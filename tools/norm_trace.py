#!/usr/bin/env python
"""P2.5's gated E-norm drift, as a CURVE across training rather than an endpoint.

WHY THIS EXISTS, AND WHY IT IS A POST-HOC TOOL RATHER THAN A TRAINING FLAG
--------------------------------------------------------------------------
`p2_deciding_measurements_handoff.md` S2, P2.5 asked for exactly one thing:

    "Add a norm trace to the cell's periodic diagnostic so that if it matters at
     10k steps there is a curve rather than an endpoint."

That did not happen.  `pace/p1_arms.sbatch:232` defaults `DIAG_INTERVAL=0` and
only `PROBE=1` forces it to 10, so the P2.3 cells ran all 24,414 steps with the
periodic architecture diagnostic OFF and there is no in-training trace to read.

The measurement is still recoverable, because `train.py` saves periodically and
NOTHING rotates the saves: the cells left checkpoints at 95000/100000/105000/
110000/115000/115966.  Running `evals/diag_dual_channel_walk.py` on each and
reading the norms off the resulting JSONs reconstructs the curve after the fact.
That is what `pace/norm_trace.sbatch` does; this file is the assembler.

WHAT THE CURVE IS FOR.  On the 400-step probes the gated arms' `e_write_norm`
climbed monotonically after the lap and the final row norm was 135.9 / 137.0
against a FLAT 112.5 on accum -- +21%, with no renorm at the operating point.
Over 400 steps that is benign.  The question P2.5 could not answer is whether it
is still climbing at 24k, because an endpoint cannot distinguish "settled high"
from "on its way up".  Those have different consequences for a 16-day 5B run:
settled is a calibration constant, still-climbing is an instability with a
schedule, and you want to know which BEFORE committing the compute.

TWO SLOPES, AND THEY ANSWER DIFFERENT QUESTIONS.  Do not conflate them.

  WITHIN a walk   -- `e_write_norm` across chunks at one checkpoint.  This is
                     the buffer's behaviour inside a single sequence: the ring
                     re-writes rows as it laps, so a post-lap climb here is the
                     gate's doing.  Reported as `within_slope`, fitted on the
                     post-lap chunks only (`--lap_chunk`, default K/W).
  ACROSS training -- the endpoint norm at successive checkpoints.  THIS is
                     P2.5's question.  Reported as `across_slope`, per 1k steps.

A flat `across_slope` with a positive `within_slope` is the benign reading: the
gate lifts norms within a sequence and training has found a stable level for
that lift.  A positive `across_slope` is the one that matters.

INPUT.  Walk JSONs as written by `diag_dual_channel_walk.py --out`.  Either

    --walk STEP=PATH [--walk STEP=PATH ...]        explicit, or
    --dir DIR                                      glob DIR/walk_*.json and
                                                   take the step from the name

STATISTICS.  `final_carry.e_row_norm` is the primary (it is the norm of the
block the next chunk actually READS).  `rows[-1].e_write_norm` is carried beside
it because it is what the probes reported and the +21% figure came from, and a
curve that disagrees between the two is itself informative.  `e_over_s0` is the
scale-free version and is the one to quote across models.

NO GPU, NO TORCH.  Pure JSON.  Run it on a login node.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
import textwrap


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------

def _step_from_name(path):
    """Pull the step out of walk_115966.json / walk-115966.json / 115966.json."""
    base = os.path.basename(path)
    m = re.search(r"(\d{3,})", base)
    return int(m.group(1)) if m else None


def load_walks(args):
    """-> [(step, path, record)], sorted by step.  Refuses duplicates."""
    items = []
    for spec in args.walk:
        if "=" not in spec:
            sys.exit("--walk wants STEP=PATH, got %r" % (spec,))
        step, path = spec.split("=", 1)
        try:
            items.append((int(step), path))
        except ValueError:
            sys.exit("--walk STEP must be an integer, got %r" % (step,))
    for d in args.dir:
        found = sorted(glob.glob(os.path.join(d, "walk_*.json")))
        if not found:
            sys.exit("no walk_*.json under %s" % (d,))
        for path in found:
            step = _step_from_name(path)
            if step is None:
                sys.exit("cannot read a step out of %r; use --walk STEP=PATH"
                         % (path,))
            items.append((step, path))

    if not items:
        sys.exit("nothing to read: pass --walk STEP=PATH or --dir DIR")

    seen = {}
    out = []
    for step, path in sorted(items):
        if step in seen:
            sys.exit("two walks claim step %d: %s and %s"
                     % (step, seen[step], path))
        seen[step] = path
        try:
            with open(path, encoding="utf-8") as fh:
                out.append((step, path, json.load(fh)))
        except FileNotFoundError:
            sys.exit("no such walk: %s" % (path,))
        except json.JSONDecodeError as exc:
            sys.exit("%s is not valid JSON (%s) -- a killed walk writes nothing, "
                     "so this is usually a truncated redirect" % (path, exc))
    return out


# --------------------------------------------------------------------------
# statistics
# --------------------------------------------------------------------------

def ols_slope(xs, ys):
    """Least-squares slope.  None when it is not defined (n<2, or x constant)."""
    pairs = [(x, y) for x, y in zip(xs, ys) if y is not None]
    n = len(pairs)
    if n < 2:
        return None
    mx = sum(x for x, _ in pairs) / n
    my = sum(y for _, y in pairs) / n
    den = sum((x - mx) ** 2 for x, _ in pairs)
    if den == 0:
        return None
    return sum((x - mx) * (y - my) for x, y in pairs) / den


def within_walk(rec, lap_chunk):
    """The per-chunk E-write norm inside ONE walk, and its post-lap slope."""
    rows = rec.get("rows") or []
    geo = rec.get("geometry") or {}
    W, K = geo.get("n_vec"), geo.get("n_slots")
    route = geo.get("route", "accum")

    # The lap: under "grow" the read block reaches K at chunk K/W, and only
    # after that does the ring start GATING rows rather than appending them.
    # Before it, a gated buffer is bit-identical to accum (buffers.py's own
    # note), so a slope fitted across the lap boundary measures the fill, not
    # the gate.
    if lap_chunk is None:
        lap_chunk = (int(K / W) if (W and K and route == "ring") else 1)

    norms = [(r.get("chunk"), r.get("e_write_norm"), r.get("e_over_s0"))
             for r in rows]
    post = [(c, n, o) for c, n, o in norms if c is not None and c >= lap_chunk]
    return {
        "route": route,
        "n_vec": W,
        "n_slots": K,
        "lap_chunk": lap_chunk,
        "chunks": [c for c, _, _ in norms],
        "e_write_norm": [n for _, n, _ in norms],
        "e_over_s0": [o for _, _, o in norms],
        "within_slope": ols_slope([float(c) for c, _, _ in post],
                                  [n for _, n, _ in post]),
        "post_lap_first": (post[0][1] if post else None),
        "post_lap_last": (post[-1][1] if post else None),
    }


def endpoint(rec):
    """The statistics that describe the block the NEXT chunk would read."""
    fc = rec.get("final_carry") or {}
    rows = rec.get("rows") or []
    last = rows[-1] if rows else {}
    health = rec.get("health") or {}
    return {
        "e_row_norm": fc.get("e_row_norm"),
        "e_centred_cosine": fc.get("e_centred_cosine"),
        "e_eff_rank_pr": fc.get("e_eff_rank_pr"),
        "e_eff_rank_entropy": fc.get("e_eff_rank_entropy"),
        "last_e_write_norm": last.get("e_write_norm"),
        "last_e_over_s0": last.get("e_over_s0"),
        "last_loss": last.get("loss"),
        "chance_margin": (health.get("margin") if isinstance(health, dict)
                          else None),
    }


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------

def _f(v, spec="9.3f", none="        -"):
    return none if v is None else format(v, spec)


def report(walks, lap_chunk, flat_tol, out=sys.stdout):
    def p(*a):
        print(*a, file=out)

    per_step = []
    for step, path, rec in walks:
        row = {"step": step, "path": path}
        row.update(endpoint(rec))
        row["within"] = within_walk(rec, lap_chunk)
        per_step.append(row)

    geo = walks[0][2].get("geometry") or {}
    p("")
    p("  buffer %s  route=%s  W=%s  K=%s  chunks=%s  chunk_len=%s"
      % (geo.get("buffer"), geo.get("route"), geo.get("n_vec"),
         geo.get("n_slots"), geo.get("chunks"), geo.get("chunk_len")))

    # --- the health veto, FIRST.  RED 10's lesson: a comparison of two
    # chance-level losses still looks like a result, and every number below is
    # a comparison.  If any walk in the series scored at chance, the curve is
    # not interpretable and no amount of slope arithmetic fixes it.
    at_chance = [r["step"] for r in per_step
                 if r["chance_margin"] is not None and r["chance_margin"] <= 0]
    if at_chance:
        p("")
        p("  *** AT CHANCE at steps: " + ", ".join(str(s) for s in at_chance))
        p("  *** chance_margin <= 0 means the walk scored no better than a")
        p("  *** uniform guess, so its norms describe a model that is not")
        p("  *** predicting.  Fix the walk before reading the curve below.")

    p("")
    p("  ACROSS TRAINING -- P2.5's question")
    p("  %8s%12s%12s%9s%13s%10s%13s"
      % ("step", "e_row_norm", "last_write", "/s0", "eff_rank_pr",
         "entropy", "centred_cos"))
    p("  " + "-" * 77)
    for r in per_step:
        p("  %8d%s%s%s%s%s%s"
          % (r["step"],
             _f(r["e_row_norm"], "12.3f", "           -"),
             _f(r["last_e_write_norm"], "12.3f", "           -"),
             _f(r["last_e_over_s0"], "9.2f", "        -"),
             _f(r["e_eff_rank_pr"], "13.2f", "            -"),
             _f(r["e_eff_rank_entropy"], "10.2f", "         -"),
             _f(r["e_centred_cosine"], "13.4f", "            -")))

    steps = [float(r["step"]) for r in per_step]
    across = {}
    for key in ("e_row_norm", "last_e_write_norm", "e_eff_rank_pr"):
        s = ols_slope(steps, [r[key] for r in per_step])
        across[key] = None if s is None else s * 1000.0   # per 1k steps

    p("")
    p("  slope per 1,000 steps:")
    for key in ("e_row_norm", "last_e_write_norm", "e_eff_rank_pr"):
        p("    %-20s%s" % (key, _f(across[key], "10.4f", "         -")))

    p("")
    p("  WITHIN EACH WALK -- the gate's per-sequence lift, not P2.5's question")
    p("  %8s%6s%16s%10s%14s"
      % ("step", "lap", "post-lap first", "last", "slope/chunk"))
    p("  " + "-" * 54)
    for r in per_step:
        w = r["within"]
        p("  %8d%6s%s%s%s"
          % (r["step"], w["lap_chunk"],
             _f(w["post_lap_first"], "16.3f", "               -"),
             _f(w["post_lap_last"], "10.3f", "         -"),
             _f(w["within_slope"], "14.4f", "             -")))

    # --- the verdict, stated in the terms P2.5 asked for --------------------
    p("")
    span = steps[-1] - steps[0]
    base = per_step[0]["e_row_norm"]
    tip = per_step[-1]["e_row_norm"]
    verdict = "UNREADABLE"
    detail = ""
    if at_chance:
        detail = "a walk in the series is at chance; fix that first."
    elif base is None or tip is None:
        detail = ("e_row_norm missing from a walk -- check the walk wrote "
                  "final_carry.")
    elif len(per_step) < 3:
        verdict = "UNDERPOWERED"
        detail = ("only %d checkpoints; a slope over fewer than 3 points is a "
                  "line through noise.  Add saves and re-run." % len(per_step))
    elif not base:
        detail = "e_row_norm is zero at the first checkpoint; nothing to divide by."
    else:
        frac = (tip - base) / base
        drift = across["e_row_norm"]
        if abs(frac) <= flat_tol:
            verdict = "SETTLED"
            detail = ("e_row_norm moved %+.2f%% over %s steps (tolerance "
                      "+/-%.0f%%).  The +21%% the 400-step probes saw is a LEVEL "
                      "the gate reaches, not a trend.  Benign for the full run; "
                      "record the level and move on."
                      % (frac * 100, format(int(span), ","), flat_tol * 100))
        elif frac > flat_tol:
            verdict = "STILL CLIMBING"
            detail = ("e_row_norm rose %+.2f%% over %s steps (%s per 1k).  This "
                      "is the reading that costs money: extrapolate to the 5B "
                      "run's step count before committing, and price a renorm "
                      "at the operating point."
                      % (frac * 100, format(int(span), ","),
                         "?" if drift is None else "%+.4f" % drift))
        else:
            verdict = "FALLING"
            detail = ("e_row_norm fell %+.2f%% over %s steps.  Unexpected -- the "
                      "probes had it climbing.  Check the walks read the arms "
                      "you think they did."
                      % (frac * 100, format(int(span), ",")))
    p("  VERDICT: %s" % verdict)
    for line in textwrap.wrap(detail, width=70):
        p("    " + line)
    p("")
    p("  A flat across-slope WITH a positive within-slope is the benign shape:")
    p("  the gate lifts norms inside a sequence and training has found a stable")
    p("  level for that lift.  Only the across-slope is P2.5's question.")

    return {"per_step": per_step, "across_slope_per_1k": across,
            "verdict": verdict, "detail": detail,
            "at_chance_steps": at_chance, "geometry": geo}


def main():
    ap = argparse.ArgumentParser(
        description="P2.5 E-norm drift across saved checkpoints (no GPU).")
    ap.add_argument("--walk", action="append", default=[], metavar="STEP=PATH",
                    help="one walk JSON and the step it came from; repeatable")
    ap.add_argument("--dir", action="append", default=[], metavar="DIR",
                    help="glob DIR/walk_*.json, step taken from the filename")
    ap.add_argument("--lap_chunk", type=int, default=None,
                    help="first post-lap chunk for the within-walk slope "
                         "(default K/W for a ring, 1 otherwise)")
    ap.add_argument("--flat_tol", type=float, default=0.05,
                    help="|relative drift| at or below this reads as SETTLED "
                         "(default 0.05)")
    ap.add_argument("--out", default=None, help="write the record as JSON")
    args = ap.parse_args()

    walks = load_walks(args)
    print("=== E-norm trace | %d checkpoints | steps %d..%d ==="
          % (len(walks), walks[0][0], walks[-1][0]))
    for step, path, _ in walks:
        print("    %8d  %s" % (step, path))

    rec = report(walks, args.lap_chunk, args.flat_tol)

    if args.out:
        d = os.path.dirname(os.path.abspath(args.out))
        os.makedirs(d, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(rec, fh, indent=2, default=str)
        print("\nwrote %s" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
