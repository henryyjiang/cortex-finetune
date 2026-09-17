"""
Compare the P1 probe arms against each other a few hundred steps into training.

WHAT "AGAINST B2" MEANS HERE, because getting this wrong invalidates the whole
comparison.  B2 itself is not a comparison target: it is a completed 5B-token
conversion on a different corpus schedule with a different step count and a
different W.  The valid contrast is the arm that carries B2'S BUFFER -- A1,
accum -- branched from the SAME heal checkpoint, on the SAME corpus, at the SAME
steps, with the SAME seed.  So `--reference a1` is the default, every delta is
reported against it, and this tool refuses to print a delta against an arm whose
step window does not overlap the reference's.

The arms are paired STEP BY STEP.  Same parent, same seed, same data order, so
at step s every arm saw the same batch; the per-step loss difference therefore
has the batch's own difficulty divided out, which is the only reason a few
hundred steps can say anything at all.

WHAT A FEW HUNDRED STEPS CAN AND CANNOT SAY.  It CAN say whether each piece of
the architecture is attached to the loss -- did either gate leave its
exactly-zero init, does Z receive gradient, is the carry collapsing, is anything
non-finite -- and those are the questions that decide whether a cell is worth
finishing.  It CANNOT say whether Z helps or whether the gate beats accum: those
need the full cells, the 2x2 (evals/eval_carry_2x2.py) and the influence horizon
(evals/eval_influence_horizon.py).  This tool prints that sentence in its own
output rather than trusting a reader to remember it, because a green probe in a
log is exactly the kind of thing that gets quoted later as "Z worked".

INPUTS, in the order it looks for them
  <dir>/cortex_diag.jsonl   train.py's periodic architecture diagnostic
                            (--cortex.diag_interval; PROBE=1 sets it)
  --csv <arm>=<file>        a wandb export (tools/pull_wandb_metrics.py) for
                            per-STEP loss, which is denser than the diagnostic
  --eval <arm>=<file>       any eval JSON written for that arm: the 2x2, the
                            carry ablation, the gate pre-registration, the
                            pre-launch gates.  Headline numbers are pulled by
                            KEY, so an unrecognised file is reported as
                            unrecognised rather than silently ignored.

USAGE
    python tools/compare_arms.py \
        --arm a1=cortex-retrofit/probe-p1-a1-accum-w16-cc8 \
        --arm a2=cortex-retrofit/probe-p1-a2-accum-w16-cc8-z \
        --arm a3=cortex-retrofit/probe-p1-a3-gated-w16k64-cc8 \
        --arm a3z=cortex-retrofit/probe-p1-a3z-gated-w16k64-cc8-z \
        --reference a1 --out eval_results/p1_probe_compare.json
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import sys
from datetime import datetime

import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

#: What each arm is, so the printed table can say it rather than leaving the
#: reader to decode a run name.  A1 is the family control: B2's buffer.
ARM_MEANING = {
    "a1": "accum, E only      -- B2's buffer, the family control",
    "a2": "accum + Z          -- does the Z channel exist at all",
    "a3": "gated ring, E only -- the buffer-geometry cell",
    "a3z": "gated ring + Z     -- accum vs the gate with Z held on",
}

#: Liveness checks.  Each is a question a few hundred steps CAN answer, and each
#: has a documented failure that produced a perfectly healthy loss curve.
LIVENESS = (
    ("gate left its zero init", "gate_left_init",
     "at gate_init='zero' both projections are EXACTLY zero, so any movement "
     "proves the gradient path exists; none means 16.8M dead parameters"),
    ("Z gate left its zero init", "gate_z_left_init",
     "the Z gate trains through the write even when the read is starved; zero "
     "movement means the Z channel is bolted on and inert"),
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("compare the P1 probe arms")
    p.add_argument("--arm", action="append", default=[], metavar="LABEL=DIR",
                   required=True)
    p.add_argument("--csv", action="append", default=[], metavar="LABEL=FILE",
                   help="wandb export with per-step train/loss")
    p.add_argument("--eval", action="append", default=[], metavar="LABEL=FILE")
    p.add_argument("--reference", default="a1",
                   help="the arm every delta is measured against.  a1 by "
                        "default because it is the one carrying B2's buffer")
    p.add_argument("--last", type=int, default=100,
                   help="window of most recent steps used for the loss summary")
    p.add_argument("--trained_depth", type=int, default=0,
                   help="mean_recurrence the arms actually trained at (8 for "
                        "the P1 probes).  Re-anchors read_live_frac onto the "
                        "sweep row measured there, because gate 3 falls back "
                        "to config.mean_recurrence -- still 32 on every "
                        "B2-family checkpoint -- when prelaunch_final was run "
                        "without --trained_depth")
    p.add_argument("--boot", type=int, default=5000)
    p.add_argument("--out", default=None)
    return p.parse_args()


def _pairs(items):
    out = {}
    for it in items:
        if "=" not in it:
            raise SystemExit(f"expected LABEL=VALUE, got {it!r}")
        k, v = it.split("=", 1)
        out.setdefault(k, []).append(v)
    return out


# ---------------------------------------------------------------------------
# reading
# ---------------------------------------------------------------------------

def read_diag(run_dir: str) -> list:
    """train.py's cortex_diag.jsonl -> a list of rows, oldest first.

    A resumed run APPENDS, so the same step can appear twice; the later row
    wins, because it is the one the run actually continued from.  Silently
    keeping both would double-count a step in every paired comparison.
    """
    path = os.path.join(run_dir, "cortex_diag.jsonl")
    if not os.path.isfile(path):
        return []
    by_step = {}
    with open(path, encoding="ascii", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue            # a torn final line after a kill; skip it
            if "step" in row:
                by_step[int(row["step"])] = row
    return [by_step[s] for s in sorted(by_step)]


def read_csv_losses(path: str) -> dict:
    """wandb export -> {step: loss}.  Stdlib csv, no pandas: this has to run in
    the cluster's cortex-retro env, which does not have it."""
    out = {}
    with open(path, encoding="utf-8", errors="replace", newline="") as fh:
        for row in csv.DictReader(fh):
            try:
                s = int(float(row["train/step"]))
                v = float(row["train/loss"])
            except (KeyError, TypeError, ValueError):
                continue
            if v == v:                      # drop NaN rows
                out[s] = v
    return out


def read_live_from_gate(gate: dict, trained_depth: int = 0) -> dict:
    """Pull read_live_frac out of gate 3 WITH the depth it was measured at.

    The bare fraction is not quotable.  `at_run_config` is anchored on
    config.mean_recurrence unless prelaunch_final was given --trained_depth, and
    on every B2-family checkpoint that field still says 32 while the arm trained
    at 8: 0.0195 against 0.546, a factor of 28.  P11 section 3 makes a null
    result for Z mean OPPOSITE things across that gap -- a starved read makes
    the null uninformative, a live one makes it damning -- so the anchor travels
    with the number, and a known trained depth re-anchors it onto the sweep row
    that was actually measured there.  The displaced value is kept, not dropped,
    so a table can be reconciled against the record it came from.
    """
    at = gate.get("at_run_config") or {}
    out = {"read_live_frac": at.get("read_live_frac"),
           "read_live_at_mr": at.get("mean_recurrence"),
           "read_live_anchor": gate.get("anchor", "unrecorded")}
    if not trained_depth or out["read_live_at_mr"] == trained_depth:
        return out
    row = next((r for r in gate.get("sweep_mean_recurrence") or []
                if r.get("mean_recurrence") == trained_depth), None)
    if row is None:
        out["read_live_note"] = (
            f"NOT re-anchored: --trained_depth {trained_depth} was given but "
            f"this record has no sweep row measured there")
        return out
    out["read_live_frac_at_run_config"] = out["read_live_frac"]
    out["read_live_frac"] = row.get("read_live_frac")
    out["read_live_at_mr"] = trained_depth
    out["read_live_note"] = (
        f"re-anchored onto the mr={trained_depth} sweep row; the record's own "
        f"at_run_config was measured at mr={at.get('mean_recurrence')}, which "
        f"is not a depth this arm ran")
    return out


def read_eval(path: str, trained_depth: int = 0) -> dict:
    """Pull the headline out of any eval JSON this repo writes, BY KEY.

    Keyed rather than positional so a file whose shape changed is reported as
    unrecognised instead of contributing a number from the wrong field.  That
    distinction is the whole reason the 'unrecognised' row exists.
    """
    with open(path, encoding="utf-8", errors="replace") as fh:
        doc = json.load(fh)
    out = {"file": os.path.basename(path)}
    if isinstance(doc, dict) and "gates" in doc:          # prelaunch_final
        out["kind"] = "prelaunch"
        out["all_passed"] = doc.get("all_passed")
        for g in doc.get("gates", []):
            if g.get("gate") == "donor_control":
                for k in ("content_delta_both", "content_delta_e",
                          "content_delta_z", "column_delta"):
                    if k in g:
                        out[k] = g[k]
            if g.get("gate") == "read_live_fraction":
                out.update(read_live_from_gate(g, trained_depth))
    elif isinstance(doc, dict) and "cells" in doc:        # carry 2x2
        out["kind"] = "carry_2x2"
        out["cells"] = doc["cells"]
    elif isinstance(doc, dict) and "spread" in doc:       # gate pre-registration
        out["kind"] = "gate_prereg"
        sp = doc["spread"]
        out["fg_across_input_std"] = sp.get("across_input_std")
        out["fg_across_input_swing"] = sp.get("across_input_swing")
        out["fg_mean"] = sp.get("mean")
        out["prereg_passed"] = doc.get("passed")
    elif isinstance(doc, dict) and "carry_delta" in doc:  # carry ablation
        out["kind"] = "carry_ablation"
        out["carry_delta"] = doc["carry_delta"]
    else:
        out["kind"] = "unrecognised"
        out["top_level_keys"] = sorted(doc)[:12] if isinstance(doc, dict) else None
    return out


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------

def paired_ci(deltas, n_boot: int = 5000, seed: int = 0):
    """Bootstrap CI on the paired mean.

    Same definition as `evals/eval_influence_horizon.paired_ci` -- pinned
    against it in tests/test_compare_arms.py rather than imported, because that
    module pulls the model-loading stack and this tool must run on a login node
    with nothing but a checkpoint directory.
    """
    if not deltas:
        return (float("nan"),) * 3
    t = torch.tensor(deltas, dtype=torch.float64)
    g = torch.Generator().manual_seed(seed)
    idx = torch.randint(len(t), (n_boot, len(t)), generator=g)
    means = t[idx].mean(dim=1)
    lo, hi = torch.quantile(means, torch.tensor([0.025, 0.975],
                                                dtype=torch.float64))
    return float(t.mean()), float(lo), float(hi)


def loss_series(arm: dict) -> dict:
    """{step: loss}, preferring the dense wandb export over the diagnostic."""
    if arm.get("csv_losses"):
        return arm["csv_losses"]
    return {int(r["step"]): float(r["loss"])
            for r in arm["diag"] if "loss" in r}


def compare_losses(arms: dict, ref: str, last: int, n_boot: int) -> dict:
    """Paired per-step loss deltas against the reference arm.

    PAIRED IS NOT OPTIONAL at this horizon.  Unpaired, the per-batch difficulty
    swamps a few-hundred-step architecture difference completely; paired, the
    batch is the same batch and divides out.  It only works because the arms
    share a parent, a seed and a data order -- so the overlap is reported, and
    an arm with no shared steps gets no delta rather than a meaningless one.
    """
    series = {k: loss_series(v) for k, v in arms.items()}
    ref_s = series.get(ref, {})
    out = {"reference": ref, "arms": {}}
    for label, s in series.items():
        rec = {"steps": len(s)}
        if s:
            steps = sorted(s)
            rec["first_step"], rec["last_step"] = steps[0], steps[-1]
            tail = steps[-last:]
            rec["mean_loss_last"] = sum(s[t] for t in tail) / len(tail)
            rec["window"] = len(tail)
        if label != ref and ref_s and s:
            shared = sorted(set(s) & set(ref_s))
            rec["shared_steps"] = len(shared)
            if shared:
                d = [s[t] - ref_s[t] for t in shared]
                m, lo, hi = paired_ci(d, n_boot)
                rec["paired_delta_vs_ref"] = m
                rec["ci95"] = [lo, hi]
                # A CI that straddles zero at this horizon is the EXPECTED
                # result and is reported as such, not as a negative finding.
                rec["separated_from_zero"] = bool(lo > 0 or hi < 0)
            else:
                rec["why_no_delta"] = (
                    "no shared steps with the reference arm -- the two did not "
                    "run the same window, so a delta would not be paired")
        out["arms"][label] = rec
    return out


def health_summary(rows: list) -> dict:
    """Collapse a diagnostic trajectory into what a launch decision needs."""
    if not rows:
        return {}
    last = rows[-1]
    out = {"diag_rows": len(rows),
           "first_step": rows[0].get("step"), "last_step": last.get("step")}
    for key in ("e_row_norm", "e_centred_cosine", "e_eff_rank_pr",
                "z_row_norm", "z_over_e_norm", "z_eff_rank_pr",
                "fg_at_bias", "fg_z_at_bias", "rows", "latent_carry",
                "z_read_grad_frac", "z_write_grad_frac"):
        if key in last:
            out[key] = last[key]
    # THE STEP a flag first went true, not just its final value: "the gate
    # trained" and "the gate took 300 steps to start training" are different
    # facts and only the trajectory holds the second.
    #
    # A check is only APPLICABLE when the arm has the thing it asks about: an
    # accum arm has no gate and an E-only arm has no Z gate, and reporting
    # those as "NOT LIVE" would put a failure banner on two arms that are
    # behaving exactly as designed -- which is how a real failure gets ignored.
    for _, key, _ in LIVENESS:
        if not any(key in r for r in rows):
            continue
        hits = [r.get("step") for r in rows if r.get(key)]
        out[key] = bool(hits)
        out[key + "_step"] = hits[0] if hits else None
    gn = [r["grad_norm"] for r in rows if "grad_norm" in r]
    if gn:
        out["grad_norm_max"] = max(gn)
        out["grad_norm_mean"] = sum(gn) / len(gn)
        out["grad_norm_nonfinite"] = sum(1 for v in gn if not math.isfinite(v))
    # The read gradient fraction is cumulative over the run, so the LAST row is
    # the run average -- the number that goes in the Z pre-registration.
    if "z_read_grad_frac" in last:
        out["z_read_grad_frac_run_avg"] = last["z_read_grad_frac"]
    return out


# ---------------------------------------------------------------------------
# printing
# ---------------------------------------------------------------------------

def _f(v, spec="9.4f", none="        -"):
    if v is None or (isinstance(v, float) and v != v):
        return none
    try:
        return format(v, spec)
    except (TypeError, ValueError):
        return str(v)[:len(none)].rjust(len(none))


def print_comparison(rec, out=sys.stdout) -> None:
    p = lambda *a: print(*a, file=out)
    labels = list(rec["arms"])
    ref = rec["loss"]["reference"]
    p("")
    p("=" * 96)
    p(f"P1 PROBE COMPARISON   reference = {ref}"
      + (f"  ({ARM_MEANING[ref]})" if ref in ARM_MEANING else ""))
    p("=" * 96)
    for label in labels:
        meaning = ARM_MEANING.get(label, rec["arms"][label].get("dir", ""))
        p(f"  {label:<5} {meaning}")
    p("")
    p("-- LOSS (paired per step; same parent, seed and data order) ------------")
    p(f"{'arm':<6}{'steps':>7}{'window':>8}{'mean loss':>11}"
      f"{'delta vs ref':>14}{'95% CI':>26}{'sep?':>6}")
    for label in labels:
        L = rec["loss"]["arms"].get(label, {})
        ci = L.get("ci95")
        ci_s = ("" if not ci else f"[{ci[0]:+.4f}, {ci[1]:+.4f}]")
        p(f"{label:<6}{L.get('steps', 0):>7}{L.get('window', 0):>8}"
          f"{_f(L.get('mean_loss_last'), '11.4f')}"
          f"{_f(L.get('paired_delta_vs_ref'), '+14.4f')}"
          f"{ci_s:>26}"
          f"{('yes' if L.get('separated_from_zero') else ('no' if 'ci95' in L else '-')):>6}")
        if L.get("why_no_delta"):
            p(f"       {L['why_no_delta']}")
    p("")
    p("-- ARCHITECTURE HEALTH (what a few hundred steps CAN answer) -----------")
    p(f"{'arm':<6}{'rows':>6}{'E rank':>8}{'E cos':>8}{'Z/E':>9}"
      f"{'fg':>7}{'fg_z':>7}{'Zread':>7}{'Zwrite':>8}{'gate?':>7}{'gateZ?':>8}")
    for label in labels:
        h = rec["arms"][label].get("health", {})
        p(f"{label:<6}{_f(h.get('rows'), '6.0f', '     -')}"
          f"{_f(h.get('e_eff_rank_pr'), '8.2f', '       -')}"
          f"{_f(h.get('e_centred_cosine'), '8.3f', '       -')}"
          f"{_f(h.get('z_over_e_norm'), '9.4f', '        -')}"
          f"{_f(h.get('fg_at_bias'), '7.3f', '      -')}"
          f"{_f(h.get('fg_z_at_bias'), '7.3f', '      -')}"
          f"{_f(h.get('z_read_grad_frac'), '7.2f', '      -')}"
          f"{_f(h.get('z_write_grad_frac'), '8.2f', '       -')}"
          f"{('yes' if h.get('gate_left_init') else ('NO' if 'gate_left_init' in h else '-')):>7}"
          f"{('yes' if h.get('gate_z_left_init') else ('NO' if 'gate_z_left_init' in h else '-')):>8}")
    p("")
    for label in labels:
        h = rec["arms"][label].get("health", {})
        bad = [n for n, k, _ in LIVENESS if k in h and not h[k]]
        if bad:
            p(f"  {label}: NOT LIVE -- {', '.join(bad)}")
            for n, k, why in LIVENESS:
                if n in bad:
                    p(f"      {why}")
        if h.get("grad_norm_nonfinite"):
            p(f"  {label}: {h['grad_norm_nonfinite']} non-finite grad-norm rows "
              f"-- those updates were SKIPPED, the run was not training through them")
    p("")
    p("-- MEMORY CARRY (from eval JSONs, when they exist) ---------------------")
    any_eval = False
    for label in labels:
        for ev in rec["arms"][label].get("evals", []):
            any_eval = True
            kind = ev.get("kind")
            body = {k: v for k, v in ev.items()
                    if k not in ("file", "kind", "top_level_keys")}
            p(f"  {label:<5} {kind:<16} {ev['file']}")
            if kind == "unrecognised":
                p(f"        keys {ev.get('top_level_keys')} -- not a shape this "
                  f"tool knows; NOT summarised rather than summarised wrongly")
            else:
                for k, v in body.items():
                    p(f"        {k} = {_f(v, '.4f') if isinstance(v, float) else v}")
    if not any_eval:
        p("  none passed in.  The carry numbers that decide the mechanism come")
        p("  from evals/eval_carry_2x2.py (E/Z factorial, matched columns) and")
        p("  evals/eval_influence_horizon.py (the replacement for the x1.31")
        p("  compounding metric, which SCORES A FIXED-WIDTH GATED BUFFER AS A")
        p("  FAILURE for plateauing, i.e. for behaving correctly).")
    p("")
    p("WHAT THIS DOES NOT SAY.  A few hundred steps cannot decide whether Z")
    p("helps or whether the gate beats accum.  Those need the full cells, the")
    p("2x2 and the influence horizon.  What it does decide is whether each arm")
    p("is worth finishing: every 'NO' above is a piece of the architecture")
    p("sitting inert behind a healthy loss curve.")
    p("")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build(arm_dirs: dict, csvs: dict, evals: dict, ref: str, last: int,
          n_boot: int, trained_depth: int = 0) -> dict:
    arms = {}
    for label, dirs in arm_dirs.items():
        d = dirs[0]
        rec = {"dir": d, "diag": read_diag(d)}
        if label in csvs:
            rec["csv_losses"] = read_csv_losses(csvs[label][0])
        rec["evals"] = [read_eval(p, trained_depth) for p in evals.get(label, [])
                        if os.path.isfile(p)]
        rec["health"] = health_summary(rec["diag"])
        arms[label] = rec
    out = {"when": datetime.now().isoformat(timespec="seconds"),
           "trained_depth": trained_depth or None,
           "arms": arms,
           "loss": compare_losses(arms, ref, last, n_boot)}
    for a in out["arms"].values():
        a.pop("diag", None)         # the trajectory is large; health keeps it
        a.pop("csv_losses", None)
    return out


def main() -> int:
    args = parse_args()
    arm_dirs = _pairs(args.arm)
    if args.reference not in arm_dirs:
        print(f"ERROR: --reference {args.reference!r} is not one of the arms "
              f"{sorted(arm_dirs)}.  Every delta is measured against it, and "
              f"it should be the arm carrying B2's buffer (a1, accum) unless "
              f"you mean something else.")
        return 1
    missing = [f"{k}={v[0]}" for k, v in arm_dirs.items()
               if not os.path.isdir(v[0])]
    if missing:
        print(f"ERROR: no such run directory: {missing}")
        return 1
    rec = build(arm_dirs, _pairs(args.csv), _pairs(args.eval),
                args.reference, args.last, args.boot, args.trained_depth)
    empty = [k for k, v in rec["arms"].items() if not v["health"]]
    if empty:
        print(f"WARNING: no cortex_diag.jsonl under {empty} -- those runs were "
              f"launched without --cortex.diag_interval, so only their loss can "
              f"be compared.  PROBE=1 in pace/p1_arms.sbatch sets it.")
    print_comparison(rec)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="ascii") as fh:
            json.dump(rec, fh, indent=2, default=str)
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
