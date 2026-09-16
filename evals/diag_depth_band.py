"""
P0.7 — is the informative depth band ABSOLUTE or RELATIVE?  One forward-only
sweep, and it blocks the Z channel's slot->depth map.

THE QUESTION
------------
P0.1 measured the usable delta band at t ~ 2..9 — but at a SINGLE T (32, on an
mr8 checkpoint, i.e. out of distribution for it).  It therefore cannot say
whether "informative" tracks the ABSOLUTE step or the FRACTION of the loop, and
the two give different write rules that no single fixed assignment serves:

  rule A, clamped absolute   slots tile t = 2..9, clamped to T-1
  rule B, relative fraction  slot j takes k_j = round(f_j * (T-1))

           T=8                T=16              T=32
  A   2..7, top lumps    2..9 (x2)         2..9 (x2) — IDENTICAL to T=16,
                                           the loop's extra depth unused
  B   1..4               2..8              3..17 — spreads into the region
                                           P0.1 calls saturated

If the band is absolute, B over-samples dead depths at large T.  If it is
relative, A throws away the loop's extra depth at T=32.  T is also SAMPLED PER
BATCH (`randomized_iteration_sampler`), so a fixed absolute depth can simply not
exist on a given step — which is why the answer has to be measured rather than
assumed.

WHAT IT MEASURES
----------------
`record_trajectory` (shared with diag_latent_scale.py, so the two probes cannot
diverge) at T = 4, 8, 16, 32, and per T the band's upper edge under two
independent criteria:

  MAGNITUDE   the last t whose ||d_t|| is at least MAG_FLOOR x ||s0||.  This is
              the criterion that reproduces P0.1's published 2..9: on that run
              d/s0 was 0.90 at t=8 and 0.45 at t=10, so MAG_FLOOR = 0.5 puts the
              edge at 9.  It is fixed HERE, before the sweep, for that reason.
  NOVELTY     the last t with cos(d_t, d_t-1) >= NOVELTY_FLOOR.  A converged
              trajectory can still be exploring; consecutive deltas running
              anti-parallel (P0.1 reached -0.95) means the late depths are one
              direction re-walked and carry nothing new for Z to write.
              -0.7 sits between P0.1's -0.63 at t=10 and -0.87 at t=16.

Both edges are reported because they can disagree, and a disagreement is itself
the finding: magnitude says "there is still signal", novelty says "it is the
same signal again".

  t_lo is FIXED AT 2 and is not measured.  d_1 was 23.8x ||s0|| in P0.1 — a
  different regime, not a bigger version of the same one — and the design
  already excludes it.  Measuring an edge the design will not use would only
  make the verdict noisier.

THE VERDICT
-----------
Rule A predicts t_hi is constant across T; rule B predicts t_hi / T is.  Scored
by coefficient of variation across the sweep, lower wins, with a 2x margin
required or the answer is INCONCLUSIVE rather than a coin flip.

RIGHT-CENSORED T VALUES ARE EXCLUDED FROM THE SCORE.  If the band is still open
at the last step (the loop ran out before the criterion was crossed) the edge is
a lower bound, not a measurement.  Keeping such a row would bias the answer in
one direction only: a truly ABSOLUTE band edge of 9 reads as 4 at T=4, which
looks like movement, while a relative edge can never be truncated at all.  The
excluded T values are printed.

AND THE ONE COMPARISON THAT ACTUALLY DISCRIMINATES is T=16 vs T=32.  At T=4 and
T=8 a clamped-absolute band and a relative band are nearly the same set — the
loop is not long enough for them to differ — so including the short T values in
the CV makes both rules look good.  The T16/T32 pair is printed separately and
is the row to read.

CAVEATS THAT SURVIVE THIS PROBE
-------------------------------
  * OUT OF DISTRIBUTION AT LARGE T.  An mr8 checkpoint driven to T=32 was never
    trained to use those depths, so a band that looks absolute may only be
    saying "this model stops at 8".  That is the same confound P0.1 carries.
    Re-run on Huginn-0125 (mean_recurrence 32) before generalising; the script
    prints the checkpoint's own mean_recurrence next to every T so the rows
    inside and outside distribution are distinguishable at a glance.
  * FLOAT32, and it is not optional.  The late deltas are ~1% of ||s_t||, inside
    bf16's relative precision; a bf16 pass invented a delta plateau at 0.37x
    that fp32 puts at 0.06x.  The quantization-floor check runs per T and names
    the contaminated rows.

USAGE
    python evals/diag_depth_band.py --model_name <ckpt> \
        --text_file ../cortex_next_phase_framework.md --seq_len 512 --batch 2

    # the generalisation check, on a model actually trained at depth 32:
    python evals/diag_depth_band.py --model_name <huginn-0125> --T 4 8 16 32 64
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
from datetime import datetime

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from evals.diag_latent_scale import record_trajectory  # noqa: E402
from evals.model_utils import _unwrap, load_checkpoint  # noqa: E402

#: Band edge criteria, fixed before the sweep.  See the header for the P0.1
#: numbers each one reproduces; changing either after seeing a result is the
#: failure this probe exists to avoid.
MAG_FLOOR = 0.5          # ||d_t|| / ||s0||
NOVELTY_FLOOR = -0.7     # cos(d_t, d_{t-1})
BAND_LO = 2              # d_1 is a different regime; the design excludes it
#: How much better one rule must score before the verdict is called.
MARGIN = 2.0


def _norm(x: torch.Tensor) -> float:
    return float(x.detach().float().flatten(0, -2).norm(dim=-1).mean())


def trajectory_stats(s0: torch.Tensor, traj: list[torch.Tensor]) -> list[dict]:
    """Per-step ||d_t||/||s0|| and cos(d_t, d_{t-1}), in float32."""
    s0m = _norm(s0)
    rows, prev, prev_d = [], s0, None
    for t, s in enumerate(traj, start=1):
        d = s - prev
        cdd = None
        if prev_d is not None:
            cdd = float(torch.nn.functional.cosine_similarity(
                d.float().flatten(0, -2), prev_d.float().flatten(0, -2),
                dim=-1).mean())
        rows.append({"t": t, "d_over_s0": _norm(d) / max(s0m, 1e-12),
                     "cos_delta": cdd, "s_norm": _norm(s)})
        prev, prev_d = s, d
    return rows


def band_edges(rows: list[dict]) -> dict:
    """Upper edge of the informative band under each criterion.

    Returns None for an edge the trajectory never crosses INSIDE the measured
    range — which is a real outcome (the band ran past T) and must not be
    silently reported as "the edge is T".
    """
    mag = [r["t"] for r in rows
           if r["t"] >= BAND_LO and r["d_over_s0"] >= MAG_FLOOR]
    nov = [r["t"] for r in rows
           if r["t"] >= BAND_LO and r["cos_delta"] is not None
           and r["cos_delta"] >= NOVELTY_FLOOR]
    T = rows[-1]["t"]
    return {
        "t_hi_magnitude": max(mag) if mag else None,
        "t_hi_novelty": max(nov) if nov else None,
        # Censored = the criterion was still satisfied at the last step, so the
        # band may extend past T and the edge is a lower bound.
        "magnitude_censored": bool(mag) and max(mag) == T,
        "novelty_censored": bool(nov) and max(nov) == T,
    }


def cv(values: list[float]) -> float:
    """Coefficient of variation.  inf on an empty or zero-mean set, so a rule
    with no usable points can never win by default."""
    if len(values) < 2:
        return float("inf")
    t = torch.tensor(values, dtype=torch.float64)
    m = float(t.mean())
    return float(t.std(unbiased=False)) / abs(m) if abs(m) > 1e-12 else float("inf")


def score_rules(per_T: dict, key: str) -> dict:
    """Absolute vs relative, on one criterion's edges.

    RIGHT-CENSORED T VALUES ARE DROPPED, and the reason is not a detail -- it
    decides the verdict.  If the band would end at t=9 but the sweep only ran
    T=4, the measured edge is 3, because the loop ran out before the band did.
    That 3 is not evidence that the edge moved; it is evidence that this T could
    not see it.  Keeping it would make a perfectly ABSOLUTE band read as
    [3, 9, 9, 9] and hand the verdict to INCONCLUSIVE (or, with a shorter sweep,
    to RELATIVE) purely by construction -- the short-T rows penalise the
    absolute rule and NOTHING penalises the relative one, since a relative edge
    is never truncated by definition.

    `band_edges` already computes the flag; this is the consumer that has to
    honour it.  The dropped set is reported rather than silently removed,
    because a sweep where every row is censored has no verdict to give.
    """
    cens_key = key.replace("t_hi_", "") + "_censored"
    usable, censored = [], []
    for t in sorted(per_T):
        if per_T[t].get(key) is None:
            continue
        (censored if per_T[t].get(cens_key) else usable).append(t)
    ts = usable
    absolute = [float(per_T[t][key]) for t in ts]
    relative = [per_T[t][key] / t for t in ts]
    cv_a, cv_b = cv(absolute), cv(relative)
    if cv_a <= cv_b / MARGIN:
        verdict = "ABSOLUTE"
    elif cv_b <= cv_a / MARGIN:
        verdict = "RELATIVE"
    else:
        verdict = "INCONCLUSIVE"
    # The pair that actually discriminates: at small T a clamped-absolute band
    # and a relative band are nearly the same set, so the full-sweep CV flatters
    # both.  16 vs 32 is where the two rules genuinely disagree.
    pair = None
    if 16 in ts and 32 in ts:
        a16, a32 = per_T[16][key], per_T[32][key]
        pair = {"t_hi_16": a16, "t_hi_32": a32,
                "absolute_ratio": a32 / a16,      # 1.0 if absolute
                "relative_ratio": (a32 / 32) / (a16 / 16)}   # 1.0 if relative
    if len(ts) < 2:
        verdict = "NO VERDICT (too few uncensored T)"
    return {"T": ts, "censored_T": censored,
            "absolute": absolute, "relative": relative,
            "cv_absolute": cv_a, "cv_relative": cv_b, "verdict": verdict,
            "discriminating_pair": pair}


def slot_depth_map(rule: str, T: int, n_slots: int, lo: int, hi: int) -> list[int]:
    """The map P0.7 exists to choose: which loop depth each summary column
    writes.

    Printed for both rules at every T so the consequence of the verdict is on
    the same page as the verdict.  Under the ring, write vector j always lands
    at rows congruent to j (mod W), so this map is what makes the depth-slice
    ablation addressable -- it is a convention, and a stable one, not a runtime
    decision.

    Mechanical note kept from the design: the n_slots slots are COLUMNS in ONE
    forward, each with its own full trajectory, so this is an n_slots x (T-1)
    grid from which one cell per row is harvested.  At small T several columns
    land on the same depth -- INDEPENDENT VIEWS of that depth, not duplicates --
    and at large T most depths go unsampled, which is the intended behaviour
    under rule A and the thing rule B changes.
    """
    if T < 2:
        return [1] * n_slots
    if rule == "absolute":
        hi_c = min(hi, T - 1)
        lo_c = min(lo, hi_c)
        span = hi_c - lo_c + 1
        return [lo_c + (j % span) for j in range(n_slots)]
    f_lo, f_hi = lo / max(hi, 1), 1.0          # fractions of the loop
    return [max(1, min(T - 1, round((f_lo + (f_hi - f_lo) * j / max(n_slots - 1, 1))
                                    * (T - 1))))
            for j in range(n_slots)]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("P0.7 depth-band invariance")
    p.add_argument("--model_name", required=True)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--T", type=int, nargs="+", default=[4, 8, 16, 32],
                   help="the sweep.  16 and 32 are the discriminating pair; "
                        "4 and 8 are context and cannot separate the rules.")
    p.add_argument("--text_file", default=None,
                   help="real prose.  Random ids give the loop nothing to "
                        "converge ON, so the trajectory shape is not the "
                        "trained one and the band is not the trained band.")
    p.add_argument("--data", default=None)
    p.add_argument("--seq_len", type=int, default=512)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--n_slots", type=int, default=16,
                   help="W, for the slot->depth map the verdict implies")
    p.add_argument("--no_prefix", action="store_true")
    p.add_argument("--dtype", default="float32",
                   choices=["float32", "bfloat16"])
    p.add_argument("--device",
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", default=None)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)
    if args.dtype == "bfloat16":
        print("WARNING: bfloat16.  The late deltas are ~1% of ||s_t||, which is "
              "inside bf16's precision -- a bf16 pass reported a plateau at "
              "0.37x that fp32 puts at 0.06x.  This probe reads exactly that "
              "region.  Do not quote these numbers.", flush=True)
    device = torch.device(args.device)
    model, cfg = load_checkpoint(args.checkpoint, args.model_name, None,
                                 getattr(torch, args.dtype), device)
    inner = _unwrap(model)
    D = int(cfg.n_embd)
    trained_T = int(getattr(cfg, "mean_recurrence", 0) or 0)

    if args.data:
        from datasets import load_from_disk
        ds = load_from_disk(args.data)
        rows = [ds[i]["input_ids"][: args.seq_len] for i in range(args.batch)]
        input_ids = torch.tensor(rows, dtype=torch.long, device=device)
        source = args.data
    elif args.text_file:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.model_name,
                                            trust_remote_code=True)
        raw = io.open(args.text_file, encoding="utf-8", errors="replace").read()
        ids = tok(raw, return_tensors=None)["input_ids"]
        need = args.batch * args.seq_len
        if len(ids) < need:
            print(f"FAILED: {args.text_file} tokenizes to {len(ids)} tokens, "
                  f"need {need}.")
            return 2
        input_ids = torch.tensor(
            [ids[i * args.seq_len:(i + 1) * args.seq_len]
             for i in range(args.batch)], dtype=torch.long, device=device)
        source = args.text_file
    else:
        input_ids = torch.randint(0, int(cfg.vocab_size) - 1,
                                  (args.batch, args.seq_len), device=device)
        source = "random ids (NOT the trained trajectory shape -- see --text_file)"

    eps = float(torch.finfo(getattr(torch, args.dtype)).eps)
    print(f"\n=== P0.7 depth band | {os.path.basename(args.model_name)} | "
          f"{source} ===")
    print(f"    D={D} seq_len={args.seq_len} batch={args.batch} "
          f"dtype={args.dtype} prefix={'off' if args.no_prefix else 'on'}")
    print(f"    checkpoint mean_recurrence = {trained_T or '?'}; any T above it "
          f"is OUT OF DISTRIBUTION\n    and a band that looks absolute there may "
          f"only be saying 'this model stops at {trained_T or '?'}'.")

    per_T, steps_by_T, suspect_by_T = {}, {}, {}
    for T in sorted(set(args.T)):
        # num_steps is forced all-no-grad by record_trajectory's caller contract
        # (T, 0): identical values to the with-grad split and far cheaper.  The
        # split decides which depths are TRAINABLE, which is a different
        # question and a different probe.
        s0, traj, _ = record_trajectory(model, input_ids, T,
                                        use_prefix=not args.no_prefix)
        if s0 is None or len(traj) != T:
            print(f"FAILED at T={T}: captured {0 if not traj else len(traj)} "
                  f"steps, asked for {T}.")
            return 2
        rows = trajectory_stats(s0, traj)
        steps_by_T[T] = rows
        per_T[T] = band_edges(rows)
        suspect_by_T[T] = [r["t"] for r in rows
                           if r["d_over_s0"] * max(_norm(s0), 1e-12)
                           < 4.0 * eps * r["s_norm"]]
        print(f"  T={T:<3d} band edge: magnitude "
              f"{per_T[T]['t_hi_magnitude']}"
              f"{'+' if per_T[T]['magnitude_censored'] else ''}, novelty "
              f"{per_T[T]['t_hi_novelty']}"
              f"{'+' if per_T[T]['novelty_censored'] else ''}"
              f"{'   [OOD]' if trained_T and T > trained_T else ''}",
              flush=True)

    mag = score_rules(per_T, "t_hi_magnitude")
    nov = score_rules(per_T, "t_hi_novelty")

    print("\n" + "=" * 78)
    print("the trajectory, per T")
    print("=" * 78)
    for T, rows in steps_by_T.items():
        head = "  t      " + "".join(f"{r['t']:>7}" for r in rows[:12])
        dline = "  d/s0   " + "".join(f"{r['d_over_s0']:>7.2f}" for r in rows[:12])
        cline = "  cos    " + "".join(
            ("      -" if r["cos_delta"] is None else f"{r['cos_delta']:>7.2f}")
            for r in rows[:12])
        print(f"T={T}" + ("  (first 12 steps)" if len(rows) > 12 else ""))
        print(head); print(dline); print(cline)
        if suspect_by_T[T]:
            lo, hi = min(suspect_by_T[T]), max(suspect_by_T[T])
            print(f"  !! QUANTIZATION FLOOR at t={lo}..{hi}: those deltas are "
                  f"within 4x of {args.dtype}'s\n     resolvable difference. "
                  f"They are measuring arithmetic, not the model.")

    print("\n" + "=" * 78)
    print("VERDICT — does the band's upper edge hold at a fixed t, or a fixed "
          "t/T?")
    print("=" * 78)
    for name, sc in (("magnitude", mag), ("novelty", nov)):
        print(f"  {name:<10} edges {dict(zip(sc['T'], [int(a) for a in sc['absolute']]))}")
        if sc["censored_T"]:
            print(f"  {'':<10} excluded (band still open at T, so the edge is a "
                  f"lower bound): {sc['censored_T']}")
        print(f"  {'':<10} CV(absolute) {sc['cv_absolute']:.3f}   "
              f"CV(relative) {sc['cv_relative']:.3f}   -> {sc['verdict']}")
        p = sc["discriminating_pair"]
        if p:
            print(f"  {'':<10} T16 vs T32 (THE row to read): t_hi "
                  f"{p['t_hi_16']} -> {p['t_hi_32']}; "
                  f"absolute ratio {p['absolute_ratio']:.2f} "
                  f"(1.0 = absolute), relative ratio "
                  f"{p['relative_ratio']:.2f} (1.0 = relative)")
        else:
            print(f"  {'':<10} no T16/T32 pair in the sweep -- the full-sweep CV "
                  f"flatters BOTH rules,\n  {'':<10} because at small T they "
                  f"select nearly the same depths.  Re-run with --T 4 8 16 32.")
    if mag["verdict"] != nov["verdict"]:
        print("\n  THE TWO CRITERIA DISAGREE, and that is a finding rather than "
              "a failure:\n  magnitude says how much the state still moves, "
              "novelty says whether the\n  movement is anywhere new.  A band "
              "that is absolute in magnitude and relative\n  in novelty means "
              "the late steps keep their size while re-walking one\n  "
              "direction -- write the NOVELTY band, since Z carries the "
              "pathway, not the\n  displacement.")

    print("\n" + "=" * 78)
    print(f"the slot -> depth map each rule implies (n_slots = {args.n_slots})")
    print("=" * 78)
    hi_ref = mag["absolute"][-1] if mag["absolute"] else 9
    for T in sorted(steps_by_T):
        a = slot_depth_map("absolute", T, args.n_slots, BAND_LO, int(hi_ref))
        b = slot_depth_map("relative", T, args.n_slots, BAND_LO, int(hi_ref))
        print(f"  T={T:<3d} A {a}")
        print(f"  {'':<5} B {b}")
    print("\n  Rows are COLUMNS of one forward, each with its own trajectory, so "
          "a repeated\n  depth is an independent view of it and not a "
          "duplicate.  Under the ring,\n  write vector j always lands at rows "
          "congruent to j (mod W), so whichever map\n  wins stays addressable "
          "and the depth-slice ablation survives.")

    report = {
        "probe": "P0.7 depth-band invariance",
        "when": datetime.now().isoformat(timespec="seconds"),
        "model_name": args.model_name,
        "checkpoint": args.checkpoint,
        "source": source,
        "trained_mean_recurrence": trained_T,
        "criteria": {"mag_floor": MAG_FLOOR, "novelty_floor": NOVELTY_FLOOR,
                     "band_lo": BAND_LO, "margin": MARGIN},
        "geometry": {"D": D, "seq_len": args.seq_len, "batch": args.batch,
                     "dtype": args.dtype, "prefix": not args.no_prefix,
                     "n_slots": args.n_slots},
        "per_T": {str(k): v for k, v in per_T.items()},
        "steps": {str(k): v for k, v in steps_by_T.items()},
        "quantization_suspect": {str(k): v for k, v in suspect_by_T.items()},
        "magnitude_rule": mag,
        "novelty_rule": nov,
    }
    out = args.out or os.path.join(
        "eval_results", f"p07_depth_band-{datetime.now():%Y%m%d-%H%M%S}",
        "results.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with io.open(out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(f"\n  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
