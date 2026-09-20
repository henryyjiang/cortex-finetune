"""
P1.0 pre-registration scorer: did the gate learn anything, and how far does it
reach?

Read `p10_gated_prereg.md` first.  That file is the pre-registration; this file
is the instrument that scores it, and the thresholds live HERE so the two cannot
drift apart.  Run it on a trained gated arm (A3') and it prints a scorecard with
the predictions, the thresholds, the measured values and what each outcome
means — all of which were written down before the arm launched.

WHAT IS PRE-REGISTERED, AND WHY THESE TWO
-----------------------------------------
1. THE TRAINED `fg` SPREAD ACROSS INPUTS.  This is the sharp one, because the
   null is EXACT rather than conventional.  At `gate_init="zero"` both gate
   projections are zero, so at step 0

       fg = sigmoid(forget_bias) = 0.731   for every row, channel and input

   with across-input standard deviation of exactly 0.  A trained gate that is
   still content-blind is an EMA with 16.8M parameters attached, which is a
   clean negative result and a publishable one.  So the measurement is the
   DECOMPOSITION, not the mean:

     across-input spread   fg varies with WHAT is being written/kept -> the gate
                           is doing gating
     within-input spread   fg varies by row and channel but is the same for
                           every input -> a learned static profile.  Still not
                           content-dependent; a per-channel decay constant is an
                           EMA with a shape.

   Reporting only `fg_p10/fg_p90` (which is what diag_gate_geometry prints at
   init) cannot tell those apart: a fixed per-channel bias produces a wide p10/p90
   spread while the gate remains blind.  That is the mistake this script exists
   to not make.

2. THE BABILONG HORIZON THRESHOLDS, which follow from the retention model
   validated on the B2 checkpoint (twice, to ~2%):

       F(d) = ig * fg ** floor(d / (K/W))

   At W=16 / K=64 a lap is 4 chunks of 512 tokens = 2,048 tokens, and with
   ig = 0.5 the requirement "at least 10% of a write still present" gives

       16k context = 32 chunks = 8 laps   ->  fg >= 0.2 ** (1/8)  = 0.81836
       32k context = 64 chunks = 16 laps  ->  fg >= 0.2 ** (1/16) = 0.90430

   The registered bars are 0.818 and 0.905; the second is that derivation
   rounded UP by 0.0007, which is the direction that cannot flatter a result, so
   it is kept rather than corrected.

   At init fg = 0.731 delivers 4.1% and 0.33% — so BOTH fail at init, by
   construction, and the arm has to move fg to pass.

   AND THE PREDICTION IS THAT IT WILL NOT.  A3' trains at cross_chunks 8, so the
   gate never sees content older than 8 chunks and there is NO GRADIENT PRESSURE
   to raise fg at all.  The pre-registered expectation is therefore: A3' shows
   NO CLIFF where accum shows one (that is the mechanism claim, and it does not
   need a long horizon), and it does NOT show a longer horizon at 16k/32k.
   Reading a failed threshold as a failure of the gated design would be reading
   a training-chain fact as an architecture fact.  The horizon claim needs the
   long-chain cell (gated K=128 at cc=16), which is a different run.

USAGE
    python evals/eval_gate_prereg.py \
        --model_name <A3' checkpoint dir> \
        --text_file ../cortex_next_phase_framework.md \
        --chunks 24 --seq_len 512 --steps 8

FLOAT32, like every other probe in this family: fg is a sigmoid of a difference
of two projections of large states, and the quantity being read is its SPREAD.
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

from cortex_memory.buffers import PrefixGatedBuffer  # noqa: E402
from evals.model_utils import _unwrap, load_checkpoint  # noqa: E402
from evals.model_utils import parse_config_overrides  # noqa: E402

# ---------------------------------------------------------------------------
# THE PRE-REGISTRATION.  Written 2026-09-16, before A1/A3' launched.  Changing a
# number here after seeing a result is the failure mode the whole file guards
# against, so each one carries its derivation rather than a citation.
# ---------------------------------------------------------------------------

#: fg at `gate_init="zero"` — sigmoid(forget_bias=1.0).  Exact, not measured.
FG_AT_INIT = 0.7310585786300049
#: ig at init — sigmoid(input_bias=0.0).
IG_AT_INIT = 0.5
#: "a write is still present" floor, as a fraction of one write.  0.10 is a
#: choice; it is fixed here so the two thresholds below are one decision and not
#: two, and every number in the doc is recomputed from it.
RETENTION_FLOOR = 0.10

#: The gate is content-dependent at all.  Null is EXACTLY 0 at init, so 0.02 is
#: a noise band and not a null hypothesis — fp32, ~10^5 samples.
FG_ACROSS_INPUT_STD_MIN = 0.02
#: The gate is content-dependent ENOUGH to act as memory within A3's own chain.
#: A3' at cross_chunks 8 with K/W=4 completes 2 laps, so a keep/drop decision
#: only separates two writes by 2x if (fg_hi/fg_lo)**2 >= 2, i.e. a p10..p90
#: swing of at least 0.41 * fg ~ 0.21 at fg ~ 0.73.  Failing THIS while passing
#: the one above is a real and anticipated outcome: a gate that learned
#: something too small to matter over two laps.
FG_ACROSS_INPUT_SWING_MIN = 0.21


def fg_threshold(context_tokens: int, chunk_len: int, lap_chunks: int,
                 ig: float = IG_AT_INIT,
                 floor: float = RETENTION_FLOOR) -> float:
    """The trained fg a gated ring needs to still hold `floor` of a write after
    `context_tokens`.

    Inverts F(d) = ig * fg ** floor(d / lap).  Returns >1 (i.e. unreachable)
    when ig alone is already below the floor, which is the honest answer rather
    than a clipped one.
    """
    laps = max(1, (context_tokens // chunk_len) // max(lap_chunks, 1))
    if ig <= floor:
        return float("inf")
    return float((floor / ig) ** (1.0 / laps))


#: The two headline cells, in tokens.  BABILong's buckets straddle the accum
#: buffer's 4,096-token cliff, which is why these two and not others.
BABILONG_CELLS = (16_384, 32_768)

#: The thresholds AS WRITTEN DOWN, at W=16 / K=64 / 512-token chunks.  They are
#: what `fg_threshold` derives, to rounding: 0.2**(1/8) = 0.81836 -> 0.818 and
#: 0.2**(1/16) = 0.90430 -> 0.905, where the second is rounded UP by 0.0007.
#: Scoring uses THESE rather than the freshly derived values, for two reasons:
#: they are the registered bar, and rounding up is the direction that cannot
#: flatter the result.  `fg_threshold` stays live so the derivation is
#: checkable and so the follow-up geometries (K=128, cc=16) have one.
PREREGISTERED_FG_MIN = {16_384: 0.818, 32_768: 0.905}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("score the P1.0 gate pre-registration")
    p.add_argument("--model_name", required=True)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--text_file", default=None,
                   help="real prose, tokenized with the checkpoint's own "
                        "tokenizer.  Random ids give the gate nothing to be "
                        "content-dependent ABOUT, so the across-input spread "
                        "would be measured on noise.")
    p.add_argument("--data", default=None, help="a packed dataset (cluster-side)")
    p.add_argument("--chunks", type=int, default=24)
    p.add_argument("--seq_len", type=int, default=512, help="tokens per chunk")
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--steps", type=int, default=8, help="T; use the arm's mr")
    p.add_argument("--dtype", default="float32",
                   choices=["float32", "bfloat16"])
    p.add_argument("--device",
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", default=None)
    p.add_argument("--set", action="append", default=[],
                   metavar="KEY=VALUE",
                   help="force a graft-building config flag, e.g. "
                        "--set use_memory=true --set prefix_memory=gated.  "
                        "REQUIRED on an OVERLAY checkpoint: train.py's "
                        "checkpoint_<step> dirs, and the _w16 branch dirs cut "
                        "from them, hold chkpt.pt and NO config.json, so "
                        "--model_name loads the BASE dir whose config carries "
                        "no cortex flags at all (use_memory is literally "
                        "'<absent>' on ckpts/olmo-retrofit-cortex) and without "
                        "these the graft builds with no buffer.  Mirror the "
                        "arm's PROBE_SETS in pace/p1_arms.sbatch.  RED 12.")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def tape_gates(model, buf, ids_chunks, num_steps, device):
    """Run the chain and record (ig, fg) at every merge the gate actually ran.

    The gate is wrapped on the INSTANCE, so what is taped is the merge the
    forward performed — not a re-derivation of it from the config, which is the
    distinction this repo keeps paying for.  Merges before the ring fills are a
    plain append and never call `gate`, so they contribute nothing and are not
    silently counted as fg = 1.
    """
    igs, fgs = [], []
    real_gate = buf.gate

    def wrapped(state, candidate):
        out, ig, fg = real_gate(state, candidate)
        igs.append(ig.detach().float().cpu())
        fgs.append(fg.detach().float().cpu())
        return out, ig, fg

    had = "gate" in vars(buf)
    buf.gate = wrapped
    try:
        with torch.no_grad():
            m_cross = None
            for xc in ids_chunks:
                out = model(input_ids=xc.to(device), num_steps=num_steps,
                            m_cross_in=m_cross, return_m_cross=True,
                            output_details={"return_logits": False,
                                            "return_latents": False,
                                            "return_head": False,
                                            "return_stats": False})
                m_cross = (out.get("m_cross") if isinstance(out, dict)
                           else getattr(out, "m_cross", None))
    finally:
        if had:
            buf.gate = real_gate
        else:
            del buf.gate
    return igs, fgs


def spread(fgs: list[torch.Tensor]) -> dict:
    """Decompose the variation in fg into across-input and within-input parts.

    `fgs` is one [B, n, D] tensor per merge.  Stack to [M*B, n, D] and treat the
    leading axis as INPUTS (a merge on a batch element: a specific state meeting
    a specific candidate) and (n, D) as the gate's own row/channel geometry.

      across_input_std   std over inputs at fixed (row, channel), then averaged.
                         Zero at init by construction.  THE headline.
      within_input_std   std over (row, channel) at fixed input, then averaged.
                         A learned static per-channel decay profile shows up
                         here and NOT above, which is the distinction a raw
                         p10/p90 cannot draw.
    """
    x = torch.cat(fgs, dim=0)                     # [N, n, D], N = merges x batch
    flat = x.flatten()
    across = x.std(dim=0, unbiased=False)         # [n, D]
    within = x.flatten(1).std(dim=1, unbiased=False)   # [N]
    per_input = x.flatten(1).mean(dim=1)          # [N] the input's own mean fg
    q = torch.tensor([0.10, 0.50, 0.90])
    return {
        "merges_taped": len(fgs),
        "samples": int(x.shape[0]),
        "mean": float(flat.mean()),
        "p10": float(flat.quantile(0.10)),
        "p50": float(flat.quantile(0.50)),
        "p90": float(flat.quantile(0.90)),
        "across_input_std": float(across.mean()),
        "across_input_std_max": float(across.max()),
        "within_input_std": float(within.mean()),
        # The swing an individual (row, channel) shows as the INPUT changes --
        # the quantity the 2x-over-two-laps effect-size bar is stated in.
        "across_input_swing": float(
            (x.quantile(0.90, dim=0) - x.quantile(0.10, dim=0)).mean()),
        "per_input_mean_p10_p50_p90": [float(v)
                                       for v in per_input.quantile(q)],
    }


def tokenize_file(path: str, model_name: str, need: int):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    raw = io.open(path, encoding="utf-8", errors="replace").read()
    ids = tok(raw, return_tensors=None)["input_ids"]
    if len(ids) < need:
        raise SystemExit(
            f"FAILED: {path} tokenizes to {len(ids)} tokens, need {need}.")
    return ids[:need]


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)
    if args.dtype == "bfloat16":
        print("WARNING: bfloat16.  The measured quantity is the SPREAD of a "
              "sigmoid, which is the kind of small difference bf16 got wrong "
              "by 6x in P0.1.  Do not quote these numbers.", flush=True)
    device = torch.device(args.device)
    overrides = parse_config_overrides(args.set)
    model, cfg = load_checkpoint(args.checkpoint, args.model_name, None,
                                 getattr(torch, args.dtype), device,
                                 config_overrides=overrides or None)
    inner = _unwrap(model)
    cortex = getattr(inner, "cortex", None)
    buf = getattr(cortex, "prefix", None) if cortex else None
    if not isinstance(buf, PrefixGatedBuffer):
        print("FAILED: this checkpoint does not carry a gated prefix buffer, so "
              "there is no gate to score.  (An accum arm is A1; the "
              "pre-registration is about A3'.)")
        return 2
    if buf.route != "ring":
        print(f"NOTE: route={buf.route!r}.  The fg thresholds below are derived "
              f"from the RING retention model F(d) = ig * fg**floor(d/(K/W)); a "
              f"dense route decays every chunk and needs its own derivation.")

    geo = buf.geometry(args.seq_len)
    lap = int(geo["lap_chunks"])
    need = args.batch * args.chunks * args.seq_len

    if args.data:
        from datasets import load_from_disk
        ds = load_from_disk(args.data)
        rows, want = [], args.chunks * args.seq_len
        for i in range(args.batch):
            r = ds[i]["input_ids"][:want]
            if len(r) < want:
                print(f"FAILED: row {i} of {args.data} has {len(r)} tokens, "
                      f"need {want}.")
                return 2
            rows.append(r)
        flat = torch.tensor(rows, dtype=torch.long)
        source = args.data
    elif args.text_file:
        ids = tokenize_file(args.text_file, args.model_name, need)
        flat = torch.tensor(ids, dtype=torch.long).view(args.batch, -1)
        source = args.text_file
    else:
        print("FAILED: pass --text_file or --data.  Random ids would measure "
              "the across-input spread of a gate reading noise, which is not "
              "the pre-registered quantity.")
        return 2

    chunks = [flat[:, i * args.seq_len:(i + 1) * args.seq_len].contiguous()
              for i in range(args.chunks)]
    num_steps = torch.tensor([args.steps, 0])

    print(f"\n=== P1.0 gate pre-registration | "
          f"{os.path.basename(args.model_name)} | {source} ===")
    print("  " + "  ".join(f"{k}={v}" for k, v in geo.items()))
    print(f"  {args.chunks} chunks x {args.seq_len} tok x batch {args.batch}, "
          f"T={args.steps}, dtype={args.dtype}")
    if args.chunks < 2 * lap:
        print(f"\nFAILED: {args.chunks} chunks is under two laps ({2 * lap}), "
              f"so the gate never fires and there is nothing to score.")
        return 2

    igs, fgs = tape_gates(model, buf, chunks, num_steps, device)
    if not fgs:
        print("\nFAILED: the gate never ran.  Either the chain is shorter than "
              "one lap or `merge` took the append branch throughout -- in "
              "which case this arm trained an accum buffer with dead gate "
              "parameters, which is the outcome train.py's cross_chunks assert "
              "exists to prevent.")
        return 2

    fg = spread(fgs)
    ig = spread(igs)

    # -- prediction 1: is the gate content-dependent at all? ---------------
    p1 = fg["across_input_std"] >= FG_ACROSS_INPUT_STD_MIN
    p1b = fg["across_input_swing"] >= FG_ACROSS_INPUT_SWING_MIN

    # -- prediction 2: the horizon thresholds ------------------------------
    cells = []
    for ctx in BABILONG_CELLS:
        # The registered bar governs PASS/FAIL.  The derived one is reported
        # beside it because the geometry may not be the one the bars were
        # written for -- a K=128 arm has an 8-chunk lap and a genuinely easier
        # threshold, and silently scoring it against the K=64 number would make
        # the follow-up cell look better than it is.
        derived = fg_threshold(ctx, args.seq_len, lap, ig=ig["mean"])
        registered = PREREGISTERED_FG_MIN.get(ctx, derived)
        laps = max(1, (ctx // args.seq_len) // max(lap, 1))
        cells.append({
            "context_tokens": ctx,
            "laps": laps,
            "fg_required": registered,
            "fg_required_derived_here": derived,
            "geometry_matches_registration": (
                args.seq_len == 512 and lap == 4),
            "fg_measured": fg["mean"],
            "retained_fraction": ig["mean"] * fg["mean"] ** laps,
            "retained_at_init": IG_AT_INIT * FG_AT_INIT ** laps,
            "passes": fg["mean"] >= registered,
        })

    report = {
        "probe": "P1.0 gate pre-registration",
        "when": datetime.now().isoformat(timespec="seconds"),
        "model_name": args.model_name,
        "checkpoint": args.checkpoint,
        "source": source,
        "geometry": geo,
        "dtype": args.dtype,
        "thresholds": {
            "fg_across_input_std_min": FG_ACROSS_INPUT_STD_MIN,
            "fg_across_input_swing_min": FG_ACROSS_INPUT_SWING_MIN,
            "retention_floor": RETENTION_FLOOR,
            "fg_at_init": FG_AT_INIT,
        },
        "fg": fg,
        "ig": ig,
        "forget_bias": float(buf.forget_bias.detach().float().mean()),
        "input_bias": float(buf.input_bias.detach().float().mean()),
        "prediction_1_content_dependent": bool(p1),
        "prediction_1b_effect_size": bool(p1b),
        "babilong_cells": cells,
    }

    print(f"\n-- the gate, trained --")
    print(f"  forget_bias {report['forget_bias']:+.4f} (init +1.0)   "
          f"input_bias {report['input_bias']:+.4f} (init 0.0)")
    print(f"  fg  mean {fg['mean']:.4f}  p10 {fg['p10']:.4f}  "
          f"p90 {fg['p90']:.4f}   (init: {FG_AT_INIT:.4f} everywhere)")
    print(f"  ig  mean {ig['mean']:.4f}                          "
          f"(init: {IG_AT_INIT:.4f} everywhere)")
    print(f"  taped {fg['merges_taped']} merges, {fg['samples']} gated rows")

    print("\n" + "=" * 78)
    print("PRE-REGISTERED PREDICTION 1 — did the gate learn to gate?")
    print("=" * 78)
    print(f"  across-input std  {fg['across_input_std']:.4f}   "
          f"threshold >= {FG_ACROSS_INPUT_STD_MIN}   "
          f"{'PASS' if p1 else 'FAIL'}")
    print(f"  within-input std  {fg['within_input_std']:.4f}   "
          f"(a learned STATIC profile lands here, not above)")
    print(f"  across-input swing (p90-p10 per row/channel) "
          f"{fg['across_input_swing']:.4f}   "
          f"threshold >= {FG_ACROSS_INPUT_SWING_MIN}   "
          f"{'PASS' if p1b else 'FAIL'}")
    if p1 and p1b:
        print("  => the gate is content-dependent AND the swing is large enough "
              "to separate\n     two writes by 2x over A3's own two laps.")
    elif p1:
        print("  => content-dependent, but the swing is too small to matter "
              "over two laps.\n     ANTICIPATED, and not a rescue: it is the "
              "outcome the short chain predicts.\n     The follow-up is the "
              "long-chain cell (gated K=128 at cc=16), not a re-analysis.")
    else:
        print("  => CLEAN NEGATIVE.  The gate is an EMA with "
              f"{geo['params'] / 1e6:.1f}M extra parameters.\n"
              "     Report it as such; it is a result, not a bug.")

    print("\n" + "=" * 78)
    print("PRE-REGISTERED PREDICTION 2 — the BABILong horizon thresholds")
    print("=" * 78)
    print(f"  lap = {lap} chunks = {lap * args.seq_len} tokens; "
          f"F(d) = ig * fg**floor(d/lap), floor = {RETENTION_FLOOR}")
    print(f"  {'context':>9}{'laps':>6}{'fg bar':>9}{'derived':>9}{'fg got':>9}"
          f"{'retained':>10}{'at init':>9}   verdict")
    for c in cells:
        print(f"  {c['context_tokens']:>9,}{c['laps']:>6}"
              f"{c['fg_required']:>9.4f}{c['fg_required_derived_here']:>9.4f}"
              f"{c['fg_measured']:>9.4f}"
              f"{c['retained_fraction']:>10.4f}{c['retained_at_init']:>9.4f}"
              f"   {'PASS' if c['passes'] else 'FAIL'}")
    if not all(c["geometry_matches_registration"] for c in cells):
        print("\n  !! THIS GEOMETRY IS NOT THE ONE THE BARS WERE REGISTERED "
              "FOR (512-tok chunks,\n     K/W = 4).  `fg bar` is still the "
              "registered number and `derived` is what\n     THIS geometry "
              "actually needs -- score against `derived` and say so.")
    print("\n  THE PRE-REGISTERED EXPECTATION IS FAIL ON BOTH ROWS, and the "
          "reason was\n  written down before the run: A3' trains at "
          "cross_chunks 8, so the gate never\n  sees content older than 8 "
          "chunks and there is no gradient pressure to raise\n  fg.  This cell "
          "demonstrates NO CLIFF (which is the mechanism claim, and does\n"
          "  not need a horizon); it does not demonstrate a LONGER horizon.  "
          "Reading a\n  FAIL here as a failure of the gated design would be "
          "reading a training-chain\n  fact as an architecture fact.  The "
          "horizon claim lives in gated K=128 at cc=16.")

    out = args.out or os.path.join(
        "eval_results", f"p10_gate_prereg-{datetime.now():%Y%m%d-%H%M%S}",
        "results.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with io.open(out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(f"\n  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
