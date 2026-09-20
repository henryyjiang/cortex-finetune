"""
Is `s0` a viable READ SITE at all?  The scale sweep P0.1 never ran.

WHY THIS EXISTS.  P2.1 (job 13295314) put donor-Z and random-Z at 0.0000 on both
dual arms -- 24 numbers, all |.| <= 0.0014 -- while random-E on the same rows
moved 0.66-1.9 nats, and `tests/test_influence_horizon.py` pins the z limb of the
damage path as landing.  So the Z null is a fact about the trained checkpoints,
and it has exactly two explanations:

  A. THE LOOP WASHES s0 OUT.  A recurrent-depth model is trained from fresh
     `trunc_normal_` noise at s0 on every forward -- path independence is the
     property that makes variable-depth inference work at all.  If the loop
     converges to the same place regardless of where it starts, then NOTHING
     written into s0 can matter, at any scale, and Z's read SITE is the defect.
  B. Z IS SIMPLY TOO SMALL.  s0 is a live input, and Z's substituted values sit
     too close to the noise they replace (P0.1: the staggered delta is 0.90x the
     `trunc_normal_` it displaces) for the swap to change anything.

These call for OPPOSITE fixes.  Under A the read has to move inside the loop --
`CortexGraft.read_into` fires every iteration and is bypassed entirely in prefix
mode -- and no amount of renorm helps.  Under B the fix is `latent_renorm=s0`
plus a scale, which costs one config flag and no parameters.  Nothing in the
project distinguishes them, and the difference is a redesign.

WHAT IT MEASURES.  No training.  Prime the chain on a real document, then score
the LAST chunk repeatedly, changing only what is substituted into s0's carried
columns:

  real          the carried Z, i.e. the model as it runs today
  noise @ std   `latent_read_null = ("noise", std, seed)`, swept over orders of
                magnitude around ||s0|| itself

and report every cell as a delta from `real`, in nats/token, with ||s0|| printed
so the stds read as multiples of the thing they replace.

HOW TO READ IT -- fixed here, before the numbers exist:

  FLAT across 3+ orders of magnitude    ->  A.  s0 is not a read site.  The loop
      is path-independent at these columns, so the Z redesign has to move the
      read in-loop.  A renorm cannot rescue it, and neither can a bigger write.
  RESPONDS above some scale             ->  B.  s0 is live and Z is just too
      quiet.  Try `latent_renorm=s0` before building anything.
  RESPONDS at Z's own scale, but the
  DONOR swap (P2.1) still moved nothing ->  s0 is live and Z's CONTENT is what
      is uninformative.  The write is the defect, not the read site.

RED 11, 2026-09-17 -- TWO BUGS, AND THE FIRST ONE FLIPPED THE VERDICT.

  1. UNITS.  The sweep asked for `base * s` where `base` was
     `cortex._z_s0_scale`, a per-token L2 ROW NORM, and handed it to
     `_null_latent` as the PER-ELEMENT std of `normal_(0, std)`.  Those differ
     by sqrt(D) = 45.25 at D=2048, so the row labelled "1x ||s0||" injected a
     row of norm 17.6 -- 45x ||s0||, and 1.8x the loop's own working state.
     `evals/eval_carry_2x2.py:255` had it right (it passes
     `cfg.init_values["std"]`); this file did not.  Fixed: the scale axis now
     means what it says, `s` x the ROW NORM, via `cortex._z_s0_rms`.
  2. THE VERDICT COUNTED CELLS THAT ONLY BREAK THE MODEL.  `span_nats` ran over
     the whole sweep, including injections whose row norm exceeds the carried E
     row (||E|| ~ 171, P0.1 fp32).  Above that the substitution is drowning E
     inside `adapter(cat([x, input_embeds]))` at the same columns -- damage to
     the E channel, not a read of s0.  The verdict is now taken over the
     IN-DISTRIBUTION cells only; the rest are printed, marked, and excluded.

  The loud cells still earn their place: they are the instrument's own proof
  that the knob reaches the model.  A sweep that is flat EVERYWHERE cannot tell
  (A) from a `latent_read_null` that never landed -- the RED 8/9/10 shape -- so
  it now returns INVALID rather than (A).

  Both bugs were live for job 13297293 (a2 + a3z).  Re-read those numbers
  through the corrected axis before quoting them: the verdict there printed as
  (B) off the 10x and 100x cells alone, which were 452x and 4525x ||s0||.

EVAL MODE IS NOT OPTIONAL.  RED 10: the walk and gate 4 run `inner.train()` and
score these checkpoints at 11.4-11.97 nats against ln(vocab) = 11.5157, while
the horizon runs `.eval()` and gets 3.19.  This tool reports its own intact loss
for the same reason the horizon now does -- a table of differences is unreadable
without the level it is a difference of.

USAGE
  python evals/diag_s0_sensitivity.py \
      --model_name ckpts/olmo-retrofit-cortex \
      --checkpoint cortex-retrofit/p1-a2-accum-w16-cc8-z/checkpoint_XXXX \
      --data data/pg19_olmo_val_len4096 --n_chunks 8 --T 8 \
      --set use_memory=true --set prefix_memory=accum --set latent_carry=true \
      --out_dir eval_results/s0_sensitivity-$(date +%Y%m%d)
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import datetime

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model_utils import load_checkpoint, to_num_steps, _unwrap  # noqa: E402
from model_utils import parse_config_overrides  # noqa: E402

#: Swept as multiples of the MEASURED ||s0|| ROW NORM per token, not as absolute
#: stds, so the row labels mean the same thing on any checkpoint width.  The
#: grid spans four orders BELOW the swamp threshold (see `swamp_norm`) plus one
#: cell above it: the cells below decide the verdict, the cell above proves the
#: knob is connected.
DEFAULT_SCALES = (0.0, 0.1, 1.0, 10.0, 100.0, 1000.0)

#: Below this span ACROSS THE IN-DISTRIBUTION CELLS, orders of magnitude of s0
#: have changed nothing and the reading is (A).  Deliberately loose: the
#: question is orders of magnitude, not a CI.
FLAT_SPAN_NATS = 0.01

#: A cell this far from `real` proves `latent_read_null` reaches the model.  If
#: NO cell clears it -- not even the deliberately oversized one -- the sweep is
#: indistinguishable from a disconnected knob and must not return (A).
KNOB_LIVE_NATS = 0.01

#: Fallback swamp threshold, as a multiple of ||s0||, for a checkpoint whose
#: modeling file predates `_e_carried_norm`.  ||E|| / ||s0|| is ~440 on the B2
#: family (171.0 / 0.3885, P0.1 fp32); 100x is a deliberately conservative
#: floor -- it can only make the in-distribution window SMALLER, never admit a
#: cell that is actually swamping E.
SWAMP_FALLBACK_MULT = 100.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("s0 read-site sensitivity")
    p.add_argument("--model_name", required=True)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--data", required=True)
    p.add_argument("--n_chunks", type=int, default=8)
    p.add_argument("--T", type=int, default=None,
                   help="RECURRENCE DEPTH.  config.mean_recurrence is 32 on "
                        "every B2-family checkpoint and no arm ran there; the "
                        "arms trained at 8.  Pass it.")
    p.add_argument("--scales", type=float, nargs="+",
                   default=list(DEFAULT_SCALES),
                   help="noise stds as MULTIPLES of the measured ||s0||")
    p.add_argument("--max_examples", type=int, default=50)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--dtype", default="float32",
                   choices=["float32", "bfloat16", "float16"],
                   help="float32: these are small differences of large states, "
                        "the class bf16 got wrong by 6x in P0.1")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    p.add_argument("--out_dir", default=None)
    return p.parse_args()


def score_chunk(model, ids, labels, carry, num_steps, device, seed):
    """One scored forward, with s0's own RNG pinned so the sweep is paired."""
    torch.manual_seed(seed)
    with torch.no_grad():
        out = model(input_ids=ids.unsqueeze(0).to(device),
                    labels=labels.unsqueeze(0).to(device),
                    m_cross_in=carry, return_m_cross=False,
                    num_steps=num_steps)
    return float(out["loss"] if isinstance(out, dict) else out.loss)


def prime(model, xs, num_steps, device, seed):
    """Run every chunk but the last, returning the carry the last one reads."""
    carry = None
    for i, ids in enumerate(xs[:-1]):
        torch.manual_seed(seed + i)
        with torch.no_grad():
            out = model(input_ids=ids.unsqueeze(0).to(device),
                        m_cross_in=carry, return_m_cross=True,
                        num_steps=num_steps)
        carry = out["m_cross"] if isinstance(out, dict) else out.m_cross
    return carry


def summarize(cells: dict, s0_mean, chance: float, scales,
              row_norms: dict = None, swamp_norm=None) -> dict:
    """Means, deltas and the A-vs-B reading, separated from the run loop so a
    unit test can exercise the verdict without a GPU.

    `row_norms` maps cell name -> the L2 ROW NORM actually injected, and
    `swamp_norm` is the norm of the carried E row it competes with
    (`cortex._e_carried_norm`).  Cells above `swamp_norm` are reported but do
    NOT decide the verdict -- see RED 11 in the module docstring.  With neither
    supplied the window falls back to SWAMP_FALLBACK_MULT x ||s0||.
    """
    mean = {k: sum(v) / len(v) for k, v in cells.items() if v}
    real = mean.get("real")
    span = (max(mean.values()) - min(mean.values())) if mean else 0.0
    delta = ({k: v - real for k, v in mean.items() if k != "real"}
             if real is not None else {})

    limit = swamp_norm if swamp_norm else None
    if not limit and s0_mean:
        limit = SWAMP_FALLBACK_MULT * s0_mean
    ood = sorted(k for k in mean
                 if k != "real" and row_norms and limit
                 and row_norms.get(k) is not None
                 and row_norms[k] > limit)
    kept = [v for k, v in mean.items() if k not in ood]
    span_in = (max(kept) - min(kept)) if kept else 0.0

    # The knob's OWN control, checked BEFORE the verdict: a sweep with nothing
    # in it that moves cannot distinguish path independence from a
    # substitution that never landed.
    knob_live = any(abs(d) > KNOB_LIVE_NATS for d in delta.values())
    at_chance = real is not None and (chance - real) < 1.0
    if at_chance:
        # RED 10: gate 4 ranked three chance-level losses and the ordering set
        # the direction of the whole P2 program.  A difference of two numbers
        # at ln(vocab) still prints a tidy table, so the level has to VETO the
        # verdict and not merely warn above it -- which is what gate 4 now does
        # (`passed: False` on random ids) and what this did not.
        reading = "INVALID_at_chance"
    elif not knob_live:
        reading = "INVALID_knob_unproven"
    elif span_in < FLAT_SPAN_NATS:
        reading = "A_path_independent"
    else:
        reading = "B_s0_live"

    return {
        "s0_norm": s0_mean,
        "swamp_norm": swamp_norm,
        "swamp_limit_used": limit,
        "injected_row_norm": dict(row_norms) if row_norms else {},
        "ood_cells": ood,
        "cells": mean,
        "delta_from_real": delta,
        "span_nats": span,
        "span_in_dist_nats": span_in,
        "knob_live": knob_live,
        "reading": reading,
        "intact": {"mean_nats": real, "chance_nats": chance,
                   "margin_below_chance": (None if real is None
                                           else chance - real),
                   "at_chance": (None if real is None
                                 else bool(chance - real < 1.0))},
    }


def main() -> int:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    overrides = parse_config_overrides(args.set)
    model, cfg = load_checkpoint(args.checkpoint, args.model_name, None,
                                 getattr(torch, args.dtype), device,
                                 config_overrides=overrides or None)
    inner = _unwrap(model)
    cortex = getattr(inner, "cortex", None)
    if cortex is None or getattr(cortex, "prefix", None) is None:
        print("FAILED: no prefix buffer on this checkpoint.  Force the "
              "geometry with --set (see pace/eval_deciding.sbatch).")
        return 2
    if not getattr(cortex, "latent_carry", False):
        print("FAILED: latent_carry is off, so `latent_init` never runs and "
              "there is no s0 substitution to be sensitive to.  This needs a "
              "dual-channel arm (a2 / a3z) and --set latent_carry=true.")
        return 2
    num_steps = to_num_steps(args.T)

    from datasets import load_from_disk
    ds = load_from_disk(args.data)
    n = min(args.max_examples, len(ds)) if args.max_examples else len(ds)

    cells: dict = {"real": []}
    for s in args.scales:
        cells["noise@%g" % s] = []
    s0_norms: list = []
    rms_vals: list = []
    e_norms: list = []
    used = 0

    for si in range(n):
        ids = torch.tensor(ds[si]["input_ids"], dtype=torch.long)
        if ids.numel() < args.n_chunks * 8:
            continue
        x, y = ids[:-1], ids[1:]
        keep = (x.numel() // args.n_chunks) * args.n_chunks
        xs = list(torch.chunk(x[:keep], args.n_chunks))
        ys = list(torch.chunk(y[:keep], args.n_chunks))
        seed = args.seed + si

        cortex.latent_read_null = None
        carry = prime(model, xs, num_steps, device, seed)
        if carry is None:
            continue
        cells["real"].append(
            score_chunk(model, xs[-1], ys[-1], carry, num_steps, device,
                        seed + 977))
        # TWO SCALES, TWO UNITS, AND RED 11 WAS MIXING THEM.  `_z_s0_scale`
        # is the per-token L2 ROW NORM -- what the row labels mean -- while
        # `_null_latent` takes a PER-ELEMENT std, which is `_z_s0_rms`.  They
        # differ by sqrt(D) = 45.25 at D=2048, so passing the first where the
        # second belongs injected 45x the norm every label claimed.
        s0 = getattr(cortex, "_z_s0_scale", None)
        rms = getattr(cortex, "_z_s0_rms", None)
        if rms is None:
            print("FAILED: the graft on this checkpoint does not record "
                  "`_z_s0_rms`, so the only scale available is a ROW NORM and "
                  "feeding it to `_null_latent` is exactly RED 11.  Re-run "
                  "tools/prepare_cortex_checkpoint.py against a cortex_graft.py "
                  "at 2026-09-17 or later.")
            return 2
        if s0:
            s0_norms.append(float(s0))
        rms_vals.append(float(rms))
        e = getattr(cortex, "_e_carried_norm", None)
        if e:
            e_norms.append(float(e))
        base = float(rms)

        for s in args.scales:
            cortex.latent_read_null = ("noise", max(base * s, 0.0), args.seed)
            cells["noise@%g" % s].append(
                score_chunk(model, xs[-1], ys[-1], carry, num_steps, device,
                            seed + 977))
        cortex.latent_read_null = None
        used += 1
        if used % 10 == 0:
            print("  %d samples" % used, flush=True)

    if not used:
        print("FAILED: no usable samples.")
        return 2

    chance = math.log(max(int(getattr(inner.config, "vocab_size", 2)), 2))
    s0_mean = (sum(s0_norms) / len(s0_norms)) if s0_norms else None
    rms_mean = (sum(rms_vals) / len(rms_vals)) if rms_vals else None
    e_mean = (sum(e_norms) / len(e_norms)) if e_norms else None
    # What each cell ACTUALLY put in, in the same unit as ||s0|| and ||E||: an
    # elementwise std of sigma gives a row of norm sigma*sqrt(D).  Reported per
    # cell so the axis can never be silently re-interpreted again.
    width = int(getattr(inner.config, "n_embd", 0) or 0)
    root_d = math.sqrt(width) if width else None
    row_norms = ({("noise@%g" % sc): rms_mean * sc * root_d
                  for sc in args.scales}
                 if (rms_mean is not None and root_d) else None)
    summary = summarize(cells, s0_mean, chance, args.scales,
                        row_norms=row_norms, swamp_norm=e_mean)
    report = {
        "instrument": "s0 read-site sensitivity",
        "when": datetime.now().isoformat(timespec="seconds"),
        "model_name": args.model_name, "checkpoint": args.checkpoint,
        "config": {"n_chunks": args.n_chunks, "T": args.T, "samples": used,
                   "dtype": args.dtype, "scales": args.scales,
                   "config_overrides": overrides},
    }
    report.update(summary)
    mean = summary["cells"]

    print("\n" + "=" * 78)
    print("s0 READ-SITE SENSITIVITY -- what does the loop care about in s0?")
    s0_txt = "unmeasured" if s0_mean is None else ("%.4f" % s0_mean)
    print("n=%d paired samples   ||s0|| = %s (per-token row L2)"
          % (used, s0_txt))
    swamp = summary["swamp_limit_used"]
    if e_mean:
        print("||E carried row|| = %.1f -- the field the substitution shares "
              "the adapter concat" % e_mean)
        print("   with.  A cell above it is DROWNING E rather than being read "
              "at s0, so it")
        print("   is marked (ood) and excluded from the verdict.  See RED 11.")
    elif swamp:
        print("||E carried row|| unmeasured (old graft); the swamp window "
              "falls back to %.1f" % swamp)
    print("intact (real Z) %.4f nats/token   chance %.4f   margin %+.4f"
          % (mean["real"], chance, chance - mean["real"]))
    if summary["intact"]["at_chance"]:
        print("  *** AT CHANCE -- this table is unreadable.  See RED 10.")
    print("=" * 78)
    print("  %-26s%12s%10s%12s"
          % ("substituted into s0", "row norm", "loss", "delta"))
    print("  %-26s%12.3f%10.4f%12.4f"
          % ("real carried Z", (s0_mean or 0.0), mean["real"], 0.0))
    for s in args.scales:
        k = "noise@%g" % s
        if k not in mean:
            continue
        rn = (summary["injected_row_norm"] or {}).get(k)
        tag = "  (ood)" if k in summary["ood_cells"] else ""
        print("  %-26s%12s%10.4f%+12.4f%s"
              % ("noise @ %g x ||s0||" % s,
                 ("%.3f" % rn) if rn is not None else "?",
                 mean[k], mean[k] - mean["real"], tag))
    print("\n  span, in-distribution cells only: %.4f nats"
          % summary["span_in_dist_nats"])
    print("  span across the whole sweep:      %.4f nats"
          % summary["span_nats"])
    if summary["reading"] == "INVALID_at_chance":
        print("  INVALID: the intact loss is at ln(vocab).  Every row of this")
        print("  table is a difference of two chance-level numbers and the")
        print("  ordering means nothing.  See RED 10 -- the loading path, not")
        print("  the carry, is what needs fixing before this run is repeated.")
    elif summary["reading"] == "INVALID_knob_unproven":
        print("  INVALID: NOTHING in this sweep moved the loss, not even the")
        print("  deliberately oversized cell.  That is what a latent_read_null")
        print("  which never reached latent_init looks like, and it is")
        print("  indistinguishable from (A).  Do NOT read it as (A) -- check")
        print("  latent_carry, n_pre, and tests/test_s0_sensitivity.py.")
    elif summary["reading"] == "A_path_independent":
        print("  READS AS (A): the loop is PATH-INDEPENDENT at these columns.")
        print("  Orders of magnitude of s0 change nothing at any scale the")
        print("  model runs at, so s0 is not a read site and no renorm rescues")
        print("  Z -- latent_renorm=s0 lands at exactly 1x, inside the flat")
        print("  region.  The read has to move in-loop (CortexGraft.read_into,")
        print("  bypassed entirely in prefix mode).")
    else:
        print("  READS AS (B): s0 is LIVE -- the loss moves with what is in it")
        print("  at a scale the model actually runs at.  Z is then too quiet")
        print("  rather than mis-sited; try latent_renorm=s0 and a scale before")
        print("  building any new module.")

    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)
        path = os.path.join(args.out_dir, "results.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)
        print("\nwrote %s" % path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
