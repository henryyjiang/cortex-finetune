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

EVAL MODE IS NOT OPTIONAL.  RED 10: the walk and gate 4 run `inner.train()` and
score these checkpoints at 11.4-11.97 nats against ln(vocab) = 11.5157, while
the horizon runs `.eval()` and gets 3.19.  This tool reports its own intact loss
for the same reason the horizon now does -- a table of differences is unreadable
without the level it is a difference of.

USAGE
  python evals/diag_s0_sensitivity.py \
      --model_name ckpts/olmo-retrofit-cortex \
      --checkpoint cortex-retrofit/probe-p1-a2-accum-w16-cc8-z/checkpoint_XXXX \
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

#: Swept as multiples of the MEASURED ||s0|| per token, not as absolute stds, so
#: the row labels mean the same thing on any checkpoint width.
DEFAULT_SCALES = (0.0, 0.1, 1.0, 10.0, 100.0)

#: Below this total span, three orders of magnitude of s0 have changed nothing
#: and the reading is (A).  Deliberately loose: the question is orders of
#: magnitude, not a CI.
FLAT_SPAN_NATS = 0.01


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


def summarize(cells: dict, s0_mean, chance: float, scales) -> dict:
    """Means, deltas and the A-vs-B reading, separated from the run loop so a
    unit test can exercise the verdict without a GPU."""
    mean = {k: sum(v) / len(v) for k, v in cells.items() if v}
    real = mean.get("real")
    span = (max(mean.values()) - min(mean.values())) if mean else 0.0
    return {
        "s0_norm": s0_mean,
        "cells": mean,
        "delta_from_real": {k: v - real for k, v in mean.items()
                            if k != "real"} if real is not None else {},
        "span_nats": span,
        "reading": "A_path_independent" if span < FLAT_SPAN_NATS else "B_s0_live",
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
        # ||s0|| per token, recorded by latent_init on the forward just run.
        s0 = getattr(cortex, "_z_s0_scale", None)
        if s0:
            s0_norms.append(float(s0))
        base = float(s0) if s0 else 1.0

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
    summary = summarize(cells, s0_mean, chance, args.scales)
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
    print("n=%d paired samples   ||s0|| = %s" % (used, s0_txt))
    print("intact (real Z) %.4f nats/token   chance %.4f   margin %+.4f"
          % (mean["real"], chance, chance - mean["real"]))
    if summary["intact"]["at_chance"]:
        print("  *** AT CHANCE -- this table is unreadable.  See RED 10.")
    print("=" * 78)
    print("  %-26s%10s%12s" % ("substituted into s0", "loss", "delta"))
    print("  %-26s%10.4f%12.4f" % ("real carried Z", mean["real"], 0.0))
    for s in args.scales:
        k = "noise@%g" % s
        if k in mean:
            print("  %-26s%10.4f%+12.4f"
                  % ("noise @ %g x ||s0||" % s, mean[k], mean[k] - mean["real"]))
    print("\n  span across the whole sweep: %.4f nats" % summary["span_nats"])
    if summary["reading"] == "A_path_independent":
        print("  READS AS (A): the loop is PATH-INDEPENDENT at these columns.")
        print("  Orders of magnitude of s0 change nothing, so s0 is not a read")
        print("  site and no renorm rescues Z.  The read has to move in-loop")
        print("  (CortexGraft.read_into, bypassed entirely in prefix mode).")
    else:
        print("  READS AS (B): s0 is LIVE -- the loss moves with what is in it.")
        print("  Z is then too quiet rather than mis-sited; try latent_renorm=s0")
        print("  and a scale before building any new module.")

    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)
        path = os.path.join(args.out_dir, "results.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)
        print("\nwrote %s" % path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
