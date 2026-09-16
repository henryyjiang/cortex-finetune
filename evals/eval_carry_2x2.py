"""
The 2x2 carry ablation: E on/off x Z on/off, at MATCHED COLUMN COUNT.

WHY A 2x2 AND NOT A NUMBER.  The single carry delta answers "is anything
carried".  The dual-channel design needs three different answers, and only a
factorial design separates them:

  E main effect   does the token-space carry (post-ln_f summary states, what B2
                  has) help?
  Z main effect   does the latent-space carry (the recurrence's own trajectory,
                  substituted into s0) help?
  INTERACTION     are they COMPLEMENTS or SUBSTITUTES?  This is the actual
                  hypothesis.  The existing memory-vs-depth trade treats them as
                  substitutes (carry-on T=8 beats carry-off T=32 by 3.6x); the
                  dual-channel claim is that they are complements, and a 2x2 is
                  the only design that can show it.

MATCHED COLUMN COUNT IS THE WHOLE DESIGN, NOT A DETAIL.  If "E off" DROPS the
carried columns instead of emptying them, the packed sequence gets shorter, the
model's own initialize_state draws a different s0 for the real tokens, and the
positions shift -- so "no information" is confounded with "shorter sequence" and
the measured effect is partly neither channel.  Every cell here splices the same
number of columns at the same positions; only the CONTENTS change.

  E null = ZEROS.  Imperfect, and say so: a zero row is still a key, scoring a
           mid-range logit rather than -inf, and it goes on absorbing ~3-5% of
           the softmax mass.  There is no better null for a channel the model
           reads as input embeddings.
  Z null = NOISE at s0's own scale.  STRICTLY BETTER than E's, and this is a
           real asymmetry worth stating in the writeup: the latent field's
           trained default IS fresh trunc_normal_ noise -- initialize_state
           writes exactly that into those columns today -- so the null is
           perfectly in-distribution, identical in column count AND in
           distribution.  Real-state-vs-noise has no sink ambiguity at all.

SCOPE, HONESTLY.  The Z axis needs a model that writes a latent channel.  Until
that exists this tool runs the E axis and REFUSES the Z rows rather than
silently reporting a 1x2 dressed up as a 2x2.  The matched-column machinery --
the part that is easy to get wrong -- is live either way, and it is what an
E-only accum-vs-gated comparison needs today.

READ-BLOCK LENGTH DIFFERS BY BUFFER, INHERENTLY.  An append buffer's read block
grows (32 -> 256 columns) while a gated one is fixed at K.  That is the
mechanism, not a confound to remove, so matched columns hold WITHIN an arm's
2x2 and NOT across an accum arm and a gated arm.  Comparing those two needs the
influence horizon (evals/eval_influence_horizon.py), not this.

USAGE
  python evals/eval_carry_2x2.py --model_name <ckpt> \
      --data data/pg19_olmo_val_len4096 --n_chunks 8 --T 8 \
      --out_dir eval_results/carry_2x2-$(date +%Y%m%d)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model_utils import load_checkpoint, to_num_steps, _unwrap  # noqa: E402

CELLS = (("E1Z1", True, True), ("E1Z0", True, False),
         ("E0Z1", False, True), ("E0Z0", False, False))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", required=True)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--data", required=True)
    p.add_argument("--n_chunks", type=int, default=8,
                   help="must match the arm's cross_chunks")
    p.add_argument("--T", type=int, default=None,
                   help="recurrence depth; pass 8 on the mr8 arms, whose "
                        "config says 32")
    p.add_argument("--max_examples", type=int, default=100)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--dtype", default="float32",
                   choices=["float32", "bfloat16"])
    p.add_argument("--boot", type=int, default=2000)
    p.add_argument("--allow_missing_z", action="store_true",
                   help="run the E axis alone and report the Z rows as "
                        "unavailable, instead of failing.")
    p.add_argument("--out_dir", default="eval_results/carry_2x2")
    return p.parse_args()


def has_latent_channel(cortex) -> bool:
    """Does this model carry a latent (Z) channel as well as a token (E) one?

    The dual-channel design writes the recurrence's trajectory at the summary
    columns and substitutes it into s0 on the next chunk.  Duck-typed on the
    hook the graft will expose, so this tool does not have to be edited when the
    channel lands -- but it must never GUESS: reporting a 2x2 when only E exists
    would turn a missing channel into a null result.
    """
    buf = getattr(cortex, "prefix", None)
    return bool(getattr(cortex, "latent_carry", False)
                and buf is not None
                and getattr(buf, "carries_latent", False))


def null_e(state: torch.Tensor) -> torch.Tensor:
    """E's null: same columns, same positions, zero contents.

    See the header for why this is the imperfect one.
    """
    return torch.zeros_like(state)


def null_z(state: torch.Tensor, std: float, seed: int) -> torch.Tensor:
    """Z's null: fresh noise at s0's own scale.

    This is the model's TRAINED DEFAULT for those columns -- initialize_state
    writes trunc_normal_(std) there on every chunk today -- so the null is
    in-distribution rather than merely length-matched.
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    n = torch.empty(state.shape, dtype=torch.float32).normal_(0.0, std,
                                                              generator=g)
    return n.to(device=state.device, dtype=state.dtype)


def chain_nll(model, cortex, xs, ys, ms, num_steps, device, seed,
              e_on: bool, z_on: bool, s0_std: float):
    """Mean NLL over chunks 2..N for one cell of the 2x2.

    Chunk 1 is excluded from the endpoint (no incoming carry either way, so
    every cell is identical there) but still RUN, because it is what produces
    the carry the later chunks read -- and it doubles as a sanity check: the
    four cells must agree on chunk 1 to floating-point noise.
    """
    torch.manual_seed(seed)
    state, tot, ntok, first = None, 0.0, 0, None
    for i, (xc, yc, mc) in enumerate(zip(xs, ys, ms)):
        m_in = state
        if m_in is not None:
            if not e_on:
                m_in = null_e(m_in)
            # Z lives in a separate field; when the channel exists the graft
            # takes it from the same slot columns.  Nulling it is a per-chunk
            # substitution, handled by the graft hook rather than here.
        cortex.latent_read_null = (None if z_on
                                   else ("noise", s0_std, seed + i))
        out = model(input_ids=xc.unsqueeze(0).to(device),
                    num_steps=num_steps, m_cross_in=m_in,
                    return_m_cross=True)
        state = (out.get("m_cross") if isinstance(out, dict)
                 else getattr(out, "m_cross", None))
        n = int(mc.sum())
        if n == 0:
            continue
        logits = (out["logits"] if isinstance(out, dict) else out.logits)[0].float()
        ce = F.cross_entropy(logits, yc.to(device), reduction="none")
        loss = float((ce * mc.to(device)).sum() / n)
        if i == 0:
            first = loss
            continue
        tot += loss * n
        ntok += n
    cortex.latent_read_null = None
    return (tot / ntok if ntok else None), first


def paired_ci(deltas, n_boot: int, seed: int = 0):
    if not deltas:
        return (float("nan"),) * 3
    t = torch.tensor(deltas, dtype=torch.float64)
    g = torch.Generator().manual_seed(seed)
    idx = torch.randint(len(t), (n_boot, len(t)), generator=g)
    means = t[idx].mean(dim=1)
    lo, hi = torch.quantile(means, torch.tensor([0.025, 0.975],
                                                dtype=torch.float64))
    return float(t.mean()), float(lo), float(hi)


def main() -> int:
    args = parse_args()
    if args.dtype == "bfloat16":
        print("WARNING: bfloat16.  Every number below is a difference of two "
              "nearly equal losses -- the quantity bf16 got wrong by 6x in "
              "P0.1.  Do not quote these.", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg = load_checkpoint(args.checkpoint, args.model_name, None,
                                 getattr(torch, args.dtype), device)
    inner = _unwrap(model)
    cortex = getattr(inner, "cortex", None)
    if cortex is None or getattr(cortex, "prefix", None) is None:
        print("FAILED: this checkpoint has no prefix buffer.")
        return 2

    z_live = has_latent_channel(cortex)
    if not z_live and not args.allow_missing_z:
        print(
            "FAILED: this model has no latent (Z) channel, so a 2x2 is not\n"
            "measurable -- only the E axis exists.  Re-run with\n"
            "  --allow_missing_z\n"
            "to get the E-on/E-off contrast at matched column count, which is\n"
            "what an accum-vs-gated comparison needs today.  Refusing by\n"
            "default so that a missing channel is never reported as a null\n"
            "result for Z.")
        return 3

    cells = CELLS if z_live else (("E1Z1", True, True), ("E0Z1", False, True))
    # s0's own scale, for Z's null.  The graft exposes the trunc_normal_ std the
    # model uses; falling back to the analytic value keeps this runnable on a
    # config that does not carry it.
    s0_std = float(getattr(cfg, "init_values", {}).get("std", 0.02)
                   if isinstance(getattr(cfg, "init_values", None), dict)
                   else 0.02)
    num_steps = to_num_steps(args.T)

    from datasets import load_from_disk
    ds = load_from_disk(args.data)
    n = len(ds) if args.max_examples == 0 else min(args.max_examples, len(ds))

    per_cell: dict[str, list[float]] = {c[0]: [] for c in cells}
    chunk1_spread, n_used = [], 0
    for si in range(n):
        ids = torch.tensor(ds[si]["input_ids"], dtype=torch.long)
        if ids.numel() < args.n_chunks * 8:
            continue
        x, y = ids[:-1], ids[1:]
        keep = (x.numel() // args.n_chunks) * args.n_chunks
        x, y = x[:keep], y[:keep]
        mask = torch.ones_like(y, dtype=torch.float32)
        xs = list(torch.chunk(x, args.n_chunks))
        ys = list(torch.chunk(y, args.n_chunks))
        ms = list(torch.chunk(mask, args.n_chunks))

        firsts, ok = [], True
        for name, e_on, z_on in cells:
            nll, first = chain_nll(model, cortex, xs, ys, ms, num_steps, device,
                                   args.seed + si, e_on, z_on, s0_std)
            if nll is None:
                ok = False
                break
            per_cell[name].append(nll)
            firsts.append(first)
        if not ok:
            for name, _, _ in cells:
                per_cell[name] = per_cell[name][:n_used]
            continue
        chunk1_spread.append(max(firsts) - min(firsts))
        n_used += 1
        if n_used % 10 == 0:
            print(f"  {n_used} samples", flush=True)

    report = {
        "instrument": "2x2 carry ablation (matched column count)",
        "when": datetime.now().isoformat(timespec="seconds"),
        "model_name": args.model_name,
        "z_channel": z_live,
        "config": {"n_chunks": args.n_chunks, "T": args.T,
                   "dtype": args.dtype, "samples": n_used,
                   "s0_std": s0_std},
        "cells": {}, "effects": {},
        "chunk1_max_spread": (max(chunk1_spread) if chunk1_spread else None),
    }
    for name, _, _ in cells:
        v = per_cell[name]
        report["cells"][name] = {
            "mean_nll": (sum(v) / len(v) if v else None), "n": len(v)}

    def effect(a, b, label, note):
        """Paired mean of (a - b) across samples."""
        if a not in per_cell or b not in per_cell:
            return
        d = [x - y for x, y in zip(per_cell[a], per_cell[b])]
        m, lo, hi = paired_ci(d, args.boot, args.seed)
        report["effects"][label] = {"mean_nats": m, "ci_lo": lo, "ci_hi": hi,
                                    "n": len(d), "note": note}

    effect("E0Z1", "E1Z1", "E_main",
           "NLL without E minus with E, Z held on.  Positive = E helps.")
    if z_live:
        effect("E1Z0", "E1Z1", "Z_main",
               "NLL without Z minus with Z, E held on.  Positive = Z helps.")
        effect("E0Z1", "E0Z0", "Z_alone",
               "Does Z work with no E?  The rung a non-recurrent baseline "
               "cannot compete on by construction.")
        d = [(a - b) - (c - e) for a, b, c, e in
             zip(per_cell["E0Z0"], per_cell["E1Z0"],
                 per_cell["E0Z1"], per_cell["E1Z1"])]
        m, lo, hi = paired_ci(d, args.boot, args.seed)
        report["effects"]["interaction"] = {
            "mean_nats": m, "ci_lo": lo, "ci_hi": hi, "n": len(d),
            "note": "E's effect without Z minus E's effect with Z.  NEGATIVE "
                    "= complements (each is worth MORE when the other is "
                    "present); positive = substitutes."}

    print(f"\n{'=' * 78}")
    print(f"2x2 carry ablation -- mean NLL over chunks 2..{args.n_chunks}, "
          f"n={n_used} paired samples")
    print(f"column count is IDENTICAL in every cell; only the contents change")
    print("=" * 78)
    for name, e_on, z_on in cells:
        c = report["cells"][name]
        print(f"  {name}  E={'on ' if e_on else 'off'}  "
              f"Z={'on ' if z_on else 'off'}   "
              f"NLL {c['mean_nll']:.5f}" if c["mean_nll"] is not None else
              f"  {name}  (no data)")
    if report["chunk1_max_spread"] is not None:
        print(f"\n  chunk-1 sanity: max spread across cells "
              f"{report['chunk1_max_spread']:.2e} "
              f"(should be ~0 -- no cell has an incoming carry there)")
    print(f"\n{'-' * 78}")
    for label, e in report["effects"].items():
        sig = "" if (e["ci_lo"] <= 0 <= e["ci_hi"]) else "  *"
        print(f"  {label:<12}{e['mean_nats']:>10.5f}  "
              f"[{e['ci_lo']:>9.5f}, {e['ci_hi']:>9.5f}]{sig}")
        print(f"               {e['note']}")
    if not z_live:
        print(f"\n  Z axis UNAVAILABLE: this model carries no latent channel, "
              f"so\n  Z_main, Z_alone and the interaction are not reported.  "
              f"The E axis\n  above is a genuine matched-column contrast; the "
              f"2x2 is not.")

    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, "results.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
