"""
Context ceiling — how much is ANY carry worth on this data, and what fraction
of it does the compressed carry recover?

Every carry number the project has (+0.024 Track A, -0.013 B1, ~0 for the ci /
ki4 arms) is quoted against zero, so a delta of +0.024 nats has never had a
scale.  AutoCompressor always reports all three points: Llama-2 ppl 5.40 with no
context, 5.07 with their compressed summaries, 4.76 with REAL full attention over
the same span.  The meaningful quantity is the fraction of 5.40->4.76 that the
compression recovers.  This measures our version of that.

For each chunk g >= 2 of the same 4x1024 PG-19 rows the carry ablation uses,
score chunk g's tokens with k real PRECEDING tokens prepended, for k in
--context_lens, plus (when the model has one) the model's own compressed carry:

  k=0        the floor: chunk g alone, no context at all
  k=128..    real tokens, natural contiguous positions -- an ORACLE, since the
             model gets the actual text the carry is trying to summarise
  carry      the true chained state from chunks 1..g-1 (== the carry ablation's
             "carried" condition)

Headline outputs
  ceiling(k)       = L(k=0) - L(k real tokens)      how much context is worth
  carry_delta      = L(k=0) - L(carry)              what we actually recover
  carry_equiv_tok  the k whose ceiling matches carry_delta, by interpolation --
                   "our 32-vector carry is worth about N tokens of real context"

Read it two ways.  If the ceiling is large and the carry recovers a few percent,
the mechanism is the problem.  If the ceiling is itself ~0.02-0.03 nats, then
Track A's +0.024 was already near it and the line is a characterised negative
with a MEASURED bound rather than an open question.  The project's own
teacher-advantage probe measured -0.98 nats at W=2048 on this base (McLeish
continued-pretrained it at max_length=1024), so a ceiling that goes NEGATIVE at
large k is an expected and reportable outcome, not a bug -- it is the concrete
statement of "the base cannot use a long window", which is the motivation for
chunking in the first place.

Needs no memory: run it on the BASE model for the pure ceiling, or on an arm to
get the ceiling and the fraction recovered in one job.

Usage:
    python evals/eval_context_ceiling.py --model_name ckpts/olmo8-cortex \\
        --data data/pg19_olmo_val_len4096 --T 8
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from model_utils import load_checkpoint, has_cross_state, to_num_steps

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Context ceiling / oracle sweep")
    p.add_argument("--checkpoint",   type=str, default=None)
    p.add_argument("--model_name",   required=True)
    p.add_argument("--memory_slots", type=int, default=None)
    p.add_argument("--T",            type=int, default=None,
                   help="Recurrence depth (None = config mean_recurrence; set it "
                        "explicitly, retrofit-derived configs inherit 32)")
    p.add_argument("--data",         required=True)
    p.add_argument("--n_chunks",     type=int, default=4,
                   help="Sub-windows per sample (match training cross_chunks)")
    p.add_argument("--context_lens", type=int, nargs="+",
                   default=[128, 256, 512, 1024],
                   help="real preceding tokens to prepend (k=0 is always run)")
    p.add_argument("--max_examples", type=int, default=50, help="0 = all rows")
    p.add_argument("--seed",         type=int, default=1234)
    p.add_argument("--out_dir",      default="eval_results/context_ceiling")
    p.add_argument("--dtype",        default="bfloat16", choices=["float32", "bfloat16"])
    p.add_argument("--device",       default=None)
    return p.parse_args()


@torch.no_grad()
def _score(model, ctx, xc, yc, mc, state, num_steps, device, seed):
    """NLL of chunk `xc` given `ctx` real preceding tokens and/or carry `state`.

    The context is prepended as ordinary tokens with natural contiguous
    positions -- this is the oracle condition, so nothing about it should be
    special-cased.  Only the chunk's own positions are scored.
    """
    torch.manual_seed(seed)                      # identical s0 across conditions
    n_tok = int(mc.sum())
    if n_tok == 0:
        return None
    ids = xc if ctx is None or ctx.numel() == 0 else torch.cat([ctx, xc])
    off = 0 if ctx is None else ctx.numel()
    out = model(input_ids=ids.unsqueeze(0).to(device), num_steps=num_steps,
                m_cross_in=state, return_m_cross=False)
    logits = out["logits"][0, off:].float()
    ce = F.cross_entropy(logits, yc.to(device), reduction="none")
    return float((ce * mc.to(device)).sum() / n_tok)


@torch.no_grad()
def carry_before(model, x, n_chunks, g, num_steps, seed, device):
    """The true chained state after chunks 1..g-1 (None when the model has no
    cross state).  Same construction as the carry ablation's carried condition."""
    torch.manual_seed(seed)
    m_cross = None
    for gi, xc in enumerate(torch.chunk(x, n_chunks)):
        if gi >= g:
            break
        out = model(input_ids=xc.unsqueeze(0).to(device), num_steps=num_steps,
                    m_cross_in=m_cross, return_m_cross=True)
        m_cross = out.get("m_cross")
    return m_cross


def _paired(a, b):
    d = torch.tensor(a) - torch.tensor(b)
    se = float(d.std() / max(len(d), 1) ** 0.5) if len(d) > 1 else 0.0
    return float(d.mean()), se


def _equiv_tokens(ceiling, carry):
    """Interpolate the k whose oracle ceiling equals the carry delta.
    ceiling: list of (k, delta) sorted by k.  Returns None when the carry is
    outside the measured range (below zero, or above the largest ceiling)."""
    if carry <= 0 or not ceiling:
        return None
    prev_k, prev_d = 0, 0.0
    for k, d in ceiling:
        if d >= carry:
            if d == prev_d:
                return float(k)
            return prev_k + (carry - prev_d) * (k - prev_k) / (d - prev_d)
        prev_k, prev_d = k, d
    return None                                   # carry exceeds every ceiling


def main() -> None:
    args = parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32

    print(f"Loading: {args.model_name}  (overlay: {args.checkpoint})")
    model, cfg = load_checkpoint(args.checkpoint, args.model_name,
                                 args.memory_slots, dtype, device)
    with_carry = has_cross_state(model)
    num_steps = to_num_steps(args.T if args.T is not None else int(cfg.mean_recurrence))
    print(f"T={int(num_steps[0])}  n_chunks={args.n_chunks}  "
          f"carry condition: {'on' if with_carry else 'OFF (no cross state)'}")
    print(f"context lens: 0 + {args.context_lens}")

    from datasets import load_from_disk
    ds = load_from_disk(args.data)
    n = len(ds) if args.max_examples == 0 else min(args.max_examples, len(ds))
    print(f"{args.data}: {len(ds)} rows, evaluating {n}")

    conds = [0] + sorted(args.context_lens)
    # per_chunk[cond][g] = per-sample losses; g indexes chunks 2..n_chunks
    per_chunk = {c: [[] for _ in range(args.n_chunks)] for c in conds}
    carry_ch = [[] for _ in range(args.n_chunks)]
    t_start = time.time()

    for si in range(n):
        row = ds[si]
        ids = torch.tensor(row["input_ids"], dtype=torch.long)
        mask = torch.tensor(row["attention_mask"], dtype=torch.float)
        x, y, ym = ids[:-1], ids[1:], mask[1:]
        seed = args.seed + si
        x_ch = torch.chunk(x, args.n_chunks)
        y_ch = torch.chunk(y, args.n_chunks)
        m_ch = torch.chunk(ym, args.n_chunks)
        off = 0
        for g in range(args.n_chunks):
            off += 0 if g == 0 else x_ch[g - 1].numel()
            if g == 0:
                continue                          # no preceding context to give
            vals = {}
            for k in conds:
                ctx = x[max(0, off - k):off] if k else None
                vals[k] = _score(model, ctx, x_ch[g], y_ch[g], m_ch[g], None,
                                 num_steps, device, seed)
            cv = None
            if with_carry:
                st = carry_before(model, x, args.n_chunks, g, num_steps, seed, device)
                cv = _score(model, None, x_ch[g], y_ch[g], m_ch[g], st,
                            num_steps, device, seed)
            if all(v is not None for v in vals.values()) and (cv is not None or not with_carry):
                for k in conds:
                    per_chunk[k][g].append(vals[k])
                if with_carry:
                    carry_ch[g].append(cv)
        if si == 0:
            dt = time.time() - t_start
            print(f"  sample 1 took {dt:.1f}s -> ETA {dt * n / 60:.0f} min")
        elif (si + 1) % 10 == 0:
            print(f"  {si + 1}/{n} samples  [{(time.time() - t_start) / 60:.0f} min]")

    # ---- per-chunk table ---------------------------------------------------
    results = {}
    hdr = f"  {'Chunk':<7} {'N':>4}" + "".join(f" {('k=' + str(k)):>9}" for k in conds)
    if with_carry:
        hdr += f" {'carry':>9}"
    print("\n" + hdr)
    print("  " + "-" * (len(hdr) - 2))
    for g in range(1, args.n_chunks):
        if not per_chunk[0][g]:
            continue
        line = f"  {g + 1:<7} {len(per_chunk[0][g]):>4}"
        entry = {"n": len(per_chunk[0][g])}
        for k in conds:
            v = float(torch.tensor(per_chunk[k][g]).mean())
            line += f" {v:>9.4f}"
            entry[f"k{k}"] = v
        if with_carry:
            v = float(torch.tensor(carry_ch[g]).mean())
            line += f" {v:>9.4f}"
            entry["carry"] = v
        results[f"chunk{g + 1}"] = entry
        print(line)

    # ---- pooled chunks 2+ --------------------------------------------------
    base = [v for g in range(1, args.n_chunks) for v in per_chunk[0][g]]
    print(f"\n  chunks 2+ pooled (n={len(base)}), delta vs k=0 "
          f"(positive = that context HELPS):")
    ceiling = []
    for k in conds[1:]:
        vals = [v for g in range(1, args.n_chunks) for v in per_chunk[k][g]]
        d, se = _paired(base, vals)
        ceiling.append((k, d))
        results[f"ceiling_k{k}"] = {"delta": d, "se": se}
        print(f"    {k:>5} real tokens : {d:+.5f}  (SE {se:.5f})")

    if with_carry:
        vals = [v for g in range(1, args.n_chunks) for v in carry_ch[g]]
        d, se = _paired(base, vals)
        results["carry_delta"] = {"delta": d, "se": se}
        print(f"    {'carry':>5}             : {d:+.5f}  (SE {se:.5f})")
        eq = _equiv_tokens(ceiling, d)
        best_k, best_d = max(ceiling, key=lambda kd: kd[1]) if ceiling else (0, 0.0)
        results["carry_equiv_tokens"] = eq
        results["best_ceiling"] = {"k": best_k, "delta": best_d}
        print()
        if eq is not None:
            print(f"  => the compressed carry is worth about {eq:.0f} tokens of REAL context")
        elif d <= 0:
            print("  => the carry is worth LESS than no context at all; no equivalent exists")
        else:
            print(f"  => the carry BEATS every measured oracle (best: {best_d:+.5f} "
                  f"at k={best_k}); the ceiling is not the binding constraint")
        if best_d > 0:
            print(f"  => it recovers {100 * d / best_d:.1f}% of the best oracle "
                  f"({best_d:+.5f} at k={best_k} real tokens)")
        else:
            print(f"  => the ORACLE ITSELF is non-positive (best {best_d:+.5f} at "
                  f"k={best_k}): real context over this span does not help this "
                  f"model, so there is no ceiling for a carry to recover.  That is "
                  f"the measured form of 'the base cannot use a long window'.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved -> {out_dir}")


if __name__ == "__main__":
    main()
