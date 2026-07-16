"""
Carried-vs-zeroed ITER-STATE ablation — the cross-window readout for the
dense/no-boundary-crossing arms (M_iter / ccot_iter, final-arms round).

These models train their memory entirely within-window (LM2-style); whether
the learned state is USEFUL across windows is an inference-time question:
at each chunk boundary the final per-position buffer is pooled (default:
last position, causally complete) and broadcast-seeded into every position
of the next chunk via CortexMemory.set_iter_carry().

For each held-out sample (max_length tokens, split into n_chunks
sub-windows), per-chunk LM loss under two conditions:

  carried : iter state pooled + seeded chunk-to-chunk
  zeroed  : every chunk starts from zero buffers (training-identical)

Chunk 1 is identical between conditions (no incoming state either way) — the
built-in delta~0 sanity check.  The within-window memory runs in BOTH
conditions, so the delta isolates exactly the cross-window carry.  Paired
per-sample design; s0 loop-init draws seeded identically across conditions.

Any cross state (M_cross/DirectCCoT/AccumCCoT) is carried identically in
both conditions, so hybrid models still get a pure iter-carry delta.

Usage:
    python evals/eval_iter_carry_ablation.py \
        --model_name cortex-retro-ft/rung1-k0-ki4-cc1-len1024-ep4/final_checkpoint \
        --data data/pg19_olmo_val_len4096
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from model_utils import load_checkpoint, to_num_steps


def _cortex(model):
    m = model
    if hasattr(m, "module"):
        m = m.module
    if hasattr(m, "_orig_mod"):
        m = m._orig_mod
    return getattr(m, "cortex", None)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Carried-vs-zeroed iter-state ablation")
    p.add_argument("--checkpoint",   type=str, default=None,
                   help="Optional train.py .pt overlay (default: eval --model_name as-is)")
    p.add_argument("--model_name",   default="EleutherAI/pythia-160m")
    p.add_argument("--memory_slots", type=int, default=None)
    p.add_argument("--T",            type=int, default=None,
                   help="Recurrence depth (None = config mean_recurrence)")
    p.add_argument("--data",         required=True,
                   help="Tokenized dataset dir (load_from_disk; rows = max_length+1 ids)")
    p.add_argument("--n_chunks",     type=int, default=4,
                   help="Sub-windows per sample (1024-token windows on 4096 rows "
                        "= the dense arms' training window length)")
    p.add_argument("--pool",         default="last", choices=["last", "mean"],
                   help="How the window's per-position buffer is pooled for carry")
    p.add_argument("--max_examples", type=int, default=200, help="0 = all rows")
    p.add_argument("--seed",         type=int, default=1234)
    p.add_argument("--out_dir",      default="eval_results/iter_carry_ablation")
    p.add_argument("--dtype",        default="bfloat16", choices=["float32", "bfloat16"])
    return p.parse_args()


@torch.no_grad()
def chunk_losses(model, cortex, x, y, ymask, n_chunks, num_steps, pool,
                 carried: bool, seed: int, device):
    """Per-chunk mean NLL for one sample under one condition.
    Cross state (if any) is carried in BOTH conditions; only the iter-state
    seeding differs."""
    torch.manual_seed(seed)          # identical s0 draws across conditions
    x_chunks = torch.chunk(x, n_chunks)
    y_chunks = torch.chunk(y, n_chunks)
    m_chunks = torch.chunk(ymask, n_chunks)
    m_cross = None
    iter_st = ci_st = None
    out_rows = []
    for xc, yc, mc in zip(x_chunks, y_chunks, m_chunks):
        if carried and (iter_st is not None or ci_st is not None):
            cortex.set_iter_carry(iter_st, ci_st)
        out = model(input_ids=xc.unsqueeze(0).to(device), num_steps=num_steps,
                    m_cross_in=m_cross, return_m_cross=True)
        m_cross = out.get("m_cross")
        if carried:
            iter_st, ci_st = cortex.get_iter_state(1, xc.shape[0], pool=pool)
        n_tok = int(mc.sum())
        if n_tok == 0:
            out_rows.append((None, 0))
            continue
        logits = out["logits"][0].float()
        ce = F.cross_entropy(logits, yc.to(device), reduction="none")
        loss = float((ce * mc.to(device)).sum() / n_tok)
        out_rows.append((loss, n_tok))
    return out_rows


def main() -> None:
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype  = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32

    print(f"Loading: {args.model_name}  (overlay: {args.checkpoint})")
    model, cfg = load_checkpoint(args.checkpoint, args.model_name,
                                 args.memory_slots, dtype, device)
    cortex = _cortex(model)
    if cortex is None or not cortex.has_iter_state:
        raise SystemExit("Model has no iter state (M_iter / ccot_iter) — "
                         "carried == zeroed by construction; nothing to ablate. "
                         "For M_cross models use eval_carry_ablation.py.")
    num_steps = to_num_steps(args.T if args.T is not None else int(cfg.mean_recurrence))
    print(f"T={int(num_steps[0])}  n_chunks={args.n_chunks}  pool={args.pool}  "
          f"m_iter={cortex.m_iter is not None}  ccot_iter={cortex.ccot_iter is not None}")

    from datasets import load_from_disk
    ds = load_from_disk(args.data)
    n = len(ds) if args.max_examples == 0 else min(args.max_examples, len(ds))
    print(f"{args.data}: {len(ds)} rows, evaluating {n}")

    per_chunk = [[] for _ in range(args.n_chunks)]

    for si in range(n):
        row  = ds[si]
        ids  = torch.tensor(row["input_ids"], dtype=torch.long)
        mask = torch.tensor(row["attention_mask"], dtype=torch.float)
        x, y, ym = ids[:-1], ids[1:], mask[1:]
        seed = args.seed + si
        rc = chunk_losses(model, cortex, x, y, ym, args.n_chunks, num_steps,
                          args.pool, True,  seed, device)
        rz = chunk_losses(model, cortex, x, y, ym, args.n_chunks, num_steps,
                          args.pool, False, seed, device)
        for i, ((lc, ntc), (lz, _)) in enumerate(zip(rc, rz)):
            if lc is not None and lz is not None:
                per_chunk[i].append((lc, lz))
        if (si + 1) % 25 == 0:
            print(f"  {si + 1}/{n} samples...")

    results = {}
    print(f"\n  {'Chunk':<7} {'N':>5} {'carried':>10} {'zeroed':>10} "
          f"{'delta(z-c)':>12} {'SE':>9}  (delta>0 => carry helps)")
    print(f"  {'-'*60}")
    agg_deltas = []
    for i, pairs in enumerate(per_chunk):
        if not pairs:
            continue
        c  = torch.tensor([p[0] for p in pairs])
        z  = torch.tensor([p[1] for p in pairs])
        d  = z - c                        # positive = carried loss is LOWER
        se = float(d.std() / max(len(d), 1) ** 0.5) if len(d) > 1 else 0.0
        results[f"chunk{i+1}"] = {
            "n": len(pairs), "carried": float(c.mean()), "zeroed": float(z.mean()),
            "delta": float(d.mean()), "se": se,
        }
        print(f"  {i+1:<7} {len(pairs):>5} {c.mean():>10.4f} {z.mean():>10.4f} "
              f"{d.mean():>12.5f} {se:>9.5f}")
        if i > 0:
            agg_deltas.append(d)
    if agg_deltas:
        d_all = torch.cat(agg_deltas)
        results["chunks2plus"] = {
            "n": len(d_all), "delta": float(d_all.mean()),
            "se": float(d_all.std() / max(len(d_all), 1) ** 0.5),
        }
        print(f"\n  chunks 2+ aggregate: delta = {d_all.mean():.5f} "
              f"(SE {results['chunks2plus']['se']:.5f})  "
              f"[chunk 1 is the identical-by-construction control]")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved -> {out_dir}")


if __name__ == "__main__":
    main()
