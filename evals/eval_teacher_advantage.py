"""
Teacher-advantage probe — does a full-window forward actually beat the
chunk-only context this model family gets at eval time?  Prerequisite check
for full-window-teacher distillation, and an LM-loss length sweep that
locates the RDM's long-context cliff (the 20260712 no-chunk evals showed
degenerate GENERATION beyond ~1.5-2k tokens; this measures whether the
next-token DISTRIBUTION degrades too, which is what a distillation teacher
would be supplying).

For each held-out PG-19 sample (4096 tokens = 4x1024 chunks) and each chunk
n >= 2, compute the mean NLL of chunk n's tokens under a single plain forward
(no carry, m_cross_in=None) whose context is the last W tokens ending at
chunk n's end, for W in --windows.  W=1024 is the chunk-only control — the
exact context a chunked student sees (minus the buffer).

  delta(W) = NLL(W=1024) - NLL(W)   [paired per sample]

  delta > 0  : longer context helps -> a W-window teacher has real signal to
               distill; delta IS the distillable headroom in nats (compare to
               the ~0.004-0.010 nats the carry currently delivers).
  delta <= 0 : the RDM's LM prediction degrades past that window -> cap
               --cortex.distill_window below the cliff (or drop distillation
               for the recall mix).

Prep (login node, once — same dataset as the carry ablation):
    python tools/prepare_pg19_dataset.py --tokenizer ckpts/olmo8-cortex \
        --out data/pg19_olmo_val_len4096 --max_length 4096 --split validation

Usage (the teacher candidate is the BASE graft dir):
    python evals/eval_teacher_advantage.py \
        --model_name ckpts/olmo8-cortex \
        --data data/pg19_olmo_val_len4096
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from model_utils import load_checkpoint, to_num_steps


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Full-window teacher-advantage probe")
    p.add_argument("--checkpoint",   type=str, default=None,
                   help="Optional train.py .pt overlay (default: eval --model_name as-is)")
    p.add_argument("--model_name",   default="ckpts/olmo8-cortex")
    p.add_argument("--memory_slots", type=int, default=None)
    p.add_argument("--T",            type=int, default=None,
                   help="Recurrence depth (None = config mean_recurrence)")
    p.add_argument("--data",         required=True,
                   help="Tokenized PG-19 dataset dir (load_from_disk; rows = max_length+1 ids)")
    p.add_argument("--n_chunks",     type=int, default=4,
                   help="Sub-windows per sample (match training cross_chunks)")
    p.add_argument("--windows",      default="1024,1536,2048,3072,4096",
                   help="Comma-separated context lengths; the smallest is the control")
    p.add_argument("--max_examples", type=int, default=150, help="0 = all rows")
    p.add_argument("--seed",         type=int, default=1234)
    p.add_argument("--out_dir",      default="eval_results/teacher_advantage")
    p.add_argument("--dtype",        default="bfloat16", choices=["float32", "bfloat16"])
    return p.parse_args()


@torch.no_grad()
def span_nll(model, x, y, ym, end, window, span_len, num_steps, seed, device):
    """Mean NLL of the last span_len positions of x[end-window:end], one plain
    forward (no carry).  Returns (nll, n_tokens) or (None, 0) if all-pad."""
    torch.manual_seed(seed)          # deterministic s0 draw
    start = end - window
    xc = x[start:end].unsqueeze(0).to(device)
    out = model(input_ids=xc, num_steps=num_steps)
    logits = out["logits"][0, -span_len:].float()
    yc = y[end - span_len:end].to(device)
    mc = ym[end - span_len:end].to(device)
    n_tok = int(mc.sum())
    if n_tok == 0:
        return None, 0
    ce = F.cross_entropy(logits, yc, reduction="none")
    return float((ce * mc).sum() / n_tok), n_tok


def main() -> None:
    args    = parse_args()
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype   = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    windows = sorted(int(w) for w in args.windows.split(","))
    w0      = windows[0]             # control = chunk-only context

    print(f"Loading: {args.model_name}  (overlay: {args.checkpoint})")
    model, cfg = load_checkpoint(args.checkpoint, args.model_name,
                                 args.memory_slots, dtype, device)
    num_steps = to_num_steps(args.T if args.T is not None else int(cfg.mean_recurrence))
    print(f"T={int(num_steps[0])}  n_chunks={args.n_chunks}  windows={windows}")

    from datasets import load_from_disk
    ds = load_from_disk(args.data)
    n = len(ds) if args.max_examples == 0 else min(args.max_examples, len(ds))
    print(f"{args.data}: {len(ds)} rows, evaluating {n}")

    # cell[(chunk_idx, W)] = list of nll paired by sample (same sample order in
    # every cell of a chunk row, so per-sample deltas vs W=w0 are exact pairs)
    cell: dict[tuple[int, int], list[float]] = {}

    for si in range(n):
        row  = ds[si]
        ids  = torch.tensor(row["input_ids"], dtype=torch.long)
        mask = torch.tensor(row["attention_mask"], dtype=torch.float)
        x, y, ym = ids[:-1], ids[1:], mask[1:]
        cl = len(x) // args.n_chunks           # chunk length (1024 @ 4096/4)
        seed = args.seed + si
        for ci in range(1, args.n_chunks):     # chunks 2..N (chunk 1 has no prefix)
            end = (ci + 1) * cl
            span = min(cl, w0)
            nlls = {}
            for W in windows:
                if W > end:                    # not enough prefix for this window
                    continue
                # same seed across W: the only varying factor is the context
                nll, n_tok = span_nll(model, x, y, ym, end, W, span,
                                      num_steps, seed, device)
                if nll is None:
                    nlls = {}
                    break                      # all-pad span: drop every cell
                nlls[W] = nll
            if w0 not in nlls:                 # control missing -> unpaired, skip
                continue
            for W, v in nlls.items():
                cell.setdefault((ci, W), []).append(v)
        if (si + 1) % 25 == 0:
            print(f"  {si + 1}/{n} samples...")

    # Paired deltas vs the control window, per (chunk, W) and aggregated per W
    results: dict[str, dict] = {}
    print(f"\n  {'chunk':<6} {'W':>6} {'N':>5} {'NLL':>9} {'delta_vs_'+str(w0):>12} {'SE':>9}"
          f"   (delta>0 => longer context helps)")
    print(f"  {'-'*56}")
    per_w: dict[int, list[torch.Tensor]] = {}
    for ci in range(1, args.n_chunks):
        base = cell.get((ci, w0))
        if not base:
            continue
        b = torch.tensor(base)
        for W in windows:
            vals = cell.get((ci, W))
            if not vals:
                continue
            v = torch.tensor(vals)
            d = b[: len(v)] - v                # paired: same sample order
            se = float(d.std() / max(len(d), 1) ** 0.5) if len(d) > 1 else 0.0
            results[f"chunk{ci+1}_w{W}"] = {
                "n": len(v), "nll": float(v.mean()),
                "delta_vs_control": float(d.mean()), "se": se,
            }
            print(f"  {ci+1:<6} {W:>6} {len(v):>5} {v.mean():>9.4f} "
                  f"{d.mean():>12.5f} {se:>9.5f}")
            if W != w0:
                per_w.setdefault(W, []).append(d)
    for W, ds_ in sorted(per_w.items()):
        d_all = torch.cat(ds_)
        results[f"aggregate_w{W}"] = {
            "n": len(d_all), "delta_vs_control": float(d_all.mean()),
            "se": float(d_all.std() / max(len(d_all), 1) ** 0.5),
        }
        print(f"\n  aggregate W={W}: delta = {d_all.mean():.5f} "
              f"(SE {results[f'aggregate_w{W}']['se']:.5f}, n={len(d_all)})")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved -> {out_dir}")


if __name__ == "__main__":
    main()
