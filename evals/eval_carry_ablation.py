"""
Carry-vs-zeroed M_cross ablation — the most sensitive "is anything carried?"
detector — plus the A0.1 oldest-vs-newest SLICE ablation for AccumCCoT
models (two-track plan, 2026-07-17).

For each held-out PG-19 sample (4096 tokens, split into the same 4x1024
sub-windows as training), compute per-chunk LM loss under paired conditions:

  carried : m_cross carried chunk-to-chunk (exactly the training forward)
  zeroed  : m_cross_in=None on every chunk (the buffer contributes nothing)
  oldest  : (--slice_ablate) carried, but the OLDEST n vectors of the
            accumulated state are dropped before every read
  newest  : (--slice_ablate) carried, but the NEWEST n vectors (the most
            recent chunk's) are dropped before every read

Chunk 1 is identical between conditions (no incoming buffer either way) — a
built-in sanity check that should show delta ~0.  Chunks 2+ are where a
working memory must lower the loss.  The paired per-sample design cancels
cross-sample variance, so deltas of ~0.001 nats are resolvable with ~100
samples.

Slice-ablation logic (AccumCCoT only — write-once rows stay separable):
the TRUE accumulated state is maintained outside the model; each chunk's
read receives the ablated view, and the chunk's newly extracted vectors
(the last accum_vecs rows of the returned state) are re-appended to the
true state, so the ablation never compounds into later writes.  Prediction
from the flat per-chunk delta of acc4v (chunk2 +0.0238 ≈ chunk4 +0.0225):
dropping the OLDEST vectors changes nothing and dropping the NEWEST kills
the delta — i.e. all signal lives in the most recent chunk's vectors and
carry_grad_chunks=2 is the binding constraint on multi-hop (gates A1-horizon).

The random loop-state init (s0 ~ N(0,sigma^2)) is seeded identically per
sample across all conditions, so the ONLY difference is the buffer.

Prep (login node, once):
    python tools/prepare_pg19_dataset.py --tokenizer ckpts/olmo8-cortex \
        --out data/pg19_olmo_val_len4096 --max_length 4096 --split validation

Usage:
    python evals/eval_carry_ablation.py \
        --model_name cortex-retro-ft/rung1-k0-acc4v-tb2-rs-ep3-rcl/final_checkpoint \
        --data data/pg19_olmo_val_len4096 \
        --slice_ablate both
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

from model_utils import load_checkpoint, has_cross_state, to_num_steps, _unwrap

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cortex_memory.chunking import ablate_vec_slice  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Carry-vs-zeroed M_cross ablation")
    p.add_argument("--checkpoint",   type=str, default=None,
                   help="Optional train.py .pt overlay (default: eval --model_name as-is)")
    p.add_argument("--model_name",   default="EleutherAI/pythia-160m")
    p.add_argument("--memory_slots", type=int, default=None)
    p.add_argument("--T",            type=int, default=None,
                   help="Recurrence depth (None = config mean_recurrence)")
    p.add_argument("--data",         required=True,
                   help="Tokenized PG-19 dataset dir (load_from_disk; rows = max_length+1 ids)")
    p.add_argument("--n_chunks",     type=int, default=4,
                   help="Sub-windows per sample (match training cross_chunks)")
    p.add_argument("--max_examples", type=int, default=200, help="0 = all rows")
    p.add_argument("--seed",         type=int, default=1234)
    p.add_argument("--out_dir",      default="eval_results/carry_ablation")
    p.add_argument("--dtype",        default="bfloat16", choices=["float32", "bfloat16"])
    # A0.1 slice ablation (AccumCCoT models only)
    p.add_argument("--slice_ablate", default="none",
                   choices=["none", "oldest", "newest", "both"],
                   help="Also run carried-with-slice-dropped conditions "
                        "(AccumCCoT only; rows are write-once and separable)")
    p.add_argument("--slice_n",      type=int, default=0,
                   help="Vectors to ablate (0 = the model's accum_vecs, i.e. "
                        "one chunk's worth)")
    p.add_argument("--slice_op",     default="drop", choices=["drop", "zero"],
                   help="drop = remove rows (clean; read renormalizes). "
                        "zero = zero rows in place (confounded: zeroed rows "
                        "still soak up softmax mass in the read)")
    return p.parse_args()


@torch.no_grad()
def chunk_losses(model, x, y, ymask, n_chunks, num_steps, cond: str,
                 seed: int, device, n_vec: int = 0, slice_n: int = 0,
                 slice_op: str = "drop"):
    """Per-chunk mean NLL for one sample under one condition
    (carried / zeroed / oldest / newest).
    Returns list of (loss, n_tokens) per chunk; None loss for all-pad chunks."""
    torch.manual_seed(seed)          # identical s0 draws across conditions
    x_chunks = torch.chunk(x, n_chunks)
    y_chunks = torch.chunk(y, n_chunks)
    m_chunks = torch.chunk(ymask, n_chunks)
    m_cross = None                   # TRUE state (never ablated)
    out_rows = []
    for xc, yc, mc in zip(x_chunks, y_chunks, m_chunks):
        if cond == "zeroed":
            m_in = None
        elif cond in ("oldest", "newest") and m_cross is not None:
            m_in = ablate_vec_slice(m_cross, slice_n, cond, slice_op)
            if m_in is not None and m_in.shape[1] == 0:
                m_in = None          # everything dropped == no buffer
        else:
            m_in = m_cross
        out = model(input_ids=xc.unsqueeze(0).to(device), num_steps=num_steps,
                    m_cross_in=m_in, return_m_cross=(cond != "zeroed"))
        if cond == "carried":
            m_cross = out.get("m_cross")
        elif cond in ("oldest", "newest"):
            # Reconstruct the true state: the returned state's last n_vec rows
            # are this chunk's newly extracted vectors (write-once append) —
            # re-append them to the UN-ablated state so the ablation affects
            # reads only and never compounds into later writes.
            ret = out.get("m_cross")
            new_rows = ret[:, -n_vec:]
            m_cross = new_rows if m_cross is None \
                else torch.cat([m_cross, new_rows], dim=1)
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
    if not has_cross_state(model):
        raise SystemExit("Model has no cross state (M_cross / DirectCCoT) — "
                         "carried == zeroed by construction; nothing to ablate.")

    conditions = ["carried", "zeroed"]
    n_vec = slice_n = 0
    if args.slice_ablate != "none":
        accum = getattr(_unwrap(model).cortex, "accum", None)
        if accum is None:
            raise SystemExit("--slice_ablate needs an AccumCCoT model (write-"
                             "once rows) — gated/overwritten states are not "
                             "separable per chunk.")
        n_vec   = int(accum.n_vec)
        slice_n = args.slice_n if args.slice_n > 0 else n_vec
        conditions += (["oldest", "newest"] if args.slice_ablate == "both"
                       else [args.slice_ablate])
        print(f"Slice ablation: {conditions[2:]}  n={slice_n} vectors "
              f"(accum_vecs={n_vec})  op={args.slice_op}")

    num_steps = to_num_steps(args.T if args.T is not None else int(cfg.mean_recurrence))
    print(f"T={int(num_steps[0])}  n_chunks={args.n_chunks}")

    from datasets import load_from_disk
    ds = load_from_disk(args.data)
    n = len(ds) if args.max_examples == 0 else min(args.max_examples, len(ds))
    print(f"{args.data}: {len(ds)} rows, evaluating {n}")

    # per_chunk[cond][i] = list of per-sample losses, index-aligned across
    # conditions (only samples where every condition produced a loss count)
    per_chunk = {c: [[] for _ in range(args.n_chunks)] for c in conditions}

    for si in range(n):
        row  = ds[si]
        ids  = torch.tensor(row["input_ids"], dtype=torch.long)
        mask = torch.tensor(row["attention_mask"], dtype=torch.float)
        x, y, ym = ids[:-1], ids[1:], mask[1:]
        seed = args.seed + si
        rows = {c: chunk_losses(model, x, y, ym, args.n_chunks, num_steps, c,
                                seed, device, n_vec, slice_n, args.slice_op)
                for c in conditions}
        for i in range(args.n_chunks):
            if all(rows[c][i][0] is not None for c in conditions):
                for c in conditions:
                    per_chunk[c][i].append(rows[c][i][0])
        if (si + 1) % 25 == 0:
            print(f"  {si + 1}/{n} samples...")

    # Paired stats per chunk index: delta(cond) = cond - carried per sample
    # (positive = removing that information HURT, i.e. it carried signal).
    others = [c for c in conditions if c != "carried"]
    results = {}
    hdr = f"  {'Chunk':<7} {'N':>5} {'carried':>10}"
    for c in others:
        hdr += f" {c:>10} {'d(' + c + ')':>12}"
    print("\n" + hdr + "   (delta>0 => that carry content helps)")
    print(f"  {'-' * (len(hdr) - 2)}")
    agg = {c: [] for c in others}
    for i in range(args.n_chunks):
        pairs = per_chunk["carried"][i]
        if not pairs:
            continue
        cvec = torch.tensor(pairs)
        entry = {"n": len(pairs), "carried": float(cvec.mean())}
        line = f"  {i + 1:<7} {len(pairs):>5} {cvec.mean():>10.4f}"
        for c in others:
            ovec = torch.tensor(per_chunk[c][i])
            d    = ovec - cvec
            se   = float(d.std() / max(len(d), 1) ** 0.5) if len(d) > 1 else 0.0
            key  = "zeroed" if c == "zeroed" else f"abl_{c}"
            entry[key] = float(ovec.mean())
            entry["delta" if c == "zeroed" else f"delta_{c}"] = float(d.mean())
            entry["se" if c == "zeroed" else f"se_{c}"] = se
            line += f" {ovec.mean():>10.4f} {d.mean():>12.5f}"
            if i > 0:
                agg[c].append(d)
        results[f"chunk{i + 1}"] = entry
        print(line)
    for c in others:
        if not agg[c]:
            continue
        d_all = torch.cat(agg[c])
        key = "chunks2plus" if c == "zeroed" else f"chunks2plus_{c}"
        results[key] = {
            "n": len(d_all), "delta": float(d_all.mean()),
            "se": float(d_all.std() / max(len(d_all), 1) ** 0.5),
        }
        print(f"\n  chunks 2+ [{c:>7}]: delta = {d_all.mean():.5f} "
              f"(SE {results[key]['se']:.5f})"
              + ("  [chunk 1 is the identical-by-construction control]"
                 if c == "zeroed" else ""))
    if args.slice_ablate != "none":
        results["slice_config"] = {"n": slice_n, "op": args.slice_op,
                                   "accum_vecs": n_vec}

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved -> {out_dir}")


if __name__ == "__main__":
    main()
