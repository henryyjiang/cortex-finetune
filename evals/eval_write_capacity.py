"""
Write-capacity diagnostic — can the carry reconstruct the chunk it just encoded?

The carry ablation answers "does the memory help predict the NEXT chunk", and a
null there has three incompatible explanations that it cannot tell apart:
the write encodes nothing, the read cannot use what is there, or there is
nothing worth carrying on natural text.  This asks the easiest possible
question instead — the information is definitely available (the model just
processed those exact tokens) and the target is the best possible one — so a
null here is unambiguous.

For each chunk g of the same 4x1024 PG-19 rows the carry ablation uses, score
chunk g's OWN tokens under four carries:

  self  state AFTER chunk g   — contains g's own summary  (reconstruction)
  prev  state BEFORE chunk g  — the ordinary condition, == the carry ablation
  none  no carry              — the floor
  shuf  another sample's post-chunk state — same shape, wrong content

  reconstruction advantage = L_prev - L_self       (the headline)
  information advantage    = L_shuf - L_self       (net of the register effect)

`shuf` matters because `none` removes the prefix COLUMNS entirely while `self`
adds 32-96 of them, so L_none - L_self confounds information transfer with a
pure extra-register / attention-sink effect.  Measured at init on a pretrained
RoPE LM, a shuffled carry was indistinguishable from a real one, so this control
is not hypothetical.

Reading the result
  L_self well below both  -> the write encodes its chunk; a flat carry delta is
                             then a READ/utility failure, not a compression one
  L_self ~= L_prev        -> the summary vectors do not contain their own chunk;
                             everything downstream of the write is moot

Works on BOTH mechanism generations.  Prefix models get prefix_write=False so
the pass reads the carry without appending a second set of summary slots; the
older bolt-on buffers (AccumCCoT / LSTMBuffer, e.g. the B1 gates) have no such
flag and simply take the state through m_cross_in, which is the same experiment.

The random loop-state init is seeded identically per sample across conditions,
so the ONLY difference is the carry.

Prep (login node, once — shared with the carry ablation):
    python tools/prepare_pg19_dataset.py --tokenizer ckpts/olmo8-cortex \
        --out data/pg19_olmo_val_len4096 --max_length 4096 --split validation

Usage:
    python evals/eval_write_capacity.py \
        --model_name cortex-retrofit/retro-b1-acc4v-tb2-mr8/model_only_chkpt_22500 \
        --data data/pg19_olmo_val_len4096
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

from model_utils import load_checkpoint, has_cross_state, to_num_steps

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CONDS = ("self", "prev", "none", "shuf")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Write-capacity (self-reconstruction) diagnostic")
    p.add_argument("--checkpoint",   type=str, default=None,
                   help="Optional train.py .pt overlay (default: eval --model_name as-is)")
    p.add_argument("--model_name",   required=True)
    p.add_argument("--memory_slots", type=int, default=None)
    p.add_argument("--T",            type=int, default=None,
                   help="Recurrence depth (None = config mean_recurrence)")
    p.add_argument("--data",         required=True,
                   help="Tokenized PG-19 dataset dir (load_from_disk; rows = max_length+1 ids)")
    p.add_argument("--n_chunks",     type=int, default=4,
                   help="Sub-windows per sample (match training cross_chunks)")
    p.add_argument("--max_examples", type=int, default=150, help="0 = all rows")
    p.add_argument("--seed",         type=int, default=1234)
    p.add_argument("--out_dir",      default="eval_results/write_capacity")
    p.add_argument("--dtype",        default="bfloat16", choices=["float32", "bfloat16"])
    p.add_argument("--device",       default=None,
                   help="cuda / cpu (default: cuda when available)")
    return p.parse_args()


def _is_prefix(model) -> bool:
    cortex = getattr(model, "cortex", None)
    if cortex is None and hasattr(model, "module"):
        cortex = getattr(model.module, "cortex", None)
    return cortex is not None and getattr(cortex, "prefix", None) is not None


@torch.no_grad()
def _score(model, xc, yc, mc, state, num_steps, prefix_mode, device):
    """Mean NLL of chunk `xc` read with carry `state`.  No write: prefix models
    drop the summary slots (prefix_write=False), bolt-on models never appended
    any.  Returns None for an all-pad chunk."""
    n_tok = int(mc.sum())
    if n_tok == 0:
        return None
    kw = {"prefix_write": False} if prefix_mode else {}
    out = model(input_ids=xc.unsqueeze(0).to(device), num_steps=num_steps,
                m_cross_in=state, return_m_cross=False, **kw)
    ce = F.cross_entropy(out["logits"][0].float(), yc.to(device), reduction="none")
    return float((ce * mc.to(device)).sum() / n_tok)


@torch.no_grad()
def states_for_sample(model, x, n_chunks, num_steps, seed, device):
    """The TRUE carry chain for one sample: states[g] is the state AFTER chunk
    g's write, states[-1] is None (before chunk 1).  Written exactly as training
    writes it — full forward, summary slots appended."""
    torch.manual_seed(seed)
    states, m_cross = [], None
    for xc in torch.chunk(x, n_chunks):
        out = model(input_ids=xc.unsqueeze(0).to(device), num_steps=num_steps,
                    m_cross_in=m_cross, return_m_cross=True)
        m_cross = out.get("m_cross")
        states.append(m_cross)
    return states


@torch.no_grad()
def chunk_losses(model, x, y, ymask, n_chunks, num_steps, seed, device,
                 prefix_mode, states, shuf_states):
    """Per-chunk NLL under the four carries.  Returns {cond: [per-chunk loss]}."""
    x_chunks = torch.chunk(x, n_chunks)
    y_chunks = torch.chunk(y, n_chunks)
    m_chunks = torch.chunk(ymask, n_chunks)
    rows = {c: [] for c in CONDS}
    for g, (xc, yc, mc) in enumerate(zip(x_chunks, y_chunks, m_chunks)):
        carries = {
            "self": states[g],                       # includes chunk g's summary
            "prev": states[g - 1] if g > 0 else None,
            "none": None,
            # a different sample's post-chunk state: same shape, wrong content.
            # index g so the shape matches under accumulation (state grows with g).
            "shuf": shuf_states[g] if shuf_states is not None else None,
        }
        for cond in CONDS:
            torch.manual_seed(seed)              # identical s0 across conditions
            rows[cond].append(_score(model, xc, yc, mc, carries[cond],
                                     num_steps, prefix_mode, device))
    return rows


def _paired(a, b):
    """mean and SE of (a - b) over paired samples."""
    d = torch.tensor(a) - torch.tensor(b)
    se = float(d.std() / max(len(d), 1) ** 0.5) if len(d) > 1 else 0.0
    return float(d.mean()), se


def main() -> None:
    args = parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32

    print(f"Loading: {args.model_name}  (overlay: {args.checkpoint})")
    model, cfg = load_checkpoint(args.checkpoint, args.model_name,
                                 args.memory_slots, dtype, device)
    if not has_cross_state(model):
        raise SystemExit("Model has no cross state — there is no write to measure.")
    prefix_mode = _is_prefix(model)
    print(f"mechanism: {'prefix splice' if prefix_mode else 'bolt-on buffer'}"
          f"  device={device}  dtype={args.dtype}")

    num_steps = to_num_steps(args.T if args.T is not None else int(cfg.mean_recurrence))
    print(f"T={int(num_steps[0])}  n_chunks={args.n_chunks}")

    from datasets import load_from_disk
    ds = load_from_disk(args.data)
    n = len(ds) if args.max_examples == 0 else min(args.max_examples, len(ds))
    print(f"{args.data}: {len(ds)} rows, evaluating {n}")

    def sample(i):
        row = ds[i]
        ids = torch.tensor(row["input_ids"], dtype=torch.long)
        mask = torch.tensor(row["attention_mask"], dtype=torch.float)
        return ids[:-1], ids[1:], mask[1:]

    # per_chunk[cond][g] = per-sample losses, index-aligned across conditions
    per_chunk = {c: [[] for _ in range(args.n_chunks)] for c in CONDS}

    # `shuf` reuses the PREVIOUS sample's chain, so seed it from the last sample
    # before the loop starts.  Without this, sample 0 would silently fall back to
    # a None carry and score `shuf` identically to `none`, inflating the
    # information delta by exactly one sample.
    xs, _, _ = sample(n - 1)
    prev_states = states_for_sample(model, xs, args.n_chunks, num_steps,
                                    args.seed + n - 1, device)
    for si in range(n):
        x, y, ym = sample(si)
        seed = args.seed + si
        states = states_for_sample(model, x, args.n_chunks, num_steps, seed, device)
        rows = chunk_losses(model, x, y, ym, args.n_chunks, num_steps, seed,
                            device, prefix_mode, states, prev_states)
        for g in range(args.n_chunks):
            if all(rows[c][g] is not None for c in CONDS):
                for c in CONDS:
                    per_chunk[c][g].append(rows[c][g])
        prev_states = states
        if (si + 1) % 25 == 0:
            print(f"  {si + 1}/{n} samples...")

    results = {}
    print("\n  {:<7} {:>5} {:>9} {:>9} {:>9} {:>9} {:>12} {:>12}".format(
        "Chunk", "N", "self", "prev", "none", "shuf", "d(recon)", "d(info)"))
    print("  " + "-" * 82)
    agg = {"recon": [], "info": []}
    for g in range(args.n_chunks):
        if not per_chunk["self"][g]:
            continue
        m = {c: torch.tensor(per_chunk[c][g]) for c in CONDS}
        recon, se_r = _paired(per_chunk["prev"][g], per_chunk["self"][g])
        info, se_i = _paired(per_chunk["shuf"][g], per_chunk["self"][g])
        results[f"chunk{g + 1}"] = {
            "n": len(per_chunk["self"][g]),
            **{c: float(m[c].mean()) for c in CONDS},
            "delta_recon": recon, "se_recon": se_r,
            "delta_info": info, "se_info": se_i,
        }
        print("  {:<7} {:>5} {:>9.4f} {:>9.4f} {:>9.4f} {:>9.4f} {:>12.5f} {:>12.5f}".format(
            g + 1, len(per_chunk["self"][g]), *[float(m[c].mean()) for c in CONDS],
            recon, info))
        if g > 0:
            agg["recon"].append(torch.tensor(per_chunk["prev"][g]) -
                                torch.tensor(per_chunk["self"][g]))
            agg["info"].append(torch.tensor(per_chunk["shuf"][g]) -
                               torch.tensor(per_chunk["self"][g]))

    # Chunk 1 has no `prev` carry, so pool chunks 2+ for the headline, matching
    # the carry ablation's aggregation.
    for key in ("recon", "info"):
        if not agg[key]:
            continue
        d = torch.cat(agg[key])
        results[f"chunks2plus_{key}"] = {
            "n": len(d), "delta": float(d.mean()),
            "se": float(d.std() / max(len(d), 1) ** 0.5),
        }
        label = ("reconstruction (prev - self)" if key == "recon"
                 else "information   (shuf - self)")
        print(f"\n  chunks 2+ {label}: {d.mean():+.5f} "
              f"(SE {results[f'chunks2plus_{key}']['se']:.5f})")

    print("\n  Large positive reconstruction delta => the write encodes its own")
    print("  chunk, so a flat carry-ablation delta is a READ failure.")
    print("  Reconstruction large but information ~0 => the gain is the extra")
    print("  prefix columns acting as registers, not the content.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved -> {out_dir}")


if __name__ == "__main__":
    main()
