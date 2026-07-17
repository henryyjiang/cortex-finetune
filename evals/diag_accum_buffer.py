"""
A0.2 — AccumCCoT buffer diagnostics at inference (two-track plan, 2026-07-17).

Confirms (or refutes) the generation-collapse hypothesis: acc4v degenerates to
function words ("and", "to the") on >= 4k chunked eval, and the suspected
mechanism is the accumulated buffer going OUT-OF-DISTRIBUTION once the chunk
count exceeds the training range (training only ever showed the read
cross_chunks-1 = 3 accumulated writes; a 8k/16k chunked eval feeds it 7-15).

For each held-out long sample, run the chunked carried forward (exactly the
chunked longcontext eval) and log per chunk index:

  n_vecs        — accumulated vector count after this chunk's write (shows
                  where the accum_max FIFO cap starts trimming)
  new_norm      — mean L2 of THIS chunk's freshly extracted vectors
  state_norm    — mean L2 over the whole accumulated state
  new_vs_old    — mean cosine similarity of this chunk's vectors to all older
                  vectors (collapse detector: drifting toward 1 = the buffer
                  is filling with near-duplicates)
  loss          — per-chunk LM loss under the carried forward
  distinct      — (--gen_tokens > 0) distinct-token ratio of a greedy
                  continuation generated with the buffer held fixed; degenerate
                  function-word loops show up as a low ratio.  Sample texts for
                  the first few samples are stored in the JSON for eyeballing.

If new_norm / new_vs_old stay flat with chunk index but generation still
degrades, the OOD story is wrong and the problem is in the read; if they
drift past chunk ~4, that motivates A1's chunk-count curriculum.

Prep (login node, once — longer windows than the training data):
    python tools/prepare_pg19_dataset.py --tokenizer ckpts/olmo8-cortex \
        --out data/pg19_olmo_val_len16384 --max_length 16384 --split validation

Usage:
    python evals/diag_accum_buffer.py \
        --model_name cortex-retro-ft/rung1-k0-acc4v-tb2-rs-ep3-rcl/final_checkpoint \
        --data data/pg19_olmo_val_len16384 --chunk_len 1024 --gen_tokens 32
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from model_utils import (load_checkpoint, has_cross_state, to_num_steps,
                         greedy_generate, _unwrap)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("AccumCCoT buffer diagnostics")
    p.add_argument("--checkpoint",   type=str, default=None)
    p.add_argument("--model_name",   required=True)
    p.add_argument("--memory_slots", type=int, default=None)
    p.add_argument("--T",            type=int, default=None)
    p.add_argument("--data",         required=True,
                   help="Tokenized long-window dataset dir (load_from_disk)")
    p.add_argument("--chunk_len",    type=int, default=1024,
                   help="Window size per chunk (match training chunk length)")
    p.add_argument("--max_examples", type=int, default=50, help="0 = all rows")
    p.add_argument("--gen_tokens",   type=int, default=32,
                   help="Greedy tokens to generate after each chunk (0 = off)")
    p.add_argument("--gen_prompt",   type=int, default=128,
                   help="Tokens of the current chunk used as the generation prompt")
    p.add_argument("--gen_save_samples", type=int, default=3,
                   help="Store generated texts for this many samples in the JSON")
    p.add_argument("--seed",         type=int, default=1234)
    p.add_argument("--out_dir",      default="eval_results/diag_accum_buffer")
    p.add_argument("--dtype",        default="bfloat16", choices=["float32", "bfloat16"])
    return p.parse_args()


def _distinct_ratio(text: str, tokenizer) -> float:
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    return len(set(ids)) / max(len(ids), 1)


@torch.no_grad()
def main() -> None:
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype  = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32

    print(f"Loading: {args.model_name}  (overlay: {args.checkpoint})")
    model, cfg = load_checkpoint(args.checkpoint, args.model_name,
                                 args.memory_slots, dtype, device)
    if not has_cross_state(model):
        raise SystemExit("Model has no cross state — nothing to diagnose.")
    accum = getattr(_unwrap(model).cortex, "accum", None)
    if accum is None:
        raise SystemExit("Not an AccumCCoT model — this diagnostic reads the "
                         "accumulated write-once state.")
    n_vec = int(accum.n_vec)
    num_steps = to_num_steps(args.T if args.T is not None else int(cfg.mean_recurrence))
    print(f"T={int(num_steps[0])}  chunk_len={args.chunk_len}  "
          f"accum_vecs={n_vec}  accum_max={int(accum.max_vecs)}")

    tokenizer = None
    if args.gen_tokens > 0:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    from datasets import load_from_disk
    ds = load_from_disk(args.data)
    n = len(ds) if args.max_examples == 0 else min(args.max_examples, len(ds))
    print(f"{args.data}: {len(ds)} rows, evaluating {n}")

    # stats[chunk_idx][key] = list over samples
    stats: dict[int, dict[str, list]] = {}
    gen_samples: list[dict] = []

    for si in range(n):
        row  = ds[si]
        ids  = torch.tensor(row["input_ids"], dtype=torch.long)
        mask = torch.tensor(row["attention_mask"], dtype=torch.float)
        x, y, ym = ids[:-1], ids[1:], mask[1:]
        n_chunks = x.shape[0] // args.chunk_len
        if n_chunks < 2:
            continue
        torch.manual_seed(args.seed + si)     # deterministic s0 draws
        m_cross = None
        sample_gens = []
        for gi in range(n_chunks):
            sl = slice(gi * args.chunk_len, (gi + 1) * args.chunk_len)
            xc, yc, mc = x[sl], y[sl], ym[sl]
            if int(mc.sum()) == 0:
                break                          # padding reached — stop the chain
            out = model(input_ids=xc.unsqueeze(0).to(device), num_steps=num_steps,
                        m_cross_in=m_cross, return_m_cross=True)
            new_state = out.get("m_cross")

            # ── buffer stats ────────────────────────────────────────────────
            st = new_state[0].float()                       # [N, D]
            new  = st[-n_vec:]
            old  = st[:-n_vec]
            rec = stats.setdefault(gi, {k: [] for k in
                    ("n_vecs", "new_norm", "state_norm", "new_vs_old", "loss",
                     "distinct")})
            rec["n_vecs"].append(st.shape[0])
            rec["new_norm"].append(float(new.norm(dim=-1).mean()))
            rec["state_norm"].append(float(st.norm(dim=-1).mean()))
            if old.shape[0] > 0:
                sim = F.cosine_similarity(new.unsqueeze(1), old.unsqueeze(0), dim=-1)
                rec["new_vs_old"].append(float(sim.mean()))

            # ── per-chunk carried LM loss ───────────────────────────────────
            logits = out["logits"][0].float()
            ce = F.cross_entropy(logits, yc.to(device), reduction="none")
            rec["loss"].append(float((ce * mc.to(device)).sum() / int(mc.sum())))

            # ── generation probe (buffer held fixed, short prompt) ──────────
            if args.gen_tokens > 0:
                prompt = xc[-args.gen_prompt:].unsqueeze(0)
                text = greedy_generate(model, tokenizer, prompt,
                                       args.gen_tokens, num_steps,
                                       m_cross=new_state)
                rec["distinct"].append(_distinct_ratio(text, tokenizer))
                if si < args.gen_save_samples:
                    sample_gens.append({"chunk": gi + 1, "text": text})

            m_cross = new_state
        if sample_gens:
            gen_samples.append({"sample": si, "generations": sample_gens})
        if (si + 1) % 10 == 0:
            print(f"  {si + 1}/{n} samples...")

    # ── aggregate + report ──────────────────────────────────────────────────
    def _ms(v):
        if not v:
            return None, None
        t = torch.tensor(v)
        return float(t.mean()), float(t.std())

    results = {}
    print(f"\n  {'Chunk':<6} {'N':>4} {'vecs':>5} {'new_norm':>9} "
          f"{'state_norm':>10} {'new_vs_old':>10} {'loss':>8} {'distinct':>9}")
    print(f"  {'-' * 66}")
    for gi in sorted(stats):
        rec = stats[gi]
        nn_m, nn_s = _ms(rec["new_norm"])
        sn_m, _    = _ms(rec["state_norm"])
        vo_m, _    = _ms(rec["new_vs_old"])
        ls_m, _    = _ms(rec["loss"])
        dr_m, _    = _ms(rec["distinct"])
        results[f"chunk{gi + 1}"] = {
            "n": len(rec["loss"]),
            "n_vecs": int(rec["n_vecs"][0]) if rec["n_vecs"] else None,
            "new_norm_mean": nn_m, "new_norm_std": nn_s,
            "state_norm_mean": sn_m, "new_vs_old_cos": vo_m,
            "loss": ls_m, "distinct_ratio": dr_m,
        }
        print(f"  {gi + 1:<6} {len(rec['loss']):>4} "
              f"{rec['n_vecs'][0] if rec['n_vecs'] else '-':>5} "
              f"{nn_m:>9.4f} {sn_m:>10.4f} "
              f"{vo_m if vo_m is not None else float('nan'):>10.4f} "
              f"{ls_m:>8.4f} "
              f"{dr_m if dr_m is not None else float('nan'):>9.3f}")
    print("\n  Reading: training only ever exposed reads to <= cross_chunks-1 "
          "writes.\n  Drift in new_norm / new_vs_old past that chunk index = "
          "the buffer goes OOD\n  (motivates A1's chunk-count curriculum); "
          "flat stats + degrading distinct\n  ratio = the problem is in the "
          "read, not the state.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "results.json", "w") as f:
        json.dump({"per_chunk": results, "gen_samples": gen_samples,
                   "config": {"chunk_len": args.chunk_len, "n_vec": n_vec,
                              "gen_tokens": args.gen_tokens}}, f, indent=2)
    print(f"\nResults saved -> {out_dir}")


if __name__ == "__main__":
    main()
