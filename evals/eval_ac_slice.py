"""
A0.1 SLICE ABLATION ON A REFERENCE MODEL — is recency dominance a property of
the mechanism, or a cortex defect?

Why this exists
---------------
On rung1-ep1 (PG-19, nc=8, 32 vectors/write) our slice ablation found that the
newest write carries ~92% of the whole carry benefit, and that the benefit does
not grow with depth (chunk2 delta 0.111 -> chunk8 delta 0.100).  AutoCompressor
(Chevalier et al., EMNLP'23) Figure 2 claims the opposite for the same
mechanism: summary accumulation "helps improve perplexity beyond one compressed
segment".

This runs OUR ablation, unchanged in protocol and sign convention, on THEIR
released model:  princeton-nlp/AutoCompressor-1.3b-30k -- OPT-1.3b fine-tuned
on 2B tokens of Books3, 30,720-token sequences, 20 segments, 50 summary
vectors, summary accumulation ON (config verified at load).

  * If AC also puts ~90% of the value in the newest summary, recency dominance
    is a property of the approach and cortex is behaving normally.
  * If AC's contribution is spread across the 20 steps, cortex has a genuine
    defect and the frozen read loop is the leading suspect.

Protocol (identical to evals/eval_carry_ablation.py)
----------------------------------------------------
    delta = ablated - carried        POSITIVE = those vectors were HELPING

The accumulated state is built ONCE per sample by the true (un-ablated) chain,
exactly as eval_carry_ablation.py:118-131 does -- the ablation affects READS
only and never compounds into later writes.  So every condition at chunk i
reads a slice of the same S_i, and the paired delta is clean.

Conditions at chunk i (S_i = the 50*i vectors written by chunks 1..i):
    carried       S_i                (full buffer)
    zeroed        empty              (no carry at all)
    drop_oldest1  S_i[:, 50:]        (drop the oldest ONE write)
    drop_newest1  S_i[:, :-50]       (drop the newest ONE write)
    keep_newest1  S_i[:, -50:]       (KEEP only the newest write)
    keep_oldest1  S_i[:, :50]        (KEEP only the oldest write)

Chunk 1 is a built-in control: no state exists, so every delta must be exactly
0.0.  Anything else means the port is wrong.

The port
--------
`AutoCompressors-main/auto_compressor.py` imports `modeling_flash_llama` at
module scope (flash-attn, unavailable on Blackwell/Windows), so the OPT forward
is ported here -- same port as evals/eval_autocompressor_carry.py, which
documents its fidelity against auto_compressor.py:85,123,294-306.

Usage:
    python evals/eval_ac_slice.py --out_dir eval_results/ac_slice --max_examples 20
"""
from __future__ import annotations

import argparse
import json
import os
import statistics as st
import time

import torch
import torch.nn.functional as F
from torch import nn

CONDITIONS = ["carried", "zeroed", "drop_oldest1", "drop_newest1",
              "keep_newest1", "keep_oldest1"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("AutoCompressor A0.1 slice ablation")
    p.add_argument("--model_name", default="princeton-nlp/AutoCompressor-1.3b-30k")
    p.add_argument("--n_chunks", type=int, default=20,
                   help="segments per sample (this model was trained at 20)")
    p.add_argument("--chunk_len", type=int, default=1536,
                   help="tokens per segment (20 x 1536 = 30,720 = its training shape)")
    p.add_argument("--max_examples", type=int, default=20,
                   help="PG-19 test books; one contiguous window is taken from each")
    p.add_argument("--skip_tokens", type=int, default=1000,
                   help="tokens skipped at the head of each book (Gutenberg front matter)")
    p.add_argument("--hf_dataset", default="emozilla/pg19-test",
                   help="parquet mirror of PG-19 test; deepmind/pg19 is script-based "
                        "and datasets>=5 refuses it")
    p.add_argument("--split", default="test")
    p.add_argument("--text_col", default="text")
    p.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16"])
    p.add_argument("--out_dir", default="eval_results/ac_slice")
    return p.parse_args()


# --------------------------------------------------------------------------
# port of auto_compressor.py:281-306 (see module docstring)
# --------------------------------------------------------------------------
class PortedOPTPositions(nn.Module):
    """Real tokens get cumsum*mask + 1; softprompt and summary slots get 1 (the
    pad row), i.e. no position.  The `position_ids` transformers computes for us
    is deliberately ignored -- it knows nothing about the placeholder slots."""

    def __init__(self, weight: torch.Tensor, cfg: dict):
        super().__init__()
        # must be a buffer, not a bare attribute, or model.to(device) skips it
        self.register_buffer("weight", weight)
        self.cfg = cfg

    def forward(self, attention_mask, past_key_values_length: int = 0, position_ids=None):
        attention_mask = attention_mask.long()
        bsz = attention_mask.size(0)
        sp_len = self.cfg["softprompt_length"]
        sm_len = self.cfg["summary_length"]
        pkv_sp = self.cfg["past_key_values_softprompt_length"]

        dev = attention_mask.device
        left = torch.ones(bsz, sp_len, dtype=torch.long, device=dev)
        right = torch.ones(bsz, sm_len, dtype=torch.long, device=dev)

        total_sp = sp_len + pkv_sp
        am = attention_mask[:, total_sp: attention_mask.size(1) - sm_len]
        positions = am.cumsum(dim=1) * am + 1
        positions = positions[:, past_key_values_length - pkv_sp:]
        positions = torch.cat([left, positions, right], dim=1)
        return F.embedding(positions, self.weight)


def _load_embed_summary(name: str, summary_length: int, hidden: int) -> torch.Tensor:
    """Loading through the stock OPTForCausalLM drops embed_summary.weight as an
    unexpected key, so fetch it straight out of the checkpoint.  Handles both the
    sharded (2.7b) and single-file (1.3b) layouts."""
    from huggingface_hub import hf_hub_download

    try:
        index = json.load(open(hf_hub_download(name, "pytorch_model.bin.index.json")))
        key = next(k for k in index["weight_map"] if k.endswith("embed_summary.weight"))
        shard = hf_hub_download(name, index["weight_map"][key])
    except Exception:
        shard, key = hf_hub_download(name, "pytorch_model.bin"), None

    try:
        sd = torch.load(shard, map_location="cpu", weights_only=True, mmap=True)
    except Exception:
        sd = torch.load(shard, map_location="cpu", weights_only=True)

    if key is None:
        key = next(k for k in sd if k.endswith("embed_summary.weight"))
    w = sd[key].clone()
    assert tuple(w.shape) == (summary_length, hidden), \
        f"embed_summary is {tuple(w.shape)}, expected {(summary_length, hidden)}"
    return w


def load_model(name: str, dtype: torch.dtype, device: str):
    from transformers import AutoTokenizer, OPTForCausalLM

    tok = AutoTokenizer.from_pretrained(name)
    try:                                     # transformers >= 5
        model = OPTForCausalLM.from_pretrained(name, dtype=dtype)
    except TypeError:                        # transformers 4.x (cortex-retro has 4.51)
        model = OPTForCausalLM.from_pretrained(name, torch_dtype=dtype)

    summary_length = int(model.config.summary_length)
    accumulate = bool(getattr(model.config, "accumulate_summary", False))
    embed_summary = _load_embed_summary(
        name, summary_length, model.config.hidden_size).to(dtype)

    cfg = {"softprompt_length": 0, "past_key_values_softprompt_length": 0,
           "summary_length": 0}
    dec = model.model.decoder
    dec.embed_positions = PortedOPTPositions(dec.embed_positions.weight.data, cfg)

    model.eval().to(device)          # config has dropout 0.1 -- eval() is required
    return tok, model, embed_summary.to(device), cfg, summary_length, accumulate


@torch.no_grad()
def forward_segment(model, embed_summary, cfg, softprompt, seg_ids,
                    summary_length: int, want_summary: bool):
    """One segment, laid out [softprompt, segment, summary_tokens] under a plain
    causal mask -- auto_compressor.py:85.  Returns (segment logits, new summary)."""
    dev = seg_ids.device
    bsz, _ = seg_ids.shape

    seg_embeds = model.model.decoder.embed_tokens(seg_ids)
    sm_len = summary_length if want_summary else 0
    if sm_len:
        ids = torch.arange(summary_length, device=dev).unsqueeze(0).expand(bsz, -1)
        sm_embeds = F.embedding(ids, embed_summary).to(seg_embeds.dtype)
    else:
        sm_embeds = seg_embeds[:, :0]

    sp_len = softprompt.size(1)
    inputs = torch.cat([softprompt.to(seg_embeds.dtype), seg_embeds, sm_embeds], dim=1)
    mask = torch.ones(bsz, inputs.size(1), dtype=torch.long, device=dev)

    cfg["softprompt_length"] = sp_len
    cfg["past_key_values_softprompt_length"] = 0
    cfg["summary_length"] = sm_len

    out = model.model(inputs_embeds=inputs, attention_mask=mask,
                      use_cache=False, return_dict=True)
    h = out.last_hidden_state
    total = h.size(1)
    seg_h = h[:, sp_len: total - sm_len]
    new_summary = h[:, total - sm_len:] if sm_len else h[:, :0]

    cfg["softprompt_length"] = 0
    cfg["summary_length"] = 0
    return model.lm_head(seg_h).float(), new_summary


def slice_state(state: torch.Tensor, cond: str, sl: int) -> torch.Tensor:
    """Mirrors cortex_memory.chunking.ablate_vec_slice with op='drop': rows are
    write-once and appended in chunk order, so 'oldest' == rows[:n]."""
    n = state.size(1)
    if cond == "carried":
        return state
    if cond == "zeroed":
        return state[:, :0]
    if cond == "drop_oldest1":
        return state[:, sl:] if n > sl else state[:, :0]
    if cond == "drop_newest1":
        return state[:, :-sl] if n > sl else state[:, :0]
    if cond == "keep_newest1":
        return state[:, -sl:]
    if cond == "keep_oldest1":
        return state[:, :sl]
    raise ValueError(cond)


@torch.no_grad()
def sample_losses(model, embed_summary, cfg, ids, n_chunks, chunk_len,
                  summary_length, accumulate):
    """Per-condition, per-chunk mean NLL for one sample.

    The true state is advanced by the `carried` pass only; every other condition
    reads a slice of that same state and writes nothing.  This is the
    read-only-ablation discipline of eval_carry_ablation.py:118-131.
    """
    dev = ids.device
    hidden = model.config.hidden_size
    state = torch.zeros(1, 0, hidden, device=dev, dtype=model.dtype)
    out = {c: [] for c in CONDITIONS}

    for i in range(n_chunks):
        seg = ids[:, i * chunk_len: (i + 1) * chunk_len]
        want_summary = i < n_chunks - 1      # last segment never needs to write

        for cond in CONDITIONS:
            sp = slice_state(state, cond, summary_length)
            # only the true chain needs its summary vectors back
            logits, new_summary = forward_segment(
                model, embed_summary, cfg, sp, seg, summary_length,
                want_summary and cond == "carried")
            # within-chunk next-token loss; identical token set across
            # conditions, so the paired delta is clean
            out[cond].append(
                float(F.cross_entropy(logits[0, :-1], seg[0, 1:], reduction="mean")))
            if cond == "carried" and want_summary:
                nxt = torch.cat([state, new_summary], dim=1) if accumulate else new_summary
        if want_summary:
            state = nxt
    return out


def build_samples(tok, args, need_tokens: int):
    """One contiguous window per PG-19 test book -- no cross-document splices."""
    from datasets import load_dataset

    ds = load_dataset(args.hf_dataset, split=args.split)
    rows = []
    for text in ds[args.text_col]:
        if not text or not text.strip():
            continue
        ids = tok(text, add_special_tokens=False)["input_ids"]
        if len(ids) < args.skip_tokens + need_tokens:
            continue
        rows.append(ids[args.skip_tokens: args.skip_tokens + need_tokens])
        if len(rows) >= args.max_examples:
            break
    return rows


def main() -> int:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32

    print(f"device={device} dtype={args.dtype}")
    tok, model, embed_summary, cfg, summary_length, accumulate = load_model(
        args.model_name, dtype, device)
    print(f"{args.model_name}")
    print(f"summary_length={summary_length}  accumulate_summary={accumulate}")
    if not accumulate:
        print("NOTE: accumulate_summary is False -- the buffer holds one write, "
              "so the slice conditions collapse onto carried/zeroed.")

    need = args.n_chunks * args.chunk_len
    samples = build_samples(tok, args, need)
    print(f"{len(samples)} samples x {need} tokens "
          f"({args.n_chunks} chunks x {args.chunk_len})  data={args.hf_dataset}")
    if not samples:
        print("no samples built -- check --hf_dataset/--split")
        return 1

    per = {c: [[] for _ in range(args.n_chunks)] for c in CONDITIONS}
    t0 = time.time()
    for n, row in enumerate(samples):
        ids = torch.tensor([row], dtype=torch.long, device=device)
        ls = sample_losses(model, embed_summary, cfg, ids, args.n_chunks,
                           args.chunk_len, summary_length, accumulate)
        for cond in CONDITIONS:
            for i, v in enumerate(ls[cond]):
                per[cond][i].append(v)
        el = time.time() - t0
        print(f"  {n + 1}/{len(samples)}  {el:.0f}s elapsed, "
              f"{el / (n + 1) * (len(samples) - n - 1):.0f}s left", flush=True)

    abl = [c for c in CONDITIONS if c != "carried"]
    results, pooled = {}, {c: [] for c in abl}

    hdr = f"  {'chunk':<7}{'vecs':>6}{'carried':>10}" + "".join(f"{c:>14}" for c in abl)
    print("\n  delta = ablated - carried   (positive = those vectors were helping)")
    print(hdr)
    for i in range(args.n_chunks):
        c = per["carried"][i]
        row = {"n": len(c), "carried": st.fmean(c),
               "vecs": i * summary_length if accumulate else min(i, 1) * summary_length}
        line = f"  chunk{i + 1:<2}{row['vecs']:>6}{st.fmean(c):>10.4f}"
        for cond in abl:
            d = [a - b for a, b in zip(per[cond][i], c)]
            row[cond] = st.fmean(per[cond][i])
            row[f"delta_{cond}"] = st.fmean(d)
            row[f"se_{cond}"] = st.stdev(d) / len(d) ** 0.5 if len(d) > 1 else 0.0
            if i > 0:
                pooled[cond].extend(d)
            line += f"{st.fmean(d):>+14.4f}"
        results[f"chunk{i + 1}"] = row
        print(line)

    print("\n  pooled over chunks 2+:")
    for cond in abl:
        d = pooled[cond]
        se = st.stdev(d) / len(d) ** 0.5 if len(d) > 1 else 0.0
        results[f"chunks2plus_{cond}"] = {"n": len(d), "delta": st.fmean(d), "se": se}
        print(f"    {cond:<14}{st.fmean(d):>+10.5f}  +/- {1.96 * se:.5f} (95%)")

    bad = [c for c in abl if results["chunk1"][f"delta_{c}"] != 0.0]
    print(f"\n  chunk-1 control: "
          f"{'OK (all deltas exactly 0)' if not bad else '*** NONZERO: ' + str(bad) + ' -- PORT IS WRONG ***'}")

    # headline: what fraction of the total carry benefit survives on the newest
    # write alone, at the deepest chunk?  (cortex rung1-ep1 chunk8: 92%)
    last = results[f"chunk{args.n_chunks}"]
    total = last["delta_zeroed"]
    if total > 0:
        frac = 1.0 - last["delta_keep_newest1"] / total
        results["headline"] = {
            "chunk": args.n_chunks, "total_carry_benefit": total,
            "cost_of_keeping_only_newest": last["delta_keep_newest1"],
            "fraction_of_benefit_from_newest_write_alone": frac}
        print(f"  chunk{args.n_chunks}: total carry benefit {total:+.4f} nats; "
              f"keeping ONLY the newest {summary_length} vectors costs "
              f"{last['delta_keep_newest1']:+.4f}")
        print(f"  -> the newest write alone is worth {100 * frac:.1f}% of the whole buffer")

    results["config"] = {
        "model": args.model_name, "n_chunks": args.n_chunks,
        "chunk_len": args.chunk_len, "summary_length": summary_length,
        "accumulate_summary": accumulate, "dtype": args.dtype,
        "data": f"{args.hf_dataset}:{args.split}", "skip_tokens": args.skip_tokens,
        "slice_op": "drop", "ablation": "read-only, never compounds into writes"}

    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir, "results.json")
    json.dump(results, open(out, "w"), indent=2)
    print(f"\nSaved -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
