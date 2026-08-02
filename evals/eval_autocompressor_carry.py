"""
Harness validation: run OUR carry-vs-zeroed ablation against a model whose
memory is KNOWN to work — princeton-nlp/AutoCompressor-2.7b-6k (EMNLP'23,
"Adapting Language Models to Compress Long Contexts").

Why this exists
---------------
`evals/eval_carry_ablation.py` reported delta = -0.0128 for retro-b1 (carry
HURTS).  That is ambiguous between "the memory is broken" and "the detector is
broken", because the detector has never been run on a positive control.  This
script closes that: same paired carried-vs-zeroed protocol, same delta sign
convention, same results.json schema — on a released model that demonstrably
carries context across segments.

    delta = zeroed - carried   (POSITIVE = the carry HELPS)

Expected: a large POSITIVE delta on chunks 2+.  Chunk 1 must be exactly 0.0
(no softprompt exists yet under either condition) — same built-in control as
our own harness.  If chunk 1 is nonzero, the port is wrong; if chunks 2+ come
back ~0, the protocol itself cannot see carried memory and every carry number
we have needs re-reading.

Why this is a re-implementation and not an import
------------------------------------------------
`AutoCompressors-main/auto_compressor.py` imports `modeling_flash_llama`
(needs flash-attn, not available on Windows) at module scope, and its mixin
targets transformers 4.34.  So the OPT forward is ported here directly.  The
port is faithful on the three things that matter:

  1. Summary vectors are produced by appending `summary_length` learned
     summary-token embeddings to the END of the segment and taking the output
     hidden states at those positions (auto_compressor.py:85,123).
  2. They are consumed by PREPENDING them to the next segment's input
     embeddings — through the model's own self-attention, no separate read
     module (auto_compressor.py:85).
  3. Softprompt and summary positions get placeholder position id 1 (the pad
     row), i.e. no real position (auto_compressor.py:294-306).  Verified that
     transformers 5.8.1 computes real-token positions as cumsum*mask + 1 too
     (modeling_opt.py:64-70 with self.offset = 2), so the arithmetic matches
     the pinned 4.34 exactly.

`accumulate_summary` is read from the checkpoint config (true for this model).

Usage (see the header of the PowerShell block in the handoff notes):
    python evals/eval_autocompressor_carry.py --out_dir eval_results/ac_control
"""
from __future__ import annotations

import argparse
import json
import os

import torch
import torch.nn.functional as F
from torch import nn


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("AutoCompressor carry-vs-zeroed positive control")
    p.add_argument("--model_name", default="princeton-nlp/AutoCompressor-2.7b-6k")
    p.add_argument("--n_chunks", type=int, default=4,
                   help="segments per sample (the 6k model was trained at 4)")
    p.add_argument("--chunk_len", type=int, default=1536,
                   help="tokens per segment (4 x 1536 = the 6k training shape)")
    p.add_argument("--max_examples", type=int, default=50,
                   help="samples; 50 matches the n our own gate ran at")
    p.add_argument("--hf_dataset", default="wikitext",
                   help="Wikipedia is one of their four training domains")
    p.add_argument("--hf_config", default="wikitext-103-raw-v1")
    p.add_argument("--split", default="test")
    p.add_argument("--text_col", default="text")
    p.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16"])
    p.add_argument("--out_dir", default="eval_results/ac_control")
    return p.parse_args()


class PortedOPTPositions(nn.Module):
    """auto_compressor.py:281-306, ported to the transformers 5.x call signature.

    Real tokens get cumsum*mask + 1; softprompt and summary slots get 1 (the
    pad row), i.e. no position.  The `position_ids` transformers computes for
    us is deliberately ignored — it has no idea about the placeholder slots.
    """

    def __init__(self, weight: torch.Tensor, cfg: dict):
        super().__init__()
        self.weight = weight
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


def load_model(name: str, dtype: torch.dtype, device: str):
    """Stock OPTForCausalLM + the checkpoint's embed_summary + the ported
    positional embedding.  Loading through the stock class drops
    embed_summary.weight (an unexpected key), so it is fetched separately."""
    from transformers import AutoTokenizer, OPTForCausalLM

    tok = AutoTokenizer.from_pretrained(name)
    model = OPTForCausalLM.from_pretrained(name, dtype=dtype)

    summary_length = int(model.config.summary_length)
    accumulate = bool(getattr(model.config, "accumulate_summary", False))

    embed_summary = _load_embed_summary(name, summary_length,
                                        model.config.hidden_size).to(dtype)

    cfg = {"softprompt_length": 0, "past_key_values_softprompt_length": 0,
           "summary_length": 0}
    dec = model.model.decoder
    dec.embed_positions = PortedOPTPositions(dec.embed_positions.weight.data, cfg)

    model.eval().to(device)          # config has dropout 0.1 — eval() is required
    embed_summary = embed_summary.to(device)
    return tok, model, embed_summary, cfg, summary_length, accumulate


def _load_embed_summary(name: str, summary_length: int, hidden: int) -> torch.Tensor:
    from huggingface_hub import hf_hub_download

    index = json.load(open(hf_hub_download(name, "pytorch_model.bin.index.json")))
    key = next(k for k in index["weight_map"] if k.endswith("embed_summary.weight"))
    shard = hf_hub_download(name, index["weight_map"][key])
    w = torch.load(shard, map_location="cpu", weights_only=True)[key]
    assert tuple(w.shape) == (summary_length, hidden), \
        f"embed_summary is {tuple(w.shape)}, expected {(summary_length, hidden)}"
    return w


@torch.no_grad()
def forward_segment(model, embed_summary, cfg, softprompt, seg_ids,
                    summary_length: int, want_summary: bool):
    """One segment. Returns (segment logits, new summary vectors).

    Layout is [softprompt, segment, summary_tokens] under a plain causal mask —
    exactly auto_compressor.py:85. Segment tokens see every softprompt vector;
    summary tokens see everything.
    """
    dev = seg_ids.device
    bsz, seg_len = seg_ids.shape

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


@torch.no_grad()
def chunk_losses(model, embed_summary, cfg, ids, n_chunks, chunk_len,
                 summary_length, accumulate, carried: bool):
    """Per-chunk mean NLL for one sample. `carried=False` is the ablation:
    the softprompt is discarded between segments (their `softprompt=None`)."""
    dev = ids.device
    hidden = model.config.hidden_size
    softprompt = torch.zeros(1, 0, hidden, device=dev, dtype=model.dtype)
    losses = []

    for i in range(n_chunks):
        seg = ids[:, i * chunk_len: (i + 1) * chunk_len]
        want_summary = i < n_chunks - 1          # last segment needs no summary
        logits, new_summary = forward_segment(
            model, embed_summary, cfg, softprompt, seg, summary_length, want_summary)

        # within-chunk next-token loss; identical token set across conditions,
        # so the paired delta is clean
        ce = F.cross_entropy(logits[0, :-1], seg[0, 1:], reduction="mean")
        losses.append(float(ce))

        if not carried:
            softprompt = torch.zeros(1, 0, hidden, device=dev, dtype=model.dtype)
        elif accumulate:
            softprompt = torch.cat([softprompt, new_summary], dim=1)
        else:
            softprompt = new_summary
    return losses


def build_samples(tok, args, need_tokens: int):
    """Concatenate the corpus and cut it into fixed-length rows."""
    from datasets import load_dataset

    ds = load_dataset(args.hf_dataset, args.hf_config, split=args.split)
    rows, buf = [], []
    for text in ds[args.text_col]:
        if not text or not text.strip():
            continue
        buf.extend(tok(text, add_special_tokens=False)["input_ids"])
        while len(buf) >= need_tokens:
            rows.append(buf[:need_tokens])
            buf = buf[need_tokens:]
            if len(rows) >= args.max_examples:
                return rows
    return rows


def main() -> int:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32

    print(f"device={device} dtype={args.dtype}")
    tok, model, embed_summary, cfg, summary_length, accumulate = load_model(
        args.model_name, dtype, device)
    print(f"summary_length={summary_length}  accumulate_summary={accumulate}")

    need = args.n_chunks * args.chunk_len
    samples = build_samples(tok, args, need)
    print(f"{len(samples)} samples x {need} tokens "
          f"({args.n_chunks} chunks x {args.chunk_len})")
    if not samples:
        print("no samples built — check --hf_dataset/--split")
        return 1

    per_chunk = {"carried": [[] for _ in range(args.n_chunks)],
                 "zeroed":  [[] for _ in range(args.n_chunks)]}

    for n, row in enumerate(samples):
        ids = torch.tensor([row], dtype=torch.long, device=device)
        for cond in ("carried", "zeroed"):
            ls = chunk_losses(model, embed_summary, cfg, ids, args.n_chunks,
                              args.chunk_len, summary_length, accumulate,
                              carried=(cond == "carried"))
            for i, v in enumerate(ls):
                per_chunk[cond][i].append(v)
        if (n + 1) % 5 == 0:
            print(f"  {n + 1}/{len(samples)}", flush=True)

    # paired stats, same convention as evals/eval_carry_ablation.py:195
    #   delta = zeroed - carried   (positive = the carry helps)
    import statistics as st

    results, all_d = {}, []
    print(f"\n  {'Chunk':<8}{'N':>5}{'carried':>11}{'zeroed':>11}{'delta':>11}{'se':>10}")
    for i in range(args.n_chunks):
        c, z = per_chunk["carried"][i], per_chunk["zeroed"][i]
        d = [zz - cc for zz, cc in zip(z, c)]
        se = st.stdev(d) / (len(d) ** 0.5) if len(d) > 1 else 0.0
        results[f"chunk{i + 1}"] = {
            "n": len(c), "carried": st.fmean(c), "zeroed": st.fmean(z),
            "delta": st.fmean(d), "se": se}
        if i > 0:
            all_d.extend(d)
        print(f"  chunk{i + 1:<3}{len(c):>8}{st.fmean(c):>11.4f}"
              f"{st.fmean(z):>11.4f}{st.fmean(d):>+11.4f}{se:>10.4f}")

    se_all = st.stdev(all_d) / (len(all_d) ** 0.5) if len(all_d) > 1 else 0.0
    results["chunks2plus"] = {"n": len(all_d), "delta": st.fmean(all_d), "se": se_all}
    results["config"] = {
        "model": args.model_name, "n_chunks": args.n_chunks,
        "chunk_len": args.chunk_len, "summary_length": summary_length,
        "accumulate_summary": accumulate,
        "data": f"{args.hf_dataset}/{args.hf_config}:{args.split}"}

    d0 = results["chunk1"]["delta"]
    print(f"\n  chunks 2+: delta = {st.fmean(all_d):+.5f} +/- {1.96 * se_all:.5f} (95%)"
          f"   [positive = the carry helps]")
    print(f"  chunk-1 control: {d0:+.6f}   "
          f"{'OK (exactly 0 as required)' if d0 == 0.0 else '*** NONZERO — PORT IS WRONG ***'}")

    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir, "results.json")
    json.dump(results, open(out, "w"), indent=2)
    print(f"\nSaved -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
