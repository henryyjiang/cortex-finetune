"""
One-time (login-node) tokenization of a long-document corpus into the format
train.py's non-parquet `--preprocessed_data_path` branch expects:

    an HF dataset on disk with columns
      input_ids      : int64 [max_length + 1]   (train.py does [:, :-1] / [:, 1:])
      attention_mask : int64 [max_length + 1]   (1 = real token, 0 = pad)

One document per row (the cortex first-pass data design: the cross-chunk carry
is document-correct with no packing).  Documents longer than the window are
truncated; shorter ones get one EOS appended (so --cortex.eos_from_tokens can
detect the document end) and are padded with EOS to full length with
attention_mask 0 over the padding (pads are excluded from the loss via the
mask, and from the M_cross write via eos_from_tokens).

Default corpus is PG-19 (deepmind/pg19, books) — almost every document fills a
4096-token window, which is exactly what makes the M_cross carry load-bearing.

Run on a LOGIN node (needs internet for the first download; ~11 GB for PG-19):

    python tools/prepare_pg19_dataset.py \
        --tokenizer ckpts/olmo8-cortex \
        --out data/pg19_olmo_len4096 \
        --max_length 4096

Then train with:
    python train.py --preprocessed_data_path data/pg19_olmo_len4096 --max_length 4096 ...
"""
from __future__ import annotations

import argparse


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tokenizer", required=True,
                    help="tokenizer source — use the graft-prepared model dir")
    ap.add_argument("--out", required=True, help="output dataset dir (save_to_disk)")
    ap.add_argument("--dataset", default="emozilla/pg19")  # complete parquet mirror; the deepmind original is script-based (datasets>=3.x refuses)
    ap.add_argument("--split", default="train")
    ap.add_argument("--text_col", default="text")
    ap.add_argument("--max_length", type=int, default=4096,
                    help="tokens per training sequence; rows are max_length+1 long")
    ap.add_argument("--max_samples", type=int, default=None,
                    help="cap the number of documents (None = all)")
    ap.add_argument("--num_proc", type=int, default=16)
    args = ap.parse_args()

    from datasets import load_dataset
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    eos = tok.eos_token_id
    assert eos is not None, "tokenizer has no eos token"
    bos = tok.bos_token_id  # may be None (OLMo-2 has none); prepended when present
    row_len = args.max_length + 1
    # bound tokenization cost: ~4 chars/token in English, 8x is ample margin
    char_cap = row_len * 8

    ds = load_dataset(args.dataset, split=args.split)
    if args.max_samples is not None:
        ds = ds.select(range(min(args.max_samples, len(ds))))
    print(f"{args.dataset}[{args.split}]: {len(ds)} documents")

    def tokenize(batch):
        out_ids, out_mask = [], []
        for text in batch[args.text_col]:
            ids = tok(text[:char_cap], add_special_tokens=False)["input_ids"]
            if bos is not None:
                ids = [bos] + ids
            if len(ids) >= row_len:
                ids = ids[:row_len]
                mask = [1] * row_len
            else:
                ids = ids + [eos]                      # mark the document end
                mask = [1] * len(ids)
                pad = row_len - len(ids)
                ids += [eos] * pad                     # eos-pad → eos_from_tokens
                mask += [0] * pad                      # pads excluded from loss
            out_ids.append(ids)
            out_mask.append(mask)
        return {"input_ids": out_ids, "attention_mask": out_mask}

    tokenized = ds.map(tokenize, batched=True, batch_size=64,
                       num_proc=args.num_proc, remove_columns=ds.column_names)
    tokenized.save_to_disk(args.out)

    n_full = sum(1 for m in tokenized[:1000]["attention_mask"] if m[-1] == 1)
    print(f"saved {len(tokenized)} rows x {row_len} tokens -> {args.out}")
    print(f"window-filling documents (first 1000 sampled): {n_full}/1000")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
