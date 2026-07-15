"""
One-time (login-node) tokenization of a short-document corpus into WRAPPED-
PACKED rows in the format train.py's non-parquet `--preprocessed_data_path`
branch expects (same schema as tools/prepare_pg19_dataset.py):

      input_ids      : int64 [max_length + 1]   (train.py does [:, :-1] / [:, 1:])
      attention_mask : int64 [max_length + 1]   (all 1 — packed rows have no pads)

Wrapped packing (McLeish's continued-pretraining recipe, shells/olmo.sh): all
documents are tokenized, joined with EOS separators, and the resulting stream
is sliced into fixed-length rows — documents wrap across row boundaries, no
padding.  This reproduces the base RDM's training distribution (the model was
continued-pretrained on packed nemotron-cc-math at max_length=1024), which is
what makes it REPLAY data for the dense/no-boundary-crossing final arms:
in-distribution content at the in-distribution window length.

Default corpus is McLeish's: nvidia/Nemotron-CC-Math-v1, subset 4plus
(quality-4-and-5 documents; his dataset name was
olmo_2_0425_1b_packed_nemotron_cc_math_v1_4plus_wrapped_packing).  The full
corpus is 133B tokens — cap with --max_tokens (streamed; only the cap is
downloaded/tokenized).

Run on a LOGIN node (needs internet):

    python tools/prepare_packed_dataset.py \
        --tokenizer ckpts/olmo8-cortex \
        --out data/nemotron_math_olmo_len1024 \
        --max_length 1024 --max_tokens 130_000_000

Then train with:
    python train.py --preprocessed_data_path data/nemotron_math_olmo_len1024 \
        --max_length 1024 ...
"""
from __future__ import annotations

import argparse


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tokenizer", required=True,
                    help="tokenizer source — use the graft-prepared model dir")
    ap.add_argument("--out", required=True, help="output dataset dir (save_to_disk)")
    ap.add_argument("--dataset", default="nvidia/Nemotron-CC-Math-v1")
    ap.add_argument("--subset", default="4plus",
                    help="dataset config name (None/'' = default config)")
    ap.add_argument("--split", default="train")
    ap.add_argument("--text_col", default="text")
    ap.add_argument("--max_length", type=int, default=1024,
                    help="tokens per training sequence; rows are max_length+1 long")
    ap.add_argument("--max_tokens", type=int, default=130_000_000,
                    help="stop after packing this many tokens (the corpus is 133B)")
    ap.add_argument("--shuffle_buffer", type=int, default=10_000,
                    help="streaming shuffle buffer (0 = no shuffle)")
    ap.add_argument("--seed", type=int, default=74)
    args = ap.parse_args()

    from datasets import Dataset, load_dataset
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    eos = tok.eos_token_id
    assert eos is not None, "tokenizer has no eos token"
    bos = tok.bos_token_id  # may be None (OLMo-2 has none); prepended when present
    row_len = args.max_length + 1

    ds = load_dataset(args.dataset, args.subset or None, split=args.split,
                      streaming=True)
    if args.shuffle_buffer:
        ds = ds.shuffle(seed=args.seed, buffer_size=args.shuffle_buffer)

    target_rows = args.max_tokens // args.max_length
    stats = {"docs": 0, "rows": 0}

    def gen():
        # generator-backed Dataset: rows go straight to arrow, so the 100M+
        # token stream never sits in Python memory
        carry: list[int] = []      # token remainder wrapping into the next row
        for ex in ds:
            text = ex[args.text_col]
            if not text:
                continue
            ids = tok(text, add_special_tokens=False)["input_ids"]
            if bos is not None:
                ids = [bos] + ids
            carry.extend(ids)
            carry.append(eos)      # document separator (eos_from_tokens-compatible)
            stats["docs"] += 1
            while len(carry) >= row_len:
                yield {"input_ids": carry[:row_len],
                       "attention_mask": [1] * row_len}
                carry = carry[row_len:]
                stats["rows"] += 1
                if stats["rows"] >= target_rows:
                    return
            if stats["docs"] % 20_000 == 0:
                print(f"  {stats['docs']} docs -> {stats['rows']}/{target_rows} rows",
                      flush=True)

    packed = Dataset.from_generator(gen)
    assert len(packed), "no rows produced — check --dataset/--subset/--text_col"
    packed.save_to_disk(args.out)
    print(f"saved {len(packed)} rows x {row_len} tokens "
          f"({len(packed) * args.max_length / 1e6:.0f}M tokens from "
          f"{stats['docs']} docs) -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
