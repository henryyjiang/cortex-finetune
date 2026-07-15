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

Two-step usage (PACE: compute nodes have no internet, and the login node's
memory cap kills the `datasets` streaming parquet reader — so download plain
files first, then pack from local shards with the low-memory pyarrow reader):

    # login node, step 1 — list the subset's shards and download a few
    # (~1B tokens/shard; 1-2 shards cover a 130M-token target):
    python tools/prepare_packed_dataset.py --list_files \
        --dataset nvidia/Nemotron-CC-Math-v1 --subset 4plus
    huggingface-cli download nvidia/Nemotron-CC-Math-v1 --repo-type dataset \
        --include "<first-shard-path-from-the-listing>"

    # login node, step 2 — tokenize + pack from the downloaded shards:
    python tools/prepare_packed_dataset.py \
        --tokenizer ckpts/olmo8-cortex \
        --out data/nemotron_math_olmo_len1024 \
        --max_length 1024 --max_tokens 130_000_000

(step 2 finds the downloaded shards in the HF cache automatically; set
HF_HOME=$SCRATCH/hf_cache in both steps.  --parquet_glob overrides with an
explicit local path glob.)

Then train with:
    python train.py --preprocessed_data_path data/nemotron_math_olmo_len1024 \
        --max_length 1024 ...
"""
from __future__ import annotations

import argparse
import glob as globlib


def _subset_files(dataset: str, subset: str) -> list[str]:
    """Parquet shard paths of one subset, by exact path-component match
    (substring would also catch 4plus_MIND when asked for 4plus)."""
    from huggingface_hub import HfApi
    return sorted(
        f for f in HfApi().list_repo_files(dataset, repo_type="dataset")
        if f.endswith(".parquet")
        and (not subset or subset in f.replace("-", "/").split("/"))
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tokenizer", help="tokenizer source — use the graft-prepared model dir")
    ap.add_argument("--out", help="output dataset dir (save_to_disk)")
    ap.add_argument("--dataset", default="nvidia/Nemotron-CC-Math-v1")
    ap.add_argument("--subset", default="4plus",
                    help="subset dir to match in shard paths ('' = all shards)")
    ap.add_argument("--list_files", action="store_true",
                    help="print the subset's shard paths and exit")
    ap.add_argument("--parquet_glob", default=None,
                    help="explicit local parquet glob (skips the HF cache lookup)")
    ap.add_argument("--text_col", default="text")
    ap.add_argument("--max_length", type=int, default=1024,
                    help="tokens per training sequence; rows are max_length+1 long")
    ap.add_argument("--max_tokens", type=int, default=130_000_000,
                    help="stop after packing this many tokens (the corpus is 133B)")
    ap.add_argument("--seed", type=int, default=74)
    args = ap.parse_args()

    if args.list_files:
        for f in _subset_files(args.dataset, args.subset):
            print(f)
        return 0
    assert args.tokenizer and args.out, "--tokenizer and --out are required"

    import pyarrow.parquet as pq
    from datasets import Dataset
    from transformers import AutoTokenizer

    # resolve local shard files: explicit glob, else whatever shards of the
    # subset are already in the HF cache (downloaded via huggingface-cli)
    if args.parquet_glob:
        files = sorted(globlib.glob(args.parquet_glob))
    else:
        from huggingface_hub import try_to_load_from_cache
        files = []
        for f in _subset_files(args.dataset, args.subset):
            local = try_to_load_from_cache(args.dataset, f, repo_type="dataset")
            if isinstance(local, str):
                files.append(local)
    assert files, ("no local parquet shards found — download some first "
                   "(see this file's header) or pass --parquet_glob")
    print(f"packing from {len(files)} local shard(s):")
    for f in files:
        print(f"  {f}")

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    eos = tok.eos_token_id
    assert eos is not None, "tokenizer has no eos token"
    bos = tok.bos_token_id  # may be None (OLMo-2 has none); prepended when present
    row_len = args.max_length + 1

    target_rows = args.max_tokens // args.max_length
    stats = {"docs": 0, "rows": 0}

    def gen():
        # generator-backed Dataset: rows go straight to arrow, and pyarrow
        # iter_batches reads small slices — neither the corpus nor the token
        # stream ever sits in memory
        carry: list[int] = []      # token remainder wrapping into the next row
        for path in files:
            pf = pq.ParquetFile(path)
            for batch in pf.iter_batches(batch_size=64, columns=[args.text_col]):
                for text in batch.column(0).to_pylist():
                    if not text:
                        continue
                    ids = tok(text, add_special_tokens=False)["input_ids"]
                    if bos is not None:
                        ids = [bos] + ids
                    carry.extend(ids)
                    carry.append(eos)  # doc separator (eos_from_tokens-compatible)
                    stats["docs"] += 1
                    while len(carry) >= row_len:
                        yield {"input_ids": carry[:row_len],
                               "attention_mask": [1] * row_len}
                        carry = carry[row_len:]
                        stats["rows"] += 1
                        if stats["rows"] >= target_rows:
                            return
                    if stats["docs"] % 20_000 == 0:
                        print(f"  {stats['docs']} docs -> "
                              f"{stats['rows']}/{target_rows} rows", flush=True)

    packed = Dataset.from_generator(gen)
    # rows are packed in corpus order — shuffle so an epoch isn't
    # snapshot/domain-ordered (cheap: arrow-backed index permutation)
    packed = packed.shuffle(seed=args.seed)
    assert len(packed), "no rows produced — check --dataset/--subset/--text_col"
    packed.save_to_disk(args.out)
    print(f"saved {len(packed)} rows x {row_len} tokens "
          f"({len(packed) * args.max_length / 1e6:.0f}M tokens from "
          f"{stats['docs']} docs) -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
