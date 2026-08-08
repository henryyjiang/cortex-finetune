"""
Corpus mix — interleave two already-prepared datasets into one drop-in dataset
for train.py (same columns/shapes; point DATA_PATH at the output).

WHY THIS EXISTS (2026-08-05).  Every Nemotron result on the books is a corpus
SWAP, not a mix: the `-nemo` arms replaced PG-19 outright.  That moved two
things at once — GSM8K recovered (10.4 -> 14.0 on acc4v, 0.8 -> 39.6 on ki4)
but the write statistics also moved hard (new-vs-old cos 0.94 -> 0.53, write
norm 5.78 -> 1.03) and the carry delta FELL (0.0237 -> 0.0205).  So we cannot
say whether Nemotron helped as replay or merely as a different corpus.  A mix
holds the memory-bearing corpus fixed and adds replay on top, which is the
question B2's arm-phase data recipe actually needs answered.

PACKING CAVEAT — READ BEFORE QUOTING A RESULT.  PG-19 packs are ONE DOCUMENT
PER ROW (prepare_pg19_dataset.py); Nemotron packs are WRAPPED (documents joined
with EOS and sliced across row boundaries, prepare_packed_dataset.py).  A mix
therefore contains BOTH packing schemes, and the ceil2 probes showed packing is
itself a ceiling lever (Nemotron peaks at +0.065 vs PG-19's +0.104 at k256,
partly for packing reasons).  Rows are not re-packed here — mixing is at row
granularity only.  Report the mix as "PG-19 (doc-per-row) + Nemotron (wrapped)".

Rows are interleaved by sampling without replacement to hit --ratio, then
shuffled once with --seed, so the mix is uniform across the epoch rather than
front-loaded with one corpus.

Run on a LOGIN node (disk only, no downloads, no GPU):

    python tools/prepare_corpus_mix.py \
        --a data/pg19_olmo_len4096 --b data/nemotron_math_olmo_len4096_4b \
        --out data/pg19_nemo50_olmo_len4096 --ratio 0.5

--ratio is B's share of the OUTPUT rows.  The script takes all of A and draws
the matching number of B rows (or caps A if B is the scarcer corpus), so the
output size is set by whichever side runs out first; it prints the arithmetic
before writing.  Use --max_rows to pin a token budget across arms.
"""
from __future__ import annotations

import argparse
import sys

from datasets import load_from_disk, concatenate_datasets


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", required=True, help="prepared dataset dir (corpus A)")
    ap.add_argument("--b", required=True, help="prepared dataset dir (corpus B)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ratio", type=float, default=0.5,
                    help="B's share of output rows (0..1); default 0.5")
    ap.add_argument("--max_rows", type=int, default=0,
                    help="cap total output rows (0 = no cap) — use to hold the "
                         "token budget fixed across arms")
    ap.add_argument("--seed", type=int, default=74)
    args = ap.parse_args()

    if not 0.0 < args.ratio < 1.0:
        print(f"ERROR: --ratio must be strictly between 0 and 1; got {args.ratio}",
              file=sys.stderr)
        return 1

    ds_a = load_from_disk(args.a)
    ds_b = load_from_disk(args.b)

    # Shape compatibility is load-bearing: train.py assumes a fixed row length,
    # and a silent mismatch would produce ragged chunks under cross_chunks.
    cols_a, cols_b = sorted(ds_a.column_names), sorted(ds_b.column_names)
    if cols_a != cols_b:
        print(f"ERROR: column mismatch\n  A {cols_a}\n  B {cols_b}", file=sys.stderr)
        return 1
    len_a, len_b = len(ds_a[0]["input_ids"]), len(ds_b[0]["input_ids"])
    if len_a != len_b:
        print(f"ERROR: row length mismatch — A {len_a} tokens, B {len_b}. "
              f"Re-prep one side at the other's --max_length.", file=sys.stderr)
        return 1

    # Output size is bounded by whichever corpus runs out first at --ratio.
    n_from_a = min(len(ds_a), int(len(ds_b) * (1 - args.ratio) / args.ratio))
    n_from_b = min(len(ds_b), int(n_from_a * args.ratio / (1 - args.ratio)))
    if args.max_rows:
        scale = min(1.0, args.max_rows / (n_from_a + n_from_b))
        n_from_a, n_from_b = int(n_from_a * scale), int(n_from_b * scale)

    total = n_from_a + n_from_b
    print(f"A {args.a}: {len(ds_a):,} rows -> taking {n_from_a:,}")
    print(f"B {args.b}: {len(ds_b):,} rows -> taking {n_from_b:,}")
    print(f"mix: {total:,} rows x {len_a} tokens = {total * len_a / 1e6:.1f}M tokens "
          f"(B share {n_from_b / total:.3f}, requested {args.ratio})")

    ds_a = ds_a.shuffle(seed=args.seed).select(range(n_from_a))
    ds_b = ds_b.shuffle(seed=args.seed).select(range(n_from_b))
    mixed = concatenate_datasets([ds_a, ds_b]).shuffle(seed=args.seed)
    mixed = mixed.flatten_indices()
    # num_proc is NOT optional at this size.  shuffle() only attaches an index
    # permutation, so save_to_disk materialises ~1M rows of ~20KB each in random
    # order across two source arrow files on Lustre — a million seeks, ~14h
    # single-process.  Same bottleneck already fixed in prepare_packed_dataset.py.
    # Keep the shuffle: it is what stops an epoch being corpus-ordered.
    mixed.save_to_disk(args.out, num_proc=8)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
