"""
P0.4, the half that was never measured: how many documents in a corpus actually
SPAN the training row, and what a doc-per-row (strided) pack would therefore
yield from it.

WHY THIS EXISTS.  P0.3 (2026-09-15) measured the cross-chunk ceiling within- vs
across-document on the real packs and found something that reframes issue 7:

    per-boundary QUALITY is the same in both corpora; only the SHARE differs.
    FineWeb-Edu's within-document ceiling is +0.23343 at k512, the 50/50 mix's
    is +0.23271 -- indistinguishable.  What separates them is that 70.1% of the
    mix's boundaries stay inside a document against 28.9% of the heal pack's.

So issue 7 is a PACKING problem, not a CORPUS problem, and more PG-19 was never
its natural fix: moving the mix to 60/40 buys +0.00099 nats, while reaching a
100%-within-document pack would buy +0.02103 -- twenty times as much.  The
cheapest route to that is FineWeb-Edu filtered to documents that span the row
(--min_tokens in prepare_pg19_dataset.py already implements it), because that is
the heal distribution and carries the lowest benchmark risk of any option in the
framework's table.

THE CATCH, AND THE ONLY REASON THIS SCRIPT EXISTS.  Doc-per-row packing throws
away every document shorter than one window.  Wrapped packing keeps ~100% of the
tokens it reads; strided doc-per-row keeps only the full windows, and a
1.1k-token FineWeb-Edu document yields ZERO of them.  PG-19 survives this
because its documents are books.  Whether FineWeb-Edu does is an empirical
question nobody has asked, and it cannot be answered from the existing pack --
wrapping already destroyed document identity there.  It has to be counted from
the source shards.

WHAT IT COUNTS, AND WHY IT MATCHES THE PACKER EXACTLY.
prepare_pg19_dataset.py calls stride_windows(ids, row_len, eos) with
row_len = max_length + 1, and that function emits, for a document of L tokens:

    floor(L / row_len)  FULL windows        (mask all ones, row_len real tokens)
    1 RAGGED window     iff L % row_len     (L % row_len + 1 real tokens, EOS-
                                             padded, mask 0 over the padding)

and --min_tokens M then keeps a row iff its mask sums to >= M.  A full window
sums to row_len; a ragged one to (L % row_len) + 1.  So this script reports the
full-window yield, which is what any min_tokens at or near max_length selects,
and it reports the ragged rows separately rather than folding them in.

Counting the full-window yield is NOT the same as counting documents over the
threshold: a 3-window document contributes three rows, not one.  Reporting
"x% of documents exceed 4096 tokens" alone would understate the yield, which is
the mistake this script exists to avoid making by hand.

SAMPLING.  Tokenizing an entire corpus to count lengths is wasteful, so by
default this reads --max_docs documents SPREAD ACROSS ALL LOCAL SHARDS (an equal
slice from each, rather than the first N of shard 0, which would be a single
crawl segment).  Exact total document counts come free from parquet footers --
no column data is read for them -- so the extrapolation denominator is exact and
only the per-document length distribution is sampled.  Pass --max_docs 0 to
count everything.

It reads the SAME local shards prepare_packed_dataset.py reads, resolved the
same way, so the number is about the corpus that is actually on disk.  Download
more shards and re-run if the yield falls short.

    python tools/count_long_docs.py \
        --tokenizer ckpts/olmo-retrofit-cortex \
        --dataset HuggingFaceFW/fineweb-edu --subset 10BT \
        --max_length 4096 --need_rows 470000

--need_rows is the decision: the post-heal mix leg needs 470,000 rows at the
locked D-fallback geometry (26,703 steps x batch_size 32 = 854,496 rows, half
from each leg, plus the 1.10x margin).  Use 366,208 to size a heal replacement.

CPU only, no GPU, no internet (the shards are already cached).  Documents are
small, so unlike the PG-19 pack this does not need a compute node -- but if a
login node kills it, pace/prepare_pg19_pack.sbatch is the template to copy.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_packed_dataset import _subset_files          # noqa: E402


def resolve_shards(dataset, subset, parquet_glob):
    """The same resolution prepare_packed_dataset.py uses, so the count is over
    the same bytes the pack would be built from."""
    import glob as globlib
    if parquet_glob:
        return sorted(globlib.glob(parquet_glob))
    from huggingface_hub import try_to_load_from_cache
    files = []
    for f in _subset_files(dataset, subset):
        local = try_to_load_from_cache(dataset, f, repo_type="dataset")
        if isinstance(local, str):
            files.append(local)
    return files


def yield_rows(lengths, row_len):
    """(full-mask rows, ragged rows, tokens in the full rows) for these docs.

    Mirrors stride_windows exactly, INCLUDING its boundary case.  The obvious
    reading -- full = floor(L / row_len), ragged = 1 iff L % row_len -- is wrong
    once: when the remainder is exactly row_len - 1, the document-end EOS
    stride_windows appends fills the window, so `mask` comes out all ones and
    that row is a FULL row, not a ragged one.  tests/test_count_long_docs.py
    pins this against the packer itself rather than against this docstring.

    It is a 1-in-row_len event and changes no verdict at 4,097, but the point of
    the script is that its arithmetic IS the packer's; a counter that drifts
    from the thing it predicts is worse than no counter.

    A document shorter than one window contributes 0 full rows and 1 ragged --
    which is precisely the volume doc-per-row packing gives up against wrapping,
    and the whole question this script exists to answer.
    """
    full = sum(L // row_len + (1 if L % row_len == row_len - 1 else 0)
               for L in lengths)
    ragged = sum(1 for L in lengths if L % row_len not in (0, row_len - 1))
    return full, ragged, full * row_len


def pct(xs, q):
    if not xs:
        return 0
    s = sorted(xs)
    return s[min(len(s) - 1, int(q * len(s)))]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--dataset", default="HuggingFaceFW/fineweb-edu")
    ap.add_argument("--subset", default="10BT")
    ap.add_argument("--parquet_glob", default=None)
    ap.add_argument("--text_col", default="text")
    ap.add_argument("--max_length", type=int, default=4096)
    ap.add_argument("--max_docs", type=int, default=200_000,
                    help="documents to tokenize, spread evenly across shards "
                         "(0 = all).  The extrapolation denominator is exact "
                         "either way; only the length distribution is sampled.")
    ap.add_argument("--batch", type=int, default=1000)
    ap.add_argument("--need_rows", type=int, default=470_000,
                    help="the row budget this leg has to fill.  470,000 = the "
                         "post-heal mix leg at D-fallback; 366,208 = the heal.")
    args = ap.parse_args()

    import pyarrow.parquet as pq
    from transformers import AutoTokenizer

    files = resolve_shards(args.dataset, args.subset, args.parquet_glob)
    assert files, ("no local parquet shards found -- download some first (see "
                   "pace/b2_retrofit.sbatch's header) or pass --parquet_glob")

    # Exact totals from the parquet FOOTERS -- no column data is read for this.
    totals = [pq.ParquetFile(f).metadata.num_rows for f in files]
    n_docs_total = sum(totals)
    print("%s[%s]: %d local shard(s), %s documents"
          % (args.dataset, args.subset, len(files), format(n_docs_total, ",")))

    per_shard = (n_docs_total if args.max_docs == 0
                 else max(1, args.max_docs // len(files)))
    row_len = args.max_length + 1
    tok = AutoTokenizer.from_pretrained(args.tokenizer)

    lengths = []
    for f, n_in_shard in zip(files, totals):
        want = min(per_shard, n_in_shard)
        got = 0
        pf = pq.ParquetFile(f)
        for batch in pf.iter_batches(batch_size=args.batch,
                                     columns=[args.text_col]):
            texts = batch.column(args.text_col).to_pylist()
            if got + len(texts) > want:
                texts = texts[:want - got]
            enc = tok(texts, add_special_tokens=False)["input_ids"]
            lengths.extend(len(ids) for ids in enc)
            got += len(texts)
            if got >= want:
                break
        print("  %s: sampled %s of %s"
              % (Path(f).name, format(got, ","), format(n_in_shard, ",")))

    n = len(lengths)
    assert n, "no documents read"
    scanned_tokens = sum(lengths)
    scale = n_docs_total / n

    print("\ndocuments tokenized : %s  (%.2f%% of the corpus)"
          % (format(n, ","), 100 / scale))
    print("tokens in them      : %.3fB  (mean %s tokens/doc)"
          % (scanned_tokens / 1e9, format(scanned_tokens // n, ",")))
    print("\ndocument length percentiles (tokens)")
    for q in (0.50, 0.75, 0.90, 0.95, 0.99):
        print("  p%-3d %9s" % (int(q * 100), format(pct(lengths, q), ",")))
    print("  max  %9s" % format(max(lengths), ","))

    print("\ndocuments that SPAN a %d-token row (>= %s tokens)"
          % (args.max_length, format(row_len, ",")))
    for m in (row_len // 2, row_len, 2 * row_len, 4 * row_len):
        k = sum(1 for L in lengths if L >= m)
        t = sum(L for L in lengths if L >= m)
        print("  >= %6s tok : %8s docs (%5.2f%%)   holding %.3fB tok "
              "(%5.2f%% of scanned)"
              % (format(m, ","), format(k, ","), 100 * k / n, t / 1e9,
                 100 * t / scanned_tokens))

    full, ragged, kept = yield_rows(lengths, row_len)
    print("\nDOC-PER-ROW YIELD at max_length %d (row_len %s), matching "
          "stride_windows" % (args.max_length, format(row_len, ",")))
    print("  full windows  : %9s rows   (%.3fB tokens = %.1f%% of the tokens "
          "read)" % (format(full, ","), kept / 1e9,
                     100 * kept / scanned_tokens))
    print("  ragged rows   : %9s  (one per document; most are far below "
          "min_tokens and would be dropped)" % format(ragged, ","))
    print("  rows/document : %.2f" % (full / n))

    est = full * scale
    print("\nEXTRAPOLATED to all %s local documents" % format(n_docs_total, ","))
    print("  full windows  : %12s rows" % format(int(est), ","))
    print("  need          : %12s rows" % format(args.need_rows, ","))
    margin = est / args.need_rows if args.need_rows else float("inf")
    print("  margin        : %12.2fx" % margin)

    if margin >= 1.10:
        print("\n  VERDICT: the long-doc leg is PURCHASABLE from the shards on "
              "disk.")
        print("  Build it with prepare_pg19_dataset.py --dataset %s "
              "--min_tokens %d," % (args.dataset, args.max_length))
        print("  then RE-RUN P0.3 on the result before trusting the share -- "
              "the yield")
        print("  says the rows exist, not that they carry cross-chunk signal.")
    else:
        short = args.need_rows * 1.10 / max(est, 1)
        print("\n  VERDICT: NOT purchasable from these shards -- short by "
              "%.1fx at a 1.10 margin." % short)
        print("  Either download ~%.1fx more shards, or drop the long-doc "
              "packing option" % short)
        print("  and keep the 50/50 wrapped mix, which P0.3 priced at only "
              "-0.021 nats")
        print("  against a perfect-within-document pack.  Do not split the "
              "difference silently.")

    print("\n  Both numbers assume the tokenizer above.  A different tokenizer "
          "shifts every")
    print("  length and therefore the yield; the packer must be run with the "
          "same one.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
