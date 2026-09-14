"""
One-time (login-node) tokenization of a long-document corpus into the format
train.py's non-parquet `--preprocessed_data_path` branch expects:

    an HF dataset on disk with columns
      input_ids      : int64 [max_length + 1]   (train.py does [:, :-1] / [:, 1:])
      attention_mask : int64 [max_length + 1]   (1 = real token, 0 = pad)

Every row is one continuous span of ONE document (the cortex first-pass data
design: the cross-chunk carry is document-correct with no packing).  A document
longer than the window is STRIDED into consecutive, non-overlapping windows —
a 250k-token book becomes ~61 rows at max_length 4096, not one.  The last
window of each document, and any document shorter than the window, gets one EOS
appended (so --cortex.eos_from_tokens can detect the document end) and is padded
with EOS to full length with attention_mask 0 over the padding (pads are
excluded from the loss via the mask, and from the M_cross write via
eos_from_tokens).

FIXED 2026-09-14.  This script used to emit `ids[:row_len]` — ONE row per
document, the first ~4,096 tokens, the rest of the book discarded — under a
`char_cap = row_len * 8` that stopped tokenizing at ~32k characters anyway.
PG-19 volumes run to hundreds of thousands of characters, so the pack held
roughly the FIRST 1% OF EACH BOOK: 28,602 rows / ~117M tokens against PG-19's
~2.5B.  Every "PG-19 cannot supply the volume" decision in the project was
downstream of that row count and none of them are safe to quote.  The
doc-per-row property the script exists for is preserved exactly; there is just
~20x more of it.

Default corpus is PG-19 (emozilla/pg19, books) — almost every document spans
many windows, which is exactly what makes the M_cross carry load-bearing.

Run on a LOGIN node (needs internet for the first download; ~11 GB for PG-19):

    python tools/prepare_pg19_dataset.py \
        --tokenizer ckpts/olmo8-cortex \
        --out data/pg19_olmo_len4096 \
        --max_length 4096

The script is corpus-agnostic — --dataset/--split/--text_col point it anywhere,
and --min_tokens keeps only documents that actually span the window, which is
what makes a corpus able to carry cross-chunk signal at all.  A code pack (a
function defined in chunk 1 and called in chunk 4 is the long-range dependency
prose does not have):

    python tools/prepare_pg19_dataset.py \
        --tokenizer ckpts/olmo8-cortex --dataset codeparrot/codeparrot-clean-valid \
        --split train --text_col content --min_tokens 3500 \
        --out data/code_olmo_val_len4096 --max_length 4096

Then train with:
    python train.py --preprocessed_data_path data/pg19_olmo_len4096 --max_length 4096 ...
"""
from __future__ import annotations

import argparse


def stride_windows(ids, row_len: int, eos: int):
    """Split one document's token ids into consecutive non-overlapping rows.

    Returns a list of `(input_ids, attention_mask)` pairs, each exactly `row_len`
    long.  Full windows are all-ones mask; the document's ragged tail gets one
    EOS appended as the document-end marker and is EOS-padded with mask 0.

    Module level, not a closure inside main(), so `tests/test_pg19_packer.py`
    can pin it: this function is the entire difference between a 28,602-row pack
    that holds the first 1% of each book and a ~600k-row one that holds all of
    it, and the failure mode is silent — a truncating packer produces a
    perfectly valid dataset that merely has no long-range structure in it.
    """
    out = []
    for start in range(0, len(ids), row_len):
        win = list(ids[start:start + row_len])
        if len(win) == row_len:
            # A full window.  No EOS marker: either the document runs on into
            # the next window, or it happened to end exactly on the boundary —
            # indistinguishable here, and the second case is rare enough to cost
            # nothing.
            out.append((win, [1] * row_len))
        else:
            win = win + [eos]                          # mark the document end
            mask = [1] * len(win)
            pad = row_len - len(win)
            win = win + [eos] * pad                    # eos-pad → eos_from_tokens
            mask = mask + [0] * pad                    # pads excluded from loss
            out.append((win, mask))
    return out


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
                    help="cap the number of documents (None = all); applied "
                         "BEFORE --min_tokens, so raise it when filtering hard")
    ap.add_argument("--min_tokens", type=int, default=0,
                    help="drop ROWS holding fewer than this many real tokens "
                         "(0 = keep all).  NOTE the unit changed with the 2026-"
                         "09-14 striding fix: this filtered whole DOCUMENTS when "
                         "a document was a row, and now filters windows — in "
                         "practice the ragged last window of each book.  A row "
                         "that does not span the window contributes no "
                         "cross-chunk signal at all: chunk g is then a different "
                         "document than g-1, and the measurable ceiling is ~0 by "
                         "construction.  Set it near max_length to keep only "
                         "full windows (at the cost of ~1 row per book, plus "
                         "every book shorter than the window).")
    ap.add_argument("--prepend_bos", action="store_true",
                    help="prepend the tokenizer's bos id to every row.  OFF by "
                         "default: prepare_packed_dataset.py does not do it, so "
                         "leaving it off is what keeps the legs of a corpus mix "
                         "consistent.  On OLMo-2 bos == eos, so turning it on "
                         "puts a document boundary at position 0 of every row.")
    ap.add_argument("--char_cap", type=int, default=0,
                    help="truncate each document to this many CHARACTERS before "
                         "tokenizing (0 = no cap, the default).  This used to be "
                         "hardwired to row_len*8 and was half of the truncation "
                         "bug: at max_length 4096 it stopped at ~32k characters, "
                         "so raising row_len alone would still have capped every "
                         "book at ~8k tokens.  Set it only to bound tokenization "
                         "cost on a pathological corpus.")
    ap.add_argument("--map_batch_size", type=int, default=8,
                    help="documents per map batch.  Lower than the usual 64 "
                         "because a batch now holds WHOLE books rather than "
                         "32k-character prefixes, and peak memory is "
                         "map_batch_size x num_proc x book length.")
    ap.add_argument("--num_proc", type=int, default=16)
    args = ap.parse_args()

    from datasets import load_dataset
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    eos = tok.eos_token_id
    assert eos is not None, "tokenizer has no eos token"
    # BOS is OPT-IN (--prepend_bos), and off by default.
    #
    # The old comment here said "may be None (OLMo-2 has none)".  It is not:
    # OLMo-2's tokenizer_config sets bos_token = <|endoftext|>, the SAME id as
    # eos (100257).  So every row silently got an EOS prepended at position 0 —
    # which prepare_packed_dataset.py does not do, giving the PG-19 and
    # FineWeb-Edu legs of a mix two different row-start conventions, and which
    # with --cortex.eos_from_tokens flags position 0 of chunk 0 as a document
    # boundary on every row.
    bos = tok.bos_token_id if args.prepend_bos else None
    if args.prepend_bos and bos == eos:
        print("WARNING: --prepend_bos, but this tokenizer's bos IS its eos "
              f"({bos}) — every row will start with an EOS.")
    row_len = args.max_length + 1
    char_cap = args.char_cap          # 0 = tokenize the whole document

    ds = load_dataset(args.dataset, split=args.split)
    if args.max_samples is not None:
        ds = ds.select(range(min(args.max_samples, len(ds))))
    n_docs = len(ds)
    print(f"{args.dataset}[{args.split}]: {n_docs} documents")

    def tokenize(batch):
        """One document in, AS MANY ROWS AS IT SPANS out.

        A batched map may return a different number of rows than it received —
        that is what turns one book into ~61 consecutive windows.  Windows are
        non-overlapping, so no token is trained on twice within an epoch, and
        each row is a contiguous span of a single document, which is the whole
        property the cross-chunk carry depends on.
        """
        out_ids, out_mask = [], []
        for text in batch[args.text_col]:
            ids = tok(text[:char_cap] if char_cap else text,
                      add_special_tokens=False)["input_ids"]
            if bos is not None:
                # First window only: bos marks the start of the DOCUMENT, not of
                # every window.  Prepending it per row would put a document
                # boundary at position 0 of every row, which with
                # --cortex.eos_from_tokens is exactly the thing the flag is
                # meant to detect (and on OLMo-2, bos == eos).
                ids = [bos] + ids
            for win, mask in stride_windows(ids, row_len, eos):
                out_ids.append(win)
                out_mask.append(mask)
        return {"input_ids": out_ids, "attention_mask": out_mask}

    tokenized = ds.map(tokenize, batched=True, batch_size=args.map_batch_size,
                       num_proc=args.num_proc, remove_columns=ds.column_names)

    if args.min_tokens:
        before = len(tokenized)
        # count real tokens, not row length — every row is row_len long, the
        # mask is what says how much of it is document.
        tokenized = tokenized.filter(
            lambda r: sum(r["attention_mask"]) >= args.min_tokens,
            num_proc=args.num_proc)
        print(f"min_tokens={args.min_tokens}: kept {len(tokenized)}/{before} rows")
        if len(tokenized) == 0:
            raise SystemExit("No row survived --min_tokens; lower it or "
                             "raise --max_samples")

    # num_proc: save_to_disk is the slow half of this script, not tokenization,
    # and it is the step that cost ~14h once on Lustre.  It matters more after
    # the striding fix than before, because the output is ~20x larger.
    # Completion marker: state.json / dataset_info.json are written LAST, so
    # shard files without them mean a killed save.
    tokenized.save_to_disk(args.out, num_proc=8)

    n_full = sum(1 for m in tokenized[:1000]["attention_mask"] if m[-1] == 1)
    print(f"saved {len(tokenized)} rows x {row_len} tokens -> {args.out}")
    print(f"  {len(tokenized) / max(n_docs, 1):.1f} rows per document "
          f"({n_docs} documents in, {len(tokenized) * args.max_length / 1e9:.2f}B "
          f"tokens of window)")
    print(f"full (unragged) windows, first 1000 sampled: {n_full}/1000")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
