"""
Z attempt 2, STEP 1 -- the synthetic positive control.  Findings doc,
"Attempt 2", Step 1.

WHY IT EXISTS.  Every Z test in attempt 1 ran on prose, where E already
captures the carry benefit, so none of them could have come back positive.
This is a task where the carried state is NECESSARY and exactly measurable,
so a mechanism that can carry anything shows it here in ~1000 updates.

THE TASK: register tracking.  A row is a stream of lines

    REG + d_1 [+ d_2 ...] = v \\n

where REG is one of R single-token register names, the d_i are digits, and v
is the register's new value mod 10 (registers start at 0, never stated).  A
value is written out ONLY when its register is updated, so:

  * an answer whose register was last updated IN THIS CHUNK is LOCAL: the
    previous value is on the page.  ~80% of answers at R=16.  These are the
    task's own positive control -- if the model cannot get these, it has not
    learned the task and no carry number means anything.
  * an answer whose register was last updated in an EARLIER chunk can only be
    produced from the CARRY.  About R per chunk (each register's first update
    in the chunk), ~16 per chunk, ~112 per 8-chunk row.

WHAT IT CAN AND CANNOT SHOW.  E can hold R digits as well as Z can, so no task
makes "Z can and E cannot" true by construction.  What it gives is a carry that
is necessary and cleanly scored.  Z's own contribution is read from the 2x2's
E-OFF cells (Z_alone), which are only in-distribution for a model trained with
--cortex.e_dropout > 0 -- see the findings doc, Step 3.  `ops_per_line` > 1
makes each answer a multi-step sum, i.e. work for the loop, not only storage.

SCORING IS CHUNKING-AGNOSTIC.  Training draws random chunk sizes, so which
answers are carry-dependent is not a property of the row.  The validation pack
therefore stores, per token, `answer_dep`: -2 for a non-answer token, else the
row position of the earliest token the answer depends on (the register's
previous answer, or this line's REG token on a first update).  The eval
(`eval_carry_2x2.py --score carry|local|answers`) classifies each answer
against ITS OWN chunk boundaries.  The TRAINING pack carries only input_ids
and attention_mask, cast to the prose pack's features, because
tools/prepare_corpus_mix.py requires identical columns.

TOKENS ARE BUILT AS IDS, NOT TEXT.  Each piece ("A".."P", "+", "=", "0".."9",
"\\n") is verified to encode to exactly ONE token and the rows are
concatenations of those ids.  Tokenizing rendered text instead would let BPE
merge digits with neighbours and the answer would no longer be one token.

    python tools/prepare_carry_task.py --tokenizer ckpts/olmo-retrofit-cortex \\
        --features_like data/pg19_fw50_olmo_len4096 \\
        --out data/carry_task_r16_len4096 --rows 4000 --seed 1
    python tools/prepare_carry_task.py ... --out data/carry_task_r16_len4096_val \\
        --rows 1000 --seed 2 --with_answer_dep

Launcher (both packs, then the J1 mix): pace/prepare_carry_task.sbatch.
"""
from __future__ import annotations

import argparse
import random
import sys
from typing import Optional

NOT_ANSWER = -2
REG_NAMES = [chr(ord("A") + i) for i in range(26)]
DIGITS = [str(i) for i in range(10)]


def piece_ids(tok, pieces) -> dict:
    """piece -> its ONE token id, or raise.  Also refuses two pieces that
    share an id, which would make an answer ambiguous with a name."""
    out = {}
    for p in pieces:
        ids = tok.encode(p, add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(f"piece {p!r} encodes to {len(ids)} tokens {ids}; "
                             "the task needs every piece to be exactly one")
        out[p] = int(ids[0])
    if len(set(out.values())) != len(out):
        raise ValueError(f"two pieces share a token id: {out}")
    return out


def make_row(rng: random.Random, n_tokens: int, reg_ids: list, digit_ids: list,
             plus: int, eq: int, nl: int, ops_per_line: int = 1,
             modulus: int = 10) -> tuple[list, list]:
    """One row of exactly n_tokens ids, and its answer_dep.

    The row is filled with whole lines and the last line is cut wherever the
    row ends; an answer cut off by the end is simply absent.  answer_dep[q] is
    NOT_ANSWER except at answer tokens, where it is the position of the
    earliest token the answer depends on.
    """
    if modulus > len(digit_ids):
        raise ValueError("modulus exceeds the digit pieces available")
    value = [0] * len(reg_ids)
    last_answer: list[Optional[int]] = [None] * len(reg_ids)
    ids: list = []
    dep: list = []
    while len(ids) < n_tokens:
        r = rng.randrange(len(reg_ids))
        start = len(ids)
        line = [reg_ids[r]]
        total = value[r]
        for _ in range(ops_per_line):
            d = rng.randrange(modulus)
            line += [plus, digit_ids[d]]
            total = (total + d) % modulus
        line += [eq]
        ans_pos = start + len(line)
        line += [digit_ids[total], nl]
        ids += line
        d_ = [NOT_ANSWER] * len(line)
        d_[ans_pos - start] = (last_answer[r] if last_answer[r] is not None
                               else start)
        dep += d_
        value[r] = total
        last_answer[r] = ans_pos
    return ids[:n_tokens], dep[:n_tokens]


def classify(answer_dep: list, n_chunks: int) -> tuple[list, list]:
    """(carry, local) masks over LABEL positions p = 0 .. len-2.

    The label at p is ids[p+1], predicted from x[p], so the prediction lives in
    chunk p // L of x; the answer depends on x[dep], visible locally iff it is
    in the same chunk.  L = the eval's own chunk length, (len-1) // n_chunks,
    the same arithmetic eval_carry_2x2 uses.  Positions past the last full
    chunk are in neither mask.
    """
    n_x = len(answer_dep) - 1
    L = n_x // n_chunks
    keep = L * n_chunks
    carry = [0] * keep
    local = [0] * keep
    for p in range(keep):
        d = answer_dep[p + 1]
        if d == NOT_ANSWER:
            continue
        if d // L < p // L:
            carry[p] = 1
        else:
            local[p] = 1
    return carry, local


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rows", type=int, required=True)
    ap.add_argument("--max_length", type=int, default=4096,
                    help="rows are max_length + 1 ids, as every pack here")
    ap.add_argument("--registers", type=int, default=16)
    ap.add_argument("--ops_per_line", type=int, default=1)
    ap.add_argument("--seed", type=int, required=True,
                    help="REQUIRED and must differ between the train and val "
                         "packs, or the validation rows are training rows")
    ap.add_argument("--features_like", default=None,
                    help="cast input_ids/attention_mask to this pack's features "
                         "so prepare_corpus_mix.py can concatenate them")
    ap.add_argument("--with_answer_dep", action="store_true",
                    help="add the answer_dep column (validation packs)")
    ap.add_argument("--report_chunks", type=int, default=8,
                    help="only for the printed carry/local counts")
    args = ap.parse_args()

    if not 1 <= args.registers <= len(REG_NAMES):
        print(f"ERROR: --registers must be 1..{len(REG_NAMES)}", file=sys.stderr)
        return 1
    from datasets import Dataset, Features, Sequence, Value, load_from_disk
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    pieces = REG_NAMES[:args.registers] + DIGITS + ["+", "=", "\n"]
    pid = piece_ids(tok, pieces)
    reg_ids = [pid[n] for n in REG_NAMES[:args.registers]]
    digit_ids = [pid[d] for d in DIGITS]
    print(f"pieces -> ids: {pid}")

    rng = random.Random(args.seed)
    n_tok = args.max_length + 1
    cols = {"input_ids": [], "attention_mask": []}
    if args.with_answer_dep:
        cols["answer_dep"] = []
    n_carry = n_local = 0
    for _ in range(args.rows):
        ids, dep = make_row(rng, n_tok, reg_ids, digit_ids, pid["+"], pid["="],
                            pid["\n"], args.ops_per_line)
        cols["input_ids"].append(ids)
        cols["attention_mask"].append([1] * n_tok)
        if args.with_answer_dep:
            cols["answer_dep"].append(dep)
        c, l_ = classify(dep, args.report_chunks)
        n_carry += sum(c)
        n_local += sum(l_)
    print(f"{args.rows} rows x {n_tok} ids | at {args.report_chunks} equal "
          f"chunks: {n_carry / args.rows:.1f} carry-dependent and "
          f"{n_local / args.rows:.1f} local answers per row")

    if args.features_like:
        ref = load_from_disk(args.features_like).features
        feats = {k: ref[k] for k in ("input_ids", "attention_mask")}
    else:
        feats = {"input_ids": Sequence(Value("int32")),
                 "attention_mask": Sequence(Value("int8"))}
    if args.with_answer_dep:
        feats["answer_dep"] = Sequence(Value("int32"))
    ds = Dataset.from_dict(cols, features=Features(feats))
    ds.save_to_disk(args.out, num_proc=1)
    print(f"wrote {args.out}  features={dict(ds.features)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
