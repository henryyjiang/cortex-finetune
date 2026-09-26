"""Two checkpoint dirs, one question: would a pack tokenized with A train
correctly under B?

WHY THIS EXISTS, AND WHY IT IS A LAUNCH GATE RATHER THAN A COMMENT.  J5 trains
on `data/j1_pg19fw50_carry20_len4096`, which `tools/prepare_carry_task.py` built
with `ckpts/olmo-retrofit-cortex`'s tokenizer -- but J5's base is
`ckpts/olmo8-cortex`.  Both descend from OLMo-2-0425-1B, so the vocabularies are
EXPECTED to be identical, and that expectation is exactly the kind this project
turns into a check: a mismatched tokenizer trains on garbage behind a perfectly
healthy loss curve, and the carry task's answers are single digit tokens whose
ids would silently become something else.

WHAT IS COMPARED, in the order that catches the most with the least:

  1. vocab_size and eos_token_id -- the coarse mismatch, and the one a wrong
     repo produces.
  2. THE CARRY TASK'S OWN PIECES.  "A".."P", "+", "=", "0".."9" and "\\n" --
     the pieces prepare_carry_task.py verifies encode to exactly one token, and
     the only ids the answer NLL is ever computed over.  If these move, the
     eval scores the wrong positions, which is a silently wrong number rather
     than a crash.
  3. A fixed probe string, encoded end to end, ids compared exactly -- the
     merge-table check: two tokenizers can agree on every single-token piece
     and still disagree on how they merge in context.

Exit 0 = the pack is safe under the second tokenizer.  Any mismatch is exit 1
with the offending ids printed, and the launcher refuses.

    python tools/check_tokenizer_match.py \\
        --pack_tokenizer ckpts/olmo-retrofit-cortex --model ckpts/olmo8-cortex
"""
from __future__ import annotations

import argparse
import sys

#: The carry task's alphabet, as tools/prepare_carry_task.py builds it: the
#: register names it uses (16 of 26), the two operators, the ten digits and the
#: line break.  Kept literal rather than imported so this tool runs before
#: anything of the training stack is importable.
CARRY_PIECES = ([chr(ord("A") + i) for i in range(16)]
                + ["+", "=", "\n"] + [str(i) for i in range(10)])

#: Long enough to exercise merges, and made of exactly the two corpora the pack
#: mixes (PG-19 prose and FineWeb-ish web text) plus a carry line.
PROBE = ("It was the best of times, it was the worst of times.\n"
         "A+7=12\nB+3=5\n"
         "The quick brown fox jumps over the lazy dog -- 2026, $4.50, "
         "https://example.com/path?q=1 &amp; more.\n")


def load(path: str):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(path, trust_remote_code=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack_tokenizer", required=True,
                    help="the dir the PACK was tokenized with "
                         "(tools/prepare_carry_task.py --tokenizer)")
    ap.add_argument("--model", required=True,
                    help="the dir the run will TRAIN with (--model_name)")
    a = ap.parse_args()

    if a.pack_tokenizer == a.model:
        print(f"  ok   same dir ({a.model}) -- nothing to compare")
        return 0

    try:
        ta, tb = load(a.pack_tokenizer), load(a.model)
    except Exception as exc:                      # noqa: BLE001 - reported, not raised
        print(f"FAIL: could not load both tokenizers: {exc}")
        return 1

    fails = []

    va, vb = len(ta), len(tb)
    if va != vb:
        fails.append(f"vocab_size {va} (pack) vs {vb} (model)")
    else:
        print(f"  ok   vocab_size {va} on both")

    ea, eb = ta.eos_token_id, tb.eos_token_id
    if ea != eb:
        fails.append(f"eos_token_id {ea} (pack) vs {eb} (model)")
    else:
        print(f"  ok   eos_token_id {ea} on both")

    moved = []
    for piece in CARRY_PIECES:
        ia = ta.encode(piece, add_special_tokens=False)
        ib = tb.encode(piece, add_special_tokens=False)
        if ia != ib:
            moved.append(f"{piece!r}: {ia} vs {ib}")
    if moved:
        fails.append("the carry task's own pieces moved -- the answer NLL "
                     "would be computed over other tokens: "
                     + "; ".join(moved[:6])
                     + (f" (+{len(moved) - 6} more)" if len(moved) > 6 else ""))
    else:
        print(f"  ok   all {len(CARRY_PIECES)} carry pieces keep their ids")

    pa = ta.encode(PROBE, add_special_tokens=False)
    pb = tb.encode(PROBE, add_special_tokens=False)
    if pa != pb:
        first = next((i for i, (x, y) in enumerate(zip(pa, pb)) if x != y),
                     min(len(pa), len(pb)))
        fails.append(f"the probe string encodes to {len(pa)} vs {len(pb)} ids, "
                     f"first difference at position {first}: "
                     f"{pa[first:first + 4]} vs {pb[first:first + 4]} -- the "
                     f"merge tables differ even where single pieces agree")
    else:
        print(f"  ok   the {len(pa)}-id probe string round-trips identically")

    for f in fails:
        print(f"  FAIL {f}")
    if fails:
        print(f"\nTOKENIZER MISMATCH ({len(fails)}): {a.pack_tokenizer} built "
              f"the pack, {a.model} would train on it.  Re-tokenize the pack "
              f"with the model's tokenizer, or the run trains on garbage "
              f"behind a healthy loss curve.")
        return 1
    print("\nTOKENIZER MATCH -- the pack is safe under the model's tokenizer.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
