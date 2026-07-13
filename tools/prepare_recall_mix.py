"""
Recall-supervised mix (signal fix #2) — splice cross-chunk key-value recall
probes into an already-prepared PG-19 dataset dir, producing a drop-in
replacement dataset for train.py (same columns/shapes, point DATA_PATH at it).

For a --frac fraction of window-filling rows:
  * a FACT sentence ('... the code word for the amber falcon is "harbor" ...')
    is spliced at a random token position inside an early chunk (uniform over
    --fact_chunks), and
  * a PROBE ('Question: What is the code word for the amber falcon?\\nAnswer:
    "harbor"') overwrites the row's final tokens (end of the LAST chunk).

Fact and probe are always in different sub-windows and more than one chunk
apart from the probe's window start, so under the training cross_chunks split
the answer token is predictable ONLY via the carried M_cross buffer.  Unlike
plain PG-19 (where the measured carry payoff is ~0.004 nats/token spread over
everything), the probe's answer token carries a ~ln(vocab_pool) nats loss
delta concentrated exactly on the memory pathway — the dense training signal
the 20260712 analysis called for.

Labels: attention_mask stays 1 over the whole augmented row (train.py derives
labels from the mask), so book text keeps its normal LM loss and the probe is
supervised like any other text.  Rows that don't fill the window are copied
through unmodified.

Run on a LOGIN node (tokenizer load only — no downloads if data is prepared):

    python tools/prepare_recall_mix.py \
        --data data/pg19_olmo_len4096 --tokenizer ckpts/olmo8-cortex \
        --out data/pg19_olmo_recall25_len4096 --frac 0.25

    # matching val set (for eval_carry_ablation on probe-bearing data):
    python tools/prepare_recall_mix.py \
        --data data/pg19_olmo_val_len4096 --tokenizer ckpts/olmo8-cortex \
        --out data/pg19_olmo_val_recall100_len4096 --frac 1.0
"""
from __future__ import annotations

import argparse
import random

# Small closed pools -> the value is a common, cleanly-tokenized word the model
# can only get from the fact (chance ~1/len(VALUES) even with perfect format
# learning).  Keys are adjective+noun pairs: enough combinations that key
# memorization across samples is useless.
ADJECTIVES = [
    "amber", "crimson", "ivory", "cobalt", "scarlet", "golden", "silver",
    "violet", "emerald", "copper", "marble", "shadow", "winter", "summer",
    "ancient", "silent", "hidden", "broken", "gentle", "hollow", "iron",
    "lonely", "narrow", "distant", "quiet", "rusty", "salted", "velvet",
]
NOUNS = [
    "falcon", "lantern", "harbor", "orchard", "compass", "anchor", "meadow",
    "bridge", "castle", "garden", "hammer", "island", "jacket", "kettle",
    "ladder", "mirror", "needle", "organ", "palace", "ribbon", "saddle",
    "temple", "valley", "wagon", "window", "candle", "basket", "engine",
]
VALUES = [
    "harbor", "willow", "ember", "granite", "sparrow", "cedar", "frost",
    "raven", "clover", "maple", "onyx", "pearl", "quartz", "reed", "slate",
    "thorn", "umber", "vine", "wren", "yarrow", "birch", "coral", "dusk",
    "fern", "gull", "heron", "iris", "juniper", "kelp", "lark", "moss",
    "nettle", "oak", "pine", "quill", "rush", "sage", "tide", "usher",
    "wharf", "yew", "zephyr", "brook", "cliff", "drift", "elm", "flint",
]

FACT_TMPL  = '\nRemember this: the code word for the {key} is "{value}".\n'
PROBE_TMPL = '\nQuestion: What is the code word for the {key}?\nAnswer: "{value}"'


def splice_facts_probe(ids, placements, probe_ids):
    """Pure token-level splice, row length preserved.

    placements: list of (pos, fact_ids) — each fact is inserted at pos
    (positions refer to the ORIGINAL ids; insertion order is high-to-low so
    earlier placements stay put).  A fact at pos ends up occupying
    [pos + sum(len of facts at lower pos), ...) — i.e. it can drift right by
    the total length of facts placed before it; callers must leave margin.
    The probe overwrites the last len(probe_ids) positions; displaced book
    tokens fall off the end."""
    row_len = len(ids)
    total = sum(len(f) for _, f in placements) + len(probe_ids)
    assert max(p for p, _ in placements) + total < row_len, "no room"
    new = list(ids)
    for pos, fact_ids in sorted(placements, key=lambda t: t[0], reverse=True):
        new = new[:pos] + list(fact_ids) + new[pos:]
    return new[: row_len - len(probe_ids)] + list(probe_ids)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data",      required=True,
                    help="prepared dataset dir (prepare_pg19_dataset.py output)")
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out",       required=True)
    ap.add_argument("--frac",      type=float, default=0.25,
                    help="fraction of window-filling rows to augment")
    ap.add_argument("--n_facts",   type=int, default=1,
                    help="facts per augmented row (probe asks about one of them)")
    ap.add_argument("--n_chunks",  type=int, default=4,
                    help="training cross_chunks (chunk geometry of the splice)")
    ap.add_argument("--fact_chunks", default="0,1,2",
                    help="0-based chunk indices eligible to hold a fact "
                         "(never the last chunk — that holds the probe)")
    ap.add_argument("--margin",    type=int, default=64,
                    help="min tokens between a fact and its chunk boundaries "
                         "(also absorbs the rightward drift from other facts "
                         "inserted at lower positions — ~20 tokens per fact)")
    ap.add_argument("--seed",      type=int, default=74)
    ap.add_argument("--num_proc",  type=int, default=1)
    args = ap.parse_args()

    from datasets import load_from_disk
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    ds = load_from_disk(args.data)
    row_len = len(ds[0]["input_ids"])          # max_length + 1
    chunk_len = (row_len - 1) // args.n_chunks # matches train.py's torch.chunk
    fact_chunks = [int(c) for c in args.fact_chunks.split(",")]
    assert all(0 <= c < args.n_chunks - 1 for c in fact_chunks), \
        "fact chunks must precede the last (probe) chunk"
    print(f"{args.data}: {len(ds)} rows x {row_len}; chunk_len={chunk_len}, "
          f"frac={args.frac}, n_facts={args.n_facts}, fact_chunks={fact_chunks}")

    stats = {"augmented": 0, "plain": 0, "not_full": 0}

    def augment(row, idx):
        rng = random.Random(args.seed * 1_000_003 + idx)  # per-row, order-stable
        ids, mask = row["input_ids"], row["attention_mask"]
        if mask[-1] != 1:                       # padded row: splice would mix
            stats["not_full"] += 1              # probe with eos-pad garbage
            return row
        if rng.random() >= args.frac:
            stats["plain"] += 1
            return row
        # distinct keys within the row; value sampled per fact
        keys = rng.sample([(a, n) for a in ADJECTIVES for n in NOUNS],
                          args.n_facts)
        facts = []
        for adj, noun in keys:
            key = f"the {adj} {noun}"
            value = rng.choice(VALUES)
            facts.append((key, value))
        # probe asks about one random fact
        q_key, q_value = facts[rng.randrange(len(facts))]
        probe_ids = tok(PROBE_TMPL.format(key=q_key, value=q_value),
                        add_special_tokens=False)["input_ids"]
        placements = []
        for key, value in facts:
            fact_ids = tok(FACT_TMPL.format(key=key, value=value),
                           add_special_tokens=False)["input_ids"]
            ci = rng.choice(fact_chunks)
            lo = ci * chunk_len + args.margin
            hi = (ci + 1) * chunk_len - args.margin - len(fact_ids)
            assert hi > lo, "chunk too small for fact + margins"
            placements.append((rng.randrange(lo, hi), fact_ids))
        new_ids = splice_facts_probe(ids, placements, probe_ids)
        stats["augmented"] += 1
        return {"input_ids": new_ids, "attention_mask": [1] * row_len}

    # num_proc=1 by default: the stats dict and rng-by-idx stay exact; the map
    # is tokenizer-bound but cheap (facts/probes are ~20 tokens each).
    out = ds.map(augment, with_indices=True, num_proc=args.num_proc)
    out.save_to_disk(args.out)
    print(f"saved {len(out)} rows -> {args.out}")
    print(f"stats: {stats}")

    # show one augmented row (ASCII-safe for the cp1252 Windows console)
    for i in range(min(len(out), 2000)):
        text = tok.decode(out[i]["input_ids"][-64:])
        if "code word" in text:
            print("sample probe tail:",
                  text[-200:].encode("ascii", "replace").decode())
            break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
