#!/usr/bin/env python
"""Did raising the carry buffer's capacity break the model?

Reads records.jsonl from every run under an eval-tag dir and prints the
diagnostics that distinguish "the memory finally has the context" from "the
prefix is now so far out of distribution that the model has stopped working".

Columns, and what a bad value looks like:

  acc %            containment accuracy -- the blunt metric, here for continuity
  gold NLL         nats/token on the gold answer.  RISING with capacity means
                   the extra context is actively hurting.  This is the only
                   column sensitive enough to move at n=100.
  p(EOS) first     probability of ending the answer before it starts.  This is
                   the mechanism behind the LongMemEval regression: carry-on
                   stops after ~10 words where carry-off runs to the cap, and
                   containment then loses because the gold string surfaces
                   later.  Rising with capacity = capacity is not the fix.
  entropy first    nats.  A collapse toward 0 or a jump toward log|V| (~10.8
                   for a 50k vocab) means the output distribution is broken and
                   the run is uninterpretable regardless of its accuracy.
  words            mean generated words -- the observable p(EOS) acts through.

Usage:
    python tools/summarize_carry_health.py --root eval_results/longcontext_b2-ammax
"""
from __future__ import annotations

import argparse
import json
import math
import statistics as st
from pathlib import Path


def load(run_dir: Path) -> list:
    recs = []
    for f in sorted(run_dir.glob("*/records.jsonl")):
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    recs.append(json.loads(line))
    return recs


def mean_of(recs, key):
    vals = [r[key] for r in recs if r.get(key) is not None]
    return st.mean(vals) if vals else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser("carry-health summary")
    ap.add_argument("--root", default="eval_results/longcontext_b2-ammax")
    ap.add_argument("--vocab", type=int, default=50304,
                    help="vocab size, for the uniform-entropy reference line")
    args = ap.parse_args()

    root = Path(args.root)
    runs = sorted(d for d in root.iterdir() if d.is_dir())
    if not runs:
        raise SystemExit(f"No run dirs under {root}")

    print(f"\n=== carry health: {root} ===")
    print(f"    {'run':<44} {'n':>6} {'acc %':>7} {'gold NLL':>9} "
          f"{'p(EOS) 1st':>11} {'entropy':>8} {'words':>7} {'chunks':>7}")
    print(f"    {'-' * 106}")
    for d in runs:
        recs = load(d)
        if not recs:
            print(f"    {d.name:<44} {'--':>6}   no records.jsonl")
            continue
        n = len(recs)
        acc = sum(1 for r in recs if r["correct"]) / n * 100
        print(f"    {d.name:<44} {n:>6,} {acc:>7.2f} "
              f"{mean_of(recs, 'gold_nll_per_tok'):>9.4f} "
              f"{mean_of(recs, 'p_eos_first'):>11.4f} "
              f"{mean_of(recs, 'entropy_first'):>8.3f} "
              f"{mean_of(recs, 'pred_words'):>7.2f} "
              f"{mean_of(recs, 'n_prime_chunks'):>7.1f}")
    print(f"\n    Uniform-distribution entropy for a {args.vocab:,}-token vocab is "
          f"{math.log(args.vocab):.2f} nats.")
    print(f"    An entropy near that, or near 0, means the run is broken and its "
          f"accuracy means nothing.\n")


if __name__ == "__main__":
    main()
