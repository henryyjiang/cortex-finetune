"""
Does a packed dataset cover the step budget?

The parquet loader does NOT wrap: train.py iterates it once inside
`for epoch in range(epochs)` and the only break is `optimizer_step >= max_steps`.
Exhaustion is therefore a CLEAN EXIT — wandb reports "finished" at half the
intended run and nothing in the loss curve says otherwise.  That is what stopped
both B1 arms at 24,414 steps / 400M tokens.  Check before queueing 48h, not after.

Rows consumed per optimizer step = batch_size (each row is one sequence).
Tokens per step = batch_size * max_length.

    python tools/check_pack.py --data data/fineweb_edu_olmo_len4096 \
        --steps 91552 --batch_size 4

    # B2 phase 0 (healing, --stop_at_step 91552) and a full arm (305,176):
    python tools/check_pack.py --data data/fineweb_edu_olmo_len4096 --steps 91552
    python tools/check_pack.py --data data/nemotron_math_olmo_len4096_4b --steps 305176

Exit code is 1 when the pack is short, so it can gate a submit script.
"""
from __future__ import annotations

import argparse
import sys


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Packed-dataset budget check")
    p.add_argument("--data", required=True, help="load_from_disk dir")
    p.add_argument("--steps", type=int, required=True,
                   help="optimizer steps the pack must cover (stop_at_step, or "
                        "max_steps for a full arm)")
    p.add_argument("--batch_size", type=int, default=4,
                   help="global batch in sequences per optimizer step")
    p.add_argument("--max_length", type=int, default=4096,
                   help="tokens per sequence (rows are max_length+1 long)")
    p.add_argument("--margin", type=float, default=1.10,
                   help="require this much headroom over the budget (1.10 = 10%%)")
    return p.parse_args()


def main() -> int:
    a = parse_args()
    from datasets import load_from_disk

    ds = load_from_disk(a.data)
    rows = len(ds)
    row_len = len(ds[0]["input_ids"])
    need_rows = a.steps * a.batch_size
    tokens = rows * a.max_length

    print(f"pack      : {a.data}")
    print(f"  rows    : {rows:,}  x {row_len} ids/row ({a.max_length} usable)")
    print(f"  tokens  : {tokens / 1e9:.3f}B")
    print(f"budget    : {a.steps:,} steps x {a.batch_size} seq "
          f"= {need_rows:,} rows = {a.steps * a.batch_size * a.max_length / 1e9:.3f}B tokens")

    if row_len != a.max_length + 1:
        print(f"  WARNING: rows are {row_len} ids, expected max_length+1 = "
              f"{a.max_length + 1}.  Wrong --max_length, or the pack was built "
              f"for a different sequence length.")

    ratio = rows / need_rows if need_rows else float("inf")
    print(f"coverage  : {ratio:.2f}x the budget "
          f"({rows - need_rows:+,} rows, {(rows - need_rows) / a.batch_size:+,.0f} steps)")

    if ratio < 1.0:
        print(f"\nFAIL: the loader will exhaust at step "
              f"{rows // a.batch_size:,} of {a.steps:,} and the run will report "
              f"'finished'.  Build a bigger pack (tools/prepare_packed_dataset.py "
              f"--max_tokens), or lower --stop_at_step.")
        return 1
    if ratio < a.margin:
        print(f"\nFAIL: only {ratio:.2f}x the budget, under the {a.margin:.2f}x "
              f"margin.  A pack sized exactly to the budget leaves no room for a "
              f"resume that replays a partial step.")
        return 1
    print("\nOK: pack covers the budget with margin.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
