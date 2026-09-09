#!/usr/bin/env python
"""Check, before submitting anything, that the requested per-bucket n exists.

This is the check whose absence produced every n=100 long-context table in the
project: those jobs asked for --max_examples 500 and got 100, because
RMT-team/BABILong holds exactly 100 rows per (task, length) -- and nothing in
the pipeline said so.  The shortfall was only visible in the totals column of
a results file written 8 GPU-hours later.

Run on a login node after evals/download_datasets.py:

    python tools/preflight_longcontext.py --want 250
    python tools/preflight_longcontext.py --want 250 --lme   # also scan LongMemEval
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def count_parquet(path: Path) -> int:
    """Row count from the parquet footer -- no need to read the data."""
    try:
        import pyarrow.parquet as pq
    except ImportError:
        raise SystemExit("pyarrow is needed to count parquet rows "
                         "(it ships with `datasets`; check the conda env).")
    return pq.ParquetFile(str(path)).metadata.num_rows


def count_json(path: Path) -> int:
    with open(path, encoding="utf-8") as fh:
        head = fh.read(1).strip()
    if head == "[":
        with open(path, encoding="utf-8") as fh:
            return len(json.load(fh))
    with open(path, encoding="utf-8") as fh:      # jsonl
        return sum(1 for line in fh if line.strip())


def scan_babilong(root: Path, tasks, lengths, want: int) -> int:
    print(f"\n=== BABILong under {root} ===")
    if not root.exists():
        print(f"  MISSING: {root} does not exist. "
              f"Run `python evals/download_datasets.py`.")
        return 1
    print(f"    {'task':<6} " + " ".join(f"{c:>7}" for c in lengths))
    worst = None
    problems = 0
    for task in tasks:
        row = []
        for cfg in lengths:
            parquet = sorted(root.glob(f"{cfg}/{task}-*.parquet"))
            js = root / "data" / task / f"{cfg}.json"
            if parquet:
                n = sum(count_parquet(p) for p in parquet)
            elif js.exists():
                n = count_json(js)
            else:
                n = 0
            row.append(n)
            worst = n if worst is None else min(worst, n)
            if n < want:
                problems += 1
        cells = " ".join((f"{n:>7,}" if n >= want else f"{n:>6,}!") for n in row)
        print(f"    {task:<6} {cells}")
    if worst is None:
        print("  No cells found at all -- wrong --babilong path?")
        return 1
    print(f"\n  smallest cell: {worst:,} rows; requested {want}/bucket")
    if problems:
        print(f"  {problems} cell(s) marked '!' cannot supply {want}. "
              f"Either lower --want, or point BABILONG_PATH at "
              f"data/babilong-1k (RMT-team/babilong-1k-samples, ~1000 "
              f"rows/cell, qa1-qa5).")
    else:
        print(f"  OK: every cell can supply {want}.")
    return problems


def scan_lme(root: Path, depth_buckets, want: int) -> int:
    print(f"\n=== LongMemEval under {root} ===")
    f = root / "longmemeval_s"
    if not f.exists():
        print(f"  MISSING: {f}. Run `python evals/download_datasets.py`.")
        return 1
    with open(f, encoding="utf-8") as fh:
        raw = json.load(fh)
    ds = raw if isinstance(raw, list) else next(iter(raw.values()))
    labels = [f"<={d}sess" for d in depth_buckets] + [f">{depth_buckets[-1]}sess"]
    counts = {lbl: 0 for lbl in labels}
    for ex in ds:
        sessions = ex.get("haystack_sessions", [])
        if isinstance(sessions, str):
            sessions = json.loads(sessions)
        depth = len(sessions)
        bucket = labels[-1]
        for i, thresh in enumerate(depth_buckets):
            if depth <= thresh:
                bucket = labels[i]
                break
        counts[bucket] += 1
    for lbl, n in counts.items():
        flag = "" if n >= want or n == 0 else "   <- short"
        print(f"    {lbl:<12} {n:>6,}{flag}")
    total = sum(counts.values())
    print(f"\n  {total:,} questions TOTAL in longmemeval_s. This is a hard "
          f"ceiling:\n  the split has no more questions, so LongMemEval cannot "
          f"be powered past it\n  regardless of --max_examples. Use LME_MAX=0 "
          f"(all) and spend added\n  compute on BABILong instead.")
    return 0


def main() -> None:
    ap = argparse.ArgumentParser("long-context eval preflight")
    ap.add_argument("--babilong", default="data/babilong-1k",
                    help="local BABILong dir (either layout is detected)")
    ap.add_argument("--lme_path", default="data/LongMemEval")
    ap.add_argument("--tasks", nargs="+",
                    default=["qa1", "qa2", "qa3", "qa4", "qa5"])
    ap.add_argument("--lengths", nargs="+",
                    default=["1k", "2k", "4k", "8k", "16k", "32k"])
    ap.add_argument("--want", type=int, default=250,
                    help="examples per bucket the run will ask for")
    ap.add_argument("--lme", action="store_true",
                    help="also scan LongMemEval (loads a large JSON)")
    args = ap.parse_args()

    problems = scan_babilong(Path(args.babilong), args.tasks, args.lengths,
                             args.want)
    if args.lme:
        problems += scan_lme(Path(args.lme_path), [5, 10, 20, 50], args.want)
    raise SystemExit(1 if problems else 0)


if __name__ == "__main__":
    main()
