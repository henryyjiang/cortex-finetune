"""Rebuild eval_babilong.py's end-of-run artefacts from streamed records.jsonl.

eval_babilong.py writes records.jsonl incrementally but results.json / summary.csv
only after the last task returns, so a job killed on walltime leaves the per-example
data intact and the aggregates missing. This reconstructs them from the records and
drops a RUN_NOTE.md recording that they were rebuilt rather than produced by the run.

    python tools/reconstruct_babilong_outputs.py <babilong_out_dir> [--label final_checkpoint]

Writes results.json, summary.csv (both schema-identical to a completed run),
scoring_summary.csv (chance / containment / rank / rank_ln / pmi), and RUN_NOTE.md.
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

norm = lambda s: " ".join(str(s).strip().lower().split())


def argmin(vals):
    return min(range(len(vals)), key=lambda i: vals[i])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("out_dir")
    p.add_argument("--label", default="final_checkpoint",
                   help="Top-level key eval_babilong.py would have used")
    p.add_argument("--note", default="killed on walltime; aggregates rebuilt from records")
    args = p.parse_args()

    out = Path(args.out_dir)
    rec_path = out / "records.jsonl"
    rows = [json.loads(l) for l in rec_path.open(encoding="utf-8")]
    if not rows:
        raise SystemExit(f"no records in {rec_path}")

    # ── results.json / summary.csv: containment, exactly what the run would write ──
    agg: dict = collections.defaultdict(lambda: collections.defaultdict(
        lambda: {"correct": 0, "total": 0}))
    for r in rows:
        cell = agg[r["task"]][r["bucket"]]
        cell["total"] += 1
        cell["correct"] += bool(r["correct"])
    for task in agg:
        for cell in agg[task].values():
            cell["accuracy"] = cell["correct"] / cell["total"] if cell["total"] else 0.0

    results = {args.label: {t: dict(b) for t, b in agg.items()}}
    (out / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    with (out / "summary.csv").open("w", encoding="utf-8") as f:
        f.write("model,task,bucket,correct,total,accuracy\n")
        for task, buckets in results[args.label].items():
            for bucket, r in buckets.items():
                f.write(f"{args.label},{task},{bucket},{r['correct']},"
                        f"{r['total']},{r['accuracy']:.4f}\n")

    # ── scoring_summary.csv: the three scorings side by side ──────────────────
    by = collections.defaultdict(list)
    for r in rows:
        by[(r["task"], r["bucket"])].append(r)

    with (out / "scoring_summary.csv").open("w", encoding="utf-8") as f:
        f.write("task,bucket,n,n_cand,chance,contain,rank,rank_ln,pmi\n")
        for (task, bucket), rs in sorted(by.items()):
            n = len(rs)
            contain = sum(1 for r in rs if r["correct"]) / n
            ch = rk = rkln = pmi = m = csize = 0
            for r in rs:
                c = r.get("candidates")
                if not c:
                    continue
                m += 1
                csize += len(c)
                g, nll, ntok = norm(r["gold"]), r["cand_nll"], r["cand_ntok"]
                ch += 1.0 / len(c)
                rk += norm(c[argmin(nll)]) == g
                rkln += norm(c[argmin([nll[i] / max(ntok[i], 1)
                                       for i in range(len(c))])]) == g
                nc = r.get("cand_nll_nocontext")
                if nc:
                    pmi += norm(c[argmin([nll[i] - nc[i]
                                          for i in range(len(c))])]) == g
            if not m:
                continue
            f.write(f"{task},{bucket},{n},{csize/m:.1f},{ch/m:.4f},{contain:.4f},"
                    f"{rk/m:.4f},{rkln/m:.4f},{pmi/m:.4f}\n")

    # ── RUN_NOTE.md: provenance, so these are never mistaken for run output ────
    cells = {f"{t}/{b}": len(v) for (t, b), v in sorted(by.items())}
    full = max(cells.values())
    short = {k: v for k, v in cells.items() if v < full}
    note = [
        "# Reconstructed outputs",
        "",
        f"`results.json`, `summary.csv` and `scoring_summary.csv` in this directory were "
        f"**rebuilt from `records.jsonl`** by `tools/reconstruct_babilong_outputs.py`, "
        f"not written by `eval_babilong.py`.",
        "",
        f"Reason: {args.note}.",
        "",
        f"- records recovered: **{len(rows)}**",
        f"- cells: {len(cells)}, full cell size {full}",
        f"- incomplete cells: {short or 'none'}",
        "",
        "`results.json` and `summary.csv` use containment scoring and are schema-identical "
        "to a completed run. `scoring_summary.csv` is additional and has no counterpart in "
        "normal output. `samples.json` is NOT reconstructed — it is a debug sample of the "
        "first five rows per bucket and is not recoverable from the records.",
    ]
    (out / "RUN_NOTE.md").write_text("\n".join(note) + "\n", encoding="utf-8")

    print(f"wrote results.json, summary.csv, scoring_summary.csv, RUN_NOTE.md -> {out}")
    print(f"  {len(rows)} records, {len(cells)} cells, incomplete: {short or 'none'}")


if __name__ == "__main__":
    main()
