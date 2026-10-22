"""
Stitch a chain of wandb links back into ONE contiguous per-step series.

Why this exists
---------------
train.py never calls wandb.init(resume=...), so every link of a chained run is
a SEPARATE wandb run with its own id.  `tools/pull_wandb_metrics.py` exports one
CSV per link, and the 5B pair is eight links per arm:

    c-chunked__eqm9i0t2.csv      1 .. 9,735
    c-chunked__teclfb1p.csv  9,701 .. 11,444      <- overlaps the one above
    ...

The overlap is not a duplicate.  A link that died at 17,017 was resumed from
checkpoint_16,800, so steps 16,801..17,017 were trained TWICE, by two different
links, and only the SECOND one's weights are on the lineage that reached the
horizon.  Concatenating the exports gives a series that re-traverses those
steps; de-duplicating by keeping the first occurrence keeps the orphaned
branch.  Both are wrong, and both are wrong in a way that looks fine on a plot.

THE RULE, and the only one that matches what the weights did: a link owns the
steps from its own first logged step up to (not including) the first step of
the link that resumed after it.  Everything past that point was rolled back by
the resume and is dropped -- counted and named in the manifest, never silently.

Stdlib only (csv, math), for the same reason tools/compare_arms.py is: it has
to run in the cluster's cortex-retro env as well as locally.

Outputs, under <exports>/<project>/stitched/
    <run>.csv            the contiguous series, same columns as the exports
                         plus link_id / link_index, so it drops straight into
                         `tools/compare_arms.py --csv <arm>=<this file>`
    <run>__links.csv     one row per link: claimed range, kept range, dropped
    paired_<a>_vs_<b>.csv   per-step paired contrast, when --pair is given

Usage
-----
    python tools/stitch_wandb_links.py --project cortex-5b \
        --pair c-chunked cortex-final --heal_end 11444 --max_steps 38147

    # the architecture trace has the same problem, from appends not exports
    python tools/stitch_wandb_links.py --project cortex-5b \
        --diag logs/final5b_diag/cortex-final_at_38146.jsonl

Ordering
--------
Links are ordered by their first logged step, which is the resume point and is
what defines the chain.  Two links sharing a first step means one of them was a
false start (launched, died before it logged past the resume point, relaunched
from the same checkpoint); the longer one is kept and the choice is printed.
`--order created_at` instead reads wandb's own creation times from the
project's summary.csv -- authoritative, but only present in summaries written
by a pull_wandb_metrics.py from 2026-10-22 or later.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import sys

STEP_KEY = "train/step"
LOSS_KEY = "train/loss"
NORM_KEY = "train/total_norm"


# --------------------------------------------------------------------------
# reading

def _num(v):
    """CSV cell -> float, or None.  wandb writes '' for a key a link never
    logged and the string 'NaN'/'Infinity' for a non-finite one; both must come
    back as None rather than raising or, worse, as the string."""
    if v is None or v == "":
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def read_link(path: str) -> dict:
    """One export CSV -> {id, name, path, cols, rows}.  Rows keep their ORIGINAL
    string cells: this tool re-emits them, and re-formatting a float through
    Python would change the digits that land in the stitched file."""
    base = os.path.basename(path)[:-len(".csv")]
    name, _, run_id = base.partition("__")
    rows, cols = [], []
    with open(path, encoding="utf-8", errors="replace", newline="") as fh:
        rd = csv.DictReader(fh)
        cols = list(rd.fieldnames or [])
        for row in rd:
            s = _num(row.get(STEP_KEY))
            if s is None:
                continue
            row["_step"] = int(s)
            rows.append(row)
    rows.sort(key=lambda r: r["_step"])
    return {"id": run_id or base, "name": name, "path": path,
            "cols": cols, "rows": rows}


def read_created_at(exports: str, project: str) -> dict:
    """{run_id: created_at} from the project's summary.csv, if it has the
    column.  Missing file or missing column -> {} and the caller falls back."""
    p = os.path.join(exports, project, "summary.csv")
    if not os.path.isfile(p):
        return {}
    out = {}
    with open(p, encoding="utf-8", errors="replace", newline="") as fh:
        for row in csv.DictReader(fh):
            if row.get("created_at") and row.get("id"):
                out[row["id"]] = row["created_at"]
    return out


# --------------------------------------------------------------------------
# the stitch

def order_links(links: list, created: dict) -> tuple:
    """Chain order, and which source produced it.

    created_at is wandb's own clock and is authoritative: it orders a rollback
    (a link relaunched from a checkpoint BEHIND the previous link's start)
    correctly, which no step-based rule can.

    Falling back to the first logged step is right for every ordinary chain,
    because the first logged step IS the resume point.  Its one tie is two
    links that resumed from the same checkpoint -- a false start and the
    relaunch.  The SHORTER goes first so the longer supersedes it: the link
    that died before logging past its own resume point is the one with nothing
    to contribute.
    """
    if created and all(l["id"] in created for l in links):
        return (sorted(links, key=lambda l: (created[l["id"]], l["rows"][0]["_step"])),
                "created_at")
    return (sorted(links, key=lambda l: (l["rows"][0]["_step"], len(l["rows"]))),
            "first_step")


def detect_containment(links: list) -> list:
    """Links whose whole span sits inside an EARLIER link's span, as ordered.

    Under resume-and-continue each link ends at or past the one before it, so
    containment means the order is wrong: link i claims steps past link i+1's
    end, which can only happen if i was launched AFTER i+1, from a checkpoint
    behind i+1's resume point.  First-step ordering cannot see that and would
    silently truncate the long link at the short one's start, losing every step
    past the short one's end.  wandb's created_at can see it, so this is a
    reason to go and get created_at, not a reason to guess.
    """
    out = []
    for i in range(len(links) - 1):
        if links[i]["rows"][-1]["_step"] > links[i + 1]["rows"][-1]["_step"]:
            out.append((links[i]["id"], links[i + 1]["id"]))
    return out


def stitch(links: list) -> dict:
    """Truncate each link at the next one's first step and concatenate.

    Returns {rows, manifest, gaps, superseded}.  `rows` is strictly increasing
    in step by construction -- asserted, not assumed, because a chain whose
    links do not nest this way is a chain this rule cannot describe and the
    caller must be told rather than handed a plausible CSV.
    """
    manifest, out, superseded = [], [], []
    for i, link in enumerate(links):
        nxt = links[i + 1]["rows"][0]["_step"] if i + 1 < len(links) else None
        kept = [r for r in link["rows"] if nxt is None or r["_step"] < nxt]
        rec = {
            "link_index": i,
            "link_id": link["id"],
            "file": os.path.basename(link["path"]),
            "claimed_first": link["rows"][0]["_step"],
            "claimed_last": link["rows"][-1]["_step"],
            "rows_claimed": len(link["rows"]),
            "kept_first": kept[0]["_step"] if kept else "",
            "kept_last": kept[-1]["_step"] if kept else "",
            "rows_kept": len(kept),
            "rows_rolled_back": len(link["rows"]) - len(kept),
        }
        manifest.append(rec)
        if not kept:
            superseded.append(link["id"])
            continue
        for r in kept:
            r["link_id"] = link["id"]
            r["link_index"] = str(i)
        out.extend(kept)

    steps = [r["_step"] for r in out]
    bad = [(steps[i], steps[i + 1]) for i in range(len(steps) - 1)
           if steps[i + 1] <= steps[i]]
    if bad:
        raise SystemExit(
            f"ERROR: the stitched series is not strictly increasing at "
            f"{bad[:5]}{' ...' if len(bad) > 5 else ''}.\n"
            f"  Links do not nest as resume-and-continue, so the ownership "
            f"rule in this tool's header does not describe this chain.  "
            f"Inspect the per-link ranges above before trusting any of it.")

    gaps = []
    for i in range(len(steps) - 1):
        if steps[i + 1] != steps[i] + 1:
            gaps.append((steps[i] + 1, steps[i + 1] - 1))
    return {"rows": out, "manifest": manifest, "gaps": gaps,
            "superseded": superseded}


def write_rows(path: str, cols: list, rows: list) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


# --------------------------------------------------------------------------
# the paired contrast

def _mean(xs):
    return sum(xs) / len(xs)


def paired_stats(deltas: list) -> dict:
    """Mean, se and t of a paired per-step difference.

    Valid ONLY while both arms consume the same rows in the same order: the row
    permutation is a pure function of (seed, len(dataset)) consumed in order, so
    optimizer step k reads the same 32 rows in either arm.  A phase boundary
    where one arm switched pack and the other did not breaks that, which is why
    the caller splits at the heal boundary instead of pooling across it.
    """
    n = len(deltas)
    if n < 2:
        return {"n": n}
    m = _mean(deltas)
    var = sum((d - m) ** 2 for d in deltas) / (n - 1)
    se = math.sqrt(var / n)
    return {"n": n, "mean": m, "sd": math.sqrt(var), "se": se,
            "t": (m / se) if se else float("inf"),
            "ci_lo": m - 1.96 * se, "ci_hi": m + 1.96 * se}


def _fmt_stats(label: str, st: dict) -> str:
    if st.get("n", 0) < 2:
        return f"  {label:22s} n={st.get('n', 0)}  (too few paired steps)"
    return (f"  {label:22s} n={st['n']:>6}  mean {st['mean']:+.4f}  "
            f"sd {st['sd']:.4f}  se {st['se']:.5f}  t {st['t']:+.1f}  "
            f"95% CI [{st['ci_lo']:+.4f}, {st['ci_hi']:+.4f}]")


def pair_runs(a_name, a_rows, b_name, b_rows, heal_end, out_path, tail=2000):
    """Join two stitched series on step and report b - a.

    NEGATIVE mean = b has the LOWER loss, i.e. b is ahead.
    """
    a = {r["_step"]: _num(r.get(LOSS_KEY)) for r in a_rows}
    b = {r["_step"]: _num(r.get(LOSS_KEY)) for r in b_rows}
    shared = sorted(s for s in a.keys() & b.keys()
                    if a[s] is not None and b[s] is not None)
    if not shared:
        print(f"  no steps where both {a_name} and {b_name} logged a finite loss")
        return None

    dcol = f"delta_{b_name}_minus_{a_name}"
    rows = [{"train/step": s, f"loss_{a_name}": a[s], f"loss_{b_name}": b[s],
             dcol: b[s] - a[s]} for s in shared]
    write_rows(out_path, ["train/step", f"loss_{a_name}", f"loss_{b_name}", dcol], rows)

    print(f"\n  paired on {len(shared):,} steps "
          f"({shared[0]:,}..{shared[-1]:,}) -> {out_path}")
    print(f"  delta = {b_name} - {a_name};  NEGATIVE means {b_name} is ahead")
    only_a = len(a) - len(shared)
    only_b = len(b) - len(shared)
    if only_a or only_b:
        print(f"  unpaired steps dropped: {only_a:,} in {a_name}, "
              f"{only_b:,} in {b_name}")

    # Split at the heal boundary.  The corpus moves under the loss there, so a
    # delta pooled across it mixes two different contrasts and the pooled mean
    # is not either of them.
    heal = [b[s] - a[s] for s in shared if s < heal_end]
    mix = [b[s] - a[s] for s in shared if s >= heal_end]
    print(_fmt_stats(f"heal  (step < {heal_end})", paired_stats(heal)))
    print(_fmt_stats(f"mix   (step >= {heal_end})", paired_stats(mix)))
    print(_fmt_stats(f"last {tail} of the mix", paired_stats(mix[-tail:])))
    print("  (pooling heal and mix into one number is the artefact that "
          "produced the retracted 'flat loss' reading -- quote the phases.)")
    return rows


# --------------------------------------------------------------------------
# the diagnostic trace

def dedup_diag(path: str, out_path: str) -> None:
    """cortex_diag.jsonl -> one row per step.

    train.py APPENDS to this file across links, so a resume re-walks the steps
    between the checkpoint and the dead link's real stop and appends them a
    second time.  The file is therefore neither unique nor monotone in `step`.
    Appends are chronological, so the LAST occurrence of a step is the one the
    surviving lineage wrote -- the same ownership rule the CSV stitch applies,
    arrived at from the other direction.
    """
    rows, bad = [], 0
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                bad += 1
    by_step = {}
    for i, r in enumerate(rows):
        s = r.get("step")
        if s is None:
            continue
        by_step[int(s)] = (i, r)           # later append wins
    keep = [r for _, (_, r) in sorted(by_step.items())]
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="ascii") as fh:
        for r in keep:
            fh.write(json.dumps(r) + "\n")
    steps = sorted(by_step)
    print(f"  {path}")
    print(f"    {len(rows):,} rows -> {len(keep):,} unique steps "
          f"({steps[0]:,}..{steps[-1]:,}), {len(rows) - len(keep):,} re-walked, "
          f"{bad} unparseable")
    print(f"    -> {out_path}")


# --------------------------------------------------------------------------

def report_run(name: str, res: dict, max_steps: int) -> None:
    man, rows, gaps = res["manifest"], res["rows"], res["gaps"]
    print(f"\n=== {name}: {len(man)} link(s) -> {len(rows):,} steps ===")
    print(f"  {'idx':>3} {'link id':12} {'claimed':>17} {'kept':>17} "
          f"{'rows':>7} {'rolled back':>12}")
    for m in man:
        claimed = f"{m['claimed_first']:,}..{m['claimed_last']:,}"
        kept = (f"{m['kept_first']:,}..{m['kept_last']:,}"
                if m["rows_kept"] else "(none -- superseded)")
        print(f"  {m['link_index']:>3} {m['link_id']:12} {claimed:>17} "
              f"{kept:>17} {m['rows_kept']:>7,} {m['rows_rolled_back']:>12,}")
    rolled = sum(m["rows_rolled_back"] for m in man)
    if rolled:
        print(f"  {rolled:,} logged steps were rolled back by a resume and are "
              f"NOT in the stitched series.")
    if res["superseded"]:
        print(f"  WARN: link(s) {', '.join(res['superseded'])} contributed "
              f"nothing -- entirely re-trained by a later link.")

    first, last = rows[0]["_step"], rows[-1]["_step"]
    missing = sum(hi - lo + 1 for lo, hi in gaps)
    print(f"  range {first:,}..{last:,}; "
          f"{'no gaps' if not gaps else f'{len(gaps)} gap(s), {missing:,} steps missing'}")
    for lo, hi in gaps[:10]:
        print(f"    missing {lo:,}..{hi:,}  ({hi - lo + 1:,} steps)")
    if len(gaps) > 10:
        print(f"    ... and {len(gaps) - 10} more")

    nan_loss = sum(1 for r in rows if _num(r.get(LOSS_KEY)) is None)
    nan_norm = sum(1 for r in rows if NORM_KEY in r and _num(r.get(NORM_KEY)) is None)
    print(f"  non-finite: {nan_loss:,} loss rows, {nan_norm:,} grad-norm rows "
          f"(a non-finite norm = train.py's guard SKIPPED that update)")
    if max_steps:
        print(f"  coverage vs horizon {max_steps:,}: "
              f"{100.0 * len(rows) / max_steps:.1f}% of steps logged, "
              f"last step is {max_steps - last:,} short")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--project", default="cortex-5b")
    ap.add_argument("--exports", default="wandb_exports")
    ap.add_argument("--out_dir", default=None,
                    help="default: <exports>/<project>/stitched")
    ap.add_argument("--runs", nargs="+", default=None,
                    help="run names to stitch (default: every name present)")
    ap.add_argument("--order", choices=("first_step", "created_at"),
                    default="first_step")
    ap.add_argument("--pair", nargs=2, metavar=("A", "B"), default=None,
                    help="write the per-step paired contrast B - A")
    ap.add_argument("--heal_end", type=int, default=11444,
                    help="phase boundary the paired stats split at")
    ap.add_argument("--tail", type=int, default=2000,
                    help="size of the trailing window in the paired report")
    ap.add_argument("--max_steps", type=int, default=38147,
                    help="horizon, for the coverage line (0 to skip)")
    ap.add_argument("--diag", nargs="+", default=None,
                    help="cortex_diag.jsonl file(s) to de-duplicate as well")
    args = ap.parse_args()

    out_dir = args.out_dir or os.path.join(args.exports, args.project, "stitched")

    if args.diag:
        print("=== diagnostic traces ===")
        for p in args.diag:
            if not os.path.isfile(p):
                print(f"  SKIP (no such file): {p}")
                continue
            base = os.path.basename(p)
            stem = base[:-len(".jsonl")] if base.endswith(".jsonl") else base
            dedup_diag(p, os.path.join(out_dir, f"{stem}__dedup.jsonl"))

    pattern = os.path.join(args.exports, args.project, "*.csv")
    paths = [p for p in sorted(glob.glob(pattern))
             if os.path.basename(p) != "summary.csv" and "__" in os.path.basename(p)]
    if not paths:
        if args.diag:
            return 0
        print(f"No link exports under {pattern}.\n"
              f"  Pull them first:  python tools/pull_wandb_metrics.py "
              f"--project {args.project} --all_attempts")
        return 1

    created = read_created_at(args.exports, args.project) if args.order == "created_at" else {}
    if args.order == "created_at" and not created:
        print("WARN: --order created_at asked for, but the project's "
              "summary.csv has no created_at column (pulled before "
              "2026-10-22).  Falling back to first_step ordering.\n")

    by_name = {}
    for p in paths:
        link = read_link(p)
        if not link["rows"]:
            print(f"WARN: {os.path.basename(p)} has no usable "
                  f"{STEP_KEY} rows -- skipped.")
            continue
        by_name.setdefault(link["name"], []).append(link)

    wanted = args.runs or sorted(by_name)
    stitched = {}
    for name in wanted:
        links = by_name.get(name)
        if not links:
            print(f"WARN: no exports named {name} in {args.project}.")
            continue
        links, ordered_by = order_links(links, created)
        firsts = [l["rows"][0]["_step"] for l in links]
        for i in range(len(firsts) - 1):
            if firsts[i] == firsts[i + 1]:
                print(f"WARN: {name}: links {links[i]['id']} and "
                      f"{links[i + 1]['id']} both start at step {firsts[i]:,} "
                      f"-- one is a false start.  Keeping the longer "
                      f"({links[i + 1]['id']}); check it before quoting the tail.")
        contained = detect_containment(links)
        if contained and ordered_by == "first_step":
            for outer, inner in contained:
                print(f"ERROR: {name}: link {inner} ({links[0]['name']}) lies "
                      f"entirely inside link {outer}'s step range.")
            print(f"  A chain cannot do that under resume-and-continue, so the "
                  f"first-step ordering is wrong here -- most likely a link was "
                  f"relaunched from a checkpoint BEHIND an earlier link's "
                  f"start.  Only wandb's clock can order that:\n"
                  f"    python tools/pull_wandb_metrics.py --project "
                  f"{args.project} --all_attempts   # writes created_at\n"
                  f"    python tools/stitch_wandb_links.py --project "
                  f"{args.project} --order created_at ...\n"
                  f"  {name} NOT stitched.")
            continue
        res = stitch(links)
        report_run(name, res, args.max_steps)

        cols = list(links[0]["cols"])
        for l in links[1:]:
            for c in l["cols"]:
                if c not in cols:
                    cols.append(c)
        cols += ["link_id", "link_index"]
        out_csv = os.path.join(out_dir, f"{name}.csv")
        write_rows(out_csv, cols, res["rows"])
        write_rows(os.path.join(out_dir, f"{name}__links.csv"),
                   list(res["manifest"][0].keys()), res["manifest"])
        print(f"  -> {out_csv}")
        print(f"  -> {os.path.join(out_dir, f'{name}__links.csv')}")
        stitched[name] = res["rows"]

    if args.pair:
        a, b = args.pair
        print(f"\n=== paired contrast: {b} vs {a} ===")
        if a not in stitched or b not in stitched:
            print(f"  need both {a} and {b} stitched; have "
                  f"{sorted(stitched) or 'none'}")
            return 1
        pair_runs(a, stitched[a], b, stitched[b], args.heal_end,
                  os.path.join(out_dir, f"paired_{a}_vs_{b}.csv"), args.tail)

    print(f"\nAll outputs under {out_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
