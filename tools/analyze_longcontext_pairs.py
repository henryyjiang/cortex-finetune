#!/usr/bin/env python
"""Paired analysis of the long-context carry-on / carry-off runs.

Reads the per-example records.jsonl written by eval_babilong.py and
eval_longmemeval.py and produces the three tables the B2 long-context section
needs:

  --pairs   carry-on vs carry-off at one T.  McNemar (exact, two-sided) on the
            discordant pairs, plus a paired bootstrap CI on the accuracy
            difference and -- when both runs were scored with --score_nll -- a
            paired t-test on the gold-answer NLL, which is the sensitive
            metric.

  --ladder  the memory-vs-depth table: every (T, carry) cell side by side with
            its per-token layer-application count, so "the carry at T=8 buys
            more than 4x the depth at T=32 does" can be read off one table
            instead of asserted.

Why paired and not the two-proportion z the old tables used: both conditions
run the SAME example through the SAME final window (split_context is called
identically; only m_cross differs), so the example is a matched pair.  Between-
example difficulty is most of the variance and it cancels in the pairing.  The
unpaired test throws that away and needs roughly 2/psi x p(1-p) times more
examples for the same power -- see tools/power_longcontext.py.

Usage:
    python tools/analyze_longcontext_pairs.py --pairs \\
        --on  eval_results/longcontext_b2-power/<arm>-T8-nc8-sl512 \\
        --off eval_results/longcontext_b2-power/<arm>-T8-nc8-sl512-nocarry

    python tools/analyze_longcontext_pairs.py --ladder \\
        --root eval_results/longcontext_b2-power --arm retro-b2-acc32-cc8-mr8
"""
from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path


# ---------------------------------------------------------------------------
# Stats (pure stdlib -- the login-node env has no scipy)
# ---------------------------------------------------------------------------

def norm_cdf(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar: binomial(b; b+c, 0.5).

    Exact rather than the chi-square approximation because the discordant
    counts here are small (tens), which is exactly where the approximation is
    anticonservative."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def mcnemar_midp(b: int, c: int) -> float:
    """Mid-p McNemar -- less conservative than the exact test, standard for
    reporting alongside it."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    point = math.comb(n, k) / (2 ** n)
    return min(1.0, 2 * (tail - 0.5 * point))


def paired_bootstrap_ci(diffs, iters=10000, alpha=0.05, seed=0):
    """Percentile CI for the mean of per-example paired differences."""
    if not diffs:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    n = len(diffs)
    means = []
    for _ in range(iters):
        s = 0.0
        for _ in range(n):
            s += diffs[rng.randrange(n)]
        means.append(s / n)
    means.sort()
    lo = means[int(alpha / 2 * iters)]
    hi = means[min(iters - 1, int((1 - alpha / 2) * iters))]
    return (lo, hi)


def paired_t(diffs):
    """(mean, se, t, two-sided p) for paired differences, normal approx on p."""
    n = len(diffs)
    if n < 2:
        return (float("nan"),) * 4
    mean = sum(diffs) / n
    var = sum((d - mean) ** 2 for d in diffs) / (n - 1)
    se = math.sqrt(var / n)
    if se == 0:
        return (mean, 0.0, float("nan"), float("nan"))
    t = mean / se
    return (mean, se, t, 2 * (1 - norm_cdf(abs(t))))


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------

def load_records(run_dir: Path) -> dict:
    """id -> record, merged across the babilong and longmemeval sub-evals.

    Both write records.jsonl; BABILong ids are 'task/bucket/row' and
    LongMemEval ids are question_ids, so they cannot collide.
    """
    recs = {}
    found = []
    # Glob rather than a fixed ("babilong", "longmemeval") pair so a run that
    # was sharded across jobs -- babilong-qa1/, babilong-qa2/, ... , which is
    # how the T=32 carry-on cell fits inside a wall-clock limit -- merges back
    # into one record set here.
    for f in sorted(run_dir.glob("*/records.jsonl")):
        found.append(f.parent.name)
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                recs[r["id"]] = r
    if not recs:
        raise SystemExit(
            f"No records.jsonl under {run_dir}.\n"
            f"The paired analysis needs per-example records; summary.csv keeps "
            f"counts only. Re-run the eval with the current harness (records "
            f"are written by default) -- pre-2026-09 runs cannot be paired "
            f"retroactively."
        )
    print(f"  loaded {len(recs):,} records from {run_dir.name} ({', '.join(found)})")
    return recs


def cell_of(r: dict) -> tuple:
    return (r.get("task", "?"), r.get("bucket", "?"))


def scored_correct(r: dict, mode: str):
    """Correctness under a scoring rule other than free-generation containment.

    mode:
      gen   0/1 containment of the gold string in the generation (as recorded)
      rank  argmin of the candidates' summed NLL -- chance is 1/len(candidates)
      norm  same, per-token normalised (the acc_norm of the MC harnesses:
            candidates differ in token count and summed NLL favours short ones)
      pmi   argmin of NLL(a | context) - NLL(a | question only), i.e. the
            candidate whose probability the CONTEXT raised most.  The prior
            drops out, which matters because the base model is flat across a
            32x context range -- these tasks are largely answered from it.

    Returns None when the record cannot support the mode, so a run without
    --rank_answers degrades to the gen column instead of erroring.
    """
    if mode == "gen":
        return bool(r["correct"])
    cands, nll = r.get("candidates"), r.get("cand_nll")
    if not cands or not nll:
        return None
    if mode == "pmi":
        nc = r.get("cand_nll_nocontext")
        if not nc:
            return None
        score = [a - b for a, b in zip(nll, nc)]
    elif mode == "norm":
        ntok = r.get("cand_ntok") or [1] * len(nll)
        score = [a / max(t, 1) for a, t in zip(nll, ntok)]
    else:
        score = list(nll)
    pick = cands[min(range(len(score)), key=lambda i: score[i])]
    return str(pick).strip().lower() == str(r.get("gold", "")).strip().lower()


def ascii_safe(s: str) -> str:
    """LongMemEval bucket labels carry a literal U+2264, which kills printing
    on a cp1252 console."""
    return s.replace("≤", "<=").encode("ascii", "replace").decode("ascii")


# ---------------------------------------------------------------------------
# Paired report
# ---------------------------------------------------------------------------

def pairs_report(on_dir: Path, off_dir: Path, bootstrap: int,
                 by_cell: bool = True) -> None:
    on, off = load_records(on_dir), load_records(off_dir)
    shared = sorted(set(on) & set(off))
    if not shared:
        raise SystemExit("The two runs share no example ids -- check that they "
                         "were run on the same tasks/buckets.")
    only_on, only_off = len(on) - len(shared), len(off) - len(shared)
    if only_on or only_off:
        print(f"  WARNING: {only_on} ids only in carry-on, {only_off} only in "
              f"carry-off; analysis uses the {len(shared):,} shared ids.")

    groups = defaultdict(list)
    for i in shared:
        groups[cell_of(on[i])].append(i)

    def block(name, ids):
        b = sum(1 for i in ids if on[i]["correct"] and not off[i]["correct"])
        c = sum(1 for i in ids if off[i]["correct"] and not on[i]["correct"])
        n = len(ids)
        acc_on = sum(1 for i in ids if on[i]["correct"]) / n * 100
        acc_off = sum(1 for i in ids if off[i]["correct"]) / n * 100
        delta = acc_on - acc_off
        psi = (b + c) / n
        p_ex = mcnemar_exact(b, c)
        p_mid = mcnemar_midp(b, c)
        print(f"    {ascii_safe(name):<16} {n:>6} {acc_on:>7.2f} {acc_off:>7.2f} "
              f"{delta:>+8.2f} {b:>5} {c:>5} {psi:>7.3f} {p_ex:>8.4f} {p_mid:>8.4f}")
        return b, c, n, psi

    print("\n=== PAIRED accuracy (McNemar) ===")
    print(f"  carry-on : {on_dir}")
    print(f"  carry-off: {off_dir}\n")
    print(f"    {'cell':<16} {'n':>6} {'on %':>7} {'off %':>7} {'delta':>8} "
          f"{'b':>5} {'c':>5} {'psi':>7} {'p_exact':>8} {'p_mid':>8}")
    print(f"    {'-' * 92}")
    if by_cell:
        for cell in sorted(groups):
            block("/".join(cell), groups[cell])
        print(f"    {'-' * 92}")
    by_task = defaultdict(list)
    for cell, ids in groups.items():
        by_task[cell[0]].extend(ids)
    for task in sorted(by_task):
        block(f"{task} pooled", by_task[task])
    print(f"    {'-' * 92}")
    b, c, n, psi = block("ALL pooled", shared)

    # b = carry-on-only wins, c = carry-off-only wins.  psi is the number the
    # power calculator was guessing at; print it so the NEXT design is sized on
    # a measurement rather than an assumption.
    print(f"\n  Discordance psi = {psi:.4f} over {n:,} pairs "
          f"({b + c:,} disagreements: {b} carry-on-only, {c} carry-off-only).")
    print(f"  Feed this back into: python tools/power_longcontext.py --plan")

    scoring_report(on, off, shared, bootstrap)

    diffs = [(1.0 if on[i]["correct"] else 0.0) - (1.0 if off[i]["correct"] else 0.0)
             for i in shared]
    mean, se, t, p = paired_t(diffs)
    lo, hi = paired_bootstrap_ci(diffs, iters=bootstrap)
    print(f"\n  Paired accuracy difference: {mean * 100:+.2f} pt "
          f"(SE {se * 100:.2f}, t={t:+.2f}, p={p:.4f})")
    print(f"  Bootstrap 95% CI ({bootstrap:,} resamples): "
          f"[{lo * 100:+.2f}, {hi * 100:+.2f}] pt")

    # ---- NLL: the metric that can actually resolve this effect --------------
    nll_ids = [i for i in shared
               if "gold_nll_per_tok" in on[i] and "gold_nll_per_tok" in off[i]]
    if not nll_ids:
        print("\n  (No gold-answer NLL in these records -- re-run with "
              "--score_nll to get the sensitive metric.)")
        return
    print(f"\n=== PAIRED gold-answer NLL (nats/token, positive = carry helps) ===")
    print(f"    {'cell':<16} {'n':>6} {'on':>9} {'off':>9} {'delta':>9} "
          f"{'SE':>8} {'t':>7} {'p':>8}")
    print(f"    {'-' * 78}")

    def nll_block(name, ids):
        d = [off[i]["gold_nll_per_tok"] - on[i]["gold_nll_per_tok"] for i in ids]
        m_on = sum(on[i]["gold_nll_per_tok"] for i in ids) / len(ids)
        m_off = sum(off[i]["gold_nll_per_tok"] for i in ids) / len(ids)
        mean, se, t, p = paired_t(d)
        print(f"    {ascii_safe(name):<16} {len(ids):>6} {m_on:>9.4f} {m_off:>9.4f} "
              f"{mean:>+9.4f} {se:>8.4f} {t:>+7.2f} {p:>8.4f}")

    if by_cell:
        cells = defaultdict(list)
        for i in nll_ids:
            cells[cell_of(on[i])].append(i)
        for cell in sorted(cells):
            nll_block("/".join(cell), cells[cell])
        print(f"    {'-' * 78}")
    nll_block("ALL pooled", nll_ids)
    print()


def scoring_report(on, off, shared, bootstrap: int) -> None:
    """The same paired contrast under every scoring rule the records support,
    and under the two stratifications that decide whether the contrast is even
    asking about the memory."""
    modes = [m for m in ("gen", "rank", "norm", "pmi")
             if scored_correct(on[shared[0]], m) is not None]

    def row(name, ids, mode):
        ok = [(scored_correct(on[i], mode), scored_correct(off[i], mode))
              for i in ids]
        ok = [(a, b) for a, b in ok if a is not None and b is not None]
        if not ok:
            return
        n = len(ok)
        a_on = sum(1 for a, _ in ok if a) / n * 100
        a_off = sum(1 for _, b in ok if b) / n * 100
        b = sum(1 for x, y in ok if x and not y)
        c = sum(1 for x, y in ok if y and not x)
        diffs = [(1.0 if x else 0.0) - (1.0 if y else 0.0) for x, y in ok]
        lo, hi = paired_bootstrap_ci(diffs, iters=bootstrap)
        print(f"    {name:<28} {mode:<5} {n:>6,} {a_on:>7.2f} {a_off:>7.2f} "
              f"{a_on - a_off:>+8.2f}  [{lo * 100:+6.2f},{hi * 100:+6.2f}] "
              f"{mcnemar_exact(b, c):>8.4f}")

    print("=== SCORING RULES and STRATA (paired, pooled) ===")
    print(f"    {'stratum':<28} {'rule':<5} {'n':>6} {'on %':>7} {'off %':>7} "
          f"{'delta':>8}  {'95% CI':>15} {'p_exact':>8}")
    print(f"    {'-' * 92}")
    for m in modes:
        row("all", shared, m)

    # Stratum 1: is the memory even on the critical path?  When the gold string
    # sits in the final window the model can answer without the carry, and those
    # examples only dilute the contrast.
    in_win = [i for i in shared if on[i].get("gold_in_final_window")]
    out_win = [i for i in shared if on[i].get("gold_in_final_window") is False]
    if in_win and out_win:
        print(f"    {'-' * 92}")
        for m in modes:
            row(f"gold NOT in final window", out_win, m)
        for m in modes:
            row(f"gold in final window", in_win, m)
        print(f"\n    'gold in final window' is the control: the answer is "
              f"readable without the carry,\n    so a delta there is not "
              f"evidence of memory. {len(in_win):,} of {len(shared):,} examples "
              f"({len(in_win) / len(shared) * 100:.1f}%).")

    # Stratum 2: did the FIFO throw the context away on this example?
    kept = [i for i in shared if on[i].get("chunks_evicted") == 0]
    lost = [i for i in shared if (on[i].get("chunks_evicted") or 0) > 0]
    if kept and lost:
        print(f"    {'-' * 92}")
        for m in modes:
            row("buffer kept all chunks", kept, m)
        for m in modes:
            row("buffer evicted chunks", lost, m)
        ev = [on[i]["chunks_evicted"] for i in lost]
        print(f"\n    {len(lost):,} examples lost writes to the FIFO "
              f"(median {sorted(ev)[len(ev) // 2]} chunks dropped). Those two "
              f"rows are\n    different regimes and pooling them is what made "
              f"the earlier tables hard to read.")
    print()


# ---------------------------------------------------------------------------
# Depth-vs-memory ladder
# ---------------------------------------------------------------------------

def layer_apps(T: int, passes: int = 1) -> int:
    """Per-token layer applications: prelude(2) + core(4)*T + coda(2) for the
    raven recurrent-depth block, i.e. 8 + 6T for one pass, matching the
    FLOP-matched ladder in pace/eval_longcontext.sbatch."""
    return passes * (8 + 6 * T)


def ladder_report(root: Path, arm: str, suffix: str, ts, ref: str | None) -> None:
    rows, missing = [], []
    for T in ts:
        for carry, tag in ((True, ""), (False, "-nocarry")):
            d = root / f"{arm}-T{T}{suffix}{tag}"
            if not d.exists():
                missing.append((T, carry, d.name))
                continue
            recs = load_records(d)
            n = len(recs)
            acc = sum(1 for r in recs.values() if r["correct"]) / n * 100
            rows.append((T, carry, layer_apps(T), n, acc, recs))
    ref_row = None
    if ref and (root / ref).exists():
        recs = load_records(root / ref)
        ref_row = (len(recs),
                   sum(1 for r in recs.values() if r["correct"]) / len(recs) * 100)

    base = next((r for r in rows if r[0] == min(ts) and r[1]), None)

    def label(T, carry):
        return f"T={T} " + ("carry" if carry else "no-carry")

    print("\n=== DEPTH vs MEMORY ladder ===")
    print(f"    {'condition':<26} {'layer-apps':>11} {'n':>7} {'acc %':>8} "
          f"{'vs T8+carry':>12}")
    print(f"    {'-' * 68}")
    for T, carry, la, n, acc, _ in rows:
        rel = f"{acc - base[4]:+.2f}" if base else "--"
        print(f"    {label(T, carry):<26} {la:>11} {n:>7,} {acc:>8.2f} {rel:>12}")
    if ref_row:
        rel = f"{ref_row[1] - base[4]:+.2f}" if base else "--"
        print(f"    {'reference (base)':<26} {layer_apps(min(ts)):>11} "
              f"{ref_row[0]:>7,} {ref_row[1]:>8.2f} {rel:>12}")
    for T, carry, name in missing:
        print(f"    {label(T, carry):<26} {layer_apps(T):>11} {'--':>7} "
              f"{'MISSING':>8}   {name}")

    # The two contrasts the table exists to make, stated rather than left
    # implied -- and tested, since every cell scores the SAME example set and
    # the deltas are therefore paired, not independent samples.
    def paired_delta(a, b):
        ids = sorted(set(a[5]) & set(b[5]))
        bb = sum(1 for i in ids if a[5][i]["correct"] and not b[5][i]["correct"])
        cc = sum(1 for i in ids if b[5][i]["correct"] and not a[5][i]["correct"])
        return a[4] - b[4], mcnemar_exact(bb, cc), len(ids)

    on8 = next((r for r in rows if r[0] == min(ts) and r[1]), None)
    off8 = next((r for r in rows if r[0] == min(ts) and not r[1]), None)
    deep = next((r for r in rows if r[0] == max(ts) and not r[1]), None)
    if on8 and off8:
        d, p, n = paired_delta(on8, off8)
        print(f"\n  MEMORY at T={on8[0]}: {d:+.2f} pt over the same-depth no-carry "
              f"control, at the same {on8[2]} layer-apps (McNemar p={p:.4f}, n={n:,}).")
    if off8 and deep:
        d, p, n = paired_delta(deep, off8)
        print(f"  DEPTH T={off8[0]}->{deep[0]} without carry: {d:+.2f} pt for "
              f"{deep[2] / off8[2]:.1f}x the layer-apps (McNemar p={p:.4f}, n={n:,}).")
        print(f"  Every cell is scored on the same examples, so the two deltas "
              f"are paired and directly comparable.")
    print()


def main() -> None:
    ap = argparse.ArgumentParser("paired long-context analysis")
    ap.add_argument("--pairs", action="store_true",
                    help="carry-on vs carry-off at one T (McNemar + NLL)")
    ap.add_argument("--ladder", action="store_true",
                    help="depth-vs-memory table across T")
    ap.add_argument("--on",  help="carry-on run dir (with --pairs)")
    ap.add_argument("--off", help="carry-off run dir (with --pairs)")
    ap.add_argument("--root", default="eval_results/longcontext_b2-power",
                    help="eval tag dir holding the run dirs (with --ladder)")
    ap.add_argument("--arm", default="retro-b2-acc32-cc8-mr8")
    ap.add_argument("--suffix", default="-nc8-sl512",
                    help="label suffix the sbatch appended after -T<N>")
    ap.add_argument("--ts", nargs="+", type=int, default=[8, 16, 32])
    ap.add_argument("--reference", default=None,
                    help="optional base-model run dir name for the ladder")
    ap.add_argument("--bootstrap", type=int, default=10000)
    ap.add_argument("--no_cells", action="store_true",
                    help="pooled rows only (skip the per-cell table)")
    args = ap.parse_args()

    if not (args.pairs or args.ladder):
        ap.error("pick --pairs and/or --ladder")
    if args.pairs:
        if not (args.on and args.off):
            ap.error("--pairs needs --on and --off")
        pairs_report(Path(args.on), Path(args.off), args.bootstrap,
                     by_cell=not args.no_cells)
    if args.ladder:
        ladder_report(Path(args.root), args.arm, args.suffix, sorted(args.ts),
                      args.reference)


if __name__ == "__main__":
    main()
