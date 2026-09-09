#!/usr/bin/env python
"""Design calculator for the long-context carry-on / carry-off eval.

Answers the only question worth asking before burning GPU-hours on a rerun:
**how many examples per bucket does the carry-vs-no-carry contrast need before
a ~1-point accuracy difference stops being indistinguishable from noise?**

Two designs are costed, because they differ by ~2x in required n:

  unpaired   two-proportion z on the bucket totals.  This is all the CURRENT
             harness supports -- summary.csv keeps counts only, so once a run
             finishes there is no way to know WHICH examples flipped.
  paired     McNemar on the discordant pairs.  Carry-on and carry-off see a
             byte-identical final window (eval_babilong.split_context is
             called identically in both; only m_cross differs), so every
             example is a matched pair and the between-example variance --
             which is most of the variance, since BABILong difficulty varies
             far more than the carry effect -- cancels.  Requires per-example
             records; see --records in eval_babilong.py.

The paired numbers depend on the DISCORDANCE rate psi = P(the two conditions
disagree on an example), which has never been measured here because the
records were never written.  So psi is swept, with the independence value
2p(1-p) as the pessimistic bound: any positive agreement between the two
conditions only makes psi smaller and the design cheaper.

Usage:
    python tools/power_longcontext.py --root eval_results/longcontext_b2-final
    python tools/power_longcontext.py --plan --cells 60
"""
from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path


# ---------------------------------------------------------------------------
# Normal quantiles (Acklam inverse-CDF; scipy is not in the login-node env)
# ---------------------------------------------------------------------------

def z_of(p: float) -> float:
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > phigh:
        return -z_of(1 - p)
    q, r = p - 0.5, (p - 0.5) ** 2
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
           (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


def norm_cdf(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


# ---------------------------------------------------------------------------
# Sample-size / MDE formulas
# ---------------------------------------------------------------------------

def n_unpaired(delta: float, p: float, alpha: float = 0.05,
               power: float = 0.80) -> float:
    """Examples PER ARM to detect `delta` in a two-proportion test at
    baseline rate p."""
    za, zb = z_of(1 - alpha / 2), z_of(power)
    p1, p2 = p, p + delta
    return (za + zb) ** 2 * (p1 * (1 - p1) + p2 * (1 - p2)) / delta ** 2


def n_paired(delta: float, psi: float, alpha: float = 0.05,
             power: float = 0.80) -> float:
    """PAIRS needed for McNemar at discordance rate psi and difference delta.
    delta = (b - c)/n, psi = (b + c)/n."""
    if psi <= delta ** 2:
        return float("inf")
    za, zb = z_of(1 - alpha / 2), z_of(power)
    return (za * math.sqrt(psi) + zb * math.sqrt(psi - delta ** 2)) ** 2 / delta ** 2


def mde_unpaired(n: float, p: float, alpha=0.05, power=0.80) -> float:
    """Smallest effect an n-per-arm unpaired test can find at `power`."""
    return (z_of(1 - alpha / 2) + z_of(power)) * math.sqrt(2 * p * (1 - p) / n)


def mde_paired(n: float, psi: float, alpha=0.05, power=0.80) -> float:
    return (z_of(1 - alpha / 2) + z_of(power)) * math.sqrt(psi / n)


def indep_psi(p: float) -> float:
    """Discordance if the two conditions were independent -- the worst case;
    any positive agreement lowers it."""
    return 2 * p * (1 - p)


def z_unpaired(c1: int, n1: int, c2: int, n2: int) -> float:
    if n1 == 0 or n2 == 0:
        return 0.0
    p1, p2 = c1 / n1, c2 / n2
    pool = (c1 + c2) / (n1 + n2)
    se = math.sqrt(pool * (1 - pool) * (1 / n1 + 1 / n2))
    return (p1 - p2) / se if se > 0 else 0.0


# ---------------------------------------------------------------------------
# Reading existing results
# ---------------------------------------------------------------------------

def read_babilong(run_dir: Path):
    """(task, bucket) -> (correct, total)"""
    f = run_dir / "babilong" / "summary.csv"
    cells = {}
    if not f.exists():
        return cells
    with open(f, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            cells[(row["task"], row["bucket"])] = (int(row["correct"]),
                                                   int(row["total"]))
    return cells


def read_longmemeval(run_dir: Path):
    f = run_dir / "longmemeval" / "summary.csv"
    cells = {}
    if not f.exists():
        return cells
    with open(f, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if int(row["total"]) > 0:
                cells[("lme", row["bucket"])] = (int(row["correct"]),
                                                 int(row["total"]))
    return cells


def pooled(cells):
    return (sum(c for c, _ in cells.values()), sum(t for _, t in cells.values()))


def ascii_safe(s: str) -> str:
    """LongMemEval bucket labels carry a literal U+2264; printing them dies on
    a cp1252 console (i.e. every Windows run of this script)."""
    return s.replace("≤", "<=").encode("ascii", "replace").decode("ascii")


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------

def observed_report(root: Path, on_label: str, off_label: str) -> None:
    on_dir, off_dir = root / on_label, root / off_label
    print("\n=== OBSERVED (n as run) ===")
    print(f"  carry-on : {on_label}")
    print(f"  carry-off: {off_label}\n")

    for name, reader in (("BABILong", read_babilong),
                         ("LongMemEval", read_longmemeval)):
        on, off = reader(on_dir), reader(off_dir)
        keys = sorted(set(on) & set(off))
        if not keys:
            print(f"  {name}: no overlapping cells\n")
            continue
        print(f"  {name}")
        print(f"    {'cell':<14} {'on':>11} {'off':>11} {'delta pt':>9} {'z':>7}")
        by_task = defaultdict(lambda: [0, 0, 0, 0])
        for k in keys:
            (c1, n1), (c2, n2) = on[k], off[k]
            d = (c1 / n1 - c2 / n2) * 100
            print(f"    {ascii_safe('/'.join(k)):<14} {c1:>4}/{n1:<6} {c2:>4}/{n2:<6} "
                  f"{d:>+9.1f} {z_unpaired(c1, n1, c2, n2):>+7.2f}")
            t = by_task[k[0]]
            t[0] += c1; t[1] += n1; t[2] += c2; t[3] += n2
        print(f"    {'-' * 54}")
        for task, (c1, n1, c2, n2) in sorted(by_task.items()):
            d = (c1 / n1 - c2 / n2) * 100
            print(f"    {task + ' pooled':<14} {c1:>4}/{n1:<6} {c2:>4}/{n2:<6} "
                  f"{d:>+9.2f} {z_unpaired(c1, n1, c2, n2):>+7.2f}")
        (C1, N1), (C2, N2) = pooled(on), pooled(off)
        d = (C1 / N1 - C2 / N2) * 100
        z = z_unpaired(C1, N1, C2, N2)
        print(f"    {'ALL pooled':<14} {C1:>4}/{N1:<6} {C2:>4}/{N2:<6} "
              f"{d:>+9.2f} {z:>+7.2f}   p={2 * (1 - norm_cdf(abs(z))):.3f}")
        print()


def design_report(p: float, deltas, n_cells: int, per_bucket_grid,
                  alpha: float, power: float) -> None:
    psi_grid = [0.05, 0.10, 0.15, indep_psi(p)]

    print(f"=== DESIGN ===  baseline p={p:.3f}, alpha={alpha}, "
          f"power={power:.0%}, {n_cells} cells pooled\n")

    print("  Pairs required, by true effect and by discordance")
    print("  psi = P(the two conditions disagree on an example):\n")
    hdr = "  ".join(f"psi={x:.3f}".rjust(11) for x in psi_grid)
    print(f"    {'effect':<11} {hdr}   {'unpaired/arm':>13}")
    for d in deltas:
        row = []
        for psi in psi_grid:
            n = n_paired(d, psi, alpha, power)
            row.append(("inf" if n == float("inf") else f"{n:,.0f}").rjust(11))
        print(f"    {d * 100:>+6.2f} pt   {'  '.join(row)}   "
              f"{n_unpaired(d, p, alpha, power):>13,.0f}")
    print(f"\n    psi={indep_psi(p):.3f} is the independence bound -- pairing buys "
          f"nothing there.\n    Real psi is lower whenever the conditions agree "
          f"more than chance, which\n    near an answer-prior floor they almost "
          f"certainly do.\n")

    print(f"  Minimum detectable effect at a given per-bucket n, pooled over "
          f"{n_cells} cells:\n")
    print(f"    {'n/bucket':>9} {'pairs':>8}   " +
          "  ".join(f"psi={x:.2f}".rjust(8) for x in psi_grid[:3]) +
          f"   {'unpaired':>9}")
    for nb in per_bucket_grid:
        tot = nb * n_cells
        mp = "  ".join(f"{mde_paired(tot, x, alpha, power) * 100:>8.2f}"
                       for x in psi_grid[:3])
        print(f"    {nb:>9,} {tot:>8,}   {mp}   "
              f"{mde_unpaired(tot, p, alpha, power) * 100:>9.2f}")
    print("\n    (all figures in accuracy points)\n")


def main() -> None:
    ap = argparse.ArgumentParser("long-context eval design calculator")
    ap.add_argument("--root", default="eval_results/longcontext_b2-final")
    ap.add_argument("--on",  default="retro-b2-acc32-cc8-mr8-T8-nc8-sl512")
    ap.add_argument("--off", default="retro-b2-acc32-cc8-mr8-T8-nc8-sl512-nocarry")
    ap.add_argument("--plan", action="store_true",
                    help="skip the observed-results section")
    ap.add_argument("--p", type=float, default=0.13,
                    help="baseline accuracy assumed by the design calc")
    ap.add_argument("--cells", type=int, default=18,
                    help="(task, length) cells pooled over: 3 tasks x 6 "
                         "lengths = 18; qa1-qa10 x 6 = 60")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--power", type=float, default=0.80)
    args = ap.parse_args()

    if not args.plan:
        observed_report(Path(args.root), args.on, args.off)

    design_report(args.p, [0.005, 0.0089, 0.01, 0.015, 0.02, 0.03, 0.05],
                  args.cells, [100, 250, 500, 1000], args.alpha, args.power)


if __name__ == "__main__":
    main()
