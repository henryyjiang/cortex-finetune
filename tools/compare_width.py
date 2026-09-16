"""
Does halving the write width cost DELIVERED capacity?

THE QUESTION, AND WHY THE OBVIOUS NUMBER IS THE WRONG ONE.
`tools/slice_summary_emb.py` reports the rank of `summary_emb` -- the learned
input embeddings appended to each chunk.  On the trained parent that measured
20.02 of 32, and the W=16 slice kept 10.90 of 16: **54% of the parent's rank at
50% of the write columns.**  So the trained write basis was NOT redundant, and
P1.0's "the slots were never carrying 32 things" -- an AT-INIT measurement of an
UNTRAINED parameter -- does not transfer.

But `summary_emb` is the write BASIS, not what reaches attention.  Three
different "effective rank" numbers live in this project and they are on
different tensors:

  write basis          `summary_emb`, the input embeddings   20.02 of 32 trained
  carried state        the post-ln_f rows the buffer holds    ~4 of 32 on B2
  delivered influence  directions the carry moves the stream  ~349 (P0.2)

The collapse from ~20 to ~4 happens IN THE FORWARD PASS, not in the parameter.
So the number that decides whether width binds is the **carried state's** rank
at W=32 against W=16 -- which is what this scores, from two runs of
`evals/diag_dual_channel_walk.py` on the SAME weights.

THE RULE IS PRE-REGISTERED.  `p11_z_probe_prereg.md` section 4 fixes the three
readings BEFORE the measurement, and the thresholds below are the same numbers,
held here as module constants so the document and the instrument cannot drift.

USAGE
    python tools/compare_width.py --parent walk-parent-w32.json --branch walk.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

#: Delivered rank retained, as a fraction of the parent's, above which the write
#: basis given up was NOT reaching the token stream.  0.90 rather than 1.0
#: because the statistic is a participation ratio over a handful of dominant
#: directions on a finite sample -- a few percent of wobble is the instrument,
#: not the architecture.
DELIVERED_RETAINED_SURVIVES = 0.90

#: Within this much of the COLUMN fraction (0.50 at 32 -> 16), width binds on
#: the delivered side and the horizon argument has to be re-made against a real
#: cost.  Between the two bands is the honest "partial" verdict.
DELIVERED_BINDS_MARGIN = 0.10


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("delivered-rank width contrast")
    p.add_argument("--parent", required=True, help="walk JSON at the wide W")
    p.add_argument("--branch", required=True, help="walk JSON at the narrow W")
    p.add_argument("--out", default=None)
    return p.parse_args()


def _row(tag, rec):
    g, c = rec.get("geometry", {}), rec.get("final_carry", {})
    return {"arm": tag, "n_vec": g.get("n_vec"), "rows": c.get("rows"),
            "eff_rank_pr": c.get("e_eff_rank_pr"),
            "eff_rank_entropy": c.get("e_eff_rank_entropy"),
            "centred_cosine": c.get("e_centred_cosine"),
            "row_norm": c.get("e_row_norm")}


def score(parent: dict, branch: dict) -> dict:
    a, b = _row("parent", parent), _row("branch", branch)
    out = {"parent": a, "branch": b}
    if a["n_vec"] and b["n_vec"] and a["n_vec"] == b["n_vec"]:
        out["verdict"] = "INVALID"
        out["why"] = (f"both walks ran at W={a['n_vec']} -- there is no width "
                      f"contrast here.  Pass --set accum_vecs on each side.")
        return out
    if not (a["eff_rank_pr"] and b["eff_rank_pr"]):
        out["verdict"] = "INVALID"
        out["why"] = ("a walk reported no carried-block rank; it probably ran "
                      "with no prefix buffer.")
        return out
    retained = b["eff_rank_pr"] / a["eff_rank_pr"]
    col_frac = b["n_vec"] / a["n_vec"]
    out["delivered_retained"] = retained
    out["column_fraction"] = col_frac
    if retained >= DELIVERED_RETAINED_SURVIVES:
        out["verdict"] = "SURVIVES"
        out["why"] = ("the delivered rank is essentially unchanged, so the "
                      "write basis given up was not reaching the token stream. "
                      "P1.0's bet holds on the axis that matters; report the "
                      "summary_emb loss as a measured cost that did not bind.")
    elif retained <= col_frac + DELIVERED_BINDS_MARGIN:
        out["verdict"] = "BINDS"
        out["why"] = ("delivered rank fell about in proportion to the columns, "
                      "so width binds on the delivered side too.  The response "
                      "is NOT reverting to the wide W -- that halves the "
                      "horizon and restores the eviction cliff the gate exists "
                      "to remove -- it is the cc=16 / K=128 cell, where the "
                      "extra laps pay for the width.")
    else:
        out["verdict"] = "PARTIAL"
        out["why"] = ("between the two registered bands: some delivered rank "
                      "was lost but less than proportionally.  Pre-registered "
                      "response is to run as specified and report the number, "
                      "NOT to pick whichever reading suits the result.")
    return out


def print_report(rec, out=sys.stdout) -> None:
    p = lambda *a: print(*a, file=out)
    f = lambda v: "     -" if v is None else f"{v:9.3f}"
    p("")
    p("  DELIVERED rank of the carried block -- the number that decides width")
    p(f"  {'arm':<8}{'W':>4}{'rows':>6}{'eff_rank_pr':>13}{'entropy':>10}"
      f"{'cos':>10}{'|row|':>10}")
    for tag in ("parent", "branch"):
        r = rec[tag]
        p(f"  {tag:<8}{(r['n_vec'] or 0):>4}{(r['rows'] or 0):>6}"
          f"{f(r['eff_rank_pr']):>13}{f(r['eff_rank_entropy']):>10}"
          f"{f(r['centred_cosine']):>10}{f(r['row_norm']):>10}")
    if "delivered_retained" in rec:
        p("")
        p(f"  delivered rank retained: {100 * rec['delivered_retained']:.0f}% "
          f"at {100 * rec['column_fraction']:.0f}% of the write columns")
    p("")
    p(f"  VERDICT: {rec['verdict']}")
    for line in rec["why"].split(". "):
        if line.strip():
            p(f"    {line.strip().rstrip('.')}.")
    p("")
    p("  This is a rank measurement, not a benchmark.  It says whether the")
    p("  carry's BASIS narrowed, not whether the arm is worse -- that is the")
    p("  cells' job (p11_z_probe_prereg.md).")
    p("")


def main() -> int:
    args = parse_args()
    with open(args.parent, encoding="utf-8") as fh:
        parent = json.load(fh)
    with open(args.branch, encoding="utf-8") as fh:
        branch = json.load(fh)
    rec = score(parent, branch)
    print_report(rec)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="ascii") as fh:
            json.dump(rec, fh, indent=2, default=str)
        print(f"wrote {args.out}")
    # An INVALID comparison is a failure; SURVIVES / PARTIAL / BINDS are all
    # legitimate RESULTS and must not fail the job -- a gate that goes red on a
    # negative finding teaches people to skip the gate.
    return 1 if rec["verdict"] == "INVALID" else 0


if __name__ == "__main__":
    raise SystemExit(main())
