#!/usr/bin/env python
"""J6's raw numbers.  No thresholds, no cuts, no verdict.

This is deliberately NOT a scorer.  J1-J5 each shipped a decision table, and
J5's headlined "NO" while its own fields supported NOT_READ -- the
pre-registered gate was `local_nll`, which is computed on answers that never
touch the buffer and so could not detect the E-only control failing to carry,
which is the exact failure it was written to catch.  Four limbs read by hand
is cheaper than another table that can be confidently wrong.

So: every cell, every paired delta, every CI, printed.  The one thing this
file asserts is that a number is what it says it is -- absolute NLLs are
printed beside every delta, because the J4 lesson is that a positive delta
between two worse-than-chance cells looks exactly like a channel.

    python tools/dump_j6.py --root eval_results/j6_readout

Reads the same results.json files evals/eval_carry_2x2.py writes, through
score_j1's helpers, so the pairing and the bootstrap are the ones every other
arm used.
"""
from __future__ import annotations

import argparse
import math
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "evals"))

from score_j1 import (  # noqa: E402
    cell_mean, effect, load_results, pg19_books,
)

LN10 = math.log(10.0)
# `delta` was cancelled 2026-09-27 (6 h node stall, no checkpoint) and
# REPLACED by J7's `stag`: raw staggered states + the read projection
# frozen at identity.  Kept in this order so the printed tables line up
# with the array indices 0-3.
LIMBS = ("ctrl", "mix", "stag", "slow")
#: Every limb trained with both channels live, so every limb is scored in the
#: same cell as J4's real limb.  ON is what the model operates in; OFF is Z
#: ablated -- "skipped" in the -off file, "another row's Z" in the -donor one.
ON, OFF, E_OFF = "E1Z1", "E1Z0", "E0Z0"
PG19_PACK = "data/pg19_olmo_validation_len4096_strided"


def run_name(limb: str) -> str:
    return f"j6-a3z-{limb}"


def f4(v):
    return "    -   " if v is None else f"{v:8.4f}"


def d4(e):
    if e is None:
        return "     -"
    return f"{e['mean']:+.4f} [{e['lo']:+.4f}, {e['hi']:+.4f}]"


def _rep(root, limb, key):
    return load_results(os.path.join(root, run_name(limb), key))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="eval_results/j6_readout")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--pg19_pack", default=PG19_PACK)
    a = ap.parse_args()
    root = a.root if os.path.isabs(a.root) else os.path.join(REPO, a.root)

    books = None
    try:
        books = pg19_books(os.path.join(REPO, a.pg19_pack))
    except Exception as exc:                                  # noqa: BLE001
        print(f"[pg19] book clusters unavailable ({exc}); PG-19 CIs will be "
              f"row-level and the ragged rows are NOT dropped -- treat the "
              f"prose block as indicative only.\n")
    clusters, ragged = (books if books else (None, frozenset()))

    present = [l for l in LIMBS if _rep(root, l, "carry-carry-off")]
    missing = [l for l in LIMBS if l not in present]
    print("=" * 78)
    print("J6 RAW NUMBERS -- no cuts applied, no verdict.  Chance = ln10 = "
          f"{LN10:.4f}")
    print("Every limb scored in its operating cell E1Z1; Z-ablated cell E1Z0.")
    print("X = NLL(ablated) - NLL(operating).  POSITIVE = the live channel is "
          "better.")
    if missing:
        print(f"\n!! MISSING LIMBS: {', '.join(missing)} -- nothing below "
              f"compares against them.")
    print("=" * 78)

    # ---- 1. absolute cells, the table that goes first ---------------------
    print("\n[1] CARRY TASK -- ABSOLUTE cell NLLs.  Read these BEFORE any "
          "delta:\n    a positive X between two worse-than-chance cells is "
          "J4's signature, not a channel.\n")
    print(f"    {'limb':6s} {'file':7s} {ON:>9s} {OFF:>9s} {E_OFF:>9s}   "
          f"{'vs chance':>10s}")
    for limb in present:
        for tag, key in (("off", "carry-carry-off"),
                         ("donor", "carry-carry-donor")):
            r = _rep(root, limb, key)
            if not r:
                continue
            on = cell_mean(r, ON)
            row = (f"    {limb:6s} {tag:7s} {f4(on)} {f4(cell_mean(r, OFF))} "
                   f"{f4(cell_mean(r, E_OFF))}")
            if tag == "off" and on is not None:
                row += f"   {LN10 - on:+10.4f}"
            print(row)

    # ---- 2. the deciding deltas -------------------------------------------
    print("\n[2] CARRY TASK -- paired within-model deltas (n=300, 95% row "
          "bootstrap).\n    X_content is the deciding cell: does the channel "
          "hold CONTENT?\n")
    print(f"    {'limb':6s} {'X_content (donor Z - own Z)':32s} "
          f"{'X_off (Z skipped - own Z)':32s} {'E_margin':>24s}")
    for limb in present:
        off_r = _rep(root, limb, "carry-carry-off")
        don_r = _rep(root, limb, "carry-carry-donor")
        xc = (effect(don_r, OFF, don_r, ON, n_boot=a.boot) if don_r else None)
        xo = (effect(off_r, OFF, off_r, ON, n_boot=a.boot) if off_r else None)
        em = (effect(off_r, E_OFF, off_r, OFF, n_boot=a.boot) if off_r else None)
        print(f"    {limb:6s} {d4(xc):32s} {d4(xo):32s} {d4(em):>24s}")
    print("\n    J4's control read X_content +0.0001 [-0.0000, +0.0004] and "
          "X_off +0.4716.\n    A limb that moved X_content is the only thing "
          "this arm was built to find.")

    # ---- 3. the positive control ------------------------------------------
    print("\n[3] LOCAL answers -- computable from the current chunk alone, so "
          "they never\n    touch the buffer.  This says the model trained; it "
          "says nothing about Z.\n")
    print(f"    {'limb':6s} {'operating':>10s} {'Z ablated':>10s}")
    for limb in present:
        r = _rep(root, limb, "carry-local-off")
        if r:
            print(f"    {limb:6s} {f4(cell_mean(r, ON)):>10s} "
                  f"{f4(cell_mean(r, OFF)):>10s}")

    # ---- 4. off-task -------------------------------------------------------
    n_books = len(set(clusters.values())) if clusters else None
    how = ("Book-clustered CIs, ragged rows dropped" if clusters
           else "ROW-level CIs, ragged rows KEPT")
    if n_books:
        how += f" ({n_books} books)"
    print(f"\n[4] PG-19 off-task.  {how}.\n    Prose has no chance baseline, "
          "so only the within-model contrasts mean anything.\n")
    print(f"    {'limb':6s} {'NLL':>9s}  {'X_content':32s} {'X_off':32s}")
    for limb in present:
        off_r = _rep(root, limb, "pg19-all-off")
        don_r = _rep(root, limb, "pg19-all-donor")
        kw = dict(clusters_of=clusters, drop_rows=ragged, n_boot=a.boot)
        xc = (effect(don_r, OFF, don_r, ON, **kw) if don_r else None)
        xo = (effect(off_r, OFF, off_r, ON, **kw) if off_r else None)
        nll = cell_mean(off_r, ON, drop_rows=ragged) if off_r else None
        print(f"    {limb:6s} {f4(nll):>9s}  {d4(xc):32s} {d4(xo):32s}")
    print("\n    J5's zonly read +0.0798 [+0.0718, +0.0913] here -- the only "
          "donor swap in\n    the project that ever cost real nats, and it "
          "was off-task and E-free.")

    # ---- 5. between limb, and the warning that goes with it ----------------
    print("\n[5] BETWEEN LIMB -- carry NLL in each limb's operating cell.\n")
    base = None
    for limb in present:
        r = _rep(root, limb, "carry-carry-off")
        v = cell_mean(r, ON) if r else None
        if limb == "ctrl":
            base = v
        delta = ("" if (v is None or base is None or limb == "ctrl")
                 else f"   {v - base:+.4f} vs ctrl")
        print(f"    {limb:6s} {f4(v)}{delta}")
    print("\n    THESE ARE BETWEEN-RUN NUMBERS.  E's own carry skill drifts "
          "0.37-0.85 nats\n    across ten same-seed runs, so a difference "
          "smaller than that is not\n    evidence of anything.  It is printed "
          "because `mix`'s deciding metric is\n    the absolute carry NLL "
          "against ctrl -- donor-mixing trains X_content\n    directly, which "
          "demotes it to a manipulation check ON THAT LIMB ONLY.")
    print("\n" + "=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
