"""J5's pre-registered rules -- ../j5_prereg.md, applied to
pace/j5_readout.sbatch's output.  Nothing here may be decided after the numbers.

WHAT J5 ASKS, IN ONE SENTENCE.  Three limbs finetuned from the same base
(`ckpts/olmo8-cortex`, whose recurrence is TRAINED) with the same budget, seed
and data order, differing only in which channels of the carry buffer they read:
Z only, E only, Z + E.  Does the latent channel carry task content when the
latent space is mature -- and, in the Z-only limb, when nothing else carries?

WHY A SEPARATE SCORER FROM score_j1.  J1/J3/J4 share one decision table because
they share one limb set (real / donor / noread) and one question (does Z ADD to
E?).  J5's limbs are the channels themselves and its primary contrast is
WITHIN model rather than between limbs, because between-limb carry differences
on this task drift 0.37-0.85 nats across same-seed runs.  Forcing that into
score_j1's table would have meant redefining "real" and "noread" per limb --
the kind of overloading that makes a table unreadable.  Everything that is
GENERIC is imported from score_j1 rather than copied: the bootstrap, the
pairing, the PG-19 book clustering, the trajectory readers, and every
threshold, so J5's numbers sit on exactly the J arms' scale.

THE THREE NUMBERS THAT DECIDE, and the order matters (../j5_prereg.md S2):

  1. ABSOLUTE.  The limb's carry NLL against ln 10 = 2.3026, the
     no-information point.  Primary because J4's within-model D values
     INVERTED when read without it: +0.8245 looked like Z helping while the
     absolute NLLs showed the limb sitting AT chance (2.3155) and the
     comparison cell WORSE than chance (3.1401).
  2. CONTENT.  X_content = NLL(another row's Z) - NLL(own Z), within model,
     same rows, paired, row bootstrap.  This is the cell that caught J4:
     0.7481 vs 0.7482, four decimals.  It is the only one that separates a
     channel from content-free capacity the model leans on.
  3. LOAD.  X_off = NLL(Z skipped) - NLL(own Z), within model.  Reported
     BESIDE content, never instead of it: J4 scored +0.4716 here while content
     scored +0.0001, and that PAIR is the signature of capacity-not-channel.

Every between-limb number (zonly vs eonly, both vs eonly) is reported and
labelled DRIFT-SUBJECT, and none of them can produce a PASS on their own.

    python evals/score_j5.py --root eval_results/j5_readout --step 4000
    python evals/score_j5.py --root eval_results/j5_readout --step 4000 \
        --also_step 2000            # the trajectory (../j5_prereg.md S1.5)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from score_j1 import (                                            # noqa: E402
    BOOT, E_CARRY_BAND, E_CARRY_MAX, LEAK_MARGIN, LN10,
    MIN_EFFECT_CARRY, NOREAD_ZMAIN_TOL, PAIR_MIN_FRAC,
    TASK_LEARNED_LOCAL_MAX, cell_mean, effect, embed_trajectory, f4, fmt,
    load_results, pg19_books,
)

ENCODING = "endpoint"
CARRY_PACK = "data/carry_task_r16_len4096_val"
PG19_PACK = "data/pg19_olmo_validation_len4096_strided"
PACKS = {"carry": CARRY_PACK, "pg19": PG19_PACK}

#: The three limbs, and THE CELL EACH ONE OPERATES IN.  This mapping is the
#: single most load-bearing thing in the file: score a limb in another limb's
#: cell and the number is an out-of-distribution condition reported as the
#: arm's result.
#:   zonly  trained with E's spliced rows zeroed  -> its cell is E0*
#:   eonly  trained with Z's columns zeroed       -> its cell is E1Z0
#:   both   trained with both live                -> its cell is E1Z1
LIMBS = ("zonly", "eonly", "both")
OPERATING_CELL = {"zonly": "E0Z1", "eonly": "E1Z0", "both": "E1Z1"}
#: The cell that ablates Z inside the same eval run.  Under --z_null off it is
#: "no read at all"; under --z_null donor it is another row's Z.  E is held at
#: the limb's own level in both, so the ONLY thing that changes is Z.
Z_ABLATED_CELL = {"zonly": "E0Z0", "eonly": "E1Z1", "both": "E1Z0"}

#: pace/j5_readout.sbatch builds EXACTLY these, in this order
#: (tests/test_j5_arm.py parses the sbatch and compares).
#: (key, limb, pack, score, z_null, n, cells)
TASKS = (
    ("zonly_carry_off",   "zonly", "carry", "carry", "off",   300, "E0Z1,E0Z0"),
    ("zonly_carry_donor", "zonly", "carry", "carry", "donor", 300, "E0Z1,E0Z0"),
    ("zonly_local_off",   "zonly", "carry", "local", "off",   100, "E0Z1,E0Z0"),
    ("zonly_pg19_off",    "zonly", "pg19",  "all",   "off",   400, "E0Z1,E0Z0"),
    ("zonly_pg19_donor",  "zonly", "pg19",  "all",   "donor", 400, "E0Z1,E0Z0"),
    ("eonly_carry_off",   "eonly", "carry", "carry", "off",   300, "E1Z1,E1Z0,E0Z0"),
    ("eonly_local_off",   "eonly", "carry", "local", "off",   100, "E1Z1,E1Z0"),
    ("eonly_pg19_off",    "eonly", "pg19",  "all",   "off",   400, "E1Z0,E0Z0"),
    ("both_carry_off",    "both",  "carry", "carry", "off",   300, "E1Z1,E1Z0,E0Z1,E0Z0"),
    ("both_carry_donor",  "both",  "carry", "carry", "donor", 300, "E1Z1,E1Z0,E0Z1,E0Z0"),
    ("both_local_off",    "both",  "carry", "local", "off",   100, "E1Z1,E1Z0"),
    ("both_pg19_off",     "both",  "pg19",  "all",   "off",   400, "E1Z1,E1Z0"),
    ("both_pg19_donor",   "both",  "pg19",  "all",   "donor", 400, "E1Z1,E1Z0"),
)


def run_name(limb: str) -> str:
    """The TRAINING run's name, exactly as pace/j5_joint.sbatch builds it."""
    if limb not in LIMBS:
        raise ValueError(f"limb must be one of {LIMBS}; got {limb!r}")
    return f"j5-a3z-{ENCODING}-{limb}"


def task_dir(root: str, task: tuple, step: int) -> str:
    _, limb, pack, score, znull, _, _ = task
    return os.path.join(root, f"step{step}", run_name(limb),
                        f"{pack}-{score}-{znull}")


# ─── vetoes ─────────────────────────────────────────────────────────────────

def veto_reasons(rep: Optional[dict], task: tuple, step: int) -> list:
    """Every reason this results file may not be read.  Empty = clean.

    score_j1.veto_reasons cannot be reused: it closes over that module's fixed
    STEP and run-name scheme, and J5 reads two steps of a different scheme.
    The CHECKS are the same ones, deliberately.
    """
    key, limb, pack, score, znull, n, cells = task
    if rep is None:
        return [f"{key}: MISSING (no results.json)"]
    cfg = rep.get("config", {})
    out = []
    if cfg.get("score") != score:
        out.append(f"{key}: scored {cfg.get('score')!r}, registered {score!r}")
    if cfg.get("z_null") != znull:
        out.append(f"{key}: z_null {cfg.get('z_null')!r}, registered {znull!r}")
    data = cfg.get("data")
    if data is not None and os.path.normpath(data) != os.path.normpath(PACKS[pack]):
        out.append(f"{key}: read {data}, registered {PACKS[pack]}")
    ck = cfg.get("checkpoint")
    want = f"{run_name(limb)}/checkpoint_{step}"
    if ck is None or not os.path.normpath(ck).replace("\\", "/").endswith(want):
        out.append(f"{key}: checkpoint {ck!r} is not .../{want}")
    if rep.get("chunk1_ok") is not True:
        out.append(f"{key}: chunk-1 veto (spread {rep.get('chunk1_max_spread')})")
    if (rep.get("health") or {}).get("at_chance"):
        out.append(f"{key}: at chance -- RED 10's shape")
    if not rep.get("z_channel"):
        out.append(f"{key}: no Z channel on this build (check the --set flags)")
    got = int(cfg.get("samples") or 0)
    if got < PAIR_MIN_FRAC * n:
        out.append(f"{key}: {got} samples of {n} requested")
    # The registered cells must all be present, or a later effect would be
    # computed from a cell that never ran -- or, worse, silently omitted.
    # config.cells is the LIST eval_carry_2x2 wrote, not the comma string the
    # launcher passed -- reading it as a string silently matched nothing.
    ran = set(cfg.get("cells") or [])
    want_cells = set(cells.split(","))
    if ran and not want_cells <= ran:
        out.append(f"{key}: cells {sorted(ran)} ran, registered {sorted(want_cells)}")
    return out


# ─── the decision table ─────────────────────────────────────────────────────

def limb_reading(carry_nll: Optional[float], x_content: Optional[dict],
                 x_off: Optional[dict]) -> str:
    """One limb's reading on the carry task, in the registered order.

    ABSOLUTE FIRST.  A limb whose carry NLL does not clear chance cannot have a
    channel however its within-model deltas read -- that is J4's +0.8245
    lesson, where a positive D sat on a comparison cell WORSE than chance.
    """
    if carry_nll is None:
        return "NOT_READ"
    if carry_nll >= LN10 - MIN_EFFECT_CARRY:
        return "AT_CHANCE"
    if x_content is None:
        return "CARRIES_no_content_cell"
    if x_content["lo"] > 0 and x_content["mean"] >= MIN_EFFECT_CARRY:
        return "CARRIES_CONTENT"
    # Clears chance, but its own content is not what did it.  Distinguish the
    # two ways that happens, because they mean different things: columns the
    # model leans on regardless of content (J4's signature), versus columns it
    # does not use at all.
    if x_off is not None and x_off["lo"] > 0 and x_off["mean"] >= MIN_EFFECT_CARRY:
        return "CAPACITY_NOT_CHANNEL"
    return "CHANNEL_UNUSED"


def overall(zonly: str, eonly_learned: bool, both: str) -> str:
    """The one-line answer to '../j5_prereg.md S0'.

    `eonly` GATES EVERYTHING.  With E's read zeroed there is no E-on cell
    inside the treatment limb to normalise against, so without a control that
    learned the task on this base, at this budget, with this fresh memory path,
    "Z did not learn" and "nothing learns here in 4,000 updates" are one number
    (../j5_prereg.md S1.1, veto 8).
    """
    if not eonly_learned:
        return ("NOT_READ: the E-only control did not learn the task on this "
                "base at this budget, so no Z number is interpretable "
                "(../j5_prereg.md S1.1)")
    if zonly == "CARRIES_CONTENT":
        if both == "CARRIES_CONTENT":
            return ("Z_LEARNS: the latent channel carries its own content with "
                    "a trained latent space, and still does beside E")
        return ("Z_LEARNS_ALONE: the latent channel carries its own content "
                "when nothing else carries, but adds no measurable content "
                "beside E")
    if zonly == "CAPACITY_NOT_CHANNEL":
        return ("Z_CAPACITY_NOT_CHANNEL: the columns are load-bearing and "
                "content-free -- J4's failure mode reproduced under the best "
                "available conditions, which makes it a STRONGER negative "
                "than J4's, not a partial pass")
    if zonly in ("AT_CHANCE", "CHANNEL_UNUSED"):
        return ("NO: the carried loop state does not learn to carry content "
                "even with a trained latent space, no competitor and full "
                "pressure on every carry row")
    return f"NOT_READ: zonly reads {zonly} -- see the vetoes"


# ─── scoring one step ───────────────────────────────────────────────────────

def score_step(root: str, step: int, books: Optional[tuple],
               n_boot: int) -> dict:
    reps, vetoes = {}, []
    for t in TASKS:
        reps[t[0]] = load_results(task_dir(root, t, step))
        vetoes += veto_reasons(reps[t[0]], t, step)
    out: dict = {"step": step, "vetoes": vetoes, "limbs": {}}
    if vetoes:
        out["reading"] = "VETOED"
        return out
    r = reps

    # ── instrument check: the E-only limb must read exactly no Z ────────────
    # Its E1Z1 and E1Z0 cells are the SAME model (latent_carry_read=false), so
    # they must agree to floating point.  A larger gap means the rebuild did
    # not apply the flag and every eonly number is about another model.
    try:
        v = effect(r["eonly_carry_off"], "E1Z1", r["eonly_carry_off"], "E1Z0",
                   n_boot=10)["mean"]
    except KeyError:
        v = None
    if v is None or not abs(v) <= NOREAD_ZMAIN_TOL:
        out["vetoes"].append(
            f"eonly Z_main {v}: the E-only limb reads something, or its cells "
            f"did not both run.  latent_carry_read=false did not apply.")
        out["reading"] = "VETOED"
        return out

    # ── the task is learnable here at all (../j5_prereg.md S1.1, veto 8) ────
    local = {limb: cell_mean(r[f"{limb}_local_off"], OPERATING_CELL[limb])
             for limb in LIMBS}
    out["local_nll"] = local
    out["task_learned"] = {limb: (local[limb] is not None
                                  and local[limb] <= TASK_LEARNED_LOCAL_MAX)
                           for limb in LIMBS}
    eonly_learned = out["task_learned"]["eonly"]

    # ── the no-information floor, measured rather than assumed ──────────────
    # zonly with BOTH channels off is the condition in which nothing can be
    # known about an earlier chunk.  If it scores carry answers below
    # ln10 - LEAK_MARGIN, something reaches across chunks that is not the
    # buffer, and every number here is measuring that instead.
    floor = cell_mean(r["zonly_carry_off"], "E0Z0")
    out["no_carry_floor"] = floor
    if floor is None or floor < LN10 - LEAK_MARGIN:
        out["vetoes"].append(
            f"THE FLOOR LEAKS: with neither channel read, carry answers score "
            f"{f4(floor)} against ln10 {LN10:.4f} (margin {LEAK_MARGIN}).  "
            f"Something carries that is not the buffer.")
        out["reading"] = "VETOED"
        return out

    # ── per limb, on the carry task, in its own cell ────────────────────────
    for limb in LIMBS:
        on, off = OPERATING_CELL[limb], Z_ABLATED_CELL[limb]
        rep_off = r[f"{limb}_carry_off"]
        rep_donor = r.get(f"{limb}_carry_donor")
        carry_nll = cell_mean(rep_off, on)
        rec: dict = {"operating_cell": on, "carry_nll": carry_nll,
                     "chance": LN10,
                     "margin_vs_chance": (None if carry_nll is None
                                          else LN10 - carry_nll)}
        if limb == "eonly":
            # No Z to ablate.  What this limb contributes instead is THE SCALE:
            # E's own carry margin, within model, on the same rows.
            rec["X_off"] = None
            rec["X_content"] = None
            rec["E_margin"] = effect(rep_off, "E0Z0", rep_off, "E1Z0",
                                     n_boot=n_boot)
            rec["reading"] = ("E_LEARNED" if eonly_learned
                              and carry_nll is not None
                              and carry_nll < LN10 - MIN_EFFECT_CARRY
                              else "E_DID_NOT_LEARN")
        else:
            rec["X_off"] = effect(rep_off, off, rep_off, on, n_boot=n_boot)
            rec["X_content"] = (None if rep_donor is None else
                                effect(rep_donor, off, rep_donor, on,
                                       n_boot=n_boot))
            if limb == "both":
                # E's margin inside the limb that has both: E off, Z off in
                # both cells, so only E changes.  Out of distribution (this
                # limb trained with E always on) and labelled as such.
                rec["E_margin_ood"] = effect(rep_off, "E0Z0", rep_off, "E1Z0",
                                             n_boot=n_boot)
                # And the cell J4's E-dropout pair approximated: Z alone on a
                # model that never trained that way.  Descriptive only -- the
                # in-distribution version of this question IS the zonly limb.
                rec["z_alone_ood"] = cell_mean(rep_off, "E0Z1")
            rec["reading"] = limb_reading(carry_nll, rec["X_content"],
                                          rec["X_off"])
        out["limbs"][limb] = rec

    # ── between-limb, reported and never decisive (S7: 0.37-0.85 nat drift) ─
    nll = {limb: out["limbs"][limb]["carry_nll"] for limb in LIMBS}
    out["between_limb"] = {
        "note": "DRIFT-SUBJECT: E's own carry skill moves 0.37-0.85 nats "
                "across ten same-seed runs.  These label nothing.",
        "carry_nll": nll,
        "zonly_minus_eonly": (None if None in (nll["zonly"], nll["eonly"])
                              else nll["zonly"] - nll["eonly"]),
        "both_minus_eonly": (None if None in (nll["both"], nll["eonly"])
                             else nll["both"] - nll["eonly"]),
    }
    # E-HEALTH, J4's audit line on the same scale: E saturating the task and E
    # having lost it are the two ways the E-only cell stops being informative.
    out["e_health"] = {"eonly_carry_nll": nll["eonly"],
                       "band": list(E_CARRY_BAND), "max": E_CARRY_MAX,
                       "degraded": (nll["eonly"] is not None
                                    and nll["eonly"] > E_CARRY_MAX)}

    # ── the off-task cell: does the content-free signature reappear on prose? ─
    if books is not None:
        clusters_of, ragged = books
        pg = {}
        for limb in LIMBS:
            on, off = OPERATING_CELL[limb], Z_ABLATED_CELL[limb]
            rep_off = r[f"{limb}_pg19_off"]
            rep_donor = r.get(f"{limb}_pg19_donor")
            if limb == "eonly":
                pg[limb] = {"nll": cell_mean(rep_off, on, ragged),
                            "E_margin": effect(rep_off, "E0Z0", rep_off, "E1Z0",
                                               clusters_of, ragged, n_boot)}
                continue
            pg[limb] = {
                "nll": cell_mean(rep_off, on, ragged),
                "X_off": effect(rep_off, off, rep_off, on, clusters_of,
                                ragged, n_boot),
                "X_content": (None if rep_donor is None else
                              effect(rep_donor, off, rep_donor, on,
                                     clusters_of, ragged, n_boot)),
            }
        pg["ragged_rows_dropped"] = len(ragged)
        out["pg19"] = pg

    out["answer"] = overall(out["limbs"]["zonly"]["reading"], eonly_learned,
                            out["limbs"]["both"]["reading"])
    out["reading"] = out["limbs"]["zonly"]["reading"]
    return out


# ─── the trajectory ─────────────────────────────────────────────────────────

def trajectory(early: dict, late: dict) -> dict:
    """The 2,000 -> 4,000 direction (../j5_prereg.md S1.5).

    IT LABELS NOTHING ON ITS OWN, and that is registered.  A NO whose numbers
    are MOVING toward the cuts is a different finding from a NO that is flat,
    and only the second one closes the design -- but neither is a PASS.
    """
    out = {"from_step": early.get("step"), "to_step": late.get("step"),
           "note": "descriptive: the direction, not a decision rule"}
    if early.get("reading") == "VETOED" or late.get("reading") == "VETOED":
        out["available"] = False
        return out
    out["available"] = True
    for limb in LIMBS:
        a, b = early["limbs"].get(limb, {}), late["limbs"].get(limb, {})
        row = {}
        if a.get("carry_nll") is not None and b.get("carry_nll") is not None:
            # NEGATIVE = improving: the NLL fell between the two read-outs.
            row["d_carry_nll"] = b["carry_nll"] - a["carry_nll"]
        for k in ("X_content", "X_off"):
            if a.get(k) and b.get(k):
                row[f"d_{k}"] = b[k]["mean"] - a[k]["mean"]
        if row:
            out[limb] = row
    return out


# ─── printing ───────────────────────────────────────────────────────────────

def print_step(v: dict) -> None:
    print("=" * 78)
    print(f"J5 VERDICT, step {v['step']} (pre-registered: evals/score_j5.py, "
          f"../j5_prereg.md)")
    print("  Each limb is scored IN ITS OWN TRAINED CELL.  X = NLL(ablated) - "
          "NLL(operating);")
    print("  POSITIVE = the live channel is better.  Chance (no information) "
          f"= ln 10 = {LN10:.4f}.")
    print("=" * 78)
    for x in v.get("vetoes", []):
        print(f"  VETO  {x}")
    if v.get("reading") == "VETOED":
        return
    print(f"  local-answer NLL (task learned <= {TASK_LEARNED_LOCAL_MAX:.4f}):")
    for limb in LIMBS:
        print(f"    {limb:<6} {f4(v['local_nll'][limb])}  "
              f"{'learned' if v['task_learned'][limb] else 'NOT LEARNED'}")
    print(f"  no-carry floor (zonly, both channels off)  {f4(v['no_carry_floor'])}"
          f"  (leak if < {LN10 - LEAK_MARGIN:.4f})")
    print("")
    for limb in LIMBS:
        rec = v["limbs"][limb]
        print(f"  --- {limb.upper()} (cell {rec['operating_cell']}) -> "
              f"{rec['reading']}")
        print(f"      carry NLL       {f4(rec['carry_nll'])}   "
              f"margin under chance {f4(rec['margin_vs_chance'])}  "
              f"(needs > {MIN_EFFECT_CARRY})")
        if rec.get("X_content") is not None:
            print(f"      X_content       {fmt(rec['X_content'])}   "
                  f"(donor's Z minus own Z -- THE deciding cell)")
        if rec.get("X_off") is not None:
            print(f"      X_off           {fmt(rec['X_off'])}   "
                  f"(Z skipped minus own Z -- load, not content)")
        if rec.get("E_margin") is not None:
            print(f"      E_margin        {fmt(rec['E_margin'])}   "
                  f"(E off minus E on -- THE SCALE every Z number is quoted "
                  f"against)")
        if rec.get("E_margin_ood") is not None:
            print(f"      E_margin (OOD)  {fmt(rec['E_margin_ood'])}   "
                  f"(this limb trained with E always on)")
        if rec.get("z_alone_ood") is not None:
            print(f"      Z alone  (OOD)  {f4(rec['z_alone_ood'])}   "
                  f"(E blanked on a limb that never trained that way)")
    b = v["between_limb"]
    print("")
    print(f"  between limbs -- {b['note']}")
    print(f"    carry NLL  zonly {f4(b['carry_nll']['zonly'])}  "
          f"eonly {f4(b['carry_nll']['eonly'])}  both {f4(b['carry_nll']['both'])}")
    print(f"    zonly - eonly  {f4(b['zonly_minus_eonly'])}     "
          f"both - eonly  {f4(b['both_minus_eonly'])}")
    eh = v["e_health"]
    print(f"    E-health: eonly carry NLL {f4(eh['eonly_carry_nll'])} "
          f"(observed band {eh['band'][0]}-{eh['band'][1]}, max {eh['max']})"
          + ("  DEGRADED" if eh["degraded"] else ""))
    if "pg19" in v:
        pg = v["pg19"]
        print(f"  PG-19 off-task (book-clustered, "
              f"{pg['ragged_rows_dropped']} ragged rows dropped)")
        for limb in LIMBS:
            row = pg[limb]
            line = f"    {limb:<6} NLL {f4(row['nll'])}"
            if row.get("X_content") is not None:
                line += f"   X_content {fmt(row['X_content'])}"
            if row.get("X_off") is not None:
                line += f"   X_off {fmt(row['X_off'])}"
            if row.get("E_margin") is not None:
                line += f"   E_margin {fmt(row['E_margin'])}"
            print(line)
    print("-" * 78)
    print(f"  ANSWER (step {v['step']}): {v['answer']}")


def print_trajectory(t: dict) -> None:
    print("")
    print(f"  TRAJECTORY {t['from_step']} -> {t['to_step']} -- {t['note']}")
    if not t.get("available"):
        print("    not available (one of the two steps was vetoed or missing)")
        return
    for limb in LIMBS:
        row = t.get(limb)
        if not row:
            continue
        bits = []
        if "d_carry_nll" in row:
            d = row["d_carry_nll"]
            # "flat" is a THIRD reading, not a rounding of one of the other
            # two: a null that does not move is the one that closes the design
            # (../j5_prereg.md S4), so it must not print as "worsening".
            way = "flat" if abs(d) < 1e-4 else ("improving" if d < 0
                                                else "worsening")
            bits.append(f"carry NLL {d:+.4f} ({way})")
        if "d_X_content" in row:
            bits.append(f"X_content {row['d_X_content']:+.4f}")
        if "d_X_off" in row:
            bits.append(f"X_off {row['d_X_off']:+.4f}")
        print(f"    {limb:<6} " + "   ".join(bits))
    print("    A null whose numbers MOVE toward the cuts is a different "
          "finding from a flat null;")
    print("    only the flat one closes the design.  Neither is a PASS.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="eval_results/j5_readout")
    ap.add_argument("--step", type=int, default=4000,
                    help="the DECIDING read-out (../j5_prereg.md S4)")
    ap.add_argument("--also_step", type=int, default=2000,
                    help="the earlier read-out, for the trajectory; 0 skips it")
    ap.add_argument("--runs_root", default="cortex-retrofit",
                    help="where the training runs keep cortex_diag.jsonl")
    ap.add_argument("--pg19", default=PG19_PACK,
                    help="the PG-19 pack, for book clusters and ragged rows; "
                         "'' skips the prose read-out")
    ap.add_argument("--boot", type=int, default=BOOT)
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    books = pg19_books(a.pg19) if a.pg19 else None
    late = score_step(a.root, a.step, books, a.boot)
    early = (score_step(a.root, a.also_step, books, a.boot)
             if a.also_step else None)

    print_step(late)
    if early is not None:
        print("")
        print_step(early)
        print_trajectory(trajectory(early, late))

    # z_embed_ratio: the read-strength trajectory, and on J5 it is the only
    # number that can say a positive reading came from a read that is still on.
    embeds = {}
    for limb in LIMBS:
        et = embed_trajectory(os.path.join(a.runs_root, run_name(limb),
                                           "cortex_diag.jsonl"))
        if et is not None:
            embeds[run_name(limb)] = et
    for run, et in embeds.items():
        lof = et["last_over_first"]
        print(f"  embed ratio {run}: ||Z col||/||E col|| {et['first']:.4f} @ "
              f"{et['first_step']} -> {et['last']:.4f} @ {et['last_step']}"
              + (f"  (x{lof:.2f})" if lof is not None else "")
              + ("  COLLAPSED" if et["collapsed"] else ""))

    rec = {"experiment": "j5", "encoding": ENCODING,
           "deciding_step": a.step, "steps": {str(a.step): late},
           "answer": late.get("answer"), "embed_ratio": embeds,
           "constants": {"LN10": LN10,
                         "TASK_LEARNED_LOCAL_MAX": TASK_LEARNED_LOCAL_MAX,
                         "MIN_EFFECT_CARRY": MIN_EFFECT_CARRY,
                         "LEAK_MARGIN": LEAK_MARGIN,
                         "E_CARRY_MAX": E_CARRY_MAX,
                         "E_CARRY_BAND": list(E_CARRY_BAND),
                         "NOREAD_ZMAIN_TOL": NOREAD_ZMAIN_TOL,
                         "PAIR_MIN_FRAC": PAIR_MIN_FRAC}}
    if early is not None:
        rec["steps"][str(a.also_step)] = early
        rec["trajectory"] = trajectory(early, late)
    out = a.out or os.path.join(a.root, "verdict.json")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(rec, fh, indent=2)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
