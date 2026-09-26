"""
J1's PRE-REGISTERED VERDICT.  The document is ../j1_prereg.md (repo root);
THIS FILE holds its thresholds and its decision table as code, committed before
any J1 read-out existed, so the rules and the instrument cannot drift -- the
same arrangement as evals/eval_gate_prereg.py for p10.

WHAT IT READS.  pace/j1_readout.sbatch runs evals/eval_carry_2x2.py once per
(limb, pack, score, z_null) and writes

    eval_results/j1_readout/<run>/<pack>-<score>-<znull>/results.json

where <run> is the TRAINING run name (j1-a3z-tokens-real, ...).  Each results
file keeps PER-SAMPLE NLLs and the pack rows they came from, so limbs trained
separately can be paired row by row here without a rerun.

EACH LIMB IS SCORED IN ITS OWN TRAINED CONDITION, E on:
    real    E1Z1               reads its own document's Z
    donor   E1Z0, z_null donor reads another document's Z (its in-batch roll
                               cannot run at the eval's batch 1)
    noread  E1Z1               carries Z, reads nothing

SIGN CONVENTION, one line, every number obeys it:
    D = NLL(comparison limb) - NLL(real limb)      POSITIVE = real is better.

    python evals/score_j1.py --root eval_results/j1_readout
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from typing import Optional

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ─── the pre-registered numbers ─────────────────────────────────────────────
#: An answer token is ONE digit (tools/prepare_carry_task.py), so an answer the
#: model has no information on costs ln(10) nats.  The task's own chance line.
LN10 = math.log(10.0)
#: TASK LEARNED: local answers (their register was updated in this chunk, so
#: the previous value is on the page) must resolve at least half the digit
#: entropy.  Above this the model has not learned the task and no carry number
#: means anything -- the carry pack's positive control.
TASK_LEARNED_LOCAL_MAX = LN10 / 2
#: Smallest effect on CARRY answers that counts, in nats: ~2% of the full
#: no-information-to-perfect range.  With ~300 rows x ~110 carry answers a
#: difference this size is resolvable; anything smaller is not a mechanism
#: worth a cortex-final design even when its CI clears zero.
MIN_EFFECT_CARRY = 0.05
#: HEADROOM: if the no-read limb (E alone) already scores carry answers below
#: this, E saturates the task and the main limbs cannot show Z at all.
HEADROOM_MIN = 0.10
#: E-OFF IS OFF: with E zeroed and no Z read, carry answers carry no
#: information, so the no-read E-dropout limb must sit near ln 10.  Further
#: below than this means something else carries register values into the
#: E-off cell, and the E-dropout verdict is not read.
LEAK_MARGIN = 0.25
#: The read gate starts at 0.1.  Below this at the end = the model refused the
#: read (the tier-1.5 oracle ended at 0.014 / 0.028).  Descriptive, except that
#: a collapsed gate beside a positive verdict is a contradiction -> AUDIT.
GATE_COLLAPSE = 0.03
#: J3 (j3_prereg.md S5).  The read's EFFECTIVE strength, z_read_ratio =
#: ||delta|| / ||x|| at the read site, can collapse through out_proj / v_proj
#: even with the gate slowed.  Collapsed = the mean of the LAST READ_WINDOW
#: diag rows below READ_COLLAPSE_FRAC x the mean of the FIRST READ_WINDOW.
#: 0.30 is the gate rule's own fraction (GATE_COLLAPSE 0.03 of the 0.1 init).
#: Windowed, not single rows: the ratio is an ACTIVATION measure off the last
#: forward before each diag row, so one row is one chunk's noise.  Same role
#: as the gate rule: descriptive, except beside a positive verdict -> AUDIT.
READ_COLLAPSE_FRAC = 0.30
READ_WINDOW = 4
#: rows paired across limbs / rows requested.  Below this the pairing lost
#: rows silently and the comparison is not the one registered.
PAIR_MIN_FRAC = 0.90
#: J4 ONLY (j4_prereg.md S5), and the reason it exists is new to J4: under
#: latent_encoding='endpoint' the Z write comes from the SAME summary columns
#: that produce E, so a carry objective on Z pushes on the very columns E is
#: read from.  J1's and J3's writes came from different positions and could not
#: do this.  E's own carry skill drifts 0.37-0.85 nats across ten same-seed
#: runs, which is more than enough to hide a regression, so the guard is the
#: OBSERVED BAND and not a tight threshold: one observed spread (0.48) above
#: the worst run on record (0.85), rounded to 1.20.  A noread limb whose carry
#: NLL sits above this has lost E, and J4 would be trading E for Z.
#:
#: IT IS AN AUDIT LINE, NOT A VETO, and deliberately: E degrading does not make
#: a NO on the E-off cell wrong (that cell has E off anyway), and it does not
#: make a positive reading wrong either, since both limbs share the regression.
#: What it changes is what J4 COSTS, which is a finding and has to be reported
#: rather than swallowed.
E_CARRY_MAX = 1.20
#: The band itself, reported beside the number so the reader can see what the
#: threshold was derived from rather than taking 1.20 on faith.
E_CARRY_BAND = (0.37, 0.85)
#: The no-read limb has no read, so its E1Z0 and E1Z1 cells are the SAME model
#: and must agree to fp noise.  A larger |Z_main| means it reads something.
NOREAD_ZMAIN_TOL = 1e-4
BOOT = 5000
PARENT_STEP, CELL_STEPS = 91552, 2000
STEP = PARENT_STEP + CELL_STEPS

CARRY_PACK = "data/carry_task_r16_len4096_val"
PG19_PACK = "data/pg19_olmo_validation_len4096_strided"

#: The read-out's task list.  pace/j1_readout.sbatch builds EXACTLY these
#: (tests/test_j1_readout.py parses the sbatch and compares), in this order.
#: (key, limb, pack, score, z_null, n, cells)
MAIN_TASKS = (
    ("real_carry_off",     "real",   "carry", "carry", "off",   300, "E1Z1,E1Z0"),
    ("real_carry_donor",   "real",   "carry", "carry", "donor", 300, "E1Z1,E1Z0"),
    ("donor_carry_donor",  "donor",  "carry", "carry", "donor", 300, "E1Z1,E1Z0"),
    ("noread_carry_off",   "noread", "carry", "carry", "off",   300, "E1Z1,E1Z0"),
    ("real_local_off",     "real",   "carry", "local", "off",   100, "E1Z1,E1Z0"),
    ("noread_local_off",   "noread", "carry", "local", "off",   100, "E1Z1,E1Z0"),
    ("real_pg19_off",      "real",   "pg19",  "all",   "off",   400, "E1Z1,E1Z0"),
    ("real_pg19_donor",    "real",   "pg19",  "all",   "donor", 400, "E1Z1,E1Z0"),
    ("donor_pg19_donor",   "donor",  "pg19",  "all",   "donor", 400, "E1Z1,E1Z0"),
    ("noread_pg19_off",    "noread", "pg19",  "all",   "off",   400, "E1Z1,E1Z0"),
)
#: The E-dropout pair trained with E blanked on 25% of rows, so its E-OFF
#: cells are in distribution and all four cells run.
EDROP_TASKS = (
    ("real_carry_off",     "real",   "carry", "carry", "off",   300, ""),
    ("real_carry_donor",   "real",   "carry", "carry", "donor", 300, ""),
    ("noread_carry_off",   "noread", "carry", "carry", "off",   300, ""),
    ("real_local_off",     "real",   "carry", "local", "off",   100, ""),
)
PACKS = {"carry": CARRY_PACK, "pg19": PG19_PACK}


#: Experiments this scorer reads.  J3 (the in-loop scratchpad) and J4 (the
#: carried Z as its own input_embeds columns) run the SAME limbs, packs, cells
#: and decision table as J1 -- their launchers are pace/j3_joint.sbatch and
#: pace/j4_joint.sbatch, their read-out is pace/j1_readout.sbatch with
#: EXPERIMENT=j3|j4 -- so the rules stay one piece of code.  Each design's own
#: thresholds go here as constants BEFORE any of its numbers exist: J3 added
#: READ_COLLAPSE_FRAC, J4 adds E_CARRY_MAX.
#:
#: WHAT IS DIFFERENT ABOUT J4, and all of it is bookkeeping rather than rules:
#:   * the encoding is 'endpoint' (--encoding endpoint), so run names read
#:     j4-a3z-endpoint-<limb>;
#:   * there is no read GATE and no injected delta, so the collapse rule reads
#:     `z_embed_ratio` instead (embed_trajectory), at the same fraction;
#:   * a Z-ONLY limb runs as a second E-dropout pair at --edrop 0.95, scored by
#:     score_edrop unchanged -- the limbs are still real/noread, so nothing in
#:     the task list or the decision table moves.
EXPERIMENTS = ("j1", "j3", "j4")


def run_name(limb: str, encoding: str = "tokens", edrop: str = "",
             experiment: str = "j1") -> str:
    """The TRAINING run's name, exactly as pace/j1_joint.sbatch (j1),
    pace/j3_joint.sbatch (j3) and pace/j4_joint.sbatch (j4) build it."""
    if experiment not in EXPERIMENTS:
        raise ValueError(f"experiment must be one of {EXPERIMENTS}; got {experiment!r}")
    tag = f"-edrop{edrop}" if edrop else ""
    return f"{experiment}-a3z-{encoding}{tag}-{limb}"


def task_dir(root: str, task: tuple, encoding: str, edrop: str,
             experiment: str = "j1") -> str:
    _, limb, pack, score, znull, _, _ = task
    return os.path.join(root, run_name(limb, encoding, edrop, experiment),
                        f"{pack}-{score}-{znull}")


# ─── statistics ─────────────────────────────────────────────────────────────

def boot_ci(deltas: list, clusters: Optional[list] = None,
            n_boot: int = BOOT, seed: int = 0) -> tuple:
    """(mean, lo, hi) of the paired deltas, 95% percentile bootstrap.

    `clusters` (same length as deltas) resamples WHOLE clusters -- PG-19 rows
    are consecutive windows of ~21 per book and are not independent, so a row
    bootstrap there would print a CI several times too narrow.  The carry
    pack's rows are independent documents by construction; pass None.
    """
    n = len(deltas)
    if n == 0:
        return (float("nan"),) * 3
    mean = sum(deltas) / n
    rng = random.Random(seed)
    if clusters is None:
        groups = [[d] for d in deltas]
    else:
        if len(clusters) != n:
            raise ValueError("clusters and deltas differ in length")
        by: dict = {}
        for d, c in zip(deltas, clusters):
            by.setdefault(c, []).append(d)
        groups = list(by.values())
    sums = [sum(g) for g in groups]
    cnts = [len(g) for g in groups]
    k = len(groups)
    stats = []
    for _ in range(n_boot):
        s = c = 0.0
        for _ in range(k):
            j = rng.randrange(k)
            s += sums[j]
            c += cnts[j]
        stats.append(s / c)
    stats.sort()
    lo = stats[int(0.025 * (n_boot - 1))]
    hi = stats[int(math.ceil(0.975 * (n_boot - 1)))]
    return mean, lo, hi


def paired(rep_a: dict, cell_a: str, rep_b: dict, cell_b: str) -> tuple:
    """Pair two results files row by row.  Returns (rows, a_vals, b_vals) over
    the rows BOTH scored, in row order."""
    def as_map(rep, cell):
        rows = rep.get("sample_rows") or []
        vals = (rep.get("per_sample") or {}).get(cell)
        if vals is None:
            raise KeyError(f"cell {cell} not in this results file "
                           f"(cells run: {rep.get('config', {}).get('cells')})")
        if len(rows) != len(vals):
            raise ValueError("sample_rows and per_sample disagree in length")
        return dict(zip(rows, vals))
    ma, mb = as_map(rep_a, cell_a), as_map(rep_b, cell_b)
    rows = sorted(set(ma) & set(mb))
    return rows, [ma[r] for r in rows], [mb[r] for r in rows]


def effect(rep_cmp: dict, cell_cmp: str, rep_real: dict, cell_real: str,
           clusters_of: Optional[dict] = None, drop_rows: frozenset = frozenset(),
           n_boot: int = BOOT) -> dict:
    """D = NLL(comparison) - NLL(real), paired.  POSITIVE = real is better."""
    rows, a, b = paired(rep_cmp, cell_cmp, rep_real, cell_real)
    keep = [i for i, r in enumerate(rows) if r not in drop_rows]
    rows = [rows[i] for i in keep]
    d = [a[i] - b[i] for i in keep]
    cl = None if clusters_of is None else [clusters_of[r] for r in rows]
    m, lo, hi = boot_ci(d, cl, n_boot)
    return {"mean": m, "lo": lo, "hi": hi, "n": len(d),
            "n_clusters": (len(set(cl)) if cl is not None else len(d)),
            "cmp_nll": (sum(a[i] for i in keep) / len(keep)) if keep else None,
            "real_nll": (sum(b[i] for i in keep) / len(keep)) if keep else None}


def cell_mean(rep: dict, cell: str, drop_rows: frozenset = frozenset()) -> Optional[float]:
    rows = rep.get("sample_rows") or []
    vals = (rep.get("per_sample") or {}).get(cell) or []
    v = [x for r, x in zip(rows, vals) if r not in drop_rows]
    return sum(v) / len(v) if v else None


# ─── vetoes ─────────────────────────────────────────────────────────────────

def veto_reasons(rep: Optional[dict], task: tuple, run: str) -> list:
    """Every reason this results file may not be read.  Empty = clean."""
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
    want = f"{run}/checkpoint_{STEP}"
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
    return out


# ─── the decision table ─────────────────────────────────────────────────────

def carry_reading(d_noread: dict, d_donor: dict, min_effect: float) -> str:
    """The main limbs' reading on a pack.  min_effect 0 = CI-only (prose)."""
    if d_noread["hi"] < 0:
        return "READ_HURTS"
    gain = d_noread["lo"] > 0 and d_noread["mean"] >= min_effect
    content = d_donor["lo"] > 0 and d_donor["mean"] >= min_effect
    if gain and content:
        return "Z_ADDS_CONTENT"
    if gain:
        return "CAPACITY_ONLY"
    return "NO_GAIN"


def edrop_reading(x_noread: dict, z_alone_donor: dict) -> str:
    """The E-dropout pair at E OFF, where Z is the only carry."""
    gain = x_noread["lo"] > 0 and x_noread["mean"] >= MIN_EFFECT_CARRY
    if not gain:
        return "Z_CANNOT_LEARN" if x_noread["hi"] >= 0 else "READ_HURTS"
    if z_alone_donor["lo"] > 0:
        return "Z_CAN_LEARN"
    return "AUDIT_gain_without_content"


def overall(main_carry: Optional[str], edrop: Optional[str]) -> str:
    """The one-line answer to 'does Z provide learnable information in J1?'

    The E-dropout pair decides when it was run and read cleanly: it is the only
    cell where Z is the sole carry, so it alone separates "Z cannot learn" from
    "Z is redundant with E".  The main limbs decide only a positive.
    """
    if edrop == "Z_CAN_LEARN":
        if main_carry == "Z_ADDS_CONTENT":
            return "YES: Z learns content, and adds it beyond E"
        return ("YES_BUT_REDUNDANT: Z learns content when E is absent; beyond "
                "E it adds nothing measurable at this budget")
    if edrop in ("Z_CANNOT_LEARN", "READ_HURTS"):
        return ("NO: the jointly trained read/write did not learn to carry "
                "content even with E removed")
    if edrop == "AUDIT_gain_without_content":
        return ("AUDIT: Z helps at E-off but another document's Z does as well "
                "-- impossible on this task without an instrument fault")
    if main_carry == "Z_ADDS_CONTENT":
        return "YES: Z adds content beyond E (E-dropout pair not read)"
    if main_carry == "TASK_NOT_LEARNED":
        return ("NOT_READ: the carry task's local answers were not learned, so "
                "no carry number means anything")
    if main_carry is None:
        return "NOT_READ: vetoed or missing -- see the vetoes"
    return ("UNDECIDED: no content gain beyond E, and without a clean E-dropout "
            "read 'redundant' and 'cannot learn' look the same")


# ─── I/O ────────────────────────────────────────────────────────────────────

def load_results(path: str) -> Optional[dict]:
    f = os.path.join(path, "results.json")
    if not os.path.exists(f):
        return None
    with open(f, encoding="utf-8") as fh:
        return json.load(fh)


def gate_trajectory(diag_path: str) -> Optional[dict]:
    """First and last z_read_gate from a run's cortex_diag.jsonl, or None."""
    if not os.path.exists(diag_path):
        return None
    pts = []
    with open(diag_path, encoding="utf-8") as fh:
        for line in fh:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            g = r.get("z_read_gate")
            if g is not None:
                pts.append((int(r.get("step", 0)), float(g)))
    if not pts:
        return None
    pts.sort()
    return {"first_step": pts[0][0], "first": pts[0][1],
            "last_step": pts[-1][0], "last": pts[-1][1],
            "collapsed": pts[-1][1] < GATE_COLLAPSE}


def read_trajectory(diag_path: str) -> Optional[dict]:
    """First and last z_read_ratio (||injected delta|| / ||x|| at the read
    site) from a run's cortex_diag.jsonl, or None when the run did not record
    it (every J1 run: the field was added for J3).

    With the gate's LR slowed (J3's gate_lr_mult), the gate cannot close the
    read quickly, but out_proj / v_proj still can -- so this is the read's
    effective strength and the number a collapse would show up in.  `first` /
    `last` are MEANS over READ_WINDOW rows at each end (see READ_COLLAPSE_FRAC);
    `collapsed` feeds the same AUDIT rule as the gate (audit_reasons).
    """
    if not os.path.exists(diag_path):
        return None
    pts = []
    with open(diag_path, encoding="utf-8") as fh:
        for line in fh:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            v = r.get("z_read_ratio")
            if v is not None:
                pts.append((int(r.get("step", 0)), float(v)))
    if not pts:
        return None
    pts.sort()
    w = max(1, min(READ_WINDOW, len(pts) // 2 or 1))
    head, tail = pts[:w], pts[-w:]
    first = sum(v for _, v in head) / len(head)
    last = sum(v for _, v in tail) / len(tail)
    lof = (last / first) if first else None
    return {"first_step": head[0][0], "first": first,
            "last_step": tail[-1][0], "last": last, "window": w,
            "last_over_first": lof,
            "collapsed": lof is not None and lof < READ_COLLAPSE_FRAC}


def embed_trajectory(diag_path: str) -> Optional[dict]:
    """J4's read strength: `z_embed_ratio` from a run's cortex_diag.jsonl, or
    None on a run that never recorded it (every J1 and J3 run).

    J4 HAS NO GATE AND NO INJECTED DELTA.  Its read is 64 extra columns of
    `input_embeds`, read by the base model's own attention, so the only way the
    model can turn the read off is to shrink the projection until those columns
    stop being scored.  `z_embed_ratio` is the spliced Z rows' norm over the E
    rows' norm at the same splice, which is ~1.0 at init by construction (rows
    rescaled to E's measured norm, identity projection) -- so unlike a gate
    there is no designed starting value to compare against and the TRAJECTORY
    is the whole reading.

    Same shape as `read_trajectory` on purpose, including READ_WINDOW and
    READ_COLLAPSE_FRAC: this is the same rule ("the read turned itself off")
    measured in the only place J4 leaves it visible, and inventing a second
    fraction for it would be a threshold chosen after the design.
    """
    if not os.path.exists(diag_path):
        return None
    pts = []
    with open(diag_path, encoding="utf-8") as fh:
        for line in fh:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            v = r.get("z_embed_ratio")
            if v is not None:
                pts.append((int(r.get("step", 0)), float(v)))
    if not pts:
        return None
    pts.sort()
    w = max(1, min(READ_WINDOW, len(pts) // 2 or 1))
    head, tail = pts[:w], pts[-w:]
    first = sum(v for _, v in head) / len(head)
    last = sum(v for _, v in tail) / len(tail)
    lof = (last / first) if first else None
    return {"first_step": head[0][0], "first": first,
            "last_step": tail[-1][0], "last": last, "window": w,
            "last_over_first": lof,
            "collapsed": lof is not None and lof < READ_COLLAPSE_FRAC}


def e_health_audit(main_v: dict, edrop_v: Optional[dict]) -> list:
    """J4 (j4_prereg.md S5): did training a Z carry on E's OWN columns cost E?

    Two readings, and the second is the one that is actually within-model:

      * the noread limb's carry NLL against E_CARRY_MAX.  Between-model (the
        comparison is to J1's and J3's runs), so the threshold is the observed
        drift band and not a tight one.
      * on the E-dropout limbs, `E_margin` -- the real limb's own carry NLL with
        E blanked minus with E on, Z off in both.  That IS within-model, it is
        in distribution (the limb trained with E dropped), and it comes free
        from cells the read-out already runs.  E_margin at or below zero on a
        limb whose E is supposed to carry the task is the unambiguous version of
        this finding.

    Audit lines, never vetoes -- see E_CARRY_MAX.
    """
    out = []
    head = main_v.get("noread_carry_nll")
    if head is not None and head > E_CARRY_MAX:
        out.append(
            f"AUDIT (E-health): the noread limb scores carry answers at "
            f"{head:.4f}, above E_CARRY_MAX {E_CARRY_MAX} -- J1/J3 sat at "
            f"{E_CARRY_BAND[0]}-{E_CARRY_BAND[1]}.  Under 'endpoint' the Z "
            f"write shares E's summary columns, so this is the shape a "
            f"regression in E would take.  J4 may be trading E for Z")
    m = (edrop_v or {}).get("E_margin")
    if m is not None and not (m["mean"] > 0 and m["lo"] > 0):
        out.append(
            f"AUDIT (E-health): within the real E-dropout limb, E buys "
            f"{m['mean']:+.4f} [{m['lo']:+.4f}, {m['hi']:+.4f}] on carry "
            f"answers (E off minus E on, Z off in both).  Not CI-backed above "
            f"zero: E is not carrying the task in the limb whose E-off cell "
            f"decides the answer")
    return out


def within_model_audit(main_v: dict) -> list:
    """J3 (j3_prereg.md S4.5).  Z_ADDS_CONTENT is a BETWEEN-limb reading, and
    J1 showed the limbs' E-carry skill can drift 0.26 nats apart along three
    same-seed trajectories while the read itself does nothing (the real limb's
    own Z removed: +0.0001).  So a main-limb Z_ADDS_CONTENT must also show up
    WITHIN the real limb -- its own Z removed (Z_main off) costs at least
    MIN_EFFECT_CARRY with CI_lo > 0 -- or the gain is the trajectory, not the
    read.  The E-dropout pair needs no such rule: Z_CAN_LEARN already requires
    its within-model donor contrast."""
    if main_v.get("carry_reading") != "Z_ADDS_CONTENT":
        return []
    w = (main_v.get("carry") or {}).get("real_Z_main_off")
    if w and w["mean"] >= MIN_EFFECT_CARRY and w["lo"] > 0:
        return []
    got = "n/a" if not w else f"{w['mean']:+.4f} [{w['lo']:+.4f}, {w['hi']:+.4f}]"
    return [f"AUDIT: main-limb Z_ADDS_CONTENT without the within-model contrast "
            f"(real limb, own Z removed: {got}; needs >= {MIN_EFFECT_CARRY} with "
            f"CI_lo > 0) -- the between-limb gain is not coming through the read"]


def answer_with_audit(answer: str, audit: list) -> str:
    """j1_prereg.md S5 and j3_prereg.md S5: a positive reading that trips an
    audit IS an AUDIT, so the ANSWER line says so.  (Until 2026-09-23 the audit
    lines printed beside an unchanged ANSWER; J1 tripped none, so its recorded
    verdict is the same either way.)"""
    if not audit:
        return answer
    return f"AUDIT (the table alone would read: {answer})"


def audit_reasons(deciding: list, gates: dict, reads: dict,
                  embeds: Optional[dict] = None) -> list:
    """A positive reading beside a read that is OFF on the same limb is a
    contradiction: a read that is off cannot be the one doing the work.  Off =
    a collapsed gate (GATE_COLLAPSE, J1's rule), a collapsed effective read
    strength (READ_COLLAPSE_FRAC, J3's -- the gate can be held open while
    out_proj closes the read instead), or, under J4, a collapsed embed ratio
    (the same fraction: J4 has neither a gate nor an injected delta, so the
    only way its read turns off is the projection shrinking)."""
    out = []
    for run in deciding:
        g = gates.get(run)
        if g is not None and g["collapsed"]:
            out.append(f"AUDIT: {run} reads positive beside a COLLAPSED read "
                       f"gate ({g['last']:.4f} < {GATE_COLLAPSE})")
        rt = reads.get(run)
        if rt is not None and rt.get("collapsed"):
            out.append(f"AUDIT: {run} reads positive beside a COLLAPSED read "
                       f"strength (|delta|/|x| x{rt['last_over_first']:.2f} "
                       f"< {READ_COLLAPSE_FRAC})")
        et = (embeds or {}).get(run)
        if et is not None and et.get("collapsed"):
            out.append(f"AUDIT: {run} reads positive beside a COLLAPSED embed "
                       f"ratio (||Z col||/||E col|| x"
                       f"{et['last_over_first']:.2f} < {READ_COLLAPSE_FRAC})")
    return out


def pg19_books(pack_path: str) -> tuple:
    """(row -> book id, ragged rows) for the strided PG-19 pack.

    eval_carry_2x2 ignores attention_mask, so on the 50 ragged book-end rows it
    scores the PADDING too.  Those rows are dropped from every PG-19 number
    here rather than letting padding into a J1 comparison.  Books come from the
    same document-end logic evals/diag_z_content.py uses (Step 0).
    """
    sys.path.insert(0, os.path.join(REPO, "evals"))
    from diag_z_content import book_ids, is_book_end, pack_eos
    from datasets import load_from_disk
    ds = load_from_disk(pack_path)
    rows = [ds[i] for i in range(len(ds))]
    eos, _ = pack_eos(rows, None)
    ends, ragged = [], set()
    for i, r in enumerate(rows):
        m = r.get("attention_mask")
        ends.append(is_book_end(r["input_ids"], m, eos))
        if m is not None and min(m) == 0:
            ragged.add(i)
    books, note = book_ids(ends)
    if note.startswith("FALLBACK"):
        raise SystemExit(f"PG-19 books: {note}  Refusing a clustered CI on "
                         "made-up clusters.")
    return dict(enumerate(books)), frozenset(ragged)


# ─── main ───────────────────────────────────────────────────────────────────

def score_main(root: str, encoding: str, books: Optional[tuple],
               n_boot: int, experiment: str = "j1") -> dict:
    reps, vetoes = {}, []
    for t in MAIN_TASKS:
        run = run_name(t[1], encoding, "", experiment)
        reps[t[0]] = load_results(task_dir(root, t, encoding, "", experiment))
        vetoes += veto_reasons(reps[t[0]], t, run)
    out: dict = {"vetoes": vetoes}
    if vetoes:
        out["reading"] = "VETOED"
        return out
    r = reps
    for key in ("noread_carry_off", "noread_pg19_off"):
        try:
            nz = effect(r[key], "E1Z0", r[key], "E1Z1", n_boot=10)["mean"]
        except KeyError:
            nz = None
        if nz is None or not abs(nz) <= NOREAD_ZMAIN_TOL:
            out["vetoes"].append(f"{key} Z_main {nz}: the no-read limb reads "
                                 "something, or its cells did not both run")
    if out["vetoes"]:
        out["reading"] = "VETOED"
        return out

    local = {k: cell_mean(r[f"{k}_local_off"], "E1Z1") for k in ("real", "noread")}
    out["local_nll"] = local
    out["task_learned"] = all(v is not None and v <= TASK_LEARNED_LOCAL_MAX
                              for v in local.values())
    head = cell_mean(r["noread_carry_off"], "E1Z1")
    out["noread_carry_nll"] = head
    # J4's E-health reading (j4_prereg.md S5).  The SAME number HEADROOM_MIN
    # bounds from below, bounded from above: E saturating the task and E having
    # lost the task are the two ways this one cell stops being informative.
    out["e_health"] = {"noread_carry_nll": head, "band": list(E_CARRY_BAND),
                       "max": E_CARRY_MAX,
                       "degraded": head is not None and head > E_CARRY_MAX}
    d_noread = effect(r["noread_carry_off"], "E1Z1", r["real_carry_off"], "E1Z1",
                      n_boot=n_boot)
    d_donor = effect(r["donor_carry_donor"], "E1Z0", r["real_carry_off"], "E1Z1",
                     n_boot=n_boot)
    # Within the real limb, descriptive: its own read removed (off) or fed
    # another document's Z (donor).  Positive = its own Z is better.
    out["carry"] = {"D_noread": d_noread, "D_donor": d_donor,
                    "real_Z_main_off": effect(r["real_carry_off"], "E1Z0",
                                              r["real_carry_off"], "E1Z1",
                                              n_boot=n_boot),
                    "real_Z_main_donor": effect(r["real_carry_donor"], "E1Z0",
                                                r["real_carry_donor"], "E1Z1",
                                                n_boot=n_boot)}
    if not out["task_learned"]:
        out["carry_reading"] = "TASK_NOT_LEARNED"
    elif head is not None and head < HEADROOM_MIN:
        out["carry_reading"] = "CEILING_E_SATURATES"
    else:
        out["carry_reading"] = carry_reading(d_noread, d_donor, MIN_EFFECT_CARRY)

    if books is not None:
        clusters_of, ragged = books
        p_noread = effect(r["noread_pg19_off"], "E1Z1", r["real_pg19_off"], "E1Z1",
                          clusters_of, ragged, n_boot)
        p_donor = effect(r["donor_pg19_donor"], "E1Z0", r["real_pg19_off"], "E1Z1",
                         clusters_of, ragged, n_boot)
        out["pg19"] = {"D_noread": p_noread, "D_donor": p_donor,
                       "real_Z_main_off": effect(r["real_pg19_off"], "E1Z0",
                                                 r["real_pg19_off"], "E1Z1",
                                                 clusters_of, ragged, n_boot),
                       "real_Z_main_donor": effect(r["real_pg19_donor"], "E1Z0",
                                                   r["real_pg19_donor"], "E1Z1",
                                                   clusters_of, ragged, n_boot),
                       "ragged_rows_dropped": len(ragged)}
        out["pg19_reading"] = carry_reading(p_noread, p_donor, 0.0)
    out["reading"] = out["carry_reading"]
    return out


def score_edrop(root: str, encoding: str, edrop: str, n_boot: int,
                experiment: str = "j1") -> Optional[dict]:
    paths = [task_dir(root, t, encoding, edrop, experiment) for t in EDROP_TASKS]
    if not any(os.path.exists(os.path.join(p, "results.json")) for p in paths):
        return None
    reps, vetoes = {}, []
    for t, p in zip(EDROP_TASKS, paths):
        reps[t[0]] = load_results(p)
        vetoes += veto_reasons(reps[t[0]], t,
                               run_name(t[1], encoding, edrop, experiment))
    out: dict = {"vetoes": vetoes}
    if vetoes:
        out["reading"] = "VETOED"
        return out
    r = reps
    nr = r["noread_carry_off"]
    for label, off, on in (("Z_main", "E1Z0", "E1Z1"), ("Z_alone", "E0Z0", "E0Z1")):
        try:
            v = effect(nr, off, nr, on, n_boot=10)["mean"]
        except KeyError:
            v = None
        if v is None or not abs(v) <= NOREAD_ZMAIN_TOL:
            out["vetoes"].append(f"edrop noread {label} {v}: the no-read limb "
                                 "reads something, or its cells did not all run")
    if out["vetoes"]:
        out["reading"] = "VETOED"
        return out
    local = cell_mean(r["real_local_off"], "E1Z1")
    floor = cell_mean(r["noread_carry_off"], "E0Z1")
    out.update(local_nll=local, noread_eoff_carry_nll=floor)
    if local is None or local > TASK_LEARNED_LOCAL_MAX:
        out["reading"] = "TASK_NOT_LEARNED"
        return out
    if floor is None or floor < LN10 - LEAK_MARGIN:
        out["vetoes"].append(f"E-OFF IS NOT OFF: the no-read E-dropout limb scores "
                             f"carry answers at {floor} with E zeroed, below "
                             f"ln10 - {LEAK_MARGIN}")
        out["reading"] = "VETOED"
        return out
    x = effect(r["noread_carry_off"], "E0Z1", r["real_carry_off"], "E0Z1",
               n_boot=n_boot)
    # Within the real E-dropout limb at E OFF: another document's Z (donor)
    # or no read (off) in place of its own.  Positive = its own Z is better.
    za_donor = effect(r["real_carry_donor"], "E0Z0", r["real_carry_donor"],
                      "E0Z1", n_boot=n_boot)
    za_off = effect(r["real_carry_off"], "E0Z0", r["real_carry_off"], "E0Z1",
                    n_boot=n_boot)
    out.update(X_noread=x, real_Z_alone_donor=za_donor, real_Z_alone_off=za_off)
    # J4's WITHIN-MODEL E-health contrast (j4_prereg.md S5), free from cells
    # this read-out already runs: the real limb's carry NLL with E blanked minus
    # with E on, Z off in BOTH, so the only thing that changes is E.  Reported
    # for every experiment -- it costs nothing and it is the number that says
    # whether the E-off cell is being read on a model whose E still works.
    out["E_margin"] = effect(r["real_carry_off"], "E0Z0",
                             r["real_carry_off"], "E1Z0", n_boot=n_boot)
    out["reading"] = edrop_reading(x, za_donor)
    return out


def f4(x) -> str:
    return "n/a" if x is None else f"{x:.4f}"


def fmt(e: Optional[dict]) -> str:
    if not e:
        return "n/a"
    if "ci_lo" in e:
        return f"{e['mean_nats']:+.4f} [{e['ci_lo']:+.4f}, {e['ci_hi']:+.4f}] n={e['n']}"
    return (f"{e['mean']:+.4f} [{e['lo']:+.4f}, {e['hi']:+.4f}] n={e['n']}"
            + (f" / {e['n_clusters']} books" if e.get("n_clusters", e["n"]) != e["n"] else ""))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", default="j1", choices=EXPERIMENTS,
                    help="j1 (joint xattn read), j3 (in-loop scratchpad) or j4 "
                         "(Z as its own input_embeds columns); sets the "
                         "run-name prefix and the default --root")
    ap.add_argument("--root", default="",
                    help="default: eval_results/<experiment>_readout")
    ap.add_argument("--encoding", default="",
                    help="default: 'endpoint' for j4 (D3's vindicated write), "
                         "'tokens' for j1/j3 -- the encoding each launcher "
                         "builds its run names with")
    ap.add_argument("--edrop", default="0.25",
                    help="the E-dropout pair's tag; scored if its results "
                         "exist.  J4's Z-ONLY limb is this same machinery at "
                         "--edrop 0.95, run as a second pass")
    ap.add_argument("--runs_root", default="cortex-retrofit",
                    help="where the training runs keep cortex_diag.jsonl")
    ap.add_argument("--pg19", default=PG19_PACK,
                    help="the PG-19 pack, for book clusters and ragged rows; "
                         "'' skips the prose read-out")
    ap.add_argument("--boot", type=int, default=BOOT)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    exp = a.experiment
    a.root = a.root or f"eval_results/{exp}_readout"
    # Defaulted here rather than in add_argument so the default can depend on
    # the experiment: J4's write IS 'endpoint' (the graft refuses anything else
    # under latent_read='embeds'), and scoring it under 'tokens' would look for
    # run directories that cannot exist and report every task MISSING.
    a.encoding = a.encoding or ("endpoint" if exp == "j4" else "tokens")

    books = pg19_books(a.pg19) if a.pg19 else None
    main_v = score_main(a.root, a.encoding, books, a.boot, exp)
    edrop_v = score_edrop(a.root, a.encoding, a.edrop, a.boot, exp)
    gates, reads, embeds = {}, {}, {}
    for limb, tag in (("real", ""), ("donor", ""), ("real", a.edrop)):
        run = run_name(limb, a.encoding, tag, exp)
        diag = os.path.join(a.runs_root, run, "cortex_diag.jsonl")
        g = gate_trajectory(diag)
        if g is not None:
            gates[run] = g
        rt = read_trajectory(diag)
        if rt is not None:
            reads[run] = rt
        et = embed_trajectory(diag)
        if et is not None:
            embeds[run] = et

    answer = overall(main_v.get("carry_reading"),
                     edrop_v.get("reading") if edrop_v else None)
    # A positive reading beside a collapsed gate on the SAME limb is a
    # contradiction: a read that is off cannot be the one doing the work.
    deciding = []
    if main_v.get("carry_reading") == "Z_ADDS_CONTENT":
        deciding.append(run_name("real", a.encoding, "", exp))
    if edrop_v and edrop_v.get("reading") == "Z_CAN_LEARN":
        deciding.append(run_name("real", a.encoding, a.edrop, exp))
    audit = (audit_reasons(deciding, gates, reads, embeds)
             + within_model_audit(main_v)
             + e_health_audit(main_v, edrop_v))
    answer = answer_with_audit(answer, audit)

    print("=" * 78)
    print(f"{exp.upper()} VERDICT (pre-registered: evals/score_j1.py, "
          f"../{exp}_prereg.md)")
    print("  D = NLL(other limb) - NLL(real limb); POSITIVE = real is better")
    print("=" * 78)
    for v in main_v.get("vetoes", []):
        print(f"  VETO  {v}")
    if main_v.get("reading") != "VETOED":
        print(f"  carry task (main limbs, E on)   -> {main_v['carry_reading']}")
        print(f"    local-answer NLL  real {f4(main_v['local_nll']['real'])}  "
              f"noread {f4(main_v['local_nll']['noread'])}  "
              f"(learned <= {TASK_LEARNED_LOCAL_MAX:.4f})")
        print(f"    noread carry NLL  {f4(main_v['noread_carry_nll'])}  "
              f"(no information = {LN10:.4f}; E saturates below {HEADROOM_MIN}; "
              f"E-health band {E_CARRY_BAND[0]}-{E_CARRY_BAND[1]}, "
              f"max {E_CARRY_MAX})")
        c = main_v["carry"]
        print(f"    D_noread          {fmt(c['D_noread'])}")
        print(f"    D_donor           {fmt(c['D_donor'])}")
        print(f"    real Z_main off   {fmt(c['real_Z_main_off'])}")
        print(f"    real Z_main donor {fmt(c['real_Z_main_donor'])}")
        if "pg19" in main_v:
            p = main_v["pg19"]
            print(f"  PG-19 val (secondary, book-clustered, {p['ragged_rows_dropped']} "
                  f"ragged rows dropped) -> {main_v['pg19_reading']}")
            print(f"    D_noread          {fmt(p['D_noread'])}")
            print(f"    D_donor           {fmt(p['D_donor'])}")
            print(f"    real Z_main off   {fmt(p['real_Z_main_off'])}")
            print(f"    real Z_main donor {fmt(p['real_Z_main_donor'])}")
    if edrop_v is None:
        print("  E-dropout pair: no results (not run, or not scored yet)")
    else:
        for v in edrop_v.get("vetoes", []):
            print(f"  VETO  (edrop) {v}")
        print(f"  E-dropout pair, carry task, E OFF -> {edrop_v['reading']}")
        if "local_nll" in edrop_v:
            print(f"    local-answer NLL  {f4(edrop_v['local_nll'])}")
        if "X_noread" in edrop_v:
            print(f"    no-carry floor    {f4(edrop_v['noread_eoff_carry_nll'])} (ln10 {LN10:.4f})")
            print(f"    X_noread          {fmt(edrop_v['X_noread'])}")
            print(f"    real Z_alone donor {fmt(edrop_v['real_Z_alone_donor'])}")
            print(f"    real Z_alone off   {fmt(edrop_v['real_Z_alone_off'])}")
        if edrop_v.get("E_margin"):
            print(f"    E_margin (within)  {fmt(edrop_v['E_margin'])}  "
                  f"(E off minus E on, Z off in both; > 0 = E still carries)")
    for run, g in gates.items():
        print(f"  read gate {run}: {g['first']:.4f} @ {g['first_step']} -> "
              f"{g['last']:.4f} @ {g['last_step']}"
              + ("  COLLAPSED" if g["collapsed"] else ""))
    for run, rt in reads.items():
        lof = rt["last_over_first"]
        print(f"  read strength {run}: |delta|/|x| {rt['first']:.4f} @ "
              f"{rt['first_step']} -> {rt['last']:.4f} @ {rt['last_step']}"
              + (f"  (x{lof:.2f})" if lof is not None else "")
              + ("  COLLAPSED" if rt["collapsed"] else ""))
    for run, et in embeds.items():
        lof = et["last_over_first"]
        print(f"  embed ratio {run}: ||Z col||/||E col|| {et['first']:.4f} @ "
              f"{et['first_step']} -> {et['last']:.4f} @ {et['last_step']}"
              + (f"  (x{lof:.2f})" if lof is not None else "")
              + ("  COLLAPSED" if et["collapsed"] else ""))
    for x in audit:
        print(f"  {x}")
    print("-" * 78)
    print(f"  ANSWER: {answer}")
    # "edrop" KEEPS ITS MEANING (the pair's scored dict): J1's and J3's
    # recorded verdicts are on disk under that key, and renaming it to add a
    # scalar beside it would silently break any later reader of those.
    rec = {"experiment": exp, "encoding": a.encoding, "edrop_tag": a.edrop,
           "answer": answer, "audit": audit, "main": main_v,
           "edrop": edrop_v, "gates": gates, "read_strength": reads,
           "embed_ratio": embeds,
           "constants": {"TASK_LEARNED_LOCAL_MAX": TASK_LEARNED_LOCAL_MAX,
                         "MIN_EFFECT_CARRY": MIN_EFFECT_CARRY,
                         "HEADROOM_MIN": HEADROOM_MIN, "LEAK_MARGIN": LEAK_MARGIN,
                         "GATE_COLLAPSE": GATE_COLLAPSE,
                         "READ_COLLAPSE_FRAC": READ_COLLAPSE_FRAC,
                         "READ_WINDOW": READ_WINDOW, "E_CARRY_MAX": E_CARRY_MAX,
                         "E_CARRY_BAND": list(E_CARRY_BAND),
                         "PAIR_MIN_FRAC": PAIR_MIN_FRAC, "STEP": STEP}}
    out = a.out or os.path.join(a.root, "verdict.json")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(rec, fh, indent=2)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
