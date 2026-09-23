"""J1's pre-registered read-out: evals/score_j1.py + pace/j1_readout.sbatch.

Written with the pre-registration, before any J1 number existed.  The scorer is
the pre-registration in code, so each decision it can print is planted here
from synthetic per-sample NLLs and checked to come out as registered -- the
instrument must be able to say every answer in its table, including NO.

Run: /c/Users/henry/miniconda3/envs/cortex-retro/python.exe -m pytest tests/test_j1_readout.py -q
"""
from __future__ import annotations

import json
import math
import os
import random
import re
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "evals"))

import score_j1 as S  # noqa: E402
from eval_carry_2x2 import compute_effects, select_cells, CELLS  # noqa: E402


def _read(path):
    with open(os.path.join(REPO, path), encoding="utf-8") as fh:
        return fh.read()


# ─── synthetic results files ────────────────────────────────────────────────

def _rep(task, run, cells: dict, rows=None, **over):
    key, limb, pack, score, znull, n, cellspec = task
    rows = list(range(n)) if rows is None else rows
    rep = {
        "z_channel": True,
        "config": {"score": score, "z_null": znull, "data": S.PACKS[pack],
                   "checkpoint": f"cortex-retrofit/{run}/checkpoint_{S.STEP}",
                   "samples": len(rows), "cells": sorted(cells)},
        "per_sample": {k: list(v) for k, v in cells.items()},
        "sample_rows": rows,
        "chunk1_ok": True, "chunk1_max_spread": 1e-7,
        "health": {"at_chance": False},
        "effects": compute_effects(cells, True, 200, 0),
    }
    for k, v in over.items():
        if k in rep["config"]:
            rep["config"][k] = v
        else:
            rep[k] = v
    return rep


def _noise(n, sd, seed):
    rng = random.Random(seed)
    return [rng.gauss(0.0, sd) for _ in range(n)]


def _world(tmp, carry, local=0.2, pg19=None, edrop=None, enc="tokens"):
    """Write every read-out file for a planted outcome.

    carry: dict limb -> mean carry-answer NLL in the limb's trained condition
    pg19:  dict limb -> mean prose NLL (default: all equal)
    edrop: dict with 'real_eoff', 'noread_eoff', 'real_eoff_donor' carry NLLs
    """
    root = str(tmp)
    pg19 = pg19 or {"real": 3.0, "donor": 3.0, "noread": 3.0}
    base = _noise(400, 0.3, 1)
    for t in S.MAIN_TASKS:
        key, limb, pack, score, znull, n, _ = t
        run = S.run_name(limb, enc)
        if score == "carry":
            mu = carry[limb]
        elif score == "local":
            mu = local
        else:
            mu = pg19[limb]
        own = [mu + base[i] + 0.01 * e for i, e in enumerate(_noise(n, 1, hash(key) % 97))]
        if limb == "noread":
            cells = {"E1Z1": own, "E1Z0": list(own)}          # no read: identical
        elif limb == "donor":
            # trained condition is E1Z0 under donor; E1Z1 = its own Z (untrained use)
            cells = {"E1Z0": own, "E1Z1": [v + 0.05 for v in own]}
        else:
            # real: E1Z1 own Z; E1Z0 = off or donor, a bit worse
            cells = {"E1Z1": own, "E1Z0": [v + 0.2 for v in own]}
        d = S.task_dir(root, t, enc, "")
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "results.json"), "w") as fh:
            json.dump(_rep(t, run, cells), fh)
    if edrop is not None:
        for t in S.EDROP_TASKS:
            key, limb, pack, score, znull, n, _ = t
            run = S.run_name(limb, enc, "0.25")
            b = _noise(n, 0.2, 7)
            if score == "local":
                c = {k: [local + x for x in b] for k in ("E1Z1", "E1Z0", "E0Z1", "E0Z0")}
            elif limb == "noread":
                v = [edrop["noread_eoff"] + x for x in b]
                c = {"E1Z1": [1.0 + x for x in b], "E1Z0": [1.0 + x for x in b],
                     "E0Z1": v, "E0Z0": list(v)}
            else:
                eoff_null = (edrop["real_eoff_donor"] if znull == "donor"
                             else edrop["noread_eoff"])
                c = {"E1Z1": [0.5 + x for x in b], "E1Z0": [0.6 + x for x in b],
                     "E0Z1": [edrop["real_eoff"] + x for x in b],
                     "E0Z0": [eoff_null + x for x in b]}
            d = S.task_dir(root, t, enc, "0.25")
            os.makedirs(d, exist_ok=True)
            with open(os.path.join(d, "results.json"), "w") as fh:
                json.dump(_rep(t, run, c), fh)
    return root


BOOKS = ({i: i // 21 for i in range(400)}, frozenset({20, 41}))


# ─── the pre-registered numbers ─────────────────────────────────────────────

class TestTheNumbers:
    def test_chance_is_one_digit(self):
        assert abs(S.LN10 - 2.302585) < 1e-6
        assert abs(S.TASK_LEARNED_LOCAL_MAX - S.LN10 / 2) < 1e-12

    def test_the_step_is_the_stop_save_of_the_launch(self):
        s = _read("pace/j1_joint.sbatch")
        assert f"PARENT_STEP=${{PARENT_STEP:-{S.PARENT_STEP}}}" in s
        assert f"CELL_STEPS=${{CELL_STEPS:-{S.CELL_STEPS}}}" in s
        assert f"STEP=${{STEP:-{S.STEP}}}" in _read("pace/j1_readout.sbatch")

    def test_thresholds_are_in_their_registered_order(self):
        assert 0 < S.MIN_EFFECT_CARRY < S.HEADROOM_MIN < S.TASK_LEARNED_LOCAL_MAX < S.LN10
        assert 0 < S.GATE_COLLAPSE < 0.1       # the gate starts at 0.1


# ─── statistics ─────────────────────────────────────────────────────────────

class TestTheBootstrap:
    def test_mean_is_the_plain_mean(self):
        m, lo, hi = S.boot_ci([1.0, 2.0, 3.0], n_boot=500)
        assert m == 2.0 and lo <= m <= hi

    def test_book_clusters_widen_a_correlated_ci(self):
        rng = random.Random(0)
        book_eff = [rng.gauss(0, 1) for _ in range(20)]
        d, cl = [], []
        for b in range(20):
            for _ in range(20):
                d.append(book_eff[b] + rng.gauss(0, 0.1))
                cl.append(b)
        _, lo_r, hi_r = S.boot_ci(d, None, 2000)
        _, lo_c, hi_c = S.boot_ci(d, cl, 2000)
        assert (hi_c - lo_c) > 2.5 * (hi_r - lo_r)

    def test_empty_is_nan_not_zero(self):
        assert all(math.isnan(x) for x in S.boot_ci([]))


class TestPairing:
    def test_rows_are_intersected_and_ordered(self):
        a = {"sample_rows": [3, 1, 2], "per_sample": {"E1Z1": [30.0, 10.0, 20.0]}}
        b = {"sample_rows": [2, 3, 9], "per_sample": {"E1Z1": [2.0, 3.0, 9.0]}}
        rows, va, vb = S.paired(a, "E1Z1", b, "E1Z1")
        assert rows == [2, 3] and va == [20.0, 30.0] and vb == [2.0, 3.0]

    def test_a_cell_that_did_not_run_raises(self):
        a = {"sample_rows": [0], "per_sample": {"E1Z1": [1.0]},
             "config": {"cells": ["E1Z1"]}}
        with pytest.raises(KeyError):
            S.paired(a, "E0Z1", a, "E1Z1")

    def test_sign_positive_means_real_is_better(self):
        worse = {"sample_rows": [0, 1], "per_sample": {"E1Z1": [2.0, 2.0]}}
        real = {"sample_rows": [0, 1], "per_sample": {"E1Z1": [1.0, 1.0]}}
        assert S.effect(worse, "E1Z1", real, "E1Z1", n_boot=50)["mean"] == 1.0

    def test_dropped_rows_never_enter(self):
        a = {"sample_rows": [0, 1], "per_sample": {"E1Z1": [9.0, 1.0]}}
        b = {"sample_rows": [0, 1], "per_sample": {"E1Z1": [0.0, 0.0]}}
        e = S.effect(a, "E1Z1", b, "E1Z1", drop_rows=frozenset({0}), n_boot=50)
        assert e["n"] == 1 and e["mean"] == 1.0


# ─── vetoes ─────────────────────────────────────────────────────────────────

class TestVetoes:
    T = S.MAIN_TASKS[0]
    RUN = S.run_name("real")

    def _clean(self, **over):
        return _rep(self.T, self.RUN, {"E1Z1": [1.0] * 300, "E1Z0": [1.1] * 300},
                    **over)

    def test_a_clean_file_passes(self):
        assert S.veto_reasons(self._clean(), self.T, self.RUN) == []

    @pytest.mark.parametrize("over, needle", [
        (dict(score="all"), "scored"),
        (dict(z_null="noise"), "z_null"),
        (dict(data="data/pg19_olmo_val_len4096"), "registered"),
        (dict(checkpoint="cortex-retrofit/j1-a3z-tokens-real/checkpoint_92000"), "checkpoint"),
        (dict(checkpoint="cortex-retrofit/j1-a3z-tokens-real-smoke/checkpoint_93552"), "checkpoint"),
        (dict(chunk1_ok=False), "chunk-1"),
        (dict(health={"at_chance": True}), "chance"),
        (dict(z_channel=False), "Z channel"),
        (dict(samples=200), "samples"),
    ])
    def test_each_defect_is_named(self, over, needle):
        reasons = S.veto_reasons(self._clean(**over), self.T, self.RUN)
        assert any(needle in r for r in reasons), reasons

    def test_a_missing_file_is_a_veto_not_a_zero(self):
        assert "MISSING" in S.veto_reasons(None, self.T, self.RUN)[0]


# ─── the decision table, planted end to end ─────────────────────────────────

class TestTheMainLimbs:
    def test_real_beats_both_is_content(self, tmp_path):
        root = _world(tmp_path, {"real": 1.0, "donor": 1.6, "noread": 1.8})
        v = S.score_main(root, "tokens", BOOKS, 300)
        assert v["carry_reading"] == "Z_ADDS_CONTENT", v

    def test_donor_as_good_as_real_is_capacity_only(self, tmp_path):
        root = _world(tmp_path, {"real": 1.0, "donor": 1.0, "noread": 1.8})
        v = S.score_main(root, "tokens", BOOKS, 300)
        assert v["carry_reading"] == "CAPACITY_ONLY"

    def test_all_equal_is_no_gain(self, tmp_path):
        root = _world(tmp_path, {"real": 1.5, "donor": 1.5, "noread": 1.5})
        assert S.score_main(root, "tokens", BOOKS, 300)["carry_reading"] == "NO_GAIN"

    def test_a_significant_but_tiny_gain_is_not_a_gain(self, tmp_path):
        root = _world(tmp_path, {"real": 1.49, "donor": 1.5, "noread": 1.5})
        assert S.score_main(root, "tokens", BOOKS, 300)["carry_reading"] == "NO_GAIN"

    def test_real_worse_than_noread_is_read_hurts(self, tmp_path):
        root = _world(tmp_path, {"real": 1.8, "donor": 1.8, "noread": 1.0})
        assert S.score_main(root, "tokens", BOOKS, 300)["carry_reading"] == "READ_HURTS"

    def test_e_saturating_the_task_is_a_ceiling(self, tmp_path):
        root = _world(tmp_path, {"real": 0.01, "donor": 0.02, "noread": 0.05})
        v = S.score_main(root, "tokens", BOOKS, 300)
        assert v["carry_reading"] == "CEILING_E_SATURATES"

    def test_an_unlearned_task_reads_nothing(self, tmp_path):
        root = _world(tmp_path, {"real": 1.0, "donor": 1.6, "noread": 1.8}, local=2.2)
        v = S.score_main(root, "tokens", BOOKS, 300)
        assert v["carry_reading"] == "TASK_NOT_LEARNED"

    def test_prose_is_read_book_clustered_without_ragged_rows(self, tmp_path):
        root = _world(tmp_path, {"real": 1.0, "donor": 1.6, "noread": 1.8},
                      pg19={"real": 2.90, "donor": 3.0, "noread": 3.0})
        v = S.score_main(root, "tokens", BOOKS, 300)
        assert v["pg19_reading"] == "Z_ADDS_CONTENT"
        assert v["pg19"]["D_noread"]["n"] == 398
        assert v["pg19"]["D_noread"]["n_clusters"] == 20

    def test_a_missing_task_vetoes_the_whole_main_reading(self, tmp_path):
        root = _world(tmp_path, {"real": 1.0, "donor": 1.6, "noread": 1.8})
        os.remove(os.path.join(S.task_dir(root, S.MAIN_TASKS[2], "tokens", ""),
                               "results.json"))
        v = S.score_main(root, "tokens", BOOKS, 300)
        assert v["reading"] == "VETOED" and any("MISSING" in x for x in v["vetoes"])

    def test_a_noread_limb_that_reads_is_vetoed(self, tmp_path):
        root = _world(tmp_path, {"real": 1.0, "donor": 1.6, "noread": 1.8})
        t = S.MAIN_TASKS[3]
        p = os.path.join(S.task_dir(root, t, "tokens", ""), "results.json")
        rep = json.load(open(p))
        rep["per_sample"]["E1Z0"] = [v + 0.1 for v in rep["per_sample"]["E1Z0"]]
        rep["effects"] = compute_effects(rep["per_sample"], True, 100, 0)
        json.dump(rep, open(p, "w"))
        v = S.score_main(root, "tokens", BOOKS, 300)
        assert v["reading"] == "VETOED" and "reads something" in v["vetoes"][0]


class TestTheEDropoutPair:
    MAIN = {"real": 1.5, "donor": 1.5, "noread": 1.5}

    def test_z_carrying_registers_at_e_off_is_can_learn(self, tmp_path):
        root = _world(tmp_path, self.MAIN, edrop={
            "real_eoff": 0.8, "noread_eoff": 2.3, "real_eoff_donor": 2.9})
        v = S.score_edrop(root, "tokens", "0.25", 300)
        assert v["reading"] == "Z_CAN_LEARN", v
        assert S.overall("NO_GAIN", v["reading"]).startswith("YES_BUT_REDUNDANT")

    def test_no_gain_at_e_off_is_cannot_learn(self, tmp_path):
        root = _world(tmp_path, self.MAIN, edrop={
            "real_eoff": 2.3, "noread_eoff": 2.3, "real_eoff_donor": 2.3})
        v = S.score_edrop(root, "tokens", "0.25", 300)
        assert v["reading"] == "Z_CANNOT_LEARN"
        assert S.overall("NO_GAIN", v["reading"]).startswith("NO:")

    def test_a_leaking_e_off_cell_is_vetoed(self, tmp_path):
        root = _world(tmp_path, self.MAIN, edrop={
            "real_eoff": 0.8, "noread_eoff": 1.5, "real_eoff_donor": 2.9})
        v = S.score_edrop(root, "tokens", "0.25", 300)
        assert v["reading"] == "VETOED" and "E-OFF IS NOT OFF" in v["vetoes"][0]

    def test_gain_without_content_is_an_audit(self, tmp_path):
        root = _world(tmp_path, self.MAIN, edrop={
            "real_eoff": 0.8, "noread_eoff": 2.3, "real_eoff_donor": 0.8})
        v = S.score_edrop(root, "tokens", "0.25", 300)
        assert v["reading"] == "AUDIT_gain_without_content"

    def test_absent_pair_is_none_and_main_alone_cannot_say_no(self, tmp_path):
        root = _world(tmp_path, self.MAIN)
        assert S.score_edrop(root, "tokens", "0.25", 300) is None
        assert S.overall("NO_GAIN", None).startswith("UNDECIDED")
        assert S.overall("Z_ADDS_CONTENT", None).startswith("YES")


class TestTheGate:
    def test_trajectory_and_collapse(self, tmp_path):
        p = tmp_path / "cortex_diag.jsonl"
        p.write_text("\n".join(json.dumps(r) for r in [
            {"step": 91575, "z_read_gate": 0.1},
            {"step": 93550, "z_read_gate": 0.02}, {"step": 91600}]))
        g = S.gate_trajectory(str(p))
        assert (g["first"], g["last"], g["collapsed"]) == (0.1, 0.02, True)

    def test_no_file_is_none(self, tmp_path):
        assert S.gate_trajectory(str(tmp_path / "nope.jsonl")) is None


# ─── the launcher is the scorer's task table ────────────────────────────────

def _bash_tasks(src: str, name: str) -> list:
    body = re.search(name + r"=\(\n(.*?)\n\)", src, re.S).group(1)
    return [tuple(l.strip().strip('"').split()) for l in body.splitlines() if l.strip()]


class TestTheLauncher:
    SRC = _read("pace/j1_readout.sbatch")

    @pytest.mark.parametrize("name, table", [("MAIN_TASKS", S.MAIN_TASKS),
                                             ("EDROP_TASKS", S.EDROP_TASKS)])
    def test_task_tables_match_row_for_row(self, name, table):
        got = _bash_tasks(self.SRC, name)
        want = [(k, l, p, s, z, str(n), c or "-") for k, l, p, s, z, n, c in table]
        assert got == want

    def test_the_array_covers_the_main_table(self):
        assert f"#SBATCH --array=0-{len(S.MAIN_TASKS) - 1}" in self.SRC

    def test_it_builds_what_j1_trained(self):
        j = _read("pace/j1_joint.sbatch")
        for flag in ("latent_encoding", "latent_tok_pool", "latent_read_znorm",
                     "latent_read_znorm_target", "latent_read_heads",
                     "latent_write_only", "latent_read_scramble",
                     "latent_s0_read", "gate_slots", "accum_vecs"):
            assert flag in j and f"--set {flag}=" in self.SRC, flag
        assert "--allow_scrambled" in self.SRC
        assert 'READ_HEADS=${READ_HEADS:-8}' in j and 'READ_HEADS=${READ_HEADS:-8}' in self.SRC
        assert 'ZNORM_TARGET=${ZNORM_TARGET:-3.0}' in j and 'ZNORM_TARGET=${ZNORM_TARGET:-3.0}' in self.SRC
        assert 'TOK_POOL=${TOK_POOL:-4}' in j and 'TOK_POOL=${TOK_POOL:-4}' in self.SRC

    def test_the_packs_are_the_registered_ones(self):
        assert f"CARRY_DATA=${{CARRY_DATA:-{S.CARRY_PACK}}}" in self.SRC
        assert f"PG19_DATA=${{PG19_DATA:-{S.PG19_PACK}}}" in self.SRC

    def test_the_run_names_are_j1s(self):
        assert "RUN=j1-a3z-${ENCODING}${TAG}-${LIMB}" in self.SRC
        assert 'RUN_NAME=j1-a3z-${ENCODING}${TAG}-${LIMB}${RUN_SUFFIX:-}' in _read("pace/j1_joint.sbatch")
        assert S.run_name("real", "tokens", "0.25") == "j1-a3z-tokens-edrop0.25-real"

    def test_the_verdict_waits_for_everything_and_names_gaps(self):
        v = _read("pace/j1_verdict.sbatch")
        assert "afterany" in v and "evals/score_j1.py" in v
        assert "--gres" not in v                 # CPU only: no checkpoint opened


class TestTheCellsFlag:
    def test_subset_and_order(self):
        assert [c[0] for c in select_cells(CELLS, "E1Z0,E1Z1")] == ["E1Z1", "E1Z0"]

    def test_empty_is_everything(self):
        assert select_cells(CELLS, "") == CELLS

    def test_unknown_and_single_cells_are_refused(self):
        with pytest.raises(SystemExit):
            select_cells(CELLS, "E1Z1,E9Z9")
        with pytest.raises(SystemExit):
            select_cells(CELLS, "E1Z1")

    def test_the_record_says_which_cells_and_checkpoint(self):
        src = _read("evals/eval_carry_2x2.py")
        assert '"checkpoint": args.checkpoint' in src
        assert '"cells": [c[0] for c in cells]' in src
