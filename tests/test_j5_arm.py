"""J5's arm: pace/j5_joint.sbatch, pace/j5_readout.sbatch, evals/score_j5.py.

Written WITH ../j5_prereg.md and before any J5 number exists.  The scorer is
the pre-registration in code, so every answer in its table is planted here from
synthetic per-sample NLLs and checked to come out as registered -- including
the ones nobody wants, because an instrument that cannot say NO is not an
instrument.

The launcher tests exist for a different reason: J5's three limbs are
distinguished by TWO config flags whose defaults are both `true`, so a limb
that lost its flag would train the `both` condition under another limb's name
and the read-out would score it without complaint.  Every check here is
something that has already gone wrong once in this project.

Run: python -m pytest tests/test_j5_arm.py -q
"""
from __future__ import annotations

import json
import os
import random
import re
import sys
import zlib

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "evals"))

import score_j5 as S  # noqa: E402
from eval_carry_2x2 import compute_effects  # noqa: E402

STEP = 4000
LN10 = S.LN10


def _read(path):
    with open(os.path.join(REPO, path), encoding="utf-8") as fh:
        return fh.read()


# ─── synthetic results files ────────────────────────────────────────────────

def _seed(key: str) -> int:
    """A STABLE per-task seed.  `hash(str)` is salted per process, which made
    J1's decision-table tests depend on which process ran them."""
    return zlib.crc32(key.encode()) % 97


def _noise(n, sd, seed):
    rng = random.Random(seed)
    return [rng.gauss(0.0, sd) for _ in range(n)]


def _rep(task, step, cells: dict, rows=None, **over):
    key, limb, pack, score, znull, n, _ = task
    rows = list(range(n)) if rows is None else rows
    rep = {
        "z_channel": True,
        "config": {"score": score, "z_null": znull, "data": S.PACKS[pack],
                   "checkpoint": (f"cortex-retrofit/{S.run_name(limb)}"
                                  f"/checkpoint_{step}"),
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


def _world(tmp, *, step=STEP, carry=None, local=0.2, floor=LN10,
           x_off=None, x_content=None, pg19=2.9, e_margin=1.5, **over):
    """Write every read-out file for a planted outcome.

    carry      limb -> the limb's carry NLL in its OWN operating cell
    x_off      limb -> how much WORSE the Z-skipped cell is (nats).  The
               load-bearing number; J4 planted +0.4716 here.
    x_content  limb -> how much WORSE the donor cell is.  THE deciding number;
               J4 planted +0.0001 here.
    floor      the zonly E0Z0 cell under --z_null off, i.e. neither channel
    e_margin   how much worse eonly is with E blanked
    """
    root = str(tmp)
    carry = {"zonly": 2.30, "eonly": 0.60, "both": 0.60, **(carry or {})}
    x_off = {"zonly": 0.0, "both": 0.0, **(x_off or {})}
    x_content = {"zonly": 0.0, "both": 0.0, **(x_content or {})}
    shared = _noise(400, 0.3, 1)                    # a row effect, so pairing bites
    for t in S.TASKS:
        key, limb, pack, score, znull, n, cellspec = t
        on, off = S.OPERATING_CELL[limb], S.Z_ABLATED_CELL[limb]
        base = [shared[i] + 0.01 * e
                for i, e in enumerate(_noise(n, 1, _seed(key)))]
        if score == "local":
            mu, gap = local, 0.0
        elif pack == "pg19":
            mu = pg19
            gap = (x_content if znull == "donor" else x_off).get(limb, 0.0)
        else:
            mu = carry[limb]
            gap = (x_content if znull == "donor" else x_off).get(limb, 0.0)
        cells = {c: None for c in cellspec.split(",")}
        for c in cells:
            if c == on:
                cells[c] = [mu + b for b in base]
            elif c == off:
                cells[c] = [mu + gap + b for b in base]
            else:
                cells[c] = [mu + b for b in base]
        if limb == "eonly":
            # latent_carry_read=false: E1Z1 and E1Z0 are the SAME model.
            cells["E1Z1"] = list(cells["E1Z0"])
            if "E0Z0" in cells:                     # E blanked: E's own margin
                cells["E0Z0"] = [mu + e_margin + b for b in base]
        if limb == "zonly" and "E0Z0" in cells and znull == "off" \
                and score == "carry":
            # THE FLOOR: neither channel read.  Planted explicitly so the leak
            # veto can be exercised, and so x_off is measured against it.
            cells["E0Z0"] = [floor + b for b in base]
        if limb == "both" and "E0Z0" in cells:
            cells["E0Z0"] = [mu + e_margin + b for b in base]
        d = S.task_dir(root, t, step)
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "results.json"), "w", encoding="utf-8") as fh:
            json.dump(_rep(t, step, cells, **over), fh)
    return root


BOOKS = ({i: i // 21 for i in range(400)}, frozenset({20, 41}))


def _score(root, step=STEP):
    return S.score_step(root, step, BOOKS, 200)


# ─── the registered numbers ─────────────────────────────────────────────────

class TestTheConstants:
    def test_chance_is_ln_ten(self):
        assert S.LN10 == pytest.approx(2.302585, abs=1e-5)

    def test_the_thresholds_are_the_j_arms_own(self):
        import score_j1 as J
        assert S.MIN_EFFECT_CARRY == J.MIN_EFFECT_CARRY == 0.05
        assert S.TASK_LEARNED_LOCAL_MAX == J.TASK_LEARNED_LOCAL_MAX
        assert S.E_CARRY_MAX == J.E_CARRY_MAX == 1.20

    def test_each_limb_has_one_operating_cell_and_one_ablation(self):
        assert set(S.OPERATING_CELL) == set(S.Z_ABLATED_CELL) == set(S.LIMBS)
        for limb in S.LIMBS:
            assert S.OPERATING_CELL[limb] != S.Z_ABLATED_CELL[limb]

    def test_the_operating_cells_are_the_trained_conditions(self):
        # zonly trained with E zeroed -> an E0 cell; eonly with Z zeroed -> Z0;
        # both with everything live -> E1Z1.  Getting this wrong reports an
        # out-of-distribution condition as the arm's result.
        assert S.OPERATING_CELL["zonly"].startswith("E0")
        assert S.OPERATING_CELL["eonly"].endswith("Z0")
        assert S.OPERATING_CELL["both"] == "E1Z1"


# ─── the decision table ─────────────────────────────────────────────────────

class TestTheDecisionTable:
    def test_a_real_channel_reads_as_carrying_content(self, tmp_path):
        v = _score(_world(tmp_path, carry={"zonly": 1.4},
                          x_off={"zonly": 0.6}, x_content={"zonly": 0.5}))
        assert v["vetoes"] == []
        assert v["limbs"]["zonly"]["reading"] == "CARRIES_CONTENT"
        assert v["answer"].startswith("Z_LEARNS")

    def test_j4s_own_failure_reads_as_capacity_not_channel(self, tmp_path):
        # The planted numbers ARE J4's: the columns cost 0.47 nats to blank and
        # 0.0001 nats to fill with another document's content.
        v = _score(_world(tmp_path, carry={"zonly": 1.4},
                          x_off={"zonly": 0.4716},
                          x_content={"zonly": 0.0001}))
        assert v["limbs"]["zonly"]["reading"] == "CAPACITY_NOT_CHANNEL"
        assert "Z_CAPACITY_NOT_CHANNEL" in v["answer"]

    def test_a_limb_at_chance_is_no_however_its_deltas_read(self, tmp_path):
        # J4's +0.8245 lesson: a large positive within-model D sitting on a
        # comparison cell WORSE than chance is not a channel.  Absolute first.
        v = _score(_world(tmp_path, carry={"zonly": 2.3155},
                          x_off={"zonly": 0.8245}, x_content={"zonly": 0.3}))
        assert v["limbs"]["zonly"]["reading"] == "AT_CHANCE"
        assert v["answer"].startswith("NO:")

    def test_a_channel_nothing_leans_on_reads_as_unused(self, tmp_path):
        # Read on `both`, where it is the reachable case: E clears chance on
        # its own, and blanking Z costs nothing.  On `zonly` the Z-skipped cell
        # IS the no-carry floor, so a limb that clears chance there necessarily
        # has a large X_off -- see the next test.
        v = _score(_world(tmp_path, carry={"both": 0.60},
                          x_off={"both": 0.0}, x_content={"both": 0.0}))
        assert v["limbs"]["both"]["reading"] == "CHANNEL_UNUSED"

    def test_on_zonly_the_ablated_cell_is_the_no_carry_floor(self, tmp_path):
        # Structural, and worth pinning: with E gone, "Z skipped" leaves the
        # model nothing at all, so X_off measures the limb's whole carry skill
        # and CANNOT be small while the limb clears chance.  That is exactly
        # why X_content, not X_off, is the deciding cell here.
        v = _score(_world(tmp_path, carry={"zonly": 1.4},
                          x_off={"zonly": 0.0}, x_content={"zonly": 0.0}))
        assert v["limbs"]["zonly"]["X_off"]["mean"] == pytest.approx(
            LN10 - 1.4, abs=0.05)
        assert v["limbs"]["zonly"]["reading"] == "CAPACITY_NOT_CHANNEL"
        assert v["answer"].startswith("Z_CAPACITY_NOT_CHANNEL")

    def test_a_significant_but_tiny_content_gain_is_not_a_gain(self, tmp_path):
        v = _score(_world(tmp_path, carry={"zonly": 1.4},
                          x_off={"zonly": 0.3}, x_content={"zonly": 0.01}))
        assert v["limbs"]["zonly"]["reading"] == "CAPACITY_NOT_CHANNEL"

    def test_content_alone_beside_e_is_the_narrower_pass(self, tmp_path):
        v = _score(_world(tmp_path, carry={"zonly": 1.4},
                          x_off={"zonly": 0.6}, x_content={"zonly": 0.5,
                                                           "both": 0.0}))
        assert v["limbs"]["both"]["reading"] != "CARRIES_CONTENT"
        assert v["answer"].startswith("Z_LEARNS_ALONE")


class TestTheControlGatesEverything:
    def test_an_eonly_that_did_not_learn_the_task_makes_everything_unreadable(
            self, tmp_path):
        # ../j5_prereg.md S1.1 / veto 8.  With E's read zeroed there is no E-on
        # cell inside the treatment limb, so without this control "Z did not
        # learn" and "nothing learns here in 4,000 updates" are one number.
        v = _score(_world(tmp_path, local=2.0, carry={"zonly": 1.4},
                          x_off={"zonly": 0.6}, x_content={"zonly": 0.5}))
        assert v["task_learned"]["eonly"] is False
        assert v["answer"].startswith("NOT_READ")

    def test_the_control_supplies_the_scale_every_z_number_is_quoted_against(
            self, tmp_path):
        v = _score(_world(tmp_path, e_margin=1.5))
        em = v["limbs"]["eonly"]["E_margin"]
        assert em["mean"] == pytest.approx(1.5, abs=0.05)
        assert em["lo"] > 0


class TestTheVetoes:
    def test_a_leaking_floor_is_vetoed(self, tmp_path):
        # Neither channel read, and the carry answers still score well below
        # chance: something reaches across chunks that is not the buffer.
        v = _score(_world(tmp_path, floor=1.0))
        assert v["reading"] == "VETOED"
        assert any("FLOOR LEAKS" in x for x in v["vetoes"])

    def test_an_eonly_that_reads_a_z_is_vetoed(self, tmp_path):
        root = _world(tmp_path)
        p = os.path.join(S.task_dir(root, S.TASKS[5], STEP), "results.json")
        rep = json.load(open(p, encoding="utf-8"))
        rep["per_sample"]["E1Z1"] = [v + 0.3 for v in rep["per_sample"]["E1Z1"]]
        json.dump(rep, open(p, "w", encoding="utf-8"))
        v = _score(root)
        assert v["reading"] == "VETOED"
        assert any("reads something" in x for x in v["vetoes"])

    def test_a_missing_task_vetoes_the_whole_step(self, tmp_path):
        root = _world(tmp_path)
        os.remove(os.path.join(S.task_dir(root, S.TASKS[0], STEP),
                               "results.json"))
        v = _score(root)
        assert v["reading"] == "VETOED"
        assert any("MISSING" in x for x in v["vetoes"])

    @pytest.mark.parametrize("over,needle", [
        ({"score": "all"}, "scored"),
        ({"z_null": "noise"}, "z_null"),
        ({"samples": 10}, "samples"),
        ({"chunk1_ok": False}, "chunk-1"),
        ({"z_channel": False}, "no Z channel"),
    ])
    def test_each_defect_is_named(self, over, needle):
        rep = _rep(S.TASKS[0], STEP, {"E0Z1": [1.0] * 300, "E0Z0": [1.0] * 300},
                   **over)
        assert any(needle in x for x in S.veto_reasons(rep, S.TASKS[0], STEP))

    def test_a_cell_that_did_not_run_is_named(self):
        rep = _rep(S.TASKS[0], STEP, {"E0Z1": [1.0] * 300})
        assert any("cells" in x for x in S.veto_reasons(rep, S.TASKS[0], STEP))

    def test_the_wrong_step_is_named(self):
        rep = _rep(S.TASKS[0], 2000, {"E0Z1": [1.0] * 300, "E0Z0": [1.0] * 300})
        assert any("checkpoint" in x for x in S.veto_reasons(rep, S.TASKS[0],
                                                             STEP))


class TestTheTrajectory:
    def test_a_moving_null_is_distinguishable_from_a_flat_one(self, tmp_path):
        early = _world(tmp_path / "a", step=2000, carry={"zonly": 2.30})
        late = _world(tmp_path / "a", step=4000, carry={"zonly": 2.10})
        t = S.trajectory(_score(early, 2000), _score(late, 4000))
        assert t["available"] is True
        assert t["zonly"]["d_carry_nll"] == pytest.approx(-0.20, abs=0.02)

    def test_it_labels_nothing_on_its_own(self, tmp_path):
        root = _world(tmp_path, carry={"zonly": 2.30})
        v = _score(root)
        # A limb at chance stays NO whatever direction it is moving in.
        assert v["answer"].startswith("NO:")

    def test_a_vetoed_step_makes_the_trajectory_unavailable(self, tmp_path):
        early = _world(tmp_path / "a", step=2000, floor=1.0)
        late = _world(tmp_path / "a", step=4000)
        t = S.trajectory(_score(early, 2000), _score(late, 4000))
        assert t["available"] is False


# ─── the launchers ──────────────────────────────────────────────────────────

class TestTheJointLauncher:
    SB = "pace/j5_joint.sbatch"

    def test_the_base_is_track_as_trained_recurrence_checkpoint(self):
        assert "MODEL=${MODEL:-ckpts/olmo8-cortex}" in _read(self.SB)

    def test_there_is_no_branch_path(self):
        # The prepared base IS the start.  A --branch_path would continue
        # another run's step counter, schedule and ramp.
        s = _read(self.SB)
        cmd = [l for l in s.splitlines() if not l.lstrip().startswith("#")]
        assert not any("--branch_path" in l for l in cmd)
        assert "PARENT_STEP=0" in s

    def test_the_recurrence_ramp_is_off_and_the_depth_is_pinned(self):
        # THE failure this arm would otherwise hit silently: from step 0 the
        # 0.25 x 305,176 ramp puts mean_recurrence at 1 for the whole run, so
        # there is no loop, hence no loop state, hence no Z.
        s = _read(self.SB)
        assert "--mean_recurrence_schedule.turn_on false" in s
        assert "--override_mean_recurrence $MEAN_REC" in s
        assert "--override_mean_backprop_depth $BACKPROP_DEPTH" in s
        assert "MEAN_REC=${MEAN_REC:-8}" in s

    def test_the_dead_ramp_field_agrees_with_the_live_depth(self):
        # train.py:608's frozen-write warning reads
        # mean_recurrence_schedule['max_mean_rec'] WITHOUT consulting turn_on, so
        # at its default of 32 it warned that the write band 2..9 was frozen on a
        # run whose effective recurrence was 8 (job 13617827).  A false alarm
        # about the exact failure mode this arm is about would sit in every
        # training log and a later reader would be right to believe it.
        s = _read(self.SB)
        assert "--mean_recurrence_schedule.max_mean_rec $MEAN_REC" in s

    def test_the_three_limbs_are_the_three_buffer_contents(self):
        s = _read(self.SB)
        assert "LIMBS=(zonly eonly both)" in s
        for limb, ecr, lcr in (("zonly", "false", "true"),
                               ("eonly", "true", "false"),
                               ("both", "true", "true")):
            m = re.search(rf"^\s*{limb}\)\s*READ_ARGS=.*$", s, re.M)
            assert m, f"no limb case for {limb}"
            assert f"--cortex.e_carry_read {ecr}" in m.group(0)
            assert f"--cortex.latent_carry_read {lcr}" in m.group(0)

    def test_every_limb_keeps_the_same_read_site_and_rescale(self):
        # The limbs must differ in the carry and in NOTHING else: same
        # encoding, same read site, same rescale target, one seed.
        s = _read(self.SB)
        assert "ENCODING=endpoint" in s
        assert "--cortex.latent_read embeds" in s
        assert "--cortex.latent_read_znorm_target $ZNORM_TARGET" in s
        assert s.count("--seed 74") == 1

    def test_the_rescale_target_has_no_default_outside_the_measuring_mode(self):
        # J4's single smoke failure was a ZNORM_TARGET inherited from another
        # branch.  On a new base there is no safe default.
        s = _read(self.SB)
        assert "elif [ -z \"$ZNORM_TARGET\" ]; then" in s
        assert re.search(r'ZNORM_TARGET=\$\{ZNORM_TARGET:-[\d.]+\}\s*# PLACEHOLDER', s)

    def test_the_budget_is_four_thousand_read_at_two_and_four(self):
        s = _read(self.SB)
        assert "CELL_STEPS=${CELL_STEPS:-4000}" in s
        assert "SAVE_INTERVAL=${SAVE_INTERVAL:-2000}" in s

    def test_the_tokenizer_preflight_refuses_the_launch(self):
        s = _read(self.SB)
        assert "tools/check_tokenizer_match.py" in s
        assert "REFUSING THE LAUNCH" in s

    def test_the_smoke_gate_checks_the_measured_proof_of_each_condition(self):
        # e_carry_read is a config flag; e_spliced_norm is the MEASURED half of
        # the same fact, and it is what catches a limb that did not run its own
        # condition.
        s = _read(self.SB)
        assert "z_e_spliced_norm" in s
        assert "z_e_carry_read" in s
        assert "VETO 1" in s and "VETO 2" in s

    def test_the_smoke_gate_catches_a_loop_that_is_not_running(self):
        assert "THE LOOP IS NOT RUNNING AT DEPTH" in _read(self.SB)


class TestTheReadoutLauncher:
    SB = "pace/j5_readout.sbatch"

    def test_the_task_table_matches_the_scorer_row_for_row(self):
        s = _read(self.SB)
        block = re.search(r"^TASKS=\((.*?)^\)", s, re.S | re.M)
        assert block, "no TASKS array"
        rows = [tuple(l.strip().strip('"').split())
                for l in block.group(1).strip().splitlines() if l.strip()]
        want = [(k, limb, pack, score, zn, str(n), cells)
                for k, limb, pack, score, zn, n, cells in S.TASKS]
        assert rows == want

    def test_the_array_covers_the_table(self):
        s = _read(self.SB)
        assert f"#SBATCH --array=0-{len(S.TASKS) - 1}" in s

    def test_only_the_eonly_limb_is_rebuilt_with_the_no_read_flag(self):
        # zonly's E0 cells reproduce its trained splice exactly; rebuilding it
        # with e_carry_read=false would collapse E1 onto E0 for nothing.
        s = _read(self.SB)
        assert s.count("--set latent_carry_read=false") == 1
        assert "--set e_carry_read" not in s

    def test_the_step_is_in_the_output_path(self):
        # Both read-outs must coexist: the trajectory is the arm's second
        # deliverable, and one overwriting the other loses it silently.
        assert "eval_results/j5_readout/step${STEP}/" in _read(self.SB)

    def test_it_refuses_without_a_rescale_target(self):
        assert 'if [ -z "$ZNORM_TARGET" ]; then' in _read(self.SB)

    def test_it_reads_the_limb_back_off_the_run(self):
        s = _read(self.SB)
        assert "z_e_carry_read" in s and "z_carry_read" in s
        assert "not the limb its name claims" in s


class TestTheVerdictLauncher:
    def test_it_scores_both_steps(self):
        s = _read("pace/j5_verdict.sbatch")
        assert "STEP=${STEP:-4000}" in s
        assert "ALSO_STEP=${ALSO_STEP:-2000}" in s
        assert "evals/score_j5.py" in s

    def test_it_skips_the_smoke_and_enorm_run_dirs(self):
        assert "*-smoke|*-enorm" in _read("pace/j5_verdict.sbatch")


class TestTheTokenizerPreflight:
    def test_it_checks_the_carry_tasks_own_alphabet(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "ctm", os.path.join(REPO, "tools/check_tokenizer_match.py"))
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        # The answer NLL is computed over exactly these, so these are the ids
        # that must not move.
        assert set("0123456789") <= set(m.CARRY_PIECES)
        assert "+" in m.CARRY_PIECES and "=" in m.CARRY_PIECES
        assert "A" in m.CARRY_PIECES and "P" in m.CARRY_PIECES
