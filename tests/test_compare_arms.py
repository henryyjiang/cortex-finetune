"""
The P1 probe comparison.

The tool's job is to make a few hundred steps say the most that a few hundred
steps honestly can, and no more.  So the tests here are about the places it
could quietly say MORE than it knows: an unpaired loss delta, a resumed run's
duplicated steps counted twice, an eval JSON of an unrecognised shape summarised
from the wrong field, a missing diagnostic reported as a zero.

Run: /c/Users/henry/miniconda3/envs/cortex-retro/python.exe -m pytest tests/test_compare_arms.py -q
"""
from __future__ import annotations

import io
import json
import os
import sys

import pytest
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "evals"))

from tools.compare_arms import (  # noqa: E402
    build, compare_losses, health_summary, paired_ci, print_comparison,
    read_diag, read_eval,
)


def _write_diag(tmp_path, label, rows):
    d = tmp_path / label
    d.mkdir(parents=True, exist_ok=True)
    with open(d / "cortex_diag.jsonl", "w", encoding="ascii") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    return str(d)


def _rows(n=10, base=3.0, slope=-0.001, start=100, every=10, **extra):
    out = []
    for i in range(n):
        r = {"step": start + i * every, "loss": base + slope * i,
             "grad_norm": 0.5, "rows": 64, "e_eff_rank_pr": 4.0,
             "e_centred_cosine": 0.9, "e_row_norm": 170.0,
             "fg_at_bias": 0.731, "gate_left_init": 0, "latent_carry": 0}
        r.update({k: (v[i] if isinstance(v, list) else v)
                  for k, v in extra.items()})
        out.append(r)
    return out


class TestReadingTheDiagnostic:

    def test_a_resumed_run_does_not_count_a_step_twice(self):
        """The jsonl is APPENDED on resume, so a crash-and-restart repeats
        steps.  Counting both would double-weight them in every paired
        comparison -- and the later row is the one the run continued from."""
        import tempfile, pathlib
        tmp = pathlib.Path(tempfile.mkdtemp())
        d = _write_diag(tmp, "a", _rows(3) + _rows(3, base=9.0))
        rows = read_diag(d)
        assert len(rows) == 3
        assert all(r["loss"] > 8.0 for r in rows)

    def test_a_torn_final_line_is_skipped_not_fatal(self):
        """A job killed mid-write leaves half a line.  Losing the diagnostic of
        a run that already cost GPU-hours because of it would be absurd."""
        import tempfile, pathlib
        tmp = pathlib.Path(tempfile.mkdtemp())
        d = _write_diag(tmp, "a", _rows(3))
        with open(os.path.join(d, "cortex_diag.jsonl"), "a",
                  encoding="ascii") as fh:
            fh.write('{"step": 999, "loss":')
        assert len(read_diag(d)) == 3

    def test_a_missing_file_is_empty_and_not_an_error(self):
        import tempfile
        assert read_diag(tempfile.mkdtemp()) == []


class TestHealthSummary:

    def test_it_reports_the_step_a_flag_first_went_true(self):
        """'the gate trained' and 'the gate took 300 steps to start' are
        different facts, and only the trajectory holds the second."""
        rows = _rows(6, gate_left_init=[0, 0, 0, 1, 1, 1])
        h = health_summary(rows)
        assert h["gate_left_init"] is True
        assert h["gate_left_init_step"] == 130

    def test_a_gate_that_never_moved_is_reported_as_false(self):
        h = health_summary(_rows(6))
        assert h["gate_left_init"] is False
        assert h["gate_left_init_step"] is None

    def test_a_check_the_arm_cannot_answer_is_omitted_not_failed(self):
        """An E-only arm has no Z gate and an accum arm has no gate at all.
        Reporting either as NOT LIVE would hang a failure banner on an arm
        behaving exactly as designed, which is how a real failure gets ignored.
        """
        h = health_summary(_rows(6))                 # E-only rows
        assert "gate_left_init" in h                 # it has a gate
        assert "gate_z_left_init" not in h           # it has no Z gate

    def test_non_finite_grad_norms_are_counted(self):
        """A non-finite total_norm means train.py SKIPPED that update: the run
        was not training through those steps, which no loss curve shows."""
        rows = _rows(4)
        rows[2]["grad_norm"] = float("inf")
        assert health_summary(rows)["grad_norm_nonfinite"] == 1

    def test_the_run_average_read_fraction_is_the_last_row(self):
        """latent_read_grad_frac is cumulative over the run, so the final row
        IS the average -- the number that goes in the Z pre-registration."""
        rows = _rows(4, z_read_grad_frac=[0.9, 0.7, 0.6, 0.52], latent_carry=1)
        assert health_summary(rows)["z_read_grad_frac_run_avg"] == 0.52

    def test_an_empty_trajectory_summarises_to_nothing(self):
        assert health_summary([]) == {}


class TestThePairedLossComparison:

    def test_the_delta_is_paired_on_shared_steps(self):
        arms = {"a1": {"diag": _rows(10, base=3.0)},
                "a3": {"diag": _rows(10, base=3.0)}}
        for r in arms["a3"]["diag"]:
            r["loss"] -= 0.02
        c = compare_losses(arms, "a1", last=5, n_boot=500)
        assert c["arms"]["a3"]["shared_steps"] == 10
        assert c["arms"]["a3"]["paired_delta_vs_ref"] == pytest.approx(-0.02)
        assert c["arms"]["a3"]["separated_from_zero"] is True

    def test_arms_with_no_shared_steps_get_no_delta(self):
        """Unpaired, the per-batch difficulty swamps the architecture effect
        entirely at this horizon -- a number here would be worse than none."""
        arms = {"a1": {"diag": _rows(5, start=100)},
                "a3": {"diag": _rows(5, start=9000)}}
        c = compare_losses(arms, "a1", last=5, n_boot=200)
        assert "paired_delta_vs_ref" not in c["arms"]["a3"]
        assert "no shared steps" in c["arms"]["a3"]["why_no_delta"]

    def test_a_straddling_ci_is_reported_as_not_separated(self):
        """The EXPECTED outcome at a few hundred steps, and it must not read as
        a negative finding."""
        torch.manual_seed(0)
        noise = (torch.randn(20) * 0.3).tolist()
        arms = {"a1": {"diag": _rows(20)},
                "a3": {"diag": _rows(20)}}
        for r, n in zip(arms["a3"]["diag"], noise):
            r["loss"] += n
        c = compare_losses(arms, "a1", last=20, n_boot=2000)
        assert c["arms"]["a3"]["separated_from_zero"] is False

    def test_the_dense_csv_wins_over_the_sampled_diagnostic(self):
        arms = {"a1": {"diag": _rows(5), "csv_losses": {i: 1.0 for i in range(50)}}}
        c = compare_losses(arms, "a1", last=10, n_boot=100)
        assert c["arms"]["a1"]["steps"] == 50

    def test_paired_ci_matches_the_influence_horizons_definition(self):
        """Same statistic, pinned rather than imported: eval_influence_horizon
        pulls the model-loading stack and this tool must run on a login node."""
        from evals.eval_influence_horizon import paired_ci as ref_ci
        d = [0.1, -0.2, 0.05, 0.3, -0.01]
        assert paired_ci(d, 500, 0) == ref_ci(d, 500, 0)


class TestTheInfluenceHorizonsPerChannelDamage:
    """The instrument that REPLACES the x1.31 compounding metric for a
    fixed-width buffer.  Until the Z channel existed it damaged the whole write,
    and a `zeros_like` on a 2D-wide write nulls Z as well -- the same defect
    eval_carry_2x2's `null_e` was fixed for, which reported an E0Z1 cell as E0Z0
    under the wrong label."""

    def test_damaging_e_leaves_z_bit_identical(self):
        from evals.eval_influence_horizon import damage_write
        D = 8
        w = torch.randn(1, 4, 2 * D)
        donor = torch.randn(1, 4, 2 * D)
        out = damage_write(w, donor, "donor", "e", D)
        assert torch.equal(out[..., :D], donor[..., :D])
        assert torch.equal(out[..., D:], w[..., D:])

    def test_damaging_z_leaves_e_bit_identical(self):
        from evals.eval_influence_horizon import damage_write
        D = 8
        w = torch.randn(1, 4, 2 * D)
        out = damage_write(w, None, "zero", "z", D)
        assert torch.equal(out[..., :D], w[..., :D])
        assert float(out[..., D:].abs().sum()) == 0.0

    def test_on_an_e_only_carry_the_choice_is_inert(self):
        from evals.eval_influence_horizon import damage_write
        D = 8
        w = torch.randn(1, 4, D)
        donor = torch.randn(1, 4, D)
        for ch in ("both", "e", "z"):
            assert torch.equal(damage_write(w, donor, "donor", ch, D), donor)


class TestReadingEvalJsons:

    def _write(self, tmp, name, doc):
        p = os.path.join(tmp, name)
        with open(p, "w", encoding="ascii") as fh:
            json.dump(doc, fh)
        return p

    def test_an_unrecognised_shape_is_named_and_not_summarised(self):
        """Summarising a file whose shape changed would put a number from the
        wrong field into a comparison table with no way to notice."""
        import tempfile
        tmp = tempfile.mkdtemp()
        ev = read_eval(self._write(tmp, "x.json", {"something": 1, "else": 2}))
        assert ev["kind"] == "unrecognised"
        assert ev["top_level_keys"] == ["else", "something"]

    def test_it_pulls_the_donor_deltas_out_of_a_prelaunch_record(self):
        import tempfile
        tmp = tempfile.mkdtemp()
        doc = {"all_passed": True, "gates": [
            {"gate": "donor_control", "content_delta_both": 0.03,
             "content_delta_e": 0.02, "content_delta_z": 0.01,
             "column_delta": 0.09},
            {"gate": "read_live_fraction",
             "at_run_config": {"read_live_frac": 0.53}}]}
        ev = read_eval(self._write(tmp, "p.json", doc))
        assert ev["kind"] == "prelaunch"
        assert ev["content_delta_z"] == 0.01
        assert ev["read_live_frac"] == 0.53


class TestThePrintedComparison:

    def _build(self, tmp):
        a1 = _write_diag(tmp, "a1", _rows(10))
        a3 = _write_diag(tmp, "a3z", _rows(10, base=2.99, latent_carry=1,
                                           z_read_grad_frac=0.53,
                                           z_write_grad_frac=1.0,
                                           z_over_e_norm=0.002,
                                           gate_z_left_init=1,
                                           gate_left_init=1,
                                           fg_z_at_bias=0.73))
        return build({"a1": [a1], "a3z": [a3]}, {}, {}, "a1", 10, 500)

    def test_it_prints_ascii_and_names_what_each_arm_is(self):
        import tempfile, pathlib
        rec = self._build(pathlib.Path(tempfile.mkdtemp()))
        buf = io.StringIO()
        print_comparison(rec, out=buf)
        text = buf.getvalue()
        text.encode("ascii")
        assert "B2's buffer, the family control" in text
        assert "gated ring + Z" in text

    def test_it_says_in_its_own_output_what_it_cannot_say(self):
        """A green probe in a log gets quoted later as 'Z worked'.  The
        disclaimer travels with the numbers or it does not travel."""
        import tempfile, pathlib
        rec = self._build(pathlib.Path(tempfile.mkdtemp()))
        buf = io.StringIO()
        print_comparison(rec, out=buf)
        assert "cannot decide whether Z" in buf.getvalue()

    def test_a_dead_gate_is_called_out_by_name(self):
        import tempfile, pathlib
        tmp = pathlib.Path(tempfile.mkdtemp())
        rec = build({"a1": [_write_diag(tmp, "a1", _rows(6))]}, {}, {},
                    "a1", 6, 200)
        buf = io.StringIO()
        print_comparison(rec, out=buf)
        assert "NOT LIVE" in buf.getvalue()
        assert "16.8M dead parameters" in buf.getvalue()

    def test_the_trajectory_is_not_carried_into_the_saved_record(self):
        """The jsonl can be thousands of rows; the saved comparison is meant to
        be readable and diffable."""
        import tempfile, pathlib
        rec = self._build(pathlib.Path(tempfile.mkdtemp()))
        assert "diag" not in rec["arms"]["a1"]
        json.dumps(rec)


class TestTheWidthContrast:
    """`tools/compare_width.py` -- the delivered-rank read that decides whether
    W=32 -> 16 cost anything that reaches attention.

    The slice measured the WRITE BASIS (`summary_emb`: 20.02 of 32 -> 10.90 of
    16, i.e. 54% at 50% of the columns) and that retires P1.0's at-init
    argument.  It does NOT decide the question, because the collapse from ~20 to
    ~4 happens in the forward pass.  The rule is pre-registered in
    p11_z_probe_prereg.md section 4; these tests hold the instrument to it.
    """

    @staticmethod
    def _walk(n_vec, pr, rows=None):
        rows = rows if rows is not None else n_vec * 4
        return {"geometry": {"n_vec": n_vec},
                "final_carry": {"rows": rows, "e_eff_rank_pr": pr,
                                "e_eff_rank_entropy": pr * 1.8,
                                "e_centred_cosine": 0.9, "e_row_norm": 170.0}}

    def test_unchanged_delivered_rank_reads_as_survives(self):
        from tools.compare_width import score
        r = score(self._walk(32, 4.10), self._walk(16, 4.02))
        assert r["verdict"] == "SURVIVES"
        assert "not reaching the token stream" in r["why"]

    def test_proportional_loss_reads_as_binds(self):
        from tools.compare_width import score
        r = score(self._walk(32, 4.10), self._walk(16, 2.05))
        assert r["verdict"] == "BINDS"
        # The pre-registered response is NOT a revert -- that would restore the
        # eviction cliff the gate exists to remove.
        assert "NOT reverting" in r["why"] and "cc=16" in r["why"]

    def test_the_middle_band_is_named_rather_than_rounded_to_a_side(self):
        """Between the bands the honest answer is 'report the number'.  An
        instrument that snapped to the nearer verdict would be choosing the
        flattering reading on the user's behalf."""
        from tools.compare_width import score
        r = score(self._walk(32, 4.00), self._walk(16, 3.00))   # 75% at 50%
        assert r["verdict"] == "PARTIAL"

    def test_two_walks_at_the_same_width_are_invalid_not_survives(self):
        """The failure that would otherwise read as the best possible result:
        forgetting --set accum_vecs on one side makes both walks W=32, the
        ratio is exactly 1.0, and it would print SURVIVES."""
        from tools.compare_width import score
        r = score(self._walk(32, 4.10), self._walk(32, 4.10))
        assert r["verdict"] == "INVALID"
        assert "no width contrast" in r["why"]

    def test_a_result_does_not_fail_the_job_but_an_invalid_comparison_does(self):
        """A gate that goes red on a negative finding teaches people to skip the
        gate.  BINDS is a result; a broken comparison is not."""
        from tools.compare_width import score
        import io as _io
        from tools.compare_width import print_report
        for verdict, rec in (("BINDS", score(self._walk(32, 4.1), self._walk(16, 2.0))),
                             ("SURVIVES", score(self._walk(32, 4.1), self._walk(16, 4.1)))):
            assert rec["verdict"] == verdict
            buf = _io.StringIO()
            print_report(rec, out=buf)
            buf.getvalue().encode("ascii")

    def test_the_thresholds_are_module_constants(self):
        """So p11 and the instrument cannot drift apart -- the same rule p10
        follows for the gate bars."""
        from tools import compare_width
        assert compare_width.DELIVERED_RETAINED_SURVIVES == 0.90
        assert compare_width.DELIVERED_BINDS_MARGIN == 0.10

    def test_a_rank_increase_is_invalid_not_survives(self):
        """REGRESSION from the 2026-09-16 19:09 job, which printed
        '897% retained' and scored SURVIVES.

        accum_max defaulted to 128 on BOTH sides, which is 8 chunks at W=16 but
        only 4 at W=32 -- so the W=32 carry held 128 rows drawn from four
        forwards (PR 1.64) against W=16's 128 rows from eight (PR 14.72).  The
        9x gap was an EVICTION artifact with no width content, and the lower
        bound alone could not see it: halving the write width cannot multiply
        delivered rank.
        """
        from tools.compare_width import score
        r = score(self._walk(32, 1.641, rows=128), self._walk(16, 14.719, rows=128))
        assert r["verdict"] == "INVALID"

    def test_a_mismatched_chunk_count_is_caught_before_the_ratio(self):
        """The cause, not just the symptom: accum_max caps ROWS, so holding it
        fixed across two widths changes the HORIZON."""
        from tools.compare_width import score
        r = score(self._walk(32, 4.0, rows=128),      # 4 chunks
                  self._walk(16, 4.0, rows=128))      # 8 chunks
        assert r["verdict"] == "INVALID"
        assert "different numbers of CHUNKS" in r["why"]
        # and it says how to fix it, per width
        assert "accum_max = chunks x W" in r["why"]

    def test_matched_chunk_counts_still_score_normally(self):
        """The guard must not swallow the comparison it exists to protect:
        accum_max = chunks x W on each side keeps the chunk count equal."""
        from tools.compare_width import score
        r = score(self._walk(32, 4.10, rows=256),     # 8 chunks
                  self._walk(16, 4.02, rows=128))     # 8 chunks
        assert r["verdict"] == "SURVIVES"
        assert r["chunks_retained"] == [8.0, 8.0]
