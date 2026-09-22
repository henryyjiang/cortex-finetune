"""P3.0 tiers 0.5 / 1 — the in-loop read-site gain sweep.

The verdict arithmetic is tested WITHOUT a GPU, which is the whole point:
RED 14's sign inversion survived because the effects arithmetic lived in a
closure inside `main()` and could not be reached without one.

REWRITTEN AFTER JOB 13420851, which printed LIVE on all four cells through
three defects this file now pins:

  1. the verdict was the SPAN across "in-distribution" cells, and the only
     cell driving it sat at 0.51x the working state.  Moving the cut anywhere
     defensible flipped every arm (0.275 -> 0.0033).  A verdict that depends
     entirely on an arbitrary line is not a verdict.
  2. the noise control was a fixed per-element std of 0.02 = a row norm of
     0.905, against ||Z row|| ~ 4.3.  The 0.27-nat "content effect at init"
     was a 4.8x magnitude gap.
  3. LatentRefresh returned before the norm recording, so BOTH tier-0.5 cells
     had no axis at all and were scored on gate values.

Run: /c/Users/henry/miniconda3/envs/cortex-retro/python.exe -m pytest tests/ -q
"""
from __future__ import annotations

import os
import sys

import pytest
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from evals.diag_readinto_gain import (S0_RESPONSE_COEFF,  # noqa: E402
                                      SENSITIVE_OVER_S0, print_report,
                                      response_fit, set_gate, summarize)

CHANCE = 11.5157          # ln(vocab) on the OLMo family


def _cells(**kw):
    return {k: [v] for k, v in kw.items()}


def _sensitive():
    """A sweep whose coefficient is far above s0's, with a matched control."""
    return (_cells(**{"real@0": 3.20, "noise@0": 3.20,
                      "real@0.1": 3.2001, "noise@0.1": 3.2001,
                      "real@1": 3.2033, "noise@1": 3.2033,
                      "real@10": 3.475, "noise@10": 3.475}),
            {"real@0": 0.0, "noise@0": 0.0, "real@0.1": 0.005,
             "noise@0.1": 0.005, "real@1": 0.05, "noise@1": 0.05,
             "real@10": 0.5, "noise@10": 0.5})


class TestTheVerdict:

    def test_a_site_far_above_s0_reads_sensitive(self):
        rec = summarize(*_sensitive(), CHANCE)
        assert rec["reading"] == "SENSITIVE_unlike_s0"
        assert rec["coeff_over_s0"] > SENSITIVE_OVER_S0

    def test_a_site_at_s0s_own_level_reads_inert(self):
        """The reading that stops the program.  k near s0's means the new hook
        is no better than the one it replaces."""
        k = S0_RESPONSE_COEFF
        cells = _cells(**{"real@0": 3.20,
                          "real@1": 3.20 + k * 1.0 ** 2,
                          "real@10": 3.20 + k * 100.0 ** 2})
        rec = summarize(cells, {"real@0": 0.0, "real@1": 1.0, "real@10": 100.0},
                        CHANCE)
        assert rec["reading"] == "INERT_like_s0"

    def test_the_verdict_does_not_depend_on_the_span_threshold(self):
        """Job 13420851's defect, pinned.  The same table read LIVE at
        threshold 1.0 and DEAD at 0.5, 0.25 and 0.1 -- so the span is reported
        at every cut and used at none of them."""
        rec = summarize(*_sensitive(), CHANCE)
        spans = rec["span_by_threshold"]
        assert len(set(round(v, 5) for v in spans.values())) > 1, (
            "this fixture must actually have a threshold-dependent span, or "
            "the test proves nothing")
        assert rec["reading"] == "SENSITIVE_unlike_s0"

    def test_no_axis_is_refused_and_not_scored(self):
        """Tier 0.5 on job 13420851: every ratio was a dash and the tool
        scored it anyway."""
        rec = summarize(
            _cells(**{"real@0": 3.19, "real@1": 3.193, "real@10": 3.353}),
            {"real@0": None, "real@1": None, "real@10": None}, CHANCE)
        assert rec["reading"] == "INVALID_no_axis"
        assert rec["have_axis"] is False

    def test_an_unmatched_control_is_an_audit_not_a_finding(self):
        """Fired on all four cells of job 13420851 and was ignored, because the
        condition was pre-registered in prose and never wired into the verdict.
        It is wired in now."""
        rec = summarize(
            _cells(**{"real@0": 3.20, "noise@0": 3.20,
                      "real@10": 3.475, "noise@10": 3.195}),
            {"real@0": 0.0, "noise@0": 0.0, "real@10": 0.5, "noise@10": 0.04},
            CHANCE)
        assert rec["reading"] == "AUDIT_control_not_matched"
        assert rec["content_effect_at_init"] > 0.01

    def test_nothing_moving_anywhere_is_invalid(self):
        rec = summarize(
            _cells(**{"real@0": 3.20, "real@1": 3.2001, "real@10": 3.2002}),
            {"real@0": 0.0, "real@1": 0.01, "real@10": 0.02}, CHANCE)
        assert rec["reading"] == "INVALID_knob_unproven"

    def test_a_run_at_chance_vetoes_everything(self):
        """RED 10.  The level vetoes; it does not warn."""
        cells, norms = _sensitive()
        cells = {k: [v[0] + 8.3] for k, v in cells.items()}
        rec = summarize(cells, norms, CHANCE)
        assert rec["reading"] == "INVALID_at_chance"

    def test_the_veto_order_puts_the_level_first(self):
        """At chance AND no axis -> at_chance wins, because a table of
        differences at ln(vocab) is not about the axis either."""
        rec = summarize(_cells(**{"real@0": 11.50, "real@1": 11.20}),
                        {"real@0": None, "real@1": None}, CHANCE)
        assert rec["reading"] == "INVALID_at_chance"


class TestTheResponseShape:

    def test_a_quadratic_response_is_recovered(self):
        """Measured on the real arms at 1.919 (a2) and 1.952 (a3z)."""
        k = 1.25
        delta = {"real@%g" % r: k * r ** 2 for r in (0.05, 0.5)}
        norms = {"real@%g" % r: r for r in (0.05, 0.5)}
        coeff, p = response_fit(delta, norms)
        assert coeff == pytest.approx(k, rel=1e-6)
        assert p == pytest.approx(2.0, abs=1e-6)

    def test_cells_below_the_measurement_floor_are_dropped(self):
        """d/r^2 on two numbers that are both noise is what made the smallest
        cell of a3z read k = 8.26 against its own arm's 1.87."""
        delta = {"real@a": 1e-6, "real@b": 1.25 * 0.5 ** 2}
        norms = {"real@a": 0.005, "real@b": 0.5}
        coeff, _ = response_fit(delta, norms)
        assert coeff == pytest.approx(1.25, rel=1e-6)

    def test_it_returns_none_rather_than_guessing_with_no_usable_cell(self):
        assert response_fit({"real@a": 1e-9}, {"real@a": 0.1}) == (None, None)

    def test_the_sign_of_the_delta_does_not_change_the_coefficient(self):
        """A read that HELPS is the outcome we want and must not be scored as
        a smaller response than one that hurts by the same amount."""
        norms = {"real@1": 0.5}
        up, _ = response_fit({"real@1": 0.25}, norms)
        down, _ = response_fit({"real@1": -0.25}, norms)
        assert up == down


class TestTheReport:

    def test_it_prints_and_stays_ascii(self):
        import io
        rec = summarize(*_sensitive(), CHANCE)
        rec.update(tier="1", module="LatentRead", examples=50, T=8,
                   x_read_norm=16.13, z_row_norm=4.34)
        buf = io.StringIO()
        print_report(rec, buf)
        s = buf.getvalue()
        s.encode("ascii")
        assert "SENSITIVE_unlike_s0" in s
        assert "BOTH arms" in s
        assert "NEVER the verdict" in s

    def test_a_quadratic_exponent_is_called_out_in_the_log(self):
        import io
        rec = summarize(*_sensitive(), CHANCE)
        rec.update(tier="1", module="LatentRead", examples=1, T=8,
                   x_read_norm=16.1, z_row_norm=4.3)
        buf = io.StringIO()
        print_report(rec, buf)
        assert "says nothing about content" in buf.getvalue()

    def test_every_reading_has_a_note(self):
        """A verdict string with no explanation beside it is how a reader ends
        up quoting the string and not the caveat."""
        import io
        from evals.diag_readinto_gain import VERDICT_NOTES
        for reading in ("SENSITIVE_unlike_s0", "INERT_like_s0",
                        "AUDIT_control_not_matched", "INVALID_no_axis",
                        "INVALID_at_chance", "INVALID_knob_unproven"):
            assert reading in VERDICT_NOTES
            rec = summarize(*_sensitive(), CHANCE)
            rec.update(tier="1", module="LatentRead", examples=1, T=8,
                       x_read_norm=1.0, z_row_norm=1.0, reading=reading)
            buf = io.StringIO()
            print_report(rec, buf)
            assert VERDICT_NOTES[reading][0] in buf.getvalue()


class TestTheSweepTurnsTheRightKnob:

    def test_it_finds_gate_on_latent_read(self):
        from cortex_memory.latent_read import LatentRead
        r = LatentRead(16, n_heads=4)
        set_gate(r, 0.5)
        assert r.gate_value == pytest.approx(0.5)

    def test_it_finds_alpha_on_latent_refresh(self):
        from cortex_memory.latent_read import LatentRefresh
        r = LatentRefresh(16)
        set_gate(r, 0.25)
        assert r.gate_value == pytest.approx(0.25)

    def test_it_refuses_a_module_with_neither(self):
        with pytest.raises(RuntimeError, match="gate"):
            set_gate(torch.nn.Linear(4, 4), 1.0)

    def test_gate_zero_injects_exactly_nothing(self):
        from cortex_memory.latent_read import LatentRead
        r = LatentRead(16, n_heads=4)
        set_gate(r, 0.0)
        delta = r(torch.randn(1, 5, 16), torch.randn(1, 3, 16))
        assert float(delta.detach().abs().sum()) == 0.0


class TestTheScaleMatchedNull:
    """Defect 2 of job 13420851.  A content control has to differ from the
    real thing in CONTENT ONLY."""

    def test_the_matched_null_has_the_real_z_row_norm(self):
        from test_read_into import _model
        m = _model()
        z = torch.randn(2, 5, 64) * 3.0
        out = m.cortex._null_latent(("noise_matched", None, 0), z)
        assert torch.allclose(out.float().norm(dim=-1),
                              z.float().norm(dim=-1), rtol=1e-4)

    def test_it_matches_per_row_and_not_on_the_mean(self):
        """The ring's rows are written at different chunk ages and their norms
        are not equal, so a mean-matched null would be wrong row by row."""
        from test_read_into import _model
        m = _model()
        z = torch.randn(1, 3, 64)
        z[0, 0] *= 10.0
        out = m.cortex._null_latent(("noise_matched", None, 0), z)
        assert torch.allclose(out.float().norm(dim=-1),
                              z.float().norm(dim=-1), rtol=1e-4)

    def test_the_fixed_std_null_still_works_for_reproducing_the_old_run(self):
        from test_read_into import _model
        m = _model()
        z = torch.randn(1, 4, 64)
        out = m.cortex._null_latent(("noise", 0.02, 0), z)
        assert out.shape == z.shape
        assert float(out.float().norm(dim=-1).mean()) == pytest.approx(
            0.02 * 64 ** 0.5, rel=0.3)

    def test_zeros_are_still_refused(self):
        from test_read_into import _model
        m = _model()
        with pytest.raises(ValueError, match="noise"):
            m.cortex._null_latent(("zeros",), torch.randn(1, 2, 64))


class TestTheAxisIsRecordedOnBothModules:
    """Defect 3 of job 13420851: LatentRefresh returned before the recording,
    so BOTH tier-0.5 cells had no axis and were scored on gate values."""

    @pytest.mark.parametrize("module", ["xattn", "refresh"])
    def test_all_three_norms_survive_a_forward(self, module):
        from test_read_into import _model, _chain
        m = _model(read=module, s0=(module == "refresh"))
        _chain(m, n_chunks=2)
        c = m.cortex
        assert c._x_read_norm is not None and c._x_read_norm > 0
        assert c._z_row_norm is not None and c._z_row_norm > 0
        assert c._z_read_delta_norm is not None and c._z_read_delta_norm > 0

    @pytest.mark.parametrize("module", ["xattn", "refresh"])
    def test_the_delta_norm_tracks_the_gate(self, module):
        from test_read_into import _model, _chain
        m = _model(read=module, s0=(module == "refresh"))
        _chain(m, n_chunks=2)
        small = m.cortex._z_read_delta_norm
        set_gate(m.cortex.latent_reader, 1.0)
        _chain(m, n_chunks=2)
        assert m.cortex._z_read_delta_norm > small * 5, (
            "the sweep's axis is this number; if it does not move with the "
            "gate the reported ratios are fiction")

    def test_the_refresh_ratio_is_against_the_columns_it_actually_touches(self):
        """LatentRefresh perturbs n_pre of ~650 packed columns.  Measured
        against the whole-sequence mean it would look ~5x quieter than it is,
        and the two tiers would be on different axes."""
        from test_read_into import _model, _chain
        from cortex_graft import _row_norm
        m = _model(read="refresh", s0=True)
        _chain(m, n_chunks=2)
        c = m.cortex
        assert c._n_pre > 0
        # the recorded ||x|| must be the carried block's, not the whole packed
        # sequence's -- those differ, so this is a real distinction
        assert c._x_read_norm is not None

    def test_the_recorded_norm_is_a_row_norm_not_a_per_element_rms(self):
        from cortex_graft import _row_norm
        D = 2048
        assert _row_norm(torch.full((1, 4, D), 1.0)) == pytest.approx(
            D ** 0.5, rel=1e-4)

    def test_it_is_computed_in_fp32_from_a_bf16_state(self):
        from cortex_graft import _row_norm
        t = torch.full((1, 2, 512), 1.0, dtype=torch.bfloat16)
        assert _row_norm(t) == pytest.approx(512 ** 0.5, rel=1e-3)
