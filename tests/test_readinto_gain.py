"""P3.0 tiers 0.5 / 1 — the in-loop read-site gain sweep.

The verdict arithmetic is tested WITHOUT a GPU, which is the whole point:
RED 14's sign inversion survived because the effects arithmetic lived in a
closure inside `main()` and could not be reached without one.  `summarize()`
is module-level here for exactly that reason.

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

from evals.diag_readinto_gain import (SWAMP_MULT_OF_X, summarize,  # noqa: E402
                                      print_report, set_gate)

CHANCE = 11.5157          # ln(vocab) on the OLMo family


def _cells(**kw):
    """name -> a one-element list, so the means are exactly what is passed."""
    return {k: [v] for k, v in kw.items()}


class TestTheVerdict:

    def test_a_responding_site_reads_live(self):
        rec = summarize(
            _cells(**{"real@0": 3.20, "noise@0": 3.20,
                      "real@0.1": 3.05, "noise@0.1": 3.06}),
            {"real@0": 0.0, "noise@0": 0.0, "real@0.1": 0.3, "noise@0.1": 0.3},
            CHANCE)
        assert rec["reading"] == "LIVE_site_has_gain"

    def test_a_flat_site_reads_dead(self):
        rec = summarize(
            _cells(**{"real@0": 3.20, "real@0.001": 3.2001,
                      "real@0.01": 3.2002, "real@10": 9.9}),
            {"real@0": 0.0, "real@0.001": 0.001, "real@0.01": 0.01,
             "real@10": 14.0},                      # the loud cell is OOD
            CHANCE)
        assert rec["reading"] == "DEAD_no_gain_in_distribution"
        assert rec["ood_cells"] == ["real@10"]
        assert rec["knob_live"], (
            "the oversized cell is what proves the knob is connected; without "
            "it a flat sweep cannot be told from one that never landed")

    def test_nothing_moving_anywhere_is_invalid_not_dead(self):
        """The RED 8/9/10 shape.  A sweep with nothing in it that moves cannot
        distinguish 'no gain' from 'the knob was never wired', and returning a
        verdict there is how a confident wrong number gets into a plan."""
        rec = summarize(
            _cells(**{"real@0": 3.20, "real@1": 3.2001, "real@10": 3.2002}),
            {"real@0": 0.0, "real@1": 0.01, "real@10": 0.02}, CHANCE)
        assert rec["reading"] == "INVALID_knob_unproven"

    def test_a_run_at_chance_vetoes_the_verdict(self):
        """RED 10: gate 4 ranked three losses that were all at ln(vocab) and
        the ordering set the direction of the P2 program.  The level VETOES,
        it does not merely warn."""
        rec = summarize(
            _cells(**{"real@0": 11.50, "real@0.1": 11.20}),
            {"real@0": 0.0, "real@0.1": 0.3}, CHANCE)
        assert rec["reading"] == "INVALID_at_chance"
        assert rec["intact"]["at_chance"] is True

    def test_the_swamp_window_excludes_cells_that_only_break_the_model(self):
        """RED 11's second bug, at the site where the competing quantity is x
        itself: above ||delta|| ~ ||x|| the injection is replacing the loop's
        working state, not being read by it."""
        rec = summarize(
            _cells(**{"real@0": 3.20, "real@1": 3.19, "real@10": 8.0}),
            {"real@0": 0.0, "real@1": SWAMP_MULT_OF_X * 0.5,
             "real@10": SWAMP_MULT_OF_X * 5},
            CHANCE)
        assert rec["ood_cells"] == ["real@10"]
        assert rec["span_in_dist_nats"] == pytest.approx(0.01)
        assert rec["span_nats"] == pytest.approx(4.81)   # 8.0 - 3.19

    def test_deltas_are_measured_from_the_intact_cell(self):
        rec = summarize(
            _cells(**{"real@0": 3.20, "real@0.1": 3.05}),
            {"real@0": 0.0, "real@0.1": 0.3}, CHANCE)
        assert rec["delta_from_intact"]["real@0.1"] == pytest.approx(-0.15)
        assert "real@0" not in rec["delta_from_intact"]


class TestTheRealVersusNoisePair:

    def test_pairs_are_matched_by_gate(self):
        rec = summarize(
            _cells(**{"real@0": 3.2, "noise@0": 3.2,
                      "real@0.1": 3.05, "noise@0.1": 3.08}),
            {k: 0.3 for k in ("real@0", "noise@0", "real@0.1", "noise@0.1")},
            CHANCE)
        assert rec["real_minus_noise"]["0.1"] == pytest.approx(-0.03)
        assert rec["content_effect_at_init"] == pytest.approx(0.03)

    def test_an_unpaired_cell_is_simply_absent(self):
        rec = summarize(_cells(**{"real@0": 3.2, "real@0.1": 3.0}),
                        {"real@0": 0.0, "real@0.1": 0.3}, CHANCE)
        assert rec["real_minus_noise"] == {}
        assert rec["content_effect_at_init"] == 0.0


class TestTheReport:

    def test_it_prints_and_stays_ascii(self):
        import io
        rec = summarize(
            _cells(**{"real@0": 3.20, "noise@0": 3.20,
                      "real@0.1": 3.05, "noise@0.1": 3.06}),
            {"real@0": 0.0, "noise@0": 0.0, "real@0.1": 0.3, "noise@0.1": 0.3},
            CHANCE)
        rec.update(tier="1", module="LatentRead", examples=50, T=8,
                   x_read_norm=10.2, z_row_norm=2.1)
        buf = io.StringIO()
        print_report(rec, buf)
        s = buf.getvalue()
        s.encode("ascii")
        assert "LIVE_site_has_gain" in s
        assert "BOTH arms" in s

    def test_a_missing_norm_prints_as_a_dash_and_not_as_zero(self):
        import io
        rec = summarize(_cells(**{"real@0": 3.2, "real@0.1": 3.0}),
                        {"real@0": None, "real@0.1": None}, CHANCE)
        rec.update(tier="0.5", module="LatentRefresh", examples=3, T=8,
                   x_read_norm=None, z_row_norm=None)
        buf = io.StringIO()
        print_report(rec, buf)
        assert "-" in buf.getvalue()


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
        """The intact cell must be the intact model, or every delta in the
        table is measured from the wrong baseline."""
        from cortex_memory.latent_read import LatentRead
        r = LatentRead(16, n_heads=4)
        set_gate(r, 0.0)
        delta = r(torch.randn(1, 5, 16), torch.randn(1, 3, 16))
        assert float(delta.detach().abs().sum()) == 0.0


class TestTheScaleRecording:
    """p30 S2 says to MEASURE the scale at the new site rather than assume it.
    These pin that the graft records it, in fp32 and as a ROW norm -- the unit
    RED 11 got wrong by sqrt(D) = 45.25."""

    def test_the_three_norms_are_recorded_after_a_forward(self):
        from test_read_into import _model, _chain
        m = _model()
        _chain(m, n_chunks=2)
        c = m.cortex
        assert c._x_read_norm is not None and c._x_read_norm > 0
        assert c._z_row_norm is not None and c._z_row_norm > 0
        assert c._z_read_delta_norm is not None

    def test_the_recorded_norm_is_a_row_norm_not_a_per_element_rms(self):
        from cortex_graft import _row_norm
        D = 2048
        t = torch.full((1, 4, D), 1.0)
        assert _row_norm(t) == pytest.approx(D ** 0.5, rel=1e-4)

    def test_it_is_computed_in_fp32_from_a_bf16_state(self):
        """The bf16 difference trap: a bf16 pass invented a delta plateau in
        P0.1 that was the rounding floor wearing the shape of a result."""
        from cortex_graft import _row_norm
        t = torch.full((1, 2, 512), 1.0, dtype=torch.bfloat16)
        assert _row_norm(t) == pytest.approx(512 ** 0.5, rel=1e-3)

    def test_the_delta_norm_tracks_the_gate(self):
        from test_read_into import _model, _chain
        m = _model()
        _chain(m, n_chunks=2)
        small = m.cortex._z_read_delta_norm
        set_gate(m.cortex.latent_reader, 1.0)
        _chain(m, n_chunks=2)
        big = m.cortex._z_read_delta_norm
        assert big > small * 5, (
            "the sweep's axis is this number; if it does not move with the "
            "gate the reported ratios are fiction")
