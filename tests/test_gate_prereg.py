"""
The P1.0 pre-registration, and the instrument that scores it.

`p10_gated_prereg.md` is the pre-registration; `evals/eval_gate_prereg.py` holds
the thresholds as constants so the two cannot drift.  What is pinned here is the
part a reader is entitled to check without trusting either file: that the
published fg thresholds FOLLOW from the retention model rather than being
transcribed, and that the fg decomposition can actually tell a content-dependent
gate apart from an EMA with a per-channel shape.  The second one matters because
the headline negative result ("the gate is an EMA with 16.8M extra parameters")
is only credible if the measurement could have said otherwise.

Run: /c/Users/henry/miniconda3/envs/cortex-retro/python.exe -m pytest tests/ -q
"""
from __future__ import annotations

import os
import sys

import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from evals.eval_gate_prereg import (  # noqa: E402
    BABILONG_CELLS, FG_ACROSS_INPUT_STD_MIN, FG_ACROSS_INPUT_SWING_MIN,
    FG_AT_INIT, IG_AT_INIT, PREREGISTERED_FG_MIN, RETENTION_FLOOR,
    fg_threshold, spread)

DOC = os.path.join(os.path.dirname(REPO), "p10_gated_prereg.md")


class TestThresholdsAreDerived:
    """The two published numbers, recomputed from the model."""

    def test_16k_needs_fg_818(self):
        # 16,384 tokens / 512-token chunks = 32 chunks; K/W = 4, so 8 laps.
        got = fg_threshold(16_384, 512, 4)
        assert abs(got - 0.818) < 5e-4, got

    def test_32k_derives_to_9043_and_the_registered_bar_rounds_it_up(self):
        """0.2**(1/16) = 0.90430.  The bar written down is 0.905 — that
        derivation rounded UP by 0.0007.  Kept rather than corrected, because
        rounding up is the direction that cannot flatter a result, and moving a
        pre-registered threshold after the fact is the thing pre-registration
        exists to prevent."""
        got = fg_threshold(32_768, 512, 4)
        assert abs(got - 0.90430) < 5e-5, got
        assert PREREGISTERED_FG_MIN[32_768] == 0.905
        assert PREREGISTERED_FG_MIN[32_768] > got          # strictly harder
        assert PREREGISTERED_FG_MIN[16_384] == 0.818

    def test_the_init_gate_fails_both_by_construction(self):
        """4% and 0.3% — the numbers the handoff quotes.  If these ever came out
        passing, the arm would have nothing to demonstrate."""
        assert abs(IG_AT_INIT * FG_AT_INIT ** 8 - 0.041) < 1e-3
        assert abs(IG_AT_INIT * FG_AT_INIT ** 16 - 0.0033) < 1e-3
        for ctx in BABILONG_CELLS:
            assert FG_AT_INIT < fg_threshold(ctx, 512, 4)
            assert FG_AT_INIT < PREREGISTERED_FG_MIN[ctx]

    def test_a_longer_lap_lowers_the_bar(self):
        """K=128 at W=16 is an 8-chunk lap, which is the follow-up cell the
        horizon claim actually lives in.  It must be an EASIER threshold, or the
        reason for running it is wrong."""
        assert fg_threshold(16_384, 512, 8) < fg_threshold(16_384, 512, 4)

    def test_an_ig_below_the_floor_is_reported_as_unreachable(self):
        """Honest rather than clipped: if the input gate alone already admits
        less than the retention floor, no forget gate can recover it."""
        assert fg_threshold(16_384, 512, 4, ig=0.05) == float("inf")

    def test_both_rows_come_from_one_floor(self):
        """The floor is a choice; it is made ONCE.  Two independently chosen
        thresholds would be two decisions dressed as a model."""
        for ctx in BABILONG_CELLS:
            laps = (ctx // 512) // 4
            assert abs(IG_AT_INIT * fg_threshold(ctx, 512, 4) ** laps
                       - RETENTION_FLOOR) < 1e-9


class TestDocAndCodeAgree:
    """The doc restates the thresholds in prose.  Cheap check that nobody
    edited one side."""

    def test_the_published_numbers_are_in_the_doc(self):
        if not os.path.exists(DOC):
            import pytest
            pytest.skip(f"{DOC} not present in this checkout")
        text = open(DOC, encoding="utf-8").read()
        assert "0.818" in text and "0.905" in text
        assert str(FG_ACROSS_INPUT_STD_MIN) in text
        assert str(FG_ACROSS_INPUT_SWING_MIN) in text


class TestSpreadDecomposition:
    """The measurement has to be able to return the negative AND the positive."""

    def _fg(self, n_inputs=64, rows=16, ch=32, across=0.0, within=0.0,
            base=FG_AT_INIT):
        """Synthesise fg tapes with a known decomposition.

        `across`  a per-input offset, identical across rows and channels.
        `within`  a per-(row, channel) offset, identical across inputs.
        """
        torch.manual_seed(0)
        a = torch.randn(n_inputs, 1, 1) * across
        w = torch.randn(1, rows, ch) * within
        x = base + a + w
        return [x[i:i + 1] for i in range(n_inputs)]

    def test_a_constant_gate_reads_as_zero_across_inputs(self):
        """The EMA null, which is exact at init."""
        s = spread(self._fg())
        assert s["across_input_std"] < 1e-6
        assert s["within_input_std"] < 1e-6
        assert abs(s["mean"] - FG_AT_INIT) < 1e-6

    def test_a_static_per_channel_profile_is_NOT_counted_as_content_dependence(self):
        """The trap the whole decomposition exists for: a learned but
        input-independent profile gives a wide p10/p90 while the gate is blind.
        A raw spread readout would call this a pass."""
        s = spread(self._fg(within=0.10))
        assert s["p90"] - s["p10"] > 0.15            # looks spread out...
        assert s["across_input_std"] < 1e-6          # ...and is still blind
        assert s["within_input_std"] > 0.05
        assert s["across_input_std"] < FG_ACROSS_INPUT_STD_MIN

    def test_content_dependence_is_detected(self):
        s = spread(self._fg(across=0.10))
        assert s["across_input_std"] > FG_ACROSS_INPUT_STD_MIN
        assert s["within_input_std"] < 1e-6

    def test_the_two_components_separate_when_both_are_present(self):
        s = spread(self._fg(across=0.08, within=0.03))
        assert abs(s["across_input_std"] - 0.08) < 0.02
        assert abs(s["within_input_std"] - 0.03) < 0.02

    def test_the_swing_bar_tracks_the_across_input_component(self):
        """1b is stated in p90-p10 PER row/channel, so it must move with the
        across-input component and ignore the static one."""
        static = spread(self._fg(within=0.30))
        assert static["across_input_swing"] < 1e-5
        live = spread(self._fg(across=0.12))
        assert live["across_input_swing"] > 0.2      # ~2.56 sigma for a normal

    def test_sample_count_is_reported(self):
        s = spread(self._fg(n_inputs=8, rows=4, ch=5))
        assert s["merges_taped"] == 8 and s["samples"] == 8
