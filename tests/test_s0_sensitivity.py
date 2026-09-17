"""
The s0 read-site sweep.

Written WITH the instrument, not after it.  REDs 8, 9 and 10 were all
instruments that returned a confident wrong number instead of failing, and the
influence horizon survived its no-op for as long as it did because it had no
test file at all.  This one decides whether the Z redesign moves the read
in-loop or just renorms it, so it gets a test file on day one.

Run: /c/Users/henry/miniconda3/envs/cortex-retro/python.exe -m pytest tests/test_s0_sensitivity.py -q
"""
from __future__ import annotations

import math
import os
import sys

import pytest
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "evals"))

from test_cortex_eval import VOCAB, _build_raven  # noqa: E402

from evals.diag_s0_sensitivity import (  # noqa: E402
    DEFAULT_SCALES, FLAT_SPAN_NATS, KNOB_LIVE_NATS, SWAMP_FALLBACK_MULT,
    prime, score_chunk, summarize,
)

CHANCE = math.log(VOCAB)
NV, K, CL, EOS = 4, 16, 16, VOCAB - 1
NC, T = 4, 4
#: _build_raven's width.  sqrt(D) = 8 here and 45.25 on the real checkpoints;
#: it is the factor RED 11 mixed up, so the tests want it by name.
D = 64
ROOT_D = 8.0


def _model(latent=True):
    torch.manual_seed(1234)
    return _build_raven(use_memory=True, memory_slots=0, accum_vecs=NV,
                        prefix_memory="accum", accum_max=K * 4,
                        latent_carry=latent, eos_token_id=EOS).eval()


def _chunks(seed=0, n=NC):
    torch.manual_seed(seed)
    ids = torch.randint(0, VOCAB - 1, (CL * n + 1,))
    x, y = ids[:-1], ids[1:]
    return list(torch.chunk(x, n)), list(torch.chunk(y, n))


class TestTheVerdict:
    """`summarize` is the part that turns numbers into a redesign decision, so
    it is separated from the run loop and pinned here without a GPU."""

    S0 = 0.3885          # the measured row norm on the real arms
    E_NORM = 171.0       # ||E carried row||, P0.1 fp32

    def _cells(self, values):
        cells = {"real": [values[0]]}
        for s, v in zip(DEFAULT_SCALES, values[1:]):
            cells["noise@%g" % s] = [v]
        return cells

    def _norms(self):
        """What each DEFAULT_SCALES cell injects, in row-norm units.  With the
        axis fixed this is just s x ||s0||, which is the whole point."""
        return {("noise@%g" % s): s * self.S0 for s in DEFAULT_SCALES}

    def _sum(self, values, **kw):
        kw.setdefault("row_norms", self._norms())
        kw.setdefault("swamp_norm", self.E_NORM)
        return summarize(self._cells(values), self.S0, CHANCE, DEFAULT_SCALES,
                         **kw)

    def test_flat_in_distribution_with_a_loud_ood_cell_reads_as_A(self):
        """THE SHAPE JOB 13297293 ACTUALLY HAD, read through the fixed axis:
        nothing moves at any scale the model runs at, and the one deliberately
        oversized cell blows up.  That is (A) plus a working knob, and the old
        span rule called it (B)."""
        out = self._sum([3.19, 3.19, 3.19, 3.19, 3.19, 3.19, 6.46])
        assert out["reading"] == "A_path_independent"
        assert out["span_in_dist_nats"] < FLAT_SPAN_NATS
        assert out["span_nats"] > 3.0, "the full span must still be reported"
        assert out["ood_cells"] == ["noise@1000"]
        assert out["knob_live"] is True

    def test_a_sweep_that_never_moves_is_invalid_not_A(self):
        """RED 8/9/10's shape: a disconnected knob prints a perfectly flat
        table, and flat is the reading that sends the project into a
        redesign.  It must refuse instead."""
        out = self._sum([3.19] * 7)
        assert out["reading"] == "INVALID_knob_unproven"
        assert out["knob_live"] is False

    def test_a_responsive_sweep_reads_as_s0_live(self):
        """Loss climbing with the injected scale, IN DISTRIBUTION, is (B)."""
        out = self._sum([3.10, 3.11, 3.14, 3.30, 4.20, 9.00, 12.0])
        assert out["reading"] == "B_s0_live"
        assert out["delta_from_real"]["noise@100"] > 5.0

    def test_an_ood_cell_alone_cannot_produce_B(self):
        """The decisive one.  Only the swamping cell moves; every cell at a
        scale the model runs at is flat.  Drowning E inside
        adapter(cat([x, input_embeds])) is not evidence that the loop reads
        s0, and before RED 11 it was scored as exactly that."""
        out = self._sum([3.19, 3.19, 3.19, 3.19, 3.19, 3.19, 9.00])
        assert out["reading"] == "A_path_independent"
        assert out["delta_from_real"]["noise@1000"] > 5.0

    def test_the_swamp_window_falls_back_to_a_multiple_of_s0(self):
        """An old graft records no ||E||.  The fallback must still exclude the
        oversized cell, and must be conservative -- never WIDER than ||E||."""
        out = self._sum([3.19, 3.19, 3.19, 3.19, 3.19, 3.19, 6.46],
                        swamp_norm=None)
        assert out["swamp_limit_used"] == SWAMP_FALLBACK_MULT * self.S0
        assert out["swamp_limit_used"] < self.E_NORM
        assert out["reading"] == "A_path_independent"

    def test_the_intact_level_rides_along(self):
        """RED 10: a table of differences is unreadable without the level it is
        a difference of, so the verdict carries one."""
        at_chance = self._sum([11.6] * 7)
        assert at_chance["intact"]["at_chance"] is True
        healthy = self._sum([3.19] * 7)
        assert healthy["intact"]["at_chance"] is False

    def test_an_at_chance_model_vetoes_the_verdict(self):
        """RED 10 again, and this is the part gate 4 had to learn too: a
        difference of two chance-level losses still prints a tidy ordering.
        The level must REFUSE the verdict, not sit in a warning above it."""
        out = self._sum([11.60, 11.60, 11.60, 11.60, 11.60, 11.60, 14.0])
        assert out["reading"] == "INVALID_at_chance"
        assert out["knob_live"] is True, (
            "the knob was live; the veto has to come from the LEVEL, so that "
            "a fixed loading path turns this run into a real reading")

    def test_no_samples_does_not_invent_a_verdict(self):
        out = summarize({"real": []}, None, CHANCE, DEFAULT_SCALES)
        assert out["cells"] == {} and out["delta_from_real"] == {}
        assert out["intact"]["mean_nats"] is None
        assert out["reading"] == "INVALID_knob_unproven"


class TestTheSweepIsPaired:
    """Every cell must differ ONLY in what went into s0.  If the sweep also
    redrew s0's other columns, or re-primed the chain, the deltas would be
    measuring the instrument's own RNG -- which is how a null becomes a
    result."""

    def test_the_same_setting_scores_the_same_twice(self):
        m = _model()
        xs, ys = _chunks()
        num_steps = torch.tensor([T, 0])
        carry = prime(m, xs, num_steps, torch.device("cpu"), 5)
        a = score_chunk(m, xs[-1], ys[-1], carry, num_steps,
                        torch.device("cpu"), 77)
        b = score_chunk(m, xs[-1], ys[-1], carry, num_steps,
                        torch.device("cpu"), 77)
        assert a == b, ("two identical scored forwards disagreed, so s0's RNG "
                        "is not pinned and every delta in the sweep is noise")

    def test_changing_the_injection_changes_the_score(self):
        """The knob has to reach the model.  A latent_read_null that did
        nothing would print a perfectly flat table and read as (A) -- the
        reading that sends the project into a redesign."""
        m = _model()
        xs, ys = _chunks()
        num_steps = torch.tensor([T, 0])
        cortex = m.cortex
        cortex.latent_read_null = None
        carry = prime(m, xs, num_steps, torch.device("cpu"), 5)
        real = score_chunk(m, xs[-1], ys[-1], carry, num_steps,
                           torch.device("cpu"), 77)
        cortex.latent_read_null = ("noise", 50.0, 3)
        loud = score_chunk(m, xs[-1], ys[-1], carry, num_steps,
                           torch.device("cpu"), 77)
        cortex.latent_read_null = None
        assert abs(loud - real) > 1e-6, (
            "a 50.0-std substitution into s0 changed the loss by nothing, so "
            "latent_read_null never reached latent_init and the sweep cannot "
            "distinguish (A) from a broken knob")

    def test_an_e_only_arm_is_refused_not_reported(self):
        """latent_init returns s0 untouched without latent_carry, so the sweep
        would be flat BY CONSTRUCTION and read as (A)."""
        m = _model(latent=False)
        assert not getattr(m.cortex, "latent_carry", False)


class TestTheScaleAxisIsARowNorm:
    """RED 11.  The sweep's x-axis says "x ||s0||", ||s0|| is a per-token ROW
    NORM, and `_null_latent` takes a PER-ELEMENT std.  Nothing asserted the
    unit, so a sqrt(D) error rode through the whole instrument and flipped the
    verdict.  These pin the unit at both ends."""

    def test_null_latent_takes_an_elementwise_std(self):
        """The contract, stated as a number: std sigma -> rows of norm
        sigma*sqrt(D), NOT rows of norm sigma."""
        m = _model()
        head = torch.zeros(1, 5, D)
        sigma = 0.25
        z = m.cortex._null_latent(("noise", sigma, 3), head)
        row = float(z.float().norm(dim=-1).mean())
        assert abs(row - sigma * ROOT_D) < 0.15 * sigma * ROOT_D, (
            "expected rows at sigma*sqrt(D)=%.3f, got %.3f -- the unit of "
            "latent_read_null's std has changed and every caller that scales "
            "it by a NORM is now wrong" % (sigma * ROOT_D, row))

    def test_the_graft_records_both_scales_and_names_the_units(self):
        """`_z_s0_scale` is a row norm, `_z_s0_rms` is per-element, and they
        differ by sqrt(D).  A consumer that grabs the wrong one is RED 11."""
        m = _model()
        xs, _ = _chunks()
        prime(m, xs, torch.tensor([T, 0]), torch.device("cpu"), 5)
        row = getattr(m.cortex, "_z_s0_scale", None)
        rms = getattr(m.cortex, "_z_s0_rms", None)
        assert row and rms, ("the graft records neither scale, so the sweep "
                             "has nothing correct to scale by")
        assert abs(row - rms * ROOT_D) < 0.05 * row, (
            "||s0||=%.4f is not rms=%.4f x sqrt(D)=%.1f; one of the two is "
            "not the unit its name claims" % (row, rms, ROOT_D))

    def test_scaling_by_the_row_norm_overshoots_by_sqrt_d(self):
        """The bug, pinned as a property rather than as a story: if a caller
        ever again passes `_z_s0_scale` where `_z_s0_rms` belongs, the injected
        row is sqrt(D) too big.  At D=2048 that is 45x."""
        m = _model()
        xs, _ = _chunks()
        prime(m, xs, torch.tensor([T, 0]), torch.device("cpu"), 5)
        row = float(m.cortex._z_s0_scale)
        rms = float(m.cortex._z_s0_rms)
        head = torch.zeros(1, 5, D)
        right = float(m.cortex._null_latent(("noise", rms, 3), head)
                      .float().norm(dim=-1).mean())
        wrong = float(m.cortex._null_latent(("noise", row, 3), head)
                      .float().norm(dim=-1).mean())
        assert abs(right - row) < 0.15 * row, (
            "scaling by the rms must reproduce ||s0||: wanted %.4f, got %.4f"
            % (row, right))
        assert wrong > 4.0 * right, (
            "scaling by the row norm must visibly overshoot; it gave %.4f "
            "against %.4f" % (wrong, right))

    def test_the_e_carried_row_scale_is_recorded(self):
        """The swamp threshold the verdict now depends on.  It has to be
        measured on the forward, and it has to dwarf ||s0|| -- that gap is why
        an oversized s0 injection looks like a read when it is really E being
        drowned inside adapter(cat([x, input_embeds]))."""
        m = _model()
        xs, ys = _chunks()
        num_steps = torch.tensor([T, 0])
        carry = prime(m, xs, num_steps, torch.device("cpu"), 5)
        score_chunk(m, xs[-1], ys[-1], carry, num_steps, torch.device("cpu"), 7)
        e = getattr(m.cortex, "_e_carried_norm", None)
        assert e, ("prefix_pack spliced a carry but recorded no ||E||, so the "
                   "sweep cannot tell an in-distribution cell from a swamping "
                   "one and falls back to a guess")
        assert e > float(m.cortex._z_s0_scale), (
            "||E||=%.3f is not above ||s0||=%.3f; if that is real the swamp "
            "window is meaningless and the verdict rule needs rethinking"
            % (e, float(m.cortex._z_s0_scale)))
