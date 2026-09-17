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
    DEFAULT_SCALES, FLAT_SPAN_NATS, prime, score_chunk, summarize,
)

CHANCE = math.log(VOCAB)
NV, K, CL, EOS = 4, 16, 16, VOCAB - 1
NC, T = 4, 4


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

    def _cells(self, values):
        cells = {"real": [values[0]]}
        for s, v in zip(DEFAULT_SCALES, values[1:]):
            cells["noise@%g" % s] = [v]
        return cells

    def test_a_flat_sweep_reads_as_path_independent(self):
        out = summarize(self._cells([3.10] * 6), 0.63, CHANCE, DEFAULT_SCALES)
        assert out["reading"] == "A_path_independent"
        assert out["span_nats"] < FLAT_SPAN_NATS

    def test_a_responsive_sweep_reads_as_s0_live(self):
        """Loss climbing with the injected scale is the (B) picture."""
        out = summarize(self._cells([3.10, 3.11, 3.14, 3.30, 4.20, 9.00]),
                        0.63, CHANCE, DEFAULT_SCALES)
        assert out["reading"] == "B_s0_live"
        assert out["delta_from_real"]["noise@100"] > 5.0

    def test_the_intact_level_rides_along(self):
        """RED 10: a table of differences is unreadable without the level it is
        a difference of, so the verdict carries one."""
        at_chance = summarize(self._cells([11.6] * 6), 0.63, CHANCE,
                              DEFAULT_SCALES)
        assert at_chance["intact"]["at_chance"] is True
        healthy = summarize(self._cells([3.19] * 6), 0.63, CHANCE,
                            DEFAULT_SCALES)
        assert healthy["intact"]["at_chance"] is False

    def test_no_samples_does_not_invent_a_verdict(self):
        out = summarize({"real": []}, None, CHANCE, DEFAULT_SCALES)
        assert out["cells"] == {} and out["delta_from_real"] == {}
        assert out["intact"]["mean_nats"] is None


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
