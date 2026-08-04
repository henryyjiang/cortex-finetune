"""
evals/eval_context_ceiling.py against a real tiny grafted raven model.

The oracle's whole job is to put the carry delta on a scale, so the things
worth pinning are the ones that would silently produce a plausible-looking
scale:

  * the k real tokens must be PRECEDING tokens, contiguous with the chunk, and
    only the chunk's own positions may be scored -- an off-by-n_ctx in the
    logit slice scores the context instead and every ceiling comes out huge;
  * k=0 must equal the no-carry condition exactly (same forward), since every
    delta is measured against it;
  * the carry condition must be the state after chunks 1..g-1, i.e. the carry
    ablation's "carried", or the two instruments stop being comparable;
  * _equiv_tokens must refuse to extrapolate -- a carry above every measured
    ceiling has no equivalent, and returning the largest k would read as a
    real measurement.

Run: /c/Users/henry/miniconda3/envs/cortex-retro/python.exe -m pytest tests/ -q
"""
from __future__ import annotations

import os
import sys

import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "evals"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from test_cortex_eval import VOCAB, _build_raven  # noqa: E402

import eval_context_ceiling as ecc  # noqa: E402

NV, CL, NC = 4, 16, 4
EOS = VOCAB - 1
DEV = torch.device("cpu")
NS = torch.tensor([2, 0])


def _model(memory=True):
    torch.manual_seed(1234)
    kw = dict(use_memory=True, memory_slots=0, prefix_memory="accum",
              accum_vecs=NV, accum_max=64, eos_token_id=EOS) if memory \
        else dict(use_memory=False, memory_slots=0)
    return _build_raven(**kw).eval()


def _sample():
    torch.manual_seed(0)
    ids = torch.randint(0, VOCAB - 1, (CL * NC + 1,))
    return ids[:-1], ids[1:], torch.ones(CL * NC)


class TestEquivTokens:
    """Pure function -- the number that goes on a slide."""

    def test_interpolates_between_measured_points(self):
        eq = ecc._equiv_tokens([(100, 0.01), (200, 0.03)], 0.02)
        assert 100 < eq < 200 and abs(eq - 150) < 1e-6

    def test_returns_the_first_k_that_reaches_it(self):
        assert ecc._equiv_tokens([(100, 0.05), (200, 0.09)], 0.05) == 100

    def test_refuses_to_extrapolate_past_the_measured_range(self):
        """A carry better than every oracle has no equivalent; reporting the
        largest k would look like a measurement."""
        assert ecc._equiv_tokens([(100, 0.01), (200, 0.02)], 0.5) is None

    def test_non_positive_carry_has_no_equivalent(self):
        assert ecc._equiv_tokens([(100, 0.01)], 0.0) is None
        assert ecc._equiv_tokens([(100, 0.01)], -0.01) is None


class TestScoring:

    def test_k0_equals_no_context(self):
        m = _model(memory=False)
        x, y, ym = _sample()
        xc, yc, mc = x[:CL], y[:CL], ym[:CL]
        a = ecc._score(m, None, xc, yc, mc, None, NS, DEV, 7)
        b = ecc._score(m, torch.zeros(0, dtype=torch.long), xc, yc, mc, None,
                       NS, DEV, 7)
        assert a == b

    def test_context_changes_the_score(self):
        m = _model(memory=False)
        x, y, ym = _sample()
        xc, yc, mc = x[CL:2 * CL], y[CL:2 * CL], ym[CL:2 * CL]
        bare = ecc._score(m, None, xc, yc, mc, None, NS, DEV, 7)
        ctx = ecc._score(m, x[:CL], xc, yc, mc, None, NS, DEV, 7)
        assert bare != ctx

    def test_only_the_chunk_is_scored(self):
        """The logit slice must drop the context, so the loss stays a per-token
        mean over the CHUNK regardless of how much context was prepended."""
        m = _model(memory=False)
        x, y, ym = _sample()
        xc, yc, mc = x[2 * CL:3 * CL], y[2 * CL:3 * CL], ym[2 * CL:3 * CL]
        vals = [ecc._score(m, x[2 * CL - k:2 * CL] if k else None, xc, yc, mc,
                           None, NS, DEV, 7) for k in (0, CL, 2 * CL)]
        # all finite, all in the same ballpark -- a slice bug makes one wild
        assert all(v is not None and v == v for v in vals)
        assert max(vals) - min(vals) < 5.0

    def test_seed_makes_conditions_paired(self):
        m = _model(memory=False)
        x, y, ym = _sample()
        xc, yc, mc = x[:CL], y[:CL], ym[:CL]
        a = ecc._score(m, None, xc, yc, mc, None, NS, DEV, 7)
        b = ecc._score(m, None, xc, yc, mc, None, NS, DEV, 7)
        assert a == b


class TestCarryBefore:

    def test_state_is_from_the_preceding_chunks_only(self):
        """chunk g's carry must hold exactly g-1 chunks' writes -- one too many
        and it leaks the chunk being scored, which is a different experiment."""
        m = _model()
        x, _, _ = _sample()
        for g in (1, 2, 3):
            st = ecc.carry_before(m, x, NC, g, NS, 7, DEV)
            assert st.shape[1] == g * NV

    def test_none_before_the_first_chunk(self):
        m = _model()
        x, _, _ = _sample()
        assert ecc.carry_before(m, x, NC, 0, NS, 7, DEV) is None

    def test_no_cross_state_yields_none(self):
        m = _model(memory=False)
        x, _, _ = _sample()
        assert ecc.carry_before(m, x, NC, 2, NS, 7, DEV) is None
