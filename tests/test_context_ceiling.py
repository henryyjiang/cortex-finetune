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

The two decompositions added on top (2026-08-04) have their own failure modes:

  * carry_plus is only meaningful if `ctx` and `state` are genuinely
    independent -- if passing both silently dropped one, the residual would
    come out equal to either the carry delta or exactly zero, and both are
    readings the experiment is meant to distinguish;
  * the position bands must partition the chunk and must average only the
    tokens the mask keeps, or a padded final chunk turns a band into nan and
    the "where do the nats live" table reads as a hole in the middle.

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


class TestBuckets:
    """Position bands -- the 'where in the chunk do the nats live' table."""

    def test_bands_partition_the_chunk(self):
        ce = torch.arange(8, dtype=torch.float)
        mc = torch.ones(8)
        assert ecc._bucket_nll(ce, mc, 4) == [0.5, 2.5, 4.5, 6.5]

    def test_mask_excludes_pads_from_the_band_mean(self):
        ce = torch.tensor([1.0, 9.0, 1.0, 9.0])
        mc = torch.tensor([1.0, 0.0, 1.0, 0.0])
        assert ecc._bucket_nll(ce, mc, 2) == [1.0, 1.0]

    def test_fully_masked_band_is_none_not_nan(self):
        """A short final chunk must leave a hole, not a nan that poisons the
        pooled mean of that band across samples."""
        ce = torch.tensor([1.0, 2.0, 3.0, 4.0])
        mc = torch.tensor([1.0, 1.0, 0.0, 0.0])
        out = ecc._bucket_nll(ce, mc, 2)
        assert out[0] == 1.5 and out[1] is None

    def test_score_fills_buckets_and_keeps_the_scalar(self):
        m = _model(memory=False)
        x, y, ym = _sample()
        xc, yc, mc = x[:CL], y[:CL], ym[:CL]
        out = []
        v = ecc._score(m, None, xc, yc, mc, None, NS, DEV, 7, out, 4)
        assert v is not None and len(out) == 1 and len(out[0]) == 4
        # the bands average back to the scalar (uniform mask, equal-size bands)
        assert abs(sum(out[0]) / 4 - v) < 1e-4

    def test_buckets_are_opt_in(self):
        m = _model(memory=False)
        x, y, ym = _sample()
        a = ecc._score(m, None, x[:CL], y[:CL], ym[:CL], None, NS, DEV, 7)
        b = ecc._score(m, None, x[:CL], y[:CL], ym[:CL], None, NS, DEV, 7, [], 0)
        assert a == b


class TestCarryPlus:
    """carry + real context -- the residual denominator."""

    def test_context_and_carry_are_independent_inputs(self):
        """Passing both must differ from passing either alone; if one were
        silently dropped the residual would be a fixed artefact."""
        m = _model()
        x, y, ym = _sample()
        g = 2
        xc, yc, mc = x[g * CL:(g + 1) * CL], y[g * CL:(g + 1) * CL], ym[g * CL:(g + 1) * CL]
        st = ecc.carry_before(m, x, NC, g, NS, 7, DEV)
        ctx = x[g * CL - CL // 2:g * CL]
        both = ecc._score(m, ctx, xc, yc, mc, st, NS, DEV, 7)
        ctx_only = ecc._score(m, ctx, xc, yc, mc, None, NS, DEV, 7)
        carry_only = ecc._score(m, None, xc, yc, mc, st, NS, DEV, 7)
        assert both != ctx_only and both != carry_only

    def test_only_the_chunk_is_scored_with_both(self):
        """The logit slice must still drop the context columns when a carry is
        also present -- the carry adds rows the offset must NOT count."""
        m = _model()
        x, y, ym = _sample()
        g = 2
        xc, yc, mc = x[g * CL:(g + 1) * CL], y[g * CL:(g + 1) * CL], ym[g * CL:(g + 1) * CL]
        st = ecc.carry_before(m, x, NC, g, NS, 7, DEV)
        vals = [ecc._score(m, x[g * CL - k:g * CL] if k else None, xc, yc, mc,
                           st, NS, DEV, 7) for k in (0, CL // 2, CL)]
        assert all(v is not None and v == v for v in vals)
        assert max(vals) - min(vals) < 5.0


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


class TestDocSplit:
    """P0.3 -- within- vs across-document.

    The split is only free because every condition was appended in lockstep,
    so ONE boolean mask selects the same instances in all of them.  The two
    failure modes that would produce a plausible-looking but wrong table are a
    mask that desynchronises the paired lists (the delta then compares chunk i
    of one condition against chunk j of another) and a partition that is not a
    partition (an instance counted in both classes, or in neither, silently
    changes the share the mix ratio is set from).
    """

    EOS = 100257

    def test_clean_row_is_within_document(self):
        x = torch.arange(64) % 500                 # no separator anywhere
        assert ecc._doc_tags(x, 32, 16, self.EOS) == (False, False)

    def test_separator_in_the_oracle_window_is_local_and_prefix(self):
        x = torch.arange(64) % 500
        x[20] = self.EOS                           # inside the last 16 before 32
        assert ecc._doc_tags(x, 32, 16, self.EOS) == (True, True)

    def test_separator_before_the_window_is_prefix_only(self):
        """The distinction the two schemes exist for: the carry chain crossed a
        document but the oracle's k tokens did not, so the ceiling and the
        carry are not labelled by the same fact."""
        x = torch.arange(64) % 500
        x[4] = self.EOS                            # before off-kmax = 16
        assert ecc._doc_tags(x, 32, 16, self.EOS) == (False, True)

    def test_separator_after_the_boundary_is_neither(self):
        """A separator inside the chunk being SCORED says nothing about what
        crossed into it."""
        x = torch.arange(64) % 500
        x[40] = self.EOS
        assert ecc._doc_tags(x, 32, 16, self.EOS) == (False, False)

    def test_kmax_window_is_clamped_at_the_row_start(self):
        x = torch.arange(64) % 500
        x[0] = self.EOS
        assert ecc._doc_tags(x, 8, 1024, self.EOS) == (True, True)

    def _fixture(self):
        # 3 chunks; index 0 is unused (chunk 1 has no preceding context).
        per_chunk = {0: [[], [1.0, 2.0, 3.0], [4.0, 5.0]],
                     128: [[], [0.9, 1.9, 2.9], [3.9, 4.9]]}
        carry = [[], [0.5, 1.5, 2.5], [3.5, 4.5]]
        return per_chunk, carry, [0, 128], 3

    def test_unfiltered_pool_keeps_every_instance_in_order(self):
        per_chunk, carry, conds, nc = self._fixture()
        base, cond, cv = ecc._pool(per_chunk, carry, conds, nc, True)
        assert base == [1.0, 2.0, 3.0, 4.0, 5.0]
        assert cond[128] == [0.9, 1.9, 2.9, 3.9, 4.9]
        assert cv == [0.5, 1.5, 2.5, 3.5, 4.5]

    def test_mask_selects_the_same_instances_in_every_condition(self):
        """Pairing survives the filter: base[i], cond[i] and carry[i] must
        still be the same (sample, chunk)."""
        per_chunk, carry, conds, nc = self._fixture()
        keep = [[], [True, False, True], [False, True]]
        base, cond, cv = ecc._pool(per_chunk, carry, conds, nc, True, keep)
        assert base == [1.0, 3.0, 5.0]
        assert cond[128] == [0.9, 2.9, 4.9]
        assert cv == [0.5, 2.5, 4.5]

    def test_within_and_across_partition_the_instances(self):
        per_chunk, carry, conds, nc = self._fixture()
        tags = [[], [(False, False), (True, True), (False, True)],
                [(True, True), (False, False)]]
        for ti in (0, 1):
            sizes = []
            for want in (False, True):
                keep = [[t[ti] == want for t in tags[g]] for g in range(nc)]
                sizes.append(len(ecc._pool(per_chunk, carry, conds, nc,
                                           True, keep)[0]))
            assert sum(sizes) == 5

    def test_no_carry_model_pools_none(self):
        per_chunk, carry, conds, nc = self._fixture()
        assert ecc._pool(per_chunk, carry, conds, nc, False)[2] is None
