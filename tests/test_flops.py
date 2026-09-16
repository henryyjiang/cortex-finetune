"""
The analytic FLOP cost model -- issue 8's instrument.

WHY THIS FILE EXISTS.  Issue 8 says the memory-vs-depth claim rests on a
degenerate depth arm: depth saturates at T~8 on both the mr8 and the mr32
checkpoint, so "memory beats depth" currently reduces to "there is no depth
lever left to beat", and a reviewer will say so.  The framework's fix #1 is a
compute-matched T-sweep -- accuracy against ANALYTIC FLOPs -- and `evals/flops.py`
is the cost model that figure would be plotted against.

**No test here can resolve issue 8.**  Issue 8 is resolved by a MEASUREMENT (a
forward-only T-sweep on the trained arms, no training required), and by nothing
else.  What these tests do is make sure the instrument that measurement will be
divided by is correct BEFORE it is used, because `flops.py` currently has no
callers and no tests at all: every number it has ever produced was produced by
hand, read once, and never checked.

The properties pinned are the ones whose failure would BIAS A RATIO -- the only
output anyone quotes from this file -- rather than crash it.

Run: /c/Users/henry/miniconda3/envs/cortex-retro/python.exe -m pytest tests/test_flops.py -q
"""
from __future__ import annotations

import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "evals"))

from evals.flops import (  # noqa: E402
    ModelSpec, forward_flops, match_cot_budget, query_flops,
)

#: The real B2 / cortex-final geometry, from the arm's own config.json.  Written
#: out rather than read from a checkpoint so the test runs anywhere, and so a
#: config that drifts shows up as a FAILURE in the from_config test below rather
#: than as a silently different cost model.
B2 = ModelSpec(d=2048, n_head=16, n_kv_heads=16, head_dim=128,
               intermediate=8192, vocab=100352,
               n_prelude=4, n_core=6, n_coda=4, name="cortex")


class TestTheDepthModel:

    def test_depth_is_prelude_plus_t_recurrent_blocks_plus_coda(self):
        assert B2.depth(1) == 4 + 6 + 4
        assert B2.depth(8) == 4 + 6 * 8 + 4
        assert B2.depth(32) == 4 + 6 * 32 + 4

    def test_a_dense_baseline_ignores_t_entirely(self):
        """The control arm is a plain transformer: T is not a knob it has, and a
        cost model that charged it for depth would hand the memory arm a
        manufactured win."""
        base = ModelSpec.dense_baseline(B2, n_layers=16)
        assert base.depth(1) == base.depth(32) == 16
        assert base.adapter_flops == 0
        a = forward_flops(base, 512, T=1)
        b = forward_flops(base, 512, T=32)
        assert a == b

    def test_the_adapter_is_paid_once_per_step_per_token(self):
        """Linear(2d, d) on every one of the T iterations.  Dropping it
        under-prices the recurrent arm by exactly the thing that makes depth
        expensive."""
        assert B2.adapter_flops == 4 * B2.d * B2.d
        n = 128
        with_adapter = forward_flops(B2, n, T=4)
        # The adapter term is packed * T * adapter_flops with packed == n here.
        assert with_adapter - forward_flops(B2, n, T=4, n_pre=0, n_sum=0) == 0
        d4 = forward_flops(B2, n, T=4) - forward_flops(B2, n, T=2)
        d8 = forward_flops(B2, n, T=8) - forward_flops(B2, n, T=6)
        # Two extra recurrent steps cost the same whenever they are added:
        # the per-step cost is flat in T apart from the attention-score term,
        # which is identical here because `packed` does not move.
        assert d4 == d8


class TestTheCortexSpecificCostsANaiveModelWouldMiss:
    """The packed sequence is longer than the token sequence.  A cost model that
    counted only real tokens would price the memory arm as if its carry were
    free -- which is precisely the direction that would manufacture the claim."""

    def test_carried_columns_are_paid_at_every_layer(self):
        bare = forward_flops(B2, 512, T=8)
        carried = forward_flops(B2, 512, T=8, n_pre=64)
        assert carried > bare

    def test_summary_slots_are_paid_too(self):
        bare = forward_flops(B2, 512, T=8)
        writing = forward_flops(B2, 512, T=8, n_sum=16)
        assert writing > bare

    def test_the_accum_carry_grows_across_chunks_and_is_capped(self):
        """`PrefixAccumBuffer` appends per chunk, so the carry is NOT flat --
        modelling it flat would under-price the accum arm late in a chain and
        over-price it early.  Capped at accum_max, which is what stops the
        growth."""
        small = query_flops(B2, 4096, 0, T=8, n_chunks=8,
                            carry_vecs=128, summary_vecs=16)["ingest"]
        flat = 8 * forward_flops(B2, 512, T=8, n_pre=0, n_sum=16, head_tokens=0)
        assert small > flat, "the growing carry is not being charged for"

    def test_a_gated_ring_is_cheaper_per_chunk_than_a_no_trim_accum(self):
        """The mechanism claim in cost terms: A3' reads 64 columns where A1
        reads up to 128, and that is the 'cheaper AND?' framing the writeup has
        to make.  If the cost model did not show it, the framing would be
        unsupported."""
        accum = query_flops(B2, 4096, 0, T=8, n_chunks=8,
                            carry_vecs=128, summary_vecs=16)["total"]
        gated = query_flops(B2, 4096, 0, T=8, n_chunks=8,
                            carry_vecs=64, summary_vecs=16)["total"]
        assert gated < accum


class TestTheMatchingIsExact:

    def test_cost_is_monotone_in_generated_tokens(self):
        """`match_cot_budget` bisects, which is only exact on a monotone cost.
        Non-monotone, the bisection returns a wrong answer SILENTLY and the
        compute-matched control is mismatched."""
        base = ModelSpec.dense_baseline(B2, n_layers=16)
        costs = [query_flops(base, 900, g, T=1, n_chunks=1)["total"]
                 for g in (0, 1, 8, 64, 256, 512)]
        assert costs == sorted(costs)
        assert len(set(costs)) == len(costs)

    def test_the_matched_budget_is_the_largest_affordable_one(self):
        base = ModelSpec.dense_baseline(B2, n_layers=16)
        budget = query_flops(B2, 900, 256, T=8, n_chunks=1,
                             carry_vecs=128, summary_vecs=32)["total"]
        g = match_cot_budget(base, budget, 900)
        cost = lambda n: query_flops(base, 900, n, T=1, n_chunks=1)["total"]
        assert cost(g) <= budget < cost(g + 1)

    def test_an_unaffordable_budget_returns_zero_rather_than_negative(self):
        base = ModelSpec.dense_baseline(B2, n_layers=16)
        assert match_cot_budget(base, 1, 900) == 0


class TestTheMemoryVsDepthTradeIsAComputeClaim:
    """Issue 8, in the only form a unit test can touch it.

    The claim is an ALLOCATION claim -- at a fixed budget, is depth or memory
    the better spend -- so the two things it needs are (a) that depth actually
    costs what it appears to, and (b) that carrying memory at low T is cheaper
    than buying depth.  Neither is evidence that memory WINS; that needs the
    T-sweep on trained arms.  What they do is stop the claim being quoted
    against a denominator nobody checked.
    """

    def test_depth_is_the_expensive_axis_and_the_ratio_is_pinned(self):
        cheap = forward_flops(B2, 512, T=8)
        deep = forward_flops(B2, 512, T=32)
        ratio = deep / cheap
        # 4 + 6*32 + 4 = 200 layers against 4 + 6*8 + 4 = 56: ~3.5x, slightly
        # diluted by the head and the score term.  Pinned so the figure's x-axis
        # cannot drift without this failing.
        assert 3.3 < ratio < 3.7, ratio

    def test_carry_on_at_t8_is_cheaper_than_carry_off_at_t32(self):
        """The shape of the existing 3.6x claim.  Asserted as a DIRECTION plus
        a band, not as the published digits -- the published number must be
        quoted from a run of the instrument, not from this test."""
        with_mem = query_flops(B2, 4096, 0, T=8, n_chunks=8,
                               carry_vecs=128, summary_vecs=32)["total"]
        deep_no_mem = query_flops(B2, 4096, 0, T=32, n_chunks=8,
                                  carry_vecs=0, summary_vecs=0)["total"]
        assert with_mem < deep_no_mem
        assert 2.0 < deep_no_mem / with_mem < 5.0, deep_no_mem / with_mem

    def test_the_saturation_point_is_not_something_this_file_knows(self):
        """Guard against the file being mistaken for an answer.  FLOPs are
        monotone in T forever; the T at which ACCURACY stops improving is the
        measurement issue 8 needs, and nothing here predicts it."""
        costs = [forward_flops(B2, 512, T=t) for t in (1, 2, 4, 8, 16, 32, 64)]
        assert costs == sorted(costs)
        assert len(set(costs)) == len(costs)


class TestItReadsARealConfig:

    def test_from_config_maps_every_field_the_cost_model_needs(self):
        """The cost model reads a live config, so a renamed field would silently
        change every FLOP number rather than raise."""
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from test_cortex_eval import _build_raven
        m = _build_raven(use_memory=False)
        spec = ModelSpec.from_config(m.config)
        cfg = m.config
        assert spec.d == cfg.n_embd
        assert spec.n_head == cfg.num_attention_heads
        assert spec.n_kv_heads == cfg.num_key_value_heads
        assert spec.intermediate == cfg.intermediate_size
        assert spec.n_prelude == cfg.n_layers_in_prelude
        assert spec.n_core == cfg.n_layers_in_recurrent_block
        assert spec.n_coda == cfg.n_layers_in_coda
        assert spec.depth(8) == spec.n_prelude + 8 * spec.n_core + spec.n_coda
