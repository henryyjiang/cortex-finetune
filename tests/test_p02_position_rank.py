"""
P0.2 probe tests — `evals/diag_position_rank.py`.

The property that makes these worth having is the same one P0.1's tests pin: a
probe that silently measures NOTHING looks exactly like a probe that measures a
null result.  `diag_position_rank` reaches into the model through two wrappers
(`prefix_pack` for the positions, `prefix_unpack` for the columns), and if
either stops firing — the splice moves, the graft is renamed, a checkpoint
ships a modeling file where the pack is inlined — the script would report an
empty sweep rather than crash.  So:

  * the layout arithmetic is pinned exactly, because an off-by-one in
    `tail` is the difference between the production layout and a neighbour;
  * the rank statistic is pinned on stacks whose answer is known by
    construction (orthonormal, identical, two clusters);
  * the degenerate case is pinned, because dividing dust by its own norm
    produces a confident 0.0 cosine for a perfectly collapsed stack — the
    opposite of the truth;
  * and the hook test asserts the captured positions ARE the requested layout,
    which fails if the wrapper stops being called at all.

Run: /c/Users/henry/miniconda3/envs/cortex-retro/python.exe -m pytest tests/test_p02_position_rank.py -v
"""
from __future__ import annotations

import os
import sys

import pytest
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "evals"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from evals.diag_position_rank import (  # noqa: E402
    READ_LAYOUTS, WRITE_LAYOUTS, layout_positions, rank_stats, run_chunk)
from evals.model_utils import _unwrap  # noqa: E402


def _pos(layout, n, S, last_real):
    return layout_positions(layout, n, S, last_real,
                            torch.device("cpu"), torch.long).tolist()


# ---------------------------------------------------------------------------
# Layout arithmetic
# ---------------------------------------------------------------------------

class TestLayouts:

    def test_zero_puts_every_slot_at_position_zero(self):
        assert _pos("zero", 4, 16, 16) == [0, 0, 0, 0]

    def test_tail_continues_the_chunk_numbering(self):
        """prefix_pack derives the slot positions from the LAST REAL position,
        not from S, so a cached prefill that does not start at 0 stays
        consistent.  This mirrors that derivation exactly."""
        assert _pos("tail", 4, 16, 16) == [17, 18, 19, 20]
        assert _pos("tail", 3, 16, 99) == [100, 101, 102]

    def test_contiguous_is_the_autocompressor_llama_layout(self):
        assert _pos("contiguous", 4, 16, 16) == [0, 1, 2, 3]

    def test_spread_spans_the_real_tokens_inclusive(self):
        p = _pos("spread", 4, 16, 16)
        assert p[0] == 0 and p[-1] == 15
        assert p == sorted(p)
        assert len(set(p)) == 4

    def test_spread_repeats_when_K_exceeds_the_span(self):
        """Integer positions cannot spread 32 slots over 8 columns.  That is a
        real property of the layout at large K and the probe must not hide it,
        because it is exactly the regime where widening K stops buying rank."""
        p = _pos("spread", 32, 8, 8)
        assert len(p) == 32
        assert len(set(p)) < 32
        assert min(p) == 0 and max(p) == 7

    def test_every_advertised_layout_is_implemented(self):
        for lay in set(WRITE_LAYOUTS) | set(READ_LAYOUTS):
            assert len(_pos(lay, 4, 16, 16)) == 4

    def test_an_unknown_layout_raises_rather_than_falling_back(self):
        with pytest.raises(ValueError):
            _pos("middle", 4, 16, 16)


# ---------------------------------------------------------------------------
# Rank statistics
# ---------------------------------------------------------------------------

class TestRankStats:

    def test_orthonormal_slots_reach_the_ceiling(self):
        """K orthonormal rows, centred, span K-1 dimensions -- centring removes
        one degree of freedom.  So the ceiling this probe can report for a
        maximally spread stack of K slots is K-1, not K, and a layout scoring
        near it is saturating rather than underperforming."""
        K, D = 8, 32
        m = torch.eye(K, D).unsqueeze(0)
        r = rank_stats(m)
        assert r["K"] == K
        assert r["eff_rank_entropy"] > K - 1.5
        assert r["centred_norm_ratio"] > 0.5

    def test_identical_slots_report_the_degenerate_limit(self):
        """A perfectly collapsed stack centres to numerical dust.  Normalising
        dust by its own norm yields an arbitrary direction and a cosine near
        0.0 -- a confident wrong answer that reads as 'uncorrelated'.  The
        guard must report one direction, fully aligned, instead."""
        m = torch.ones(1, 8, 32)
        r = rank_stats(m)
        assert r["centred_norm_ratio"] < 1e-6
        assert r["eff_rank_entropy"] == pytest.approx(1.0)
        assert r["eff_rank_pr"] == pytest.approx(1.0)
        assert r["centred_cosine"] == pytest.approx(1.0)

    def test_two_clusters_resolve_to_about_two_directions(self):
        K, D = 8, 32
        a, b = torch.zeros(D), torch.zeros(D)
        a[0], b[1] = 1.0, 1.0
        m = torch.stack([a] * 4 + [b] * 4).unsqueeze(0)
        r = rank_stats(m)
        assert 1.0 <= r["eff_rank_entropy"] <= 2.5

    def test_a_large_shared_component_does_not_inflate_the_cosine(self):
        """The reason the statistic is centred.  Eight orthogonal directions
        sitting on a big common offset are eight directions; an uncentred
        cosine would call them one."""
        K, D = 8, 32
        m = torch.eye(K, D) + 50.0
        r = rank_stats(m.unsqueeze(0))
        assert r["eff_rank_entropy"] > K - 1.5
        assert r["centred_cosine"] < 0.1

    def test_batch_lanes_are_averaged_not_concatenated(self):
        m = torch.eye(8, 32).unsqueeze(0).repeat(3, 1, 1)
        r1 = rank_stats(m[:1])
        r3 = rank_stats(m)
        assert r3["K"] == 8
        assert r3["eff_rank_entropy"] == pytest.approx(
            r1["eff_rank_entropy"], rel=1e-4)


# ---------------------------------------------------------------------------
# The hooks -- these fail if the splice stops firing
# ---------------------------------------------------------------------------

def _prefix_model(n_vec=4):
    """A tiny REAL raven with the prefix graft.

    eos_token_id is not optional: the prefix buffer seeds summary_emb from a
    real token embedding (AutoCompressor uses EOS) and refuses to build without
    one.  _build_raven reports that refusal as "cannot build raven base
    (transformers skew?)", so a missing eos_token_id here looks exactly like a
    library version problem and silently skips every hook test below."""
    from test_cortex_eval import VOCAB, _build_raven
    return _build_raven(use_memory=True, memory_slots=0,
                        prefix_memory="accum", accum_vecs=n_vec,
                        accum_max=32, h_T_proj=True,
                        eos_token_id=VOCAB - 1)


class TestHooks:

    @pytest.mark.parametrize("layout", WRITE_LAYOUTS)
    def test_write_positions_are_the_requested_layout(self, layout):
        """The load-bearing test.  If prefix_pack is renamed, inlined into the
        modeling file, or simply not called, `captured` comes back empty and
        this fails -- instead of the probe reporting a tidy sweep of nothing."""
        model = _prefix_model(n_vec=4)
        inner = _unwrap(model)
        S = 16
        ids = torch.randint(0, 200, (1, S))
        cap, out = run_chunk(model, inner, ids, num_steps=2, m_cross_in=None,
                             write_layout=layout, read_layout="zero")
        assert "write_cols" in cap, "prefix_unpack hook never fired"
        assert cap["n_sum"] == 4 and cap["n_pre"] == 0
        assert cap["write_cols"].shape[1] == 4
        got = cap["pos"][cap["n_pre"] + cap["S"]:]
        assert got == _pos(layout, 4, S, cap["pos"][cap["n_pre"] + S - 1])

    def test_read_side_sees_a_populated_carry_on_chunk_two(self):
        model = _prefix_model(n_vec=4)
        inner = _unwrap(model)
        S = 16
        c1 = torch.randint(0, 200, (1, S))
        c2 = torch.randint(0, 200, (1, S))
        _, out1 = run_chunk(model, inner, c1, 2, None, "tail", "zero")
        carry = getattr(out1, "m_cross", None)
        assert carry is not None and carry.shape[1] == 4
        cap, _ = run_chunk(model, inner, c2, 2, carry, "tail", "contiguous")
        assert "read_cols" in cap, "the carried columns were never captured"
        assert cap["n_pre"] == 4
        assert cap["pos"][:4] == [0, 1, 2, 3]

    def test_the_wrappers_leave_no_instance_shadow(self):
        """Restoring by assigning the bound method back would leave a permanent
        entry in the instance __dict__ that shadows the class method forever --
        so a later probe patching the class would be silently overridden, and
        the model would hold a reference cycle to its own bound method."""
        model = _prefix_model(n_vec=4)
        inner = _unwrap(model)
        cortex = inner.cortex
        assert "prefix_pack" not in vars(cortex)
        run_chunk(model, inner, torch.randint(0, 200, (1, 16)), 2, None,
                  "tail", "zero")
        assert "prefix_pack" not in vars(cortex), "wrapper left behind"
        assert "prefix_unpack" not in vars(cortex), "wrapper left behind"

    def test_a_model_without_a_prefix_buffer_fails_loudly(self):
        from test_cortex_eval import _build_raven
        model = _build_raven(use_memory=False)
        inner = _unwrap(model)
        if getattr(inner, "cortex", None) is None:
            with pytest.raises(RuntimeError, match="prefix_pack"):
                run_chunk(model, inner, torch.randint(0, 200, (1, 16)), 2,
                          None, "tail", "zero")
