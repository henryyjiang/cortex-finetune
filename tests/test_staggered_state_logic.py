"""J7's write logic, with no Raven base — so it runs OFF the cluster.

`tests/test_staggered_state.py` checks the mechanism through a real packed
forward, which needs the cluster's transformers and SKIPS on a dev box.  The
three pieces of arithmetic J7 adds are pure, so they are pinned here too, where
they actually execute:

  * the slot -> depth map and its clamp,
  * `_latent_staggered_write`'s column selection,
  * `latent_write_grad_frac`'s PER-SLOT count, which is the silent-wrong-number
    site: the branch it replaces returns 1.0 whenever the last loop iteration is
    trainable, which is always, over a staggered band that may be entirely
    inside the no-grad prefix.

The methods are bound to a stub carrying only the attributes they read.  That is
deliberate: it is what lets this file run with no model, and it fails loudly if
one of them starts reaching for graft state it did not use before.

Run: python -m pytest tests/test_staggered_state_logic.py -q
"""
from __future__ import annotations

import os
import sys

import pytest
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from cortex_graft import CortexMemory  # noqa: E402
from cortex_memory.latent_embed import LatentEmbedRead  # noqa: E402


class _Prefix:
    def __init__(self, n_vec):
        self.n_vec = n_vec


class _Stub:
    """Only what J7's three methods read."""

    def __init__(self, encoding="staggered_state", lo=2, hi=9, rule="absolute",
                 n_vec=16):
        self.latent_encoding = encoding
        self.latent_depth_lo, self.latent_depth_hi = lo, hi
        self.latent_depth_rule = rule
        self.prefix = _Prefix(n_vec)
        self._z_depths_used = []
        self._z_grad = []

    depth_map = CortexMemory.latent_depth_map
    staggered = CortexMemory._latent_staggered_write
    # a property on the class: fetch the underlying function to call by hand
    grad_frac = CortexMemory.latent_write_grad_frac.fget

    def latent_depth_map(self, T, n_slots):
        return _Stub.depth_map(self, T, n_slots)


def _tape(T, n_vec=16, D=4):
    """Tape entry k-1 is filled with the value k, so a drawn depth is readable
    straight off the returned tensor."""
    return [torch.full((2, n_vec, D), float(k)) for k in range(1, T + 1)]


class TestTheDepthMap:

    def test_absolute_cycles_the_band(self):
        s = _Stub(lo=2, hi=4, n_vec=6)
        assert s.latent_depth_map(9, 6) == [2, 3, 4, 2, 3, 4]

    def test_a_short_loop_clamps_the_top_of_the_band(self):
        s = _Stub(lo=2, hi=9)
        got = s.latent_depth_map(5, 16)          # T-1 == 4
        assert max(got) == 4 and min(got) == 2

    def test_a_loop_shorter_than_the_band_floor_collapses_to_a_point(self):
        """hi_c = min(hi, T-1) then lo_c = min(lo, hi_c): at T=2 the band is the
        single depth 1, and n_depths == 1 is what the gate must catch."""
        s = _Stub(lo=2, hi=9)
        assert set(s.latent_depth_map(2, 16)) == {1}

    def test_T_below_two_is_all_depth_one(self):
        s = _Stub()
        assert s.latent_depth_map(1, 16) == [1] * 16


class TestTheStaggeredWrite:

    def test_slot_j_takes_the_depth_the_map_asked_for(self):
        s = _Stub(lo=2, hi=4, n_vec=6)
        out = s.staggered(_tape(9, n_vec=6))
        assert [int(out[0, j, 0]) for j in range(6)] == [2, 3, 4, 2, 3, 4]

    def test_slot_j_takes_column_j(self):
        s = _Stub(lo=1, hi=3, n_vec=4)
        tape = _tape(6, n_vec=4)
        for k, t in enumerate(tape, start=1):
            for j in range(4):
                t[:, j, :] = k * 10 + j
        out = s.staggered(tape)
        want = [k * 10 + j for j, k in enumerate(s.latent_depth_map(6, 4))]
        assert [int(out[0, j, 0]) for j in range(4)] == want

    def test_the_shape_is_one_row_per_slot(self):
        s = _Stub(n_vec=16)
        out = s.staggered(_tape(8))
        assert out.shape == (2, 16, 4)

    def test_it_records_the_depths_it_drew(self):
        s = _Stub(lo=2, hi=4, n_vec=6)
        s.staggered(_tape(9, n_vec=6))
        assert s._z_depths_used == [2, 3, 4]

    def test_it_never_indexes_past_the_tape(self):
        s = _Stub(lo=2, hi=9, n_vec=16)
        s.staggered(_tape(3))                     # would ask for 2..9
        assert max(s._z_depths_used) <= 3

    def test_a_delta_tape_goes_through_the_same_code(self):
        """'delta' and 'staggered_state' must not drift apart in the slot ->
        depth convention; they share this method for that reason."""
        a, b = _Stub(encoding="delta", lo=2, hi=4, n_vec=6), _Stub(lo=2, hi=4, n_vec=6)
        ta, tb = _tape(9, n_vec=6), _tape(9, n_vec=6)
        torch.testing.assert_close(a.staggered(ta), b.staggered(tb))


class TestTheGradFraction:
    """The branch that exists because the old one lies about a staggered write."""

    def test_all_trainable_is_one(self):
        s = _Stub(lo=1, hi=9)
        s._z_grad = [True] * 8
        assert s.grad_frac() == pytest.approx(1.0)

    def test_a_band_wholly_inside_the_no_grad_prefix_is_zero(self):
        s = _Stub(lo=1, hi=2, n_vec=16)
        s._z_grad = [False] * 6 + [True] * 2      # depths 1..6 frozen
        assert s.grad_frac() == pytest.approx(0.0)

    def test_the_old_shortcut_would_have_said_one_on_that_case(self):
        """Pinning the defect, not just the fix: with `endpoint` the same
        _z_grad returns 1.0, because endpoint draws the last (trainable) step.
        If these two ever agree, the per-slot branch stopped running."""
        frozen = _Stub(encoding="endpoint")
        frozen._z_grad = [False] * 6 + [True] * 2
        assert frozen.grad_frac() == pytest.approx(1.0)
        stag = _Stub(lo=1, hi=2, n_vec=16)
        stag._z_grad = [False] * 6 + [True] * 2
        assert stag.grad_frac() == pytest.approx(0.0)

    def test_a_partly_frozen_band_lands_between(self):
        """T=8, band 1..8 clamps to 1..7, so 8 slots draw [1,2,3,4,5,6,7,1].
        With depths 1-4 frozen, the live slots are the ones at 5, 6 and 7."""
        s = _Stub(lo=1, hi=8, n_vec=8)
        s._z_grad = [False] * 4 + [True] * 4      # depths 1-4 frozen, 5-8 live
        assert s.grad_frac() == pytest.approx(3 / 8)

    def test_it_counts_slots_not_tape_entries(self):
        """16 slots over a 7-deep band double-cover the low depths, so one live
        depth can supply TWO live slots -- the slot count is what describes the
        write, and it is not the tape count."""
        s = _Stub(lo=1, hi=8, n_vec=16)
        s._z_grad = [False] * 6 + [True, False]   # only depth 7 is live
        assert s.grad_frac() == pytest.approx(2 / 16)

    def test_the_last_tape_entry_is_never_drawn(self):
        """`hi_c = min(hi, T - 1)`, so the final entry is out of reach however
        wide the band is asked to be.  For 'delta' that was deliberate -- d_T
        sits at the saturated end P0.1 says carries least.  For a raw-state
        write it excludes s_T, which is precisely E's pre-coda twin, so the
        clamp happens to remove the one depth this arm does NOT want.  Pinned
        because it is load-bearing by accident, and a future widening of the
        band would silently put the twin back in."""
        s = _Stub(lo=1, hi=99, n_vec=16)
        s.staggered(_tape(8))
        assert max(s._z_depths_used) == 7
        live_only_at_the_end = _Stub(lo=1, hi=99, n_vec=16)
        live_only_at_the_end._z_grad = [False] * 7 + [True]
        assert live_only_at_the_end.grad_frac() == pytest.approx(0.0)

    def test_an_empty_tape_is_zero_not_a_crash(self):
        s = _Stub()
        assert s.grad_frac() == pytest.approx(0.0)

    def test_delta_still_averages_the_whole_tape(self):
        s = _Stub(encoding="delta")
        s._z_grad = [False, False, True, True]
        assert s.grad_frac() == pytest.approx(0.5)


class TestTheFrozenIdentityProjection:
    """LatentEmbedRead is a plain nn.Module, so the AC claim is checkable here."""

    def test_identity_init_is_an_exact_no_op(self):
        m = LatentEmbedRead(8)
        z = torch.randn(2, 5, 8)
        torch.testing.assert_close(m(z), z, rtol=1e-6, atol=1e-6)

    def test_zero_rows_stay_exactly_zero(self):
        """The J4 null is zeros at those columns, and unwritten ring rows are
        exactly zero; a bias would break both, which is why there is none."""
        m = LatentEmbedRead(8)
        z = torch.zeros(2, 5, 8)
        assert float(m(z).detach().abs().sum()) == 0.0

    def test_there_is_no_bias_to_learn_a_constant_with(self):
        assert LatentEmbedRead(8).proj.bias is None

    def test_freezing_removes_the_gradient(self):
        """The input requires grad so a graph exists at all -- the point is that
        the PARAMETER gets nothing, not that the module is detached."""
        m = LatentEmbedRead(8)
        for p in m.parameters():
            p.requires_grad_(False)
        m(torch.randn(2, 5, 8, requires_grad=True)).sum().backward()
        assert all(p.grad is None for p in m.parameters())

    def test_trainable_does_receive_a_gradient(self):
        """The contrast, so the test above cannot pass by the graph being dead."""
        m = LatentEmbedRead(8)
        m(torch.randn(2, 5, 8)).sum().backward()
        assert all(p.grad is not None and torch.count_nonzero(p.grad) > 0
                   for p in m.parameters())

    def test_freezing_changes_no_shapes(self):
        a, b = LatentEmbedRead(8), LatentEmbedRead(8)
        for p in a.parameters():
            p.requires_grad_(False)
        assert sorted(a.state_dict()) == sorted(b.state_dict())
        assert (sum(p.numel() for p in a.parameters())
                == sum(p.numel() for p in b.parameters()))
