"""
P1.0 gated-buffer geometry: writes-per-chunk decoupled from buffer width.

Every property here is one that a wrong configuration hides behind a healthy
loss curve, which is why they are tests and not comments:

  * the old shape (K == W) still behaves exactly as it did, so landing P1.0
    cannot silently move an arm that did not ask for it;
  * a ring's FIRST LAP is bit-identical to PrefixAccumBuffer, which is the
    property that makes an accum arm and a gated arm diverge at eviction and
    nowhere else -- if it ever stops holding, A1-vs-A3 stops being one-variable;
  * gate_init="zero" really is the constant EMA at step 0, AND survives
    train.py's reset_cortex_graft_init, which calls reset_parameters() on every
    cortex submodule and would otherwise put kaiming weights back (bug class 1:
    a designed init clobbered after construction, invisible in the curve);
  * the depth->row map is stable across chunks, which is what keeps the
    depth-slice ablation alive under gating;
  * a dense route with no symmetry breaker is rank-pinned -- the degeneracy is
    a proof, and this pins it numerically so nobody re-derives it;
  * the cross_chunks >= 2 x lap constraint actually fires.  Without it a run
    trains an accum buffer with dead gate parameters attached.

Run: /c/Users/henry/miniconda3/envs/cortex-retro/python.exe -m pytest tests/ -q
"""
from __future__ import annotations

import io
import os
import sys

import pytest
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, 'evals'))

from cortex_memory.buffers import PrefixAccumBuffer, PrefixGatedBuffer
from diag_gate_geometry import ids_from_pack  # noqa: E402

B, D = 2, 64


def _writes(n, w, seed=0):
    g = torch.Generator().manual_seed(seed)
    return [torch.randn(B, w, D, generator=g) for _ in range(n)]


def _replay(buf, writes):
    state = None
    for v in writes:
        state = buf.merge(state, v)
    return state


# ---------------------------------------------------------------------------
# Backward compatibility: nothing moves at the old shape
# ---------------------------------------------------------------------------

class TestOldShapeUnchanged:

    def test_k_equals_w_is_the_pre_p10_merge(self):
        """At n_slots == n_vec the ring scatter is the identity, so the update
        is exactly fg*state + ig*new_vecs over the whole buffer."""
        torch.manual_seed(0)
        buf = PrefixGatedBuffer(D, n_vec=8, gate_init="default")
        vs = _writes(3, 8)
        got = _replay(buf, vs)

        ref = None
        for v in vs:
            if ref is None:
                ref = v
                continue
            comb = buf.gate_proj_in(v) + buf.gate_proj_mem(torch.tanh(ref))
            ig_l, fg_l = comb.chunk(2, dim=-1)
            ref = (torch.sigmoid(fg_l + buf.forget_bias) * ref
                   + torch.sigmoid(ig_l + buf.input_bias) * v)
        assert torch.equal(got, ref)

    def test_default_n_slots_tracks_n_vec(self):
        buf = PrefixGatedBuffer(D, n_vec=16)
        assert buf.n_slots == 16
        assert buf.geometry(512)["lap_chunks"] == 1


# ---------------------------------------------------------------------------
# The first lap IS the accum buffer
# ---------------------------------------------------------------------------

class TestGrowFirstLap:

    def test_first_lap_matches_accum_then_diverges_at_eviction(self):
        """The whole basis of the A1-vs-A3 contrast.  Identical while the ring
        is filling; different from the first chunk that would have evicted."""
        torch.manual_seed(0)
        W, K = 16, 64
        gated = PrefixGatedBuffer(D, n_vec=W, n_slots=K)
        accum = PrefixAccumBuffer(D, n_vec=W, max_vecs=K)
        vs = _writes(6, W)

        g = a = None
        for i, v in enumerate(vs):
            g, a = gated.merge(g, v), accum.merge(a, v)
            if i < K // W:                       # chunks 1..4, the filling lap
                assert g.shape == a.shape, f"shape diverged on chunk {i + 1}"
                assert torch.equal(g, a), f"content diverged on chunk {i + 1}"
            else:                                # chunk 5+: accum trims, ring gates
                assert g.shape == (B, K, D)
                assert not torch.equal(g, a)

    def test_grow_allocates_no_slot_init(self):
        """An unused [K, D] parameter would ride in every state dict and, under
        the meta-device load path, could come back as garbage."""
        assert not hasattr(PrefixGatedBuffer(D, 16, 64, fill="grow"), "slot_init")
        assert hasattr(PrefixGatedBuffer(D, 16, 64, fill="init"), "slot_init")

    def test_fill_init_pads_to_full_width_immediately(self):
        buf = PrefixGatedBuffer(D, n_vec=16, n_slots=64, fill="init")
        s = buf.merge(None, torch.randn(B, 16, D))
        assert s.shape == (B, 64, D)

    def test_width_is_fixed_once_full(self):
        buf = PrefixGatedBuffer(D, n_vec=16, n_slots=64)
        s = _replay(buf, _writes(20, 16))
        assert s.shape == (B, 64, D)


# ---------------------------------------------------------------------------
# Gate initialisation
# ---------------------------------------------------------------------------

class TestGateInit:

    def test_zero_init_is_exactly_the_ema(self):
        """Step 0 must be s' = sigmoid(1)*s + sigmoid(0)*c, uniform over rows
        and channels -- a live memory, not a no-op read."""
        buf = PrefixGatedBuffer(D, n_vec=8, gate_init="zero")
        s, c = torch.randn(B, 8, D), torch.randn(B, 8, D)
        out, ig, fg = buf.gate(s, c)
        assert torch.allclose(ig, torch.full_like(ig, 0.5), atol=1e-6)
        assert torch.allclose(fg, torch.full_like(fg, torch.sigmoid(torch.tensor(1.0))),
                              atol=1e-6)
        assert torch.allclose(out, 0.7310586 * s + 0.5 * c, atol=1e-5)

    def test_zero_init_is_not_a_dead_read(self):
        """Distinguishes this from LSTMBuffer's zero-init out_proj: the carried
        state must still depend on what was written."""
        buf = PrefixGatedBuffer(D, n_vec=8, n_slots=32, gate_init="zero")
        a = _replay(buf, _writes(6, 8, seed=1))
        b = _replay(buf, _writes(6, 8, seed=2))
        assert not torch.allclose(a, b)
        assert a.abs().mean() > 0

    def test_default_init_spreads_the_gates(self):
        torch.manual_seed(0)
        buf = PrefixGatedBuffer(D, n_vec=8, gate_init="default")
        _, _, fg = buf.gate(torch.randn(B, 8, D) * 3, torch.randn(B, 8, D))
        assert fg.std() > 0.05, "kaiming gate should not be constant"

    def test_apply_gate_init_restores_zero_after_reset_parameters(self):
        """train.py's reset_cortex_graft_init calls reset_parameters() on every
        cortex submodule to undo post_init's non-finite clobber.  That puts
        kaiming weights back into both gate projections; the buffer has to be
        able to re-apply its own designed init afterwards."""
        buf = PrefixGatedBuffer(D, n_vec=8, gate_init="zero")
        for m in buf.modules():
            if m is not buf and callable(getattr(m, "reset_parameters", None)):
                m.reset_parameters()
        assert buf.gate_proj_in.weight.abs().sum() > 0, "reset should have clobbered"
        buf.apply_gate_init()
        assert buf.gate_proj_in.weight.abs().sum() == 0
        assert buf.gate_proj_mem.weight.abs().sum() == 0
        assert buf.gate_proj_in.bias.abs().sum() == 0
        assert float(buf.forget_bias.detach()) == 1.0
        assert float(buf.input_bias.detach()) == 0.0


# ---------------------------------------------------------------------------
# Sparse routing: the depth->row map, and what it buys
# ---------------------------------------------------------------------------

class TestRingAddressing:

    def test_depth_row_map_is_stable_across_chunks(self):
        """Write vector j must land in rows congruent to j (mod W) on EVERY
        chunk.  This is what keeps the depth-slice ablation alive under gating;
        if it drifts, a depth ablation silently ablates a mixture.

        Isolated by differencing: at gate_init="zero" the gate is a constant, so
        perturbing ONLY write-vector j moves exactly the rows that received it.
        """
        W, K = 4, 16
        buf = PrefixGatedBuffer(D, n_vec=W, n_slots=K, gate_init="zero")
        state = _replay(buf, _writes(K // W, W))           # fill the lap
        for c in range(K // W, K // W + 6):
            for j in range(W):
                bumped = torch.zeros(B, W, D)
                bumped[:, j] = 1.0
                a = buf.merge(state.clone(), torch.zeros(B, W, D))
                buf._chunk -= 1                            # same chunk, twice
                b = buf.merge(state.clone(), bumped)
                buf._chunk -= 1
                moved = torch.nonzero(
                    (b - a).abs().sum(-1)[0]).flatten().tolist()
                assert moved, f"write {j} reached no row on chunk {c}"
                assert all(r % W == j for r in moved), (
                    f"chunk {c}, write {j} landed in rows {moved}, "
                    f"which are not all congruent to {j} mod {W}")
                assert set(moved) <= set(buf.depth_rows(j)), (
                    f"depth_rows({j}) = {buf.depth_rows(j)} does not cover "
                    f"the rows the write actually reached: {moved}")
            state = buf.merge(state, torch.randn(B, W, D))

    def test_depth_rows_matches_where_writes_land(self):
        W, K = 8, 32
        buf = PrefixGatedBuffer(D, n_vec=W, n_slots=K)
        assert buf.depth_rows(0) == [0, 8, 16, 24]
        assert buf.depth_rows(7) == [7, 15, 23, 31]
        with pytest.raises(ValueError):
            buf.depth_rows(W)

    def test_depth_rows_refuses_dense_routes(self):
        buf = PrefixGatedBuffer(D, n_vec=8, n_slots=32, route="mix")
        with pytest.raises(ValueError, match="no depth->row map"):
            buf.depth_rows(0)

    def test_ring_touches_exactly_w_rows_per_chunk(self):
        W, K = 8, 32
        buf = PrefixGatedBuffer(D, n_vec=W, n_slots=K, gate_init="zero")
        state = _replay(buf, _writes(K // W, W))       # fill the lap
        for _ in range(3):
            before = state.clone()
            state = buf.merge(state, torch.randn(B, W, D))
            changed = int((state - before).abs().sum(-1)[0].gt(0).sum())
            assert changed == W, f"gated {changed} rows, expected {W}"

    def test_ring_requires_a_whole_number_of_laps(self):
        with pytest.raises(ValueError, match="multiple"):
            PrefixGatedBuffer(D, n_vec=16, n_slots=40, route="ring")

    def test_cursor_resets_on_a_new_sequence(self):
        """`state is None` is the only reset signal the buffer gets, so a second
        sequence must write the same rows the first one did."""
        W, K = 8, 32
        buf = PrefixGatedBuffer(D, n_vec=W, n_slots=K, gate_init="zero")
        vs = _writes(7, W, seed=3)
        first = _replay(buf, vs)
        second = _replay(buf, vs)
        assert torch.equal(first, second)


# ---------------------------------------------------------------------------
# Dense routing, kept as the control -- and its degeneracy
# ---------------------------------------------------------------------------

class TestDenseRoute:

    def test_dense_refreshes_every_row(self):
        buf = PrefixGatedBuffer(D, n_vec=8, n_slots=32, route="mix",
                                gate_init="zero")
        state = buf.merge(None, torch.randn(B, 8, D))
        before = state.clone()
        state = buf.merge(state, torch.randn(B, 8, D))
        assert int((state - before).abs().sum(-1)[0].gt(0).sum()) == 32

    def test_no_symmetry_breaker_pins_the_rank(self):
        """route_init_std=0 IS the tile routing: rows r and r+W get the same
        candidate from the same state under a pointwise gate, so they are equal
        forever and the state can never exceed rank W.  A proof, pinned."""
        W, K = 4, 16
        buf = PrefixGatedBuffer(D, n_vec=W, n_slots=K, route="mix",
                                route_init_std=0.0, gate_init="zero")
        state = _replay(buf, _writes(6, W))
        for r in range(W):
            for copy in range(r + W, K, W):
                assert torch.allclose(state[:, r], state[:, copy], atol=1e-5)
        assert torch.linalg.matrix_rank(
            state[0] - state[0].mean(0, keepdim=True)) <= W

    def test_symmetry_breaker_lifts_it(self):
        torch.manual_seed(0)
        W, K = 4, 16
        buf = PrefixGatedBuffer(D, n_vec=W, n_slots=K, route="mix",
                                route_init_std=0.5, gate_init="zero")
        state = _replay(buf, _writes(6, W))
        assert not torch.allclose(state[:, 0], state[:, W], atol=1e-5)


# ---------------------------------------------------------------------------
# Shape contract and accounting
# ---------------------------------------------------------------------------

class TestContract:

    def test_write_width_is_enforced(self):
        buf = PrefixGatedBuffer(D, n_vec=16, n_slots=64)
        with pytest.raises(ValueError, match="buffer writes 16"):
            buf.merge(None, torch.randn(B, 8, D))

    def test_accum_shaped_carry_is_refused(self):
        buf = PrefixGatedBuffer(D, n_vec=16, n_slots=64)
        _replay(buf, _writes(4, 16))
        with pytest.raises(ValueError, match="fixed-width"):
            buf.merge(torch.randn(B, 128, D), torch.randn(B, 16, D))

    def test_narrower_than_one_write_is_refused(self):
        with pytest.raises(ValueError, match="cannot hold fewer rows"):
            PrefixGatedBuffer(D, n_vec=32, n_slots=16)

    def test_geometry_reports_the_p10_numbers(self):
        g = PrefixGatedBuffer(D, n_vec=16, n_slots=64).geometry(512)
        assert g["compression"] == 32.0
        assert g["read_cols_per_chunk"] == 64
        assert g["write_cols_per_chunk"] == 16
        assert g["packed_cols"] == 64 + 512 + 16
        assert g["lap_chunks"] == 4
        assert g["min_cross_chunks"] == 8

    def test_n_slots_is_not_a_parameter_shape_under_ring(self):
        """The reason K is a runtime knob and W is not: nothing under the ring
        route has a gate_slots dimension, so a K=64 checkpoint's weights load
        into a K=128 buffer unchanged."""
        small = PrefixGatedBuffer(D, n_vec=16, n_slots=64)
        big = PrefixGatedBuffer(D, n_vec=16, n_slots=256)
        assert small.state_dict().keys() == big.state_dict().keys()
        for k, v in small.state_dict().items():
            assert v.shape == big.state_dict()[k].shape, k
        big.load_state_dict(small.state_dict())

    def test_mix_route_does_freeze_k(self):
        a = PrefixGatedBuffer(D, n_vec=16, n_slots=64, route="mix")
        b = PrefixGatedBuffer(D, n_vec=16, n_slots=128, route="mix")
        assert a.state_dict()["route_mix"].shape != b.state_dict()["route_mix"].shape

    def test_gradient_reaches_the_memory_side_over_three_chunks(self):
        """gate_proj_mem only sees gradient through a chain of >= 3 chunks.  If
        this ever fails, carry_grad_chunks is too short and the memory half of
        the gate trains on nothing."""
        buf = PrefixGatedBuffer(D, n_vec=8, n_slots=16, gate_init="zero")
        state = None
        for v in _writes(4, 8):
            state = buf.merge(state, v)
        state.sum().backward()
        assert buf.gate_proj_mem.weight.grad is not None
        assert buf.gate_proj_mem.weight.grad.abs().sum() > 0


# ---------------------------------------------------------------------------
# The retention model the geometry was chosen on
# ---------------------------------------------------------------------------

class TestRetentionModel:

    def test_matches_ig_times_fg_to_the_power_of_laps(self):
        """F(d) = ig * fg**floor(d / (K/W)) at gate_init=zero.  W=16/K=64 was
        locked on this curve, so it should break if the update ever changes."""
        W, K, N = 4, 16, 24
        lap = K // W
        buf = PrefixGatedBuffer(D, n_vec=W, n_slots=K, gate_init="zero")
        vs = _writes(N, W, seed=7)
        donor = _writes(N, W, seed=99)
        base = _replay(buf, vs)
        ceil = float((vs[0] - donor[0]).norm() / vs[0].norm())

        # Steady state only: content written during the first lap is
        # APPENDED at full strength, not gated in at ig, so it follows
        # 1.0 * fg**laps instead.  N=24 keeps every d below in the
        # gated regime.
        for d in (0, lap, 2 * lap, 3 * lap, 4 * lap):
            swapped = list(vs)
            swapped[N - 1 - d] = donor[N - 1 - d]
            moved = float((_replay(buf, swapped) - base).detach().norm()
                          / vs[0].norm() / ceil)
            want = 0.5 * 0.7310586 ** (d // lap)
            assert abs(moved - want) < 0.03, (
                f"d={d}: measured {moved:.3f}, model {want:.3f}")


# ---------------------------------------------------------------------------
# Graft level: the config keys actually reach the buffer
# ---------------------------------------------------------------------------

class TestGraftWiring:
    """A key the graft does not read is the quiet failure here: the run trains
    a DIFFERENT buffer from the one the sbatch asked for, with a healthy loss
    curve and a config.json that says otherwise."""

    def _cortex(self, **kw):
        from test_cortex_graft import AttnRaven
        return AttnRaven(use_memory=True, memory_slots=0,
                         prefix_memory="gated", eos_token_id=7, **kw).cortex

    def test_gate_keys_reach_the_buffer(self):
        c = self._cortex(accum_vecs=4, gate_slots=16, gate_route="ring",
                         gate_norm="rms", gate_init="default", gate_fill="init")
        assert c.prefix.n_vec == 4
        assert c.prefix.n_slots == 16
        assert c.prefix.route == "ring"
        assert c.prefix.gate_norm == "rms"
        assert c.prefix.gate_init == "default"
        assert c.prefix.fill == "init"

    def test_defaults_reproduce_the_pre_p10_shape(self):
        c = self._cortex(accum_vecs=8)
        assert c.prefix.n_slots == 8, "gate_slots unset must mean K == W"
        assert c.prefix.route == "ring" and c.prefix.gate_init == "zero"

    def test_gate_slots_zero_means_n_vec(self):
        c = self._cortex(accum_vecs=8, gate_slots=0)
        assert c.prefix.n_slots == 8


class TestPackSourceAndOverrides:
    """The A(d) probe died in four seconds on its first real run, twice over:
    its only prose source was --text_file pointing above the repo (true on a
    laptop, absent on PACE), and it loaded with NO config overrides against a
    base dir whose config carries no cortex flags -- so even with prose it would
    have exited 2 with "no prefix buffer".  Both are pinned here because both
    were invisible until a GPU allocation went to waste on them.
    """

    def _pack(self, tmp_path, rows=8, row_len=10):
        import datasets
        ds = datasets.Dataset.from_dict(
            {"input_ids": [[r * 100 + c for c in range(row_len)]
                           for r in range(rows)]})
        d = str(tmp_path / "pack")
        ds.save_to_disk(d)
        return d

    def test_the_donor_rows_cannot_overlap_the_main_tapes_rows(self, tmp_path):
        """The whole point of skip_rows.  A donor that shares rows with the main
        tape is a near-copy of it, and A(d) then measures rounding error and
        comes back flat for a reason that is not the buffer."""
        d = self._pack(tmp_path)
        main = ids_from_pack(d, 20, 0)
        donor = ids_from_pack(d, 20, 2)
        assert len(main) == len(donor) == 20
        assert not (set(main) & set(donor))

    def test_a_pack_too_small_for_the_geometry_refuses_instead_of_padding(self, tmp_path):
        """Silently returning fewer tokens would make the last chunks repeats of
        the first, which reads as a buffer that remembers everything."""
        d = self._pack(tmp_path)
        with pytest.raises(SystemExit) as e:
            ids_from_pack(d, 10_000, 0)
        assert "rows" in str(e.value)

    def test_the_probe_accepts_data_and_set(self):
        """Argument-level, because the failure was at argument level: the flag
        either exists on the parser or the job dies on the node."""
        import diag_gate_geometry as g
        src = io.open(g.__file__, encoding="utf-8").read()
        assert '"--data"' in src
        assert '"--set"' in src
        assert "parse_config_overrides(args.set)" in src
        assert "config_overrides=overrides or None" in src

    def test_the_launcher_passes_both(self):
        s = io.open(os.path.join(REPO, "pace", "gate_geometry_ad.sbatch"),
                    encoding="utf-8").read()
        assert '--data "$DATA"' in s
        assert "--set use_memory=true" in s
        assert "text_file" not in s.split("cd $SLURM_SUBMIT_DIR")[1], (
            "the launcher still reaches for a file above the repo")
