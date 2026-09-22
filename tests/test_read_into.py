"""P3.0 — the in-loop Z read at `CortexGraft.read_into`.

These are the eight gates `p30_readinto_prereg.md` §4 fixes BEFORE any GPU
time is spent, plus one the pre-registration did not have (the ablation null
following the read to its new site).  They are written with the module rather
than after it, because ten of this project's REDs were instrument-side and
four of them -- 8, 9, 10, 11 -- returned a CONFIDENT WRONG NUMBER instead of
failing.  A test that can only be reached from a GPU is how RED 14 survived.

GATE 3 IS THE RESULT, NOT A TEST.  Printed beside the s0 row -- read gradient
6.1e-2 at split (0,T) and EXACTLY 0.0 at (1,T-1), (2,T-2), (3,T-3) -- it is
the single table that justifies moving the read at all.

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

from test_cortex_eval import VOCAB, _build_raven  # noqa: E402

from cortex_memory.latent_read import (LatentRead, LatentRefresh,  # noqa: E402
                                       written_row_mask)

NV, K, CL, EOS = 4, 16, 16, VOCAB - 1
SPLITS = [(0, 8), (1, 7), (2, 6), (3, 5)]      # the four the s0 probe measured


def _model(read="xattn", depth="none", s0=False, latent=True, **kw):
    torch.manual_seed(1234)
    flags = dict(use_memory=True, memory_slots=0, accum_vecs=NV,
                 prefix_memory="gated", gate_slots=K, gate_route="ring",
                 gate_init="zero", gate_fill="grow",
                 latent_carry=latent, eos_token_id=EOS)
    if latent:
        flags.update(latent_s0_read=s0, latent_read=read,
                     latent_read_depth=depth, latent_read_heads=4)
    flags.update(kw)
    return _build_raven(**flags).train()


def _ids(n_chunks=2, eos_at=None):
    torch.manual_seed(0)
    ids = torch.randint(0, VOCAB - 1, (1, CL * n_chunks))
    if eos_at is not None:
        ids[0, eos_at] = EOS
    return ids


def _chain(m, n_chunks=2, num_steps=(0, 8), labels=False, eos_at=None):
    """Run the chunk chain; return (carries, losses).  Chunk 1 carries nothing,
    so anything about the READ must be asserted on chunk 2 or later."""
    ids = _ids(n_chunks, eos_at)
    mc, carries, losses = None, [], []
    for xc in [c.contiguous() for c in torch.chunk(ids, n_chunks, dim=1)]:
        kw = {"labels": xc} if labels else {}
        if eos_at is not None:
            # `eos_mask` is an explicit forward kwarg the trainer computes;
            # without it `begin()` builds no EOS masks at all and the read is
            # unmasked, which is the state every arm before P3.0 ran in.
            kw["eos_mask"] = (xc == EOS)
        out = m(xc, num_steps=torch.tensor(list(num_steps)),
                m_cross_in=mc, return_m_cross=True, **kw)
        mc = out["m_cross"]
        carries.append(mc)
        if labels:
            losses.append(out["loss"])
    return carries, losses


def _spy_read(m):
    """Record (x_in, x_out, current_step) for every read_into call."""
    seen = []
    real = m.cortex.read_into

    def spy(x, current_step=None):
        before = x.detach().clone()
        out = real(x, current_step)
        seen.append((before, out.detach().clone(), current_step))
        return out

    m.cortex.read_into = spy
    return seen


# ---------------------------------------------------------------------------
# gate 1 — read_into is not a no-op in prefix mode
# ---------------------------------------------------------------------------

class TestGate1NotANoOp:

    def test_read_into_changes_x_once_there_is_a_carry(self):
        """In prefix mode `cortex_graft.py` sets m_cross = accum = ccot_direct
        = None and memory_slots_iter = 0, so every pre-P3.0 branch of read_into
        is dead: a wired, gradient-carrying, EOS-masked hook with nothing
        plugged into it.  This asserts something is now plugged in."""
        m = _model()
        seen = _spy_read(m)
        _chain(m, n_chunks=2)
        assert seen, "read_into never fired at all"
        moved = [not torch.equal(a, b) for a, b, _ in seen]
        # Chunk 1 carries nothing, so its reads are legitimately no-ops.
        assert any(moved), "read_into never moved x on any iteration"

    def test_it_is_still_a_no_op_before_anything_is_carried(self):
        m = _model()
        seen = _spy_read(m)
        _chain(m, n_chunks=1)
        assert seen
        assert all(torch.equal(a, b) for a, b, _ in seen), (
            "chunk 1 has no carry; the read must inject exactly nothing rather "
            "than attending to unwritten rows")

    def test_refresh_mode_also_moves_x(self):
        m = _model(read="refresh", s0=True)
        seen = _spy_read(m)
        _chain(m, n_chunks=2)
        assert any(not torch.equal(a, b) for a, b, _ in seen)


# ---------------------------------------------------------------------------
# gate 2 — it fires once per iteration
# ---------------------------------------------------------------------------

class TestGate2OncePerIteration:

    @pytest.mark.parametrize("split", SPLITS)
    def test_call_count_is_the_whole_loop(self, split):
        m = _model()
        seen = _spy_read(m)
        _chain(m, n_chunks=2, num_steps=split)
        per_chunk = len(seen) // 2
        assert per_chunk == sum(split), (
            f"read_into fired {per_chunk}x for a {split} loop; it must fire on "
            "every iteration of BOTH loops -- that is the property that makes "
            "the read gradient live")

    def test_the_graft_counter_agrees_with_the_spy(self):
        """`_z_inloop_n` is what health.latent_runtime reports, so it is the
        number a run log will carry.  It must not drift from reality."""
        m = _model()
        _chain(m, n_chunks=2, num_steps=(2, 6))
        assert m.cortex._z_inloop_n == 8

    def test_current_step_is_passed_and_counts_up(self):
        m = _model()
        seen = _spy_read(m)
        _chain(m, n_chunks=1, num_steps=(3, 5))
        steps = [int(s) for _, _, s in seen]
        assert steps == list(range(8)), (
            "core_block_forward already has current_step and already passes it "
            "to iter_write; depth-matched addressing needs it here too")


# ---------------------------------------------------------------------------
# gate 3 — THE THESIS: the read has a gradient at every split
# ---------------------------------------------------------------------------

class TestGate3TheReadHasAGradient:

    @pytest.mark.parametrize("split", SPLITS)
    def test_read_params_get_gradient_at_every_split(self, split):
        """The whole redesign in one assertion.

        At s0 the read gradient is EXACTLY 0.0 the moment the sampler draws a
        single no-grad step, because `iterate_forward` runs the no-grad
        iterations FIRST and Z enters once, upstream of them.  `read_into`
        fires inside both loops, and `num_steps_with_grad` = min(s, Poisson+1)
        >= 1 on every training batch, so the read's parameters are always in
        the graph."""
        m = _model()
        _, losses = _chain(m, n_chunks=2, num_steps=split, labels=True)
        m.zero_grad(set_to_none=True)
        losses[-1].backward()
        r = m.cortex.latent_reader
        for name, p in [("q_proj", r.q_proj.weight), ("k_proj", r.k_proj.weight),
                        ("v_proj", r.v_proj.weight), ("out_proj", r.out_proj.weight),
                        ("gate", r.gate)]:
            assert p.grad is not None, f"{name} got no grad at split {split}"
            assert float(p.grad.abs().sum()) > 0.0, (
                f"{name} grad is exactly zero at split {split} -- the read is "
                "severed, which is the s0 failure this site exists to fix")

    def test_the_s0_site_still_loses_its_gradient_and_that_is_the_contrast(self):
        """The other half of the table.  Not a property of the new code -- a
        regression guard on the measurement that motivated it, so the claim
        cannot rot silently."""
        m = _model(read="none", s0=True, latent=True)
        _, losses = _chain(m, n_chunks=2, num_steps=(1, 7), labels=True)
        m.zero_grad(set_to_none=True)
        losses[-1].backward()
        gz = m.cortex.prefix.gate_proj_in_z.weight.grad
        assert gz is None or float(gz.abs().sum()) == 0.0, (
            "one no-grad step must still zero the s0 read's gradient; if this "
            "fails the measurement that justified P3.0 no longer holds")


# ---------------------------------------------------------------------------
# gate 4 — the read is always live
# ---------------------------------------------------------------------------

class TestGate4AlwaysLive:

    @pytest.mark.parametrize("split", SPLITS)
    def test_read_grad_frac_is_one(self, split):
        m = _model()
        _chain(m, n_chunks=2, num_steps=split)
        assert m.cortex.latent_read_measured
        assert m.cortex.latent_read_grad_frac == 1.0

    def test_s0_only_still_reports_the_sampled_fraction(self):
        """`read_grad_frac` means different things at the two sites, which is
        why health.latent_runtime reports `read_site` beside it."""
        m = _model(read="none", s0=True)
        _chain(m, n_chunks=2, num_steps=(2, 6))
        assert m.cortex.latent_read_grad_frac == 0.0


# ---------------------------------------------------------------------------
# gate 5 — an unwritten row injects nothing
# ---------------------------------------------------------------------------

class TestGate5UnwrittenRows:

    def test_all_zero_z_gives_exactly_zero_and_not_nan(self):
        """A fully-masked softmax is 0/0.  If that NaN were allowed to travel
        it would land in the loss on the FIRST CHUNK of every sequence."""
        r = LatentRead(16, n_heads=4)
        x = torch.randn(2, 5, 16)
        delta = r(x, torch.zeros(2, 3, 16))
        assert torch.equal(delta, torch.zeros_like(delta))

    def test_a_mixed_batch_does_not_leak_nan(self):
        r = LatentRead(16, n_heads=4)
        z = torch.randn(2, 3, 16)
        z[1] = 0.0                      # row 1 of the batch is on its chunk 1
        delta = r(torch.randn(2, 5, 16), z)
        assert torch.isfinite(delta).all()
        assert float(delta[1].detach().abs().sum()) == 0.0

    def test_unwritten_rows_take_no_attention_mass(self):
        """Not decoration: a zero row is not a neutral key -- it scores a
        mid-range logit and takes softmax mass BY COUNT.  Under fill='grow'
        most rows are unwritten early in every sequence."""
        r = LatentRead(16, n_heads=4)
        x = torch.randn(1, 5, 16)
        z = torch.randn(1, 4, 16)
        z_padded = torch.cat([z, torch.zeros(1, 12, 16)], dim=1)
        assert torch.allclose(r(x, z), r(x, z_padded), atol=1e-6)

    def test_written_row_mask_is_exact(self):
        z = torch.zeros(1, 3, 8)
        z[0, 1] = 1e-9
        assert written_row_mask(z).tolist() == [[False, True, False]]

    def test_refresh_leaves_unwritten_columns_alone(self):
        r = LatentRefresh(8, alpha_init=0.5)
        x = torch.randn(1, 6, 8)
        z = torch.randn(1, 2, 8)
        z[0, 1] = 0.0
        out = r(x, z, n_pre=2)
        assert torch.equal(out[0, 1], x[0, 1])
        assert not torch.equal(out[0, 0], x[0, 0])
        assert torch.equal(out[:, 2:], x[:, 2:])


# ---------------------------------------------------------------------------
# gate 6 — the EOS mask applies
# ---------------------------------------------------------------------------

class TestGate6EosMask:

    def test_the_mask_is_lifted_onto_the_packed_layout(self):
        """`begin()` builds the mask from the REAL sequence length, before
        prefix_pack adds n_pre carried columns and n_sum summary slots, so the
        stored mask is short by n_pre + n_sum.  Nothing caught this before
        because every read_into branch was dead in prefix mode."""
        m = _model()
        captured = {}
        real = m.cortex._packed_read_mask

        def spy(x):
            out = real(x)
            captured.setdefault("mask", out)
            captured["width"] = x.shape[1]
            return out

        m.cortex._packed_read_mask = spy
        _chain(m, n_chunks=2, eos_at=CL + 4)
        mask = captured["mask"]
        assert mask is not None
        assert mask.shape[1] == captured["width"], (
            "the lifted mask must cover the PACKED sequence, not the raw one")
        n_pre, n_sum = m.cortex._n_pre, m.cortex._n_sum
        assert float(mask[0, :n_pre].min()) == 1.0, (
            "carried columns sit before every real token, so they are before "
            "the first EOS by construction")
        assert float(mask[0, n_pre + 5]) == 0.0, (
            "a position past the chunk's first EOS must not read the previous "
            "document's carry")

    def test_masked_positions_receive_exactly_zero(self):
        r = LatentRead(16, n_heads=4)
        x, z = torch.randn(1, 6, 16), torch.randn(1, 3, 16)
        mask = torch.ones(1, 6, 1)
        mask[0, 3:] = 0.0
        delta = r(x, z, read_mask=mask)
        assert float(delta[0, 3:].detach().abs().sum()) == 0.0
        assert float(delta[0, :3].detach().abs().sum()) > 0.0


# ---------------------------------------------------------------------------
# gate 7 — depth-matched selects what the rule says
# ---------------------------------------------------------------------------

class TestGate7DepthMatched:

    def test_rows_match_latent_depth_map_exactly(self):
        """Checked AGAINST `latent_depth_map`, never against a reimplementation
        of it -- the same reason the depth map is cross-checked against
        `diag_depth_band.slot_depth_map` instead of being trusted to stay in
        step with it."""
        m = _model(depth="matched")
        c = m.cortex
        T, W = 8, c.prefix.n_vec
        depths = c.latent_depth_map(T, W)
        for step in range(T):
            rows = c.latent_read_rows(T, K, step)
            assert rows == [r for r in range(K) if depths[r % W] == step + 1]

    def test_iteration_i_reads_the_delta_it_is_the_counterpart_of(self):
        """Iteration i consumes s_i and produces s_{i+1}, so the depth it wants
        is d_{i+1} -- an off-by-one here would silently read a neighbouring
        depth and the loss curve would look fine."""
        m = _model(depth="matched")
        c = m.cortex
        c.latent_depth_lo, c.latent_depth_hi = 3, 3     # every slot at depth 3
        assert c.latent_read_rows(8, K, 2) == list(range(K))
        assert c.latent_read_rows(8, K, 1) == []
        assert c.latent_read_rows(8, K, 3) == []

    def test_a_depth_no_row_carries_injects_exactly_zero(self):
        m = _model(depth="matched")
        m.cortex.latent_depth_lo = m.cortex.latent_depth_hi = 3
        seen = _spy_read(m)
        _chain(m, n_chunks=2, num_steps=(0, 8))
        second = seen[len(seen) // 2:]
        moved = [i for i, (a, b, _) in enumerate(second) if not torch.equal(a, b)]
        assert moved == [2], (
            f"only iteration 2 carries depth 3; iterations {moved} moved x")

    def test_matched_without_current_step_raises_rather_than_falling_back(self):
        """A silent fallback to depth-agnostic would run a different arm than
        the config names, and nothing downstream could tell."""
        m = _model(depth="matched")
        _chain(m, n_chunks=2)                   # populate the carry
        with pytest.raises(RuntimeError, match="current_step"):
            m.cortex.read_into(torch.randn(1, m.cortex._n_pre + CL + NV, 64))

    def test_a_stale_modeling_file_is_refused_on_every_mode(self):
        """Job 13425548 ran against a checkpoint copy of the modeling file
        that predates P3.0 -- `read_into(x)`, one argument.  Depth was 'none',
        so the run did not use the step and the staleness was INVISIBLE; it
        would have surfaced as a wrong arm the first time a matched-depth cell
        ran.  prepare_cortex_checkpoint.py's docstring already calls the stale
        copy 'not a hypothetical'.  Refused on every mode now, not just
        matched."""
        for depth in ("none", "matched"):
            m = _model(depth=depth)
            _chain(m, n_chunks=2)
            with pytest.raises(RuntimeError, match="predates P3.0|current_step"):
                m.cortex.read_into(
                    torch.randn(1, m.cortex._n_pre + CL + NV, 64))


# ---------------------------------------------------------------------------
# gate 8 — E-only is untouched
# ---------------------------------------------------------------------------

class TestGate8EOnlyUntouched:

    def test_no_read_module_and_byte_identical_output(self):
        """cortex-final ships E-only.  An opt-in that moved the default would
        invalidate every arm on record, including the P2.3 cells."""
        a = _model(latent=False)
        b = _model(latent=False)
        assert a.cortex.latent_reader is None
        assert not any("latent_reader" in n for n, _ in a.named_parameters())
        ca, _ = _chain(a, n_chunks=2)
        cb, _ = _chain(b, n_chunks=2)
        assert torch.equal(ca[-1], cb[-1])

    def test_z_on_with_read_none_reproduces_the_s0_arm(self):
        """The P1 arms must still be reproducible bit for bit."""
        a = _model(read="none", s0=True)
        b = _model(read="none", s0=True)
        assert a.cortex.latent_reader is None
        ca, _ = _chain(a, n_chunks=2)
        cb, _ = _chain(b, n_chunks=2)
        assert torch.equal(ca[-1], cb[-1])


# ---------------------------------------------------------------------------
# gate 9 — the ablation null follows the read to its new site
# ---------------------------------------------------------------------------

class TestGate9TheNullFollowsTheRead:
    """NOT in the pre-registration, and it is the one that would have become
    RED 15.  `latent_read_null` is the contract eval_carry_2x2.py and
    eval_influence_horizon.py set to turn Z OFF.  It used to live inside
    latent_init because s0 was the only site.  A second site that bypassed it
    would report 'Z off' while feeding the model real Z -- a confident wrong
    number, in the exact shape of REDs 8/9/10, firing on the first P3.0 arm."""

    def test_the_null_reaches_the_in_loop_read(self):
        m = _model()
        base, _ = _chain(m, n_chunks=2, num_steps=(0, 8), labels=True)
        m.cortex.latent_read_null = ("noise", 0.02, 0)
        nulled, _ = _chain(m, n_chunks=2, num_steps=(0, 8), labels=True)
        assert not torch.equal(base[-1], nulled[-1]), (
            "setting latent_read_null changed nothing, so the in-loop read is "
            "bypassing the ablation hook and every Z-off number it produces "
            "would be a Z-ON number under the wrong label")

    def test_the_null_is_deterministic_for_a_seed(self):
        m = _model()
        m.cortex.latent_read_null = ("noise", 0.02, 7)
        a, _ = _chain(m, n_chunks=2)
        b, _ = _chain(m, n_chunks=2)
        assert torch.equal(a[-1], b[-1])

    def test_zeros_are_still_refused_as_z_s_null(self):
        m = _model()
        m.cortex.latent_read_null = ("zeros",)
        with pytest.raises(ValueError, match="noise"):
            _chain(m, n_chunks=2)


# ---------------------------------------------------------------------------
# the configuration surface
# ---------------------------------------------------------------------------

class TestTheConfigSurface:

    # Built through CortexMemory directly and NOT through _build_raven: that
    # helper turns any construction failure into pytest.skip (transformers
    # skew), so a "this config must be refused" test routed through it passes
    # by SKIPPING whether or not the guard exists.  Found by reading -rs.
    def _graft(self, **flags):
        from types import SimpleNamespace
        from cortex_graft import CortexMemory
        cfg = SimpleNamespace(
            n_embd=64, use_memory=True, memory_slots=0, accum_vecs=NV,
            prefix_memory="gated", gate_slots=K, eos_token_id=EOS,
            summary_init_token=EOS, **flags)
        return CortexMemory(cfg)

    def test_read_without_latent_carry_is_refused(self):
        with pytest.raises(ValueError, match="latent_carry"):
            self._graft(latent_carry=False, latent_read="xattn")

    def test_a_write_nothing_reads_is_refused(self):
        with pytest.raises(ValueError, match="read NOWHERE"):
            self._graft(latent_carry=True, latent_s0_read=False,
                        latent_read="none")

    def test_matched_without_xattn_is_refused(self):
        with pytest.raises(ValueError, match="matched"):
            self._graft(latent_carry=True, latent_read="refresh",
                        latent_read_depth="matched")

    def test_out_proj_is_not_zero_init(self):
        """The x0.90 shape.  LSTMBuffer zero-init'd out_proj, the read became a
        literal no-op that had to be discovered from nothing, and B2 measured
        unfreezing it at x1.33 -- the largest mechanism effect in the project.
        The real failure was a zero-init read whose GRADIENT PATH WAS SEVERED,
        and this site is where it is not, but the init stays nonzero anyway."""
        r = _model().cortex.latent_reader
        assert float(r.out_proj.weight.detach().abs().sum()) > 0.0
        assert r.gate_value == pytest.approx(0.1)

    def test_designed_init_survives_the_post_init_clobber(self):
        """Bug class 1, now covering a fifth module.  `reset_cortex_graft_init`
        calls reset_parameters() on every submodule, which puts KAIMING back
        into q/k/v/out and would silently discard this module's chosen init --
        exactly what it used to do to gate_init='zero' on the ring."""
        from cortex_graft import reset_cortex_graft_init
        m = _model()
        r = m.cortex.latent_reader
        torch.nn.init.constant_(r.gate, 99.0)
        torch.nn.init.constant_(r.out_proj.weight, 99.0)
        reset_cortex_graft_init(m)
        assert r.gate_value == pytest.approx(0.1)
        assert float(r.out_proj.weight.abs().max()) < 1.0
        assert torch.isfinite(r.out_proj.weight).all()

    def test_health_reports_the_site_and_the_gate(self):
        from cortex_memory.health import latent_runtime
        m = _model()
        _chain(m, n_chunks=2)
        h = latent_runtime(m.cortex)
        assert h["read_site"] == "xattn"
        assert h["read_grad_frac"] == 1.0
        assert h["read_gate"] == pytest.approx(0.1)
        assert h["read_calls"] == 8

    def test_suppressing_the_z_read_covers_the_in_loop_site(self):
        """`tools/prelaunch_final._SuppressZRead` is gate 1's instrument: it
        turns the Z read off and asserts the loss matches the E-only arm.  It
        used to unhook `latent_init` only.  On a P3.0 arm that leaves the
        in-loop read running, so "Z off" and "Z on" would genuinely produce the
        same loss and the gate would PASS while measuring nothing."""
        from tools.prelaunch_final import _SuppressZRead
        m = _model()
        _chain(m, n_chunks=2)                       # populate the carry
        live, _ = _chain(m, n_chunks=2)
        with _SuppressZRead(m.cortex):
            assert m.cortex.latent_reader is None
            off, _ = _chain(m, n_chunks=2)
        assert m.cortex.latent_reader is not None, "the reader must come back"
        assert not torch.equal(live[-1], off[-1]), (
            "suppression changed nothing, so the gate is blind on this arm")
