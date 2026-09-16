"""
The Z channel — the dual-channel (token + latent) carry.

Z writes the recurrence's TRAJECTORY at the summary columns and reads it by
substituting into `s0` at the carried columns.  Almost every way it can be wrong
is silent: a detached tape gives a gradient-free carry behind a healthy loss
curve, a read that never fires gives a write nothing consumes, a shared gate
gives Z a keep/drop decision made on evidence about E.  None of those change the
loss curve's shape.  So these tests check the MECHANISM — what the packed
forward actually did — rather than the configuration that asked for it.

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

from evals.diag_depth_band import slot_depth_map  # noqa: E402

NV, K, CL, EOS = 4, 16, 16, VOCAB - 1
NC_GATED = 8              # K/W = 4, so two laps


def _model(latent=True, gated=True, **kw):
    torch.manual_seed(1234)
    common = dict(use_memory=True, memory_slots=0, accum_vecs=NV,
                  latent_carry=latent, eos_token_id=EOS)
    if gated:
        common.update(prefix_memory="gated", gate_slots=K, gate_route="ring",
                      gate_init="zero", gate_fill="grow")
    else:
        common.update(prefix_memory="accum", accum_max=K)
    common.update(kw)
    return _build_raven(**common).train()


def _ids(n_chunks=NC_GATED):
    torch.manual_seed(0)
    return torch.randint(0, VOCAB - 1, (1, CL * n_chunks))


def _chain(m, n_chunks=NC_GATED, num_steps=(0, 4), labels=False):
    """Run the chunk chain, returning every carry the forward produced."""
    ids = _ids(n_chunks)
    xs = [c.contiguous() for c in torch.chunk(ids, n_chunks, dim=1)]
    mc, carries, losses = None, [], []
    for xc in xs:
        kw = {}
        if labels:
            kw["labels"] = xc
        out = m(xc, num_steps=torch.tensor(list(num_steps)),
                m_cross_in=mc, return_m_cross=True, **kw)
        mc = out["m_cross"]
        carries.append(mc)
        if labels:
            losses.append(out["loss"])
    return carries, losses


def _spy_s0(m):
    """Capture s0 before and after the Z substitution, for every chunk."""
    seen = []
    real = m.cortex.latent_init

    def spy(s0, num_steps_no_grad=None):
        before = s0.detach().clone()
        out = real(s0, num_steps_no_grad)
        seen.append((before, out.detach().clone()))
        return out

    m.cortex.latent_init = spy
    return seen, (lambda: setattr(m.cortex, "latent_init", real))


class TestTheCarryShape:

    def test_the_carry_is_two_channels_wide(self):
        m = _model()
        D = m.cortex.prefix.hidden_size
        carries, _ = _chain(m)
        assert carries[-1].shape[-1] == 2 * D
        assert carries[-1].shape[1] == K            # rows unchanged by Z

    def test_e_only_is_unchanged_by_the_feature_existing(self):
        """The default must stay byte-identical to what B2 ran.  Z is opt-in and
        an opt-in that moved the default would invalidate every arm on record."""
        m = _model(latent=False)
        D = m.cortex.prefix.hidden_size
        carries, _ = _chain(m)
        assert carries[-1].shape[-1] == D
        assert not m.cortex.prefix.carries_latent
        assert not hasattr(m.cortex.prefix, "gate_proj_in_z")

    def test_row_count_is_the_ring_geometry_regardless_of_z(self):
        """Z rides the SAME rows through the SAME pointer, so it must not move
        the read block by a single column -- that co-location is the economic
        argument for carrying both channels."""
        a, _ = _chain(_model(latent=False))
        b, _ = _chain(_model(latent=True))
        assert [c.shape[1] for c in a] == [c.shape[1] for c in b]

    def test_an_e_only_carry_reaching_a_dual_channel_buffer_raises(self):
        """The mismatch that must never be tolerated: half of E read as a latent
        state is finite, plausible and wrong."""
        m = _model()
        D = m.cortex.prefix.hidden_size
        with pytest.raises(ValueError, match="E-only carry reached"):
            m.cortex.prefix.split_channels(torch.zeros(1, K, D))

    def test_a_dual_channel_carry_reaching_an_e_only_buffer_raises(self):
        m = _model(latent=False)
        D = m.cortex.prefix.hidden_size
        with pytest.raises(ValueError, match="latent_carry true"):
            m.cortex.prefix.split_channels(torch.zeros(1, K, 2 * D))

    def test_accum_carries_z_too(self):
        """A2 (accum + Z) is the cell that keeps the depth-slice ablation as a
        WITHIN-run measurement, so Z must not be gated-only."""
        m = _model(gated=False)
        D = m.cortex.prefix.hidden_size
        carries, _ = _chain(m, n_chunks=4)
        assert carries[-1].shape[-1] == 2 * D
        assert carries[-1].shape[1] == 4 * NV       # still write-once append


class TestTheRead:
    """The substitution into s0 -- the half that makes the design free."""

    def test_s0s_carried_columns_become_the_carried_z(self):
        m = _model()
        D = m.cortex.prefix.hidden_size
        seen, restore = _spy_s0(m)
        try:
            carries, _ = _chain(m, n_chunks=3)
        finally:
            restore()
        # chunk 2 reads what chunk 1 wrote
        before, after = seen[1]
        n_pre = carries[0].shape[1]
        assert torch.allclose(after[:, :n_pre].float(),
                              carries[0][..., D:].float(), atol=1e-5)

    def test_the_noise_it_replaces_is_actually_replaced(self):
        """If the substitution silently no-op'd, Z would be a write nothing
        consumes -- and the loss curve would look exactly the same."""
        m = _model()
        seen, restore = _spy_s0(m)
        try:
            carries, _ = _chain(m, n_chunks=3)
        finally:
            restore()
        before, after = seen[1]
        n_pre = carries[0].shape[1]
        assert not torch.allclose(before[:, :n_pre].float(),
                                  after[:, :n_pre].float(), atol=1e-6)

    def test_only_the_carried_columns_move(self):
        """The real tokens' and summary slots' s0 must be untouched: Z is a
        substitution into a field that already exists, not an addition."""
        m = _model()
        seen, restore = _spy_s0(m)
        try:
            carries, _ = _chain(m, n_chunks=3)
        finally:
            restore()
        before, after = seen[1]
        n_pre = carries[0].shape[1]
        assert torch.equal(before[:, n_pre:], after[:, n_pre:])

    def test_chunk_one_has_nothing_to_read_and_says_so(self):
        m = _model()
        seen, restore = _spy_s0(m)
        try:
            _chain(m, n_chunks=2)
        finally:
            restore()
        before, after = seen[0]
        assert torch.equal(before, after)

    def test_the_sequence_does_not_get_longer(self):
        """The whole point: Z costs no columns.  An E-only and a dual-channel
        run must present the same packed width to the loop."""
        a, b = _model(latent=False), _model(latent=True)
        wa, wb = [], []
        for m, acc in ((a, wa), (b, wb)):
            real = m.cortex.prefix.merge
            m.cortex.prefix.merge = (
                lambda st, nv, nl=None, _r=real, _a=acc: (
                    _a.append(nv.shape[1]) or _r(st, nv, nl)))
            _chain(m, n_chunks=3)
        assert wa == wb


class TestTheWrite:

    def test_the_tape_is_not_detached(self):
        """`latent_states` in the modeling file is `.clone().detach()`, and
        reusing it for Z would give a gradient-free carry behind a healthy loss
        curve.  The slice taken inside the loop has to be live."""
        m = _model()
        _, losses = _chain(m, n_chunks=3, labels=True)
        torch.stack(losses).mean().backward()
        # gradient reaching the loop's own weights through a live Z path
        loop = [p for n, p in m.named_parameters()
                if "core_block" in n and p.grad is not None]
        assert loop and all(torch.isfinite(p.grad).all() for p in loop)

    def test_grad_frac_is_one_when_the_whole_loop_is_trainable(self):
        m = _model()
        _chain(m, n_chunks=3, num_steps=(0, 4))
        assert m.cortex.latent_write_grad_frac == 1.0

    def test_grad_frac_is_zero_when_the_loop_is_all_no_grad(self):
        """mr32 with mean_backprop_depth 8 puts the whole write band in the
        frozen region.  That is a legitimate CHOICE, and this is the number that
        makes it a decision instead of a discovery."""
        m = _model()
        _chain(m, n_chunks=3, num_steps=(4, 0))
        assert m.cortex.latent_write_grad_frac == 0.0

    def test_grad_frac_is_partial_on_a_split_loop(self):
        m = _model()
        _chain(m, n_chunks=3, num_steps=(2, 2))
        frac = m.cortex.latent_write_grad_frac
        assert 0.0 < frac < 1.0

    def test_requires_grad_on_the_carry_is_NOT_the_indicator(self):
        """THE TRAP.  The carry is one tensor holding both channels, and E is
        live even when the loop ran entirely under no_grad (E comes through the
        coda, outside the loop).  `cat` marks the WHOLE tensor requires_grad, so
        reading requires_grad off the carry reports True for a Z channel that
        has no gradient path at all.  latent_write_grad_frac is the honest
        number; this pins the difference so nobody re-derives it the wrong way.
        """
        m = _model()
        D = m.cortex.prefix.hidden_size
        carries, _ = _chain(m, n_chunks=2, num_steps=(4, 0))
        assert carries[-1][..., D:].requires_grad          # says True...
        assert m.cortex.latent_write_grad_frac == 0.0      # ...and is frozen

    def test_one_cell_per_slot_is_harvested_from_the_grid(self):
        """The W slots are COLUMNS of one forward, each with its own trajectory,
        so the tape is a W x (T-1) grid and slot j must contribute column j --
        not the whole row, and not column 0 W times."""
        m = _model()
        _chain(m, n_chunks=2, num_steps=(0, 5))
        cx = m.cortex
        tape = cx._z_tape
        # T deltas, not T-1: latent_init captures s_0, so d_1 = s_1 - s_0 is a
        # real first step rather than a missing one.
        assert len(tape) == 5
        assert all(t.shape[1] == NV for t in tape)
        # ...but the map is asked for the LOOP LENGTH and clamps to T-1, so d_T
        # is never selected.  That is the spec (rule B is written as
        # round(f_j * (T-1))) and the tape index must follow it, not the tape.
        depths = cx.latent_depth_map(len(tape), NV)
        assert max(depths) <= len(tape) - 1
        w = cx.latent_write()
        for j, k in enumerate(depths):
            assert torch.equal(w[:, j], tape[k - 1][:, j])

    def test_an_empty_tape_raises_rather_than_writing_zeros(self):
        """A checkpoint carrying a modeling file that predates the Z hooks would
        otherwise train a dual-channel buffer whose Z half is never written --
        the exact silent class as the prefix_pack failure that ran 9 hours with
        no cross-segment memory."""
        m = _model()
        m.cortex._z_tape = []
        m.cortex._n_sum = NV
        with pytest.raises(RuntimeError, match="tape is EMPTY"):
            m.cortex.latent_write()


class TestSeparateGates:

    def test_z_has_its_own_gate_parameters(self):
        m = _model()
        buf = m.cortex.prefix
        assert buf.gate_proj_in_z is not buf.gate_proj_in
        assert buf.forget_bias_z is not buf.forget_bias

    def test_both_gates_receive_gradient(self):
        m = _model()
        _, losses = _chain(m, labels=True)
        torch.stack(losses).mean().backward()
        buf = m.cortex.prefix
        for name in ("gate_proj_in", "gate_proj_mem",
                     "gate_proj_in_z", "gate_proj_mem_z"):
            g = getattr(buf, name).weight.grad
            assert g is not None and float(g.norm()) > 0, name

    def test_zero_init_covers_the_z_gate_too(self):
        """apply_gate_init is what train.py calls to undo post_init's clobber.
        If it missed the Z pair, the Z gate would start at kaiming -- millions of
        untrained parameters making per-channel keep/drop decisions about a
        write they know nothing about, which is the exact thing gate_init=zero
        exists to prevent on the E side."""
        m = _model()
        buf = m.cortex.prefix
        torch.nn.init.normal_(buf.gate_proj_in_z.weight)
        torch.nn.init.normal_(buf.gate_proj_mem_z.weight)
        buf.apply_gate_init()
        assert float(buf.gate_proj_in_z.weight.detach().abs().sum()) == 0.0
        assert float(buf.gate_proj_mem_z.weight.detach().abs().sum()) == 0.0
        assert float(buf.forget_bias_z.detach()) == 1.0
        assert float(buf.input_bias_z.detach()) == 0.0

    def test_the_two_gates_can_disagree(self):
        """The reason they are separate: E rows are ~171 in norm and Z deltas
        ~0.35, so one projection reading both would be dominated by E.  With
        different weights the channels must be able to reach different fg."""
        m = _model()
        buf = m.cortex.prefix
        torch.manual_seed(3)
        torch.nn.init.normal_(buf.gate_proj_mem.weight, std=0.5)
        torch.nn.init.zeros_(buf.gate_proj_mem_z.weight)
        st = torch.randn(1, NV, buf.hidden_size)
        cd = torch.randn(1, NV, buf.hidden_size)
        _, _, fg_e = buf.gate(st, cd)
        _, _, fg_z = buf.gate_latent(st, cd)
        assert not torch.allclose(fg_e, fg_z)

    def test_gate_latent_refuses_on_an_e_only_buffer(self):
        m = _model(latent=False)
        buf = m.cortex.prefix
        st = torch.randn(1, NV, buf.hidden_size)
        with pytest.raises(ValueError, match="E-only buffer"):
            buf.gate_latent(st, st)

    def test_the_ring_pointer_is_shared(self):
        """Separate gates, ONE pointer -- that co-location is what keeps row r
        holding the same chunk's and same slot's (E, Z) pair."""
        m = _model()
        D = m.cortex.prefix.hidden_size
        carries, _ = _chain(m)
        prev, cur = carries[-2], carries[-1]
        moved_e = (cur[..., :D] - prev[..., :D]).norm(dim=-1)[0] > 1e-4
        moved_z = (cur[..., D:] - prev[..., D:]).norm(dim=-1)[0] > 1e-8
        assert torch.equal(moved_e, moved_z)


class TestTheNull:
    """Z's ablation null, which the 2x2 depends on."""

    def test_the_null_replaces_the_carried_z(self):
        m = _model()
        D = m.cortex.prefix.hidden_size
        carries, _ = _chain(m, n_chunks=2)
        m.cortex.latent_read_null = ("noise", 0.02, 7)
        seen, restore = _spy_s0(m)
        try:
            _chain(m, n_chunks=2)
        finally:
            restore()

    def test_zeros_are_refused_as_a_null(self):
        """E's null is zeros and Z's is noise, and the asymmetry is the point:
        a zero key still scores a mid-range logit and absorbs softmax mass,
        while noise is the model's own trained default for these columns."""
        m = _model()
        with pytest.raises(ValueError, match="kind must be 'noise'"):
            m.cortex._null_latent(("zeros",), torch.zeros(1, 2, 8))

    def test_the_null_is_reproducible_from_its_seed(self):
        m = _model()
        head = torch.zeros(1, 4, m.cortex.prefix.hidden_size)
        a = m.cortex._null_latent(("noise", 0.02, 11), head)
        b = m.cortex._null_latent(("noise", 0.02, 11), head)
        c = m.cortex._null_latent(("noise", 0.02, 12), head)
        assert torch.equal(a, b) and not torch.equal(a, c)

    def test_the_2x2s_duck_typed_contract_is_satisfied(self):
        """evals/eval_carry_2x2.py refuses to report a 2x2 unless it finds all
        three of these.  Refusing is right -- a missing channel reported as a
        null result for Z would be worse than no measurement -- so the contract
        has to be checked, not assumed."""
        sys.path.insert(0, os.path.join(REPO, "evals"))
        from evals.eval_carry_2x2 import has_latent_channel
        assert has_latent_channel(_model(latent=True).cortex)
        assert not has_latent_channel(_model(latent=False).cortex)


class TestDepthMap:

    def test_it_agrees_with_the_probe_that_chooses_it(self):
        """P0.7's probe prints a map and the graft writes one.  If they ever
        diverge, the probe would be choosing between rules the run does not
        implement."""
        m = _model()
        cx = m.cortex
        for T in (4, 8, 16, 32):
            for rule in ("absolute", "relative"):
                cx.latent_depth_rule = rule
                assert cx.latent_depth_map(T, 16) == slot_depth_map(
                    rule, T, 16, cx.latent_depth_lo, cx.latent_depth_hi)

    def test_every_depth_exists_in_the_loop_it_is_asked_of(self):
        """T is SAMPLED PER BATCH, so a map that requests depth 9 on a batch
        that drew T=4 would index past the tape.  Both rules must clamp."""
        m = _model()
        for rule in ("absolute", "relative"):
            m.cortex.latent_depth_rule = rule
            for T in (2, 3, 4, 8, 16, 32):
                assert all(1 <= k <= T - 1
                           for k in m.cortex.latent_depth_map(T, 16))

    def test_a_short_loop_still_produces_a_write(self):
        """The end-to-end form of the clamp: T=2 leaves exactly one usable
        delta, and the forward must not raise."""
        m = _model()
        carries, _ = _chain(m, n_chunks=2, num_steps=(0, 2))
        assert carries[-1].shape[1] == 2 * NV


class TestTheReadGradientAsymmetry:
    """MEASURED 2026-09-16, and in no earlier design note: a SINGLE no-grad step
    cuts Z's read gradient to exactly zero, while E's is unaffected.

    E re-enters the loop at every iteration through `input_embeds` (the adapter
    concatenates the carried E columns on each pass), so its read is refreshed
    inside the gradient window however long the no-grad prefix is.  Z enters
    ONCE, at s0, and the no-grad iterations run FIRST -- so with n >= 1 there is
    no path from the carried Z to the loss at all.

    This is a property of the architecture, not a tuning accident, and it cannot
    be configured away: `randomized_iteration_sampler` gives n = 0 only when the
    sampled p <= mean_backprop_depth, and p has a Poisson tail.  So it has to be
    MEASURED and reported, which is what these tests pin.
    """

    def _gate_grad(self, num_steps):
        m = _model()
        _, losses = _chain(m, n_chunks=NC_GATED, num_steps=num_steps,
                           labels=True)
        torch.stack(losses).mean().backward()
        buf = m.cortex.prefix
        gz = buf.gate_proj_in_z.weight.grad
        ge = buf.gate_proj_in.weight.grad
        return (0.0 if gz is None else float(gz.norm())), float(ge.norm()), m

    def test_a_fully_trainable_loop_gives_the_z_gate_gradient(self):
        z, e, m = self._gate_grad((0, 4))
        assert z > 0 and e > 0
        assert m.cortex.latent_read_grad_frac == 1.0

    def test_one_no_grad_step_is_enough_to_kill_it(self):
        """EXACTLY zero, not merely small -- which is why this is a structural
        fact and not a magnitude to be tuned."""
        z, e, m = self._gate_grad((1, 3))
        assert z == 0.0
        assert e > 0.0                       # ...and E is untouched
        assert m.cortex.latent_read_grad_frac == 0.0

    def test_the_e_read_survives_every_split(self):
        for split in ((0, 4), (1, 3), (2, 2), (3, 1)):
            _, e, _ = self._gate_grad(split)
            assert e > 0.0, split

    def test_the_fraction_is_counted_over_batches_not_reset_per_forward(self):
        """The quantity that matters is the training-run average: on a given
        batch the read is either fully live or fully dead, and it is the MIX
        over batches that decides how much signal the read side gets."""
        m = _model()
        _chain(m, n_chunks=2, num_steps=(0, 4))     # 2 live forwards
        _chain(m, n_chunks=2, num_steps=(2, 2))     # 2 dead forwards
        assert m.cortex.latent_read_grad_frac == 0.5
        assert m.cortex.latent_read_measured

    def test_an_unmeasured_read_is_not_reported_as_a_dead_one(self):
        """A modeling-file snapshot that predates the `num_steps_no_grad`
        argument leaves the fraction at 0.0, which must not be read as 'the
        read is dead'."""
        m = _model()
        assert not m.cortex.latent_read_measured
        assert m.cortex.latent_read_grad_frac == 0.0

    def test_the_forward_still_uses_z_when_the_read_has_no_gradient(self):
        """The cost is gradient signal, NOT the mechanism: the carried Z is
        substituted into s0 on every batch either way.  If this ever stopped
        being true, a dead read would also be a dead feature."""
        m = _model()
        seen, restore = _spy_s0(m)
        try:
            carries, _ = _chain(m, n_chunks=3, num_steps=(2, 2))
        finally:
            restore()
        before, after = seen[1]
        n_pre = carries[0].shape[1]
        assert not torch.equal(before[:, :n_pre], after[:, :n_pre])
