"""J7 — the staggered raw-state write, and the frozen (AutoCompressor) read.

Two changes, and both of them are silent-failure shaped.

`latent_encoding='staggered_state'` writes s_{k_j} at slot j for depths k_j
across the band, where `endpoint` wrote s_T at every slot and `delta` wrote
d_{k_j}.  The failure it replaces is not a crash: depth staggering was
configured on J4's and J5's arms and did NOTHING, because every encoding but
'delta' short-circuited before the depth map ran.  So these tests check that the
write actually drew more than one depth, that the depths it drew are the ones
the map asked for, and that the reported grad fraction is the PER-SLOT number
and not the last-step shortcut, which would print 1.0 over a frozen write.

`latent_embed_frozen=True` freezes the read projection at its identity init,
which makes the splice algebraically AutoCompressor's: rescale, then concatenate.
The parameter deliberately stays in the module, so the frozen and trainable arms
have the same state_dict and the same optimizer count.

Run: python -m pytest tests/test_staggered_state.py -q
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

NV, K, CL, EOS = 4, 16, 16, VOCAB - 1
NC = 8


def _model(encoding="staggered_state", lo=2, hi=9, frozen=False, **kw):
    torch.manual_seed(1234)
    common = dict(use_memory=True, memory_slots=0, accum_vecs=NV,
                  latent_carry=True, eos_token_id=EOS,
                  prefix_memory="gated", gate_slots=K, gate_route="ring",
                  gate_init="zero", gate_fill="grow",
                  latent_encoding=encoding,
                  latent_depth_rule="absolute",
                  latent_depth_lo=lo, latent_depth_hi=hi,
                  latent_s0_read=False, latent_read="embeds",
                  latent_read_znorm="rms", latent_read_znorm_target=136.0,
                  latent_embed_frozen=frozen)
    common.update(kw)
    return _build_raven(**common).train()


def _run(m, num_steps=(0, 6), n_chunks=NC, labels=False):
    torch.manual_seed(0)
    ids = torch.randint(0, VOCAB - 1, (2, CL * n_chunks))
    xs = [c.contiguous() for c in torch.chunk(ids, n_chunks, dim=1)]
    mc, out = None, None
    for xc in xs:
        kw = {"labels": xc} if labels else {}
        out = m(xc, num_steps=torch.tensor(list(num_steps)),
                m_cross_in=mc, return_m_cross=True, **kw)
        mc = out["m_cross"]
    return mc, out


class TestTheTape:

    def test_the_state_tape_is_taped_at_all(self):
        m = _model()
        _run(m)
        assert len(m.cortex._z_states) > 1

    def test_state_and_delta_tapes_stay_index_aligned(self):
        """_z_states[k-1] is s_k and _z_tape[k-1] is s_k - s_{k-1}, so the two
        lists must be the same length.  They are appended in one branch for
        exactly this reason; a drift of one would silently shift every depth."""
        m = _model()
        _run(m)
        c = m.cortex
        assert len(c._z_states) == len(c._z_tape) == len(c._z_grad)

    def test_the_state_tape_reconstructs_the_delta_tape(self):
        """The strongest statement that the two tapes describe one trajectory:
        differencing consecutive taped STATES must reproduce the taped deltas."""
        m = _model()
        _run(m)
        c = m.cortex
        for i in range(1, len(c._z_tape)):
            got = c._z_states[i] - c._z_states[i - 1]
            torch.testing.assert_close(got, c._z_tape[i], rtol=1e-4, atol=1e-5)

    def test_the_tape_is_not_detached(self):
        """A detached state tape is a gradient-free carry behind a healthy loss
        curve -- the bug class this project names 'class 2'."""
        m = _model()
        _run(m)
        assert any(t.requires_grad for t in m.cortex._z_states)

    def test_the_tape_resets_between_forwards(self):
        m = _model()
        _run(m, n_chunks=2)
        first = len(m.cortex._z_states)
        _run(m, n_chunks=2)
        assert len(m.cortex._z_states) == first


class TestTheStagger:

    def test_the_write_draws_more_than_one_depth(self):
        """The whole point.  `endpoint` drew one depth and reported nothing;
        n_depths == 1 here means the stagger collapsed."""
        m = _model()
        _run(m)
        assert len(m.cortex._z_depths_used) > 1

    def test_the_depths_drawn_are_the_depth_maps(self):
        m = _model()
        _run(m)
        c = m.cortex
        want = c.latent_depth_map(len(c._z_states), c.prefix.n_vec)
        T = len(c._z_states)
        want = sorted({min(max(k, 1), T) for k in want})
        assert c._z_depths_used == want

    def test_the_band_is_respected(self):
        m = _model(lo=2, hi=4)
        _run(m, num_steps=(0, 6))
        assert min(m.cortex._z_depths_used) >= 2
        assert max(m.cortex._z_depths_used) <= 4

    def test_endpoint_reports_no_depths_and_staggered_does(self):
        """The inert-config failure, stated as a test: `endpoint` never runs the
        depth map, so a run that believes it staggered must be able to tell."""
        e = _model(encoding="endpoint")
        _run(e)
        assert e.cortex._z_depths_used == []
        s = _model(encoding="staggered_state")
        _run(s)
        assert s.cortex._z_depths_used != []

    def test_a_short_loop_clamps_rather_than_indexing_out_of_range(self):
        m = _model(lo=2, hi=9)
        _run(m, num_steps=(0, 2))
        assert m.cortex._z_depths_used
        assert max(m.cortex._z_depths_used) <= len(m.cortex._z_states)

    def test_slot_j_takes_column_j(self):
        """Under the ring, write vector j always lands at rows congruent to j
        (mod W), so the slot -> column convention is what makes the depth ->
        row map addressable."""
        m = _model()
        _run(m)
        c = m.cortex
        w = c.latent_write()
        T = len(c._z_states)
        depths = c.latent_depth_map(T, c.prefix.n_vec)
        for j, k in enumerate(depths):
            want = c._z_states[min(max(k, 1), T) - 1][:, j:j + 1]
            torch.testing.assert_close(w[:, j:j + 1], want)

    def test_delta_and_staggered_share_the_slot_to_depth_convention(self):
        d = _model(encoding="delta")
        _run(d)
        s = _model(encoding="staggered_state")
        _run(s)
        assert (d.cortex.latent_depth_map(len(d.cortex._z_tape), NV)
                == s.cortex.latent_depth_map(len(s.cortex._z_states), NV))


class TestTheGradFraction:

    def test_it_is_per_slot_and_not_the_last_step_shortcut(self):
        """With a no-grad prefix long enough to swallow the band, the honest
        number is 0.0.  The `endpoint` shortcut would return 1.0, because the
        LAST iteration is always trainable -- that is the silent-wrong-number
        site this branch exists to close."""
        m = _model(lo=1, hi=2)
        _run(m, num_steps=(6, 2))
        assert m.cortex.latent_write_grad_frac == pytest.approx(0.0)

    def test_a_fully_trainable_loop_reports_one(self):
        m = _model(lo=1, hi=3)
        _run(m, num_steps=(0, 6))
        assert m.cortex.latent_write_grad_frac == pytest.approx(1.0)

    def test_a_partly_frozen_band_reports_between(self):
        m = _model(lo=1, hi=9)
        _run(m, num_steps=(3, 3))
        frac = m.cortex.latent_write_grad_frac
        assert 0.0 < frac < 1.0

    def test_endpoint_still_reports_one_on_the_same_split(self):
        """The contrast that makes the number above meaningful rather than a
        regression: endpoint draws depth T, which is always trainable."""
        m = _model(encoding="endpoint")
        _run(m, num_steps=(6, 2))
        assert m.cortex.latent_write_grad_frac == pytest.approx(1.0)


class TestTheFrozenProjection:

    def test_frozen_means_requires_grad_false(self):
        m = _model(frozen=True)
        assert all(not p.requires_grad for p in m.cortex.latent_embed.parameters())

    def test_trainable_is_still_the_default(self):
        m = _model(frozen=False)
        assert all(p.requires_grad for p in m.cortex.latent_embed.parameters())

    def test_the_parameter_set_is_identical_either_way(self):
        """The frozen and trainable arms must differ in requires_grad and
        NOTHING else -- same state_dict keys, same optimizer count."""
        a = _model(frozen=True)
        b = _model(frozen=False)
        assert (sorted(a.state_dict().keys()) == sorted(b.state_dict().keys()))
        assert (sum(p.numel() for p in a.parameters())
                == sum(p.numel() for p in b.parameters()))

    def test_frozen_at_identity_is_an_exact_no_op(self):
        """Which is the claim that makes this the AutoCompressor splice:
        proj @ z == z, so the spliced column is the rescaled row itself."""
        m = _model(frozen=True)
        z = torch.randn(2, K, m.cortex.latent_embed.hidden_size)
        torch.testing.assert_close(m.cortex.latent_embed(z), z,
                                   rtol=1e-5, atol=1e-5)

    def test_a_frozen_projection_receives_no_gradient(self):
        m = _model(frozen=True)
        _, out = _run(m, labels=True)
        out["loss"].backward()
        assert all(p.grad is None or torch.count_nonzero(p.grad) == 0
                   for p in m.cortex.latent_embed.parameters())

    def test_the_reset_path_cannot_unfreeze_it(self):
        """reset_cortex_graft_init re-applies the designed init; an arm that
        came back trainable there would be a control wearing this arm's name."""
        from cortex_graft import reset_cortex_graft_init
        m = _model(frozen=True)
        reset_cortex_graft_init(m)
        assert all(not p.requires_grad for p in m.cortex.latent_embed.parameters())

    def test_the_rescale_is_untouched_by_freezing(self):
        """Dropping the rotation is not dropping the scale fix: Z entering at
        ~10 against E's ~136 in one concatenation IS the measured s0 failure."""
        m = _model(frozen=True)
        assert m.cortex.latent_read_znorm == "rms"
        assert m.cortex.latent_read_znorm_target == pytest.approx(136.0)


class TestItStillBuildsAndRuns:

    def test_the_carry_is_still_two_channels_wide(self):
        m = _model()
        mc, _ = _run(m)
        assert mc.shape[-1] == 2 * m.config.n_embd
        assert mc.shape[1] == K

    def test_a_loss_still_flows_to_the_write(self):
        m = _model()
        _, out = _run(m, labels=True)
        out["loss"].backward()
        g = m.cortex.prefix.gate_proj_in_z.weight.grad
        assert g is not None and torch.count_nonzero(g) > 0

    def test_an_unknown_encoding_still_raises(self):
        # _expect_value_error, or _build_raven turns the ValueError this test
        # exists to catch into a skip labelled "transformers skew?" -- which it
        # did on job 13673654, where this was the 1 skipped of 27 on a cluster
        # whose environment was fine.
        with pytest.raises(ValueError, match="latent_encoding"):
            _model(encoding="staggered",          # near-miss of the real name
                   _expect_value_error=True)

    def test_embeds_admits_the_new_encoding(self):
        m = _model(encoding="staggered_state")
        assert m.cortex.latent_read == "embeds"
