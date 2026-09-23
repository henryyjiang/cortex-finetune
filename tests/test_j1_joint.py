"""
J1, the joint run (findings doc, "Attempt 2", Step 2): the four graft switches
it adds, pinned before any of them trains.

  latent_encoding    delta | endpoint | tokens      what Z IS
  latent_read_znorm  none | rms                     Z's scale at the xattn read
  latent_write_only                                 the no-read limb, on purpose
  e_dropout                                         blank the spliced E, training only

The property J1 exists to test is that the loss reaches Z's WRITE in chunk n
through Z's READ in chunk n+1.  So that path is pinned here with E's path cut,
and pinned ABSENT on the no-read limb -- a no-read limb that still carried a
read gradient would be a second treatment arm with the wrong label.

Run: /c/Users/henry/miniconda3/envs/cortex-retro/python.exe -m pytest tests/test_j1_joint.py -q
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

from cortex_graft import rescale_rows  # noqa: E402
from cortex_memory.health import latent_runtime  # noqa: E402

NV, K, CL, EOS, D = 4, 16, 16, VOCAB - 1, 64
T = 4


def _model(limb="real", encoding="delta", gated=True, **kw):
    torch.manual_seed(1234)
    flags = dict(use_memory=True, memory_slots=0, accum_vecs=NV,
                 latent_carry=True, eos_token_id=EOS,
                 latent_encoding=encoding, latent_tok_pool=2)
    if gated:
        flags.update(prefix_memory="gated", gate_slots=K, gate_route="ring",
                     gate_init="zero", gate_fill="grow")
    else:
        flags.update(prefix_memory="accum", accum_max=K * 4)
    if limb == "noread":
        flags.update(latent_read="none", latent_s0_read=False,
                     latent_write_only=True)
    else:
        flags.update(latent_read="xattn", latent_s0_read=False,
                     latent_read_heads=4,
                     latent_read_scramble=(limb == "donor"))
    flags.update(kw)
    return _build_raven(**flags).train()


def _ids(n=2, batch=2, seed=0):
    torch.manual_seed(seed)
    return torch.randint(0, VOCAB - 1, (batch, CL * n))


def _fwd(m, xc, mc=None, steps=(0, T), **kw):
    return m(xc, num_steps=torch.tensor(list(steps)), m_cross_in=mc,
             return_m_cross=True, **kw)


def _capture_writes(m):
    """Wrap prefix.merge to record (new_vecs, new_latent) per forward."""
    buf, real = m.cortex.prefix, m.cortex.prefix.merge
    got = []

    def wrapped(state, new_vecs, new_latent=None):
        got.append((new_vecs, new_latent))
        return real(state, new_vecs, new_latent)
    buf.merge = wrapped
    return got


# ─── config validation ──────────────────────────────────────────────────────

def _cortex(**kw):
    """Build the graft ALONE.  Not through _build_raven: that helper turns any
    ValueError into a pytest SKIP, so a must-raise test written against it
    goes green without checking anything (the first draft of this file did)."""
    from types import SimpleNamespace
    from cortex_graft import CortexMemory
    base = dict(n_embd=D, use_memory=True, memory_slots=0, accum_vecs=NV,
                prefix_memory="gated", gate_slots=K, eos_token_id=EOS,
                summary_init_token=EOS, latent_carry=True,
                latent_read="xattn", latent_s0_read=False,
                latent_read_heads=4)
    base.update(kw)
    return CortexMemory(SimpleNamespace(**base))


class TestTheSwitchesRefuseNonsense:
    def test_defaults_reproduce_every_arm_on_record(self):
        c = _cortex(latent_read="none", latent_s0_read=True)
        assert (c.latent_encoding, c.latent_read_znorm, c.latent_write_only,
                c.e_dropout, c.latent_tok_pool) == ("delta", "none", False,
                                                    0.0, 4)

    @pytest.mark.parametrize("kw, match", [
        (dict(latent_encoding="summary"), "latent_encoding must be"),
        (dict(latent_read_znorm="l2"), "latent_read_znorm must be"),
        (dict(e_dropout=1.0), "e_dropout must be"),
        (dict(e_dropout=-0.1), "e_dropout must be"),
        (dict(latent_tok_pool=0), "latent_tok_pool"),
        (dict(latent_read_znorm="rms", latent_read_znorm_target=-1.0),
         "znorm_target"),
    ])
    def test_bad_values_raise(self, kw, match):
        with pytest.raises(ValueError, match=match):
            _cortex(**kw)

    def test_an_encoding_without_a_z_channel_raises(self):
        with pytest.raises(ValueError, match="no Z channel"):
            _cortex(latent_carry=False, latent_read="none",
                    latent_s0_read=True, latent_encoding="endpoint")

    def test_znorm_without_the_xattn_read_raises(self):
        with pytest.raises(ValueError, match="needs latent_read='xattn'"):
            _cortex(latent_read="refresh", latent_read_znorm="rms")

    def test_write_only_with_a_read_on_raises(self):
        with pytest.raises(ValueError, match="NO-READ limb"):
            _cortex(latent_write_only=True)

    def test_write_only_with_the_s0_read_on_raises(self):
        with pytest.raises(ValueError, match="NO-READ limb"):
            _cortex(latent_read="none", latent_s0_read=True,
                    latent_write_only=True)

    def test_the_no_read_combination_still_raises_without_the_flag(self):
        # The accident stays an error; only the deliberate limb builds.
        with pytest.raises(ValueError, match="read NOWHERE"):
            _cortex(latent_read="none")

    def test_the_deliberate_no_read_limb_builds(self):
        c = _cortex(latent_read="none", latent_write_only=True)
        assert c.latent_reader is None and c.latent_write_only

    def test_e_dropout_without_a_prefix_buffer_raises(self):
        with pytest.raises(ValueError, match="e_dropout"):
            _cortex(prefix_memory="", latent_carry=False, latent_read="none",
                    latent_s0_read=True, e_dropout=0.2)


# ─── the encodings ──────────────────────────────────────────────────────────

class TestTheEncodingsWriteWhatTheySay:
    def _write_and_latents(self, encoding):
        m = _model(encoding=encoding).eval()
        got = _capture_writes(m)
        with torch.no_grad():
            out = _fwd(m, _ids(1), output_details={
                "return_logits": False, "return_latents": True,
                "return_head": False, "return_stats": False})
        return m, got[-1][1], out.latent_states

    def test_endpoint_is_s_T_at_the_summary_columns(self):
        m, z, ls = self._write_and_latents("endpoint")
        assert torch.allclose(z, ls[:, -NV:], atol=1e-6)

    def test_tokens_is_the_pooled_last_real_tokens(self):
        m, z, ls = self._write_and_latents("tokens")
        n_sum, pool = NV, 2
        L = NV * pool
        tok = ls[:, -n_sum - L:-n_sum]
        want = tok.reshape(tok.shape[0], NV, pool, D).mean(dim=2)
        assert z.shape == (2, NV, D)
        assert torch.allclose(z, want, atol=1e-6)

    def test_delta_is_unchanged(self):
        m, z, ls = self._write_and_latents("delta")
        # The staggered deltas are small next to the state; the endpoint is not.
        assert not torch.allclose(z, ls[:, -NV:], atol=1e-3)

    def test_tokens_refuses_a_chunk_too_short_to_pool(self):
        m = _model(encoding="tokens", latent_tok_pool=8).eval()   # 32 > 16
        with pytest.raises(ValueError, match="only"):
            with torch.no_grad():
                _fwd(m, _ids(1))

    @pytest.mark.parametrize("encoding", ["endpoint", "tokens"])
    def test_a_state_write_is_in_the_gradient_window_at_any_split(self, encoding):
        m = _model(encoding=encoding)
        _fwd(m, _ids(1), steps=(2, T - 2))
        assert m.cortex.latent_write_grad_frac == 1.0
        assert latent_runtime(m.cortex)["encoding"] == encoding


# ─── THE property: read in n+1 trains the write in n ────────────────────────

def _z_path_grads(limb, encoding):
    """Chunk 2's loss, backpropagated with E's half of the carry DETACHED, so
    every gradient that reaches chunk 1 came through the Z write and read.

    Returns (grad at the carried Z rows, grad at chunk 1's final loop state).
    The first is the read's pressure on the carry; the second is that pressure
    arriving inside chunk 1's loop.  A delta write is taken at depths 2..T-1,
    so it does not depend on the FINAL state and its second grad is zero by
    construction -- for it, the carry grad plus a live write graph is the test.
    """
    m = _model(limb=limb, encoding=encoding)
    x1, x2 = [c.contiguous() for c in torch.chunk(_ids(2), 2, dim=1)]
    out1 = _fwd(m, x1)
    s1 = m.cortex._z_last_x
    s1.retain_grad()
    mc = out1["m_cross"]
    z1 = mc[..., D:]
    z1.retain_grad()
    live_write = z1.grad_fn is not None
    mc = torch.cat([mc[..., :D].detach(), z1], dim=-1)
    out2 = _fwd(m, x2, mc=mc, labels=x2)
    out2["loss"].backward()
    return z1.grad, s1.grad, live_write


def _nz(g):
    return g is not None and float(g.abs().sum()) > 0


class TestTheReadTrainsTheWrite:
    @pytest.mark.parametrize("encoding", ["delta", "endpoint", "tokens"])
    def test_real_limb_puts_read_pressure_on_a_live_write(self, encoding):
        gz, gs, live = _z_path_grads("real", encoding)
        assert live, "the carried Z has no graph back into chunk 1's loop"
        assert _nz(gz)
        if encoding != "delta":
            assert _nz(gs)

    @pytest.mark.parametrize("encoding", ["delta", "endpoint", "tokens"])
    def test_no_read_limb_puts_none(self, encoding):
        gz, gs, _ = _z_path_grads("noread", encoding)
        assert not _nz(gz) and not _nz(gs)

    def test_donor_limb_trains_the_other_documents_write(self):
        """The roll sends each sequence's read to ANOTHER sequence's Z, so the
        gradient still reaches a write -- just not this document's.  Pinned so
        nobody 'fixes' it into a no-read limb: the donor's write is shaped
        under read pressure, with the content mismatched."""
        gz, gs, _ = _z_path_grads("donor", "endpoint")
        assert _nz(gz) and _nz(gs)


# ─── the no-read limb ───────────────────────────────────────────────────────

class TestTheNoReadLimb:
    def test_z_is_written_and_carried_at_full_width(self):
        m = _model(limb="noread")
        x1, x2 = torch.chunk(_ids(2), 2, dim=1)
        mc = _fwd(m, x1.contiguous())["m_cross"]
        assert mc.shape[-1] == 2 * D
        assert float(mc[..., D:].abs().sum()) > 0
        assert m.cortex.latent_reader is None

    def test_the_z_half_changes_nothing_downstream(self):
        m = _model(limb="noread").eval()
        x1, x2 = [c.contiguous() for c in torch.chunk(_ids(2), 2, dim=1)]
        with torch.no_grad():
            mc = _fwd(m, x1)["m_cross"]
            a = _fwd(m, x2, mc=mc)["logits"]
            mc2 = torch.cat([mc[..., :D], torch.randn_like(mc[..., D:])], -1)
            torch.manual_seed(5)
            b = _fwd(m, x2, mc=mc2)["logits"]
            torch.manual_seed(5)
            a = _fwd(m, x2, mc=mc)["logits"]
        assert torch.equal(a, b)


# ─── the Z norm ─────────────────────────────────────────────────────────────

class TestTheZNorm:
    def test_rows_land_on_the_target_and_zero_rows_stay_zero(self):
        z = torch.randn(2, 5, D) * 30
        z[:, 3] = 0.0
        out = rescale_rows(z, 3.0)
        n = out.norm(dim=-1)
        assert torch.allclose(n[:, [0, 1, 2, 4]], torch.full((2, 4), 3.0), atol=1e-4)
        assert torch.equal(out[:, 3], torch.zeros(2, D))

    def test_the_reader_sees_rescaled_rows(self):
        m = _model(encoding="endpoint", latent_read_znorm="rms",
                   latent_read_znorm_target=3.0).eval()
        seen = []
        real = m.cortex.latent_reader.forward

        def spy(x, z, read_mask=None):
            seen.append(z.detach().clone())
            return real(x, z, read_mask=read_mask)
        m.cortex.latent_reader.forward = spy
        x1, x2 = [c.contiguous() for c in torch.chunk(_ids(2), 2, dim=1)]
        with torch.no_grad():
            mc = _fwd(m, x1)["m_cross"]
            _fwd(m, x2, mc=mc)
        assert seen
        n = seen[-1].norm(dim=-1)
        written = n > 0
        assert torch.allclose(n[written], torch.full_like(n[written], 3.0),
                              atol=1e-4)


# ─── E-dropout ──────────────────────────────────────────────────────────────

class TestEDropout:
    def _two_chunks(self, m):
        x1, x2 = [c.contiguous() for c in torch.chunk(_ids(2), 2, dim=1)]
        mc = _fwd(m, x1)["m_cross"]
        return x2, mc

    def test_training_blanks_the_spliced_e_and_counts_it(self):
        m = _model(gated=False, e_dropout=0.999)
        x2, mc = self._two_chunks(m)
        seen = {}
        real = m.cortex.prefix.split_channels

        def spy(state):
            e, z = real(state)
            seen.setdefault("e", e)
            return e, z
        m.cortex.prefix.split_channels = spy
        torch.manual_seed(0)
        out = _fwd(m, x2, mc=mc)
        del m.cortex.prefix.split_channels
        assert m.cortex._e_dropped == 2
        assert latent_runtime(m.cortex)["e_dropped"] == 2
        # ONLY THE READ IS BLANKED: accum appends, so the carried rows of the
        # new state are the incoming E, untouched.
        n_old = mc.shape[1]
        assert torch.equal(out["m_cross"][:, :n_old, :D], mc[:, :, :D])

    def test_eval_never_drops(self):
        m = _model(gated=False, e_dropout=0.999).eval()
        x2, mc = self._two_chunks(m)
        with torch.no_grad():
            _fwd(m, x2, mc=mc)
        assert m.cortex._e_dropped == 0

    def test_a_dropped_row_splices_the_2x2_e_off_null(self):
        """The null is ZEROS -- what eval_carry_2x2's E-off cell splices -- so a
        model trained with E-dropout has SEEN the condition the 2x2 scores it
        on.  Checked on the packed embeddings the prelude actually receives."""
        m = _model(gated=False, e_dropout=0.999)
        x2, mc = self._two_chunks(m)
        real = m.cortex.prefix_pack
        got = {}

        def spy(*a, **kw):
            out = real(*a, **kw)
            got["packed"], got["n_pre"] = out[0], out[2]
            return out
        m.cortex.prefix_pack = spy
        _fwd(m, x2, mc=mc)
        del m.cortex.prefix_pack
        n_pre = got["n_pre"]
        assert n_pre == mc.shape[1] > 0
        assert torch.equal(got["packed"][:, :n_pre],
                           torch.zeros_like(got["packed"][:, :n_pre]))


# ─── the launcher ───────────────────────────────────────────────────────────

class TestTheLauncher:
    SB = os.path.join(REPO, "pace", "j1_joint.sbatch")

    def _src(self):
        with open(self.SB, encoding="utf-8") as fh:
            return fh.read()

    def test_each_limb_maps_to_its_flags(self):
        s = self._src()
        assert "latent_read_scramble true" in s          # donor
        assert "latent_write_only true" in s             # no-read
        assert "--cortex.latent_s0_read false" in s

    def test_it_branches_from_the_heal_w16_slice(self):
        assert "retro-b2-heal/checkpoint_91552_w16" in self._src()

    def test_the_donor_limb_needs_two_rows_per_micro_batch(self):
        s = self._src()
        assert "MICRO_BS" in s and "-lt 2" in s

    def test_diag_interval_is_always_passed(self):
        # The memory file's standing warning: without it the in-flight
        # monitoring silently does not happen.
        assert "--cortex.diag_interval" in self._src()

    def test_no_var_assignment_prefix_on_the_python_line(self):
        for line in self._src().splitlines():
            if line.strip().startswith("python train.py"):
                assert "=" not in line.split("python")[0]
