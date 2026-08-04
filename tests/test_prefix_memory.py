"""
AutoCompressor-faithful prefix memory (2026-08-02) — module + graft level.

These replace the graft tests for the retired AccumCCoT / GatedAccumBuffer /
DirectCCoT paths.  The property that makes them worth having is that a WRONG
prefix splice is invisible in training: the loss curve stays healthy whether
the carry reaches the model or not, whether the summary slots see the chunk or
not, and whether a resume reset the write path or not.  So each test below
pins one thing that cannot be read off a curve:

  * the carry is LIVE at init (no zero-init read module to open) — the whole
    point of the rewrite;
  * the packed sequence is stripped back to real tokens before the head, so
    labels need no padding;
  * summary slots get position 0 and real tokens 1..S;
  * summary_emb is seeded from wte and a reload does NOT re-seed it;
  * ended documents do not leak their vectors across the boundary;
  * append rows stay separable (the slice-detach horizon and the eval slice
    ablation both depend on it), while gated width stays constant.

Run: /c/Users/henry/miniconda3/envs/cortex-retro/python.exe -m pytest tests/ -q
"""
from __future__ import annotations

import os
import sys

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, os.path.dirname(__file__))

from cortex_graft import CortexMemory, resolve_summary_init_token
from cortex_memory.buffers import PrefixAccumBuffer, PrefixGatedBuffer
from cortex_memory.chunking import detach_old_vecs
from test_cortex_graft import B, H, S, VOCAB, AttnRaven, _ids

NV = 4
EOS = VOCAB - 1


def _model(mode="accum", n_vec=NV, accum_max=32, **kw):
    return AttnRaven(use_memory=True, memory_slots=0, prefix_memory=mode,
                     accum_vecs=n_vec, accum_max=accum_max,
                     eos_token_id=EOS, **kw)


# ---------------------------------------------------------------------------
# Buffer modules
# ---------------------------------------------------------------------------

class TestPrefixBuffers:

    def test_accum_merge_appends_and_caps(self):
        buf = PrefixAccumBuffer(H, n_vec=NV, max_vecs=2 * NV)
        v = torch.randn(B, NV, H)
        s1 = buf.merge(None, v)
        s2 = buf.merge(s1, torch.randn(B, NV, H))
        s3 = buf.merge(s2, torch.randn(B, NV, H))
        assert s1.shape == (B, NV, H) and s2.shape == (B, 2 * NV, H)
        assert s3.shape == (B, 2 * NV, H)              # FIFO trim
        assert torch.allclose(s3[:, :NV], s2[:, NV:])  # oldest dropped

    def test_gated_merge_keeps_width_and_adopts_first_chunk(self):
        buf = PrefixGatedBuffer(H, n_vec=NV)
        v1 = torch.randn(B, NV, H)
        s1 = buf.merge(None, v1)
        assert torch.allclose(s1, v1)                  # nothing to gate against
        s2 = buf.merge(s1, torch.randn(B, NV, H))
        assert s2.shape == (B, NV, H) and not torch.allclose(s1, s2)

    def test_no_tanh_or_layernorm_bounding(self):
        """Infidelity #3: AC's summaries are raw hidden states.  A merge that
        re-bounded them would pin the write norm the way the old extraction
        did (0.23% across-sample variation in the 2026-08-02 diag)."""
        buf = PrefixAccumBuffer(H, n_vec=NV, max_vecs=32)
        big = torch.randn(B, NV, H) * 50.0
        assert torch.allclose(buf.merge(None, big), big)

    def test_seed_copies_the_token_row_everywhere(self):
        buf = PrefixAccumBuffer(H, n_vec=NV, max_vecs=32)
        wte = torch.randn(VOCAB, H)
        assert not bool(buf.summary_seeded)
        buf.init_from_token_embedding(wte, EOS)
        assert torch.allclose(buf.summary_emb, wte[EOS].expand(NV, H))
        assert bool(buf.summary_seeded)

    def test_summary_emb_is_exempt_from_decay_and_muon(self):
        buf = PrefixAccumBuffer(H, n_vec=NV, max_vecs=32)
        assert getattr(buf.summary_emb, "_no_weight_decay", False)


# ---------------------------------------------------------------------------
# Build / flag handling
# ---------------------------------------------------------------------------

class TestBuild:

    def test_accum_and_gated_build_and_own_the_cross_state(self):
        for mode, cls in (("accum", PrefixAccumBuffer), ("gated", PrefixGatedBuffer)):
            m = _model(mode)
            assert isinstance(m.cortex.prefix, cls)
            assert m.cortex.m_cross is None and m.cortex.accum is None
            assert m.cortex.ccot_direct is None and m.cortex.has_cross_state

    def test_no_read_module(self):
        """Infidelity #1: the read is the base model's own attention over the
        prepended vectors.  A stray out_proj would mean the bolt-on channel is
        back and has to be discovered from zero all over again."""
        m = _model()
        assert not hasattr(m.cortex.prefix, "out_proj")
        assert not hasattr(m.cortex.prefix, "q_proj")

    @pytest.mark.parametrize("dead,slots,attr", [("accum_ccot", 0, "accum"),
                                                 ("gated_accum", 4, "m_cross"),
                                                 ("ccot_direct", 0, "ccot_direct")])
    def test_retired_flags_still_LOAD(self, dead, slots, attr):
        """The graft must still build these.  Every Track-A and B1 checkpoint
        carries one of them in its config.json, so raising here (as it did from
        2026-08-02 to 2026-08-04) makes the whole historical results table
        unloadable — the write-capacity diagnostic could not open the very
        checkpoint it was written to explain.  The no-silent-null protection
        moved to train.py, which refuses to START a run on a retired mechanism;
        building the real buffer cannot produce a silent null because the
        mechanism is genuinely present."""
        m = AttnRaven(use_memory=True, memory_slots=slots, **{dead: True})
        assert getattr(m.cortex, attr) is not None
        assert m.cortex.has_cross_state

    @pytest.mark.parametrize("dead", ["accum_ccot", "gated_accum", "ccot_direct"])
    def test_retired_flags_cannot_be_combined_with_prefix(self, dead):
        """Two different cross-segment mechanisms at once: prefix mode would
        silently win and the legacy flag would read as honoured."""
        with pytest.raises(ValueError, match="legacy"):
            AttnRaven(use_memory=True, memory_slots=0, prefix_memory="accum",
                      accum_vecs=NV, eos_token_id=EOS, **{dead: True})

    def test_unknown_mode_raises(self):
        with pytest.raises(ValueError, match="prefix_memory"):
            AttnRaven(use_memory=True, memory_slots=0, prefix_memory="bogus")

    def test_seed_token_resolution(self):
        from types import SimpleNamespace
        cfg = SimpleNamespace(eos_token_id=7, vocab_size=VOCAB)
        assert resolve_summary_init_token(cfg) == 7
        cfg.eos_token_id = [9, 11]                        # some configs carry a list
        assert resolve_summary_init_token(cfg) == 9
        cfg.summary_init_token = 3                        # explicit wins
        assert resolve_summary_init_token(cfg) == 3

    def test_seed_token_missing_fails_at_build(self):
        with pytest.raises(ValueError, match="eos_token_id"):
            AttnRaven(use_memory=True, memory_slots=0, prefix_memory="accum",
                      accum_vecs=NV, accum_max=32)

    def test_seed_token_out_of_vocab_fails(self):
        from types import SimpleNamespace
        with pytest.raises(ValueError, match="outside the vocabulary"):
            resolve_summary_init_token(
                SimpleNamespace(eos_token_id=VOCAB + 5, vocab_size=VOCAB))


# ---------------------------------------------------------------------------
# The splice
# ---------------------------------------------------------------------------

class TestSplice:

    def test_logits_cover_only_real_tokens(self):
        """Labels are NOT padded for the carry/summary columns, so the unpack
        must strip them; a shape slip here shifts every label by n_prefix."""
        m = _model()
        out = m(_ids(), (0, 1), return_m_cross=True)
        assert out["logits"].shape == (B, S, VOCAB)

        # second chunk: n_prefix is now non-zero, which is when a bad strip bites
        out2 = m(_ids(), (0, 1), m_cross_in=out["m_cross"].detach())
        assert out2["logits"].shape == (B, S, VOCAB)

    def test_carry_at_zero_tokens_shifted_slots_at_the_tail(self):
        """Position layout (prefix_pos='tail', 2026-08-04).  The carry stays at
        0 — it is at the FRONT, so real tokens already query it at positive
        offsets — while the summary slots continue the token numbering so the
        WRITE also reads at positive offsets.  See prefix_pack for why position 0
        for a trailing slot is not the RoPE analog of AutoCompressor's OPT trick."""
        m = _model()
        m.cortex.begin(torch.randn(B, 2 * NV, H), None, S,
                       torch.device("cpu"), torch.float32)
        _, pos, n_pre, n_sum = m.cortex.prefix_pack(
            torch.zeros(B, S, H), torch.arange(S).unsqueeze(0))
        assert (n_pre, n_sum) == (2 * NV, NV)
        assert torch.all(pos[:, :n_pre] == 0)
        assert torch.equal(pos[0, n_pre:n_pre + S], torch.arange(1, S + 1))
        assert torch.equal(pos[0, -n_sum:], torch.arange(S + 1, S + 1 + NV))
        # still inside the trained window: S + n_vec, not S + n_pre + n_vec
        assert int(pos.max()) == S + NV

    def test_layout_is_carry_tokens_summary(self):
        m = _model()
        carry = torch.randn(B, NV, H)
        m.cortex.begin(carry, None, S, torch.device("cpu"), torch.float32)
        emb = torch.randn(B, S, H)
        packed, _, n_pre, n_sum = m.cortex.prefix_pack(emb, torch.arange(S).unsqueeze(0))
        assert packed.shape == (B, NV + S + NV, H)
        assert torch.allclose(packed[:, :n_pre], carry)
        assert torch.allclose(packed[:, n_pre:n_pre + S], emb)
        assert torch.allclose(packed[:, -n_sum:],
                              m.cortex.prefix.summary_emb.expand(B, NV, H))

    def test_emb_scale_applies_to_slots_only(self):
        m = _model()
        carry = torch.randn(B, NV, H)
        m.cortex.begin(carry, None, S, torch.device("cpu"), torch.float32)
        packed, _, n_pre, n_sum = m.cortex.prefix_pack(
            torch.zeros(B, S, H), torch.arange(S).unsqueeze(0), emb_scale=4.0)
        assert torch.allclose(packed[:, :n_pre], carry)           # hidden states: raw
        assert torch.allclose(packed[:, -n_sum:],
                              m.cortex.prefix.summary_emb.expand(B, NV, H) * 4.0)

    def test_carry_is_live_at_init(self):
        """No zero-init gate stands between the carry and the model — this is
        the single biggest difference from every earlier buffer, all of which
        started as an exact no-op."""
        m = _model()
        ids = _ids()
        mc = m(ids, (0, 2), return_m_cross=True)["m_cross"].detach()
        torch.manual_seed(5); a = m(ids, (0, 2), m_cross_in=mc)
        torch.manual_seed(5); b = m(ids, (0, 2), m_cross_in=None)
        assert not torch.allclose(a["logits"], b["logits"])


# ---------------------------------------------------------------------------
# Carry across a chunk chain
# ---------------------------------------------------------------------------

class TestChain:

    def test_accum_state_grows_by_n_vec_per_chunk(self):
        m = _model(accum_max=64)
        mc = None
        for i in range(1, 5):
            mc = m(_ids(), (0, 1), m_cross_in=mc, return_m_cross=True)["m_cross"]
            assert mc.shape == (B, i * NV, H)

    def test_accum_fifo_caps_at_accum_max(self):
        m = _model(accum_max=2 * NV)
        mc = None
        for _ in range(4):
            mc = m(_ids(), (0, 1), m_cross_in=mc, return_m_cross=True)["m_cross"]
        assert mc.shape == (B, 2 * NV, H)

    def test_gated_width_is_constant(self):
        m = _model("gated")
        mc = None
        for _ in range(4):
            mc = m(_ids(), (0, 1), m_cross_in=mc, return_m_cross=True)["m_cross"]
            assert mc.shape == (B, NV, H)

    def test_write_grad_through_chain(self):
        m = _model()
        ids, labels = _ids(), _ids()
        out1 = m(ids, (0, 2), return_m_cross=True)
        out2 = m(ids, (0, 2), labels=labels, m_cross_in=out1["m_cross"])
        out2["loss"].backward()
        g = m.cortex.prefix.summary_emb.grad
        assert g is not None and torch.isfinite(g).all() and g.norm() > 0

    def test_single_chunk_gives_the_write_path_no_grad(self):
        """Same requirement as every earlier carry: cross_chunks > 1 or the
        write is never on the loss path.  train.py asserts it; this is why."""
        m = _model()
        ids, labels = _ids(), _ids()
        m(ids, (0, 2), labels=labels, return_m_cross=True)["loss"].backward()
        g = m.cortex.prefix.summary_emb.grad
        assert g is None or g.norm() == 0

    def test_gated_feedback_needs_three_chunks(self):
        m = _model("gated")
        ids = _ids()
        mc, losses = None, []
        for _ in range(3):
            out = m(ids, (0, 2), labels=_ids(), m_cross_in=mc, return_m_cross=True)
            mc = out["m_cross"]
            losses.append(out["loss"])
        torch.stack(losses).mean().backward()
        g = m.cortex.prefix.gate_proj_mem.weight.grad
        assert g is not None and g.norm() > 0

    def test_rows_stay_separable_for_the_detach_horizon(self):
        """train.py slice-detaches rows older than carry_grad_chunks chunks,
        and eval_carry_ablation reconstructs the true state by re-appending the
        last n_vec returned rows.  Both are sound only because merge APPENDS:
        every earlier row must survive a write bit-for-bit."""
        m = _model(accum_max=64)
        mc = None
        for i in range(3):
            prev = mc
            mc = m(_ids(), (0, 1), m_cross_in=mc, return_m_cross=True)["m_cross"]
            if prev is not None:
                assert torch.equal(mc[:, :-NV], prev)      # older rows untouched
            assert mc.shape[1] == (i + 1) * NV
        # and the horizon keeps only the newest grad_chunks * n_vec rows live
        cut = detach_old_vecs(mc, NV, grad_chunks=1)
        assert cut.shape == mc.shape
        g = torch.autograd.grad(cut.sum(), m.cortex.prefix.summary_emb,
                                retain_graph=True, allow_unused=True)[0]
        full = torch.autograd.grad(mc.sum(), m.cortex.prefix.summary_emb,
                                   allow_unused=True)[0]
        assert full is not None and full.norm() > 0
        assert g is None or g.norm() < full.norm()


# ---------------------------------------------------------------------------
# Packed documents
# ---------------------------------------------------------------------------

class TestEos:
    """Document-boundary policy for the PREFIX carry.

    Changed 2026-08-04.  The old policy treated any EOS in the chunk as a full
    memory reset, which on EOS-separated packed data (what B2 trains on) switched
    the read off for the majority of chunks — see CortexMemory._carried_state.
    The default is now to carry across boundaries, matching what the backbone's
    own attention already does inside a chunk; prefix_eos_reset=True restores the
    old policy and these tests pin both.
    """

    def test_carry_survives_a_boundary_by_default(self):
        m = _model()
        ids = _ids(b=1)
        prev = m(ids, (0, 1), return_m_cross=True)["m_cross"].detach()
        eos = torch.zeros(1, S, dtype=torch.bool)
        eos[0, S - 1] = True                       # doc ends on the last position
        out = m(ids, (0, 1), m_cross_in=prev, return_m_cross=True, eos_mask=eos)
        # the previous chunk's rows are still there, un-zeroed
        assert torch.allclose(out["m_cross"][:, :NV], prev[:, :NV])
        # ...and this chunk still contributed a real (non-zero) summary
        assert out["m_cross"][:, -NV:].abs().sum() > 0

    def test_legacy_reset_zeroes_an_ended_document(self):
        m = _model(prefix_eos_reset=True)
        ids = _ids(b=1)
        prev = m(ids, (0, 1), return_m_cross=True)["m_cross"].detach()
        eos = torch.zeros(1, S, dtype=torch.bool)
        eos[0, S - 1] = True
        out = m(ids, (0, 1), m_cross_in=prev, return_m_cross=True, eos_mask=eos)
        assert torch.allclose(out["m_cross"], torch.zeros_like(out["m_cross"]))

    def test_open_document_keeps_carrying(self):
        m = _model()
        ids = _ids(b=1)
        prev = m(ids, (0, 1), return_m_cross=True)["m_cross"].detach()
        eos = torch.zeros(1, S, dtype=torch.bool)
        eos[0, S // 2] = True                      # a doc ends mid-chunk, another opens
        out = m(ids, (0, 1), m_cross_in=prev, return_m_cross=True, eos_mask=eos)
        assert out["m_cross"].abs().sum() > 0

    def test_legacy_reset_is_per_lane(self):
        m = _model(prefix_eos_reset=True)
        ids = _ids(b=2)
        prev = m(ids, (0, 1), return_m_cross=True)["m_cross"].detach()
        eos = torch.zeros(2, S, dtype=torch.bool)
        eos[0, S - 1] = True                       # lane 0 ends, lane 1 continues
        out = m(ids, (0, 1), m_cross_in=prev, return_m_cross=True, eos_mask=eos)
        assert torch.allclose(out["m_cross"][0], torch.zeros_like(out["m_cross"][0]))
        assert out["m_cross"][1].abs().sum() > 0

    def test_default_never_appends_a_zero_row(self):
        """A zero row would be spliced back as a zero-valued attention target,
        which still takes softmax mass (a zero key scores 0, not -inf)."""
        m = _model()
        ids = _ids(b=2)
        eos = torch.zeros(2, S, dtype=torch.bool)
        eos[0, S - 1] = True                       # the degenerate empty-suffix lane
        out = m(ids, (0, 1), m_cross_in=None, return_m_cross=True, eos_mask=eos)
        assert out["m_cross"][0].abs().sum() > 0


# ---------------------------------------------------------------------------
# Resume safety
# ---------------------------------------------------------------------------

class TestSeedingPersistence:

    def test_seeded_from_wte_on_first_forward(self):
        m = _model()
        assert not bool(m.cortex.prefix.summary_seeded)
        m(_ids(), (0, 1))
        assert bool(m.cortex.prefix.summary_seeded)
        assert torch.allclose(m.cortex.prefix.summary_emb,
                              m.wte.weight[EOS].expand(NV, H))

    def test_reload_does_not_reseed_a_trained_buffer(self):
        """The failure this guards: summary_seeded as a plain attribute would
        be False again after every resume, so each link of a multi-job chain
        would reset the write path to wte[eos] while the curve looked fine."""
        m = _model()
        m(_ids(), (0, 1))
        with torch.no_grad():
            m.cortex.prefix.summary_emb.add_(1.0)
        trained = m.cortex.prefix.summary_emb.clone()

        m2 = _model()
        assert "cortex.prefix.summary_seeded" in m.state_dict()
        m2.load_state_dict(m.state_dict())
        m2(_ids(), (0, 1))                          # would re-seed if the flag reset
        assert torch.allclose(m2.cortex.prefix.summary_emb, trained)

    def test_accum_params_are_a_strict_subset_of_gated(self):
        """The premise of the shared healing phase (pace/b2_retrofit.sbatch):
        healing runs as accum, then BOTH arms branch off it.  That is only
        legitimate if accum trains everything gated needs except the gate."""
        acc = set(_model("accum").cortex.prefix.state_dict())
        gat = set(_model("gated").cortex.prefix.state_dict())
        assert acc < gat
        assert gat - acc == {"gate_proj_in.weight", "gate_proj_in.bias",
                             "gate_proj_mem.weight", "gate_proj_mem.bias",
                             "forget_bias", "input_bias"}

    def test_healing_checkpoint_branches_into_the_gated_arm(self):
        """train.py's --branch_path loads with strict=False and rejects any
        missing key outside `cortex.`.  Verify that the ONLY thing a gated arm
        misses from an accum healing checkpoint is its gate, and that the
        healed summary embeddings survive the branch unchanged (they must not
        be re-seeded — the arm inherits 1.5B tokens of training in them)."""
        heal = _model("accum")
        heal(_ids(), (0, 1))                        # seed + "train"
        with torch.no_grad():
            heal.cortex.prefix.summary_emb.add_(0.5)
        healed = heal.cortex.prefix.summary_emb.clone()

        arm = _model("gated")
        missing, unexpected = arm.load_state_dict(heal.state_dict(), strict=False)
        assert not unexpected
        assert all(k.startswith("cortex.") for k in missing)
        assert all("prefix." in k for k in missing)
        # gate keeps its designed init; LM2 3.3 forget bias +1 biases to retain
        assert torch.allclose(arm.cortex.prefix.forget_bias, torch.ones(1))
        arm(_ids(), (0, 1))
        assert torch.allclose(arm.cortex.prefix.summary_emb, healed)

    def test_fresh_build_still_seeds_after_a_partial_load(self):
        """A base graft dir carries no cortex tensors: the flag must arrive
        False so the seeding still runs."""
        m = _model()
        m2 = _model()
        sd = {k: v for k, v in m.state_dict().items() if not k.startswith("cortex.")}
        m2.load_state_dict(sd, strict=False)
        assert not bool(m2.cortex.prefix.summary_seeded)
        m2(_ids(), (0, 1))
        assert torch.allclose(m2.cortex.prefix.summary_emb,
                              m2.wte.weight[EOS].expand(NV, H))
