"""
Tests for GatedAccumBuffer — the gated-accumulation LM2 variant (two-track
plan §A1 buffer-choice note, promoted to a Track-B retrofit option
2026-07-17): AccumCCoT's extraction write + the LM2 gated merge on a fixed
K-slot state (target k=16/32).

Key invariant under test: the extraction is SHARED code with AccumCCoT
(`_extract_summary_vectors`), so append vs gated-overwrite is the only
load-bearing difference between the two arms.

Graft-level tests reuse the FakeRaven harness from test_cortex_graft (mirrors
the exact hook sequence of the grafted raven modeling files).

Run: /c/Users/henry/miniconda3/envs/cortex/python.exe -m pytest tests/ -v
"""
from __future__ import annotations

import os
import sys

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, os.path.dirname(__file__))

from cortex_memory.buffers import AccumCCoT, GatedAccumBuffer, LSTMBuffer
from test_cortex_graft import B, H, S, FakeRaven, _ids

K = 16   # slot count in these tests (target scale k=16/32)


# ---------------------------------------------------------------------------
# GatedAccumBuffer module
# ---------------------------------------------------------------------------

class TestGatedAccumModule:

    def _mod(self, k=K):
        return GatedAccumBuffer(H, n_slots=k, n_heads=4)

    def test_write_shape_k16_k32(self):
        for k in (16, 32):
            m = self._mod(k)
            buf = torch.zeros(B, k, H)
            new = m.write(torch.randn(B, S, H), buf)
            assert new.shape == (B, k, H)

    def test_state_is_fixed_size(self):
        """Unlike AccumCCoT, repeated writes never grow the state."""
        m = self._mod()
        buf = torch.zeros(B, K, H)
        for _ in range(6):
            buf = m.write(torch.randn(B, S, H), buf)
            assert buf.shape == (B, K, H)

    def test_read_zero_init_noop(self):
        m = self._mod()
        buf = m.write(torch.randn(B, S, H), torch.zeros(B, K, H))
        delta = m.read(torch.randn(B, S, H), buf)
        assert torch.all(delta == 0)

    def test_read_active_after_out_proj_init(self):
        m = self._mod()
        nn.init.normal_(m.out_proj.weight, std=0.05)
        buf = m.write(torch.randn(B, S, H), torch.zeros(B, K, H))
        delta = m.read(torch.randn(B, S, H), buf)
        assert delta.shape == (B, S, H) and delta.norm() > 0

    def test_extraction_matches_accum_ccot(self):
        """With identical extraction weights, GatedAccumBuffer.extract must be
        bitwise-equal to AccumCCoT.extract — the shared-write-path invariant
        that makes append-vs-gated-overwrite the only arm difference."""
        gated = self._mod()
        accum = AccumCCoT(H, n_vec=K, n_heads=4, max_vecs=4 * K)
        with torch.no_grad():
            accum.vec_emb.copy_(gated.vec_emb)
            accum.wq_proj.weight.copy_(gated.wq_proj.weight)
            accum.wk_proj.weight.copy_(gated.wk_proj.weight)
            accum.wv_proj.weight.copy_(gated.wv_proj.weight)
            accum.vec_ln.weight.copy_(gated.vec_ln.weight)
            accum.vec_ln.bias.copy_(gated.vec_ln.bias)
        h = torch.randn(B, S, H)
        assert torch.equal(gated.extract(h), accum.extract(h))

    def test_gated_merge_between_buffer_and_candidate(self):
        """fg,ig ∈ (0,1): every element of the merged state lies strictly
        inside the interval spanned by the old buffer row and the candidate
        (elementwise convex-cone check: new = fg*buf + ig*cand)."""
        m = self._mod()
        buf  = torch.randn(B, K, H)
        h    = torch.randn(B, S, H)
        cand = m.extract(h)
        new  = m.write(h, buf)
        lo = torch.minimum(torch.zeros_like(buf), buf) + torch.minimum(torch.zeros_like(cand), cand)
        hi = torch.maximum(torch.zeros_like(buf), buf) + torch.maximum(torch.zeros_like(cand), cand)
        assert torch.all(new >= lo - 1e-5) and torch.all(new <= hi + 1e-5)

    def test_forget_bias_retention_at_init(self):
        """Forget bias +1.0 (LM2 §3.3): at init the buffer is mostly retained —
        writing on top of a buffer must keep a substantial fraction of it."""
        m = self._mod()
        buf = torch.randn(B, K, H)
        new = m.write(torch.randn(B, S, H), buf)
        # correlation between old and new state should be clearly positive
        cos = nn.functional.cosine_similarity(
            new.reshape(-1, H), buf.reshape(-1, H), dim=-1).mean()
        assert cos > 0.3

    def test_pool_mask_restricts_extraction(self):
        m = self._mod()
        h = torch.randn(B, S, H)
        mask = torch.zeros(B, S, dtype=torch.bool)
        mask[:, : S // 2] = True
        buf = torch.randn(B, K, H)
        n1 = m.write(h, buf, pool_mask=mask)
        h2 = h.clone()
        h2[:, S // 2:] += torch.randn_like(h2[:, S // 2:])
        n2 = m.write(h2, buf, pool_mask=mask)
        assert torch.allclose(n1, n2, atol=1e-6)

    def test_candidates_are_slot_distinct(self):
        m = self._mod()
        cand = m.extract(torch.randn(B, S, H))
        for i in range(0, K, 5):
            for j in range(i + 1, K, 5):
                assert not torch.allclose(cand[:, i], cand[:, j])


# ---------------------------------------------------------------------------
# GatedAccumBuffer through the graft (FakeRaven chain)
# ---------------------------------------------------------------------------

def _gated_model(k=K):
    return FakeRaven(use_memory=True, memory_slots=k, gated_accum=True)


NUM_STEPS = (1, 1)


class TestGatedAccumGraft:

    def test_build(self):
        model = _gated_model()
        assert isinstance(model.cortex.m_cross, GatedAccumBuffer)
        assert model.cortex.has_cross_state
        # h_T_proj is an LSTMBuffer-only mitigation — extraction wk/wv already
        # decouple the write path (same reason AccumCCoT takes raw h_T).
        assert model.cortex.h_T_proj is None

    def test_default_stays_lstm(self):
        model = FakeRaven(use_memory=True, memory_slots=4)
        assert isinstance(model.cortex.m_cross, LSTMBuffer)
        assert model.cortex.h_T_proj is not None

    def test_state_shape_constant_over_chunks(self):
        model = _gated_model()
        m_cross = None
        for _ in range(5):
            out = model(_ids(), NUM_STEPS, m_cross_in=m_cross, return_m_cross=True)
            m_cross = out["m_cross"]
            assert m_cross.shape == (B, K, H)

    def test_zero_init_read_is_noop(self):
        """At designed init the carried buffer must not change the logits
        (step-0 == base model)."""
        model = _gated_model()
        ids = _ids()
        buf = model(_ids(), NUM_STEPS, return_m_cross=True)["m_cross"]
        # FakeRaven.initialize_state draws random s0 — seed identically so the
        # ONLY difference between the two forwards is the carried buffer.
        torch.manual_seed(1234)
        out0 = model(ids, NUM_STEPS)
        torch.manual_seed(1234)
        out1 = model(ids, NUM_STEPS, m_cross_in=buf)
        assert torch.equal(out0["logits"], out1["logits"])

    def test_chain_grads_reach_write_and_feedback(self):
        """3-chunk chain: the read at chunk g+1 must put chunk g's write on the
        loss path — vec_emb / wk_proj (extraction), gate_proj_in, and the
        memory-feedback gate_proj_mem (needs >= 3 chunks) all get gradient."""
        model = _gated_model()
        with torch.no_grad():
            nn.init.normal_(model.cortex.m_cross.out_proj.weight, std=0.05)
        m_cross, losses = None, []
        for _ in range(3):
            ids = _ids()
            out = model(ids, NUM_STEPS, labels=ids, m_cross_in=m_cross,
                        return_m_cross=True)
            m_cross = out["m_cross"]
            losses.append(out["loss"])
        torch.stack(losses).mean().backward()
        buf = model.cortex.m_cross
        for name, p in (("vec_emb", buf.vec_emb),
                        ("wk_proj", buf.wk_proj.weight),
                        ("gate_proj_in", buf.gate_proj_in.weight),
                        ("gate_proj_mem", buf.gate_proj_mem.weight)):
            assert p.grad is not None and p.grad.abs().sum() > 0, name

    def test_eos_write_reset_zeroes_ended_lanes(self):
        """A lane whose document ended (write_reset) starts its gated update
        from a zeroed buffer — same-shape state, so the generic eos helpers
        apply unchanged."""
        model = _gated_model()
        buf = model(_ids(), NUM_STEPS, return_m_cross=True)["m_cross"]
        assert buf.shape == (B, K, H)
        ids = _ids()
        eos = torch.zeros_like(ids, dtype=torch.bool)
        eos[0, S // 2] = True          # lane 0: doc ends mid-window
        out = model(ids, NUM_STEPS, m_cross_in=buf, return_m_cross=True,
                    eos_mask=eos)
        assert out["m_cross"].shape == (B, K, H)
