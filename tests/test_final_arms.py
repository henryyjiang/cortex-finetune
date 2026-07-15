"""
Tests for the final-arms round (2026-07-14 plan, post-signal-round-null):

  AccumCCoT   — AutoCompressor-style accumulating multi-vector carry (Arm A)
  ccot_iter   — per-position Coconut carry across loop iterations (dense 2x2)
  chunking    — randomized segmenting + stop-gradient-after-N helpers
                (imported by train.py's cortex_fwd_bwd — real code, no mirror)

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

from cortex_memory.buffers import AccumCCoT
from cortex_memory.chunking import detach_old_vecs, random_chunk_sizes
from test_cortex_graft import B, H, S, FakeRaven, _ids

NV = 3   # accum_vecs in these tests


# ---------------------------------------------------------------------------
# AccumCCoT module
# ---------------------------------------------------------------------------

class TestAccumCCoTModule:

    def _mod(self, max_vecs=12):
        return AccumCCoT(H, n_vec=NV, n_heads=4, max_vecs=max_vecs)

    def test_extract_shape(self):
        m = self._mod()
        vecs = m.extract(torch.randn(B, S, H))
        assert vecs.shape == (B, NV, H)

    def test_append_accumulates_and_caps(self):
        m = self._mod(max_vecs=2 * NV)
        v = m.extract(torch.randn(B, S, H))
        s1 = m.append(None, v)
        assert s1.shape == (B, NV, H)
        s2 = m.append(s1, v)
        assert s2.shape == (B, 2 * NV, H)
        s3 = m.append(s2, v)                      # over the cap → FIFO trim
        assert s3.shape == (B, 2 * NV, H)
        # oldest rows dropped: s3 == rows [NV:] of the uncapped concat
        assert torch.allclose(s3[:, :NV], s2[:, NV:])

    def test_read_zero_init_noop(self):
        m = self._mod()
        state = m.append(None, m.extract(torch.randn(B, S, H)))
        delta = m.read(torch.randn(B, S, H), state)
        assert torch.all(delta == 0)

    def test_read_active_after_out_proj_init(self):
        m = self._mod()
        nn.init.normal_(m.out_proj.weight, std=0.05)
        state = m.append(None, m.extract(torch.randn(B, S, H)))
        delta = m.read(torch.randn(B, S, H), state)
        assert delta.shape == (B, S, H) and delta.norm() > 0

    def test_pool_mask_restricts_extraction(self):
        """Vectors must not change when masked-out positions change."""
        m = self._mod()
        h = torch.randn(B, S, H)
        mask = torch.zeros(B, S, dtype=torch.bool)
        mask[:, : S // 2] = True                  # only the first half is poolable
        v1 = m.extract(h, pool_mask=mask)
        h2 = h.clone()
        h2[:, S // 2:] += torch.randn_like(h2[:, S // 2:])
        v2 = m.extract(h2, pool_mask=mask)
        assert torch.allclose(v1, v2, atol=1e-6)

    def test_vectors_are_slot_distinct(self):
        m = self._mod()
        vecs = m.extract(torch.randn(B, S, H))
        for i in range(NV):
            for j in range(i + 1, NV):
                assert not torch.allclose(vecs[:, i], vecs[:, j])


# ---------------------------------------------------------------------------
# AccumCCoT through the graft (FakeRaven chain)
# ---------------------------------------------------------------------------

def _accum_model():
    return FakeRaven(use_memory=True, memory_slots=0, accum_ccot=True,
                     accum_vecs=NV, accum_max=12)


class TestAccumGraft:

    def test_build_and_precedence(self):
        # accum_ccot takes precedence over ccot_direct at K=0
        model = FakeRaven(use_memory=True, memory_slots=0, accum_ccot=True,
                          ccot_direct=True, accum_vecs=NV, accum_max=12)
        assert model.cortex.accum is not None
        assert model.cortex.ccot_direct is None
        assert model.cortex.has_cross_state

    def test_state_grows_across_chain(self):
        model = _accum_model()
        ids = _ids()
        s1 = model(ids, (0, 1), return_m_cross=True)["m_cross"]
        assert s1.shape == (B, NV, H)
        s2 = model(ids, (0, 1), m_cross_in=s1, return_m_cross=True)["m_cross"]
        assert s2.shape == (B, 2 * NV, H)
        # accumulation, not overwrite: the first chunk's vectors survive
        assert torch.allclose(s2[:, :NV], s1)

    def test_zero_init_carry_is_noop(self):
        model = _accum_model()
        ids = _ids()
        state = torch.randn(B, 2 * NV, H)
        torch.manual_seed(3); a = model(ids, (0, 2), m_cross_in=state)
        torch.manual_seed(3); b = model(ids, (0, 2), m_cross_in=None)
        assert torch.allclose(a["logits"], b["logits"])

    def test_carry_changes_logits_when_active(self):
        model = _accum_model()
        nn.init.normal_(model.cortex.accum.out_proj.weight, std=0.05)
        ids = _ids()
        state = model(ids, (0, 2), return_m_cross=True)["m_cross"].detach()
        torch.manual_seed(4); a = model(ids, (0, 2), m_cross_in=state)
        torch.manual_seed(4); b = model(ids, (0, 2), m_cross_in=None)
        assert not torch.allclose(a["logits"], b["logits"])

    def test_write_grad_through_chain(self):
        model = _accum_model()
        nn.init.normal_(model.cortex.accum.out_proj.weight, std=0.05)
        ids, labels = _ids(), _ids()
        out1 = model(ids, (0, 2), return_m_cross=True)
        out2 = model(ids, (0, 2), labels=labels, m_cross_in=out1["m_cross"])
        out2["loss"].backward()
        g = model.cortex.accum.vec_emb.grad
        assert g is not None and g.norm() > 0


# ---------------------------------------------------------------------------
# ccot_iter (per-position Coconut carry across loop iterations)
# ---------------------------------------------------------------------------

def _ci_model():
    return FakeRaven(use_memory=True, memory_slots=0, ccot_iter=True)


class TestCCoTIter:

    def test_build_no_cross_state(self):
        model = _ci_model()
        assert model.cortex.ccot_iter is not None
        # dense within-window memory only — nothing crosses segments
        assert not model.cortex.has_cross_state
        assert model(_ids(), (0, 2), return_m_cross=True)["m_cross"] is None

    def test_zero_init_noop(self):
        """in_proj zero-init → removing the module changes nothing at step 0."""
        model = _ci_model()
        ids = _ids()
        torch.manual_seed(5); a = model(ids, (0, 2))
        mod = model.cortex.ccot_iter
        model.cortex.ccot_iter = None
        torch.manual_seed(5); b = model(ids, (0, 2))
        model.cortex.ccot_iter = mod
        assert torch.allclose(a["logits"], b["logits"])

    def test_active_after_read_init(self):
        model = _ci_model()
        nn.init.normal_(model.cortex.ccot_iter.in_proj.weight, std=0.05)
        ids = _ids()
        torch.manual_seed(6); a = model(ids, (0, 2))
        mod = model.cortex.ccot_iter
        model.cortex.ccot_iter = None
        torch.manual_seed(6); b = model(ids, (0, 2))
        model.cortex.ccot_iter = mod
        assert not torch.allclose(a["logits"], b["logits"])

    def test_first_iteration_reads_nothing(self):
        """With a single loop iteration there is no previous write to read —
        the module must be inert even with an active in_proj (Coconut: the
        first forward has no latent thought to consume)."""
        model = _ci_model()
        nn.init.normal_(model.cortex.ccot_iter.in_proj.weight, std=0.05)
        ids = _ids()
        torch.manual_seed(7); a = model(ids, (0, 1))
        mod = model.cortex.ccot_iter
        model.cortex.ccot_iter = None
        torch.manual_seed(7); b = model(ids, (0, 1))
        model.cortex.ccot_iter = mod
        assert torch.allclose(a["logits"], b["logits"])

    def test_per_position_causality(self):
        """FakeRaven's layers are position-local, so with a per-position carry
        a perturbation at the last position must not move earlier positions'
        logits.  A pooled (leaky) carry would fail this."""
        model = _ci_model()
        nn.init.normal_(model.cortex.ccot_iter.in_proj.weight, std=0.05)
        ids = _ids()
        ids2 = ids.clone()
        ids2[:, -1] = (ids2[:, -1] + 1) % 200
        torch.manual_seed(8); a = model(ids, (0, 3))
        torch.manual_seed(8); b = model(ids2, (0, 3))
        assert torch.allclose(a["logits"][:, :-1], b["logits"][:, :-1], atol=1e-6)
        assert not torch.allclose(a["logits"][:, -1], b["logits"][:, -1])

    def test_mutually_exclusive_with_m_iter_allowed_at_graft(self):
        """The graft itself allows coexistence (train.py asserts them apart);
        both modules build and forward runs."""
        model = FakeRaven(use_memory=True, memory_slots=0, ccot_iter=True,
                          memory_slots_iter=2)
        out = model(_ids(), (0, 2))
        assert out["logits"].shape == (B, S, 200)


# ---------------------------------------------------------------------------
# chunking helpers (imported by train.py — real code under test)
# ---------------------------------------------------------------------------

class TestChunking:

    def test_random_sizes_sum_and_bounds(self):
        g = torch.Generator().manual_seed(0)
        for _ in range(50):
            sizes = random_chunk_sizes(4096, 4, generator=g)
            assert len(sizes) == 4
            assert sum(sizes) == 4096
            assert all(s > 0 for s in sizes)
            base = 4096 // 4
            assert all(base // 2 <= s <= 3 * base // 2 + 1 for s in sizes)

    def test_single_chunk_passthrough(self):
        assert random_chunk_sizes(4096, 1) == [4096]

    def test_jitter_actually_varies(self):
        g = torch.Generator().manual_seed(0)
        draws = {tuple(random_chunk_sizes(4096, 4, generator=g)) for _ in range(20)}
        assert len(draws) > 1

    def test_detach_passthrough(self):
        assert detach_old_vecs(None, NV, 2) is None
        s = torch.randn(B, 2 * NV, H, requires_grad=True)
        assert detach_old_vecs(s, NV, 0) is s          # 0 = full BPTT
        assert detach_old_vecs(s, NV, 2) is s          # nothing older than horizon

    def test_detach_slices_gradient(self):
        """Only the newest grad_chunks × n_vec rows may pass gradient."""
        c1 = torch.randn(B, NV, H, requires_grad=True)
        c2 = torch.randn(B, NV, H, requires_grad=True)
        c3 = torch.randn(B, NV, H, requires_grad=True)
        state = torch.cat([c1, c2, c3], dim=1)
        out = detach_old_vecs(state, NV, grad_chunks=1)
        assert out.shape == state.shape
        assert torch.allclose(out, state)              # values unchanged
        out.sum().backward()
        # detached rows contribute exactly-zero gradient to their sources
        # (autograd routes zeros through the cat, so .grad is 0, not None)
        assert c1.grad is None or torch.all(c1.grad == 0)
        assert c2.grad is None or torch.all(c2.grad == 0)
        assert c3.grad is not None and c3.grad.norm() > 0
