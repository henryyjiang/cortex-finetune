"""
Phase-2 tests: the cross-chunk segment chain + loop-freeze logic.

train.py hard-fails at import on a no-CUDA box (device health check), so these
replicate the EXACT logic of train.py's `cortex_fwd_bwd` (chunk → carry M_cross
un-detached → stack-mean → one backward) and `set_loop_trainable` against the
FakeRaven from test_cortex_graft, and assert the load-bearing properties:

  * the M_cross write path receives gradient ONLY through the multi-chunk chain
    (it gets none from a single chunk — exactly why cross_chunks>1 is required);
  * freezing the loop (adapter + core_block) stops loop grads while memory still
    trains.

Run: /c/Users/henry/miniconda3/envs/cortex-retro/python.exe -m pytest tests/ -q
"""
from __future__ import annotations

import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from tests.test_cortex_graft import FakeRaven, _ids, _activate_cross_read, B, K, H, S


def _chunk_chain_backward(model, input_ids, labels, n_chunks, num_steps=(0, 3),
                          eos_id=None):
    """Mirror of train.py cortex_fwd_bwd: one graph over all chunks, M_cross
    carried un-detached, single backward on the mean chunk loss.  eos_id set
    mirrors --cortex.eos_from_tokens (eos_mask derived per chunk)."""
    x_chunks = torch.chunk(input_ids, n_chunks, dim=1)
    y_chunks = torch.chunk(labels, n_chunks, dim=1)
    m_cross = None
    losses = []
    for xc, yc in zip(x_chunks, y_chunks):
        out = model(xc, num_steps, labels=yc, m_cross_in=m_cross, return_m_cross=True,
                    eos_mask=(xc == eos_id) if eos_id is not None else None)
        m_cross = out["m_cross"]                       # un-detached carry
        if (yc != -100).any():
            losses.append(out["loss"])
    total = torch.stack(losses).mean()
    total.backward()
    return total


def _set_loop_trainable(model, trainable: bool) -> int:
    """Mirror of train.py set_loop_trainable (selector: adapter / core_block)."""
    n = 0
    for name, p in model.named_parameters():
        if ("adapter" in name) or ("core_block" in name):
            p.requires_grad_(trainable)
            n += 1
    return n


class TestCrossChunkChain:

    def test_write_path_trains_through_chain(self):
        model = FakeRaven(use_memory=True, memory_slots=4)
        _activate_cross_read(model)
        model.zero_grad()
        _chunk_chain_backward(model, _ids(s=24), _ids(s=24), n_chunks=3)
        g = model.cortex.m_cross.gate_proj_in.weight.grad
        assert g is not None and g.norm() > 0, "M_cross write path got no gradient from the chain"

    def test_single_chunk_gives_write_path_no_grad(self):
        """With one chunk the written buffer is never read in the same backward,
        so the write path gets zero gradient — the reason cross_chunks>1 matters."""
        model = FakeRaven(use_memory=True, memory_slots=4)
        _activate_cross_read(model)
        model.zero_grad()
        _chunk_chain_backward(model, _ids(s=24), _ids(s=24), n_chunks=1)
        g = model.cortex.m_cross.gate_proj_in.weight.grad
        assert g is None or g.norm() == 0, "single-chunk should not train the write path"

    def test_three_chunks_train_feedback_gates(self):
        """>=3 chunks are needed for the forget/memory-feedback params (the first
        write sees a zero buffer → zero feedback-gate grad at n=2)."""
        model = FakeRaven(use_memory=True, memory_slots=4)
        _activate_cross_read(model)
        model.zero_grad()
        _chunk_chain_backward(model, _ids(s=24), _ids(s=24), n_chunks=3)
        g = model.cortex.m_cross.gate_proj_mem.weight.grad
        assert g is not None and g.norm() > 0, "memory-feedback gate not trained at n_chunks=3"


class TestEosFromTokens:
    """Mirror of --cortex.eos_from_tokens: eos_mask derived from token ids and
    passed per chunk, so padded (pad == eos) sequences write only the open
    document suffix and reset the carry once the document ends."""

    EOS = 0  # any fixed vocab id works for the fake model

    def _ids_no_eos(self, s):
        ids = _ids(s=s)
        ids[ids == self.EOS] = 1
        return ids

    def test_chain_still_trains_with_eos_mask(self):
        model = FakeRaven(use_memory=True, memory_slots=4)
        _activate_cross_read(model)
        model.zero_grad()
        ids = self._ids_no_eos(24)          # no EOS anywhere → full carry
        _chunk_chain_backward(model, ids, ids.clone(), n_chunks=3, eos_id=self.EOS)
        g = model.cortex.m_cross.gate_proj_in.weight.grad
        assert g is not None and g.norm() > 0

    def test_all_pad_chunk_zeroes_carry(self):
        """A fully-padded chunk (doc ended earlier) must hand a zero buffer to
        the next chunk instead of a pool over pad states."""
        model = FakeRaven(use_memory=True, memory_slots=4)
        ids = self._ids_no_eos(24)
        ids[:, 8:] = self.EOS               # doc ends inside chunk 1 of 3
        x_chunks = torch.chunk(ids, 3, dim=1)
        m_cross = None
        for xc in x_chunks:
            out = model(xc, (0, 2), m_cross_in=m_cross, return_m_cross=True,
                        eos_mask=(xc == self.EOS))
            m_cross = out["m_cross"]
        assert m_cross is not None and torch.all(m_cross == 0)


class TestL2SP:
    """Mirror of --cortex.l2sp_coeff: coeff * ||theta_loop - theta_base||^2 added
    to the backward objective, anchoring the unfrozen loop to its snapshot."""

    COEFF = 0.1

    def _pairs(self, model):
        return [(p, p.detach().clone()) for n, p in model.named_parameters()
                if ("adapter" in n) or ("core_block" in n)]

    def test_penalty_pulls_loop_toward_anchor(self):
        model = FakeRaven(use_memory=True, memory_slots=4)
        _activate_cross_read(model)
        pairs = self._pairs(model)
        with torch.no_grad():                    # displace the loop off the anchor
            model.adapter.weight.add_(0.5)
        model.zero_grad()
        pen = torch.stack([(p - ref).pow(2).sum() for p, ref in pairs]).sum()
        (self.COEFF * pen).backward()
        g = model.adapter.weight.grad
        # gradient of coeff*||p - ref||^2 is 2*coeff*(p - ref) = 2*0.1*0.5
        assert g is not None and torch.allclose(g, torch.full_like(g, 0.1), atol=1e-6)
        # the penalty must not touch non-loop params
        assert model.lm_head.weight.grad is None
        assert model.cortex.m_cross.gate_proj_in.weight.grad is None

    def test_penalty_is_inert_while_loop_frozen(self):
        model = FakeRaven(use_memory=True, memory_slots=4)
        _activate_cross_read(model)
        pairs = self._pairs(model)
        _set_loop_trainable(model, trainable=False)
        model.zero_grad()
        # mirror of train.py: objective = total + coeff * pen; pen is constant
        # (no grad_fn) when every loop param is frozen — backward must not fail
        x_chunks = torch.chunk(_ids(s=24), 3, dim=1)
        m_cross, losses = None, []
        for xc in x_chunks:
            out = model(xc, (0, 3), labels=xc.clone(), m_cross_in=m_cross, return_m_cross=True)
            m_cross = out["m_cross"]
            losses.append(out["loss"])
        total = torch.stack(losses).mean()
        pen = torch.stack([(p - ref).pow(2).sum() for p, ref in pairs]).sum()
        assert not pen.requires_grad
        (total + self.COEFF * pen).backward()
        assert model.adapter.weight.grad is None
        assert model.cortex.m_cross.gate_proj_in.weight.grad is not None


class TestLoopFreeze:

    def test_freeze_stops_loop_grad_keeps_memory(self):
        model = FakeRaven(use_memory=True, memory_slots=4)
        _activate_cross_read(model)
        n = _set_loop_trainable(model, trainable=False)
        assert n > 0
        model.zero_grad()
        _chunk_chain_backward(model, _ids(s=24), _ids(s=24), n_chunks=3)
        # loop (adapter + core_block) frozen → no grad
        assert model.adapter.weight.grad is None
        for blk in model.core_block:
            assert blk.weight.grad is None
        # memory still trains
        assert model.cortex.m_cross.gate_proj_in.weight.grad is not None
        # coda / head still train (not part of the loop)
        assert model.lm_head.weight.grad is not None

    def test_unfreeze_restores_loop_grad(self):
        model = FakeRaven(use_memory=True, memory_slots=4)
        _activate_cross_read(model)
        _set_loop_trainable(model, trainable=False)
        _set_loop_trainable(model, trainable=True)        # staged unfreeze
        model.zero_grad()
        _chunk_chain_backward(model, _ids(s=24), _ids(s=24), n_chunks=3)
        assert model.adapter.weight.grad is not None
        assert model.core_block[0].weight.grad is not None
