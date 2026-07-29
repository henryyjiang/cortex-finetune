"""
B0 — retrofit wiring tests on the REAL raven modeling file (two-track plan
§B0): the memory candidates (AccumCCoT / GatedAccumBuffer k=32) grafted into
a tiny real RavenForCausalLM, exercising the actual iterate_forward /
core_block_forward hook path the conversion training will run.

The load-bearing invariant is the parity-gate mechanism: with zero-init
memory reads, a memory-ON model with the SAME base weights produces
bit-identical logits to memory-OFF at step 0 — so the memory-augmented
conversion starts ON the McLeish loss curve, and any early divergence in the
B0 run is a wiring bug, not recurrence instability.

Reuses the tiny-raven builder from test_cortex_eval (skips cleanly on
transformers skew).

Run: /c/Users/henry/miniconda3/envs/cortex-retro/python.exe -m pytest tests/test_b0_retrofit.py -v
"""
from __future__ import annotations

import os
import sys

import pytest
import torch
import torch.nn as nn

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from test_cortex_eval import _build_raven, VOCAB

NS = torch.tensor([2, 1])          # small fixed recurrence for determinism


def _ids(s=32):
    torch.manual_seed(99)
    return torch.randint(1, VOCAB, (1, s))


def _copy_base_weights(dst, src):
    """Overlay src's base weights onto dst (cortex.* keys stay dst's own —
    exactly how from_pretrained loads a memory-free checkpoint, strict=False)."""
    missing, unexpected = dst.load_state_dict(src.state_dict(), strict=False)
    assert not unexpected
    assert all(k.startswith("cortex") for k in missing)


@pytest.fixture(scope="module")
def base():
    torch.manual_seed(0)
    return _build_raven(use_memory=False)


class TestStep0Parity:

    def _parity(self, base, **mem_flags):
        mem = _build_raven(use_memory=True, **mem_flags)
        _copy_base_weights(mem, base)
        ids = _ids()
        torch.manual_seed(7)
        out_base = base(input_ids=ids, num_steps=NS)
        torch.manual_seed(7)
        out_mem = mem(input_ids=ids, num_steps=NS, return_m_cross=True)
        assert torch.equal(out_base["logits"], out_mem["logits"])
        return mem, ids, out_mem

    def test_accum_step0_equals_base(self, base):
        """B0 gate mechanism, accum candidate: zero-init read => identical
        logits, even WITH a carried state from a previous chunk."""
        mem, ids, out1 = self._parity(base, memory_slots=0, accum_ccot=True,
                                      accum_vecs=4, accum_max=64)
        # second chunk WITH the carried state must still match the base
        torch.manual_seed(8)
        out_base2 = base(input_ids=ids, num_steps=NS)
        torch.manual_seed(8)
        out_mem2 = mem(input_ids=ids, num_steps=NS,
                       m_cross_in=out1["m_cross"], return_m_cross=True)
        assert torch.equal(out_base2["logits"], out_mem2["logits"])

    def test_gated32_step0_equals_base(self, base):
        """B0 gate mechanism, gated k=32 candidate."""
        mem, ids, out1 = self._parity(base, memory_slots=32, gated_accum=True)
        assert out1["m_cross"].shape[1] == 32
        torch.manual_seed(8)
        out_base2 = base(input_ids=ids, num_steps=NS)
        torch.manual_seed(8)
        out_mem2 = mem(input_ids=ids, num_steps=NS,
                       m_cross_in=out1["m_cross"], return_m_cross=True)
        assert torch.equal(out_base2["logits"], out_mem2["logits"])


class TestRealForwardChain:

    def test_accum_state_grows_through_iterate_forward(self, base):
        mem = _build_raven(use_memory=True, memory_slots=0, accum_ccot=True,
                           accum_vecs=4, accum_max=64)
        m_cross = None
        for chunk in range(3):
            out = mem(input_ids=_ids(), num_steps=NS,
                      m_cross_in=m_cross, return_m_cross=True)
            m_cross = out["m_cross"]
            assert m_cross.shape[1] == 4 * (chunk + 1)

    def test_cotraining_grads_reach_loop_and_memory(self, base):
        """The Track-B point: backbone fully plastic + memory co-trained.
        A 3-chunk chain must deliver gradient BOTH to the extraction params
        (vec_emb) and to the recurrent core (core_block), in one graph."""
        mem = _build_raven(use_memory=True, memory_slots=0, accum_ccot=True,
                           accum_vecs=4, accum_max=64).train()
        with torch.no_grad():
            nn.init.normal_(mem.cortex.accum.out_proj.weight, std=0.05)
        m_cross, losses = None, []
        for _ in range(3):
            ids = _ids()
            out = mem(input_ids=ids, num_steps=NS, labels=ids,
                      m_cross_in=m_cross, return_m_cross=True)
            m_cross = out["m_cross"]
            losses.append(out["loss"])
        torch.stack(losses).mean().backward()
        assert mem.cortex.accum.vec_emb.grad is not None \
            and mem.cortex.accum.vec_emb.grad.abs().sum() > 0
        core = [p for n, p in mem.named_parameters()
                if "core_block" in n and p.grad is not None
                and p.grad.abs().sum() > 0]
        assert core, "no core_block param received gradient — loop not plastic"

    def test_cotraining_grads_reach_loop_and_memory_gated(self, base):
        """Same co-training invariant for the GatedAccumBuffer candidate.

        accum had this coverage and gated did not (noted 2026-07-29 while
        picking the B1 arm).  gated is wired as MEMORY_MODE=gated and is the
        planned third arm, so the gap is worth closing before it runs.

        Two gated-specific points beyond the accum version:
          * the buffer lives on cortex.m_cross (LSTMBuffer's slot), not
            cortex.accum, so it travels a different graft path;
          * gate_proj_mem is the memory-FEEDBACK side and only receives
            gradient through a chain of >= 3 segments (documented on both
            GatedAccumBuffer and LSTMBuffer).  The 3-chunk chain here is
            exactly that minimum, so a zero grad would mean the feedback gate
            never trains — which is why carry_grad_chunks stays 0 for gated.
        """
        mem = _build_raven(use_memory=True, memory_slots=32,
                           gated_accum=True).train()
        buf = mem.cortex.m_cross
        with torch.no_grad():
            nn.init.normal_(buf.out_proj.weight, std=0.05)
        m_cross, losses = None, []
        for _ in range(3):
            ids = _ids()
            out = mem(input_ids=ids, num_steps=NS, labels=ids,
                      m_cross_in=m_cross, return_m_cross=True)
            m_cross = out["m_cross"]
            losses.append(out["loss"])
        torch.stack(losses).mean().backward()

        assert buf.vec_emb.grad is not None and buf.vec_emb.grad.abs().sum() > 0, \
            "extraction queries received no gradient"
        assert buf.gate_proj_mem.weight.grad is not None \
            and buf.gate_proj_mem.weight.grad.abs().sum() > 0, \
            "memory-feedback gate received no gradient over a 3-segment chain"
        core = [p for n, p in mem.named_parameters()
                if "core_block" in n and p.grad is not None
                and p.grad.abs().sum() > 0]
        assert core, "no core_block param received gradient — loop not plastic"

    def test_gated_state_is_fixed_size_through_iterate_forward(self, base):
        """The append-vs-gated contrast, asserted: accum's state GROWS by
        accum_vecs per chunk (test above), gated's stays at n_slots."""
        mem = _build_raven(use_memory=True, memory_slots=32, gated_accum=True)
        m_cross = None
        for _ in range(3):
            out = mem(input_ids=_ids(), num_steps=NS,
                      m_cross_in=m_cross, return_m_cross=True)
            m_cross = out["m_cross"]
            assert m_cross.shape[1] == 32
