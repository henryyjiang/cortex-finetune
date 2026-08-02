"""
Phase-1 integration tests for the cortex_graft hooks.

These validate the GRAFT LOGIC (read/write timing, M_cross carry across calls,
M_iter causality, EOS masking, the disabled-is-a-no-op guarantee) against a
minimal fake model whose iterate_forward / core_block_forward / forward mirror
the EXACT hook sequence applied to raven_modeling_minimal_{olmo,llama}.py.  No
OLMo/Llama weights or RavenConfig needed.

A real 1B-checkpoint smoke test (use_memory=False reproduces published logits)
runs separately once a checkpoint is downloaded — see cortex_migration_plan.md
Phase-1 checklist.

Run: /c/Users/henry/miniconda3/envs/cortex/python.exe -m pytest tests/ -v
"""
from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from cortex_graft import CortexMemory, memory_enabled

H = 32
K = 4
B = 2
S = 16
VOCAB = 200


class FakeRaven(nn.Module):
    """Mirrors RavenForCausalLM's prelude/adapter/core_block/coda flow and the
    cortex hook sequence.  The graft edits to the real model files reproduce
    exactly the `if self.cortex is not None: ...` calls used here."""

    def __init__(self, **mem_flags):
        super().__init__()
        config = SimpleNamespace(n_embd=H, **mem_flags)
        self.config = config
        self.wte        = nn.Embedding(VOCAB, H)
        self.adapter    = nn.Linear(H * 2, H)
        self.core_block = nn.ModuleList([nn.Linear(H, H) for _ in range(2)])
        self.coda       = nn.ModuleList([nn.Linear(H, H) for _ in range(1)])
        self.lm_head    = nn.Linear(H, VOCAB, bias=False)
        self.cortex = CortexMemory(config) if memory_enabled(config) else None

    def initialize_state(self, input_embeds):
        return torch.randn_like(input_embeds)

    def core_block_forward(self, x, input_embeds):
        x = self.adapter(torch.cat([x, input_embeds], dim=-1))
        if self.cortex is not None:                       # ← graft hook (read)
            x = self.cortex.read_into(x)
        for block in self.core_block:
            x = block(x)
        if self.cortex is not None:                       # ← graft hook (M_iter write)
            self.cortex.iter_write(x)
        return x

    def iterate_forward(self, input_embeds, num_steps):
        n, k = num_steps
        x = self.initialize_state(input_embeds)
        with torch.no_grad():
            for _ in range(n):
                x = self.core_block_forward(x, input_embeds)
        for _ in range(k):
            x = self.core_block_forward(x, input_embeds)
        return x

    def forward(self, input_ids, num_steps, labels=None,
                m_cross_in=None, return_m_cross=False, eos_mask=None):
        input_embeds = self.wte(input_ids)
        n_prefix = n_summary = 0
        if self.cortex is not None:                       # ← graft hook (begin)
            self.cortex.begin(m_cross_in, eos_mask, input_ids.shape[1],
                              input_ids.device, input_embeds.dtype)
            # ← graft hook (summary seeding, once, from the loaded wte)
            if (self.cortex.prefix is not None
                    and not bool(self.cortex.prefix.summary_seeded)):
                self.cortex.init_summary_from_embedding(self.wte.weight)
            # ← graft hook (prefix splice).  emb_scale is 1 here; the real
            # model passes self.emb_scale.
            position_ids = torch.arange(input_ids.shape[1]).unsqueeze(0)
            input_embeds, _, n_prefix, n_summary = self.cortex.prefix_pack(
                input_embeds, position_ids, 1.0)
        x = self.iterate_forward(input_embeds, num_steps)
        h_T = x
        new_m_cross = None
        # ← graft hook (M_cross write); prefix memory writes after the coda
        if self.cortex is not None and self.cortex.prefix is None:
            new_m_cross = self.cortex.cross_write(h_T)
        for block in self.coda:
            x = block(x)
        if self.cortex is not None and self.cortex.prefix is not None:
            # ← graft hook (prefix unpack: summary states ARE the new carry)
            x, new_m_cross = self.cortex.prefix_unpack(x, n_prefix, n_summary)
        logits = self.lm_head(x).float()
        out = {"logits": logits}
        if labels is not None:
            out["loss"] = nn.functional.cross_entropy(
                logits.reshape(-1, VOCAB), labels.reshape(-1), ignore_index=-100)
        if return_m_cross:
            out["m_cross"] = new_m_cross
        return out


class CausalBlock(nn.Module):
    """Single-head causal self-attention + a linear."""

    def __init__(self) -> None:
        super().__init__()
        self.qkv = nn.Linear(H, 3 * H, bias=False)
        self.weight = nn.Parameter(torch.zeros(1))   # so freeze selectors match
        self.out = nn.Linear(H, H)

    def forward(self, x):
        # Written out rather than via scaled_dot_product_attention, whose CPU
        # kernels are not available for every dtype/shape these tests use.
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        scores = (q @ k.transpose(-2, -1)) / (H ** 0.5)
        causal = torch.ones(x.shape[1], x.shape[1], dtype=torch.bool).tril()
        scores = scores.masked_fill(~causal, torch.finfo(scores.dtype).min)
        return self.out(x + nn.functional.softmax(scores, dim=-1) @ v)


class AttnRaven(FakeRaven):
    """FakeRaven with real causal attention in the loop and coda.

    FakeRaven is deliberately POSITION-WISE, which is fine for the buffers that
    inject an additive delta but useless for prefix memory: its read IS the base
    model's attention over the prepended vectors, so in a position-wise model
    the carry columns can never reach a real token.  A "carry changes the
    logits" assertion would then pass for the wrong reason — a longer packed
    sequence draws a different initialize_state, which moves the logits all by
    itself.  Prefix tests use this subclass so the mixing is genuine.

    It is a SUBCLASS rather than a change to FakeRaven because two EOS tests and
    the loop-freeze tests are written against position-wise behaviour.
    """

    def __init__(self, **mem_flags):
        super().__init__(**mem_flags)
        self.core_block = nn.ModuleList([CausalBlock() for _ in range(2)])
        self.coda       = nn.ModuleList([CausalBlock() for _ in range(1)])


def _ids(b=B, s=S):
    return torch.randint(0, VOCAB, (b, s))


def _activate_cross_read(model, std=0.05):
    with torch.no_grad():
        if model.cortex.m_cross is not None:
            nn.init.normal_(model.cortex.m_cross.out_proj.weight, std=std)
        if model.cortex.ccot_direct is not None:
            nn.init.normal_(model.cortex.ccot_direct.in_proj.weight, std=std)


def _activate_iter_read(model, std=0.05):
    with torch.no_grad():
        nn.init.normal_(model.cortex.m_iter.out_proj.weight, std=std)


# ---------------------------------------------------------------------------
# Disabled = exact no-op
# ---------------------------------------------------------------------------

class TestDisabled:

    def test_default_off(self):
        model = FakeRaven()
        assert model.cortex is None

    def test_deterministic_when_off(self):
        model = FakeRaven()
        ids = _ids()
        torch.manual_seed(0); a = model(ids, (0, 2))
        torch.manual_seed(0); b = model(ids, (0, 2))
        assert torch.allclose(a["logits"], b["logits"])

    def test_no_m_cross_when_off(self):
        model = FakeRaven()
        out = model(_ids(), (0, 2), return_m_cross=True)
        assert out.get("m_cross") is None


# ---------------------------------------------------------------------------
# M_cross (LM2 buffer)
# ---------------------------------------------------------------------------

class TestMCross:

    def test_enabled(self):
        model = FakeRaven(use_memory=True, memory_slots=K)
        assert model.cortex.m_cross is not None
        assert model.cortex.has_cross_state

    def test_output_shape(self):
        model = FakeRaven(use_memory=True, memory_slots=K)
        out = model(_ids(), (0, 1), return_m_cross=True)
        assert out["m_cross"].shape == (B, K, H)

    def test_carry_changes_next_call(self):
        model = FakeRaven(use_memory=True, memory_slots=K)
        _activate_cross_read(model)
        ids = _ids()
        mc = model(ids, (0, 2), return_m_cross=True)["m_cross"].detach()
        torch.manual_seed(1); a = model(ids, (0, 2), m_cross_in=mc)
        torch.manual_seed(1); b = model(ids, (0, 2), m_cross_in=None)
        assert not torch.allclose(a["logits"], b["logits"])

    def test_zero_carry_is_noop_at_init(self):
        """out_proj zero-init → carrying a buffer changes nothing at step 0."""
        model = FakeRaven(use_memory=True, memory_slots=K)
        ids = _ids()
        mc = torch.randn(B, K, H)
        torch.manual_seed(2); a = model(ids, (0, 2), m_cross_in=mc)
        torch.manual_seed(2); b = model(ids, (0, 2), m_cross_in=None)
        assert torch.allclose(a["logits"], b["logits"])

    def test_write_grad_through_chain(self):
        model = FakeRaven(use_memory=True, memory_slots=K)
        _activate_cross_read(model)
        ids, labels = _ids(), _ids()
        out1 = model(ids, (0, 2), return_m_cross=True)
        out2 = model(ids, (0, 2), labels=labels, m_cross_in=out1["m_cross"])  # no detach
        out2["loss"].backward()
        g = model.cortex.m_cross.gate_proj_in.weight.grad
        assert g is not None and g.norm() > 0


# ---------------------------------------------------------------------------
# M_iter (per-position short-term)
# ---------------------------------------------------------------------------

class TestMIter:

    def test_enabled(self):
        model = FakeRaven(use_memory=True, memory_slots_iter=K)
        assert model.cortex.m_iter is not None
        assert not model.cortex.has_cross_state          # M_iter is not cross-state

    def test_not_in_output(self):
        model = FakeRaven(use_memory=True, memory_slots_iter=K)
        out = model(_ids(), (0, 2), return_m_cross=True)
        assert out.get("m_cross") is None

    def test_more_iterations_changes_output(self):
        model = FakeRaven(use_memory=True, memory_slots_iter=K)
        _activate_iter_read(model)
        ids = _ids()
        torch.manual_seed(0); o1 = model(ids, (0, 1))
        torch.manual_seed(0); o3 = model(ids, (0, 3))
        assert not torch.allclose(o1["logits"], o3["logits"])

    def test_resets_between_calls(self):
        model = FakeRaven(use_memory=True, memory_slots_iter=K)
        _activate_iter_read(model)
        ids = _ids()
        torch.manual_seed(7); a = model(ids, (0, 2))
        torch.manual_seed(7); b = model(ids, (0, 2))
        assert torch.allclose(a["logits"], b["logits"])

    def test_causal(self):
        """Perturbing the last token must not change earlier-position logits."""
        model = FakeRaven(use_memory=True, memory_slots_iter=K)
        _activate_iter_read(model)
        a = _ids(b=1)
        b = a.clone(); b[0, -1] = (b[0, -1] + 1) % VOCAB
        torch.manual_seed(11); oa = model(a, (0, 3))
        torch.manual_seed(11); ob = model(b, (0, 3))
        assert torch.allclose(oa["logits"][0, :-1], ob["logits"][0, :-1], atol=1e-5)


# ---------------------------------------------------------------------------
# EOS handling through the real loop
# ---------------------------------------------------------------------------

class TestEOS:

    E = 7

    def test_no_eos_is_noop(self):
        model = FakeRaven(use_memory=True, memory_slots=K)
        ids = _ids()
        no_eos = torch.zeros(B, S, dtype=torch.bool)
        torch.manual_seed(2); a = model(ids, (0, 2), return_m_cross=True)
        torch.manual_seed(2); b = model(ids, (0, 2), return_m_cross=True, eos_mask=no_eos)
        assert torch.allclose(a["logits"], b["logits"])
        assert torch.allclose(a["m_cross"], b["m_cross"])

    def test_write_pools_only_open_suffix(self):
        model = FakeRaven(use_memory=True, memory_slots=K)
        e = self.E
        eos = torch.zeros(1, S, dtype=torch.bool); eos[0, e] = True
        a = _ids(b=1)
        b = a.clone(); b[0, :e + 1] = (b[0, :e + 1] + 17) % VOCAB   # change ended-doc prefix
        torch.manual_seed(4); mca = model(a, (0, 2), return_m_cross=True, eos_mask=eos)["m_cross"]
        torch.manual_seed(4); mcb = model(b, (0, 2), return_m_cross=True, eos_mask=eos)["m_cross"]
        assert torch.allclose(mca, mcb, atol=1e-5)

    def test_read_masked_after_first_eos(self):
        model = FakeRaven(use_memory=True, memory_slots=K)
        _activate_cross_read(model)
        e = self.E
        ids = _ids(b=1)
        eos = torch.zeros(1, S, dtype=torch.bool); eos[0, e] = True
        carried = torch.randn(1, K, H)
        torch.manual_seed(8); ob = model(ids, (0, 2), m_cross_in=carried, eos_mask=eos)
        torch.manual_seed(8); on = model(ids, (0, 2), m_cross_in=None, eos_mask=eos)
        # continuing doc (<= first EOS) gets the injection → differs
        assert not torch.allclose(ob["logits"][0, :e + 1], on["logits"][0, :e + 1])
        # fresh docs (> first EOS) read nothing → identical
        assert torch.allclose(ob["logits"][0, e + 1:], on["logits"][0, e + 1:], atol=1e-5)

    def test_eos_at_last_carries_zero(self):
        model = FakeRaven(use_memory=True, memory_slots=K)
        ids = _ids(b=1)
        eos = torch.zeros(1, S, dtype=torch.bool); eos[0, S - 1] = True
        mc = model(ids, (0, 2), return_m_cross=True, eos_mask=eos)["m_cross"]
        assert torch.allclose(mc, torch.zeros_like(mc))


# ---------------------------------------------------------------------------
# Both buffers together
# ---------------------------------------------------------------------------

class TestBoth:

    def test_both_active_runs_and_grads(self):
        model = FakeRaven(use_memory=True, memory_slots=K, memory_slots_iter=K)
        _activate_cross_read(model)
        _activate_iter_read(model)
        ids, labels = _ids(), _ids()
        out1 = model(ids, (0, 3), return_m_cross=True)
        out2 = model(ids, (0, 3), labels=labels, m_cross_in=out1["m_cross"])
        out2["loss"].backward()
        assert out1["m_cross"].shape == (B, K, H)
        assert model.cortex.m_cross.gate_proj_in.weight.grad is not None
        assert model.cortex.m_iter.gate_proj_in.weight.grad is not None
