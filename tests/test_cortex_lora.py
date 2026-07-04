"""
LoRA-on-loop (rung 1b) tests: build_loop_lora / LoopLoRA against FakeRaven.

The graft contract mirrored here: the grafted RavenForCausalLM.__init__ calls
`self.cortex_lora = build_loop_lora(self, config)` after the transformer is
built; hooks add (alpha/r) * B(Ax) to every loop linear; B is zero-init so
step 0 is exactly the base model; param names avoid the 'adapter'/'core_block'
substrings so set_loop_trainable() leaves LoRA trainable while the base loop
freezes.

Run: /c/Users/henry/miniconda3/envs/cortex-retro/python.exe -m pytest tests/ -q
"""
from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from cortex_graft import build_loop_lora
from tests.test_cortex_graft import FakeRaven, _ids
from tests.test_cortex_train import _chunk_chain_backward, _set_loop_trainable


def _lora_model(rank=8, alpha=16.0, **mem_flags):
    model = FakeRaven(use_memory=True, memory_slots=4, lora_rank=rank,
                      lora_alpha=alpha, **mem_flags)
    # mirror of the grafted __init__ line
    model.cortex_lora = build_loop_lora(model, model.config)
    assert model.cortex_lora is not None
    return model


class TestLoopLoRA:

    def test_disabled_when_rank_zero(self):
        model = FakeRaven(use_memory=True, memory_slots=4)
        assert build_loop_lora(model, model.config) is None

    def test_exact_noop_at_init(self):
        """B=0 -> the hooks add exact zeros; logits bit-identical to pre-LoRA."""
        model = FakeRaven(use_memory=True, memory_slots=4, lora_rank=8, lora_alpha=16.0)
        ids = _ids()
        torch.manual_seed(0); before = model(ids, (0, 2))["logits"]
        model.cortex_lora = build_loop_lora(model, model.config)
        torch.manual_seed(0); after = model(ids, (0, 2))["logits"]
        assert torch.equal(before, after)

    def test_nonzero_B_changes_output(self):
        model = _lora_model()
        ids = _ids()
        torch.manual_seed(0); base = model(ids, (0, 2))["logits"]
        with torch.no_grad():
            for B in model.cortex_lora.B.values():
                B.normal_(std=0.05)
        torch.manual_seed(0); adapted = model(ids, (0, 2))["logits"]
        assert not torch.allclose(base, adapted)

    def test_names_dodge_loop_freeze(self):
        """set_loop_trainable freezes by 'adapter'/'core_block' substrings —
        LoRA param names must contain neither, and must contain 'cortex' (for
        the Adam-side / memory-LR routing)."""
        model = _lora_model()
        names = [n for n, _ in model.named_parameters() if "cortex_lora" in n]
        assert names, "LoRA params not registered on the model"
        for n in names:
            assert "adapter" not in n and "core_block" not in n, n
            assert "cortex" in n
        _set_loop_trainable(model, trainable=False)
        for n, p in model.named_parameters():
            if "cortex_lora" in n:
                assert p.requires_grad, f"{n} was frozen by set_loop_trainable"

    def test_lora_trains_while_base_loop_frozen(self):
        """The rung-1b configuration: base loop frozen, LoRA + memory train.
        At B=0 only B gets gradient (A's grad flows through B); after B moves,
        A trains too."""
        model = _lora_model()
        _set_loop_trainable(model, trainable=False)
        model.zero_grad()
        _chunk_chain_backward(model, _ids(s=24), _ids(s=24), n_chunks=3)
        assert model.adapter.weight.grad is None            # base loop frozen
        B0 = next(iter(model.cortex_lora.B.values()))
        assert B0.grad is not None and B0.grad.norm() > 0   # LoRA trains
        # memory still trains alongside
        assert model.cortex.m_cross.gate_proj_in.weight.grad is not None

        with torch.no_grad():
            for B in model.cortex_lora.B.values():
                B.normal_(std=0.05)
        model.zero_grad()
        _chunk_chain_backward(model, _ids(s=24), _ids(s=24), n_chunks=3)
        A0 = next(iter(model.cortex_lora.A.values()))
        assert A0.grad is not None and A0.grad.norm() > 0

    def test_state_dict_roundtrip(self):
        """Resume/eval contract: a fresh model+LoRA loads a trained state dict
        and reproduces outputs (hooks are rebuilt by construction)."""
        src = _lora_model()
        with torch.no_grad():
            for B in src.cortex_lora.B.values():
                B.normal_(std=0.05)
        ids = _ids()
        torch.manual_seed(0); want = src(ids, (0, 2))["logits"]

        dst = _lora_model()
        dst.load_state_dict(src.state_dict())
        torch.manual_seed(0); got = dst(ids, (0, 2))["logits"]
        assert torch.allclose(want, got)
