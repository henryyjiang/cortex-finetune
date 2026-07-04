"""
Phase-0 unit tests for the ported cortex_memory pillars.

These exercise the components in ISOLATION (no host model) — buffer write/read
mechanics, the EOS masking helpers, the recurrence samplers, and Muon routing.
The forward-level integration tests (causality, full EOS carry through the
grafted RavenForCausalLM) live in Phase 1's tests/test_cortex_graft.py.

Run with the cortex conda env:
  /c/Users/henry/miniconda3/envs/cortex/python.exe -m pytest tests/ -v
"""
from __future__ import annotations

import os
import sys

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from cortex_memory import (
    LSTMBuffer,
    DirectCCoT,
    LTIInjection,
    Muon,
    _zeropower_via_newtonschulz5,
    sample_num_steps,
    sample_batch_steps,
    get_current_mean_recurrence,
    enforce_mu_bwd,
    compute_eos_masks,
    apply_write_reset,
    apply_valid_write,
)

H = 32   # hidden size (divisible by n_heads=4)
K = 4    # memory slots
B = 2    # batch size
S = 16   # sequence length


def _activate_read(buf: LSTMBuffer, std: float = 0.05) -> None:
    """out_proj is zero-init → read is a no-op; give it weight to make read live."""
    with torch.no_grad():
        nn.init.normal_(buf.out_proj.weight, std=std)


def _activate_direct(d: DirectCCoT, std: float = 0.05) -> None:
    """in_proj is zero-init → carry is inert; activate it."""
    with torch.no_grad():
        nn.init.normal_(d.in_proj.weight, std=std)


# ---------------------------------------------------------------------------
# 1. LSTMBuffer
# ---------------------------------------------------------------------------

class TestLSTMBuffer:

    def test_write_shape(self):
        buf = LSTMBuffer(H, K)
        h_T = torch.randn(B, S, H)
        new = buf.write(h_T, torch.zeros(B, K, H))
        assert new.shape == (B, K, H)

    def test_read_shape(self):
        buf = LSTMBuffer(H, K)
        delta = buf.read(torch.randn(B, S, H), torch.randn(B, K, H))
        assert delta.shape == (B, S, H)

    def test_out_proj_zero_init_read_is_noop(self):
        buf = LSTMBuffer(H, K)
        delta = buf.read(torch.randn(B, S, H), torch.randn(B, K, H))
        assert torch.allclose(delta, torch.zeros_like(delta)), (
            "out_proj must be zero-init so the read injection is a no-op at step 0"
        )

    def test_forget_bias_init_positive(self):
        buf = LSTMBuffer(H, K)
        assert float(buf.forget_bias.detach()) == pytest.approx(1.0), "LM2 §3.3: forget bias init +1.0"

    def test_read_live_after_activation(self):
        buf = LSTMBuffer(H, K)
        _activate_read(buf)
        delta = buf.read(torch.randn(B, S, H), torch.randn(B, K, H))
        assert not torch.allclose(delta, torch.zeros_like(delta))

    def test_write_uses_h_T(self):
        """Different h_T → different written buffer."""
        buf = LSTMBuffer(H, K)
        b0 = torch.zeros(B, K, H)
        n1 = buf.write(torch.randn(B, S, H), b0)
        n2 = buf.write(torch.randn(B, S, H), b0)
        assert not torch.allclose(n1, n2)

    def test_per_sequence_independence(self):
        """Each sequence's write depends only on its own h_T (no batch averaging)."""
        buf = LSTMBuffer(H, K)
        h_T = torch.randn(B, S, H)
        new = buf.write(h_T, torch.zeros(B, K, H))
        assert not torch.allclose(new[0], new[1])

    def test_slots_are_distinct(self):
        """Slot-query cross-attention + slot_emb residual → K distinct slots
        after a write (guards the K=4 > K=1 requirement, framework §4.4)."""
        buf = LSTMBuffer(H, K)
        new = buf.write(torch.randn(B, S, H), torch.zeros(B, K, H))
        for i in range(K):
            for j in range(i + 1, K):
                assert not torch.allclose(new[0, i], new[0, j], atol=1e-4), (
                    f"slots {i} and {j} collapsed to identical content"
                )

    def test_write_grad_through_two_call_chain(self):
        """write params get gradient only when the written (un-detached) buffer
        is read by a later call that affects the loss — the cross-segment chain."""
        buf = LSTMBuffer(H, K)
        _activate_read(buf)
        mc = buf.write(torch.randn(B, S, H), torch.zeros(B, K, H))   # call 1 (no detach)
        delta = buf.read(torch.randn(B, S, H), mc)                   # call 2 reads it
        delta.pow(2).mean().backward()
        assert buf.gate_proj_in.weight.grad is not None
        assert buf.gate_proj_in.weight.grad.norm() > 0

    def test_pool_mask_restricts_write(self):
        """Pooling over only a suffix ignores the masked-out prefix."""
        buf = LSTMBuffer(H, K)
        h_T = torch.randn(1, S, H)
        mask = torch.zeros(1, S, dtype=torch.bool)
        mask[0, 8:] = True
        h_T2 = h_T.clone()
        h_T2[0, :8] = torch.randn(8, H)        # change only the masked-out prefix
        torch.manual_seed(0)
        a = buf.write(h_T, torch.zeros(1, K, H), pool_mask=mask)
        torch.manual_seed(0)
        b = buf.write(h_T2, torch.zeros(1, K, H), pool_mask=mask)
        assert torch.allclose(a, b, atol=1e-5)

    def test_m_iter_per_position_shape(self):
        """M_iter folds S into the batch: [B*S, 1, D] buffers."""
        buf = LSTMBuffer(H, 1)
        h = torch.randn(B * S, 1, H)
        new = buf.write(h, torch.zeros(B * S, 1, H))
        assert new.shape == (B * S, 1, H)


# ---------------------------------------------------------------------------
# 2. DirectCCoT
# ---------------------------------------------------------------------------

class TestDirectCCoT:

    def test_write_shape(self):
        d = DirectCCoT(H)
        state = d.write(torch.randn(B, S, H))
        assert state.shape == (B, 1, H)

    def test_state_proj_identity_init(self):
        d = DirectCCoT(H)
        assert torch.allclose(d.state_proj.weight, torch.eye(H))

    def test_in_proj_zero_init(self):
        d = DirectCCoT(H)
        assert torch.allclose(d.in_proj.weight, torch.zeros(H, H))

    def test_read_noop_at_init(self):
        d = DirectCCoT(H)
        delta = d.read(torch.randn(B, 1, H))
        assert torch.allclose(delta, torch.zeros_like(delta))

    def test_read_live_after_activation(self):
        d = DirectCCoT(H)
        _activate_direct(d)
        delta = d.read(torch.randn(B, 1, H))
        assert not torch.allclose(delta, torch.zeros_like(delta))

    def test_state_proj_grad_through_chain(self):
        d = DirectCCoT(H)
        _activate_direct(d)
        state = d.write(torch.randn(B, S, H))   # no detach
        delta = d.read(state)
        delta.pow(2).mean().backward()
        g = d.state_proj.weight.grad
        assert g is not None and g.norm() > 0

    def test_pool_mask_suffix_only(self):
        d = DirectCCoT(H)
        h = torch.randn(1, S, H)
        mask = torch.zeros(1, S, dtype=torch.bool)
        mask[0, 8:] = True
        h2 = h.clone()
        h2[0, :8] = torch.randn(8, H)
        a = d.write(h, pool_mask=mask)
        b = d.write(h2, pool_mask=mask)
        assert torch.allclose(a, b, atol=1e-5)


# ---------------------------------------------------------------------------
# 3. EOS helpers
# ---------------------------------------------------------------------------

class TestEOSHelpers:

    def test_no_eos_full_carry(self):
        """All-False eos → read everywhere, pool everywhere, no reset."""
        eos = torch.zeros(B, S, dtype=torch.bool)
        crm, pool, reset, valid = compute_eos_masks(eos, S, eos.device, torch.float32)
        assert torch.allclose(crm, torch.ones(B, S, 1))
        assert pool.all()
        assert not reset.any()
        assert valid.all()

    def test_read_mask_stops_after_first_eos(self):
        eos = torch.zeros(1, S, dtype=torch.bool)
        eos[0, 7] = True
        crm, _, _, _ = compute_eos_masks(eos, S, eos.device, torch.float32)
        assert crm[0, :8, 0].all()          # positions <= first EOS read carried state
        assert not crm[0, 8:, 0].any()      # later positions read nothing

    def test_pool_mask_open_suffix_only(self):
        eos = torch.zeros(1, S, dtype=torch.bool)
        eos[0, 7] = True
        _, pool, reset, valid = compute_eos_masks(eos, S, eos.device, torch.float32)
        assert not pool[0, :8].any()        # ended doc's prefix excluded
        assert pool[0, 8:].all()            # open doc's suffix included
        assert reset[0]
        assert valid[0]

    def test_eos_at_last_position_carries_nothing(self):
        eos = torch.zeros(1, S, dtype=torch.bool)
        eos[0, S - 1] = True
        _, _, _, valid = compute_eos_masks(eos, S, eos.device, torch.float32)
        assert not valid[0], "empty open suffix → valid_write False"

    def test_apply_write_reset_zeroes_ended_lane(self):
        mc = torch.randn(2, K, H)
        reset = torch.tensor([True, False])
        out = apply_write_reset(mc, reset)
        assert torch.allclose(out[0], torch.zeros(K, H))
        assert torch.allclose(out[1], mc[1])

    def test_apply_valid_write_zeroes_empty_suffix(self):
        mc = torch.randn(2, K, H)
        valid = torch.tensor([False, True])
        out = apply_valid_write(mc, valid)
        assert torch.allclose(out[0], torch.zeros(K, H))
        assert torch.allclose(out[1], mc[1])

    def test_apply_valid_write_none_is_noop(self):
        mc = torch.randn(2, K, H)
        assert torch.allclose(apply_valid_write(mc, None), mc)


# ---------------------------------------------------------------------------
# 4. Samplers
# ---------------------------------------------------------------------------

class TestSamplers:

    def test_num_steps_valid(self):
        for step in range(50):
            n, k = sample_num_steps(step, mean_recurrence=8, mean_backprop_depth=4)
            assert n >= 0 and k >= 1
            assert k <= 4                       # k = min(T, mu_bwd)

    def test_num_steps_deterministic_per_step(self):
        a = sample_num_steps(123, 8, 4)
        b = sample_num_steps(123, 8, 4)
        assert a == b

    def test_batch_steps_length_and_independence(self):
        steps = sample_batch_steps(7, batch_size=8, mean_recurrence=16, mean_backprop_depth=8)
        assert len(steps) == 8
        assert len(set(steps)) > 1, "per-sequence T should vary within the batch"

    def test_curriculum_ramps(self):
        assert get_current_mean_recurrence(0, 8, 100) == 1
        mid = get_current_mean_recurrence(50, 8, 100)
        assert 1 < mid < 8
        assert get_current_mean_recurrence(100, 8, 100) == 8
        assert get_current_mean_recurrence(999, 8, 100) == 8

    def test_curriculum_disabled(self):
        assert get_current_mean_recurrence(0, 8, 0) == 8

    def test_enforce_mu_bwd(self):
        assert enforce_mu_bwd(8) == 4
        assert enforce_mu_bwd(7) == 4          # ceil(7/2)
        assert enforce_mu_bwd(16) == 8


# ---------------------------------------------------------------------------
# 5. Muon
# ---------------------------------------------------------------------------

class TestMuon:

    def test_newtonschulz_near_orthogonal(self):
        G = torch.randn(16, 16)
        O = _zeropower_via_newtonschulz5(G)
        prod = O @ O.T
        # near-orthogonal: O O^T ≈ I (loose tolerance, 5 NS steps)
        assert (prod - torch.eye(16)).abs().mean() < 0.15

    def test_routing_2d_vs_1d(self):
        w2d = nn.Parameter(torch.randn(8, 8))
        w1d = nn.Parameter(torch.randn(8))
        flagged = nn.Parameter(torch.randn(8, 8))
        flagged._no_weight_decay = True
        assert Muon._use_muon(w2d)
        assert not Muon._use_muon(w1d)
        assert not Muon._use_muon(flagged), "structural/SSM params must skip Muon"

    def test_optimizer_step_reduces_loss(self):
        torch.manual_seed(0)
        model = nn.Sequential(nn.Linear(16, 16), nn.ReLU(), nn.Linear(16, 1))
        opt = Muon(model.parameters(), lr=1e-2)
        x = torch.randn(32, 16)
        y = torch.randn(32, 1)
        losses = []
        for _ in range(20):
            opt.zero_grad()
            loss = (model(x) - y).pow(2).mean()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        assert losses[-1] < losses[0], "Muon failed to reduce a trivial loss"


# ---------------------------------------------------------------------------
# 6. LTIInjection (Path-1 / from-scratch only — sanity that it is portable)
# ---------------------------------------------------------------------------

class TestLTI:

    def test_spectral_norm_below_one(self):
        lti = LTIInjection(H)
        assert lti.spectral_norm() < 1.0

    def test_decay_init_target(self):
        lti = LTIInjection(H)
        # init decay ≈ sqrt(1/5) ≈ 0.447
        assert lti.contraction_factor() == pytest.approx(0.447, abs=0.02)

    def test_forward_shape(self):
        lti = LTIInjection(H)
        out = lti(torch.randn(B, S, H), torch.randn(B, S, H))
        assert out.shape == (B, S, H)

    def test_structural_params_flagged(self):
        lti = LTIInjection(H)
        assert getattr(lti.A_log, "_no_weight_decay", False)
        assert getattr(lti.dt_bias, "_no_weight_decay", False)
        assert getattr(lti.B, "_no_weight_decay", False)
