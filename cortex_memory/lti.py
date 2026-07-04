"""
Parcae LTI injection — included for completeness and Path-1 from-scratch
experiments.  NOT used by default in the finetuning (Path 2) graft: it is a
*construction-time* contraction guarantee that cannot be applied to a
pretrained loop (the raven base already has its own concat-adapter injection
and stable init).  See cortex_migration_plan.md §0 / §4 (Gap 1).

Ported verbatim from cortex-main/model.py.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _init_dt_bias(tensor: torch.Tensor, decay_target: float = math.sqrt(1.0 / 5.0)) -> None:
    """
    Inverse-softplus init so that softplus(dt_bias) · exp(A_log=0) = 1 gives
    the target initial decay = exp(-softplus(dt_bias)).

    decay_target = sqrt(1/5) ≈ 0.447  (Parcae §4.1)

    Derivation:
      decay = exp(-softplus(dt))
      → softplus(dt) = -log(decay_target)            let x = -log(decay_target)
      → dt = softplus⁻¹(x) = x + log(-expm1(-x))
    """
    with torch.no_grad():
        x   = -math.log(decay_target)                 # ≈ 0.8047 for decay=0.447
        inv = x + math.log(-math.expm1(-x))           # inverse softplus ≈ 0.212
        tensor.fill_(inv)


class LTIInjection(nn.Module):
    """
    Stable input injection via a discrete LTI system (Parcae §4.1).

    h_{t+1} = exp(-dt·A) ⊙ h_t  +  dt · (z₀ @ B.T)

    Parameterization ensures ρ(Ā) < 1 for all parameter values:
      A  = exp(A_log)         > 0  always  (A_log ∈ ℝ, init = 0 → A = 1)
      dt = softplus(dt_bias)  > 0  always  (init → decay ≈ 0.447)
      decay = exp(-dt·A)      ∈ (0,1)  always

    B is identity-initialized; all three are excluded from weight decay.
    """

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.A_log   = nn.Parameter(torch.zeros(hidden_size))
        self.dt_bias = nn.Parameter(torch.empty(hidden_size))
        self.B       = nn.Parameter(torch.eye(hidden_size))

        _init_dt_bias(self.dt_bias)   # decay ≈ 0.447 at init

        # Excluded from weight decay and Muon Newton-Schulz (structural params)
        self.A_log._no_weight_decay   = True
        self.dt_bias._no_weight_decay = True
        self.B._no_weight_decay       = True

    def forward(self, h: torch.Tensor, z0: torch.Tensor) -> torch.Tensor:
        dt    = F.softplus(self.dt_bias)               # (D,)
        decay = torch.exp(-dt * torch.exp(self.A_log)) # (D,) in (0, 1)
        return h * decay + dt * (z0 @ self.B.T)

    @torch.no_grad()
    def spectral_norm(self) -> float:
        """Max decay factor — should stay < 1 throughout training."""
        dt = F.softplus(self.dt_bias)
        return torch.exp(-dt * torch.exp(self.A_log)).max().item()

    @torch.no_grad()
    def contraction_factor(self) -> float:
        """Mean decay factor across hidden dimensions."""
        dt = F.softplus(self.dt_bias)
        return torch.exp(-dt * torch.exp(self.A_log)).mean().item()
