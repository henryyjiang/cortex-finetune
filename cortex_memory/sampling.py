"""
Recurrence sampling + curriculum (Parcae Algorithm 4).

Ported from cortex-main/train.py.  The host retrofitting-recurrence repo has
its own num_steps sampler; these are provided so a cortex run can match the
Parcae distribution exactly (opt-in).  Pure functions, no torch.nn state.
"""
from __future__ import annotations

import math

import torch


def sample_num_steps(
    optimizer_step:      int,
    mean_recurrence:     int,
    mean_backprop_depth: int,
) -> tuple[int, int]:
    """
    Parcae Algorithm 4 (App. H): sample total T first, then derive n and k.

        T ~ LogNormal-Poisson(µ_rec)      [heavy-tailed, mean ≈ µ_rec]
        n = max(T - µ_bwd, 0)             [no-grad steps]
        k = min(T,   µ_bwd)               [grad steps]

    This avoids the distributional mismatch in McLeish Algorithm 3, where
    setting k = µ_bwd as a constant truncates and compresses the forward-
    pass distribution, hurting generalisation to other test-time depths.

    Returns (n, k) as plain Python ints.
    """
    seed = 514229 + optimizer_step
    gen  = torch.Generator(device="cpu").manual_seed(seed % (2**31 - 1))

    sigma = 0.5
    mu    = math.log(max(mean_recurrence, 1)) - (sigma ** 2) / 2
    rate  = torch.zeros(1).log_normal_(mean=mu, std=sigma, generator=gen)
    T     = max(1, int(torch.poisson(rate, generator=gen).item()))

    n = max(T - mean_backprop_depth, 0)
    k = min(T, mean_backprop_depth)
    return n, k


def sample_batch_steps(
    optimizer_step:      int,
    batch_size:          int,
    mean_recurrence:     int,
    mean_backprop_depth: int,
) -> list[tuple[int, int]]:
    """
    Per-sequence variant: sample independent T_i for each sequence i.

    Returns list[(n_i, k_i)] of length batch_size.
    Each sequence gets a different depth from the same Λ, faithfully
    approximating E_{T~Λ}[loss] within the batch rather than collapsing
    to a single T per micro-batch.  Ref: Parcae §4.2, Appendix G.
    """
    steps = []
    for i in range(batch_size):
        seed = 514229 + optimizer_step * 10007 + i
        gen  = torch.Generator(device="cpu").manual_seed(seed % (2**31 - 1))

        sigma = 0.5
        mu    = math.log(max(mean_recurrence, 1)) - (sigma ** 2) / 2
        rate  = torch.zeros(1).log_normal_(mean=mu, std=sigma, generator=gen)
        T     = max(1, int(torch.poisson(rate, generator=gen).item()))

        n = max(T - mean_backprop_depth, 0)
        k = min(T, mean_backprop_depth)
        steps.append((n, k))
    return steps


def get_current_mean_recurrence(
    step:           int,
    target_mean:    int,
    curriculum_steps: int,
) -> int:
    """
    Linear ramp from 1 → target_mean over curriculum_steps optimizer steps.
    After curriculum_steps, holds constant at target_mean.

    Ref: McLeish et al. §4.2 — scheduling mean recurrence is both data-
         and compute-efficient, reducing FLOPs for the same loss.
    """
    if curriculum_steps <= 0 or step >= curriculum_steps:
        return target_mean
    frac = step / curriculum_steps
    return max(1, round(1 + frac * (target_mean - 1)))


def enforce_mu_bwd(mean_recurrence: int) -> int:
    """
    µbwd = ⌈µrec / 2⌉

    Parcae Appendix I shows that growing µrec without growing µbwd
    proportionally degrades performance at high test-time T.
    µbwd = ⌈µrec/2⌉ is the validated choice across all their ablations.
    """
    return math.ceil(mean_recurrence / 2)
