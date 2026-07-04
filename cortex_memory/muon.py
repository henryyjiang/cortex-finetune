"""
Muon optimizer (Momentum Orthogonalised by Newton-Schulz).

Ported from cortex-main/train.py.  Opt-in only — the host repo's AdamW /
ELLISAdam remain the default.  Muon was a Parcae-from-scratch artifact; it is
provided here so it can be toggled on for experimentation via a flag, not
because it is recommended for finetuning a pretrained loop.

Routing: 2D+ weight matrices NOT tagged `_no_weight_decay` get the Newton-Schulz
orthogonalised update; everything else (1D params, biases, norms, embeddings,
structural/SSM params, and any param tagged `_no_weight_decay`) falls back to a
built-in AdamW.  Tag embedding-like / structural params with
`param._no_weight_decay = True` to keep them on the AdamW path.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def _zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """
    Newton-Schulz quintic iteration — approximates G · (G^T G)^{-1/2},
    i.e. the nearest orthogonal matrix to G in the Frobenius norm.

    5 steps suffice for near-orthogonality in practice.
    Ref: Keller Jordan 2024 (https://github.com/KellerJordan/Muon).
    """
    assert G.ndim >= 2
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.float() / (G.norm() + 1e-7)
    if G.shape[0] > G.shape[1]:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        X = a * X + b * (A @ X) + c * (A @ A @ X)
    if G.shape[0] > G.shape[1]:
        X = X.T
    return X.to(G.dtype)


class Muon(torch.optim.Optimizer):
    """
    Muon: Momentum Orthogonalised by Newton-Schulz.

    For 2D+ weight matrices that are not SSM/structural (_no_weight_decay)
    and not embeddings: applies Newton-Schulz to the gradient, then a
    momentum-style update.  This effectively orthogonalises the update
    direction and removes the need to tune a separate LR per layer.

    For everything else (1D params, biases, LayerNorm scales, SSM params,
    and embedding tables tagged with _no_weight_decay): falls back to AdamW.

    Ref: Keller Jordan 2024; Prairie et al. (Parcae) 2025 with momentum
         warmup and weight decay annealing.

    Args
    ----
    params          : parameter groups
    lr              : learning rate applied to Muon updates (and AdamW fallback)
    momentum        : Muon momentum (subject to external warmup schedule)
    nesterov        : use Nesterov momentum (default True)
    ns_steps        : Newton-Schulz iterations (5 is sufficient)
    weight_decay    : applied to Muon params; SSM params always skip WD
    """

    def __init__(
        self,
        params,
        lr:           float = 3e-4,
        momentum:     float = 0.95,
        nesterov:     bool  = True,
        ns_steps:     int   = 5,
        weight_decay: float = 0.0,
    ) -> None:
        defaults = dict(
            lr=lr, momentum=momentum, nesterov=nesterov,
            ns_steps=ns_steps, weight_decay=weight_decay,
        )
        super().__init__(params, defaults)

    @staticmethod
    def _use_muon(p: nn.Parameter) -> bool:
        """True for 2D+ weight matrices that aren't structural/SSM params."""
        return p.ndim >= 2 and not getattr(p, "_no_weight_decay", False)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr       = group["lr"]
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            ns_steps = group["ns_steps"]
            wd       = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad

                if self._use_muon(p):
                    # ── Muon update ─────────────────────────────────────
                    state = self.state[p]
                    if "buf" not in state:
                        state["buf"] = torch.zeros_like(g)
                    buf = state["buf"]
                    buf.mul_(momentum).add_(g)

                    # Nesterov: lookahead along the momentum direction
                    g_eff = g.add(buf, alpha=momentum) if nesterov else buf

                    # Newton-Schulz orthogonalisation
                    update = _zeropower_via_newtonschulz5(g_eff, steps=ns_steps)
                    # RMS-match the update to a typical AdamW step.  The
                    # orthogonalised update has per-element RMS ~ 1/sqrt(max(m,n)),
                    # i.e. ~sqrt(max(m,n)) too small.  Paired with an AdamW-sized LR
                    # (3e-4) the weight matrices moved ~30x slower per step than the
                    # embeddings on the Adam fallback, leaving the bulk of the
                    # network undertrained (the LAMBADA-catastrophic / BLIMP-mild
                    # signature vs official Pythia at matched tokens).  Rescaling to
                    # a shape-independent RMS of ~0.2 makes one AdamW-sized LR + WD
                    # correct across every matrix shape, instead of needing a
                    # separate ~0.02 Muon LR.  Ref: Liu et al. 2025, "Muon is
                    # Scalable for LLM Training" (the 0.2*sqrt(max(m,n)) rule).
                    # (Synced from cortex-main 635118f, 2026-06-28.)
                    update.mul_(0.2 * max(g.shape[0], g.shape[1]) ** 0.5)

                    if wd != 0.0:
                        p.mul_(1.0 - lr * wd)
                    p.add_(update, alpha=-lr)

                else:
                    # ── AdamW fallback (biases, LN, embeddings, SSM) ────
                    state = self.state[p]
                    if "step" not in state:
                        state["step"]       = 0
                        state["exp_avg"]    = torch.zeros_like(g)
                        state["exp_avg_sq"] = torch.zeros_like(g)

                    state["step"] += 1
                    # Parcae uses (0.95, 0.95) for the Adam fallback.
                    # beta1 is kept constant (not subject to Muon momentum warmup):
                    # the warmup intentionally dips to 0.85 to let the Muon gradient
                    # buffer start fresh, which is harmful for Adam's EMA stability.
                    beta1, beta2, eps = 0.95, 0.95, 1e-8

                    state["exp_avg"].mul_(beta1).add_(g, alpha=1 - beta1)
                    state["exp_avg_sq"].mul_(beta2).addcmul_(g, g, value=1 - beta2)

                    t    = state["step"]
                    bias = (1 - beta2 ** t) ** 0.5 / (1 - beta1 ** t)
                    denom = state["exp_avg_sq"].sqrt().add_(eps)

                    # WD only for 2D+ non-flagged params; 1D params (biases,
                    # LayerNorm gamma/beta, scalar gates) and SSM params are exempt.
                    # 2D+ non-SSM params are already handled by the Muon path above.
                    if wd != 0.0 and not getattr(p, "_no_weight_decay", False) and p.ndim >= 2:
                        p.mul_(1.0 - lr * wd)
                    p.addcdiv_(state["exp_avg"], denom, value=-lr * bias)

        return loss
