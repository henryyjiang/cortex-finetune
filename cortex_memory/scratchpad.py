"""J3 -- the in-loop latent scratchpad.

Findings doc "Latent Memory (Z) Attempt 1: Findings", Attempt 2, design J3:
"a gated latent state written at iteration t and read at t+1 via the disabled
`iter_write` / `read_into` hooks, carried across the chunk boundary".  Ranked
first as a final design because it is the only one where memory and recurrence
share one state and its write learns under read pressure WITHIN a chunk, so the
interface does not depend on the cross-chunk graph.

WHY J1 SAYS THIS HAS TO EXIST (j1_prereg.md S10).  J1's Z write was the loop
state itself (s_T at the last real tokens), and the only gradient that could
shape it for reading came through the NEXT chunk's read.  The read gate closed
in ~270 updates and took that gradient with it; with E removed, Z carried
nothing.  Here the write has a second, within-chunk consumer: the read at
iteration t+1 reads what iteration t wrote, on every loop step of every chunk,
so the write parameters train whether or not the cross-chunk read has found
anything yet.  What gets CARRIED is the same scratchpad, so the cross-chunk
read inherits a write that was already shaped for reading.

THE DESIGN, AND WHY EACH PART IS FORCED
---------------------------------------

* ONE ROW PER PACKED POSITION.  The within-chunk read must not leak the
  future.  A pooled scratchpad written from the chunk's tokens and read back
  by those tokens would give position i a summary of positions > i -- label
  leakage that the loss would reward and that would look like a working
  memory.  Per-position rows cannot cross the causal mask: position i writes
  row i and reads row i.  (Cross-position mixing is left to the base model's
  own causal attention, which already does it every layer.)

* THE CARRY RE-ENTERS EVERY ITERATION.  The carried rows are keys of the same
  read on every loop step, beside the own row, exactly as E re-enters through
  `input_embeds`.  Seeding the scratchpad's first state from the carry instead
  would repeat the s0 failure: the no-grad iterations run first, so a carry
  that enters once has no gradient path on any batch with a no-grad prefix.

* THE CARRY RIDES THE EXISTING Z CHANNEL.  At the end of the chunk the
  scratchpad rows at the last W * latent_tok_pool real tokens are pooled to W
  rows -- the same positions J1's `tokens` encoding took from s_T -- and merged
  into the gated ring's Z half through its own gate.  So E/Z co-location, the
  ring horizon, the donor roll, `latent_read_null` and every eval that already
  scores J1 apply unchanged.  (`latent_encoding='scratch'`.)

* THE WRITE IS AN LM2 GATE OVER DEPTH, ZERO-INITIALISED.

      m_t = fg * m_{t-1} + ig * s_t,   [ig, fg] = sigmoid(W_in s_t + W_mem n(m_{t-1}) + b)

  The candidate is the loop state s_t itself (post core layers), so the only
  new write parameters are the gates.  At `gate_init` zero the weights are
  zero and step 0 is the constant EMA  m_t = 0.731 m_{t-1} + 0.5 s_t:  a
  working memory from the first step, not a no-op to be discovered -- the
  same reasoning as the ring's gate_init='zero'.  m_0 is zero and zero rows
  are masked as keys, so iteration 1 reads nothing from the scratchpad.

  The memory side is RMS-normalised, not tanh'd: s_t rows are ~10 in norm
  over D = 2048 (~0.2 per element), where tanh is the identity anyway, but a
  gated accumulator's norm drifts with depth and the gate should see what a
  row holds, not how big it has become.

COST.  Two D -> 2D projections per position per iteration (16.8M parameters at
D = 2048, plus Adam state), and one [B, S, D] state per iteration in the graph.
Price it (`pace/j3_price_oom.sbatch`) before launching.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn


class LatentScratchpad(nn.Module):
    """Per-position gated latent state over loop depth (J3's write)."""

    def __init__(self, hidden_size: int, forget_bias_init: float = 1.0) -> None:
        super().__init__()
        if not math.isfinite(float(forget_bias_init)):
            raise ValueError(
                f"scratchpad forget_bias_init must be finite; got "
                f"{forget_bias_init!r}")
        self.hidden_size = int(hidden_size)
        self.forget_bias_init = float(forget_bias_init)
        self.gate_proj_in = nn.Linear(hidden_size, 2 * hidden_size)
        self.gate_proj_mem = nn.Linear(hidden_size, 2 * hidden_size)
        self.forget_bias = nn.Parameter(torch.ones(1))
        self.input_bias = nn.Parameter(torch.zeros(1))
        self.apply_designed_init()

    def apply_designed_init(self) -> list:
        """(Re-)apply the designed init; returns log tags.

        MUST be called from reset_cortex_graft_init, for the same reason as
        PrefixGatedBuffer.apply_gate_init: that function calls
        reset_parameters() on every cortex submodule, which puts kaiming
        weights back into both projections and discards the zero init.
        """
        for proj in (self.gate_proj_in, self.gate_proj_mem):
            nn.init.zeros_(proj.weight)
            nn.init.zeros_(proj.bias)
        nn.init.constant_(self.forget_bias, self.forget_bias_init)
        nn.init.zeros_(self.input_bias)
        return [f"latent_scratch.[gates=0 (EMA fg={self.ema_forget:.3f} "
                f"ig=0.500), forget_bias={self.forget_bias_init}]"]

    @property
    def ema_forget(self) -> float:
        """fg at the zero init: the per-ITERATION retention of the scratchpad."""
        return 1.0 / (1.0 + math.exp(-self.forget_bias_init))

    @staticmethod
    def _rms(m: torch.Tensor) -> torch.Tensor:
        return m * torch.rsqrt(
            m.float().pow(2).mean(-1, keepdim=True).clamp_min(1e-12)
        ).to(m.dtype)

    def forward(self, s: torch.Tensor,
                m_prev: Optional[torch.Tensor] = None) -> torch.Tensor:
        """s [B,S,D] (loop state after the core layers), m_prev [B,S,D] or
        None (the first iteration) -> m [B,S,D]."""
        if m_prev is None:
            m_prev = torch.zeros_like(s)
        elif m_prev.shape != s.shape:
            raise ValueError(
                f"scratchpad state {tuple(m_prev.shape)} does not match the loop "
                f"state {tuple(s.shape)}.  The scratchpad is per packed position "
                "and is reset every forward; a mismatch means it survived into "
                "a different chunk's layout.")
        m_prev = m_prev.to(dtype=s.dtype)
        combined = self.gate_proj_in(s) + self.gate_proj_mem(self._rms(m_prev))
        ig_l, fg_l = combined.chunk(2, dim=-1)
        ig = torch.sigmoid(ig_l + self.input_bias.to(s.dtype))
        fg = torch.sigmoid(fg_l + self.forget_bias.to(s.dtype))
        return fg * m_prev + ig * s
