"""
cortex_memory — base-model-agnostic memory components ported from cortex-main.

Self-contained, pure-torch building blocks for grafting episodic latent memory
onto the retrofitting-recurrence (raven) Pre/Loop/Coda model.  Nothing here
imports or depends on the host model; the graft lives in cortex_graft.py and is
gated behind config flags (default off — the base repo is unchanged until you
opt in).

Components
----------
  LSTMBuffer    — LM2-style K-slot LSTM-gated memory (M_cross / M_iter)
  DirectCCoT    — Coconut-style K=0 carry (single carried vector)
  LTIInjection  — Parcae LTI injection (Path-1 / from-scratch only; unused by
                  default — see cortex_migration_plan.md §0)
  Muon          — Newton-Schulz optimizer (opt-in; host AdamW is default)
  sampling      — Parcae Algorithm-4 recurrence sampling + curriculum
  eos           — EOS-aware cross-state document handling helpers

See ../cortex_migration_plan.md for the full plan.
"""
from __future__ import annotations

from .buffers import LSTMBuffer, DirectCCoT
from .lti import LTIInjection, _init_dt_bias
from .muon import Muon, _zeropower_via_newtonschulz5
from .sampling import (
    sample_num_steps,
    sample_batch_steps,
    get_current_mean_recurrence,
    enforce_mu_bwd,
)
from .eos import compute_eos_masks, apply_write_reset, apply_valid_write

__all__ = [
    "LSTMBuffer",
    "DirectCCoT",
    "LTIInjection",
    "_init_dt_bias",
    "Muon",
    "_zeropower_via_newtonschulz5",
    "sample_num_steps",
    "sample_batch_steps",
    "get_current_mean_recurrence",
    "enforce_mu_bwd",
    "compute_eos_masks",
    "apply_write_reset",
    "apply_valid_write",
]
