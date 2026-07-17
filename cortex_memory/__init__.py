"""
cortex_memory — base-model-agnostic memory components ported from cortex-main.

Self-contained, pure-torch building blocks for grafting episodic latent memory
onto the retrofitting-recurrence (raven) Pre/Loop/Coda model.  Nothing here
imports or depends on the host model; the graft lives in cortex_graft.py and is
gated behind config flags (default off — the base repo is unchanged until you
opt in).

Components
----------
  LSTMBuffer       — LM2-style K-slot LSTM-gated memory (M_cross / M_iter)
  DirectCCoT       — Coconut-style K=0 carry (single carried vector)
  AccumCCoT        — AutoCompressor-style accumulating multi-vector carry
  GatedAccumBuffer — gated-accumulation LM2 variant (extraction write + LM2
                     gated merge on K fixed slots; append-vs-overwrite arm)
  LTIInjection  — Parcae LTI injection (Path-1 / from-scratch only; unused by
                  default — see cortex_migration_plan.md §0)
  Muon          — Newton-Schulz optimizer (opt-in; host AdamW is default)
  sampling      — Parcae Algorithm-4 recurrence sampling + curriculum
  eos           — EOS-aware cross-state document handling helpers
  chunking      — cross-chunk chain helpers (randomized segmenting, stop-grad
                  horizon, eval-side slice ablation)

See ../cortex_migration_plan.md for the full plan.
"""
from __future__ import annotations

from .buffers import LSTMBuffer, DirectCCoT, AccumCCoT, GatedAccumBuffer
from .chunking import random_chunk_sizes, detach_old_vecs, ablate_vec_slice
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
    "AccumCCoT",
    "GatedAccumBuffer",
    "random_chunk_sizes",
    "detach_old_vecs",
    "ablate_vec_slice",
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
