"""
cortex_memory — base-model-agnostic memory components ported from cortex-main.

Self-contained, pure-torch building blocks for grafting episodic latent memory
onto the retrofitting-recurrence (raven) Pre/Loop/Coda model.  Nothing here
imports or depends on the host model; the graft lives in cortex_graft.py and is
gated behind config flags (default off — the base repo is unchanged until you
opt in).

Components
----------
LIVE (buffers.py) — what a new run may select:
  PrefixAccumBuffer — AutoCompressor-faithful append (accumulate_summary)
  PrefixGatedBuffer — the same write with an LM2 gate at constant width
                      (merge is the ONLY difference, so append-vs-gated is a
                      controlled comparison)

LOAD-PATH ONLY (legacy.py) — required to open Track-A / B1 checkpoints;
`train.py` refuses to START a run on any of them.  Read legacy.py's docstring
before touching these; deleting them orphans the historical results table:
  LSTMBuffer       — LM2-style K-slot LSTM-gated memory (M_cross / M_iter)
  DirectCCoT       — Coconut-style K=0 carry (single carried vector)
  AccumCCoT        — AutoCompressor-style accumulating multi-vector carry
  GatedAccumBuffer — gated-accumulation LM2 variant (extraction write + LM2
                     gated merge on K fixed slots; append-vs-overwrite arm)

  LTIInjection  — Parcae LTI injection (Path-1 / from-scratch only; unused by
                  default; the from-scratch line it served is closed)
  Muon          — Newton-Schulz optimizer (opt-in; host AdamW is default)
  sampling      — Parcae Algorithm-4 recurrence sampling + curriculum
  eos           — EOS-aware cross-state document handling helpers
  chunking      — cross-chunk chain helpers (randomized segmenting, stop-grad
                  horizon, eval-side slice ablation)

The original porting plan is archived at `archive/planning/cortex_migration_plan.md`
in the project root; README.md is the current source of record.
"""
from __future__ import annotations

from .buffers import PrefixAccumBuffer, PrefixGatedBuffer
from .legacy import LSTMBuffer, DirectCCoT, AccumCCoT, GatedAccumBuffer
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
    "PrefixAccumBuffer",
    "PrefixGatedBuffer",
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
