"""
Cortex memory buffers -- base-model-agnostic, pure torch.

These modules know nothing about the host model: they operate on [B, S, D]
hidden states and [B, K, D] buffers.  They are grafted into RavenForCausalLM
via cortex_graft.py and gated behind config flags (default off).

THE LIVE SURFACE (what a new run may select):

  _PrefixBufferBase  -- shared summary-token machinery
  PrefixAccumBuffer  -- AutoCompressor-faithful append (accumulate_summary)
  PrefixGatedBuffer  -- the same write with an LM2 gate at constant width

Subclasses differ ONLY in `merge`, which is what makes append-vs-gated a
controlled comparison.

THE LEGACY SURFACE lives in `legacy.py` -- LSTMBuffer, DirectCCoT, AccumCCoT,
GatedAccumBuffer.  They are re-exported below so existing imports keep working,
but they are load-path only: `train.py` refuses to start a new run on them, and
they cannot be deleted without orphaning every Track-A and B1 checkpoint.  See
legacy.py's docstring before touching them.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# Legacy buffers, quarantined to legacy.py on 2026-09-14.  Re-exported so that
# every existing `from cortex_memory.buffers import ...` keeps working -- the
# graft's load path for Track-A / B1 checkpoints depends on these names.
from .legacy import (  # noqa: F401
    LSTMBuffer,
    DirectCCoT,
    AccumCCoT,
    GatedAccumBuffer,
    _extract_summary_vectors,
)


# ---------------------------------------------------------------------------
# AutoCompressor-faithful prefix buffers
# (supersede AccumCCoT / GatedAccumBuffer; those two stay only so the Track-A
#  and B1 arms remain reproducible — new runs should not select them)
# ---------------------------------------------------------------------------
#
# WHY THESE EXIST
# ---------------
# The B1 co-training round measured carry = -0.0128 nats (z = -9.3): carrying
# the buffer made the model WORSE than zeroing it.  Reading
# AutoCompressors-main/auto_compressor.py against our implementation turned up
# four divergences from the published, working design.  Each is fixed here.
#
#   1. THE READ PATH.  AutoCompressor PREPENDS its summary vectors to the next
#      segment's input embeddings (auto_compressor.py:85) — they are consumed
#      by the model's OWN pretrained self-attention at every layer.  There is
#      no read module.  We used a separate cross-attention block with a
#      zero-init out_proj injected inside the recurrent loop, so the entire
#      communication channel had to be discovered from nothing.  These buffers
#      have NO read path: the modeling file prepends `state` to input_embeds
#      and the base model's own attention does the reading.
#
#   2. THE WRITE PATH.  AutoCompressor APPENDS learned summary-token embeddings
#      to the segment and takes their output hidden states
#      (auto_compressor.py:123) — the summary is written by the model's own
#      full-depth forward.  We used a bolt-on extraction cross-attention.  Here
#      `summary_emb` is appended to the input by the modeling file and the
#      resulting post-ln_f hidden states ARE the new vectors.
#
#   3. NO tanh(LayerNorm(.)) BOUNDING.  Our extraction ended in
#      tanh(vec_ln(attended + vec_emb)), pinning the write norm by
#      construction — the 2026-08-02 buffer diag measured write-norm variation
#      across samples at 0.23% of the mean, the flattest of any arm run.
#      AutoCompressor applies no bounding: its summary vectors are raw hidden
#      states living in the same space as token embeddings, which is exactly
#      what feeding them back as input embeddings requires.
#
#   4. INITIALISATION.  AutoCompressor seeds every summary embedding with the
#      EOS token embedding (auto_compressor.py:49) — a real, in-distribution
#      vector the pretrained model already knows how to process.  Ours were
#      normal_(std=0.02) noise.  init_from_token_embedding does it their way;
#      the graft calls it once the base weights are loaded.
#
# WHAT IS DELIBERATELY NOT COPIED
# -------------------------------
# AutoCompressor gives summary/softprompt slots the pad position (no position).
# OPT has learned absolute position embeddings; we have RoPE, so the modeling
# file's analog is position 0 (identity rotation) for prefix and summary slots,
# with real tokens taking 1..S.  That also keeps every position index inside
# the trained 1024 window even though the padded sequence is longer than it.
#
# STATE SHAPE CONTRACT (unchanged from AccumCCoT, on purpose)
# -----------------------------------------------------------
# `state` stays [B, N, D] with write-once rows, so train.py's stop-gradient
# horizon (detach_old_vecs, --cortex.carry_grad_chunks) keeps working
# untouched, and so do apply_write_reset / apply_valid_write in the graft.


class _PrefixBufferBase(nn.Module):
    """Shared summary-token machinery for the two prefix buffers.

    Subclasses differ ONLY in `merge` — append (AutoCompressor's
    accumulate_summary) vs the LM2 gated update at constant memory.  Keeping
    everything else identical is what makes the two arms a clean contrast.
    """

    def __init__(self, hidden_size: int, n_vec: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.n_vec = n_vec

        # The analog of AutoCompressor's `embed_summary` (auto_compressor.py:47).
        # Overwritten by init_from_token_embedding once the base weights exist;
        # this normal_ only matters for unit tests that build the buffer
        # standalone.  Embedding-like: no weight decay, no Muon Newton-Schulz.
        self.summary_emb = nn.Parameter(torch.empty(n_vec, hidden_size))
        nn.init.normal_(self.summary_emb, std=0.02)
        self.summary_emb._no_weight_decay = True

        # PERSISTENT so it survives save/load.  The seeding below can only run
        # on the first forward (wte is random at __init__), and a plain Python
        # attribute would be False again after every resume — which would
        # re-seed a TRAINED summary_emb back to wte[eos] at the start of every
        # link in the resume chain, silently resetting the write path while the
        # loss curve stayed healthy.  A registered buffer rides in the state
        # dict, so a restored checkpoint reports itself already seeded.
        self.register_buffer("summary_seeded",
                             torch.zeros((), dtype=torch.bool), persistent=True)

    @torch.no_grad()
    def init_from_token_embedding(self, wte_weight: torch.Tensor,
                                  token_id: int) -> None:
        """Seed every summary embedding with one real token's embedding.

        AutoCompressor uses EOS (auto_compressor.py:49).  All n_vec rows start
        identical and differentiate through training, exactly as theirs do.
        """
        assert wte_weight.shape[1] == self.hidden_size, (
            f"embedding dim {wte_weight.shape[1]} != buffer dim {self.hidden_size}")
        self.summary_emb.data.copy_(
            wte_weight[token_id].to(self.summary_emb.dtype)
            .unsqueeze(0).expand(self.n_vec, -1)
        )
        self.summary_seeded.fill_(True)

    def summary_slots(self, batch_size: int, dtype: torch.dtype,
                      device: torch.device) -> torch.Tensor:
        """[B, n_vec, D] summary-token embeddings to APPEND to this chunk's
        input.  Their post-ln_f hidden states come back through `merge`."""
        return (self.summary_emb.unsqueeze(0)
                .expand(batch_size, -1, -1).to(device=device, dtype=dtype))

    def merge(self, state: Optional[torch.Tensor],
              new_vecs: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class PrefixAccumBuffer(_PrefixBufferBase):
    """AutoCompressor's accumulating carry, faithfully.

    `merge` concatenates this chunk's summary vectors onto the carried state
    and never overwrites — AutoCompressor's accumulate_summary=true
    (auto_compressor.py:230, the default in their run/train.sh).  The FIFO cap
    only bounds unbounded chunk chains at eval; in training the state is
    expected to stay under it.

    Rows are write-once, so per-chunk slices stay separable and train.py can
    slice-detach vectors older than --cortex.carry_grad_chunks — their
    "stop-gradient after N compression steps" (substep_trainer.py:166).
    """

    def __init__(self, hidden_size: int, n_vec: int = 32,
                 max_vecs: int = 128) -> None:
        super().__init__(hidden_size, n_vec)
        assert max_vecs >= n_vec, "max_vecs must hold at least one chunk's write"
        self.max_vecs = max_vecs

    def merge(self, state: Optional[torch.Tensor],
              new_vecs: torch.Tensor) -> torch.Tensor:
        """state [B, N, D] (None on chunk 1) + [B, n_vec, D] -> [B, N', D]."""
        out = new_vecs if state is None else torch.cat([state, new_vecs], dim=1)
        if out.shape[1] > self.max_vecs:
            out = out[:, -self.max_vecs:]          # FIFO: drop the oldest
        return out


class PrefixGatedBuffer(_PrefixBufferBase):
    """LM2-gated fixed-K carry over the same AutoCompressor write/read paths.

    Identical to PrefixAccumBuffer except that `merge` is the LM2 gated update
    (create_gates, LM2 section 3.3) at CONSTANT memory instead of an append:

        combined = gate_proj_in(candidate) + gate_proj_mem(tanh(state))
        ig, fg   = chunk(combined, 2, -1);  sigmoid(. + bias)
        state'   = fg * state + ig * candidate

    n_slots == n_vec, so the update is per-slot: slot i weighs what it would
    write against what it already holds.  Forget bias +1.0 biases toward
    retention at init.  gate_proj_mem (the memory-feedback side) only receives
    gradient through a chain of >= 3 chunks — keep carry_grad_chunks >= 2.

    Because merge overwrites in place, rows are NOT separable across chunks;
    train.py detaches the whole state every carry_grad_chunks chunks instead of
    slicing (truncated BPTT), which the existing loop already handles.
    """

    def __init__(self, hidden_size: int, n_vec: int = 32) -> None:
        super().__init__(hidden_size, n_vec)
        self.n_slots = n_vec

        self.gate_proj_in  = nn.Linear(hidden_size, hidden_size * 2)
        self.gate_proj_mem = nn.Linear(hidden_size, hidden_size * 2)
        self.forget_bias   = nn.Parameter(torch.ones(1))    # +1.0, LM2 3.3
        self.input_bias    = nn.Parameter(torch.zeros(1))

    def merge(self, state: Optional[torch.Tensor],
              new_vecs: torch.Tensor) -> torch.Tensor:
        """state [B, K, D] (None on chunk 1) + [B, K, D] -> [B, K, D]."""
        if state is None:
            # First chunk: nothing to retain, so the gate would only attenuate
            # a candidate it has no memory to weigh against.  Adopt it whole —
            # what AutoCompressor effectively does when softprompt is empty.
            return new_vecs
        combined = (self.gate_proj_in(new_vecs)
                    + self.gate_proj_mem(torch.tanh(state)))       # [B, K, 2D]
        ig_logits, fg_logits = combined.chunk(2, dim=-1)
        ig = torch.sigmoid(ig_logits + self.input_bias)
        fg = torch.sigmoid(fg_logits + self.forget_bias)
        return fg * state + ig * new_vecs

