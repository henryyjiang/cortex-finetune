"""
EOS-aware cross-state document handling (cortex-main model #14).

Refactored out of CortexGPT.forward into standalone functions so the host-model
graft and the unit tests share one implementation.

Packed segments can contain several documents.  The carried cross-state belongs
to the document continuing from the previous segment, and the state written for
the next segment must describe the document still open at this segment's end.
Three per-lane masks implement this:

  read mask   — positions p <= first EOS may read the carried state; later
                positions belong to documents that started inside this segment
                and read nothing (== empty buffer).
  pool mask   — only positions p > last EOS (the open document's suffix) are
                pooled into the write.
  write reset — the incoming buffer (the ended document's state) is excluded
                from the gated update when an EOS occurred.
  valid write — the open-document suffix may be empty (EOS at the final
                position); then nothing is carried and the written state is 0.

First-pass finetuning uses one-document-per-sequence data with eos_mask=None
(full carry, correct) — these helpers only fire on packed multi-doc segments.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch


def compute_eos_masks(
    eos_mask: torch.Tensor,
    seq_len: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    eos_mask : [B, S] bool — True at the last token (EOS) of each packed document.

    Returns
    -------
    cross_read_mask : [B, S, 1] in `dtype`  — multiply the read delta by this
                      (1.0 for positions <= first EOS, 0.0 after).
    pool_mask       : [B, S] bool           — positions the write may pool over.
    write_reset     : [B] bool              — lanes where a document ended
                      (their incoming buffer is excluded from the gated update).
    valid_write     : [B] bool              — lanes with a non-empty open suffix
                      (lanes that are False carry a zero state forward).
    """
    eos_b   = eos_mask.bool()
    B       = eos_b.shape[0]
    S       = seq_len
    has_eos = eos_b.any(dim=1)                                            # [B]
    pos     = torch.arange(S, device=device)

    first_eos = torch.where(
        has_eos, eos_b.int().argmax(dim=1),
        torch.full((B,), S - 1, dtype=torch.long, device=device),
    )
    cross_read_mask = (
        (pos.unsqueeze(0) <= first_eos.unsqueeze(1))                      # [B, S]
        .unsqueeze(-1).to(dtype)                                          # [B, S, 1]
    )

    last_eos  = S - 1 - eos_b.flip(1).int().argmax(dim=1)                 # valid where has_eos
    pool_mask = (pos.unsqueeze(0) > last_eos.unsqueeze(1)) | (~has_eos).unsqueeze(1)
    valid_write = pool_mask.any(dim=1)                                    # [B]
    write_reset = has_eos
    return cross_read_mask, pool_mask, write_reset, valid_write


def apply_write_reset(
    m_cross_in: torch.Tensor, write_reset: torch.Tensor
) -> torch.Tensor:
    """Exclude the ended document's state from the gated update.
    Multiplication (rather than indexing) keeps the graph alive for the
    non-reset lanes."""
    return m_cross_in * (~write_reset).view(-1, 1, 1).to(m_cross_in.dtype)


def apply_valid_write(
    new_m_cross: torch.Tensor, valid_write: Optional[torch.Tensor]
) -> torch.Tensor:
    """Lanes whose open-document suffix is empty carry nothing forward."""
    if valid_write is None:
        return new_m_cross
    return new_m_cross * valid_write.view(-1, 1, 1).to(new_m_cross.dtype)
