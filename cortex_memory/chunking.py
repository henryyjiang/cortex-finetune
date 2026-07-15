"""
Chunk-chain helpers for the cross-chunk training loop (train.py cortex_fwd_bwd).

Pure functions, importable off-cluster — train.py imports these so the unit
tests exercise the real implementation instead of a mirror copy.

  random_chunk_sizes  — AutoCompressor-style randomized segmenting
  detach_old_vecs     — AutoCompressor-style stop-gradient after N chunks
                        (AccumCCoT write-once state; exact slice detach)
"""
from __future__ import annotations

from typing import List, Optional

import torch


def random_chunk_sizes(
    seq_len: int,
    n_chunks: int,
    generator: Optional[torch.Generator] = None,
    jitter: float = 0.25,
) -> List[int]:
    """Randomized segmenting (AutoCompressor §3: training on variable-length
    segments makes the carry robust to segmentation at eval time).

    Returns n_chunks sizes summing to seq_len, each boundary jittered
    uniformly by up to ±jitter of the even chunk size.  jitter=0.25 keeps
    every chunk within [0.5, 1.5]× the even size, so no chunk degenerates.
    The same sizes apply to the whole micro-batch (tensors split along dim 1).
    """
    assert n_chunks >= 1
    base = seq_len // n_chunks
    if n_chunks == 1 or base < 4:
        return [seq_len]
    max_j = max(1, int(base * jitter))
    bounds = [0]
    for i in range(1, n_chunks):
        j = int(torch.randint(-max_j, max_j + 1, (1,), generator=generator).item())
        b = i * base + j
        # keep boundaries strictly increasing and inside the sequence
        b = max(bounds[-1] + 1, min(b, seq_len - (n_chunks - i)))
        bounds.append(b)
    bounds.append(seq_len)
    return [bounds[i + 1] - bounds[i] for i in range(n_chunks)]


def detach_old_vecs(
    state: Optional[torch.Tensor],
    n_vec: int,
    grad_chunks: int,
) -> Optional[torch.Tensor]:
    """AutoCompressor's "stop-gradient after N compression steps" for the
    AccumCCoT write-once state: rows are appended in chunk order, so the
    newest grad_chunks × n_vec rows keep their graph and everything older is
    detached ("for learning to compress the useful information in S_i it is
    sufficient to predict the tokens in the adjacent S_{i+1}" — Chevalier et
    al. 2023; no quality penalty, large graph-memory saving).

    grad_chunks == 0 disables (full-chain BPTT, the pre-existing behavior).
    """
    if state is None or grad_chunks <= 0:
        return state
    keep = grad_chunks * n_vec
    if state.shape[1] <= keep:
        return state
    return torch.cat([state[:, :-keep].detach(), state[:, -keep:]], dim=1)
