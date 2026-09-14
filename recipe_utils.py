"""Pure helpers for the training recipe — the parts of train.py worth testing.

`train.py` cannot be imported on a box without CUDA (device-health check at
import), wandb and torchdata, which is why `tests/` historically re-implemented
its logic instead of calling it — and re-implemented logic tests nothing.
Everything here depends only on torch, so `tests/test_recipe_utils.py` exercises
the code the training run actually executes.

Added 2026-09-14 with the fixes from `../recipe_sweep_findings.md`.
"""
from __future__ import annotations

import math
from typing import Optional, Sequence

import torch


def resolve_warmup_steps(scheduler_args: dict, max_training_steps: int) -> int:
    """LR warmup in steps: `warmup_steps` if set, else `warmup` x the horizon.

    Warmup is the one schedule term that should not scale with the horizon: it
    exists to let Adam's second moment mature and to survive the conversion
    transient, and both are counted in steps.  The inherited `warmup=0.0025`
    gives 125 steps at McLeish's 50,000-step horizon and 764 at B2's 305,176 —
    but only 96 at the post-batch-fix 38,147.  OLMo-2 itself warmed up for 4,000.
    """
    explicit = int(scheduler_args.get("warmup_steps", 0) or 0)
    if explicit > 0:
        return min(explicit, max_training_steps)
    return math.ceil(scheduler_args["warmup"] * max_training_steps)


def fast_forward_indices(
    order: Sequence[int], data_start_step: int, micro_batch_size: int
) -> Optional[list]:
    """The unconsumed tail of a resume's sampler order, or None if nothing to skip.

    `data_start_step` counts dataloader ITEMS (micro-batches), not rows: the row
    cursor is `data_start_step * micro_batch_size`.  With micro_batch_size=1 —
    every cortex run to date — the two coincide, which is why B2's logs read
    `Step: 431736` for 431,732 skipped rows.

    Raises when the cursor is past the end of `order`, which is the launch-blocker
    case: a corpus switch on a resume carries the cursor into a pack that may be
    smaller than it.

    Correctness rests on `order` being a pure function of (seed, len(dataset)) and
    therefore identical on every link of the chain.  A dataset whose length
    changed gets a DIFFERENT permutation, which is why such a link must reset the
    position rather than slice into it.
    """
    if data_start_step <= 1:
        return None
    consumed = data_start_step * micro_batch_size
    if consumed >= len(order):
        raise ValueError(
            f"resume cursor is past the end of the dataset: {consumed:,} rows "
            f"already consumed but the pack holds only {len(order):,}"
        )
    return list(order[consumed:])


def reduce_chunk_losses(
    chunk_losses: Sequence[torch.Tensor],
    chunk_tokens: Sequence[int],
    mode: str = "token",
) -> torch.Tensor:
    """Combine per-chunk token-mean losses into one row loss.

    "token"  weights each chunk by its unmasked-label count, so the result equals
             the token mean over the whole row — what the model's own loss does on
             the non-chunked path, and invariant to how the row was split.
    "chunk"  the pre-2026-09-14 plain mean of per-chunk means.  Equal to "token"
             only when every chunk holds the same number of unmasked labels.

    The weights are plain counts, off the autograd graph, so this changes the
    reduction and nothing else.
    """
    stacked = torch.stack(list(chunk_losses))
    if mode == "chunk":
        return stacked.mean()
    if mode != "token":
        raise ValueError(f"chunk_loss_reduction must be 'token' or 'chunk', got {mode!r}")
    if len(chunk_tokens) != len(chunk_losses):
        raise ValueError(
            f"chunk_tokens has {len(chunk_tokens)} entries for "
            f"{len(chunk_losses)} losses"
        )
    w = torch.tensor(list(chunk_tokens), device=stacked.device, dtype=stacked.dtype)
    return (stacked * w).sum() / w.sum()
