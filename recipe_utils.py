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


def worst_case_num_steps(
    mean_recurrence: int, mean_backprop_depth: int
) -> tuple[int, int]:
    """The `(n_no_grad, k_with_grad)` split that costs the most memory.

    `num_steps_sampler` (train.py:1463) draws `p` from a Poisson-of-lognormal and
    returns `k = min(s, p)`, `n = max(p - s, 0)` with `s = mean_backprop_depth`.
    So **k is hard-capped at s** however far the tail of p runs: the worst case
    for retained activations — the only case an OOM gate may measure — is `k = s`
    exactly, and a smoke that samples instead of pinning it measures a random
    draw from the distribution rather than its ceiling.

    `n` is unbounded above but runs under `no_grad`, so it costs wall-clock and
    not retained graph; it is returned at its mean-recurrence value so the
    smoke's seconds-per-step number stays representative.

    `sheduler_n_k_handler` (train.py:1488) clamps `s` down to the current mean
    recurrence whenever the ramp has not yet reached the backprop depth, so the
    same clamp is applied here — otherwise a smoke run early in the ramp would
    claim a deeper graph than the run can ever build.
    """
    if mean_recurrence < 1:
        mean_recurrence = 1
    k = min(int(mean_backprop_depth), int(mean_recurrence))
    n = max(int(mean_recurrence) - k, 0)
    return n, k


def carry_rows(chunk_index: int, accum_vecs: int, accum_max: int) -> int:
    """Rows of carry a write-once accum buffer presents to chunk `chunk_index`.

    Mirrors `AccumPrefixBuffer.__call__` (cortex_memory/buffers.py:185): rows are
    appended one chunk's worth at a time and FIFO-trimmed to `accum_max`.
    `chunk_index` is 0-based, so chunk 0 reads nothing.

    Worth its own helper because the trim is the branch that
    `b2_retrofit.sbatch:409` designed OUT of existence: the arm sets
    `ACCUM_MAX=256 = 8 x 32`, i.e. `accum_max = cross_chunks * accum_vecs`, with
    the comment "moves WITH cross_chunks".  So the FIFO never fired in training
    and fired on every eval example past 4,096 tokens — half of write-count
    drift.  Carrying that convention to cross_chunks 16 means `accum_max 512`
    and 512 carry rows on the last chunk, which is a memory fact before it is a
    mechanism one: the read at chunk 15 attends over chunk + 512 columns.
    """
    if chunk_index <= 0:
        return 0
    return min(chunk_index * int(accum_vecs), int(accum_max))


def gated_carry_rows(chunk_index: int, n_vec: int, n_slots: int,
                     fill: str = "grow") -> int:
    """Rows a gated ring presents to chunk `chunk_index`.  The twin of
    `carry_rows`, and it is a DIFFERENT function, not the same one with a
    different cap.

    An accum buffer appends and then FIFO-trims, so its width is
    `min(i * W, accum_max)` and the trim is a branch the run can be configured
    never to take.  A gated ring under `fill="grow"` grows the same way for
    exactly one lap and is then PINNED at `n_slots` forever, because merge
    overwrites in place — there is no trim to avoid and no configuration in
    which the width keeps climbing.  Under `fill="init"` the block is
    full-width from chunk 1.

    The two happen to agree while `i * W < n_slots`, which is precisely the
    "first lap is bit-identical to accum" property A1 vs A3' is built on; they
    part company at the chunk where accum would start dropping rows.  Reusing
    `carry_rows` with `accum_max=n_slots` would give the right number by
    accident and the wrong reason, and it would be silently wrong the moment
    `fill="init"` is selected.
    """
    if chunk_index <= 0:
        return 0 if fill == "grow" else int(n_slots)
    if fill != "grow":
        return int(n_slots)
    return min(int(chunk_index) * int(n_vec), int(n_slots))


def select_fwd_bwd_path(non_recurrent_model: bool, cross_chunks: int) -> str:
    """Which forward/backward implementation train.py dispatches to.

    Returns "non_rec", "cortex" (the chunk chain) or "tight" (one whole-row
    forward).  Mirrors train.py's dispatch, and exists as a helper ONLY because
    the condition is the difference between a control and a second experiment.

    Until 2026-09-14 the middle branch read `use_memory AND cross_chunks > 1`,
    so `--cortex.use_memory false` silently turned CHUNKING off as well: the
    no-memory control would have trained on whole 4,096-token rows against the
    memory model's 8x512 chunks.  Chunk length is the biggest lever the ceiling
    probes ever found, so such a run differs in two variables and isolates
    neither.  `use_memory` is deliberately NOT a parameter here: the chunk chain
    already handles a memory-less model, and re-admitting the flag is exactly
    the regression this guards.
    """
    if non_recurrent_model:
        return "non_rec"
    return "cortex" if int(cross_chunks) > 1 else "tight"


def control_has_memory(use_memory: bool, has_cortex: bool) -> bool:
    """True when a run declared no-memory but built the memory module anyway.

    The negative twin of train.py's "use_memory set but cortex is None" guard,
    and the more expensive of the two failures: a memory run that secretly has
    no memory wastes GPU-hours, but a CONTROL that secretly has memory corrupts
    every delta measured against it — the memory model's advantage vanishes and
    the result reads as "memory does not work".  Both look like a healthy loss
    curve, which is why this is checked at step 0 rather than discovered at eval.

    The 16 persisted cortex flags are loaded from the checkpoint's config.json,
    not from the command line, so `--cortex.use_memory false` is one override
    against a config that may still select a mechanism.
    """
    return (not use_memory) and has_cortex
