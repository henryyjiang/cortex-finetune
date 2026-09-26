"""Based on https://github.com/seal-rg/recurrent-pretraining/blob/main/finetuning_simple_example.py"""

####################################################################################################
# Imports.
####################################################################################################

import json
import time

global_start_time = time.monotonic()
import os
import socket
from typing import Any, Optional
from functools import partial
import sys
import datetime
import shutil
import inspect
import subprocess
import torch
import wandb
import math
from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler, AutoConfig
from datasets import load_dataset, Dataset, load_from_disk
from contextlib import nullcontext
from stateful_parquet_dataset import get_parquet_dataloader
from cortex_graft import reset_cortex_graft_init as _reset_cortex_graft_init
from cortex_graft import set_read_only_trainable as _set_read_only_trainable
from cortex_memory.chunking import random_chunk_sizes, detach_old_vecs
from cortex_memory.health import training_diag
from recipe_utils import (
    chunk_loss_weights,
    control_has_memory,
    fast_forward_indices,
    reduce_chunk_losses,
    resolve_warmup_steps,
    select_fwd_bwd_path,
)
from dataclasses import dataclass, field
from jsonargparse import CLI
from ellisadam import ELLISAdam

torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_math_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_cudnn_sdp(True)

# Check device health immediately after loading torch and standard libraries without loading cuda/hip/dist:
nvml_count = torch.cuda.device_count()
if nvml_count < 1:
    raise ValueError(f"Node failure! Device manager init failed on {socket.gethostname()}")

end_time = time.monotonic()
if int(os.getenv("SLURM_PROCID", "0")) == 0:
    print(f"{time.ctime()[:-5]}: Time to load libraries: {end_time - global_start_time:.02f} seconds.")


@dataclass
class CLISettings:
    run_name: str = "default-run"
    out_path: str = "huginn_llama"
    resume_path: Optional[str] = None
    # branch_path: start a NEW arm from a shared earlier run (the two-phase
    # retrofit: one healing phase, then two memory arms).  Unlike resume_path
    # it tolerates a model that has grown parameters — the gated arm adds the
    # LM2 gate on top of the healing phase's summary embeddings — and it does
    # NOT restore the optimizer or the dataloader position, because the arms
    # switch to a different dataset at the branch.  The LR schedule, the
    # mean-recurrence ramp and the step counter DO continue, which is the
    # whole point: both arms sit at the same place on one schedule.
    # Applied identically to every arm, so the branch cannot advantage one.
    branch_path: Optional[str] = None
    save_n_mins_before_timeout: Optional[int] = None
    # data
    preprocessed_data_path: Optional[str] = None
    dataset_location: str = "openai/gsm8k"
    dataset_args: dict[str, Any] = field(
        default_factory=lambda: dict(q_col="question", a_col="answer")
    )
    dataset_config: str = "main"
    max_length: Optional[int] = None
    max_samples: Optional[int] = None
    # impl
    micro_batch_size: int = 2
    compile: bool = False
    # training
    max_steps: int = 0
    # stop_at_step: end THIS phase early without moving the schedule horizon.
    # max_steps drives both the LR cosine and the mean-recurrence ramp, so it
    # must stay fixed across every phase and every resume of a multi-phase run;
    # this is the knob for "heal for 91,552 steps, then branch".  0 = off.
    stop_at_step: int = 0
    epochs: int = 1
    batch_size: int = 32
    optim_config: dict[str, Any] = field(
        default_factory=lambda: dict(lr=5e-7, weight_decay=1e-4, betas=(0.9, 0.95), eps=1e-8)
    )
    # warmup / cooldown are FRACTIONS of max_steps; min_lr = min_lr_ratio * lr.
    # warmup_steps (0 = off) overrides the fraction with an absolute step count.
    # It exists because warmup is the one schedule term that should NOT scale
    # with the horizon: it is there to let Adam's second moment mature and to
    # survive the conversion transient, and both are counted in STEPS.  The
    # inherited warmup=0.0025 gives 125 steps at McLeish's 50,000-step horizon
    # and 764 at B2's 305,176 — but only 96 at the post-batch-fix 38,147, for a
    # conversion that starts at loss ~10.3 with first-decile grad-norms of
    # 3.0-4.3 against a steady-state 1.2.  OLMo-2 itself warmed up for 4,000.
    scheduler_args: dict[float, Any] = field(
        default_factory=lambda: dict(warmup=0.1, cooldown=0.1, min_lr_ratio=0.001, warmup_steps=0)
    )
    save_interval: int = -1
    model_name: str = "smcleish/Recurrent-TinyLlama-3T-untrained"
    wandb_disabled: bool = False
    seed: int = 74
    fix_num_steps: bool = False
    # P3.0 tier 1.5.  "n,k" forces num_steps = [n, k] on every batch, e.g.
    # "0,8" = zero no-grad prefix at the trained depth, which makes the read
    # gradient live on every iteration of every batch.  Empty = off, and off
    # is the default so nothing that exists changes.  Distinct from
    # fix_num_steps (hardcoded [0,1], for compile warmup) because reusing that
    # would run the oracle probe at a single recurrent step.
    force_num_steps: str = ""
    # P3.0 tier 1.5.  Freeze everything except cortex.latent_reader, so the
    # oracle probe measures the READ and nothing else can absorb the signal.
    train_read_only: bool = False
    init_from_scratch: bool = False
    take_loss_over_all_tokens: bool = False # for chat templated datasets default is to only supervise assistant tokens
    max_grad_norm: float = 1.0
    # Abort after this many CONSECUTIVE non-finite (nan/inf) optimizer updates.
    # Each such update is skipped (weights untouched) rather than applied — nan
    # grads survive grad-clipping (nan * coef = nan) and would otherwise poison
    # the weights permanently.  A run that is nan from step 1 fails fast here
    # instead of burning hours training on garbage; transient bf16 overflow in
    # the deep recurrent unroll is skipped and training continues.
    max_nonfinite_skips: int = 20
    # Override the checkpoint's recurrence depths for the num_steps sampler (0 =
    # use model_config).  override_mean_backprop_depth shortens the TBPTT window
    # (fewer grad steps retained) — the lever for the step-500 unfreeze OOM: once
    # the loop is trainable, activations are retained across num_steps_with_grad ×
    # cross_chunks, and lowering the grad depth cuts that memory (and tames the
    # BPTT gradient).  Forward compute (mean_recurrence) is unchanged — the
    # no-grad prefix grows to keep total recurrence constant.
    override_mean_backprop_depth: int = 0
    override_mean_recurrence: int = 0
    bf16_true: bool = False
    compile_warmup_routine: bool = False
    no_amp: bool = True
    is_parquet_dataset: bool = False
    ignore_past_parquet_dataset: bool = False
    # reset_dataset_position: the NON-PARQUET twin of ignore_past_parquet_dataset,
    # and the fix for the 2026-09-14 recipe audit's finding 1.
    #
    # On the map-style path `data_start_step` is the CUMULATIVE count of
    # dataloader items drawn since the last reset (the skip loop increments it
    # too), and the only reset is a branch.  A resume that switches corpus
    # therefore fast-forwards that many items into the NEW pack before serving
    # anything: B2's 199,485 link skipped 431,732 rows of fw_nemo50 (verified in
    # logs/Report-12190106.out, first `Step: 431736`).  Two consequences —
    # roughly half the new pack is never seen, and the pack must hold
    # `cursor + steps*batch_size` rows or the loader runs dry.  Exhaustion is a
    # CLEAN EXIT here (see the guard at the end of train()), so a pack that is
    # too small produces a run wandb marks "finished" at a fraction of budget.
    #
    # `--ignore_past_parquet_dataset` does NOT cover this: train.py:561 gates it
    # on cfg.is_parquet_dataset, so it was a no-op on every B2 link.
    #
    # Pass this ONLY on the link that switches corpus.  Every later same-corpus
    # link must restore its position or it replays the pack from the top.
    reset_dataset_position: bool = False
    parquet_dataset_max_tokens: Optional[int] = None
    ignore_past_scheduler: bool = False
    mean_recurrence_schedule: dict[float, Any] = field(
        default_factory=lambda: dict(turn_on=False, warmup=0.1, max_mean_rec=32, warmup_type="linear")
    )
    mean_backprop_depth_schedule: dict[float, Any] = field(
        default_factory=lambda: dict(turn_on=False, warmup=0.1, max_backprop=8, start=1)
    )
    no_monkeypatch_on_jonas_init: bool = False
    throttle: bool = False
    non_recurrent_model: bool = False
    muon: dict[float, Any] = field(
        default_factory=lambda: dict(use_muon=False, lr=0.005, weight_decay=1e-4)
    )
    use_ellis_adam: dict[float, Any] = field(
        default_factory=lambda: dict(use_ellis_adam=False, decouple_wd=True, tensor_wise_gradient_normalization=False, tensor_wise_finite_check=False, running_init=True, atan_adam=True, update_clipping=True,)
    )
    parquet_epoching_flag_use_with_real_caution: int = 1
    # --- cortex memory graft (flag-gated; default OFF → vanilla retrofitting-recurrence) ---
    # use_memory          : master switch (also set as a config attr so the grafted
    #                       RavenForCausalLM builds self.cortex).
    # memory_slots        : K for the LM2 M_cross buffer (0 disables).
    # memory_slots_iter   : K for the per-position M_iter buffer (0 disables).
    # ccot_direct         : K=0 Coconut carry (only when memory_slots == 0).
    # cross_chunks        : split each sequence into N consecutive sub-windows and
    #                       carry M_cross un-detached between them in ONE backward.
    #                       This is what puts the M_cross write path on the loss graph
    #                       (>=3 needed to train the forget/feedback gates).  1 = off.
    # freeze_loop         : freeze adapter + core_block (the recurrent loop) — train
    #                       memory (+ coda/embeds) only.  Experiment-ladder rung 1.
    #                       (A staged-unfreeze knob, freeze_loop_until_step, was
    #                       removed 2026-09-14: never set by any sbatch, absent from
    #                       every wandb export, and training-only so no checkpoint
    #                       carried it.  Split into two runs instead — rung 1 frozen,
    #                       then resume with freeze_loop=false.)
    # eos_from_tokens     : derive eos_mask from the token ids (== tokenizer.eos)
    #                       and pass it into each chunk forward, so the M_cross
    #                       write pools only the open document suffix and resets
    #                       across doc/pad boundaries.  Off = eos_mask=None (full
    #                       carry; correct for one-doc sequences that fill the
    #                       window, and what the evals use).  Turn on when data
    #                       has padded short docs (pad == eos) or packed docs.
    # (l2sp_coeff, the experiment-ladder rung-3 L2-SP anchor, was removed
    #  2026-09-14.  Training-only, so no checkpoint's config.json carried it and
    #  nothing became unloadable; its launcher pace/rung3_l2sp.sbatch was already
    #  deleted; the one run that used it, rung3-k4-l2sp1e-3, is recorded in
    #  wandb_exports/cortex-retro-ft/summary.csv.  It also cost a branch inside
    #  cortex_fwd_bwd that was evaluated every micro-step and could never be true.)
    # memory_lr           : dedicated LR for ALL newly-added cortex params (memory
    #                       buffers + LoRA; selected by "cortex" in the param name).
    #                       Fresh zero-init modules on a pretrained base want
    #                       ~10-50x the base LR.  Their group uses weight_decay=0
    #                       (decay would pull the identity-init projections toward
    #                       zero).  0 = off (cortex params ride the default group).
    #                       NOTE: changes optimizer param-group structure — keep it
    #                       identical across runs that resume from each other.
    # lora_rank/lora_alpha: LoRA-on-loop (rung 1b): low-rank adapters on every
    #                       loop linear, base loop stays frozen (use with
    #                       freeze_loop=true), B zero-init -> step-0 == base.
    #                       rank 0 = off.
    # accum_ccot          : AutoCompressor-style accumulating carry (final-
    #                       arms round, Arm A): each chunk compressed into
    #                       accum_vecs summary vectors APPENDED to the carried
    #                       state (never overwritten; direct pathway from
    #                       every chunk to every later chunk).  Requires
    #                       memory_slots == 0 and not ccot_direct.
    # accum_vecs          : summary vectors extracted per chunk (AC used 50
    #                       per 1024-2048-token segment at 2.7-7B scale; 4-8
    #                       is proportionate at 1B with D=2048).
    # accum_max           : FIFO cap on accumulated vectors — only binds at
    #                       eval on long chunk chains (training asserts
    #                       cross_chunks * accum_vecs <= accum_max).
    #                       ACCUM ONLY; the gated buffer's capacity is
    #                       gate_slots and its cap never "binds", it gates.
    # gate_slots          : K for prefix_memory=gated — carried columns held,
    #                       i.e. the FIXED read cost per chunk, decoupled from
    #                       accum_vecs (the write width) by P1.0 on 2026-09-16.
    #                       0 = same as accum_vecs (the pre-P1.0 shape, which
    #                       forced every row to be overwritten every chunk and
    #                       measured strictly worse than accum on every axis).
    #                       Under gate_route=ring nothing has a gate_slots
    #                       dimension, so unlike accum_vecs it is a runtime knob
    #                       and can be swept at eval.
    # gate_route          : 'ring' (sparse FIFO-indexed write, the default and
    #                       the recommendation) or 'mix' (dense learned [K,W]
    #                       routing, kept as the control).  See the
    #                       PrefixGatedBuffer docstring for the three reasons
    #                       sparse wins — key selectivity, truncated BPTT, and
    #                       keeping the depth->row map.
    # gate_norm           : what the memory half of the gate sees — 'tanh'
    #                       (LM2 as published), 'rms', 'none'.  60-75% of the
    #                       real state's entries sit past |2| where tanh is
    #                       saturated, so this is a measured question.
    # gate_init           : 'zero' makes step 0 the constant EMA
    #                       0.731*state + 0.5*candidate and leaves the read
    #                       fully live; 'default' restores kaiming gate weights.
    # gate_forget_bias    : the forget gate's bias at INIT, i.e. the buffer's
    #                       horizon before training moves it: fg =
    #                       sigmoid(bias), retention F(d) = ig * fg**floor(d/(K/W)).
    #                       1.0 is LM2 as published (fg 0.731, half-life ~2.2
    #                       chunks).  At cross_chunks 8 the gate never sees
    #                       content older than 8 chunks, so NOTHING pushes this
    #                       up during training and the init value is the whole
    #                       horizon -- BABILong 16k wants >= 1.50, 32k >= 2.25.
    #                       Raising it is free in parameters and cost; it is a
    #                       DECISION, and it changes what an arm is comparable
    #                       to, so do not move it mid-comparison.
    # gate_fill           : rows a ring has not reached on its first lap —
    #                       'grow' (emit only written rows; the buffer IS
    #                       PrefixAccumBuffer for exactly one lap, so an accum
    #                       arm and a gated arm diverge at eviction and nowhere
    #                       else) or 'init' (pad from a learned slot_init).
    # gated_accum         : gated-accumulation LM2 variant (two-track plan
    #                       buffer-choice note): the K-slot M_cross becomes a
    #                       GatedAccumBuffer — AccumCCoT's extraction write
    #                       (learned query embeddings over loop-final h_T)
    #                       merged by the LM2 gate instead of appended.
    #                       Requires memory_slots > 0 (target k=16/32).
    #                       Extraction is shared code with AccumCCoT, so
    #                       append vs gated-overwrite is the only difference
    #                       between the arms.  State is mixed (rows not
    #                       separable), so carry_grad_chunks falls back to
    #                       whole-carry TBPTT-N detach, like the LSTMBuffer.
    # carry_grad_chunks   : stop-gradient horizon in chunks (AutoCompressor:
    #                       gradients stop after 2 compression steps — no
    #                       quality penalty, big graph-memory saving; frees
    #                       memory for denser cadence, e.g. cross_chunks=8).
    #                       accum_ccot: exact per-chunk slice detach (write-
    #                       once rows stay separable).  LSTMBuffer/DirectCCoT
    #                       (mixed state, rows not separable): the whole
    #                       carry is detached every N chunks = TBPTT-N.
    #                       0 = full-chain BPTT (pre-existing behavior).
    # random_segments     : randomized segmenting (AutoCompressor): jitter the
    #                       chunk boundaries ±25% of the even size each micro-
    #                       batch, so the carry is robust to variable segment
    #                       lengths at eval.
    cortex: dict[str, Any] = field(
        default_factory=lambda: dict(
            use_memory=False, memory_slots=0, memory_slots_iter=0, memory_heads=4,
            ccot_direct=False, h_T_proj=True, cross_chunks=1,
            freeze_loop=False, eos_from_tokens=False,
            memory_lr=0.0, lora_rank=0, lora_alpha=32.0,
            accum_ccot=False, accum_vecs=4, accum_max=64,
            gated_accum=False,
            # prefix_memory: '' | 'accum' | 'gated' — the AutoCompressor-
            # faithful carry (2026-08-02).  'accum' = PrefixAccumBuffer
            # (append), 'gated' = PrefixGatedBuffer (LM2 gate at fixed width).
            # Selecting it disables every other cross-segment path: the carry
            # is spliced into the token stream and read by the base model's own
            # attention, so there is no read module to configure.
            prefix_memory="",
            # The gated buffer geometry (P1.0).  Defaults reproduce the
            # pre-P1.0 shape (gate_slots 0 => K == W) so nothing moves
            # unless a run asks for it; the recommended arm is
            #   accum_vecs 16, gate_slots 64, gate_route ring.
            gate_slots=0, gate_route="ring", gate_norm="tanh",
            gate_init="zero", gate_fill="grow", gate_forget_bias=1.0,
            # THE Z CHANNEL (the dual-channel carry).  latent_carry false is
            # byte-identical to B2 and to every arm run so far; true widens the
            # carried tensor to 2D (E at [...,:D], Z at [...,D:]), adds a second
            # gate on the gated buffer, and turns on the s0 substitution in the
            # modeling file.
            #
            # latent_depth_rule is the one knob P0.7 exists to set
            # (evals/diag_depth_band.py).  It defaults to "absolute" because
            # that is what P0.1 measured (band t ~ 2..9 at a single T), NOT
            # because the question is settled -- P0.1 could not distinguish
            # "steps 2..9 whatever T is" from "the first quarter of the loop",
            # and the two diverge sharply at T=32.  The banner says so on every
            # run.  Do not launch a Z arm before P0.7 reports.
            latent_carry=False, latent_depth_rule="absolute",
            latent_depth_lo=2, latent_depth_hi=9, latent_renorm="none",
            # P3.0 — WHERE Z IS READ.  The s0 site is MEASURED DEAD (job
            # 13297293, read through RED 11's units fix: deleting the carried
            # columns outright costs +4.1e-5 / -2.5e-5 nats, opposite signs
            # across the two arms, i.e. noise), and a single no-grad step cuts
            # its read gradient to exactly zero.  Both defaults below reproduce
            # every arm on record anyway -- latent_s0_read True, latent_read
            # 'none' -- so nothing moves unless an arm asks.
            #
            #   latent_read  'refresh' = Option 0, one scalar re-adding Z into
            #                the carried columns every iteration (tests whether
            #                the loop WASHES s0 rather than whether s0 has no
            #                gain); 'xattn' = Option 1, the LatentRead module.
            #   latent_read_depth  'matched' reads only the rows whose taped
            #                depth equals this iteration.  A FLAG and not a
            #                code change, same reasoning as latent_depth_rule.
            latent_s0_read=True, latent_read="none", latent_read_depth="none",
            latent_read_heads=8, latent_read_gate_init=0.1,
            # P3.0 tier 1.5's CONTROL arm: the read sees another document's
            # carried Z (a roll of the batch), so treatment and control
            # differ in CONTENT CORRESPONDENCE and nothing else -- same
            # module, same capacity, same seed, same data order.
            latent_read_scramble=False,
            # Z ATTEMPT 2 (J1).  Defaults reproduce every arm on record; see
            # CortexMemory.__init__ for what each switch does and why.
            #   latent_encoding    'delta' | 'endpoint' | 'tokens' (Step 0 picks)
            #   latent_read_znorm  'rms' rescales Z rows to a fixed row norm
            #                      before the xattn read, so encodings differ in
            #                      content and not in scale
            #   latent_write_only  J1's no-read limb, deliberately
            #   e_dropout          blank the spliced E rows of this share of
            #                      sequences per forward (training only)
            latent_encoding="delta", latent_tok_pool=4,
            latent_read_znorm="none", latent_read_znorm_target=3.0,
            latent_write_only=False, e_dropout=0.0,
            # Z ATTEMPT 2 (J3), cortex_memory/scratchpad.py.  latent_read
            # 'scratch' + latent_encoding 'scratch' is the in-loop scratchpad;
            #   latent_carry_read          False = J3's no-read limb (carry
            #                              unread, scratchpad read kept)
            #   latent_read_gate_lr_mult   the read gate's LR multiplier, as a
            #                              reparameterisation (a param group
            #                              would break the branch's LambdaLR)
            #   latent_scratch_forget_bias the scratchpad's per-iteration fg
            latent_carry_read=True, latent_read_gate_lr_mult=1.0,
            latent_scratch_forget_bias=1.0,
            # Z ATTEMPT 2 (J4), cortex_memory/latent_embed.py.  NO NEW KEY:
            # latent_read='embeds' splices the carried Z as its own K columns of
            # `input_embeds` beside E's, so the base model's OWN pretrained
            # attention does the reading -- the same path that already reads E
            # selectively on every loop iteration.  It requires
            # latent_encoding='endpoint' (D3's vindicated write),
            # latent_s0_read=false, and latent_read_znorm='rms' with
            # latent_read_znorm_target at E's MEASURED row norm (~171); the
            # graft raises on each, because every one of them would otherwise
            # run a different design under J4's name.  latent_carry_read=false
            # is J4's no-read limb and splices zeros at the same columns.
            # diag_interval: every N optimizer steps, log the architecture's
            # health (carry rank + per-channel norms, whether either gate has
            # left its exactly-zero init, the Z read/write gradient fractions)
            # to wandb AND to <out_path>/<run_name>/cortex_diag.jsonl.
            #
            # 0 = OFF, and off is the default so every arm on record keeps its
            # exact behaviour.  It is not free -- one SVD of a [K, D] matrix per
            # call -- and it is not optional for a PROBE: every quantity that
            # decides whether an arm is worth finishing moves in the first few
            # hundred steps and is settled thereafter, so reading them off the
            # final checkpoint answers the question too late and shows no
            # trajectory.  pace/p1_arms.sbatch sets it under PROBE=1.
            diag_interval=0,
            # summary_init_token: token whose embedding seeds the summary
            # slots (AutoCompressor uses EOS).  -1 = take config.eos_token_id;
            # set it explicitly when the checkpoint config carries none, which
            # RavenConfig by default does not.
            summary_init_token=-1,
            # prefix_pos / prefix_eos_reset: both are now SINGLE-VALUED.  Their
            # alternate branches were retired 2026-09-15 (the graft asserts the
            # value instead of branching on it) after
            # pace/check_tier3_compat.sh came back clear across 148 surviving
            # config.json files on scratch.  They stay HERE, and in the persist
            # list below, because both keys are in every memory checkpoint's
            # config.json and the graft is their load path — a flag in the
            # 16-flag persist list may be quarantined but never removed.
            #
            # prefix_pos 'tail': the trailing summary slots sit at S+1..S+n_vec,
            # continuing the chunk's numbering, so the write reads the chunk at
            # POSITIVE relative offsets.  The retired 'zero' layout put
            # everything non-token at position 0, i.e. the whole write in a
            # negative-offset RoPE regime the base model never saw.
            prefix_pos="tail",
            # prefix_eos_reset False: the carry crosses document boundaries, as
            # the backbone's own attention already does inside a chunk.  The
            # retired True zeroed the WHOLE incoming carry on any chunk holding
            # an EOS, which switched the read off for ~60% of chunks on
            # EOS-separated packed data.  See CortexMemory._carried_state.
            prefix_eos_reset=False,
            carry_grad_chunks=0, random_segments=False,
            # window_backward (J1, 2026-09-22): call backward once per carry
            # WINDOW instead of once per row.  cortex_fwd_bwd keeps every
            # chunk's loss -- and with it that chunk's whole activation graph --
            # until the single backward at the end, so carry_grad_chunks cuts
            # gradient paths but frees NOTHING: cc16 OOMed at carry_grad_chunks
            # 2 and 1 alike (13304813/13304814), and J1 at micro 2 OOMed on the
            # no-read limb too (13456835, 138.71 of 139.80 GiB).  On a GATED
            # buffer the whole carry detaches every carry_grad_chunks chunks,
            # which splits the row's graph into independent windows; backprop
            # each as it closes and the peak holds one window, not the row.
            # EXACT: the row loss is a weighted sum over chunks with weights
            # fixed up front (recipe_utils.chunk_loss_weights), so the window
            # gradients add up to the one-backward gradient, to fp summation
            # order.  Refused at startup where it cannot apply (accum's slice
            # detach overlaps windows; carry_grad_chunks 0 has none) and under
            # DDP, whose reducer expects one backward per sync step.  Default
            # off, so every arm on record keeps its exact code path.
            window_backward=False,
            # chunk_loss_reduction: how the per-chunk losses in cortex_fwd_bwd
            # combine into the row's loss.
            #   "token" (default since 2026-09-14) — weight each chunk by its
            #       unmasked label count, so the row's loss equals the token mean
            #       over the whole row.  This is what the model's own loss does
            #       on the non-chunked path (tightly_scoped_fwd_bwd), so the two
            #       paths agree, and it is invariant to how the row is split —
            #       which matters because cross_chunks varies across arms.
            #   "chunk" — the pre-2026-09-14 behaviour, a plain mean of per-chunk
            #       token-means.  Identical to "token" only when every chunk
            #       holds the same number of unmasked labels.
            # Magnitude, measured: on today's PG-19 pack ~1% of rows are short at
            # max_length 4096 (~5% at 8192), so the two agree almost everywhere.
            # After the planned striding fix to prepare_pg19_dataset.py the LAST
            # window of every book is ragged, taking that to ~1 in 8 rows — which
            # is why this is a flag and why the default moved before the pack was
            # built rather than after.
            chunk_loss_reduction="token",
        )
    )

    def __post_init__(self):
        assert self.micro_batch_size <= self.batch_size, "batch size must be less than micro batch size"

        self.amp_args = {"device_type": "cuda", "dtype": torch.bfloat16}
        if self.no_amp:
            # https://github.com/Lightning-AI/pytorch-lightning/pull/20921
            # https://github.com/pytorch/pytorch/issues/65766
            self.amp_args["enabled"] = False
            self.amp_args["cache_enabled"] = False
        else:
            # i.e. we haven't turned amp off
            self.amp_args["enabled"] = True
            self.amp_args["cache_enabled"] = self.compile and (not self.bf16_true) # can only use cache if compiled and in float32

        assert self.batch_size % self.micro_batch_size == 0, "grad accum steps must be an int"
        assert not (self.max_steps and self.stop_at_step
                    and self.stop_at_step > self.max_steps), (
            "stop_at_step is an early stop inside the max_steps horizon")
        assert not (self.resume_path and self.branch_path), (
            "resume_path continues THIS run; branch_path starts a new arm from "
            "another run.  Set exactly one."
        )
        # Retired mechanisms: refuse to START a run on one.  The check lives here
        # rather than in cortex_graft.CortexMemory because the graft is also the
        # LOAD path for every Track-A and B1 checkpoint, whose config.json still
        # carries these flags — raising there made the entire historical results
        # table unloadable (found 2026-08-04, when the write-capacity diagnostic
        # could not open the checkpoint it was written to explain).  New training
        # runs are the only place the flag is a mistake rather than a fact.
        for dead, replacement in (("accum_ccot", "--cortex.prefix_memory accum"),
                                  ("gated_accum", "--cortex.prefix_memory gated"),
                                  ("ccot_direct", "(retired; no replacement)")):
            if self.cortex.get(dead):
                raise ValueError(
                    f"cortex.{dead} selects a mechanism retired on 2026-08-02. "
                    f"Use {replacement} for new runs.  (Evaluating an existing "
                    f"checkpoint that carries this flag still works — the graft "
                    f"builds the legacy buffer for the load path.)")
        if self.cortex["prefix_memory"]:
            assert self.cortex["prefix_memory"] in ("accum", "gated"), (
                "cortex.prefix_memory must be '', 'accum' or 'gated'; got "
                f"{self.cortex['prefix_memory']!r}"
            )
            assert self.cortex["use_memory"], "cortex.prefix_memory needs use_memory=true"
            assert self.cortex["cross_chunks"] > 1, (
                "cortex.prefix_memory is a cross-segment carry — it only trains "
                "through the chunk chain (cross_chunks > 1)"
            )
            assert not (self.cortex["accum_ccot"] or self.cortex["gated_accum"]
                        or self.cortex["ccot_direct"] or self.cortex["memory_slots"]), (
                "cortex.prefix_memory replaces every other cross-segment path — "
                "set memory_slots=0 and accum_ccot/gated_accum/ccot_direct=false"
            )
            if self.cortex["prefix_memory"] == "accum":
                # The FIFO must hold the whole chain, or early chunks are
                # silently dropped and the arm stops being an accumulation test.
                assert (self.cortex["cross_chunks"] * self.cortex["accum_vecs"]
                        <= self.cortex["accum_max"]), (
                    f"accum_max ({self.cortex['accum_max']}) must hold "
                    f"cross_chunks x accum_vecs "
                    f"({self.cortex['cross_chunks']} x {self.cortex['accum_vecs']})"
                )
            if self.cortex["prefix_memory"] == "gated":
                for key, allowed in (("gate_route", ("ring", "mix")),
                                     ("gate_norm", ("tanh", "rms", "none")),
                                     ("gate_init", ("zero", "default")),
                                     ("gate_fill", ("grow", "init"))):
                    assert self.cortex[key] in allowed, (
                        f"cortex.{key} must be one of {allowed}; got "
                        f"{self.cortex[key]!r}")
                fb = float(self.cortex["gate_forget_bias"])
                assert math.isfinite(fb) and abs(fb) <= 10.0, (
                    f"cortex.gate_forget_bias ({fb}) is outside [-10, 10]: "
                    "sigmoid saturates well before that, so a value out here "
                    "is a typo, and a saturated forget gate either never "
                    "forgets or never keeps")
                K = int(self.cortex["gate_slots"]) or int(self.cortex["accum_vecs"])
                W = int(self.cortex["accum_vecs"])
                assert K >= W, (
                    f"cortex.gate_slots ({K}) < cortex.accum_vecs ({W}): the "
                    "gated buffer cannot hold fewer rows than one chunk writes")
                if self.cortex["gate_route"] == "ring":
                    assert K % W == 0, (
                        f"cortex.gate_slots ({K}) must be a multiple of "
                        f"cortex.accum_vecs ({W}) under gate_route=ring, or a "
                        "partial lap writes a different row set every time "
                        "round and the depth->row map breaks")
                    # THE CONSTRAINT NOTHING ELSE CHECKS.  A ring's gate only
                    # starts acting on lap 2 -- the first lap is a plain append
                    # (gate_fill=grow) or a pad (init).  If the chain is shorter
                    # than two laps the gate NEVER FIRES in training and the run
                    # trains an accum buffer with dead gate parameters bolted
                    # on, behind a perfectly healthy loss curve.  The gated
                    # analogue of "raising accum_max moves the cliff, it does
                    # not train eviction".
                    lap = K // W
                    assert self.cortex["cross_chunks"] >= 2 * lap, (
                        f"cortex.cross_chunks ({self.cortex['cross_chunks']}) "
                        f"must be at least 2 x the ring lap ({lap} = gate_slots "
                        f"{K} / accum_vecs {W}) = {2 * lap}, or the gate never "
                        "fires during training and receives zero gradient.  "
                        "Lower gate_slots, raise accum_vecs, or raise "
                        "cross_chunks.")
        if self.cortex["latent_carry"]:
            assert self.cortex["prefix_memory"] in ("accum", "gated"), (
                "cortex.latent_carry needs --cortex.prefix_memory accum|gated. "
                "Z is read by substituting into the CARRIED COLUMNS of s0, so "
                "without a prefix buffer there are no carried columns: the read "
                "would be a no-op and the arm would train a write that nothing "
                "consumes.")
            assert self.cortex["latent_depth_rule"] in ("absolute", "relative"), (
                f"cortex.latent_depth_rule must be 'absolute' or 'relative'; "
                f"got {self.cortex['latent_depth_rule']!r}")
            assert self.cortex["latent_renorm"] in ("none", "s0"), (
                f"cortex.latent_renorm must be 'none' or 's0'; got "
                f"{self.cortex['latent_renorm']!r}")
            lo, hi = (int(self.cortex["latent_depth_lo"]),
                      int(self.cortex["latent_depth_hi"]))
            assert 1 <= lo <= hi, (
                f"cortex.latent_depth_lo/hi ({lo}/{hi}) must satisfy 1 <= lo "
                f"<= hi.  Depth 0 is s0 itself, which is what Z REPLACES, not "
                f"something to write.")
            assert lo >= 2, (
                f"cortex.latent_depth_lo is {lo}.  d_1 measured 23.8x the noise "
                f"it replaces -- a different regime, not a larger version of "
                f"the same one -- and writing it would put one slot three "
                f"orders of magnitude off the rest.  Set 2 or higher, or turn "
                f"on --cortex.latent_renorm s0 deliberately.")
            # The Z write only trains while the depths it samples are inside the
            # gradient window, and the no-grad steps run FIRST, so the trainable
            # region is always the LAST mean_backprop_depth iterations.  This is
            # a config-level warning, not an assert, because a frozen Z write is
            # a legitimate CHOICE at high recurrence -- it just has to be one.
            mr = int(self.mean_recurrence_schedule.get("max_mean_rec", 0) or 0)
            # The model config owns mean_backprop_depth and is not loaded yet,
            # so use the override when one is set and otherwise assume the
            # retrofit checkpoints' 8.  Named as an assumption rather than read
            # as a fact: a wrong guess here costs a spurious warning, while
            # staying silent costs a frozen write path nobody decided on.
            depth = int(self.override_mean_backprop_depth or 8)
            if mr and mr - depth >= hi:
                print(f"[cortex] WARNING: latent_carry with max_mean_rec {mr} "
                      f"and mean_backprop_depth {depth} (assumed; set "
                      f"--override_mean_backprop_depth to be sure) freezes the first "
                      f"{mr - depth} loop steps, which covers the whole write "
                      f"band {lo}..{hi}.  Z becomes a FROZEN FEATURE EXTRACTOR: "
                      f"the model can learn to USE it, never to SHAPE it.  That "
                      f"is the mirror of the frozen-READ failure that cost "
                      f"x0.90 -> x1.33.  Raise mean_backprop_depth or accept it "
                      f"IN THE PRE-REGISTRATION -- it changes what a null "
                      f"result for Z means.")
        if self.cortex["accum_ccot"]:
            assert self.cortex["memory_slots"] == 0 and not self.cortex["ccot_direct"], (
                "cortex.accum_ccot replaces the K-slot buffer / DirectCCoT — "
                "set memory_slots=0 and ccot_direct=false"
            )
            assert self.cortex["cross_chunks"] > 1, (
                "cortex.accum_ccot is a cross-segment carry — it only trains "
                "through the chunk chain (cross_chunks > 1)"
            )
            assert self.cortex["cross_chunks"] * self.cortex["accum_vecs"] \
                <= self.cortex["accum_max"], (
                "accum_max must hold every chunk's vectors during training "
                "(cross_chunks * accum_vecs) — the FIFO cap is an eval device, "
                "silently trimming during training would break the stop-grad "
                "slice bookkeeping"
            )
        if self.cortex["gated_accum"]:
            assert self.cortex["memory_slots"] > 0, (
                "cortex.gated_accum swaps the K-slot buffer's write path — "
                "it needs memory_slots > 0 (target k=16/32)"
            )
            assert not self.cortex["accum_ccot"], (
                "cortex.gated_accum and accum_ccot are the two arms of the "
                "append-vs-gated-overwrite comparison — run one at a time"
            )
            assert self.cortex["cross_chunks"] > 1, (
                "cortex.gated_accum is a cross-segment carry — it only trains "
                "through the chunk chain (cross_chunks > 1)"
            )
        if self.cortex["random_segments"]:
            assert self.cortex["cross_chunks"] > 1, (
                "cortex.random_segments varies the chunk boundaries — needs "
                "cross_chunks > 1"
            )
        if self.is_parquet_dataset:
            assert (self.parquet_dataset_max_tokens is not None) or (self.max_steps != 0), "if using parquet need to specify max tokens or max steps"
            assert self.max_length is not None, "if using parquet need to specify max_length of context"

        if self.non_recurrent_model:
            assert not self.throttle, "Can't use throttle with non_recurrent_model"
            assert not self.mean_backprop_depth_schedule["turn_on"], "Can't use mean_backprop_depth_schedule with non_recurrent_model"
            assert not self.mean_recurrence_schedule["turn_on"], "Can't use mean_recurrence_schedule with non_recurrent_model"
            assert not self.compile_warmup_routine, "Can't use compile_warmup_routine with non_recurrent_model"

            self.no_monkeypatch_on_jonas_init = True # turn off for normal models

@dataclass
class Message:
    role: str
    content: str

def get_flux_timeleft():
    result = subprocess.run(
        ["flux", "job", "timeleft"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
        text=True
    )
    return int(result.stdout.strip())

def parse_slurm_timeleft(raw: str) -> int:
    """Seconds from a Slurm `squeue -o %L` duration: [[D-]HH:]MM:SS."""
    raw = raw.strip()
    days, _, rest = raw.rpartition("-")
    parts = [int(p) for p in rest.split(":")]
    while len(parts) < 3:
        parts.insert(0, 0)
    h, m, s = parts
    return ((int(days) if days else 0) * 24 + h) * 3600 + m * 60 + s

def _query_slurm_deadline():
    """Job end time as epoch seconds, or None if it cannot be determined.

    Prefers $SLURM_JOB_END_TIME (exported by many Slurm builds, epoch seconds,
    no subprocess at all) and falls back to ONE squeue call.
    """
    end = os.getenv("SLURM_JOB_END_TIME")
    if end:
        try:
            return float(end)
        except ValueError:
            pass
    job_id = os.getenv("SLURM_JOB_ID")
    if job_id is None:
        return None
    try:
        result = subprocess.run(
            ["squeue", "-h", "-j", job_id, "-o", "%L"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            check=False, text=True, timeout=60,
        )
        if result.returncode == 0 and result.stdout.strip():
            return time.time() + parse_slurm_timeleft(result.stdout)
    except Exception:
        pass
    return None


def _query_flux_deadline():
    try:
        result = subprocess.run(
            ["flux", "job", "timeleft"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            check=False, text=True, timeout=60,
        )
        if result.returncode == 0 and result.stdout.strip():
            return time.time() + int(result.stdout.strip())
    except Exception:
        pass
    return None


_timeleft_deadline = None      # epoch seconds; resolved exactly once
_timeleft_resolved = False


def get_timeleft():
    """Seconds of walltime left, on either scheduler.

    Resolve the deadline ONCE, then do pure local clock arithmetic.  Two
    incidents drove this shape, both worth not repeating:

    1. Upstream only spoke Flux (LLNL).  On PACE/Slurm there is no `flux`
       binary, so `subprocess.run(..., check=True)` raised FileNotFoundError
       and killed the run at the first step that checked.

    2. Replacing it with a per-step `squeue` call was WORSE (B1 jobs
       11561763/64, 2026-07-30).  check_if_save runs every optimizer step, so
       two jobs at ~2,000 steps/h issued ~4,000 squeue calls an hour between
       them.  At 04:01:02, 7.1h in, the controller returned non-zero to BOTH
       jobs inside the same millisecond and check=True turned that transient
       scheduler hiccup into a fatal CalledProcessError.  Training was
       perfectly healthy at the time; ~4,600 steps of progress were lost.

    So: no repeated subprocess calls, and a helper whose whole job is to
    PROTECT a run must never be able to kill it.  Any failure to determine the
    deadline disables pre-timeout saves (returns inf) rather than raising --
    the periodic save_interval checkpoints are the fallback.
    """
    global _timeleft_deadline, _timeleft_resolved
    if not _timeleft_resolved:
        _timeleft_resolved = True
        _timeleft_deadline = (_query_slurm_deadline()
                              if os.getenv("SLURM_JOB_ID") is not None
                              else _query_flux_deadline())
        if _timeleft_deadline is None:
            print("[timeout-save] could not determine the walltime deadline; "
                  "pre-timeout saves are DISABLED (periodic save_interval "
                  "checkpoints still run).")
        else:
            print(f"[timeout-save] walltime deadline resolved: "
                  f"{_timeleft_deadline - time.time():.0f}s from now")
    if _timeleft_deadline is None:
        return float("inf")
    return _timeleft_deadline - time.time()

has_completed_timeout_save = False
def check_if_save(save_n_mins_before_timeout):
    global has_completed_timeout_save
    if (save_n_mins_before_timeout * 60 > get_timeleft()) and (not has_completed_timeout_save):
        has_completed_timeout_save = True
        return True
    return False

def save_model_only(cfg, state, chkpt_name):
    unwrapped_model = get_unwrapped_model(state)
    unwrapped_model.save_pretrained(f"{cfg.out_path}/{cfg.run_name}/{chkpt_name}")
    state["tokenizer"].save_pretrained(f"{cfg.out_path}/{cfg.run_name}/{chkpt_name}")

def save_checkpoint(state, agg_vars_dict, cfg):
    # agg_vars_dict = {"data_start_step": data_start_step, "optimizer_step": optimizer_step, "total_tokens": total_tokens, "total_tokens_with_loss": total_tokens_with_loss}
    step = agg_vars_dict["optimizer_step"]
    if cfg.is_parquet_dataset:
        # have to call this on all nodes as there is an internal gather
        dataloader_state = state["dataloader"].state_dict()
    else:
        dataloader_state = None
    
    if cfg.muon["use_muon"]:
        # muon does an all gather on saving
        optim_state_dict = state["optimizer"].state_dict()
    elif is_main_process():
        optim_state_dict = state["optimizer"].state_dict()

    if not is_main_process():
        return

    extras = {}
    if cfg.mean_recurrence_schedule["turn_on"]:
        extras["mean_recurrence_scheduler"] = state["mean_recurrence_scheduler"].state_dict()
    if cfg.mean_backprop_depth_schedule["turn_on"]:
        extras["mean_backprop_depth_scheduler"] = state["mean_backprop_depth_scheduler"].state_dict()

    unwrap = get_unwrapped_model(state)
    ckpt = dict(
        model=unwrap.state_dict(),
        optimizer=optim_state_dict,
        scheduler=state["scheduler"].state_dict(),
        dataloader=dataloader_state,
        rng_state=torch.get_rng_state(),
        cuda_rng_state=torch.cuda.get_rng_state_all(),
        agg_vars_dict=agg_vars_dict,
        cfg=cfg.__dict__, # for provenance
        **extras,
    )

    chkpt_dir = f"{cfg.out_path}/{cfg.run_name}/checkpoint_{step}"
    os.makedirs(chkpt_dir, exist_ok=True)
    torch.save(ckpt, f"{chkpt_dir}/chkpt.pt")
    print(f"[rank 0] Saved checkpoint @ step {step:,}")

def load_checkpoint(state, cfg, device, branch: bool = False):
    path = cfg.branch_path if branch else cfg.resume_path
    ckpt = torch.load(f"{path}/chkpt.pt", map_location=device)
    unwrap = get_unwrapped_model(state)
    # tools/slice_summary_emb.py narrows a trained summary_emb (W=32 -> 16) so a
    # P1 arm can branch at a width the parent never trained at.  It stamps this
    # record on the checkpoint.  A BRANCH is the intended use and is announced
    # loudly; a RESUME is not, because the optimizer state in that file still
    # carries the PARENT's [W_old, D] moments for a parameter that is now
    # [W_new, D], and load_state_dict would either raise here or, on a future
    # optimizer that reshapes silently, carry the wrong second moment for the
    # whole write path behind a healthy loss curve.
    sliced = ckpt.get("cortex_slice")
    if sliced is not None:
        if not branch:
            raise RuntimeError(
                f"resume_path={path} is a summary_emb-SLICED checkpoint "
                f"({sliced.get('W_old')} -> {sliced.get('W_new')} write "
                f"columns, written {sliced.get('when')}).  Its optimizer state "
                f"still has the parent's width.  Slices are for --branch_path; "
                f"resume the PARENT run instead, or branch off this one.")
        if is_main_process():
            print(f"[branch] summary_emb was SLICED {sliced.get('W_old')} -> "
                  f"{sliced.get('W_new')} rows ({sliced.get('rule')}), "
                  f"{sliced.get('when')}, from {sliced.get('src')}")
            k, a = sliced.get("kept") or {}, sliced.get("all") or {}
            if k and a:
                print(f"[branch]   effective rank {k.get('participation_rank', 0):.2f} "
                      f"of the parent's {a.get('participation_rank', 0):.2f}; "
                      f"kept-row centred cos {k.get('centred_cos_mean', 0):+.4f}")
    if branch:
        # The arm may have MORE parameters than the phase it branches from
        # (gated adds the LM2 gate on top of the shared summary embeddings).
        # Everything else must match exactly: an unexpected key means the
        # checkpoint is from a different architecture, and a missing key
        # outside the memory module means part of the backbone silently kept
        # its freshly-initialised weights.
        missing, unexpected = unwrap.load_state_dict(ckpt["model"], strict=False)
        bad = [k for k in missing if not k.startswith("cortex.")]
        if unexpected or bad:
            raise RuntimeError(
                f"branch_path={path} does not match this arm's model: "
                f"unexpected={unexpected[:8]} non-memory missing={bad[:8]}"
            )
        if is_main_process():
            print(f"[branch] loaded weights from {path}; "
                  f"{len(missing)} new memory params start fresh: {missing}")
            print("[branch] optimizer state NOT restored (param set changed); "
                  "LR schedule, recurrence ramp and step counter continue")
    else:
        unwrap.load_state_dict(ckpt["model"], strict=True)
        state["optimizer"].load_state_dict(ckpt["optimizer"])

    # A train_read_only branch (the P3.0 oracle probe) is a NEW optimisation
    # problem, not a continuation: different optimizer (AdamW, not Muon), a
    # different param-group count, and its own max_steps horizon.  Job
    # 13434102 died here on both counts -- the parent's LambdaLR carried one
    # lr_lambda per MUON group and the probe's AdamW has one group
    # (IndexError), and had it loaded, the inherited step counter (91,952)
    # would have been past max_steps=1000: ONE optimizer step at min LR,
    # then a clean "DONE" and a null result by construction.
    fresh = branch and cfg.train_read_only
    if fresh and is_main_process():
        print("[branch] train_read_only: LR schedule, recurrence schedules and "
              "step counter start FRESH (the parent's belong to a different "
              "optimizer and horizon)")

    if not fresh:
        if cfg.mean_recurrence_schedule["turn_on"] and ("mean_recurrence_scheduler" in ckpt):
            state["mean_recurrence_scheduler"].load_state_dict(ckpt["mean_recurrence_scheduler"])
        if cfg.mean_backprop_depth_schedule["turn_on"] and ("mean_backprop_depth_scheduler" in ckpt):
            state["mean_backprop_depth_scheduler"].load_state_dict(ckpt["mean_backprop_depth_scheduler"])

    if not cfg.ignore_past_scheduler and not fresh:
        n_saved = len(ckpt["scheduler"].get("lr_lambdas") or [])
        n_now = len(state["optimizer"].param_groups)
        if n_saved and n_saved != n_now:
            raise RuntimeError(
                f"{path}: the saved LR scheduler has {n_saved} param groups and "
                f"this run's optimizer has {n_now} (optimizer or memory_lr/"
                f"throttle changed).  LambdaLR cannot map one onto the other.  "
                f"Pass --ignore_past_scheduler true if a fresh schedule is intended.")
        state["scheduler"].load_state_dict(ckpt["scheduler"])
    # A branch switches datasets (healing corpus -> the arms' corpus), so the
    # saved parquet position is meaningless and restoring it would skip into
    # the middle of a file that has nothing to do with it.
    if cfg.is_parquet_dataset and not cfg.ignore_past_parquet_dataset and not branch:
        state["dataloader"].load_state_dict(ckpt["dataloader"])

    torch.set_rng_state(ckpt["rng_state"].to("cpu"))
    torch.cuda.set_rng_state_all([rng.to("cpu") for rng in ckpt["cuda_rng_state"]])
    print(f"{'Branched' if branch else 'Resumed'} from {path}")
    agg = dict(ckpt["agg_vars_dict"])
    if fresh:
        agg.update(optimizer_step=0, total_tokens=0,
                   total_tokens_with_loss=0, elapsed_time=0.0)
    if branch:
        agg["data_start_step"] = 1        # fresh corpus, read it from the top
    elif cfg.reset_dataset_position:
        # Same run, new corpus (see CLISettings.reset_dataset_position).  The
        # restored cursor was taken against the OLD pack; keeping it would skip
        # that many rows into the new one.
        if is_main_process():
            print(f"[data] reset_dataset_position: discarding restored dataloader "
                  f"cursor {agg['data_start_step']:,} — reading "
                  f"{cfg.preprocessed_data_path} from row 0")
        agg["data_start_step"] = 1
    return agg

def is_main_process():
    if torch.distributed.is_initialized():
        return torch.distributed.get_rank() == 0
    else:
        return True

def seed_everything(seed):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed) 
    torch.manual_seed(seed)

def get_unwrapped_model(state):
    if isinstance(state, dict):
        return state["model"].module if state["distributed"] else state["model"]
    else:
        # Detect the wrapper structurally (DDP `.module` / compile `_orig_mod`)
        # rather than using is_initialized() as a DDP proxy: single-GPU Muon
        # runs now form a 1-process group, so is_initialized() is True even
        # though the model is unwrapped.
        return get_unwrapped_model_from_module(state)


####################################################################################################
# Main driver functions.
####################################################################################################
# DEFAULT_SYS_PROMPT = "You are a helpful assistant that can help users with mathematical reasoning."
DEFAULT_SYS_PROMPT = "You are a helpful assistant that can assist users with mathematical reasoning."

def initialize_state_monkeypatch(self, input_embeds, scale: float = 1.0, patched_std: float = 0.008703882797784892, patched_embed_scale: float = 1.0):
    """
    Patch to fixes the std to the Huginn value and remove the embed scaling
    """
    x = torch.randn_like(input_embeds)
    std = patched_std * scale
    if std > 0:
        torch.nn.init.trunc_normal_(x, mean=0.0, std=std, a=-3 * std, b=3 * std)
        if patched_embed_scale != 1:
            x = x * self.emb_scale
    else:
        x.zero_()
    return x


def set_loop_trainable(model, trainable: bool) -> int:
    """Freeze/unfreeze the recurrent loop (adapter + core_block) in place.
    Everything else (memory, coda, embeddings, norms) keeps its grad state.
    Returns the number of loop parameters toggled.  Cortex experiment-ladder
    rung 1 = freeze loop, train memory only."""
    target = get_unwrapped_model_from_module(model)
    n = 0
    for name, p in target.named_parameters():
        if ("adapter" in name) or ("core_block" in name):
            p.requires_grad_(trainable)
            n += 1
    return n


def set_read_only_trainable(model) -> tuple:
    """Rank-0-logging wrapper over cortex_graft.set_read_only_trainable.

    The body lives in cortex_graft.py for the same reason
    reset_cortex_graft_init's does: train.py imports wandb, so ANY helper that
    lives here is unreachable from the unit suite and from the login-node
    tools.  A freeze whose correctness cannot be tested is a freeze that
    silently trains the wrong parameter set.
    """
    return _set_read_only_trainable(get_unwrapped_model_from_module(model))


def reset_cortex_graft_init(model):
    """Rank-0-logging wrapper over cortex_graft.reset_cortex_graft_init.

    The body moved to cortex_graft.py on 2026-09-16 so that every consumer which
    builds the graft through from_pretrained can apply it -- notably
    tools/smoke_prefix_real.py, which cannot import this module (wandb) and was
    therefore smoking a buffer whose gate post_init had left at forget_bias
    = -2.2e12.  Nothing about the reset itself changed; see the docstring there.
    """
    _reset_cortex_graft_init(model, log=print if is_main_process() else None)


def get_unwrapped_model_from_module(model):
    """Unwrap DDP / torch.compile to reach named_parameters with stable names."""
    m = model
    if hasattr(m, "module"):
        m = m.module
    if hasattr(m, "_orig_mod"):
        m = m._orig_mod
    return m


def _resolve_warmup_steps(cfg, max_training_steps: int) -> int:
    """LR warmup in steps — see recipe_utils.resolve_warmup_steps.

    The mean-recurrence and backprop-depth ramps deliberately keep using their
    own fractions: those are curricula over the run, not optimizer warmup.
    """
    return resolve_warmup_steps(cfg.scheduler_args, max_training_steps)


def startup(cfg: CLISettings):
    """The main setup function for the training script."""
    seed_everything(cfg.seed)
    ##########    Comms              ##############
    rank = int(os.getenv("SLURM_PROCID", os.getenv("RANK", "0")))
    local_device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")
    if torch.cuda.device_count() > 1:
        distributed = True
        torch.distributed.init_process_group(
            backend="nccl",
            rank=rank,
            world_size=int(os.getenv("SLURM_NTASKS", os.getenv("WORLD_SIZE", -1))),
            device_id=local_device,  # this immediately forms the NCCL communicator, crucial based on Sean's testing
            timeout=datetime.timedelta(hours=0.5 if cfg.is_parquet_dataset else 2), # 2hrs should be good to process for ~20M samples-ish
        )
        world_size = torch.distributed.get_world_size()
        print(f"Comms formed on rank {rank} with device {local_device} out of world size {world_size}.")
    else:
        world_size = 1
        distributed = False
        # The host MuonWithAuxAdam (pip `muon`, Keller Jordan's distributed
        # optimizer) calls dist.get_world_size() inside .step(); on a single-GPU
        # run torch.distributed is otherwise never initialized, so it raises
        # "Default process group has not been initialized".  Form a trivial
        # 1-process NCCL group so Muon runs at world_size=1.  `distributed`
        # stays False, so no DDP wrap / DistributedSampler / no_sync / metric
        # all-reduce path engages (those key off the local `distributed` var).
        if cfg.muon["use_muon"] and not torch.distributed.is_initialized():
            os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
            # Derive a per-job port from SLURM_JOB_ID: gpu-a100 nodes are shared,
            # so co-located single-GPU arms would otherwise all bind the same
            # hardcoded TCPStore port and collide (EADDRINUSE).
            default_port = 20000 + int(os.getenv("SLURM_JOB_ID", "0")) % 40000
            os.environ.setdefault("MASTER_PORT", str(default_port))
            torch.distributed.init_process_group(
                backend="nccl", rank=0, world_size=1, device_id=local_device,
            )

    weight_dtype = torch.float32
    if cfg.bf16_true:
        torch.set_default_dtype(torch.bfloat16)
        weight_dtype = torch.bfloat16
    torch.cuda.set_device(local_device)

    ########## Model and tokenizer ##############
    config = AutoConfig.from_pretrained(cfg.model_name, trust_remote_code=True)
    # cortex: push memory flags onto the config so the grafted RavenForCausalLM
    # builds self.cortex.  Requires the model dir to use the grafted modeling file
    # (see tools/prepare_cortex_checkpoint.py).  Default OFF → no-op.
    if cfg.cortex["use_memory"]:
        for _k in ("use_memory", "memory_slots", "memory_slots_iter",
                   "memory_heads", "ccot_direct", "h_T_proj",
                   "lora_rank", "lora_alpha",
                   "accum_ccot", "accum_vecs", "accum_max",
                   "gated_accum", "prefix_memory", "summary_init_token",
                   "prefix_pos", "prefix_eos_reset",
                   # P1.0 gated geometry.  These MUST persist: the graft reads
                   # them from config.json on every load, so a key missing here
                   # silently rebuilds a DIFFERENT buffer on resume or at eval
                   # (K back to W, route back to ring, gate back to kaiming)
                   # behind a healthy loss curve.
                   "gate_slots", "gate_route", "gate_norm", "gate_init",
                   "gate_fill",
                   # The Z channel.  These MUST persist for the same reason the
                   # gate keys do, and one of them harder: latent_carry changes
                   # the carried tensor's WIDTH and the buffer's PARAMETER SET,
                   # so a resume or an eval that rebuilt the graft without it
                   # would construct an E-only buffer, drop the Z gate's weights
                   # as unexpected keys, and run a half-width carry.
                   "latent_carry", "latent_depth_rule", "latent_depth_lo",
                   "latent_depth_hi", "latent_renorm",
                   # P3.0's read site.  latent_read is in the same class as
                   # latent_carry -- it ADDS PARAMETERS (16.8M for xattn), so a
                   # key missing here would rebuild the graft without the read
                   # module on resume or at eval, drop its weights as
                   # unexpected keys, and score an arm that reads nothing while
                   # reporting the arm that does.
                   "latent_s0_read", "latent_read", "latent_read_depth",
                   "latent_read_heads", "latent_read_gate_init",
                   # The control arm MUST persist: a scrambled-read cell
                   # reloaded without it becomes the treatment arm, and the
                   # two would be indistinguishable after the fact.
                   "latent_read_scramble",
                   # J1.  latent_encoding changes what the carried Z IS, and
                   # latent_write_only is the only thing that lets a no-read
                   # limb's config build at all -- an eval rebuilt without it
                   # raises.  e_dropout is training-only in effect but persists
                   # so the checkpoint records the condition it trained under.
                   "latent_encoding", "latent_tok_pool", "latent_read_znorm",
                   "latent_read_znorm_target", "latent_write_only",
                   "e_dropout",
                   # J3.  latent_read_gate_lr_mult persists HARDEST of these:
                   # the stored gate parameter is gate/mult, so a checkpoint
                   # rebuilt at mult 1.0 would read a 0.01-mult gate at 100x.
                   # latent_carry_read is the only thing that lets the no-read
                   # limb rebuild as itself; the forget bias records the arm.
                   "latent_carry_read", "latent_read_gate_lr_mult",
                   "latent_scratch_forget_bias"):
            setattr(config, _k, cfg.cortex[_k])
        if is_main_process():
            print(f"[cortex] memory ON: K={cfg.cortex['memory_slots']} "
                  f"K_iter={cfg.cortex['memory_slots_iter']} "
                  f"ccot_direct={cfg.cortex['ccot_direct']} "
                  f"accum_ccot={cfg.cortex['accum_ccot']} "
                  f"(vecs={cfg.cortex['accum_vecs']}/max={cfg.cortex['accum_max']}) "
                  f"gated_accum={cfg.cortex['gated_accum']} "
                  f"prefix_memory={cfg.cortex['prefix_memory'] or 'off'} "
                  + (f"gate(K={cfg.cortex['gate_slots'] or cfg.cortex['accum_vecs']}"
                     f",route={cfg.cortex['gate_route']}"
                     f",norm={cfg.cortex['gate_norm']}"
                     f",init={cfg.cortex['gate_init']}"
                     f",fill={cfg.cortex['gate_fill']}) "
                     if cfg.cortex['prefix_memory'] == 'gated' else "")
                  + (f"latent(Z on, rule={cfg.cortex['latent_depth_rule']}"
                     f",band={cfg.cortex['latent_depth_lo']}.."
                     f"{cfg.cortex['latent_depth_hi']}"
                     f",renorm={cfg.cortex['latent_renorm']}"
                     f",read={'s0+' if cfg.cortex['latent_s0_read'] else ''}"
                     f"{cfg.cortex['latent_read']}"
                     + (f"/{cfg.cortex['latent_read_depth']}"
                        if cfg.cortex['latent_read'] == 'xattn' else "")
                     + (",SCRAMBLED(control arm)"
                        if cfg.cortex['latent_read_scramble'] else "")
                     + (",WRITE-ONLY(no-read limb)"
                        if cfg.cortex['latent_write_only'] else "")
                     + (",CARRY-UNREAD(no-read limb)"
                        if not cfg.cortex['latent_carry_read'] else "")
                     + (f",gate_lr_mult={cfg.cortex['latent_read_gate_lr_mult']}"
                        if cfg.cortex['latent_read_gate_lr_mult'] != 1.0 else "")
                     + (f",scratch_fb={cfg.cortex['latent_scratch_forget_bias']}"
                        if cfg.cortex['latent_read'] == 'scratch' else "")
                     + (",EMBEDS(J4: Z as its own input_embeds columns)"
                        if cfg.cortex['latent_read'] == 'embeds' else "")
                     + f",enc={cfg.cortex['latent_encoding']}"
                     + (f",znorm={cfg.cortex['latent_read_znorm_target']}"
                        if cfg.cortex['latent_read_znorm'] == 'rms' else "")
                     + (f",e_dropout={cfg.cortex['e_dropout']}"
                        if cfg.cortex['e_dropout'] else "")
                     + ") "
                     if cfg.cortex['latent_carry'] else "") +
                  f"prefix_pos={cfg.cortex['prefix_pos']} "
                  f"prefix_eos_reset={cfg.cortex['prefix_eos_reset']} "
                  f"cross_chunks={cfg.cortex['cross_chunks']} "
                  f"carry_grad_chunks={cfg.cortex['carry_grad_chunks']} "
                  f"random_segments={cfg.cortex['random_segments']} "
                  f"lora_rank={cfg.cortex['lora_rank']}")
            if cfg.cortex["latent_carry"]:
                print("[cortex] Z channel ON.  latent_depth_rule="
                      f"{cfg.cortex['latent_depth_rule']!r} is P0.7's question "
                      "(evals/diag_depth_band.py): P0.1 measured the band at a "
                      "SINGLE T and cannot say whether 'informative' tracks the "
                      "absolute step or the fraction of the loop.  If P0.7 has "
                      "not reported, this run is asserting an answer it does "
                      "not have -- record which, in this run's sbatch header.")
    if cfg.init_from_scratch:
        # https://huggingface.co/smcleish/Recurrent-Llama-3.2-2-4-2-untrained/blob/main/raven_modeling_minimal_with_init.py
        if cfg.non_recurrent_model:
            pass
        else:
            config.auto_map["AutoModelForCausalLM"] = "raven_modeling_minimal_with_init.RavenForCausalLM"
            # Redirect to a different modelling file as for Llama we need to hardcode emb_scale=1.0, which we do in the regular modelling file
        model = AutoModelForCausalLM.from_config(
            config,
            trust_remote_code=True,
            attn_implementation="sdpa",
        )
        if not cfg.no_monkeypatch_on_jonas_init:
            from types import MethodType
            model.initialize_state = MethodType(initialize_state_monkeypatch, model)

        model.to(device=local_device, dtype=weight_dtype)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            cfg.model_name,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            device_map=local_device,
            torch_dtype=weight_dtype,
            attn_implementation="sdpa",
            config=config,
        )

    # cortex: fail loud if memory was requested but the graft didn't build
    # (model dir lacks the grafted modeling file, or cortex_graft failed to
    # import) — otherwise training silently runs as a no-memory baseline.
    if cfg.cortex["use_memory"] and getattr(model, "cortex", None) is None:
        raise RuntimeError(
            "cfg.cortex.use_memory is set but model.cortex is None — the grafted "
            "RavenForCausalLM did not build the memory module. Ensure model_name "
            "points at a graft-prepared dir (tools/prepare_cortex_checkpoint.py) "
            "and that cortex_graft imports from the repo root."
        )
    # cortex: the NEGATIVE twin of the guard above, and the defence for the
    # control run.  Bug class 2 has now silently voided four runs, and it cuts
    # both ways: a memory run that secretly has no memory, and a CONTROL that
    # secretly has memory, both produce a perfectly healthy loss curve.  The
    # second is the more expensive mistake, because the control's whole job is
    # to be the zero — a control with a live carry would make the memory model's
    # delta vanish and the finding would read as "memory does not work."
    #
    # `use_memory false` must leave cortex UNBUILT, not merely unused: the graft
    # returns early, so there are no summary slots, no memory parameters in the
    # optimizer, and no prefix columns in any chunk's attention.  Anything else
    # is a different model with the same flag.
    if control_has_memory(cfg.cortex["use_memory"],
                          getattr(model, "cortex", None) is not None):
        raise RuntimeError(
            "cfg.cortex.use_memory is false but model.cortex is not None — this "
            "run would train WITH memory while every log, checkpoint config and "
            "table says it is the no-memory control.  The graft built the module "
            "anyway, which means a cortex flag in the checkpoint's config.json "
            "is still selecting a mechanism (check prefix_memory, memory_slots, "
            "accum_ccot, gated_accum): the 16 persisted flags come from the "
            "checkpoint, not from the command line, and use_memory is the only "
            "one the CLI is overriding here."
        )
    if (cfg.cortex["use_memory"] and cfg.cortex["lora_rank"] > 0
            and getattr(model, "cortex_lora", None) is None):
        raise RuntimeError(
            "cfg.cortex.lora_rank is set but model.cortex_lora is None — the "
            "grafted modeling file predates the LoRA graft. Re-run "
            "tools/prepare_cortex_checkpoint.py to refresh the model dir."
        )
    # cortex: the guard above catches a MISSING module; this one catches a stale
    # forward, which is invisible until eval.  cortex_graft.py is imported live
    # from the repo root, but the modeling file is a SNAPSHOT in the model dir
    # (trust_remote_code loads it from there, not from convert_pretrained_model/).
    # A snapshot predating the prefix rewrite still builds cortex.prefix — the
    # module comes from the live graft, so `model.cortex is not None` passes —
    # but its forward() never calls prefix_pack/prefix_unpack.  The summary slots
    # are then never spliced into the stream: no gradient reaches cortex.prefix,
    # and cross_write returns None for a prefix arm (memory_slots=0 leaves no
    # bolt-on buffer to fall back on).  The run trains as a plain no-memory
    # baseline with a perfectly healthy loss curve.
    #
    # That is what happened to the rung1-pfxaccum32-* arms (2026-08-07): every
    # config flag correct, ckpts/olmo8-cortex holding a pre-rewrite snapshot, and
    # the saved checkpoints came back with summary_seeded still False and
    # summary_emb still at its N(0, 0.02) init from reset_cortex_inits below.
    # Cost was three training runs plus a night of evals, so fail at step 0.
    if cfg.cortex["use_memory"] and cfg.cortex["prefix_memory"]:
        fwd = type(model).forward
        # Signature first: it always works, and the snapshot's tell is that it
        # has no prefix_write/prefix_read knobs at all.  Source is the semantic
        # check but needs a retrievable file, so it only ever adds a failure.
        stale = "prefix_write" not in inspect.signature(fwd).parameters
        if not stale:
            try:
                stale = "prefix_pack" not in inspect.getsource(fwd)
            except (OSError, TypeError):
                pass
        if stale:
            try:
                where = inspect.getfile(fwd)
            except TypeError:
                where = f"the modeling file in {cfg.model_name}"
            raise RuntimeError(
                f"cfg.cortex.prefix_memory={cfg.cortex['prefix_memory']!r} but "
                f"{where} has no prefix_pack in forward() — this model dir's "
                "snapshot of raven_modeling_minimal_cortex.py predates the "
                "prefix rewrite, so the summary slots would never be spliced "
                "and cortex.prefix would train no weights (silently, with a "
                "healthy loss curve).  Refresh the snapshot:\n"
                "  cp convert_pretrained_model/raven_modeling_minimal_<variant>.py "
                f"{cfg.model_name}/raven_modeling_minimal_cortex.py\n"
                "then re-run tools/prepare_eval_checkpoint.py against any "
                "checkpoints already derived from it."
            )

    # cortex: undo post_init's clobbering of the graft's designed inits (must run
    # AFTER from_pretrained, BEFORE the freeze / optimizer build so the optimizer
    # sees the intended weights).  Skipped on --resume
    # (a resumed checkpoint carries the trained cortex weights, not fresh ones).
    if cfg.cortex["use_memory"] and cfg.resume_path is None:
        reset_cortex_graft_init(model)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # cortex: the config's special-token ids must agree with the tokenizer's.
    #
    # Every smcleish/Recurrent-* config inherits HUGINN's ids (bos 65504, eos
    # 65505, pad 65509) while carrying an OLMo-2 tokenizer whose real eos is
    # 100257.  They are valid ids in a 100,352 vocab, so nothing raises — but
    # 65505 decodes to " creek", and resolve_summary_init_token() falls back to
    # config.eos_token_id when summary_init_token < 0.  B2 therefore seeded its
    # prefix summary embeddings from " creek" instead of EOS, which is one of
    # the four AutoCompressor divergences B2 existed to fix.
    #
    # Only fires where the seed is actually about to be applied: a fresh prefix
    # run with no explicit token.  A resumed or eval-loaded checkpoint has
    # summary_seeded set and skips the seeding entirely, so it is unaffected.
    if (cfg.cortex["use_memory"] and cfg.cortex["prefix_memory"]
            and int(cfg.cortex["summary_init_token"]) < 0):
        cfg_eos = getattr(config, "eos_token_id", None)
        if isinstance(cfg_eos, (list, tuple)):
            cfg_eos = cfg_eos[0] if cfg_eos else None
        if cfg_eos != tokenizer.eos_token_id:
            raise RuntimeError(
                f"Refusing to seed the prefix summary embeddings from "
                f"config.eos_token_id={cfg_eos} ({tokenizer.decode([cfg_eos])!r}) "
                f"when the tokenizer's eos is {tokenizer.eos_token_id} "
                f"({tokenizer.decode([tokenizer.eos_token_id])!r}).  The "
                f"Recurrent-* configs carry Huginn's ids; pass "
                f"--cortex.summary_init_token {tokenizer.eos_token_id} "
                f"(AutoCompressor uses EOS), or re-prepare the checkpoint with "
                f"tools/prepare_cortex_checkpoint.py, which now fixes the ids."
            )

    # cortex: optionally freeze the recurrent loop (train memory + coda only).
    # Done on the unwrapped model BEFORE the DDP wrap.  There is no staged
    # unfreeze: to run rung 1 then rung 2, split into two runs (frozen, then
    # resume with freeze_loop=false).
    if cfg.cortex["use_memory"] and cfg.cortex["freeze_loop"]:
        n_frozen = set_loop_trainable(model, trainable=False)
        if is_main_process():
            print(f"[cortex] froze {n_frozen} loop (adapter+core_block) params")
    if cfg.train_read_only:
        if cfg.cortex["freeze_loop"]:
            raise ValueError(
                "train_read_only and freeze_loop are different freezes and "
                "combining them is a configuration error, not a stricter "
                "freeze: freeze_loop leaves memory and coda TRAINING, which "
                "train_read_only then overrides wholesale.  Pick one.")
        n_train, n_frozen = set_read_only_trainable(model)
        if is_main_process():
            print(f"[cortex] TIER 1.5 ORACLE PROBE: training {n_train} params "
                  f"under cortex.latent_reader, froze {n_frozen}.  Nothing "
                  f"else can absorb the signal -- and nothing else will "
                  f"improve, so this run's loss is NOT comparable to a cell's.")

    ##########  Distribute model   ##############
    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_device], find_unused_parameters=not cfg.compile, gradient_as_bucket_view=True)
    if cfg.compile:
        model = torch.compile(model, fullgraph=False, dynamic=False, mode="max-autotune-no-cudagraphs")
    ##########     Optimizer       ##############
    if cfg.use_ellis_adam["use_ellis_adam"]:
        optimizer = ELLISAdam(
            params=model.parameters(),
            **{k: v for k, v in cfg.optim_config.items() if k != "eps"},
            **{k: v for k, v in cfg.use_ellis_adam.items() if k != "use_ellis_adam"},
        )

    elif cfg.muon["use_muon"]:
        from muon import MuonWithAuxAdam

        body_params = []
        non_body_params = []
        norms = []
        cortex_params = []  # memory buffers + LoRA; only split out when memory_lr > 0
        memory_lr = float(cfg.cortex["memory_lr"]) if cfg.cortex["use_memory"] else 0.0

        if cfg.non_recurrent_model:
            if ("TinyLlama-1.1B-intermediate-step-1431k-3T" in cfg.model_name) or ("Llama-3.2-1B" in cfg.model_name) or ("OLMo-2" in cfg.model_name):
                for n, p in model.named_parameters():
                    if ("norm" in n) or ("bias" in n):
                        norms.append(p)
                    elif ("embed_tokens" in n) or ("lm_head" in n):
                        non_body_params.append(p)
                    else:
                        body_params.append((n,p))
            else:
                for n, p in model.named_parameters():
                    if ("norm" in n) or ("bias" in n):
                        norms.append(n)
                    elif ("embed_tokens" in n) or ("lm_head" in n):
                        non_body_params.append(n)
                    else:
                        body_params.append(n)
                if is_main_process():
                    print(model)
                    print("="*70)
                    print(norms)
                    print("="*70)
                    print(non_body_params)
                    print("="*70)
                    print(body_params)
                assert False, "Model not allowed for muon"
        else:
            # if a huginn
            non_recur_body_params = []  # split out only when cfg.throttle
            for n, p in model.named_parameters():
                if "cortex" in n:
                    # cortex memory + LoRA params — keep on the Adam side: they
                    # are zero/identity-init and several are tagged
                    # _no_weight_decay; Newton-Schulz orthogonalisation is
                    # inappropriate for them.  With memory_lr > 0 they get their
                    # own adam group (higher LR, no WD); else they ride the aux
                    # group as before.
                    cortex_params.append(p)
                elif ("norm" in n) or ("ln_f" in n) or ("Wqkv.bias" in n):
                    norms.append(p)
                elif ("wte" in n) or ("lm_head" in n):
                    non_body_params.append(p)
                elif cfg.throttle and not (("adapter" in n) or ("core_block" in n)):
                    # throttle scales param_groups[0] by 1/mean-k, which must hit
                    # ONLY the recurrent loop — keep prelude/coda body params in
                    # their own muon group so they get the full LR.
                    non_recur_body_params.append((n, p))
                else:
                    body_params.append((n,p))

        # body_params = sorted(body_params, key=lambda x: x.size(), reverse=True)
        # Took sorting out of the init so that it is deterministic
        body_params.sort(key=lambda np: (-np[1].numel(), tuple(np[1].shape), np[0]))
        body_params = [p for _, p in body_params]
        param_groups = [
            dict(params=body_params, use_muon=True, lr=cfg.muon["lr"], weight_decay=cfg.muon["weight_decay"], no_sorting_in_init=False),
        ]
        if cfg.throttle and not cfg.non_recurrent_model and non_recur_body_params:
            non_recur_body_params.sort(key=lambda np: (-np[1].numel(), tuple(np[1].shape), np[0]))
            param_groups.append(
                dict(params=[p for _, p in non_recur_body_params], use_muon=True, lr=cfg.muon["lr"], weight_decay=cfg.muon["weight_decay"], no_sorting_in_init=False)
            )
        if not (memory_lr > 0):
            non_body_params = non_body_params + cortex_params
        # eps is passed EXPLICITLY.  Until 2026-09-14 it was omitted, and
        # MuonWithAuxAdam.__init__ then filled in its own default of 1e-10 — so
        # `optim_config.eps` was dead config and every cortex and McLeish run
        # actually trained at 1e-10, against OLMo-2's pretraining value of 1e-8.
        # Third instance of the recipe audit's rule: find where a value is
        # CONSUMED, not where it is set.  (`eps` is inside MuonWithAuxAdam's
        # allowed-key assert, so passing it is safe.)
        param_groups.append(
            dict(params=non_body_params + norms, use_muon=False, lr=cfg.optim_config["lr"], betas=cfg.optim_config["betas"], eps=cfg.optim_config["eps"], weight_decay=cfg.optim_config["weight_decay"]),
        )
        if memory_lr > 0 and cortex_params:
            # dedicated group for the fresh cortex params: higher LR, no weight
            # decay (decay would pull the identity-init projections toward zero).
            param_groups.append(
                dict(params=cortex_params, use_muon=False, lr=memory_lr, betas=cfg.optim_config["betas"], eps=cfg.optim_config["eps"], weight_decay=0.0),
            )
        optimizer = MuonWithAuxAdam(param_groups)

        ## Need to save all states on all ranks, see: https://github.com/KellerJordan/Muon/issues/46
        def gather(self):
            if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
                return
            world = torch.distributed.get_world_size()

            for group in self.param_groups:
                if not group["use_muon"]:
                    continue

                params = group["params"]
                # Make sure every rank has a buffer tensor to receive the broadcast.
                for p in params:
                    st = self.state[p]
                    if "momentum_buffer" not in st:
                        st["momentum_buffer"] = torch.zeros_like(p)

                # For param i, the canonical copy lives on rank (i % world).
                for i, p in enumerate(params):
                    src = i % world
                    torch.distributed.broadcast(self.state[p]["momentum_buffer"], src=src)

        optimizer.register_state_dict_pre_hook(gather)
    else:
        # print(model.named_parameters())
        optim_config = cfg.optim_config.copy()
        memory_lr = float(cfg.cortex["memory_lr"]) if cfg.cortex["use_memory"] else 0.0

        def _is_cortex(n):
            # cortex memory + LoRA params — split into their own group (higher
            # LR, no weight decay) only when memory_lr > 0
            return memory_lr > 0 and ("cortex" in n)

        cortex_params = [p for n, p in model.named_parameters() if _is_cortex(n)]
        if cfg.throttle:
            recur_params = []
            non_recur_params = []
            for n, p in model.named_parameters():
                if _is_cortex(n):
                    continue
                if ("adapter" in n) or ("core_block" in n):
                    recur_params.append(p)
                else:
                    non_recur_params.append(p)
            params = [
                {"params": recur_params,  "lr": cfg.optim_config["lr"]},
                {"params": non_recur_params, "lr": cfg.optim_config["lr"]},
            ]
            optim_config.pop("lr")
        elif cortex_params:
            rest = [p for n, p in model.named_parameters() if not _is_cortex(n)]
            params = [{"params": rest, "lr": cfg.optim_config["lr"]}]
            optim_config.pop("lr")
        else:
            params = model.parameters()
        if cortex_params:
            params.append({"params": cortex_params, "lr": memory_lr, "weight_decay": 0.0})
        optimizer = torch.optim.AdamW(params, **optim_config)

    ##########     Data            ##############
    def format_and_tokenize_examples(examples):
        conversations = []
        for idx in range(len(examples[cfg.dataset_args["q_col"]])):
            if cfg.dataset_args["q_col"] != "text":
                messages = [
                    Message(role="system", content=DEFAULT_SYS_PROMPT),
                    Message(role="user", content=examples[cfg.dataset_args["q_col"]][idx].strip()),
                    Message(role="Huginn", content=examples[cfg.dataset_args["a_col"]][idx].strip()),
                ]
            else:
                messages = tokenizer.bos_token + examples[cfg.dataset_args["q_col"]][idx].strip()
            conversations.append(messages)
        
        if cfg.dataset_args["q_col"] != "text":
            chat_encoding = tokenizer.apply_chat_template(
                conversations,
                tokenize=True,
                add_generation_prompt=False,
                return_assistant_tokens_mask=True,
                padding="max_length",
                max_length=cfg.max_length + 1,
                return_tensors="pt",
                return_dict=True,
                truncation=True,
            )
            if cfg.take_loss_over_all_tokens:
                chat_encoding["assistant_masks"] = chat_encoding["attention_mask"]
        else:
            chat_encoding = tokenizer(
                conversations,
                padding="max_length",
                max_length=cfg.max_length + 1,
                return_tensors="pt",
                truncation=True,
                
            )
            chat_encoding["assistant_masks"] = chat_encoding["attention_mask"].clone()

        return {
            "token_ids": chat_encoding["input_ids"],
            "mask": chat_encoding["assistant_masks"],
            "attention_mask": chat_encoding["attention_mask"],
        }

    if cfg.preprocessed_data_path is None:
        cfg.token_id_col_name = "token_ids"
        dataset_save_dir = f"{cfg.out_path}/{cfg.run_name}/dataset"
        if is_main_process(): # only load to rank 0 to begin
            try:
                dataset: Dataset = load_dataset(cfg.dataset_location, cfg.dataset_config)["train"]  # type: ignore
            except:
                dataset: Dataset = load_from_disk(cfg.dataset_location, cfg.dataset_config)  # type: ignore

            if cfg.max_samples is not None:
                dataset = dataset.select(range(cfg.max_samples))

            if os.path.exists(dataset_save_dir): # delete any old dataset
                shutil.rmtree(dataset_save_dir)

            tokenized_dataset = dataset.map(
                format_and_tokenize_examples,
                num_proc=16,
                remove_columns=dataset.column_names,
                batched=True,
                batch_size=1024,
            )

        if distributed: # load the dataset to other ranks
            if is_main_process():
                tokenized_dataset.save_to_disk(dataset_save_dir)
            torch.distributed.barrier()
            tokenized_dataset = load_from_disk(dataset_save_dir)
            torch.distributed.barrier()
    else:
        cfg.token_id_col_name = "input_ids"
        if cfg.is_parquet_dataset:
            assert cfg.max_samples is None, "cannot have max samples for parquet dataset"
            tokenized_dataset = get_parquet_dataloader(world_size, rank, cfg.micro_batch_size, cfg.preprocessed_data_path, num_epochs=cfg.parquet_epoching_flag_use_with_real_caution)
        else:
            tokenized_dataset = load_from_disk(cfg.preprocessed_data_path)
            if cfg.max_samples is not None:
                dataset = dataset.select(range(cfg.max_samples))

    if not cfg.is_parquet_dataset:
        tokenized_dataset.set_format("pt")

    dataloader_generator = torch.Generator()
    dataloader_generator.manual_seed(cfg.seed)
    # epoch_order: the explicit shuffle permutation for the single-process
    # map-style path.  None on every other path (parquet / DistributedSampler),
    # which is what fast_forward_dataloader() keys off.
    epoch_order = None
    if cfg.is_parquet_dataset:
        dataloader = tokenized_dataset
    elif distributed:
        sampler = torch.utils.data.DistributedSampler(
            tokenized_dataset,
            shuffle=not cfg.is_parquet_dataset,
            num_replicas=world_size,
            rank=rank,
            seed=cfg.seed,
        )
        dataloader = torch.utils.data.DataLoader(
            tokenized_dataset,
            batch_size=cfg.micro_batch_size,
            sampler=sampler,
            pin_memory=True,
            generator=dataloader_generator,
        )
    else:
        # An EXPLICIT permutation instead of shuffle=True.  Two reasons, both
        # from the 2026-09-14 recipe audit:
        #
        #  * it is O(1)-sliceable, which is what lets a resume fast-forward in
        #    constant time instead of re-walking every consumed row.  The old
        #    path iterated and discarded at ~68 rows/s: 3h03m on B2's last arm
        #    link and 11.9h across the arm's six links, 4.1% of its GPU-hours.
        #  * shuffle=True routes through RandomSampler, whose permutation cannot
        #    be reproduced outside the DataLoader iterator without depending on
        #    torch internals (_BaseDataLoaderIter draws its base_seed from the
        #    same generator BEFORE the sampler runs).  Owning the permutation
        #    removes that dependency.
        #
        # The order is still a pure function of (cfg.seed, len(dataset)), so it
        # is identical on every link of a resume chain — which is what makes the
        # slice correct.  It is NOT the order a pre-2026-09-14 run would have
        # drawn, so do not resume an older run across this change.
        epoch_order = torch.randperm(
            len(tokenized_dataset), generator=dataloader_generator
        ).tolist()
        dataloader = torch.utils.data.DataLoader(
            tokenized_dataset,  # type: ignore
            batch_size=cfg.micro_batch_size,
            sampler=epoch_order,
            pin_memory=True,
        )

    ##########     Scheduler       ##############
    if cfg.is_parquet_dataset:
        if cfg.max_steps:
            max_training_steps = cfg.max_steps
        else:
            max_training_steps = max(1, math.ceil(cfg.parquet_dataset_max_tokens / world_size / cfg.max_length))
        num_warmup_steps = _resolve_warmup_steps(cfg, max_training_steps)
        num_decay_steps = math.ceil(cfg.scheduler_args["cooldown"] * max_training_steps)
    else:
        if cfg.max_steps:
            max_training_steps = cfg.max_steps
        else:
            accumulation_steps = max(1, cfg.batch_size // cfg.micro_batch_size)
            num_update_steps_per_epoch = math.ceil(len(dataloader) / accumulation_steps)
            max_training_steps = cfg.epochs * num_update_steps_per_epoch
        num_warmup_steps = _resolve_warmup_steps(cfg, max_training_steps)
        num_decay_steps = math.ceil(cfg.scheduler_args["cooldown"] * max_training_steps)

    scheduler = get_scheduler(
        name="warmup_stable_decay",
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=max_training_steps,
        scheduler_specific_kwargs={"num_decay_steps":num_decay_steps, "min_lr_ratio": cfg.scheduler_args["min_lr_ratio"]},
    )

    state = {
        "model": model,
        "optimizer": optimizer,
        "tokenizer": tokenizer,
        "dataloader": dataloader,
        "distributed": distributed,
        "scheduler": scheduler,
        # Kept so train() can rebuild a fast-forwarded loader on a resume; both
        # are None on the parquet and DistributedSampler paths.
        "dataset": None if cfg.is_parquet_dataset else tokenized_dataset,
        "dataloader_order": epoch_order,
    }

    if cfg.mean_recurrence_schedule["turn_on"]:
        # make a dummy optimizer of one param 
        num_warmup_steps = math.ceil(cfg.mean_recurrence_schedule["warmup"] * max_training_steps)
        mean_recurrence_scheduler = get_scheduler(
            name="warmup_stable_decay",
            optimizer=torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=float(cfg.mean_recurrence_schedule["max_mean_rec"])),
            num_warmup_steps=num_warmup_steps,
            num_training_steps=max_training_steps,
            scheduler_specific_kwargs={"num_decay_steps":0, "min_lr_ratio":0, "warmup_type": cfg.mean_recurrence_schedule["warmup_type"]},
        )
        state["mean_recurrence_scheduler"] = mean_recurrence_scheduler
    
    if cfg.mean_backprop_depth_schedule["turn_on"]:
        # make a dummy optimizer of one param 
        num_warmup_steps = math.ceil(cfg.mean_backprop_depth_schedule["warmup"] * max_training_steps)

        max_depth = cfg.mean_backprop_depth_schedule["max_backprop"]
        start = max(1.0, cfg.mean_backprop_depth_schedule["start"] - 1) # start at one below so we get the right value out of the scheduler after the first step
        min_lr_ratio = max(0.0, min(1.0, start / max_depth))

        mean_backprop_depth_scheduler = get_scheduler(
            name="warmup_stable_decay",
            optimizer=torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=float(max_depth)),
            num_warmup_steps=num_warmup_steps,
            num_training_steps=max_training_steps,
            scheduler_specific_kwargs={"num_decay_steps":0, "min_lr_ratio":min_lr_ratio},
        )
        state["mean_backprop_depth_scheduler"] = mean_backprop_depth_scheduler
        state["mean_backprop_depth_scheduler"].step() # take the first step so we get 2 out of the scheduler and not 1

    cfg.world_size = world_size
    if is_main_process():
        wandb.init(
            project=cfg.out_path,
            name=cfg.run_name,
            config=cfg,
            dir=cfg.out_path,
            mode="disabled" if cfg.wandb_disabled else "online",
        )

    return state, local_device


def distributed_and_agg_metrics(metrics_to_agg_data_step, metrics_to_agg_optim_step):
    keys_to_mean = ["loss", "log_ppl"]

    distributed = torch.distributed.is_initialized()
    rank = int(os.getenv("SLURM_PROCID", os.getenv("RANK", "0")))
    local_device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")

    def _sync(value: float, op=torch.distributed.ReduceOp.SUM) -> float:
        """Synchronise a scalar across ranks and return the reduced result."""
        if distributed:
            tensor = torch.tensor(value, dtype=torch.float64, device=local_device)
            torch.distributed.all_reduce(tensor, op=op)
            return tensor.item()
        return value
    

    aggregated = {}
    # metrics_to_agg_data_step
    for key, local_list in metrics_to_agg_data_step.items():
        if not local_list:
            continue

        local_sum = float(sum(local_list))
        local_count = float(len(local_list))

        global_sum = _sync(local_sum)
        global_count = _sync(local_count)

        aggregated[key] = global_sum / (max(global_count, 1.0) if key in keys_to_mean else 1.0)

        local_list.clear()

    # metrics_to_agg_optim_step
    for key, val in metrics_to_agg_optim_step.items():
        if key in keys_to_mean:
            # we don't pass this anymore as it is global anyway but is example of how to use avg
            aggregated[key] = _sync(val, op=torch.distributed.ReduceOp.AVG)
        else:
            aggregated[key] = _sync(val)

    return aggregated

def get_steps_compiling(data_step, device):
    if data_step > 600:
        exit()
    n = data_step % 300
    k =  min(8, n)
    print(f"Warming up sampling step={data_step}, n={n}, k={k}")
    return  torch.tensor([n,k], device=device)

def num_steps_sampler(data_step, mean_recurrence, mean_backprop_depth, cfg):
    """
    Sampling num steps in a checkpointable way
    https://github.com/seal-rg/recurrent-pretraining/blob/main/recpre/model_dynamic.py#L1250
    """
    t = max(mean_recurrence - mean_backprop_depth, 0)
    s = mean_backprop_depth
    
    seed_n = 514229 + data_step 
    seed_k = 317811 + data_step   

    n_generator = torch.Generator(device="cpu")
    n_generator.manual_seed(seed_n % (2**31 - 1))
    k_generator = torch.Generator(device="cpu")
    k_generator.manual_seed(seed_k % (2**31 - 1))

    sigma = 0.5
    mu = math.log(t + s) - (sigma**2 / 2)
    rate = torch.zeros((1,)).log_normal_(mean=mu, std=sigma, generator=n_generator)
    p = torch.poisson(torch.tensor([rate], dtype=torch.float), generator=n_generator) + 1
    n = torch.clamp(p - s, min=0)
    k = torch.as_tensor(torch.minimum(torch.as_tensor(s), p))

    return n.to(dtype=torch.long), k.to(dtype=torch.long)

def sheduler_n_k_handler(state, cfg, model_config):
    if cfg.mean_recurrence_schedule["turn_on"]:
        new_mean_rec = math.ceil(state["mean_recurrence_scheduler"].get_last_lr()[0])
    else:
        new_mean_rec = model_config.mean_recurrence

    if cfg.mean_backprop_depth_schedule["turn_on"]:
        mean_backprop_depth = math.ceil(state["mean_backprop_depth_scheduler"].get_last_lr()[0])
    else:
        mean_backprop_depth = model_config.mean_backprop_depth

    if new_mean_rec <= 0:
        # schedule starts at 0
        new_mean_rec = 1

    if (new_mean_rec - mean_backprop_depth) < 0:
        # t = max(mean_recurrence - mean_backprop_depth, 0) messes up the schedule so we catch that here
        return partial(num_steps_sampler, mean_recurrence=new_mean_rec, mean_backprop_depth=new_mean_rec, cfg=cfg), new_mean_rec, new_mean_rec
    else:
        return partial(num_steps_sampler, mean_recurrence=new_mean_rec, mean_backprop_depth=mean_backprop_depth, cfg=cfg), new_mean_rec, mean_backprop_depth

def fast_forward_dataloader(state, cfg, data_start_step):
    """Position a map-style loader after `data_start_step` micro-batches, in O(1).

    Returns a new DataLoader over the unconsumed tail of this run's permutation,
    or None when the fast path does not apply (parquet loader, DistributedSampler,
    or nothing to skip) — the caller then falls back to iterate-and-discard.

    `data_start_step` counts dataloader ITEMS, i.e. micro-batches, not rows; the
    row cursor is `data_start_step * micro_batch_size`.  With micro_batch_size=1
    (every cortex run to date) the two coincide, which is why B2's logs read
    `Step: 431736` for 431,732 skipped rows.

    Correctness rests on the permutation being a pure function of
    (cfg.seed, len(dataset)) and therefore identical on every link of the chain —
    see the epoch_order comment in startup().  A dataset whose length changed
    (a corpus switch) gets a DIFFERENT permutation, which is exactly why such a
    link must pass --reset_dataset_position rather than slice into it.
    """
    order = state.get("dataloader_order")
    if order is None:
        return None
    consumed = data_start_step * cfg.micro_batch_size
    try:
        tail = fast_forward_indices(order, data_start_step, cfg.micro_batch_size)
    except ValueError as exc:
        raise RuntimeError(
            f"{exc} ({cfg.preprocessed_data_path}).  If this link switches "
            f"corpus, pass --reset_dataset_position true; otherwise the pack is "
            f"too small — tools/check_pack.py --resume_rows {consumed}."
        ) from None
    if tail is None:
        return None
    if is_main_process():
        print(f"[data] fast-forward: skipping {consumed:,} consumed rows, "
              f"{len(tail):,} remain "
              f"({len(tail) // max(1, cfg.batch_size):,} optimizer steps)")
    return torch.utils.data.DataLoader(
        state["dataset"],
        batch_size=cfg.micro_batch_size,
        sampler=tail,
        pin_memory=True,
    )


def train(state, device, cfg, data_start_step=1, optimizer_step=0, total_tokens_from_restart=0, total_tokens_with_loss_from_restart=0, elapsed_time_from_restart=0.0):
    model, optimizer = state["model"], state["optimizer"]
    model.train()

    accumulation_steps = cfg.batch_size // cfg.micro_batch_size
    optimizer_step = optimizer_step
    step_time = time.monotonic()
    total_tokens = 0
    total_tokens_with_loss = 0
    tokens_in_step = 0
    k_mean_tracker = [0,0]
    consecutive_nonfinite = 0   # run-abort guard: see max_nonfinite_skips
    elapsed_time = 0.0
    # The last chunk chain's carry, kept ONLY when --cortex.diag_interval is on.
    # Detached, so no graph is retained; [K, 2D] at most, so the memory is
    # nothing.  It is held because the diagnostic runs at the wandb log site,
    # which is outside the closure that produced it.
    cortex_diag_state = {"carry": None}

    output_details = {
        "return_logits": False,
        "return_latents": False,
        "return_head": False,
        # get_stats() runs softmax + log over the full [B, T, vocab] logits every
        # forward — an ~0.8-1.2 GB transient on an OLMo-size vocab, on top of the
        # fp32 logits already held for the loss.  That transient is what tipped
        # the loop-touching rungs (1b/2/3) over the 80 GB ceiling (they OOM'd
        # 20-392 MB short), and its `prob_entropy = ... probs.log()` amplifies
        # nan.  It is diagnostic-only (nothing in the wandb log reads it — the
        # num_steps counters below are taken straight from the sampler), so keep
        # it OFF during training.  Flip to True only for one-off inspection.
        "return_stats": False,
    }

    metrics_to_agg_data_step = {
        "loss": [],
        "log_ppl": [],
    }
    world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1

    model_config = get_unwrapped_model(model).config
    # Apply recurrence-depth overrides in one place so the sampler, scheduler and
    # logging all see them.  Mutating model_config only affects the sampled
    # num_steps (the forward reads num_steps, not these fields directly).
    if cfg.override_mean_backprop_depth > 0:
        if is_main_process():
            print(f"[cortex] override mean_backprop_depth {model_config.mean_backprop_depth}"
                  f" -> {cfg.override_mean_backprop_depth} (shorter TBPTT window)")
        model_config.mean_backprop_depth = cfg.override_mean_backprop_depth
    if cfg.override_mean_recurrence > 0:
        if is_main_process():
            print(f"[cortex] override mean_recurrence {model_config.mean_recurrence}"
                  f" -> {cfg.override_mean_recurrence}")
        model_config.mean_recurrence = cfg.override_mean_recurrence
    if cfg.mean_recurrence_schedule["turn_on"] or cfg.mean_backprop_depth_schedule["turn_on"]:
        num_steps_sampler_partial, new_mean_rec, new_backprop_depth = sheduler_n_k_handler(state, cfg, model_config)
    elif cfg.non_recurrent_model:
        new_mean_rec, new_backprop_depth = model_config.num_hidden_layers, model_config.num_hidden_layers
    else:
        new_mean_rec = model_config.mean_recurrence
        new_backprop_depth = model_config.mean_backprop_depth
        num_steps_sampler_partial = partial(num_steps_sampler, mean_recurrence=new_mean_rec, mean_backprop_depth=new_backprop_depth, cfg=cfg)

    # P3.0 tier 1.5's forced split.  Parsed ONCE, here, so a malformed value
    # fails before the data loader and the wandb run exist rather than on the
    # first forward of a queued job -- the same reason summary_init_token is
    # resolved at build time.
    _FORCED_NUM_STEPS = None
    if cfg.force_num_steps:
        try:
            _n, _k = (int(v) for v in str(cfg.force_num_steps).split(","))
        except Exception:
            raise ValueError(
                f"force_num_steps must be 'n,k' (e.g. '0,8'); got "
                f"{cfg.force_num_steps!r}")
        if _n < 0 or _k < 1:
            raise ValueError(
                f"force_num_steps needs n >= 0 and k >= 1; got {_n},{_k}.  "
                "k < 1 would put the whole loop outside the gradient.")
        _FORCED_NUM_STEPS = torch.tensor([_n, _k], dtype=torch.long)
        if is_main_process():
            print(f"[cortex] force_num_steps: every batch runs [{_n}, {_k}].  "
                  f"This OVERRIDES the sampler, so mean_recurrence and "
                  f"mean_backprop_depth no longer describe this run.")
            if _n == 0:
                print("[cortex]   n=0: the Z read gradient is live on EVERY "
                      "batch (latent_read_grad_frac 1.0 by construction, not "
                      "by the sampler's ~0.55 at mr8).")

    # window_backward: refuse every geometry where it would silently do nothing
    # or do the wrong thing.  A memory flag that no-ops prices one run and
    # trains another -- the OOM comes back at step N of a queued job.
    if cfg.cortex["window_backward"]:
        _gc = int(cfg.cortex["carry_grad_chunks"])
        _why = None
        if not cfg.cortex["use_memory"]:
            _why = "use_memory is false, so there is no carry to window"
        elif bool(cfg.cortex["accum_ccot"]) or cfg.cortex["prefix_memory"] == "accum":
            _why = ("the accum buffer slice-detaches row by row, so its "
                    "gradient windows OVERLAP and there is no point at which a "
                    "window's graph is unreachable")
        elif _gc <= 0:
            _why = ("carry_grad_chunks is 0 (full-chain BPTT): there is no "
                    "detach, so the whole row is one window")
        elif state["distributed"]:
            _why = ("DDP all-reduces on the first backward of a sync step; a "
                    "second backward in the same step would miss it")
        if _why is not None:
            raise ValueError(f"--cortex.window_backward true cannot apply here: {_why}.")
        if is_main_process():
            _cc = int(cfg.cortex["cross_chunks"])
            print(f"[cortex] window_backward: one backward per {_gc}-chunk carry "
                  f"window ({-(-_cc // _gc)} per row), so the peak holds "
                  f"{min(_gc, _cc)} of {_cc} chunks' activations.  Same gradient "
                  f"as one backward per row, to fp summation order.")

    # The resume cursor is only meaningful within one pass over the data:
    # data_step restarts at 1 on every epoch, so "resume at item N" does not name
    # a position once there is more than one epoch.  Before 2026-09-14 this
    # silently re-applied the skip at the start of every later epoch.  Nothing
    # has ever combined the two (only the closed Track-A -ep3/-ep4 arms set
    # epochs > 1, and none of them resumed), so refuse rather than guess.
    if cfg.epochs > 1 and data_start_step > 1:
        raise RuntimeError(
            f"Cannot resume (data_start_step={data_start_step:,}) with "
            f"epochs={cfg.epochs}: the dataloader cursor is per-epoch and the "
            f"restored value does not identify a position.  Use epochs=1 and a "
            f"pack sized for the full budget, which is what every retrofit run "
            f"does."
        )

    # Resume positioning.  Prefer the O(1) slice; fall back to iterate-and-discard
    # only where the permutation is not ours (DistributedSampler).  See
    # fast_forward_dataloader().
    epoch_loader = state["dataloader"]
    enumerate_start = (data_start_step + 1) if cfg.is_parquet_dataset else 1
    slow_skip = (data_start_step != 1) and (not cfg.is_parquet_dataset)
    if slow_skip:
        fast = fast_forward_dataloader(state, cfg, data_start_step)
        if fast is not None:
            epoch_loader, enumerate_start, slow_skip = fast, data_start_step + 1, False

    reached_target = False
    data_step = 0           # defined up front so the exhaustion guard can read it
    for epoch in range(cfg.epochs):
        if epoch > 0:
            # data_step restarts at 1 inside this loop, so a resume's skip would
            # re-fire on every later epoch.  Only epoch 0 is a continuation.
            epoch_loader, enumerate_start, slow_skip = state["dataloader"], 1, False
        for data_step, inputs in enumerate(epoch_loader, start=enumerate_start):
            if slow_skip and (data_step <= data_start_step):
                # not first_run and not parquet_run and is less than the restart
                continue

            # Realize the input and labels tensors.
            input_ids = inputs[cfg.token_id_col_name][:, :-1].to(dtype=torch.long, device=device, non_blocking=True)
            # Need to take into account the assistant and attention if sequences are being padded
            if cfg.preprocessed_data_path is None:
                mask = ~(inputs["mask"].bool() & inputs["attention_mask"].bool())
            else:
                mask = ~inputs["attention_mask"].bool()

            labels = torch.where(mask[:, 1:], -100, inputs[cfg.token_id_col_name][:, 1:]).to(
                dtype=torch.long, device=device, non_blocking=True
            )
            total_tokens_with_loss += (labels != -100).sum().item()

            tokens_in_step += input_ids.numel()
            is_accumulating = (data_step % accumulation_steps != 0)
 
            if cfg.force_num_steps:
                # P3.0 tier 1.5: a FIXED split, zero no-grad prefix, so every
                # iteration carries read gradient.  A deliberate departure from
                # the training distribution -- this is an existence test, not a
                # cell -- and it is why the tier is not a Tier 2 result.
                #
                # Separate from `fix_num_steps`, which is hardcoded to [0,1]
                # and exists for compile warmup.  Reusing that flag would have
                # meant running the oracle probe at ONE recurrent step.
                num_steps = _FORCED_NUM_STEPS.to(model.device)
            elif cfg.fix_num_steps:
                num_steps = torch.tensor([0,1], device=model.device)
            elif cfg.compile_warmup_routine:
                num_steps = get_steps_compiling(data_step, model.device)
            elif not cfg.non_recurrent_model:
                num_steps = num_steps_sampler_partial(data_step)

            if cfg.throttle:
                k_mean_tracker[0] += num_steps[1]
                k_mean_tracker[1] += 1

            # The actual compute step of  Forward, loss, and backward computation:
            def tightly_scoped_fwd_bwd(model, input_ids, labels):
                with model.no_sync() if is_accumulating and state["distributed"] else nullcontext():
                    with torch.autocast(**cfg.amp_args):
                        outputs = model(input_ids, labels=labels, num_steps=num_steps, output_details=output_details)

                    (outputs["loss"] / accumulation_steps).backward()
                    # num_steps = [n_no_grad, n_with_grad] — the same tensor the
                    # model unpacks in iterate_forward — so read the counts from
                    # it directly rather than from output_details stats (now off).
                    return outputs["loss"].detach(), outputs["log_ppl"].detach(), int(num_steps[0]), int(num_steps[1])
            
            def non_rec_fwd_bwd(model, input_ids, labels):
                with model.no_sync() if is_accumulating and state["distributed"] else nullcontext():
                    with torch.autocast(**cfg.amp_args):
                        logits = model(input_ids).logits

                    loss = torch.nn.functional.cross_entropy(
                        logits.view(-1, logits.shape[-1]), labels.view(-1), ignore_index=-100
                    ) # copied from Huginn code to be sure

                    (loss / accumulation_steps).backward()
                    log_ppl = loss.clone().detach().exp()
                    return loss.detach(), log_ppl, model_config.num_hidden_layers, model_config.num_hidden_layers

            def cortex_fwd_bwd(model, input_ids, labels):
                # Cross-chunk segment chain: split the sequence into N consecutive
                # sub-windows and carry M_cross UN-detached between them, building one
                # graph over all chunks so chunk g+1's read back-props into chunk g's
                # write (the only way the M_cross write path gets gradient).  One
                # backward at the end.  First-pass data is one-doc-per-sequence so
                # eos_mask is None (full carry).
                n_chunks = int(cfg.cortex["cross_chunks"])
                eos_id = state["tokenizer"].eos_token_id if cfg.cortex["eos_from_tokens"] else None
                grad_chunks = int(cfg.cortex["carry_grad_chunks"])
                # Write-once state => the stop-gradient horizon can slice-detach
                # rows older than grad_chunks chunks instead of detaching the
                # whole carry.  True for the prefix ACCUM buffer (rows appended,
                # never overwritten) and for the retired accum_ccot; false for
                # the gated buffers, whose merge mixes rows.  NOTE: keying this
                # off accum_ccot alone silently downgraded prefix-accum to
                # whole-carry TBPTT — the flag it read was retired 2026-08-02.
                accum_on    = (bool(cfg.cortex["accum_ccot"])
                               or cfg.cortex["prefix_memory"] == "accum")
                # Rows appended per chunk: prefix buffers use accum_vecs too.
                vecs_per_chunk = int(cfg.cortex["accum_vecs"])
                # window_backward (J1): backprop each carry window as it closes.
                # Only on the whole-carry detach below -- that detach is what
                # makes the windows independent.  Validated at startup, so a
                # geometry where it cannot apply never gets this far.
                window_bwd = (bool(cfg.cortex["window_backward"])
                              and grad_chunks > 0 and not accum_on)
                with model.no_sync() if is_accumulating and state["distributed"] else nullcontext():
                    # .contiguous(): torch.chunk/split return non-contiguous views
                    # and the model's loss does labels.view(-1), which requires
                    # contiguity.
                    if cfg.cortex["random_segments"]:
                        # AutoCompressor-style randomized segmenting: jitter the
                        # boundaries ±25% each micro-batch (global RNG; ranks
                        # may draw different sizes — harmless, every param
                        # participates in every micro-step either way).
                        sizes = random_chunk_sizes(input_ids.shape[1], n_chunks)
                        x_chunks = [c.contiguous() for c in torch.split(input_ids, sizes, dim=1)]
                        y_chunks = [c.contiguous() for c in torch.split(labels, sizes, dim=1)]
                    else:
                        x_chunks = [c.contiguous() for c in torch.chunk(input_ids, n_chunks, dim=1)]
                        y_chunks = [c.contiguous() for c in torch.chunk(labels, n_chunks, dim=1)]
                    m_cross = None
                    chunk_losses = []
                    chunk_tokens = []       # unmasked labels per kept chunk
                    n_ng = n_wg = 0
                    if window_bwd:
                        # The row loss's weights, fixed before any chunk runs,
                        # so each window's share is known when it closes.
                        chunk_w = chunk_loss_weights(
                            [int((yc != -100).sum()) for yc in y_chunks],
                            mode=cfg.cortex["chunk_loss_reduction"])
                        window = []         # w_i * loss_i, graph attached
                        total = None        # detached sum of flushed windows
                    for gi, (xc, yc) in enumerate(zip(x_chunks, y_chunks)):
                        # Stop-gradient horizon (AutoCompressor: predicting the
                        # adjacent segment suffices to learn compression).
                        if grad_chunks > 0 and m_cross is not None:
                            if accum_on:
                                # write-once rows: exact slice detach of vectors
                                # older than grad_chunks chunks
                                m_cross = detach_old_vecs(
                                    m_cross, vecs_per_chunk, grad_chunks)
                            elif gi % grad_chunks == 0:
                                if window_bwd and window:
                                    # Nothing from chunk gi on can reach this
                                    # window's graph once the carry detaches on
                                    # the next line, so backprop it now and let
                                    # its activations go before the next window
                                    # is built.
                                    part = torch.stack(window).sum()
                                    (part / accumulation_steps).backward()
                                    total = part.detach() if total is None else total + part.detach()
                                    window = []
                                # gated/overwritten state (rows not separable):
                                # detach the whole carry every grad_chunks
                                # chunks = truncated BPTT with window N
                                m_cross = m_cross.detach()
                        with torch.autocast(**cfg.amp_args):
                            out = model(xc, labels=yc, num_steps=num_steps,
                                        m_cross_in=m_cross, return_m_cross=True,
                                        eos_mask=(xc == eos_id) if eos_id is not None else None,
                                        output_details=output_details)
                        # .get(): when no cross-state is active (e.g. K=0 and
                        # ccot_direct=False) the model omits the m_cross field,
                        # so bracket-indexing would KeyError — carry None instead.
                        m_cross = out.get("m_cross")             # carried, un-detached
                        # counts from the sampler tensor (stats dict now off — see
                        # output_details); matches what iterate_forward unpacked.
                        n_ng, n_wg = int(num_steps[0]), int(num_steps[1])
                        n_valid = int((yc != -100).sum())
                        if n_valid:                             # skip fully-masked chunks
                            if window_bwd:
                                window.append(out["loss"] * chunk_w[gi])
                            else:
                                chunk_losses.append(out["loss"])
                                chunk_tokens.append(n_valid)
                    if window_bwd:
                        if cfg.cortex["diag_interval"] and m_cross is not None:
                            cortex_diag_state["carry"] = m_cross.detach()
                        if window:
                            part = torch.stack(window).sum()
                            (part / accumulation_steps).backward()
                            total = part.detach() if total is None else total + part.detach()
                        if total is None:
                            # every chunk fully masked -- same guard as below
                            z = torch.zeros((), device=input_ids.device)
                            return z, z, n_ng, n_wg
                        return total, total.exp(), n_ng, n_wg
                    if not chunk_losses:
                        # Every chunk fully label-masked (-100): unreachable with
                        # one-doc-per-sequence data, guarded so torch.stack([])
                        # cannot crash.  Skip the backward (zero contribution).
                        # NOTE: under DDP a skipped backward on the sync micro-step
                        # would desync the all-reduce — only safe because the
                        # documented first-pass data never produces an all-masked
                        # micro-batch.
                        z = torch.zeros((), device=input_ids.device)
                        return z, z, n_ng, n_wg
                    if cfg.cortex["diag_interval"] and m_cross is not None:
                        # Detach: the diagnostic reads geometry, never gradient,
                        # and holding the graph here would keep the whole chain
                        # alive past the backward.
                        cortex_diag_state["carry"] = m_cross.detach()
                    total = reduce_chunk_losses(
                        chunk_losses, chunk_tokens,
                        mode=cfg.cortex["chunk_loss_reduction"])
                    (total / accumulation_steps).backward()
                    return total.detach(), total.detach().exp(), n_ng, n_wg

            # NOT `use_memory AND cross_chunks > 1` (the pre-2026-09-14 form).
            # That conjunct made `--cortex.use_memory false` silently turn
            # CHUNKING off as well: the "control" would have trained on full
            # 4096-token windows against the memory model's 8x512 chunks, and
            # chunk length is the biggest lever the ceiling probes ever found.
            # That is not a control, it is a second experiment
            # (cortex_next_phase_framework.md §1).
            #
            # cortex_fwd_bwd already handles a memory-less chain: the modeling
            # file guards `if self.cortex is not None` at the splice, the read
            # and the unpack; `out.get("m_cross")` returns None BY DESIGN; and
            # detach_old_vecs never fires because `m_cross is not None` is
            # false.  So the chunk chain with no carry is exactly C-chunked.
            _path = select_fwd_bwd_path(cfg.non_recurrent_model,
                                        cfg.cortex["cross_chunks"])
            fwd_bwd_func = {"non_rec": non_rec_fwd_bwd,
                            "cortex": cortex_fwd_bwd,
                            "tight": tightly_scoped_fwd_bwd}[_path]
            loss, log_ppl, num_steps_no_grad, num_steps_with_grad = fwd_bwd_func(model, input_ids, labels)

            # logging
            metrics_to_agg_data_step["loss"].append(loss.item())
            metrics_to_agg_data_step["log_ppl"].append(log_ppl.item())

            if not is_accumulating:
                if cfg.throttle:
                    # NOTE: this is only okay to do as k is the same at each step on all ranks
                    # this will break if k is not the same on all ranks at all steps

                    g = optimizer.param_groups[0] # recur params first, then non recur when initing optim
                    denom = max(1, int(k_mean_tracker[0] / k_mean_tracker[1])) # mean k for this batch
                    g["lr"] = g["lr"] / denom
                    k_mean_tracker  = [0, 0]

                    lrs = [pg["lr"] for pg in optimizer.param_groups]
                    wandb_lr_log  = {"train/lr_recur": lrs[0], "train/lr_nonrecur": lrs[1]}
                else:
                    lrs = [pg["lr"] for pg in optimizer.param_groups]
                    wandb_lr_log  = {"train/lr_recur": lrs[0], "train/lr_nonrecur": lrs[0]}
                # Every group by index, because the two names above both read
                # group 0 (Muon) when throttle is off — so until 2026-09-14 NO
                # logged quantity showed the AdamW group's LR or memory_lr, and a
                # change to either would have been invisible in wandb.
                wandb_lr_log.update({f"train/lr_group{i}": lr for i, lr in enumerate(lrs)})


                total_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=cfg.max_grad_norm,
                    norm_type=2.0,
                ).item()
                grad_clip_coef = min(1.0, float(cfg.max_grad_norm) / (total_norm + 1e-12))

                # Non-finite guard.  clip_grad_norm_ returns a nan/inf total_norm
                # whenever ANY grad is nan/inf (nan loss -> nan grads, or bf16
                # overflow in the deep recurrent unroll -> inf grads).  Applying
                # such grads is unrecoverable: grad clipping does not sanitize nan
                # (nan * coef = nan), so a single bad step poisons every weight
                # for the rest of the run (the 500-step all-nan rung-2 failure).
                # Skip the update entirely instead; count consecutive skips so a
                # run that is nan/inf from step 1 aborts fast, while a transient
                # overflow is dropped and training carries on.
                if math.isfinite(total_norm):
                    optimizer.step()
                    consecutive_nonfinite = 0
                else:
                    consecutive_nonfinite += 1
                    if is_main_process():
                        print(f"[guard] step {optimizer_step + 1}: non-finite grad-norm "
                              f"({total_norm}) — update SKIPPED "
                              f"({consecutive_nonfinite}/{cfg.max_nonfinite_skips})")
                    if consecutive_nonfinite >= cfg.max_nonfinite_skips:
                        raise RuntimeError(
                            f"Aborting: {consecutive_nonfinite} consecutive non-finite "
                            f"updates (last grad-norm={total_norm}). The recurrent unroll "
                            f"is diverging (bf16 overflow / nan) — lower the loop LR, "
                            f"reduce mean_backprop_depth, or keep the m_cross carry + "
                            f"logits in fp32."
                        )

                optimizer.zero_grad(set_to_none=True)
                state["scheduler"].step()
                optimizer_step += 1

                if cfg.mean_recurrence_schedule["turn_on"] or cfg.mean_backprop_depth_schedule["turn_on"]:
                    if cfg.mean_recurrence_schedule["turn_on"]:
                        state["mean_recurrence_scheduler"].step()
                    if cfg.mean_backprop_depth_schedule["turn_on"]:
                        state["mean_backprop_depth_scheduler"].step()
                    num_steps_sampler_partial, new_mean_rec, new_backprop_depth = sheduler_n_k_handler(state, cfg, model_config)

            if not is_accumulating:
                time_taken = (time.monotonic() - step_time)
                time_interval = time_taken / accumulation_steps
                tok_sec = tokens_in_step / time_taken
                elapsed_time += time_taken
                # peak over THIS accumulation window (reset below) — makes the
                # phase-1 -> phase-2 activation jump at the staged unfreeze
                # readable straight off the log.
                peak_gib = torch.cuda.max_memory_allocated(device) / 2**30
                torch.cuda.reset_peak_memory_stats(device)
                print(
                    f"GPU: {model.device} | Step: {data_step:4d} | Updates: {optimizer_step:4d} | Time/step: {time_interval:2.4f}"
                    f" | Tok/sec={tok_sec:9.2f} | Loss: {loss:2.4f} / log-ppl: {log_ppl:2.4f} | Grad-Norm {total_norm:2.4f} | ClipCoef {grad_clip_coef:1.4f}"
                    f" | Peak-Mem {peak_gib:5.2f}GiB"
                )
                total_tokens += tokens_in_step
                step_time = time.monotonic()
                tokens_in_step = 0

                agg_metrics = distributed_and_agg_metrics(metrics_to_agg_data_step, {"total_tokens_with_loss": total_tokens_with_loss, "total_tokens": total_tokens, "tokens_per_second": tok_sec})
                total_tokens_to_log = total_tokens_from_restart + agg_metrics.pop("total_tokens")
                total_tokens_with_loss_to_log = total_tokens_with_loss_from_restart + agg_metrics.pop("total_tokens_with_loss")
                elapsed_time_to_log = elapsed_time_from_restart + elapsed_time

                # THE ARCHITECTURE'S HEALTH, every diag_interval steps.
                # Off by default (diag_interval=0), so no arm on record changes.
                # It goes to wandb AND to a jsonl beside the checkpoints: wandb
                # is for looking at a trajectory, the jsonl is what
                # tools/compare_arms.py reads back when the run is over and is
                # the only copy that survives a project being moved.
                cortex_diag_row = {}
                di = int(cfg.cortex["diag_interval"])
                if di and optimizer_step % di == 0 and is_main_process():
                    cortex_diag_row = training_diag(
                        getattr(get_unwrapped_model(model), "cortex", None),
                        cortex_diag_state["carry"])
                    if cortex_diag_row:
                        cortex_diag_row["step"] = optimizer_step
                        cortex_diag_row["loss"] = float(loss.item())
                        cortex_diag_row["grad_norm"] = float(total_norm)
                        cortex_diag_row["num_steps_no_grad"] = int(num_steps_no_grad)
                        cortex_diag_row["num_steps_with_grad"] = int(num_steps_with_grad)
                        _dp = f"{cfg.out_path}/{cfg.run_name}/cortex_diag.jsonl"
                        os.makedirs(os.path.dirname(_dp), exist_ok=True)
                        with open(_dp, "a", encoding="ascii") as _fh:
                            print(json.dumps(cortex_diag_row), file=_fh)

                if is_main_process():
                    wandb.log({
                        **{f"cortex/{k}": v for k, v in cortex_diag_row.items()
                           if k != "step"},
                        "train/step": optimizer_step,
                        "train/epoch": epoch,
                        "train/lr": state["scheduler"].get_last_lr()[1 if cfg.throttle else 0],
                        "train/total_tokens": total_tokens_to_log,
                        "train/total_tokens_with_loss": total_tokens_with_loss_to_log,
                        "train/total_tokens_no_loss": total_tokens_to_log - total_tokens_with_loss_to_log,
                        "train/total_samples": data_step * cfg.micro_batch_size * world_size,
                        "train/num_steps_no_grad": num_steps_no_grad,
                        "train/num_steps_with_grad": num_steps_with_grad,
                        "train/total_norm": total_norm,
                        "train/grad_clip_coef": grad_clip_coef,
                        "train/grad_clip_max_norm": cfg.max_grad_norm,
                        "train/mean_recurrence": new_mean_rec,
                        "train/mean_backprop_depth": new_backprop_depth,
                        "train/elapsed_time": elapsed_time_to_log,
                        **{f"train/{k}": v for k,v in agg_metrics.items()},
                        **wandb_lr_log,
                    }, step=optimizer_step)

                    if (cfg.save_interval != -1) and (optimizer_step % cfg.save_interval == 0):
                        save_model_only(cfg, state, f"model_only_chkpt_{optimizer_step}")

                if (cfg.save_interval != -1) and (optimizer_step % (2 * cfg.save_interval) == 0):
                    # have to call save_checkpoint on all ranks for the dataloader
                    save_checkpoint(state, {"data_start_step": data_step, "optimizer_step": optimizer_step, "total_tokens": total_tokens_to_log, "total_tokens_with_loss": total_tokens_with_loss_to_log, "elapsed_time": elapsed_time_to_log}, cfg)

                if cfg.save_n_mins_before_timeout is not None:
                    if check_if_save(cfg.save_n_mins_before_timeout):
                        save_checkpoint(state, {"data_start_step": data_step, "optimizer_step": optimizer_step, "total_tokens": total_tokens_to_log, "total_tokens_with_loss": total_tokens_with_loss_to_log, "elapsed_time": elapsed_time_to_log}, cfg)
                        if torch.distributed.is_initialized():
                            torch.distributed.barrier()

            if cfg.max_steps and optimizer_step >= cfg.max_steps:
                reached_target = True
                break
            # Early stop that does NOT move the schedule horizon.  max_steps is
            # the denominator for the LR cosine AND the mean-recurrence ramp, so
            # shortening it to end a phase early would re-run cooldown per phase
            # (the 2026-06-24 sawtooth, baked into the weights).  This lets the
            # healing phase end at its own step count while every phase of the
            # run stays on one 305,176-step schedule.
            if cfg.stop_at_step and optimizer_step >= cfg.stop_at_step:
                if is_main_process():
                    print(f"[stop_at_step] reached {optimizer_step} — ending this "
                          f"phase (schedule horizon max_steps={cfg.max_steps} "
                          f"unchanged)")
                # Make sure the phase's end is resumable/branchable even if it
                # does not land on a save_interval boundary.
                save_checkpoint(state, {"data_start_step": data_step,
                                        "optimizer_step": optimizer_step,
                                        "total_tokens": total_tokens_to_log,
                                        "total_tokens_with_loss": total_tokens_with_loss_to_log,
                                        "elapsed_time": elapsed_time_to_log}, cfg)
                reached_target = True
                break
        if reached_target:
            break

    # EXHAUSTION GUARD (recipe audit, finding 1).  Falling off the end of the
    # dataloader used to be indistinguishable from finishing: the loop ends,
    # train() returns, save_model_only writes "final_checkpoint" and wandb marks
    # the run FINISHED — at whatever fraction of the budget the pack covered.
    # That is what stopped both B1 arms at 24,414 steps / 400M tokens while the
    # loss curve looked perfectly healthy.  Fail loudly instead.
    if not reached_target and (cfg.max_steps or cfg.stop_at_step):
        target = cfg.stop_at_step or cfg.max_steps
        raise RuntimeError(
            f"Dataloader exhausted at optimizer step {optimizer_step:,} of "
            f"{target:,} ({100 * optimizer_step / max(1, target):.1f}% of the "
            f"budget) after {data_step:,} micro-batches.\n"
            f"  The pack at {cfg.preprocessed_data_path} is too small for this "
            f"link.  Size it as  cursor_at_link_start + steps_to_serve * "
            f"batch_size  — see tools/check_pack.py --resume_rows — and note "
            f"that a corpus switch on a RESUME carries the cursor forward "
            f"unless --reset_dataset_position true is passed."
        )

    model.eval()
    return state


def main():
    """Encapsulates main scope away from import calls."""

    # Configuration loader
    cfg: CLISettings = CLI(CLISettings)

    # Print system setup
    if is_main_process():
        print("--------------------------------------------------------------------")
        print(f"------------------ Launching run {cfg.run_name}------------------")
        print("--------------------------------------------------------------------")
        print("--------------------------------------------------------------------")
        print(f"Platform: {sys.platform}, Python: {sys.version.split(' (')[0]}, PyTorch: {torch.__version__}")
        print(f"CPU threads: {torch.get_num_threads()}, GPUs: {torch.cuda.device_count()} on {socket.gethostname()}.")
        driver = f"HIP/ROCM {torch.version.hip}" if torch.version.hip else f"CUDA: {torch.version.cuda}"
        print(f"GPU : {torch.cuda.get_device_name()}. {driver}.")

    # set flags
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True  # Should be true anyway
    torch._dynamo.config.optimize_ddp = "python_reducer"
    # have to use the below two together as we do error if we compile the gradient states the no_grad/grad step
    torch._dynamo.config.compiled_autograd = False # didn't work for Jonas ever...
    # torch._dynamo.config.error_on_recompile = True # Here's to hoping

    train_time = time.monotonic()
    # Set up dist and load model and tokenizer into state
    state, device = startup(cfg)
    data_start_step, optimizer_step, total_tokens, total_tokens_with_loss, elapsed_time = 1, 0, 0, 0, 0.0
    if cfg.resume_path is not None or cfg.branch_path is not None:
        agg_dict = load_checkpoint(state, cfg, device, branch=cfg.resume_path is None)
        data_start_step, optimizer_step, total_tokens, total_tokens_with_loss, elapsed_time = agg_dict["data_start_step"], agg_dict["optimizer_step"], agg_dict["total_tokens"], agg_dict["total_tokens_with_loss"], agg_dict["elapsed_time"]
        # max_steps is an ABSOLUTE horizon.  A restored counter already at or
        # past it runs one optimizer step and exits "reached_target" -- a
        # finished-looking run that trained nothing.
        if cfg.max_steps and optimizer_step >= cfg.max_steps:
            raise RuntimeError(
                f"restored optimizer_step={optimizer_step:,} is already >= "
                f"max_steps={cfg.max_steps:,}: this run would train for one "
                f"step and report success.  max_steps is absolute, not NEW steps.")
        # cfg.max_steps = optimizer_step + cfg.max_steps # make max_steps max NEW steps

    # train
    state = train(state, device, cfg, data_start_step, optimizer_step, total_tokens, total_tokens_with_loss, elapsed_time)
    save_model_only(cfg, state, "final_checkpoint")

    # Now exit
    if is_main_process():
        print("--------------------------------------------------------------------")
        print(f"Training time: {str(datetime.timedelta(seconds=time.monotonic() - train_time))} ")
        max_alloc = f"{torch.cuda.max_memory_allocated(device) / float(1024**3):,.3f} GB"
        max_reserved = f"{torch.cuda.max_memory_reserved(device) / float(1024**3):,.3f} GB"
        print(f"Max. Mem allocated: {max_alloc}. Max. Mem reserved: {max_reserved}.")
        print("--------------------------------------------------------------------")
        wandb.finish()
        dataset_save_dir = f"{cfg.out_path}/{cfg.run_name}/dataset"
        if os.path.exists(dataset_save_dir):
            shutil.rmtree(dataset_save_dir)


def shutdown():
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
    print(f"---------Total time: {str(datetime.timedelta(seconds=time.monotonic() - global_start_time))} ---------")
    print("-----------------Shutdown complete.--------------------------")


def guarded_main():
    try:
        run_name = main()
        print("--------------------------------------------------------------------")
        print(f"Run {run_name} finished without error.")
    except BaseException:
        print("--------------------------------------------------------------------")
        print("Run finished with errors.")
        raise
    finally:
        shutdown()  # guarantee NCCL deconstruction


if __name__ == "__main__":
    guarded_main()