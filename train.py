"""Based on https://github.com/seal-rg/recurrent-pretraining/blob/main/finetuning_simple_example.py"""

####################################################################################################
# Imports.
####################################################################################################

import time

global_start_time = time.monotonic()
import os
import socket
from typing import Any, Optional
from functools import partial
import sys
import datetime
import shutil
import subprocess
import torch
import wandb
import math
from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler, AutoConfig
from datasets import load_dataset, Dataset, load_from_disk
from contextlib import nullcontext
from stateful_parquet_dataset import get_parquet_dataloader
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
    epochs: int = 1
    batch_size: int = 32
    optim_config: dict[str, Any] = field(
        default_factory=lambda: dict(lr=5e-7, weight_decay=1e-4, betas=(0.9, 0.95), eps=1e-8)
    )
    scheduler_args: dict[float, Any] = field(
        default_factory=lambda: dict(warmup=0.1, cooldown=0.1, min_lr_ratio=0.001)
    ) # min_lr = min_lr_ratio * lr
    save_interval: int = -1
    model_name: str = "smcleish/Recurrent-TinyLlama-3T-untrained"
    wandb_disabled: bool = False
    seed: int = 74
    fix_num_steps: bool = False
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
    # Debug: register forward hooks that report the FIRST module to emit a
    # non-finite output from finite inputs (i.e. the op that births the nan/inf),
    # and run fwd+bwd under torch.autograd.set_detect_anomaly to localize the
    # backward.  Slow — use with --max_steps 2-3 for a quick locate, then off.
    debug_nan: bool = False
    bf16_true: bool = False
    compile_warmup_routine: bool = False
    no_amp: bool = True
    is_parquet_dataset: bool = False
    ignore_past_parquet_dataset: bool = False
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
    # freeze_loop_until_step : staged unfreeze — keep the loop frozen until this
    #                       optimizer step, then unfreeze it (0 = never auto-unfreeze).
    # eos_from_tokens     : derive eos_mask from the token ids (== tokenizer.eos)
    #                       and pass it into each chunk forward, so the M_cross
    #                       write pools only the open document suffix and resets
    #                       across doc/pad boundaries.  Off = eos_mask=None (full
    #                       carry; correct for one-doc sequences that fill the
    #                       window, and what the evals use).  Turn on when data
    #                       has padded short docs (pad == eos) or packed docs.
    # l2sp_coeff          : L2-SP anchor (experiment-ladder rung 3): add
    #                       coeff * ||theta_loop - theta_loop^base||^2 to the loss,
    #                       anchoring the unfrozen loop to the PRETRAINED weights
    #                       (snapshot taken from model_name at startup, i.e. the
    #                       base graft dir, BEFORE any --resume_path load).  0 = off.
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
    cortex: dict[str, Any] = field(
        default_factory=lambda: dict(
            use_memory=False, memory_slots=0, memory_slots_iter=0, memory_heads=4,
            ccot_direct=False, h_T_proj=True, cross_chunks=1,
            freeze_loop=False, freeze_loop_until_step=0, eos_from_tokens=False,
            l2sp_coeff=0.0, memory_lr=0.0, lora_rank=0, lora_alpha=32.0,
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
        if self.cortex["l2sp_coeff"] > 0:
            assert self.cortex["use_memory"] and self.cortex["cross_chunks"] > 1, (
                "cortex.l2sp_coeff is only applied inside the cross-chunk fwd/bwd "
                "path (requires cortex.use_memory and cortex.cross_chunks > 1)"
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

has_completed_timeout_save = False
def check_if_save(save_n_mins_before_timeout):
    global has_completed_timeout_save
    if (save_n_mins_before_timeout * 60 > get_flux_timeleft()) and (not has_completed_timeout_save):
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

def load_checkpoint(state, cfg, device):
    ckpt = torch.load(f"{cfg.resume_path}/chkpt.pt", map_location=device)
    unwrap = get_unwrapped_model(state)
    unwrap.load_state_dict(ckpt["model"], strict=True)
    state["optimizer"].load_state_dict(ckpt["optimizer"])

    if cfg.mean_recurrence_schedule["turn_on"] and ("mean_recurrence_scheduler" in ckpt):
        state["mean_recurrence_scheduler"].load_state_dict(ckpt["mean_recurrence_scheduler"])
    if cfg.mean_backprop_depth_schedule["turn_on"] and ("mean_backprop_depth_scheduler" in ckpt):
        state["mean_backprop_depth_scheduler"].load_state_dict(ckpt["mean_backprop_depth_scheduler"])

    if not cfg.ignore_past_scheduler:
        state["scheduler"].load_state_dict(ckpt["scheduler"])
    if cfg.is_parquet_dataset and not cfg.ignore_past_parquet_dataset:
        state["dataloader"].load_state_dict(ckpt["dataloader"])

    torch.set_rng_state(ckpt["rng_state"].to("cpu"))
    torch.cuda.set_rng_state_all([rng.to("cpu") for rng in ckpt["cuda_rng_state"]])
    print(f"Resumed from {cfg.resume_path}")
    return ckpt["agg_vars_dict"]

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


def install_nan_localizer(model):
    """Register forward hooks on every leaf module; the FIRST one that emits a
    non-finite output while ALL its inputs were finite is the op that birthed the
    nan/inf (propagated nan downstream is ignored).  Debug-only (cfg.debug_nan) —
    the per-module .all() checks are too slow for a real run.  Pairs with
    set_detect_anomaly (backward) in train()."""
    target = get_unwrapped_model_from_module(model)
    flag = {"reported": False}

    def _finite(t):
        return (not torch.is_tensor(t)) or torch.isfinite(t).all().item()

    def hook(module, inputs, output):
        if flag["reported"]:
            return
        outs = output if isinstance(output, (tuple, list)) else (output,)
        out_bad = any(torch.is_tensor(o) and not torch.isfinite(o).all().item() for o in outs)
        if out_bad and all(_finite(i) for i in inputs):
            flag["reported"] = True
            if is_main_process():
                print(f"[nan-localizer] FIRST non-finite born in module "
                      f"'{module._nan_name}' ({type(module).__name__}) — inputs "
                      f"were finite, so this op produced the nan/inf.")

    for name, m in target.named_modules():
        if not list(m.children()):          # leaf modules only
            m._nan_name = name
            m.register_forward_hook(hook)
    if is_main_process():
        print("[nan-localizer] forward hooks installed (debug_nan=True — expect a slowdown)")
    return flag


def reset_cortex_graft_init(model):
    """Undo post_init's clobbering of the cortex graft's initialization, on the
    live model after from_pretrained.

    RavenForCausalLM.__init__ builds CortexMemory (designed inits so the memory
    read is a no-op and step-0 == the base model) and THEN calls post_init().
    HF's _init_weights treats every cortex tensor as a freshly-'missing' key (the
    "newly initialized: ['cortex.h_T_proj.weight', 'cortex.m_cross.cand_ln1.bias',
    ...]" load warning) and re-initializes it with the raven DEPTH-SCALED scheme.
    The graft modules have no valid layer index, so that scheme hands them an
    effectively-infinite std -> NON-FINITE weights.  That is the confirmed
    forward-nan source, and it hits EVERY cortex op in turn (localizer found
    h_T_proj first, then m_cross.cand_ln1, ...), so restoring hand-picked tensors
    is whack-a-mole.  Instead reset the WHOLE cortex subtree:
      (1) every submodule back to its nn default (Linear->kaiming, LayerNorm->
          weight 1/bias 0) — finite, in place, dtype/device preserved;
      (2) re-apply the few explicit designed inits the graft sets by hand.
    Mirrors cortex_graft.CortexMemory + cortex_memory.buffers; skipped on
    --resume (a resumed ckpt carries trained, not fresh, cortex weights).  LoRA
    A/B are bare Parameters _init_weights never touches (B stays 0), left alone."""
    cortex = getattr(get_unwrapped_model_from_module(model), "cortex", None)
    if cortex is None:
        return
    # (1) undo the non-finite clobber: nn defaults for every submodule.
    n_reset = 0
    for m in cortex.modules():
        if m is not cortex and callable(getattr(m, "reset_parameters", None)):
            m.reset_parameters(); n_reset += 1
    # (2) re-apply the graft's explicit designed inits (mirror the source).
    fixed = []
    if getattr(cortex, "h_T_proj", None) is not None:
        torch.nn.init.eye_(cortex.h_T_proj.weight); fixed.append("h_T_proj=eye")
    for buf_name in ("m_cross", "m_iter"):                     # LSTMBuffer
        buf = getattr(cortex, buf_name, None)
        if buf is None:
            continue
        torch.nn.init.zeros_(buf.out_proj.weight)              # read starts at 0
        torch.nn.init.normal_(buf.slot_emb, std=0.02)
        torch.nn.init.ones_(buf.forget_bias)                  # LM2 §3.3 forget bias +1
        torch.nn.init.zeros_(buf.input_bias)
        fixed.append(f"{buf_name}.[out_proj=0,slot_emb~N,forget_bias=1,input_bias=0]")
    ccot = getattr(cortex, "ccot_direct", None)                # DirectCCoT (K=0)
    if ccot is not None:
        torch.nn.init.eye_(ccot.state_proj.weight); fixed.append("ccot.state_proj=eye")
        torch.nn.init.zeros_(ccot.in_proj.weight); fixed.append("ccot.in_proj=0")
    # (3) insurance: nothing in cortex should be non-finite now — warn loudly if
    #     some module lacked reset_parameters and slipped through.
    bad = [n for n, p in cortex.named_parameters() if not torch.isfinite(p).all()]
    if is_main_process():
        print(f"[cortex] reset {n_reset} cortex submodules to nn defaults + "
              f"re-applied designed inits {fixed} (undo post_init clobber)")
        if bad:
            print(f"[cortex] WARNING: {len(bad)} cortex params STILL non-finite "
                  f"after reset (no reset_parameters?): {bad[:12]}")


def get_unwrapped_model_from_module(model):
    """Unwrap DDP / torch.compile to reach named_parameters with stable names."""
    m = model
    if hasattr(m, "module"):
        m = m.module
    if hasattr(m, "_orig_mod"):
        m = m._orig_mod
    return m


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
                   "lora_rank", "lora_alpha"):
            setattr(config, _k, cfg.cortex[_k])
        if is_main_process():
            print(f"[cortex] memory ON: K={cfg.cortex['memory_slots']} "
                  f"K_iter={cfg.cortex['memory_slots_iter']} "
                  f"ccot_direct={cfg.cortex['ccot_direct']} "
                  f"cross_chunks={cfg.cortex['cross_chunks']} "
                  f"lora_rank={cfg.cortex['lora_rank']}")
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
    if (cfg.cortex["use_memory"] and cfg.cortex["lora_rank"] > 0
            and getattr(model, "cortex_lora", None) is None):
        raise RuntimeError(
            "cfg.cortex.lora_rank is set but model.cortex_lora is None — the "
            "grafted modeling file predates the LoRA graft. Re-run "
            "tools/prepare_cortex_checkpoint.py to refresh the model dir."
        )

    # cortex: undo post_init's clobbering of the graft's designed inits (must run
    # AFTER from_pretrained, BEFORE the L2-SP snapshot / freeze / optimizer build
    # so the anchor + optimizer see the intended weights).  Skipped on --resume
    # (a resumed checkpoint carries the trained cortex weights, not fresh ones).
    if cfg.cortex["use_memory"] and cfg.resume_path is None:
        reset_cortex_graft_init(model)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # cortex: optionally freeze the recurrent loop (train memory + coda only).
    # Done on the unwrapped model BEFORE the DDP wrap.  Staged unfreeze (if
    # freeze_loop_until_step > 0) re-enables it later inside train(), which
    # under DDP requires re-wrapping the model (the reducer only registers
    # params that required grad at construction) — that re-wrap cannot reach
    # through torch.compile, so forbid the combination up front.
    if (cfg.cortex["use_memory"] and cfg.cortex["freeze_loop"]
            and cfg.cortex["freeze_loop_until_step"] > 0
            and distributed and cfg.compile):
        raise RuntimeError(
            "cortex.freeze_loop_until_step > 0 with DDP requires re-wrapping the "
            "model at the unfreeze step, which is not supported under "
            "torch.compile. Run with --compile=false, or split into two runs "
            "(rung 1 frozen, then resume with freeze_loop=false)."
        )
    if cfg.cortex["use_memory"] and cfg.cortex["freeze_loop"]:
        n_frozen = set_loop_trainable(model, trainable=False)
        if is_main_process():
            print(f"[cortex] froze {n_frozen} loop (adapter+core_block) params; "
                  f"unfreeze at step {cfg.cortex['freeze_loop_until_step'] or 'never'}")

    # cortex: L2-SP anchor snapshot (rung 3).  Taken here — after loading
    # model_name (the base graft dir) but BEFORE any --resume_path load — so the
    # reference is always the PRETRAINED loop, even when resuming a rung-1/2
    # checkpoint.  Pairs hold live param references (stable through DDP/compile
    # wrapping) next to their frozen base copies.
    l2sp_pairs = None
    if cfg.cortex["l2sp_coeff"] > 0:
        l2sp_pairs = [
            (p, p.detach().clone())
            for n, p in model.named_parameters()
            if ("adapter" in n) or ("core_block" in n)
        ]
        if is_main_process():
            print(f"[cortex] L2-SP anchor on {len(l2sp_pairs)} loop tensors "
                  f"(coeff={cfg.cortex['l2sp_coeff']})")

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
        param_groups.append(
            dict(params=non_body_params + norms, use_muon=False, lr=cfg.optim_config["lr"], betas=cfg.optim_config["betas"], weight_decay=cfg.optim_config["weight_decay"]),
        )
        if memory_lr > 0 and cortex_params:
            # dedicated group for the fresh cortex params: higher LR, no weight
            # decay (decay would pull the identity-init projections toward zero).
            param_groups.append(
                dict(params=cortex_params, use_muon=False, lr=memory_lr, betas=cfg.optim_config["betas"], weight_decay=0.0),
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
        dataloader = torch.utils.data.DataLoader(
            tokenized_dataset,  # type: ignore
            batch_size=cfg.micro_batch_size,
            shuffle=not cfg.is_parquet_dataset,
            pin_memory=True,
            generator=dataloader_generator,
        )

    ##########     Scheduler       ##############
    if cfg.is_parquet_dataset:
        if cfg.max_steps:
            max_training_steps = cfg.max_steps
        else:
            max_training_steps = max(1, math.ceil(cfg.parquet_dataset_max_tokens / world_size / cfg.max_length))
        num_warmup_steps = math.ceil(cfg.scheduler_args["warmup"] * max_training_steps)
        num_decay_steps = math.ceil(cfg.scheduler_args["cooldown"] * max_training_steps)
    else:
        if cfg.max_steps:
            max_training_steps = cfg.max_steps
        else:
            accumulation_steps = max(1, cfg.batch_size // cfg.micro_batch_size)
            num_update_steps_per_epoch = math.ceil(len(dataloader) / accumulation_steps)
            max_training_steps = cfg.epochs * num_update_steps_per_epoch
        num_warmup_steps = math.ceil(cfg.scheduler_args["warmup"] * max_training_steps)
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
        "l2sp_pairs": l2sp_pairs,
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

    if cfg.debug_nan:
        install_nan_localizer(model)

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

    for epoch in range(cfg.epochs):
        for data_step, inputs in enumerate(state["dataloader"], start=(data_start_step + 1) if cfg.is_parquet_dataset else 1):
            if (data_start_step != 1) and (not cfg.is_parquet_dataset) and (data_step <= data_start_step):
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
 
            if cfg.fix_num_steps:
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
                with model.no_sync() if is_accumulating and state["distributed"] else nullcontext():
                    # .contiguous(): torch.chunk returns non-contiguous views and
                    # the model's loss does labels.view(-1), which requires contiguity.
                    x_chunks = [c.contiguous() for c in torch.chunk(input_ids, n_chunks, dim=1)]
                    y_chunks = [c.contiguous() for c in torch.chunk(labels, n_chunks, dim=1)]
                    m_cross = None
                    chunk_losses = []
                    n_ng = n_wg = 0
                    for xc, yc in zip(x_chunks, y_chunks):
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
                        if (yc != -100).any():                  # skip fully-masked chunks
                            chunk_losses.append(out["loss"])
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
                    total = torch.stack(chunk_losses).mean()
                    # L2-SP anchor (rung 3): pull the unfrozen loop toward the
                    # pretrained weights.  Added to the backward objective only —
                    # `total` (the logged loss) stays pure LM loss so curves are
                    # comparable across rungs.  While the loop is frozen the
                    # penalty is a constant and contributes no gradient.
                    objective = total
                    if state["l2sp_pairs"]:
                        pen = torch.stack(
                            [(p - ref).pow(2).sum() for p, ref in state["l2sp_pairs"]]
                        ).sum()
                        objective = total + float(cfg.cortex["l2sp_coeff"]) * pen
                    (objective / accumulation_steps).backward()
                    return total.detach(), total.detach().exp(), n_ng, n_wg

            if cfg.non_recurrent_model:
                fwd_bwd_func = non_rec_fwd_bwd
            elif cfg.cortex["use_memory"] and int(cfg.cortex["cross_chunks"]) > 1:
                fwd_bwd_func = cortex_fwd_bwd
            else:
                fwd_bwd_func = tightly_scoped_fwd_bwd
            with torch.autograd.set_detect_anomaly(True) if cfg.debug_nan else nullcontext():
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
                    if cfg.debug_nan and is_main_process():
                        # Localize which params carry the non-finite grad (the
                        # forward-hook + anomaly mode miss an inf GRADIENT, e.g.
                        # rung1b's BPTT explosion through the in-loop LoRA).
                        bad = [n for n, p in get_unwrapped_model_from_module(model).named_parameters()
                               if p.grad is not None and not torch.isfinite(p.grad).all()]
                        print(f"[nan-localizer] {len(bad)} param tensors have non-finite grads; "
                              f"first offenders: {bad[:12]}")
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

                # cortex: staged unfreeze — re-enable the loop once past the
                # configured step (rung 2 of the experiment ladder).
                if (cfg.cortex["use_memory"] and cfg.cortex["freeze_loop"]
                        and cfg.cortex["freeze_loop_until_step"] > 0
                        and optimizer_step == cfg.cortex["freeze_loop_until_step"]):
                    n_unfrozen = set_loop_trainable(model, trainable=True)
                    if state["distributed"]:
                        # DDP's reducer only registers params that required grad
                        # at wrap time; without a re-wrap the newly-unfrozen loop
                        # grads would never all-reduce and ranks silently drift.
                        # Params are identical across ranks here (frozen ones
                        # untouched, trained ones just synced), and the optimizer
                        # holds references to the underlying params, so a fresh
                        # wrapper is safe.  startup() forbids this path under
                        # torch.compile.
                        model = torch.nn.parallel.DistributedDataParallel(
                            get_unwrapped_model_from_module(model),
                            device_ids=[device],
                            find_unused_parameters=True,
                            gradient_as_bucket_view=True,
                        )
                        state["model"] = model
                    if is_main_process():
                        print(f"[cortex] step {optimizer_step}: unfroze {n_unfrozen} loop params"
                              + (" (re-wrapped DDP)" if state["distributed"] else ""))

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
                print(
                    f"GPU: {model.device} | Step: {data_step:4d} | Updates: {optimizer_step:4d} | Time/step: {time_interval:2.4f}"
                    f" | Tok/sec={tok_sec:9.2f} | Loss: {loss:2.4f} / log-ppl: {log_ppl:2.4f} | Grad-Norm {total_norm:2.4f} | ClipCoef {grad_clip_coef:1.4f}"
                )
                total_tokens += tokens_in_step
                step_time = time.monotonic()
                tokens_in_step = 0

                agg_metrics = distributed_and_agg_metrics(metrics_to_agg_data_step, {"total_tokens_with_loss": total_tokens_with_loss, "total_tokens": total_tokens, "tokens_per_second": tok_sec})
                total_tokens_to_log = total_tokens_from_restart + agg_metrics.pop("total_tokens")
                total_tokens_with_loss_to_log = total_tokens_with_loss_from_restart + agg_metrics.pop("total_tokens_with_loss")
                elapsed_time_to_log = elapsed_time_from_restart + elapsed_time

                if is_main_process():
                    wandb.log({
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
                break

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
    if cfg.resume_path is not None:
        agg_dict = load_checkpoint(state, cfg, device)
        data_start_step, optimizer_step, total_tokens, total_tokens_with_loss, elapsed_time = agg_dict["data_start_step"], agg_dict["optimizer_step"], agg_dict["total_tokens"], agg_dict["total_tokens_with_loss"], agg_dict["elapsed_time"]
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