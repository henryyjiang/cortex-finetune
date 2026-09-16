"""Peak-memory and wall-clock gate for the config-D row-length decision.

The last open item in `cortex_next_phase_framework.md` §10 is not a design
question but an OOM question: does 8,192 x cross_chunks 16 fit on one H200 at
`micro_batch_size 1`?  This measures it, data-free, before any pack is built —
a failure here costs an hour, a failure after the corpus is rebuilt costs a week.

WHY NOT tools/smoke_prefix_real.py.  That script already runs the chunk chain
and prints peak VRAM, so the framework's "not yet written" is stale as far as
the chain goes.  But it measures a DIFFERENT run than the one we are sizing, in
three ways that all point the same direction — under-reporting:

  1. `pace/smoke_prefix_real.sbatch` passes `--dtype bfloat16`, so the weights
     are bf16.  Neither `pace/b2_retrofit.sbatch` nor `shells/olmo.sh` sets
     `--bf16_true`, and train.py:127 defaults it False, so **the real run holds
     fp32 weights and fp32 grads** under a bf16 autocast (`--no_amp false`).
     Same model, ~2x the resident parameter memory.
  2. It carries **no optimizer state**.  MuonWithAuxAdam holds a momentum buffer
     per Muon param and Adam's two moments per aux param; on a 1B model that is
     tens of GiB, and it is resident during the backward of every later step.
     Its own sbatch header says so ("It carries no optimizer state") — fine for
     a code-path check, not for a ceiling.
  3. It pins `num_steps = [T//2, T - T//2]`, which is not the worst case.  The
     sampler caps `k` at `mean_backprop_depth` (see `worst_case_num_steps`), and
     the deepest retained graph the run can build is `k = s`, not `s/2`.

So this script measures the steady-state peak of a REAL micro-step: fp32 weights,
bf16 autocast, worst-case recurrence split, the actual Muon+Adam groups, and one
optimizer step taken before the measured step so its state is already resident.

It is a MEASUREMENT, not an assertion.  Exit 0 means "the measurement completed"
— fit or OOM, both are answers.  Exit 1 means the script itself broke.  That is
what lets the sbatch sweep three geometries in one job without a blanket
`|| true` that would also hide a real failure.

Cluster (H200; an A100-80 answer is not the answer to this question):

    sbatch pace/smoke_geometry_oom.sbatch

One geometry by hand, on an interactive GPU node:

    python tools/smoke_geometry_oom.py --max_length 8192 --cross_chunks 16

Correctness of the numbers rests on this file mirroring train.py; every mirrored
block names the lines it was copied from.  If train.py's optimizer grouping or
`cortex_fwd_bwd` changes, this drifts silently — `tests/test_smoke_geometry_oom.py`
pins the parts that can be pinned without a GPU.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import torch  # noqa: E402

from cortex_memory.chunking import detach_old_vecs  # noqa: E402
from recipe_utils import (  # noqa: E402
    carry_rows,
    gated_carry_rows,
    reduce_chunk_losses,
    worst_case_num_steps,
)

GIB = 2 ** 30


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("config-D geometry OOM gate")
    p.add_argument("--model_name", default="ckpts/olmo-retrofit-cortex",
                   help="graft-prepared checkpoint dir — the control's base")
    # geometry
    p.add_argument("--max_length", type=int, default=8192, help="tokens per row")
    p.add_argument("--cross_chunks", type=int, default=16)
    p.add_argument("--micro_batch_size", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=16,
                   help="gradient accumulation; used only to project wall clock")
    p.add_argument("--max_steps", type=int, default=38147,
                   help="horizon; used only to project wall clock")
    # recurrence
    p.add_argument("--mean_recurrence", type=int, default=8,
                   help="MAX_MEAN_REC — the ramp's endpoint, i.e. the expensive end")
    p.add_argument("--backprop_depth", type=int, default=0,
                   help="0 = read mean_backprop_depth off the checkpoint config, "
                        "which is what train.py does when nothing overrides it")
    # memory mechanism (measure the MEMORY model: strictly the larger footprint)
    p.add_argument("--use_memory", default="true", choices=["true", "false"])
    p.add_argument("--prefix_memory", default="accum", choices=["accum", "gated"])
    p.add_argument("--accum_vecs", type=int, default=32)
    p.add_argument("--accum_max", type=int, default=0,
                   help="0 = cross_chunks x accum_vecs, which is B2's convention "
                        "(b2_retrofit.sbatch:409, 'moves WITH cross_chunks') and "
                        "the LARGER footprint: the FIFO never trims, so the last "
                        "chunk reads over every earlier write.  Measure this, not "
                        "a capped buffer — a gate must be conservative.")
    p.add_argument("--gate_slots", type=int, default=0,
                   help="K for --prefix_memory gated: carried columns held.  "
                        "0 = same as --accum_vecs (the pre-P1.0 shape).  A3' "
                        "runs 64.  Ignored by the accum buffer, which is capped "
                        "by --accum_max instead.")
    p.add_argument("--latent_carry", action="store_true",
                   help="price the Z CHANNEL.  Not free and not obviously "
                        "cheap: the carried tensor widens to 2D (small), the "
                        "trajectory tape adds T x [B, n_vec, D] (about 1 MB at "
                        "T=8/W=16/D=2048, also small) -- but the SECOND GATE is "
                        "two more [D, 2D] Linears, i.e. ~33.5M parameters with "
                        "their optimizer state, which is not small.  A Z arm "
                        "that was priced on an E-only run was not priced.")
    p.add_argument("--gate_route", default="ring", choices=["ring", "mix"])
    p.add_argument("--gate_norm", default="tanh", choices=["tanh", "rms", "none"])
    p.add_argument("--gate_init", default="zero", choices=["zero", "default"])
    p.add_argument("--gate_fill", default="grow", choices=["grow", "init"])
    p.add_argument("--carry_grad_chunks", type=int, default=0,
                   help="0 = cross_chunks // 2, B2's 50%% convention (4 of 8 in "
                        "the arm).  This is a MEMORY parameter, not only a "
                        "gradient one: it sets how many chunks of carry keep "
                        "their graph, so 8 of 16 retains twice what 4 of 8 did.")
    # optimizer (b2_retrofit.sbatch:478-486)
    p.add_argument("--with_optimizer", default="true", choices=["true", "false"],
                   help="false measures the framework's literal spec (fwd+bwd "
                        "only) and will UNDER-report the run by the optimizer "
                        "state; true is the number to decide on")
    p.add_argument("--muon_lr", type=float, default=1e-3)
    p.add_argument("--adam_lr", type=float, default=5e-5)
    p.add_argument("--memory_lr", type=float, default=5e-4)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    # measurement
    p.add_argument("--warmup_steps", type=int, default=1,
                   help="micro-steps run BEFORE the measured one, so optimizer "
                        "state and the caching allocator are at steady state. "
                        "0 measures a first step and under-reports.")
    p.add_argument("--json_out", default="",
                   help="append one JSON line per run; the sbatch summarises it")
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def build_model(args, use_memory: bool):
    """Mirror of train.py's model build for the cortex path.

    Precision mirrors train.py:856-859 with `bf16_true` at its default False:
    fp32 weights, bf16 only inside autocast.  This is the single largest
    difference from `smoke_prefix_real.py` and it is the whole point of the file.
    """
    from transformers import AutoConfig, AutoModelForCausalLM

    cfg = AutoConfig.from_pretrained(args.model_name, trust_remote_code=True)
    # The 16 persisted cortex flags (CLEANUP_AUDIT's rule): set the ones this
    # geometry depends on, leave the rest at the checkpoint's values.
    for k, v in (("use_memory", use_memory),
                 ("memory_slots", 0), ("memory_slots_iter", 0),
                 ("prefix_memory", args.prefix_memory),
                 ("accum_vecs", args.accum_vecs),
                 ("accum_max", args.accum_max),
                 ("gate_slots", args.gate_slots), ("gate_route", args.gate_route),
                 ("gate_norm", args.gate_norm), ("gate_init", args.gate_init),
                 ("gate_fill", args.gate_fill),
                 ("latent_carry", args.latent_carry),
                 ("prefix_pos", "tail"), ("prefix_eos_reset", False)):
        setattr(cfg, k, v)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, trust_remote_code=True, config=cfg,
        torch_dtype=torch.float32)
    return model.to(args.device).train(), cfg


def build_param_groups(model, muon_lr: float, adam_lr: float, memory_lr: float,
                       betas=(0.9, 0.95), eps: float = 1e-8,
                       muon_wd: float = 1e-4, adam_wd: float = 1e-4):
    """Mirror of train.py:1078-1132, the recurrent (huginn) branch.

    `throttle` is off (train.py default, and the recipe audit says it must stay
    off) and `non_recurrent_model` is false, so the two branches dropped here are
    dead for every run in the plan.  `eps` defaults to 1e-8 because that is what
    train.py now passes explicitly — before 2026-09-14 the key was omitted and
    MuonWithAuxAdam filled in its own 1e-10.

    Returned as plain dicts so a CPU test can check the PARTITION without
    constructing Muon, which needs a process group.
    """
    body, non_body, norms, cortex_params = [], [], [], []
    for n, p in model.named_parameters():
        if "cortex" in n:
            cortex_params.append(p)
        elif ("norm" in n) or ("ln_f" in n) or ("Wqkv.bias" in n):
            norms.append(p)
        elif ("wte" in n) or ("lm_head" in n):
            non_body.append(p)
        else:
            body.append((n, p))
    body.sort(key=lambda np: (-np[1].numel(), tuple(np[1].shape), np[0]))
    body = [p for _, p in body]

    groups = [dict(params=body, use_muon=True, lr=muon_lr,
                   weight_decay=muon_wd, no_sorting_in_init=False)]
    if not (memory_lr > 0):
        non_body = non_body + cortex_params
    groups.append(dict(params=non_body + norms, use_muon=False, lr=adam_lr,
                       betas=betas, eps=eps, weight_decay=adam_wd))
    if memory_lr > 0 and cortex_params:
        groups.append(dict(params=cortex_params, use_muon=False, lr=memory_lr,
                           betas=betas, eps=eps, weight_decay=0.0))
    return groups


def init_trivial_process_group():
    """train.py:845-855.  MuonWithAuxAdam calls dist.get_world_size() inside
    .step(); without a group it raises rather than assuming world_size 1."""
    if torch.distributed.is_initialized():
        return
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault(
        "MASTER_PORT", str(20000 + int(os.getenv("SLURM_JOB_ID", "0")) % 40000))
    torch.distributed.init_process_group(
        backend="nccl", rank=0, world_size=1, device_id=torch.device("cuda", 0))


def micro_step(model, x, y, eos_id, n_chunks, num_steps, carry_grad_chunks,
               accum_vecs, accumulation_steps, amp_args, use_memory,
               write_once=True):
    """Mirror of train.py's `cortex_fwd_bwd` (1701-1785), minus DDP and the
    all-masked guard (synthetic rows carry no -100 labels).

    `random_segments` is off and `accum_on` is True for the prefix accum buffer,
    both as B2 ran them.  The autocast placement matters: it wraps the forward
    only, exactly as at train.py:1755, so the loss reduction and the backward
    run outside it.
    """
    x_chunks = [c.contiguous() for c in torch.chunk(x, n_chunks, dim=1)]
    y_chunks = [c.contiguous() for c in torch.chunk(y, n_chunks, dim=1)]
    m_cross, chunk_losses, chunk_tokens = None, [], []
    for gi, (xc, yc) in enumerate(zip(x_chunks, y_chunks)):
        # train.py's `accum_on` dispatch.  It is a MEMORY fact here, not just a
        # correctness one: a slice detach frees the older chunks' graphs one
        # block at a time, while a gated buffer's whole-state detach keeps every
        # chunk since the last detach alive at once.  Pricing the gated arm with
        # the accum branch would under-count the peak by most of a lap.
        if carry_grad_chunks > 0 and m_cross is not None:
            if write_once:
                m_cross = detach_old_vecs(m_cross, accum_vecs, carry_grad_chunks)
            elif gi % carry_grad_chunks == 0:
                m_cross = m_cross.detach()
        with torch.autocast(**amp_args):
            out = model(xc, labels=yc, num_steps=num_steps,
                        m_cross_in=m_cross, return_m_cross=True,
                        eos_mask=(xc == eos_id),
                        output_details={"return_logits": False,
                                        "return_latents": False,
                                        "return_head": False,
                                        "return_stats": False})
        m_cross = out.get("m_cross")
        if use_memory and m_cross is None:
            raise RuntimeError(
                "forward returned no m_cross with return_m_cross=True and the "
                "prefix buffer active — the checkpoint's copy of "
                "raven_modeling_minimal_cortex.py predates the prefix rewrite. "
                "Re-run tools/prepare_cortex_checkpoint.py.  (Same silent "
                "failure smoke_prefix_real.py exists to catch; here a memory run "
                "that secretly has no memory would measure the CONTROL's "
                "footprint and green-light a geometry that cannot fit.)")
        chunk_losses.append(out["loss"])
        chunk_tokens.append(int((yc != -100).sum()))
    total = reduce_chunk_losses(chunk_losses, chunk_tokens, mode="token")
    (total / accumulation_steps).backward()
    return float(total.detach()), m_cross


def fmt_gib(n_bytes: int) -> str:
    return f"{n_bytes / GIB:.2f} GiB"


def main() -> int:
    args = parse_args()
    use_memory = args.use_memory == "true"
    with_optimizer = args.with_optimizer == "true"

    if not args.device.startswith("cuda") or not torch.cuda.is_available():
        print("This gate measures CUDA peak memory; run it on a GPU node.")
        return 1
    if args.max_length % args.cross_chunks:
        print(f"max_length {args.max_length} is not divisible by cross_chunks "
              f"{args.cross_chunks}: torch.chunk would emit uneven chunks and "
              f"the geometry would not be the one you think you measured.")
        return 1

    torch.manual_seed(0)
    torch.set_float32_matmul_precision("high")          # train.py:1991
    total_mem = torch.cuda.get_device_properties(0).total_memory
    dev_name = torch.cuda.get_device_properties(0).name
    chunk_len = args.max_length // args.cross_chunks
    # Resolve the two mechanism parameters that scale WITH cross_chunks in
    # b2_retrofit.sbatch.  Both are memory parameters, and defaulting either to
    # B2's literal number instead of B2's rule would measure a geometry nobody
    # is going to run.
    if args.accum_max <= 0:
        args.accum_max = args.cross_chunks * args.accum_vecs
    if args.carry_grad_chunks <= 0:
        args.carry_grad_chunks = max(1, args.cross_chunks // 2)

    print(f"=== geometry gate: {args.max_length} tok x cc{args.cross_chunks} "
          f"({chunk_len}-tok chunks) | memory={use_memory} "
          f"| optimizer={with_optimizer} ===")
    print(f"  device: {dev_name}, {fmt_gib(total_mem)} total")

    t0 = time.time()
    model, cfg = build_model(args, use_memory)
    print(f"  loaded fp32 in {time.time() - t0:.1f}s "
          f"({sum(p.numel() for p in model.parameters()) / 1e9:.2f}B params)")

    cortex = getattr(model, "cortex", None)
    if use_memory:
        if cortex is None:
            print("  FAIL: use_memory true but model.cortex is None — the graft "
                  "did not load, and this would measure the control.")
            return 1
        eos_id = cortex.summary_init_token
        n_vec = cortex.prefix.n_vec
        gated = args.prefix_memory == "gated"
        n_slots = int(getattr(cortex.prefix, "n_slots", n_vec))
    else:
        gated, n_slots = False, 0
        # The negative guard, mirroring the positive one: a control that
        # secretly has memory and a memory run that secretly does not both look
        # like a healthy loss curve (recurring bug class 2).
        if cortex is not None:
            print("  FAIL: use_memory false but model.cortex is not None.")
            return 1
        # Inert on this path: with no cortex the model ignores eos_mask
        # entirely.  Note the value is Huginn's 65505 (' creek') and not OLMo's
        # real 100257 — the retrofit configs carry the wrong special-token ids
        # (recipe_sweep_findings §4).  It costs nothing here; it is not a licence
        # to copy this line anywhere the mask is read.
        eos_id = getattr(cfg, "eos_token_id", 100257)
        n_vec = 0

    depth = args.backprop_depth or int(getattr(cfg, "mean_backprop_depth", 8))
    n_ng, k_wg = worst_case_num_steps(args.mean_recurrence, depth)
    num_steps = torch.tensor([n_ng, k_wg], device=model.device)
    if not use_memory:
        peak_carry = 0
    elif gated:
        peak_carry = gated_carry_rows(args.cross_chunks - 1, n_vec, n_slots,
                                      args.gate_fill)
    else:
        peak_carry = carry_rows(args.cross_chunks - 1, n_vec, args.accum_max)
    src = "--backprop_depth" if args.backprop_depth else "the checkpoint config"
    print(f"  recurrence: mean {args.mean_recurrence}, backprop depth {depth} "
          f"(from {src}) -> worst-case num_steps "
          f"[{n_ng} no-grad, {k_wg} with-grad]")
    if gated:
        print(f"  buffer: GATED ring, W {n_vec}, K {n_slots}, "
              f"route {args.gate_route}, norm {args.gate_norm}, "
              f"init {args.gate_init}, fill {args.gate_fill}, "
              f"carry_grad_chunks {args.carry_grad_chunks} of "
              f"{args.cross_chunks}")
    else:
        print(f"  buffer: accum_vecs {n_vec}, accum_max {args.accum_max}, "
              f"carry_grad_chunks {args.carry_grad_chunks} of "
              f"{args.cross_chunks}")
    print(f"  max packed sequence: {chunk_len} + {peak_carry} carry rows = "
          f"{chunk_len + peak_carry} "
          f"(max_position_embeddings={getattr(cfg, 'max_position_embeddings', '?')})")
    if use_memory and gated:
        lap = n_slots // max(n_vec, 1)
        # The same bound train.py asserts and smoke_prefix_real.py refuses on.
        # Priced here too, because a geometry whose gate never fires has the
        # WRONG FOOTPRINT as well as the wrong mechanism: it never reaches the
        # whole-state-detach regime, so the number this script prints would not
        # be the number the run faces.
        if args.cross_chunks < 2 * lap:
            print(f"  FAIL: cross_chunks {args.cross_chunks} < 2 laps "
                  f"({2 * lap} = 2 x K/W).  The gate never fires, so this would "
                  f"price an append buffer with dead gate parameters.")
            return 1
        print(f"  note: lap is {lap} chunks; the gate fires from chunk {lap} on, "
              f"and the read block is PINNED at {n_slots} columns from there. "
              f"Against a no-trim accum on the same chain "
              f"({args.cross_chunks * n_vec} columns) this is "
              f"{100 * n_slots / max(args.cross_chunks * n_vec, 1):.0f}% of the "
              f"read cost -- the pair is 'cheaper AND?', never 'equal cost AND?'.")
        print(f"  note: +{sum(p.numel() for p in cortex.prefix.parameters()) / 1e6:.1f}M "
              f"buffer parameters, which the optimizer holds two moments for. "
              f"That is the one axis on which the gated arm is the LARGER of "
              f"the two, and the reason this gate is not a formality.")
    elif use_memory:
        if args.cross_chunks * n_vec > args.accum_max:
            print(f"  note: the FIFO trim FIRES from chunk "
                  f"{args.accum_max // max(n_vec, 1)} on — eviction is "
                  f"in-distribution, which it never was in B2, and this is the "
                  f"SMALLER of the two footprints.")
        else:
            print(f"  note: the FIFO never trims at this setting "
                  f"({args.cross_chunks} x {n_vec} <= accum_max "
                  f"{args.accum_max}) — B2's convention, the larger footprint, "
                  f"and the reason eviction was out-of-distribution at eval. "
                  f"Pass --accum_max 128 to price the capped alternative.")

    optimizer = None
    if with_optimizer:
        from muon import MuonWithAuxAdam
        init_trivial_process_group()
        groups = build_param_groups(
            model, args.muon_lr, args.adam_lr,
            args.memory_lr if use_memory else 0.0)
        shape = ", ".join(
            f"{len(g['params'])}{'m' if g['use_muon'] else 'a'}" for g in groups)
        optimizer = MuonWithAuxAdam(groups)
        print(f"  optimizer: MuonWithAuxAdam, {len(groups)} groups ({shape})")

    # amp_args as train.py:305-314 builds them under `--no_amp false`,
    # `--compile false`: autocast ON, cache OFF.
    amp_args = {"device_type": "cuda", "dtype": torch.bfloat16,
                "enabled": True, "cache_enabled": False}

    B, S = args.micro_batch_size, args.max_length
    ids = torch.randint(0, cfg.vocab_size, (B, S + 1), device=model.device)
    for gi in range(1, args.cross_chunks):          # one doc boundary per chunk
        ids[:, gi * chunk_len + chunk_len // (gi + 1)] = eos_id
    x, y = ids[:, :-1], ids[:, 1:]

    def one(tag):
        t = time.time()
        loss, _ = micro_step(model, x, y, eos_id, args.cross_chunks, num_steps,
                             args.carry_grad_chunks, n_vec, args.batch_size,
                             amp_args, use_memory, write_once=not gated)
        if optimizer is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        dt = time.time() - t
        print(f"  {tag}: loss {loss:.4f}  {dt:.1f}s  "
              f"peak {fmt_gib(torch.cuda.max_memory_allocated())}", flush=True)
        return dt

    status, peak_alloc, peak_res, step_s = "FITS", 0, 0, 0.0
    try:
        for i in range(args.warmup_steps):
            one(f"warmup {i + 1}/{args.warmup_steps}")
        torch.cuda.reset_peak_memory_stats()
        step_s = one("measured")
        peak_alloc = torch.cuda.max_memory_allocated()
        peak_res = torch.cuda.max_memory_reserved()
    except RuntimeError as e:
        # Catch the allocator's OOM and nothing else: the m_cross guard above
        # raises RuntimeError too, and swallowing THAT would report a silently
        # memory-less run as a clean fit.  OutOfMemoryError is a RuntimeError
        # subclass, so one `except` with an explicit test covers both the typed
        # and the string-only forms across torch versions.
        oom = isinstance(e, getattr(torch.cuda, "OutOfMemoryError", ()))
        if not (oom or "out of memory" in str(e).lower()):
            raise
        status = "OOM"
        peak_res = torch.cuda.max_memory_reserved()
        print(f"  OOM: {str(e).splitlines()[0]}")

    print("\n-- result --")
    if status == "FITS":
        head = total_mem - peak_res
        # One optimizer step is batch_size micro-steps, but step_s already
        # includes an optimizer step, so this OVER-counts by (batch_size - 1)
        # optimizer steps.  Stated as an upper bound rather than silently.
        opt_step_s = step_s * args.batch_size
        days = opt_step_s * args.max_steps / 86400
        print(f"  peak allocated : {fmt_gib(peak_alloc)}")
        print(f"  peak reserved  : {fmt_gib(peak_res)}  "
              f"({100 * peak_res / total_mem:.0f}% of {fmt_gib(total_mem)}, "
              f"{fmt_gib(head)} headroom)")
        print(f"  micro-step     : {step_s:.1f}s  -> {opt_step_s / 60:.1f} min "
              f"per optimizer step at batch_size {args.batch_size} (upper "
              f"bound: the optimizer step is counted once per micro-step)")
        print(f"  projection     : {days:.1f} days for {args.max_steps:,} steps "
              f"= {days / 2:.1f} links of 48h")
    else:
        print(f"  peak reserved before OOM: {fmt_gib(peak_res)} of "
              f"{fmt_gib(total_mem)}")
    print(f"\nRESULT {status} max_length={args.max_length} "
          f"cross_chunks={args.cross_chunks} chunk_len={chunk_len} "
          f"memory={use_memory} optimizer={with_optimizer} "
          f"peak_alloc_gib={peak_alloc / GIB:.2f} "
          f"peak_res_gib={peak_res / GIB:.2f} step_s={step_s:.1f}")

    if args.json_out:
        with open(args.json_out, "a") as f:
            f.write(json.dumps({
                "status": status, "device": dev_name,
                "max_length": args.max_length, "cross_chunks": args.cross_chunks,
                "chunk_len": chunk_len, "micro_batch_size": args.micro_batch_size,
                "use_memory": use_memory, "with_optimizer": with_optimizer,
                "mean_recurrence": args.mean_recurrence, "backprop_depth": depth,
                "num_steps": [n_ng, k_wg], "carry_rows": peak_carry,
                "accum_vecs": n_vec, "accum_max": args.accum_max,
                "buffer": ("gated" if gated else "accum") if use_memory else "none",
                "gate_slots": n_slots if gated else 0,
                "gate_route": args.gate_route if gated else "",
                "gate_fill": args.gate_fill if gated else "",
                "latent_carry": bool(args.latent_carry),
                "carry_grad_chunks": args.carry_grad_chunks,
                "peak_alloc_gib": round(peak_alloc / GIB, 2),
                "peak_res_gib": round(peak_res / GIB, 2),
                "total_gib": round(total_mem / GIB, 2),
                "step_s": round(step_s, 2),
            }) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
