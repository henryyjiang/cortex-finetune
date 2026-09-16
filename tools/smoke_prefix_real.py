"""
One-step smoke of the PREFIX memory path on the REAL 1B checkpoint.

Closes a standing open thread: prefix memory has only ever run on toy-size
weights (n_embd=64, S=32, max_position_embeddings=128).  The 2026-08-04
prefix_pos change makes that gap sharper — the tail layout puts summary slots at
S+1..S+n_vec, i.e. position 1056 at S=1024, which no toy config ever reached.

Runs train.py's cortex_fwd_bwd chain (chunk, carry un-detached, detach_old_vecs,
one backward) against the real weights and asserts the things that are SILENT
when broken:

  * the graft actually loaded          — a failed `import cortex_graft` leaves
                                         cortex=None and the run becomes a
                                         no-memory baseline with a healthy curve
  * prefix_pos / prefix_eos_reset      — the 2026-08-04 defaults are live
  * summary_emb seeded from wte[eos]   — not noise, not left at post_init random
  * the carry ACCUMULATES              — 32 -> 64 -> 96 rows
  * the carry SURVIVES a doc boundary  — Fix A, on real weights: chunk g+1's
                                         state must still contain chunk g's rows
                                         verbatim even when the chunk holds EOS
  * summary_emb receives gradient      — the write path is on the loss
  * loop params receive gradient       — freeze_loop=false is in effect
  * everything is finite               — no NaN from the untrained arrangement

THE GATED GEOMETRY (--prefix_memory gated, 2026-09-16, P1.0).
A3' runs a sparse ring at W=16 / K=64, and NONE of the accum checks above
describe it: the carry stops growing at K, rows are overwritten in place, and
"preserved" is SUPPOSED to go False.  Running the accum smoke and calling the
gated arm covered would be the same mistake as the config-vs-modeling-file one
this script exists for, so the gated path asserts its OWN invariants — every one
of them a property of the packed forward, not of the config:

  * the carry GROWS then STOPS         — W per chunk up to K, then fixed forever
                                         (fill="grow": lap 1 IS PrefixAccumBuffer)
  * the gate actually FIRES            — the chain must be >= 2 laps, or the run
                                         trains 16.8M dead parameters behind a
                                         healthy loss curve.  Asserted, not hoped.
  * the ring writes the RIGHT rows     — after the first lap, exactly the W rows
                                         at (chunk*W + 0..W) mod K change and the
                                         other K-W come back BIT-IDENTICAL.  That
                                         is the sparse ring's defining property,
                                         and the depth->row map rests on it.
  * depth_rows(j) matches the write    — rows congruent to j (mod W), so the
                                         depth-slice ablation still addresses.
  * the gate receives gradient         — gate_proj_mem only trains through a
                                         chain of >= 3 chunks; a zero grad here
                                         means carry_grad_chunks is too short.

Login node (no GPU): defaults are small (S=256/chunk, T=2) so this finishes in
a couple of minutes on CPU.  It exercises the code path, not the model quality.

    module load anaconda3 && conda activate cortex-retro
    export HF_HOME=$SCRATCH/hf_cache HF_HUB_OFFLINE=1
    python tools/smoke_prefix_real.py --model_name ckpts/olmo-retrofit-cortex

Full training geometry (needs a GPU; submit it or grab an interactive node):
    python tools/smoke_prefix_real.py --model_name ckpts/olmo-retrofit-cortex \
        --full --device cuda --dtype bfloat16

A3's geometry.  The gate needs two laps, so cross_chunks 8 is the minimum
at K/W=4 and the script refuses a shorter chain:
    python tools/smoke_prefix_real.py --model_name ckpts/olmo-retrofit-cortex \
        --prefix_memory gated --accum_vecs 16 --gate_slots 64 --cross_chunks 8

Run from the repo root so the grafted modeling file's `import cortex_graft`
resolves.  Exit code is 1 on any failed check.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from cortex_graft import reset_cortex_graft_init  # noqa: E402
from cortex_memory.buffers import PrefixGatedBuffer  # noqa: E402
from cortex_memory.chunking import detach_old_vecs  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Real-checkpoint prefix-memory smoke")
    p.add_argument("--model_name", default="ckpts/olmo-retrofit-cortex",
                   help="graft-prepared checkpoint dir (B2's base)")
    p.add_argument("--chunk_len", type=int, default=256, help="tokens per chunk")
    p.add_argument("--cross_chunks", type=int, default=4)
    p.add_argument("--T", type=int, default=2, help="recurrence for the smoke")
    p.add_argument("--accum_vecs", type=int, default=32)
    p.add_argument("--accum_max", type=int, default=128)
    p.add_argument("--carry_grad_chunks", type=int, default=2)
    p.add_argument("--prefix_memory", default="accum", choices=["accum", "gated"])
    p.add_argument("--gate_slots", type=int, default=0,
                   help="K, carried columns (gated only).  0 = same as "
                        "--accum_vecs, the pre-P1.0 shape.  A3' runs 64.")
    p.add_argument("--gate_route", default="ring", choices=["ring", "mix"])
    p.add_argument("--gate_norm", default="tanh", choices=["tanh", "rms", "none"])
    p.add_argument("--gate_init", default="zero", choices=["zero", "default"])
    p.add_argument("--gate_fill", default="grow", choices=["grow", "init"])
    p.add_argument("--latent_carry", action="store_true",
                   help="the Z channel: carry the recurrence trajectory as well "
                        "as the token-space summary, and read it by "
                        "substituting into s0.  Adds the modeling-file checks "
                        "for the two loop hooks it needs.")
    p.add_argument("--latent_depth_rule", default="absolute",
                   choices=["absolute", "relative"])
    p.add_argument("--latent_renorm", default="none", choices=["none", "s0"])
    p.add_argument("--device", default="cpu")
    p.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    p.add_argument("--full", action="store_true",
                   help="B2's real geometry: chunk_len 1024, T 8 (needs a GPU)")
    p.add_argument("--quick", action="store_true",
                   help="chunk_len 64 — every check still fires, ~4x less compute "
                        "than the default; use this on a CPU login node")
    return p.parse_args()


def cortex_weights_are_fresh(loading_info: dict) -> bool:
    """Did this checkpoint supply its own cortex weights?

    `missing_keys` is HF's list of parameters the checkpoint did NOT provide, so
    a cortex key in it means the graft was built fresh and the designed init
    must be applied.  A cortex key ABSENT from it means the checkpoint carried
    trained weights, and applying the reset would wipe the very buffer the smoke
    was pointed at -- then report, truthfully and uselessly, that a freshly
    initialised buffer behaves correctly.

    Asked of the loader rather than inferred from the directory name or from
    `summary_seeded` (which post_init can clobber, and which is the thing the
    reset itself clears).
    """
    return any(k.startswith("cortex.")
               for k in (loading_info or {}).get("missing_keys", []))


def build_model(args):
    from transformers import AutoConfig, AutoModelForCausalLM

    cfg = AutoConfig.from_pretrained(args.model_name, trust_remote_code=True)
    for k, v in (("use_memory", True), ("memory_slots", 0), ("memory_slots_iter", 0),
                 ("prefix_memory", args.prefix_memory),
                 ("accum_vecs", args.accum_vecs), ("accum_max", args.accum_max),
                 ("gate_slots", args.gate_slots), ("gate_route", args.gate_route),
                 ("gate_norm", args.gate_norm), ("gate_init", args.gate_init),
                 ("gate_fill", args.gate_fill),
                 ("latent_carry", args.latent_carry),
                 ("latent_depth_rule", args.latent_depth_rule),
                 ("latent_renorm", args.latent_renorm),
                 ("prefix_pos", "tail"), ("prefix_eos_reset", False)):
        setattr(cfg, k, v)
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    model, info = AutoModelForCausalLM.from_pretrained(
        args.model_name, trust_remote_code=True, config=cfg, torch_dtype=dtype,
        output_loading_info=True)
    model = model.to(args.device).train()
    # DID THIS CHECKPOINT CARRY TRAINED CORTEX WEIGHTS?
    #
    # The reset below is correct for a graft-prepared BASE, whose config.json
    # enables memory but whose weights contain no cortex tensors at all.  On a
    # TRAINED checkpoint (anything through tools/prepare_eval_checkpoint.py) it
    # would be destructive: it would wipe the buffer the smoke was pointed at
    # and then report, truthfully and uselessly, that a freshly initialised
    # buffer behaves correctly.
    #
    # `missing_keys` is the exact signal -- HF lists every parameter the
    # checkpoint did NOT supply -- so this asks the loader rather than guessing
    # from the directory name or from `summary_seeded` (which post_init can
    # clobber).
    fresh = cortex_weights_are_fresh(info)
    # APPLY THE DESIGNED INIT, exactly as train.py does on a fresh run.
    #
    # Without this the smoke measures a model the run will never build.  The
    # checkpoint carries no cortex tensors, so HF reports them "newly
    # initialized" and post_init hands the graft the raven DEPTH-SCALED scheme,
    # which has no valid layer index for these modules.  Measured on the tiny
    # fixture: forget_bias = -2.2e12, i.e. fg identically ZERO -- a gate that
    # forgets everything on every write, and a forget_bias whose gradient
    # underflows to exactly 0.  Every structural check (ring rows, shapes,
    # depth map) still PASSED against it, which is precisely why this was worth
    # finding here: the smoke was green on a buffer with the wrong semantics.
    #
    # A resumed or trained checkpoint carries real cortex weights and must NOT
    # be reset; this script always loads a graft-prepared BASE, which never
    # does, so the reset is unconditional here and guarded on `resume_path is
    # None` in train.py.
    if fresh:
        reset_cortex_graft_init(model, log=print)
    else:
        print("[cortex] checkpoint supplied its own cortex weights -- NOT "
              "resetting.\n         The designed-init check below is skipped; "
              "this is a trained buffer,\n         and what it holds is the "
              "thing under test.")
    model._cortex_was_fresh = fresh
    return model, cfg


def _changed_rows(new: torch.Tensor, prev: torch.Tensor) -> list[int]:
    """Row indices of `prev` that `new` did not return verbatim.

    Compared in float32 on ROW NORMS, relatively: a carried row is a post-`ln_f`
    state with norm ~171, so an absolute atol that suits a unit-scale tensor
    would call every row unchanged.  A row the sparse ring did not touch is a
    literal pass-through (index_copy is out-of-place), so the honest expectation
    for it is bit-identical and any real tolerance is generous; a row the gate
    DID touch moves by ~50% of its norm at gate_init="zero".  The two cases are
    four orders of magnitude apart, so this is not a judgement call.
    """
    n = min(new.shape[1], prev.shape[1])
    a = new[:, :n].detach().float()                       # [B, n, D]
    b = prev[:, :n].detach().float()
    rel = (a - b).norm(dim=-1) / b.norm(dim=-1).clamp_min(1e-6)   # [B, n]
    return [int(i) for i in torch.nonzero(rel.amax(dim=0) > 1e-3).flatten()]


def run_chain(model, x, y, eos_id, n_chunks, num_steps, carry_grad_chunks,
              accum_vecs, device, verbose=False, write_once=True):
    """train.py's cortex_fwd_bwd, minus DDP/autocast/L2-SP.  Returns a report.

    verbose prints per-chunk progress: on a CPU login node one chunk of the 1B
    model takes minutes, and a silent run is indistinguishable from a hang.

    `preserved` and `changed` are two views of the same comparison and BOTH are
    reported, because the two buffers make opposite predictions about it: accum
    says every overlapping row comes back verbatim (preserved True, changed
    empty), while a gated ring past its first lap says exactly W named rows moved
    and the rest did not.  A single boolean cannot express the second one, and a
    gated arm checked with the accum assertion would fail for the right reason
    with the wrong message.
    """
    x_chunks = [c.contiguous() for c in torch.chunk(x, n_chunks, dim=1)]
    y_chunks = [c.contiguous() for c in torch.chunk(y, n_chunks, dim=1)]

    m_cross, losses, shapes, preserved, changed = None, [], [], [], []
    for gi, (xc, yc) in enumerate(zip(x_chunks, y_chunks)):
        t0 = time.time()
        if verbose:
            print(f"  chunk {gi + 1}/{n_chunks} ({xc.shape[1]} tok, carry "
                  f"{0 if m_cross is None else m_cross.shape[1]}) ...",
                  end="", flush=True)
        prev = None if m_cross is None else m_cross.detach().clone()
        # The stop-gradient horizon, EXACTLY as train.py's cortex_fwd_bwd
        # dispatches it -- a slice detach for write-once rows, a whole-state
        # detach for a merge that overwrites them.  Running the accum branch on
        # a gated buffer would detach rows by age they do not have, and
        # `gate_proj_mem` (which only trains through a chain of >= 3 chunks)
        # would quietly get a different graph here than in training.
        if carry_grad_chunks > 0 and m_cross is not None:
            if write_once:
                m_cross = detach_old_vecs(m_cross, accum_vecs, carry_grad_chunks)
            elif gi % carry_grad_chunks == 0:
                m_cross = m_cross.detach()
        out = model(xc.to(device), labels=yc.to(device), num_steps=num_steps,
                    m_cross_in=m_cross, return_m_cross=True,
                    eos_mask=(xc == eos_id).to(device))
        # .get, not ["m_cross"]: ModelOutput drops None-valued fields, so
        # bracket-indexing raises KeyError instead of returning None (train.py
        # documents the same trap).  None here means the forward never ran the
        # prefix splice -- see the modeling-file check in main().
        m_cross = out.get("m_cross")
        if m_cross is None:
            raise RuntimeError(
                "forward returned no m_cross with return_m_cross=True and a "
                "prefix buffer active.  The checkpoint dir's COPY of "
                "raven_modeling_minimal_cortex.py predates the prefix rewrite, "
                "so prefix_pack/prefix_unpack never ran and this model has NO "
                "cross-segment memory.  Re-run tools/prepare_cortex_checkpoint.py "
                "against the base.")
        losses.append(out["loss"])
        shapes.append(tuple(m_cross.shape))
        # Fix A on real weights: the rows written by earlier chunks must come
        # back verbatim, even from a chunk that contained a document boundary.
        if prev is not None:
            rows = _changed_rows(m_cross, prev)
            changed.append(rows)
            preserved.append(not rows)
        if verbose:
            print(f" loss {float(losses[-1].detach()):.4f}  "
                  f"[{time.time() - t0:.1f}s]", flush=True)
    total = torch.stack(losses).mean()
    if verbose:
        print("  backward (one pass over the whole chain) ...", end="", flush=True)
    t0 = time.time()
    total.backward()
    if verbose:
        print(f" [{time.time() - t0:.1f}s]", flush=True)
    return {"losses": [float(l.detach()) for l in losses], "shapes": shapes,
            "preserved": preserved, "changed": changed,
            "total": float(total.detach())}


def main() -> int:
    args = parse_args()
    if args.full:
        args.chunk_len, args.T = 1024, 8
    elif args.quick:
        args.chunk_len = 64
    torch.manual_seed(0)
    if args.device == "cpu":
        # ~14 distinct layers applied 4 + 6*T + 4 times over the packed sequence,
        # so this is minutes per chunk at 1B on a shared login core.  Say so
        # before the first long silence rather than after it.
        print(f"note: CPU run, expect ~{max(1, args.chunk_len // 64)}-"
              f"{max(2, args.chunk_len // 24)} min total at chunk_len="
              f"{args.chunk_len}, T={args.T}.  --quick is ~4x faster; "
              f"--device cuda on an interactive GPU node is ~100x.")

    print(f"loading {args.model_name} ({args.dtype}, {args.device}) ...")
    t0 = time.time()
    model, cfg = build_model(args)
    print(f"  loaded in {time.time() - t0:.1f}s")

    ok = True

    def check(label, cond, detail=""):
        nonlocal ok
        ok &= bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {label}" + (f"  {detail}" if detail else ""))

    cortex = getattr(model, "cortex", None)
    print("\n-- graft --")
    check("cortex built (not a silent no-memory run)", cortex is not None)
    if cortex is None:
        print("\nFAILED: run from the repo root so `import cortex_graft` resolves.")
        return 1
    check("prefix buffer active", cortex.prefix is not None,
          f"{type(cortex.prefix).__name__}")
    # The designed init actually took.  post_init leaves these at ~1e12 on a
    # base checkpoint (they are "newly initialized" keys with no layer index),
    # and a buffer running at forget_bias = -2.2e12 has fg identically zero --
    # structurally correct, semantically nothing like the run.
    bad = [n for n, p in cortex.named_parameters()
           if not torch.isfinite(p).all() or float(p.detach().abs().max()) > 1e4]
    if getattr(model, "_cortex_was_fresh", True):
        check("cortex parameters are at their designed init (not post_init's)",
              not bad, f"out of range: {bad[:4]}" if bad else "")
    else:
        # A trained buffer is allowed any finite value; only non-finite is a
        # failure, and the 1e4 bound would be a false alarm on it.
        nonfinite = [n for n, p in cortex.named_parameters()
                     if not torch.isfinite(p).all()]
        check("trained cortex parameters are finite", not nonfinite,
              f"{nonfinite[:4]}" if nonfinite else "")
    check("prefix_pos == 'tail' (2026-08-04)", cortex.prefix_pos == "tail",
          f"got {cortex.prefix_pos!r}")
    check("prefix_eos_reset is off (2026-08-04)", cortex.prefix_eos_reset is False,
          f"got {cortex.prefix_eos_reset!r}")

    # The checks above read cortex_graft.py, which is imported LIVE from the repo
    # root -- they pass whenever the graft is current, regardless of the model.
    # But raven_modeling_minimal_cortex.py is COPIED into each checkpoint dir by
    # prepare_cortex_checkpoint.py, so a dir prepared before the 2026-08-02
    # prefix rewrite carries a forward() that never calls prefix_pack.  Then
    # cortex.prefix exists, every flag check passes, the loss curve looks
    # healthy, and the run has NO cross-segment memory: recurring bug class 2.
    # Nothing else in this script could tell the difference, so check the source
    # of the forward that actually got loaded.
    import inspect
    try:
        src = inspect.getsource(type(model).forward)
        loaded_from = inspect.getfile(type(model))
    except (OSError, TypeError):
        src, loaded_from = "", "<unavailable>"
    check("loaded modeling file splices the prefix",
          "prefix_pack" in src and "prefix_unpack" in src,
          os.path.basename(loaded_from))
    if args.latent_carry:
        # THE SAME BUG CLASS, ONE CHANNEL DOWN.  Z needs two hooks that live in
        # the modeling file and nowhere else: `latent_init` (the s0
        # substitution, in iterate_forward) and `iter_write(x, current_step)`
        # (the trajectory tape, in core_block_forward).  A checkpoint dir
        # carrying a modeling-file SNAPSHOT from before those hooks would build
        # a dual-channel buffer, allocate its second gate, pass every config
        # check -- and carry a Z half that is never written and never read.  The
        # loss curve would be perfectly healthy.  cortex_graft.py raises on an
        # empty tape at the first merge, which catches it, but this names it
        # before the forward runs and says what to do about it.
        try:
            loop_src = inspect.getsource(type(model).iterate_forward)
            core_src = inspect.getsource(type(model).core_block_forward)
        except (OSError, TypeError):
            loop_src = core_src = ""
        # The SPLIT must be passed too, not just the call.  An intermediate
        # snapshot that calls latent_init(x) without num_steps_no_grad leaves
        # latent_read_grad_frac at 0.0 -- which reads as "the Z read is dead"
        # when it actually means "nobody measured it".  Those are opposite
        # conclusions and the difference is one argument.
        check("loaded modeling file substitutes Z into s0 (latent_init)",
              "latent_init" in loop_src)
        check("...and passes the no-grad split, so the read can be measured",
              "latent_init(x, " in loop_src or "num_steps_no_grad)" in loop_src)
        check("loaded modeling file tapes the trajectory "
              "(iter_write with current_step)",
              "iter_write" in core_src and "current_step" in core_src)
        if ("latent_init" not in loop_src or "current_step" not in core_src
                or "num_steps_no_grad" not in loop_src):
            print(f"\n     {loaded_from}\n"
                  f"     predates the Z hooks.  Re-prepare it:\n"
                  f"       python tools/prepare_cortex_checkpoint.py "
                  f"--variant olmo \\\n"
                  f"           --src {args.model_name} --dst {args.model_name}")
            return 1
    if "prefix_pack" not in src:
        print(f"\n     {loaded_from}\n"
              f"     predates the prefix rewrite.  Re-prepare it:\n"
              f"       python tools/prepare_cortex_checkpoint.py --variant olmo \\\n"
              f"           --src {args.model_name} --dst {args.model_name}\n"
              f"     (--src == --dst is safe: it re-copies the modeling file and\n"
              f"      re-patches config.json in place.)")
        return 1

    eos_id = cortex.summary_init_token
    buf = cortex.prefix
    gated = isinstance(buf, PrefixGatedBuffer)
    n_vec = buf.n_vec
    S = args.chunk_len * args.cross_chunks
    print(f"\n-- geometry --\n  n_embd={cfg.n_embd} vocab={cfg.vocab_size} eos={eos_id}"
          f"\n  {args.cross_chunks} chunks x {args.chunk_len} tok, n_vec={n_vec}, T={args.T}"
          f"\n  max packed position = {args.chunk_len + n_vec} "
          f"(block_size={getattr(cfg, 'block_size', '?')}, "
          f"max_position_embeddings={getattr(cfg, 'max_position_embeddings', '?')})")
    if gated:
        geo = buf.geometry(args.chunk_len)
        print("  " + "  ".join(f"{k}={v}" for k, v in geo.items()))
        # THE CONSTRAINT NOTHING ELSE CHECKS.  The gate first acts on lap 2
        # (lap 1 is a plain append under fill="grow"), so a chain shorter than
        # two laps exercises an ACCUM buffer with the gate's parameters bolted
        # on and never fired.  Stop here rather than report a green smoke for a
        # geometry whose defining mechanism did not run.
        need = int(geo["min_cross_chunks"])
        if args.cross_chunks < need:
            print(f"\nFAILED: cross_chunks={args.cross_chunks} is shorter than "
                  f"two laps ({need} = 2 x {geo['lap_chunks']}), so the gate "
                  f"NEVER FIRES in this smoke.  Everything below would pass "
                  f"while measuring an append buffer.  Re-run with "
                  f"--cross_chunks {need} (train.py asserts the same bound).")
            return 1

    # Real-ish batch: random ids with EOS separators, so the document-boundary
    # path is exercised rather than skipped.
    ids = torch.randint(0, cfg.vocab_size, (1, S + 1))
    # One document boundary inside every chunk after the first, at a different
    # offset each time.  Derived from cross_chunks rather than hardcoded: fixed
    # indices assumed >= 3 chunks and went out of bounds at --cross_chunks 2.
    for gi in range(1, args.cross_chunks):
        off = args.chunk_len // (gi + 1)
        ids[0, gi * args.chunk_len + off] = eos_id
    x, y = ids[:, :-1], ids[:, 1:]

    num_steps = torch.tensor([args.T // 2, args.T - args.T // 2])
    print(f"\n-- forward/backward chain (num_steps={num_steps.tolist()}) --")
    t0 = time.time()
    rep = run_chain(model, x, y, eos_id, args.cross_chunks, num_steps,
                    args.carry_grad_chunks, n_vec, args.device, verbose=True,
                    write_once=not gated)
    print(f"  {time.time() - t0:.1f}s")

    print("\n-- checks --")
    check("summary_emb seeded on first forward",
          bool(cortex.prefix.summary_seeded))
    wte = model.transformer.wte.weight
    if getattr(model, "_cortex_was_fresh", True):
        check("summary_emb == wte[eos] (AutoCompressor init)",
              torch.allclose(cortex.prefix.summary_emb.detach().float(),
                             wte[eos_id].detach().float().unsqueeze(0).expand(n_vec, -1),
                             atol=1e-3))
    else:
        # On a trained checkpoint the rows SHOULD have moved off the seed --
        # asserting they still match it would be asserting the run did nothing.
        seed = wte[eos_id].detach().float().unsqueeze(0).expand(n_vec, -1)
        moved = float((cortex.prefix.summary_emb.detach().float() - seed).norm()
                      / seed.norm().clamp_min(1e-12))
        check("summary_emb has moved off the wte[eos] seed (it trained)",
              moved > 1e-3, f"relative distance {moved:.3f}")
    if not gated:
        row_width = cfg.n_embd * (2 if args.latent_carry else 1)
        want = [(1, (g + 1) * n_vec, row_width)
                for g in range(args.cross_chunks)]
        check("carry accumulates one write per chunk", rep["shapes"] == want,
              f"{rep['shapes']}")
        check("carry survives document boundaries (Fix A)",
              all(rep["preserved"]), f"{rep['preserved']}")
    else:
        K, W = buf.n_slots, n_vec
        # 1. GROW THEN STOP.  W rows per chunk until the ring is full, then the
        #    read block is fixed forever -- the property that makes the gated
        #    arm the cheaper one, and the first thing to break if an
        #    accum-shaped carry ever reaches a gated buffer.
        row_width = cfg.n_embd * (2 if args.latent_carry else 1)
        want = [(1, min((g + 1) * W, K), row_width)
                for g in range(args.cross_chunks)]
        check("carry grows by W per chunk and stops at K",
              rep["shapes"] == want, f"{rep['shapes']}")

        # 2. THE RING WRITES THE RIGHT ROWS.  At chunk g the buffer's cursor is
        #    g, so the gated rows are (g*W .. g*W+W) mod K and every other row
        #    must come back as a literal pass-through.  Computed here from the
        #    RULE rather than read back from the buffer, so a cursor that
        #    drifted (it is deliberately non-persistent) is CAUGHT and not
        #    merely described.
        expect, got = [], rep["changed"]
        for g in range(1, args.cross_chunks):
            if g * W < K:
                expect.append([])                       # still filling: append
            else:
                expect.append([((g * W) % K + i) % K for i in range(W)])
        ring_ok = [sorted(c) for c in got] == expect
        fired = sum(1 for c in got if c)
        check("ring gates exactly its own W rows, passes the other K-W through",
              ring_ok,
              f"changed={got}" if not ring_ok else f"{fired} gated chunks")

        # 3. THE GATE FIRED.  Implied by (2) when (2) passes, and the line to
        #    read first when it does not: a silently-appending gated run is the
        #    exact failure this section exists to prevent.
        check("the gate actually fired at least once", fired > 0,
              f"{fired} of {args.cross_chunks - 1} carried chunks")

        # 4. DEPTH -> ROW MAP.  Gating kills CHUNK separability, not DEPTH
        #    separability; the depth-slice ablation that picks Z's write depths
        #    addresses rows through this map, so it is load-bearing and not a
        #    convenience.
        dr_ok = all(buf.depth_rows(j) == [r for r in range(K) if r % W == j]
                    for j in range(W))
        check("depth_rows(j) == rows congruent to j (mod W)", dr_ok)

        # 5. THE GATE IS ON THE LOSS.  gate_proj_mem sees gradient only through
        #    a chain of >= 3 chunks, so a zero here means carry_grad_chunks (or
        #    the chain) is too short and the forget gate would never train --
        #    16.8M parameters sitting behind a healthy loss curve.
        for name in ("gate_proj_in", "gate_proj_mem"):
            gg = getattr(buf, name).weight.grad
            check(f"{name} has gradient",
                  gg is not None and torch.isfinite(gg).all()
                  and float(gg.norm()) > 0,
                  f"|g|={float(gg.norm()):.3e}" if gg is not None else "None")
        fb = buf.forget_bias.grad
        check("forget_bias has gradient (the horizon is trainable)",
              fb is not None and torch.isfinite(fb).all() and float(fb.norm()) > 0,
              f"|g|={float(fb.norm()):.3e}" if fb is not None else "None")
    check("all chunk losses finite",
          all(l == l and abs(l) != float("inf") for l in rep["losses"]),
          " ".join(f"{l:.3f}" for l in rep["losses"]))

    if args.latent_carry:
        D = buf.hidden_size
        # 1. THE CARRY IS TWO CHANNELS WIDE, and the row count did NOT move.
        #    Z rides the same rows through the same ring pointer, so if the read
        #    block grew, Z stopped being free and the design's central claim is
        #    gone.
        check("carry is 2D wide and the read block is unchanged",
              all(sh[-1] == 2 * D for sh in rep["shapes"]),
              f"last dim {rep['shapes'][-1][-1]} vs 2 x {D}")
        # 2. THE WRITE IS ON THE LOSS.  `latent_states` in the modeling file is
        #    detached; if the tape were taken from it, Z would be a
        #    gradient-free carry behind a healthy loss curve.
        frac = cortex.latent_write_grad_frac
        check("the Z write is inside the gradient window",
              frac > 0.0,
              f"grad_frac {frac:.2f} of taped steps")
        if frac == 0.0:
            print("       ^ every taped step ran under no_grad, so Z is a FROZEN "
                  "FEATURE EXTRACTOR:\n         the model can learn to USE it, "
                  "never to SHAPE it.  Legitimate at high\n         recurrence, "
                  "but it must be a DECISION in the pre-registration.")
        if args.latent_carry and frac < 1.0:
            print(f"       note: {100 * (1 - frac):.0f}% of the write band is "
                  f"frozen at this num_steps split.")
        # 3. THE DEPTH MAP IS VALID FOR THE LOOP IT RAN IN.  T is sampled per
        #    batch in training, so a map that requests a depth the forward never
        #    reached would index past the tape.
        depths = cortex.latent_depth_map(len(cortex._z_tape), n_vec)
        check("every written depth exists in the loop",
              all(1 <= k <= len(cortex._z_tape) for k in depths),
              f"depths {depths} over {len(cortex._z_tape)} taped steps")
        # 4. THE READ GRADIENT, which is a DIFFERENT and sharper constraint
        #    than the write's.  A single no-grad step cuts Z's read gradient to
        #    exactly zero, because Z enters the loop once (at s0) and the no-grad
        #    iterations run first.  E is unaffected -- it re-enters through
        #    input_embeds on every iteration.  So this is reported, not asserted:
        #    n > 0 is the normal case in training and is not a failure.
        rfrac = cortex.latent_read_grad_frac
        if not cortex.latent_read_measured:
            # 0.0 here would be indistinguishable from a dead read, and the two
            # call for opposite actions.  Say UNMEASURED.
            print("  [INFO] Z read gradient: UNMEASURED (the loaded modeling "
                  "file does not pass\n         the no-grad split to "
                  "latent_init).  This is NOT a zero.")
            rfrac = 1.0          # do not suppress the wiring check on an unknown
        else:
            print(f"  [INFO] Z read gradient live on {100 * rfrac:.0f}% of "
                  f"forwards (num_steps=[{int(num_steps[0])}, "
                  f"{int(num_steps[1])}])")
        if cortex.latent_read_measured and rfrac < 1.0:
            print("         A SINGLE no-grad step zeroes it: Z enters at s0 and "
                  "the no-grad\n         iterations run FIRST, so everything "
                  "downstream is detached.  E re-enters\n         through "
                  "input_embeds every iteration and is unaffected.  In training "
                  "n=0\n         only when the sampled p <= "
                  "mean_backprop_depth, so this fraction is a\n         "
                  "TRAINING-RUN AVERAGE that belongs in the pre-registration.")
        # 5. BOTH GATES TRAIN -- but the Z gate can only be judged when the read
        #    was live on this split.  One projection reading E (norm ~171) and Z
        #    (~0.35) together would be dominated by E, so separate gates are the
        #    design and a dead Z gate is how that silently reverts.
        if gated and rfrac > 0:
            for name in ("gate_proj_in_z", "gate_proj_mem_z"):
                gz = getattr(buf, name).weight.grad
                check(f"{name} has gradient",
                      gz is not None and torch.isfinite(gz).all()
                      and float(gz.norm()) > 0,
                      f"|g|={float(gz.norm()):.3e}" if gz is not None else "None")
        elif gated:
            # A ZERO Z-GATE GRADIENT HERE PROVES NOTHING, so do not report it as
            # a pass OR a fail -- run the one split that can answer the
            # question instead.  Two different questions were being conflated:
            #
            #   "is the Z gate WIRED at all?"      an architecture question, and
            #                                     answerable only at (0, T)
            #   "how often does it get gradient?"  a statistics question about
            #                                     the training schedule, which
            #                                     the fraction above answers
            #
            # The first one is what a smoke exists to check, and it would
            # otherwise be permanently skipped: the default split is
            # [T//2, T-T//2], which never has n = 0.
            print("  [INFO] the default split has no live Z read, so the Z "
                  "gate cannot be judged\n         from it.  Re-running one "
                  "chain at num_steps=[0, T] to answer the\n         "
                  "ARCHITECTURE question (is the gate wired?) separately from "
                  "the\n         SCHEDULE question (how often is it live?).")
            model.zero_grad(set_to_none=True)
            rep2 = run_chain(model, x, y, eos_id, args.cross_chunks,
                             torch.tensor([0, args.T]), args.carry_grad_chunks,
                             n_vec, args.device, write_once=not gated)
            check("all chunk losses finite at num_steps=[0, T]",
                  all(l == l for l in rep2["losses"]))
            for name in ("gate_proj_in_z", "gate_proj_mem_z"):
                gz = getattr(buf, name).weight.grad
                check(f"{name} is wired (gradient at num_steps=[0, T])",
                      gz is not None and torch.isfinite(gz).all()
                      and float(gz.norm()) > 0,
                      f"|g|={float(gz.norm()):.3e}" if gz is not None else "None")
            if cortex.latent_read_measured:
                print(f"       (Z read was live on "
                      f"{100 * cortex.latent_read_grad_frac:.0f}% of all "
                      f"forwards including this pass)")

    # NOTE: when the Z wiring pass above ran, the gradients below are from THAT
    # chain (the first chain's were cleared for it).  Both chains exercise the
    # same write path, so every check still means what it says -- but they are
    # not the numbers from the [T//2, T-T//2] split, and a reader comparing
    # magnitudes across runs needs to know which chain produced them.
    g = cortex.prefix.summary_emb.grad
    check("summary_emb has gradient (write is on the loss)",
          g is not None and torch.isfinite(g).all() and float(g.norm()) > 0,
          f"|g|={float(g.norm()):.3e}" if g is not None else "None")

    loop_g = [p.grad for n, p in model.named_parameters()
              if "core_block" in n and p.grad is not None]
    check("loop params have gradient (freeze_loop=false)", len(loop_g) > 0,
          f"{len(loop_g)} tensors")
    check("no NaN/Inf in any gradient",
          all(torch.isfinite(p.grad).all() for p in model.parameters()
              if p.grad is not None))

    if args.device.startswith("cuda"):
        print(f"\n  peak VRAM: {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")

    print(f"\n{'SMOKE PASSED' if ok else 'SMOKE FAILED'}"
          f"  (mean loss {rep['total']:.4f}; a fresh conversion starts ~10.3)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
