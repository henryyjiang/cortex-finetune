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

Login node (no GPU): defaults are small (S=256/chunk, T=2) so this finishes in
a couple of minutes on CPU.  It exercises the code path, not the model quality.

    module load anaconda3 && conda activate cortex-retro
    export HF_HOME=$SCRATCH/hf_cache HF_HUB_OFFLINE=1
    python tools/smoke_prefix_real.py --model_name ckpts/olmo-retrofit-cortex

Full training geometry (needs a GPU; submit it or grab an interactive node):
    python tools/smoke_prefix_real.py --model_name ckpts/olmo-retrofit-cortex \
        --full --device cuda --dtype bfloat16

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
    p.add_argument("--device", default="cpu")
    p.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    p.add_argument("--full", action="store_true",
                   help="B2's real geometry: chunk_len 1024, T 8 (needs a GPU)")
    p.add_argument("--quick", action="store_true",
                   help="chunk_len 64 — every check still fires, ~4x less compute "
                        "than the default; use this on a CPU login node")
    return p.parse_args()


def build_model(args):
    from transformers import AutoConfig, AutoModelForCausalLM

    cfg = AutoConfig.from_pretrained(args.model_name, trust_remote_code=True)
    for k, v in (("use_memory", True), ("memory_slots", 0), ("memory_slots_iter", 0),
                 ("prefix_memory", args.prefix_memory),
                 ("accum_vecs", args.accum_vecs), ("accum_max", args.accum_max),
                 ("prefix_pos", "tail"), ("prefix_eos_reset", False)):
        setattr(cfg, k, v)
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, trust_remote_code=True, config=cfg, torch_dtype=dtype)
    return model.to(args.device).train(), cfg


def run_chain(model, x, y, eos_id, n_chunks, num_steps, carry_grad_chunks,
              accum_vecs, device, verbose=False):
    """train.py's cortex_fwd_bwd, minus DDP/autocast/L2-SP.  Returns a report.

    verbose prints per-chunk progress: on a CPU login node one chunk of the 1B
    model takes minutes, and a silent run is indistinguishable from a hang."""
    x_chunks = [c.contiguous() for c in torch.chunk(x, n_chunks, dim=1)]
    y_chunks = [c.contiguous() for c in torch.chunk(y, n_chunks, dim=1)]

    m_cross, losses, shapes, preserved = None, [], [], []
    for gi, (xc, yc) in enumerate(zip(x_chunks, y_chunks)):
        t0 = time.time()
        if verbose:
            print(f"  chunk {gi + 1}/{n_chunks} ({xc.shape[1]} tok, carry "
                  f"{0 if m_cross is None else m_cross.shape[1]}) ...",
                  end="", flush=True)
        prev = None if m_cross is None else m_cross.detach().clone()
        if carry_grad_chunks > 0 and m_cross is not None:
            m_cross = detach_old_vecs(m_cross, accum_vecs, carry_grad_chunks)
        out = model(xc.to(device), labels=yc.to(device), num_steps=num_steps,
                    m_cross_in=m_cross, return_m_cross=True,
                    eos_mask=(xc == eos_id).to(device))
        m_cross = out["m_cross"]
        losses.append(out["loss"])
        shapes.append(tuple(m_cross.shape))
        # Fix A on real weights: the rows written by earlier chunks must come
        # back verbatim, even from a chunk that contained a document boundary.
        if prev is not None:
            preserved.append(bool(torch.allclose(
                m_cross[:, :prev.shape[1]].detach().float(), prev.float(),
                atol=1e-3, rtol=1e-3)))
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
            "preserved": preserved, "total": float(total.detach())}


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
    check("prefix_pos == 'tail' (2026-08-04)", cortex.prefix_pos == "tail",
          f"got {cortex.prefix_pos!r}")
    check("prefix_eos_reset is off (2026-08-04)", cortex.prefix_eos_reset is False,
          f"got {cortex.prefix_eos_reset!r}")

    eos_id = cortex.summary_init_token
    n_vec = cortex.prefix.n_vec
    S = args.chunk_len * args.cross_chunks
    print(f"\n-- geometry --\n  n_embd={cfg.n_embd} vocab={cfg.vocab_size} eos={eos_id}"
          f"\n  {args.cross_chunks} chunks x {args.chunk_len} tok, n_vec={n_vec}, T={args.T}"
          f"\n  max packed position = {args.chunk_len + n_vec} "
          f"(block_size={getattr(cfg, 'block_size', '?')}, "
          f"max_position_embeddings={getattr(cfg, 'max_position_embeddings', '?')})")

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
                    args.carry_grad_chunks, n_vec, args.device, verbose=True)
    print(f"  {time.time() - t0:.1f}s")

    print("\n-- checks --")
    check("summary_emb seeded on first forward",
          bool(cortex.prefix.summary_seeded))
    wte = model.transformer.wte.weight
    check("summary_emb == wte[eos] (AutoCompressor init)",
          torch.allclose(cortex.prefix.summary_emb.detach().float(),
                         wte[eos_id].detach().float().unsqueeze(0).expand(n_vec, -1),
                         atol=1e-3))
    want = [(1, (g + 1) * n_vec, cfg.n_embd) for g in range(args.cross_chunks)]
    check("carry accumulates one write per chunk", rep["shapes"] == want,
          f"{rep['shapes']}")
    check("carry survives document boundaries (Fix A)",
          all(rep["preserved"]), f"{rep['preserved']}")
    check("all chunk losses finite",
          all(l == l and abs(l) != float("inf") for l in rep["losses"]),
          " ".join(f"{l:.3f}" for l in rep["losses"]))

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
