"""
P0.1 — the latent-channel scale check.  cortex_next_phase_framework.md §6.

WHAT IT DECIDES, AND WHY IT BLOCKS EVERYTHING IN ISSUE 1.
The dual-channel design reads Z by SUBSTITUTING it into the carried columns of
`s0`, replacing the `trunc_normal_` noise that `initialize_state` puts there.
Zero new parameters, live at init, sequence length unchanged -- which is the
whole reason it avoids the zero-init bootstrap that killed every earlier latent
design.  But a substitution is only in-distribution if the thing substituted IN
is on the same scale as the thing taken OUT.  Nobody has measured that.

What is known is the OTHER channel: E writes post-`ln_f` states whose norms were
measured at 143.7-144.6, against a nominal ||s0|| of ~0.63 at D=2048 -- a factor
of ~230.  That figure is token-space and says nothing about the latent
trajectory.  If the latent deltas are also orders of magnitude off `s0`, Z needs
a parameter-free renorm and the design changes before a line of it is built.  If
they are within a small factor, the substitution is clean as drawn.

WHAT IT MEASURES.  One forward pass, recording the recurrence trajectory:

  ||s0||          the trunc_normal_ init -- the scale Z must match
  ||s_t||         the state after each loop iteration, t = 1..T
  ||d_t||         ||s_t - s_{t-1}||, the STAGGERED DELTAS -- this is what Z is
                  actually made of (framework issue 1: "write Z from the
                  trajectory, not the endpoint"), so ||d||/||s0|| is the number
                  that decides the renorm, not ||s_T||/||s0||
  cos(s_t,s_t-1)  trajectory convergence -- a fixed point that arrives early
                  means the state stops moving
  cos(d_t,d_t-1)  THE ONE THAT DECIDES THE DEPTH SLICE.  A converged trajectory
                  (cos(s,s)->1) can still be exploring, if each small step goes
                  somewhere new; consecutive deltas near-parallel means the late
                  depths are one direction re-walked and carry no new
                  information for Z to write.  Near 0 means they are exploring
                  and a staggered write over late depths is buying variety.
  ||m_cross||     the E channel's actual write (the summary slots' post-`ln_f`
                  states, i.e. the carry itself), on the same page for scale

All norms are per-token L2 (mean over batch and position), so they are
comparable across D and directly comparable to the 143.7-144.6 figure.

WHERE THE STATES ACTUALLY ARE, since two of them are easy to mix up:
  * `out.latent_states` is `x.clone().detach()` taken BEFORE the coda and
    before `ln_f` -- it is s_T, NOT a post-`ln_f` state.  This script asserts it
    against the last hooked step, which is a real check that the hook captured
    the true trajectory rather than something adjacent to it.
  * `out.hidden_states` (needs `return_head: True`) is post-`ln_f` AND after
    prefix_unpack, so it is real-token columns only.
  * `out.m_cross` (needs `return_m_cross=True`) is the summary slots' post-
    `ln_f` states -- the E write.  That is the one to compare against 143.7.

TRAPS THIS SCRIPT IS WRITTEN AROUND.
  * `latent_states` is DETACHED.  Reading norms off it is fine (this is
    forward-only), but do not copy the access pattern into the training write --
    a detached Z is a gradient-free carry sitting behind a healthy loss curve,
    which is bug class 2 in a new hat.  The write must take a non-detached slice
    from inside the loop.
  * The trajectory is captured by wrapping `core_block_forward`, because
    `iterate_forward` returns only the endpoint and `xk.detach()` (the
    second-to-last state).  Everything between is discarded, and that is exactly
    what Z wants.
  * `num_steps` is forced to (T, 0) -- all no-grad.  The values are identical to
    the with-grad split and it is much cheaper.  The split decides WHICH depths
    are trainable, not what they contain; that is the depth-slice ablation's
    question, not this one.
  * RUN THIS IN FLOAT32.  It is the default for a reason.  Late deltas are on
    the order of 1% of ||s_t||, and bf16 carries ~8 mantissa bits, so a bf16
    subtraction of two converged states is substantially rounding.  The first
    bf16 run of this probe reported a delta plateau at 0.37x ||s0|| that fp32
    put at 0.06x -- a 6x overstatement -- and reported cos(d_t,d_t-1) = -0.62
    where fp32 says -0.95.  The plateau was the quantization floor wearing the
    shape of a result.  The QUANTIZATION FLOOR block below flags this
    automatically; do not silence it, fix the dtype.

Usage (local, 1B checkpoint, no cluster needed):
    python evals/diag_latent_scale.py \
        --model_name ../checkpoints/retrofit/retro-b2-acc32-cc8-mr8-final_checkpoint \
        --steps 32 --seq_len 512 --batch 4

    # against a real corpus rather than random ids:
    python evals/diag_latent_scale.py --model_name <ckpt> \
        --data data/fineweb_edu_olmo_len4096
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
from datetime import datetime

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evals.model_utils import _unwrap, load_checkpoint  # noqa: E402


def _token_norms(x: torch.Tensor) -> torch.Tensor:
    """Per-token L2 over the hidden dim -> [B*S].  float32 so a bf16 forward is
    not quantised by the statistic measuring it."""
    return x.detach().float().flatten(0, -2).norm(dim=-1)


def _stats(x) -> dict | None:
    if x is None:
        return None
    n = _token_norms(x)
    return {
        "mean": float(n.mean()),
        "std": float(n.std()),
        "p05": float(n.quantile(0.05)),
        "p50": float(n.median()),
        "p95": float(n.quantile(0.95)),
    }


def record_trajectory(model, input_ids, num_steps: int, use_prefix: bool = True):
    """One forward, capturing the state after every loop iteration.

    Returns (s0, [s_1..s_T], out).  Every captured tensor is detached; this is a
    measurement, not a training path.
    """
    inner = _unwrap(model)
    if not hasattr(inner, "core_block_forward"):
        raise RuntimeError(
            "core_block_forward is not on the model class -- the modeling "
            "file's loop entry point has moved.  Re-read iterate_forward "
            "before trusting anything this script prints.")

    traj: list[torch.Tensor] = []
    holder: dict = {}
    real_core = inner.core_block_forward
    real_init = inner.initialize_state

    def wrapped_core(x, *a, **kw):
        # x is the state ENTERING this iteration; the first call sees s0.
        holder.setdefault("s0", x.detach())
        out = real_core(x, *a, **kw)
        traj.append((out[0] if isinstance(out, tuple) else out).detach())
        return out

    def wrapped_init(input_embeds, scale: float = 1.0):
        s = real_init(input_embeds, scale=scale)
        holder.setdefault("s0", s.detach())
        return s

    # Restore by DELETING the shadow, not by assigning the bound method back.
    # `inner.core_block_forward = real_core` looks like a restore but leaves a
    # permanent entry in the instance __dict__ that shadows the class method
    # forever -- so a later patch of the class (another probe, a test) would be
    # silently overridden by this stale binding, and the model would hold a
    # reference cycle to its own bound method.  Snapshot whether the name was an
    # instance attribute to begin with and put back exactly that.
    had_core = "core_block_forward" in vars(inner)
    had_init = "initialize_state" in vars(inner)
    inner.core_block_forward = wrapped_core
    inner.initialize_state = wrapped_init
    try:
        with torch.no_grad():
            out = model(
                input_ids=input_ids,
                num_steps=torch.tensor([num_steps, 0]),
                output_details={"return_logits": False, "return_latents": True,
                                "return_head": True, "return_stats": False},
                return_m_cross=True,
                prefix_write=use_prefix,
                prefix_read=use_prefix,
            )
    finally:
        if had_core:
            inner.core_block_forward = real_core
        else:
            del inner.core_block_forward
        if had_init:
            inner.initialize_state = real_init
        else:
            del inner.initialize_state

    return holder.get("s0"), traj, out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", required=True)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--data", default=None,
                    help="a packed dataset (cluster-side)")
    ap.add_argument("--text_file", default=None,
                    help="a UTF-8 text file, tokenized with the checkpoint's "
                         "own tokenizer.  Real prose, no pack needed -- use "
                         "this off-cluster.  Random ids are a poor proxy: they "
                         "give the loop nothing to converge ON, so the "
                         "trajectory shape is not the trained one.")
    ap.add_argument("--steps", type=int, default=32, help="T, loop iterations")
    ap.add_argument("--seq_len", type=int, default=512)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--no_prefix", action="store_true",
                    help="run with prefix_write/prefix_read off -- the pure "
                         "backbone trajectory over exactly seq_len columns, no "
                         "summary slots appended")
    ap.add_argument("--dtype", default="float32",
                    choices=["bfloat16", "float32"],
                    help="float32 by DEFAULT and it matters: the late deltas "
                         "are ~1%% of ||s_t||, which is inside bf16's relative "
                         "precision.  A bf16 run reports a delta PLATEAU that "
                         "is the quantization floor, not the model (measured "
                         "2026-09-15: 0.37x s0 in bf16 against 0.06x in fp32 "
                         "at t=32, a 6x overstatement).")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)

    # load_checkpoint returns (model, config) -- it is already .eval() and on
    # the requested device/dtype.
    model, cfg = load_checkpoint(args.checkpoint, args.model_name, None,
                                 dtype, device)
    inner = _unwrap(model)
    D = int(cfg.n_embd)
    std = float(cfg.init_values["std"])

    if args.data:
        from datasets import load_from_disk
        ds = load_from_disk(args.data)
        rows = [ds[i]["input_ids"][: args.seq_len] for i in range(args.batch)]
        input_ids = torch.tensor(rows, dtype=torch.long, device=device)
        source = args.data
    elif args.text_file:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.model_name,
                                            trust_remote_code=True)
        raw = io.open(args.text_file, encoding="utf-8", errors="replace").read()
        ids = tok(raw, return_tensors=None)["input_ids"]
        need = args.batch * args.seq_len
        if len(ids) < need:
            print(f"FAILED: {args.text_file} tokenizes to {len(ids)} tokens, "
                  f"need {need} for batch={args.batch} seq_len={args.seq_len}.")
            return 2
        input_ids = torch.tensor(
            [ids[i * args.seq_len:(i + 1) * args.seq_len]
             for i in range(args.batch)], dtype=torch.long, device=device)
        source = f"{args.text_file} ({len(ids)} tokens available)"
    else:
        input_ids = torch.randint(0, int(cfg.vocab_size) - 1,
                                  (args.batch, args.seq_len), device=device)
        source = "random ids"

    s0, traj, out = record_trajectory(model, input_ids, args.steps,
                                      use_prefix=not args.no_prefix)
    if s0 is None or not traj:
        print("FAILED: captured no trajectory -- the hook did not fire.")
        return 2
    if len(traj) != args.steps:
        print(f"FAILED: captured {len(traj)} steps, asked for {args.steps}.  "
              "num_steps is not reaching iterate_forward as expected.")
        return 2

    # Self-check: latent_states is s_T taken before the coda, so it must equal
    # the last hooked step exactly.  If it does not, the hook is capturing
    # something adjacent to the trajectory and every number below is wrong.
    ls = getattr(out, "latent_states", None)
    if ls is not None:
        if not torch.allclose(ls.float(), traj[-1].float(), atol=1e-3, rtol=1e-3):
            print("FAILED self-check: out.latent_states != last hooked step.  "
                  "The hook is not on the real trajectory.")
            return 2
        print("[selfcheck] out.latent_states == last hooked step: OK")

    s0_stats = _stats(s0)
    s0m = s0_stats["mean"]
    steps = []
    prev = s0
    prev_d = None
    for t, s in enumerate(traj, start=1):
        d = s - prev
        cos = torch.nn.functional.cosine_similarity(
            s.float().flatten(0, -2), prev.float().flatten(0, -2), dim=-1)
        dd = None
        if prev_d is not None:
            dd = float(torch.nn.functional.cosine_similarity(
                d.float().flatten(0, -2), prev_d.float().flatten(0, -2),
                dim=-1).mean())
        steps.append({"t": t, "s": _stats(s), "delta": _stats(d),
                      "cos_with_prev": float(cos.mean()),
                      "cos_delta_with_prev_delta": dd})
        prev, prev_d = s, d

    # QUANTIZATION FLOOR.  A delta is only measurable if it is well above the
    # representational granularity of the states it is the difference of.  eps
    # is the dtype's relative spacing, so eps * ||s_t|| is roughly the smallest
    # per-component difference the subtraction can resolve; scaled by sqrt(D)
    # for the per-token L2.  Flag any step whose delta is within 4x of it.
    eps = float(torch.finfo(dtype).eps)
    suspect = []
    for r in steps:
        floor = eps * r["s"]["mean"] * (D ** 0.5) / (D ** 0.5)  # eps * ||s||
        r["quant_floor"] = floor
        r["floor_ratio"] = r["delta"]["mean"] / floor if floor > 0 else float("inf")
        if r["floor_ratio"] < 4.0:
            suspect.append(r["t"])

    m_cross = _stats(getattr(out, "m_cross", None))
    post_tok = _stats(getattr(out, "hidden_states", None))
    sT = steps[-1]["s"]["mean"]
    d_first = steps[0]["delta"]["mean"]
    d_mid = steps[len(steps) // 2]["delta"]["mean"]
    d_last = steps[-1]["delta"]["mean"]

    report = {
        "probe": "P0.1 latent scale",
        "when": datetime.now().isoformat(timespec="seconds"),
        "model_name": args.model_name,
        "checkpoint": args.checkpoint,
        "source": source,
        "geometry": {"D": D, "init_std": std, "T": args.steps,
                     "seq_len": args.seq_len, "batch": args.batch,
                     "dtype": args.dtype, "prefix": not args.no_prefix},
        "analytic_s0_norm": std * D ** 0.5,
        "s0": s0_stats,
        "m_cross_E_write": m_cross,
        "post_ln_f_real_tokens": post_tok,
        "steps": steps,
        "ratios": {
            "sT_over_s0": sT / s0m,
            "delta_first_over_s0": d_first / s0m,
            "delta_mid_over_s0": d_mid / s0m,
            "delta_last_over_s0": d_last / s0m,
            "E_write_over_s0": (m_cross["mean"] / s0m) if m_cross else None,
        },
        "quantization": {
            "dtype": args.dtype,
            "eps": eps,
            "steps_within_4x_of_floor": suspect,
        },
    }

    out_path = args.out or os.path.join(
        "eval_results", f"p01_latent_scale-{datetime.now():%Y%m%d-%H%M%S}",
        "results.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n=== P0.1 latent scale | {os.path.basename(args.model_name)} "
          f"| {source} ===")
    print(f"    D={D} init_std={std:.6g} T={args.steps} seq_len={args.seq_len} "
          f"batch={args.batch} dtype={args.dtype} "
          f"prefix={'off' if args.no_prefix else 'on'}")
    print()
    print(f"  ||s0||         {s0m:10.4f}   (analytic std*sqrt(D) = "
          f"{std * D ** 0.5:.4f})")
    print(f"  ||s_T||        {sT:10.4f}   {sT / s0m:8.2f}x s0")
    if m_cross:
        print(f"  ||m_cross||    {m_cross['mean']:10.4f}   "
              f"{m_cross['mean'] / s0m:8.2f}x s0   <- the E channel's write")
    if post_tok:
        print(f"  ||post-ln_f||  {post_tok['mean']:10.4f}   "
              f"{post_tok['mean'] / s0m:8.2f}x s0   (real tokens)")
    print()
    print("  t     ||s_t||     ||d_t||     d/s0    cos(s,s-1)   cos(d,d-1)")
    for r in steps:
        dd = r["cos_delta_with_prev_delta"]
        print(f"  {r['t']:<4d}{r['s']['mean']:10.4f}{r['delta']['mean']:12.4f}"
              f"{r['delta']['mean'] / s0m:10.2f}{r['cos_with_prev']:13.4f}"
              f"{'' if dd is None else f'{dd:13.4f}'}")
    print()
    if suspect:
        lo, hi = min(suspect), max(suspect)
        print(f"  !! QUANTIZATION FLOOR: at t={lo}..{hi} the delta is within 4x "
              f"of {args.dtype}'s\n     resolvable difference "
              f"(eps={eps:.2e}).  Those rows are measuring arithmetic, not the\n"
              f"     model.  Re-run with --dtype float32 before quoting any of "
              f"them.\n")
    print(f"  VERDICT INPUT: the staggered deltas Z is made of run "
          f"{d_first / s0m:.1f}x (first) / {d_mid / s0m:.1f}x (mid) / "
          f"{d_last / s0m:.1f}x (last)")
    print("  the scale of the noise they would replace in s0.")
    print("  Near 1x: substitution is in-distribution as drawn, no renorm.")
    print("  Far from 1x: Z needs a parameter-free renorm, decided BEFORE the "
          "write is built.")
    print(f"\n  wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
