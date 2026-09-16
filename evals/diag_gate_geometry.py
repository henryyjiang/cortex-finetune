"""
P1.0 -- gated-buffer geometry: writes-per-chunk x width x routing x horizon.
cortex_next_phase_framework.md S6 (P1.0), control_launch_handoff.md S4.

WHAT THIS DECIDES.  P1.0 asks how to make each chunk's write CHEAPER so that
the horizon extends, and the blocker it named is a shape, not a config:
`PrefixGatedBuffer` used to set `n_slots = n_vec`, so writes-per-chunk was
locked to buffer width and "16 cheap writes into a 128-slot state" could not be
expressed.  That is now decoupled (buffers.py), which opens a routing question
-- how do W candidate vectors reach K state rows -- with four answers on the
table.  This probe prices them against each other AND against the B2 accum
buffer that is actually in production, on ONE axis that matters:

    replace the write at chunk N-d with a DONOR document's write, replay, and
    measure how far the final state moved.

    A(d) = || s_N(write_{N-d} := donor) - s_N ||  /  || one chunk's write ||
    R(d) = the same displacement, over || s_N ||

"how much of the state at chunk N still traces to the write d chunks back."
It is the buffer-level form of the influence-horizon instrument that has to
replace the x1.31 compounding metric, because a fixed-width gated state should
PLATEAU and the AC-style metric scores a plateau as failure.  A(d) is the
HEADLINE: R(d) is diluted by width (a buffer holding twice as many chunks shows
half the share of each while remembering strictly more), so R is comparable only
within a fixed K, while A is in units of one write and compares across cells.

The shapes to expect, and the reason the comparison is worth running:

  accum  a STEP.  Rows are write-once, so A(d) is flat while chunk N-d is still
         in the FIFO and EXACTLY ZERO the moment it is evicted.  That cliff is
         the issue-4 step function: +4.33/+2.00/+1.00 carry delta inside the
         4,096-token horizon, -1.67/-1.00/-6.50 outside it.
  dense  ("mix") EXPONENTIAL.  Every row is multiplied by fg on every chunk,
         so A(d) ~ fg^d with no cliff and no plateau -- and every row ends up a
         superposition of every chunk, which is what makes it a blurred
         attention key rather than a selectable one.
  ring   a STAIRCASE.  A row is untouched for K/W chunks and merely GATED when
         its turn comes, so retention follows F(d) = ig * fg**floor(d/(K/W)).
         At gate_init="zero" that is exact at step 0 and it is what the locked
         W=16/K=64 geometry was chosen on.

HOW THE WRITES ARE OBTAINED.  Forward-only, on the real checkpoint: the model
is run chunk-by-chunk over real prose with its own trained accum carry, and the
post-ln_f states at the summary columns -- the actual write, `prefix_unpack`'s
`new_vecs` -- are taped.  A SECOND tape is collected from a different document
to serve as the donor.  Buffers are then replayed over the tape offline, so
every configuration sees byte-identical writes and the only thing varying is
the merge.

FIVE THINGS THIS PROBE CANNOT TELL YOU, stated here so they are not discovered
in the writeup:

  0. gate_norm IS NOT MEASURABLE HERE, BY CONSTRUCTION.  At gate_init="zero"
     both projections are zero, so tanh/rms/none produce IDENTICAL retention --
     the rows agreeing is a tautology, not a finding.  What the probe CAN say
     about it is `tanh_saturated_frac`: the share of state entries past |2|,
     where tanh' < 0.08 and the forget gate is reading sign(state) rather than
     magnitude.  That is what predicts whether the choice will matter once the
     weights move off zero.
  1. THE GATES ARE UNTRAINED.  Every gated number is an AT-INIT number
     (forget_bias +1.0, input_bias 0, plain Linear gates).  It prices the
     ARCHITECTURE's retention at initialisation and separates the routings from
     each other; it does not predict where training moves them.  P1.0 says so:
     retention needs short training cells, and this is what runs first and for
     free.
  2. THE WRITES COME FROM AN ACCUM-TRAINED MODEL.  A gated arm would produce
     its own write distribution.  Holding the tape fixed is what makes the
     comparison controlled, and it is also its ceiling.
  3. W < 32 IS A PROXY.  The tape has the checkpoint's trained 32 summary
     columns; a W=16 cell takes the FIRST 16 of them.  Under the `tail` layout
     slot j attends to slots < j, so a prefix is at least a coherent subset,
     but a real W=16 model would train its own 16.  Cross-W rows are indicative;
     within-W rows (routing A/B at fixed W) are clean.
  4. STATE-SPACE DISPLACEMENT IS NOT LOSS.  A(d) measures energy in the carry,
     not nats at the head.  A row the model ignores still counts here.  Turning
     this into nats is the job of the loss-level influence horizon that has to
     exist before A3 -- this probe is its cheap upstream half.

FLOAT32 THROUGHOUT, and it is not optional: A(d) is a DIFFERENCE OF TWO LARGE
NEARLY-EQUAL STATES, which is the exact quantity a bf16 pass got wrong in P0.1
(a delta plateau at 0.37x that fp32 puts at 0.06x).  --dtype bfloat16 exists
only so that mistake can be reproduced deliberately; it prints a warning.

USAGE
  python evals/diag_gate_geometry.py \
      --model_name ../checkpoints/retrofit/retro-b2-acc32-cc8-mr8-final_checkpoint \
      --text_file ../cortex_next_phase_framework.md \
      --donor_file ../recipe_sweep_findings.md \
      --chunks 24 --seq_len 512 --steps 8 \
      --out eval_results/p10_gate_geometry-$(date +%Y%m%d)

  # no GPU / no checkpoint: exercises every code path on synthetic writes
  python evals/diag_gate_geometry.py --synthetic --chunks 24
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

from cortex_memory.buffers import PrefixAccumBuffer, PrefixGatedBuffer  # noqa: E402


# ---------------------------------------------------------------------------
# The configurations under test
# ---------------------------------------------------------------------------
#
# Every cell is (label, kind, kwargs).  The accum cells are the incumbents:
# b2 is EXACTLY the production geometry, the other two are the P1.0 table's
# cheap-write rows, so "does the gate beat accum" and "does a cheaper accum
# write beat the current one" are answered on the same axes.
#
# read_cols is what the cell costs in packed sequence length per chunk, and it
# is NOT equal across cells by construction -- that is the mechanism (accum's
# read block grows to accum_max, a gated one never does), not a confound to
# remove.  It is reported next to every retention number so the comparison is
# never read as free.

def build_cells(D: int, chunk_len: int, cells: list[str] | None):
    out = []

    def add(label, mod, note):
        out.append({"label": label, "buf": mod, "note": note})

    # -- incumbents: what is in production, and the accum alternatives ---
    add("accum W32 K256 (B2)", PrefixAccumBuffer(D, 32, 256),
        "the production geometry: 16:1 compression, hard 8-chunk horizon")
    add("accum W16 K64", PrefixAccumBuffer(D, 16, 64),
        "COST-MATCHED accum control: same read block as the locked cell, so "
        "the gate's contribution is not confounded with buying more columns")
    add("accum W16 K512", PrefixAccumBuffer(D, 16, 512),
        "P1.0 row: 32:1 and a 32-chunk horizon, but +448 read columns")

    # -- THE LOCKED CELL, and the knobs around it -------------------------
    add("gated W16 K64 ring", PrefixGatedBuffer(D, 16, 64, route="ring"),
        "LOCKED 2026-09-16: 32:1 write, lap 4, 64 read cols, 592 packed")
    add("gated W16 K64 ring/rms",
        PrefixGatedBuffer(D, 16, 64, route="ring", gate_norm="rms"),
        "same, with the saturated tanh on the memory side replaced")
    add("gated W16 K64 ring/kaiming",
        PrefixGatedBuffer(D, 16, 64, route="ring", gate_init="default"),
        "same, with PyTorch's gate init.  Scores HIGHER on retention for a "
        "bad reason -- see the warning under the geometry table")
    add("gated W16 K64 ring/slotinit",
        PrefixGatedBuffer(D, 16, 64, route="ring", fill="init"),
        "same, but the first lap pads from slot_init instead of growing")

    # -- the sparse-vs-dense contrast, at matched W and K -----------------
    add("gated W16 K64 mix", PrefixGatedBuffer(D, 16, 64, route="mix"),
        "DENSE control: every row refreshed every chunk")
    add("gated W16 K64 mix/tile",
        PrefixGatedBuffer(D, 16, 64, route="mix", route_init_std=0.0),
        "dense with NO symmetry breaker -- provably rank-pinned at W, kept to "
        "show the degeneracy numerically")

    # -- the pre-P1.0 shape, i.e. what A3 would have run ------------------
    add("gated W32 K32 (pre-P1.0)", PrefixGatedBuffer(D, 32, 32),
        "the only gated buffer the code could express before P1.0")

    if cells:
        keep = [c for c in out if any(k in c["label"] for k in cells)]
        if not keep:
            raise SystemExit(f"--cell matched nothing; labels are "
                             f"{[c['label'] for c in out]}")
        out = keep
    for c in out:
        b = c["buf"]
        if isinstance(b, PrefixAccumBuffer):
            c["geom"] = {
                "n_vec": b.n_vec, "n_slots": b.max_vecs, "route": "append",
                "gate_norm": "-", "compression": chunk_len / b.n_vec,
                "read_cols_per_chunk": b.max_vecs,
                "write_cols_per_chunk": b.n_vec,
                "packed_cols": b.max_vecs + chunk_len + b.n_vec,
                "refresh_period": b.max_vecs // b.n_vec,
                "params": sum(p.numel() for p in b.parameters()),
            }
        else:
            c["geom"] = b.geometry(chunk_len)
    return out


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def rank_stats(mat: torch.Tensor) -> dict:
    """[B, K, D] -> centred cosine + two effective ranks, averaged over lanes.

    Centring first: K vectors sharing a large common component read as ~1.0
    uncentred however much independent structure sits on top.  Same statistic
    as diag_position_rank.py, so the numbers are comparable to P0.2's.
    """
    m = mat.detach().float()
    cos, ent, pr = [], [], []
    for b in range(m.shape[0]):
        c = m[b] - m[b].mean(0, keepdim=True)
        K = c.shape[0]
        n = c.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        g = (c / n) @ (c / n).T
        off = ~torch.eye(K, dtype=torch.bool, device=g.device)
        cos.append(float(g[off].mean()))
        s = torch.linalg.svdvals(c).clamp_min(0)
        tot = float(s.sum())
        if tot <= 0:
            ent.append(1.0)
            pr.append(1.0)
            continue
        p = s / s.sum()
        nz = p[p > 0]
        ent.append(float(torch.exp(-(nz * nz.log()).sum())))
        s2 = s ** 2
        pr.append(float((s2.sum() ** 2) / (s2 ** 2).sum().clamp_min(1e-30)))
    n = len(cos)
    return {"centred_cosine": sum(cos) / n,
            "eff_rank_entropy": sum(ent) / n,
            "eff_rank_pr": sum(pr) / n}


def replay(buf, tape: torch.Tensor, swap_at: int | None = None,
           donor: torch.Tensor | None = None) -> torch.Tensor:
    """Run a buffer over a [N, B, W, D] write tape, optionally substituting the
    donor's write at chunk index `swap_at`.  Returns the final state."""
    if hasattr(buf, "_chunk"):
        buf._chunk = 0
    state = None
    with torch.no_grad():
        for i in range(tape.shape[0]):
            w = donor[i] if (swap_at is not None and i == swap_at) else tape[i]
            state = buf.merge(state, w)
    return state


def gate_report(buf, state: torch.Tensor, new_vecs: torch.Tensor) -> dict:
    """What the gate actually does on one real merge, at init.

    `sat` is the fraction of |state| entries above 2.0, where tanh' < 0.08:
    past that the memory side of the gate is reading sign(state) and the forget
    gate cannot see how much a row holds.  It is the number that decides whether
    gate_norm should stay at LM2's published tanh.
    """
    if not isinstance(buf, PrefixGatedBuffer):
        return {}
    with torch.no_grad():
        W = new_vecs.shape[1]
        cand = (new_vecs if buf.route == "ring"
                else buf.route_candidate(new_vecs))
        sub = (state.index_select(
                   1, ((buf._chunk * W) % buf.n_slots
                       + torch.arange(W, device=state.device)) % buf.n_slots)
               if buf.route == "ring" else state)
        _, ig, fg = buf.gate(sub, cand)
        return {
            "ig_mean": float(ig.mean()), "fg_mean": float(fg.mean()),
            "fg_p10": float(fg.flatten().quantile(0.10)),
            "fg_p90": float(fg.flatten().quantile(0.90)),
            "tanh_saturated_frac": float((state.abs() > 2.0).float().mean()),
            "state_abs_mean": float(state.abs().mean()),
            # How hard the routing rescales a write before the gate sees it.
            # attn averages W candidates, so at init it delivers roughly
            # 1/sqrt(W) of a real write -- a scale defect, not an opinion.
            "cand_norm_ratio": float(cand.norm(dim=-1).mean()
                                     / new_vecs.norm(dim=-1).mean()),
        }


def horizon_curve(buf, tape: torch.Tensor, donor: torch.Tensor,
                  depths: list[int]) -> dict:
    """Two retention curves per cell, because one of them is confounded.

    Both come from the same substitution: replace the write at chunk N-1-d with
    a donor document's write, replay, and measure how far the final state moved.

      R(d)  displacement as a SHARE of the state norm.  This is what the model
            reads -- the fraction of the carry block traceable to chunk N-d --
            but it is diluted by width: a buffer holding twice as many chunks
            has half the share of each while remembering strictly more.  Do not
            compare R across cells with different K.
      A(d)  displacement in units of ONE CHUNK'S WRITE ENERGY.  Dilution-free,
            so it IS comparable across cells: A(d) ~ sqrt(2) means chunk N-d is
            present at full write strength (sqrt(2) because an uncorrelated
            donor both removes the original and adds itself), and A(d) ~ 0
            means it is gone.  THIS IS THE HEADLINE CURVE.

    Summaries: `horizon_10pct` is the largest sampled d with A(d) >= 10% of
    A(1) -- censored at max(depths), and reported as such.  `retained_writes`
    is the trapezoidal integral of A over d, i.e. how many chunks' worth of
    write energy the state still carries; it is the number a PLATEAU should win
    on and a CLIFF should not, and it is the replacement for the x1.31
    compounding metric at buffer level.
    """
    N = tape.shape[0]
    base = replay(buf, tape).float()
    s_norm = base.flatten(1).norm(dim=-1)                       # [B]
    w_norm = tape.float().flatten(2).norm(dim=-1).mean(0)       # [B]
    # The CEILING: how far apart this chunk's write and the donor's actually
    # are, in the same units.  A buffer that retained chunk N-d perfectly and
    # in one copy would score exactly this.  It is NOT sqrt(2): the trained
    # write path is ~98% a constant direction, so two different documents'
    # writes differ by only ~0.2 of a write norm, and every A(d) below has to
    # be read against that, not against full orthogonality.
    ceil = ((tape.float() - donor.float()).flatten(2).norm(dim=-1).mean(0)
            / w_norm.clamp_min(1e-12))                          # [B]
    R, A, Fr = {}, {}, {}
    for d in depths:
        if d >= N:
            continue
        s = replay(buf, tape, swap_at=N - 1 - d, donor=donor).float()
        dist = (s - base).flatten(1).norm(dim=-1)               # [B]
        R[d] = float((dist / s_norm.clamp_min(1e-12)).mean())
        A[d] = float((dist / w_norm.clamp_min(1e-12)).mean())
        Fr[d] = float((dist / (w_norm * ceil).clamp_min(1e-12)).mean())
    if not A:
        return {"R": {}, "A": {}, "F": {}, "swap_ceiling": float(ceil.mean()),
                "horizon_10pct": 0, "horizon_censored": False,
                "retained_writes": 0.0, "final_state_norm": float(s_norm.mean())}
    ks = sorted(A)
    a1 = A[ks[0]]
    live = [d for d in ks if A[d] >= 0.10 * a1] or [0]
    mass = sum(0.5 * (Fr[ks[i]] + Fr[ks[i + 1]]) * (ks[i + 1] - ks[i])
               for i in range(len(ks) - 1))
    return {"R": {str(k): R[k] for k in ks},
            "A": {str(k): A[k] for k in ks},
            "F": {str(k): Fr[k] for k in ks},
            "swap_ceiling": float(ceil.mean()),
            "horizon_10pct": max(live),
            "horizon_censored": max(live) == ks[-1],
            "retained_writes": mass,
            "final_state_norm": float(s_norm.mean())}


# ---------------------------------------------------------------------------
# Tape collection (the only part that needs the model)
# ---------------------------------------------------------------------------

def tokenize_file(path: str, model_name: str, need: int):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    raw = io.open(path, encoding="utf-8", errors="replace").read()
    ids = tok(raw, return_tensors=None)["input_ids"]
    if len(ids) < need:
        raise SystemExit(
            f"FAILED: {path} tokenizes to {len(ids)} tokens, need {need}.  "
            f"Lower --chunks/--batch/--seq_len or pass a longer file.")
    return ids


def collect_tape(model, inner, path: str, args, device) -> torch.Tensor:
    """[n_chunks, B, 32, D] of the checkpoint's own summary writes.

    The chain runs with the model's PRODUCTION accum carry, so each chunk's
    write is conditioned on a realistic read -- which is the closest available
    stand-in for what a trained gated arm would see.
    """
    ids = tokenize_file(path, args.model_name,
                        args.batch * args.seq_len * args.chunks)
    per = args.seq_len * args.chunks
    rows = [ids[b * per:(b + 1) * per] for b in range(args.batch)]
    ids_t = torch.tensor(rows, dtype=torch.long, device=device)

    cortex = inner.cortex
    real_unpack = cortex.prefix_unpack
    grabbed: dict = {}

    def wrapped_unpack(x, n_pre, n_sum):
        if n_sum:
            grabbed["w"] = x[:, x.shape[1] - n_sum:].detach().float().cpu()
        return real_unpack(x, n_pre, n_sum)

    tape, carry = [], None
    cortex.prefix_unpack = wrapped_unpack
    try:
        for c in range(args.chunks):
            chunk = ids_t[:, c * args.seq_len:(c + 1) * args.seq_len]
            with torch.no_grad():
                out = model(input_ids=chunk,
                            num_steps=torch.tensor([args.steps, 0]),
                            m_cross_in=carry,
                            return_m_cross=True,
                            output_details={"return_logits": False,
                                            "return_latents": False,
                                            "return_head": True,
                                            "return_stats": False},
                            prefix_write=True, prefix_read=carry is not None)
            if "w" not in grabbed:
                raise SystemExit(
                    "FAILED: prefix_unpack captured no summary columns -- the "
                    "prefix splice has moved.  Re-read cortex_graft.py.")
            tape.append(grabbed.pop("w"))
            carry = out.m_cross
            print(f"  chunk {c + 1}/{args.chunks}  carry={tuple(carry.shape)}",
                  flush=True)
    finally:
        cortex.prefix_unpack = real_unpack
    return torch.stack(tape)


# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default=None)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--text_file", default=None,
                    help="real prose for the main tape")
    ap.add_argument("--donor_file", default=None,
                    help="a DIFFERENT document, for the donor tape.  Defaults "
                         "to the second half of --text_file, which is weaker: "
                         "same author, same register.")
    ap.add_argument("--synthetic", action="store_true",
                    help="skip the model; replay gaussian writes.  Exercises "
                         "every path, quotes nothing.")
    ap.add_argument("--chunks", type=int, default=24)
    ap.add_argument("--seq_len", type=int, default=512,
                    help="512 is B2's trained chunk length (4096 / cc8)")
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--steps", type=int, default=8,
                    help="T.  8 is the mr8 arms' TRAINED depth; the config "
                         "says 32 and evaluating there is finding 0c.")
    ap.add_argument("--dtype", default="float32",
                    choices=["float32", "bfloat16"])
    ap.add_argument("--device",
                    default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--cell", action="append", default=None,
                    help="repeatable substring filter on cell labels")
    ap.add_argument("--tape", default=None,
                    help="cache path; written after collection, reused if it "
                         "exists and matches the requested shape")
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    if args.dtype == "bfloat16":
        print("WARNING: bfloat16.  R(d) is a difference of two large nearly-"
              "equal states, which is the quantity bf16 got wrong in P0.1 by "
              "6x.  Do not quote anything below.", flush=True)

    D, source = 2048, "synthetic gaussian writes (NOT the trained regime)"
    if args.synthetic:
        D = 256
        tape = torch.randn(args.chunks, args.batch, 32, D)
        donor = torch.randn(args.chunks, args.batch, 32, D)
    else:
        if not args.model_name or not args.text_file:
            raise SystemExit("need --model_name and --text_file (or --synthetic)")
        cached = (args.tape and os.path.exists(args.tape))
        if cached:
            blob = torch.load(args.tape)
            tape, donor, source = blob["tape"], blob["donor"], blob["source"]
            if tape.shape[0] < args.chunks:
                raise SystemExit(
                    f"cached tape has {tape.shape[0]} chunks, need {args.chunks}")
            tape, donor = tape[:args.chunks], donor[:args.chunks]
            D = tape.shape[-1]
            print(f"[tape] reused {args.tape}  {tuple(tape.shape)}")
        else:
            from evals.model_utils import load_checkpoint, _unwrap
            device = torch.device(args.device)
            model, cfg = load_checkpoint(args.checkpoint, args.model_name, None,
                                         getattr(torch, args.dtype), device)
            inner = _unwrap(model)
            if getattr(getattr(inner, "cortex", None), "prefix", None) is None:
                print("FAILED: this checkpoint has no prefix buffer, so there "
                      "are no writes to tape.")
                return 2
            D = int(cfg.n_embd)
            print(f"[tape] main <- {args.text_file}", flush=True)
            tape = collect_tape(model, inner, args.text_file, args, device)
            donor_path = args.donor_file or args.text_file
            print(f"[tape] donor <- {donor_path}", flush=True)
            if args.donor_file:
                donor = collect_tape(model, inner, args.donor_file, args, device)
            else:
                # Fall back to a lane roll: weaker (same document) but it keeps
                # the probe runnable with one file.  Flagged in `source`.
                donor = tape.roll(1, dims=1).roll(args.chunks // 2, dims=0)
            source = (f"{args.text_file} / donor "
                      f"{args.donor_file or 'lane-rolled self (WEAK DONOR)'}")
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if args.tape:
                torch.save({"tape": tape, "donor": donor, "source": source},
                           args.tape)

    tape, donor = tape.float(), donor.float()
    N = tape.shape[0]
    depths = [d for d in [1, 2, 3, 4, 6, 8, 12, 16, 20, 24, 28, 32] if d < N]

    cells = build_cells(D, args.seq_len, args.cell)
    report = {
        "probe": "P1.0 gated buffer geometry",
        "when": datetime.now().isoformat(timespec="seconds"),
        "model_name": args.model_name,
        "source": source,
        "geometry": {"D": D, "chunks": N, "seq_len": args.seq_len,
                     "batch": args.batch, "T": args.steps,
                     "dtype": args.dtype, "synthetic": args.synthetic},
        "write_stats": {
            "write_norm_mean": float(tape.norm(dim=-1).mean()),
            "write_abs_mean": float(tape.abs().mean()),
            "donor_cos": float(
                torch.nn.functional.cosine_similarity(
                    tape.flatten(0, 2), donor.flatten(0, 2), dim=-1).mean()),
        },
        "cells": {},
    }

    for c in cells:
        buf, W = c["buf"], c["geom"]["n_vec"]
        sub = tape[:, :, :W], donor[:, :, :W]
        final = replay(buf, sub[0])
        rec = dict(c["geom"])
        rec["note"] = c["note"]
        rec.update(rank_stats(final))
        rec["state_norm"] = float(final.float().norm())
        rec["state_norm_per_row"] = float(final.float().norm(dim=-1).mean())
        rec.update(horizon_curve(buf, sub[0], sub[1], depths))
        prev = replay(buf, sub[0][:-1])
        rec.update(gate_report(buf, prev, sub[0][-1]))
        report["cells"][c["label"]] = rec
        print(f"  {c['label']:<26} rank {rec['eff_rank_entropy']:6.2f}  "
              f"h10 {rec['horizon_10pct']:>3}  writes-held "
              f"{rec['retained_writes']:6.2f}",
              flush=True)

    # -- tables -------------------------------------------------------------
    def row(label, rec, key):
        r = rec[key]
        cols = "".join(f"{r.get(str(d), float('nan')):7.3f}" for d in depths)
        return f"  {label:<26}{cols}"

    hdr = f"  {'cell':<26}" + "".join(f"{('d=' + str(d)):>7}" for d in depths)

    print("\n" + "=" * 100)
    print("F(d) -- fraction of chunk N-d's DISTINGUISHABLE content still in the "
          "state.  THE HEADLINE CURVE.")
    print("  1.0 = fully retained, 0 = gone, >1 = DUPLICATED into several rows "
          "(read it next to `rank`).")
    print("  accum should CLIFF, dense gates should DECAY, ring should STAIRCASE "
          "above dense at equal width.")
    print("=" * 100)
    print(hdr)
    for label, rec in report["cells"].items():
        print(row(label, rec, "F"))

    print("\n" + "=" * 100)
    print("R(d) -- the same displacement as a SHARE of the state norm (what the "
          "model actually reads).")
    print("  DILUTED BY WIDTH: a wider buffer scores lower per chunk while "
          "holding strictly more.  Compare within K.")
    print("=" * 100)
    print(hdr)
    for label, rec in report["cells"].items():
        print(row(label, rec, "R"))

    print("\n" + "=" * 100)
    print("geometry, cost and capacity")
    print("=" * 100)
    print(f"  {'cell':<26}{'W':>4}{'K':>6}{'cmpr':>7}{'read':>6}{'pack':>7}"
          f"{'rank':>7}{'cos':>7}{'h10':>6}{'wrts':>7}{'params':>10}  flag")
    for label, rec in report["cells"].items():
        print(f"  {label:<26}{rec['n_vec']:>4}{rec['n_slots']:>6}"
              f"{rec['compression']:>7.0f}{rec['read_cols_per_chunk']:>6}"
              f"{rec['packed_cols']:>7}{rec['eff_rank_entropy']:>7.2f}"
              f"{rec['centred_cosine']:>7.3f}"
              f"{str(rec['horizon_10pct']) + ('+' if rec['horizon_censored'] else ''):>6}"
              f"{rec['retained_writes']:>7.2f}{rec['params']:>10}"
              f"  {'RANK-PINNED' if rec['eff_rank_entropy'] < 1.5 * rec['n_vec'] else ''}")

    print("\n  NOTE: `wrts` is gameable by a high-variance gate init -- kaiming\n"
          "  leaves a random subset of channels near-frozen, which reads as\n"
          "  retention but is not memory design.  Read it next to `rank`, and\n"
          "  treat a RANK-PINNED row's mass as duplication, not capacity.")

    gated = {k: v for k, v in report["cells"].items() if "ig_mean" in v}
    if gated:
        print("\n" + "=" * 100)
        print("gate at init (untrained -- this is an ARCHITECTURE reading, not "
              "a prediction)")
        print("=" * 100)
        print(f"  {'cell':<26}{'ig':>7}{'fg':>7}{'fg p10':>8}{'fg p90':>8}"
              f"{'|s|mean':>9}{'tanh sat':>10}{'cand/wr':>9}")
        for label, rec in gated.items():
            print(f"  {label:<26}{rec['ig_mean']:>7.3f}{rec['fg_mean']:>7.3f}"
                  f"{rec['fg_p10']:>8.3f}{rec['fg_p90']:>8.3f}"
                  f"{rec['state_abs_mean']:>9.3f}"
                  f"{rec['tanh_saturated_frac']:>10.3f}"
                  f"{rec['cand_norm_ratio']:>9.3f}")

    if args.out:
        os.makedirs(args.out, exist_ok=True)
        with io.open(os.path.join(args.out, "results.json"), "w",
                     encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)
        print(f"\nwrote {os.path.join(args.out, 'results.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
