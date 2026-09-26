"""
Z attempt 2, D3 -- DOES THE CARRIED Z HOLD THE REGISTER FILE?  Forward-only.
j4_handoff.md, step 1.  Decides what J4 WRITES.

WHY.  J1 and J3 both returned NO on the E-off cell: with E removed, Z carried
nothing about the registers (X_noread +0.0003 both times).  That is equally
explained by "the read could not find the content" and "the write never held
it".  Both designs carried the loop state at the chunk's LAST 64 REAL TOKENS
(J1 s_T, J3 the scratchpad there), positions whose states are shaped for
next-token prediction, while E is written from summary slots that exist to
summarise the chunk.  Nobody has checked whether the carried Z contains the
register values.  This does, with E as the positive control.

THE TASK'S SHAPE MATTERS (tools/prepare_carry_task.py, R = 16, one op per
line).  A 512-token chunk holds ~85 lines (measured), so nearly every
register is updated inside every chunk, and what the next chunk needs is each
register's LAST value.  Measured on 200 generated rows at 8 chunks: 49.7% of
those last updates land in the chunk's final 64 tokens -- exactly the window the
tokens/scratch write pools -- 49.9% earlier in the chunk, 0.4% in an earlier
chunk, 0.06% never.  So every number is also reported by WHERE the register was
last updated:

    recent   last update inside the chunk's final `tok_rows` tokens  (~50%)
    chunk    last update earlier in this chunk                       (~50%)
    older    last update in an earlier chunk                          (0.4%)
    never    never updated (value 0)                                  (0.06%)

A last-tokens write can be expected to hold at most the `recent` half; E and
Z_end, written from columns that attend over the whole chunk, can hold both.

ENCODINGS, per (row, chunk), each [W, D] and probed flattened:
    E        the E write (merge's new_vecs) -- THE POSITIVE CONTROL: E demonstrably
             carries this task (carry NLL 0.37-0.85 vs ln10 2.30 in J1/J3)
    Z_write  the Z write this checkpoint was trained with (merge's new_latent):
             J1 = tokens (s_T pooled), J3 = scratch (m_T pooled)
    Z_tok    s_T at the last `tok_rows` real tokens, pooled by `tok_pool`
    Z_end    s_T at the SUMMARY columns -- the latent twin of E, never yet carried
and, from chunk RING_FROM_CHUNK on (the ring is full), the carried state the
NEXT chunk reads:
    E_ring   the ring's E half [K, D]
    Z_ring   the ring's Z half [K, D]

THE PROBE.  For every register a 10-way linear classifier: dual (kernel) ridge
on per-dim z-scored features onto the one-hot value, argmax.  Five-fold, split
BY ROW (every row is an independent document; no row is on both sides), lambda
by row-grouped CV inside the training folds, statistics from training folds
only.  Every sample gets one out-of-fold prediction.  The baseline is the
majority value per register from the training folds.  Margins are accuracy
minus that baseline; CIs are a row bootstrap.

THE CUTS (chosen, written before the run -- j4_handoff.md step 1):
    PROBE HEALTH   E's margin over all registers >= HEALTH_MARGIN, CI_lo > 0.
                   Otherwise nothing below is read.
    HOLDS          a Z encoding's margin CI_lo > 0 AND >= HOLDS_FRAC x E's margin
    PARTIAL        CI_lo > 0, below that
    NONE           CI includes 0

WHAT EACH OUTCOME TRIGGERS (j4_handoff.md):
    Z_write HOLDS            the write was fine and the READ failed -> J4 reads it
                             through the pretrained attention (input_embeds)
    Z_write not, Z_end HOLDS the write POSITION was wrong -> J4 writes Z_end
    no latent encoding HOLDS the loop state does not hold the register file at
                             the chunk end -> the WRITE must change before J4
    PROBE BROKEN             fix the probe; read nothing

PROBE (b), --incremental -- THE REDUNDANCY PROBE (j4_prereg.md S4.6, S9.1).
D3 showed Z_end's register information is PRESENT.  It never showed it is
ADDITIONAL, and S4.6 holds a J4 PASS against exactly that: a re-coded copy of E
still passes the E-off cell, because a copy carries the task when the original
is blanked.  Opt-in, so a D3 re-run without it is bit-identical to the committed
results.  Three readings on the SAME captures, no extra forward pass:

    INCREMENT   margin([E, Z_end]) - margin(E), a PAIRED row bootstrap.  THIS
                IS THE REGISTERED READING: knob-free, threshold-free, no
                reduction and no ridge of its own.
                  ADDITIONAL  CI_lo > 0   Z_end exposes register content E does not
                  REDUNDANT   CI spans 0  nothing beyond E
                  Z_END_EMPTY Z_end does not decode at all -- nothing to be
                              redundant WITH; read D3 first
    resid_frac  the share of Z_end's reduced variance that survives PROJECTING
                E out, out of fold.  ~1e-6 means Z_end's subspace lies inside
                E's: a linear re-coding, the sharpest REDUNDANT there is.
    E_pca       the reduction's control.  The reduction is a rotation and the
                linear kernel is invariant to one, so at k >= rank it is
                lossless; E_pca below E's margin means RESID_PCA_K was too
                small, never that reducing distorted anything.

WHY THE RESIDUAL IS A VARIANCE SHARE AND NOT A PROBE.  Kernel ridge is jointly
scale-equivariant in (K, lambda) and this probe picks lambda relative to its own
kernel, so probing a residual cannot tell "no signal" from "the same signal,
1e-6 as large": a numerically-zero residual decodes exactly as well as the
original.  A ridge is also the wrong operator -- it SHRINKS where this needs a
PROJECTION -- and on raw features (d = W x D ~ 65k against ~1.7k training rows)
"regress E out" is not even identified.  Hence the PCA reduction, the projection
at lam_rel 1e-6, and a magnitude reading rather than a decode.

ONE STATED LIMIT.  Two INDEPENDENTLY NOISY views of one code do carry more than
one, so this probe would read that ADDITIONAL -- correctly.  It cannot arise
between E and Z_end: E is coda + ln_f applied to Z_end at the same columns, a
DETERMINISTIC map, so E holds no noise Z_end does not.  ADDITIONAL there means
"the coda made register content linearly inaccessible that Z_end still
exposes", not "statistically independent content".  tests/test_z3_registers.py
pins both cases.

SELF-CHECKS, each refuses the run rather than reporting around it:
  * the register replay's answer positions must equal the pack's own
    `answer_dep` answer positions, row by row;
  * every replayed update must satisfy v == (previous v + d) mod 10;
  * on a 'tokens' checkpoint, Z_write must equal Z_tok (same columns, same
    pooling), or the capture is not reading the columns the write reads;
  * capture()'s own layout and tape checks (diag_z_content.py).

USAGE
  python evals/diag_z_registers.py --model_name ckpts/olmo-retrofit-cortex \\
      --checkpoint cortex-retrofit/j3-a3z-scratch-real/checkpoint_93552 \\
      --data data/carry_task_r16_len4096_val <--set flags> --out_dir ...
  python evals/diag_z_registers.py ... --incremental        # + probe (b)
  python evals/diag_z_registers.py --summarize eval_results/z3_registers
  Launcher: pace/diag_z_registers.sbatch (array over four trained checkpoints).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from datetime import datetime
from typing import Optional

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "tools"))

N_REGS = 16
N_VALUES = 10
#: PROBE HEALTH: E's accuracy margin over the majority baseline, all registers.
#: E carries the task at 0.37-0.85 nats (~43-69% mass on the right digit), so
#: a working linear probe on its write should clear 0.15 easily.  Chosen.
HEALTH_MARGIN = 0.15
#: HOLDS: a Z encoding's margin at least this share of E's.  Chosen.
HOLDS_FRAC = 0.5
WRITE_ENCODINGS = ("E", "Z_write", "Z_tok", "Z_end")
RING_ENCODINGS = ("E_ring", "Z_ring")
Z_WRITES = ("Z_write", "Z_tok", "Z_end")
#: 1-based chunk from which the ring is full (K/W = 64/16 = 4 laps' worth).
RING_FROM_CHUNK = 4
CATEGORIES = ("recent", "chunk", "older", "never")
LAM_REL = (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)
#: PROBE (b), THE REDUNDANCY PROBE -- j4_prereg.md S4.6 and S9.1.  D3 showed
#: Z_end's register information is PRESENT.  It never showed it is ADDITIONAL,
#: and S4.6 holds a J4 PASS against exactly that: if Z_end is largely a re-coded
#: copy of E it still passes the E-off cell, because a copy carries the task
#: when the original is blanked.  Two readings on the SAME captures:
#:
#:   INCREMENT  margin([E, Z_end]) - margin(E), a PAIRED row bootstrap (the two
#:              probes see the same rows, so the difference is paired and its CI
#:              is far tighter than the two margins' CIs apart).
#:   RESIDUAL   Z_end's margin after E is regressed out of it -- per fold, on
#:              TRAINING rows only (_resid_kernel), so a residual that decodes
#:              cannot be leakage.
#:
#: Both cuts are CI-based and threshold-free ON PURPOSE.  D3's one thresholded
#: cut (HOLDS_FRAC 0.5) is precisely where its mechanical majority verdict and
#: its substantive pattern disagreed; this probe does not repeat that.
#:   ADDITIONAL   both CI_lo > 0    Z_end holds register information E does not
#:   REDUNDANT    both CIs span 0   consistent with a re-coded copy of E
#:   MIXED        exactly one       report both, no single label
#:   Z_END_EMPTY  Z_end's own margin CI spans 0 -- nothing to be redundant WITH
#: INCREMENT_SHARE (increment / E's margin) is DESCRIPTIVE and never a cut.
#:
#: STATED CHOICE: [E, Z_end] concatenates two equal-width blocks ([16, D] each),
#: and _kernel z-scores per dim then divides by the total dim count, so the two
#: blocks enter at equal weight per dimension.  The RESIDUAL reading does not
#: depend on that weighting at all, which is why both are reported.
#: THE REGISTERED READING IS THE INCREMENT.  It is knob-free: no reduction, no
#: ridge of its own, threshold-free, and paired.  S9.1 offers "[E, Z_end]
#: concatenated vs E alone, OR Z_end's residual after regressing out E"; the
#: residual needs a dimension choice to be identified at all (see
#: _resid_kernel), so it is REPORTED as a diagnostic and never labels the run.
#: A residual that disagrees with the increment is printed as a disagreement,
#: which is the one thing D3 taught about mechanical labels.
RESID_PCA_K = 128
RESID_LAM_REL = 1e-6
#: The residual is read as a VARIANCE SHARE, not by probing it.  Kernel ridge is
#: jointly scale-equivariant in (K, lambda) and the probe picks lambda relative
#: to its own kernel, so probing a residual cannot distinguish "no signal" from
#: "the same signal, 10^-6 as large": a numerically-zero residual still decodes.
#: What is well posed is HOW MUCH of Z_end's reduced variance survives
#: projecting E out, out of fold.  Below RESID_FRAC_MIN the residual holds no
#: direction of its own and its probe is not readable -- which is itself the
#: sharpest possible REDUNDANT, and threshold-free in practice (a re-coding
#: lands at ~1e-6, not near this cut).
RESID_FRAC_MIN = 1e-3
INCREMENTAL_ENCODINGS = ("E", "E_plus_Z_end", "E_pca", "Z_end_resid_E")


# ─── the task: replaying the registers ──────────────────────────────────────

def piece_table(tok, n_regs: int = N_REGS) -> dict:
    """The task's single-token pieces, via the generator's own checker."""
    from prepare_carry_task import DIGITS, REG_NAMES, piece_ids
    ids = piece_ids(tok, REG_NAMES[:n_regs] + DIGITS + ["+", "=", "\n"])
    return {"regs": [ids[r] for r in REG_NAMES[:n_regs]],
            "digits": [ids[d] for d in DIGITS],
            "plus": ids["+"], "eq": ids["="], "nl": ids["\n"]}


def replay(ids: list, pt: dict) -> list:
    """Every register update in a row: [(answer_pos, reg, value)], in order.

    A line is REG + d [+ d ...] = v NL, rows start on a line boundary and the
    last line may be cut.  The register is the token at the line start; the
    value is the digit after '='.  Checked twice: v == (previous v + sum d)
    mod 10 here, and the answer positions against the pack's answer_dep in
    `check_answer_positions`.
    """
    reg_of = {t: i for i, t in enumerate(pt["regs"])}
    dig_of = {t: i for i, t in enumerate(pt["digits"])}
    value = [0] * len(pt["regs"])
    out, cur, acc = [], None, 0
    for p, t in enumerate(ids):
        if (p == 0 or ids[p - 1] == pt["nl"]) and t in reg_of:
            cur, acc = reg_of[t], 0
        elif cur is not None and t in dig_of and p > 0 and ids[p - 1] == pt["plus"]:
            acc += dig_of[t]
        elif t == pt["eq"] and cur is not None and p + 1 < len(ids) \
                and ids[p + 1] in dig_of:
            v = dig_of[ids[p + 1]]
            if v != (value[cur] + acc) % N_VALUES:
                raise ValueError(
                    f"replay: at position {p + 1} register {cur} reads {v}, but "
                    f"its previous value {value[cur]} plus {acc} gives "
                    f"{(value[cur] + acc) % N_VALUES}.  The line grammar is not "
                    "what this parser assumes.")
            value[cur] = v
            out.append((p + 1, cur, v))
    return out


def check_answer_positions(updates: list, answer_dep: list,
                           not_answer: int = -2) -> None:
    got = [p for p, _, _ in updates]
    want = [p for p, d in enumerate(answer_dep) if d != not_answer]
    if got != want:
        n = next((i for i, (a, b) in enumerate(zip(got, want)) if a != b),
                 min(len(got), len(want)))
        raise ValueError(
            f"replay found {len(got)} answers, the pack's answer_dep marks "
            f"{len(want)}; first disagreement at index {n}.  The replay is not "
            "reading the rows the way the generator wrote them.")


def chunk_targets(updates: list, n_chunks: int, L: int,
                  tok_rows: int) -> list:
    """Per chunk i (0-based): (values [N_REGS], categories [N_REGS]) as of the
    chunk's END.  Chunk i's input is ids[i*L:(i+1)*L] (x = ids[:-1], chunked
    exactly as eval_carry_2x2 and diag_z_content chunk it), so an update is
    visible to chunk i iff its answer position is < (i+1)*L."""
    out = []
    for i in range(n_chunks):
        end, start = (i + 1) * L, i * L
        values, last = [0] * N_REGS, [None] * N_REGS
        for p, r, v in updates:
            if p < end:
                values[r], last[r] = v, p
        cats = []
        for r in range(N_REGS):
            p = last[r]
            cats.append("never" if p is None else
                        "recent" if p >= end - tok_rows else
                        "chunk" if p >= start else "older")
        out.append((values, cats))
    return out


# ─── the capture ────────────────────────────────────────────────────────────

def collect(model, cortex, rows: list, pt: dict, n_chunks: int, num_steps,
            device, seed: int, tok_rows: int, tok_pool: int,
            log_every: int = 25) -> tuple[list, dict]:
    """rows: [(row_index, ids list, answer_dep list or None)].
    Returns (samples, self_check).  One sample per (row, chunk); the ring
    encodings only from chunk RING_FROM_CHUNK on."""
    from diag_z_content import capture
    D = int(cortex.prefix.hidden_size)
    enc = str(getattr(cortex, "latent_encoding", "delta"))
    samples, check = [], {"tokens_write_equals_z_tok": None}
    for n_done, (ri, ids, dep) in enumerate(rows):
        updates = replay(ids, pt)
        if dep is not None:
            check_answer_positions(updates, dep)
        x = torch.tensor(ids[:-1], dtype=torch.long)
        L = x.numel() // n_chunks
        targets = chunk_targets(updates, n_chunks, L, tok_rows)
        state = None
        for i, xc in enumerate(torch.chunk(x[:L * n_chunks], n_chunks)):
            a = capture(model, cortex, xc, state, num_steps, device,
                        seed + 1_000_003 * ri + 101 * i, tok_rows, tok_pool)
            if enc == "tokens" and check["tokens_write_equals_z_tok"] is None:
                ok = bool(torch.allclose(a["Z_delta"], a["Z_tok"],
                                         atol=1e-4, rtol=1e-4))
                check["tokens_write_equals_z_tok"] = ok
                if not ok:
                    raise RuntimeError(
                        "self-check failed: on a 'tokens' checkpoint the Z write "
                        "must equal s_T pooled at the last tok_rows tokens, and "
                        "it does not -- the capture is not reading the columns "
                        "the write reads (check --tok_rows/--tok_pool).")
            values, cats = targets[i]
            rec = {"row": ri, "chunk": i, "values": values, "cats": cats,
                   "E": a["E"].half(), "Z_write": a["Z_delta"].half(),
                   "Z_tok": a["Z_tok"].half(), "Z_end": a["Z_end"].half()}
            state = a["state"]
            if i + 1 >= RING_FROM_CHUNK and state is not None:
                s = state[0].float().cpu()
                rec["E_ring"], rec["Z_ring"] = s[:, :D].half(), s[:, D:].half()
            samples.append(rec)
        if log_every and (n_done + 1) % log_every == 0:
            print(f"  {n_done + 1}/{len(rows)} rows", flush=True)
    return samples, check


# ─── the probe ──────────────────────────────────────────────────────────────

def row_folds(rows: torch.Tensor, n_folds: int) -> torch.Tensor:
    """Fold id per sample, by ROW: every sample of a row lands in one fold."""
    uniq = torch.unique(rows)
    fold_of = {int(r): i % n_folds for i, r in enumerate(uniq.tolist())}
    return torch.tensor([fold_of[int(r)] for r in rows.tolist()])


def _zs(X: torch.Tensor, tr: torch.Tensor, device) -> torch.Tensor:
    """Per-dim z-score with TRAINING-fold statistics only."""
    X = X.to(device=device, dtype=torch.float32)
    mu = X[tr].mean(dim=0)
    sd = X[tr].std(dim=0).clamp_min(1e-6)
    return (X - mu) / sd


def _kernel(X: torch.Tensor, tr: torch.Tensor, device) -> torch.Tensor:
    X = _zs(X, tr, device)
    return (X @ X.T).double() / X.shape[1]


def _pca_scores(X: torch.Tensor, tr: torch.Tensor, device,
                k: int) -> torch.Tensor:
    """Top-`k` PCA scores, with the centring and the basis fit on the TRAINING
    rows only.  The features are [W, D] flattened -- tens of thousands of dims
    -- so the [d, d] covariance is never formed: eigh of the training Gram
    gives the same basis, and every score is an inner product.

        G = Xc_tr Xc_tr^T = U L U^T,  V_k = Xc_tr^T U_k L_k^-1/2
        S  = Xc V_k = (Xc Xc_tr^T) U_k L_k^-1/2
    """
    Xz = _zs(X, tr, device)
    ti = torch.nonzero(tr.to(Xz.device), as_tuple=True)[0]
    Xc = (Xz - Xz.index_select(0, ti).mean(dim=0)).double()
    Xtr = Xc.index_select(0, ti)
    ev, U = torch.linalg.eigh(Xtr @ Xtr.T)
    keep = ev > ev.max() * 1e-10
    ev, U = ev[keep], U[:, keep]
    kk = min(k, ev.numel())
    ev, U = ev[-kk:], U[:, -kk:]                       # eigh returns ascending
    return (Xc @ Xtr.T) @ U / ev.clamp_min(1e-12).sqrt()


def _pca_kernel(X: torch.Tensor, tr: torch.Tensor, device, k: int) -> torch.Tensor:
    """The linear kernel of X's top-`k` training-fit PCA scores."""
    S = _pca_scores(X, tr, device, k)
    return (S @ S.T) / S.shape[1]


def _resid_kernel(X: torch.Tensor, C: torch.Tensor, tr: torch.Tensor, device,
                  k: int = None, lam_rel: float = None) -> torch.Tensor:
    """The kernel of X AFTER C is PROJECTED OUT of it, both reduced to their
    top-`k` training-fit PCA scores first (probe (b)'s RESIDUAL).

    WHY THE REDUCTION IS NOT OPTIONAL.  On the raw features d = W x D is ~65k
    against ~1.7k training samples, so `C_tr` spans the whole sample space and
    "regress C out of X" is not identified: at any small ridge the training
    residual is ~0 whatever the true relation.  And a ridge is the wrong
    operator anyway -- it SHRINKS where this needs a PROJECTION.  The probe
    z-scores its features and picks its own lambda by CV, so it is scale-free:
    a residual that is a shrunken copy of the signal decodes exactly as well as
    the signal, and a re-coded copy would read ADDITIONAL.  Reducing both sides
    to k << n_tr components makes the least-squares fit identified, and then a
    deterministic re-coding of C cancels to ~0 as it must.

    The cost of the reduction is that it can only speak about the retained
    subspaces; `E_pca` is probed beside it as the control (if E's margin
    survives the reduction, the subspace that carries the registers is inside
    it).  This is why the REGISTERED reading is the INCREMENT and this residual
    is reported as a diagnostic -- j4_prereg.md S9.1 offers either.
    """
    k = RESID_PCA_K if k is None else k
    lam_rel = RESID_LAM_REL if lam_rel is None else lam_rel
    Sx = _pca_scores(X, tr, device, k)
    Sc = _pca_scores(C, tr, device, k)
    ti = torch.nonzero(tr.to(Sx.device), as_tuple=True)[0]
    Xt, Ct = Sx.index_select(0, ti), Sc.index_select(0, ti)
    A = Ct.T @ Ct
    A = A + lam_rel * A.diagonal().mean() * torch.eye(
        A.shape[0], dtype=A.dtype, device=A.device)
    B = torch.linalg.solve(A, Ct.T @ Xt)
    R = Sx - Sc @ B
    return (R @ R.T) / R.shape[1]


def resid_frac(X: torch.Tensor, C: torch.Tensor, rows: torch.Tensor, device,
               n_folds: int = 5, k: int = RESID_PCA_K,
               lam_rel: float = RESID_LAM_REL) -> float:
    """The share of X's reduced variance that survives projecting C out of it,
    measured OUT OF FOLD (the projection is fit on training rows, the share is
    read on held-out rows).  ~0 means X's retained subspace lies inside C's:
    X is a linear re-coding of C and holds no direction of its own."""
    folds = row_folds(rows, n_folds)
    num = den = 0.0
    for f in range(n_folds):
        tr, te = folds != f, folds == f
        if not bool(te.any()):
            continue
        Sx = _pca_scores(X, tr, device, k)
        Sc = _pca_scores(C, tr, device, k)
        ti = torch.nonzero(tr.to(Sx.device), as_tuple=True)[0]
        Xt, Ct = Sx.index_select(0, ti), Sc.index_select(0, ti)
        A = Ct.T @ Ct
        A = A + lam_rel * A.diagonal().mean() * torch.eye(
            A.shape[0], dtype=A.dtype, device=A.device)
        R = Sx - Sc @ torch.linalg.solve(A, Ct.T @ Xt)
        ted = te.to(Sx.device)
        num += float(R[ted].pow(2).sum())
        den += float(Sx[ted].pow(2).sum())
    return num / den if den > 0 else float("nan")


def probe(X: torch.Tensor, Y: torch.Tensor, rows: torch.Tensor, device,
          n_folds: int = 5, lam_rel=LAM_REL,
          covar: Optional[torch.Tensor] = None,
          resid_lam_rel: float = RESID_LAM_REL,
          pca_k: Optional[int] = None) -> dict:
    """X [n, d], Y [n, N_REGS] values, rows [n].  Out-of-fold predictions for
    every sample from a kernel ridge onto the one-hot values, lambda chosen by
    row-grouped CV inside each training fold; the majority baseline the same
    way.  Returns {"pred", "maj"} [n, N_REGS] and the lambdas chosen.

    `covar` [n, d_c] (probe (b)): project it out of X per fold, on that fold's
    training rows only, and probe the residual.  `pca_k` alone: probe X's top-k
    training-fit PCA scores (the reduction's own control).  With both None this
    function is bit-identical to the D3 version that produced the committed
    results."""
    n = X.shape[0]
    folds = row_folds(rows, n_folds)
    Yoh = torch.nn.functional.one_hot(Y, N_VALUES).reshape(n, -1).double()
    pred = torch.full_like(Y, -1)
    maj = torch.full_like(Y, -1)
    lams_chosen = []
    lam_rel_t = torch.tensor(lam_rel, dtype=torch.float64)
    for f in range(n_folds):
        tr, te = folds != f, folds == f
        if not bool(te.any()):
            continue
        if covar is not None:
            K = _resid_kernel(X, covar, tr, device, pca_k, resid_lam_rel)
        elif pca_k is not None:
            K = _pca_kernel(X, tr, device, pca_k)
        else:
            K = _kernel(X, tr, device)
        trd, ted = tr.to(K.device), te.to(K.device)
        Ktr = K[trd][:, trd]
        lams = lam_rel_t.to(K.device) * Ktr.diagonal().mean()
        inner = row_folds(rows[tr], min(n_folds - 1, 4)).to(K.device)
        Ytr = Yoh[tr].to(K.device)
        cv = torch.zeros(len(lams), dtype=torch.float64, device=K.device)
        for g in range(int(inner.max()) + 1):
            fit, val = inner != g, inner == g
            if not bool(val.any()) or not bool(fit.any()):
                continue
            ev, U = torch.linalg.eigh(Ktr[fit][:, fit])
            mu = Ytr[fit].mean(dim=0)
            UtY = U.T @ (Ytr[fit] - mu)
            KvU = Ktr[val][:, fit] @ U
            for li, lam in enumerate(lams):
                p = KvU @ (UtY / (ev + lam).unsqueeze(1)) + mu
                cv[li] += (Ytr[val] - p).pow(2).sum()
        li = int(torch.argmin(cv))
        lams_chosen.append(float(lam_rel_t[li]))
        ev, U = torch.linalg.eigh(Ktr)
        mu = Ytr.mean(dim=0)
        scores = K[ted][:, trd] @ U @ ((U.T @ (Ytr - mu)) / (ev + lams[li]).unsqueeze(1)) + mu
        pred[te] = scores.reshape(-1, N_REGS, N_VALUES).argmax(dim=-1).cpu()
        mode = torch.mode(Y[tr], dim=0).values                    # [N_REGS]
        maj[te] = mode.unsqueeze(0).expand(int(te.sum()), -1)
    return {"pred": pred, "maj": maj, "lambda_rel": lams_chosen}


def boot_margin(correct: torch.Tensor, maj_correct: torch.Tensor,
                rows: torch.Tensor, mask: torch.Tensor, n_boot: int,
                seed: int) -> Optional[dict]:
    """Accuracy, baseline and margin over the (sample, register) cells in
    `mask`, with a ROW bootstrap.  None when the mask is empty."""
    uniq, inv = torch.unique(rows, return_inverse=True)
    m = mask.double()
    c = torch.zeros(len(uniq), dtype=torch.float64).index_add_(
        0, inv, (correct.double() * m).sum(1))
    b = torch.zeros(len(uniq), dtype=torch.float64).index_add_(
        0, inv, (maj_correct.double() * m).sum(1))
    k = torch.zeros(len(uniq), dtype=torch.float64).index_add_(0, inv, m.sum(1))
    if float(k.sum()) == 0:
        return None
    g = torch.Generator().manual_seed(seed)
    idx = torch.randint(0, len(uniq), (n_boot, len(uniq)), generator=g)
    marg = (c[idx].sum(1) - b[idx].sum(1)) / k[idx].sum(1).clamp_min(1)
    lo, hi = torch.quantile(marg, torch.tensor([0.025, 0.975], dtype=torch.float64))
    return {"acc": float(c.sum() / k.sum()), "baseline": float(b.sum() / k.sum()),
            "margin": float((c.sum() - b.sum()) / k.sum()),
            "lo": float(lo), "hi": float(hi), "cells": int(k.sum())}


def boot_delta(correct_a: torch.Tensor, correct_b: torch.Tensor,
               rows: torch.Tensor, mask: torch.Tensor, n_boot: int,
               seed: int) -> Optional[dict]:
    """PAIRED row bootstrap on accuracy(a) - accuracy(b) over the cells in
    `mask` (probe (b)'s INCREMENT).  Both encodings are scored on the SAME
    resampled rows, so the shared row-to-row variance cancels; the majority
    baseline is identical for both, so this difference IS the margin
    difference.  Same generator and seed as boot_margin, so the resamples
    match the margins reported beside it.  None when the mask is empty."""
    uniq, inv = torch.unique(rows, return_inverse=True)
    m = mask.double()
    a = torch.zeros(len(uniq), dtype=torch.float64).index_add_(
        0, inv, (correct_a.double() * m).sum(1))
    b = torch.zeros(len(uniq), dtype=torch.float64).index_add_(
        0, inv, (correct_b.double() * m).sum(1))
    k = torch.zeros(len(uniq), dtype=torch.float64).index_add_(0, inv, m.sum(1))
    if float(k.sum()) == 0:
        return None
    g = torch.Generator().manual_seed(seed)
    idx = torch.randint(0, len(uniq), (n_boot, len(uniq)), generator=g)
    d = (a[idx].sum(1) - b[idx].sum(1)) / k[idx].sum(1).clamp_min(1)
    lo, hi = torch.quantile(d, torch.tensor([0.025, 0.975], dtype=torch.float64))
    return {"delta": float((a.sum() - b.sum()) / k.sum()),
            "lo": float(lo), "hi": float(hi), "cells": int(k.sum())}


def status(z: Optional[dict], e: Optional[dict]) -> str:
    if z is None or e is None:
        return "n/a"
    if z["lo"] <= 0:
        return "NONE"
    return "HOLDS" if z["margin"] >= HOLDS_FRAC * e["margin"] else "PARTIAL"


def healthy(e: Optional[dict]) -> bool:
    return e is not None and e["margin"] >= HEALTH_MARGIN and e["lo"] > 0


def reading(rep: dict) -> str:
    """The per-checkpoint reading, from the WRITE probes (all registers)."""
    w = rep["write"]
    if not healthy(w["E"]["all"]):
        return "PROBE_BROKEN"
    s = {k: w[k]["status"] for k in Z_WRITES}
    if s["Z_write"] == "HOLDS":
        return "Z_WRITE_HOLDS"
    if s["Z_end"] == "HOLDS":
        return "ONLY_Z_END_HOLDS"
    if s["Z_tok"] == "HOLDS":
        return "ONLY_Z_TOK_HOLDS"
    return "NO_LATENT_HOLDS"


def redundancy(rep: dict) -> str:
    """Probe (b)'s reading (j4_prereg.md S4.6).  Both cuts CI-based."""
    w, inc = rep.get("write") or {}, rep.get("incremental") or {}
    if not inc or not w:
        return "NOT_RUN"
    if not healthy(w.get("E", {}).get("all")):
        return "PROBE_BROKEN"
    zend = w.get("Z_end", {}).get("all")
    if zend is None or zend["lo"] <= 0:
        return "Z_END_EMPTY"
    d = inc.get("increment", {}).get("all")
    if d is None:
        return "NOT_RUN"
    return "ADDITIONAL" if d["lo"] > 0 else "REDUNDANT"


TRIGGERS_B = {
    "ADDITIONAL": "Z_end holds register information E does not -- a J4 PASS "
                  "reads as a real second channel (j4_prereg.md S4.6)",
    "REDUNDANT": "no detectable information beyond E -- a J4 PASS reads as "
                 "YES_BUT_REDUNDANT: an expensive re-coded copy, not a channel",
    "Z_END_EMPTY": "Z_end does not decode the registers here at all, so there "
                   "is nothing for it to be redundant WITH -- read D3 first",
    "PROBE_BROKEN": "E does not decode either: the probe is broken -- read nothing",
    "NOT_RUN": "probe (b) was not run on this checkpoint (--incremental)",
}


TRIGGERS = {
    "Z_WRITE_HOLDS": "the carried write held the registers and the READ failed "
                     "-> J4 reads this write through input_embeds",
    "ONLY_Z_END_HOLDS": "the write POSITION was wrong -> J4 writes Z_end (s_T at "
                        "the summary columns)",
    "ONLY_Z_TOK_HOLDS": "s_T at the last tokens holds them but this checkpoint's "
                        "write lost them (the scratchpad) -> J4 writes s_T",
    "NO_LATENT_HOLDS": "the loop state at the chunk end does not hold the "
                       "register file -> the WRITE must change before J4",
    "PROBE_BROKEN": "E does not decode either: the probe is broken -- read nothing",
}


def report_encodings(samples: list, names: tuple, device, n_folds: int,
                     n_boot: int, seed: int) -> dict:
    use = [s for s in samples if names[0] in s]
    if not use:
        return {}
    rows = torch.tensor([s["row"] for s in use])
    Y = torch.tensor([s["values"] for s in use], dtype=torch.long)
    cats = [s["cats"] for s in use]
    masks = {"all": torch.ones(len(use), N_REGS, dtype=torch.bool)}
    for c in CATEGORIES:
        masks[c] = torch.tensor([[x == c for x in cs] for cs in cats])
    out, correct = {}, {}
    for name in names:
        X = torch.stack([s[name].reshape(-1) for s in use])
        p = probe(X, Y, rows, device, n_folds)
        correct[name] = (p["pred"] == Y)
        maj_ok = (p["maj"] == Y)
        out[name] = {"lambda_rel": p["lambda_rel"],
                     **{k: boot_margin(correct[name], maj_ok, rows, m, n_boot, seed)
                        for k, m in masks.items()}}
        print(f"  probed {name}: acc {out[name]['all']['acc']:.3f} vs baseline "
              f"{out[name]['all']['baseline']:.3f}", flush=True)
    e_key = names[0]
    for name in names:
        out[name]["status"] = ("control" if name == e_key else
                               status(out[name]["all"], out[e_key]["all"]))
    out["_n"] = {"samples": len(use), "rows": int(torch.unique(rows).numel()),
                 **{c: int(masks[c].sum()) for c in CATEGORIES}}
    return out


def report_incremental(samples: list, device, n_folds: int, n_boot: int,
                       seed: int, resid_lam_rel: float = RESID_LAM_REL,
                       pca_k: int = RESID_PCA_K) -> dict:
    """Probe (b): is Z_end's register content ADDITIONAL to E's, or a copy?

    Four probes on the same (row, chunk) samples the write block used: E alone,
    [E, Z_end] concatenated, E reduced to `pca_k` components (the reduction's
    control) and Z_end with E projected out inside the same reduction.  The
    REGISTERED reading is the INCREMENT, the paired difference of the first
    two; the residual is a diagnostic beside it (see _resid_kernel)."""
    use = [s for s in samples if "E" in s and "Z_end" in s]
    if not use:
        return {}
    rows = torch.tensor([s["row"] for s in use])
    Y = torch.tensor([s["values"] for s in use], dtype=torch.long)
    cats = [s["cats"] for s in use]
    masks = {"all": torch.ones(len(use), N_REGS, dtype=torch.bool)}
    for c in CATEGORIES:
        masks[c] = torch.tensor([[x == c for x in cs] for cs in cats])
    Xe = torch.stack([s["E"].reshape(-1) for s in use])
    Xz = torch.stack([s["Z_end"].reshape(-1) for s in use])
    #                       features            covar   pca_k
    feats = {"E":             (Xe,                        None, None),
             "E_plus_Z_end":  (torch.cat([Xe, Xz], 1),    None, None),
             "E_pca":         (Xe,                        None, pca_k),
             "Z_end_resid_E": (Xz,                        Xe,   pca_k)}
    out, correct = {}, {}
    for name, (X, cov, pk) in feats.items():
        p = probe(X, Y, rows, device, n_folds, covar=cov,
                  resid_lam_rel=resid_lam_rel, pca_k=pk)
        correct[name] = (p["pred"] == Y)
        maj_ok = (p["maj"] == Y)
        out[name] = {"lambda_rel": p["lambda_rel"],
                     **{k: boot_margin(correct[name], maj_ok, rows, m, n_boot, seed)
                        for k, m in masks.items()}}
        print(f"  probed {name}: acc {out[name]['all']['acc']:.3f} vs baseline "
              f"{out[name]['all']['baseline']:.3f}", flush=True)
    out["increment"] = {k: boot_delta(correct["E_plus_Z_end"], correct["E"],
                                      rows, m, n_boot, seed)
                        for k, m in masks.items()}
    em = out["E"]["all"]["margin"]
    inc = out["increment"]["all"]
    out["increment_share"] = (inc["delta"] / em) if (inc and em > 0) else None
    out["resid_lam_rel"] = resid_lam_rel
    out["pca_k"] = pca_k
    # The reduction's own control: does E still decode after it?
    out["pca_retains_E"] = (out["E_pca"]["all"]["margin"]
                            >= HOLDS_FRAC * out["E"]["all"]["margin"])
    out["resid_frac"] = resid_frac(Xz, Xe, rows, device, n_folds, pca_k,
                                   resid_lam_rel)
    out["resid_readable"] = out["resid_frac"] >= RESID_FRAC_MIN
    r = out["Z_end_resid_E"]["all"]
    out["residual_disagrees"] = bool(out["resid_readable"]
                                     and (r["lo"] > 0) != (inc["lo"] > 0))
    out["_n"] = {"samples": len(use), "rows": int(torch.unique(rows).numel())}
    return out


# ─── output ─────────────────────────────────────────────────────────────────

def _fmt(e: Optional[dict]) -> str:
    if not e:
        return "      n/a"
    return (f"{e['acc']:.3f} (base {e['baseline']:.3f}) margin {e['margin']:+.3f} "
            f"[{e['lo']:+.3f},{e['hi']:+.3f}]")


def print_report(rep: dict) -> None:
    print("=" * 78)
    print(f"D3 REGISTER PROBE -- {rep['checkpoint']}")
    print(f"  encoding {rep['latent_encoding']} | {rep['write']['_n']['rows']} rows, "
          f"{rep['write']['_n']['samples']} (row, chunk) samples | cells by last "
          "update: " + ", ".join(f"{c} {rep['write']['_n'][c]}" for c in CATEGORIES))
    print("=" * 78)
    for block, names in (("write", WRITE_ENCODINGS), ("ring", RING_ENCODINGS)):
        if not rep.get(block):
            continue
        print(f"  {block.upper()} (all registers | recent | chunk)")
        for name in names:
            r = rep[block][name]
            print(f"    {name:8} {r['status']:8} all    {_fmt(r['all'])}")
            for c in ("recent", "chunk"):
                print(f"    {'':8} {'':8} {c:6} {_fmt(r[c])}")
    if rep.get("incremental"):
        inc = rep["incremental"]
        print("-" * 78)
        print("  PROBE (b) REDUNDANCY -- is Z_end's content ADDITIONAL to E's?")
        for name in INCREMENTAL_ENCODINGS:
            print(f"    {name:14} all    {_fmt(inc[name]['all'])}")
        d = inc["increment"]["all"]
        sh = inc.get("increment_share")
        print(f"    {'INCREMENT':14} [E,Z_end] - E  {d['delta']:+.3f} "
              f"[{d['lo']:+.3f},{d['hi']:+.3f}] (paired)"
              + (f"  = {sh:+.1%} of E's margin" if sh is not None else ""))
        for c in ("recent", "chunk"):
            dc = inc["increment"][c]
            if dc:
                print(f"    {'':14} {c:6}         {dc['delta']:+.3f} "
                      f"[{dc['lo']:+.3f},{dc['hi']:+.3f}]")
        print(f"    RESIDUAL diagnostic: E PROJECTED out inside the top-"
              f"{inc['pca_k']} PCA subspace, fit on training rows only "
              f"(lam_rel {inc['resid_lam_rel']})")
        print(f"      Z_end variance surviving the projection, out of fold: "
              f"{inc['resid_frac']:.2e}"
              + ("" if inc["resid_readable"] else
                 f"  < {RESID_FRAC_MIN}: Z_end's retained subspace lies INSIDE "
                 "E's, so it is a linear re-coding and the margin above it is "
                 "not readable (a scale-free probe decodes a 1e-6 residual too)"))
        print(f"      the reduction keeps E's signal ({HOLDS_FRAC} x E's "
              f"margin): {'YES' if inc['pca_retains_E'] else 'NO -- the '
              'residual line is not readable, raise pca_k'}")
        if inc["residual_disagrees"]:
            print("      NOTE: the residual and the increment DISAGREE.  The "
                  "label follows the increment (registered); the disagreement "
                  "is itself the finding -- report both.")
    print("-" * 78)
    print(f"  PROBE HEALTH (E margin >= {HEALTH_MARGIN}, CI_lo > 0): "
          f"{'OK' if healthy(rep['write']['E']['all']) else 'FAILED'}")
    print(f"  READING: {rep['reading']} -- {TRIGGERS[rep['reading']]}")
    if rep.get("redundancy"):
        print(f"  PROBE (b): {rep['redundancy']} -- {TRIGGERS_B[rep['redundancy']]}")


def summarize(root: str) -> int:
    reps = []
    for f in sorted(glob.glob(os.path.join(root, "*", "results.json"))):
        with open(f, encoding="utf-8") as fh:
            reps.append(json.load(fh))
    if not reps:
        print(f"no results.json under {root}")
        return 1
    print("=" * 78)
    print(f"D3 SUMMARY -- {len(reps)} checkpoints (margin over majority, all registers)")
    print("=" * 78)
    print(f"  {'checkpoint':40} {'enc':8} {'E':>7} {'Z_write':>16} {'Z_tok':>16} "
          f"{'Z_end':>16} {'Z_ring':>16}  reading")
    for r in reps:
        w, g = r["write"], r.get("ring") or {}
        cell = lambda d: (f"{d['all']['margin']:+.3f} {d['status'][:7]:>7}"
                          if d and d.get("all") else "n/a")
        run = r["checkpoint"].rstrip("/").split("/")[-2] if "/" in r["checkpoint"] else r["checkpoint"]
        print(f"  {run:40} {r['latent_encoding']:8} {w['E']['all']['margin']:+7.3f} "
              f"{cell(w['Z_write']):>16} {cell(w['Z_tok']):>16} {cell(w['Z_end']):>16} "
              f"{cell(g.get('Z_ring')):>16}  {r['reading']}")
    readings = {r["reading"] for r in reps}
    print("-" * 78)
    for x in sorted(readings):
        print(f"  {x}: {TRIGGERS[x]}")
    if any(r.get("incremental") for r in reps):
        print("=" * 78)
        print("  PROBE (b) REDUNDANCY -- increment = margin([E,Z_end]) - margin(E), paired")
        print(f"  {'checkpoint':40} {'E':>7} {'[E,Z_end]':>10} {'increment':>22} "
              f"{'Z_end|E':>16}  probe (b)")
        for r in reps:
            inc = r.get("incremental")
            if not inc:
                continue
            run = (r["checkpoint"].rstrip("/").split("/")[-2]
                   if "/" in r["checkpoint"] else r["checkpoint"])
            d, rs = inc["increment"]["all"], inc["Z_end_resid_E"]["all"]
            print(f"  {run:40} {inc['E']['all']['margin']:+7.3f} "
                  f"{inc['E_plus_Z_end']['all']['margin']:+10.3f} "
                  f"{d['delta']:+7.3f} [{d['lo']:+.3f},{d['hi']:+.3f}] "
                  f"{rs['margin']:+7.3f} {'*' if rs['lo'] > 0 else ' ':>2}     "
                  f"{r.get('redundancy', 'NOT_RUN')}")
        for x in sorted({r.get("redundancy", "NOT_RUN") for r in reps
                         if r.get("incremental")}):
            print(f"  {x}: {TRIGGERS_B[x]}")
    return 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--summarize", default="",
                   help="print the cross-checkpoint table from <dir>/*/results.json and exit")
    p.add_argument("--model_name", default="")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--data", default="data/carry_task_r16_len4096_val")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                   help="graft flags, as pace/j1_readout.sbatch sets them")
    p.add_argument("--n_chunks", type=int, default=8)
    p.add_argument("--T", type=int, default=8)
    p.add_argument("--max_examples", type=int, default=300)
    p.add_argument("--tok_rows", type=int, default=64)
    p.add_argument("--tok_pool", type=int, default=4)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--boot", type=int, default=2000)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--dtype", default="float32", choices=["float32"])
    p.add_argument("--out_dir", default="eval_results/z3_registers")
    p.add_argument("--allow_scrambled", action="store_true")
    p.add_argument("--incremental", action="store_true",
                   help="also run probe (b), the redundancy probe (j4_prereg.md "
                        "S4.6/S9.1): [E, Z_end] against E alone, and Z_end with "
                        "E regressed out.  Opt-in, so a D3 re-run without it is "
                        "bit-identical to the committed results.")
    p.add_argument("--resid_lam_rel", type=float, default=RESID_LAM_REL,
                   help="probe (b)'s residual ridge, relative to the covariate "
                        "Gram's mean diagonal (registered: %(default)s)")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.summarize:
        return summarize(args.summarize)
    if not args.model_name:
        raise SystemExit("--model_name is required unless --summarize")
    from transformers import AutoTokenizer
    from cortex_memory.health import refuse_if_scrambled
    from eval_carry_2x2 import has_latent_channel
    from model_utils import _unwrap, load_checkpoint, parse_config_overrides, to_num_steps

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg = load_checkpoint(args.checkpoint, args.model_name, None,
                                 torch.float32, device,
                                 config_overrides=parse_config_overrides(args.set) or None)
    inner = _unwrap(model)
    cortex = getattr(inner, "cortex", None)
    refuse_if_scrambled(cortex, "z_registers", args.allow_scrambled)
    if cortex is None or getattr(cortex, "prefix", None) is None \
            or not has_latent_channel(cortex):
        print("FAILED: no prefix buffer with a Z channel on this build (check --set).")
        return 2
    pt = piece_table(AutoTokenizer.from_pretrained(args.model_name,
                                                  trust_remote_code=True))

    from datasets import load_from_disk
    ds = load_from_disk(args.data)
    n = min(args.max_examples or len(ds), len(ds))
    has_dep = "answer_dep" in ds.column_names
    rows = [(i, list(ds[i]["input_ids"]),
             list(ds[i]["answer_dep"]) if has_dep else None) for i in range(n)]
    print(f"[z3] {args.data}: {n} rows, answer_dep {'checked' if has_dep else 'ABSENT'} | "
          f"encoding {cortex.latent_encoding} | T={args.T}", flush=True)
    samples, check = collect(model, cortex, rows, pt, args.n_chunks,
                             to_num_steps(args.T), device, args.seed,
                             args.tok_rows, args.tok_pool)
    rep = {"instrument": "Z attempt 2 D3: register probe (j4_handoff.md step 1)",
           "when": datetime.now().isoformat(timespec="seconds"),
           "model_name": args.model_name, "checkpoint": args.checkpoint,
           "data": args.data, "sets": args.set,
           "latent_encoding": str(cortex.latent_encoding),
           "config": {k: getattr(args, k) for k in (
               "n_chunks", "T", "max_examples", "tok_rows", "tok_pool", "folds",
               "boot", "seed")},
           "cuts": {"HEALTH_MARGIN": HEALTH_MARGIN, "HOLDS_FRAC": HOLDS_FRAC,
                    "RING_FROM_CHUNK": RING_FROM_CHUNK,
                    "RESID_LAM_REL": args.resid_lam_rel,
                    "RESID_PCA_K": RESID_PCA_K,
                    "RESID_FRAC_MIN": RESID_FRAC_MIN},
           "self_check": check, "answer_dep_checked": has_dep}
    rep["write"] = report_encodings(samples, WRITE_ENCODINGS, device, args.folds,
                                    args.boot, args.seed)
    rep["ring"] = report_encodings(samples, RING_ENCODINGS, device, args.folds,
                                   args.boot, args.seed)
    rep["reading"] = reading(rep)
    if args.incremental:
        print("  probe (b): the redundancy probe (j4_prereg.md S4.6)", flush=True)
        rep["incremental"] = report_incremental(samples, device, args.folds,
                                                args.boot, args.seed,
                                                args.resid_lam_rel)
        rep["redundancy"] = redundancy(rep)
    print_report(rep)
    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, "results.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(rep, fh, indent=2)
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
