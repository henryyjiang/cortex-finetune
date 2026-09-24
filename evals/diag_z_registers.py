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


def _kernel(X: torch.Tensor, tr: torch.Tensor, device) -> torch.Tensor:
    X = X.to(device=device, dtype=torch.float32)
    mu = X[tr].mean(dim=0)
    sd = X[tr].std(dim=0).clamp_min(1e-6)
    X = (X - mu) / sd
    return (X @ X.T).double() / X.shape[1]


def probe(X: torch.Tensor, Y: torch.Tensor, rows: torch.Tensor, device,
          n_folds: int = 5, lam_rel=LAM_REL) -> dict:
    """X [n, d], Y [n, N_REGS] values, rows [n].  Out-of-fold predictions for
    every sample from a kernel ridge onto the one-hot values, lambda chosen by
    row-grouped CV inside each training fold; the majority baseline the same
    way.  Returns {"pred", "maj"} [n, N_REGS] and the lambdas chosen."""
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
    print("-" * 78)
    print(f"  PROBE HEALTH (E margin >= {HEALTH_MARGIN}, CI_lo > 0): "
          f"{'OK' if healthy(rep['write']['E']['all']) else 'FAILED'}")
    print(f"  READING: {rep['reading']} -- {TRIGGERS[rep['reading']]}")


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
                    "RING_FROM_CHUNK": RING_FROM_CHUNK},
           "self_check": check, "answer_dep_checked": has_dep}
    rep["write"] = report_encodings(samples, WRITE_ENCODINGS, device, args.folds,
                                    args.boot, args.seed)
    rep["ring"] = report_encodings(samples, RING_ENCODINGS, device, args.folds,
                                   args.boot, args.seed)
    rep["reading"] = reading(rep)
    print_report(rep)
    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, "results.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(rep, fh, indent=2)
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
