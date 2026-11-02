"""
Post-mortem diagnostics 2c + 2d: WHERE the chunk-1 tax lives, and WHAT breaks
when the ring runs past its trained depth.  Forward-only.

WHY (2026-10-28).  Two of the 5B pair's results have no mechanism yet.

  THE CHUNK-1 TAX.  cortex-final pays ~+0.06 nats over the control on chunk 1
  (fwedu +0.059, pg19 +0.057 at 38,146) and the gap WIDENED over training
  (+0.039 / +0.049 at 11,444).  Every MC benchmark is a single forward, i.e.
  all chunk 1, which is where the 1-4 pt losses come from.  Under
  gate_fill=grow chunk 1 carries NO columns at all, while chunks 2+ always
  carry 16-64.  Two readings, and they predict different fixes:
    register  the model leans on carried columns as attention sinks /
              registers whatever they hold, and chunk 1 is simply missing
              them.  Prediction: ANY carry (another document's, or a mean
              one) recovers most of the tax.  Fix: a learned default carry.
    capacity  the shared weights compromise between two regimes and chunk 1
              lost.  Prediction: a foreign carry recovers nothing (or hurts).
  --parts chunk1 scores chunk 1 under five carries and profiles the tax by
  position-in-chunk (early-position tax = sink/register shape).  It also scores
  chunks 2..8 WITHOUT carry, which separates "chunk 1 is hard" from "this model
  is worse whenever the carry is absent".  It saves two fixed carries
  (mean64.pt, generic64.pt) that eval_multiple_choice.py --prefix_carry feeds
  to the MC suite -- the benchmark-level version of the same test.

  THE PRIMING-DEPTH COLLAPSE.  cortex beats the control at the trained 2 ring
  laps, is at parity by 8, worse at 16 and degenerate at ~51 laps (LongMemEval,
  first-token entropy 6.48 vs 3.26).  --parts drift runs single books for
  --drift_chunks chunks and logs, per chunk: NLL with and without carry,
  predictive entropy, per-channel carry norms / effective rank / centred
  cosine (cortex_memory.health.carry_health), and MEASURED row turnover at the
  merge (||row_after - row_before|| / ||row_before|| over the rows this chunk
  wrote) -- the runtime face of the trained gates (E fg 0.17, Z 0.45).  Which
  statistic leaves its trained range first is the mechanism.

DEPTH TESTS (2026-11-02, v2_buffer_handoff.md Part A).  WHY the carry turns
harmful from chunk ~27 on 202-chunk books when E_CLAMP / E_STORE moved the
crossover only to 29 / 30.  All on the drift's streams and seeds, so every
number pairs chunk for chunk with chunk1_drift/ (--drift_baseline checks it).
  gates     A1: the drift again, with each channel's gate call spied on: mean
            fg / ig over the touched rows, the share of entries with fg+ig > 1,
            and ||write|| / ||stored row it lands in||.  Gating starts at chunk
            5 under fill=grow, so chunks 5..8 are the trained range.
  restart   A2/A3: at each --depths chunk c, score c on the long memory
            (chained from chunk 1), a FRESH one (chunks c-7..c-1 read from
            empty), and the two channel swaps (long E + fresh Z, fresh E +
            long Z).  At c=8 fresh IS long (same chunks, same seeds): checked.
  recency   A4: at c, the lift memory gives the rare forms of the recent
            window (c-4..c-1) and of the old one (1..min(8, c-5)) over a
            control book's, log-mass with memory minus without (interp part F's
            rule).  Shares the restart forwards when both run.
  forced    A5: the drift with every gated row's update overridden: `fifo`
            replaces it (fg=0, ig=1); `hardgate` replaces it if its trained fg,
            averaged over dims, is < 0.5, else keeps it bit-identical.
  Reading: handoff Part A.  Every part writes results.json as it finishes.

CONTROL ARM.  Loaded without --set it has no prefix buffer; chunk1 then
reports only the no-carry profiles (its chunk-1 and chunks-2..8 NLL by
position), which is the comparison the cortex numbers are read against.  drift
refuses on it.

READING IT
  chunk1  `tax_recovered` = (none - X) for X in donor/mean/generic, nats.  Put
          it next to the control-vs-cortex chunk-1 gap from the control run's
          own JSON (same rows, same seed).  Recovered >= half the gap
          -> register; ~0 or negative -> capacity.
          `nocarry_gap_by_chunk` flat across chunks 1..8 -> the tax is
          "carry absent", not "chunk 1".
  drift   the first chunk where `carry_gain` (nocarry - carry) goes negative,
          and which of e_row_norm / e_eff_rank_pr / z_* / turnover has left
          its chunks-2..8 range by then.

    python evals/diag_chunk1_drift.py --model_name ckpts/olmo-retrofit-cortex \
        --checkpoint cortex-5b/cortex-final/checkpoint_38146 $SETS \
        --data data/pg19_olmo_validation_len4096_strided --T 8 \
        --parts chunk1,drift --out_dir eval_results/pm/...
"""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import sys
from datetime import datetime

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evals.model_utils import (_unwrap, load_checkpoint,  # noqa: E402
                               parse_config_overrides, to_num_steps)
from cortex_memory.health import (carry_health, chance_margin,  # noqa: E402
                                  expected_ring_rows)

N_POS_BINS = 8


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model_name", required=True)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--data", required=True)
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    p.add_argument("--n_chunks", type=int, default=8)
    p.add_argument("--T", type=int, required=True,
                   help="MANDATORY: configs inherit mean_recurrence 32")
    p.add_argument("--parts", default="chunk1,drift")
    p.add_argument("--max_rows", type=int, default=200,
                   help="chunk1: eval rows (from the front of the pack)")
    p.add_argument("--bank_rows", type=int, default=64,
                   help="chunk1: rows from the BACK of the pack that build the "
                        "mean / generic carries, disjoint from the eval rows")
    p.add_argument("--drift_chunks", type=int, default=202,
                   help="202 = LongMemEval's depth, ~51 ring laps")
    p.add_argument("--drift_streams", type=int, default=6)
    p.add_argument("--depths", default="8,16,32,64,128",
                   help="restart/recency: target chunks (1-based), each >= 8 "
                        "and 8 + a multiple of the ring's lap")
    p.add_argument("--forced_modes", default="fifo,hardgate")
    p.add_argument("--drift_baseline", default=None,
                   help="chunk1_drift/results.json to pair gates/forced/restart "
                        "against (same streams and seeds)")
    p.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--boot", type=int, default=2000)
    p.add_argument("--out_dir", required=True)
    return p.parse_args()


# --------------------------------------------------------------------------
# pure helpers (tests/test_chunk1_drift.py)
# --------------------------------------------------------------------------

def row_chunks(ids: torch.Tensor, n_chunks: int):
    """Exactly the 2x2 / influence-horizon split: x = ids[:-1], y = ids[1:],
    trimmed to a multiple of n_chunks, torch.chunk'ed."""
    x, y = ids[:-1], ids[1:]
    keep = (x.numel() // n_chunks) * n_chunks
    return list(torch.chunk(x[:keep], n_chunks)), list(torch.chunk(y[:keep], n_chunks))


def pos_bins(ce: torch.Tensor, n_bins: int = N_POS_BINS) -> list:
    """Mean of a per-token vector over n_bins equal position bins."""
    return [float(b.mean()) for b in torch.chunk(ce.float(), n_bins)]


def find_streams(full_flags: list, rows_needed: int, max_streams: int) -> list:
    """Runs of consecutive FULL rows (a strided pack's one-book spans).

    A row with padding ends a book (prepare_pg19_dataset.py pads only a book's
    last window), so a stream is a run of full rows that does not cross one.
    Returns [(start_row, rows_needed)] -- at most one stream per book, so the
    streams are independent documents.
    """
    out, run_start = [], None
    for i, full in enumerate(list(full_flags) + [False]):
        if full and run_start is None:
            run_start = i
        if not full and run_start is not None:
            if i - run_start >= rows_needed:
                out.append((run_start, rows_needed))
                if len(out) >= max_streams:
                    return out
            run_start = None
    return out


def turnover(before: torch.Tensor | None, after: torch.Tensor,
             rows: list, hidden: int) -> dict:
    """||after - before|| / ||before|| on the rows this merge wrote, per channel.

    Measured, not read off the gate parameters: it is what the ring actually did
    to its contents at this chunk.  ~1 = the row was replaced, ~0 = retained.
    Rows the ring had not reached yet (no `before`) are skipped.
    """
    if before is None or not rows:
        return {}
    n = before.shape[1]
    rows = [r for r in rows if r < n]
    if not rows:
        return {}
    b, a = before[0, rows].float(), after[0, rows].float()
    out = {}
    for name, sl in (("e", slice(0, hidden)), ("z", slice(hidden, 2 * hidden))):
        if b.shape[-1] < sl.stop:
            continue
        num = (a[:, sl] - b[:, sl]).norm(dim=-1)
        den = b[:, sl].norm(dim=-1).clamp_min(1e-12)
        out[f"{name}_turnover"] = float((num / den).mean())
    return out


def paired_ci(deltas, n_boot: int, seed: int = 0):
    if not deltas:
        return (float("nan"),) * 3
    t = torch.tensor(deltas, dtype=torch.float64)
    g = torch.Generator().manual_seed(seed)
    idx = torch.randint(len(t), (n_boot, len(t)), generator=g)
    m = t[idx].mean(dim=1)
    lo, hi = torch.quantile(m, torch.tensor([0.025, 0.975], dtype=torch.float64))
    return float(t.mean()), float(lo), float(hi)


FRESH_CHUNKS = 7        # restart: the fresh memory reads chunks c-7 .. c-1
GATE_MODES = ("trained", "fifo", "hardgate")


def check_depths(depths: list, lap: int) -> None:
    """Refuse restart/recency depths whose fresh memory is not comparable.

    c <= 7 would read before chunk 1.  The fresh chain runs 7 merges and the
    long one c-1, so their ring pointers agree (row r holds the same chunk's
    write in both, which the row-wise swap and cosine need) only when
    (c - 8) is a multiple of the lap.
    """
    for c in depths:
        if c <= FRESH_CHUNKS:
            raise ValueError(f"depth {c}: the fresh memory reads chunks "
                             f"{c - FRESH_CHUNKS}..{c - 1}, which starts before chunk 1")
        if (c - FRESH_CHUNKS - 1) % lap:
            raise ValueError(f"depth {c}: (c - 8) is not a multiple of the "
                             f"{lap}-chunk lap, so long and fresh rings are rotated")


def gate_stats(state, cand, ig, fg, ch: str) -> dict:
    """A1 over the rows one gate call touched: mean fg / ig, the share of
    entries with fg + ig > 1, and ||write|| / ||stored row it lands in||."""
    fg, ig = fg.detach().float(), ig.detach().float()
    ratio = (cand.detach().float().norm(dim=-1)
             / state.detach().float().norm(dim=-1).clamp_min(1e-12))
    return {f"{ch}_fg": float(fg.mean()), f"{ch}_ig": float(ig.mean()),
            f"{ch}_fg_ig_gt1": float((fg + ig > 1).float().mean()),
            f"{ch}_write_over_stored": float(ratio.mean())}


def forced_update(state, cand, fg, mode: str):
    """A5: the gated rows under a forced write rule.  `hardgate` keeps a row
    bit-identical when its dim-mean trained fg >= 0.5, else replaces it."""
    if mode == "fifo":
        return cand
    if mode == "hardgate":
        keep = fg.float().mean(dim=-1, keepdim=True) >= 0.5
        return torch.where(keep, state, cand)
    raise ValueError(f"unknown forced mode {mode!r}")


class gate_spy:
    """Wrap the buffer's per-channel gate calls (E: `gate`, Z: `gate_latent`)
    for one forward: record gate_stats, and in a forced mode replace the
    update.  Only the touched rows ever reach a gate (the ring index_copy's
    the rest through), so untouched rows stay bit-identical in every mode."""

    def __init__(self, buf, mode: str = "trained"):
        if mode not in GATE_MODES:
            raise ValueError(f"gate mode {mode!r} not in {GATE_MODES}")
        self.buf, self.mode, self.stats = buf, mode, {}

    def _wrap(self, name: str, ch: str):
        real = getattr(type(self.buf), name).__get__(self.buf, type(self.buf))

        def g(state, candidate):
            out, ig, fg = real(state, candidate)
            st = gate_stats(state, candidate, ig, fg, ch)
            if self.mode != "trained":
                out = forced_update(state, candidate, fg, self.mode)
                st[f"{ch}_replaced"] = float(
                    (fg.float().mean(dim=-1) < 0.5).float().mean()
                    if self.mode == "hardgate" else 1.0)
            self.stats.update(st)
            return out, ig, fg
        return g

    def __enter__(self):
        self.buf.gate = self._wrap("gate", "e")
        if getattr(self.buf, "carries_latent", False):
            self.buf.gate_latent = self._wrap("gate_latent", "z")
        return self

    def __exit__(self, *exc):
        for k in ("gate", "gate_latent"):
            self.buf.__dict__.pop(k, None)
        return False


def centred_row_cos(a: torch.Tensor, b: torch.Tensor, hidden: int) -> dict:
    """Mean cosine between matching rows of two carries, each centred on its
    own row mean first, per channel ([1, K, D*] each)."""
    out = {}
    for name, sl in (("e", slice(0, hidden)), ("z", slice(hidden, 2 * hidden))):
        if a.shape[-1] < sl.stop:
            continue
        x, y = a[0, :, sl].float(), b[0, :, sl].float()
        x, y = x - x.mean(0, keepdim=True), y - y.mean(0, keepdim=True)
        out[f"{name}_cos"] = float(F.cosine_similarity(x, y, dim=-1).mean())
    return out


# --------------------------------------------------------------------------
# model plumbing
# --------------------------------------------------------------------------

def seed_for(base: int, k: int) -> int:
    """One s0 seed per (row, chunk position), shared by EVERY condition at that
    position, so carry / no-carry / donor differ only in the carry and never in
    the trunc_normal_ draw (the 2x2's chunk-1 veto, applied per chunk)."""
    return base * 1009 + k


def fwd(model, x, state, num_steps, device, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    with torch.no_grad():
        out = model(input_ids=x.unsqueeze(0).to(device), num_steps=num_steps,
                    m_cross_in=state, return_m_cross=True)
    logits = (out["logits"] if isinstance(out, dict) else out.logits)[0].float()
    new = out.get("m_cross") if isinstance(out, dict) else getattr(out, "m_cross", None)
    return logits, new


RING_POINTER_ATTRS = ("_chunk", "_z_chunk", "_z_pending")


class preserve_ring_pointer:
    """Snapshot and restore the gated ring's write pointer around a side forward.

    The pointer is mutable state ON THE BUFFER (buffers.py merge): a forward
    with m_cross_in=None resets `_chunk` to 1 and every merge advances it.  The
    drift loop interleaves a no-carry forward into a running chain, and without
    this every later write of the chain lands on the same rows (16..31) --
    caught in the smoke as turnover reading exactly 0.0 on the rows the ring
    rule says were written.
    """

    def __init__(self, buf):
        self.buf = buf

    def __enter__(self):
        self.saved = {k: (list(v) if isinstance(v, list) else v)
                      for k, v in self.buf.__dict__.items()
                      if k in RING_POINTER_ATTRS}
        return self

    def __exit__(self, *exc):
        for k, v in self.saved.items():
            setattr(self.buf, k, v)
        return False


def ce_of(logits, y, device):
    return F.cross_entropy(logits, y.to(device), reduction="none")


def chain_states(model, xs, num_steps, device, seed):
    """States ENTERING each chunk (None for chunk 1) on the row's own chain."""
    st, out = None, []
    for k, x in enumerate(xs):
        out.append(st)
        _, st = fwd(model, x, st, num_steps, device, seed_for(seed, k))
    return out


# --------------------------------------------------------------------------
# part chunk1
# --------------------------------------------------------------------------

def part_chunk1(model, ds, args, num_steps, device, has_mem, vocab):
    n_eval = min(args.max_rows, len(ds) - args.bank_rows)
    if n_eval < 2:
        raise SystemExit(f"pack too small: {len(ds)} rows for {args.max_rows} "
                         f"eval + {args.bank_rows} bank")
    if n_eval < args.max_rows:
        print(f"WARN: only {n_eval} eval rows (asked {args.max_rows}); the bank "
              f"takes the last {args.bank_rows}")
    res = {"n_eval_rows": 0, "conditions": {}, "nocarry_by_chunk": {},
           "carry_by_chunk": {}, "pos_profile": {}}
    mean64 = generic64 = None
    if has_mem:
        # The bank: states entering chunk 5 (a full 64-row ring) of rows from
        # the BACK of the pack, so no eval row's own content is in them.
        acc, nb = None, 0
        for bi in range(len(ds) - args.bank_rows, len(ds)):
            ids = torch.tensor(ds[bi]["input_ids"], dtype=torch.long)
            xs, _ = row_chunks(ids, args.n_chunks)
            sts = chain_states(model, xs[:5], num_steps, device, args.seed + bi)
            s5 = sts[4].float()
            if generic64 is None:
                generic64 = sts[4].detach().clone()
            acc = s5 if acc is None else acc + s5
            nb += 1
        mean64 = (acc / nb).to(generic64.dtype)
        torch.save(mean64.cpu(), os.path.join(args.out_dir, "mean64.pt"))
        torch.save(generic64.cpu(), os.path.join(args.out_dir, "generic64.pt"))
        hm, hg = carry_health(mean64, mean64.shape[-1] // 2), \
            carry_health(generic64, generic64.shape[-1] // 2)
        res["bank"] = {"rows": nb, "mean64_health": hm, "generic64_health": hg}
        print(f"[chunk1] bank of {nb} rows: mean64 e_row_norm "
              f"{hm.get('e_row_norm'):.2f} vs generic64 {hg.get('e_row_norm'):.2f} "
              f"(averaging shrinks norms -- read mean64 with that in mind)")

    conds = ["none"] + (["donor64", "donor16", "mean64", "generic64"] if has_mem else [])
    per = {c: [] for c in conds}
    prof = {c: [[] for _ in range(N_POS_BINS)] for c in conds}
    noc = {k: [] for k in range(args.n_chunks)}
    car = {k: [] for k in range(args.n_chunks)}
    prof_later = [[] for _ in range(N_POS_BINS)]
    for si in range(n_eval):
        ids = torch.tensor(ds[si]["input_ids"], dtype=torch.long)
        xs, ys = row_chunks(ids, args.n_chunks)
        seed = args.seed + si
        # every row's chain on its own carry (the normal condition), and every
        # chunk standalone (no carry), same s0 seed per chunk position
        st = None
        for k, (x, y) in enumerate(zip(xs, ys)):
            lg, st_next = fwd(model, x, st, num_steps, device, seed_for(seed, k))
            ce = ce_of(lg, y, device)
            car[k].append(float(ce.mean()))
            if k > 0:
                for b, v in enumerate(pos_bins(ce)):
                    prof_later[b].append(v)
            st = st_next if has_mem else None
        for k, (x, y) in enumerate(zip(xs, ys)):
            lg, _ = fwd(model, x, None, num_steps, device, seed_for(seed, k))
            noc[k].append(float(ce_of(lg, y, device).mean()))
        # chunk 1 under each carry; one s0 seed for all of them (paired)
        donor_sts = None
        if has_mem:
            dj = (si + n_eval // 2) % n_eval
            dids = torch.tensor(ds[dj]["input_ids"], dtype=torch.long)
            dxs, _ = row_chunks(dids, args.n_chunks)
            donor_sts = chain_states(model, dxs[:5], num_steps, device, args.seed + dj)
        for c in conds:
            carry = {"none": None,
                     "donor64": donor_sts[4] if donor_sts else None,
                     "donor16": donor_sts[1] if donor_sts else None,
                     "mean64": mean64, "generic64": generic64}[c]
            lg, _ = fwd(model, xs[0], carry, num_steps, device, seed_for(seed, 0))
            ce = ce_of(lg, ys[0], device)
            per[c].append(float(ce.mean()))
            for b, v in enumerate(pos_bins(ce)):
                prof[c][b].append(v)
        res["n_eval_rows"] += 1
        if res["n_eval_rows"] % 25 == 0:
            print(f"  chunk1: {res['n_eval_rows']} rows", flush=True)

    mean = lambda v: sum(v) / len(v) if v else None
    for c in conds:
        res["conditions"][c] = {"chunk1_nll": mean(per[c]),
                                "pos_profile": [mean(b) for b in prof[c]],
                                "per_row": per[c]}
        if c != "none":
            m, lo, hi = paired_ci([a - b for a, b in zip(per["none"], per[c])],
                                  args.boot, args.seed)
            res["conditions"][c]["tax_recovered"] = {
                "mean_nats": m, "ci_lo": lo, "ci_hi": hi,
                "note": "NLL(chunk1, no carry) - NLL(chunk1, this carry). "
                        "Positive = this carry lowers chunk-1 loss."}
    res["nocarry_by_chunk"] = {str(k + 1): {"nll": mean(v), "per_row": v}
                               for k, v in noc.items()}
    res["carry_by_chunk"] = {str(k + 1): {"nll": mean(v), "per_row": v}
                             for k, v in car.items()}
    res["pos_profile"]["chunks2plus_with_carry"] = [mean(b) for b in prof_later]
    res["health"] = chance_margin([res["conditions"]["none"]["chunk1_nll"]], vocab)

    print(f"\n{'=' * 78}\nCHUNK-1 TAX  (n={res['n_eval_rows']} rows, "
          f"{'cortex' if has_mem else 'control: no-carry profiles only'})\n{'=' * 78}")
    for c in conds:
        r = res["conditions"][c]
        tr = r.get("tax_recovered")
        extra = (f"   recovered {tr['mean_nats']:+.4f} [{tr['ci_lo']:+.4f}, "
                 f"{tr['ci_hi']:+.4f}]") if tr else ""
        print(f"  {c:<10} chunk-1 NLL {r['chunk1_nll']:.4f}{extra}")
    print("  by position bin (64-token bins), chunk 1 no carry vs chunks 2+ with carry:")
    print("   " + " ".join(f"{v:6.3f}" for v in res["conditions"]["none"]["pos_profile"]))
    print("   " + " ".join(f"{v:6.3f}" for v in res["pos_profile"]["chunks2plus_with_carry"]))
    print("  no-carry NLL by chunk: " + " ".join(
        f"{res['nocarry_by_chunk'][str(k + 1)]['nll']:.3f}" for k in range(args.n_chunks)))
    print("  carry NLL by chunk:    " + " ".join(
        f"{res['carry_by_chunk'][str(k + 1)]['nll']:.3f}" for k in range(args.n_chunks)))
    return res


# --------------------------------------------------------------------------
# part drift
# --------------------------------------------------------------------------

def book_streams(ds, args):
    """The drift's single-book streams and chunk length.  Every depth part
    calls this with the same args, so its streams are the drift's."""
    chunk_len = (len(ds[0]["input_ids"]) - 1) // args.n_chunks
    rows_needed = math.ceil((args.drift_chunks + 1) / args.n_chunks) + 1
    has_mask = "attention_mask" in ds.column_names
    flags = [(not has_mask) or (ds[i]["attention_mask"][-1] == 1)
             for i in range(len(ds))]
    streams = find_streams(flags, rows_needed, args.drift_streams)
    if not streams:
        raise SystemExit(f"no run of {rows_needed} consecutive full rows in "
                         f"{args.data}: cannot build a {args.drift_chunks}-chunk "
                         f"single-book stream")
    return streams, chunk_len


def stream_xy(ds, start: int, nrows: int):
    toks = torch.cat([torch.tensor(ds[r]["input_ids"][:-1], dtype=torch.long)
                      for r in range(start, start + nrows)])
    return toks[:-1], toks[1:]


def part_drift(model, buf, ds, args, num_steps, device, hidden, gate_mode=None):
    """gate_mode None is the drift as recorded; "trained" adds A1's gate
    statistics; "fifo" / "hardgate" override the update (A5)."""
    streams, chunk_len = book_streams(ds, args)
    print(f"[drift] {len(streams)} single-book streams of {args.drift_chunks} "
          f"chunks (rows {[s for s, _ in streams]}).  ASSUMES the pack keeps "
          f"each book's windows in order, as prepare_pg19_dataset.py writes it.")
    n_slots = int(getattr(buf, "n_slots", getattr(buf, "max_vecs", 64)))
    n_vec = int(buf.n_vec)
    per_chunk = [dict() for _ in range(args.drift_chunks)]
    keys_acc = {}

    def add(c, k, v):
        if v is None:
            return
        keys_acc.setdefault((c, k), []).append(float(v))

    real_merge = type(buf).merge.__get__(buf, type(buf))
    for s_i, (start, nrows) in enumerate(streams):
        x_all, y_all = stream_xy(ds, start, nrows)
        st = None
        for c in range(args.drift_chunks):
            x = x_all[c * chunk_len:(c + 1) * chunk_len]
            y = y_all[c * chunk_len:(c + 1) * chunk_len]
            if x.numel() < chunk_len:
                break
            seen = {}

            def spy(state, new_vecs, new_latent=None):
                out = real_merge(state, new_vecs, new_latent)
                seen["before"], seen["after"] = state, out
                return out

            sd = seed_for(args.seed + 99991 * s_i, c)
            buf.merge = spy
            gs = gate_spy(buf, gate_mode) if gate_mode else None
            try:
                with gs or contextlib.nullcontext():
                    lg, st_new = fwd(model, x, st, num_steps, device, sd)
            finally:
                buf.__dict__.pop("merge", None)
            for k, v in (gs.stats.items() if gs else ()):
                add(c, k, v)
            ce = ce_of(lg, y, device)
            probs = lg.softmax(-1)
            ent = -(probs * probs.clamp_min(1e-30).log()).sum(-1)
            with preserve_ring_pointer(buf):
                lg0, _ = fwd(model, x, None, num_steps, device, sd)
            ce0 = ce_of(lg0, y, device)
            add(c, "nll_carry", ce.mean())
            add(c, "nll_nocarry", ce0.mean())
            add(c, "carry_gain", ce0.mean() - ce.mean())
            add(c, "entropy", ent.mean())
            add(c, "first_tok_entropy", ent[0])
            if st is not None:
                h = carry_health(st, hidden)
                for k in ("rows", "e_row_norm", "e_eff_rank_pr", "e_centred_cos_mean",
                          "z_row_norm", "z_eff_rank_pr", "z_centred_cos_mean",
                          "z_over_e_norm"):
                    add(c, k, h.get(k))
            if "after" in seen:
                # chunk c (0-based) is the c-th merge the ring has run
                rows = expected_ring_rows(c, n_vec, n_slots)
                tv = turnover(seen["before"], seen["after"], rows, hidden)
                # hardgate keeps rows by design, so a 0.0 there is not a pointer bug
                if (gate_mode in (None, "trained") and tv
                        and seen["before"].shape[1] == n_slots
                        and tv.get("e_turnover") == 0.0):
                    raise RuntimeError(
                        f"chunk {c + 1}: the rows the ring rule says were written "
                        f"({rows[0]}..{rows[-1]}) did not change -- the ring pointer "
                        f"and the rule disagree, so every turnover number is wrong")
                for k, v in tv.items():
                    add(c, k, v)
            st = st_new
        print(f"  drift stream {s_i + 1}/{len(streams)} done", flush=True)

    out = {"streams": streams, "chunk_len": chunk_len, "per_chunk": []}
    for c in range(args.drift_chunks):
        row = {"chunk": c + 1, "laps": round(c / max(n_slots // n_vec, 1), 2)}
        for (cc, k), v in keys_acc.items():
            if cc == c:
                row[k] = sum(v) / len(v)
                row[k + "_n"] = len(v)
        out["per_chunk"].append(row)
    neg = [r["chunk"] for r in out["per_chunk"]
           if r.get("carry_gain") is not None and r["chunk"] > 1 and r["carry_gain"] < 0]
    out["first_negative_gain_chunk"] = neg[0] if neg else None

    print(f"\n{'=' * 78}\nDRIFT  ({len(streams)} books x {args.drift_chunks} chunks)\n{'=' * 78}")
    cols = ["nll_carry", "nll_nocarry", "carry_gain", "entropy", "first_tok_entropy",
            "e_row_norm", "e_eff_rank_pr", "z_row_norm", "z_eff_rank_pr",
            "e_turnover", "z_turnover"]
    print("  chunk laps " + " ".join(f"{k[:11]:>11}" for k in cols))
    for c in [1, 2, 3, 4, 5, 8, 12, 16, 24, 32, 48, 64, 96, 128, 160, 202]:
        if c > args.drift_chunks:
            continue
        r = out["per_chunk"][c - 1]
        print(f"  {c:>5} {r['laps']:>4} " + " ".join(
            f"{r[k]:>11.4f}" if r.get(k) is not None else f"{'-':>11}" for k in cols))
    print(f"\n  first chunk where the carry HURTS (carry_gain < 0): "
          f"{out['first_negative_gain_chunk']}")
    if gate_mode:
        out["gate_mode"] = gate_mode
        gcols = [f"{ch}_{k}" for ch in ("e", "z") for k in
                 ("fg", "ig", "fg_ig_gt1", "write_over_stored", "replaced")]
        gcols = [k for k in gcols if any(k in r for r in out["per_chunk"])]
        print(f"\n  gates ({gate_mode}), over the rows each merge touched:")
        print("  chunk " + " ".join(f"{k[:13]:>13}" for k in gcols))
        for c in [5, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 160, 202]:
            if c > args.drift_chunks:
                continue
            r = out["per_chunk"][c - 1]
            print(f"  {c:>5} " + " ".join(
                f"{r[k]:>13.4f}" if r.get(k) is not None else f"{'-':>13}" for k in gcols))
    return out


def vs_baseline(out: dict, path: str | None) -> dict | None:
    """Pair a gates/forced drift against chunk1_drift/'s, chunk for chunk.

    nll_nocarry never reads the carry, so on the same streams and seeds it must
    match the baseline: its max gap is the pairing check.
    """
    if not path:
        return None
    with open(path, encoding="utf-8") as fh:
        base = json.load(fh)["drift"]
    if base["streams"] != [list(s) for s in out["streams"]]:
        raise SystemExit(f"{path} ran on streams {base['streams']}, this run on "
                         f"{out['streams']}: not paired")
    bpc = {r["chunk"]: r for r in base["per_chunk"]}
    rows, gaps = [], []
    for r in out["per_chunk"]:
        b = bpc.get(r["chunk"])
        if b is None or r.get("carry_gain") is None or b.get("carry_gain") is None:
            continue
        gaps.append(abs(r["nll_nocarry"] - b["nll_nocarry"]))
        rows.append({"chunk": r["chunk"], "carry_gain": r["carry_gain"],
                     "baseline_carry_gain": b["carry_gain"],
                     "delta": r["carry_gain"] - b["carry_gain"]})
    res = {"baseline": path,
           "baseline_first_negative_gain_chunk": base.get("first_negative_gain_chunk"),
           "first_negative_gain_chunk": out.get("first_negative_gain_chunk"),
           "pairing_max_nocarry_gap": max(gaps) if gaps else None, "per_chunk": rows}
    print(f"\n  vs {path}: first harmful chunk {res['first_negative_gain_chunk']} "
          f"(baseline {res['baseline_first_negative_gain_chunk']}); pairing check "
          f"max |nll_nocarry gap| = {res['pairing_max_nocarry_gap']}")
    by = {r["chunk"]: r for r in rows}
    for c in (2, 8, 32, 64, 202):
        if c in by:
            r = by[c]
            print(f"    chunk {c:>3}: gain {r['carry_gain']:+.4f} vs "
                  f"{r['baseline_carry_gain']:+.4f}  (delta {r['delta']:+.4f})")
    return res


def depth_pairing(res: dict, path: str | None) -> dict | None:
    """restart/recency's long chain and no-carry forwards vs chunk1_drift/'s
    nll_carry / nll_nocarry at the same chunks: both gaps should be ~0."""
    if not path:
        return None
    with open(path, encoding="utf-8") as fh:
        bpc = {r["chunk"]: r for r in json.load(fh)["drift"]["per_chunk"]}
    gaps = {str(c): {"long_vs_carry": d["nll_long"]["mean"] - bpc[int(c)]["nll_carry"],
                     "nocarry": d["nll_nocarry"]["mean"] - bpc[int(c)]["nll_nocarry"]}
            for c, d in res["depths"].items() if bpc.get(int(c), {}).get("nll_carry") is not None}
    print(f"\n  pairing vs {path} (both ~0 or the long chain was perturbed): " + "  ".join(
        f"c{c} {g['long_vs_carry']:+.2e}/{g['nocarry']:+.2e}" for c, g in gaps.items()))
    return {"baseline": path, "gaps": gaps}


# --------------------------------------------------------------------------
# parts restart (A2/A3) and recency (A4)
# --------------------------------------------------------------------------

def part_depth(model, buf, ds, args, num_steps, device, hidden, do_restart,
               do_recency, vocab_text=None):
    """One long chain per stream; at each depth c, the side forwards both
    parts need.  Long-chain seeds are the drift's, so nll_long at c is
    chunk1_drift's nll_carry at c."""
    from evals.diag_memory_interp import (RARE_MAX_DF, bag_of_forms,
                                          build_form_table, frequent_forms,
                                          restrict_forms, set_logmass, splice_z)
    depths = sorted({int(d) for d in args.depths.split(",") if d.strip()})
    n_slots = int(getattr(buf, "n_slots", 64))
    check_depths(depths, max(n_slots // int(buf.n_vec), 1))
    if depths[-1] > args.drift_chunks:
        raise SystemExit(f"depth {depths[-1]} > --drift_chunks {args.drift_chunks}")
    streams, chunk_len = book_streams(ds, args)
    xys = [stream_xy(ds, s, n) for s, n in streams]
    chunk = lambda s, k: (xys[s][0][k * chunk_len:(k + 1) * chunk_len],  # noqa: E731
                          xys[s][1][k * chunk_len:(k + 1) * chunk_len])
    print(f"[depth] {len(streams)} streams (rows {[s for s, _ in streams]}), "
          f"depths {depths}, restart={do_restart} recency={do_recency}")
    if do_recency:
        if len(streams) < 2:
            raise SystemExit("recency needs >= 2 streams: the control set is "
                             "another book's")
        form_all, forms = build_form_table(vocab_text, 1)
        form_all = form_all.to(device)
        nf = len(forms)
        bags = [[bag_of_forms(chunk(s, k)[0], form_all, nf) for k in range(depths[-1])]
                for s in range(len(streams))]
        freq = frequent_forms([b for bs in bags for b in bs], RARE_MAX_DF)
        rare = restrict_forms(form_all, ~freq)
        rbags = [[bag_of_forms(chunk(s, k)[0], rare, nf) for k in range(depths[-1])]
                 for s in range(len(streams))]

    names = ("long", "fresh", "longE_freshZ", "freshE_longZ")
    rows = {c: [] for c in depths}
    for s_i in range(len(streams)):
        sd = lambda k: seed_for(args.seed + 99991 * s_i, k)  # noqa: E731
        st = None
        for k in range(depths[-1]):
            x, y = chunk(s_i, k)
            lg, st_next = fwd(model, x, st, num_steps, device, sd(k))
            c = k + 1
            if c in rows:
                r = {"stream": s_i}
                with preserve_ring_pointer(buf):
                    lg0, _ = fwd(model, x, None, num_steps, device, sd(k))
                    r["nll_nocarry"] = float(ce_of(lg0, y, device).mean())
                    r["nll_long"] = float(ce_of(lg, y, device).mean())
                    if do_restart:
                        fresh = None
                        for j in range(k - FRESH_CHUNKS, k):
                            _, fresh = fwd(model, chunk(s_i, j)[0], fresh,
                                           num_steps, device, sd(j))
                        if fresh.shape != st.shape:
                            raise RuntimeError(f"fresh carry {tuple(fresh.shape)} vs "
                                               f"long {tuple(st.shape)} at chunk {c}")
                        if c == FRESH_CHUNKS + 1:
                            # same chunks, same seeds: anything else is a pointer bug
                            gap = float((fresh.float() - st.float()).abs().max())
                            if gap > 1e-5 * float(st.float().abs().max()):
                                raise RuntimeError(f"chunk 8: fresh != long (max gap "
                                                   f"{gap:.3g}); the restart is not paired")
                        r.update(centred_row_cos(st, fresh, hidden))
                        carries = {"fresh": fresh}
                        if fresh.shape[-1] == 2 * hidden:
                            carries["longE_freshZ"] = splice_z(st, fresh)
                            carries["freshE_longZ"] = splice_z(fresh, st)
                        for nm, carry in carries.items():
                            lgx, _ = fwd(model, x, carry, num_steps, device, sd(k))
                            r[f"nll_{nm}"] = float(ce_of(lgx, y, device).mean())
                        for nm, carry in (("long", st), ("fresh", fresh)):
                            h = carry_health(carry, hidden)
                            for key in ("e_row_norm", "z_row_norm"):
                                if h.get(key) is not None:
                                    r[f"{nm}_{key}"] = h[key]
                if do_recency:
                    # old stops before recent, or at c <= 12 the shared chunks empty both sets
                    win = {"recent": range(k - 4, k), "old": range(0, min(8, k - 4))}
                    o = (s_i + 1) % len(streams)
                    u = {nm: torch.stack([rbags[s_i][j] for j in w]).any(0)
                         for nm, w in win.items()}
                    u["ctl"] = torch.stack([rbags[o][j] for j in win["recent"]]).any(0)
                    here = bag_of_forms(x, form_all, nf)
                    lp, lp0 = lg.log_softmax(-1), lg0.log_softmax(-1)
                    for nm in u:
                        others = torch.stack([u[q] for q in u if q != nm]).any(0)
                        m = u[nm] & ~here & ~others
                        ids = torch.nonzero((form_all >= 0) & m[form_all.clamp_min(0)]).squeeze(1)
                        r[f"n_{nm}"] = int(m.sum())
                        r[f"d_{nm}"] = (float((set_logmass(lp, ids)
                                               - set_logmass(lp0, ids)).double().mean())
                                        if ids.numel() else None)
                    for nm in ("recent", "old"):
                        ok = r[f"d_{nm}"] is not None and r["d_ctl"] is not None
                        r[f"{nm}_lift"] = r[f"d_{nm}"] - r["d_ctl"] if ok else None
                rows[c].append(r)
            st = st_next
        print(f"  depth stream {s_i + 1}/{len(streams)} done", flush=True)

    def ci(vals):
        vals = [v for v in vals if v is not None]
        m, lo, hi = paired_ci(vals, args.boot, args.seed)
        return {"mean": m, "ci_lo": lo, "ci_hi": hi, "n": len(vals)}

    restart = {"streams": streams, "fresh_chunks": FRESH_CHUNKS, "depths": {}}
    recency = {"streams": streams, "rare_rule": f"alphabetic form in < "
               f"{RARE_MAX_DF:.0%} of the streams' chunks 1..{depths[-1]}",
               "windows": "recent c-4..c-1, old 1..min(8, c-5), ctl = next stream's c-4..c-1; "
                          "sets disjoint and absent from chunk c", "depths": {}}
    for c, rs in rows.items():
        base = {"nll_nocarry": ci([r["nll_nocarry"] for r in rs]),
                "nll_long": ci([r["nll_long"] for r in rs])}
        if do_restart:
            d = dict(base)
            for nm in names[1:]:
                if all(f"nll_{nm}" in r for r in rs):
                    d[f"nll_{nm}"] = ci([r[f"nll_{nm}"] for r in rs])
            # positive = the swapped-in fresh part makes chunk c cheaper
            d["fresh_minus_long_gain"] = ci([r["nll_long"] - r["nll_fresh"] for r in rs])
            for nm, lab in (("longE_freshZ", "fresh_z_recovers"),
                            ("freshE_longZ", "fresh_e_recovers")):
                if f"nll_{nm}" in d:
                    d[lab] = ci([r["nll_long"] - r[f"nll_{nm}"] for r in rs])
            for key in ("e_cos", "z_cos", "long_e_row_norm", "fresh_e_row_norm",
                        "long_z_row_norm", "fresh_z_row_norm"):
                if all(key in r for r in rs):
                    d[key] = ci([r[key] for r in rs])
            d["per_stream"] = rs
            restart["depths"][str(c)] = d
        if do_recency:
            d = dict(base)
            for key in ("recent_lift", "old_lift", "d_recent", "d_old", "d_ctl"):
                d[key] = ci([r[key] for r in rs])
            for key in ("n_recent", "n_old", "n_ctl"):
                d[key] = sum(r[key] for r in rs) / len(rs)
            d["per_stream"] = [{k: r[k] for k in r if k.startswith(("stream", "d_", "n_", "recent", "old"))}
                               for r in rs]
            recency["depths"][str(c)] = d

    f = lambda v: f"{v['mean']:+.4f}" if v and v.get("mean") is not None \
        and not math.isnan(v["mean"]) else "      -"  # noqa: E731
    print(f"\n{'=' * 78}\nDEPTH TESTS  ({len(streams)} books)\n{'=' * 78}")
    if do_restart:
        print("  restart: gain = NLL(long) - NLL(X), + = X cheaper than the long memory")
        print("  chunk  nocarry     long  fresh-gain  freshZ-gain  freshE-gain    e_cos    z_cos")
        for c in depths:
            d = restart["depths"][str(c)]
            print(f"  {c:>5} {d['nll_nocarry']['mean']:8.4f} {d['nll_long']['mean']:8.4f}"
                  f"    {f(d['fresh_minus_long_gain'])}      {f(d.get('fresh_z_recovers'))}"
                  f"      {f(d.get('fresh_e_recovers'))}  {f(d.get('e_cos'))}  {f(d.get('z_cos'))}")
    if do_recency:
        print("  recency: lift = d(set) - d(ctl), d = log-mass with memory - without")
        print("  chunk  recent_lift  old_lift   n_recent  n_old  n_ctl")
        for c in depths:
            d = recency["depths"][str(c)]
            print(f"  {c:>5}     {f(d['recent_lift'])}   {f(d['old_lift'])}   "
                  f"{d['n_recent']:8.0f} {d['n_old']:6.0f} {d['n_ctl']:6.0f}")
    return (restart if do_restart else None), (recency if do_recency else None)


def main() -> int:
    args = parse_args()
    if args.dtype == "bfloat16":
        print("WARNING: bfloat16 -- smoke only; these are differences of near-equal losses.")
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    overrides = parse_config_overrides(args.set)
    model, cfg = load_checkpoint(args.checkpoint, args.model_name, None,
                                 getattr(torch, args.dtype), device,
                                 config_overrides=overrides or None)
    inner = _unwrap(model)
    cortex = getattr(inner, "cortex", None)
    buf = getattr(cortex, "prefix", None) if cortex is not None else None
    has_mem = buf is not None
    if overrides and not has_mem:
        print("FAILED: --set was given (a memory arm) but no prefix buffer built.")
        return 2
    hidden = int(buf.hidden_size) if has_mem else int(getattr(inner.config, "n_embd", 2048))
    vocab = int(getattr(inner.config, "vocab_size", 0) or 0)
    num_steps = to_num_steps(args.T)
    from datasets import load_from_disk
    ds = load_from_disk(args.data)
    parts = [p.strip() for p in args.parts.split(",") if p.strip()]

    report = {"instrument": "post-mortem chunk-1 tax + drift",
              "when": datetime.now().isoformat(timespec="seconds"),
              "model_name": args.model_name, "checkpoint": args.checkpoint,
              "has_memory": has_mem,
              "config": {"T": args.T, "n_chunks": args.n_chunks, "data": args.data,
                         "dtype": args.dtype, "overrides": overrides,
                         "seed": args.seed}}
    path = os.path.join(args.out_dir, "results.json")

    def save():
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)

    unknown = set(parts) - {"chunk1", "drift", "gates", "restart", "recency", "forced"}
    if unknown:
        print(f"FAILED: unknown --parts {sorted(unknown)}")
        return 2
    bad_modes = {m.strip() for m in args.forced_modes.split(",") if m.strip()} \
        - {"fifo", "hardgate"}
    if "forced" in parts and bad_modes:
        print(f"FAILED: unknown --forced_modes {sorted(bad_modes)}")
        return 2
    depth_parts = set(parts) - {"chunk1", "drift"}
    if depth_parts and not has_mem:
        print(f"FAILED: {sorted(depth_parts)} need a memory arm; this model has no buffer.")
        return 2
    if "chunk1" in parts:
        report["chunk1"] = part_chunk1(model, ds, args, num_steps, device, has_mem, vocab)
    if "drift" in parts:
        if not has_mem:
            print("drift SKIPPED: no carry on this model (control) -- nothing to drift.")
        else:
            report["drift"] = part_drift(model, buf, ds, args, num_steps, device, hidden)
    if "gates" in parts:
        save()
        g = part_drift(model, buf, ds, args, num_steps, device, hidden, "trained")
        g["vs_baseline"] = vs_baseline(g, args.drift_baseline)
        report["gates"] = g
    if "restart" in parts or "recency" in parts:
        save()
        vocab_text = None
        if "recency" in parts:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(args.model_name)
            V = int(inner.lm_head.weight.shape[0])
            vocab_text = tok.batch_decode([[i] for i in range(V)],
                                          clean_up_tokenization_spaces=False)
        rs, rc = part_depth(model, buf, ds, args, num_steps, device, hidden,
                            "restart" in parts, "recency" in parts, vocab_text)
        pair = depth_pairing(rs or rc, args.drift_baseline)
        for name, res in (("restart", rs), ("recency", rc)):
            if res is not None:
                res["vs_baseline"] = pair
                report[name] = res
    if "forced" in parts:
        report["forced"] = {}
        for mode in [m.strip() for m in args.forced_modes.split(",") if m.strip()]:
            save()
            fd = part_drift(model, buf, ds, args, num_steps, device, hidden, mode)
            fd["vs_baseline"] = vs_baseline(fd, args.drift_baseline)
            report["forced"][mode] = fd
    save()
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
