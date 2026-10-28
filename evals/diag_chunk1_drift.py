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

def part_drift(model, buf, ds, args, num_steps, device, hidden):
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
        toks = torch.cat([torch.tensor(ds[r]["input_ids"][:-1], dtype=torch.long)
                          for r in range(start, start + nrows)])
        x_all, y_all = toks[:-1], toks[1:]
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
            try:
                lg, st_new = fwd(model, x, st, num_steps, device, sd)
            finally:
                buf.__dict__.pop("merge", None)
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
                if tv and seen["before"].shape[1] == n_slots and tv.get("e_turnover") == 0.0:
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
    return out


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
    if "chunk1" in parts:
        report["chunk1"] = part_chunk1(model, ds, args, num_steps, device, has_mem, vocab)
    if "drift" in parts:
        if not has_mem:
            print("drift SKIPPED: no carry on this model (control) -- nothing to drift.")
        else:
            report["drift"] = part_drift(model, buf, ds, args, num_steps, device, hidden)
    path = os.path.join(args.out_dir, "results.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
