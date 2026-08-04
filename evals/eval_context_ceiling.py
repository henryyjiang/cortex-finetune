"""
Context ceiling — how much is ANY carry worth on this data, and what fraction
of it does the compressed carry recover?

Every carry number the project has (+0.024 Track A, -0.013 B1, ~0 for the ci /
ki4 arms) is quoted against zero, so a delta of +0.024 nats has never had a
scale.  AutoCompressor always reports all three points: Llama-2 ppl 5.40 with no
context, 5.07 with their compressed summaries, 4.76 with REAL full attention over
the same span.  The meaningful quantity is the fraction of 5.40->4.76 that the
compression recovers.  This measures our version of that.

For each chunk g >= 2 of the same 4x1024 PG-19 rows the carry ablation uses,
score chunk g's tokens with k real PRECEDING tokens prepended, for k in
--context_lens, plus (when the model has one) the model's own compressed carry:

  k=0        the floor: chunk g alone, no context at all
  k=128..    real tokens, natural contiguous positions -- an ORACLE, since the
             model gets the actual text the carry is trying to summarise
  carry      the true chained state from chunks 1..g-1 (== the carry ablation's
             "carried" condition)

Headline outputs
  ceiling(k)       = L(k=0) - L(k real tokens)      how much context is worth
  carry_delta      = L(k=0) - L(carry)              what we actually recover
  carry_equiv_tok  the k whose ceiling matches carry_delta, by interpolation --
                   "our 32-vector carry is worth about N tokens of real context"

Two optional decompositions, both off by default (2026-08-04).  The headline
fraction-recovered treats the ceiling as one number, and it is not: it is spread
over positions inside the chunk, and part of it is ALREADY supplied by the
tokens the model can see anyway.

  --carry_plus K   also score (carry AND K real preceding tokens) together, and
                   report the RESIDUAL   L(K alone) - L(carry + K).
                   The plain carry_delta is measured against a k=0 floor no
                   deployed model ever sits at -- the last K tokens of the
                   previous chunk are free at inference.  If the residual is ~0
                   the carry is a local-context surrogate and the addressable
                   headroom is (best ceiling - ceiling(K)), not the whole
                   ceiling.  That changes the denominator, i.e. the number the
                   paper reports.

  --pos_buckets N  split every condition's NLL into N equal position bands
                   inside the chunk.  Costs no extra forwards.  Context
                   substitution is a boundary effect, so the oracle's nats
                   should be front-loaded; read the CARRY's profile against the
                   ORACLE's, not against flat.  Tracking it = stitching the
                   seam; staying flat = supplying something that is not the
                   previous chunk's text.  It also prices the chunk-length lever
                   -- halving the chunk doubles the boundaries, so a
                   front-loaded ceiling means total recoverable nats scale with
                   the number of chunks, not just the per-boundary delta.

A NOTE ON WHAT THE CEILING BOUNDS (2026-08-04).  The oracle bounds SUBSTITUTING
FOR THE PREVIOUS CHUNK'S TEXT.  It is not a bound on a carry used as working
memory for the model's own intermediate computation: the measured ceiling is
this base model's ability to exploit raw tokens, and that ability collapses at a
2048-token window (-0.977).  A carry holding precomputed results can exceed it,
because it delivers what the model would otherwise spend depth re-deriving.
Recovery above 100% is therefore a reportable outcome, not a bug -- it is what
the "carry BEATS every measured oracle" branch below exists to say.

Read it two ways.  If the ceiling is large and the carry recovers a few percent,
the mechanism is the problem.  If the ceiling is itself ~0.02-0.03 nats, then
Track A's +0.024 was already near it and the line is a characterised negative
with a MEASURED bound rather than an open question.  The project's own
teacher-advantage probe measured -0.98 nats at W=2048 on this base (McLeish
continued-pretrained it at max_length=1024), so a ceiling that goes NEGATIVE at
large k is an expected and reportable outcome, not a bug -- it is the concrete
statement of "the base cannot use a long window", which is the motivation for
chunking in the first place.

Needs no memory: run it on the BASE model for the pure ceiling, or on an arm to
get the ceiling and the fraction recovered in one job.

Usage:
    python evals/eval_context_ceiling.py --model_name ckpts/olmo8-cortex \\
        --data data/pg19_olmo_val_len4096 --T 8
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from model_utils import load_checkpoint, has_cross_state, to_num_steps

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Context ceiling / oracle sweep")
    p.add_argument("--checkpoint",   type=str, default=None)
    p.add_argument("--model_name",   required=True)
    p.add_argument("--memory_slots", type=int, default=None)
    p.add_argument("--T",            type=int, default=None,
                   help="Recurrence depth (None = config mean_recurrence; set it "
                        "explicitly, retrofit-derived configs inherit 32)")
    p.add_argument("--data",         required=True)
    p.add_argument("--n_chunks",     type=int, default=4,
                   help="Sub-windows per sample (match training cross_chunks)")
    p.add_argument("--context_lens", type=int, nargs="+",
                   default=[128, 256, 512, 1024],
                   help="real preceding tokens to prepend (k=0 is always run)")
    p.add_argument("--carry_plus", type=int, nargs="*", default=[],
                   help="also score carry AND k real preceding tokens together, "
                        "and report the residual L(k alone) - L(carry + k) -- "
                        "what the carry adds ON TOP of context that is free at "
                        "inference.  One extra forward per k per chunk.")
    p.add_argument("--pos_buckets", type=int, default=0,
                   help="split each chunk's NLL into N equal position bands "
                        "(0 = off).  Free -- no extra forwards.")
    p.add_argument("--max_examples", type=int, default=50, help="0 = all rows")
    p.add_argument("--seed",         type=int, default=1234)
    p.add_argument("--out_dir",      default="eval_results/context_ceiling")
    p.add_argument("--dtype",        default="bfloat16", choices=["float32", "bfloat16"])
    p.add_argument("--device",       default=None)
    return p.parse_args()


def _bucket_nll(ce, mc, n_buckets):
    """Mean NLL inside each of `n_buckets` equal position bands of the chunk.
    A band whose tokens are all masked out yields None rather than a nan, so a
    short final chunk cannot poison the pooled band means."""
    out = []
    for idx in torch.chunk(torch.arange(ce.numel()), n_buckets):
        w = float(mc[idx].sum())
        out.append(float((ce[idx] * mc[idx]).sum() / w) if w > 0 else None)
    return out


@torch.no_grad()
def _score(model, ctx, xc, yc, mc, state, num_steps, device, seed,
           bucket_out=None, n_buckets=0):
    """NLL of chunk `xc` given `ctx` real preceding tokens and/or carry `state`.

    The context is prepended as ordinary tokens with natural contiguous
    positions -- this is the oracle condition, so nothing about it should be
    special-cased.  Only the chunk's own positions are scored.

    `ctx` and `state` are independent: passing both is the carry_plus condition
    (the carry on top of context the model would have anyway), passing neither
    is the k=0 floor.

    When `bucket_out` is given it receives the per-position-band means for this
    same forward -- the decomposition is a slice of the CE vector already
    computed, so it costs nothing.
    """
    torch.manual_seed(seed)                      # identical s0 across conditions
    n_tok = int(mc.sum())
    if n_tok == 0:
        return None
    ids = xc if ctx is None or ctx.numel() == 0 else torch.cat([ctx, xc])
    off = 0 if ctx is None else ctx.numel()
    out = model(input_ids=ids.unsqueeze(0).to(device), num_steps=num_steps,
                m_cross_in=state, return_m_cross=False)
    logits = out["logits"][0, off:].float()
    ce = F.cross_entropy(logits, yc.to(device), reduction="none")
    mcd = mc.to(device)
    if bucket_out is not None and n_buckets:
        bucket_out.append(_bucket_nll(ce, mcd, n_buckets))
    return float((ce * mcd).sum() / n_tok)


@torch.no_grad()
def carry_before(model, x, n_chunks, g, num_steps, seed, device):
    """The true chained state after chunks 1..g-1 (None when the model has no
    cross state).  Same construction as the carry ablation's carried condition."""
    torch.manual_seed(seed)
    m_cross = None
    for gi, xc in enumerate(torch.chunk(x, n_chunks)):
        if gi >= g:
            break
        out = model(input_ids=xc.unsqueeze(0).to(device), num_steps=num_steps,
                    m_cross_in=m_cross, return_m_cross=True)
        m_cross = out.get("m_cross")
    return m_cross


def _paired(a, b):
    d = torch.tensor(a) - torch.tensor(b)
    se = float(d.std() / max(len(d), 1) ** 0.5) if len(d) > 1 else 0.0
    return float(d.mean()), se


def _equiv_tokens(ceiling, carry):
    """Interpolate the k whose oracle ceiling equals the carry delta.
    ceiling: list of (k, delta) sorted by k.  Returns None when the carry is
    outside the measured range (below zero, or above the largest ceiling)."""
    if carry <= 0 or not ceiling:
        return None
    prev_k, prev_d = 0, 0.0
    for k, d in ceiling:
        if d >= carry:
            if d == prev_d:
                return float(k)
            return prev_k + (carry - prev_d) * (k - prev_k) / (d - prev_d)
        prev_k, prev_d = k, d
    return None                                   # carry exceeds every ceiling


def main() -> None:
    args = parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32

    print(f"Loading: {args.model_name}  (overlay: {args.checkpoint})")
    model, cfg = load_checkpoint(args.checkpoint, args.model_name,
                                 args.memory_slots, dtype, device)
    with_carry = has_cross_state(model)
    num_steps = to_num_steps(args.T if args.T is not None else int(cfg.mean_recurrence))
    print(f"T={int(num_steps[0])}  n_chunks={args.n_chunks}  "
          f"carry condition: {'on' if with_carry else 'OFF (no cross state)'}")
    print(f"context lens: 0 + {args.context_lens}")

    cplus_lens = sorted(args.carry_plus) if with_carry else []
    if args.carry_plus and not with_carry:
        print("  (--carry_plus ignored: this model has no cross state)")
    elif cplus_lens:
        print(f"carry_plus:   carry + {cplus_lens} real tokens")
    nb = max(0, args.pos_buckets)
    if nb:
        print(f"position bands: {nb} per chunk")

    from datasets import load_from_disk
    ds = load_from_disk(args.data)
    n = len(ds) if args.max_examples == 0 else min(args.max_examples, len(ds))
    print(f"{args.data}: {len(ds)} rows, evaluating {n}")

    conds = [0] + sorted(args.context_lens)
    # per_chunk[cond][g] = per-sample losses; g indexes chunks 2..n_chunks
    per_chunk = {c: [[] for _ in range(args.n_chunks)] for c in conds}
    carry_ch = [[] for _ in range(args.n_chunks)]
    cplus_ch = {k: [[] for _ in range(args.n_chunks)] for k in cplus_lens}
    # pos_rows[label] = one band-list per accepted (sample, chunk), appended in
    # lockstep across labels so bands stay paired the same way the scalars are.
    labels = ([f"k{k}" for k in conds]
              + (["carry"] if with_carry else [])
              + [f"carry+{k}" for k in cplus_lens])
    pos_rows = {lab: [] for lab in labels} if nb else {}
    t_start = time.time()

    for si in range(n):
        row = ds[si]
        ids = torch.tensor(row["input_ids"], dtype=torch.long)
        mask = torch.tensor(row["attention_mask"], dtype=torch.float)
        x, y, ym = ids[:-1], ids[1:], mask[1:]
        seed = args.seed + si
        x_ch = torch.chunk(x, args.n_chunks)
        y_ch = torch.chunk(y, args.n_chunks)
        m_ch = torch.chunk(ym, args.n_chunks)
        off = 0
        for g in range(args.n_chunks):
            off += 0 if g == 0 else x_ch[g - 1].numel()
            if g == 0:
                continue                          # no preceding context to give
            bands = {lab: [] for lab in labels} if nb else {}
            vals = {}
            for k in conds:
                ctx = x[max(0, off - k):off] if k else None
                vals[k] = _score(model, ctx, x_ch[g], y_ch[g], m_ch[g], None,
                                 num_steps, device, seed,
                                 bands.get(f"k{k}"), nb)
            cv, cpv = None, {}
            if with_carry:
                st = carry_before(model, x, args.n_chunks, g, num_steps, seed, device)
                cv = _score(model, None, x_ch[g], y_ch[g], m_ch[g], st,
                            num_steps, device, seed, bands.get("carry"), nb)
                for k in cplus_lens:
                    # the SAME carry, plus context the model has for free at
                    # inference -- one extra forward, no extra write chain
                    cpv[k] = _score(model, x[max(0, off - k):off], x_ch[g],
                                    y_ch[g], m_ch[g], st, num_steps, device,
                                    seed, bands.get(f"carry+{k}"), nb)
            ok = (all(v is not None for v in vals.values())
                  and (cv is not None or not with_carry)
                  and all(v is not None for v in cpv.values()))
            if ok:
                for k in conds:
                    per_chunk[k][g].append(vals[k])
                if with_carry:
                    carry_ch[g].append(cv)
                    for k in cplus_lens:
                        cplus_ch[k][g].append(cpv[k])
                for lab in pos_rows:
                    pos_rows[lab].append(bands[lab][0])
        if si == 0:
            dt = time.time() - t_start
            print(f"  sample 1 took {dt:.1f}s -> ETA {dt * n / 60:.0f} min")
        elif (si + 1) % 10 == 0:
            print(f"  {si + 1}/{n} samples  [{(time.time() - t_start) / 60:.0f} min]")

    # ---- per-chunk table ---------------------------------------------------
    results = {}
    hdr = f"  {'Chunk':<7} {'N':>4}" + "".join(f" {('k=' + str(k)):>9}" for k in conds)
    if with_carry:
        hdr += f" {'carry':>9}"
    print("\n" + hdr)
    print("  " + "-" * (len(hdr) - 2))
    for g in range(1, args.n_chunks):
        if not per_chunk[0][g]:
            continue
        line = f"  {g + 1:<7} {len(per_chunk[0][g]):>4}"
        entry = {"n": len(per_chunk[0][g])}
        for k in conds:
            v = float(torch.tensor(per_chunk[k][g]).mean())
            line += f" {v:>9.4f}"
            entry[f"k{k}"] = v
        if with_carry:
            v = float(torch.tensor(carry_ch[g]).mean())
            line += f" {v:>9.4f}"
            entry["carry"] = v
        results[f"chunk{g + 1}"] = entry
        print(line)

    # ---- pooled chunks 2+ --------------------------------------------------
    base = [v for g in range(1, args.n_chunks) for v in per_chunk[0][g]]
    print(f"\n  chunks 2+ pooled (n={len(base)}), delta vs k=0 "
          f"(positive = that context HELPS):")
    ceiling = []
    for k in conds[1:]:
        vals = [v for g in range(1, args.n_chunks) for v in per_chunk[k][g]]
        d, se = _paired(base, vals)
        ceiling.append((k, d))
        results[f"ceiling_k{k}"] = {"delta": d, "se": se}
        print(f"    {k:>5} real tokens : {d:+.5f}  (SE {se:.5f})")

    if with_carry:
        vals = [v for g in range(1, args.n_chunks) for v in carry_ch[g]]
        d, se = _paired(base, vals)
        results["carry_delta"] = {"delta": d, "se": se}
        print(f"    {'carry':>5}             : {d:+.5f}  (SE {se:.5f})")
        eq = _equiv_tokens(ceiling, d)
        best_k, best_d = max(ceiling, key=lambda kd: kd[1]) if ceiling else (0, 0.0)
        results["carry_equiv_tokens"] = eq
        results["best_ceiling"] = {"k": best_k, "delta": best_d}
        print()
        if eq is not None:
            print(f"  => the compressed carry is worth about {eq:.0f} tokens of REAL context")
        elif d <= 0:
            print("  => the carry is worth LESS than no context at all; no equivalent exists")
        else:
            print(f"  => the carry BEATS every measured oracle (best: {best_d:+.5f} "
                  f"at k={best_k}); the ceiling is not the binding constraint")
        if best_d > 0:
            print(f"  => it recovers {100 * d / best_d:.1f}% of the best oracle "
                  f"({best_d:+.5f} at k={best_k} real tokens)")
        else:
            print(f"  => the ORACLE ITSELF is non-positive (best {best_d:+.5f} at "
                  f"k={best_k}): real context over this span does not help this "
                  f"model, so there is no ceiling for a carry to recover.  That is "
                  f"the measured form of 'the base cannot use a long window'.")

        # ---- carry on top of free context ---------------------------------
        if cplus_lens:
            print(f"\n  carry + real context (n={len(base)}).  residual = what the "
                  f"carry adds\n  ON TOP of context that is free at inference:")
            for k in cplus_lens:
                vals = [v for g in range(1, args.n_chunks) for v in cplus_ch[k][g]]
                alone = [v for g in range(1, args.n_chunks) for v in per_chunk[k][g]]
                tot, se_t = _paired(base, vals)          # vs the k=0 floor
                res, se_r = _paired(alone, vals)         # vs k real tokens alone
                results[f"carry_plus_k{k}"] = {
                    "delta_vs_k0": tot, "se_vs_k0": se_t,
                    "residual_vs_k_alone": res, "se_residual": se_r,
                }
                frac = (f"  = {100 * res / d:.0f}% of the standalone carry delta"
                        if d > 0 else "  (standalone carry delta <= 0)")
                print(f"    carry+{k:<5} : total {tot:+.5f} (SE {se_t:.5f})   "
                      f"residual {res:+.5f} (SE {se_r:.5f}){frac}")
            print("  A residual near zero means the carry is a LOCAL-CONTEXT")
            print("  SURROGATE: it re-supplies what the last k tokens already give,")
            print("  so the headroom worth chasing is (best ceiling - ceiling(k)),")
            print("  not the whole ceiling.")

    # ---- position bands ----------------------------------------------------
    if nb and pos_rows.get("k0"):
        print(f"\n  position bands inside the chunk (n={len(pos_rows['k0'])} "
              f"chunk-instances), delta vs k=0 in the SAME band:")
        edges = [f"{100 * i // nb}-{100 * (i + 1) // nb}%" for i in range(nb)]
        print("  " + f"{'condition':<14}" + "".join(f"{e:>12}" for e in edges))
        results["position_bands"] = {"n_buckets": nb, "bands": edges, "delta": {}}
        for lab in labels:
            if lab == "k0":
                continue
            row, keep = [], []
            for j in range(nb):
                a = [r[j] for r, s in zip(pos_rows["k0"], pos_rows[lab])
                     if r[j] is not None and s[j] is not None]
                b = [s[j] for r, s in zip(pos_rows["k0"], pos_rows[lab])
                     if r[j] is not None and s[j] is not None]
                keep.append(_paired(a, b)[0] if a else None)
                row.append(f"{keep[-1]:>+12.5f}" if a else f"{'-':>12}")
            results["position_bands"]["delta"][lab] = keep
            print("  " + f"{lab:<14}" + "".join(row))
        print("  Compare the carry's PROFILE to the oracle's, not to flat.  A")
        print("  front-loaded oracle is the boundary effect; a carry that tracks")
        print("  it is stitching the seam, a carry that stays flat across bands")
        print("  is supplying something that is not the previous chunk's text.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved -> {out_dir}")


if __name__ == "__main__":
    main()
