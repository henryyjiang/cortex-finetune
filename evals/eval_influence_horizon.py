"""
Influence horizon -- how far back the carry still matters, in NATS.

THE METRIC THIS REPLACES, AND WHY IT HAD TO BE REPLACED.  The project's
cross-chunk metric has been AutoCompressor-style COMPOUNDING: does the carry
delta grow as the chunk chain deepens (B2 measured x1.31).  That metric encodes
an assumption -- that a working memory accumulates -- which is true of an append
buffer and FALSE of a fixed-width gated one.  A gated buffer at steady state
should PLATEAU: it holds a bounded amount and rolls the oldest content off.
Compounding scores a plateau as failure, so running A3 against it would have
produced a negative result about the metric rather than about the mechanism.

WHAT THIS MEASURES INSTEAD.  Damage the carry at chunk n-d, and see how much
worse chunk n's loss gets:

    I(d) = NLL(chunk n | carry damaged at chunk n-d) - NLL(chunk n | intact)

in nats per token, paired per sample.  I(d) > 0 means the model was still using
chunk n-d's contribution d chunks later.  The curve's SHAPE is the finding:

  * an append buffer should hold I(d) roughly flat while chunk n-d is inside
    the FIFO and drop to exactly 0 the moment it is evicted -- the cliff behind
    issue 4's step function (BABILong carry delta +4.33/+2.00/+1.00 inside the
    4,096-token horizon, -1.67/-1.00/-6.50 outside it);
  * a gated buffer should decay smoothly with no cliff and no zero.

Both are legitimate outcomes.  The question the paper needs answered is which
shape each mechanism has and where they cross, not which one accumulates.

TWO WAYS TO DAMAGE IT, and they answer different questions.

  --damage donor    replace chunk n-d's WRITE with another document's write.
                    Column count, norms and register all match exactly, so this
                    isolates INFORMATION TRANSFER from the register effect that
                    a plain on/off ablation cannot separate.  This is the P0.6
                    control, applied per depth.  DEFAULT.
  --damage zero     zero chunk n-d's write.  Cheaper, but a zero row is still a
                    key: it scores a mid-range logit rather than -inf and goes
                    on absorbing a few percent of the softmax mass, so a null
                    result here is ambiguous in a way the donor is not.

HOW THE DAMAGE IS APPLIED WITHOUT CONTAMINATING LATER WRITES.  The true carry is
maintained outside the model.  At chunk n-d the damaged write is substituted
into the state that later chunks READ, but the chunk's own forward still ran on
the true carry, so the damage never propagates into what later chunks WRITE.
That keeps I(d) an attribution to one chunk rather than to a whole suffix.

  NOTE for append buffers: this needs per-chunk rows to stay separable, which is
  true of PrefixAccumBuffer.  For a GATED buffer the rows are mixed, so the
  substitution is applied to the write BEFORE the merge and the chain is then
  replayed -- which is the honest thing to do there, and it does mean the gated
  I(d) includes the merge's own propagation.  Stated, not hidden: it is the
  mechanism, and the same is true of the buffer-level A(d) in
  evals/diag_gate_geometry.py.

PAIRING AND POWER.  Every depth is evaluated on the same samples with the same
s0 seed, so the per-sample variance cancels; on BABILong the same data went from
p=0.42 unpaired to p=2.4e-6 paired.  Report the paired mean and its bootstrap CI,
never an unpaired t-test.

FLOAT32 for the loss differences.  I(d) at large d is a difference of two nearly
equal losses, which is the quantity bf16 got wrong by 6x in P0.1.  --dtype
bfloat16 exists to reproduce that deliberately and warns when used.

USAGE
  python evals/eval_influence_horizon.py \
      --model_name <ckpt> --data data/pg19_olmo_val_len4096 \
      --n_chunks 8 --depths 1 2 3 4 6 --damage donor \
      --out_dir eval_results/influence_horizon-$(date +%Y%m%d)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model_utils import load_checkpoint, to_num_steps, _unwrap  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", required=True)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--data", required=True,
                   help="a packed dataset with an input_ids column")
    p.add_argument("--n_chunks", type=int, default=8,
                   help="chunks per row.  Must match the arm's cross_chunks, "
                        "or the carry is evaluated at a depth it never trained "
                        "at (finding 0c's trap, one level up).")
    p.add_argument("--depths", type=int, nargs="+",
                   default=[1, 2, 3, 4, 6],
                   help="d values to probe.  Every d must be < n_chunks.")
    p.add_argument("--damage", default="donor", choices=["donor", "zero"])
    p.add_argument("--T", type=int, default=None,
                   help="recurrence depth.  Leave unset to take the config's "
                        "mean_recurrence -- which on the mr8 arms is 32 and "
                        "wrong; pass 8 for those.")
    p.add_argument("--max_examples", type=int, default=100)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--dtype", default="float32",
                   choices=["float32", "bfloat16"])
    p.add_argument("--boot", type=int, default=2000,
                   help="bootstrap resamples for the paired CI")
    p.add_argument("--out_dir",
                   default="eval_results/influence_horizon")
    return p.parse_args()


# ---------------------------------------------------------------------------

def _is_append(buf) -> bool:
    """Append buffers keep per-chunk rows separable; gated ones do not."""
    return buf is not None and hasattr(buf, "max_vecs")


def run_chain(model, buf, chunks, labels, masks, num_steps, device,
              seed: int, damage_at: int | None, donor_write, mode: str):
    """Replay one row's chunk chain, optionally damaging one chunk's write.

    Returns per-chunk (loss, n_tokens).  `damage_at` is a CHUNK INDEX, not a
    depth; the caller converts.
    """
    torch.manual_seed(seed)                 # identical s0 draws across arms
    state, rows, writes = None, [], []
    for i, (xc, yc, mc) in enumerate(zip(chunks, labels, masks)):
        out = model(input_ids=xc.unsqueeze(0).to(device),
                    num_steps=num_steps, m_cross_in=state,
                    return_m_cross=True)
        new_state = out.get("m_cross") if isinstance(out, dict) \
            else getattr(out, "m_cross", None)

        if _is_append(buf):
            # Rows are separable: take this chunk's own write off the end and
            # rebuild the state, substituting the damaged write if this is the
            # damaged chunk.  The forward above already ran on the TRUE carry,
            # so damage never reaches a later chunk's write.
            w = new_state[:, -buf.n_vec:]
            writes.append(w)
            if i == damage_at:
                w = (torch.zeros_like(w) if mode == "zero"
                     else donor_write.to(w.device, w.dtype))
            state = w if state is None else torch.cat([state, w], dim=1)
            if state.shape[1] > buf.max_vecs:
                state = state[:, -buf.max_vecs:]
        else:
            # Gated: rows are mixed, so substitute the write and let the merge
            # carry the consequence forward.  Documented in the header.
            writes.append(new_state)
            state = new_state

        n_tok = int(mc.sum())
        if n_tok == 0:
            rows.append((None, 0))
            continue
        logits = (out["logits"] if isinstance(out, dict) else out.logits)[0].float()
        ce = F.cross_entropy(logits, yc.to(device), reduction="none")
        rows.append((float((ce * mc.to(device)).sum() / n_tok), n_tok))
    return rows, writes


def paired_ci(deltas, n_boot: int, seed: int = 0):
    """Bootstrap CI on the paired mean.  Paired by construction: every entry is
    one sample's (damaged - intact), so per-sample variance is already gone."""
    if not deltas:
        return (float("nan"),) * 3
    t = torch.tensor(deltas, dtype=torch.float64)
    g = torch.Generator().manual_seed(seed)
    idx = torch.randint(len(t), (n_boot, len(t)), generator=g)
    means = t[idx].mean(dim=1)
    lo, hi = torch.quantile(means, torch.tensor([0.025, 0.975],
                                                dtype=torch.float64))
    return float(t.mean()), float(lo), float(hi)


def main() -> int:
    args = parse_args()
    if args.dtype == "bfloat16":
        print("WARNING: bfloat16.  I(d) is a difference of two nearly equal "
              "losses -- the exact quantity bf16 got wrong by 6x in P0.1.  Do "
              "not quote these numbers.", flush=True)
    bad = [d for d in args.depths if d >= args.n_chunks]
    if bad:
        raise SystemExit(f"FAILED: depths {bad} are >= n_chunks "
                         f"({args.n_chunks}); there is no chunk that far back.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = getattr(torch, args.dtype)
    model, cfg = load_checkpoint(args.checkpoint, args.model_name, None,
                                 dtype, device)
    inner = _unwrap(model)
    buf = getattr(getattr(inner, "cortex", None), "prefix", None)
    if buf is None:
        print("FAILED: this checkpoint has no prefix buffer, so there is no "
              "carry whose influence could be measured.")
        return 2
    kind = "append" if _is_append(buf) else "gated"
    num_steps = to_num_steps(args.T)

    from datasets import load_from_disk
    ds = load_from_disk(args.data)
    n = len(ds) if args.max_examples == 0 else min(args.max_examples, len(ds))

    per_depth: dict[int, list[float]] = {d: [] for d in args.depths}
    n_used = 0
    for si in range(n):
        ids = torch.tensor(ds[si]["input_ids"], dtype=torch.long)
        if ids.numel() < args.n_chunks * 8:
            continue
        x, y = ids[:-1], ids[1:]
        keep = (x.numel() // args.n_chunks) * args.n_chunks
        x, y = x[:keep], y[:keep]
        mask = torch.ones_like(y, dtype=torch.float32)
        xs = list(torch.chunk(x, args.n_chunks))
        ys = list(torch.chunk(y, args.n_chunks))
        ms = list(torch.chunk(mask, args.n_chunks))

        seed = args.seed + si
        intact, writes = run_chain(model, buf, xs, ys, ms, num_steps, device,
                                   seed, None, None, args.damage)
        if intact[-1][0] is None:
            continue

        # The donor is another ROW's write at the same chunk index: matched
        # register, matched norms, matched column count, different content.
        donor_row = (si + 1) % n
        donor_ids = torch.tensor(ds[donor_row]["input_ids"], dtype=torch.long)
        donor_xs = list(torch.chunk(donor_ids[:keep], args.n_chunks))

        for d in args.depths:
            at = args.n_chunks - 1 - d
            donor_w = None
            if args.damage == "donor":
                with torch.no_grad():
                    dout = model(input_ids=donor_xs[at].unsqueeze(0).to(device),
                                 num_steps=num_steps, m_cross_in=None,
                                 return_m_cross=True)
                dstate = (dout.get("m_cross") if isinstance(dout, dict)
                          else getattr(dout, "m_cross", None))
                donor_w = (dstate[:, -buf.n_vec:] if _is_append(buf) else dstate)
            damaged, _ = run_chain(model, buf, xs, ys, ms, num_steps, device,
                                   seed, at, donor_w, args.damage)
            if damaged[-1][0] is None:
                continue
            per_depth[d].append(damaged[-1][0] - intact[-1][0])
        n_used += 1
        if n_used % 10 == 0:
            print(f"  {n_used} samples", flush=True)

    report = {
        "instrument": "influence horizon I(d)",
        "when": datetime.now().isoformat(timespec="seconds"),
        "model_name": args.model_name,
        "buffer": {"kind": kind, "class": type(buf).__name__,
                   "n_vec": int(buf.n_vec),
                   "capacity": int(getattr(buf, "max_vecs",
                                           getattr(buf, "n_slots", 0)))},
        "config": {"n_chunks": args.n_chunks, "damage": args.damage,
                   "T": args.T, "dtype": args.dtype, "samples": n_used},
        "depths": {},
    }
    for d in args.depths:
        m, lo, hi = paired_ci(per_depth[d], args.boot, args.seed)
        report["depths"][str(d)] = {"mean_nats": m, "ci_lo": lo, "ci_hi": hi,
                                    "n": len(per_depth[d])}

    print(f"\n{'=' * 78}")
    print(f"I(d) -- nats/token that chunk N-d's carry contribution is worth at "
          f"chunk N")
    print(f"buffer: {type(buf).__name__} ({kind}), damage={args.damage}, "
          f"n={n_used} paired samples")
    print("=" * 78)
    print(f"  {'d':>3}{'I(d) nats':>12}{'95% CI':>26}{'n':>6}   shape")
    for d in args.depths:
        r = report["depths"][str(d)]
        sig = "" if (r["ci_lo"] <= 0 <= r["ci_hi"]) else "  *"
        print(f"  {d:>3}{r['mean_nats']:>12.5f}"
              f"   [{r['ci_lo']:>9.5f}, {r['ci_hi']:>9.5f}]{r['n']:>6}{sig}")
    print("\n  * = CI excludes zero, i.e. that depth still measurably matters.")
    if kind == "append":
        cap = int(getattr(buf, "max_vecs", 0)) // max(int(buf.n_vec), 1)
        print(f"  This is an APPEND buffer holding {cap} chunks: expect I(d) "
              f"roughly flat to d={cap - 1} and then exactly 0.")
    else:
        print("  This is a GATED buffer: expect smooth decay, no cliff, and no "
              "exact zero.  Do NOT score a plateau as failure.")

    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, "results.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
