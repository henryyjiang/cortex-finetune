"""
Recall-probe read-out for the post-mortem `recall` limb (and every other limb,
and 38,146 itself as the zero point).  Forward-only.

The pack (tools/prepare_recall_mix.py via pace/pm_prep_packs.sh) puts two
facts -- 'the code word for <key> is "<value>"' -- in chunks 0..5 of an 8-chunk
row and a probe about ONE of them at the very end of chunk 7.  The answer is
reachable only through the carry: the fact is 2..7 chunks back, i.e. 0.5..1.75
ring laps, straddling the one-lap mark where the trained gates (E fg 0.17)
forget.  So answer NLL BY DISTANCE is a direct retention curve.

Conditions, all scoring the same answer tokens of chunk 8:
  carry    the row's own chain (the trained condition)
  nocarry  chunk 8 alone -- the answer-prior floor (and the control arm's only
           condition; it has no carry)
  donor    chunk 8 reading another row's carry (same columns and norms, wrong
           facts) -- separates "the carry helps the FORMAT" from "the carry
           holds THIS fact"
  carry - donor  is the number that says the fact survived.

Per row: answer NLL (mean over the answer tokens, closing quote included) and
`first_tok_hit` (argmax at the answer's first position == its first token).
Aggregated by distance (chunks between the fact's chunk and chunk 8).

    python evals/score_recall_probe.py --model_name ckpts/olmo-retrofit-cortex \
        --checkpoint cortex-pm/pm-recall/checkpoint_40146 $SETS \
        --data data/pg19_val_recall_len4096 --T 8 --out_dir ...
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evals.model_utils import (_unwrap, load_checkpoint,  # noqa: E402
                               parse_config_overrides, to_num_steps)
from evals.diag_chunk1_drift import (paired_ci, preserve_ring_pointer,  # noqa: E402
                                     row_chunks, seed_for)

PROBE_RE = re.compile(r'Question: What is the code word for (.+?)\?\nAnswer: "')


def locate(tok, xs, ys):
    """-> (answer label positions in the LAST chunk, fact chunk index, key) or None.

    Positions are indices into chunk 8's label vector ys[-1]; the answer starts
    at the first label whose prefix text ends with 'Answer: "'.
    """
    last_x, last_y = xs[-1].tolist(), ys[-1].tolist()
    n = len(last_y)
    tail_start = max(0, n - 48)
    text_before = lambda j: tok.decode(last_x[tail_start:j + 1])
    start = None
    for j in range(tail_start, n):
        if text_before(j).endswith('Answer: "'):
            start = j
            break
    if start is None:
        return None
    m = PROBE_RE.search(tok.decode(last_x[tail_start:start + 1]))
    if not m:
        return None
    key = m.group(1)
    needle = f"the code word for {key} is"
    fact_chunk = None
    for ci in range(len(xs) - 1):
        if needle in tok.decode(xs[ci].tolist()):
            fact_chunk = ci
    if fact_chunk is None:
        return None
    return list(range(start, n)), fact_chunk, key


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model_name", required=True)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--data", required=True)
    p.add_argument("--set", action="append", default=[])
    p.add_argument("--n_chunks", type=int, default=8)
    p.add_argument("--T", type=int, required=True)
    p.add_argument("--max_rows", type=int, default=400)
    p.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--boot", type=int, default=2000)
    p.add_argument("--out_dir", required=True)
    args = p.parse_args()

    from datasets import load_from_disk
    from transformers import AutoTokenizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ov = parse_config_overrides(args.set)
    model, cfg = load_checkpoint(args.checkpoint, args.model_name, None,
                                 getattr(torch, args.dtype), device,
                                 config_overrides=ov or None)
    tok = AutoTokenizer.from_pretrained(args.model_name)
    inner = _unwrap(model)
    buf = getattr(getattr(inner, "cortex", None), "prefix", None)
    has_mem = buf is not None
    if ov and not has_mem:
        print("FAILED: --set given but no prefix buffer built")
        return 2
    num_steps = to_num_steps(args.T)
    ds = load_from_disk(args.data)
    n = min(args.max_rows, len(ds))

    def run(x, state, sd):
        torch.manual_seed(sd)
        with torch.no_grad():
            out = model(input_ids=x[None].to(device), num_steps=num_steps,
                        m_cross_in=state, return_m_cross=True)
        return out["logits"][0].float(), out.get("m_cross")

    def states_into_last(xs, base):
        st = None
        for k, x in enumerate(xs[:-1]):
            _, st = run(x, st, seed_for(base, k))
        return st

    conds = ["carry", "nocarry", "donor"] if has_mem else ["nocarry"]
    recs, skipped = [], 0
    for si in range(n):
        ids = torch.tensor(ds[si]["input_ids"], dtype=torch.long)
        xs, ys = row_chunks(ids, args.n_chunks)
        loc = locate(tok, xs, ys)
        if loc is None:
            skipped += 1
            continue
        pos, fact_chunk, key = loc
        base = args.seed + si
        rec = {"row": si, "key": key, "fact_chunk": fact_chunk,
               "distance": args.n_chunks - 1 - fact_chunk}
        own = states_into_last(xs, base) if has_mem else None
        donor = None
        if has_mem:
            dj = (si + n // 2) % n
            dxs, _ = row_chunks(torch.tensor(ds[dj]["input_ids"], dtype=torch.long),
                                args.n_chunks)
            donor = states_into_last(dxs, args.seed + dj)
        for c in conds:
            st = {"carry": own, "nocarry": None, "donor": donor}[c]
            with (preserve_ring_pointer(buf) if has_mem else _null()):
                lg, _ = run(xs[-1], st, seed_for(base, args.n_chunks - 1))
            y = ys[-1].to(device)
            ce = F.cross_entropy(lg, y, reduction="none")
            rec[f"{c}_nll"] = float(ce[pos].mean())
            rec[f"{c}_hit"] = bool(int(lg[pos[0]].argmax()) == int(y[pos[0]]))
        recs.append(rec)
        if len(recs) % 50 == 0:
            print(f"  {len(recs)} rows", flush=True)

    def agg(rows):
        out = {"n": len(rows)}
        for c in conds:
            v = [r[f"{c}_nll"] for r in rows]
            out[f"{c}_nll"] = sum(v) / len(v) if v else None
            out[f"{c}_acc"] = (sum(r[f"{c}_hit"] for r in rows) / len(rows)) if rows else None
        if has_mem and rows:
            for a, b in (("donor", "carry"), ("nocarry", "carry")):
                m, lo, hi = paired_ci([r[f"{a}_nll"] - r[f"{b}_nll"] for r in rows],
                                      args.boot, args.seed)
                out[f"{a}_minus_{b}"] = {"mean": m, "ci_lo": lo, "ci_hi": hi}
        return out

    by_d = {}
    for r in recs:
        by_d.setdefault(r["distance"], []).append(r)
    report = {"instrument": "recall probe by distance",
              "when": datetime.now().isoformat(timespec="seconds"),
              "checkpoint": args.checkpoint, "has_memory": has_mem,
              "config": {"T": args.T, "data": args.data, "overrides": ov},
              "skipped_rows": skipped, "all": agg(recs),
              "by_distance": {str(d): agg(v) for d, v in sorted(by_d.items())},
              "records": recs}
    print(f"\n{'=' * 70}\nRECALL PROBE  n={len(recs)} (skipped {skipped})\n{'=' * 70}")
    print(f"  {'dist':>4} {'n':>4} " + " ".join(f"{c + '_nll':>12} {c + '_acc':>9}" for c in conds)
          + ("   donor-carry [95% CI]" if has_mem else ""))
    for d, a in [("all", report["all"])] + list(report["by_distance"].items()):
        line = f"  {d:>4} {a['n']:>4} " + " ".join(
            f"{a[c + '_nll']:>12.3f} {a[c + '_acc']:>9.3f}" for c in conds)
        if has_mem and "donor_minus_carry" in a:
            x = a["donor_minus_carry"]
            line += f"   {x['mean']:+.3f} [{x['ci_lo']:+.3f}, {x['ci_hi']:+.3f}]"
        print(line)
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "results.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(f"wrote {args.out_dir}/results.json")
    if not recs:
        print("FAILED: no row had a locatable probe and fact")
        return 1
    return 0


class _null:
    def __enter__(self):
        return self

    def __exit__(self, *e):
        return False


if __name__ == "__main__":
    raise SystemExit(main())
