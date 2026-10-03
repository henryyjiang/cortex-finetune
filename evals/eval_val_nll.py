"""Held-out chunked NLL, on the SAME geometry the run trains at.

THE INSTRUMENT BOTH 5B ARMS SHARE.  It is memory-agnostic by construction --
it calls the model's own forward with `m_cross_in` threaded chunk to chunk, and
that argument is ignored when `self.cortex is None` -- so the control and
cortex-final are scored by ONE piece of code on ONE pack.  That matters more
than it sounds: the paper's headline is their difference, and a difference
between two instruments is not a difference between two models.

WHAT IT IS FOR
--------------
1. IS THE MIX BREAKING THE MODEL?  The control switches corpus at step 11,444
   from FineWeb-Edu to a 50/50 PG-19 + FineWeb-Edu mix, and a rising training
   loss there is expected and uninformative -- it is a different corpus.  Run
   this on a FineWeb-Edu val pack and a PG-19 val pack SEPARATELY and the
   ambiguity disappears:

       PG-19 down, FineWeb-Edu up a little   the mix is working; the FW rise is
                                             ordinary forgetting, and the mix
                                             still contains 50% FineWeb-Edu so
                                             it should be small
       PG-19 down, FineWeb-Edu up a lot      catastrophic forgetting of the heal
                                             corpus; the 1.71 FW epoch count is
                                             not buying what it should
       BOTH up                               the model is degrading.  Stop and
                                             look at grad norms and the LR
                                             schedule before spending another
                                             link.
       PG-19 flat                            the mix is not teaching anything;
                                             check the switch link actually
                                             read the mix pack (watch_run.py).

2. PER-CHUNK POSITION, which is where a carry shows up at all.  Chunk 1 runs
   with NO incoming carry in both arms, so its NLL is pure local LM quality and
   is the honest "did the model break" number.  Chunks 2..N are the only place
   a memory can pay.  A control that is healthy and a memory arm that is
   healthy look IDENTICAL at chunk 1 and diverge after it; a memory arm that is
   quietly broken is worse everywhere, which chunk 1 exposes.

3. THE PAIRED DELTA against an earlier checkpoint (--baseline).  Per-sample
   NLLs are stored, so two runs of this tool on the SAME pack with the SAME
   sample order pair row by row and the bootstrap runs over the paired
   DIFFERENCES.  That cancels between-document variance, which is what makes
   ~0.01 nats resolvable at a few hundred samples instead of needing thousands.

WHAT IT IS NOT.  It cannot attribute a memory arm's advantage to the carry --
a lower NLL could be the extra columns acting as free capacity, which is
exactly what X_off measures and X_content does not.  For that, run
evals/eval_carry_2x2.py on the same checkpoint (pace/midrun_check.sbatch runs
both).  This tool prints that sentence in its own output.

TRAPS IT IS WRITTEN AROUND
  * --T IS MANDATORY ON A RETROFIT CHECKPOINT.  Those configs inherit
    mean_recurrence=32 from the untrained base, so an unset T evaluates an mr8
    run at 4x its trained depth and 4x the wall clock.  Finding 0c; it cost the
    B1 carry headline.
  * n_chunks must match the run's cross_chunks or the model is read at a
    geometry it never trained at.  Both 5B arms are cross_chunks 8.
  * labels are SHIFTED here.  The modeling file does not shift (RED 10).
  * chance margin is checked: a probe within 0.25 nats of ln(vocab) is scoring
    noise, and every delta computed from it is a difference of two noise levels.

USAGE
    python evals/eval_val_nll.py --model_name ckpts/olmo-retrofit-cortex \
        --checkpoint cortex-5b/c-chunked/model_only_chkpt_12000 \
        --data data/fineweb_edu_olmo_val_len4096 --n_chunks 8 --T 8 \
        --max_examples 400 --label fwedu@12000 \
        --out eval_results/val_nll/c-chunked/fwedu_12000.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from datetime import datetime

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model_utils import load_checkpoint, to_num_steps, _unwrap  # noqa: E402
from model_utils import parse_config_overrides  # noqa: E402
from cortex_memory.health import chance_margin  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model_name", required=True,
                   help="the graft-prepared BASE dir (its modeling file is "
                        "what gets loaded), not the checkpoint")
    p.add_argument("--checkpoint", default=None,
                   help="a train.py checkpoint dir or chkpt.pt to overlay")
    p.add_argument("--data", required=True, help="load_from_disk val pack")
    p.add_argument("--n_chunks", type=int, default=8,
                   help="MUST match the run's cross_chunks (both 5B arms: 8)")
    p.add_argument("--T", type=int, required=True,
                   help="recurrence depth.  MANDATORY: retrofit configs carry "
                        "mean_recurrence=32 and both 5B arms train at 8.")
    p.add_argument("--max_examples", type=int, default=400)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--dtype", default="float32",
                   choices=["float32", "bfloat16"])
    p.add_argument("--boot", type=int, default=2000)
    p.add_argument("--no_carry", action="store_true",
                   help="thread no carry between chunks even on a memory "
                        "model -- the in-arm carry-off floor.  On the control "
                        "this changes nothing and the tool says so.")
    p.add_argument("--expect_memory", action="store_true",
                   help="REFUSE to score if the rebuilt model has no prefix "
                        "buffer.  Pass this on the cortex arm.  Without it a "
                        "graft that failed to build is not an error here -- "
                        "the file is memory-agnostic by design, so it quietly "
                        "becomes the control and the arms' difference is zero "
                        "by construction rather than by measurement.")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                   help="force graft-building config flags.  REQUIRED on a "
                        "memory checkpoint: --model_name loads the BASE dir's "
                        "config, which carries no cortex flags at all.  The "
                        "control needs NONE of these, and passing them to a "
                        "control checkpoint would build a mechanism that "
                        "never trained.")
    p.add_argument("--label", default=None,
                   help="a name for this measurement, carried into the JSON "
                        "and printed in the paired table")
    p.add_argument("--baseline", default=None,
                   help="an earlier JSON from this tool on the SAME pack.  "
                        "Reports the paired per-sample delta with a bootstrap "
                        "CI over the shared rows.")
    p.add_argument("--out", default=None, help="write the record as JSON")
    return p.parse_args()


def boot_ci(vals, boot: int, seed: int, level: float = 0.95):
    """Percentile bootstrap of the mean.  Returns (mean, lo, hi)."""
    if not vals:
        return None, None, None
    n = len(vals)
    mean = sum(vals) / n
    if boot <= 0 or n < 2:
        return mean, None, None
    rng = random.Random(seed)
    means = []
    for _ in range(boot):
        s = 0.0
        for _ in range(n):
            s += vals[rng.randrange(n)]
        means.append(s / n)
    means.sort()
    lo = means[int((1.0 - level) / 2 * boot)]
    hi = means[min(boot - 1, int((1.0 + level) / 2 * boot))]
    return mean, lo, hi


def chain_nll(model, xs, ys, num_steps, device, seed, carry: bool):
    """Per-chunk NLL down one row, with the carry threaded exactly as trained.

    The seed is set once per row and controls `initialize_state`'s
    trunc_normal_ draw for s0, so two checkpoints scored at the same seed see
    the same loop initialisation and the pairing is not diluted by it.
    """
    torch.manual_seed(seed)
    state, per_chunk, tot, ntok = None, [], 0.0, 0
    for xc, yc in zip(xs, ys):
        # no_grad is load-bearing: `state` carries the graph to the next chunk,
        # so without it the whole chain's activations are held at once.
        with torch.no_grad():
            out = model(input_ids=xc.unsqueeze(0).to(device),
                        num_steps=num_steps,
                        m_cross_in=state if carry else None,
                        return_m_cross=True)
        new = (out.get("m_cross") if isinstance(out, dict)
               else getattr(out, "m_cross", None))
        state = new if carry else None
        logits = (out["logits"] if isinstance(out, dict) else out.logits)[0].float()
        ce = F.cross_entropy(logits, yc.to(device), reduction="none")
        n = int(yc.numel())
        loss = float(ce.mean())
        per_chunk.append(loss)
        tot += loss * n
        ntok += n
    return per_chunk, (tot / ntok if ntok else None)


def paired_report(cur: dict, base_path: str, boot: int, seed: int) -> dict:
    """Pair this run against an earlier JSON, row by row, on the same pack."""
    try:
        with open(base_path) as fh:
            base = json.load(fh)
    except OSError as exc:
        print(f"  (--baseline unreadable: {exc})")
        return {}
    for key in ("data", "n_chunks", "T"):
        if base["config"].get(key) != cur["config"].get(key):
            print(f"  REFUSING the paired delta: baseline {key}="
                  f"{base['config'].get(key)!r} against this run's "
                  f"{cur['config'].get(key)!r}.  Pairing across a different "
                  f"pack or geometry compares two different questions.")
            return {}
    a = {r["row"]: r for r in base["per_sample"]}
    b = {r["row"]: r for r in cur["per_sample"]}
    shared = sorted(set(a) & set(b))
    if not shared:
        print("  (--baseline shares no sample rows with this run)")
        return {}
    d_all = [b[i]["nll"] - a[i]["nll"] for i in shared]
    d_c1 = [b[i]["per_chunk"][0] - a[i]["per_chunk"][0] for i in shared]
    d_rest = [(sum(b[i]["per_chunk"][1:]) - sum(a[i]["per_chunk"][1:]))
              / max(1, len(b[i]["per_chunk"]) - 1) for i in shared]
    out = {"baseline": base_path, "baseline_label": base.get("label"),
           "shared_rows": len(shared)}
    for name, vals in (("all", d_all), ("chunk1", d_c1), ("chunks2plus", d_rest)):
        m, lo, hi = boot_ci(vals, boot, seed)
        out[name] = {"delta": m, "lo": lo, "hi": hi}
    print()
    print(f"  PAIRED against {base.get('label') or base_path} "
          f"({len(shared)} shared rows).  Positive = THIS checkpoint is WORSE.")
    print(f"    {'window':<14} {'delta':>10}  {'95% CI':>22}")
    for name in ("all", "chunk1", "chunks2plus"):
        r = out[name]
        ci = (f"[{r['lo']:+.4f}, {r['hi']:+.4f}]"
              if r["lo"] is not None else "-")
        print(f"    {name:<14} {r['delta']:>+10.4f}  {ci:>22}")
    print("    chunk1 is the honest 'did the model break' number: no carry "
          "exists there in either arm.")
    return out


def main() -> int:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32 if args.dtype == "float32" else torch.bfloat16
    overrides = parse_config_overrides(args.set)

    # TWO values: load_checkpoint returns (model, config).  Binding the tuple
    # to `model` does NOT fail where it happens -- `_unwrap` hands a tuple
    # straight back and `getattr(tuple, "cortex")` is None, so the cortex arm
    # reports `memory=NO (control path)` and scores with the carry OFF.  The
    # TypeError at the first forward is the lucky part: without it this file
    # would emit plausible NLLs for both arms with the memory disabled in
    # both, and the headline difference would be zero BY CONSTRUCTION.
    model, _cfg = load_checkpoint(args.checkpoint, args.model_name, None, dtype,
                                  device, config_overrides=overrides or None)
    cortex = getattr(_unwrap(model), "cortex", None)
    has_memory = cortex is not None and getattr(cortex, "prefix", None) is not None
    carry = has_memory and not args.no_carry
    print(f"[val_nll] memory={'yes' if has_memory else 'NO (control path)'}  "
          f"carry={'threaded' if carry else 'off'}")
    if args.expect_memory and not has_memory:
        print("FAILED: --expect_memory, but the rebuilt model has no prefix "
              "buffer, so this would score the CONTROL path and report it as "
              "the cortex arm.  Check the --set list reached the graft "
              "(use_memory/prefix_memory/accum_vecs above) and that the "
              "overlay applied with 0 missing keys.")
        return 2
    if args.no_carry and not has_memory:
        print("[val_nll] --no_carry on a model with no memory changes nothing; "
              "the number is the same as without it.")

    num_steps = to_num_steps(args.T)
    from datasets import load_from_disk
    ds = load_from_disk(args.data)
    n = len(ds) if args.max_examples == 0 else min(args.max_examples, len(ds))

    per_sample, chunk_cols = [], [[] for _ in range(args.n_chunks)]
    for si in range(n):
        ids = torch.tensor(ds[si]["input_ids"], dtype=torch.long)
        if ids.numel() < args.n_chunks * 8:
            continue
        # SHIFTED: the modeling file does not shift, and passing labels=ids is
        # RED 10 -- the loss pins to ln(vocab) and every delta is noise.
        x, y = ids[:-1], ids[1:]
        keep = (x.numel() // args.n_chunks) * args.n_chunks
        xs = list(torch.chunk(x[:keep], args.n_chunks))
        ys = list(torch.chunk(y[:keep], args.n_chunks))
        pc, nll = chain_nll(model, xs, ys, num_steps, device,
                            args.seed + si, carry)
        if nll is None:
            continue
        per_sample.append({"row": si, "nll": nll, "per_chunk": pc})
        for ci, v in enumerate(pc):
            chunk_cols[ci].append(v)
        if len(per_sample) % 50 == 0:
            print(f"  {len(per_sample)} samples", flush=True)

    if not per_sample:
        print("FAILED: no scoreable rows in the pack.")
        return 2

    all_nll = [r["nll"] for r in per_sample]
    rest = [sum(r["per_chunk"][1:]) / max(1, len(r["per_chunk"]) - 1)
            for r in per_sample]
    m_all, lo_all, hi_all = boot_ci(all_nll, args.boot, args.seed)
    m_c1, lo_c1, hi_c1 = boot_ci([r["per_chunk"][0] for r in per_sample],
                                 args.boot, args.seed + 1)
    m_rest, lo_rest, hi_rest = boot_ci(rest, args.boot, args.seed + 2)

    vocab = int(getattr(_unwrap(model).config, "vocab_size", 0) or 0)
    health = chance_margin(all_nll, vocab)

    record = {
        "instrument": "held-out chunked val NLL",
        "when": datetime.now().isoformat(timespec="seconds"),
        "label": args.label,
        "model_name": args.model_name,
        "config": {"data": args.data, "checkpoint": args.checkpoint,
                   "n_chunks": args.n_chunks, "T": args.T,
                   "dtype": args.dtype, "samples": len(per_sample),
                   "has_memory": has_memory, "carry": carry,
                   "overrides": overrides},
        "nll": {"all": {"mean": m_all, "lo": lo_all, "hi": hi_all},
                "chunk1": {"mean": m_c1, "lo": lo_c1, "hi": hi_c1},
                "chunks2plus": {"mean": m_rest, "lo": lo_rest, "hi": hi_rest}},
        "per_chunk": [
            {"chunk": i + 1, "mean": (sum(c) / len(c)) if c else None,
             "n": len(c)} for i, c in enumerate(chunk_cols)],
        "health": health,
        "per_sample": per_sample,
    }

    print()
    print(f"=== val NLL | {args.label or args.checkpoint or args.model_name} ===")
    print(f"    pack {args.data}   {len(per_sample)} rows   "
          f"T={args.T}  n_chunks={args.n_chunks}")
    print(f"    {'window':<14} {'NLL':>9}  {'95% CI':>22}")
    for name, m, lo, hi in (("all", m_all, lo_all, hi_all),
                            ("chunk1 (no carry)", m_c1, lo_c1, hi_c1),
                            ("chunks 2+", m_rest, lo_rest, hi_rest)):
        ci = f"[{lo:.4f}, {hi:.4f}]" if lo is not None else "-"
        print(f"    {name:<14} {m:>9.4f}  {ci:>22}")
    print("    per chunk: " + "  ".join(
        f"{r['chunk']}:{r['mean']:.3f}" for r in record["per_chunk"]
        if r["mean"] is not None))

    if health.get("at_chance"):
        print(f"    !! AT CHANCE: mean {health['mean_nll']:.3f} against "
              f"ln(vocab) {health['chance']:.3f} (margin "
              f"{health['margin']:.3f} < 0.25).  This is scoring NOISE and "
              f"every delta from it is a difference of two noise levels.")
    else:
        print(f"    chance margin {health['margin']:.3f} nats "
              f"(mean {health['mean_nll']:.3f} vs ln(vocab) "
              f"{health['chance']:.3f})")

    if args.baseline:
        record["paired"] = paired_report(record, args.baseline, args.boot,
                                         args.seed + 3)

    print()
    if has_memory and carry:
        print("    NOTE a lower NLL here does NOT attribute the gain to the "
              "carry's CONTENTS -- extra columns are free capacity, which is "
              "what X_off measures.  Pair this with evals/eval_carry_2x2.py.")

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump(record, fh, indent=2)
        print(f"    wrote {args.out}")
    return 1 if health.get("at_chance") else 0


if __name__ == "__main__":
    sys.exit(main())
