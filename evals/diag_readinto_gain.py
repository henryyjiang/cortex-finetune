"""
P3.0 tiers 0.5 and 1 — DOES THE IN-LOOP READ SITE HAVE GAIN?

No training.  One forward-only sweep, ~10 min on one GPU, and it is the gate
that stands between here and the 8 GPU-hour oracle probe.

WHAT IT ANSWERS.  `pace/diag_s0.sbatch` established that `s0` is not a read
site: deleting its carried columns outright costs +4.1e-5 / -2.5e-5 nats
(opposite signs on the two arms, i.e. noise), and the loss stays flat out to
45x ||s0|| -- 1.8x the loop's own working state -- turning on only at an
injected norm of 175.8 against ||E|| = 171.0, where the substitution is
DROWNING E rather than being read.  P3.0's whole premise is that
`CortexGraft.read_into` is different.  That premise has never been measured.
This measures it, before anything is trained.

  tier 1    (--module xattn)    LatentRead AT INIT.  Sweep its scalar `gate`
                                over orders of magnitude and watch the loss.
  tier 0.5  (--module refresh)  LatentRefresh, the one-scalar Option 0 read.
                                Sweep `alpha`.  This is the arm that isolates
                                REPETITION from LEARNED EXTRACTION -- it tests
                                whether s0's columns are washed because the
                                loop OVERWRITES them at iteration 1, which the
                                s0 sweep cannot rule out.

THE AXIS IS A ROW NORM, MEASURED, NOT A GATE VALUE.  A gate of 0.1 means
nothing on its own.  The graft now records `_x_read_norm` (the post-adapter
state the delta is added to), `_z_row_norm` (what is being read) and
`_z_read_delta_norm` (what actually got injected), all fp32, so every cell is
reported as ||delta|| / ||x||.  RED 11 was exactly this: a sweep whose labels
meant something other than what it injected, off by sqrt(D) = 45.25.

THE CONTROL, AND IT IS THE POINT.  Every gate value is scored TWICE -- once on
the real carried Z and once with `latent_read_null` noise at the same scale.
At init the module is untrained, so the two are EXPECTED to agree; that is not
a failure, it is the baseline the oracle probe will be measured against.  What
the pair buys here is the ability to distinguish "this site has no gain" from
"the knob never landed", which is the RED 8/9/10 shape and the reason the s0
tool returns INVALID rather than (A) when nothing in it moves.

HOW TO READ IT -- FIXED HERE, BEFORE THE NUMBERS EXIST.

  RESPONDS at ||delta||/||x|| well below 1   ->  THE SITE HAS GAIN.  Proceed to
      tier 1.5, the oracle read probe.  This is the row that goes beside the s0
      table and justifies the redesign.
  FLAT across orders, up to ||delta|| ~ ||x||  ->  the new site is dead too.
      This CONTRADICTS the pre-registration's premise, and p30 S5 row 3 says
      what to do about it: stop and audit the instrument before spending
      anything.  Do not proceed to tier 1.5 on a flat tier 1.
  NOTHING MOVES AT ANY SCALE, including the oversized cell -> INVALID.  The
      knob is not connected; fix the instrument, not the architecture.
  REAL and NOISE separate AT INIT           ->  suspicious, not exciting.  An
      untrained random projection should not prefer real content.  Audit before
      reporting.

ONLY A READING THAT HOLDS ON BOTH ARMS IS ABOUT THE SITE.  The s0 sweep
pre-registered this and it caught the arm-dependence that exposed its own
swamp region.  Run a2 AND a3z; the flat region agreed to five decimals there
while the loud region was 7x apart.

EVAL MODE IS NOT OPTIONAL, AND THE LEVEL IS REPORTED.  RED 10: the walk and
gate 4 ran `inner.train()` and ranked three losses that were all at ln(vocab).
A table of differences is unreadable without the level it is a difference of,
so the intact loss and the margin below chance are printed and a run at chance
VETOES the verdict rather than being warned about above it.

USAGE
  python evals/diag_readinto_gain.py \
      --model_name ckpts/olmo-retrofit-cortex \
      --checkpoint cortex-retrofit/<run>/checkpoint_XXXX \
      --data data/pg19_olmo_val_len4096 --n_chunks 8 --T 8 \
      --module xattn \
      --set use_memory=true --set prefix_memory=gated --set latent_carry=true \
      --set latent_read=xattn --set latent_s0_read=false \
      --out_dir eval_results/readinto_gain-$(date +%Y%m%d)
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import datetime

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model_utils import load_checkpoint, to_num_steps, _unwrap  # noqa: E402
from model_utils import parse_config_overrides  # noqa: E402

#: Gate values, spanning four orders.  0.0 is the intact control (the module
#: present but contributing exactly nothing), and the top cell is deliberately
#: oversized so the sweep can prove its own knob is connected -- a sweep that
#: is flat EVERYWHERE cannot tell "no gain" from "never landed".
DEFAULT_GATES = (0.0, 0.001, 0.01, 0.1, 1.0, 10.0)

#: Below this span across the IN-DISTRIBUTION cells, orders of magnitude of
#: injection have changed nothing.  Same value and same reasoning as the s0
#: tool: the question is orders of magnitude, not a CI.
FLAT_SPAN_NATS = 0.01

#: A cell must move at least this much for the knob to count as connected.
KNOB_LIVE_NATS = 0.01

#: Cells whose injected row norm exceeds this multiple of ||x|| are printed but
#: do NOT decide the verdict.  Above ||x|| the delta is not being READ, it is
#: replacing the loop's working state -- the same distinction RED 11 drew
#: between reading s0 and drowning E, at the site where the competing quantity
#: is x itself rather than the carried E row.
SWAMP_MULT_OF_X = 1.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("in-loop read-site gain (P3.0 tiers 0.5 / 1)")
    p.add_argument("--model_name", required=True)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--data", required=True)
    p.add_argument("--n_chunks", type=int, default=8)
    p.add_argument("--T", type=int, default=None,
                   help="RECURRENCE DEPTH.  config.mean_recurrence is 32 on "
                        "every B2-family checkpoint and is INHERITED from the "
                        "base recipe -- no arm ran there, they trained at 8.  "
                        "RED 8 was this exact omission reporting read_live "
                        "0.0195 for arms that ran 0.546.  Pass it.")
    p.add_argument("--module", choices=["xattn", "refresh"], default="xattn",
                   help="xattn = tier 1 (LatentRead at init); refresh = tier "
                        "0.5 (LatentRefresh, Option 0)")
    p.add_argument("--gates", type=float, nargs="+", default=list(DEFAULT_GATES),
                   help="the module's scalar gate / alpha")
    p.add_argument("--max_examples", type=int, default=50)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--dtype", default="float32",
                   choices=["float32", "bfloat16", "float16"],
                   help="float32: these are small differences of large states, "
                        "the class bf16 got wrong by 6x in P0.1")
    p.add_argument("--noise_std", type=float, default=0.02,
                   help="per-ELEMENT std for the Z null (NOT a row norm)")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    p.add_argument("--out_dir", default=None)
    return p.parse_args()


def summarize(cells: dict, norms: dict, chance: float) -> dict:
    """Means, deltas and the reading.  MODULE-LEVEL, not a closure inside
    main(), and that is RED 14's lesson rather than a style preference: the
    effects arithmetic there lived inside main(), so the code producing every
    number the instrument reported had no test and needed a GPU to reach.  A
    sign inversion survived in it for two days.

    `cells` maps cell name -> list of per-example nats.  `norms` maps cell name
    -> the MEASURED ||delta||/||x|| ratio for that cell.
    """
    mean = {k: sum(v) / len(v) for k, v in cells.items() if v}
    base = mean.get("real@0")
    delta = ({k: v - base for k, v in mean.items() if k != "real@0"}
             if base is not None else {})

    ood = sorted(k for k, r in norms.items()
                 if r is not None and r > SWAMP_MULT_OF_X and k in mean)
    kept = [v for k, v in mean.items() if k not in ood]
    span_in = (max(kept) - min(kept)) if kept else 0.0
    span_all = (max(mean.values()) - min(mean.values())) if mean else 0.0

    knob_live = any(abs(d) > KNOB_LIVE_NATS for d in delta.values())
    at_chance = base is not None and (chance - base) < 1.0

    # The real-vs-noise pair, per gate.  At init these SHOULD agree; a
    # separation is a reason to audit, not to celebrate.
    pairs = {}
    for k in mean:
        if k.startswith("real@"):
            g = k.split("@", 1)[1]
            other = "noise@" + g
            if other in mean:
                pairs[g] = mean[k] - mean[other]
    content_at_init = max((abs(v) for v in pairs.values()), default=0.0)

    if at_chance:
        reading = "INVALID_at_chance"
    elif not knob_live:
        reading = "INVALID_knob_unproven"
    elif span_in < FLAT_SPAN_NATS:
        reading = "DEAD_no_gain_in_distribution"
    else:
        reading = "LIVE_site_has_gain"

    return {
        "cells": mean,
        "delta_from_intact": delta,
        "delta_norm_over_x": dict(norms),
        "ood_cells": ood,
        "span_nats": span_all,
        "span_in_dist_nats": span_in,
        "knob_live": knob_live,
        "real_minus_noise": pairs,
        "content_effect_at_init": content_at_init,
        "reading": reading,
        "intact": {"mean_nats": base, "chance_nats": chance,
                   "margin_below_chance": (None if base is None
                                           else chance - base),
                   "at_chance": (None if base is None
                                 else bool(chance - base < 1.0))},
    }


def score_chunk(model, ids, labels, carry, num_steps, device, seed):
    torch.manual_seed(seed)
    with torch.no_grad():
        out = model(input_ids=ids.unsqueeze(0).to(device),
                    labels=labels.unsqueeze(0).to(device),
                    m_cross_in=carry, return_m_cross=False,
                    num_steps=num_steps)
    return float(out["loss"] if isinstance(out, dict) else out.loss)


def prime(model, xs, num_steps, device, seed):
    """Run every chunk but the last, returning the carry the last one reads."""
    carry = None
    for i, ids in enumerate(xs[:-1]):
        torch.manual_seed(seed + i)
        with torch.no_grad():
            out = model(input_ids=ids.unsqueeze(0).to(device),
                        m_cross_in=carry, return_m_cross=True,
                        num_steps=num_steps)
        carry = out["m_cross"] if isinstance(out, dict) else out.m_cross
    return carry


def set_gate(reader, value: float) -> None:
    p = getattr(reader, "gate", None)
    if p is None:
        p = getattr(reader, "alpha", None)
    if p is None:
        raise RuntimeError(
            f"{type(reader).__name__} has neither `gate` nor `alpha`; the "
            "sweep does not know what to turn.")
    with torch.no_grad():
        p.fill_(float(value))


def main() -> int:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    overrides = parse_config_overrides(args.set)
    model, cfg = load_checkpoint(args.checkpoint, args.model_name, None,
                                 getattr(torch, args.dtype), device,
                                 config_overrides=overrides or None)
    inner = _unwrap(model)
    cortex = getattr(inner, "cortex", None)
    if cortex is None or getattr(cortex, "prefix", None) is None:
        print("FAILED: no prefix buffer on this checkpoint.  Force the "
              "geometry with --set (see pace/diag_readinto.sbatch).")
        return 2
    if not getattr(cortex, "latent_carry", False):
        print("FAILED: latent_carry is off, so nothing is carried for the "
              "in-loop read to read.  This needs a dual-channel arm (a2 / "
              "a3z) and --set latent_carry=true.")
        return 2
    reader = getattr(cortex, "latent_reader", None)
    if reader is None:
        print("FAILED: no in-loop read module on this model.  Pass "
              f"--set latent_read={args.module}.  Without it `read_into`'s Z "
              "branch never fires and this sweep would report a flat curve "
              "for a module that does not exist -- the RED 8/9/10 shape.")
        return 2
    want = {"xattn": "LatentRead", "refresh": "LatentRefresh"}[args.module]
    if type(reader).__name__ != want:
        print(f"FAILED: --module {args.module} wants {want} but the model "
              f"built {type(reader).__name__}.  The config and the flag "
              "disagree; fix the --set rather than reporting the wrong tier.")
        return 2
    num_steps = to_num_steps(args.T)

    from datasets import load_from_disk
    ds = load_from_disk(args.data)
    n = min(args.max_examples, len(ds)) if args.max_examples else len(ds)

    names = []
    for g in args.gates:
        names.append("real@%g" % g)
        names.append("noise@%g" % g)
    cells = {k: [] for k in names}
    ratios = {k: [] for k in names}
    x_norms, z_norms = [], []
    used = 0

    for si in range(n):
        ids = torch.tensor(ds[si]["input_ids"], dtype=torch.long)
        if ids.numel() < args.n_chunks * 8:
            continue
        x, y = ids[:-1], ids[1:]
        keep = (x.numel() // args.n_chunks) * args.n_chunks
        xs = list(torch.chunk(x[:keep], args.n_chunks))
        ys = list(torch.chunk(y[:keep], args.n_chunks))
        seed = args.seed + si

        # Prime with the module OFF, so every cell reads the same carry and the
        # sweep is measuring the READ and not a chain that diverged upstream.
        cortex.latent_read_null = None
        set_gate(reader, 0.0)
        carry = prime(model, xs, num_steps, device, seed)
        if carry is None:
            continue

        for g in args.gates:
            set_gate(reader, g)
            for kind in ("real", "noise"):
                cortex.latent_read_null = (None if kind == "real"
                                           else ("noise", args.noise_std, seed))
                name = "%s@%g" % (kind, g)
                cells[name].append(
                    score_chunk(model, xs[-1], ys[-1], carry, num_steps,
                                device, seed + 977))
                xn = getattr(cortex, "_x_read_norm", None)
                dn = getattr(cortex, "_z_read_delta_norm", None)
                if xn and dn is not None:
                    ratios[name].append(dn / xn)
                if kind == "real" and g == 0.0:
                    if xn:
                        x_norms.append(xn)
                    zn = getattr(cortex, "_z_row_norm", None)
                    if zn:
                        z_norms.append(zn)
        cortex.latent_read_null = None
        set_gate(reader, 0.0)
        used += 1

    if not used:
        print("FAILED: no example survived chunking.  Check --data and "
              "--n_chunks.")
        return 2

    mean_ratio = {k: (sum(v) / len(v) if v else None) for k, v in ratios.items()}
    chance = math.log(int(getattr(cfg, "vocab_size", 0)) or 1)
    rec = summarize(cells, mean_ratio, chance)
    rec.update({
        "module": type(reader).__name__,
        "tier": {"xattn": "1", "refresh": "0.5"}[args.module],
        "examples": used,
        "T": args.T,
        "gates": list(args.gates),
        "x_read_norm": (sum(x_norms) / len(x_norms)) if x_norms else None,
        "z_row_norm": (sum(z_norms) / len(z_norms)) if z_norms else None,
        "checkpoint": args.checkpoint,
        "when": datetime.now().isoformat(timespec="seconds"),
    })
    print_report(rec)

    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)
        with open(os.path.join(args.out_dir, "readinto_gain.json"), "w") as f:
            json.dump(rec, f, indent=2)
        print(f"\nwrote {args.out_dir}/readinto_gain.json")
    return 0


def print_report(rec: dict, out=sys.stdout) -> None:
    """ASCII only.  These logs are read through `grep` on a login node."""
    w = out.write
    w("\n=== P3.0 tier %s | in-loop read-site gain | %s ===\n"
      % (rec["tier"], rec["module"]))
    w("    examples %d, T=%s\n" % (rec["examples"], rec["T"]))
    xn, zn = rec.get("x_read_norm"), rec.get("z_row_norm")
    w("    ||x|| at the read site  %s   <- what the delta is added to\n"
      % ("%.4f" % xn if xn else "-"))
    w("    ||Z row||               %s   <- what is being read\n"
      % ("%.4f" % zn if zn else "-"))
    w("    intact %.4f nats, chance %.4f, margin %s\n"
      % (rec["intact"]["mean_nats"] or float("nan"),
         rec["intact"]["chance_nats"],
         "%.4f" % rec["intact"]["margin_below_chance"]
         if rec["intact"]["margin_below_chance"] is not None else "-"))
    w("\n    %-14s %12s %12s %10s\n"
      % ("cell", "nats", "d(intact)", "|d|/||x||"))
    for k in sorted(rec["cells"], key=lambda s: (float(s.split("@")[1]), s)):
        r = rec["delta_norm_over_x"].get(k)
        w("    %-14s %12.5f %12.5f %10s%s\n"
          % (k, rec["cells"][k], rec["delta_from_intact"].get(k, 0.0),
             "%.4f" % r if r is not None else "-",
             "   OOD" if k in rec["ood_cells"] else ""))
    w("\n    span (in-distribution cells only)  %.5f nats\n"
      % rec["span_in_dist_nats"])
    w("    knob proven connected              %s\n" % rec["knob_live"])
    w("    largest real-vs-noise gap at init  %.5f nats\n"
      % rec["content_effect_at_init"])
    w("\n    READING: %s\n" % rec["reading"])
    if rec["reading"] == "DEAD_no_gain_in_distribution":
        w("    The new site is flat too.  This CONTRADICTS p30's premise;\n"
          "    S5 row 3 says stop and audit the instrument before spending\n"
          "    anything on tier 1.5.\n")
    elif rec["reading"] == "LIVE_site_has_gain":
        w("    Proceed to tier 1.5, the oracle read probe.  Print this row\n"
          "    beside the s0 table: that pair is the redesign's evidence.\n")
    else:
        w("    No verdict.  Fix the instrument, not the architecture.\n")
    w("\n    A reading is about the SITE only if it holds on BOTH arms.\n"
      "    Run a2 and a3z before quoting this.\n")


if __name__ == "__main__":
    raise SystemExit(main())
