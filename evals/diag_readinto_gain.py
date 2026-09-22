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

THE CONTROL, AND IT IS THE POINT.  Every gate value is scored TWICE -- once
on the real carried Z and once on a null whose ROW NORM IS MATCHED TO THE REAL
Z, per row.  At init the module is untrained, so the two are EXPECTED to agree
to within noise; a separation means the CONTROL is broken, not that the read
works, and it is now a refusal rather than a footnote.  Job 13420851 used a
fixed per-element std of 0.02 -- a row norm of 0.905 against ||Z row|| ~ 4.3 --
and reported the resulting 4.8x magnitude gap as a 0.27-nat "content effect".

WHAT THIS CAN AND CANNOT DECIDE.  At init the read module is RANDOM, so
content-specificity is untestable here by construction: nothing random can
prefer real Z over shuffled Z.  The only question available is SENSITIVITY --
does anything injected at this site survive to the loss?  That is the question
s0 failed, and it is P3.0's necessary condition.  Separating damage from
reading is tier 1.5's job, and no amount of cleverness here substitutes for it.

HOW TO READ IT -- FIXED HERE, BEFORE THE NUMBERS EXIST.

  The statistic is k in `d_nats ~ k * r^p`, where r = ||delta||/||x||.  It is
  used INSTEAD of the span because the span depends entirely on where the
  in-distribution line is drawn: job 13420851 read LIVE at one cut and DEAD at
  every other, off the same table.  k is scale-free under p = 2, which is what
  the response measures at 1.919 / 1.952.

  SENSITIVE_unlike_s0         k >= 100x s0's 8.6e-06.  Something injected here
      survives to the loss.  Necessary condition met; proceed to tier 1.5.
  INERT_like_s0               k at s0's own level.  This CONTRADICTS p30's
      premise -- S5 row 3: stop and audit before spending the 8 GPU-hours.
  AUDIT_control_not_matched   real and noise separate AT INIT, which a random
      projection cannot do.  The control is broken; fix it and re-run.
  INVALID_no_axis             no ||delta||/||x|| was recorded, so the cells
      could only be scored on gate values.  Both tier-0.5 cells of job
      13420851 were in this state and were scored anyway.
  INVALID_knob_unproven       nothing moved anywhere, including the oversized
      cell -- indistinguishable from a knob that was never wired.
  INVALID_at_chance           RED 10.  The level vetoes the verdict.

  p is reported beside k as a diagnostic.  p ~ 2 says the response is driven
  by the SIZE of the injection and nothing else.  At init that is EXPECTED and
  is not evidence against the design; it is a reminder that this sweep cannot
  speak to content.

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

#: A cell must move at least this much for the knob to count as connected.
KNOB_LIVE_NATS = 0.01

#: THE HEADLINE STATISTIC, AND IT IS SCALE-FREE ON PURPOSE.
#:
#: The first run of this sweep (job 13420851) decided LIVE/DEAD off the SPAN
#: across "in-distribution" cells, with in-distribution meaning ||delta|| <=
#: ||x||.  That verdict was worthless: the only cell driving it sat at 0.51x
#: the working state, and moving the threshold anywhere defensible flipped
#: every arm to DEAD (span 0.275 -> 0.0033).  A verdict that depends entirely
#: on where you draw an arbitrary line is not a verdict -- which is RED 11's
#: second bug ("the verdict counted cells that only break the model")
#: reappearing in the tool written to avoid it.
#:
#: The response is quadratic in the injected magnitude -- measured log-log
#: slope 1.919 (a2) and 1.952 (a3z) -- so `k = d_nats / r^2` is CONSTANT
#: across the sweep and does not depend on which cells are kept.  That is the
#: number to compare between sites.
#:
#: s0's reference value, from the same arms (job 13297293, read through RED
#: 11's corrected axis): at r = 1.8x the loop's working state (||s_t|| ~ 10,
#: P0.1 fp32) the loss moved 2.8e-5 nats on a2, so k_s0 = 2.8e-5 / 1.8^2.
S0_RESPONSE_COEFF = 8.6e-06

#: How far above s0 the coefficient must sit before the site counts as
#: responsive.  Two orders is deliberately coarse: the measured gap is FIVE
#: orders, so nothing here turns on the exact figure.
SENSITIVE_OVER_S0 = 100.0

#: A real-vs-noise gap this large AT INIT is an AUDIT trigger, not a finding.
#: An untrained random projection cannot prefer real content, so a separation
#: means the control is not controlling -- on job 13420851 it meant the noise
#: was 4.8x quieter than the Z it stood in for.  Pre-registered in the first
#: version of this file's docstring and NOT wired into the verdict, which is
#: why the run printed LIVE over the top of its own audit condition.
CONTENT_AT_INIT_NATS = 0.01

#: Thresholds the span is reported at, so its dependence on the cut is VISIBLE
#: rather than silent.  Never used for the verdict.
SPAN_THRESHOLDS = (1.0, 0.5, 0.25, 0.1)


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
    p.add_argument("--null_kind", default="noise_matched",
                   choices=["noise_matched", "noise"],
                   help="noise_matched scales the null to the REAL Z row norm, "
                        "per row, so the control differs in CONTENT ONLY.  "
                        "Plain noise uses a fixed per-element std and is kept "
                        "only to reproduce job 13420851, whose 0.27-nat "
                        "'content effect at init' was the 4.8x scale gap.")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    p.add_argument("--out_dir", default=None)
    return p.parse_args()


def response_fit(delta: dict, norms: dict) -> tuple:
    """(coefficient k, log-log exponent p) for d_nats ~ k * r^p.

    Only cells whose |d| clears the measurement floor and whose r is nonzero
    are used -- below the floor, d/r^2 is the ratio of two numbers that are
    both noise, which is what made the smallest cell of job 13420851 read
    k = 8.26 against its own arm's 1.87.

    p is the diagnostic and k is the headline.  p ~ 2 says the response is
    driven by the SIZE of the injection and nothing else, which is what any
    random perturbation does near a local minimum; it does NOT by itself mean
    the site is useless, because at init the module is random and a quadratic
    is the only thing it could produce.  It does mean this sweep cannot speak
    to content.  k is scale-free under p = 2 and is what compares to s0.
    """
    # REAL CELLS ONLY.  The noise cells are the CONTROL and belong beside the
    # curve, not inside it -- and once the null is row-norm matched they land
    # at nearly the same r, which silently made the exponent undefined
    # (log(r2/r1) with r2 == r1) the first time this was written.
    pts = [(norms[k], abs(d)) for k, d in delta.items()
           if k.startswith("real@") and norms.get(k) and abs(d) > 1e-4]
    if not pts:
        return None, None
    ks = sorted(d / (r ** 2) for r, d in pts)
    k = (ks[len(ks) // 2] if len(ks) % 2
         else 0.5 * (ks[len(ks) // 2 - 1] + ks[len(ks) // 2]))
    p = None
    if len(pts) >= 2:
        pts.sort()
        # The MOST SEPARATED pair, not the two largest: the widest lever arm
        # in log r gives the least noisy slope, and adjacent cells can share
        # an r.
        (r1, d1), (r2, d2) = pts[0], pts[-1]
        if r1 > 0 and r2 > r1 and d1 > 0:
            p = math.log(d2 / d1) / math.log(r2 / r1)
    return k, p


def summarize(cells: dict, norms: dict, chance: float) -> dict:
    """Means, deltas and the reading.  MODULE-LEVEL, not a closure inside
    main(), and that is RED 14's lesson rather than a style preference: the
    effects arithmetic there lived inside main(), so the code producing every
    number the instrument reported had no test and needed a GPU to reach.  A
    sign inversion survived in it for two days.

    `cells` maps cell name -> list of per-example nats.  `norms` maps cell name
    -> the MEASURED ||delta||/||x|| ratio for that cell.

    WHAT THIS CAN AND CANNOT DECIDE.  At init the read module is random, so
    CONTENT-SPECIFICITY IS NOT TESTABLE HERE BY CONSTRUCTION -- a random
    projection cannot prefer real Z over shuffled Z, and any sweep that appears
    to show it is reporting an artifact.  The only question available at init
    is SENSITIVITY: does anything injected at this site survive to the loss?
    That is the question s0 failed and it is the necessary condition for P3.0.
    Separating damage from reading is tier 1.5's job, not this one's.
    """
    mean = {k: sum(v) / len(v) for k, v in cells.items() if v}
    base = mean.get("real@0")
    delta = ({k: v - base for k, v in mean.items() if k != "real@0"}
             if base is not None else {})

    # Span at several cuts, REPORTED AND NEVER USED FOR THE VERDICT.  On job
    # 13420851 the same table read LIVE at one cut and DEAD at every other.
    spans = {}
    for thr in SPAN_THRESHOLDS:
        kept = [v for k, v in mean.items()
                if norms.get(k) is None or norms[k] <= thr]
        spans["%g" % thr] = (max(kept) - min(kept)) if kept else 0.0

    knob_live = any(abs(d) > KNOB_LIVE_NATS for d in delta.values())
    at_chance = base is not None and (chance - base) < 1.0
    have_axis = any(v is not None for v in norms.values())
    coeff, exponent = response_fit(delta, norms)

    pairs = {}
    for k in mean:
        if k.startswith("real@"):
            g = k.split("@", 1)[1]
            other = "noise@" + g
            if other in mean:
                pairs[g] = mean[k] - mean[other]
    content_at_init = max((abs(v) for v in pairs.values()), default=0.0)

    over_s0 = (coeff / S0_RESPONSE_COEFF) if coeff else None

    if at_chance:
        reading = "INVALID_at_chance"
    elif not have_axis:
        # Tiers 0.5's failure on job 13420851: LatentRefresh returned before
        # the norm recording, so every ratio printed as a dash and the cells
        # were scored on gate values alone -- which the launcher header
        # explicitly says not to read.  A sweep with no axis has no verdict.
        reading = "INVALID_no_axis"
    elif not knob_live:
        reading = "INVALID_knob_unproven"
    elif content_at_init > CONTENT_AT_INIT_NATS:
        # An untrained projection preferring real content is not a result, it
        # is a broken control.  This condition was pre-registered from the
        # first version of this file and left out of the verdict; it fired on
        # all four cells of job 13420851 while the tool printed LIVE.
        reading = "AUDIT_control_not_matched"
    elif over_s0 is not None and over_s0 >= SENSITIVE_OVER_S0:
        reading = "SENSITIVE_unlike_s0"
    else:
        reading = "INERT_like_s0"

    return {
        "cells": mean,
        "delta_from_intact": delta,
        "delta_norm_over_x": dict(norms),
        "span_by_threshold": spans,
        "knob_live": knob_live,
        "have_axis": have_axis,
        "response_coeff": coeff,
        "response_exponent": exponent,
        "s0_reference_coeff": S0_RESPONSE_COEFF,
        "coeff_over_s0": over_s0,
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
                # ROW-NORM MATCHED, not a fixed std.  A per-element std of
                # 0.02 is a row of norm 0.905 against ||Z row|| ~ 4.3, so the
                # first run of this sweep compared a big perturbation against
                # a small one and reported the difference as a content effect.
                cortex.latent_read_null = (None if kind == "real"
                                           else (args.null_kind, None, seed))
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
    nl = "\n"
    w(nl + "=== P3.0 tier %s | in-loop read-site gain | %s ===" % (
        rec["tier"], rec["module"]) + nl)
    w("    examples %d, T=%s%s" % (rec["examples"], rec["T"], nl))
    xn, zn = rec.get("x_read_norm"), rec.get("z_row_norm")
    w("    ||x|| at the read site  %s   <- what the delta is added to%s"
      % ("%.4f" % xn if xn else "-", nl))
    w("    ||Z row||               %s   <- what is being read%s"
      % ("%.4f" % zn if zn else "-", nl))
    w("    intact %.4f nats, chance %.4f, margin %s%s"
      % (rec["intact"]["mean_nats"] or float("nan"),
         rec["intact"]["chance_nats"],
         "%.4f" % rec["intact"]["margin_below_chance"]
         if rec["intact"]["margin_below_chance"] is not None else "-", nl))
    w(nl + "    %-14s %12s %12s %10s%s"
      % ("cell", "nats", "d(intact)", "|d|/||x||", nl))
    for k in sorted(rec["cells"], key=lambda s: (float(s.split("@")[1]), s)):
        r = rec["delta_norm_over_x"].get(k)
        w("    %-14s %12.5f %12.5f %10s%s"
          % (k, rec["cells"][k], rec["delta_from_intact"].get(k, 0.0),
             "%.4f" % r if r is not None else "-", nl))

    # THE RESPONSE SHAPE IS THE HEADLINE, NOT THE SPAN.  k is scale-free under
    # p = 2 and is what compares across sites; the span depends entirely on
    # where the in-distribution line is drawn, and job 13420851 read LIVE at
    # one cut and DEAD at every other.
    w(nl + "    RESPONSE SHAPE   d_nats ~ k * r^p" + nl)
    w("      k (scale-free)   %s%s"
      % ("%.4g" % rec["response_coeff"] if rec["response_coeff"] else "-", nl))
    w("      k for s0         %.4g   (job 13297293, RED 11 axis)%s"
      % (rec["s0_reference_coeff"], nl))
    w("      k / k_s0         %s%s"
      % ("%.4g" % rec["coeff_over_s0"] if rec["coeff_over_s0"] else "-", nl))
    p = rec["response_exponent"]
    w("      p (log-log)      %s%s%s"
      % ("%.3f" % p if p else "-",
         "   <- pure magnitude response; says nothing about content"
         if p and 1.7 <= p <= 2.3 else "", nl))

    w(nl + "    SPAN BY THRESHOLD (reported, NEVER the verdict)" + nl)
    for thr in SPAN_THRESHOLDS:
        w("      keep |d|/||x|| <= %-5s  span %.5f nats%s"
          % ("%g" % thr, rec["span_by_threshold"].get("%g" % thr, 0.0), nl))

    w(nl + "    knob proven connected              %s%s"
      % (rec["knob_live"], nl))
    w("    axis present                       %s%s" % (rec["have_axis"], nl))
    w("    real-vs-noise gap at init          %.5f nats  (want ~0)%s"
      % (rec["content_effect_at_init"], nl))

    w(nl + "    READING: %s%s" % (rec["reading"], nl))
    for line in VERDICT_NOTES.get(rec["reading"], VERDICT_NOTES["_default"]):
        w("    " + line + nl)
    w(nl + "    A reading is about the SITE only if it holds on BOTH arms." + nl)
    w("    Run a2 and a3z before quoting this." + nl)


#: What each reading means, kept beside the verdict rather than in prose so
#: the log says the same thing the pre-registration does.
VERDICT_NOTES = {
    "SENSITIVE_unlike_s0": [
        "Something injected here SURVIVES TO THE LOSS, which is what s0",
        "failed.  That is the necessary condition for P3.0 and it is ALL an",
        "at-init sweep can establish: the module is RANDOM here, so",
        "content-specificity is untestable by construction.  Separating",
        "damage from reading is tier 1.5's job.  Proceed.",
    ],
    "INERT_like_s0": [
        "The new site responds no more than s0 did.  This CONTRADICTS p30's",
        "premise; S5 row 3 says stop and audit the instrument before",
        "spending the 8 GPU-hours on tier 1.5.",
    ],
    "AUDIT_control_not_matched": [
        "A random projection cannot prefer real content, so this gap is the",
        "CONTROL failing, not a finding.  Check that the null is row-norm",
        "matched to the real Z before reading anything else.",
    ],
    "INVALID_no_axis": [
        "Every ||delta||/||x|| is missing, so the cells were scored on gate",
        "values -- the one thing the launcher says not to read.",
    ],
    "INVALID_at_chance": [
        "The intact loss is at ln(vocab).  Nothing in the table is a result",
        "about the read site (RED 10).",
    ],
    "INVALID_knob_unproven": [
        "Nothing moved anywhere, not even the oversized cell.  That is",
        "indistinguishable from a knob that was never wired.",
    ],
    "_default": ["No verdict.  Fix the instrument, not the architecture."],
}

if __name__ == "__main__":
    raise SystemExit(main())
