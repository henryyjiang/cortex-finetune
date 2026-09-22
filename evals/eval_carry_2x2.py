"""
The 2x2 carry ablation: E on/off x Z on/off, at MATCHED COLUMN COUNT.

WHY A 2x2 AND NOT A NUMBER.  The single carry delta answers "is anything
carried".  The dual-channel design needs three different answers, and only a
factorial design separates them:

  E main effect   does the token-space carry (post-ln_f summary states, what B2
                  has) help?
  Z main effect   does the latent-space carry (the recurrence's own trajectory,
                  substituted into s0) help?
  INTERACTION     are they COMPLEMENTS or SUBSTITUTES?  This is the actual
                  hypothesis.  The existing memory-vs-depth trade treats them as
                  substitutes (carry-on T=8 beats carry-off T=32 by 3.6x); the
                  dual-channel claim is that they are complements, and a 2x2 is
                  the only design that can show it.

MATCHED COLUMN COUNT IS THE WHOLE DESIGN, NOT A DETAIL.  If "E off" DROPS the
carried columns instead of emptying them, the packed sequence gets shorter, the
model's own initialize_state draws a different s0 for the real tokens, and the
positions shift -- so "no information" is confounded with "shorter sequence" and
the measured effect is partly neither channel.  Every cell here splices the same
number of columns at the same positions; only the CONTENTS change.

  E null = ZEROS.  Imperfect, and say so: a zero row is still a key, scoring a
           mid-range logit rather than -inf, and it goes on absorbing ~3-5% of
           the softmax mass.  There is no better null for a channel the model
           reads as input embeddings.
  Z null = NOISE at s0's own scale.  STRICTLY BETTER than E's, and this is a
           real asymmetry worth stating in the writeup: the latent field's
           trained default IS fresh trunc_normal_ noise -- initialize_state
           writes exactly that into those columns today -- so the null is
           perfectly in-distribution, identical in column count AND in
           distribution.  Real-state-vs-noise has no sink ambiguity at all.

SCOPE, HONESTLY.  The Z axis needs a model that writes a latent channel.  Until
that exists this tool runs the E axis and REFUSES the Z rows rather than
silently reporting a 1x2 dressed up as a 2x2.  The matched-column machinery --
the part that is easy to get wrong -- is live either way, and it is what an
E-only accum-vs-gated comparison needs today.

READ-BLOCK LENGTH DIFFERS BY BUFFER, INHERENTLY.  An append buffer's read block
grows (32 -> 256 columns) while a gated one is fixed at K.  That is the
mechanism, not a confound to remove, so matched columns hold WITHIN an arm's
2x2 and NOT across an accum arm and a gated arm.  Comparing those two needs the
influence horizon (evals/eval_influence_horizon.py), not this.

USAGE
  python evals/eval_carry_2x2.py --model_name <ckpt> \
      --data data/pg19_olmo_val_len4096 --n_chunks 8 --T 8 \
      --out_dir eval_results/carry_2x2-$(date +%Y%m%d)
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

from cortex_memory.health import refuse_if_scrambled  # noqa: E402
from model_utils import load_checkpoint, to_num_steps, _unwrap  # noqa: E402
from model_utils import parse_config_overrides  # noqa: E402
from cortex_memory.health import chance_margin  # noqa: E402

CELLS = (("E1Z1", True, True), ("E1Z0", True, False),
         ("E0Z1", False, True), ("E0Z0", False, False))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", required=True)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--data", required=True)
    p.add_argument("--n_chunks", type=int, default=8,
                   help="must match the arm's cross_chunks")
    p.add_argument("--T", type=int, default=None,
                   help="recurrence depth; pass 8 on the mr8 arms, whose "
                        "config says 32")
    p.add_argument("--max_examples", type=int, default=100)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--dtype", default="float32",
                   choices=["float32", "bfloat16"])
    p.add_argument("--boot", type=int, default=2000)
    p.add_argument("--chunk1_tol", type=float, default=1e-4,
                   help="max allowed spread in the chunk-1 loss across cells. "
                        "Every cell runs chunk 1 with no incoming carry, so "
                        "they must agree there; a larger spread means the "
                        "cells differ in something other than the carry's "
                        "contents and no effect is interpretable")
    p.add_argument("--allow_missing_z", action="store_true",
                   help="run the E axis alone and report the Z rows as "
                        "unavailable, instead of failing.")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                   help="force a graft-building config flag, e.g. "
                        "--set use_memory=true --set prefix_memory=gated.  "
                        "REQUIRED on any arm checkpoint: --model_name loads the "
                        "BASE dir's config, which carries no cortex flags at "
                        "all (use_memory is '<absent>' on "
                        "ckpts/olmo-retrofit-cortex), so without these the "
                        "graft builds with NO prefix buffer and the run dies "
                        "with 'this checkpoint has no prefix buffer'.  Mirror "
                        "pace/p1_arms.sbatch's PROBE_SETS for the arm.")
    p.add_argument("--out_dir", default="eval_results/carry_2x2")
    p.add_argument("--allow_scrambled", action="store_true",
                   help="score a checkpoint whose read is wired to ANOTHER "
                        "document's Z (tier 1.5's shuffled limb).  Without "
                        "it such a checkpoint is REFUSED: every content "
                        "number on it is about the control arm and nothing "
                        "in the output would say so.")
    return p.parse_args()


def has_latent_channel(cortex) -> bool:
    """Does this model carry a latent (Z) channel as well as a token (E) one?

    The dual-channel design writes the recurrence's trajectory at the summary
    columns and substitutes it into s0 on the next chunk.  Duck-typed on the
    hook the graft will expose, so this tool does not have to be edited when the
    channel lands -- but it must never GUESS: reporting a 2x2 when only E exists
    would turn a missing channel into a null result.
    """
    buf = getattr(cortex, "prefix", None)
    return bool(getattr(cortex, "latent_carry", False)
                and buf is not None
                and getattr(buf, "carries_latent", False))


def null_e(state: torch.Tensor, hidden_size: int) -> torch.Tensor:
    """E's null: same columns, same positions, zero contents.

    ONLY THE E HALF.  On a dual-channel carry the tensor is [B, K, 2D] with Z at
    [..., D:], and a plain `zeros_like` would null BOTH channels -- silently
    turning the E0Z1 cell into E0Z0 wearing the wrong label, i.e. reporting a
    1x2 as a 2x2.  That is exactly what `has_latent_channel` refuses by default
    to prevent, one level up, so it has to be handled here as well.

    `hidden_size` is REQUIRED rather than inferred.  A [B, K, 2D] carry and a
    [B, K, D] carry from a model with twice the width are indistinguishable from
    the tensor alone, and the wrong guess produces a finite, plausible, wrong
    number.  Pass cortex.prefix.hidden_size.

    See the header for why zeros is the IMPERFECT null: a zero key still scores a
    mid-range logit rather than -inf and goes on absorbing ~3-5% of the softmax
    mass, so the E axis is confounded in a way Z's noise null is not.
    """
    D = state.shape[-1]
    if D == hidden_size:                                  # E-only carry
        return torch.zeros_like(state)
    if D != 2 * hidden_size:
        raise ValueError(
            f"carry is {D}-wide, expected {hidden_size} (E-only) or "
            f"{2 * hidden_size} (E+Z)")
    return torch.cat([torch.zeros_like(state[..., :hidden_size]),
                      state[..., hidden_size:]], dim=-1)


def null_z(state: torch.Tensor, std: float, seed: int) -> torch.Tensor:
    """Z's null: fresh noise at s0's own scale.

    This is the model's TRAINED DEFAULT for those columns -- initialize_state
    writes trunc_normal_(std) there on every chunk today -- so the null is
    in-distribution rather than merely length-matched.
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    n = torch.empty(state.shape, dtype=torch.float32).normal_(0.0, std,
                                                              generator=g)
    return n.to(device=state.device, dtype=state.dtype)


def chain_nll(model, cortex, xs, ys, ms, num_steps, device, seed,
              e_on: bool, z_on: bool, s0_std: float, hidden_size: int):
    """Mean NLL over chunks 2..N for one cell of the 2x2.

    Chunk 1 is excluded from the endpoint (no incoming carry either way, so
    every cell is identical there) but still RUN, because it is what produces
    the carry the later chunks read -- and it doubles as a sanity check: the
    four cells must agree on chunk 1 to floating-point noise.
    """
    torch.manual_seed(seed)
    state, tot, ntok, first = None, 0.0, 0, None
    for i, (xc, yc, mc) in enumerate(zip(xs, ys, ms)):
        m_in = state
        if m_in is not None:
            if not e_on:
                m_in = null_e(m_in, hidden_size)
            # Z lives in a separate field; when the channel exists the graft
            # takes it from the same slot columns.  Nulling it is a per-chunk
            # substitution, handled by the graft hook rather than here.
        cortex.latent_read_null = (None if z_on
                                   else ("noise", s0_std, seed + i))
        # no_grad IS LOAD-BEARING, not tidiness.  `state` carries the graph to
        # the next chunk, so without this the chain holds every chunk's graph at
        # once -- the footprint that OOMed the W=32 walk at 139.78 GiB on an
        # H200.  This tool never backprops.
        with torch.no_grad():
            out = model(input_ids=xc.unsqueeze(0).to(device),
                        num_steps=num_steps, m_cross_in=m_in,
                        return_m_cross=True)
        state = (out.get("m_cross") if isinstance(out, dict)
                 else getattr(out, "m_cross", None))
        n = int(mc.sum())
        if n == 0:
            continue
        logits = (out["logits"] if isinstance(out, dict) else out.logits)[0].float()
        ce = F.cross_entropy(logits, yc.to(device), reduction="none")
        loss = float((ce * mc.to(device)).sum() / n)
        if i == 0:
            first = loss
            continue
        tot += loss * n
        ntok += n
    cortex.latent_read_null = None
    return (tot / ntok if ntok else None), first


def paired_ci(deltas, n_boot: int, seed: int = 0):
    if not deltas:
        return (float("nan"),) * 3
    t = torch.tensor(deltas, dtype=torch.float64)
    g = torch.Generator().manual_seed(seed)
    idx = torch.randint(len(t), (n_boot, len(t)), generator=g)
    means = t[idx].mean(dim=1)
    lo, hi = torch.quantile(means, torch.tensor([0.025, 0.975],
                                                dtype=torch.float64))
    return float(t.mean()), float(lo), float(hi)


#: SIGN CONVENTION, ONE LINE, AND EVERY ROW OBEYS IT:
#:      effect = NLL(channel OFF) - NLL(channel ON),  so POSITIVE = it helps.
#: The one row that did not obey it (Z_alone, inverted until 2026-09-19) is why
#: this is a module-level function with tests rather than a closure in main().
EFFECTS = (
    ("E_main", "E0Z1", "E1Z1", True,
     "NLL without E minus with E, Z held on.  Positive = E helps."),
    # The E-only wording for the SAME row.  On a model with no latent channel
    # the cells are still named E1Z1/E0Z1 -- the labels are the 2x2's, not the
    # model's -- so the default note above would tell a reader of results.json
    # that "Z held on" when the record's own z_channel field says false.
    # Swapped in by compute_effects when z_live is False.
    ("Z_main", "E1Z0", "E1Z1", False,
     "NLL without Z minus with Z, E held on.  Positive = Z helps."),
    ("Z_alone", "E0Z0", "E0Z1", False,
     "NLL without Z minus with Z, E held OFF.  Positive = Z helps on its own.  "
     "The rung a non-recurrent baseline cannot compete on by construction."),
)


def compute_effects(per_cell: dict, z_live: bool, n_boot: int,
                    seed: int = 0) -> dict:
    """The 2x2's main effects and interaction, paired across samples.

    EXTRACTED FROM main() ON PURPOSE.  It lived there as a closure, which meant
    the arithmetic that produces every number this instrument reports had no
    test -- and a sign error sat in the Z_alone row undetected because of it.
    A function that takes a dict of lists needs no model, no GPU and no
    checkpoint, so there is no excuse for it to be untested.

    THE INTERACTION.  (E's effect without Z) minus (E's effect with Z).
    NEGATIVE means COMPLEMENTS: E is worth more when Z is present.  Positive
    means substitutes.  This is the design's actual hypothesis, and it is the
    one term whose sign cannot be read off a single column.
    """
    out = {}

    for label, off, on, always, note in EFFECTS:
        if not (always or z_live):
            continue
        if off not in per_cell or on not in per_cell:
            continue
        d = [x - y for x, y in zip(per_cell[off], per_cell[on])]
        if not d:
            continue
        m, lo, hi = paired_ci(d, n_boot, seed)
        if label == "E_main" and not z_live:
            note = ("NLL without E minus with E.  Positive = E helps.  THIS "
                    "MODEL HAS NO Z CHANNEL: the E1Z1/E0Z1 cell labels are the "
                    "2x2's, not the model's, and this row is a 1x2.")
        out[label] = {"mean_nats": m, "ci_lo": lo, "ci_hi": hi,
                      "n": len(d), "note": note}

    if z_live and all(k in per_cell for k in
                      ("E0Z0", "E1Z0", "E0Z1", "E1Z1")):
        d = [(a - b) - (c - e) for a, b, c, e in
             zip(per_cell["E0Z0"], per_cell["E1Z0"],
                 per_cell["E0Z1"], per_cell["E1Z1"])]
        if d:
            m, lo, hi = paired_ci(d, n_boot, seed)
            out["interaction"] = {
                "mean_nats": m, "ci_lo": lo, "ci_hi": hi, "n": len(d),
                "note": "E's effect without Z minus E's effect with Z.  "
                        "NEGATIVE = complements (each is worth MORE when the "
                        "other is present); positive = substitutes."}
    return out


def main() -> int:
    args = parse_args()
    if args.dtype == "bfloat16":
        print("WARNING: bfloat16.  Every number below is a difference of two "
              "nearly equal losses -- the quantity bf16 got wrong by 6x in "
              "P0.1.  Do not quote these.", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    overrides = parse_config_overrides(args.set)
    model, cfg = load_checkpoint(args.checkpoint, args.model_name, None,
                                 getattr(torch, args.dtype), device,
                                 config_overrides=overrides or None)
    inner = _unwrap(model)
    cortex = getattr(inner, "cortex", None)
    refuse_if_scrambled(cortex, "carry_2x2", args.allow_scrambled)
    if cortex is None or getattr(cortex, "prefix", None) is None:
        print("FAILED: this checkpoint has no prefix buffer.")
        return 2

    # E's null must know where the E half ends: on a dual-channel carry the
    # tensor is [B, K, 2D] and zeroing all of it would null Z as well, reporting
    # a 1x2 as a 2x2.  Taken from the BUFFER rather than the config, because the
    # buffer is what actually built the carry.
    hidden_size = int(cortex.prefix.hidden_size)
    z_live = has_latent_channel(cortex)
    if not z_live and not args.allow_missing_z:
        print(
            "FAILED: this model has no latent (Z) channel, so a 2x2 is not\n"
            "measurable -- only the E axis exists.  Re-run with\n"
            "  --allow_missing_z\n"
            "to get the E-on/E-off contrast at matched column count, which is\n"
            "what an accum-vs-gated comparison needs today.  Refusing by\n"
            "default so that a missing channel is never reported as a null\n"
            "result for Z.")
        return 3

    cells = CELLS if z_live else (("E1Z1", True, True), ("E0Z1", False, True))
    num_steps = to_num_steps(args.T)

    from datasets import load_from_disk
    ds = load_from_disk(args.data)

    # ---- Z's null has to be at s0's MEASURED scale, and this is RED 11 -----
    # `_null_latent` takes a PER-ELEMENT std.  This tool used to pass
    # `cfg.init_values["std"]` with a 0.02 fallback: a weight-init number, not
    # s0's rms, and off by whatever the two happen to differ by on the day.
    # The consequence is not noise, it is a false null -- diag_s0_sensitivity
    # measured that the substitution site has NO GAIN until the injected norm
    # reaches ~175.8 against ||E|| = 171.0, so a null at 0.02 lands deep in the
    # flat region, every Z cell equals its E twin, and the table says "Z carries
    # nothing" about the instrument rather than the model.
    #
    # `_z_s0_rms` is only populated by a forward, so prime one first.  Same
    # guard and same wording as diag_s0_sensitivity, on purpose: two tools
    # disagreeing about which scale s0 has is how RED 11 happened.
    s0_std = None
    if z_live:
        prime_ids = torch.tensor(ds[0]["input_ids"][:512], dtype=torch.long)
        with torch.no_grad():
            model(input_ids=prime_ids.unsqueeze(0).to(device),
                  num_steps=num_steps, m_cross_in=None, return_m_cross=False)
        rms = getattr(cortex, "_z_s0_rms", None)
        row = getattr(cortex, "_z_s0_scale", None)
        if rms is None:
            print("FAILED: the graft on this checkpoint does not record "
                  "`_z_s0_rms`, so the only scale available is a ROW NORM and "
                  "feeding it to `_null_latent` is exactly RED 11.  Re-run "
                  "tools/prepare_cortex_checkpoint.py against a cortex_graft.py "
                  "at 2026-09-17 or later.")
            return 2
        s0_std = float(rms)
        print(f"[2x2] Z null at s0's MEASURED per-element rms {s0_std:.6g}"
              + (f" (row norm {float(row):.4g}, sqrt(D) apart -- do not swap "
                 f"them)" if row else ""))
    n = len(ds) if args.max_examples == 0 else min(args.max_examples, len(ds))

    per_cell: dict[str, list[float]] = {c[0]: [] for c in cells}
    chunk1_spread, n_used = [], 0
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

        firsts, ok = [], True
        for name, e_on, z_on in cells:
            nll, first = chain_nll(model, cortex, xs, ys, ms, num_steps, device,
                                   args.seed + si, e_on, z_on, s0_std,
                                   hidden_size)
            if nll is None:
                ok = False
                break
            per_cell[name].append(nll)
            firsts.append(first)
        if not ok:
            for name, _, _ in cells:
                per_cell[name] = per_cell[name][:n_used]
            continue
        chunk1_spread.append(max(firsts) - min(firsts))
        n_used += 1
        if n_used % 10 == 0:
            print(f"  {n_used} samples", flush=True)

    report = {
        "instrument": "2x2 carry ablation (matched column count)",
        "when": datetime.now().isoformat(timespec="seconds"),
        "model_name": args.model_name,
        "z_channel": z_live,
        "config": {"n_chunks": args.n_chunks, "T": args.T,
                   "dtype": args.dtype, "samples": n_used,
                   "s0_std": s0_std},
        "cells": {}, "effects": {},
        "chunk1_max_spread": (max(chunk1_spread) if chunk1_spread else None),
    }

    # --- the two VETOES.  Both exist because this project's failure mode is an
    # instrument that returns a confident wrong number rather than failing.
    #
    # 1. CHUNK-1 AGREEMENT.  Every cell runs chunk 1 with no incoming carry, so
    #    the four must agree there to floating-point noise.  A spread that is
    #    not ~0 means the cells differ in something OTHER than the carry's
    #    contents -- a shifted position, a different s0, a dropped column --
    #    and then no effect below it is interpretable.  pace/eval_carry_2x2.sbatch
    #    has always told the reader to check this by eye; checking it by eye is
    #    how REDs 8, 9 and 10 survived.
    # 2. CHANCE LEVEL.  RED 10 was two at-chance losses differenced into a
    #    confident number.  Same check the walk got afterwards, same threshold,
    #    same module -- keep them in step.
    for name, _, _ in cells:
        v = per_cell[name]
        report["cells"][name] = {
            "mean_nll": (sum(v) / len(v) if v else None), "n": len(v)}

    # AFTER the cells loop, not before: chance_margin reads report["cells"],
    # and an empty list makes it return all-None -- a health check that is
    # silently absent rather than failing, which is the exact shape of defect
    # it was added to catch.
    spread = report["chunk1_max_spread"]
    report["chunk1_ok"] = (None if spread is None
                           else bool(spread <= args.chunk1_tol))
    vocab = int(getattr(getattr(inner, "config", None), "vocab_size", 0) or 0)
    report["health"] = chance_margin(
        [c["mean_nll"] for c in report["cells"].values()], vocab)

    report["effects"] = compute_effects(per_cell, z_live, args.boot, args.seed)

    print(f"\n{'=' * 78}")
    print(f"2x2 carry ablation -- mean NLL over chunks 2..{args.n_chunks}, "
          f"n={n_used} paired samples")
    print(f"column count is IDENTICAL in every cell; only the contents change")
    print("=" * 78)
    for name, e_on, z_on in cells:
        c = report["cells"][name]
        print(f"  {name}  E={'on ' if e_on else 'off'}  "
              f"Z={'on ' if z_on else 'off'}   "
              f"NLL {c['mean_nll']:.5f}" if c["mean_nll"] is not None else
              f"  {name}  (no data)")
    if report["chunk1_max_spread"] is not None:
        print(f"\n  chunk-1 sanity: max spread across cells "
              f"{report['chunk1_max_spread']:.2e} "
              f"(should be ~0 -- no cell has an incoming carry there)")
        if report["chunk1_ok"] is False:
            print(f"\n  *** CHUNK-1 VETO: spread {spread:.3e} exceeds "
                  f"--chunk1_tol {args.chunk1_tol:.0e}.")
            print("  *** The cells differ in something OTHER than the carry's")
            print("  *** contents -- a shifted position, a different s0, a")
            print("  *** dropped column.  NO EFFECT BELOW IS INTERPRETABLE.")
    h = report["health"]
    if h.get("at_chance"):
        print(f"\n  *** AT CHANCE: mean NLL {h['mean_nll']:.4f} against "
              f"ln(vocab) {h['chance']:.4f} (margin {h['margin']:.4f}).")
        print("  *** Every effect below is a difference of two noise levels.")
        print("  *** This is RED 10's shape.  Fix the scoring, not the table.")
    elif h.get("margin") is not None:
        print(f"  health: mean NLL {h['mean_nll']:.4f}, "
              f"{h['margin']:.2f} nats below chance")
    print(f"\n{'-' * 78}")
    for label, e in report["effects"].items():
        sig = "" if (e["ci_lo"] <= 0 <= e["ci_hi"]) else "  *"
        print(f"  {label:<12}{e['mean_nats']:>10.5f}  "
              f"[{e['ci_lo']:>9.5f}, {e['ci_hi']:>9.5f}]{sig}")
        print(f"               {e['note']}")
    if not z_live:
        print(f"\n  Z axis UNAVAILABLE: this model carries no latent channel, "
              f"so\n  Z_main, Z_alone and the interaction are not reported.  "
              f"The E axis\n  above is a genuine matched-column contrast; the "
              f"2x2 is not.")

    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, "results.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
