"""
Did the Z-channel architecture actually train?  The post-run half of
`PROBE=1 sbatch pace/p1_arms.sbatch`.

WHAT A SHORT RUN CAN AND CANNOT ANSWER
--------------------------------------
It CANNOT say whether Z helps.  That needs the full cell, the 2x2
(`evals/eval_carry_2x2.py`) and the influence horizon, and a few hundred steps
of a 5B-token recipe is nowhere near enough signal.  This tool refuses to
produce a number that could be read as a verdict on the mechanism, and says so
in its own output, because a green probe sitting in a log is exactly the kind of
thing that gets quoted later as "Z worked".

What it CAN answer is whether the machinery is live — every question of the form
"is this parameter block actually attached to the loss", which is the failure
class this project keeps paying for and which a healthy loss curve never
reveals:

  1. did the Z GATE leave its initialisation?  At gate_init="zero" both Z
     projections are exactly zero, so any movement at all is proof the gradient
     reached them.  Zero movement after N steps with a finite gradient is not
     possible; zero movement means no gradient path.

     THIS CHECK APPLIES TO THE GATED ARMS ONLY.  Z is PARAMETER-FREE on the
     accum buffer: PrefixAccumBuffer.merge concatenates E and Z on the last dim
     and owns no gate, so a2 (accum + Z) has no `_z` tensors to inspect and
     never will.  Requiring them there fails a healthy arm; requiring them
     wherever the buffer is merely `gated` fails a healthy E-only a3.  Both are
     "the instrument disagreed with the architecture", which is the bug class
     this file exists to catch and is therefore the one it must not commit.
     For a2, liveness is evals/diag_dual_channel_walk.py's question (2D carry
     width, |Z|/s0, z-read) plus the ValueError join_channels raises when the
     in-loop latent write never fired.
  2. did `summary_emb` keep training?  It is the shared write path, and a probe
     that branched at the wrong width would have re-seeded it from wte[eos] --
     which shows up here as the parent-to-probe distance being enormous rather
     than small.
  3. did the E gate keep training alongside it?  Z must not come at the cost of
     the channel that demonstrably works (+0.1673 carry on B2).
  4. is anything non-finite?
  5. is the carry the shape the config asked for?

USAGE
    python tools/check_latent_probe.py \
        --probe cortex-retrofit/p1-a2-accum-w16-cc8-z/checkpoint_115966 \
        --parent cortex-retrofit/retro-b2-heal/checkpoint_91552_w16

Exit code 0 = every live check passed, 1 = at least one failed.  A failure here
means the run trained SOMETHING while part of the architecture sat inert, and
the full launch must not go ahead on it.
"""
from __future__ import annotations

import argparse
import os

import torch

#: Parameters that must move if their gradient path exists.  Checked as a
#: relative change so the threshold does not depend on the tensor's scale.
REL_MOVE_MIN = 1e-6
#: `summary_emb` moving by MORE than this against the parent means it was not
#: continued but RE-SEEDED -- the silent branch-width failure.
RESEED_REL_MAX = 0.5


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("check a Z-channel architecture probe")
    p.add_argument("--probe", required=True,
                   help="the probe's checkpoint dir (holding chkpt.pt)")
    p.add_argument("--parent", default=None,
                   help="the dir it branched from.  Optional, and worth "
                        "passing: without it the re-seed check cannot run.")
    p.add_argument("--steps_min", type=int, default=1,
                   help="fail if the probe advanced fewer optimizer steps than "
                        "this past its parent")
    return p.parse_args()


def _load(path: str) -> dict:
    f = os.path.join(path, "chkpt.pt")
    if not os.path.isfile(f):
        raise SystemExit(f"FAILED: no chkpt.pt under {path}")
    return torch.load(f, map_location="cpu", weights_only=False)


def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    """||a - b|| / ||b||, in float32."""
    a, b = a.detach().float(), b.detach().float()
    den = float(b.norm())
    return float((a - b).norm()) / (den if den > 1e-12 else 1.0)


def main() -> int:
    args = parse_args()
    probe = _load(args.probe)
    model = probe["model"]
    # ONE load of the parent, not two.  This used to read the file again below
    # just to get `optimizer_step`, which held ~5.5 GB of a second copy of a
    # 1.4B-parameter fp32 state dict for the length of the call and pushed the
    # tool over the LOGIN NODE's memory cap -- where it was killed, and the
    # kill read as the probe failing.  Keep the whole dict; take both things
    # off it.  See pace/check_p1_probes.sbatch.
    parent_ckpt = _load(args.parent) if args.parent else None
    parent = parent_ckpt["model"] if parent_ckpt is not None else None

    ok = True

    def check(label, cond, detail=""):
        nonlocal ok
        ok &= bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {label}"
              + (f"  {detail}" if detail else ""))

    step = probe.get("agg_vars_dict", {}).get("optimizer_step", 0)
    prefix = {k: v for k, v in model.items() if k.startswith("cortex.prefix.")}
    latent_keys = [k for k in prefix if k.endswith("_z")
                   or k.endswith("_z.weight") or k.endswith("_z.bias")]
    gated = any("gate_proj_in." in k for k in prefix)
    z_params = bool(latent_keys)

    # WHAT THE ARM ASKED FOR.  Read here, before any check uses it, because
    # PARAMETER PRESENCE CANNOT STAND IN FOR IT: Z is parameter-free on the
    # accum buffer.  PrefixAccumBuffer.merge only join_channels-concatenates E
    # and Z on the last dim (cortex_memory/buffers.py), so an accum+Z arm owns
    # no `_z` tensors and never will, while PrefixGatedBuffer allocates a
    # SECOND GATE (gate_proj_*_z, forget_bias_z, input_bias_z) under the same
    # flag.  Inferring "is this a Z arm" from `_z` keys therefore fails A2 by
    # construction, and inferring it from `gated` fails A3 the same way.
    cfg = probe.get("cfg", {}) or {}
    cortex_cfg = cfg.get("cortex", {}) if isinstance(cfg, dict) else {}
    want_z = bool(cortex_cfg.get("latent_carry"))
    #: Z parameters exist ONLY where a gated arm asked for Z.  That is the only
    #: configuration in which "config says Z, parameters say no" is a defect.
    expect_z_params = gated and want_z

    print(f"\n=== Z probe | {args.probe} ===")
    print(f"    optimizer_step {step:,}  |  buffer: "
          f"{'gated' if gated else 'accum'}, "
          f"{'E+Z' if (want_z or z_params) else 'E only'}"
          + ("  (Z is parameter-free here)" if want_z and not gated else ""))
    if args.parent:
        pstep = parent_ckpt.get("agg_vars_dict", {}).get("optimizer_step", 0)
        print(f"    parent step {pstep:,} -> +{step - pstep:,} steps")
        check("the probe advanced past its parent", step - pstep >= args.steps_min,
              f"{step - pstep} steps")

    print("\n-- is every block attached to the loss? --")

    if not z_params:
        if expect_z_params:
            print("  [SKIP] no Z parameters in this checkpoint, and this arm "
                  "ASKED for them.")
            check("the Z gate exists", False,
                  "gated arm with --cortex.latent_carry true, but no _z "
                  "parameters -- latent_carry never reached the graft; check "
                  "the run's [cortex] banner line")
        elif want_z:
            print("  [SKIP] no Z parameters, and NONE ARE EXPECTED: this is an "
                  "accum+Z arm\n         (a2), whose Z channel is parameter-"
                  "free -- merge concatenates E and Z\n         on the last "
                  "dim and owns no gate.  Whether Z is LIVE here is a\n"
                  "         question for evals/diag_dual_channel_walk.py (2D "
                  "carry width, |Z|/s0,\n         z-read), and for the fact "
                  "that join_channels RAISES when the latent\n         write "
                  "never fired.  It is not one this tool can answer.")
        else:
            print("  [SKIP] no Z parameters, and the config did not ask for "
                  "any: an E-only\n         arm (a1/a3).  Every Z check below "
                  "is correctly skipped, and NONE of\n         this is a "
                  "failure.")

    # 1. THE Z GATE.  At gate_init="zero" the projections start at EXACTLY zero,
    #    so any nonzero weight is proof the gradient arrived.  This is the
    #    sharpest check available and it needs no parent.
    for name in ("cortex.prefix.gate_proj_in_z.weight",
                 "cortex.prefix.gate_proj_mem_z.weight"):
        if name not in model:
            continue
        w = model[name].detach().float()
        moved = float(w.abs().sum())
        check(f"{name.split('.')[-2]} left its zero init", moved > 0.0,
              f"|W|_1 = {moved:.4e}")
        if moved == 0.0:
            print("         ^ exactly zero after training means NO GRADIENT "
                  "PATH, not a small one.\n           The Z gate is inert and "
                  "the run trained an un-gated latent carry.")

    for name in ("cortex.prefix.forget_bias_z", "cortex.prefix.input_bias_z"):
        if name in model:
            print(f"    {name.split('.')[-1]} = "
                  f"{float(model[name].detach().float()):+.5f}  "
                  f"(init {'+1.0' if 'forget' in name else '0.0'})")

    # 2/3. THE SHARED WRITE PATH AND THE E GATE, against the parent.
    if parent is not None:
        for name in ("cortex.prefix.summary_emb",
                     "cortex.prefix.gate_proj_in.weight",
                     "cortex.prefix.gate_proj_mem.weight"):
            if name not in model or name not in parent:
                continue
            if model[name].shape != parent[name].shape:
                check(f"{name} kept its shape", False,
                      f"{tuple(model[name].shape)} vs parent "
                      f"{tuple(parent[name].shape)}")
                continue
            r = _rel(model[name], parent[name])
            check(f"{name.split('cortex.prefix.')[-1]} moved", r > REL_MOVE_MIN,
                  f"rel change {r:.3e}")
            if name.endswith("summary_emb") and r > RESEED_REL_MAX:
                check("summary_emb was CONTINUED, not re-seeded", False,
                      f"rel change {r:.3f} > {RESEED_REL_MAX} -- this is the "
                      f"width-mismatch failure: the branch dropped the trained "
                      f"write path and the graft re-seeded from wte[eos]")

    # 4. FINITENESS, across the whole model and not just the buffer.
    bad = [k for k, v in model.items()
           if torch.is_tensor(v) and v.is_floating_point()
           and not torch.isfinite(v).all()]
    check("every parameter is finite", not bad, f"{bad[:6]}" if bad else "")

    # 5. THE SHAPES THE CONFIG ASKED FOR.
    if cortex_cfg:
        if gated:
            check("latent_carry in the config matches the parameters found",
                  want_z == z_params,
                  f"config {want_z}, parameters {z_params}")
        elif z_params:
            # The reverse mismatch, and the only one an accum arm can show.
            check("an accum buffer owns no _z parameters", False,
                  f"config {want_z}, found {sorted(latent_keys)}")
        w_cfg = int(cortex_cfg.get("accum_vecs", 0) or 0)
        if w_cfg and "cortex.prefix.summary_emb" in model:
            got = int(model["cortex.prefix.summary_emb"].shape[0])
            check("summary_emb width matches accum_vecs", got == w_cfg,
                  f"{got} vs {w_cfg}")

    print("\n" + "=" * 72)
    if ok:
        print("PROBE PASSED — the architecture is LIVE: every block that should")
        print("be attached to the loss is attached to it.")
    else:
        print("PROBE FAILED — part of the architecture sat inert while the run")
        print("trained.  Do not launch the full cell on this configuration.")
    print("")
    print("THIS IS NOT A RESULT ABOUT Z.  A few hundred steps cannot say whether")
    print("the channel helps; that is what the full cell, evals/eval_carry_2x2.py")
    print("and evals/eval_influence_horizon.py are for.  Do not quote this run.")
    print("=" * 72)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
