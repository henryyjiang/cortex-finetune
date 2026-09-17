"""
The gate between "cortex-final is wired" and "start a full training run".

The unit tests and the real-checkpoint smoke establish that the architecture is
wired.  This establishes the four things neither of them can, each of which has
cost this project a round at least once and none of which is visible in a loss
curve:

  1. Z-OFF EQUIVALENCE -- the highest-value check here and the cheapest.
     `latent_carry=True` with the Z READ suppressed must reproduce the E-only
     model's loss to floating-point noise.  E's path through the forward is
     supposed to be untouched by the Z channel: only the E half is spliced into
     the token stream, the two channels have separate gates, and Z enters the
     loop at `s0` and nowhere else.  If the losses differ, Z is perturbing
     something it should not be touching, and NOTHING ELSE IN THE SUITE CAN
     LOCALISE THAT -- every other instrument would report it as "the Z arm is
     different", which is what the Z arm is supposed to be.

  2. CHECKPOINT ROUND-TRIP through config.json.  Bug class 2c was exactly this
     and no unit test would have caught it, because the defect lives in what the
     REBUILD reads, not in what the code computes.  `latent_carry` changes the
     carried tensor's width AND the buffer's parameter set, so a resume or an
     eval that rebuilt the graft without it constructs an E-only buffer, drops
     the Z gate's weights as unexpected keys, and runs a half-width carry.  The
     round trip goes through the JSON and loads back with strict=True, so a
     dropped key raises instead of being silently tolerated.

  3. THE REAL no_grad SPLIT DISTRIBUTION.  "Roughly half the batches carry a Z
     read gradient at mr8/depth 8" is ARITHMETIC off the Poisson tail, not a
     measurement.  It belongs in the Z pre-registration because it changes what
     a null result for Z means, and a pre-registration is not the place for a
     number nobody measured.

  4. THE DONOR / SHUFFLED-CARRY CONTROL (P0.6), per channel.  It never ran on
     B2.  On a corpus where across-document context helps about as much as
     within-document, a carry delta cannot distinguish information transfer from
     a register effect, and this is the only instrument that can.  Here it is
     run at MATCHED COLUMN COUNT and PER CHANNEL -- donor E, donor Z, donor
     both -- because "the carry matters" and "the carry's CONTENT matters" are
     different claims, and for a dual-channel carry so are "E's content matters"
     and "Z's content matters".

WHAT THIS IS NOT.  It cannot say whether Z helps, or whether the gate beats
accum.  Those need the full cells, the 2x2 and the influence horizon.  Every
check here can come back negative, which is the point: a probe that can only
report "it ran" is not a probe.

USAGE
    python tools/prelaunch_final.py --model_name ckpts/olmo-retrofit-cortex \
        --checkpoint cortex-retrofit/probe-p1-a3z-.../checkpoint_91952 \
        --chunks 8 --chunk_len 256 --out eval_results/prelaunch.json

Exit code 0 = every gate passed, 1 = at least one failed.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime

import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "evals"))
sys.path.insert(0, os.path.join(REPO, "tools"))

from cortex_memory.health import (  # noqa: E402
    carry_health, latent_runtime, read_live_fraction, split_carry,
)

#: Z-off equivalence bar, in nats of per-chunk loss.  The two forwards are the
#: same arithmetic in the same order, so the honest expectation is bit-identical
#: and this is a float32 reduction-order band, not a tolerance for "close
#: enough".  A difference above it is a real second path from Z into E.
Z_OFF_MAX_DELTA = 1e-5

#: The Z gate's parameter names -- the exact set an E-only rebuild must report
#: as unexpected, and the exact set a dual-channel rebuild must NOT.
Z_GATE_KEYS = ("gate_proj_in_z.weight", "gate_proj_in_z.bias",
               "gate_proj_mem_z.weight", "gate_proj_mem_z.bias",
               "forget_bias_z", "input_bias_z")


# ---------------------------------------------------------------------------
# 1. Z-off equivalence
# ---------------------------------------------------------------------------

class _SuppressZRead:
    """Run the Z machinery but throw the READ away.

    Not the same thing as `latent_read_null`, and the difference matters.  The
    null substitutes NOISE, which is Z's ablation condition -- in-distribution,
    matched columns, and deliberately not a no-op.  This returns `s0` UNCHANGED,
    which is what an E-only model's `s0` is, so the comparison is exact rather
    than distributional.

    The real `latent_init` still runs, so `_z_prev` is captured, the tape fills
    and every counter stays live: what is removed is precisely the substitution.
    """

    def __init__(self, cortex):
        self.cortex = cortex
        self._real = cortex.latent_init
        self._had = "latent_init" in vars(cortex)

    def __enter__(self):
        def suppressed(s0, num_steps_no_grad=None):
            self._real(s0, num_steps_no_grad)
            return s0
        self.cortex.latent_init = suppressed
        return self

    def __exit__(self, *exc):
        if self._had:
            self.cortex.latent_init = self._real
        else:
            del self.cortex.latent_init
        return False


def _chain_losses(model, chunks, num_steps, seed=0):
    """Per-chunk losses down one chain, with the s0 draw pinned per chunk.

    Seeding before every forward is not belt-and-braces: `initialize_state`
    draws trunc_normal_ noise for the WHOLE packed sequence, so two arms that
    consumed different amounts of randomness earlier would get different s0 for
    the real tokens and the comparison would measure the RNG.
    """
    losses, carry = [], None
    with torch.no_grad():
        for i, ids in enumerate(chunks):
            torch.manual_seed(seed + i)
            kw = {} if num_steps is None else {"num_steps": num_steps}
            out = model(input_ids=ids, labels=ids, m_cross_in=carry,
                        return_m_cross=True, **kw)
            carry = out["m_cross"] if isinstance(out, dict) else out.m_cross
            loss = out["loss"] if isinstance(out, dict) else out.loss
            losses.append(float(loss))
    return losses, carry


def check_z_off_equivalence(model_z, cortex_z, model_e, chunks,
                            num_steps=None, seed=0) -> dict:
    """Gate 1.  Z-on-with-the-read-suppressed vs E-only, chunk by chunk."""
    with _SuppressZRead(cortex_z):
        lz, carry_z = _chain_losses(model_z, chunks, num_steps, seed)
    le, carry_e = _chain_losses(model_e, chunks, num_steps, seed)
    deltas = [abs(a - b) for a, b in zip(lz, le)]
    D = cortex_z.prefix.hidden_size
    ez, _ = split_carry(carry_z, D)
    ee, _ = split_carry(carry_e, D)
    e_max = float((ez.float() - ee.float()).abs().max()) if ee is not None else None
    return {
        "gate": "z_off_equivalence",
        "loss_z_suppressed": lz,
        "loss_e_only": le,
        "max_abs_delta": max(deltas) if deltas else 0.0,
        "threshold": Z_OFF_MAX_DELTA,
        # The loss is a scalar summary; the E half of the carry is the tensor the
        # next chunk actually reads, so a difference there is the sharper signal
        # and is reported even when the losses agree.
        "e_carry_max_abs_delta": e_max,
        "passed": bool(deltas and max(deltas) <= Z_OFF_MAX_DELTA
                       and (e_max is None or e_max <= Z_OFF_MAX_DELTA)),
    }


# ---------------------------------------------------------------------------
# 2. checkpoint round-trip
# ---------------------------------------------------------------------------

def check_roundtrip(model, chunks, num_steps=None, seed=0,
                    workdir: str = None) -> dict:
    """Gate 2.  config -> JSON -> config -> fresh model -> strict load -> forward.

    Through the JSON ON PURPOSE.  The graft reads `gate_slots`, `gate_route`,
    `latent_carry` and the rest off `config.json` on every load, so a key that
    train.py forgot to persist rebuilds a DIFFERENT buffer at resume and at eval
    -- K back to W, route back to default, an E-only buffer where a
    dual-channel one was trained -- behind a healthy loss curve.  Recomputing
    the config in memory would not touch that path at all.

    strict=True is the assertion.  An E-only rebuild reports the six Z gate
    tensors as UNEXPECTED keys, and a strict load raises on them; a
    non-strict load would drop them and run.
    """
    cls, cfg_cls = type(model), type(model.config)
    js = model.config.to_json_string()
    raw = json.loads(js)
    if workdir:
        os.makedirs(workdir, exist_ok=True)
        with open(os.path.join(workdir, "config.json"), "w",
                  encoding="ascii") as fh:
            fh.write(js)
    # THE SURGERY IS PART OF THE PATH, so the round trip performs it.
    # `save_pretrained` serialises `tie_word_embeddings` whenever it differs
    # from the transformers default, and `RavenConfig.__init__` ALSO passes it
    # explicitly to super() while forwarding **kwargs -- so a raw saved dir
    # raises a duplicate-kwarg TypeError on reload.  That is why
    # tools/prepare_eval_checkpoint.py exists and pops it, and a gate that
    # skipped the pop would fail for a reason every checkpoint already has,
    # drowning the reason it is here for.  `raw_from_dict_error` records
    # whether the unprepared config was loadable, so the dependency on that
    # tool is reported rather than assumed.
    prepared = dict(raw)
    prepared.pop("tie_word_embeddings", None)
    try:
        cfg_cls.from_dict(dict(raw))
        raw_err = None
    except Exception as e:                      # noqa: BLE001
        raw_err = f"{type(e).__name__}: {e}"
    cfg2 = cfg_cls.from_dict(prepared)
    rebuilt = cls(cfg2)
    sd = {k: v.detach().clone() for k, v in model.state_dict().items()}
    err = None
    try:
        rebuilt.load_state_dict(sd, strict=True)
    except Exception as e:                      # noqa: BLE001 -- reported, not raised
        err = f"{type(e).__name__}: {e}"
    rebuilt = _like(model, rebuilt)
    rebuilt.train(model.training)
    out = {"gate": "checkpoint_roundtrip", "strict_load_error": err,
           "raw_from_dict_error": raw_err,
           "needs_prepare_eval_checkpoint": raw_err is not None}
    # Every flag that changes what the graft BUILDS must survive to the JSON.
    # The list is prepare_eval_checkpoint's, which tests/test_eval_checkpoint_flags.py
    # already pins against train.py's persist list -- so this gate checks the
    # third leg: that the value actually reached a serialised config.
    from prepare_eval_checkpoint import CORTEX_FLAGS  # noqa: E402
    present = [k for k in CORTEX_FLAGS if hasattr(model.config, k)]
    out["flags_missing_from_json"] = [k for k in present if k not in raw]
    out["flags_in_json"] = {k: raw[k] for k in present if k in raw}

    src_buf = model.cortex.prefix
    new_buf = getattr(getattr(rebuilt, "cortex", None), "prefix", None)
    out["buffer_rebuilt"] = new_buf is not None and type(new_buf) is type(src_buf)
    if new_buf is not None:
        for key in ("n_vec", "n_slots", "route", "gate_norm", "gate_init",
                    "fill", "carries_latent"):
            a, b = getattr(src_buf, key, None), getattr(new_buf, key, None)
            out[f"geom_{key}"] = [a, b]
        out["geometry_survived"] = all(
            getattr(src_buf, k, None) == getattr(new_buf, k, None)
            for k in ("n_vec", "n_slots", "route", "gate_norm", "gate_init",
                      "fill", "carries_latent"))
        out["z_gate_present"] = all(
            any(n.endswith(k) for n, _ in new_buf.named_parameters())
            for k in Z_GATE_KEYS) if src_buf.carries_latent else None
    else:
        out["geometry_survived"] = False

    la, ca = _chain_losses(model, chunks, num_steps, seed)
    lb, cb = _chain_losses(rebuilt, chunks, num_steps, seed)
    out["loss_before"] = la
    out["loss_after"] = lb
    out["max_abs_delta"] = max(abs(x - y) for x, y in zip(la, lb))
    D = src_buf.hidden_size
    out["carry_width"] = [None if ca is None else int(ca.shape[-1]),
                          None if cb is None else int(cb.shape[-1])]
    out["carry_max_abs_delta"] = (
        None if (ca is None or cb is None or ca.shape != cb.shape)
        else float((ca.float() - cb.float()).abs().max()))
    out["both_channels_survived"] = (
        ca is not None and cb is not None and ca.shape == cb.shape
        and (not src_buf.carries_latent or ca.shape[-1] == 2 * D))
    out["passed"] = bool(
        err is None and out["geometry_survived"] and out["both_channels_survived"]
        and not out["flags_missing_from_json"]
        and out["max_abs_delta"] <= Z_OFF_MAX_DELTA
        and (out["carry_max_abs_delta"] or 0.0) <= Z_OFF_MAX_DELTA)
    return out


# ---------------------------------------------------------------------------
# 3. the real no-grad split distribution
# ---------------------------------------------------------------------------

def check_read_live(model, mean_recurrence=None, mean_backprop_depth=None,
                    n_samples: int = 4000) -> dict:
    """Gate 3.  Measured, and it has no pass/fail -- it has a NUMBER.

    There is no threshold because there is no right answer: a low fraction is a
    fact about the sampler, not a defect, and it cannot be configured away (p
    has a Poisson tail, so no finite mean_backprop_depth makes n == 0 always).
    What would be a defect is launching without knowing it, so this gate passes
    as long as the number was actually measured, and prints it loudly.
    """
    cfg = model.config
    mr = int(mean_recurrence if mean_recurrence is not None else cfg.mean_recurrence)
    bd = int(mean_backprop_depth if mean_backprop_depth is not None
             else cfg.mean_backprop_depth)
    at_run = read_live_fraction(model, n_samples, mr, bd)
    sweep = []
    for m in sorted({mr, 8, 16, 32}):
        sweep.append(read_live_fraction(model, max(n_samples // 4, 250), m, bd))
    return {"gate": "read_live_fraction", "at_run_config": at_run,
            "sweep_mean_recurrence": sweep, "passed": True,
            "note": "put read_live_frac in the Z pre-registration: it changes "
                    "what a null result for Z means."}


# ---------------------------------------------------------------------------
# 4. the donor / shuffled-carry control, per channel
# ---------------------------------------------------------------------------

def _swap(real, donor, D, channel: str):
    """Matched-column carry substitution on one channel or both."""
    if channel == "both":
        return donor
    e_r, z_r = split_carry(real, D)
    e_d, z_d = split_carry(donor, D)
    if z_r is None:
        return donor if channel == "e" else real
    if channel == "e":
        return torch.cat([e_d, z_r], dim=-1)
    return torch.cat([e_r, z_d], dim=-1)


def check_donor_control(model, cortex, chunks_a, chunks_b, num_steps=None,
                        seed=0) -> dict:
    """Gate 4 (P0.6).  Does the carry's CONTENT matter, channel by channel?

    Prime two independent chains, then score the SAME final chunk under its own
    carry and under the other chain's -- same shape, same column count, same
    positions, wrong content.  `none` is reported too, but it DROPS the columns,
    so `real - none` confounds information transfer with a register/attention-
    sink effect and `real - donor` does not.  That gap is the whole reason this
    control exists.
    """
    D = cortex.prefix.hidden_size
    _, carry_a = _chain_losses(model, chunks_a[:-1], num_steps, seed)
    _, carry_b = _chain_losses(model, chunks_b[:-1], num_steps, seed + 500)
    last = chunks_a[-1]
    if carry_a is None or carry_b is None or carry_a.shape != carry_b.shape:
        return {"gate": "donor_control", "passed": False,
                "why": "the two chains produced carries of different shapes, so "
                       "a matched-column swap is not defined"}

    def score(state):
        torch.manual_seed(seed + len(chunks_a))
        with torch.no_grad():
            kw = {} if num_steps is None else {"num_steps": num_steps}
            out = model(input_ids=last, labels=last, m_cross_in=state,
                        return_m_cross=False, **kw)
        return float(out["loss"] if isinstance(out, dict) else out.loss)

    dual = carry_a.shape[-1] == 2 * D
    cells = {"real": score(carry_a), "none": score(None),
             "donor_both": score(_swap(carry_a, carry_b, D, "both"))}
    if dual:
        cells["donor_e"] = score(_swap(carry_a, carry_b, D, "e"))
        cells["donor_z"] = score(_swap(carry_a, carry_b, D, "z"))
    out = {"gate": "donor_control", "dual_channel": dual, "loss": cells,
           "content_delta_both": cells["donor_both"] - cells["real"],
           "column_delta": cells["none"] - cells["real"]}
    if dual:
        out["content_delta_e"] = cells["donor_e"] - cells["real"]
        out["content_delta_z"] = cells["donor_z"] - cells["real"]
    # PASSES on being measurable, not on being positive.  A near-zero content
    # delta on an untrained or briefly-trained arm is a RESULT (the carry is a
    # register, not information), and a gate that failed on it would train
    # people to skip the gate.
    out["passed"] = True
    out["note"] = ("content_delta > 0 means the carry's CONTENT helps; "
                   "content_delta ~ 0 with column_delta > 0 means the columns "
                   "help and the content does not -- the register effect this "
                   "control exists to expose.")
    return out


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------

def print_report(gates, out=sys.stdout) -> bool:
    p = lambda *a: print(*a, file=out)
    ok = True
    p("")
    p("=" * 78)
    p("CORTEX-FINAL PRE-LAUNCH GATES")
    p("=" * 78)
    for g in gates:
        name = g.get("gate", "?")
        passed = g.get("passed")
        ok = ok and bool(passed)
        p(f"[{'PASS' if passed else 'FAIL'}] {name}")
        if name == "z_off_equivalence":
            p(f"       max |loss(Z suppressed) - loss(E only)| = "
              f"{g['max_abs_delta']:.3e}  (bar {g['threshold']:.0e})")
            if g.get("e_carry_max_abs_delta") is not None:
                p(f"       max |E carry difference|             = "
                  f"{g['e_carry_max_abs_delta']:.3e}")
            if not passed:
                p("       Z is perturbing E's path.  Only the E half is spliced")
                p("       into the token stream and the gates are separate, so a")
                p("       difference here is a second path, not a tolerance.")
        elif name == "checkpoint_roundtrip":
            p(f"       strict load: {g['strict_load_error'] or 'clean'}")
            if g.get("needs_prepare_eval_checkpoint"):
                p("       raw saved config is NOT loadable as-is (expected): "
                  "run tools/prepare_eval_checkpoint.py on every dir the evals "
                  "read.")
            if g.get("flags_missing_from_json"):
                p(f"       FLAGS LOST IN SERIALISATION: "
                  f"{g['flags_missing_from_json']}")
            p(f"       geometry survived: {g['geometry_survived']}  "
              f"both channels: {g['both_channels_survived']}  "
              f"carry width {g['carry_width']}")
            p(f"       max |loss before - after| = {g['max_abs_delta']:.3e}")
        elif name == "read_live_fraction":
            a = g["at_run_config"]
            p(f"       at mr={a['mean_recurrence']} depth={a['mean_backprop_depth']}: "
              f"read_live_frac = {a['read_live_frac']:.3f} "
              f"({a['samples']} samples)")
            for s in g["sweep_mean_recurrence"]:
                p(f"         mr={s['mean_recurrence']:>3}  live "
                  f"{s['read_live_frac']:.3f}  no_grad mean "
                  f"{s['no_grad_mean']:.2f}  total steps "
                  f"{s['total_steps_mean']:.2f}")
            p("       MEASURED, not derived.  Write it into the Z pre-registration.")
        elif name == "donor_control":
            if "loss" in g:
                c = g["loss"]
                p("       " + "  ".join(f"{k}={v:.4f}" for k, v in c.items()))
                p(f"       content delta (donor-real, both) = "
                  f"{g['content_delta_both']:+.4f}")
                if g.get("dual_channel"):
                    p(f"       content delta E = {g['content_delta_e']:+.4f}   "
                      f"content delta Z = {g['content_delta_z']:+.4f}")
                p(f"       column  delta (none -real)       = "
                  f"{g['column_delta']:+.4f}   <- confounded, for contrast only")
            else:
                p(f"       {g.get('why')}")
    p("")
    p("These gates say the MACHINERY is live and the checkpoint survives a")
    p("round trip.  None of them says whether Z helps or whether the gate beats")
    p("accum -- that needs the full cells, the 2x2 and the influence horizon.")
    p("")
    return ok


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("cortex-final pre-launch gates")
    p.add_argument("--model_name", required=True)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--chunks", type=int, default=8)
    p.add_argument("--chunk_len", type=int, default=256)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--T", type=int, default=8)
    p.add_argument("--samples", type=int, default=4000,
                   help="draws for the no-grad split distribution")
    p.add_argument("--dtype", default="float32",
                   choices=["float32", "bfloat16", "float16"],
                   help="float32 by default and it is not a style choice: every "
                        "bar here is a small difference of large states, the "
                        "class of quantity a bf16 pass got wrong by 6x in P0.1")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--skip", action="append", default=[],
                   choices=["z_off", "roundtrip", "read_live", "donor"],
                   help="skip a gate.  Recorded in the JSON, so a skipped gate "
                        "can never read as a passed one.")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                   help="force a graft-building config flag, e.g. "
                        "--set latent_carry=true --set gate_slots=64.  THE "
                        "POINT: the gates can then run on the PARENT the arms "
                        "branch from, before any Z checkpoint exists -- which "
                        "is when a pre-launch gate is worth running.  Every "
                        "override is printed and recorded in the JSON.")
    p.add_argument("--out", default=None)
    return p.parse_args()


# The KEY=VALUE parser lives in evals/model_utils.py beside the
# `config_overrides` argument it feeds, so this tool and
# evals/diag_dual_channel_walk.py cannot parse the same flag differently.
from model_utils import parse_config_overrides as _parse_set  # noqa: E402


def _like(model, other):
    """Put a freshly constructed model on the SOURCE model's device and dtype.

    `type(model)(cfg)` builds on CPU, and `load_state_dict` COPIES INTO the
    existing parameters, so the rebuild stays on CPU no matter where the weights
    came from.  Feeding it the CUDA batches everything else uses then dies with
    "found at least two devices" -- after the gate has already loaded two 1B
    models.  Cheap to get right, and invisible until it runs on a GPU.
    """
    p = next(model.parameters())
    return other.to(device=p.device, dtype=p.dtype)


def build_e_twin(model):
    """An E-only model from the same weights, for gate 1.

    Rebuilt from the same config with `latent_carry` off, then loaded
    non-strictly: the six Z gate tensors MUST come back as unexpected keys and
    nothing else may.  That list is itself evidence -- an E-only rebuild that
    reported no unexpected keys would mean the Z gate was never allocated, and
    the twin would be comparable for the wrong reason.
    """
    cfg_cls = type(model.config)
    d = json.loads(model.config.to_json_string())
    # See check_roundtrip: RavenConfig cannot read back its own
    # `tie_word_embeddings`, which is why tools/prepare_eval_checkpoint.py pops
    # it.  The twin is not the thing under test here, so it takes the same
    # prepared config every eval takes.
    d.pop("tie_word_embeddings", None)
    d["latent_carry"] = False
    twin = type(model)(cfg_cls.from_dict(d))
    res = twin.load_state_dict(model.state_dict(), strict=False)
    unexpected = [k for k in res.unexpected_keys]
    stray = [k for k in unexpected
             if not any(k.endswith(z) for z in Z_GATE_KEYS)]
    if stray:
        raise RuntimeError(
            "an E-only rebuild dropped keys that are not the Z gate: "
            f"{stray[:8]}.  The twin is not comparable to the Z model.")
    twin = _like(model, twin)
    twin.train(model.training)
    return twin, unexpected


def main() -> int:
    args = parse_args()
    from model_utils import (explain_missing_cortex, load_checkpoint,  # noqa: E402
                             _unwrap)
    device = torch.device(args.device)
    overrides = _parse_set(args.set)
    model, cfg = load_checkpoint(args.checkpoint, args.model_name, None,
                                 getattr(torch, args.dtype), device,
                                 config_overrides=overrides or None)
    inner = _unwrap(model)
    cortex = getattr(inner, "cortex", None)
    if cortex is None or cortex.prefix is None:
        print("ERROR: no prefix buffer on this model -- nothing to run.")
        print("  " + explain_missing_cortex(cfg, overrides))
        return 1
    inner.train()
    torch.manual_seed(args.seed)
    V = inner.config.vocab_size
    n_tok = args.chunks * args.chunk_len
    mk = lambda: [c.contiguous().to(device) for c in torch.chunk(
        torch.randint(0, V - 1, (args.batch, n_tok)), args.chunks, dim=1)]
    chunks_a, chunks_b = mk(), mk()
    num_steps = torch.tensor([0, int(args.T)])

    gates, notes = [], {}
    if "z_off" not in args.skip:
        if not cortex.latent_carry:
            notes["z_off"] = ("skipped automatically: this checkpoint is E-only, "
                              "so there is no Z read to suppress")
        else:
            twin, unexpected = build_e_twin(inner)
            g = check_z_off_equivalence(inner, cortex, twin, chunks_a,
                                        num_steps, args.seed)
            g["e_twin_unexpected_keys"] = unexpected
            gates.append(g)
            del twin
    if "roundtrip" not in args.skip:
        gates.append(check_roundtrip(inner, chunks_a[:2], num_steps, args.seed))
    if "read_live" not in args.skip:
        gates.append(check_read_live(inner, n_samples=args.samples))
    if "donor" not in args.skip:
        gates.append(check_donor_control(inner, cortex, chunks_a, chunks_b,
                                         num_steps, args.seed))

    ok = print_report(gates)
    for k, v in notes.items():
        print(f"[skip] {k}: {v}")
    if args.skip:
        print(f"[skip] gates skipped by request: {sorted(set(args.skip))}")
    rec = {"when": datetime.now().isoformat(timespec="seconds"),
           "args": vars(args), "config_overrides": overrides,
           "skipped": sorted(set(args.skip)),
           "auto_skipped": notes, "gates": gates,
           "carry": carry_health(None, 0), "latent": latent_runtime(cortex),
           "all_passed": ok}
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="ascii") as fh:
            json.dump(rec, fh, indent=2, default=str)
        print(f"wrote {args.out}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
