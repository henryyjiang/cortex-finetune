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
  --damage random   replace it with Gaussian noise carrying the SAME per-row
                    norm.  This is a SENSITIVITY CONTROL ON THE INSTRUMENT, not
                    a third result: Z is read by substitution into `s0`, where
                    P0.1 put the staggered delta at 0.90x the trunc_normal_
                    noise it replaces, so "Z moved nothing" and "nothing could
                    move anything at this scale" are the same picture until
                    donor and random are shown to separate.  Read it first, and
                    read a Z ablation only through it.

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
import math
import os
import sys
from datetime import datetime

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model_utils import (  # noqa: E402
    load_checkpoint, parse_config_overrides, to_num_steps, _unwrap,
)


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
    p.add_argument("--damage", default="donor",
                   choices=["donor", "zero", "random"])
    p.add_argument("--damage_channel", default="both",
                   choices=["both", "e", "z"],
                   help="WHICH CHANNEL to damage on a dual-channel carry.  "
                        "'both' was the only behaviour before the Z channel "
                        "existed and it CONFLATES the two: a horizon measured "
                        "by wrecking E and Z together cannot say which one "
                        "carries the distance.  On an E-only carry all three "
                        "are the same thing and the choice is inert.")
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
    p.add_argument("--out_dir",
                   default="eval_results/influence_horizon")
    return p.parse_args()


# ---------------------------------------------------------------------------

def _is_append(buf) -> bool:
    """Append buffers keep per-chunk rows separable; gated ones do not."""
    return buf is not None and hasattr(buf, "max_vecs")


def scaled_noise(ref, gen: torch.Generator):
    """Gaussian noise carrying the SAME per-row norm as `ref`.

    The scale is the whole point.  Z is read by SUBSTITUTION into `s0`, where
    P0.1 measured the staggered delta at 0.90x the `trunc_normal_` noise it
    replaces -- so "Z moved nothing" and "nothing could move anything at this
    scale" are the same picture, and only a control that swaps in noise AT Z'S
    OWN SCALE can tell them apart.  Per ROW, not per tensor: the rows are
    separately addressable keys and a global rescale would let one row's norm
    set every other row's.

    Drawn from an EXPLICIT generator on the CPU, never the global RNG.  Both
    chains call `torch.manual_seed(seed)` so their `initialize_state` draws
    line up; a `randn_like` here would consume from that stream in the damaged
    chain only, shift every later s0, and quietly break the pairing that the
    whole instrument's power rests on.
    """
    z = torch.randn(tuple(ref.shape), generator=gen, dtype=torch.float32)
    z = z.to(ref.device, ref.dtype)
    rn = ref.norm(dim=-1, keepdim=True)
    zn = z.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return z * (rn / zn)


def damage_write(w, donor_write, mode: str, channel: str, hidden_size: int,
                 gen: torch.Generator | None = None):
    """Apply the damage to one channel of a write, leaving the other intact.

    Splitting here rather than at the call site because EVERY damage mode needs
    it and a `zeros_like` on a 2D-wide write nulls Z as well -- the same defect
    eval_carry_2x2's `null_e` was fixed for, which reported an E0Z1 cell as
    E0Z0 under the wrong label.
    """
    if mode == "zero":
        dst = torch.zeros_like(w)
    elif mode == "random":
        if gen is None:
            raise ValueError("--damage random needs an explicit generator; see "
                             "scaled_noise for why it must not use the global "
                             "RNG")
        dst = scaled_noise(w, gen)
    else:
        dst = donor_write.to(w.device, w.dtype)
    if channel == "both" or w.shape[-1] == hidden_size:
        return dst
    D = hidden_size
    if channel == "e":
        return torch.cat([dst[..., :D], w[..., D:]], dim=-1)
    return torch.cat([w[..., :D], dst[..., D:]], dim=-1)


def damage_pair(e, z, donor_e, donor_z, mode: str, channel: str,
                gen: torch.Generator | None = None):
    """Damage a PRE-merge write pair (E, Z), channel by channel.

    The gated buffer's rows are mixed by the gate, so there is no post-hoc row
    to substitute -- the damage has to go in before `merge`, where E and Z are
    still two separate tensors.  Returns (e', z').
    """
    def one(t, donor):
        if t is None:
            return None
        if mode == "zero":
            return torch.zeros_like(t)
        if mode == "random":
            if gen is None:
                raise ValueError("--damage random needs an explicit generator")
            return scaled_noise(t, gen)
        if donor is None:
            raise ValueError("donor damage needs the donor's own pre-merge "
                             "write for this channel")
        return donor.to(t.device, t.dtype)

    return (one(e, donor_e) if channel in ("both", "e") else e,
            one(z, donor_z) if channel in ("both", "z") else z)


class _patched_merge:
    """Swap `buf.merge` for the duration of one forward, then put it back.

    Restores the EXACT prior state -- an instance attribute if there was one,
    otherwise none at all, so the class's bound method takes over again.  The
    buffer carries a ring pointer (`_chunk`) that the real merge advances, so
    every wrapper here has to call through rather than reimplement.
    """

    _MISSING = object()

    def __init__(self, buf, fn):
        self.buf, self.fn = buf, fn

    def __enter__(self):
        self.prev = self.buf.__dict__.get("merge", self._MISSING)
        self.buf.merge = self.fn
        return self

    def __exit__(self, *exc):
        if self.prev is self._MISSING:
            self.buf.__dict__.pop("merge", None)
        else:
            self.buf.merge = self.prev
        return False


def capture_write(model, buf, ids, num_steps, device):
    """One chunk, no carry -> the (E, Z) pair the buffer WOULD have merged.

    The pre-merge write is the only thing a gated donor swap can be built from:
    what the model returns is the post-gate ring, which is a mixture of this
    chunk's write and everything still in the slots.
    """
    grabbed = {}

    real = type(buf).merge.__get__(buf, type(buf))

    def spy(state, new_vecs, new_latent=None):
        grabbed["e"], grabbed["z"] = new_vecs, new_latent
        return real(state, new_vecs, new_latent)

    with _patched_merge(buf, spy), torch.no_grad():
        model(input_ids=ids.unsqueeze(0).to(device), num_steps=num_steps,
              m_cross_in=None, return_m_cross=True)
    return grabbed.get("e"), grabbed.get("z")


def run_chain(model, buf, chunks, labels, masks, num_steps, device,
              seed: int, damage_at: int | None, donor_write, mode: str,
              channel: str = "both", hidden_size: int = 0,
              gen: torch.Generator | None = None, donor_latent=None):
    """Replay one row's chunk chain, optionally damaging one chunk's write.

    Returns (per-chunk (loss, n_tokens), per-chunk PRE-merge (E, Z) writes).
    `damage_at` is a CHUNK INDEX, not a depth; the caller converts.

    THE GATED PATH USED TO BE A SILENT NO-OP.  Until 2026-09-17 the `else`
    branch below appended the merged state and assigned it, and never looked at
    `damage_at` -- so on A3/A3' every damage mode did nothing, I(d) came back
    exactly 0.0 at every depth, and the table read "the gated carry does not
    matter anywhere", which is the most expensive false negative this instrument
    could produce.  The header already described the intended treatment
    ("applied to the write BEFORE the merge"); only the code disagreed.  It is
    now applied where the header says, through the single `prefix.merge` call
    site in cortex_graft.py.
    """
    torch.manual_seed(seed)                 # identical s0 draws across arms
    state, rows, writes = None, [], []
    real_merge = type(buf).merge.__get__(buf, type(buf))
    for i, (xc, yc, mc) in enumerate(zip(chunks, labels, masks)):
        seen = {}

        def merge_hook(st, new_vecs, new_latent=None, _i=i):
            seen["e"], seen["z"] = new_vecs, new_latent
            if _i == damage_at and not _is_append(buf):
                new_vecs, new_latent = damage_pair(
                    new_vecs, new_latent, donor_write, donor_latent,
                    mode, channel, gen)
            return real_merge(st, new_vecs, new_latent)

        with _patched_merge(buf, merge_hook):
            out = model(input_ids=xc.unsqueeze(0).to(device),
                        num_steps=num_steps, m_cross_in=state,
                        return_m_cross=True)
        new_state = out.get("m_cross") if isinstance(out, dict) \
            else getattr(out, "m_cross", None)
        writes.append((seen.get("e"), seen.get("z")))

        if _is_append(buf):
            # Rows are separable: take this chunk's own write off the end and
            # rebuild the state, substituting the damaged write if this is the
            # damaged chunk.  The forward above already ran on the TRUE carry,
            # so damage never reaches a later chunk's write.  Left as row
            # surgery rather than folded into the hook: it is the tested path,
            # and the two are equivalent for a write-once FIFO.
            w = new_state[:, -buf.n_vec:]
            if i == damage_at:
                w = damage_write(w, donor_write, mode, channel, hidden_size,
                                 gen)
            state = w if state is None else torch.cat([state, w], dim=1)
            if state.shape[1] > buf.max_vecs:
                state = state[:, -buf.max_vecs:]
        else:
            # Gated: the rows are mixed by the gate, so the damage went in
            # above, before the merge, and the merge carries the consequence
            # forward.  Documented in the header.
            state = new_state

        n_tok = int(mc.sum())
        if n_tok == 0:
            rows.append((None, 0))
            continue
        logits = (out["logits"] if isinstance(out, dict) else out.logits)[0].float()
        ce = F.cross_entropy(logits, yc.to(device), reduction="none")
        rows.append((float((ce * mc.to(device)).sum() / n_tok), n_tok))
    return rows, writes


def intact_health(nll: list[float], vocab: int) -> dict:
    """The absolute loss the whole I(d) table is a DIFFERENCE OF.

    I(d) prints a tidy CI whether the model underneath predicts or not, and the
    2026-09-16 probe batch is the record of what that costs: the prelaunch
    walks and gate 4 scored these checkpoints at 11.4-11.97 nats -- ln(vocab)
    is 11.52, i.e. CHANCE -- in the same job whose training loss was 2.78.
    Nothing gated on the absolute level, so a ranking of three chance-level
    losses read as "the carry is anti-informative" and ordered a program.

    This is that gate.  It does not fail the run: a level is a measurement, and
    a tool that went red on it would teach people to skip it.  It says the
    number and, when the model is at chance, says the table is unreadable.
    """
    if not nll:
        return {"mean_nats": None, "chance_nats": None, "at_chance": None}
    mean = sum(nll) / len(nll)
    chance = math.log(max(int(vocab), 2))
    return {"mean_nats": mean, "chance_nats": chance,
            "margin_below_chance": chance - mean,
            "at_chance": bool(chance - mean < 1.0), "n": len(nll)}


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
    overrides = parse_config_overrides(args.set)
    model, cfg = load_checkpoint(args.checkpoint, args.model_name, None,
                                 dtype, device,
                                 config_overrides=overrides or None)
    inner = _unwrap(model)
    buf = getattr(getattr(inner, "cortex", None), "prefix", None)
    if buf is None:
        print("FAILED: this checkpoint has no prefix buffer, so there is no "
              "carry whose influence could be measured.")
        if not args.set:
            print("  and --set was not given.  --model_name loads the BASE "
                  "dir's config, which carries no cortex flags (use_memory is "
                  "'<absent>' there), so an arm checkpoint needs its geometry "
                  "forced: --set use_memory=true --set prefix_memory=... .  "
                  "See pace/eval_deciding.sbatch, which builds them per arm.")
        return 2
    kind = "append" if _is_append(buf) else "gated"
    num_steps = to_num_steps(args.T)
    D_HIDDEN = int(buf.hidden_size)
    dual = bool(getattr(buf, "carries_latent", False))
    if args.damage_channel != "both" and not dual:
        print(f"FAILED: --damage_channel {args.damage_channel} on an E-only "
              f"carry.  There is no second channel to spare, and running it as "
              f"'both' under a per-channel label is how a 1x2 becomes a 2x2.")
        return 2
    if dual:
        print(f"[carry] dual-channel (E+Z); damaging {args.damage_channel}.  "
              f"A horizon measured on 'both' cannot say which channel carries "
              f"the distance -- run e and z as well before quoting one.")

    from datasets import load_from_disk
    ds = load_from_disk(args.data)
    n = len(ds) if args.max_examples == 0 else min(args.max_examples, len(ds))

    per_depth: dict[int, list[float]] = {d: [] for d in args.depths}
    intact_nll: list[float] = []
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
        intact_nll.append(float(intact[-1][0]))

        # The donor is another ROW's write at the same chunk index: matched
        # register, matched norms, matched column count, different content.
        donor_row = (si + 1) % n
        donor_ids = torch.tensor(ds[donor_row]["input_ids"], dtype=torch.long)
        donor_xs = list(torch.chunk(donor_ids[:keep], args.n_chunks))

        for d in args.depths:
            at = args.n_chunks - 1 - d
            donor_w = donor_z = None
            if args.damage == "donor":
                if _is_append(buf):
                    with torch.no_grad():
                        dout = model(
                            input_ids=donor_xs[at].unsqueeze(0).to(device),
                            num_steps=num_steps, m_cross_in=None,
                            return_m_cross=True)
                    dstate = (dout.get("m_cross") if isinstance(dout, dict)
                              else getattr(dout, "m_cross", None))
                    donor_w = dstate[:, -buf.n_vec:]
                else:
                    # A gated donor cannot be a slice of the returned state:
                    # that state is the post-gate ring, i.e. a MIXTURE.  Take
                    # the donor's pre-merge write instead.
                    donor_w, donor_z = capture_write(model, buf, donor_xs[at],
                                                     num_steps, device)
            # One generator per (sample, depth), so the noise is reproducible
            # from --seed and independent of the chain RNG the pairing needs.
            gen = (torch.Generator().manual_seed(args.seed + 7919 * si + d)
                   if args.damage == "random" else None)
            damaged, _ = run_chain(model, buf, xs, ys, ms, num_steps, device,
                                   seed, at, donor_w, args.damage,
                                   args.damage_channel, D_HIDDEN, gen, donor_z)
            if damaged[-1][0] is None:
                continue
            per_depth[d].append(damaged[-1][0] - intact[-1][0])
        n_used += 1
        if n_used % 10 == 0:
            print(f"  {n_used} samples", flush=True)

    report = {
        "instrument": "influence horizon I(d)",
        "damage_channel": args.damage_channel,
        "dual_channel_carry": dual,
        "when": datetime.now().isoformat(timespec="seconds"),
        "model_name": args.model_name,
        "buffer": {"kind": kind, "class": type(buf).__name__,
                   "n_vec": int(buf.n_vec),
                   "capacity": int(getattr(buf, "max_vecs",
                                           getattr(buf, "n_slots", 0))),
                   "damage_applied": ("row substitution after the write"
                                      if kind == "append" else
                                      "pre-merge write, through prefix.merge")},
        "config": {"n_chunks": args.n_chunks, "damage": args.damage,
                   "T": args.T, "dtype": args.dtype, "samples": n_used,
                   "config_overrides": overrides},
        "intact": intact_health(intact_nll,
                                getattr(inner.config, "vocab_size", 0)),
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
    _ih = report["intact"]
    if _ih["mean_nats"] is not None:
        print(f"  intact loss {_ih['mean_nats']:.4f} nats/token   chance "
              f"(ln vocab) {_ih['chance_nats']:.4f}   margin "
              f"{_ih['margin_below_chance']:+.4f}")
        if _ih["at_chance"]:
            print("  *** THE MODEL IS AT CHANCE.  Every I(d) below is a "
                  "difference of two")
            print("  *** chance-level losses and is NOT readable as a result.  "
                  "These checkpoints")
            print("  *** train at ~2.8 nats; an eval path that scores them at "
                  "~11.5 is broken,")
            print("  *** and the carry is not what it is measuring.  Fix the "
                  "load, then re-run.")
        print("")
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
