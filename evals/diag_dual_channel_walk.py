"""
The 8-chunk instrumented walk through a cortex-final carry.

A PRINTED TABLE, NOT A PASS/FAIL.  That is deliberate.  Every other instrument
in this repo scores a threshold; this one exists so the numbers can be eyeballed
against expectation before a full run is launched, because the failures that
cost this project the most were not threshold failures -- they were quantities
nobody had ever looked at.  The table prints what the packed forward actually
did to each channel on each chunk:

  rows touched      EXPECTED (from the ring rule, derived independently in
                    cortex_memory.health.expected_ring_rows) vs ACTUAL (rows
                    whose contents changed between the merge's input and its
                    output).  Asking the buffer which rows it wrote and then
                    checking that it wrote them is not a test.
  ||E write||       and ||Z write||, each as a ratio to ||s0||, because that
                    ratio is what P0.1 measured and what decided `latent_renorm`
                    ("none" at the operating point: 0.90x at T=8).  A Z write
                    that arrives at 20x s0 is a write rule landing outside the
                    measured band, and it is invisible in any loss.
  tape / depths     how many trajectory deltas the loop taped and which depths
                    the depth map selected from them.  A tape of length 0 with
                    latent_carry on means the modeling file predates the hooks.
  read_live         did THIS chunk's Z read have a gradient path at all
                    (num_steps_no_grad == 0).  The E/Z read asymmetry: one
                    no-grad step cuts Z's read gradient to exactly zero, while
                    E re-enters through input_embeds every iteration.
  write_grad_frac   share of the taped steps that were inside the gradient
                    window.  0.0 means Z is a frozen feature extractor for this
                    batch -- a legitimate configuration, but one that changes
                    what a null result for Z means.

CLOSING BLOCK
  effective rank + centred cosine of the carry's E half, against B2's measured
  ~4 of 32 and cos 0.96; the same for Z, where the delta write is mean-centred
  so the cosine measures duplication rather than anisotropy; whether `s0`'s head
  equals the carried Z (the read actually fired); and per-parameter gradient
  norms for `summary_emb`, both E gate projections and both Z gate projections.
  A grad of None is reported as None and never as 0.0 -- "no gradient path" and
  "small gradient" are different diagnoses.

USAGE
  # real checkpoint, the geometry a P1 arm runs
  python evals/diag_dual_channel_walk.py --model_name ckpts/olmo-retrofit-cortex \
      --checkpoint cortex-retrofit/p1-a3z-.../checkpoint_115966 \
      --chunks 8 --chunk_len 256 --T 8 --out eval_results/walk.json

  # forward-only (no backward), for a quick look on a small machine
  python evals/diag_dual_channel_walk.py --model_name ... --no_backward

fp32 by default, and that is not a style choice: the quantities here are
differences of large states, and a bf16 pass invented a delta plateau in P0.1
that was the rounding floor wearing the shape of a result.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cortex_memory.health import (  # noqa: E402
    carry_health, chance_margin, expected_ring_rows, gate_param_health,
    grad_norms, latent_runtime, split_carry,
)
from model_utils import shift_labels  # noqa: E402

#: Parameters the closing block reports a gradient norm for, by suffix.  The
#: list is the architecture's load-bearing set: the shared write path, both E
#: gate projections, both Z gate projections and the four biases.
WATCHED = ("summary_emb", "forget_bias", "input_bias",
           "gate_proj_in.weight", "gate_proj_mem.weight",
           "forget_bias_z", "input_bias_z",
           "gate_proj_in_z.weight", "gate_proj_mem_z.weight")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("dual-channel (E/Z) instrumented walk")
    p.add_argument("--model_name", required=True)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--chunks", type=int, default=8,
                   help="chain length.  8 is the default because a K/W=4 ring "
                        "completes two laps in it, so the gate fires")
    p.add_argument("--chunk_len", type=int, default=256)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--T", type=int, default=8,
                   help="loop depth.  Passed as (0, T) so the whole loop is in "
                        "the gradient and the Z read is LIVE -- the walk is "
                        "about what the machinery does when it can, not about "
                        "the sampler.  Use --sampled for the real split.")
    p.add_argument("--sampled", action="store_true",
                   help="let the model's own randomized_iteration_sampler pick "
                        "the split, so read_live varies chunk to chunk the way "
                        "it does in training")
    p.add_argument("--text_file", default=None,
                   help="real prose to walk over, re-tokenized with this "
                        "model's own tokenizer")
    p.add_argument("--data", default=None,
                   help="a PACKED dataset to walk over -- preferred, because it "
                        "is already tokenized with this model's tokenizer and it "
                        "is the corpus the arms train on.  ONE OF --data OR "
                        "--text_file IS REQUIRED unless --random_ids is given.")
    p.add_argument("--random_ids", action="store_true",
                   help="walk over RANDOM token ids.  Almost never what you "
                        "want: the loop has nothing to converge on, so the "
                        "trajectory is not the trained one and neither is the "
                        "carry built from it.  Measured 2026-09-16: a random-id "
                        "walk sat at loss 11.6-11.9 against ln(vocab) = 11.52, "
                        "i.e. the model was seeing noise, and every rank number "
                        "in that table described noise.  Kept for unit tests "
                        "and for checking plumbing on a toy model.")
    p.add_argument("--no_backward", action="store_true")
    p.add_argument("--dtype", default="float32",
                   choices=["float32", "bfloat16", "float16"])
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                   help="force a graft-building config flag, e.g. "
                        "--set accum_vecs=16 --set latent_carry=true.  REQUIRED "
                        "to walk a SLICED checkpoint: summary_emb is a "
                        "parameter shape, so a [16, D] slice loaded against a "
                        "config that still says 32 is a size mismatch, not a "
                        "silent one -- but the config the base dir carries is "
                        "the PARENT's, not the branch's.")
    p.add_argument("--out", default=None, help="write the record as JSON")
    return p.parse_args()


# ---------------------------------------------------------------------------
# instrumentation
# ---------------------------------------------------------------------------

class _Tap:
    """Wraps the live buffer and graft so the walk records what the forward DID.

    Instance-level wrapping, the same convention `eval_gate_prereg.tape_gates`
    uses: what is taped is the merge the forward performed, not a
    re-derivation of it from the config.
    """

    def __init__(self, cortex):
        self.cortex = cortex
        self.buf = cortex.prefix
        self.merges = []          # one dict per merge
        self.reads = []           # one dict per latent_init
        self._real_merge = self.buf.merge
        self._real_init = cortex.latent_init
        # Whether the instance already shadowed the class method, so the exit
        # restores the object to what it was rather than leaving a bound method
        # pinned as an instance attribute.
        self._had_merge = "merge" in vars(self.buf)
        self._had_init = "latent_init" in vars(cortex)

    def __enter__(self):
        def merge(state, new_vecs, new_latent=None):
            rec = {
                "chunk_index": int(getattr(self.buf, "_chunk", 0)),
                "rows_in": 0 if state is None else int(state.shape[1]),
                "e_write_norm": float(
                    new_vecs.detach().float().norm(dim=-1).mean()),
                "z_write_norm": (None if new_latent is None else float(
                    new_latent.detach().float().norm(dim=-1).mean())),
            }
            out = self._real_merge(state, new_vecs, new_latent)
            rec["rows_out"] = int(out.shape[1])
            rec["width"] = int(out.shape[-1])
            if state is not None and state.shape[1] == out.shape[1]:
                moved = (out.detach() - state.detach()).abs().sum(dim=(0, 2))
                rec["rows_touched"] = [i for i, v in enumerate(moved.tolist())
                                       if v != 0.0]
            else:
                # A growing block: every row past the old height is new, and
                # nothing below it may move.  That is the first lap, where a
                # gated ring is bit-identical to PrefixAccumBuffer.
                rec["rows_touched"] = list(range(rec["rows_in"], rec["rows_out"]))
                if state is not None:
                    rec["prefix_preserved"] = bool(torch.equal(
                        state.detach(), out.detach()[:, :state.shape[1]]))
            self.merges.append(rec)
            return out

        def latent_init(s0, num_steps_no_grad=None):
            before = s0.detach().clone()
            out = self._real_init(s0, num_steps_no_grad)
            rec = {
                "n_pre": int(self.cortex._n_pre),
                "num_steps_no_grad": (None if num_steps_no_grad is None
                                      else int(num_steps_no_grad)),
                "read_live": (None if num_steps_no_grad is None
                              else int(num_steps_no_grad) == 0),
                "s0_scale": float(before.float().flatten(0, -2)
                                  .norm(dim=-1).mean()),
                "s0_changed": bool(not torch.equal(before, out.detach())),
            }
            self.reads.append((rec, out.detach()[:, :self.cortex._n_pre].clone()))
            return out

        self.buf.merge = merge
        self.cortex.latent_init = latent_init
        return self

    def __exit__(self, *exc):
        if self._had_merge:
            self.buf.merge = self._real_merge
        else:
            del self.buf.merge
        if self._had_init:
            self.cortex.latent_init = self._real_init
        else:
            del self.cortex.latent_init
        return False


# ---------------------------------------------------------------------------
# the walk
# ---------------------------------------------------------------------------

def walk(model, cortex, chunks, num_steps=None, backward: bool = True,
         labels: bool = True) -> dict:
    """Run the chain and return the record.  `chunks` is a list of [B, S] ids.

    Takes the model AND the graft rather than digging the graft out of the
    model, so a unit test can walk a toy model built in-process and the CLI can
    walk a 1B checkpoint through exactly the same code.
    """
    buf = cortex.prefix
    D = buf.hidden_size
    W, K = buf.n_vec, getattr(buf, "n_slots", None)
    is_ring = getattr(buf, "route", None) == "ring"
    rows = []
    carry = None
    losses = []
    prev_z = None
    with _Tap(cortex) as tap:
        for i, ids in enumerate(chunks):
            kw = {"labels": shift_labels(ids)} if labels else {}
            if num_steps is not None:
                kw["num_steps"] = num_steps
            out = model(input_ids=ids, m_cross_in=carry, return_m_cross=True,
                        **kw)
            carry = out["m_cross"] if isinstance(out, dict) else out.m_cross
            if labels:
                loss = out["loss"] if isinstance(out, dict) else out.loss
                losses.append(loss)
            m = tap.merges[-1] if tap.merges else {}
            r, z_read = (tap.reads[-1] if tap.reads else ({}, None))
            rt = latent_runtime(cortex)
            row = {
                "chunk": i,
                "rows_in": m.get("rows_in"),
                "rows_out": m.get("rows_out"),
                "carry_width": m.get("width"),
                "width_is_2d": m.get("width") == 2 * D,
                "rows_touched": m.get("rows_touched"),
                "rows_expected": (expected_ring_rows(m.get("chunk_index", i), W, K)
                                  if is_ring and m.get("rows_in") == K else None),
                "e_write_norm": m.get("e_write_norm"),
                "z_write_norm": m.get("z_write_norm"),
                "s0_scale": r.get("s0_scale"),
                "read_live": r.get("read_live"),
                "s0_changed": r.get("s0_changed"),
                "num_steps_no_grad": r.get("num_steps_no_grad"),
                "tape_len": rt.get("tape_len"),
                "write_grad_frac": rt.get("write_grad_frac"),
                "loss": (float(losses[-1].detach()) if labels else None),
            }
            if row["rows_expected"] is not None and row["rows_touched"] is not None:
                row["rows_match"] = (sorted(row["rows_touched"])
                                     == sorted(row["rows_expected"]))
            for key, num in (("e_over_s0", row["e_write_norm"]),
                             ("z_over_s0", row["z_write_norm"])):
                row[key] = (None if (num is None or not row["s0_scale"])
                            else num / row["s0_scale"])
            if rt.get("latent_carry") and rt.get("tape_len"):
                row["depths"] = cortex.latent_depth_map(int(rt["tape_len"]), W)
            # DID THE READ ACTUALLY DELIVER THE CARRIED Z?  s0's head must equal
            # the Z half of the carry the PREVIOUS chunk produced.  This is the
            # check the design rests on: the read is a substitution into a field
            # that already exists, so if it silently did not happen, Z is a
            # write nothing consumes and the loss would not notice.
            if z_read is not None and prev_z is not None and z_read.shape[1]:
                n = min(z_read.shape[1], prev_z.shape[1])
                d = (z_read[:, :n].float()
                     - prev_z[:, :n].to(z_read.dtype).float()).abs().max()
                row["z_read_max_abs_err"] = float(d)
            rows.append(row)
            prev_z = None
            if carry is not None:
                _, prev_z = split_carry(carry.detach(), D)

    rec = {
        "geometry": {
            "hidden_size": D, "n_vec": W, "n_slots": K,
            "route": getattr(buf, "route", "accum"),
            "carries_latent": bool(getattr(buf, "carries_latent", False)),
            "buffer": type(buf).__name__,
            "chunks": len(chunks),
            "chunk_len": int(chunks[0].shape[1]),
        },
        "rows": rows,
        "final_carry": carry_health(carry, D),
        "gate_params": gate_param_health(buf),
        "latent": latent_runtime(cortex),
        # RED 10: the level, not only the deltas.  Every reading this
        # table feeds -- the width contrast, gate 4, the rank columns --
        # is a comparison, and a comparison of two chance-level losses
        # still looks like a result.
        "health": chance_margin(
            [r.get("loss") for r in rows],
            int(getattr(getattr(model, "config", None), "vocab_size", 0) or 0)),
    }
    if backward and losses:
        # ONE backward over the SUMMED chain loss.  Stronger than a per-chunk
        # backward for the question this block asks -- "is this parameter block
        # attached to the loss at all" -- because a parameter that only receives
        # gradient through a chain of >= 3 chunks (gate_proj_mem does) would
        # report None under a per-chunk backward and read as dead.
        torch.stack([l for l in losses]).sum().backward()
        g = grad_norms(buf, prefix="prefix.")
        rec["grads"] = {k: v for k, v in g.items()
                        if any(k.endswith(w) for w in WATCHED)}
        rec["grad_none"] = sorted(k for k, v in rec["grads"].items() if v is None)
        rec["grad_zero"] = sorted(k for k, v in rec["grads"].items()
                                  if v is not None and v == 0.0)
    return rec


# ---------------------------------------------------------------------------
# printing
# ---------------------------------------------------------------------------

def _f(v, spec="8.3f", none="     -  "):
    return none if v is None else format(v, spec)


def print_walk(rec, out=sys.stdout) -> None:
    g = rec["geometry"]
    p = lambda *a: print(*a, file=out)
    p("")
    p("=" * 100)
    p(f"DUAL-CHANNEL WALK  {g['buffer']}  W={g['n_vec']} K={g['n_slots']} "
      f"route={g['route']}  Z={'on' if g['carries_latent'] else 'OFF'}  "
      f"{g['chunks']} chunks x {g['chunk_len']} tokens")
    p("=" * 100)
    p(f"{'ch':>2} {'rows':>9} {'width':>6} {'touched':>9} {'as ring?':>9} "
      f"{'|E|/s0':>8} {'|Z|/s0':>8} {'tape':>5} {'wgrad':>6} {'read':>5} "
      f"{'z-read err':>11} {'loss':>8}")
    for r in rec["rows"]:
        touched = r["rows_touched"]
        ring = ("-" if r.get("rows_match") is None
                else ("yes" if r["rows_match"] else "NO"))
        p(f"{r['chunk']:>2} "
          f"{str(r['rows_in']) + '->' + str(r['rows_out']):>9} "
          f"{('2D' if r['width_is_2d'] else 'D'):>6} "
          f"{(len(touched) if touched is not None else 0):>9} "
          f"{ring:>9} "
          f"{_f(r['e_over_s0'], '8.2f')} {_f(r['z_over_s0'], '8.2f')} "
          f"{_f(r['tape_len'], '5d', '    -')} "
          f"{_f(r['write_grad_frac'], '6.2f', '     -')} "
          f"{('yes' if r['read_live'] else ('no' if r['read_live'] is False else '-')):>5} "
          f"{_f(r.get('z_read_max_abs_err'), '11.2e', '          -')} "
          f"{_f(r['loss'], '8.4f')}")
    h = rec.get("health") or {}
    if h.get("margin") is not None:
        p("")
        p(f"   mean NLL {h['mean_nll']:.4f}   chance (ln vocab) {h['chance']:.4f}"
          f"   margin {h['margin']:+.4f}")
        if h.get("at_chance"):
            # RED 10.  Printed four times so it cannot be scrolled past: every
            # number in this table is a statistic OF NOISE when this fires.
            for _ in range(4):
                p("   *** AT CHANCE -- this walk scored nothing.  Every rank, "
                  "norm and delta below is noise. ***")
    c = rec["final_carry"]
    p("")
    p("-- final carry ------------------------------------------------------")
    p(f"   rows {c.get('rows')}  channels {c.get('channels')}")
    p(f"   E: row_norm {_f(c.get('e_row_norm'), '.3f', '-')}  "
      f"centred_cos {_f(c.get('e_centred_cosine'), '.4f', '-')}  "
      f"eff_rank_pr {_f(c.get('e_eff_rank_pr'), '.2f', '-')}  "
      f"entropy {_f(c.get('e_eff_rank_entropy'), '.2f', '-')}")
    if c.get("channels") == 2:
        p(f"   Z: row_norm {_f(c.get('z_row_norm'), '.3f', '-')}  "
          f"centred_cos {_f(c.get('z_centred_cosine'), '.4f', '-')}  "
          f"eff_rank_pr {_f(c.get('z_eff_rank_pr'), '.2f', '-')}  "
          f"entropy {_f(c.get('z_eff_rank_entropy'), '.2f', '-')}")
        p(f"   Z/E norm ratio {_f(c.get('z_over_e_norm'), '.4f', '-')}  "
          f"unwritten Z rows {c.get('z_zero_rows')}")
    gp = rec.get("gate_params") or {}
    if gp:
        p("")
        p("-- gate parameters --------------------------------------------------")
        p(f"   E  fg(bias) {_f(gp.get('fg_at_bias'), '.4f', '-')}  "
          f"ig(bias) {_f(gp.get('ig_at_bias'), '.4f', '-')}  "
          f"|W_in| {_f(gp.get('gate_in_w_norm'), '.3e', '-')}  "
          f"|W_mem| {_f(gp.get('gate_mem_w_norm'), '.3e', '-')}  "
          f"left init: {gp.get('gate_left_init')}")
        if "fg_z_at_bias" in gp:
            p(f"   Z  fg(bias) {_f(gp.get('fg_z_at_bias'), '.4f', '-')}  "
              f"ig(bias) {_f(gp.get('ig_z_at_bias'), '.4f', '-')}  "
              f"|W_in| {_f(gp.get('gate_in_z_w_norm'), '.3e', '-')}  "
              f"|W_mem| {_f(gp.get('gate_mem_z_w_norm'), '.3e', '-')}  "
              f"left init: {gp.get('gate_z_left_init')}")
    if "grads" in rec:
        p("")
        p("-- gradient norms (one backward over the summed chain loss) ---------")
        for k in sorted(rec["grads"]):
            v = rec["grads"][k]
            tag = "  <-- NO GRADIENT PATH" if v is None else (
                "  <-- exactly zero" if v == 0.0 else "")
            p(f"   {k:<34} {_f(v, '.6e', 'None'):>14}{tag}")
    lat = rec.get("latent") or {}
    if lat.get("latent_carry"):
        p("")
        p(f"   Z read grad frac over this walk: {lat['read_grad_frac']:.2f} "
          f"(measured={lat['read_measured']})  write grad frac "
          f"{lat['write_grad_frac']:.2f}")
        p("   read_grad_frac is a property of the SAMPLER, not of the walk: one")
        p("   no-grad step cuts Z's read gradient to exactly zero, and E's is")
        p("   unaffected because E re-enters through input_embeds every step.")
    p("")
    p("This table is evidence about the MACHINERY, not about whether Z helps.")
    p("That needs the full cell, the 2x2 and the influence horizon.")
    p("")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _chunks_from_args(args, model, tok_vocab):
    torch.manual_seed(args.seed)
    if args.data:
        from datasets import load_from_disk
        ds = load_from_disk(args.data)
        need = args.chunks * args.chunk_len
        rows = [ds[i]["input_ids"][:need] for i in range(args.batch)]
        if min(len(r) for r in rows) < need:
            raise SystemExit(
                f"{args.data} rows are shorter than {need} tokens; lower "
                f"--chunks or --chunk_len.")
        ids = torch.tensor(rows, dtype=torch.long)
    elif args.text_file:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.model_name,
                                            trust_remote_code=True)
        need = args.chunks * args.chunk_len * args.batch + 8
        text = open(args.text_file, encoding="utf-8", errors="ignore").read()
        ids = tok(text, return_tensors="pt").input_ids[0]
        while ids.numel() < need:
            ids = torch.cat([ids, ids])
        ids = ids[:need - 8].reshape(args.batch, -1)
        ids = ids[:, :args.chunks * args.chunk_len]
    elif args.random_ids:
        ids = torch.randint(0, tok_vocab - 1,
                            (args.batch, args.chunks * args.chunk_len))
    else:
        raise SystemExit(
            "no prose source.  Pass --data <packed dataset> (preferred) or "
            "--text_file <path>.  --random_ids exists but gives the loop "
            "nothing to converge on, so every number in the table would "
            "describe noise rather than the trained trajectory.")
    return [c.contiguous() for c in torch.chunk(ids, args.chunks, dim=1)]


def main() -> int:
    args = parse_args()
    from model_utils import (explain_missing_cortex,  # noqa: E402
                             load_checkpoint, parse_config_overrides,
                             to_num_steps, _unwrap)
    device = torch.device(args.device)
    overrides = parse_config_overrides(args.set)
    model, cfg = load_checkpoint(args.checkpoint, args.model_name, None,
                                 getattr(torch, args.dtype), device,
                                 config_overrides=overrides or None)
    inner = _unwrap(model)
    cortex = getattr(inner, "cortex", None)
    if cortex is None or cortex.prefix is None:
        print("ERROR: no prefix buffer on this model -- nothing to run.")
        print("  " + explain_missing_cortex(cfg, overrides))
        return 1
    # TRAIN MODE, on purpose.  In eval the sampler returns (mean_recurrence, 0),
    # so every chunk would report read_live=no for a reason that has nothing to
    # do with the architecture.
    inner.train()
    chunks = _chunks_from_args(args, inner, inner.config.vocab_size)
    chunks = [c.to(device) for c in chunks]
    num_steps = None if args.sampled else to_num_steps(args.T)
    if num_steps is not None:
        # (0, T): the whole loop inside the gradient window, so the walk reports
        # what the machinery does when it CAN.  --sampled reports what training
        # actually sees.
        num_steps = torch.tensor([0, int(args.T)])
    rec = walk(inner, cortex, chunks, num_steps=num_steps,
               backward=not args.no_backward)
    rec["args"] = vars(args)
    rec["when"] = datetime.now().isoformat(timespec="seconds")
    print_walk(rec)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="ascii") as fh:
            json.dump(rec, fh, indent=2, default=str)
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
