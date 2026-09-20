"""
P0.2 — slot-slot effective rank across position layouts.
cortex_next_phase_framework.md §5 (issue 9) and §6.

WHAT IT DECIDES, AND WHY K IS DOWNSTREAM OF IT.
`n_vec` is a parameter shape, so K is chosen once and lived with.  The number
that says whether K buys anything is not K: it is how many INDEPENDENT
DIRECTIONS the K slots actually resolve into, and that is set by the position
layout.  The figures already on record:

    slot positions          slot-slot cosine   effective rank (of 32)
    all 0   (B2 attempt 1)        0.94                2.1
    tail, S+1..S+n (B2 final)      --                 4.2   <- the real arm
    AC-Llama contiguous           0.22                4.2
    spread over 0..S-1            0.23               19.7   <- never run

A ~5x rank difference from an integer assignment.  At 4.2 of 32 the buffer is
buying four directions and paying for thirty-two; widening to K=64 before the
layout is fixed buys nothing but VRAM.  So: measure rank across layouts, THEN
choose K.

THE WRITE SIDE AND THE READ SIDE ARE DIFFERENT QUESTIONS.  `prefix_pack` puts
the packed sequence together as [carried vectors | real tokens | summary slots]:

  * the SUMMARY SLOTS (write side) sit at the tail, positions S+1..S+n_vec.
    Spreading them changes what each slot sees under the causal mask — a slot
    at position p attends to real tokens 1..p and to the slots before it.  That
    is a feature: staggered spans instead of thirty-two views of the same
    prefix.
  * the CARRIED VECTORS (read side) sit at the FRONT and ALL AT POSITION 0.
    That is the same layout that measured 2.1/32 on the write side and was
    retired there in 2026-09-15 — and `prefix_pack`'s own docstring concedes
    it: "the carry has no recency order in position space; that is a separate,
    unmeasured question."  This probe is that measurement.

So the two sides are swept separately and reported separately.  Do not collapse
them into one recommendation.

WHAT IT MEASURES.  For a [K, D] stack of slot states, per batch lane, averaged:

  centred cosine   mean off-diagonal cosine AFTER subtracting the mean slot.
                   Centring matters: K vectors that share a large common
                   component read as ~1.0 uncentred no matter how much
                   independent structure sits on top of it.  This is the
                   statistic the 0.94 / 0.22 / 0.23 figures are in.
  eff_rank_entropy exp(H) of the singular-value distribution p_i = s_i / sum s
                   (Roy & Vetterli).  Max K, reached only at a flat spectrum.
                   THIS IS THE HEADLINE — it is the one that reads "of 32".
  eff_rank_pr      participation ratio (sum s^2)^2 / sum s^4.  A second
                   estimator with a different tail sensitivity; reported so a
                   layout that wins on one and loses on the other is visible
                   rather than averaged away.
  singular_values  the top of the spectrum, so any of the above can be
                   recomputed later without re-running the probe.

  A caveat that belongs in the writeup, not a footnote: the 4.2 / 19.7 figures
  came from an ad-hoc script that was never committed, so this probe cannot
  claim to reproduce them digit for digit.  It measures all layouts the same
  way in one process, which is what makes the COMPARISON sound; treat the
  absolute numbers as this probe's own scale.

HOW THE LAYOUTS ARE APPLIED.  `cortex.prefix_pos` is asserted to be "tail" at
build time and the branch was retired, so there is nothing to configure — this
probe wraps `prefix_pack` and rewrites the position ids it returns, exactly the
way P0.1 wraps `core_block_forward`.  The model is untouched, no checkpoint is
modified, and the retired branch is not resurrected.

  zero        every slot at position 0                       (B2 attempt 1)
  tail        S+1 .. S+K, continuing the chunk numbering      (B2 final, write)
  contiguous  0 .. K-1                                        (AC-Llama)
  spread      evenly spaced over 0 .. S-1                     (never run)

For the read side, "tail" is meaningless (the carried vectors precede the real
tokens), so the read sweep runs zero / contiguous / spread.  `zero` is the
production layout there.

TWO MODES, BECAUSE K CANNOT BE SWEPT ON A TRAINED CHECKPOINT.
  default        the checkpoint's own trained `summary_emb`, K = n_vec.  This
                 is the real arm and the number that describes the model we
                 have.
  --init_slots K  replace the slots with K copies of wte[eos] — the AT-INIT
                 condition the 19.7/32 figure was measured in, and the only
                 honest way to ask "would K=64 resolve more directions" without
                 a K=64 model.  Use 32 / 64 / 128 and read the curve.  Rank
                 that saturates well below K says width is not the binding
                 constraint; rank that keeps climbing says it might be.

TRAPS THIS SCRIPT IS WRITTEN AROUND.
  * RUN IT IN FLOAT32 (the default).  Singular values of a near-collinear
    stack span orders of magnitude, and an entropy over a bf16 spectrum is
    reading quantization in the tail.  Same lesson as P0.1's delta plateau.
  * The carry is NOT hidden state — it is passed in as `m_cross_in` and comes
    back as `out.m_cross`, so the chunk chain is driven explicitly here.  There
    is no buffer to reset between layouts, and each layout gets a clean chunk 1.
  * The READ side needs a populated buffer, so it runs a two-chunk chain: chunk
    1 writes under the production tail layout, chunk 2 reads under the layout
    being swept.  Sweeping the read layout on chunk 1 would measure nothing —
    n_pre is 0 there.
  * `out.m_cross` is the MERGED buffer, not this chunk's write: under accum,
    chunk 2's m_cross is 64 vectors, not 32.  Both sides are therefore read off
    a `prefix_unpack` wrapper, which sees the packed columns themselves and
    cannot confuse a merge for a write.
  * Restore by DELETING the instance shadow, not by assigning the bound method
    back — the same reference-cycle trap P0.1 documents.

Usage (local, 1B checkpoint, no cluster needed):
    python evals/diag_position_rank.py \
        --model_name ../checkpoints/retrofit/retro-b2-acc32-cc8-mr8-final_checkpoint \
        --text_file ../cortex_literature_review.md --seq_len 512 --batch 2

    # price K at init:
    python evals/diag_position_rank.py --model_name <ckpt> \
        --text_file <file> --init_slots 32 --init_slots 64 --init_slots 128
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
from datetime import datetime

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evals.model_utils import _unwrap, load_checkpoint  # noqa: E402
from evals.model_utils import parse_config_overrides  # noqa: E402

WRITE_LAYOUTS = ("zero", "tail", "contiguous", "spread")
READ_LAYOUTS = ("zero", "contiguous", "spread")


# ---------------------------------------------------------------------------
# Position layouts
# ---------------------------------------------------------------------------

def layout_positions(layout: str, n: int, S: int, last_real: int,
                     device, dtype) -> torch.Tensor:
    """The position ids for `n` slot columns, as a [n] tensor.

    `S` is the real-token count and `last_real` the highest real position in
    use, so "tail" continues a cached prefill's numbering rather than assuming
    the chunk started at 0 -- the same derivation prefix_pack makes.
    """
    if n <= 0:
        return torch.zeros(0, device=device, dtype=dtype)
    if layout == "zero":
        return torch.zeros(n, device=device, dtype=dtype)
    if layout == "tail":
        return last_real + torch.arange(1, n + 1, device=device, dtype=dtype)
    if layout == "contiguous":
        return torch.arange(n, device=device, dtype=dtype)
    if layout == "spread":
        # Evenly spaced over the real span, endpoints included.  Integer
        # positions, so at n > S there are unavoidable repeats -- that is a
        # real property of the layout at large K, not a bug to paper over.
        if S <= 1:
            return torch.zeros(n, device=device, dtype=dtype)
        step = (S - 1) / max(n - 1, 1)
        idx = (torch.arange(n, device=device, dtype=torch.float32) * step)
        return idx.round().to(dtype)
    raise ValueError(f"unknown layout {layout!r}")


# ---------------------------------------------------------------------------
# Rank statistics
# ---------------------------------------------------------------------------

def rank_stats(mat: torch.Tensor, keep_sv: int = 8) -> dict:
    """Slot-geometry statistics for a [B, K, D] stack, averaged over B.

    Everything is computed in float32 on the centred stack.  Centring is the
    difference between "these vectors point the same way" and "these vectors
    share an offset", and only the first is a capacity problem.
    """
    x = mat.detach().float()
    B, K, _ = x.shape
    cos_all, er_all, pr_all, sv_all, ratio_all = [], [], [], [], []
    for b in range(B):
        m = x[b]
        c = m - m.mean(dim=0, keepdim=True)
        raw_n = m.norm(dim=-1).mean().clamp_min(1e-30)
        cen_n = c.norm(dim=-1).mean()
        ratio = float(cen_n / raw_n)
        ratio_all.append(ratio)
        if ratio < 1e-6:
            # Every slot is the same vector: the centred stack is numerical
            # dust and its cosine/rank are the dust's geometry, not the
            # model's.  Report the honest limit -- one direction, perfectly
            # aligned -- instead of the 0.0 cosine that dividing noise by its
            # own norm would produce.
            cos_all.append(1.0)
            er_all.append(1.0)
            pr_all.append(1.0)
            sv_all.append([0.0] * min(keep_sv, K))
            continue
        nrm = c.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        u = c / nrm
        g = u @ u.T
        off = ~torch.eye(K, dtype=torch.bool, device=g.device)
        cos_all.append(float(g[off].mean()))

        s = torch.linalg.svdvals(c)
        s = s.clamp_min(0)
        tot = s.sum()
        if float(tot) <= 0:
            er_all.append(1.0)
            pr_all.append(1.0)
            sv_all.append([0.0] * min(keep_sv, K))
            continue
        p = s / tot
        nz = p[p > 0]
        er_all.append(float(torch.exp(-(nz * nz.log()).sum())))
        s2 = (s ** 2)
        pr_all.append(float((s2.sum() ** 2) / (s2 ** 2).sum().clamp_min(1e-30)))
        sv_all.append([float(v) for v in s[:keep_sv]])

    mean_sv = [sum(col) / len(col) for col in zip(*sv_all)]
    return {
        "K": int(K),
        "centred_cosine": sum(cos_all) / len(cos_all),
        "eff_rank_entropy": sum(er_all) / len(er_all),
        "eff_rank_pr": sum(pr_all) / len(pr_all),
        "singular_values": mean_sv,
        # mean ||slot - mean slot|| / mean ||slot||.  Near 0 means the slots are
        # one vector plus dust and every statistic above is at its degenerate
        # limit; near 1 means the shared component is small and the centred
        # numbers describe the whole stack.
        "centred_norm_ratio": sum(ratio_all) / len(ratio_all),
    }


# ---------------------------------------------------------------------------
# The forward, with the layout spliced in
# ---------------------------------------------------------------------------

def run_chunk(model, inner, input_ids, num_steps: int, m_cross_in,
              write_layout: str, read_layout: str,
              init_slot_emb: torch.Tensor | None = None,
              zero_carry: bool = False):
    """One forward with the slot positions rewritten.

    Returns (captured, out) where captured holds the packed columns for the
    carried block and the summary block, read off a prefix_unpack wrapper.
    """
    if zero_carry and m_cross_in is not None:
        # The LENGTH-MATCHED null: keep every carried column, empty its
        # contents.  Dropping the columns instead would shorten the packed
        # sequence, which changes initialize_state's draw for the real tokens
        # and puts an s0 difference inside what is supposed to be the carry's
        # contribution.  (A zero key still scores a mid-range logit rather than
        # -inf, so this null absorbs a few percent of softmax mass -- it is the
        # project's standing null, not a perfect one.)
        # E-only by construction: this probe runs on single-channel
        # checkpoints, and zeros_like on a DUAL-channel carry would null the Z
        # trajectory as well as the token summary -- an "E off" condition that
        # is silently "everything off".  Assert rather than branch: the probe
        # has no Z-aware interpretation to offer, so a dual-channel checkpoint
        # here means the wrong tool, not a special case.
        assert m_cross_in.shape[-1] == int(cfg.n_embd), (
            f"carry is {m_cross_in.shape[-1]}-wide, expected {cfg.n_embd}: this "
            f"is a dual-channel (E+Z) checkpoint and this probe only "
            f"interprets the token channel.  Use evals/eval_carry_2x2.py, whose "
            f"null_e nulls the E half only.")
        m_cross_in = torch.zeros_like(m_cross_in)
    cortex = getattr(inner, "cortex", None)
    if cortex is None or not hasattr(cortex, "prefix_pack"):
        raise RuntimeError(
            "cortex.prefix_pack is not on the model -- the prefix splice has "
            "moved.  Re-read cortex_graft.py before trusting anything below.")

    captured: dict = {}
    real_pack = cortex.prefix_pack
    real_unpack = cortex.prefix_unpack

    def wrapped_pack(input_embeds, position_ids, emb_scale=1.0,
                     write=True, read=True):
        packed, pos, n_pre, n_sum = real_pack(
            input_embeds, position_ids, emb_scale=emb_scale,
            write=write, read=read)
        S = input_embeds.shape[1]
        if n_sum and init_slot_emb is not None:
            # Swap the trained slots for K EOS copies, and re-cut the packed
            # tensor so n_sum reflects the requested K.
            slots = (init_slot_emb.unsqueeze(0)
                     .expand(packed.shape[0], -1, -1)
                     .to(device=packed.device, dtype=packed.dtype) * emb_scale)
            packed = torch.cat([packed[:, :n_pre + S], slots], dim=1)
            n_sum = slots.shape[1]
        last_real = int(pos[0, n_pre + S - 1]) if S > 0 else 0
        dev, dt = pos.device, pos.dtype
        new_pos = pos[:, :n_pre + S]
        if n_pre:
            rp = layout_positions(read_layout, n_pre, S, last_real, dev, dt)
            new_pos = torch.cat(
                [rp.unsqueeze(0).expand(pos.shape[0], -1), new_pos[:, n_pre:]],
                dim=1)
        if n_sum:
            wp = layout_positions(write_layout, n_sum, S, last_real, dev, dt)
            new_pos = torch.cat(
                [new_pos, wp.unsqueeze(0).expand(pos.shape[0], -1)], dim=1)
        captured["n_pre"], captured["n_sum"], captured["S"] = n_pre, n_sum, S
        captured["pos"] = new_pos[0].tolist()
        return packed, new_pos, n_pre, n_sum

    def wrapped_unpack(x, n_pre, n_sum):
        if n_pre:
            captured["read_cols"] = x[:, :n_pre].detach()
        if n_sum:
            captured["write_cols"] = x[:, n_pre + captured["S"]:].detach()
        return real_unpack(x, n_pre, n_sum)

    had_pack = "prefix_pack" in vars(cortex)
    had_unpack = "prefix_unpack" in vars(cortex)
    cortex.prefix_pack = wrapped_pack
    cortex.prefix_unpack = wrapped_unpack
    try:
        with torch.no_grad():
            out = model(input_ids=input_ids,
                        num_steps=torch.tensor([num_steps, 0]),
                        m_cross_in=m_cross_in,
                        return_m_cross=True,
                        output_details={"return_logits": False,
                                        "return_latents": False,
                                        "return_head": True,
                                        "return_stats": False},
                        prefix_write=True, prefix_read=m_cross_in is not None)
    finally:
        if had_pack:
            cortex.prefix_pack = real_pack
        else:
            del cortex.prefix_pack
        if had_unpack:
            cortex.prefix_unpack = real_unpack
        else:
            del cortex.prefix_unpack
    return captured, out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def load_input_ids(args, cfg, device):
    if args.data:
        from datasets import load_from_disk
        ds = load_from_disk(args.data)
        rows = [ds[i]["input_ids"][: args.seq_len] for i in range(args.batch)]
        return torch.tensor(rows, dtype=torch.long, device=device), args.data
    if args.text_file:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.model_name,
                                            trust_remote_code=True)
        raw = io.open(args.text_file, encoding="utf-8", errors="replace").read()
        ids = tok(raw, return_tensors=None)["input_ids"]
        need = args.batch * args.seq_len * 2          # two chunks
        if len(ids) < need:
            raise SystemExit(
                f"FAILED: {args.text_file} tokenizes to {len(ids)} tokens, "
                f"need {need} for batch={args.batch} seq_len={args.seq_len} "
                f"over two chunks.")
        rows = [ids[i * args.seq_len * 2:(i + 1) * args.seq_len * 2]
                for i in range(args.batch)]
        return (torch.tensor(rows, dtype=torch.long, device=device),
                f"{args.text_file} ({len(ids)} tokens available)")
    # Random ids give the slots nothing coherent to summarise, so the write-side
    # geometry is not the trained one.  Allowed, flagged, never quoted.
    return (torch.randint(0, int(cfg.vocab_size) - 1,
                          (args.batch, args.seq_len * 2), device=device),
            "random ids (NOT the trained regime -- do not quote)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", required=True)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--data", default=None, help="a packed dataset")
    ap.add_argument("--text_file", default=None,
                    help="a UTF-8 text file, tokenized with the checkpoint's "
                         "own tokenizer.  Real prose, no pack needed.")
    ap.add_argument("--seq_len", type=int, default=512,
                    help="real tokens per chunk.  512 is the B2 arm's trained "
                         "chunk length (max_length 4096 / cross_chunks 8).")
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--steps", type=int, default=8,
                    help="T.  Defaults to 8, the TRAINED depth of the mr8 "
                         "arms.  Leaving this at the config's mean_recurrence "
                         "would evaluate at 32 -- finding 0c, the same trap "
                         "T_OVERRIDE exists for in eval_basic.sbatch.")
    ap.add_argument("--init_slots", type=int, action="append", default=None,
                    metavar="K",
                    help="Repeatable.  Run the at-init condition with K "
                         "EOS-seeded slots instead of the trained summary_emb, "
                         "to price K.  e.g. --init_slots 32 --init_slots 64")
    ap.add_argument("--eos_id", type=int, default=None,
                    help="token to seed --init_slots from; default is the "
                         "tokenizer's eos_token_id, then config.eos_token_id")
    ap.add_argument("--dtype", default="float32",
                    choices=["bfloat16", "float32"],
                    help="float32 by DEFAULT and it matters: an entropy over a "
                         "bf16 singular-value spectrum reads quantization in "
                         "the tail as structure.")
    ap.add_argument("--device",
                    default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=None)
    ap.add_argument("--set", action="append", default=[],
                    metavar="KEY=VALUE",
                    help="force a graft-building config flag, e.g. "
                         "--set use_memory=true --set prefix_memory=gated.  "
                         "REQUIRED on an OVERLAY checkpoint: train.py's "
                         "checkpoint_<step> dirs, and the _w16 branch dirs cut "
                         "from them, hold chkpt.pt and NO config.json, so "
                         "--model_name loads the BASE dir whose config carries "
                         "no cortex flags at all (use_memory is literally "
                         "'<absent>' on ckpts/olmo-retrofit-cortex) and without "
                         "these the graft builds with no buffer.  Mirror the "
                         "arm's PROBE_SETS in pace/p1_arms.sbatch.  RED 12.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)

    overrides = parse_config_overrides(args.set)
    model, cfg = load_checkpoint(args.checkpoint, args.model_name, None,
                                 dtype, device,
                                 config_overrides=overrides or None)
    inner = _unwrap(model)
    cortex = getattr(inner, "cortex", None)
    if cortex is None or getattr(cortex, "prefix", None) is None:
        print("FAILED: this checkpoint has no prefix buffer -- there are no "
              "slots to measure.  P0.2 needs a prefix-memory arm.")
        return 2
    n_vec = int(cortex.prefix.n_vec)

    ids, source = load_input_ids(args, cfg, device)
    S = args.seq_len
    chunk1, chunk2 = ids[:, :S], ids[:, S:2 * S]

    eos_id = args.eos_id
    if eos_id is None:
        eos_id = int(getattr(cfg, "eos_token_id", 0) or 0)

    report = {
        "probe": "P0.2 position layout / slot rank",
        "when": datetime.now().isoformat(timespec="seconds"),
        "model_name": args.model_name,
        "checkpoint": args.checkpoint,
        "source": source,
        "geometry": {"D": int(cfg.n_embd), "n_vec": n_vec, "T": args.steps,
                     "seq_len": S, "batch": args.batch, "dtype": args.dtype,
                     "eos_id": eos_id},
        "write_side": {},
        "read_side": {},
        "init_sweep": {},
    }

    # --- WRITE SIDE.  Chunk 1, no carry, layout swept. ----------------------
    for lay in WRITE_LAYOUTS:
        cap, out = run_chunk(model, inner, chunk1, args.steps, None,
                             write_layout=lay, read_layout="zero")
        if "write_cols" not in cap:
            print("FAILED: captured no summary columns -- the prefix_unpack "
                  "hook did not fire.")
            return 2
        if cap["n_sum"] != n_vec:
            print(f"FAILED: expected {n_vec} summary columns, packed "
                  f"{cap['n_sum']}.")
            return 2
        report["write_side"][lay] = rank_stats(cap["write_cols"])
        report["write_side"][lay]["slot_positions"] = \
            cap["pos"][cap["n_pre"] + cap["S"]:]

    # --- READ SIDE.  Chunk 1 writes under tail, chunk 2 reads. --------------
    _, out1 = run_chunk(model, inner, chunk1, args.steps, None,
                        write_layout="tail", read_layout="zero")
    carry = getattr(out1, "m_cross", None)
    if carry is None or carry.shape[1] == 0:
        print("FAILED: chunk 1 produced no carry, so the read side cannot be "
              "measured.")
        return 2
    for lay in READ_LAYOUTS:
        cap, out_on = run_chunk(model, inner, chunk2, args.steps, carry,
                                write_layout="tail", read_layout=lay)
        if "read_cols" not in cap:
            print("FAILED: captured no carried columns on chunk 2.")
            return 2
        _, out_off = run_chunk(model, inner, chunk2, args.steps, carry,
                               write_layout="tail", read_layout=lay,
                               zero_carry=True)
        h_on = getattr(out_on, "hidden_states", None)
        h_off = getattr(out_off, "hidden_states", None)
        if h_on is None or h_off is None:
            print("FAILED: no post-ln_f token states -- return_head did not "
                  "reach the output.")
            return 2
        # DELIVERED rank, which is the read side's actual question.  The
        # carried columns' OWN geometry is nearly layout-invariant (they enter
        # as a ~440x-s0 residual that attention barely perturbs, P0.1), so
        # measuring it answers nothing.  What the layout can change is how many
        # directions the REAL TOKENS retrieve, so measure that difference
        # directly: delta = tokens with the carry - tokens with it emptied,
        # over the S token positions.
        #
        # NOT capped at K.  A first-guess reading says the carry can only move
        # the tokens inside the <=K-dimensional span its columns present to
        # attention -- but that holds at the first attention output and nowhere
        # after it.  MLPs, the recurrence and every later layer mix that
        # contribution into the full residual space, and the measurement says
        # so: K=32 columns deliver a delta of effective rank ~349 over 512
        # token positions.  So this number is NOT "how many of the 32 slots
        # arrived"; it is how many directions of the token stream the carry
        # ends up moving.  The `K` column below is the ROW COUNT (S), not the
        # buffer width.
        delta = (h_on.float() - h_off.float())
        rec = rank_stats(delta)
        rec["column_geometry"] = rank_stats(cap["read_cols"])
        rec["delta_norm"] = float(delta.norm(dim=-1).mean())
        rec["token_norm"] = float(h_on.float().norm(dim=-1).mean())
        rec["slot_positions"] = cap["pos"][:cap["n_pre"]]
        report["read_side"][lay] = rec

    # --- INIT SWEEP.  Price K on EOS-seeded slots. --------------------------
    if args.init_slots:
        wte = inner.get_input_embeddings().weight
        for K in args.init_slots:
            slot_emb = wte[eos_id].detach().unsqueeze(0).expand(K, -1)
            report["init_sweep"][str(K)] = {}
            for lay in WRITE_LAYOUTS:
                cap, _ = run_chunk(model, inner, chunk1, args.steps, None,
                                   write_layout=lay, read_layout="zero",
                                   init_slot_emb=slot_emb)
                report["init_sweep"][str(K)][lay] = rank_stats(
                    cap["write_cols"])

    out_path = args.out or os.path.join(
        "eval_results", f"p02_position_rank-{datetime.now():%Y%m%d-%H%M%S}",
        "results.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    # --- Report -------------------------------------------------------------
    def table(title, block, note=""):
        print(f"\n  {title}")
        if note:
            print(f"  {note}")
        print(f"    {'layout':<12}{'K':>4}{'centred cos':>13}"
              f"{'eff_rank(H)':>13}{'eff_rank(PR)':>14}{'of K':>8}")
        for lay, r in block.items():
            print(f"    {lay:<12}{r['K']:>4}{r['centred_cosine']:>13.4f}"
                  f"{r['eff_rank_entropy']:>13.2f}{r['eff_rank_pr']:>14.2f}"
                  f"{r['eff_rank_entropy'] / r['K']:>8.1%}")

    print(f"\n=== P0.2 position layout | {os.path.basename(args.model_name)} "
          f"| {source} ===")
    print(f"    D={cfg.n_embd} n_vec={n_vec} T={args.steps} seq_len={S} "
          f"batch={args.batch} dtype={args.dtype}")

    table("WRITE SIDE — the summary slots (production layout: tail)",
          report["write_side"])
    table("READ SIDE — directions the carry DELIVERS into the tokens "
          "(production: zero)",
          report["read_side"],
          "rank of (tokens with carry - tokens with carry zeroed), chunk 2; "
          "K is the ceiling")
    print("    (K above is the TOKEN COUNT, not the buffer width: the "
          "carry's influence is\n     not confined to a 32-dim subspace by "
          "the time it reaches ln_f.)")
    for lay, r in report["read_side"].items():
        print(f"    {lay:<12}delta/token norm "
              f"{r['delta_norm'] / max(r['token_norm'], 1e-9):>8.3%}"
              f"   column geometry eff_rank(H) "
              f"{r['column_geometry']['eff_rank_entropy']:>6.2f}")
    for K, block in report["init_sweep"].items():
        table(f"AT INIT, K={K} EOS-seeded slots", block,
              "prices width: rank that saturates below K says K is not "
              "the binding constraint")

    print("\n  HOW TO READ IT.  The write side chooses the slot layout; the "
          "read side\n  chooses the carry layout, and they are independent "
          "decisions.  Pick K only\n  after the best layout's rank is known — "
          "a layout at 4/32 is not short of\n  width, and one near K is.")
    print(f"\n  wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
