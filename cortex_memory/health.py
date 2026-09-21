"""
Health statistics for a cortex-final carry -- the numbers that are invisible in
the loss curve.

WHY THIS IS A MODULE AND NOT A FUNCTION IN EACH PROBE.  Every failure this
project has paid for was a piece of the architecture sitting inert behind a
perfectly healthy loss: a zero-init read that was a literal no-op, a detached
tape carrying no gradient, a re-seeded `summary_emb` throwing away 91,552 steps,
a gate whose init was clobbered by post_init.  The instruments that catch those
have to agree with each other, and four copies of "effective rank" that drifted
apart would be a fifth failure of the same kind.  So the statistics live here,
in a module with no host-model dependency and no argparse, and the probes
(`evals/diag_dual_channel_walk.py`, `evals/diag_gate_geometry.py`,
`tools/prelaunch_final.py`, `tools/compare_arms.py`) and `train.py` all import
them.

NOTHING HERE READS A CONFIG.  Every function takes the tensors or the live
module the forward actually produced.  "The config said gate_slots=64" and "the
merge wrote 16 rows into a 64-row state" are different claims and only the
second one is evidence.
"""
from __future__ import annotations

import math
from typing import Optional

#: How many leading eigenvalue shares `rank_stats` records.  Eight is
#: enough to see whether the head decays smoothly or falls off a cliff
#: after one row, which is the whole question PR cannot answer.
TOP_K = 8

import torch


# ---------------------------------------------------------------------------
# geometry of a carried state
# ---------------------------------------------------------------------------

def rank_stats(mat: torch.Tensor) -> dict:
    """[B, K, D] -> centred cosine + two effective ranks, averaged over lanes.

    CENTRED FIRST, and that is the whole point of the statistic: K vectors
    sharing a large common component read as cosine ~1.0 however much
    independent structure sits on top of it, which is exactly the anisotropy
    that muddied the 0.97 reading on post-`ln_f` states.  After centring, the
    cosine measures DUPLICATION.

    Two effective ranks because they answer different questions and disagree
    informatively: `eff_rank_entropy` (exp of the spectral entropy) is sensitive
    to the whole tail, `eff_rank_pr` (participation ratio) is dominated by the
    top few directions.  B2's accum carry measures ~4 of 32 on the second.
    """
    m = mat.detach().float()
    cos, ent, pr, ent2, spec = [], [], [], [], []
    for b in range(m.shape[0]):
        c = m[b] - m[b].mean(0, keepdim=True)
        K = c.shape[0]
        if K < 2:
            cos.append(0.0)
            ent.append(1.0)
            pr.append(1.0)
            ent2.append(1.0)
            spec.append([1.0] + [0.0] * (TOP_K - 1))
            continue
        n = c.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        g = (c / n) @ (c / n).T
        off = ~torch.eye(K, dtype=torch.bool, device=g.device)
        cos.append(float(g[off].mean()))
        s = torch.linalg.svdvals(c).clamp_min(0)
        tot = float(s.sum())
        if tot <= 0:
            ent.append(1.0)
            pr.append(1.0)
            ent2.append(1.0)
            spec.append([0.0] * TOP_K)
            continue
        p = s / s.sum()
        nz = p[p > 0]
        ent.append(float(torch.exp(-(nz * nz.log()).sum())))
        s2 = s ** 2
        pr.append(float((s2.sum() ** 2) / (s2 ** 2).sum().clamp_min(1e-30)))
        # PR AND ENTROPY WERE NEVER ON THE SAME QUANTITY, which is why their
        # "disagreement in direction" was never evidence: PR is a participation
        # ratio over the EIGENVALUES s^2, while `eff_rank_entropy` above is the
        # spectral entropy of p ~ s.  So PR << entropy is generic.  The s^2
        # entropy is the comparable one; the s entropy stays because records
        # already written quote it, and silently changing a reported statistic
        # is how two instruments end up disagreeing about the same checkpoint.
        p2 = s2 / s2.sum().clamp_min(1e-30)
        nz2 = p2[p2 > 0]
        ent2.append(float(torch.exp(-(nz2 * nz2.log()).sum())))
        # The spectrum head itself, as SHARES of total variance.  This is what
        # decides the open question: a PR of ~2 over 256 rows is either one
        # outlier row dominating, in which case top1 is most of the mass and PR
        # is describing that row rather than the width, or it is genuine
        # concentration, in which case the shares decay smoothly.  No amount of
        # re-reading PR can tell those apart.
        head = p2[:TOP_K].tolist()
        spec.append(head + [0.0] * (TOP_K - len(head)))
    n = len(cos)
    mean_spec = [sum(row[i] for row in spec) / n for i in range(TOP_K)]
    return {"centred_cosine": sum(cos) / n,
            "eff_rank_entropy": sum(ent) / n,
            "eff_rank_pr": sum(pr) / n,
            "eff_rank_entropy_sq": sum(ent2) / n,
            "top1_share": mean_spec[0],
            "spectrum_top": mean_spec}


def split_carry(state: Optional[torch.Tensor], hidden_size: int):
    """[B, K, D] or [B, K, 2D] -> (E, Z or None), WITHOUT consulting a buffer.

    Deliberately independent of `_PrefixBufferBase.split_channels`: this is the
    instrument, that is the model.  A probe that asked the buffer to split the
    tensor could not report a width the buffer disagrees with, which is the one
    thing a checkpoint/config mismatch looks like.
    """
    if state is None:
        return None, None
    got = state.shape[-1]
    if got == hidden_size:
        return state, None
    if got == 2 * hidden_size:
        return state[..., :hidden_size], state[..., hidden_size:]
    raise ValueError(
        f"carry width {got} is neither D ({hidden_size}) nor 2D "
        f"({2 * hidden_size}) -- this is not a cortex carry")


def carry_health(state: Optional[torch.Tensor], hidden_size: int) -> dict:
    """Per-channel geometry of one carried state.

    Reported per channel and never pooled.  E rows are post-`ln_f` hidden states
    at norm ~171 on the B2 checkpoint; Z rows are trajectory deltas at ~0.35.  A
    single pooled norm over a 2D-wide tensor is a number about E with a rounding
    error named Z, and it would hide a dead Z channel completely.
    """
    if state is None:
        return {"rows": 0, "channels": 0}
    e, z = split_carry(state, hidden_size)
    out = {"rows": int(e.shape[1]), "channels": 2 if z is not None else 1,
           "e_row_norm": float(e.detach().float().norm(dim=-1).mean())}
    out.update({f"e_{k}": v for k, v in rank_stats(e).items()})
    if z is not None:
        zf = z.detach().float()
        out["z_row_norm"] = float(zf.norm(dim=-1).mean())
        # Rows a ring has not reached yet hold exactly zero in the Z half (see
        # PrefixGatedBuffer._slot_init_block) and the read falls back to noise
        # for them.  Counting them separates "Z is dead" from "Z has not
        # travelled that far round the ring yet" -- opposite diagnoses.
        out["z_zero_rows"] = int((zf.abs().sum(dim=-1) == 0).sum())
        out["z_over_e_norm"] = out["z_row_norm"] / max(out["e_row_norm"], 1e-12)
        out.update({f"z_{k}": v for k, v in rank_stats(z).items()})
    return out


# ---------------------------------------------------------------------------
# the gate's parameters
# ---------------------------------------------------------------------------

#: `gate_init="zero"` sets both projections to exactly zero, so any movement at
#: all proves a gradient reached them.  Relative, so the bar does not depend on
#: the tensor's scale.
GATE_MOVED_MIN = 1e-6


def gate_param_health(buf, init_ref: Optional[dict] = None) -> dict:
    """What the gate's PARAMETERS say, with no forward pass.

    The forward-pass quantity (the across-input spread of fg) is what
    `evals/eval_gate_prereg.py` scores and it is strictly better evidence.  This
    is the cheap companion that can run every N training steps: at
    gate_init="zero" both projections are exactly zero, so `*_w_norm > 0` is
    proof the gradient path exists, and it costs nothing to compute.

    `forget_bias` is reported as the sigmoid it becomes, because that is the
    quantity the retention model and the pre-registered BABILong bars are
    written in (F(d) = ig * fg ** floor(d / lap)).
    """
    out = {}
    if not hasattr(buf, "gate_proj_in"):
        return out
    out["fg_at_bias"] = float(torch.sigmoid(buf.forget_bias.detach()).mean())
    out["ig_at_bias"] = float(torch.sigmoid(buf.input_bias.detach()).mean())
    out["gate_in_w_norm"] = float(buf.gate_proj_in.weight.detach().float().norm())
    out["gate_mem_w_norm"] = float(buf.gate_proj_mem.weight.detach().float().norm())
    out["gate_left_init"] = bool(
        out["gate_in_w_norm"] > GATE_MOVED_MIN
        or out["gate_mem_w_norm"] > GATE_MOVED_MIN)
    if getattr(buf, "carries_latent", False) and hasattr(buf, "gate_proj_in_z"):
        out["fg_z_at_bias"] = float(torch.sigmoid(buf.forget_bias_z.detach()).mean())
        out["ig_z_at_bias"] = float(torch.sigmoid(buf.input_bias_z.detach()).mean())
        out["gate_in_z_w_norm"] = float(
            buf.gate_proj_in_z.weight.detach().float().norm())
        out["gate_mem_z_w_norm"] = float(
            buf.gate_proj_mem_z.weight.detach().float().norm())
        out["gate_z_left_init"] = bool(
            out["gate_in_z_w_norm"] > GATE_MOVED_MIN
            or out["gate_mem_z_w_norm"] > GATE_MOVED_MIN)
    out["summary_emb_norm"] = float(buf.summary_emb.detach().float().norm())
    if init_ref:
        for k, v in list(out.items()):
            if k.endswith("_norm") and k in init_ref:
                out[f"d_{k}"] = v - float(init_ref[k])
    return out


def grad_norms(module, prefix: str = "") -> dict:
    """Per-parameter grad norms, None-safe.

    A parameter with `grad is None` after a backward is not a small gradient, it
    is NO GRADIENT PATH, and the two have to be distinguishable in the table --
    so it is reported as None rather than 0.0.
    """
    out = {}
    for name, p in module.named_parameters():
        if p.grad is None:
            out[prefix + name] = None
        else:
            out[prefix + name] = float(p.grad.detach().float().norm())
    return out


# ---------------------------------------------------------------------------
# the Z channel's runtime counters
# ---------------------------------------------------------------------------

def latent_runtime(cortex) -> dict:
    """The Z channel's per-forward counters, straight off the live graft.

    `read_measured` is not decoration.  `latent_read_grad_frac` returns 0.0 both
    when the read is dead and when the modeling file never reported the
    no-grad split (an older snapshot inside a prepared checkpoint), and those
    call for opposite actions -- re-prepare the checkpoint, or write the number
    into the pre-registration.
    """
    if cortex is None or not getattr(cortex, "latent_carry", False):
        return {"latent_carry": False}
    return {
        "latent_carry": True,
        "read_measured": bool(cortex.latent_read_measured),
        "read_grad_frac": float(cortex.latent_read_grad_frac),
        "write_grad_frac": float(cortex.latent_write_grad_frac),
        "tape_len": len(getattr(cortex, "_z_tape", [])),
        "n_pre": int(getattr(cortex, "_n_pre", 0)),
        "n_sum": int(getattr(cortex, "_n_sum", 0)),
        "s0_scale": (float(cortex._z_s0_scale)
                     if getattr(cortex, "_z_s0_scale", None) is not None
                     else None),
        # P3.0.  `read_site` is what makes `read_grad_frac` interpretable: the
        # same 1.0 means "the sampler happened to draw no no-grad prefix all
        # run" at s0 and "live by construction" in-loop, and only this field
        # distinguishes them.
        "read_site": (("s0+" if getattr(cortex, "latent_s0_read", True) else "")
                      + str(getattr(cortex, "latent_read", "none"))),
        "read_depth": str(getattr(cortex, "latent_read_depth", "none")),
        # The read gate, reported per step exactly as `forget_bias` is for the
        # E ring -- because the pre-registration fixes its reading IN ADVANCE:
        # a gate that collapses toward zero over training is the model saying
        # it does not want the read, and that is a legitimate negative.
        "read_gate": (float(cortex.latent_reader.gate_value)
                      if getattr(cortex, "latent_reader", None) is not None
                      else None),
        "read_calls": int(getattr(cortex, "_z_inloop_n", 0)),
        "matched_rows": getattr(cortex, "_z_matched_rows", None),
    }


#: MEASURED 2026-09-16 from the real sampler, 50,000 draws per cell, five seeds
#: at the operating point (0.5456 / 0.5446 / 0.5471 / 0.5438 / 0.5413).  Keyed
#: (mean_recurrence, mean_backprop_depth) -> share of training batches whose Z
#: READ has any gradient path at all, i.e. whose sampled num_steps_no_grad == 0.
#:
#: Pinned here, and cross-checked in tests/test_training_diag.py, because
#: p11_z_probe_prereg.md quotes these numbers and a pre-registration whose
#: figures have quietly drifted from the sampler is worse than none.
#:
#: THE FINDING IN THE TABLE: ~0.55 is a CEILING, not a tuning point.  Whenever
#: mean_backprop_depth >= mean_recurrence the fraction sits at 0.546/0.570/0.582
#: (depth 8/16/32) and never approaches 1, because with t = max(mr - s, 0) = 0
#: the Poisson rate is centred on s and P(Poisson(s) <= s-1) ~ 0.5 for any s.
#: So raising mean_backprop_depth past mean_recurrence buys ~3 points at a real
#: cost in graph memory.  The lever that moves it is mean_recurrence: mr8 gives
#: 0.545, mr32 gives 0.015 -- at which point Z is a frozen feature extractor.
MEASURED_READ_LIVE = {
    (8, 8): 0.5445,
    (8, 4): 0.1670,
    (8, 16): 0.5697,
    (16, 8): 0.1541,
    (32, 8): 0.0152,
    (32, 16): 0.1428,
    (32, 32): 0.5821,
}


def read_live_fraction(model, n_samples: int = 4000,
                       mean_recurrence: Optional[int] = None,
                       mean_backprop_depth: Optional[int] = None,
                       seed: int = 0) -> dict:
    """MEASURE the share of training batches whose Z read has a gradient path.

    "Roughly half at mr8 / depth 8" is arithmetic off the Poisson tail, not a
    measurement, and it is a number that goes into the Z pre-registration --
    because it changes what a null result for Z means.  So sample the model's
    OWN `randomized_iteration_sampler` rather than reimplementing it: a mirror
    of the sampler is one more thing that can drift out of step with the file it
    mirrors.

    n == 0 is the condition.  The no-grad iterations run FIRST, and a single one
    of them detaches everything downstream of `s0`; Z enters once, at `s0`, so
    one no-grad step cuts its read gradient to exactly zero.  E is unaffected --
    it re-enters through `input_embeds` on every iteration.

    The sampler is only random in TRAINING mode (in eval it returns
    (mean_recurrence, 0), which would report 100% and mean nothing), so this
    forces train mode and restores whatever was there.
    """
    sampler = getattr(model, "randomized_iteration_sampler", None)
    if sampler is None:
        raise ValueError("model has no randomized_iteration_sampler")
    cfg = model.config
    was_training = model.training
    old = (cfg.mean_recurrence, cfg.mean_backprop_depth)
    want_mr = int(mean_recurrence) if mean_recurrence is not None else int(old[0])
    want_bd = (int(mean_backprop_depth) if mean_backprop_depth is not None
               else int(old[1]))
    g_state = torch.random.get_rng_state()
    try:
        cfg.mean_recurrence, cfg.mean_backprop_depth = want_mr, want_bd
        model.train()
        torch.manual_seed(seed)
        live = 0
        n_hist, k_hist = [], []
        for _ in range(n_samples):
            n, k = sampler()
            n_i, k_i = int(n), int(k)
            live += int(n_i == 0)
            n_hist.append(n_i)
            k_hist.append(k_i)
    finally:
        torch.random.set_rng_state(g_state)
        cfg.mean_recurrence, cfg.mean_backprop_depth = old
        model.train(was_training)
    nt = torch.tensor(n_hist, dtype=torch.float)
    kt = torch.tensor(k_hist, dtype=torch.float)
    return {
        "samples": n_samples,
        "mean_recurrence": want_mr,
        "mean_backprop_depth": want_bd,
        "read_live_frac": live / max(n_samples, 1),
        "no_grad_mean": float(nt.mean()),
        "no_grad_p50": float(nt.median()),
        "with_grad_mean": float(kt.mean()),
        "total_steps_mean": float((nt + kt).mean()),
    }


# ---------------------------------------------------------------------------
# the periodic training-time diagnostic
# ---------------------------------------------------------------------------

def training_diag(cortex, carry: Optional[torch.Tensor] = None) -> dict:
    """One flat row of scalars: what the architecture is doing at this step.

    WHY THE TRAINING LOOP NEEDS THIS AT ALL.  Every quantity that decides
    whether a cortex-final arm is worth finishing -- did the gate leave its
    exactly-zero init, is the carry collapsing towards rank 4, is the Z channel
    attached to the loss, what fraction of batches carry a Z read gradient --
    moves over the FIRST FEW HUNDRED STEPS and then is settled.  Reading them
    off the final checkpoint answers the question too late and cannot show a
    trajectory; reading them off the loss curve does not answer it at all.

    Flat, scalar and JSON-safe on purpose: the same row goes to wandb (where it
    is a line on a chart) and to the run's jsonl (where tools/compare_arms.py
    reads it back).  Booleans become 0/1 so wandb will plot them.

    Costs one SVD of a [K, D] matrix per call, which is why it is behind an
    interval and not on every step.
    """
    if cortex is None or getattr(cortex, "prefix", None) is None:
        return {}
    buf = cortex.prefix
    row = {"rows": 0}
    if carry is not None:
        row.update(carry_health(carry.detach(), buf.hidden_size))
    row.update(gate_param_health(buf))
    lat = latent_runtime(cortex)
    row.update({f"z_{k}": v for k, v in lat.items() if k != "latent_carry"})
    row["latent_carry"] = lat.get("latent_carry", False)
    out = {}
    for k, v in row.items():
        if isinstance(v, bool):
            out[k] = int(v)
        elif isinstance(v, (int, float)) and v == v:      # drop NaN
            out[k] = v
    return out


# ---------------------------------------------------------------------------
# the ring's addressing, derived independently of the buffer
# ---------------------------------------------------------------------------

def expected_ring_rows(chunk_index: int, n_vec: int, n_slots: int) -> list:
    """Rows chunk `chunk_index` (0-based, counting merges the gate ran) writes.

    Derived here from the RULE so the walk can check the buffer against it.
    Asking the buffer which rows it wrote and then checking that the buffer
    wrote them is not a test.
    """
    return [((chunk_index * n_vec) % n_slots + j) % n_slots
            for j in range(n_vec)]


def chance_margin(losses, vocab_size: int) -> dict:
    """Is this probe scoring a MODEL, or is it scoring noise?

    Returns the mean NLL, the chance level ln(vocab) and the margin between
    them.  `at_chance` is the veto: a probe within 0.25 nats of ln(vocab) is
    measuring nothing, and every delta computed from it -- carry vs none, donor
    vs real, parent vs branch -- is a difference of two noise levels.

    Added after RED 10, where the walk and gate 4 ran at 11.4-11.97 against
    ln(100352) = 11.5157 and nothing noticed, because no instrument on that
    path ever checked the LEVEL.  `evals/eval_influence_horizon.intact_health`
    is the same check on the clean path; keep the two in step.
    """
    vals = [float(x) for x in losses if x is not None]
    if not vals or not vocab_size:
        return {"mean_nll": None, "chance": None, "margin": None,
                "at_chance": None}
    chance = math.log(float(vocab_size))
    mean = sum(vals) / len(vals)
    return {"mean_nll": mean, "chance": chance, "margin": chance - mean,
            "at_chance": bool(chance - mean < 0.25)}
