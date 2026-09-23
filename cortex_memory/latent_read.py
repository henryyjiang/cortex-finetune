"""The in-loop Z read (P3.0).

`p30_readinto_prereg.md` is the pre-registration; this file is what it §2 asks
for.  The one-paragraph version of why it exists:

THE s0 READ SITE IS DEAD, MEASURED.  `pace/diag_s0.sbatch` (job 13297293, read
through RED 11's units fix) says deleting s0's carried columns outright costs
+4.1e-5 nats on a2 and -2.5e-5 on a3z -- opposite signs, i.e. noise -- and the
site stays flat out to 45x ||s0||, turning on only at an injected norm of 175.8
against ||E|| = 171.0, which is the substitution drowning E inside
`adapter(cat([x, input_embeds]))` rather than being read.  A site whose content
can be deleted for free is not a read site.  On top of that, Z enters ONCE at
s0 and `iterate_forward` runs its no-grad iterations FIRST, so a single no-grad
step cuts Z's read gradient to exactly 0.0 (measured: 6.1e-2 at split (0,T),
0.0 at (1,T-1), (2,T-2), (3,T-3), against E's ~4e-2 at every split).

`CortexGraft.read_into` fixes both: it fires after the adapter and before the
core blocks on EVERY iteration of both loops, so `num_steps_with_grad >= 1`
makes the read gradient live by construction, and it injects against ||x|| ~ 10
with Z rows of ~1-5 in the [2,9] band instead of ||s0|| = 0.39 against ||E|| =
171.

TWO MODULES, AND THE CHEAP ONE IS NOT A WARM-UP.  `LatentRefresh` (Option 0)
isolates REPETITION from LEARNED EXTRACTION: one scalar, re-adding Z into the
carried columns every iteration, which tests the one mechanism the s0 sweep
cannot rule out -- that s0's carried columns are washed because the loop
overwrites them at iteration 1, not because the site has no gain.
`LatentRead` (Option 1) is the cross-attention module that asks whether an
in-loop read can extract anything at all.  Full width first: a null from the
cheapest variant says nothing about the richest, which is the same reasoning
that put W=16 and not W=1 in the buffer.  Shrink after, not before.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def written_row_mask(z: torch.Tensor) -> torch.Tensor:
    """[B, K, D] -> [B, K] bool, True where the row has ever been written.

    Rows a gated ring has not reached yet are EXACTLY zero
    (`PrefixGatedBuffer._slot_init_block`), and a zero row is not a neutral
    key: it still scores a mid-range logit and takes softmax mass by count.
    That is the same objection this codebase already makes about E's zero null
    in `eval_carry_2x2.py`, and it bites harder here because `fill='grow'`
    means the early chunks of every sequence are mostly unwritten.  So
    unwritten rows are MASKED OUT of the attention rather than attended to.
    """
    return z.detach().abs().sum(dim=-1) != 0


class LatentRefresh(nn.Module):
    """Option 0 — the parameter-free in-loop refresh, one learned scalar.

    `x[:, :n_pre] += alpha * Z`, every iteration.  Parameter-free in the sense
    that matters (no projection, nothing to bootstrap from zero); the single
    scalar exists so the model can say how much it wants, and so that
    `alpha -> 0` over training is a readable negative rather than an
    unfalsifiable "it was never on".

    Only the carried columns are touched, which is why this takes no EOS mask:
    the carried block sits BEFORE every real token, so it is before the first
    EOS by construction and `compute_eos_masks` would mark all of it live.
    """

    def __init__(self, hidden_size: int, alpha_init: float = 0.1) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.alpha_init = float(alpha_init)
        self.alpha = nn.Parameter(torch.full((1,), float(alpha_init)))

    @property
    def gate_value(self) -> float:
        """The reported number, same role `forget_bias` plays for the E ring."""
        return float(self.alpha.detach().float().reshape(-1)[0])

    def forward(self, x: torch.Tensor, z: torch.Tensor,
                n_pre: int) -> torch.Tensor:
        if n_pre <= 0:
            return x
        if z.shape[1] != n_pre:
            raise ValueError(
                f"LatentRefresh got {z.shape[1]} Z rows for {n_pre} carried "
                "columns.  The refresh is a per-column re-add and needs them "
                "to correspond; a mismatch means the Z rows were selected "
                "(depth-matched) before being handed here, which is a mode "
                "combination that does not make sense -- matching selects a "
                "SUBSET of rows and the refresh addresses columns.")
        written = written_row_mask(z).unsqueeze(-1).to(x.dtype)
        delta = self.alpha.to(x.dtype) * z.to(x.dtype) * written
        return torch.cat([x[:, :n_pre] + delta, x[:, n_pre:]], dim=1)


class LatentRead(nn.Module):
    """Option 1 — cross-attention from the loop state into the carried Z rows.

    Queries from post-adapter `x` (every packed position), keys/values from the
    carried Z, additive through `out_proj` and a learned scalar gate.
    4 x D^2 = 16.8M parameters at D = 2048 -- report it against the TRAINABLE
    set, never against the model size, and note that a3z already carries
    ~16.8M of Z-side gate on the WRITE path.  The read module is what would
    make those earn out.

    DO NOT ZERO-INIT `out_proj`.  That is the x0.90 shape: `LSTMBuffer` zeroed
    it, the read became a literal no-op that had to be discovered from nothing,
    and B2 later measured the cost of a severed read at x0.90 -> x1.33.  What
    was actually wrong there was a zero-init read whose GRADIENT PATH WAS
    SEVERED (`freeze_loop=true`), and `read_into` is the site where it is not.
    So: small nonzero init, plus a scalar gate at a small positive value, so
    the read is live from step 0 and bounded.  The gate is a reported number
    per step, and the pre-registration fixes its reading in advance -- if it
    collapses toward zero over training, the model is telling us it does not
    want the read, and that is a legitimate negative result.
    """

    def __init__(self, hidden_size: int, n_heads: int = 8,
                 gate_init: float = 0.1, proj_std: float = 0.02,
                 gate_lr_mult: float = 1.0) -> None:
        super().__init__()
        if hidden_size % n_heads:
            raise ValueError(
                f"LatentRead hidden_size ({hidden_size}) must divide by "
                f"n_heads ({n_heads}).")
        if not (gate_lr_mult > 0 and math.isfinite(gate_lr_mult)):
            raise ValueError(
                f"LatentRead gate_lr_mult must be finite and > 0; got "
                f"{gate_lr_mult!r}.  To freeze the gate, use a small value -- "
                "zero would divide the stored parameter by zero.")
        self.hidden_size = int(hidden_size)
        self.n_heads = int(n_heads)
        self.head_dim = self.hidden_size // self.n_heads
        self.gate_init = float(gate_init)
        # THE GATE'S LEARNING RATE, AS A REPARAMETERISATION.  The effective
        # gate is `gate_lr_mult * self.gate`, with `self.gate` stored at
        # gate_init / gate_lr_mult.  Adam's step on a parameter is ~lr whatever
        # the gradient's magnitude, so scaling the parameterisation by c scales
        # the gate's per-step movement by c: this IS an LR multiplier on the
        # one scalar, under the optimizer every cortex parameter uses.
        #
        # Why it exists (J1, j1_prereg.md S10): the gate fell 0.10 -> 0.00 in
        # ~270 updates on every read limb, LINEARLY, at ~3.9e-4 per update --
        # i.e. Adam at memory_lr 5e-4 stepping at full speed on a sign-stable
        # gradient.  The collapse speed was the optimizer's, not a measure of
        # how much the read hurt, and it shut the write's gradient (which
        # scales with the gate) before the write could learn.
        #
        # Why not a param group with its own LR: a branch restores the
        # parent's LambdaLR, which carries one lr_lambda per group, and
        # load_checkpoint refuses a group-count mismatch.  A new group would
        # force --ignore_past_scheduler, i.e. a different LR schedule -- a
        # second variable.  1.0 (the default) is bit-identical to every arm
        # on record.  It MUST persist into config.json: a checkpoint rebuilt
        # at a different value reads the stored parameter at the wrong scale.
        self.gate_lr_mult = float(gate_lr_mult)

        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        for proj in (self.q_proj, self.k_proj, self.v_proj, self.out_proj):
            nn.init.normal_(proj.weight, std=proj_std)
        self.gate = nn.Parameter(
            torch.full((1,), float(gate_init) / self.gate_lr_mult))
        # J3 diagnostic: mean attention mass on the position's OWN scratchpad
        # key, set only on the `own` path.  Detached, last call wins.
        self._own_share_t: Optional[torch.Tensor] = None

    @property
    def own_share(self) -> Optional[float]:
        """Share of the read's attention on the own scratchpad key (J3), 1.0
        when no carried row is live.  None until the `own` path has run."""
        t = self._own_share_t
        return None if t is None else float(t)

    def effective_gate(self) -> torch.Tensor:
        """The gate the forward multiplies by (see gate_lr_mult)."""
        if self.gate_lr_mult == 1.0:
            return self.gate
        return self.gate * self.gate_lr_mult

    @property
    def gate_value(self) -> float:
        return float(self.effective_gate().detach().float().reshape(-1)[0])

    def forward(self, x: torch.Tensor, z: Optional[torch.Tensor],
                read_mask: Optional[torch.Tensor] = None,
                own: Optional[torch.Tensor] = None) -> torch.Tensor:
        """x [B,S,D] queries, z [B,K,D] keys/values -> delta [B,S,D].

        `read_mask` is [B,S,1] over the PACKED layout: zero at positions that
        must not read the previous document's carry.  Returns the delta rather
        than x+delta so the caller owns the residual and a test can assert the
        contribution is exactly zero.

        `own` [B,S,D] is J3's per-position scratchpad: ONE extra key per query
        position, that position's own row, sharing the softmax with the carried
        rows.  With `own=None` this is exactly the J1 read (the branch below is
        untouched); see `_forward_with_own` for how the mask changes meaning.
        """
        if own is not None:
            return self._forward_with_own(x, z, read_mask, own)
        B, S, D = x.shape
        K = z.shape[1]
        if z.shape[0] != B or z.shape[-1] != D:
            raise ValueError(
                f"LatentRead got x {tuple(x.shape)} and z {tuple(z.shape)}; "
                "batch and hidden size must agree.")

        written = written_row_mask(z)                       # [B,K]
        if not bool(written.any()):
            # EVERY row unwritten -- chunk 1 of a sequence under fill='grow'.
            # Returning zeros explicitly rather than letting an all-masked
            # softmax produce NaN: a fully-masked row of `attn` would be
            # 0/0, and the resulting NaN would propagate into the loss on the
            # FIRST CHUNK of every sequence.  This is also gate 5 in the
            # pre-registration -- Z = 0 must inject exactly 0.
            return x.new_zeros(x.shape)

        z = z.to(dtype=x.dtype)
        nh, hd = self.n_heads, self.head_dim
        q = self.q_proj(x).view(B, S, nh, hd).transpose(1, 2)
        k = self.k_proj(z).view(B, K, nh, hd).transpose(1, 2)
        v = self.v_proj(z).view(B, K, nh, hd).transpose(1, 2)

        logits = (q @ k.transpose(-2, -1)) / math.sqrt(hd)   # [B,nh,S,K]
        logits = logits.masked_fill(
            ~written[:, None, None, :].expand_as(logits), float("-inf"))
        attn = F.softmax(logits, dim=-1)
        # A batch row with no written slots is impossible here (the all-zero
        # case returned above), but a MIXED batch can have one: B rows are
        # independent sequences and only some may be on chunk 1.  Those rows
        # are all -inf, so scrub their NaN instead of letting it travel.
        attn = torch.nan_to_num(attn, nan=0.0)
        out = (attn @ v).transpose(1, 2).contiguous().view(B, S, D)
        delta = self.effective_gate().to(x.dtype) * self.out_proj(out)
        if read_mask is not None:
            delta = delta * read_mask.to(delta.dtype)
        return delta

    def _forward_with_own(self, x: torch.Tensor, z: Optional[torch.Tensor],
                          read_mask: Optional[torch.Tensor],
                          own: torch.Tensor) -> torch.Tensor:
        """The J3 read: carried rows [B,K,D] (or None) plus one OWN key per
        query position [B,S,D], in one softmax.

        CAUSAL BY CONSTRUCTION.  Position i's extra key is position i's own
        scratchpad row, never another position's, so nothing inside the chunk
        crosses the causal mask through this read.  That is the whole reason
        the scratchpad is per-position: a pooled row written from the chunk's
        tokens and read back by those same tokens would hand every position
        the future.

        THE EOS MASK REMOVES CARRIED KEYS, NOT THE READ.  On the J1 path a
        masked position's whole delta is zeroed, because the carry was all it
        read.  Here a position past the first EOS must still read its OWN row
        (that is its own current document), so the mask sets the carried
        logits to -inf at those positions and leaves the own key alone.

        A position with no live key at all (own row exactly zero -- the first
        iteration -- and no carried rows it may read) gets exactly zero, via
        the same nan scrub as the J1 path.
        """
        B, S, D = x.shape
        if own.shape != x.shape:
            raise ValueError(
                f"LatentRead own rows {tuple(own.shape)} must match x "
                f"{tuple(x.shape)}: one scratchpad row per packed position.")
        nh, hd = self.n_heads, self.head_dim
        own = own.to(dtype=x.dtype)
        q = self.q_proj(x).view(B, S, nh, hd).transpose(1, 2)          # [B,nh,S,hd]
        ko = self.k_proj(own).view(B, S, nh, hd).transpose(1, 2)
        vo = self.v_proj(own).view(B, S, nh, hd).transpose(1, 2)
        own_logit = (q * ko).sum(-1, keepdim=True) / math.sqrt(hd)     # [B,nh,S,1]
        own_live = own.detach().abs().sum(dim=-1) != 0                 # [B,S]
        own_logit = own_logit.masked_fill(
            ~own_live[:, None, :, None].expand_as(own_logit), float("-inf"))

        K = 0
        if z is not None:
            written = written_row_mask(z)                              # [B,K]
            if bool(written.any()):
                if z.shape[0] != B or z.shape[-1] != D:
                    raise ValueError(
                        f"LatentRead got x {tuple(x.shape)} and z "
                        f"{tuple(z.shape)}; batch and hidden size must agree.")
                K = z.shape[1]
                z = z.to(dtype=x.dtype)
                kc = self.k_proj(z).view(B, K, nh, hd).transpose(1, 2)
                vc = self.v_proj(z).view(B, K, nh, hd).transpose(1, 2)
                lc = (q @ kc.transpose(-2, -1)) / math.sqrt(hd)        # [B,nh,S,K]
                keep = written[:, None, None, :].expand_as(lc)
                if read_mask is not None:
                    pos_ok = read_mask[..., 0].to(torch.bool)          # [B,S]
                    keep = keep & pos_ok[:, None, :, None]
                lc = lc.masked_fill(~keep, float("-inf"))
                logits = torch.cat([lc, own_logit], dim=-1)
            else:
                logits = own_logit
        else:
            logits = own_logit
        attn = torch.nan_to_num(F.softmax(logits, dim=-1), nan=0.0)
        out = attn[..., K:] * vo                                       # own share
        if K:
            out = out + attn[..., :K] @ vc
        # Kept as a tensor: float() here would sync the GPU on every loop
        # iteration of every chunk.  `own_share` converts on demand (diag).
        self._own_share_t = attn[..., K:].detach().float().mean()
        out = out.transpose(1, 2).contiguous().view(B, S, D)
        return self.effective_gate().to(x.dtype) * self.out_proj(out)


def apply_designed_init(reader: Optional[nn.Module]) -> list:
    """Re-apply the read module's hand-picked init.  Returns log tags.

    EXISTS BECAUSE OF BUG CLASS 1.  `RavenForCausalLM.__init__` builds the
    graft and THEN calls `post_init()`, which treats every cortex tensor as a
    freshly-missing key and re-initialises it with raven's DEPTH-SCALED scheme;
    the graft modules have no valid layer index, so that scheme hands them an
    effectively-infinite std.  `reset_cortex_graft_init` undoes it by calling
    `reset_parameters()` on every submodule -- which puts KAIMING back into
    q/k/v/out and silently discards the init this module's docstring argues
    for, exactly as it did to `gate_init='zero'` on the ring before
    `apply_gate_init` existed.  The module owns its init; this is the hook.
    """
    if reader is None:
        return []
    if isinstance(reader, LatentRefresh):
        nn.init.constant_(reader.alpha, reader.alpha_init)
        return [f"latent_reader.alpha={reader.alpha_init}"]
    if isinstance(reader, LatentRead):
        for proj in (reader.q_proj, reader.k_proj, reader.v_proj,
                     reader.out_proj):
            nn.init.normal_(proj.weight, std=0.02)
        # The STORED parameter is gate_init / gate_lr_mult; the effective gate
        # is gate_init either way.  Re-initialising the stored value to
        # gate_init would start a gate_lr_mult=0.01 arm at 100x its gate.
        nn.init.constant_(reader.gate, reader.gate_init / reader.gate_lr_mult)
        tags = ["latent_reader.[qkv,out~N(0,0.02) NOT zero -- see LatentRead]",
                f"latent_reader.gate={reader.gate_init}"]
        if reader.gate_lr_mult != 1.0:
            tags.append(f"latent_reader.gate_lr_mult={reader.gate_lr_mult}")
        return tags
    raise TypeError(f"unknown latent read module {type(reader).__name__}")
