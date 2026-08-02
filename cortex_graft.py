"""
cortex_graft — wires cortex_memory into the raven (RavenForCausalLM) model.

Design goals
------------
* Flag-gated, default OFF.  With use_memory=False (the default) the host model
  is byte-for-byte unchanged: RavenForCausalLM.__init__ creates `self.cortex =
  None` and every hook is a guarded no-op.
* No dependency on a modified RavenConfig.  All settings are read from the
  config via getattr with safe defaults, so an unmodified checkpoint config.json
  works.  Enable memory either by editing config.json or by passing flags to
  from_pretrained(..., use_memory=True, memory_slots=4, ...).
* All per-call runtime (the carried buffer, EOS masks, the per-position M_iter
  buffer) lives on the CortexMemory instance and is reset at the start of every
  forward, so core_block_forward / iterate_forward keep their original
  signatures (important: they are also used by generation).

Config flags (getattr defaults)
-------------------------------
  use_memory          : bool = False   master switch
  memory_slots        : int  = 0       K for M_cross (LM2 buffer); 0 disables
  memory_slots_iter   : int  = 0       K for M_iter (per-position); 0 disables
  memory_heads        : int  = 4       attention heads in both buffers
  ccot_direct         : bool = False   K=0 Coconut carry (only when memory_slots==0)
  accum_ccot          : bool = False   AutoCompressor-style accumulating carry
                                       (only when memory_slots==0, replaces
                                       ccot_direct's single overwritten vector)
  accum_vecs          : int  = 4       summary vectors extracted per chunk
  accum_max           : int  = 64      FIFO cap on accumulated vectors (eval)
  gated_accum         : bool = False   gated-accumulation LM2 variant: the K-slot
                                       M_cross becomes a GatedAccumBuffer —
                                       AccumCCoT's extraction write, LM2 gated
                                       merge (requires memory_slots > 0; target
                                       k=16/32).  h_T_proj is skipped for it.
  h_T_proj            : bool = True     R4 mitigation projection before M_cross write
  lora_rank           : int  = 0       LoRA-on-loop rank (0 disables; see LoopLoRA)
  lora_alpha          : float = 32     LoRA scaling numerator (scale = alpha/rank)

Hook points in RavenForCausalLM (see the grafted model files):
  forward()           : cortex.begin(...) before iterate_forward;
                        new_m_cross = cortex.cross_write(h_T) after it;
                        m_cross surfaced in the output.
  core_block_forward(): x = cortex.read_into(x) after the adapter, before the
                        core layers; cortex.iter_write(x) after the core layers.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn

from cortex_memory.buffers import LSTMBuffer, PrefixAccumBuffer, PrefixGatedBuffer
from cortex_memory.eos import compute_eos_masks, apply_write_reset, apply_valid_write


def memory_enabled(config) -> bool:
    """Master switch — read once in RavenForCausalLM.__init__."""
    return bool(getattr(config, "use_memory", False))


def resolve_summary_init_token(config) -> int:
    """Which token's embedding seeds the prefix summary slots.

    AutoCompressor uses EOS (auto_compressor.py:49) — a real, in-distribution
    vector the pretrained model already knows how to process.  RavenConfig
    leaves eos_token_id commented out (raven_config_minimal.py:96), so on a
    converted checkpoint it is whatever the source config carried, and on a
    hand-built config it is None.  Resolve explicitly and fail loudly rather
    than silently seeding from token 0 or from noise: a badly seeded summary
    slot degrades the carry without producing any visible symptom in the loss
    curve, which is the failure mode this project keeps paying for.

    Order: cortex.summary_init_token (>= 0) wins, else config.eos_token_id
    (first entry if it is a list, as some configs carry several).
    """
    explicit = int(getattr(config, "summary_init_token", -1) or -1)
    if explicit >= 0:
        tok = explicit
    else:
        eos = getattr(config, "eos_token_id", None)
        if isinstance(eos, (list, tuple)):
            eos = eos[0] if eos else None
        if eos is None:
            raise ValueError(
                "prefix memory needs a token to seed its summary embeddings, but "
                "the config has no eos_token_id.  Pass "
                "--cortex.summary_init_token <id> (AutoCompressor uses EOS)."
            )
        tok = int(eos)
    vocab = int(getattr(config, "vocab_size", 0) or 0)
    if vocab and not (0 <= tok < vocab):
        raise ValueError(
            f"summary init token {tok} is outside the vocabulary (size {vocab})"
        )
    return tok


# ── LoRA-on-loop (experiment-ladder rung 1b) ────────────────────────────────
#
# Low-rank adapters on every nn.Linear inside the recurrent loop (adapter +
# core_block) so the loop can ADAPT to the memory's presence without unfreezing
# the pretrained weights: out = Wx + (alpha/r) * B(Ax), base W frozen, B
# zero-init -> exact no-op at step 0 (step-0 == base model, like the memory
# read).  Config-driven from __init__ so save_pretrained / from_pretrained /
# resume all rebuild the hooks and load A/B automatically.

def _loop_linears(model):
    """Yield (name, module) for every nn.Linear under the loop (adapter +
    core_block).  Falls back to direct attributes for test doubles that lack
    the transformer ModuleDict."""
    tr = getattr(model, "transformer", model)
    for root_name in ("adapter", "core_block"):
        root = getattr(tr, root_name, None)
        if root is None:
            continue
        if isinstance(root, nn.Linear):
            yield root_name, root
        else:
            for n, m in root.named_modules():
                if isinstance(m, nn.Linear):
                    yield f"{root_name}.{n}", m


class LoopLoRA(nn.Module):
    """Holds the A/B parameters and installs additive forward hooks on the
    loop linears.  Param keys replace 'adapter'->'adpt' and 'core_block'->'loop'
    so train.py's set_loop_trainable() (which freezes by those substrings)
    leaves the LoRA parameters trainable; the keys still contain 'cortex' via
    the module name, routing them to the Adam side / memory-LR group."""

    def __init__(self, model, config) -> None:
        super().__init__()
        r     = int(getattr(config, "lora_rank", 0))
        alpha = float(getattr(config, "lora_alpha", 32))
        assert r > 0
        self.rank  = r
        self.scale = alpha / r
        self.A = nn.ParameterDict()
        self.B = nn.ParameterDict()
        self._handles = []
        for name, lin in _loop_linears(model):
            key = (name.replace("core_block", "loop").replace("adapter", "adpt")
                       .replace(".", "_"))
            A = nn.Parameter(torch.empty(r, lin.in_features))
            nn.init.kaiming_uniform_(A, a=math.sqrt(5))   # standard LoRA init
            B = nn.Parameter(torch.zeros(lin.out_features, r))
            self.A[key] = A
            self.B[key] = B
            self._handles.append(lin.register_forward_hook(self._make_hook(key)))

    def _make_hook(self, key: str):
        def hook(_mod, inputs, output):
            x = inputs[0]
            return output + (x @ self.A[key].t() @ self.B[key].t()) * self.scale
        return hook

    def remove(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []


def build_loop_lora(model, config) -> Optional[LoopLoRA]:
    """Called from the grafted RavenForCausalLM.__init__ (after the transformer
    is built).  Returns None unless use_memory and lora_rank > 0."""
    if not memory_enabled(config) or int(getattr(config, "lora_rank", 0)) <= 0:
        return None
    return LoopLoRA(model, config)


class CortexMemory(nn.Module):
    """Holds the memory modules + per-call runtime and exposes the four hooks
    (begin / read_into / iter_write / cross_write) used by the grafted model."""

    def __init__(self, config) -> None:
        super().__init__()
        D  = config.n_embd
        K  = int(getattr(config, "memory_slots", 0))
        Ki = int(getattr(config, "memory_slots_iter", 0))
        nh = int(getattr(config, "memory_heads", 4))
        self.memory_slots      = K
        self.memory_slots_iter = Ki

        # M_cross: the original LM2 K-slot buffer (K>0), kept as the
        # pre-AutoCompressor baseline arm.  DirectCCoT, AccumCCoT and
        # GatedAccumBuffer were removed 2026-08-02 — superseded by the Prefix*
        # pair below (see cortex_memory/buffers.py for why).
        # RETIRED FLAGS.  These selected buffers that no longer exist.  Failing
        # loudly matters more than usual here: silently ignoring one of them
        # yields a run with NO cross-segment memory that still logs a healthy
        # loss curve, i.e. a null result that looks like a real measurement.
        for dead, replacement in (("accum_ccot", "--cortex.prefix_memory accum"),
                                  ("gated_accum", "--cortex.prefix_memory gated"),
                                  ("ccot_direct", "(removed; no replacement)")):
            if bool(getattr(config, dead, False)):
                raise ValueError(
                    f"cortex.{dead} was retired on 2026-08-02 along with the buffer "
                    f"it selected. Use {replacement} instead."
                )

        self.m_cross = LSTMBuffer(D, K, nh) if K > 0 else None
        self.accum = None        # retained attrs: the hooks below still branch
        self.ccot_direct = None  # on them, and eval/test code reads them
        # M_iter: per-position short-term buffer (independent of M_cross).
        self.m_iter = LSTMBuffer(D, Ki, nh) if Ki > 0 else None

        # ── AutoCompressor-faithful prefix memory ───────────────────────────
        # --cortex.prefix_memory accum|gated selects it and DISABLES every
        # path above: there is no read module and no extraction attention, so
        # read_into/cross_write are bypassed entirely.  The carry is spliced
        # into the token stream by the modeling file (prefix_pack) and read
        # back out of the post-ln_f states (prefix_unpack).
        pmode = str(getattr(config, "prefix_memory", "") or "").lower()
        n_vec = int(getattr(config, "accum_vecs", 32))
        if pmode == "accum":
            self.prefix = PrefixAccumBuffer(
                D, n_vec, int(getattr(config, "accum_max", 128)))
        elif pmode == "gated":
            self.prefix = PrefixGatedBuffer(D, n_vec)
        elif pmode:
            raise ValueError(f"prefix_memory must be '', 'accum' or 'gated'; got {pmode!r}")
        else:
            self.prefix = None
        if self.prefix is not None:
            # Prefix mode owns the cross-segment state exclusively.
            self.m_cross = self.accum = self.ccot_direct = None
            # Resolve now, at build time, so a missing or invalid seed token fails
            # before the data loader and the wandb run exist rather than on the
            # first forward of a queued 48h job.
            self.summary_init_token = resolve_summary_init_token(config)
        else:
            self.summary_init_token = -1

        # R4 dual-role mitigation: project h_T before the M_cross write so the
        # buffer path and the coda path see independent representations.
        # Identity-init → no-op at step 0.  LSTMBuffer mode only — the
        # GatedAccumBuffer's extraction wk/wv already decouple the write path
        # from the coda path (same reason AccumCCoT takes raw h_T).
        if isinstance(self.m_cross, LSTMBuffer) and bool(getattr(config, "h_T_proj", True)):
            self.h_T_proj = nn.Linear(D, D, bias=False)
            nn.init.eye_(self.h_T_proj.weight)
            self.h_T_proj.weight._no_weight_decay = True
        else:
            self.h_T_proj = None

        self._reset_runtime()

    @property
    def has_cross_state(self) -> bool:
        return (self.m_cross is not None or self.ccot_direct is not None
                or self.accum is not None or self.prefix is not None)

    def init_summary_from_embedding(self, wte_weight: torch.Tensor) -> None:
        """Seed the summary embeddings from a real token (AutoCompressor uses
        EOS).  Called by the model AFTER the base weights are loaded — at
        __init__ time wte is still random, so doing it there would copy noise.
        The token id was resolved and validated in __init__."""
        if self.prefix is not None:
            self.prefix.init_from_token_embedding(wte_weight, self.summary_init_token)

    # ── prefix-memory hooks (called from the model's forward) ───────────────
    def prefix_pack(self, input_embeds: torch.Tensor,
                    position_ids: torch.Tensor,
                    emb_scale: float = 1.0):
        """Splice the carry into the token stream, AutoCompressor-style.

        Layout: [carried vectors | real tokens | summary slots].  Under the
        model's causal mask this reproduces auto_compressor.py:85 exactly —
        real tokens see every carried vector, and the summary slots at the end
        see the whole chunk.  The summary slots also see each other causally,
        which is the ONLY thing that stops n_vec identically-initialised slots
        from collapsing into copies of one vector.

        Positions: 0 for carried and summary slots (RoPE identity — our analog
        of AC's pad-position trick), 1..S for real tokens.  Every index stays
        inside the trained window even though the packed sequence is longer.

        emb_scale: the model multiplies wte(ids) by config.init_values
        ["embed_scale"] BEFORE this splice, so a summary slot seeded from
        wte[eos] would otherwise enter the network 1/emb_scale times smaller
        than the very token it was copied from — silently undoing the
        in-distribution initialisation.  Scaling the slots here makes
        `summary_emb == wte[eos]` enter exactly as an EOS token would.  The
        CARRIED vectors are deliberately NOT scaled: they are post-ln_f hidden
        states, not embeddings, and AutoCompressor feeds its summaries straight
        back into inputs_embeds unscaled.  For the converted OLMo checkpoints
        emb_scale is 1.0 anyway (convert_olmo.py:80 pins it), so this only
        matters for other bases and for toy configs.

        Returns (packed_embeds, packed_position_ids, n_prefix, n_summary).
        """
        if self.prefix is None:
            return input_embeds, position_ids, 0, 0

        B, S, _ = input_embeds.shape
        parts, n_pre = [], 0

        state = self._carried_state()
        if state is not None and state.shape[1] > 0:
            state = state.to(device=input_embeds.device, dtype=input_embeds.dtype)
            parts.append(state)
            n_pre = state.shape[1]
        parts.append(input_embeds)

        slots = self.prefix.summary_slots(B, input_embeds.dtype, input_embeds.device)
        if emb_scale != 1.0:
            slots = slots * emb_scale
        parts.append(slots)
        n_sum = slots.shape[1]

        pos = position_ids[:, :S] + 1
        pos = torch.cat([pos.new_zeros(pos.shape[0], n_pre), pos,
                         pos.new_zeros(pos.shape[0], n_sum)], dim=1)
        return torch.cat(parts, dim=1), pos, n_pre, n_sum

    def prefix_unpack(self, x: torch.Tensor, n_pre: int, n_sum: int):
        """Split the post-ln_f states back apart.

        The summary slots' final hidden states ARE the new carry — no
        extraction module, no tanh/LN bounding (auto_compressor.py:123).
        Returns (real_token_states, new_carry).
        """
        if self.prefix is None or n_sum == 0:
            return x, None
        end = x.shape[1] - n_sum
        real, new_vecs = x[:, n_pre:end], x[:, end:]
        if self._valid_write is not None:
            # Lane's open suffix is empty -> nothing to carry from this chunk.
            new_vecs = new_vecs * self._valid_write.view(-1, 1, 1).to(new_vecs.dtype)
        return real, self.prefix.merge(self._carried_state(), new_vecs)

    def _carried_state(self) -> Optional[torch.Tensor]:
        """Incoming carry with ended-document lanes zeroed (a finished doc's
        vectors must not leak into the next one)."""
        state = self._cross_buf
        if state is not None and self._write_reset is not None:
            state = state * (~self._write_reset).view(-1, 1, 1).to(state.dtype)
        return state

    # ── per-call runtime ────────────────────────────────────────────────────
    def _reset_runtime(self) -> None:
        self._cross_buf:       Optional[torch.Tensor] = None  # carried M_cross [B,K,D]/[B,1,D]/[B,N,D]
        self._cross_read_mask: Optional[torch.Tensor] = None  # [B,S,1]
        self._pool_mask:       Optional[torch.Tensor] = None  # [B,S] bool
        self._write_reset:     Optional[torch.Tensor] = None  # [B] bool
        self._valid_write:     Optional[torch.Tensor] = None  # [B] bool
        self._iter_buf:        Optional[torch.Tensor] = None  # [B*S,Ki,D]

    def begin(
        self,
        m_cross_in: Optional[torch.Tensor],
        eos_mask: Optional[torch.Tensor],
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        """Call once at the start of forward(), before iterate_forward."""
        self._reset_runtime()
        self._cross_buf = m_cross_in
        if eos_mask is not None and self.has_cross_state:
            crm, pool, reset, valid = compute_eos_masks(eos_mask, seq_len, device, dtype)
            self._cross_read_mask = crm
            self._pool_mask       = pool
            self._write_reset     = reset
            self._valid_write     = valid

    # ── hooks called inside core_block_forward ──────────────────────────────
    def read_into(self, x: torch.Tensor) -> torch.Tensor:
        """Additive memory reads, injected after the adapter and before the
        core layers (cortex first-layer injection).  Returns the updated x."""
        # M_cross / DirectCCoT / AccumCCoT cross-segment read (masked to the
        # continuing doc)
        if self.m_cross is not None and self._cross_buf is not None:
            delta = self.m_cross.read(x, self._cross_buf)
            x = x + (delta * self._cross_read_mask if self._cross_read_mask is not None else delta)
        elif self.accum is not None and self._cross_buf is not None \
                and self._cross_buf.shape[1] > 0:
            delta = self.accum.read(x, self._cross_buf)
            x = x + (delta * self._cross_read_mask if self._cross_read_mask is not None else delta)
        elif self.ccot_direct is not None and self._cross_buf is not None:
            delta = self.ccot_direct.read(self._cross_buf)            # [B,1,D] broadcast
            x = x + (delta * self._cross_read_mask if self._cross_read_mask is not None else delta)

        # M_iter per-position short-term read (zero at the first iteration)
        if self.m_iter is not None:
            B, S, D = x.shape
            if self._iter_buf is None:
                self._iter_buf = x.new_zeros(B * S, self.memory_slots_iter, D)
            x = x + self.m_iter.read(x.reshape(B * S, 1, D), self._iter_buf).reshape(B, S, D)
        return x

    def iter_write(self, x: torch.Tensor) -> None:
        """Write each position's state into its own M_iter slots, after the
        core layers (end of one loop iteration)."""
        if self.m_iter is None:
            return
        B, S, D = x.shape
        if self._iter_buf is None:
            self._iter_buf = x.new_zeros(B * S, self.memory_slots_iter, D)
        self._iter_buf = self.m_iter.write(x.reshape(B * S, 1, D), self._iter_buf)

    # ── hook called in forward() after iterate_forward ──────────────────────
    def cross_write(self, h_T: torch.Tensor) -> Optional[torch.Tensor]:
        """Write h_T into M_cross / DirectCCoT / AccumCCoT.  Returns the new
        buffer (to be carried into the next segment) or None when no
        cross-state is active."""
        B, S, D = h_T.shape
        new_m_cross: Optional[torch.Tensor] = None

        if self.accum is not None:
            # Accumulation changes the state's slot dim, so the generic
            # apply_write_reset/apply_valid_write (which assume old and new
            # buffers share a shape) don't apply — equivalent per-lane
            # semantics inline: reset lanes zero their OLD rows (the ended
            # document's vectors carry nothing; rows can't be dropped
            # per-lane without ragged shapes), invalid-write lanes zero the
            # NEWLY appended rows (empty open suffix → nothing to carry).
            state = self._cross_buf
            if state is not None and self._write_reset is not None:
                keep = (~self._write_reset).view(B, 1, 1).to(state.dtype)
                state = state * keep
            new_vecs = self.accum.extract(h_T, self._pool_mask)
            if self._valid_write is not None:
                new_vecs = new_vecs * self._valid_write.view(B, 1, 1).to(new_vecs.dtype)
            return self.accum.append(state, new_vecs)

        if self.m_cross is not None:
            h_T_w = self.h_T_proj(h_T) if self.h_T_proj is not None else h_T
            if self._cross_buf is None:
                write_in = h_T.new_zeros(B, self.memory_slots, D)
            elif self._write_reset is not None:
                write_in = apply_write_reset(self._cross_buf, self._write_reset)
            else:
                write_in = self._cross_buf
            new_m_cross = self.m_cross.write(h_T_w, write_in, self._pool_mask)
        elif self.ccot_direct is not None:
            new_m_cross = self.ccot_direct.write(h_T, self._pool_mask)

        if new_m_cross is not None and self._valid_write is not None:
            new_m_cross = apply_valid_write(new_m_cross, self._valid_write)
        return new_m_cross
