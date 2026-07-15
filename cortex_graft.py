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
  ccot_iter           : bool = False   per-position Coconut carry ACROSS LOOP
                                       ITERATIONS (dense, within-window — the
                                       DirectCCoT twin of M_iter; no cross-
                                       segment state, trains at cross_chunks=1)
  accum_ccot          : bool = False   AutoCompressor-style accumulating carry
                                       (only when memory_slots==0, replaces
                                       ccot_direct's single overwritten vector)
  accum_vecs          : int  = 4       summary vectors extracted per chunk
  accum_max           : int  = 64      FIFO cap on accumulated vectors (eval)
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

from cortex_memory.buffers import LSTMBuffer, DirectCCoT, AccumCCoT
from cortex_memory.eos import compute_eos_masks, apply_write_reset, apply_valid_write


def memory_enabled(config) -> bool:
    """Master switch — read once in RavenForCausalLM.__init__."""
    return bool(getattr(config, "use_memory", False))


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

        # M_cross: LM2 K-slot buffer (K>0) XOR DirectCCoT K=0 carry XOR
        # AccumCCoT accumulating carry (K=0; takes precedence over ccot_direct
        # — train.py asserts they are not both set).
        self.m_cross = LSTMBuffer(D, K, nh) if K > 0 else None
        self.accum   = (
            AccumCCoT(D, int(getattr(config, "accum_vecs", 4)), nh,
                      int(getattr(config, "accum_max", 64)))
            if (K == 0 and bool(getattr(config, "accum_ccot", False))) else None
        )
        self.ccot_direct = (
            DirectCCoT(D)
            if (K == 0 and self.accum is None
                and bool(getattr(config, "ccot_direct", False))) else None
        )
        # M_iter: per-position short-term buffer (independent of M_cross).
        self.m_iter = LSTMBuffer(D, Ki, nh) if Ki > 0 else None
        # ccot_iter: per-position DirectCCoT carried across LOOP ITERATIONS —
        # the Coconut-faithful twin of M_iter (dense within-window read/write,
        # no slots/gates, no cross-segment state).  Sequence dim folds into
        # the batch exactly like M_iter, so causality holds by construction.
        self.ccot_iter = (
            DirectCCoT(D) if bool(getattr(config, "ccot_iter", False)) else None
        )

        # R4 dual-role mitigation: project h_T before the M_cross write so the
        # buffer path and the coda path see independent representations.
        # Identity-init → no-op at step 0.  LM2 mode only.
        if self.m_cross is not None and bool(getattr(config, "h_T_proj", True)):
            self.h_T_proj = nn.Linear(D, D, bias=False)
            nn.init.eye_(self.h_T_proj.weight)
            self.h_T_proj.weight._no_weight_decay = True
        else:
            self.h_T_proj = None

        self._reset_runtime()

    @property
    def has_cross_state(self) -> bool:
        return (self.m_cross is not None or self.ccot_direct is not None
                or self.accum is not None)

    # ── per-call runtime ────────────────────────────────────────────────────
    def _reset_runtime(self) -> None:
        self._cross_buf:       Optional[torch.Tensor] = None  # carried M_cross [B,K,D]/[B,1,D]/[B,N,D]
        self._cross_read_mask: Optional[torch.Tensor] = None  # [B,S,1]
        self._pool_mask:       Optional[torch.Tensor] = None  # [B,S] bool
        self._write_reset:     Optional[torch.Tensor] = None  # [B] bool
        self._valid_write:     Optional[torch.Tensor] = None  # [B] bool
        self._iter_buf:        Optional[torch.Tensor] = None  # [B*S,Ki,D]
        self._ccot_iter_buf:   Optional[torch.Tensor] = None  # [B*S,1,D]

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

        # ccot_iter per-position read of the previous loop iteration's carry
        # (no read at the first iteration — nothing written yet, matching
        # Coconut where the first forward has no latent thought to consume).
        if self.ccot_iter is not None and self._ccot_iter_buf is not None:
            B, S, D = x.shape
            x = x + self.ccot_iter.read(self._ccot_iter_buf).reshape(B, S, D)
        return x

    def iter_write(self, x: torch.Tensor) -> None:
        """Write each position's state into its own M_iter slots / ccot_iter
        carry, after the core layers (end of one loop iteration)."""
        if self.m_iter is not None:
            B, S, D = x.shape
            if self._iter_buf is None:
                self._iter_buf = x.new_zeros(B * S, self.memory_slots_iter, D)
            self._iter_buf = self.m_iter.write(x.reshape(B * S, 1, D), self._iter_buf)
        if self.ccot_iter is not None:
            B, S, D = x.shape
            # DirectCCoT.write pools over the sequence dim — folding S into
            # the batch makes that a per-position identity pool (mean of 1).
            self._ccot_iter_buf = self.ccot_iter.write(x.reshape(B * S, 1, D))

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
