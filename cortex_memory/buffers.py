"""
Cortex memory buffers — base-model-agnostic, pure torch.

Ported verbatim from cortex-main/model.py (LSTMBuffer, DirectCCoT).  These
modules know nothing about the host model: they operate on [B, S, D] hidden
states and [B, K, D] buffers.  They are grafted into RavenForCausalLM via
cortex_graft.py and gated behind config flags (default off).

  LSTMBuffer  — LM2-style K-slot LSTM-gated memory (M_cross / M_iter)
  DirectCCoT  — Coconut-style K=0 carry (single carried vector, no slots)
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# LSTM Buffer (LM2-style, arXiv 2502.06049)
# ---------------------------------------------------------------------------

class LSTMBuffer(nn.Module):
    """
    K-slot LSTM-gated memory buffer.

    Write: per-slot candidates via cross-attention — each slot (its content
           plus a learned slot embedding) QUERIES the sequence states, so the
           K slots extract distinct information by construction.  This is the
           relational-memory-style update that LM2's MemoryModule descends
           from (memory attends over inputs), replacing an earlier pooled
           design where one mean vector was broadcast to all slots and slot
           updates were near-redundant (threatening the K=4 > K=1 requirement,
           framework §4.4).  The gated update itself is LSTM-style with both
           gates receiving a combined signal from the pooled input *and* the
           current buffer state (memory feedback) — matches LM2 create_gates:
           gate_in = f(inputs) + g(tanh(memory)), one combined 2·D projection
           split evenly into ig/fg.

    Read:  cross-attention — sequence tokens query the K buffer slots —
           result additively injected into the loop state.
           (Cleaner than LM2's forced-square design; no seq_len==K constraint.)

    Granularity is the caller's choice: M_cross passes [B, S, D] (one buffer
    per sequence, pooled write — only safe because its content is read by a
    strictly-later segment).  M_iter folds the sequence dim into the batch and
    passes [B*S, 1, D] (one buffer per position) — required for causality,
    since M_iter is read again at earlier positions within the same forward.

    Key LM2 §3 details preserved
    ------------------------------
    - Memory feedback: tanh(buffer) projected into gate signal each write.
    - Combined gate projection split: both gates share the same intermediate
      representation, coupling their retain/update decisions (LM2 create_gates).
    - Forget gate bias +1.0: biases toward retention at init (LM2 §3.3).
    - out_proj zero-init: read injection is a no-op at step 0, preserving the
      pretrained transformer output at the start of training.
    """

    def __init__(self, hidden_size: int, n_slots: int, n_heads: int = 4) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.n_slots     = n_slots
        self.n_heads     = n_heads
        self.head_dim    = hidden_size // n_heads
        assert hidden_size % n_heads == 0

        # ── Write path ───────────────────────────────────────────────────────
        # Both gates are derived from the same combined signal (LM2 create_gates).
        # gate_in = gate_proj_in(h_pool) + gate_proj_mem(tanh(buffer))
        # The 2·D output is split in half: first D → input gate, second D → forget gate.
        self.gate_proj_in  = nn.Linear(hidden_size, hidden_size * 2)          # input side
        self.gate_proj_mem = nn.Linear(hidden_size, hidden_size * 2)          # memory side
        self.forget_bias   = nn.Parameter(torch.ones(1))   # +1.0 per LM2 §3.3
        self.input_bias    = nn.Parameter(torch.zeros(1))

        # Candidate via slot-query cross-attention (LM2 attend_over_memory /
        # relational memory): each slot queries the sequence states, so the K
        # candidates are slot-distinct by construction.  Slot identity comes
        # from learned slot embeddings (added to both the query and the
        # candidate residual — without the residual term, slots whose buffer
        # content is identical, e.g. all-zero at the first write, would
        # receive identical updates forever and collapse to K copies).
        self.slot_emb = nn.Parameter(torch.empty(n_slots, hidden_size))
        nn.init.normal_(self.slot_emb, std=0.02)
        # Embedding-like: exempt from weight decay and Muon Newton-Schulz.
        self.slot_emb._no_weight_decay = True

        self.wq_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.wk_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.wv_proj = nn.Linear(hidden_size, hidden_size, bias=False)

        # Refinement after attention (LM2 attend_over_memory: LN → MLP → LN —
        # attended_memory_layernorm + 2-layer ReLU MLP + layernorm2).
        self.cand_ln1   = nn.LayerNorm(hidden_size)
        self.cand_mlp1  = nn.Linear(hidden_size, hidden_size)
        self.cand_mlp2  = nn.Linear(hidden_size, hidden_size)
        self.cand_ln2   = nn.LayerNorm(hidden_size)

        # ── Read path ────────────────────────────────────────────────────────
        self.q_proj  = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj  = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj  = nn.Linear(hidden_size, hidden_size, bias=False)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        nn.init.zeros_(self.out_proj.weight)  # additive injection starts at zero

    def write(
        self,
        h_T: torch.Tensor,
        buffer: torch.Tensor,
        pool_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        h_T      : [B, S, D]  — final Loop state
        buffer   : [B, K, D]  — current K-slot buffer
        pool_mask: [B, S] bool, optional — positions the write may use
                   (restricts both the gate-input pooling and the candidate
                   attention to the still-open document's suffix in packed
                   segments).  None = use all positions.
        Returns updated buffer [B, K, D].

        Gate computation (LM2 create_gates):
          gate_in  = gate_proj_in(mean(h_T)) [B,K,2D]
                   + gate_proj_mem(tanh(buffer)) [B,K,2D]     ← both gates, combined signal
          ig, fg   = chunk(sigmoid(gate_in + bias), 2, dim=-1)     each [B, K, D]

        Candidate (slot-query cross-attention, then LM2 LN → MLP → LN):
          q         = wq(buffer + slot_emb)                ← slot-distinct queries
          attended  = MHA(q, wk(h_T), wv(h_T))             ← [B, K, D], masked by pool_mask
          cand      = LN1(buffer + slot_emb + attended)    ← residual keeps slot identity
          cand      = LN2(cand + mlp2(relu(mlp1(cand))))   ← MLP refinement
          candidate = tanh(cand)

          new_buf  = fg ⊙ buffer  +  ig ⊙ candidate
        """
        B, S, D = h_T.shape
        K, nh, hd = self.n_slots, self.n_heads, self.head_dim

        # Pool sequence → single summary vector [B, D] for the gate input side
        if pool_mask is None:
            h_pool = h_T.mean(dim=1)
        else:
            m = pool_mask.to(h_T.dtype).unsqueeze(-1)                  # [B, S, 1]
            h_pool = (h_T * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)

        # Input side: [B, 2D] → expand to [B, K, 2D]
        in_signal  = self.gate_proj_in(h_pool).unsqueeze(1).expand(-1, K, -1)  # [B, K, 2D]

        # Memory side: tanh(buffer) [B, K, D] → [B, K, 2D]  (LM2 line 281: tanh before proj)
        mem_signal = self.gate_proj_mem(torch.tanh(buffer))                    # [B, K, 2D]

        # Combined → split into ig/fg (both gates share the same intermediate repr)
        combined = in_signal + mem_signal                                       # [B, K, 2D]
        ig_logits, fg_logits = combined.chunk(2, dim=-1)                       # each [B, K, D]

        ig = torch.sigmoid(ig_logits + self.input_bias)                        # [B, K, D]
        fg = torch.sigmoid(fg_logits + self.forget_bias)                       # [B, K, D]

        # Candidate: each slot queries the sequence states, so the K candidates
        # are slot-distinct by construction (relational-memory-style write).
        slots = buffer + self.slot_emb.unsqueeze(0)                            # [B, K, D]
        q = self.wq_proj(slots).view(B, K, nh, hd).transpose(1, 2)             # [B, nh, K, hd]
        k = self.wk_proj(h_T).view(B, S, nh, hd).transpose(1, 2)               # [B, nh, S, hd]
        v = self.wv_proj(h_T).view(B, S, nh, hd).transpose(1, 2)               # [B, nh, S, hd]

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(hd)                     # [B, nh, K, S]
        if pool_mask is not None:
            # Restrict attention to the allowed positions.  Rows with NO
            # allowed position would softmax over all -inf (NaN): attend
            # unmasked instead and zero the result below — forward() also
            # zeroes the whole written row for such lanes via valid_write.
            valid     = pool_mask.any(dim=1)                                   # [B]
            safe_mask = pool_mask | (~valid).unsqueeze(1)                      # [B, S]
            scores    = scores.masked_fill(
                ~safe_mask.view(B, 1, 1, S), torch.finfo(scores.dtype).min
            )
        attended = F.softmax(scores, dim=-1) @ v                               # [B, nh, K, hd]
        attended = attended.transpose(1, 2).contiguous().view(B, K, D)
        if pool_mask is not None:
            attended = attended * valid.view(B, 1, 1).to(attended.dtype)

        # Refinement (LM2 attend_over_memory):
        #   memory = LN(memory + attended_memory)      ← attended_memory_layernorm
        #   memory = LN(memory + relu(fc2(relu(fc1(memory)))))  ← layernorm2
        cand      = self.cand_ln1(slots + attended)            # LN1, slot-identity residual
        mlp_out   = self.cand_mlp2(F.relu(self.cand_mlp1(cand)))
        candidate = torch.tanh(self.cand_ln2(cand + mlp_out))  # LN2 with MLP residual

        return fg * buffer + ig * candidate

    def read(self, h: torch.Tensor, buffer: torch.Tensor) -> torch.Tensor:
        """
        h     : [B, S, D]  — current queries
        buffer: [B, K, D]  — K-slot memory (keys/values)
        Returns [B, S, D] delta to add into h.
        """
        B, S, D = h.shape
        K, nh, hd = self.n_slots, self.n_heads, self.head_dim

        q = self.q_proj(h).view(B, S, nh, hd).transpose(1, 2)
        k = self.k_proj(buffer).view(B, K, nh, hd).transpose(1, 2)
        v = self.v_proj(buffer).view(B, K, nh, hd).transpose(1, 2)

        attn = F.softmax((q @ k.transpose(-2, -1)) / math.sqrt(hd), dim=-1)
        out  = (attn @ v).transpose(1, 2).contiguous().view(B, S, D)
        return self.out_proj(out)


# ---------------------------------------------------------------------------
# Direct CCoT carry (Cortex K=0 mode — Coconut-style, no LM2 machinery)
# ---------------------------------------------------------------------------

class DirectCCoT(nn.Module):
    """
    Direct cross-token CCoT state for K=0 Cortex (framework §4.4: the
    "Coconut-equivalent" — additive injection of a single carried vector,
    no slots, no gates, no cross-attention).

    Used when memory_slots == 0 so that Cortex K=0 remains an architectural
    superset of the Parcae baseline (which carries nothing) instead of being
    identical to it.

    write: state = state_proj(mean_S(h_T))   [B, 1, D]   (overwrite, stateless)
           state_proj is identity-init and doubles as the R4 dual-role
           mitigation: the carry path sees a projected h_T while the Coda
           sees the raw h_T (same role as h_T_proj in the LM2 buffer mode).
    read:  h + in_proj(state), broadcast over positions.
           in_proj is zero-init so the injection is a no-op at step 0.

    Trains through the same cross-chunk segment chain as the LM2 buffer:
    segment g+1's read of the un-detached state puts state_proj on the loss
    path.  Unlike the LM2 buffer there are no memory-feedback parameters, so
    n_chunks >= 2 suffices to train the whole module.
    """

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.state_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.in_proj    = nn.Linear(hidden_size, hidden_size, bias=False)
        nn.init.eye_(self.state_proj.weight)
        nn.init.zeros_(self.in_proj.weight)
        # Identity-init structural projection (same treatment as h_T_proj /
        # Parcae's C matrix): no weight decay, no Muon Newton-Schulz.
        # in_proj is a regular learned projection and stays in Muon (like the
        # LSTMBuffer's zero-init out_proj).
        self.state_proj.weight._no_weight_decay = True

    def write(
        self, h_T: torch.Tensor, pool_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """h_T [B, S, D] → state [B, 1, D].  Overwrites any previous state.
        pool_mask [B, S] optionally restricts pooling to the open document's
        suffix (same semantics as LSTMBuffer.write)."""
        if pool_mask is None:
            pooled = h_T.mean(dim=1, keepdim=True)
        else:
            m = pool_mask.to(h_T.dtype).unsqueeze(-1)                  # [B, S, 1]
            pooled = ((h_T * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)).unsqueeze(1)
        return self.state_proj(pooled)

    def read(self, state: torch.Tensor) -> torch.Tensor:
        """state [B, 1, D] → delta [B, 1, D], broadcast-added over positions."""
        return self.in_proj(state)
