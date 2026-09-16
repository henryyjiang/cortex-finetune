"""
Cortex memory buffers -- base-model-agnostic, pure torch.

These modules know nothing about the host model: they operate on [B, S, D]
hidden states and [B, K, D] buffers.  They are grafted into RavenForCausalLM
via cortex_graft.py and gated behind config flags (default off).

THE LIVE SURFACE (what a new run may select):

  _PrefixBufferBase  -- shared summary-token machinery
  PrefixAccumBuffer  -- AutoCompressor-faithful append (accumulate_summary)
  PrefixGatedBuffer  -- the same write with an LM2-style gate at constant
                        width, and (P1.0, 2026-09-16) writes-per-chunk
                        decoupled from that width via n_slots/route

Subclasses differ only in `merge` -- and, for the gated buffer at
n_slots == n_vec, in nothing else at all, which is what makes append-vs-gated
a controlled comparison.  Widening n_slots past n_vec adds a routing stage;
that is a second variable, so an A/B on the merge rule must not move both at
once.  `PrefixGatedBuffer.geometry()` reports what a configuration costs.

THE LEGACY SURFACE lives in `legacy.py` -- LSTMBuffer, DirectCCoT, AccumCCoT,
GatedAccumBuffer.  They are re-exported below so existing imports keep working,
but they are load-path only: `train.py` refuses to start a new run on them, and
they cannot be deleted without orphaning every Track-A and B1 checkpoint.  See
legacy.py's docstring before touching them.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# Legacy buffers, quarantined to legacy.py on 2026-09-14.  Re-exported so that
# every existing `from cortex_memory.buffers import ...` keeps working -- the
# graft's load path for Track-A / B1 checkpoints depends on these names.
from .legacy import (  # noqa: F401
    LSTMBuffer,
    DirectCCoT,
    AccumCCoT,
    GatedAccumBuffer,
    _extract_summary_vectors,
)


# ---------------------------------------------------------------------------
# AutoCompressor-faithful prefix buffers
# (supersede AccumCCoT / GatedAccumBuffer; those two stay only so the Track-A
#  and B1 arms remain reproducible — new runs should not select them)
# ---------------------------------------------------------------------------
#
# WHY THESE EXIST
# ---------------
# The B1 co-training round measured carry = -0.0128 nats (z = -9.3): carrying
# the buffer made the model WORSE than zeroing it.  Reading
# AutoCompressors-main/auto_compressor.py against our implementation turned up
# four divergences from the published, working design.  Each is fixed here.
#
#   1. THE READ PATH.  AutoCompressor PREPENDS its summary vectors to the next
#      segment's input embeddings (auto_compressor.py:85) — they are consumed
#      by the model's OWN pretrained self-attention at every layer.  There is
#      no read module.  We used a separate cross-attention block with a
#      zero-init out_proj injected inside the recurrent loop, so the entire
#      communication channel had to be discovered from nothing.  These buffers
#      have NO read path: the modeling file prepends `state` to input_embeds
#      and the base model's own attention does the reading.
#
#   2. THE WRITE PATH.  AutoCompressor APPENDS learned summary-token embeddings
#      to the segment and takes their output hidden states
#      (auto_compressor.py:123) — the summary is written by the model's own
#      full-depth forward.  We used a bolt-on extraction cross-attention.  Here
#      `summary_emb` is appended to the input by the modeling file and the
#      resulting post-ln_f hidden states ARE the new vectors.
#
#   3. NO tanh(LayerNorm(.)) BOUNDING.  Our extraction ended in
#      tanh(vec_ln(attended + vec_emb)), pinning the write norm by
#      construction — the 2026-08-02 buffer diag measured write-norm variation
#      across samples at 0.23% of the mean, the flattest of any arm run.
#      AutoCompressor applies no bounding: its summary vectors are raw hidden
#      states living in the same space as token embeddings, which is exactly
#      what feeding them back as input embeddings requires.
#
#   4. INITIALISATION.  AutoCompressor seeds every summary embedding with the
#      EOS token embedding (auto_compressor.py:49) — a real, in-distribution
#      vector the pretrained model already knows how to process.  Ours were
#      normal_(std=0.02) noise.  init_from_token_embedding does it their way;
#      the graft calls it once the base weights are loaded.
#
# WHAT IS DELIBERATELY NOT COPIED
# -------------------------------
# AutoCompressor gives summary/softprompt slots the pad position (no position).
# OPT has learned absolute position embeddings; we have RoPE, so the modeling
# file's analog is position 0 (identity rotation) for prefix and summary slots,
# with real tokens taking 1..S.  That also keeps every position index inside
# the trained 1024 window even though the padded sequence is longer than it.
#
# STATE SHAPE CONTRACT (unchanged from AccumCCoT, on purpose)
# -----------------------------------------------------------
# `state` stays [B, N, D] with write-once rows, so train.py's stop-gradient
# horizon (detach_old_vecs, --cortex.carry_grad_chunks) keeps working
# untouched, and so do apply_write_reset / apply_valid_write in the graft.


class _PrefixBufferBase(nn.Module):
    """Shared summary-token machinery for the two prefix buffers.

    Subclasses differ in `merge` — append (AutoCompressor's
    accumulate_summary) vs the LM2-style gated update at constant memory — and
    the gated one additionally owns the write-to-row routing P1.0 introduced.
    At n_slots == n_vec that routing is the identity and the two arms differ in
    nothing but the merge rule; a gated buffer's FIRST LAP is bit-identical to
    PrefixAccumBuffer at the same (n_vec, n_slots), so the two diverge exactly
    at eviction.
    """

    def __init__(self, hidden_size: int, n_vec: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.n_vec = n_vec

        # The analog of AutoCompressor's `embed_summary` (auto_compressor.py:47).
        # Overwritten by init_from_token_embedding once the base weights exist;
        # this normal_ only matters for unit tests that build the buffer
        # standalone.  Embedding-like: no weight decay, no Muon Newton-Schulz.
        self.summary_emb = nn.Parameter(torch.empty(n_vec, hidden_size))
        nn.init.normal_(self.summary_emb, std=0.02)
        self.summary_emb._no_weight_decay = True

        # PERSISTENT so it survives save/load.  The seeding below can only run
        # on the first forward (wte is random at __init__), and a plain Python
        # attribute would be False again after every resume — which would
        # re-seed a TRAINED summary_emb back to wte[eos] at the start of every
        # link in the resume chain, silently resetting the write path while the
        # loss curve stayed healthy.  A registered buffer rides in the state
        # dict, so a restored checkpoint reports itself already seeded.
        self.register_buffer("summary_seeded",
                             torch.zeros((), dtype=torch.bool), persistent=True)

    @torch.no_grad()
    def init_from_token_embedding(self, wte_weight: torch.Tensor,
                                  token_id: int) -> None:
        """Seed every summary embedding with one real token's embedding.

        AutoCompressor uses EOS (auto_compressor.py:49).  All n_vec rows start
        identical and differentiate through training, exactly as theirs do.
        """
        assert wte_weight.shape[1] == self.hidden_size, (
            f"embedding dim {wte_weight.shape[1]} != buffer dim {self.hidden_size}")
        self.summary_emb.data.copy_(
            wte_weight[token_id].to(self.summary_emb.dtype)
            .unsqueeze(0).expand(self.n_vec, -1)
        )
        self.summary_seeded.fill_(True)

    def summary_slots(self, batch_size: int, dtype: torch.dtype,
                      device: torch.device) -> torch.Tensor:
        """[B, n_vec, D] summary-token embeddings to APPEND to this chunk's
        input.  Their post-ln_f hidden states come back through `merge`."""
        return (self.summary_emb.unsqueeze(0)
                .expand(batch_size, -1, -1).to(device=device, dtype=dtype))

    def merge(self, state: Optional[torch.Tensor],
              new_vecs: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class PrefixAccumBuffer(_PrefixBufferBase):
    """AutoCompressor's accumulating carry, faithfully.

    `merge` concatenates this chunk's summary vectors onto the carried state
    and never overwrites — AutoCompressor's accumulate_summary=true
    (auto_compressor.py:230, the default in their run/train.sh).  The FIFO cap
    only bounds unbounded chunk chains at eval; in training the state is
    expected to stay under it.

    Rows are write-once, so per-chunk slices stay separable and train.py can
    slice-detach vectors older than --cortex.carry_grad_chunks — their
    "stop-gradient after N compression steps" (substep_trainer.py:166).
    """

    def __init__(self, hidden_size: int, n_vec: int = 32,
                 max_vecs: int = 128) -> None:
        super().__init__(hidden_size, n_vec)
        assert max_vecs >= n_vec, "max_vecs must hold at least one chunk's write"
        self.max_vecs = max_vecs

    def merge(self, state: Optional[torch.Tensor],
              new_vecs: torch.Tensor) -> torch.Tensor:
        """state [B, N, D] (None on chunk 1) + [B, n_vec, D] -> [B, N', D]."""
        out = new_vecs if state is None else torch.cat([state, new_vecs], dim=1)
        if out.shape[1] > self.max_vecs:
            out = out[:, -self.max_vecs:]          # FIFO: drop the oldest
        return out


class PrefixGatedBuffer(_PrefixBufferBase):
    """LM2-style gated carry with writes-per-chunk DECOUPLED from buffer width.

    `merge` is the LM2 gated update (create_gates, LM2 section 3.3) at CONSTANT
    memory instead of an append:

        combined = gate_proj_in(candidate) + gate_proj_mem(norm(state))
        ig, fg   = chunk(combined, 2, -1);  sigmoid(. + bias)
        state'   = fg * state + ig * candidate

    THE P1.0 CHANGE (2026-09-16).  Until now `self.n_slots = n_vec` and merge was
    [B,K,D] + [B,K,D] -> [B,K,D], so writes-per-chunk was LOCKED to buffer width
    and two different quantities wore one name:

        n_vec    summary columns APPENDED per chunk -> compression (chunk_len /
                 n_vec) and per-chunk write cost.  A PARAMETER SHAPE
                 (`summary_emb`), so it can never change after training.
        n_slots  carried columns PREPENDED per chunk -> capacity and the fixed
                 per-chunk read cost.  Under route="ring" NO parameter has an
                 n_slots dimension, so it is a RUNTIME knob, sweepable at eval
                 the way `accum_max` is.

    At n_slots == n_vec with route="ring" this class is bit-identical to the
    pre-P1.0 version, and for its first lap it is bit-identical to
    `PrefixAccumBuffer(D, n_vec, n_slots)`.  `tests/test_gate_geometry.py` pins
    both.

    ROUTING.  How W = n_vec candidate vectors reach K = n_slots state rows.

      "ring"   (default, and the recommended one) SPARSE: chunk c gates rows
               [cW, cW+W) mod K and leaves the other K-W rows untouched -- not
               decayed, not gated, passed through.
      "mix"    DENSE control: candidate = M @ new_vecs for a learned [K, W]
               matrix, so every row is refreshed every chunk.

    WHY SPARSE IS THE DEFAULT.  Three reasons, in the order they actually bind.

      1. THE READ IS ATTENTION, AND SUPERPOSED ROWS ARE NOT SELECTABLE.  A dense
         route makes every row a weighted sum of EVERY chunk seen so far.  These
         rows are consumed as keys and values by the base model's own attention
         (that is the whole point of the AutoCompressor read), and a row that
         superposes thirty chunks cannot be selected against.  This is not
         hypothetical: B2's accum rows already measure centred cosine 0.96 and
         effective rank ~4 of 32, and the documented active-harm mechanism is
         "many near-identical keys take softmax mass by count".  Dense gating
         makes that pathology structural rather than incidental.
      2. TRUNCATED BPTT CANNOT TEACH A DENSE ROUTE TO REMEMBER.  A dense route's
         horizon is entirely `fg^d`, so a long horizon must be LEARNED -- but
         `carry_grad_chunks` bounds credit assignment at 2-3 chunks, so there is
         no gradient path long enough to learn one.  The ring's horizon is
         mechanical (K/W chunks before a row is even touched) and needs no
         gradient at all.  This is the same structural fact behind
         AutoCompressor's own result that stop-gradient after ~2 compression
         steps costs nothing: their retention is mechanical too.
      3. IT KEEPS THE DEPTH-SLICE ABLATION.  Under the ring, write vector j
         always lands at rows congruent to j (mod W), forever -- see
         `depth_rows`.  So if the j-th write carries loop depth k_j, that depth
         stays addressable and the ablation that picks the write depths still
         works.  Dense routing destroys the mapping.  (What gating does kill,
         under either route, is CHUNK separability -- which chunk wrote a row.
         Those are different slices and the two have been conflated.)

    What sparse gives up is content-based allocation: the ring's schedule is
    content-blind.  That is real, but it is unreachable at carry_grad_chunks=2
    for reason 2, so it is not a capability being traded away today.

    "attn" (K learned queries cross-attending the W candidates) is deliberately
    NOT implemented.  It measured worst of every routing on retention, costs 4D^2
    parameters, and it is not what published LM2 does anyway -- LM2's memory is
    [N, N], identity-initialised, updated per LAYER, and added to the attention
    output rather than spliced into the sequence (see lm2/src/memory.py).  Adding
    it here would buy neither the mechanism nor the citation.

    GEOMETRY CONSTRAINT, and nothing else checks it: the gate only starts acting
    on lap 2, so

        cross_chunks >= 2 * (n_slots / n_vec)

    or the ring never completes a lap during training and the gate receives ZERO
    gradient -- you would have trained an accum buffer with dead gate parameters
    attached.  train.py asserts this.

    CHUNK 1, AND THE FIRST LAP.  The candidate is adopted whole (AutoCompressor's
    behaviour when the softprompt is empty: nothing to retain, so a gate could
    only attenuate).  Under "ring" with K > W that leaves rows unreached for the
    first K/W chunks, and `fill` says what happens to them:

      "grow"  (default)  they do not exist yet.  The read block starts at W and
              grows by W per chunk until it reaches K, then is fixed forever.
              For that first lap the buffer is bit-identical to
              `PrefixAccumBuffer(D, W, K)`, so an accum arm and a gated arm
              branched from the same checkpoint diverge at EXACTLY the chunk
              where accum starts dropping its oldest rows and the ring starts
              gating them instead.  That is the cleanest available contrast for
              the horizon claim.
      "init"  they are filled from `slot_init`, a learned [K, D] parameter
              seeded from the same real token as `summary_emb`.  NOT the
              default, and the reason is measured: the carried block holds
              post-`ln_f` states with row norm ~171 on the B2 checkpoint, while
              `wte` rows are ~9.8 and the trained `summary_emb` ~9.06.  An
              EOS-seeded row enters the read block ~19x shorter than the real
              rows beside it -- nearly the dead key AutoCompressor's EOS trick
              exists to avoid, because that trick seeds an INPUT EMBEDDING and
              these rows are not embeddings.  Zeros would be worse: a zero key
              scores a mid-range logit rather than -inf, the documented
              3-5%-of-softmax-mass artifact.  `slot_init` is only allocated when
              fill="init", so the default carries no unused parameter.

    RETENTION MODEL.  Content enters through the input gate and decays once per
    lap through the forget gate:

        F(d) = ig * fg ** floor(d / (K/W))

    At gate_init="zero" that is exact at step 0 (ig = 0.500, fg = 0.731), and it
    reproduces the measured donor-swap curve to ~2%.  At W=16/K=64 it gives 0.50
    retained for 0-3 chunks, 0.27 at 8-11, 0.10 at ~21, 0.05 at ~29 -- against
    accum's 1.0-then-exactly-zero cliff at 8.  `evals/diag_gate_geometry.py`
    measures it; do not requote the model where the measurement exists.

    TRAINING NOTES.  merge overwrites rows in place, so rows are not separable by
    chunk and train.py detaches the whole state every carry_grad_chunks chunks
    (truncated BPTT) rather than slicing.  `gate_proj_mem` only receives gradient
    through a chain of >= 3 chunks -- keep carry_grad_chunks >= 2 or it never
    trains.  Under "ring" an untouched row keeps a live graph edge for up to K/W
    chunks, which the whole-state detach already bounds.
    """

    ROUTES = ("ring", "mix")

    def __init__(self, hidden_size: int, n_vec: int = 32,
                 n_slots: Optional[int] = None, route: str = "ring",
                 gate_norm: str = "tanh", gate_init: str = "zero",
                 fill: str = "grow", route_init_std: float = 0.02) -> None:
        super().__init__(hidden_size, n_vec)
        self.n_slots = int(n_slots) if n_slots else int(n_vec)
        if self.n_slots < n_vec:
            raise ValueError(
                f"n_slots ({self.n_slots}) < n_vec ({n_vec}): the gated buffer "
                "cannot hold fewer rows than one chunk writes.  Lower "
                "--cortex.accum_vecs instead; the point of decoupling is cheap "
                "writes into a WIDE state, not the reverse.")
        if route not in self.ROUTES:
            raise ValueError(f"route must be one of {self.ROUTES}; got {route!r}")
        if gate_norm not in ("tanh", "rms", "none"):
            raise ValueError(
                f"gate_norm must be 'tanh', 'rms' or 'none'; got {gate_norm!r}")
        if gate_init not in ("zero", "default"):
            raise ValueError(
                f"gate_init must be 'zero' or 'default'; got {gate_init!r}")
        if fill not in ("grow", "init"):
            raise ValueError(f"fill must be 'grow' or 'init'; got {fill!r}")
        if route == "ring" and self.n_slots % n_vec:
            raise ValueError(
                f"route='ring' needs n_slots ({self.n_slots}) to be a multiple "
                f"of n_vec ({n_vec}) -- a partial lap would write a different "
                "row set every time round and break the depth->row map that "
                "`depth_rows` depends on.  Use route='mix' for a ragged ratio.")
        self.route = route
        self.gate_norm = gate_norm
        self.gate_init = gate_init
        self.fill = fill

        self.gate_proj_in  = nn.Linear(hidden_size, hidden_size * 2)
        self.gate_proj_mem = nn.Linear(hidden_size, hidden_size * 2)
        self.forget_bias   = nn.Parameter(torch.ones(1))    # +1.0, LM2 3.3
        self.input_bias    = nn.Parameter(torch.zeros(1))
        self.apply_gate_init()

        if fill == "init":
            self.slot_init = nn.Parameter(torch.empty(self.n_slots, hidden_size))
            nn.init.normal_(self.slot_init, std=0.02)
            self.slot_init._no_weight_decay = True

        if route == "mix":
            # Tile pattern + noise.  The tile part keeps the routing
            # in-distribution at init (each row starts as ONE real candidate,
            # not an average of W of them); the noise is the symmetry breaker.
            # WITHOUT IT THIS IS PROVABLY DEGENERATE: rows r and r+W would
            # receive the same candidate from the same state under a pointwise
            # gate, so their trajectories would be equal forever and the state
            # would be pinned at rank <= W.  route_init_std=0.0 reproduces that
            # on purpose in the tests.
            m = torch.zeros(self.n_slots, n_vec)
            m[torch.arange(self.n_slots), torch.arange(self.n_slots) % n_vec] = 1.0
            m = m + torch.randn_like(m) * route_init_std
            self.route_mix = nn.Parameter(m)

        # Chunk counter for the ring cursor.  NOT a parameter and NOT persistent:
        # it is per-SEQUENCE runtime, and `state is None` (chunk 1) is the reset
        # signal, so it re-synchronises at the start of every forward without the
        # graft having to reach in.  A stale value would only rotate WHICH rows a
        # chunk writes, never how many.
        self._chunk = 0

    def apply_gate_init(self) -> None:
        """(Re-)apply the designed gate initialisation.

        MUST be called from train.py's reset_cortex_graft_init: that function
        calls reset_parameters() on every cortex submodule to undo post_init's
        non-finite clobber, which puts kaiming weights back into both gate
        projections and silently discards gate_init="zero".  Bug class 1.
        """
        nn.init.ones_(self.forget_bias)
        nn.init.zeros_(self.input_bias)
        if self.gate_init != "zero":
            return
        # Zero the WEIGHTS and biases of both projections, so step 0 is exactly
        # the constant EMA  s' = sigmoid(1)*s + sigmoid(0)*c = 0.731*s + 0.5*c,
        # uniform over rows and channels.
        #
        # This is NOT the zero-init failure the project already paid for.
        # LSTMBuffer zero-init'ed `out_proj`, which made the READ a literal
        # no-op that had to be discovered from nothing.  Here the read is
        # untouched and fully live -- an EMA is a working memory -- and what is
        # switched off at init is only the gate's CONTENT SENSITIVITY, which
        # training then earns rather than unlearns.  The alternative (PyTorch's
        # kaiming default) measures, on the real write tape, as fg already
        # spread p10 0.13 / p90 0.98: millions of untrained parameters making
        # strong per-channel keep/drop decisions about a write they know nothing
        # about, on top of a candidate that is ~98% a constant direction.
        for proj in (self.gate_proj_in, self.gate_proj_mem):
            nn.init.zeros_(proj.weight)
            nn.init.zeros_(proj.bias)

    # -- addressing ---------------------------------------------------------

    def depth_rows(self, j: int) -> list[int]:
        """Rows that always receive write-vector j, under route='ring'.

        Chunk c writes vector j to row (cW + j) mod K, and cW mod K cycles
        through multiples of W, so row r holds write-index r mod W for every
        chunk.  If the j-th summary column is written at loop depth k_j, this is
        the row set to ablate to remove depth k_j -- which is why gating does not
        cost the depth-slice ablation.
        """
        if self.route != "ring":
            raise ValueError(
                f"depth_rows is only defined for route='ring'; this buffer is "
                f"{self.route!r}, which mixes every write into every row and has "
                "no depth->row map.")
        if not 0 <= j < self.n_vec:
            raise ValueError(f"write index {j} outside 0..{self.n_vec - 1}")
        return [r for r in range(self.n_slots) if r % self.n_vec == j]

    # -- the update ---------------------------------------------------------

    def route_candidate(self, new_vecs: torch.Tensor) -> torch.Tensor:
        """[B, n_vec, D] -> [B, n_slots, D] dense candidate (route='mix')."""
        return torch.einsum("kw,bwd->bkd",
                            self.route_mix.to(new_vecs.dtype), new_vecs)

    def _mem_side(self, state: torch.Tensor) -> torch.Tensor:
        """What the memory half of the gate sees.

        "tanh" is LM2 as published.  It is a knob because `state` here is a stack
        of post-`ln_f` hidden states whose scale is set by the base model, not a
        normalised memory bank: measured on the B2 checkpoint, 60-75% of entries
        are past |2|, where tanh' < 0.08, so the forget gate is reading
        sign(state) and cannot see HOW MUCH a row holds.  "rms" removes the
        question at no measured cost in retention.  Do not guess which --
        evals/diag_gate_geometry.py reports the saturated fraction.
        """
        if self.gate_norm == "tanh":
            return torch.tanh(state)
        if self.gate_norm == "rms":
            return state * torch.rsqrt(
                state.float().pow(2).mean(-1, keepdim=True).clamp_min(1e-12)
            ).to(state.dtype)
        return state

    def gate(self, state: torch.Tensor, candidate: torch.Tensor):
        """The LM2 update on matched [B, n, D] tensors.

        Returns (state', ig, fg) so diagnostics can read the gates without a
        second forward -- the trained spread of fg across rows and inputs is the
        pre-registered test of whether the gate learned anything or collapsed to
        an EMA with extra parameters.
        """
        combined = (self.gate_proj_in(candidate)
                    + self.gate_proj_mem(self._mem_side(state)))
        ig_logits, fg_logits = combined.chunk(2, dim=-1)
        ig = torch.sigmoid(ig_logits + self.input_bias)
        fg = torch.sigmoid(fg_logits + self.forget_bias)
        return fg * state + ig * candidate, ig, fg

    def merge(self, state: Optional[torch.Tensor],
              new_vecs: torch.Tensor) -> torch.Tensor:
        """state [B, K, D] (None on chunk 1) + [B, W, D] -> [B, K', D].

        K' is K except during a "grow" first lap, where it is the number of rows
        written so far.
        """
        B, W, _ = new_vecs.shape
        K = self.n_slots
        if W != self.n_vec:
            raise ValueError(
                f"merge got {W} new vectors, buffer writes {self.n_vec}.  The "
                "write width is summary_emb's shape -- it cannot change after "
                "training.")

        if state is None:                       # chunk 1: adopt whole
            self._chunk = 1
            if self.route != "ring":
                return self.route_candidate(new_vecs)
            if K == W or self.fill == "grow":
                return new_vecs
            out = (self.slot_init.to(device=new_vecs.device,
                                     dtype=new_vecs.dtype)
                   .unsqueeze(0).expand(B, -1, -1).clone())
            out[:, :W] = new_vecs
            return out

        if self.route == "ring" and self.fill == "grow" and state.shape[1] < K:
            self._chunk += 1                    # still filling: append, as accum does
            return torch.cat([state, new_vecs], dim=1)

        if state.shape[1] != K:
            raise ValueError(
                f"carried state has {state.shape[1]} rows, buffer holds {K}.  A "
                "gated buffer's read block is fixed-width once filled; a "
                "mismatch means an accum-shaped carry reached a gated buffer.")

        if self.route != "ring":
            merged, _, _ = self.gate(state, self.route_candidate(new_vecs))
            self._chunk += 1
            return merged

        # Sparse ring write: gate ONLY the W rows this chunk touches and leave
        # the other K-W rows exactly as they are.  index_copy is out-of-place, so
        # autograd sees untouched rows as a pass-through rather than as a gate
        # applied with fg == 1.
        idx = ((self._chunk * W) % K
               + torch.arange(W, device=state.device)) % K
        sub, _, _ = self.gate(state.index_select(1, idx), new_vecs)
        out = state.index_copy(1, idx, sub)
        self._chunk += 1
        return out

    # -- seeding ------------------------------------------------------------

    @torch.no_grad()
    def init_from_token_embedding(self, wte_weight: torch.Tensor,
                                  token_id: int) -> None:
        """Seed summary_emb, and slot_init when it exists.

        See the `fill` note above for why an embedding-scaled `slot_init` is the
        wrong scale for the carry block, and why "grow" is the default.
        """
        super().init_from_token_embedding(wte_weight, token_id)
        if hasattr(self, "slot_init"):
            self.slot_init.data.copy_(
                wte_weight[token_id].to(self.slot_init.dtype)
                .unsqueeze(0).expand(self.n_slots, -1)
            )

    # -- accounting ---------------------------------------------------------

    def geometry(self, chunk_len: int) -> dict:
        """The numbers P1.0 is trading off, in one place."""
        lap = self.n_slots // self.n_vec if self.route == "ring" else 1
        return {
            "n_vec": self.n_vec,
            "n_slots": self.n_slots,
            "route": self.route,
            "gate_norm": self.gate_norm,
            "gate_init": self.gate_init,
            "fill": self.fill,
            "compression": chunk_len / max(self.n_vec, 1),
            "read_cols_per_chunk": self.n_slots,
            "write_cols_per_chunk": self.n_vec,
            "packed_cols": self.n_slots + chunk_len + self.n_vec,
            # Chunks before a ring row is revisited.  Dense routings touch every
            # row every chunk, so their lap is 1.
            "lap_chunks": lap,
            "min_cross_chunks": 2 * lap,
            "params": sum(p.numel() for p in self.parameters()),
        }
