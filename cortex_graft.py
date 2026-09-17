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
  accum_max           : int  = 64      FIFO cap on accumulated vectors (eval).
                                       ACCUM ONLY -- the gated buffer's capacity
                                       is gate_slots.
  gate_slots          : int  = 0       K for prefix_memory=gated: carried columns
                                       held, i.e. the fixed read cost per chunk.
                                       0 means "same as accum_vecs", the
                                       pre-P1.0 shape.  DECOUPLED from accum_vecs
                                       (the write width) since 2026-09-16.
                                       Under gate_route=ring no parameter has a
                                       gate_slots dimension, so unlike
                                       accum_vecs it can be changed after
                                       training and swept at eval.
  gate_route          : str  = "ring"  how accum_vecs writes reach gate_slots
                                       rows.  "ring" = sparse FIFO-indexed write
                                       (default), "mix" = dense learned [K,W]
                                       routing.  See the PrefixGatedBuffer
                                       docstring for why sparse is the default.
  gate_norm           : str  = "tanh"  what the memory half of the gate sees:
                                       "tanh" (LM2 as published), "rms", "none".
  gate_init           : str  = "zero"  "zero" makes step 0 the constant EMA
                                       0.731*state + 0.5*candidate; "default"
                                       restores PyTorch's kaiming gate weights.
  gate_fill           : str  = "grow"  rows a ring has not reached on its first
                                       lap: "grow" emits only written rows (the
                                       buffer IS PrefixAccumBuffer for one lap),
                                       "init" pads from a learned slot_init.
  latent_carry        : bool = False   THE Z CHANNEL.  Carry the recurrence's
                                       TRAJECTORY alongside the token-space
                                       summary, and read it by substituting into
                                       `s0` at the carried columns.  Widens the
                                       carried tensor's last dim to 2D (E at
                                       [...,:D], Z at [...,D:]).  Default off,
                                       and off is byte-identical to B2.
  latent_depth_rule   : str  = "absolute"
                                       how slot j picks the loop depth k_j it
                                       writes.  "absolute" tiles the measured
                                       band and clamps to T-1; "relative" takes
                                       round(f_j * (T-1)).  P0.7
                                       (evals/diag_depth_band.py) DECIDES THIS —
                                       it is a flag precisely so the answer is a
                                       config change and not a code change.
  latent_depth_lo/hi  : int  = 2 / 9   the band, in absolute loop steps.  P0.1,
                                       fp32: d_1 is 23.8x ||s0|| (a different
                                       regime, excluded) and past t~12 the
                                       deltas are 0.1x with cos(d_t,d_t-1)
                                       reaching -0.95, a near-pure two-cycle
                                       oscillation.
  latent_renorm       : str  = "none"  "none" substitutes the delta as measured
                                       — P0.1 puts it at 0.90x the noise it
                                       replaces AT T=8, so it is in distribution
                                       as drawn and needs no parameter.  "s0"
                                       rescales each row to the replaced noise's
                                       own RMS, which is what a write depth in
                                       the saturated tail (0.06x at t=32) would
                                       need.
  prefix_pos          : str  = "tail"  ONLY value; asserted, not branched on.
                                       Where the trailing summary slots sit in
                                       POSITION space: "tail" = continue the
                                       chunk's numbering (S+1..S+n_vec), "zero"
                                       = the pre-2026-08-04 layout, everything
                                       non-token at position 0.  See prefix_pack.
  prefix_eos_reset    : bool = False   ONLY value; asserted, not branched on.
                                       Would zero the WHOLE incoming carry on any chunk
                                       containing an EOS.  Was unconditional
                                       before 2026-08-04; see _carried_state.
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
                        core layers; cortex.iter_write(x, current_step) after
                        the core layers.
  iterate_forward()   : x = cortex.latent_init(x) immediately after
                        initialize_state — the Z READ.  It has to live there
                        and nowhere else: initialize_state runs AFTER
                        prefix_pack, so the carried columns already exist in the
                        packed sequence and are filled with fresh noise that
                        nothing reads.  Substituting into them costs no
                        sequence length and no parameters.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn

from cortex_memory.buffers import (AccumCCoT, DirectCCoT, GatedAccumBuffer,
                                   LSTMBuffer, PrefixAccumBuffer, PrefixGatedBuffer)
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

        # LEGACY cross-segment buffers (Track A, B1).  Superseded by the Prefix*
        # pair below, but NOT deleted: every Track-A and B1 checkpoint carries
        # accum_ccot / gated_accum / ccot_direct in its config.json, so refusing
        # to build them makes the entire published results table unloadable and
        # un-re-measurable.  That is exactly what happened between 2026-08-02 and
        # 2026-08-04, when this block raised instead: the write-capacity
        # diagnostic could not even open the checkpoint whose -0.01282 carry
        # delta it was written to explain.
        #
        # The original raise existed to prevent a SILENT no-memory run.  Building
        # the real buffer cannot cause that — the mechanism is genuinely present
        # — so the protection belongs where the risk actually is: train.py's
        # config validation refuses to START a new run on a retired mechanism,
        # while loading an old checkpoint for evaluation works.
        legacy_accum  = bool(getattr(config, "accum_ccot", False))
        legacy_gated  = bool(getattr(config, "gated_accum", False))
        legacy_direct = bool(getattr(config, "ccot_direct", False))
        n_vec_legacy  = int(getattr(config, "accum_vecs", 4))
        max_vec_legacy = int(getattr(config, "accum_max", 64))

        if legacy_gated and K > 0:
            self.m_cross = GatedAccumBuffer(D, K, nh)
        elif K > 0:
            self.m_cross = LSTMBuffer(D, K, nh)
        else:
            self.m_cross = None
        # AccumCCoT and DirectCCoT are the K == 0 mechanisms; accum wins if both
        # are somehow set, matching the pre-retirement selection order.
        self.accum = AccumCCoT(D, n_vec_legacy, nh, max_vec_legacy) \
            if (legacy_accum and K == 0) else None
        self.ccot_direct = DirectCCoT(D) \
            if (legacy_direct and K == 0 and self.accum is None) else None
        if legacy_accum or legacy_gated or legacy_direct:
            which = ("accum_ccot" if legacy_accum else
                     "gated_accum" if legacy_gated else "ccot_direct")
            print(f"[cortex] LEGACY buffer active: {which}. Superseded by "
                  f"--cortex.prefix_memory; supported for loading and evaluating "
                  f"Track-A / B1 checkpoints, not for new training runs.")
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
        # The Z channel.  Read here (not lazily) because it changes a PARAMETER
        # SET -- the gated buffer allocates a second gate for it -- and a flag
        # that changed the parameter set after the optimizer was built would
        # leave the new tensors untrained with no symptom.
        self.latent_carry = bool(getattr(config, "latent_carry", False))
        self.latent_depth_rule = str(
            getattr(config, "latent_depth_rule", "absolute") or "absolute")
        self.latent_depth_lo = int(getattr(config, "latent_depth_lo", 2))
        self.latent_depth_hi = int(getattr(config, "latent_depth_hi", 9))
        self.latent_renorm = str(getattr(config, "latent_renorm", "none") or "none")
        if self.latent_depth_rule not in ("absolute", "relative"):
            raise ValueError(
                f"cortex.latent_depth_rule must be 'absolute' or 'relative'; "
                f"got {self.latent_depth_rule!r}.  P0.7 "
                f"(evals/diag_depth_band.py) decides which.")
        if self.latent_renorm not in ("none", "s0"):
            raise ValueError(
                f"cortex.latent_renorm must be 'none' or 's0'; got "
                f"{self.latent_renorm!r}")
        if self.latent_carry and not pmode:
            raise ValueError(
                "cortex.latent_carry needs a prefix buffer (--cortex."
                "prefix_memory accum|gated).  Z is read by substituting into "
                "the CARRIED COLUMNS of s0, and without a prefix buffer there "
                "are no carried columns -- the read would be a no-op and the "
                "arm would train a write nothing consumes.")
        if pmode == "accum":
            self.prefix = PrefixAccumBuffer(
                D, n_vec, int(getattr(config, "accum_max", 128)),
                carries_latent=self.latent_carry)
        elif pmode == "gated":
            # gate_slots 0 => the pre-P1.0 shape (K == W).  Everything else
            # defaults to the P1.0 recommendation; see the PrefixGatedBuffer
            # docstring for why, and train.py for the cross_chunks assert that
            # stops a configuration where the gate never fires.
            self.prefix = PrefixGatedBuffer(
                D, n_vec,
                n_slots=int(getattr(config, "gate_slots", 0) or 0) or None,
                route=str(getattr(config, "gate_route", "ring") or "ring"),
                gate_norm=str(getattr(config, "gate_norm", "tanh") or "tanh"),
                gate_init=str(getattr(config, "gate_init", "zero") or "zero"),
                fill=str(getattr(config, "gate_fill", "grow") or "grow"),
                forget_bias_init=float(
                    getattr(config, "gate_forget_bias", 1.0) or 1.0),
                carries_latent=self.latent_carry)
        elif pmode:
            raise ValueError(f"prefix_memory must be '', 'accum' or 'gated'; got {pmode!r}")
        else:
            self.prefix = None
        if self.prefix is not None:
            if legacy_accum or legacy_gated or legacy_direct:
                raise ValueError(
                    "cortex.prefix_memory cannot be combined with the legacy "
                    "accum_ccot / gated_accum / ccot_direct flags — they are "
                    "different cross-segment mechanisms and prefix mode would "
                    "silently win.  Clear the legacy flag from config.json.")
            # Prefix mode owns the cross-segment state exclusively.
            self.m_cross = self.accum = self.ccot_direct = None
            # Resolve now, at build time, so a missing or invalid seed token fails
            # before the data loader and the wandb run exist rather than on the
            # first forward of a queued 48h job.
            self.summary_init_token = resolve_summary_init_token(config)
            # prefix_pos / prefix_eos_reset: both keys stay in the 16-flag persist
            # list and must keep LOADING — they are in every memory checkpoint's
            # config.json and the graft is their load path.  The BRANCHES they
            # used to select are gone, retired 2026-09-15 after
            # pace/check_tier3_compat.sh came back CLEAR across 148 surviving
            # config.json files on scratch: nothing carries the old values, and
            # the cancelled 4vop2ym8 run they existed to reproduce was rm -rf'd.
            # So read the key and ASSERT the modern value rather than branch on
            # it.  A config that still carries an old value is a real surprise
            # and should stop the job, not silently select a retired code path.
            self.prefix_pos = str(getattr(config, "prefix_pos", "tail") or "tail").lower()
            if self.prefix_pos != "tail":
                raise ValueError(
                    f"cortex.prefix_pos must be 'tail'; got {self.prefix_pos!r}.  "
                    "The 'zero' layout was retired 2026-09-15: it put the summary "
                    "slots at position 0 at the END of the sequence, so under RoPE "
                    "they queried every real token at a NEGATIVE relative offset.  "
                    "See the prefix_pack docstring for why that is not "
                    "AutoCompressor's pad-position trick.")
            self.prefix_eos_reset = bool(getattr(config, "prefix_eos_reset", False))
            if self.prefix_eos_reset:
                raise ValueError(
                    "cortex.prefix_eos_reset=true was retired 2026-09-15.  It zeroed "
                    "the WHOLE incoming carry on any chunk containing an EOS, which "
                    "switched the read OFF for roughly 60% of B2's training chunks "
                    "while every eval saw a live carry.  See the _carried_state "
                    "docstring.")
        else:
            self.summary_init_token = -1
            self.prefix_pos = "tail"
            self.prefix_eos_reset = False

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

        # The 2x2's Z-ablation hook.  A plain attribute rather than a config
        # key: it is set per CHUNK by the eval and must never persist into a
        # checkpoint.  Declared here so `cortex.latent_read_null = ...` is
        # setting something that exists, and so the E-only build carries the
        # same surface (evals/eval_carry_2x2.py duck-types on latent_carry to
        # decide whether a 2x2 is measurable at all, and refuses rather than
        # reporting a 1x2 as one).
        self.latent_read_null = None
        # Counters for the E/Z read asymmetry (see latent_init).  OUTSIDE
        # _reset_runtime on purpose: the useful quantity is the fraction over
        # BATCHES, and _reset_runtime runs once per forward.
        self._z_read_live = False
        self._z_read_n = 0
        self._z_read_live_n = 0
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
                    emb_scale: float = 1.0,
                    write: bool = True,
                    read: bool = True):
        """Splice the carry into the token stream, AutoCompressor-style.

        Layout: [carried vectors | real tokens | summary slots].  Under the
        model's causal mask this reproduces auto_compressor.py:85 exactly —
        real tokens see every carried vector, and the summary slots at the end
        see the whole chunk.  The summary slots also see each other causally,
        which is the ONLY thing that stops n_vec identically-initialised slots
        from collapsing into copies of one vector.

        Positions: carried vectors at 0, real tokens at 1..S, and the summary
        slots at S+1..S+n_vec, i.e. continuing the chunk's own numbering.  This
        is the only layout; prefix_pos is asserted to be "tail" at build time.

        The old layout ("zero", retired 2026-09-15) put the summary slots at
        position 0 as well, on
        the theory that this was our RoPE analog of AutoCompressor's pad-position
        trick.  It is not.  That trick lives in
        OPTLearnedPositionalEmbeddingWithPadding, and OPT positions are ADDITIVE,
        so "no position" means adding a zero vector.  AutoCompressor's own RoPE
        model (LlamaAutoCompressorModel) overrides nothing and uses plain
        contiguous positions.  Under RoPE, a slot at position 0 sitting at the
        END of the sequence queries every real token at a NEGATIVE relative
        offset (-1 .. -S) — a regime a causal LM never sees in training, since
        pos_q >= pos_k always holds.  Measured on a pretrained RoPE LM, that
        layout also collapses the slots onto each other: centred slot-slot
        cosine 0.94 and effective rank 2.1 of 32, versus 0.22 / 4.2 with the
        tail layout.  n_vec was buying ~2 independent directions, not 32.

        The CARRIED vectors stay at position 0 under both layouts.  They sit at
        the FRONT, so real tokens already query them at positive offsets (+1..+S)
        — the read side was never in the untrained regime — and keeping the real
        tokens at 1..S is what lets a cached decode step continue the prefill's
        numbering without knowing how many carried columns the prefill had
        (see the read=False note below).  The cost is that the carry has no
        recency order in position space; that is a separate, unmeasured question.

        Every index stays inside the trained window: n_pre + S + n_vec columns
        span positions 0..S+n_vec (1056 at S=1024, n_vec=32), against a 1024
        continued-pretraining window and a ~1.5k usable range.

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

        write / read exist for KV-CACHED GENERATION and both default to True,
        so the training call site is unchanged (bit-identical) by construction.

          write=False  omit the trailing summary slots.  During decoding the
            buffer is held fixed and the write is discarded anyway, so the
            slots are pure wasted compute — but with a KV cache they are worse
            than wasted.  A cached single-token query runs with is_causal=False
            (raven_modeling_minimal_olmo.py:450) and therefore attends to EVERY
            cached key; slots left in the cache would silently be read by the
            generated tokens, which the uncached causal mask never permits.
          read=False   omit the carried vectors.  Only for INCREMENTAL cached
            steps, where the carry is already resident in the cache from the
            prefill call and re-splicing it would double-count it and corrupt
            the cache's absolute position keys.

        The +1 position shift is applied whenever prefix memory is active, even
        when nothing is spliced, so that a cached decode step continues the
        prefill's 1..S numbering rather than restarting at S.

        Returns (packed_embeds, packed_position_ids, n_prefix, n_summary).
        """
        if self.prefix is None:
            return input_embeds, position_ids, 0, 0

        B, S, _ = input_embeds.shape
        parts, n_pre, n_sum = [], 0, 0

        if read:
            state = self._carried_state()
            if state is not None and state.shape[1] > 0:
                state = state.to(device=input_embeds.device, dtype=input_embeds.dtype)
                # ONLY THE E HALF ENTERS THE TOKEN STREAM.  Z is not an
                # embedding and is never spliced as a column: it is read by
                # substitution into `s0` at these very columns (latent_init),
                # which is what makes the read free.  Splicing it here instead
                # would double the sequence length and put trajectory deltas at
                # norm ~0.35 next to hidden states at norm ~171, where attention
                # would simply not see them.
                e_state, _ = self.prefix.split_channels(state)
                parts.append(e_state)
                n_pre = e_state.shape[1]
                # THE SCALE THE s0 SUBSTITUTION COMPETES WITH.  RED 11: the
                # s0 sweep's response turns on where a substituted row's norm
                # reaches THIS number, because `core_block_forward` runs
                # `adapter(cat([x, input_embeds]))` and the carried E row
                # lives in the `input_embeds` half of the very same columns
                # the Z read substitutes into.  Above it the injection is
                # drowning E, not being read at s0, and the instrument cannot
                # tell those apart without the number.  Diagnostic only --
                # nothing branches on it.
                self._e_carried_norm = float(
                    e_state.detach().float().flatten(0, -2).norm(dim=-1).mean())
        parts.append(input_embeds)

        if write:
            slots = self.prefix.summary_slots(B, input_embeds.dtype, input_embeds.device)
            if emb_scale != 1.0:
                slots = slots * emb_scale
            parts.append(slots)
            n_sum = slots.shape[1]

        pos = position_ids[:, :S] + 1
        if n_sum and pos.shape[1] > 0:
            # Continue the chunk's numbering, so every summary->token offset is
            # positive.  Unconditional since 2026-09-15 — prefix_pos is asserted
            # to be 'tail' at build time, so there is no other layout to select.  Derived from the LAST real position rather than from S
            # so a cached prefill (whose position_ids do not start at 0) stays
            # consistent.
            sum_pos = pos[:, -1:] + torch.arange(
                1, n_sum + 1, device=pos.device, dtype=pos.dtype).unsqueeze(0)
        else:
            sum_pos = pos.new_zeros(pos.shape[0], n_sum)
        pos = torch.cat([pos.new_zeros(pos.shape[0], n_pre), pos, sum_pos], dim=1)
        packed = parts[0] if len(parts) == 1 else torch.cat(parts, dim=1)
        # The Z read and the Z write both address the packed layout by column,
        # and iterate_forward does not receive it -- so record it here, in the
        # one function that computes it.
        self._n_pre, self._n_sum = n_pre, n_sum
        return packed, pos, n_pre, n_sum

    def prefix_unpack(self, x: torch.Tensor, n_pre: int, n_sum: int):
        """Split the post-ln_f states back apart.

        The summary slots' final hidden states ARE the new carry — no
        extraction module, no tanh/LN bounding (auto_compressor.py:123).

        n_sum == 0 is the write=False (generation) case: there is no new carry,
        but the n_pre carried columns must STILL be stripped or the head would
        emit logits for them.  Harmless for greedy decoding, which only reads
        logits[:, -1] and the carry sits at the front — but it silently breaks
        any loss or full-sequence scoring, so strip unconditionally.
        Returns (real_token_states, new_carry).
        """
        if self.prefix is None:
            return x, None
        end = x.shape[1] - n_sum
        if n_sum == 0:
            return x[:, n_pre:], None
        real, new_vecs = x[:, n_pre:end], x[:, end:]
        # The write is NOT masked in prefix mode.  Zeroing a lane whose open
        # suffix is empty (the chunk ends ON an EOS) used to happen here under
        # prefix_eos_reset, and it was actively harmful: pool_mask never
        # restricts the prefix write anyway — the summary slots are read by the
        # model's own causal attention over the WHOLE chunk — so it discarded a
        # perfectly good summary and appended a row of zeros to an accumulating
        # state, which then got spliced back as zero-valued attention targets.
        # Rare (it needs the chunk's last token to be an EOS) but silent.
        # Branch retired 2026-09-15; `_valid_write` is still computed in begin()
        # and still consumed by the bolt-on buffer path below.
        return real, self.prefix.merge(self._carried_state(), new_vecs,
                                       self.latent_write())

    def _carried_state(self) -> Optional[torch.Tensor]:
        """The incoming carry, as the prefix splice and the merge should see it.

        Until 2026-08-04 this unconditionally zeroed the carry for any lane whose
        chunk contained an EOS.  That is far too blunt for prefix memory and it
        was switching the read OFF for the majority of B2's training chunks:

          * `write_reset` is `has_eos` for the WHOLE chunk, so one document
            boundary anywhere zeroed the carry for every position — including the
            prefix of tokens that legitimately continue the previous document.
            tools/prepare_packed_dataset.py joins documents with EOS separators,
            so on FineWeb-Edu (~1.1k-token documents, 1024-token chunks) roughly
            60% of chunks lost their carry entirely; at 500-token documents, 87%.
          * the per-position mask that HAS the right semantics
            (`cross_read_mask`: positions <= the first EOS may read) is computed
            in begin() and then discarded here — only read_into consumes it, and
            that is a no-op in prefix mode.  The bolt-on buffers used it, so this
            was a regression introduced by the prefix rewrite, not a carry-over.
          * zeroing does not REMOVE the columns.  A zero key scores 0 against
            every query — a mid-range logit, not -inf — so 64-96 dead columns
            went on absorbing ~3-5% of the softmax mass at every layer.
          * the evals never pass eos_mask, so eval always saw a live carry while
            training mostly did not.

        Default is now OFF, i.e. the carry is read by the whole chunk.  The
        justification is that the backbone already does exactly this: the modeling
        file runs with prepared_attn_mask=None and is_causal=True, so base
        attention is UNMASKED across document boundaries inside a chunk.  Resetting
        the memory at boundaries held it to a stricter standard than the model it
        is grafted into.  AutoCompressor and RMT likewise carry across packed
        boundaries.

        prefix_eos_reset=True used to restore the old behaviour.  That branch was
        RETIRED 2026-09-15 (pace/check_tier3_compat.sh clear across 148 configs);
        the flag still loads and is now asserted false at build time.  The
        principled middle option — a per-position block mask so post-boundary
        tokens cannot attend to the carry columns — needs the flex_attention path
        that the modeling file currently leaves disabled; see the note in
        prefix_pack.

        The WRITE side is likewise unmasked in prefix mode: `pool_mask` and
        `_valid_write` are still computed in begin() and still consumed by the
        bolt-on buffer path, but prefix_unpack applies neither — see the comment
        there.
        """
        return self._cross_buf

    # ── per-call runtime ────────────────────────────────────────────────────
    def _reset_runtime(self) -> None:
        self._cross_buf:       Optional[torch.Tensor] = None  # carried M_cross [B,K,D]/[B,1,D]/[B,N,D]
        self._cross_read_mask: Optional[torch.Tensor] = None  # [B,S,1]
        self._pool_mask:       Optional[torch.Tensor] = None  # [B,S] bool
        self._write_reset:     Optional[torch.Tensor] = None  # [B] bool
        self._valid_write:     Optional[torch.Tensor] = None  # [B] bool
        self._iter_buf:        Optional[torch.Tensor] = None  # [B*S,Ki,D]
        # --- Z channel per-call runtime ---------------------------------
        self._n_pre = self._n_sum = 0     # packed layout, set by prefix_pack
        self._z_prev:  Optional[torch.Tensor] = None   # [B,n_sum,D], s_{t-1}
        self._z_tape:  list = []          # index t-1 -> d_t at the summary cols
        self._z_grad:  list = []          # was step t inside the gradient window
        self._z_s0_scale: Optional[float] = None       # ||s0|| per TOKEN (row L2), fp32
        self._z_s0_rms:   Optional[float] = None       # s0 per-ELEMENT rms, fp32
        self._e_carried_norm: Optional[float] = None   # ||E carried row||, fp32

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

    def iter_write(self, x: torch.Tensor,
                   current_step: Optional[int] = None) -> None:
        """Write each position's state into its own M_iter slots, and tape the
        recurrence trajectory for the Z channel.  Called after the core layers,
        i.e. at the end of one loop iteration.

        `current_step` is optional so that a checkpoint carrying an OLDER COPY
        of the modeling file (which calls this with one argument) keeps loading
        instead of dying on a TypeError.  It is not used for the tape -- the
        tape is positional and appends in order -- so nothing silently depends
        on a value an old file cannot supply.
        """
        if self.m_iter is not None:
            B, S, D = x.shape
            if self._iter_buf is None:
                self._iter_buf = x.new_zeros(B * S, self.memory_slots_iter, D)
            self._iter_buf = self.m_iter.write(x.reshape(B * S, 1, D),
                                               self._iter_buf)
        self._tape_latent(x)

    # ── the Z channel ──────────────────────────────────────────────────────
    #
    # THE WRITE IS THE TRAJECTORY, NOT THE ENDPOINT, and that is settled on two
    # independent grounds.  Information: the endpoint is what E already carries
    # (post-`ln_f`), so writing s_T into Z would carry the same conclusion twice
    # in two coordinate systems.  Scale: measured fp32 on the B2 checkpoint,
    # ||s_T|| is 28.1x the `trunc_normal_` noise Z substitutes for, while the
    # staggered delta at T=8 -- the trained depth -- is 0.90x it.  The endpoint
    # would need a renorm; the deltas are in distribution as drawn.
    #
    # THE TAPE IS EVERY STEP, AND THE DEPTH RULE SELECTS FROM IT AFTERWARDS.
    # The obvious alternative is to capture only the depths the rule asks for,
    # which needs T up front and therefore another modeling-file hook.  Taping
    # all of them costs T x [B, n_sum, D] -- about 1 MB at T=8, n_sum=16,
    # D=2048 -- because ONLY THE SUMMARY COLUMNS ARE TAPED, not the sequence.
    # In exchange, the slot->depth rule becomes a pure post-hoc selection, so
    # P0.7's answer is a config change and never a modeling-file change.

    def _tape_latent(self, x: torch.Tensor) -> None:
        """Record d_t = s_t - s_{t-1} at the summary columns.

        NOT DETACHED, and that is the whole point.  `latent_states` in the
        modeling file is `x.clone().detach()`, and reusing it here would give a
        gradient-free carry sitting behind a healthy loss curve -- bug class 2
        in a new hat.  The slice taken here is live, so the write path is on the
        loss for every step inside the gradient window.

        Which steps ARE inside it is not a detail: iterate_forward runs the
        first `num_steps_no_grad` iterations under torch.no_grad() and those run
        FIRST, so the trainable region is always the LAST `mean_backprop_depth`
        iterations.  At mr8 (where the P1 cells branch) that is the whole loop;
        at mr32 it is steps 25-32, i.e. exactly the saturated ones, and Z
        becomes a FROZEN FEATURE EXTRACTOR -- the model can learn to USE it but
        never to SHAPE it, which is the mirror of the frozen-READ failure that
        cost x0.90 -> x1.33.  `torch.is_grad_enabled()` is recorded per step so
        that is a REPORTED NUMBER (latent_write_grad_frac) and not a discovery.
        """
        if not self.latent_carry or self.prefix is None or not self._n_sum:
            return
        cur = x[:, -self._n_sum:]
        if self._z_prev is not None:
            self._z_tape.append(cur - self._z_prev)
            self._z_grad.append(bool(torch.is_grad_enabled()))
        self._z_prev = cur

    def latent_init(self, s0: torch.Tensor,
                    num_steps_no_grad: Optional[int] = None) -> torch.Tensor:
        """THE Z READ.  Substitute the carried trajectory into `s0`'s carried
        columns, replacing the noise `initialize_state` put there.

        Zero new parameters, nothing zero-initialised, sequence length
        unchanged.  This is the property the whole design rests on: the earlier
        latent buffer failed because its read was a MODULE that had to be
        discovered from an exactly-zero injection (x0.90), and unfreezing that
        read is what B2 measured at x1.33.  Here the read is a substitution into
        a field that already exists and is already consumed -- the carried
        columns are queried by real tokens at positive offsets, and
        `core_block_forward`'s `adapter(cat([x, input_embeds]))` is a pretrained
        consumer of exactly this concatenation, on every one of the T
        iterations.

        Also captures s_0 at the summary columns, so the first taped delta is
        d_1 = s_1 - s_0 and is measured against the state the loop ACTUALLY
        started from (post-substitution), not against the noise that was
        discarded.

        THE E/Z READ ASYMMETRY — MEASURED 2026-09-16, and it is not in any
        earlier design note.  `num_steps_no_grad` is taken so this can be
        reported rather than discovered:

            a SINGLE no-grad step cuts Z's read gradient to EXACTLY zero.

        The reason is structural, not a tuning problem.  E re-enters the loop on
        every iteration through `input_embeds` -- `core_block_forward` does
        `adapter(cat([x, input_embeds]))`, and the carried E columns live in
        `input_embeds` -- so E's read is refreshed inside the gradient window no
        matter how long the no-grad prefix is.  Z enters ONCE, at `s0`, and
        `iterate_forward` runs the no-grad iterations FIRST; anything downstream
        of a `torch.no_grad()` block is detached, so with n >= 1 there is no path
        from the carried Z to the loss at all.  Measured on the real loop: E's
        gate gradient is ~4e-2 at every split, Z's is 6.1e-2 at (0, T) and
        EXACTLY 0.0 at (1, T-1), (2, T-2) and (3, T-3).

        What this costs.  `randomized_iteration_sampler` gives n = 0 exactly when
        the sampled p <= mean_backprop_depth, so at mr8 / depth 8 roughly half of
        training batches carry a Z read gradient and half carry none; at mr32 /
        depth 8, essentially none do.  It CANNOT be configured away -- p has a
        Poisson tail, so no finite mean_backprop_depth makes n = 0 always -- and
        raising mean_backprop_depth only raises the fraction, at the cost of
        graph memory.

        This does not make Z inert: the forward uses the carried Z on EVERY
        batch, so the model still benefits from it, and the write still trains
        on the steps inside the window.  What is reduced is how much gradient
        signal shapes the READ side.  `latent_read_grad_frac` is the number;
        put it in the pre-registration, because it changes what a null result
        for Z means.
        """
        if self.prefix is None:
            return s0
        if self.latent_carry and num_steps_no_grad is not None:
            self._z_read_live = int(num_steps_no_grad) == 0
            self._z_read_n += 1
            self._z_read_live_n += int(self._z_read_live)
        if self.latent_carry and self._n_sum:
            # s0 at the summary columns -- the tape's starting point.  Taken
            # before any substitution so that d_1 is a real first step.
            self._z_prev = s0[:, -self._n_sum:]
            # TWO SCALES, TWO UNITS, AND THEY DIFFER BY sqrt(D) = 45.25 AT
            # D=2048.  `_z_s0_scale` is a per-token L2 ROW NORM (what a reader
            # means by "the size of s0"); `_z_s0_rms` is the PER-ELEMENT rms,
            # which is what `normal_(0, std)` in `_null_latent` takes.  RED 11
            # was `diag_s0_sensitivity.py` feeding the first into the second,
            # so every row of its sweep injected 45x the norm it claimed and
            # the flat region looked far narrower than it is.  Both are
            # recorded here, suffixed, so the next consumer has to pick.
            f32 = s0.detach().float()
            self._z_s0_scale = float(f32.flatten(0, -2).norm(dim=-1).mean())
            self._z_s0_rms = float(f32.pow(2).mean().sqrt())
        n_pre = self._n_pre
        if not self.latent_carry or not n_pre:
            return s0

        head = s0[:, :n_pre]
        null = getattr(self, "latent_read_null", None)
        if null is not None:
            # Z's null is FRESH NOISE at s0's own scale -- the model's trained
            # default for these columns, in distribution, identical column
            # count.  Contrast E's null, which is zeros: a zero key still scores
            # a mid-range logit and absorbs ~3-5% of the softmax mass, so the E
            # axis of the 2x2 is the confounded one and the Z axis is clean.
            # The asymmetry is stated in eval_carry_2x2.py and must be stated in
            # the writeup too.
            z = self._null_latent(null, head)
        else:
            _, z = self.prefix.split_channels(self._carried_state())
            if z is None:
                return s0
            z = z.to(device=s0.device, dtype=s0.dtype)
            if z.shape[1] != n_pre:
                raise ValueError(
                    f"carried Z has {z.shape[1]} rows but {n_pre} carried "
                    f"columns were spliced.  E and Z share the ring pointer and "
                    f"must share the row count; a mismatch means the two "
                    f"channels were merged at different widths.")
            # Rows never written carry exactly zero (see
            # PrefixGatedBuffer._slot_init_block).  A zero column in s0 is NOT
            # the trained default -- the model has only ever seen noise there --
            # so fall back to the noise rather than substituting a dead field.
            unwritten = (z.detach().abs().sum(dim=-1, keepdim=True) == 0)
            if bool(unwritten.any()):
                z = torch.where(unwritten, head, z)
        if self.latent_renorm == "s0":
            z = self._renorm_to(z, head)
        return torch.cat([z, s0[:, n_pre:]], dim=1)

    def _null_latent(self, null, head: torch.Tensor) -> torch.Tensor:
        """Z's ablation null.  `latent_read_null` is the contract
        evals/eval_carry_2x2.py sets: ("noise", std, seed)."""
        kind = null[0] if isinstance(null, (tuple, list)) else str(null)
        if kind != "noise":
            raise ValueError(
                f"latent_read_null kind must be 'noise'; got {kind!r}.  Zeros "
                "are E's null and the wrong one for Z -- a zeroed latent field "
                "is not a state the model has ever seen, while noise is its own "
                "trained default for these columns.")
        std = float(null[1]) if len(null) > 1 else 0.02
        seed = int(null[2]) if len(null) > 2 else 0
        g = torch.Generator(device="cpu").manual_seed(seed)
        n = torch.empty(head.shape, dtype=torch.float32).normal_(
            0.0, std, generator=g)
        return n.to(device=head.device, dtype=head.dtype)

    @staticmethod
    def _renorm_to(z: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        """Rescale each Z row to the per-row norm of the field it replaces.

        NOT THE DEFAULT, on a measurement: at T=8 the staggered delta is already
        0.90x the noise, so the substitution is in distribution as drawn and a
        renorm would be a parameter-free transform applied for no reason.  It
        exists because the ratio is strongly depth-dependent (23.8x at t=1, 0.06x
        at t=32), so a write rule that lands in the saturated tail -- or a model
        run far outside the depth it trained at -- would need it.  Decide it with
        evals/diag_depth_band.py, do not leave it to taste.
        """
        zn = z.float().norm(dim=-1, keepdim=True).clamp_min(1e-12)
        rn = ref.detach().float().norm(dim=-1, keepdim=True)
        return (z.float() * (rn / zn)).to(z.dtype)

    def latent_write(self) -> Optional[torch.Tensor]:
        """Harvest the tape into one [B, n_vec, D] write, one depth per slot.

        The n_vec slots are COLUMNS of a single forward, each with its own full
        trajectory, so this is an n_vec x (T-1) grid and exactly one cell per
        row is taken: slot j contributes column j of the delta at depth k_j.  A
        depth that two slots share is therefore TWO INDEPENDENT VIEWS of it --
        different sequence columns, different trajectories -- and not a
        duplicate, which is why sampling an 8-wide band with 16 slots is the
        design and not a waste.

        Staggered, not uniform.  Uniform depth would give semantically identical
        (E_j, Z_j) pairs but sample one depth and turn the depth-slice ablation
        into a BETWEEN-run comparison (several training runs instead of one).
        Staggered buys the depth axis from 16 rows and keeps the ablation
        WITHIN-run; under the ring, write vector j always lands at rows
        congruent to j (mod W), so the convention is stable and addressable.
        """
        if not self.latent_carry or self.prefix is None:
            return None
        if not self._z_tape:
            raise RuntimeError(
                "latent_carry is on but the trajectory tape is EMPTY: "
                "iter_write never fired, so the checkpoint's copy of the "
                "modeling file predates the Z hooks and this run would train a "
                "dual-channel buffer whose Z half is never written.  Re-run "
                "tools/prepare_cortex_checkpoint.py against the base.  (Same "
                "silent class as the prefix_pack failure that ran 9 hours with "
                "no cross-segment memory.)")
        # The tape holds T deltas: d_1 = s_1 - s_0 (s_0 captured by latent_init)
        # through d_T.  The depth map is asked for the LOOP LENGTH and clamps to
        # T-1, so d_T is deliberately never selected -- that is the rule as
        # written down (rule B is specified as k_j = round(f_j * (T-1))), and
        # matching the specification beats reclaiming one delta at the saturated
        # end of the trajectory, which is the end P0.1 says carries least.
        T_loop = len(self._z_tape)
        W = self.prefix.n_vec
        depths = self.latent_depth_map(T_loop, W)
        rows = []
        for j, k in enumerate(depths):
            d = self._z_tape[min(max(k, 1), T_loop) - 1]
            rows.append(d[:, j:j + 1])
        return torch.cat(rows, dim=1)

    def latent_depth_map(self, T: int, n_slots: int) -> list:
        """slot -> loop depth.  The two rules P0.7 exists to choose between.

        Mirrors evals/diag_depth_band.slot_depth_map; that file is the probe and
        this is the consumer, and the two are checked against each other in
        tests/test_latent_carry.py rather than trusted to stay in step.
        """
        lo, hi = self.latent_depth_lo, self.latent_depth_hi
        if T < 2:
            return [1] * n_slots
        if self.latent_depth_rule == "absolute":
            hi_c = min(hi, T - 1)
            lo_c = min(lo, hi_c)
            span = hi_c - lo_c + 1
            return [lo_c + (j % span) for j in range(n_slots)]
        f_lo = lo / max(hi, 1)
        return [max(1, min(T - 1, round((f_lo + (1.0 - f_lo)
                                         * j / max(n_slots - 1, 1)) * (T - 1))))
                for j in range(n_slots)]

    @property
    def latent_read_grad_frac(self) -> float:
        """Share of forwards SO FAR whose Z read had any gradient path at all.

        Counted across forwards rather than reset per call, because the quantity
        that matters is a training-run average: on a given batch the read is
        either fully live or fully dead (see latent_init), and it is the mix
        over batches that decides how much signal the read side gets.

        Reads 0.0 before any forward with a known split -- the modeling file has
        to pass `num_steps_no_grad`, so an old snapshot reports 0.0 rather than a
        wrong number.  `latent_read_measured` says which.
        """
        if not self._z_read_n:
            return 0.0
        return self._z_read_live_n / self._z_read_n

    @property
    def latent_read_measured(self) -> bool:
        """False when the modeling file never told us the split, so
        latent_read_grad_frac's 0.0 means 'unknown' and not 'dead'."""
        return self._z_read_n > 0

    @property
    def latent_write_grad_frac(self) -> float:
        """Share of the taped steps that were inside the gradient window.

        0.0 means Z is a frozen feature extractor for this batch.  That is a
        legitimate configuration -- at mr32 it is unavoidable without raising
        mean_backprop_depth -- but it changes what a null result for Z MEANS,
        so it belongs in the pre-registration and in the training log, not in a
        post-hoc explanation.
        """
        if not self._z_grad:
            return 0.0
        return sum(1 for g in self._z_grad if g) / len(self._z_grad)

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


def _unwrap_for_reset(model):
    """Reach the real module through DDP / torch.compile wrappers."""
    m = model
    if hasattr(m, "module"):
        m = m.module
    if hasattr(m, "_orig_mod"):
        m = m._orig_mod
    return m


def reset_cortex_graft_init(model, log=None):
    """Undo post_init's clobbering of the cortex graft's initialization, on the
    live model after from_pretrained.

    LIVES HERE, NOT IN train.py (moved 2026-09-16).  Every consumer that builds
    the graft with from_pretrained needs it, not just the trainer:
    tools/smoke_prefix_real.py was running the buffer with post_init's values --
    measured at forget_bias = -2.2e12, i.e. fg identically ZERO, a gate that
    forgets everything on every write -- while asserting that the gate behaved
    like the run's.  It could not import train.py to fix that (train.py needs
    wandb and the smoke runs on a login node), and a second copy of this logic
    would drift.  train.py keeps a thin wrapper that supplies its rank-0 guard.

    `log` is a print-like callable or None (silent); train.py passes print only
    on rank 0.

    RavenForCausalLM.__init__ builds CortexMemory (designed inits so the memory
    read is a no-op and step-0 == the base model) and THEN calls post_init().
    HF's _init_weights treats every cortex tensor as a freshly-'missing' key (the
    "newly initialized: ['cortex.h_T_proj.weight', 'cortex.m_cross.cand_ln1.bias',
    ...]" load warning) and re-initializes it with the raven DEPTH-SCALED scheme.
    The graft modules have no valid layer index, so that scheme hands them an
    effectively-infinite std -> NON-FINITE weights.  That is the confirmed
    forward-nan source, and it hits EVERY cortex op in turn (localizer found
    h_T_proj first, then m_cross.cand_ln1, ...), so restoring hand-picked tensors
    is whack-a-mole.  Instead reset the WHOLE cortex subtree:
      (1) every submodule back to its nn default (Linear->kaiming, LayerNorm->
          weight 1/bias 0) — finite, in place, dtype/device preserved;
      (2) re-apply the few explicit designed inits the graft sets by hand.
    Mirrors cortex_graft.CortexMemory + cortex_memory.buffers; skipped on
    --resume (a resumed ckpt carries trained, not fresh, cortex weights).

    LoRA (cortex_lora) MUST be re-initialized here too — the original "bare
    Parameters _init_weights never touches, B stays 0" analysis was WRONG under
    the meta-device from_pretrained path: the skeleton is built on meta (the
    __init__ kaiming/zeros are no-ops), missing keys are materialized via
    to_empty() = UNINITIALIZED memory, and _init_weights skips ParameterDicts —
    so A/B keep whatever bytes the allocator hands them.  Driver-zeroed fresh
    pages make MOST tensors read as zeros; recycled blocks carry ~1e19 garbage,
    with run-to-run membership.  Root cause of the entire rung1b failure family:
    garbage in an A row -> finite grad_B ~1e19 -> inf fp32 grad-norm every step
    (healthy forward, B~0); garbage in a B -> forward nan from step 1.  Note the
    non-finite sweep below can NOT catch this — the garbage is FINITE."""
    unwrapped = _unwrap_for_reset(model)
    lora = getattr(unwrapped, "cortex_lora", None)
    if lora is not None:
        for A in lora.A.values():
            torch.nn.init.kaiming_uniform_(A, a=math.sqrt(5))
        for B in lora.B.values():
            torch.nn.init.zeros_(B)
        if log is not None:
            log(f"[cortex] re-initialized {len(lora.A)} LoRA A/B pairs "
                  f"(A~kaiming, B=0) — undo to_empty() garbage from the "
                  f"meta-device load path")
    cortex = getattr(unwrapped, "cortex", None)
    if cortex is None:
        return
    # (1) undo the non-finite clobber: nn defaults for every submodule.
    n_reset = 0
    for m in cortex.modules():
        if m is not cortex and callable(getattr(m, "reset_parameters", None)):
            m.reset_parameters(); n_reset += 1
    # (2) re-apply the graft's explicit designed inits (mirror the source).
    def _read_init(w, tag):
        torch.nn.init.zeros_(w)
        return f"{tag}=0"
    fixed = []
    if getattr(cortex, "h_T_proj", None) is not None:
        torch.nn.init.eye_(cortex.h_T_proj.weight); fixed.append("h_T_proj=eye")
    for buf_name in ("m_cross", "m_iter"):          # LSTMBuffer / GatedAccumBuffer
        buf = getattr(cortex, buf_name, None)
        if buf is None:
            continue
        read_tag = _read_init(buf.out_proj.weight, "out_proj")  # memory read
        if hasattr(buf, "slot_emb"):                # LSTMBuffer
            torch.nn.init.normal_(buf.slot_emb, std=0.02)
            emb_tag = "slot_emb~N"
        else:                                       # GatedAccumBuffer (vec_emb queries)
            torch.nn.init.normal_(buf.vec_emb, std=0.02)
            emb_tag = "vec_emb~N"
        torch.nn.init.ones_(buf.forget_bias)                  # LM2 §3.3 forget bias +1
        torch.nn.init.zeros_(buf.input_bias)
        fixed.append(f"{buf_name}.[{read_tag},{emb_tag},forget_bias=1,input_bias=0]")
    ccot = getattr(cortex, "ccot_direct", None)                # DirectCCoT (K=0)
    if ccot is not None:
        torch.nn.init.eye_(ccot.state_proj.weight); fixed.append("ccot.state_proj=eye")
        fixed.append("ccot." + _read_init(ccot.in_proj.weight, "in_proj"))
    acc = getattr(cortex, "accum", None)                       # AccumCCoT
    if acc is not None:
        torch.nn.init.normal_(acc.vec_emb, std=0.02)
        fixed.append("accum.[vec_emb~N," + _read_init(acc.out_proj.weight, "out_proj") + "]")
    pre = getattr(cortex, "prefix", None)          # PrefixAccum / PrefixGated
    if pre is not None:
        # summary_emb here is a FINITE placeholder only — it is re-seeded from
        # wte[eos] on the first forward (see the modeling file).  This function
        # runs on fresh builds ONLY (caller guards on resume_path is None), so
        # clearing summary_seeded is both safe and necessary: the base graft dir
        # carries no cortex tensors, and a bool buffer materialised from meta
        # can come back as garbage -> read True -> seeding silently skipped.
        torch.nn.init.normal_(pre.summary_emb, std=0.02)
        pre.summary_seeded.fill_(False)
        tags = ["summary_emb~N(reseeded from wte on first forward)"]
        if hasattr(pre, "apply_gate_init"):                    # PrefixGatedBuffer
            # Step (1) above called reset_parameters() on gate_proj_in/mem,
            # which puts KAIMING weights back into both projections -- silently
            # discarding gate_init="zero" and leaving millions of untrained
            # parameters making per-channel keep/drop decisions at step 0.  The
            # buffer owns its designed init; re-apply it here.
            pre.apply_gate_init()
            tags += ["forget_bias=1", "input_bias=0",
                     f"gate_init={pre.gate_init}"]
            if getattr(pre, "carries_latent", False):
                # apply_gate_init covers the Z gate too -- step (1) above called
                # reset_parameters() on gate_proj_in_z/mem_z exactly as it does
                # on the E pair, so BOTH need re-zeroing.  Tagged separately
                # because "the gate was re-initialised" and "both gates were
                # re-initialised" are different claims in a run log.
                tags += ["forget_bias_z=1", "input_bias_z=0",
                         f"gate_init_z={pre.gate_init}"]
            if hasattr(pre, "slot_init"):                      # gate_fill=init
                torch.nn.init.normal_(pre.slot_init, std=0.02)
                tags += ["slot_init~N(reseeded from wte on first forward)"]
        elif hasattr(pre, "forget_bias"):                      # legacy gated shape
            torch.nn.init.ones_(pre.forget_bias)               # LM2 3.3 forget bias +1
            torch.nn.init.zeros_(pre.input_bias)
            tags += ["forget_bias=1", "input_bias=0"]
        fixed.append("prefix.[" + ",".join(tags) + "]")
    # (3) insurance: nothing in cortex should be non-finite now — warn loudly if
    #     some module lacked reset_parameters and slipped through.
    bad = [n for n, p in cortex.named_parameters() if not torch.isfinite(p).all()]
    if log is not None:
        log(f"[cortex] reset {n_reset} cortex submodules to nn defaults + "
              f"re-applied designed inits {fixed} (undo post_init clobber)")
        if bad:
            log(f"[cortex] WARNING: {len(bad)} cortex params STILL non-finite "
                  f"after reset (no reset_parameters?): {bad[:12]}")
