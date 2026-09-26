"""J4 -- the carried Z as extra columns of `input_embeds`.

Findings doc "Latent Memory (Z) Attempt 1: Findings", Attempt 2, design J4
(final rank 2): "Z gets its own K columns in `input_embeds`, projected to E's
scale, re-entering every iteration as E does."  j4_handoff.md step 2.

WHY THIS EXISTS, AND WHY IT IS THE LAST READ DESIGN WORTH TRYING
----------------------------------------------------------------
Three read sites have now been measured and two of them are dead:

  s0            deleting the carried columns costs +4.1e-5 / -2.5e-5 nats
                (opposite signs across arms, i.e. noise), and a single no-grad
                loop iteration cuts the read gradient to EXACTLY zero.
  a fresh       J1's xattn read: gate 0.10 -> 0 in ~270 updates.  J3 held the
  module        gate open with gate_lr_mult and the read stayed ~UNIFORM --
                own row 1/65, i.e. no selection learned in 2,000 updates.
  THIS ONE      the base model's own pretrained attention, which already reads
                E's 64 carried columns selectively on every one of the T loop
                iterations, through `adapter(cat([x, input_embeds]))`.

Lesson 2 of the handoff, in one line: PREFER A READER THAT ALREADY KNOWS HOW TO
ATTEND.  J4 adds no reader at all.  The only new parameter is the projection
below, and its job is coordinates and scale, not selection.

WHAT D3 SETTLED, AND WHAT IT DID NOT
------------------------------------
D3 (evals/diag_z_registers.py, job 13528227) probed the register file out of
each candidate write.  The carried write J1 and J3 trained -- the loop state at
the chunk's last 64 real tokens -- holds ~1% of E's margin: nothing, with the
only flicker (+0.007) on registers last updated INSIDE those last 64 tokens and
flat (-0.001) on anything earlier.  A recency smear, not a register file.  The
read was never the bottleneck.  `Z_end` -- s_T at the SUMMARY columns, the
pre-coda twin of E -- decodes the registers above chance in all four
checkpoints (22-54% of E's margin).  So J4 writes `latent_encoding='endpoint'`
and reads it here.

What D3 did NOT show is that `Z_end` is ADDITIONAL to E rather than a re-coded
copy of it (cortex_graft.py:1231's information objection, unrefuted).  That is
j4_prereg.md's open item and what its Step 0 incremental probe measures.  It
does not change this module; it changes how a PASS is read.

THE DESIGN, AND WHY EACH PART IS FORCED
---------------------------------------
* SCALE FIRST, BEFORE ANYTHING ELSE.  E's carried rows enter at row norm ~171
  (measured, B2, fp32).  `Z_end` rows are PRE-coda loop states at norm ~10-11
  -- 17x smaller.  The s0 site's whole failure was a 0.39-norm row competing
  with a 171-norm one inside the same concatenation: attention did not see it,
  and the sweep only "turned on" at an injected norm of 175.8, which is
  drowning E rather than being read.  So every Z row is rescaled to
  `latent_read_znorm_target` BEFORE reaching this module -- in the graft, by
  `rescale_rows`, the same helper and the same flag J1/J3's xattn read uses
  (cortex_graft.py:1132), so there is one rescale in the codebase and the
  ordering is visible at the call site.  The target is the MEASURED ||E||, not
  a guess.  Rescaling BEFORE the projection means the entering norm is exactly
  the target at step 0 and can then MOVE as the projection trains -- which is
  the readable negative: `z_embed_ratio` in the diag is J4's read-strength
  number, standing where J1's gate and J3's read_ratio stood.

* ONE PROJECTION, IDENTITY-INITIALISED, NO BIAS.  Z is in a different
  coordinate system from E (pre-coda vs post-coda + ln_f), so it needs a map;
  D^2 = 4.19M parameters at D = 2048.  Identity init, NOT zero: a zero-init
  read is the x0.90 shape this project has already paid for twice (LSTMBuffer's
  zeroed out_proj, then B2's x1.33 for unfreezing it) -- the read must be LIVE
  at step 0, not a no-op to be discovered.  Identity + rescale means step 0
  splices a correctly-scaled copy of the pre-coda summary state, which is a
  real read from the first batch, and training rotates it from there.

  NO BIAS, and that is load-bearing in two places, not tidiness:
    - the null.  J4's Z null is ZEROS at the same columns (j4_prereg.md S6;
      omission would change the positions and the sequence length, so the
      cells would differ in geometry and not in the carry).  A bias would turn
      the null into a learned constant vector -- an E0Z0 cell quietly reading
      something -- and score_j1.py's NOREAD_ZMAIN_TOL veto is exactly the test
      that would then fail, or worse, not fail.
    - unwritten ring rows.  A gated ring that has not come round yet holds
      EXACTLY zero (PrefixGatedBuffer._slot_init_block).  With no bias those
      rows enter as exact zeros, which is precisely what E's own unwritten rows
      already do in the same splice -- so E and Z are treated identically and
      there is nothing new to explain.

* THE GRAFT DOES THE SPLICING, NOT THIS MODULE.  `prefix_pack` owns the packed
  layout and the position ids; this is a pure [B, K, D] -> [B, K, D] map with
  no state, so it cannot disagree with the layout.  The null is applied
  UPSTREAM of it, in `_latent_z_rows`, for the reason that function's docstring
  gives: a read path that bypassed the null would report "Z off" while feeding
  the model real Z.

COST.  D^2 = 4.19M parameters (+ Adam state), and 64 more packed columns:
592 -> 656 at chunk 512, W=16, K=64, about +11% sequence and so roughly +10%
activations across the whole loop.  Price it (pace/j4_price_oom.sbatch) before
launching -- J3 already sat at 92.4 of 140 GiB.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class LatentEmbedRead(nn.Module):
    """J4's read: carried Z rows -> embedding-scale columns for `input_embeds`.

    Stateless, shape-preserving, and deliberately NOT the rescale: the graft
    rescales the rows to `latent_read_znorm_target` before calling this, exactly
    as it does for the xattn read, so `rescale_rows` has one home and the
    ordering (scale, then rotate) is readable where the splice happens.
    """

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        # bias=False: see the module docstring -- the null and the unwritten
        # ring rows both depend on zeros in giving zeros out.
        self.proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.apply_designed_init()

    def apply_designed_init(self) -> list:
        """(Re-)apply the designed init; returns log tags.

        MUST be called from reset_cortex_graft_init, for the same reason
        LatentScratchpad.apply_designed_init must: that function calls
        reset_parameters() on every cortex submodule, which would leave kaiming
        here and run an init nobody chose -- a random rotation of Z at the right
        norm, which is a control arm, not the design.
        """
        nn.init.eye_(self.proj.weight)
        self.proj.weight._no_weight_decay = True
        return ["latent_embed.[proj=I, bias=none]"]

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z [B, K, D] (rescaled Z rows, null already applied) -> [B, K, D].

        Exact-zero rows in give exact-zero rows out: `rescale_rows` clamps the
        divisor rather than filling the row, and there is no bias.
        """
        if z.dim() != 3 or z.shape[-1] != self.hidden_size:
            raise ValueError(
                f"LatentEmbedRead got {tuple(z.shape)}; expected [B, K, "
                f"{self.hidden_size}].  The Z rows are the ring's Z half and "
                "must arrive at the model's own width.")
        return self.proj(z)
