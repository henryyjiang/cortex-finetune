#!/bin/bash
#SBATCH --account=gts-aivanova7-lab
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128GB
#SBATCH --gres=gpu:H200:1
#SBATCH --constraint=H200
#SBATCH -t 48:00:00
#SBATCH -q inferno
#SBATCH -J cf5b
#SBATCH -o logs/Report-%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=henryyjiang42@gmail.com

# ============================================================================
# CORTEX-FINAL at 5B -- the dual-channel memory arm.  ARM 2 OF 2.
#
#   PHASE=heal ZNORM_TARGET=<measured> sbatch pace/cortex_final_5b.sbatch
#   PHASE=heal RESUME_PATH=cortex-5b/cortex-final/checkpoint_XXXX \
#       ZNORM_TARGET=<measured> sbatch pace/cortex_final_5b.sbatch
#   # the corpus switch, ONCE, at step 11,444:
#   PHASE=mix RESUME_PATH=... EXTRA_ARGS="--reset_dataset_position true" \
#       ZNORM_TARGET=<measured> sbatch pace/cortex_final_5b.sbatch
#
# Sibling: pace/control_5b.sbatch.  The shared recipe lives in
# pace/final_5b_common.sh and is written ONCE, so the two arms cannot drift
# apart.  `diff pace/control_5b.sbatch pace/cortex_final_5b.sbatch` is the
# complete difference between the two runs.
#
# *** SEQUENCING: THE CONTROL RUNS FIRST, AND THIS ARM DOES NOT RUN BESIDE IT. ***
# Not a technical constraint -- two H200s would work -- but the deliberate
# order, so that the control's throughput, its corpus-switch behaviour and its
# mid-run read-outs are all known before the memory arm's compute is committed.
# The advisory check below looks for a live `c5b` job and warns; it does not
# refuse, because a legitimate overlap exists (the control's last link still
# draining while this one queues).
#
# THE ARCHITECTURE, FROZEN 2026-09-28, BAND WIDENED 2026-09-30
# ------------------------------------------------------------
# This is J6's `stag` limb (J7) at 5B.  The pre-run test program is CLOSED;
# do not reopen it here.  The next spend is training.
#
#   gated dual-channel buffer, W=16 writes/chunk into K=64 ring slots
#   E  post-ln_f summary states, spliced as input_embeds columns   (B2's write)
#   Z  RAW STAGGERED STATES s_k, one depth per slot, band 1..9
#   Z read: `embeds` -- its own 64 columns beside E's, read by the model's own
#           pretrained attention at every one of the T loop iterations
#   latent_embed FROZEN at identity: the splice is literally AutoCompressor's,
#           and at identity the module is an exact no-op, so the parameter set,
#           the state_dict and the optimizer count are unchanged from a
#           trainable-projection arm.  Only requires_grad differs.
#
# The evidence, and its caveats, in one place so a writeup cannot overclaim:
#   * J7 measured X_content +0.2215 [+0.2094, +0.2340] -- a donor's Z worth
#     LESS than no Z at all (0.9226 vs 0.9160), against 0% content share on
#     every projection-training limb.  First time in the project that Z's
#     CONTENTS paid for anything.
#   * `stag` differs from the control limb in TWO flags (encoding AND freeze)
#     and no cell isolates either.  The span test FALSIFIED the pre-registered
#     laundering story -- the trained map moved Z 37% AWAY from E's span -- and
#     the D3 probe put the cause in the WRITE (Z_write +0.168 vs +0.034).  So:
#     "the staggered write carries more, laundering ruled out", NOT "freezing
#     unlocked content".
#   * +0.2215 was measured at latent_depth_lo=2.  This run trains lo=1, which
#     re-maps all 16 slots (span 6 -> 7), so the number does NOT transfer
#     as-is.  Accepted risk, recorded before the launch, not an oversight.
#   * hi=9 is INERT at T=8 (hi_c = min(hi, T-1) = 7), so the realised band is
#     1..7: seven depths over 16 slots.  Neither configured endpoint now does
#     the job it was originally set for.
#   * Z has ~32.8M tokens of training behind it against E's ~1.5B.  That 46x
#     is the whole reason this run exists at 5B rather than at 2,000 updates.
#
# ZNORM_TARGET IS A MEASUREMENT, NOT A CONSTANT -- AND IT IS A LAUNCH GATE
# -----------------------------------------------------------------------
# --cortex.latent_read_znorm_target rescales every carried Z row to a FIXED
# norm before the splice, so Z and E enter the same concatenation at comparable
# scale.  Get it wrong and the failure is the measured s0 pathology: Z entering
# ~17x below the field it competes with is flat to nothing, and Z entering
# above E drowns it.  The number is E's own carried row norm -- and it is
# LINEAGE-SPECIFIC: 171.0 on B2, 136.0 on the b2 heal branch, a 26% spread
# between two checkpoints of ONE lineage.  J4's single smoke failure was this
# number inherited from a different branch.  cortex-final descends from
# neither; it starts fresh from ckpts/olmo-retrofit-cortex, where nobody has
# measured it.  So this script REFUSES to launch without it:
#
#     sbatch pace/measure_znorm.sbatch      # ~5 min of H200, prints the number
#
# THEN WATCH IT.  `cortex/z_e_carried_norm` is logged every DIAG_INTERVAL steps
# and tools/watch_run.py FAILS on drift past +-15% of the target.  This is not
# theoretical: over J7's 2,000 steps it moved 135.5 -> 144.1 (+6.3%), ending
# +6.0% away from the 136.0 it launched at, and this run is ~19x longer.  A
# settled norm is a calibration constant; a climbing one is an instability with
# a schedule, and you want to know which INSIDE the heal phase.  This is P2.5's
# question, asked live instead of reconstructed from checkpoints after the fact
# (tools/norm_trace.py exists because the P2.3 cells ran 24,414 steps with
# diag_interval 0).  If the band breaks, the decision is whether to re-target
# on the next link -- and WHATEVER YOU CHOOSE, RECORD IT, because changing the
# constant mid-chain changes the arm.
#
# MICRO_BATCH_SIZE 2, MEASURED.  Job 13662406_2 (J7's stag limb at this exact
# geometry) peaked at 90.84 GiB of an H200's 138.7 and ran 5,077-6,552 tok/s.
# window_backward is what buys it -- the peak holds one 4-chunk carry window,
# not the whole row.  The control runs 1 for the reason its own header gives.
# NEVER change MICRO_BS mid-chain; the common file stamps and enforces it.
#
# LINK BUDGET.  5.0B / ~5,500 tok/s = ~252 GPU-h at ~47.7 usable h per link
# = ~6 links.  The recurrence ramp makes the first 9,537 steps cheaper.
#
# PRE-FLIGHT (once, before the first submit)
#   git pull
#   sbatch pace/check_packs.sbatch                                # both packs
#   sbatch pace/measure_znorm.sbatch                              # the gate
#   BRANCH_PATH=cortex-retrofit/retro-b2-heal/checkpoint_91552_w16 \
#       sbatch pace/prelaunch_final.sbatch                        # the wiring gate
#
# MONITORING
#   after every link   tools/watch_run.py  (this script calls it on exit)
#   at 4 checkpoints   ARM=cortex STEP=<n> ZNORM_TARGET=<same> \
#                          sbatch pace/midrun_check.sbatch
#   the bars           ../final_5b_launch.md S4
# ============================================================================

cd $SLURM_SUBMIT_DIR

ARM=cortex
source pace/final_5b_common.sh
final5b_shared

RUN_NAME=${RUN_NAME:-cortex-final}
# MEASURED at this exact geometry: job 13662406_2 peaked at 90.84 GiB of 138.7
# and ran 5,077-6,552 tok/s.  window_backward is what buys it.
MICRO_BS_DEFAULT=2

# --- the sequencing advisory (warn, never refuse) --------------------------
if command -v squeue >/dev/null 2>&1; then
    LIVE=$(squeue -u "$USER" -h -n c5b -o "%i %T" 2>/dev/null | head -3)
    if [ -n "$LIVE" ]; then
        echo ""
        echo "  ~~  A control link (job name c5b) is in the queue:"
        printf '  ~~    %s\n' $LIVE
        echo "  ~~  The plan is control FIRST, memory arm after.  This is a"
        echo "  ~~  warning, not a refusal -- the control's last link draining"
        echo "  ~~  while this one queues is a legitimate overlap.  If you did"
        echo "  ~~  not mean to start the memory arm yet, scancel this job."
        echo ""
    fi
fi

# --- ZNORM_TARGET: no default, by design (see the header) ------------------
if [ -z "${ZNORM_TARGET:-}" ]; then
    echo "ERROR: cortex-final needs ZNORM_TARGET set explicitly."
    echo "  It is E's MEASURED carried row norm on THIS lineage, and every"
    echo "  carried Z row is rescaled to it before the splice.  Inherited"
    echo "  values are wrong: 171.0 was B2's, 136.0 was the b2 heal branch's,"
    echo "  and this run starts fresh from $MODEL.  Measure it first:"
    echo "      sbatch pace/measure_znorm.sbatch"
    echo "  then relaunch with ZNORM_TARGET=<the number it prints>."
    exit 1
fi

ACCUM_VECS=${ACCUM_VECS:-16}           # W: writes per chunk (a PARAMETER SHAPE)
GATE_SLOTS=${GATE_SLOTS:-64}           # K: ring capacity
ACCUM_MAX=$(( ACCUM_VECS * CROSS_CHUNKS ))
CARRY_GRAD_CHUNKS=${CARRY_GRAD_CHUNKS:-4}
LATENT_LO=${LATENT_LO:-1}              # widened from 2 on 2026-09-30
LATENT_HI=${LATENT_HI:-9}              # inert at T=8 (hi_c = min(hi,T-1) = 7)
LAP=$(( GATE_SLOTS / ACCUM_VECS ))

# --- geometry pre-flights specific to the gated ring -----------------------
if [ $(( GATE_SLOTS % ACCUM_VECS )) -ne 0 ]; then
    echo "ERROR: gate_slots ($GATE_SLOTS) is not a multiple of accum_vecs" \
         "($ACCUM_VECS): a partial lap writes a different row set every time"
    echo "       round and the depth->row map breaks."; exit 1
fi
if [ "$CROSS_CHUNKS" -lt $(( 2 * LAP )) ]; then
    echo "ERROR: cross_chunks ($CROSS_CHUNKS) < 2 x lap ($(( 2 * LAP )))." \
         "The ring's gate only acts from lap 2; below this it NEVER FIRES in"
    echo "       training and the run trains an accum buffer with dead gate"
    echo "       parameters bolted on, behind a healthy loss curve."; exit 1
fi

ARM_ARGS="--cortex.use_memory true"
ARM_ARGS="$ARM_ARGS --cortex.memory_lr 5e-4 --cortex.memory_slots 0"
ARM_ARGS="$ARM_ARGS --cortex.prefix_memory gated"
ARM_ARGS="$ARM_ARGS --cortex.accum_vecs $ACCUM_VECS"
ARM_ARGS="$ARM_ARGS --cortex.accum_max $ACCUM_MAX"
ARM_ARGS="$ARM_ARGS --cortex.gate_slots $GATE_SLOTS"
ARM_ARGS="$ARM_ARGS --cortex.gate_route ring --cortex.gate_norm tanh"
ARM_ARGS="$ARM_ARGS --cortex.gate_init zero --cortex.gate_fill grow"
# The Z channel, frozen 2026-09-28.
ARM_ARGS="$ARM_ARGS --cortex.latent_carry true"
ARM_ARGS="$ARM_ARGS --cortex.latent_carry_read true --cortex.e_carry_read true"
ARM_ARGS="$ARM_ARGS --cortex.latent_encoding staggered_state"
ARM_ARGS="$ARM_ARGS --cortex.latent_embed_frozen true"
ARM_ARGS="$ARM_ARGS --cortex.latent_depth_rule absolute"
ARM_ARGS="$ARM_ARGS --cortex.latent_depth_lo $LATENT_LO"
ARM_ARGS="$ARM_ARGS --cortex.latent_depth_hi $LATENT_HI"
ARM_ARGS="$ARM_ARGS --cortex.latent_renorm none"
ARM_ARGS="$ARM_ARGS --cortex.latent_s0_read false"
ARM_ARGS="$ARM_ARGS --cortex.latent_read embeds"
ARM_ARGS="$ARM_ARGS --cortex.latent_read_znorm rms"
ARM_ARGS="$ARM_ARGS --cortex.latent_read_znorm_target $ZNORM_TARGET"
ARM_ARGS="$ARM_ARGS --cortex.latent_read_scramble false"
ARM_ARGS="$ARM_ARGS --cortex.e_dropout 0.0"
# One backward per closing carry window, so the peak holds 4 of 8 chunks'
# activations instead of the whole row.  This is what buys MICRO_BS 2.
ARM_ARGS="$ARM_ARGS --cortex.carry_grad_chunks $CARRY_GRAD_CHUNKS"
ARM_ARGS="$ARM_ARGS --cortex.window_backward true"
ARM_ARGS="$ARM_ARGS --cortex.prefix_eos_reset false --cortex.prefix_pos tail"
ARM_ARGS="$ARM_ARGS --cortex.freeze_loop false"

ARM_BANNER="    mechanism:  gated W=$ACCUM_VECS K=$GATE_SLOTS lap=$LAP accum_max=$ACCUM_MAX grad_chunks=$CARRY_GRAD_CHUNKS window_backward=true
    Z:          staggered_state, band $LATENT_LO..$LATENT_HI (absolute), read=embeds, proj FROZEN at identity
    Z scale:    znorm=rms target=$ZNORM_TARGET  <- MEASURED on this lineage?  watch cortex/z_e_carried_norm"

final5b_launch
exit $?
