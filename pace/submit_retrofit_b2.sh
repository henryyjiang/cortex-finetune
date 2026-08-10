#!/bin/bash
# B2 driver — two-phase retrofit with AutoCompressor-faithful prefix memory.
#
#   heal    one shared healing phase on FineWeb-Edu, prefix accum, to 91,552
#           steps (1.5B tokens).  Full recurrence 8 lands at step 76,294,
#           INSIDE this phase, so both arms start at constant recurrence.
#           Runs at cross_chunks=4 and STAYS there — it is mid-flight.
#   branch  arm A (accum) and arm B (gated) both branch from the SAME healing
#           checkpoint onto Nemotron math and run to 305,176 steps (5B), at
#           cross_chunks=8 (512-token chunks).
#   gate    carry ablation + slice conditions + buffer diagnostic on arm A,
#           automatically at the arms' geometry.
#
# GEOMETRY CHANGE AT THE BRANCH (2026-08-04, ceil2 probes).  The oracle ceiling
# is 2.2x larger at 512-token chunks and does not collapse; Track A read at 8
# chunks nearly doubles its carry delta (+0.0433 vs +0.0237) despite having
# trained at 4.  The heal is not restarted — that measurement was itself taken
# across exactly this mismatch.  Full reasoning, and the two flags that must
# move with cross_chunks, are in the CHUNK GEOMETRY block of b2_retrofit.sbatch.
#
# Why a shared healing phase is legitimate: accum's parameters are a strict
# SUBSET of gated's (summary_emb vs summary_emb + the LM2 gate), so healing as
# accum trains everything both arms need except the gate.  The branch loads
# weights non-strictly, discards the optimizer state (its parameter set
# changed) and the dataloader position (the corpus changed), and CONTINUES the
# LR schedule, the recurrence ramp and the step counter.  Both arms branch
# identically, so nothing about the branch can favour one of them.
#
# BUDGET SHAPE, as decided 2026-08-02 — ACCUM ONLY to start (~6 x 48h H200
# jobs, ~5B tokens).  See the ARM SEQUENCING block in b2_retrofit.sbatch for
# why accum and not gated; the short version is that dropping to one arm
# removes the between-arm comparison, so the arm that carries diagnostics wins.
#   heal      ~2 jobs   0 -> 91,552          shared, paid once
#   accum     ~4 jobs   91,552 -> 305,176
#   gate      2 x 4h eval jobs, at ANY checkpoint after the branch
# `branch gated` later costs 4 more jobs and NO extra healing — it forks the
# same checkpoint, so it stays a controlled comparison however long you wait.
# Running both at once is still supported: `branch` with no argument.
# Nothing past the current stage is ever committed; every stage is one command.
#
# The first post-branch checkpoint is already a valid read (the recurrence ramp
# finishes inside healing, so every arm checkpoint is at constant recurrence 8).
# That is the early bail point: ~3 jobs spent, not 6.
#
# MAX_STEPS stays 305,176 for EVERY link, including the healing phase, because
# it is the horizon for both the LR cosine and the recurrence ramp.  The
# healing phase ends via STOP_AT_STEP, which does not move that horizon.  This
# script never passes anything else.
#
# Usage (from ~/cortex-finetune on the cluster):
#   bash pace/submit_retrofit_b2.sh heal            # start / continue healing
#   bash pace/submit_retrofit_b2.sh status
#   bash pace/submit_retrofit_b2.sh branch          # both arms off the heal ckpt
#   bash pace/submit_retrofit_b2.sh branch accum    # ... or one arm
#   bash pace/submit_retrofit_b2.sh resume          # continue both arms
#   bash pace/submit_retrofit_b2.sh resume gated    # ... or one arm
#   bash pace/submit_retrofit_b2.sh gate            # evals at arm A's newest
#   bash pace/submit_retrofit_b2.sh gate 150000     # ... or at a named step

set -e
STAGE=${1:-}
ARG=${2:-}

SBATCH=pace/b2_retrofit.sbatch
OUT=cortex-retrofit
BASE_CKPT=ckpts/olmo-retrofit-cortex
HEAL_DATA=data/fineweb_edu_olmo_len4096
ARM_DATA=data/fw_nemo50_olmo_len4096
TOTAL_STEPS=305176          # 5B / 16,384 — the schedule horizon, never varies
HEAL_STEPS=91552            # 1.5B — where the healing phase stops
FULL_REC_STEP=76294         # 0.25 * TOTAL_STEPS — full recurrence 8
TOK_PER_STEP=16384

HEAL_RUN=retro-b2-heal
ARMS_ALL="accum gated"
# Arm run names MUST match what b2_retrofit.sbatch derives
# (retro-b2-${ARM}-cc${CROSS_CHUNKS}-mr${MAX_MEAN_REC}).  ARM_CROSS_CHUNKS is
# the one value to change if the geometry ever moves again — the sbatch's arm
# default and this must agree or 'status' / 'resume' / 'gate' look at empty dirs.
ARM_CROSS_CHUNKS=8          # 512-token chunks; healing stays at 4.  See the
                            # CHUNK GEOMETRY block in b2_retrofit.sbatch.
declare -A RUN_OF=(
    [accum]=retro-b2-acc32-cc${ARM_CROSS_CHUNKS}-mr8
    [gated]=retro-b2-gate32-cc${ARM_CROSS_CHUNKS}-mr8
)

# --- newest RESUMABLE checkpoint (checkpoint_<step>/chkpt.pt) ---------------
# save_interval also writes weights-only model_only_chkpt_* dirs, but only
# checkpoint_* carries optimizer + scheduler + dataloader, so only those can
# resume or be branched from.  Sorted NUMERICALLY (checkpoint_9500 must not
# outrank checkpoint_150000).
newest_ckpt() {
    ls -d $OUT/$1/checkpoint_* 2>/dev/null \
        | sed 's/.*checkpoint_//' | sort -n | tail -1
}

newest_model_only() {
    ls -d $OUT/$1/model_only_chkpt_* 2>/dev/null \
        | sed 's/.*chkpt_//' | sort -n | tail -1
}

require() {
    for p in "$@"; do
        [ -e "$p" ] || { echo "ERROR: missing prep: $p"; \
            echo "  See the PREP block in $SBATCH."; exit 1; }
    done
}

case "$STAGE" in
heal)
    require $BASE_CKPT $HEAL_DATA
    step=$(newest_ckpt $HEAL_RUN)
    if [ -z "$step" ]; then
        PHASE=heal STOP_AT_STEP=$HEAL_STEPS sbatch $SBATCH
        echo "Submitted: $HEAL_RUN (step 0 -> $HEAL_STEPS)"
    elif [ "$step" -ge "$HEAL_STEPS" ]; then
        echo "Healing already complete at step $step — run 'branch'."
        exit 0
    else
        PHASE=heal STOP_AT_STEP=$HEAL_STEPS \
            RESUME_PATH=$OUT/$HEAL_RUN/checkpoint_$step sbatch $SBATCH
        echo "Submitted: $HEAL_RUN resuming from step $step (-> $HEAL_STEPS)"
    fi
    cat <<EOF

First-hour checks (wandb project cortex-retrofit):
  * loss starts ~10.26 — the untrained ARRANGEMENT of pretrained OLMo weights.
    Anything much lower means the wrong base checkpoint (olmo8-cortex is the
    already-retrofitted one; using it turns this into a finetune).
  * the log line "[cortex] memory ON: ... prefix_memory=accum" MUST appear.
    Without it the run has no memory at all and still trains a healthy curve —
    that is the single most expensive silent failure available here.
  * train/mean_recurrence climbs and reaches 8 at step $FULL_REC_STEP.
  * train/total_norm bounded.  No stabilizers by design; B1's accum arm showed
    heavy-tail gnorm spikes (max 123.7 vs the control's 8.7) and this run goes
    to HIGHER recurrence than B1 ever reached.
EOF
    ;;

status)
    echo "=== queue ==="
    squeue -u "$USER" -o "%.10i %.30j %.8T %.10M %.10L %R" || true
    echo
    echo "=== progress ==="
    for run in $HEAL_RUN ${RUN_OF[accum]} ${RUN_OF[gated]}; do
        [ -d "$OUT/$run" ] || continue
        step=$(newest_ckpt $run)
        mo=$(newest_model_only $run)
        printf "  %-24s resumable=%-8s evaluable=%-8s" "$run" "${step:-none}" "${mo:-none}"
        if [ -n "$step" ]; then
            printf "  %sM tok  %s%% of %s" \
                "$(( step * TOK_PER_STEP / 1000000 ))" \
                "$(( step * 100 / TOTAL_STEPS ))" "$TOTAL_STEPS"
        fi
        echo
    done
    echo
    echo "  healing ends $HEAL_STEPS | full recurrence $FULL_REC_STEP | horizon $TOTAL_STEPS"
    ;;

branch)
    require $BASE_CKPT $ARM_DATA
    step=$(newest_ckpt $HEAL_RUN)
    [ -n "$step" ] || { echo "ERROR: no checkpoint under $OUT/$HEAL_RUN — run 'heal'."; exit 1; }
    if [ "$step" -lt "$FULL_REC_STEP" ]; then
        echo "ERROR: healing is at step $step, before full recurrence ($FULL_REC_STEP)."
        echo "  Branching now would put the arms on the ramp, which is exactly"
        echo "  what made B1's intermediate gate uninterpretable.  Run 'heal'."
        exit 1
    fi
    if [ "$step" -lt "$HEAL_STEPS" ]; then
        echo "WARNING: healing stopped at $step, short of the planned $HEAL_STEPS."
        echo "  Past full recurrence, so the arms are still comparable; branching."
    fi
    # BOTH arms must branch from the SAME checkpoint or they are not controlled.
    branch_from=$OUT/$HEAL_RUN/checkpoint_$step
    for arm in ${ARG:-$ARMS_ALL}; do
        run=${RUN_OF[$arm]}
        [ -z "$(newest_ckpt $run)" ] || { echo "SKIP $run — already started; use 'resume'."; continue; }
        PHASE=arm MEMORY_MODE=$arm BRANCH_PATH=$branch_from sbatch $SBATCH
        echo "Submitted: $run branching from $HEAL_RUN step $step (-> $TOTAL_STEPS)"
    done
    echo
    cat <<EOF

Branch checks:
  * The arms' first logged loss should sit AT the healing phase's last value
    plus the corpus shift (FineWeb-Edu -> Nemotron math) AND the geometry shift
    (cross_chunks 4 -> $ARM_CROSS_CHUNKS, i.e. $(( 4096 / ARM_CROSS_CHUNKS ))-token chunks: less local
    context per chunk, so expect ~+0.06 nats from that alone).  Not ~10.
  * The banner must read "geometry: cross_chunks=$ARM_CROSS_CHUNKS ... accum_max=256
    carry_grad_chunks=4".  cross_chunks alone will not start — train.py asserts
    cross_chunks x accum_vecs <= accum_max, and the sbatch pre-flights it.
  * Both arms must print the same '[branch] loaded weights from ...' step, and
    the gated arm must list its gate parameters as the only fresh ones.
  * Run the smoke gate at the NEW geometry first if you have not already:
      python tools/smoke_prefix_real.py --cross_chunks $ARM_CROSS_CHUNKS --accum_max 256
EOF
    ;;

resume)
    for arm in ${ARG:-$ARMS_ALL}; do
        run=${RUN_OF[$arm]}
        step=$(newest_ckpt $run)
        [ -n "$step" ] || { echo "ERROR: no checkpoint under $OUT/$run — use 'branch'."; exit 1; }
        if [ "$step" -ge "$TOTAL_STEPS" ]; then
            echo "SKIP $run — already at step $step."
            continue
        fi
        PHASE=arm MEMORY_MODE=$arm RESUME_PATH=$OUT/$run/checkpoint_$step sbatch $SBATCH
        echo "Submitted: $run resuming from step $step (-> $TOTAL_STEPS)"
    done
    ;;

gate)
    # The carry ablation is the readout: train loss separated no Track-A arm
    # (2.3426-2.3468 across all three) and will not separate these either.
    #
    # Only the ACCUM arm gets the slice ablation and the buffer diagnostic —
    # both need write-once rows and refuse a gated state, since a gated merge
    # makes the k-th chunk's contribution unrecoverable.  The gated arm gets
    # the plain carry ablation.
    run=${RUN_OF[accum]}
    step=${ARG:-$(newest_model_only $run)}
    [ -n "$step" ] || { echo "ERROR: no model_only_chkpt_* under $OUT/$run yet."; exit 1; }
    ckpt_name=model_only_chkpt_$step
    if [ ! -d "$OUT/$run/$ckpt_name" ]; then
        echo "ERROR: no $OUT/$run/$ckpt_name (save_interval is 2500)."
        echo "  present: $(ls -d $OUT/$run/model_only_chkpt_* 2>/dev/null \
            | sed 's/.*chkpt_//' | sort -n | tr '\n' ' ')"
        exit 1
    fi

    # Geometry MUST follow the arms, or the gate reads an 8-trained arm at the
    # 4-chunk default and reports a number for a regime that was never trained.
    # T_EVAL likewise: retrofit-derived configs inherit mean_recurrence=32.
    ARM_CHUNK_LEN=$(( 4096 / ARM_CROSS_CHUNKS ))
    COMMON="OUT_ROOT=$OUT CKPT_NAME=$ckpt_name BASE=$BASE_CKPT RUN=$run EVAL_TAG=b2-$step"
    env $COMMON SLICE_ABLATE=both N_CHUNKS=$ARM_CROSS_CHUNKS T_EVAL=8 \
        sbatch pace/eval_carry_ablation.sbatch
    echo "Submitted carry ablation + slice conditions: $run @ step $step" \
         "(n_chunks=$ARM_CROSS_CHUNKS, T=8)"
    env $COMMON CHUNK_LEN=$ARM_CHUNK_LEN sbatch pace/diag_accum_buffer.sbatch
    echo "Submitted buffer diagnostic:                 $run @ step $step" \
         "(chunk_len=$ARM_CHUNK_LEN)"

    gated_run=${RUN_OF[gated]}
    gstep=$(newest_model_only $gated_run)
    if [ -n "$gstep" ]; then
        env OUT_ROOT=$OUT CKPT_NAME=model_only_chkpt_$gstep BASE=$BASE_CKPT \
            RUN=$gated_run EVAL_TAG=b2-$gstep N_CHUNKS=$ARM_CROSS_CHUNKS T_EVAL=8 \
            sbatch pace/eval_carry_ablation.sbatch
        echo "Submitted carry ablation (no slices):        $gated_run @ step $gstep"
    fi

    cat <<EOF

Reading the gate (tag b2-$step; carry ablation is a 30-min job, the buffer
diagnostic still books 4h):
  * This is the first read in the project where the cross-track comparison is
    CLEAN: both arms train at constant recurrence 8 from step $FULL_REC_STEP,
    matching Track A's fixed recurrence.
  * COMPARE AT THE MATCHED GEOMETRY.  These arms train at cross_chunks=8, so
    the reference is Track A read at 8 chunks (+0.0433 +- 0.0012), NOT its
    published 4-chunk +0.0237.  Track A's 4-chunk numbers stay the reference
    only for a 4-chunk read.  Gated Track A (+0.0283) and B1 (-0.0128) were
    both measured at 4 and are not directly comparable to this gate.
  * THE ~0.1-NAT BAR IS RETRACTED (2026-08-04) — it sat ABOVE the measurable
    oracle ceiling, so no mechanism could ever have cleared it.  Report the
    FRACTION OF THE CEILING RECOVERED instead.  At cross_chunks=8 the ceiling
    on this instrument is +0.1485, so Track A's 29.1% is the number to beat;
    AutoCompressor is at 50% at 7B.  Run eval_context_ceiling with N_CHUNKS=8
    on the same checkpoint to get the arm's own denominator rather than
    borrowing Track A's.
  * Better still, quote the CONDITIONAL fraction: given the last 128 tokens
    (free at inference) Track A still adds +0.00717 +- 0.00075, recovering 37%
    of the headroom that is actually left.  CARRY_PLUS="128 256" on the ceiling
    eval produces it.
  * A NEGATIVE delta would repeat B1 and should stop the chain — but check the
    "[cortex] memory ON" line and the branch log first: a memory-off run and a
    memory-that-hurts run look identical in the loss curve.
  * Slice conditions, Track-A reference on the same instrument: total 0.024 /
    drop-oldest ~0.003 / drop-newest ~0.014, i.e. recency-dominated with
    redundant gist writes (cosine 0.94).  A different SHAPE here is the
    interesting outcome: it means co-training through the conversion changed
    what gets written, which is the thing bolt-on could not test.
  * The buffer diagnostic runs on PG-19 while the arms train on Nemotron, so
    absolute norms and losses are off-distribution.  Read the SHAPE across
    chunk depth.  Note the prefix write is unbounded by design (no tanh/LN), so
    unlike the old arms the norms CAN drift — that is signal, not a bug.
EOF
    ;;

*)
    sed -n '2,58p' "$0"
    exit 1
    ;;
esac
