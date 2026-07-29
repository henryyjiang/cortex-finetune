#!/bin/bash
# B1 driver — retrofit conversion with memory co-trained from step 0.
#
# ARMS: accum (AccumCCoT 4-vec, tb2) + off (no-carry control).  See the "WHY
# ACCUM AND NOT GATED" block in pace/b1_retrofit.sbatch — the short version is
# that the slice ablation and the buffer diagnostic both refuse gated states, so
# a gated flagship would yield one number and no explanation.  gated stays
# wired and is the natural third arm if accum clears the 20k gate.
#
# BUDGET SHAPE (this is the "save compute" plan):
#   tranche 1   2 x 48h training jobs, one per arm, step 0 -> wherever 48h ends
#   gate        2 x 4h eval jobs on the accum arm (cheap; carry ablation with
#               slice conditions, plus the buffer diagnostic)
#   DECIDE      continue, add gated as a third arm, or stop
#   tranche 2   2 x 48h resumes to finish 50000 steps
# At the 2-4k tok/s band a single 48h job covers ~21k-42k steps, so ONE job per
# arm should reach the 20k gate.  Finishing 50000 steps takes 4-6 x 48h jobs
# across both arms.  Nothing past tranche 1 is committed by `start`.
#
# B1 is a resume chain: MAX_STEPS stays 50000 for EVERY link, because it is the
# horizon for both the LR schedule and the mean-recurrence ramp — varying it
# re-runs cooldown per segment (the 2026-06-24 sawtooth artifact, baked into
# weights).  This script never passes anything else.
#
# Usage (from ~/cortex-finetune on the cluster):
#   bash pace/submit_retrofit_b1.sh start         # tranche 1: both arms
#   bash pace/submit_retrofit_b1.sh status        # steps reached + queue
#   bash pace/submit_retrofit_b1.sh gate          # evals at the newest ckpt
#   bash pace/submit_retrofit_b1.sh gate 20000    # ... or at a named step
#   bash pace/submit_retrofit_b1.sh resume        # tranche 2: continue both
#   bash pace/submit_retrofit_b1.sh resume accum  # ... or one arm
#   bash pace/submit_retrofit_b1.sh add-gated     # third arm, only after a gate
#
# Read b1_retrofit.sbatch's gate section before the first `gate`: the
# recurrence ramp runs to step 37500, so a pre-37500 read is INTERNAL (accum vs
# off on the same ramp point) and must NOT be compared against Track A's
# 0.024-0.0283 @ 350M.

set -e
STAGE=${1:-}
ARG=${2:-}

SBATCH=pace/b1_retrofit.sbatch
OUT=cortex-retrofit
BASE_CKPT=ckpts/olmo-retrofit-cortex
TOTAL_STEPS=50000
TOK_PER_STEP=16384

ARMS_ALL="accum off"
declare -A RUN_OF=(
    [accum]=retro-b1-acc4v-tb2-mr8
    [off]=retro-b1-base-mr8
    [gated]=retro-b1-k32-ga-mr8
)

# --- newest RESUMABLE checkpoint (checkpoint_<step>/chkpt.pt) ---------------
# save_interval also writes weights-only model_only_chkpt_* dirs, but only
# checkpoint_* carries optimizer + scheduler + dataloader, so only those can
# resume.  Sorted numerically, NOT lexically (checkpoint_9500 must not outrank
# checkpoint_15000).
newest_ckpt() {
    ls -d $OUT/$1/checkpoint_* 2>/dev/null \
        | sed 's/.*checkpoint_//' | sort -n | tail -1
}

# --- newest EVALUABLE checkpoint (model_only_chkpt_<step>) ------------------
newest_model_only() {
    ls -d $OUT/$1/model_only_chkpt_* 2>/dev/null \
        | sed 's/.*chkpt_//' | sort -n | tail -1
}

require_prep() {
    for p in $BASE_CKPT data/nemotron_math_olmo_len4096; do
        [ -e "$p" ] || { echo "ERROR: missing prep: $p"; \
            echo "  See the 'Prep (login node, once)' block in $SBATCH."; exit 1; }
    done
}

submit_arm() {
    local arm=$1 run=${RUN_OF[$1]}
    if [ -n "$(newest_ckpt $run)" ]; then
        echo "SKIP $run — already has checkpoints; use 'resume'."
        return
    fi
    MEMORY_MODE=$arm sbatch $SBATCH
    echo "Submitted: $run (mode=$arm, step 0 -> $TOTAL_STEPS)"
}

case "$STAGE" in
start)
    require_prep
    for arm in $ARMS_ALL; do submit_arm $arm; done
    cat <<'EOF'

First-hour checks (wandb project cortex-retrofit):
  * both arms start ~10.26 — the untrained converted checkpoint
  * the two curves TRACK each other early.  Divergence there is a wiring bug,
    not instability: B0 already showed parity within ~0.007 nats over the full
    10.4 -> 1.06 descent, with matched grad norms.
  * train/total_norm bounded.  No stabilizers are on, by design (principle
    0.1.3) — reintroduce one at a time only if instability actually shows.
  * H200 confirmed in the job's nvidia-smi line.  A100-80GB will OOM once the
    recurrence ramp climbs (both B0 arms died at 78.4GiB, including the
    no-carry one — it is the 4x1024-chunk backward, not the carry graph).
EOF
    ;;

status)
    echo "=== queue ==="
    squeue -u "$USER" -o "%.10i %.30j %.8T %.10M %.10L %R" || true
    echo
    echo "=== progress ==="
    for arm in $ARMS_ALL gated; do
        run=${RUN_OF[$arm]}
        [ -d "$OUT/$run" ] || continue
        step=$(newest_ckpt $run)
        mo=$(newest_model_only $run)
        printf "  %-26s resumable=%-7s evaluable=%-7s" "$run" "${step:-none}" "${mo:-none}"
        if [ -n "$step" ]; then
            printf "  %sM tok  %s%% of %s" \
                "$(( step * TOK_PER_STEP / 1000000 ))" \
                "$(( step * 100 / TOTAL_STEPS ))" "$TOTAL_STEPS"
        fi
        echo
    done
    echo
    echo "Recurrence ramp completes at step 37500 (0.75 * $TOTAL_STEPS)."
    ;;

resume)
    arms=${ARG:-$ARMS_ALL}
    for arm in $arms; do
        run=${RUN_OF[$arm]}
        step=$(newest_ckpt $run)
        [ -n "$step" ] || { echo "ERROR: no checkpoint_*/chkpt.pt under $OUT/$run — use 'start'."; exit 1; }
        if [ "$step" -ge "$TOTAL_STEPS" ]; then
            echo "SKIP $run — already at step $step."
            continue
        fi
        RESUME_PATH=$OUT/$run/checkpoint_$step MEMORY_MODE=$arm sbatch $SBATCH
        echo "Submitted: $run resuming from step $step (-> $TOTAL_STEPS)"
    done
    ;;

gate)
    # The carry ablation is the readout — train loss separated no Track-A arm
    # and will not separate these.  Only the accum arm gets it: `off` has no
    # cross state and eval_carry_ablation.py refuses such models by design
    # (carried == zeroed identically).  `off` is the comparison via its wandb
    # LOSS curve at the same step.
    #
    # Choosing accum is what makes the next two possible at all — both refuse
    # gated states.  This is the diagnostic payoff of that decision.
    run=${RUN_OF[accum]}
    step=${ARG:-$(newest_model_only $run)}
    if [ -z "$step" ]; then
        echo "ERROR: no model_only_chkpt_* under $OUT/$run yet."
        exit 1
    fi
    ckpt_name=model_only_chkpt_$step
    if [ ! -d "$OUT/$run/$ckpt_name" ]; then
        echo "ERROR: no $OUT/$run/$ckpt_name (save_interval is 2500)."
        echo "  present: $(ls -d $OUT/$run/model_only_chkpt_* 2>/dev/null \
            | sed 's/.*chkpt_//' | sort -n | tr '\n' ' ')"
        exit 1
    fi

    COMMON="OUT_ROOT=$OUT CKPT_NAME=$ckpt_name BASE=$BASE_CKPT RUN=$run EVAL_TAG=b1-$step"

    # SLICE_ABLATE=both adds the drop-oldest / drop-newest conditions in the
    # same paired pass — free relative to a plain carry ablation, and it is the
    # recency-vs-accumulation question that Track A answered at 0.024.
    env $COMMON SLICE_ABLATE=both sbatch pace/eval_carry_ablation.sbatch
    echo "Submitted carry ablation + slice conditions: $run @ step $step"

    env $COMMON sbatch pace/diag_accum_buffer.sbatch
    echo "Submitted buffer diagnostic:                 $run @ step $step"

    cat <<EOF

Reading the gate (tag b1-$step, ~4h each):
  * accum carry delta > 0 with off's loss curve at or above accum's at step
    $step  ->  co-trained memory is carrying something.  Continue: 'resume'.
  * carry delta ~0  ->  co-trained memory carries nothing.  STOP the chain;
    that conclusion does not need the remaining steps.
  * Do NOT compare this delta against Track A's 0.0283 — the recurrence ramp
    runs to step 37500, so a pre-37500 checkpoint is at partial recurrence
    while every Track-A arm trained at fixed recurrence 8.
  * Track-A reference values for the slice conditions, same instrument:
    total 0.024 / drop-oldest ~0.003 / drop-newest ~0.014 (recency-dominated,
    redundant gist writes at cosine 0.94).  A DIFFERENT shape here is the
    interesting outcome — it would mean co-training changed what gets written.
  * The diagnostic runs on PG-19 while B1 trains on nemotron, so absolute
    norms/loss are off-distribution.  Read the SHAPE across chunk depth.
EOF
    ;;

add-gated)
    # Third arm, deliberately gated behind a gate read: it re-uses the same
    # `off` control, so its marginal cost is one arm.
    run=${RUN_OF[gated]}
    [ -z "$(newest_ckpt $run)" ] || { echo "SKIP $run — already started."; exit 0; }
    require_prep
    MEMORY_MODE=gated sbatch $SBATCH
    echo "Submitted: $run (mode=gated, step 0 -> $TOTAL_STEPS)"
    echo "NOTE: gated gets the plain carry ablation ONLY — the slice ablation"
    echo "  and buffer diagnostic both refuse overwritten states."
    ;;

*)
    sed -n '2,40p' "$0"
    exit 1
    ;;
esac
