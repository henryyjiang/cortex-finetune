#!/bin/bash
# P0.5 — GSM8K + the MC suite vs training step, across the B2 checkpoint ladder.
# cortex_next_phase_framework.md §6.  2026-09-15.
#
# WHAT IT DECIDES.  Issue 3: GSM8K collapsed to 2.80% on b2-final against the
# static-depth OLMo-2 baseline's 35.25%, and the project has never been able to
# say whether that is a BUDGET fact or a MECHANISM fact.  McLeish's own ladder
# for this exact base, arrangement and train recurrence says 8.95% at 1.05B,
# 21.99% at 5.24B, 44.35% at 50.3B -- i.e. the capability arrives late and this
# run stopped at 5.0B.  If cortex's curve tracks that shape and merely sits
# below it, GSM8K is a conversion-budget artifact, the issue closes, and C-plain
# is never needed.  If the curve is FLAT while McLeish's rises, the mechanism is
# eating it and that is a finding.  Either way the answer is forward-only and
# the checkpoints already exist.
#
# THE REFERENCE CURVE EXISTS AND IS NOT WHAT ITS FILENAME SUGGESTS.
# paper_plots/data/olmo_50k_steps.jsonl is McLeish's GSM8K+MATH ladder, NOT a
# loss curve, and it pins the reference batch at 1,048,576 tok/step.
# paper_plots/ is gone from the working tree; recover it with
#   git cat-file -p bf9e9b0195b8db27ad3944f433aba7e20c216a94
#
# STEP -> TOKENS.  B2 ran 4 x 4096 = 16,384 tokens/optimizer-step, so
# step x 16,384 is the x-axis.  Landmarks: 64,100 = 1.05B (McLeish's first
# rung), 91,552 = 1.500B (the heal/arm boundary), 305,176 = 5.000B (b2-final).
# Plot against TOKENS, not steps, or the comparison to a 1,048,576-tok/step
# reference is off by 64x.
#
# TWO OVERRIDES ARE MANDATORY AND BOTH HAVE ALREADY COST A HEADLINE.
#   T_OVERRIDE=8   retrofit configs inherit mean_recurrence=32 from the
#                  UNTRAINED base (finding 0c).  Unset, this evaluates an mr8
#                  arm at T=32 -- 4x the trained depth and 4x the wall clock.
#   PREP_BASE=ckpts/olmo-retrofit-cortex
#                  prepare_eval_checkpoint.py merges a base dir's modeling file
#                  and config INTO the checkpoint.  Merging olmo8-cortex into a
#                  retrofit checkpoint is silent and wrong.  This is the
#                  RETROFIT line, so it takes the retrofit base.
# Both are set here and neither should be overridden without a reason written
# down.
#
# THE LADDER IS TWO RUNS, NOT ONE.  retro-b2-heal covers 2,500..90,000 and
# retro-b2-acc32-cc8-mr8 covers 92,500..305,000 (+ final_checkpoint).  They are
# one continuous training chain, so they plot as one curve -- but they live in
# separate run dirs and the heal half has NO arm corpus and NO full recurrence
# before step 9,537-equivalent.  Label the boundary on the plot.
#
# DO NOT PRUNE THE LADDER.  save_checkpoint never rotates and these
# model_only_chkpt_* dirs are this probe's only data (framework §9).
#
# SUBSAMPLING.  The full ladder is ~122 rungs at 2,500-step spacing; at ~2-4h
# per rung that is not a thing anyone should submit.  STRIDE picks every Nth
# rung.  The default 10 (= 25,000 steps = 0.41B) gives ~13 jobs and resolves a
# curve; drop to 5 only if the shape is ambiguous.
#
#   bash pace/submit_p05_ladder.sh                  # dry run, prints the plan
#   GO=1 bash pace/submit_p05_ladder.sh             # submit
#   GO=1 STRIDE=5 bash pace/submit_p05_ladder.sh    # denser
#   GO=1 SKIP_MC=true bash pace/submit_p05_ladder.sh   # GSM8K only, ~3x faster
#   GO=1 ONLY=arm bash pace/submit_p05_ladder.sh    # skip the heal half
set -eu
cd "$(dirname "$0")/.."

TAG=${TAG:-p05-$(date +%Y%m%d)}
OUT_ROOT=${OUT_ROOT:-cortex-retrofit}
HEAL_RUN=${HEAL_RUN:-retro-b2-heal}
ARM_RUN=${ARM_RUN:-retro-b2-acc32-cc8-mr8}
PREP_BASE=${PREP_BASE:-ckpts/olmo-retrofit-cortex}
T_OVERRIDE=${T_OVERRIDE:-8}
GSM8K_MAX=${GSM8K_MAX:-500}
SKIP_MC=${SKIP_MC:-false}
STRIDE=${STRIDE:-10}
ONLY=${ONLY:-all}
GO=${GO:-0}
TOK_PER_STEP=${TOK_PER_STEP:-16384}

echo "=== P0.5 ladder | tag=$TAG | T=$T_OVERRIDE | prep_base=$PREP_BASE ==="
echo "    stride=$STRIDE  gsm8k_max=$GSM8K_MAX  skip_mc=$SKIP_MC  only=$ONLY"
echo "    tokens/step=$TOK_PER_STEP"
echo

# Discover rungs NUMERICALLY, never lexically -- model_only_chkpt_9500 must not
# outrank model_only_chkpt_150000.  sort -t_ -k4 -n keys on the trailing integer.
rungs_for() {
    local run="$1"
    [ -d "$OUT_ROOT/$run" ] || return 0
    find "$OUT_ROOT/$run" -maxdepth 1 -type d -name 'model_only_chkpt_*' \
        -printf '%f\n' 2>/dev/null \
        | sed 's/^model_only_chkpt_//' \
        | grep -E '^[0-9]+$' \
        | sort -n
}

PLAN=()
add_run() {
    local run="$1" i=0
    local steps; steps=$(rungs_for "$run")
    [ -n "$steps" ] || { echo "  (no rungs under $OUT_ROOT/$run)"; return 0; }
    local n; n=$(echo "$steps" | wc -l)
    echo "  $run: $n rungs, $(echo "$steps" | head -1) .. $(echo "$steps" | tail -1)"
    for s in $steps; do
        # Keep every STRIDEth rung, and ALWAYS keep the last one of each run --
        # the heal's last rung is the 1.5B boundary point and the arm's is the
        # closest thing to b2-final in the ladder.
        if [ $((i % STRIDE)) -eq 0 ] || [ "$s" = "$(echo "$steps" | tail -1)" ]; then
            PLAN+=("$run:$s")
        fi
        i=$((i + 1))
    done
}

[ "$ONLY" = "arm" ] || add_run "$HEAL_RUN"
[ "$ONLY" = "heal" ] || add_run "$ARM_RUN"

if [ "${#PLAN[@]}" -eq 0 ]; then
    echo
    echo "REFUSING TO SUBMIT: no checkpoints found under $OUT_ROOT/."
    echo "This is the empty-scan failure -- nothing was looked at, which is not"
    echo "the same as nothing being there.  Check OUT_ROOT and the run names,"
    echo "and confirm the ladder was not pruned (framework §9 says keep it)."
    exit 2
fi

echo
echo "  will submit ${#PLAN[@]} jobs:"
for spec in "${PLAN[@]}"; do
    run=${spec%%:*}; step=${spec#*:}
    tok=$(( step * TOK_PER_STEP ))
    printf '    %-28s step %7s  = %6.3fB tokens\n' "$run" "$step" \
        "$(echo "$tok" | awk '{printf "%.3f", $1/1e9}')"
done
echo

if [ "$GO" != "1" ]; then
    echo "  DRY RUN.  Re-run with GO=1 to submit."
    echo "  Each job is one pace/eval_basic.sbatch: 5 MC tasks + GSM8K at"
    echo "  $GSM8K_MAX examples, T=$T_OVERRIDE.  Results land in"
    echo "  eval_results/basic_${TAG}-<run>-<step>/."
    exit 0
fi

for spec in "${PLAN[@]}"; do
    run=${spec%%:*}; step=${spec#*:}
    # The var-assignment PREFIX on the sbatch command itself, not a plain
    # assignment on its own line: a plain assignment is a shell variable and
    # sbatch never sees it.  That is the ARM_DATA trap, and it cost 1.7B tokens
    # of training on the wrong corpus.
    RUN="$run" \
    OUT_ROOT="$OUT_ROOT" \
    CKPT_NAME="model_only_chkpt_${step}" \
    PREP_BASE="$PREP_BASE" \
    T_OVERRIDE="$T_OVERRIDE" \
    GSM8K_MAX="$GSM8K_MAX" \
    SKIP_MC="$SKIP_MC" \
    EVAL_TAG="${TAG}-${run}-${step}" \
        sbatch pace/eval_basic.sbatch
done

echo
echo "  Submitted ${#PLAN[@]} jobs.  Read the banner on each:"
echo "    grep -m1 '^=== basic eval' logs/Report-<job>.out"
echo "  and CONFIRM it names model_only_chkpt_<step>, not final_checkpoint --"
echo "  eval_basic defaults to final_checkpoint and a dropped CKPT_NAME would"
echo "  silently evaluate the same model N times into N different output dirs."
