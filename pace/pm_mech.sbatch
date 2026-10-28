#!/bin/bash
#SBATCH --account=gts-aivanova7-lab
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96GB
#SBATCH --gres=gpu:H200:1
#SBATCH --constraint=H200
#SBATCH -t 10:00:00
#SBATCH -q inferno
#SBATCH -J pm_mech
#SBATCH -o logs/Report-%A_%a.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=henryyjiang42@gmail.com

# ============================================================================
# POST-MORTEM, FORWARD-ONLY MECHANISM BATTERY -- one task per checkpoint.
#
#   sbatch --array=0-3 pace/pm_mech.sbatch
#     [0] cortex@38146   [1] cortex@11444   [2] control@38146   [3] control@11444
#
# Tasks 1 and 3 ARE the heal-end vs final comparison (Henry, 2026-10-28): the
# same instruments on the heal boundary (FineWeb-Edu only, znorm 98.0) and on
# the horizon, so every number below has a "what did the 26,700 mix-phase
# updates change" twin.  Caveat that travels with every heal-vs-final delta:
# PG-19 was OUT OF DOMAIN at 11,444 (heal-was-fineweb-only), so a PG-19 change
# across the two is training AND domain, not training alone.
#
# What each step answers (results land in eval_results/pm/<run>/step_<n>_T8/):
#   chunk1_drift/   WHERE THE CHUNK-1 TAX LIVES and WHAT BREAKS PAST THE TRAINED
#                   DEPTH (evals/diag_chunk1_drift.py header).  Control tasks
#                   run chunk1 only: the no-carry profiles the cortex numbers
#                   are read against.
#   horizon_{e,z,both}/  INFLUENCE HORIZON I(d), d = 1..7, donor damage, per
#                   channel: the trained memory horizon in nats, to set against
#                   the gate-predicted staircase (E fg 0.17, Z 0.45 at 38,146).
#   e_content/      the 2x2 with --e_null donor: E's CONTENT effect with E ON
#                   (in distribution; E_main under zeros is not -- e_dropout 0).
#   ceiling/        (38146 only) the oracle context ceiling at nc=8: how much
#                   cross-chunk signal PG-19 offers these models at all, i.e.
#                   the denominator under "memory does not beat the control".
#   mc_{none,mean64,generic64}/  (cortex@38146) the MC suite with chunk 1 given
#                   a fixed carry.  If the carry closes the 1-4 pt gap, the
#                   benchmark deficit is a missing-register effect a learned
#                   default carry would fix, not lost capability.  `none` reruns
#                   the benchmark condition on THIS load path so the three are
#                   paired item for item (records.json).
#   mc_none/        (both arms @11444) heal-end MC, for the heal-vs-final table.
#
# COST.  Measured 2026-10-28: 0.285 s per 512-token chunk forward (64-row
# carry, T=8, fp32) on a shared 5070 Ti -> ~0.2 s on an H200 (fp32, no TF32).
# Plus a checkpoint load per python step: the 10-24 benchmark logs show 6-11
# min to load shards off Lustre, so budget ~8 min x 9 loads on the cortex task.
#   cortex@38146  chunk1 ~18m + drift ~8m + horizon 3x~24m + e_content ~32m
#                 + ceiling ~12m + MC 3x~25m + loads ~1.2h   ~= 5 h
#   cortex@11444  the same minus 2 MC conditions             ~= 4 h
#   control@38146 chunk1 ~11m + ceiling ~12m + loads         ~= 45 min
#   control@11444 chunk1 ~11m + MC ~25m + loads              ~= 1 h
# 10 h wall is 2x the slowest task.  Steps run in sequence and each one's rc
# is kept, so one failed step does not cost the others.
# ============================================================================

cd $SLURM_SUBMIT_DIR
source pace/pm_common.sh
pm_env

CASES=("cortex 38146" "cortex 11444" "control 38146" "control 11444")
IDX=${SLURM_ARRAY_TASK_ID:-0}
read ARM STEP <<< "${CASES[$IDX]}"
[ -n "$ARM" ] || { echo "no case at index $IDX (use --array=0-3)"; exit 0; }
pm_resolve "$ARM" "$STEP" || exit 1

echo "=== pm_mech | $ARM @ $STEP | $CKPT -> $RES ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
git rev-parse --short HEAD 2>/dev/null | sed 's/^/    commit: /'
RC_ALL=0
step() { echo ""; echo "--- [$ARM@$STEP] $1  ($(date +%H:%M:%S))"; }

# ---- chunk-1 tax + drift ---------------------------------------------------
step "chunk1 + drift"
PARTS=chunk1,drift
[ "$ARM" = "control" ] && PARTS=chunk1
python evals/diag_chunk1_drift.py --model_name "$MODEL" --checkpoint "$CKPT" \
    $SETS --data "$PG19_VAL" --T "$T" --parts "$PARTS" \
    --max_rows 200 --bank_rows 64 --drift_chunks 202 --drift_streams 6 \
    --dtype float32 --out_dir "$RES/chunk1_drift" || RC_ALL=$?

if [ "$ARM" = "cortex" ]; then
    # ---- influence horizon, per channel ------------------------------------
    for CH in e z both; do
        step "influence horizon, channel $CH"
        python evals/eval_influence_horizon.py --model_name "$MODEL" \
            --checkpoint "$CKPT" $SETS --data "$PG19_VAL" --n_chunks 8 \
            --depths 1 2 3 4 5 6 7 --damage donor --damage_channel "$CH" \
            --T "$T" --max_examples 100 --dtype float32 \
            --out_dir "$RES/horizon_$CH" || RC_ALL=$?
    done
    # ---- E's content effect, in distribution --------------------------------
    step "2x2 E-donor (E_content)"
    python evals/eval_carry_2x2.py --model_name "$MODEL" --checkpoint "$CKPT" \
        $SETS --data "$PG19_VAL" --n_chunks 8 --T "$T" --max_examples 400 \
        --boot 2000 --dtype float32 --e_null donor --cells E1Z1,E0Z1 \
        --out_dir "$RES/e_content" || RC_ALL=$?
fi

# ---- the oracle ceiling (38146 only: the HF dir carries the 38146 config) ---
if [ "$STEP" = "38146" ]; then
    step "context ceiling"
    python evals/eval_context_ceiling.py \
        --model_name "cortex-5b/$RUN/final_checkpoint" --checkpoint "$CKPT" \
        --data "$PG19_VAL" --n_chunks 8 --context_lens 128 256 512 \
        --max_examples 100 --pos_buckets 4 --T "$T" \
        --out_dir "$RES/ceiling" || RC_ALL=$?
fi

# ---- MC: fixed-carry test (cortex@38146) and heal-end MC (both @11444) -----
MC_TASKS="hellaswag winogrande arc_easy arc_challenge piqa"
mc() {   # $1 = label, $2.. = extra args
    local label=$1; shift
    step "MC $label"
    python evals/eval_multiple_choice.py --model_name "$MODEL" \
        --checkpoint "$CKPT" $SETS --tasks $MC_TASKS --T "$T" \
        --out_dir "$RES/mc_$label" "$@" || RC_ALL=$?
}
if [ "$ARM" = "cortex" ] && [ "$STEP" = "38146" ]; then
    mc none
    for C in mean64 generic64; do
        if [ -f "$RES/chunk1_drift/$C.pt" ]; then
            mc "$C" --prefix_carry "$RES/chunk1_drift/$C.pt"
        else
            echo "SKIPPED mc_$C: chunk1 did not write $C.pt"; RC_ALL=1
        fi
    done
elif [ "$STEP" = "11444" ]; then
    mc none
fi

echo ""
echo "=== pm_mech $ARM@$STEP finished rc=$RC_ALL  ($(date +%H:%M:%S)) -> $RES ==="
exit $RC_ALL
