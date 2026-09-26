#!/bin/bash
#SBATCH --account=gts-aivanova7-lab
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH -t 0:30:00
#SBATCH -q inferno
#SBATCH -J j1_verdict
#SBATCH -o logs/Report-%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=henryyjiang42@gmail.com

# J1's VERDICT -- evals/score_j1.py over the read-out, CPU only (json + the
# PG-19 pack's row masks for book clusters; no checkpoint is opened).
#
#   sbatch --dependency=afterany:$R[:$RE] pace/j1_verdict.sbatch
#   EXPERIMENT=j3 sbatch --dependency=afterany:$R3[:$R3E] pace/j1_verdict.sbatch
#       (J3, the in-loop scratchpad: same rules, run prefix j3-, ENCODING scratch)
#   EXPERIMENT=j4 sbatch --dependency=afterany:$R4[:$R4E] pace/j1_verdict.sbatch
#       (J4, Z as its own input_embeds columns: run prefix j4-, ENCODING endpoint)
#   EXPERIMENT=j4 E_DROPOUT=0.95 VERDICT_OUT=eval_results/j4_readout/verdict_edrop0.95.json \
#       sbatch --dependency=afterany:$R4Z pace/j1_verdict.sbatch
#       (J4's Z-ONLY pair.  VERDICT_OUT is REQUIRED on this pass: without it the
#        0.95 run would overwrite the 0.25 run's verdict.json, and the two
#        answers are different experiments on the same main limbs.)
#
# afterany, not afterok, ON PURPOSE: a failed read-out task must still produce
# a verdict that NAMES the missing piece, not a job that never starts.
#
# Also copies each J1 run's cortex_diag.jsonl from scratch into logs/j1_diag/,
# so the read-gate trajectory reaches git (tier 1.5's never did).

cd $SLURM_SUBMIT_DIR

module load anaconda3
conda activate cortex-retro
export SCRATCH=${SCRATCH:-$HOME/scratch}
export HF_HOME=$SCRATCH/hf_cache
export HF_HUB_OFFLINE=1

EXPERIMENT=${EXPERIMENT:-j1}
case "$EXPERIMENT" in
    j1) ENCODING=${ENCODING:-tokens} ;;
    j3) ENCODING=${ENCODING:-scratch} ;;
    j4) ENCODING=${ENCODING:-endpoint} ;;
    *) echo "ERROR: EXPERIMENT must be j1|j3|j4; got '$EXPERIMENT'"; exit 1 ;;
esac
E_DROPOUT=${E_DROPOUT:-0.25}
OUT_ROOT=${OUT_ROOT:-cortex-retrofit}
# Where the verdict json lands.  Empty = score_j1's default, <root>/verdict.json.
# Set it whenever a SECOND E-dropout value is scored against the same main limbs
# (J4's Z-only pair at 0.95), or the second pass silently replaces the first.
VERDICT_OUT=${VERDICT_OUT:-}
OUT_ARG=""
if [ -n "$VERDICT_OUT" ]; then OUT_ARG="--out $VERDICT_OUT"; fi

mkdir -p logs/${EXPERIMENT}_diag
for d in $OUT_ROOT/${EXPERIMENT}-a3z-${ENCODING}-* $OUT_ROOT/${EXPERIMENT}-a3z-${ENCODING}-edrop*; do
    case "$d" in *-smoke) continue ;; esac
    [ -f "$d/cortex_diag.jsonl" ] && cp "$d/cortex_diag.jsonl" "logs/${EXPERIMENT}_diag/$(basename $d).jsonl"
done
ls -la logs/${EXPERIMENT}_diag/ 2>/dev/null

python evals/score_j1.py --experiment "$EXPERIMENT" \
    --root eval_results/${EXPERIMENT}_readout --encoding "$ENCODING" \
    --edrop "$E_DROPOUT" --runs_root "$OUT_ROOT" $OUT_ARG
RC=$?
echo ""
echo "Push for the next session:"
echo "  git add eval_results/${EXPERIMENT}_readout logs/${EXPERIMENT}_diag logs/Report-*.out && git commit -m '${EXPERIMENT} readout' && git push"
exit $RC
