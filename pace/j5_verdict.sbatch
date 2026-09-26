#!/bin/bash
#SBATCH --account=gts-aivanova7-lab
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH -t 0:30:00
#SBATCH -q inferno
#SBATCH -J j5_verdict
#SBATCH -o logs/Report-%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=henryyjiang42@gmail.com

# J5's VERDICT -- evals/score_j5.py over both read-outs, CPU only (json + the
# PG-19 pack's row masks for book clusters; no checkpoint is opened).
#
#   sbatch --dependency=afterany:$R2:$R4 pace/j5_verdict.sbatch
#
# afterany, not afterok, ON PURPOSE: a failed read-out task must still produce
# a verdict that NAMES the missing piece, not a job that never starts.
#
# BOTH STEPS IN ONE PASS.  --step 4000 is the DECIDING read-out; --also_step
# 2000 adds the trajectory, which labels nothing on its own but separates a
# null that is MOVING toward the cuts from a null that is flat
# (../j5_prereg.md S1.5, S4).  Set ALSO_STEP=0 to score the deciding step alone.
#
# Also copies each J5 run's cortex_diag.jsonl into logs/j5_diag/, so the
# embed-ratio trajectory and the e_spliced_norm proof reach git.

cd $SLURM_SUBMIT_DIR

module load anaconda3
conda activate cortex-retro
export SCRATCH=${SCRATCH:-$HOME/scratch}
export HF_HOME=$SCRATCH/hf_cache
export HF_HUB_OFFLINE=1

OUT_ROOT=${OUT_ROOT:-cortex-retrofit}
ROOT=${ROOT:-eval_results/j5_readout}
STEP=${STEP:-4000}
ALSO_STEP=${ALSO_STEP:-2000}
VERDICT_OUT=${VERDICT_OUT:-}
OUT_ARG=""
if [ -n "$VERDICT_OUT" ]; then OUT_ARG="--out $VERDICT_OUT"; fi

mkdir -p logs/j5_diag
for d in $OUT_ROOT/j5-a3z-endpoint-*; do
    case "$d" in *-smoke|*-enorm) continue ;; esac
    [ -f "$d/cortex_diag.jsonl" ] && cp "$d/cortex_diag.jsonl" "logs/j5_diag/$(basename $d).jsonl"
done
ls -la logs/j5_diag/ 2>/dev/null

python evals/score_j5.py --root "$ROOT" --step "$STEP" \
    --also_step "$ALSO_STEP" --runs_root "$OUT_ROOT" $OUT_ARG
RC=$?
echo ""
echo "Push for the next session:"
echo "  git add eval_results/j5_readout logs/j5_diag logs/Report-*.out && git commit -m 'j5 readout' && git push"
exit $RC
