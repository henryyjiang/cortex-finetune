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

ENCODING=${ENCODING:-tokens}
E_DROPOUT=${E_DROPOUT:-0.25}
OUT_ROOT=${OUT_ROOT:-cortex-retrofit}

mkdir -p logs/j1_diag
for d in $OUT_ROOT/j1-a3z-${ENCODING}-* $OUT_ROOT/j1-a3z-${ENCODING}-edrop*; do
    case "$d" in *-smoke) continue ;; esac
    [ -f "$d/cortex_diag.jsonl" ] && cp "$d/cortex_diag.jsonl" "logs/j1_diag/$(basename $d).jsonl"
done
ls -la logs/j1_diag/ 2>/dev/null

python evals/score_j1.py --root eval_results/j1_readout --encoding "$ENCODING" \
    --edrop "$E_DROPOUT" --runs_root "$OUT_ROOT"
RC=$?
echo ""
echo "Push for the next session:"
echo "  git add eval_results/j1_readout logs/j1_diag logs/Report-*.out && git commit -m 'j1 readout' && git push"
exit $RC
