#!/bin/bash
#SBATCH --account=gts-aivanova7-lab
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=64GB
#SBATCH --gres=gpu:A100:1
#SBATCH -t 4:00:00
#SBATCH -q inferno
#SBATCH -o logs/Report-%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=henryyjiang42@gmail.com

# Carried-vs-zeroed ITER-STATE ablation for ONE dense-arm model — the
# cross-window readout for the M_iter / ccot_iter (no-boundary-crossing)
# final arms.  Pools each window's final per-position buffer and seeds the
# next window; paired against zero-seeded chunks.  Chunk 1 = control.
#
#   RUN=rung1-k0-ki4-cc1-len1024-ep4  sbatch pace/eval_iter_carry.sbatch
#   RUN=rung1-k0-ci-cc1-len1024-ep4   sbatch pace/eval_iter_carry.sbatch
#   RUN=rung1-k0-ki4-cc1-len1024-nemo sbatch pace/eval_iter_carry.sbatch
#   RUN=rung1-k0-ci-cc1-len1024-nemo  sbatch pace/eval_iter_carry.sbatch
#   # recall-heavy probe data (decisive):
#   RUN=<name> EVAL_TAG=$(date +%Y%m%d)-rcl100 \
#       DATA=data/pg19_olmo_val_recall100_len4096 sbatch pace/eval_iter_carry.sbatch
# (models without M_iter/ccot_iter are refused by design — use
#  pace/eval_carry_ablation.sbatch for M_cross/CCoT models.)

cd $SLURM_SUBMIT_DIR

module load anaconda3
module load cuda/12.1.1
conda activate cortex-retro

export SCRATCH=${SCRATCH:-$HOME/scratch}
export HF_HOME=$SCRATCH/hf_cache
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

[ -n "$RUN" ] || { echo "Set RUN=<run_name>"; exit 1; }
EVAL_TAG=${EVAL_TAG:-$(date +%Y%m%d)}
DATA=${DATA:-data/pg19_olmo_val_len4096}
POOL=${POOL:-last}
MODEL_DIR="cortex-retro-ft/$RUN/final_checkpoint"

python tools/prepare_eval_checkpoint.py --base ckpts/olmo8-cortex \
    --dirs "$MODEL_DIR"

OUT="eval_results/iter_carry_${EVAL_TAG}/${RUN}"
mkdir -p "$OUT"
echo "=== iter-carry ablation: $RUN (pool=$POOL) -> $OUT ==="

python evals/eval_iter_carry_ablation.py \
    --model_name "$MODEL_DIR" \
    --data "$DATA" \
    --pool "$POOL" \
    --out_dir "$OUT"
