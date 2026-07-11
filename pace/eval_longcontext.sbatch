#!/bin/bash
#SBATCH --account=gts-aivanova7-lab
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=64GB
#SBATCH --gres=gpu:A100:1
#SBATCH -t 16:00:00
#SBATCH -q inferno
#SBATCH -o logs/Report-%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=henryyjiang42@gmail.com

# Long-context eval (BABILong qa1-qa3 + LongMemEval) for ONE model — the
# headline K>0 vs K=0 memory numbers.  One model per job (BLIMP lesson from
# cortex-main: multi-model eval jobs hit the 24 h wall).
#
#   RUN=rung1-k4 EVAL_TAG=20260710 sbatch pace/eval_longcontext.sbatch
#   BASE=true    EVAL_TAG=20260710 sbatch pace/eval_longcontext.sbatch   # unfinetuned base control
#
# Prereqs (login node): datasets downloaded (evals/download_datasets.py) and
# checkpoints fixed for eval loading (tools/prepare_eval_checkpoint.py) — the
# fix also runs here idempotently as a belt-and-suspenders (pure file ops).
# T is NOT passed: evals read mean_recurrence (8) + memory flags from the
# checkpoint config; memory carry is auto-detected via has_cross_state.

cd $SLURM_SUBMIT_DIR

module load anaconda3
module load cuda/12.1.1
conda activate cortex-retro

export SCRATCH=${SCRATCH:-$HOME/scratch}   # not set by default on PACE Phoenix
export HF_HOME=$SCRATCH/hf_cache
export HF_HUB_OFFLINE=1                 # compute nodes: cache only
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

EVAL_TAG=${EVAL_TAG:-$(date +%Y%m%d)}
BASE=${BASE:-false}
if [ "$BASE" = "true" ]; then
    LABEL="base"
    MODEL_DIR="ckpts/olmo8-cortex"
else
    [ -n "$RUN" ] || { echo "Set RUN=<run_name> (or BASE=true)"; exit 1; }
    LABEL="$RUN"
    MODEL_DIR="cortex-retro-ft/$RUN/final_checkpoint"
    python tools/prepare_eval_checkpoint.py --base ckpts/olmo8-cortex \
        --dirs "$MODEL_DIR"
fi

OUT_ROOT="eval_results/longcontext_${EVAL_TAG}/${LABEL}"
mkdir -p "$OUT_ROOT"
echo "=== long-context eval: $LABEL  ($MODEL_DIR) -> $OUT_ROOT ==="

python evals/eval_babilong.py \
    --model_name "$MODEL_DIR" \
    --dataset_path data/BABILong \
    --tasks qa1 qa2 qa3 \
    --out_dir "$OUT_ROOT/babilong"

python evals/eval_longmemeval.py \
    --model_name "$MODEL_DIR" \
    --dataset_path data/LongMemEval \
    --out_dir "$OUT_ROOT/longmemeval"
