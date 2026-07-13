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

# Teacher-advantage probe — full-window vs chunk-only NLL on PG-19 val for the
# BASE RDM (the distillation-teacher candidate).  Answers two questions in one
# job: (a) is there distillable signal in a longer context window, and how many
# nats; (b) where is the model's LM-loss long-context cliff (the generation
# cliff was ~1.5-2k; distill_window must sit below the LM-loss cliff).
#
# Prereq (login node, once — same dataset as the carry ablation):
#   python tools/prepare_pg19_dataset.py --tokenizer ckpts/olmo8-cortex \
#       --out data/pg19_olmo_val_len4096 --max_length 4096 --split validation
#
#   sbatch pace/eval_teacher_advantage.sbatch                 # base (default)
#   RUN=rung1-k4-v2 sbatch pace/eval_teacher_advantage.sbatch # finetuned model

cd $SLURM_SUBMIT_DIR

module load anaconda3
module load cuda/12.1.1
conda activate cortex-retro

export SCRATCH=${SCRATCH:-$HOME/scratch}
export HF_HOME=$SCRATCH/hf_cache
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

EVAL_TAG=${EVAL_TAG:-$(date +%Y%m%d)}
DATA=${DATA:-data/pg19_olmo_val_len4096}
WINDOWS=${WINDOWS:-1024,1536,2048,3072,4096}

if [ -n "$RUN" ]; then
    MODEL_DIR="cortex-retro-ft/$RUN/final_checkpoint"
    python tools/prepare_eval_checkpoint.py --base ckpts/olmo8-cortex \
        --dirs "$MODEL_DIR"
    LABEL="$RUN"
else
    MODEL_DIR="ckpts/olmo8-cortex"
    LABEL="base"
fi

OUT="eval_results/teacher_advantage_${EVAL_TAG}/${LABEL}"
mkdir -p "$OUT"
echo "=== teacher advantage: $LABEL windows=$WINDOWS -> $OUT ==="

python evals/eval_teacher_advantage.py \
    --model_name "$MODEL_DIR" \
    --data "$DATA" \
    --windows "$WINDOWS" \
    --out_dir "$OUT"
