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

# A0.2 — AccumCCoT buffer diagnostics (two-track plan): per-chunk buffer
# stats (vector count, norms, new-vs-old cosine) + carried loss + greedy
# generation probes on LONG chunk chains, to confirm/refute the buffer-OOD
# mechanism behind the >= 4k generation collapse.
#
# Prereq (login node, once — windows LONGER than training's 4096):
#   python tools/prepare_pg19_dataset.py --tokenizer ckpts/olmo8-cortex \
#       --out data/pg19_olmo_val_len16384 --max_length 16384 --split validation
#
#   RUN=rung1-k0-acc4v-tb2-rs-ep3-rcl sbatch pace/diag_accum_buffer.sbatch

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
DATA=${DATA:-data/pg19_olmo_val_len16384}
CHUNK_LEN=${CHUNK_LEN:-1024}
GEN_TOKENS=${GEN_TOKENS:-32}
MODEL_DIR="cortex-retro-ft/$RUN/final_checkpoint"

python tools/prepare_eval_checkpoint.py --base ckpts/olmo8-cortex \
    --dirs "$MODEL_DIR"

OUT="eval_results/diag_accum_buffer_${EVAL_TAG}/${RUN}"
mkdir -p "$OUT"
echo "=== accum buffer diagnostics: $RUN -> $OUT ==="

python evals/diag_accum_buffer.py \
    --model_name "$MODEL_DIR" \
    --data "$DATA" \
    --chunk_len "$CHUNK_LEN" \
    --gen_tokens "$GEN_TOKENS" \
    --out_dir "$OUT"
