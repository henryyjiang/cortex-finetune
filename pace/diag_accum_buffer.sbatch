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
#
# B1 / retrofit checkpoints (2026-07-29): OUT_ROOT / CKPT_NAME / BASE
# overrides, Track-A values as defaults.  B1 writes to cortex-retrofit, gates
# read an intermediate model_only_chkpt_<step>, and B1 derives from the
# UNTRAINED conversion (ckpts/olmo-retrofit-cortex) — merging the wrong base's
# config into the eval checkpoint is silent and wrong.
#   OUT_ROOT=cortex-retrofit CKPT_NAME=model_only_chkpt_20000 \
#       BASE=ckpts/olmo-retrofit-cortex RUN=retro-b1-acc4v-tb2-mr8 \
#       EVAL_TAG=b1-20000 sbatch pace/diag_accum_buffer.sbatch
# NOTE for B1 reads: this diagnostic runs on PG-19, while B1 trains on
# nemotron, so the ABSOLUTE loss/norms are off-distribution (same caveat as
# the acc4v-nemo read: cos 0.53 vs 0.94, norms 1.03 vs 5.81).  What transfers
# is the SHAPE across chunk depth — drift, collapse, generation degradation.

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
OUT_ROOT=${OUT_ROOT:-cortex-retro-ft}
CKPT_NAME=${CKPT_NAME:-final_checkpoint}
BASE=${BASE:-ckpts/olmo8-cortex}
MODEL_DIR="$OUT_ROOT/$RUN/$CKPT_NAME"

[ -d "$MODEL_DIR" ] || { echo "No such checkpoint: $MODEL_DIR"; exit 1; }

python tools/prepare_eval_checkpoint.py --base "$BASE" \
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
