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

# Carry-vs-zeroed M_cross ablation for ONE model — per-chunk PG-19 val loss
# with the buffer carried vs discarded (paired, seeded; chunk 1 = control).
# The most sensitive detector of whether the buffer carries ANYTHING.
#
# Prereq (login node, once):
#   python tools/prepare_pg19_dataset.py --tokenizer ckpts/olmo8-cortex \
#       --out data/pg19_olmo_val_len4096 --max_length 4096 --split validation
#
#   RUN=rung1-k4-v2 sbatch pace/eval_carry_ablation.sbatch
#   RUN=rung1-k0-ccot-v2 sbatch pace/eval_carry_ablation.sbatch
#   RUN=rung2-k4-unfreeze500 sbatch pace/eval_carry_ablation.sbatch
# (rung1-k0-v2 has no cross state — the script refuses it by design.)
#
# A0.1 slice ablation (AccumCCoT models only, two-track plan): also run the
# carried-with-oldest/newest-vectors-dropped conditions — is all the signal
# in the most recent chunk's vectors?
#   RUN=rung1-k0-acc4v-tb2-rs-ep3-rcl SLICE_ABLATE=both \
#       sbatch pace/eval_carry_ablation.sbatch
# SLICE_N (default 0 = one chunk's worth) / SLICE_OP (drop|zero) refine it.

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
MODEL_DIR="cortex-retro-ft/$RUN/final_checkpoint"

python tools/prepare_eval_checkpoint.py --base ckpts/olmo8-cortex \
    --dirs "$MODEL_DIR"

OUT="eval_results/carry_ablation_${EVAL_TAG}/${RUN}"
mkdir -p "$OUT"
echo "=== carry ablation: $RUN -> $OUT ==="

SLICE_ABLATE=${SLICE_ABLATE:-none}
SLICE_N=${SLICE_N:-0}
SLICE_OP=${SLICE_OP:-drop}

python evals/eval_carry_ablation.py \
    --model_name "$MODEL_DIR" \
    --data "$DATA" \
    --slice_ablate "$SLICE_ABLATE" \
    --slice_n "$SLICE_N" \
    --slice_op "$SLICE_OP" \
    --out_dir "$OUT"
