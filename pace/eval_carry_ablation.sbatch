#!/bin/bash
#SBATCH --account=gts-aivanova7-lab
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=64GB
#SBATCH --gres=gpu:A100:1
#SBATCH -t 0:30:00
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
#
# Reading an arm OFF its training geometry (N_CHUNKS > its cross_chunks) needs
# ACCUM_MAX raised too, or the FIFO trims the early chunks' writes away and the
# result reflects buffer capacity rather than the model — see the block below.
#   RUN=rung1-pfxaccum32-cc4-tb2-rs-pg19-ep1 N_CHUNKS=8 ACCUM_MAX=256 \
#       T_EVAL=8 SLICE_ABLATE=both sbatch pace/eval_carry_ablation.sbatch
#
# B1 / retrofit checkpoints (2026-07-29): three things differ from a Track-A
# rung run, so each is an override with the Track-A value as the default —
#   OUT_ROOT  Track A writes to cortex-retro-ft; B1 writes to cortex-retrofit.
#   CKPT_NAME B1 gates read an INTERMEDIATE model_only_chkpt_<step>, not the
#             final_checkpoint (the chain is still running).
#   BASE      B1 derives from the UNTRAINED conversion, not the trained olmo8
#             RDM.  Merging the wrong base's config into the eval checkpoint is
#             silent and wrong, so this must track the run's --model_name.
#   OUT_ROOT=cortex-retrofit CKPT_NAME=model_only_chkpt_20000 \
#       BASE=ckpts/olmo-retrofit-cortex RUN=retro-b1-k32-ga-mr8 \
#       EVAL_TAG=b1-20000 sbatch pace/eval_carry_ablation.sbatch

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
OUT_ROOT=${OUT_ROOT:-cortex-retro-ft}
CKPT_NAME=${CKPT_NAME:-final_checkpoint}
BASE=${BASE:-ckpts/olmo8-cortex}
MODEL_DIR="$OUT_ROOT/$RUN/$CKPT_NAME"

[ -d "$MODEL_DIR" ] || { echo "No such checkpoint: $MODEL_DIR"; exit 1; }

python tools/prepare_eval_checkpoint.py --base "$BASE" \
    --dirs "$MODEL_DIR"

OUT="eval_results/carry_ablation_${EVAL_TAG}/${RUN}"
mkdir -p "$OUT"
echo "=== carry ablation: $RUN -> $OUT ==="

SLICE_ABLATE=${SLICE_ABLATE:-none}
SLICE_N=${SLICE_N:-0}
SLICE_OP=${SLICE_OP:-drop}
# T_EVAL: recurrence to evaluate at.  Unset = the checkpoint's own
# config.mean_recurrence, which is NOT always the recurrence the arm trained at:
# checkpoints derived from ckpts/olmo-retrofit-cortex inherit mean_recurrence=32
# from the UNTRAINED base config (confirmed 2026-08-04 on the B1 gate, which
# prints 32 while B1 was an mr8 run sitting at ~recurrence 4 at step 22,500).
# Evaluating a memory arm at 8x its training depth is a real confound — set this.
T_EVAL=${T_EVAL:-}
# N_CHUNKS: the chunk geometry to read at.  MUST match the arm's training
# cross_chunks.  It was not passed at all before 2026-08-04, silently taking the
# python default of 4 — fine while every arm trained at cross_chunks=4, wrong
# the moment one does not (B2's arms run at 8 / 512-token chunks).  Reading an
# 8-trained arm at 4 is the same class of silent config mismatch as evaluating
# at the wrong recurrence.
N_CHUNKS=${N_CHUNKS:-4}
# ACCUM_MAX: the buffer's FIFO cap at eval.  Unset = the checkpoint's own, which
# is its TRAINING geometry (cross_chunks x accum_vecs).  That is wrong whenever
# N_CHUNKS exceeds the arm's training cross_chunks: a cc=4 arm carries cap 128,
# so reading it at N_CHUNKS=8 keeps only the newest 4 chunks' writes and drops
# the rest before they can be measured.  Its "oldest" slice is then 4 chunks
# back, not 7, and comparing its depth profile to a cc=8 arm (cap 256) measures
# BUFFER CAPACITY rather than what either model learned.  Set it to
# N_CHUNKS x accum_vecs when reading off-geometry.
ACCUM_MAX=${ACCUM_MAX:-}

python evals/eval_carry_ablation.py \
    --model_name "$MODEL_DIR" \
    --data "$DATA" \
    --n_chunks "$N_CHUNKS" \
    --slice_ablate "$SLICE_ABLATE" \
    --slice_n "$SLICE_N" \
    --slice_op "$SLICE_OP" \
    ${T_EVAL:+--T "$T_EVAL"} \
    ${ACCUM_MAX:+--accum_max "$ACCUM_MAX"} \
    --out_dir "$OUT"
