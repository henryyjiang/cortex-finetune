#!/bin/bash
#SBATCH --account=gts-aivanova7-lab
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=64GB
#SBATCH --gres=gpu:A100:1
#SBATCH --constraint=A100-80GB
#SBATCH -t 16:00:00
#SBATCH -q inferno
#SBATCH -o logs/Report-%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=henryyjiang42@gmail.com

# Rung 1 — frozen loop + trainable memory (+prelude/coda/embeds).
# Question: does K>0 beat K=0 at all?  The loop stays byte-identical to the
# base RDM, so any delta is attributable to the memory module.
#
# Arms (submit in parallel):
#   sbatch pace/rung1_frozen_loop.sbatch                     # K=4 LM2 buffer
#   MEMORY_SLOTS=0 CCOT_DIRECT=true sbatch pace/...          # K=0 DirectCCoT
#   MEMORY_SLOTS=0 sbatch pace/...                           # no-memory control
#     (control still runs use_memory + the same cross_chunks windows, so data
#      exposure and token budget match; it just carries nothing.)

cd $SLURM_SUBMIT_DIR

module load anaconda3
module load cuda/12.1.1
conda activate cortex-retro

export SCRATCH=${SCRATCH:-$HOME/scratch}   # not set by default on PACE Phoenix
export HF_HOME=$SCRATCH/hf_cache
export HF_HUB_OFFLINE=1                 # compute nodes: cache only
export WANDB_DIR=$SCRATCH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MEMORY_SLOTS=${MEMORY_SLOTS:-4}
MEMORY_SLOTS_ITER=${MEMORY_SLOTS_ITER:-0}   # short-term M_iter (per loop iteration); 0 = off
CCOT_DIRECT=${CCOT_DIRECT:-false}
# Buffer-depth knobs: training only ever showed the buffer <= cross_chunks-1
# accumulated writes before its last read, so eval-time states (16+ writes)
# were off-distribution.  CROSS_CHUNKS=8 on the existing 4096 data gives
# 512-token windows (no re-prep, same memory).  For 1024-token windows at
# depth 8, prep data/pg19_olmo_len8192 (prepare_pg19_dataset.py
# --max_length 8192) and set MAX_LENGTH=8192 DATA_PATH=data/pg19_olmo_len8192
# CROSS_CHUNKS=8 — retained graph doubles; use the H200 variant if A100-80 OOMs.
MAX_LENGTH=${MAX_LENGTH:-4096}
CROSS_CHUNKS=${CROSS_CHUNKS:-4}
DATA_PATH=${DATA_PATH:-data/pg19_olmo_len4096}
ARM="k${MEMORY_SLOTS}"; [ "$CCOT_DIRECT" = "true" ] && ARM="${ARM}-ccot"
[ "$MEMORY_SLOTS_ITER" -gt 0 ] && ARM="${ARM}-ki${MEMORY_SLOTS_ITER}"
[ "$CROSS_CHUNKS" -ne 4 ] && ARM="${ARM}-cc${CROSS_CHUNKS}"
[ "$MAX_LENGTH" -ne 4096 ] && ARM="${ARM}-len${MAX_LENGTH}"
RUN_NAME=${RUN_NAME:-rung1-${ARM}}

# throttle stays ON in every rung (harmless on the frozen loop) so optimizer
# param-group structure is identical across rungs/resumes.
python train.py \
    --run_name $RUN_NAME \
    --out_path cortex-retro-ft \
    --model_name ckpts/olmo8-cortex \
    --preprocessed_data_path $DATA_PATH \
    --max_length $MAX_LENGTH \
    --epochs 1 \
    --micro_batch_size 1 \
    --batch_size 16 \
    --optim_config.lr 1e-4 \
    --muon.use_muon true \
    --muon.lr 0.001 \
    --throttle true \
    --no_amp false \
    --max_grad_norm 1.0 \
    --save_interval 500 \
    --seed 74 \
    --cortex.use_memory true \
    --cortex.memory_lr 5e-4 \
    --cortex.memory_slots $MEMORY_SLOTS \
    --cortex.memory_slots_iter $MEMORY_SLOTS_ITER \
    --cortex.ccot_direct $CCOT_DIRECT \
    --cortex.cross_chunks $CROSS_CHUNKS \
    --cortex.eos_from_tokens true \
    --cortex.freeze_loop true
