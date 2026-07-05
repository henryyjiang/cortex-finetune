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
CCOT_DIRECT=${CCOT_DIRECT:-false}
ARM="k${MEMORY_SLOTS}"; [ "$CCOT_DIRECT" = "true" ] && ARM="${ARM}-ccot"
RUN_NAME=${RUN_NAME:-rung1-${ARM}}

# throttle stays ON in every rung (harmless on the frozen loop) so optimizer
# param-group structure is identical across rungs/resumes.
python train.py \
    --run_name $RUN_NAME \
    --out_path cortex-retro-ft \
    --model_name ckpts/olmo8-cortex \
    --preprocessed_data_path data/pg19_olmo_len4096 \
    --max_length 4096 \
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
    --cortex.ccot_direct $CCOT_DIRECT \
    --cortex.cross_chunks 4 \
    --cortex.eos_from_tokens true \
    --cortex.freeze_loop true
