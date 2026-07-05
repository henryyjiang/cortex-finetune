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

# Rung 3 — rung 2 + L2-SP anchor (the stability fallback).
# Identical staged-unfreeze recipe, plus coeff * ||theta_loop - theta_base||^2
# on the backward objective, anchoring the unfrozen loop to the pretrained
# weights.  Only needed if rung 2 destabilizes after the unfreeze; can also be
# submitted in parallel as insurance (the logged train/loss stays pure LM loss,
# so curves are directly comparable to rung 2).
#
# L2SP_COEFF: start at 1e-3; if the loop still drifts/blows up try 1e-2, if it
# pins the loop too hard (loss identical to rung 1) try 1e-4.

cd $SLURM_SUBMIT_DIR

module load anaconda3
module load cuda/12.1.1
conda activate cortex-retro

export SCRATCH=${SCRATCH:-$HOME/scratch}   # not set by default on PACE Phoenix
export HF_HOME=$SCRATCH/hf_cache
export HF_HUB_OFFLINE=1
export WANDB_DIR=$SCRATCH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MEMORY_SLOTS=${MEMORY_SLOTS:-4}
FREEZE_UNTIL=${FREEZE_UNTIL:-500}
L2SP_COEFF=${L2SP_COEFF:-1e-3}
RUN_NAME=${RUN_NAME:-rung3-k${MEMORY_SLOTS}-l2sp${L2SP_COEFF}}

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
    --cortex.cross_chunks 4 \
    --cortex.eos_from_tokens true \
    --cortex.freeze_loop true \
    --cortex.freeze_loop_until_step $FREEZE_UNTIL \
    --cortex.l2sp_coeff $L2SP_COEFF
