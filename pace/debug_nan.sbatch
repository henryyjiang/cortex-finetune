#!/bin/bash
#SBATCH --account=gts-aivanova7-lab
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=64GB
#SBATCH --gres=gpu:A100:1
#SBATCH --constraint=A100-80GB
#SBATCH -t 00:30:00
#SBATCH -q inferno
#SBATCH -o logs/Report-%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=henryyjiang42@gmail.com

# ── NaN/Inf localizer run ────────────────────────────────────────────────────
# Runs ONE of the three diverging rungs for a few steps with --debug_nan, which:
#   * installs forward hooks that name the FIRST module to emit a non-finite
#     output from finite inputs  -> pins a FORWARD nan (rung2 / rung3), and
#   * runs fwd+bwd under torch.autograd.set_detect_anomaly (backward nan), and
#   * on a non-finite grad-norm, lists the param tensors carrying non-finite
#     grads -> pins an INF GRADIENT (rung1b's BPTT explosion).
# Short (max_steps 3), no checkpoints written (save_interval -1).
#
# Usage:
#   RUNG=1b sbatch pace/debug_nan.sbatch          # forward-finite / inf-grad case
#   RUNG=2  sbatch pace/debug_nan.sbatch          # forward-nan case
#   RUNG=3  sbatch pace/debug_nan.sbatch          # forward-nan + L2-SP
#   RUNG=2 FIX_NUM_STEPS=true sbatch pace/debug_nan.sbatch   # depth-[0,1] sanity
#
# FIX_NUM_STEPS=true pins num_steps=[0,1] (one grad step, ~no recurrence): if the
# run goes finite, the divergence is depth-driven; if it still nan's, it's not.

cd $SLURM_SUBMIT_DIR

module load anaconda3
module load cuda/12.1.1
conda activate cortex-retro

export SCRATCH=${SCRATCH:-$HOME/scratch}
export HF_HOME=$SCRATCH/hf_cache
export HF_HUB_OFFLINE=1
export WANDB_DIR=$SCRATCH
export WANDB_MODE=offline          # debug run — don't clutter the wandb project
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# The nan is persistent (it fired on all 20 steps last run), so it will
# reproduce here without forcing determinism.  This env var is a no-op unless
# torch.use_deterministic_algorithms(True) is set in code; left in as a harmless
# prerequisite in case we later add a deterministic toggle.
export CUBLAS_WORKSPACE_CONFIG=:4096:8

RUNG=${RUNG:-2}
FIX_NUM_STEPS=${FIX_NUM_STEPS:-false}
MEMORY_SLOTS=${MEMORY_SLOTS:-4}

# rung-specific cortex flags (mirror pace/rung1b_lora_loop | rung2_staged_unfreeze | rung3_l2sp)
case "$RUNG" in
  1b) RUNG_ARGS="--cortex.lora_rank 16 --cortex.lora_alpha 32" ;;
  2)  RUNG_ARGS="--cortex.freeze_loop_until_step 500" ;;
  3)  RUNG_ARGS="--cortex.freeze_loop_until_step 500 --cortex.l2sp_coeff 1e-3" ;;
  *)  echo "Set RUNG to one of: 1b, 2, 3 (got '$RUNG')"; exit 1 ;;
esac

RUN_NAME=${RUN_NAME:-debug-nan-rung${RUNG}$([ "$FIX_NUM_STEPS" = "true" ] && echo "-fixk")}

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
    --save_interval -1 \
    --max_steps 3 \
    --seed 74 \
    --debug_nan true \
    --fix_num_steps $FIX_NUM_STEPS \
    --cortex.use_memory true \
    --cortex.memory_lr 5e-4 \
    --cortex.memory_slots $MEMORY_SLOTS \
    --cortex.cross_chunks 4 \
    --cortex.eos_from_tokens true \
    --cortex.freeze_loop true \
    $RUNG_ARGS
