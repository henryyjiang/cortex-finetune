#!/bin/bash
#SBATCH --account=gts-aivanova7-lab
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=64GB
# H200 (141 GB): the staged unfreeze needs ~2 GB+ of headroom the 80 GB cards
# don't have (loop grads materialize on the first post-unfreeze backward).
# NOT H100 — those are 80 GB, same ceiling as the A100s.  h200 is its own gres
# type, so no memory-variant --constraint is needed (unlike the A100 40/80 split).
#SBATCH --gres=gpu:H200:1
#SBATCH -t 16:00:00
#SBATCH -q inferno
#SBATCH -o logs/Report-%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=henryyjiang42@gmail.com

# Rung 2 — staged unfreeze, single run from the base RDM.
# Phase 1 (steps 0..FREEZE_UNTIL): loop frozen, memory warms up (a built-in
# mini rung-1, so the loop is never hit with gradients from a random memory).
# Phase 2 (after FREEZE_UNTIL): loop unfreezes; throttle gives it the
# layer-wise LR (muon.lr / mean-k per batch).  Question: does the unfrozen
# loop train without blowing up?
#
# Watch train/total_norm and train/loss after step $FREEZE_UNTIL — if it
# diverges there, rung 3 (same run + L2-SP anchor) is the fallback.

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
# BACKPROP_DEPTH>0 shortens the TBPTT window to fit the post-unfreeze activation
# memory (the step-500 OOM lever); 0 = use the checkpoint's mean_backprop_depth.
# MAX_STEPS>0 caps optimizer steps (for the fast freeze_until=5 memory probe).
BACKPROP_DEPTH=${BACKPROP_DEPTH:-0}
MAX_STEPS=${MAX_STEPS:-0}
# MEM_SNAPSHOT=true records allocator history and dumps a memory_viz pickle on
# OOM (plus Peak-Mem per update in the log) — turn on for the memory probes.
MEM_SNAPSHOT=${MEM_SNAPSHOT:-false}
RUN_NAME=${RUN_NAME:-rung2-k${MEMORY_SLOTS}-unfreeze${FREEZE_UNTIL}$([ "$BACKPROP_DEPTH" != "0" ] && echo "-bpd${BACKPROP_DEPTH}")}

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
    --override_mean_backprop_depth $BACKPROP_DEPTH \
    --record_memory_snapshot $MEM_SNAPSHOT \
    --max_steps $MAX_STEPS
