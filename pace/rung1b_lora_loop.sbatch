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

# Rung 1b — LoRA-on-loop: the middle rung between frozen (rung 1) and fully
# unfrozen (rung 2).  The base loop stays frozen (freeze_loop=true) but adapts
# to the memory's presence through low-rank hooks on every loop linear
# (out = Wx + (alpha/r)*B(Ax), B zero-init -> step-0 == base).  LoRA params
# ride the memory-LR group (they contain 'cortex' in the name), so they train
# at MEMORY_LR alongside the buffers.  Directly addresses the historical
# failure mode: the unfrozen pretrained loop wouldn't train.

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
LORA_RANK=${LORA_RANK:-16}
LORA_ALPHA=${LORA_ALPHA:-32}
# BACKPROP_DEPTH>0 shortens the TBPTT window (0 = checkpoint's depth) — the
# fallback lever if the LoRA grads genuinely explode through the unroll.
# MAX_STEPS>0 caps optimizer steps (for cheap probes).  MEM_SNAPSHOT=true
# records allocator history + dumps a memory_viz pickle on OOM.
BACKPROP_DEPTH=${BACKPROP_DEPTH:-0}
MAX_STEPS=${MAX_STEPS:-0}
MEM_SNAPSHOT=${MEM_SNAPSHOT:-false}
# DEBUG_NAN=true installs the forward-nan localizer + adapter max|input|
# tracker (heavy sync overhead — probes only).
DEBUG_NAN=${DEBUG_NAN:-false}
RUN_NAME=${RUN_NAME:-rung1b-k${MEMORY_SLOTS}-lora${LORA_RANK}-a${LORA_ALPHA}$([ "$BACKPROP_DEPTH" != "0" ] && echo "-bpd${BACKPROP_DEPTH}")}

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
    --cortex.lora_rank $LORA_RANK \
    --cortex.lora_alpha $LORA_ALPHA \
    --cortex.cross_chunks 4 \
    --cortex.eos_from_tokens true \
    --cortex.freeze_loop true \
    --override_mean_backprop_depth $BACKPROP_DEPTH \
    --record_memory_snapshot $MEM_SNAPSHOT \
    --debug_nan $DEBUG_NAN \
    --max_steps $MAX_STEPS
