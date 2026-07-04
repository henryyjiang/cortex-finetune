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

export HF_HOME=$SCRATCH/hf_cache
export HF_HUB_OFFLINE=1
export WANDB_DIR=$SCRATCH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MEMORY_SLOTS=${MEMORY_SLOTS:-4}
LORA_RANK=${LORA_RANK:-16}
LORA_ALPHA=${LORA_ALPHA:-32}
RUN_NAME=${RUN_NAME:-rung1b-k${MEMORY_SLOTS}-lora${LORA_RANK}}

python train.py \
    --run_name $RUN_NAME \
    --out_path cortex-retro-ft \
    --model_name ckpts/olmo8-cortex \
    --preprocessed_data_path data/pg19_olmo_len4096 \
    --max_length 4096 \
    --epochs 1 \
    --micro_batch_size 2 \
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
    --cortex.freeze_loop true
