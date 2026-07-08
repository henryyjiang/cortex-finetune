#!/bin/bash
#SBATCH --account=gts-aivanova7-lab
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=64GB
#SBATCH --gres=gpu:A100:1
#SBATCH --constraint=A100-80GB
#SBATCH -t 1:00:00
#SBATCH -q inferno
#SBATCH -o logs/Report-%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=henryyjiang42@gmail.com

# Fast unfreeze-OOM memory probe on the 80 GB A100 — the diagnostic arm, run in
# PARALLEL with the real rung2/rung3 H200 submissions (unknown H200 queue; this
# backfills fast on the plentiful A100s thanks to the 1 h wall time).
#
# Same recipe as rung2, but: unfreeze at step 5, stop at step 10, memory
# instrumentation ON.  Deliverables, whether or not it OOMs:
#   - Peak-Mem per update in the log -> true phase-1 steady state vs the
#     79.25 GiB ceiling (the headroom number everything hinges on)
#   - on OOM: <out_path>/<run_name>-oom-upd*-n*-k*.pickle -> open at
#     https://pytorch.org/memory_viz to see which stacks hold the ~78 GB base
# If the snapshot reveals a removable multi-GB consumer, rung2/3 can come back
# to 80 GB cards; until then the H200 jobs are the primary path.

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
FREEZE_UNTIL=${FREEZE_UNTIL:-5}
MAX_STEPS=${MAX_STEPS:-10}
# 0 = the checkpoint's mean_backprop_depth (the real recipe) — the right setting
# for measuring; the k=1/k=2 probes already proved the OOM is k-independent.
BACKPROP_DEPTH=${BACKPROP_DEPTH:-0}
MEM_SNAPSHOT=${MEM_SNAPSHOT:-true}
RUN_NAME=${RUN_NAME:-probe-mem-k${MEMORY_SLOTS}-unfreeze${FREEZE_UNTIL}$([ "$BACKPROP_DEPTH" != "0" ] && echo "-bpd${BACKPROP_DEPTH}")}

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
