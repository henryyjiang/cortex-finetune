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

# B0 — retrofit-recurrence parity run (two-track plan §B0): McLeish's OLMo
# conversion recipe (shells/olmo.sh, Figure 6/7) run from his UNTRAINED
# converted checkpoint, with the memory buffer co-trained from step 0 and the
# backbone fully plastic.  The memory read is zero-init, so step-0 == McLeish
# baseline exactly (tests/test_b0_retrofit.py asserts this bitwise on the real
# modeling file).
#
# GATE B0: the memory arm's wandb loss curve must TRACK the no-carry control
# arm for the early steps.  Divergence = memory-wiring bug (not recurrence
# instability — McLeish already paid that cost).  Per design principle 0.1.3,
# NO stabilizers here (no throttle/L2-SP/spectral reg); watch train/total_norm
# and reintroduce one at a time only if instability shows.
#
# Arms (submit all three; the control gives the parity reference on OUR data):
#   MEMORY_MODE=accum sbatch pace/b0_retrofit_parity.sbatch   # AccumCCoT (4 vec, tb2)
#   MEMORY_MODE=gated sbatch pace/b0_retrofit_parity.sbatch   # GatedAccumBuffer k=32
#   MEMORY_MODE=off   sbatch pace/b0_retrofit_parity.sbatch   # no-carry control
# (MEMORY_MODE=off still runs use_memory + the same chunked 1024-token windows
#  — it just carries nothing, so data exposure and window length match.)
#
# Prep (login node, once):
#   python tools/prepare_cortex_checkpoint.py \
#       --src smcleish/Recurrent-OLMo-2-0425-untrained \
#       --dst ckpts/olmo-retrofit-cortex --variant olmo
#   python tools/prepare_packed_dataset.py --tokenizer ckpts/olmo-retrofit-cortex \
#       --out data/nemotron_math_olmo_len4096 --max_length 4096 \
#       --max_tokens 400_000_000
#
# Recipe provenance (kept from shells/olmo.sh): lr 5e-5, muon lr 0.001,
# scheduler cooldown 0.6 / warmup 0.0025, max_grad_norm 1.0,
# mean_recurrence_schedule warmup 0.75 "1-sqrt" (max_mean_rec 8 = the
# train-recurrence-8 variant Track A's base uses).  Deviations, both shared
# by ALL arms so the parity comparison is internal: (a) data is OUR nemotron
# 4plus packing (McLeish's parquet lives on LLNL storage), rows at 4096 =
# 4 x 1024-token chunks so each forward keeps McLeish's 1024 window while the
# memory chains across them; (b) batch_size 4 x 4096 tokens = 16,384
# tokens/step, matching McLeish's 16 x 1024.  B1 = same script with
# MAX_STEPS=50000 (and a longer walltime / resume chain).

cd $SLURM_SUBMIT_DIR

module load anaconda3
module load cuda/12.1.1
conda activate cortex-retro

export SCRATCH=${SCRATCH:-$HOME/scratch}
export HF_HOME=$SCRATCH/hf_cache
export HF_HUB_OFFLINE=1
export WANDB_DIR=$SCRATCH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MEMORY_MODE=${MEMORY_MODE:-accum}          # accum | gated | off
MAX_STEPS=${MAX_STEPS:-1500}               # parity gate; B1 = 50000
MAX_MEAN_REC=${MAX_MEAN_REC:-8}
DATA_PATH=${DATA_PATH:-data/nemotron_math_olmo_len4096}
MODEL=${MODEL:-ckpts/olmo-retrofit-cortex}
CROSS_CHUNKS=${CROSS_CHUNKS:-4}
MAX_LENGTH=${MAX_LENGTH:-4096}

case "$MEMORY_MODE" in
accum)
    MEMORY_SLOTS=0; ACCUM_CCOT=true;  GATED_ACCUM=false; CARRY_GRAD_CHUNKS=2
    ARM="acc4v-tb2";;
gated)
    MEMORY_SLOTS=32; ACCUM_CCOT=false; GATED_ACCUM=true; CARRY_GRAD_CHUNKS=0
    ARM="k32-ga";;
off)
    MEMORY_SLOTS=0; ACCUM_CCOT=false; GATED_ACCUM=false; CARRY_GRAD_CHUNKS=0
    ARM="base";;
*)
    echo "MEMORY_MODE must be accum|gated|off"; exit 1;;
esac
RUN_NAME=${RUN_NAME:-retro-b0-${ARM}-mr${MAX_MEAN_REC}}

python train.py \
    --run_name $RUN_NAME \
    --out_path cortex-retrofit \
    --model_name $MODEL \
    --preprocessed_data_path $DATA_PATH \
    --max_length $MAX_LENGTH \
    --epochs 1 \
    --max_steps $MAX_STEPS \
    --micro_batch_size 1 \
    --batch_size 4 \
    --optim_config.lr 5e-5 \
    --scheduler_args.cooldown 0.6 \
    --scheduler_args.warmup 0.0025 \
    --muon.use_muon true \
    --muon.lr 0.001 \
    --no_amp false \
    --max_grad_norm 1.0 \
    --compile false \
    --save_interval 500 \
    --seed 74 \
    --mean_recurrence_schedule.turn_on true \
    --mean_recurrence_schedule.warmup 0.75 \
    --mean_recurrence_schedule.max_mean_rec $MAX_MEAN_REC \
    --mean_recurrence_schedule.warmup_type "1-sqrt" \
    --cortex.use_memory true \
    --cortex.memory_lr 5e-4 \
    --cortex.memory_slots $MEMORY_SLOTS \
    --cortex.accum_ccot $ACCUM_CCOT \
    --cortex.gated_accum $GATED_ACCUM \
    --cortex.carry_grad_chunks $CARRY_GRAD_CHUNKS \
    --cortex.cross_chunks $CROSS_CHUNKS \
    --cortex.eos_from_tokens true \
    --cortex.freeze_loop false
