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

# Pre-submit gate for a B2 phase-0 launch: run the prefix path on the REAL 1B
# checkpoint at the geometry the run will use, and fail loudly on anything that
# a healthy loss curve would otherwise hide.
#
# WHY THIS IS NOT OPTIONAL (2026-08-04).  ckpts/olmo-retrofit-cortex was carrying
# a raven_modeling_minimal_cortex.py that predated the 2026-08-02 prefix rewrite.
# cortex_graft.py is imported LIVE from the repo root, but that modeling file is
# a SNAPSHOT copied into the checkpoint dir — so cortex.prefix existed, every
# --cortex.* flag was correct, the wandb config verified, and forward() never
# called prefix_pack.  The first phase-0 attempt ran 9 hours with NO
# cross-segment memory and a perfectly normal loss curve.  Recurring bug class 2.
# This script is the only check in the pipeline that reads the MODELING FILE
# rather than the config.
#
#   sbatch pace/smoke_prefix_real.sbatch
#   MODEL=ckpts/olmo8-cortex sbatch pace/smoke_prefix_real.sbatch   # Track-A base
#   FULL=1 sbatch pace/smoke_prefix_real.sbatch                     # 4 x 1024, T=8
#
# Default geometry is 2 x 1024 at T=8, which reaches max packed position 1056 —
# the case no toy config ever exercised, and the one the prefix_pos='tail' change
# introduced.  FULL=1 runs the true 4-chunk training shape; that is the backward
# graph that OOMed both B0 arms at 78.4GiB on an A100-80, so it may not fit here.
# It carries no optimizer state, so it is worth trying, but an OOM there is a
# memory limit and not a code failure.

cd $SLURM_SUBMIT_DIR

module load anaconda3
module load cuda/12.1.1
conda activate cortex-retro

export SCRATCH=${SCRATCH:-$HOME/scratch}
export HF_HOME=$SCRATCH/hf_cache
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODEL=${MODEL:-ckpts/olmo-retrofit-cortex}
CHUNK_LEN=${CHUNK_LEN:-1024}
CROSS_CHUNKS=${CROSS_CHUNKS:-2}
T_SMOKE=${T_SMOKE:-8}
if [ -n "$FULL" ]; then CHUNK_LEN=1024; CROSS_CHUNKS=4; fi

echo "=== prefix smoke: $MODEL | ${CROSS_CHUNKS} x ${CHUNK_LEN} tok | T=${T_SMOKE} ==="
echo "prefix_pack occurrences in the checkpoint's modeling file:"
grep -c prefix_pack "$MODEL/raven_modeling_minimal_cortex.py" \
    || echo "  (no modeling file at $MODEL — is it graft-prepared?)"

python tools/smoke_prefix_real.py \
    --model_name "$MODEL" \
    --chunk_len "$CHUNK_LEN" \
    --cross_chunks "$CROSS_CHUNKS" \
    --T "$T_SMOKE" \
    --device cuda --dtype bfloat16
