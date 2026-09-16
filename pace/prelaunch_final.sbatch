#!/bin/bash
#SBATCH --account=gts-aivanova7-lab
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=64GB
#SBATCH --gres=gpu:H200:1
#SBATCH --constraint=H200
#SBATCH -t 02:00:00
#SBATCH -q inferno
#SBATCH -o logs/Report-%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=henryyjiang42@gmail.com

# THE GATE BETWEEN "cortex-final is wired" AND "start a full training run".
#
# Runs the whole pre-launch program in one job, in dependency order, on the
# PARENT the P1 arms branch from -- i.e. before any arm exists.  That timing is
# the point: a gate that only runs on a finished arm is a post-mortem.
#
#   BRANCH_PATH=cortex-retrofit/retro-b2-heal/checkpoint_91552_w16 \
#       sbatch pace/prelaunch_final.sbatch
#
# WHAT IT RUNS, and why each step is where it is
#
#   0. the unit suite.  Cheap, and it is the only step that can fail for a
#      reason that has nothing to do with the checkpoint.  Running it first
#      means a red later step is about the weights.
#   1. smoke_prefix_real, BOTH buffers and the latent path.  Reads the
#      MODELING FILE, which is the artifact a checkpoint actually carries -- a
#      config can be perfect while the snapshot predates the hooks, which is
#      how a 9-hour run once trained with no cross-segment memory at all.
#   2. the OOM geometry gate.  Answers "does this fit" before anything long is
#      submitted, at the P1 geometries.
#   3. the 8-chunk dual-channel walk, at the arm's own geometry, forced to
#      latent_carry even though the parent is E-only.  A printed table: rows
#      touched against the ring rule, write norms as a ratio to ||s0||, the
#      tape and its depths, whether s0's head equals the carried Z, and the
#      per-parameter gradient norms.
#   4. the four pre-launch gates (tools/prelaunch_final.py): Z-off
#      equivalence, the config round trip, the MEASURED no-grad split, and the
#      per-channel donor control.
#
# EVERY STEP CAN COME BACK NEGATIVE.  A probe that can only report "it ran" is
# not a probe, and the entire history of this project is failures that were
# invisible in the loss curve.
#
# The gates run at the ARM's geometry via --set, not at the parent's, and the
# override is printed by model_utils and recorded in the JSON.  That is
# deliberate: what is being gated is the configuration about to be launched.

cd $SLURM_SUBMIT_DIR

module load anaconda3
module load cuda/12.1.1
conda activate cortex-retro

export SCRATCH=${SCRATCH:-$HOME/scratch}
export HF_HOME=$SCRATCH/hf_cache
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODEL=${MODEL:-ckpts/olmo-retrofit-cortex}
BRANCH_PATH=${BRANCH_PATH:-}
ACCUM_VECS=${ACCUM_VECS:-16}
GATE_SLOTS=${GATE_SLOTS:-64}
CROSS_CHUNKS=${CROSS_CHUNKS:-8}
CHUNK_LEN=${CHUNK_LEN:-256}
MAX_MEAN_REC=${MAX_MEAN_REC:-8}
OUT=${OUT:-eval_results/prelaunch-$(date +%Y%m%d-%H%M)}
SKIP_SMOKE=${SKIP_SMOKE:-}

mkdir -p $OUT logs
RC=0

echo "=== cortex-final pre-launch | model=$MODEL branch=${BRANCH_PATH:-<base>} ==="
echo "    gating the ARM's geometry: W=$ACCUM_VECS K=$GATE_SLOTS route=ring"
echo "    cross_chunks=$CROSS_CHUNKS chunk_len=$CHUNK_LEN T=$MAX_MEAN_REC"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo ""
echo "######## 0. unit suite ########"
python -m pytest tests/ -q || RC=1

if [ -z "$SKIP_SMOKE" ]; then
    echo ""
    echo "######## 1. real-checkpoint smoke: accum, gated, latent ########"
    python tools/smoke_prefix_real.py --model_name $MODEL \
        --prefix_memory accum --accum_vecs $ACCUM_VECS \
        --cross_chunks $CROSS_CHUNKS --device cuda --dtype bfloat16 || RC=1
    python tools/smoke_prefix_real.py --model_name $MODEL \
        --prefix_memory gated --accum_vecs $ACCUM_VECS \
        --gate_slots $GATE_SLOTS --cross_chunks $CROSS_CHUNKS \
        --device cuda --dtype bfloat16 || RC=1
    python tools/smoke_prefix_real.py --model_name $MODEL \
        --prefix_memory gated --accum_vecs $ACCUM_VECS \
        --gate_slots $GATE_SLOTS --cross_chunks $CROSS_CHUNKS \
        --latent_carry --device cuda --dtype bfloat16 || RC=1

    echo ""
    echo "######## 2. OOM geometry gate, INCLUDING the Z arms ########"
    # P1_Z=1 is the new row and the reason this step is not a formality: every
    # memory number this gate has ever produced came from an E-only run, and the
    # second gate is ~33.5M more parameters with optimizer state.
    P1=1 P1_Z=1 GEOMETRIES= bash pace/smoke_geometry_oom.sbatch || RC=1
fi

# From here on the gates need the weights the arm will actually start from.
CKPT_ARG=""
if [ -n "$BRANCH_PATH" ]; then
    if [ ! -f "$BRANCH_PATH/chkpt.pt" ]; then
        echo "ERROR: no chkpt.pt under BRANCH_PATH=$BRANCH_PATH"; exit 1
    fi
    CKPT_ARG="--checkpoint $BRANCH_PATH/chkpt.pt"
fi

# The Z geometry is FORCED here.  The parent is E-only, so without --set these
# gates would silently test the thing that is already known to work.
SETS="--set latent_carry=true --set prefix_memory=gated \
      --set accum_vecs=$ACCUM_VECS --set gate_slots=$GATE_SLOTS \
      --set gate_route=ring --set gate_init=zero --set gate_fill=grow"

echo ""
echo "######## 3. the 8-chunk dual-channel walk ########"
python evals/diag_dual_channel_walk.py --model_name $MODEL $CKPT_ARG \
    --chunks $CROSS_CHUNKS --chunk_len $CHUNK_LEN --T $MAX_MEAN_REC \
    --out $OUT/walk.json || RC=1

echo ""
echo "######## 4. the pre-launch gates ########"
python tools/prelaunch_final.py --model_name $MODEL $CKPT_ARG $SETS \
    --chunks $CROSS_CHUNKS --chunk_len $CHUNK_LEN --T $MAX_MEAN_REC \
    --out $OUT/prelaunch.json || RC=1

echo ""
if [ $RC -eq 0 ]; then
    echo "ALL PRE-LAUNCH GATES PASSED.  Results in $OUT."
    echo "Next: bash pace/submit_p1_probes.sh   (the four architecture probes)"
else
    echo "PRE-LAUNCH FAILED (rc=$RC).  Do NOT launch on this.  Read the step"
    echo "that failed above -- each one fails for exactly one reason, and every"
    echo "one of them is invisible in a loss curve."
fi
exit $RC
