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
#   3b. THE WIDTH CONTRAST (only when PARENT_PATH is set).  The same walk on
#      the UNSLICED parent, so the DELIVERED rank of the carried block can be
#      read at W=32 against W=16.  The slice reported that halving W keeps only
#      54% of summary_emb's rank, which retires P1.0's at-init argument -- but
#      summary_emb is the write BASIS, and what decides whether width binds is
#      what reaches attention.  p11_z_probe_prereg.md section 4 fixes the
#      reading before it runs.
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
# The UNSLICED parent BRANCH_PATH was cut from.  Set it and step 3b runs the
# width contrast (W=PARENT_VECS vs W=ACCUM_VECS) off the SAME weights, which
# is the only way the comparison isolates width from training.
PARENT_PATH=${PARENT_PATH:-}
PARENT_VECS=${PARENT_VECS:-32}
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
# --set accum_vecs is NOT optional on a sliced branch: summary_emb is a
# parameter SHAPE, and the config the base dir carries is the PARENT's (W=32).
# A [16, D] slice loaded against it is a size mismatch, so the walk would die
# before printing a single row.
python evals/diag_dual_channel_walk.py --model_name $MODEL $CKPT_ARG \
    --set accum_vecs=$ACCUM_VECS \
    --chunks $CROSS_CHUNKS --chunk_len $CHUNK_LEN --T $MAX_MEAN_REC \
    --out $OUT/walk.json || RC=1

if [ -n "$PARENT_PATH" ]; then
    echo ""
    echo "######## 3b. THE WIDTH CONTRAST: W=$PARENT_VECS vs W=$ACCUM_VECS ########"
    # WHY THIS EXISTS.  tools/slice_summary_emb.py reported that halving W keeps
    # 54% of the parent's summary_emb rank at 50% of the columns -- the trained
    # write basis was NOT redundant, which retires P1.0's at-init "the slots were
    # never carrying 32 things" argument.
    #
    # But summary_emb is the WRITE BASIS, not what reaches attention.  The number
    # that decides whether width binds is the DELIVERED rank: the effective rank
    # of the carried post-ln_f block, ~4 of 32 on B2.  ~4 at both widths means
    # the basis given up never reached the token stream.  p11_z_probe_prereg.md
    # section 4 fixes the reading BEFORE this runs.
    #
    # BOTH WALKS COME OFF THE SAME STEP.  The newest weights-only dir in the heal
    # run is model_only_chkpt_90000, 1,552 steps short of the parent the slice
    # was cut from; using it would confound width with training, so both sides
    # read the 91552 chkpt.pt.
    python evals/diag_dual_channel_walk.py --model_name $MODEL \
        --checkpoint $PARENT_PATH/chkpt.pt --set accum_vecs=$PARENT_VECS \
        --chunks $CROSS_CHUNKS --chunk_len $CHUNK_LEN --T $MAX_MEAN_REC \
        --no_backward --out $OUT/walk-parent-w$PARENT_VECS.json || RC=1
    python tools/compare_width.py --parent $OUT/walk-parent-w$PARENT_VECS.json \
        --branch $OUT/walk.json || RC=1
fi
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
