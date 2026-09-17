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
# WHAT THE WALK READS.  Random ids are not an option: the loop has nothing to
# converge on, so the trajectory is not the trained one and every rank number
# describes noise.  The 19:09 job ran that way and sat at loss 11.6-11.9 against
# ln(vocab) = 11.52.  DATA is preferred (already tokenized with this model's
# tokenizer, and it is the corpus the arms train on).
DATA=${DATA:-data/pg19_fw50_olmo_len4096}
TEXT=${TEXT:-}
if [ -n "$DATA" ] && [ -d "$DATA" ]; then
    PROSE="--data $DATA"
elif [ -n "$TEXT" ] && [ -f "$TEXT" ]; then
    PROSE="--text_file $TEXT"
else
    echo "ERROR: no prose source for the walk.  DATA=$DATA is not a directory"
    echo "  and TEXT=$TEXT is not a file.  Set DATA to a packed dataset."
    exit 1
fi
ACCUM_VECS=${ACCUM_VECS:-16}
GATE_SLOTS=${GATE_SLOTS:-64}
CROSS_CHUNKS=${CROSS_CHUNKS:-8}
CHUNK_LEN=${CHUNK_LEN:-256}
MAX_MEAN_REC=${MAX_MEAN_REC:-8}
OUT=${OUT:-eval_results/prelaunch-$(date +%Y%m%d-%H%M)}
SKIP_SMOKE=${SKIP_SMOKE:-}

mkdir -p $OUT logs
RC=0
# The width contrast is REPORTED, not gated, and the distinction is deliberate.
# It measures W=32 vs W=16 for the P1.0 decision, and p11_z_probe_prereg.md S4
# already fixes the response to every one of its verdicts: NONE of them changes
# this round's arms, because A1 and A3' both run W=16 and width is held constant
# across the comparison the probes exist to make.  Folding it into RC would turn
# an unresolved measurement into "the pre-launch failed", which is the same
# confusion the gate-1 NaN caused -- an exit code that cannot tell "the claim
# failed" from "this is not the claim being gated".
WIDTH_RC=0

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
# use_memory IS THE MASTER SWITCH -- cortex_graft.memory_enabled reads it and
# nothing else, and this base dir carries no cortex flags at all.  Without it
# the graft builds NOTHING, the checkpoint's cortex tensors load as "unexpected
# keys" and are dropped, and every gate below runs against a no-memory baseline
# while reporting only "no prefix buffer" (job 13266470).
SETS="--set use_memory=true --set memory_slots=0"
SETS="$SETS --set latent_carry=true --set prefix_memory=gated"
SETS="$SETS --set accum_vecs=$ACCUM_VECS --set gate_slots=$GATE_SLOTS"
SETS="$SETS --set gate_route=ring --set gate_init=zero --set gate_fill=grow"
echo ""
echo "######## 3. the 8-chunk dual-channel walk ########"
# AT THE ARM'S GEOMETRY, gate and Z included.  The parent's weights are accum
# E-only, so the gate and Z parameters come up freshly initialised -- which is
# the point: this step checks the MACHINERY (does the ring write the rows the
# rule predicts, does s0's head equal the carried Z, do all nine watched
# parameters have a gradient path), not this checkpoint's quality.
# Running it as accum/E-only, as the 2026-09-16 19:09 job did, exercises
# neither the gate nor Z and the table's Z columns all print "-".
python evals/diag_dual_channel_walk.py --model_name $MODEL $CKPT_ARG \
    $SETS $PROSE \
    --chunks $CROSS_CHUNKS --chunk_len $CHUNK_LEN --T $MAX_MEAN_REC \
    --out $OUT/walk.json || RC=1

if [ -n "$PARENT_PATH" ]; then
    echo ""
    echo "######## 3b. THE WIDTH CONTRAST: W=$PARENT_VECS vs W=$ACCUM_VECS ########"
    # ACCUM ON BOTH SIDES, because this step compares WEIGHTS and the weights
    # on both sides are accum.  It is the delivered-rank read that p11 section 4
    # pre-registers, not a machinery check.
    #
    # accum_max MUST BE SET PER WIDTH.  It caps ROWS, not chunks, so leaving it
    # at one value across two widths silently changes the HORIZON: 128 rows is 8
    # chunks at W=16 but only 4 at W=32.  The 19:09 job did exactly that and
    # produced "897% retained", scored SURVIVES -- a carry built from half as
    # many forwards is far more redundant, and none of that gap is about width.
    # chunks x W keeps the chunk count equal, which compare_width now enforces.
    BRANCH_MAX=$(( CROSS_CHUNKS * ACCUM_VECS ))
    PARENT_MAX=$(( CROSS_CHUNKS * PARENT_VECS ))
    echo "    accum_max: $PARENT_MAX at W=$PARENT_VECS, $BRANCH_MAX at W=$ACCUM_VECS"
    echo "    (both retain $CROSS_CHUNKS chunks -- that is what makes it a width contrast)"
    ACC_SETS="--set use_memory=true --set memory_slots=0 --set prefix_memory=accum"
    python evals/diag_dual_channel_walk.py --model_name $MODEL $CKPT_ARG \
        $ACC_SETS --set accum_vecs=$ACCUM_VECS --set accum_max=$BRANCH_MAX $PROSE \
        --chunks $CROSS_CHUNKS --chunk_len $CHUNK_LEN --T $MAX_MEAN_REC \
        --no_backward --out $OUT/walk-branch-w$ACCUM_VECS.json || RC=1
    python evals/diag_dual_channel_walk.py --model_name $MODEL \
        --checkpoint $PARENT_PATH/chkpt.pt \
        $ACC_SETS --set accum_vecs=$PARENT_VECS --set accum_max=$PARENT_MAX $PROSE \
        --chunks $CROSS_CHUNKS --chunk_len $CHUNK_LEN --T $MAX_MEAN_REC \
        --no_backward --out $OUT/walk-parent-w$PARENT_VECS.json || RC=1
    python tools/compare_width.py \
        --parent $OUT/walk-parent-w$PARENT_VECS.json \
        --branch $OUT/walk-branch-w$ACCUM_VECS.json \
        --out $OUT/width_contrast.json || RC=1
fi
echo ""
echo "######## 4. the pre-launch gates ########"
python tools/prelaunch_final.py --model_name $MODEL $CKPT_ARG $SETS \
    --trained_depth $MAX_MEAN_REC \
    --chunks $CROSS_CHUNKS --chunk_len $CHUNK_LEN --T $MAX_MEAN_REC \
    --out $OUT/prelaunch.json || RC=1

echo ""
if [ $RC -eq 0 ]; then
    echo "ALL PRE-LAUNCH GATES PASSED.  Results in $OUT."
    if [ $WIDTH_RC -ne 0 ]; then
        echo ""
        echo "NOTE: the width contrast (3b) did not produce a usable verdict."
        echo "  That is REPORTED, not gating: p11 section 4 fixes the response to"
        echo "  every width verdict and none of them changes this round's arms,"
        echo "  because A1 and A3' both run W=16.  Read it, do not block on it."
    fi
    echo "Next: bash pace/submit_p1_probes.sh   (the four architecture probes)"
else
    echo "PRE-LAUNCH FAILED (rc=$RC).  Do NOT launch on this.  Read the step"
    echo "that failed above -- each one fails for exactly one reason, and every"
    echo "one of them is invisible in a loss curve."
fi
exit $RC
