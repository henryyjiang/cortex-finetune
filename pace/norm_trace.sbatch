#!/bin/bash
#SBATCH --account=gts-aivanova7-lab
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H200:1
#SBATCH --mem=64GB
#SBATCH -t 2:00:00
#SBATCH -q inferno
#SBATCH -J normtrace
#SBATCH -o logs/Report-%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=henryyjiang42@gmail.com

# P2.5's E-NORM DRIFT, RECONSTRUCTED FROM THE SAVED CHECKPOINTS.
#
#   ARM=a3 sbatch pace/norm_trace.sbatch      # the gated cell -- the question
#   ARM=a1 sbatch pace/norm_trace.sbatch      # the accum cell -- the control
#
# WHY THIS IS A SEPARATE JOB AND NOT A TRAINING FLAG.  `p2_deciding_measurements
# _handoff.md` S2, P2.5 asked for a norm trace in the cell's periodic
# diagnostic, "so that if it matters at 10k steps there is a curve rather than
# an endpoint".  It never happened: `pace/p1_arms.sbatch:232` defaults
# DIAG_INTERVAL=0 and only PROBE=1 forces it to 10, so both P2.3 cells ran all
# 24,414 steps with the diagnostic OFF.  Nothing rotates the saves, so the curve
# is recoverable after the fact -- walk each saved checkpoint, read the norms off
# the JSONs, fit the trend.  That is all this job is.
#
# THE ACCUM ARM IS NOT OPTIONAL.  On the 400-step probes the gated arms sat at
# 135.9 / 137.0 against a FLAT 112.5 on accum.  "Gated norms rose 8%" means
# nothing without knowing what accum did over the same window on the same data:
# if both rose the same amount it is the corpus or the LR schedule, not the
# gate.  Run both arms and read them together, the way P2.3's own cells were
# read.  A one-arm trace is not a result.
#
# WHAT COMES BACK.  tools/norm_trace.py prints two slopes and they answer
# different questions -- ACROSS training (P2.5's question: is the norm still
# climbing at 24k?) and WITHIN a walk (the gate's per-sequence lift, which was
# always expected to be positive).  Do not quote the within-slope as the drift.
#
#   SETTLED         the +21% is a level the gate reaches.  Record it, move on.
#   STILL CLIMBING  extrapolate to the 5B step count BEFORE committing 16 days,
#                   and price a renorm at the operating point.
#   UNDERPOWERED    fewer than 3 checkpoints resolved.  Check STEPS below.
#   UNREADABLE      a walk scored at chance -- RED 10's shape.  Fix that first;
#                   norms off a non-predicting model describe nothing.

cd $SLURM_SUBMIT_DIR

module load anaconda3
module load cuda/12.1.1
conda activate cortex-retro

export SCRATCH=${SCRATCH:-$HOME/scratch}
export HF_HOME=$SCRATCH/hf_cache
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODEL=${MODEL:-ckpts/olmo-retrofit-cortex}
OUT_ROOT=${OUT_ROOT:-cortex-retrofit}
ARM=${ARM:-a3}
ACCUM_VECS=${ACCUM_VECS:-16}
GATE_SLOTS=${GATE_SLOTS:-64}
CROSS_CHUNKS=${CROSS_CHUNKS:-8}
# --T 8 IS NOT OPTIONAL.  These checkpoints' config.json says mean_recurrence
# 32 and no arm ran there; read_live_frac is 0.546 at mr8 and 0.0195 at mr32.
# That is RED 8, and it is the same trap in every eval in this directory.
T=${T:-8}
# 4096 / 8 = 512.  The chunk length the cells TRAINED at.  The walk's own
# default is 256, which would walk a geometry no arm ran.
CHUNK_LEN=${CHUNK_LEN:-$(( 4096 / CROSS_CHUNKS ))}
# --data IS NOT OPTIONAL EITHER.  With neither --data nor --text_file the walk
# falls back to RANDOM IDS, which was one of the five pre-launch reds: norms
# measured on noise are not norms measured on text, and nothing in the output
# says which you got.
DATA=${DATA:-data/pg19_olmo_val_len4096}
STAMP=${STAMP:-$(date +%Y%m%d)}

# The P2.3 cells save under the BARE name.  `probe-` is the 400-step
# architecture probe and scoring it here would draw a drift curve through six
# copies of the wrong model -- that is RED 13, and it is why MIN_STEP exists.
case "$ARM" in
a1)  RUN=${RUN:-p1-a1-accum-w${ACCUM_VECS}-cc${CROSS_CHUNKS}}
     PREFIX_MODE=accum ;;
a3)  RUN=${RUN:-p1-a3-gated-w${ACCUM_VECS}k${GATE_SLOTS}-cc${CROSS_CHUNKS}}
     PREFIX_MODE=gated ;;
*)   echo "ERROR: ARM must be a1 or a3 (the two P2.3 cells); got '$ARM'"
     exit 1 ;;
esac

RUN_DIR="$OUT_ROOT/$RUN"
[ -d "$RUN_DIR" ] || { echo "ERROR: no run dir '$RUN_DIR'."; exit 1; }
[ -d "$DATA" ]    || { echo "ERROR: no such pack: $DATA"; exit 1; }

# Which checkpoints.  Default: every checkpoint_<step> in the run dir, numeric
# sort.  STEPS="95000 115966" restricts it.
MIN_STEP=${MIN_STEP:-95000}
if [ -n "${STEPS:-}" ]; then
    CKPTS=""
    for s in $STEPS; do
        d="$RUN_DIR/checkpoint_$s"
        [ -d "$d" ] || { echo "ERROR: no such checkpoint: $d"; exit 1; }
        CKPTS="$CKPTS $d"
    done
else
    CKPTS=$(ls -d $RUN_DIR/checkpoint_* 2>/dev/null | sort -t_ -k2 -n)
fi
[ -n "$CKPTS" ] || { echo "ERROR: no checkpoints under $RUN_DIR"; exit 1; }

ACCUM_MAX=${ACCUM_MAX:-$(( ACCUM_VECS * CROSS_CHUNKS ))}
SETS="--set use_memory=true --set memory_slots=0"
SETS="$SETS --set prefix_memory=$PREFIX_MODE"
SETS="$SETS --set accum_vecs=$ACCUM_VECS"
if [ "$PREFIX_MODE" = "accum" ]; then
    SETS="$SETS --set accum_max=$ACCUM_MAX"
else
    SETS="$SETS --set gate_slots=$GATE_SLOTS"
    SETS="$SETS --set gate_route=${GATE_ROUTE:-ring}"
    SETS="$SETS --set gate_norm=${GATE_NORM:-tanh}"
    SETS="$SETS --set gate_init=${GATE_INIT:-zero}"
    SETS="$SETS --set gate_fill=${GATE_FILL:-grow}"
fi

OUT=${OUT:-eval_results/norm_trace-$STAMP/$ARM}
mkdir -p "$OUT"

echo "=== P2.5 norm trace | arm=$ARM | $PREFIX_MODE | T=$T ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "    run:        $RUN_DIR"
echo "    data:       $DATA"
echo "    geometry:   W=$ACCUM_VECS chunks=$CROSS_CHUNKS chunk_len=$CHUNK_LEN"
echo "    sets:       $SETS"
echo "    out:        $OUT"
echo "    checkpoints:"
for c in $CKPTS; do echo "        $c"; done
echo ""

RC=0
KEPT=0
for CKPT in $CKPTS; do
    STEP=$(basename "$CKPT" | sed 's/^checkpoint_//; s/_.*$//')
    case "$STEP" in ''|*[!0-9]*)
        echo "--- skip $CKPT (no step number in the name)"; continue ;;
    esac
    if [ "$MIN_STEP" != "0" ] && [ "$STEP" -lt "$MIN_STEP" ]; then
        echo "--- skip step $STEP (below MIN_STEP=$MIN_STEP)"; continue
    fi
    echo "--- walking step $STEP"
    python evals/diag_dual_channel_walk.py \
        --model_name "$MODEL" --checkpoint "$CKPT" $SETS \
        --data "$DATA" \
        --chunks "$CROSS_CHUNKS" --chunk_len "$CHUNK_LEN" --T "$T" \
        --dtype float32 --no_backward \
        --out "$OUT/walk_${STEP}.json"
    if [ $? -ne 0 ]; then
        echo "    WALK FAILED at step $STEP -- continuing so the other points"
        echo "    still land; the assembler will report what it got."
        RC=1
    else
        KEPT=$(( KEPT + 1 ))
    fi
done

echo ""
echo "=== $KEPT walks written; assembling the curve ==="
if [ "$KEPT" -lt 2 ]; then
    echo "ERROR: $KEPT usable walk(s).  Nothing to fit."; exit 1
fi

# --no_backward above: this job never asks "is the gate attached to the loss"
# (the walk's gradient block does that, and P1 already answered it on all four
# arms).  It asks only what the norms are, and a backward over six checkpoints
# is six chain graphs for nothing.
python tools/norm_trace.py --dir "$OUT" --out "$OUT/trace.json"
ARC=$?

echo ""
if [ $ARC -eq 0 ] && [ $RC -eq 0 ]; then
    echo "NORM TRACE ($ARM) DONE -> $OUT/trace.json"
    echo ""
    echo "NOW RUN THE OTHER ARM AND READ THEM TOGETHER.  A gated drift is only"
    echo "a gate result if accum's drift over the same steps is flatter; if"
    echo "both moved, it is the schedule or the corpus."
elif [ $ARC -eq 0 ]; then
    echo "NORM TRACE ($ARM) PARTIAL -- some walks failed, curve is from the rest."
else
    echo "ASSEMBLY FAILED rc=$ARC"
fi
exit $ARC
