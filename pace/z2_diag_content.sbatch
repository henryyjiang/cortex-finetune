#!/bin/bash
#SBATCH --account=gts-aivanova7-lab
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H200:1
#SBATCH --mem=128GB
#SBATCH -t 3:00:00
#SBATCH -q inferno
#SBATCH -J z2_content
#SBATCH -o logs/Report-%A_%a.out
#SBATCH --array=0-1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=henryyjiang42@gmail.com

# Z ATTEMPT 2, STEP 0: D1 (seed vs document) and D2 (what Z holds beyond E).
# Findings doc, section "Attempt 2", Step 0.  Forward-only.
#
#   sbatch pace/z2_diag_content.sbatch          # [0] a2  [1] a3z
#   sbatch --array=1 pace/z2_diag_content.sbatch   # a3z only
#
# Scores the P1 Z arms' 400-step PROBES.  These are the only a2/a3z checkpoints
# that exist (the P2.3 cells a1/a3 are E-only), so the probe- prefix is RIGHT
# here -- the opposite of RED 13.  They read Z at s0 and have no LatentRead, so
# only the geometry and Z-channel flags are forced.
#
# LOOK FOR, in order:
#   [z2] pack ... books: document ends: ~50 books  -- NOT "FALLBACK" (split would leak)
#   [z2] reseed check ... OK                    -- else D1 measures something else
#   the D1 table: ratio per encoding and per depth
#   the D2 table: "probe health" first, then the content-gain rows
#
# Decision (findings doc, Step 0): J1 runs on the Z encoding with the lowest D1
# ratio; if no encoding gains content in D2, the write must change before any
# read is tested.

cd $SLURM_SUBMIT_DIR

module load anaconda3
module load cuda/12.1.1
conda activate cortex-retro

export SCRATCH=${SCRATCH:-$HOME/scratch}
export HF_HOME=$SCRATCH/hf_cache
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODEL=${MODEL:-ckpts/olmo-retrofit-cortex}
DATA=${DATA:-data/pg19_olmo_validation_len4096_strided}
OUT_ROOT=${OUT_ROOT:-cortex-retrofit}
T=${T:-8}                      # the probes' trained depth (config says 32)
CROSS_CHUNKS=${CROSS_CHUNKS:-8}
ACCUM_VECS=${ACCUM_VECS:-16}
GATE_SLOTS=${GATE_SLOTS:-64}
MAX_EXAMPLES=${MAX_EXAMPLES:-0}
D1_ROWS=${D1_ROWS:-200}
BOOT=${BOOT:-2000}

ARMS=(a2 a3z)
IDX=${SLURM_ARRAY_TASK_ID:-0}
ARM=${ARMS[$IDX]}
if [ -z "$ARM" ]; then echo "no arm at index $IDX (use --array=0-1)"; exit 0; fi

# LITERAL names, as in pace/p30_noread.sbatch, so a grep finds them.
case "$ARM" in
    a2)  RUN=probe-p1-a2-accum-w16-cc8-z;        PREFIX_MODE=accum ;;
    a3z) RUN=probe-p1-a3z-gated-w16k64-cc8-z;    PREFIX_MODE=gated ;;
esac
# NUMERIC sort on the step (checkpoint_9999 must not outrank checkpoint_91952).
CKPT=$(ls -d $OUT_ROOT/$RUN/checkpoint_* 2>/dev/null | sort -t_ -k2 -n | tail -1)
[ -n "$CKPT" ] && [ -d "$CKPT" ] || { echo "ERROR: no checkpoint under $OUT_ROOT/$RUN"; exit 1; }
[ -d "$DATA" ] || { echo "ERROR: no such pack: $DATA"; exit 1; }

# The geometry has to be FORCED: --model_name loads the base dir, whose config
# carries no cortex flags, and use_memory is the master switch (RED 12).
# Mirrors pace/eval_carry_2x2.sbatch.
SETS="--set use_memory=true --set memory_slots=0"
SETS="$SETS --set prefix_memory=$PREFIX_MODE"
SETS="$SETS --set accum_vecs=$ACCUM_VECS"
if [ "$PREFIX_MODE" = "accum" ]; then
    SETS="$SETS --set accum_max=$(( ACCUM_VECS * CROSS_CHUNKS ))"
else
    SETS="$SETS --set gate_slots=$GATE_SLOTS --set gate_route=ring"
    SETS="$SETS --set gate_norm=tanh --set gate_init=zero --set gate_fill=grow"
fi
SETS="$SETS --set latent_carry=true --set latent_depth_rule=absolute"
SETS="$SETS --set latent_depth_lo=2 --set latent_depth_hi=9 --set latent_renorm=none"

OUT=${OUT:-eval_results/z2_content/$ARM}
echo "=== z2 D1+D2 | arm=$ARM | T=$T | n_chunks=$CROSS_CHUNKS ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "    checkpoint: $CKPT"
echo "    data:       $DATA"
echo "    sets:       $SETS"
echo "    out:        $OUT"
echo ""

python evals/diag_z_content.py \
    --model_name "$MODEL" --checkpoint "$CKPT" --data "$DATA" $SETS \
    --n_chunks "$CROSS_CHUNKS" --T "$T" \
    --max_examples "$MAX_EXAMPLES" --d1_rows "$D1_ROWS" --boot "$BOOT" \
    --dtype float32 --out_dir "$OUT"
RC=$?

echo ""
case $RC in
    0) echo "z2 ($ARM) DONE -> $OUT/results.json" ;;
    3) echo "REFUSED: no Z channel on this build -- check the --set flags." ;;
    4) echo "RESEED CHECK FAILED: D1 is not measuring s0 noise.  See above." ;;
    *) echo "z2 ($ARM) FAILED rc=$RC" ;;
esac
exit $RC
