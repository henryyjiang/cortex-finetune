#!/bin/bash
#SBATCH --account=gts-aivanova7-lab
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=48GB
#SBATCH --gres=gpu:A100:1
#SBATCH -t 1:00:00
#SBATCH -q inferno
#SBATCH -o logs/Report-%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=henryyjiang42@gmail.com

# P0.7 — is the informative depth band ABSOLUTE or RELATIVE?
# Forward-only, no optimizer, no training.  It BLOCKS the Z channel's
# slot->depth map, which is why it is cheap and why it runs first.
#
#   sbatch pace/diag_depth_band.sbatch
#   MODEL=<other ckpt> sbatch pace/diag_depth_band.sbatch
#   HUGINN=1 sbatch pace/diag_depth_band.sbatch     # + the mr32 generalisation
#
# P0.1 put the usable delta band at t ~ 2..9, but it ran at a SINGLE T (32), so
# it cannot say whether "informative" means steps 2..9 whatever T is, or the
# first quarter of the loop whatever T is.  The two imply different write rules
# and no single fixed assignment serves both; at T=32 they diverge sharply.
#
# THE CONFOUND THIS JOB CANNOT REMOVE ON ITS OWN.  The B2 checkpoints trained at
# mean_recurrence 8, so every T above 8 is OUT OF DISTRIBUTION for them and a
# band that looks absolute may only be saying "this model stops at 8" — which
# is exactly the shape rule A predicts, for the wrong reason.  HUGINN=1 adds the
# same sweep on Huginn-0125 (mean_recurrence 32), where t=2..32 is all in
# distribution.  IF THE TWO MODELS DISAGREE, the band is a property of the
# training depth and not of recurrence, and the write rule has to be stated
# relative to the arm's own trained depth rather than in absolute steps.
# Do not generalise off the B2 row alone.
#
# REAL PROSE, NOT RANDOM IDS.  Random ids give the loop nothing to converge ON,
# so the trajectory shape is not the trained one and the band is not the trained
# band.  The repo's own planning docs are the same text the other probes in this
# family use, which keeps the trajectories comparable across probes.
#
# FLOAT32, and the script defaults to it.  The late deltas are ~1% of ||s_t||,
# inside bf16's relative precision — a bf16 pass invented a delta plateau at
# 0.37x that fp32 puts at 0.06x, and this probe reads exactly that region.

cd $SLURM_SUBMIT_DIR

module load anaconda3
module load cuda/12.1.1
conda activate cortex-retro

export SCRATCH=${SCRATCH:-$HOME/scratch}
export HF_HOME=$SCRATCH/hf_cache
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODEL=${MODEL:-../checkpoints/retrofit/retro-b2-acc32-cc8-mr8-final_checkpoint}
TEXT=${TEXT:-../cortex_next_phase_framework.md}
DATA=${DATA:-}

# WHERE THE PROSE COMES FROM, and why there are two answers.
#
# DATA is preferred when it is set: the packed corpus is already tokenized
# with the arm's OWN tokenizer, it is the exact distribution the cells train
# on, and it is on the cluster -- where the repo-root planning docs are NOT,
# because they live outside this repo.  Measuring 'the trained band' on the
# training corpus is the more defensible reading anyway.
#
# TEXT stays for the HUGINN row, which CANNOT use a pack: huginn-0125's
# tokenizer differs from the pack's, so ids built for one model are noise to
# the other.  --text_file re-tokenizes with each model's own tokenizer, which
# is exactly why the script takes a path instead of ids.
#
# WHICHEVER YOU USE, USE THE SAME ONE FOR EVERY PROBE IN THIS FAMILY.  The
# band is a property of the trajectory, and the trajectory is a property of
# the text -- so trajectories measured on different sources are not
# comparable, and this line is the one that decides it.
if [ -n "$DATA" ]; then
    SRC_ARGS="--data $DATA"
    SRC_DESC="$DATA (packed; the arms' own corpus and tokenizer)"
elif [ -f "$TEXT" ]; then
    SRC_ARGS="--text_file $TEXT"
    SRC_DESC="$TEXT"
else
    echo "ERROR: no prose source.  TEXT=$TEXT does not exist (the repo-root"
    echo "  planning docs live outside this repo and are usually not on the"
    echo "  cluster).  Set DATA=<packed dataset> -- e.g."
    echo "    DATA=data/pg19_fw50_olmo_len4096 sbatch pace/diag_depth_band.sbatch"
    echo "  RANDOM IDS ARE NOT A FALLBACK: they give the loop nothing to"
    echo "  converge on, so the trajectory is not the trained one and neither"
    echo "  is the band."
    exit 1
fi
SEQ_LEN=${SEQ_LEN:-512}
BATCH=${BATCH:-2}
N_SLOTS=${N_SLOTS:-16}                 # W, so the printed map is A3's shape
# The depth the checkpoint ACTUALLY TRAINED AT.  config.json does not carry it:
# a B2-family checkpoint reads mean_recurrence 32 from the base recipe while its
# ramp capped max_mean_rec at 8, so reading it off the config marks NO row as
# out of distribution when in fact everything past 8 is -- and the late
# trajectory a depth band is read from sits entirely in that gap.
TRAINED_DEPTH=${TRAINED_DEPTH:-0}
T_SWEEP=${T_SWEEP:-"4 8 16 32"}
OUT=${OUT:-eval_results/p07_depth_band_${SLURM_JOB_ID}}

echo "=== P0.7 depth band | $MODEL | T in $T_SWEEP ==="
echo "    trained depth: ${TRAINED_DEPTH:-unknown (pass TRAINED_DEPTH)}"
echo "    prose source: $SRC_DESC"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python evals/diag_depth_band.py \
    --model_name "$MODEL" \
    $SRC_ARGS \
    --trained_depth $TRAINED_DEPTH \
    --T $T_SWEEP \
    --seq_len "$SEQ_LEN" \
    --batch "$BATCH" \
    --n_slots "$N_SLOTS" \
    --dtype float32 \
    --out "$OUT/b2/results.json"
RC=$?

if [ -n "$HUGINN" ]; then
    # The generalisation check.  Huginn-0125 trained at mean_recurrence 32, so
    # the whole sweep is in distribution for it and the "this model stops at 8"
    # reading is unavailable.  Its tokenizer differs, so --text_file is
    # re-tokenized with ITS tokenizer, which is correct and is why the script
    # takes a path rather than ids.
    HUGINN_MODEL=${HUGINN_MODEL:-tomg-group-umd/huginn-0125}
    echo ""
    echo "######## generalisation: $HUGINN_MODEL (mean_recurrence 32) ########"
    # The Huginn row CANNOT fall back to a pack: huginn-0125 tokenizes
    # differently, so ids built for the arm's tokenizer are noise to it.
    if [ ! -f "$TEXT" ]; then
        echo "SKIP: the generalisation row needs TEXT, and $TEXT is missing."
        echo "  A packed dataset cannot substitute -- different tokenizer."
        echo "  Without it, read the B2 verdict with the confound in mind:"
        echo "  B2 trained at mr8, so ABSOLUTE is what the confound itself"
        echo "  predicts and only a RELATIVE verdict is decisive alone."
        RC=1
    else
        python evals/diag_depth_band.py \
            --model_name "$HUGINN_MODEL" \
            --text_file "$TEXT" \
            --trained_depth $TRAINED_DEPTH \
            --T $T_SWEEP \
            --seq_len "$SEQ_LEN" \
            --batch "$BATCH" \
            --n_slots "$N_SLOTS" \
            --dtype float32 \
            --no_prefix \
            --out "$OUT/huginn/results.json" || RC=1
    fi
    echo ""
    echo "COMPARE THE TWO VERDICTS BEFORE FIXING f_j.  Agreement generalises the"
    echo "band; disagreement means it tracks the model's own TRAINED depth, and"
    echo "the write rule must be stated relative to that rather than in absolute"
    echo "steps.  Either way the answer goes in the Z pre-registration BEFORE the"
    echo "write is built, not after."
fi

exit $RC
