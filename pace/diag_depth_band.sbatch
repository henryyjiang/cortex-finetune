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
SEQ_LEN=${SEQ_LEN:-512}
BATCH=${BATCH:-2}
N_SLOTS=${N_SLOTS:-16}                 # W, so the printed map is A3's shape
T_SWEEP=${T_SWEEP:-"4 8 16 32"}
OUT=${OUT:-eval_results/p07_depth_band_${SLURM_JOB_ID}}

echo "=== P0.7 depth band | $MODEL | T in $T_SWEEP ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python evals/diag_depth_band.py \
    --model_name "$MODEL" \
    --text_file "$TEXT" \
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
    python evals/diag_depth_band.py \
        --model_name "$HUGINN_MODEL" \
        --text_file "$TEXT" \
        --T $T_SWEEP \
        --seq_len "$SEQ_LEN" \
        --batch "$BATCH" \
        --n_slots "$N_SLOTS" \
        --dtype float32 \
        --no_prefix \
        --out "$OUT/huginn/results.json" || RC=1
    echo ""
    echo "COMPARE THE TWO VERDICTS BEFORE FIXING f_j.  Agreement generalises the"
    echo "band; disagreement means it tracks the model's own TRAINED depth, and"
    echo "the write rule must be stated relative to that rather than in absolute"
    echo "steps.  Either way the answer goes in the Z pre-registration BEFORE the"
    echo "write is built, not after."
fi

exit $RC
