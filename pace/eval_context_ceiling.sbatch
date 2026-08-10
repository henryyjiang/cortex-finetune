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

# Context ceiling (2026-08-04) — the missing denominator.
#
# WHY THIS EXISTS.  Every carry number is quoted against zero: +0.02370 Track A,
# -0.01282 B1, ~0 for ci/ki4.  None of them has a scale, so "+0.024 nats" cannot
# be called strong or weak.  AutoCompressor always reports three points — 5.40
# no context, 5.07 compressed, 4.76 REAL full attention — and the number that
# means something is the fraction of the gap the compression recovers.  This
# sweeps k real preceding tokens as the oracle and reports the carry delta
# against it, including "the carry is worth about N tokens of real context".
#
# Both outcomes are publishable and neither is a bug:
#   large ceiling, small recovery -> the MECHANISM is the problem
#   ceiling ~0.02-0.03            -> Track A was already near it; the line is a
#                                    characterised negative with a MEASURED bound
#   ceiling <= 0                  -> real context over this span does not help
#                                    this model at all.  Expected: the
#                                    teacher-advantage probe measured -0.98 nats
#                                    at W=2048 because McLeish continued-
#                                    pretrained at max_length=1024.  That is the
#                                    quantitative form of "chunk+carry is the
#                                    only long-context path", which is the
#                                    paper's motivation — worth having as a
#                                    number rather than an assertion.
#
# The pure ceiling, on the two bases (no memory needed — run these first):
#   MODEL=ckpts/olmo8-cortex T_EVAL=8 EVAL_TAG=base-olmo8 \
#       sbatch pace/eval_context_ceiling.sbatch
#   MODEL=ckpts/olmo-retrofit-cortex T_EVAL=8 EVAL_TAG=base-retrofit \
#       sbatch pace/eval_context_ceiling.sbatch
#
# Ceiling AND fraction recovered, on an arm (RUN form, same layout as the other
# eval sbatches):
#   RUN=rung1-k0-acc4v-tb2-rs-ep3-rcl T_EVAL=8 EVAL_TAG=trackA \
#       sbatch pace/eval_context_ceiling.sbatch
#   OUT_ROOT=cortex-retrofit CKPT_NAME=model_only_chkpt_22500 \
#       BASE=ckpts/olmo-retrofit-cortex RUN=retro-b1-acc4v-tb2-mr8 T_EVAL=4 \
#       EVAL_TAG=b1-22500 sbatch pace/eval_context_ceiling.sbatch
#
# ALWAYS set T_EVAL.  Checkpoints derived from ckpts/olmo-retrofit-cortex inherit
# mean_recurrence=32 from the untrained base config; B1 is an mr8 run sitting at
# ~recurrence 4 at step 22,500.  Comparing a ceiling measured at one depth to a
# carry delta measured at another is meaningless.
#
# ~21 forwards per sample at n=50, no backward graph — minutes, not hours.
# CONTEXT_LENS overrides the sweep (default "128 256 512 1024").
#
# Two decompositions of the same measurement, both off by default (2026-08-04),
# for the question "how do we recover MORE of the available nats":
#   CARRY_PLUS="128 256"  also scores carry AND k real tokens together, and
#                         reports the residual over k-alone.  The k=0 floor is
#                         not where a deployed model sits — the tail of the
#                         previous chunk is free — so a residual near zero says
#                         the carry is re-supplying local context and the real
#                         headroom is (best ceiling - ceiling(k)).  Costs one
#                         extra forward per k per chunk.
#   POS_BUCKETS=4         splits each chunk's NLL into 4 position bands.  Free.
#                         Cross-chunk information is a boundary effect; this is
#                         where you see whether the oracle's nats are all in the
#                         first band and whether the carry delivers there.

cd $SLURM_SUBMIT_DIR

module load anaconda3
module load cuda/12.1.1
conda activate cortex-retro

export SCRATCH=${SCRATCH:-$HOME/scratch}
export HF_HOME=$SCRATCH/hf_cache
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

EVAL_TAG=${EVAL_TAG:-$(date +%Y%m%d)}
DATA=${DATA:-data/pg19_olmo_val_len4096}
OUT_ROOT=${OUT_ROOT:-cortex-retro-ft}
CKPT_NAME=${CKPT_NAME:-final_checkpoint}
BASE=${BASE:-ckpts/olmo8-cortex}
MAX_EXAMPLES=${MAX_EXAMPLES:-50}
N_CHUNKS=${N_CHUNKS:-4}
CONTEXT_LENS=${CONTEXT_LENS:-"128 256 512 1024"}
CARRY_PLUS=${CARRY_PLUS:-}
POS_BUCKETS=${POS_BUCKETS:-0}
T_EVAL=${T_EVAL:-}

# MODEL= points straight at a base dir (pure ceiling, no arm); otherwise RUN=
# selects an arm checkpoint under OUT_ROOT and gets prepared like the others.
if [ -n "$MODEL" ]; then
    MODEL_DIR="$MODEL"
    TAG_NAME=$(basename "$MODEL")
else
    [ -n "$RUN" ] || { echo "Set MODEL=<base dir> or RUN=<run_name>"; exit 1; }
    MODEL_DIR="$OUT_ROOT/$RUN/$CKPT_NAME"
    TAG_NAME="$RUN"
    python tools/prepare_eval_checkpoint.py --base "$BASE" --dirs "$MODEL_DIR"
fi

[ -d "$MODEL_DIR" ] || { echo "No such checkpoint: $MODEL_DIR"; exit 1; }

OUT="eval_results/context_ceiling_${EVAL_TAG}/${TAG_NAME}"
mkdir -p "$OUT"
echo "=== context ceiling: $MODEL_DIR -> $OUT ==="

python evals/eval_context_ceiling.py \
    --model_name "$MODEL_DIR" \
    --data "$DATA" \
    --n_chunks "$N_CHUNKS" \
    --context_lens $CONTEXT_LENS \
    --max_examples "$MAX_EXAMPLES" \
    ${CARRY_PLUS:+--carry_plus $CARRY_PLUS} \
    --pos_buckets "$POS_BUCKETS" \
    ${T_EVAL:+--T "$T_EVAL"} \
    --out_dir "$OUT"
