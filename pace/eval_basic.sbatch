#!/bin/bash
#SBATCH --account=gts-aivanova7-lab
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=64GB
#SBATCH --gres=gpu:A100:1
#SBATCH -t 16:00:00
#SBATCH -q inferno
#SBATCH -o logs/Report-%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=henryyjiang42@gmail.com

# Short-context sanity eval (MC: hellaswag/winogrande/arc/piqa + GSM8K) for
# ONE model — checks the finetune didn't degrade the base's general ability
# (the memory hypothesis itself is tested by eval_longcontext.sbatch).
#
#   RUN=rung1-k4 EVAL_TAG=20260710 sbatch pace/eval_basic.sbatch
#   BASE=true    EVAL_TAG=20260710 sbatch pace/eval_basic.sbatch
#
# GSM8K generates 256 tokens/example at T=8 — capped via GSM8K_MAX (default
# 500 of 1319) to keep the job inside the wall time.
#
# Mixed CCoT+CoT ablation (cross-state models only): CCOT_PASSES=N runs N
# silent latent passes over the prompt (M_cross carried pass-to-pass) before
# the explicit CoT generation.  Results land in gsm8k-ccot${N}/ so the plain
# run is kept.  MC is skipped (SKIP_MC honored) since only GSM8K changes.
# Compute-matched base control: the base has no cross state (passes are
# no-ops), so match total loop iterations instead via T_OVERRIDE=$((N*T)):
#   RUN=rung1-k4 CCOT_PASSES=4 EVAL_TAG=...            sbatch pace/eval_basic.sbatch
#   BASE=true    T_OVERRIDE=32 EVAL_TAG=...            sbatch pace/eval_basic.sbatch

cd $SLURM_SUBMIT_DIR

module load anaconda3
module load cuda/12.1.1
conda activate cortex-retro

export SCRATCH=${SCRATCH:-$HOME/scratch}   # not set by default on PACE Phoenix
export HF_HOME=$SCRATCH/hf_cache
export HF_HUB_OFFLINE=1                 # compute nodes: cache only
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

EVAL_TAG=${EVAL_TAG:-$(date +%Y%m%d)}
GSM8K_MAX=${GSM8K_MAX:-500}
CCOT_PASSES=${CCOT_PASSES:-0}
T_OVERRIDE=${T_OVERRIDE:-}
SKIP_MC=${SKIP_MC:-false}
BASE=${BASE:-false}
if [ "$BASE" = "true" ]; then
    LABEL="base"
    MODEL_DIR="ckpts/olmo8-cortex"
else
    [ -n "$RUN" ] || { echo "Set RUN=<run_name> (or BASE=true)"; exit 1; }
    LABEL="$RUN"
    MODEL_DIR="cortex-retro-ft/$RUN/final_checkpoint"
    python tools/prepare_eval_checkpoint.py --base ckpts/olmo8-cortex \
        --dirs "$MODEL_DIR"
fi

OUT_ROOT="eval_results/basic_${EVAL_TAG}/${LABEL}"
mkdir -p "$OUT_ROOT"
echo "=== basic eval: $LABEL  ($MODEL_DIR) -> $OUT_ROOT ==="

T_ARGS=()
[ -n "$T_OVERRIDE" ] && T_ARGS+=(--T "$T_OVERRIDE")

GSM8K_OUT="$OUT_ROOT/gsm8k"
GSM8K_ARGS=()
if [ "$CCOT_PASSES" -gt 0 ]; then
    GSM8K_OUT="$OUT_ROOT/gsm8k-ccot${CCOT_PASSES}"
    GSM8K_ARGS+=(--ccot_passes "$CCOT_PASSES")
    SKIP_MC=true    # only GSM8K differs in a CCoT ablation run
elif [ -n "$T_OVERRIDE" ]; then
    GSM8K_OUT="$OUT_ROOT/gsm8k-T${T_OVERRIDE}"
    SKIP_MC=true    # compute-matched control run: only GSM8K needed
fi

if [ "$SKIP_MC" != "true" ]; then
    python evals/eval_multiple_choice.py \
        --model_name "$MODEL_DIR" \
        --tasks hellaswag winogrande arc_easy arc_challenge piqa \
        --out_dir "$OUT_ROOT/multiple_choice"
fi

python evals/eval_gsm8k.py \
    --model_name "$MODEL_DIR" \
    --max_examples $GSM8K_MAX \
    "${T_ARGS[@]}" "${GSM8K_ARGS[@]}" \
    --out_dir "$GSM8K_OUT"
