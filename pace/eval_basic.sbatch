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
#
# B1 / B2 retrofit checkpoints (2026-08-31): the same three overrides the
# carry-ablation / ceiling / diagnostic sbatches already carry, with the
# Track-A value as the default.  This script and eval_longcontext.sbatch were
# the last two still hard-wired to cortex-retro-ft/<run>/final_checkpoint.
#   OUT_ROOT   runs root.  Track A: cortex-retro-ft.  B1/B2: cortex-retrofit.
#   CKPT_NAME  final_checkpoint, or an intermediate model_only_chkpt_<step>.
#   PREP_BASE  the graft-prepared dir whose modeling file + config are merged
#              into the checkpoint by prepare_eval_checkpoint.py.  MUST track
#              the run's own --model_name: merging olmo8-cortex into a
#              retrofit checkpoint is silent and wrong.
#   BASE_CKPT  the model evaluated when BASE=true.  Stays olmo8-cortex even
#              for a B2 comparison — olmo-retrofit-cortex is the UNTRAINED
#              arrangement (loss ~10.3) and is not a benchmark control.
# T_OVERRIDE IS MANDATORY ON EVERY RETROFIT-DERIVED CHECKPOINT.  Those configs
# inherit mean_recurrence=32 from the untrained base (finding 0c, cost the B1
# carry headline), so an unset T evaluates an mr8 arm at T=32 — 4x the depth
# it trained at and 4x the wall clock.  Setting it no longer skips the MC
# tasks; see the SKIP_MC block below.
#   OUT_ROOT=cortex-retrofit CKPT_NAME=final_checkpoint \
#       PREP_BASE=ckpts/olmo-retrofit-cortex RUN=retro-b2-acc32-cc8-mr8 \
#       T_OVERRIDE=8 EVAL_TAG=b2-final sbatch pace/eval_basic.sbatch

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
# Few-shot count in the GSM8K prompt.  8 is the harness standard and the only
# value that is comparable to any published number -- leave it alone unless you
# are running the CHUNK-LENGTH PROBE.  That probe: the recipe never shows the
# model more than max_length/cross_chunks contiguous tokens (512 on the B2 arm),
# and the 8-shot prompt crosses that bound while the 4-shot one does not, so a
# sweep 0/1/2/4/8 that RECOVERS below the bound localises the GSM8K floor to the
# chunk geometry.  P0.5 already ruled out conversion budget, so this is the
# standing next hypothesis and it costs no training.
#   N_SHOT="0 1 2 4 8" SCORE_ONLY=true SKIP_MC=true T_OVERRIDE=8 \
#     EVAL_TAG=nshot-b2final sbatch pace/eval_basic.sbatch
# One job, one checkpoint load, five rungs of gold-CoT NLL.  Minutes, not hours.
N_SHOT=${N_SHOT:-8}
# Skip the generative half entirely.  See the cost note above the invocation.
SKIP_GSM8K=${SKIP_GSM8K:-false}
# Score the gold CoT instead of decoding one: one forward per example, and a
# continuous metric.  The ONLY sane way to run the n-shot sweep -- GSM8K
# accuracy sits at 1-4%, where no affordable n can resolve a break.
SCORE_ONLY=${SCORE_ONLY:-false}
T_OVERRIDE=${T_OVERRIDE:-}
SKIP_MC=${SKIP_MC:-false}
BASE=${BASE:-false}
OUT_ROOT=${OUT_ROOT:-cortex-retro-ft}
CKPT_NAME=${CKPT_NAME:-final_checkpoint}
PREP_BASE=${PREP_BASE:-ckpts/olmo8-cortex}
BASE_CKPT=${BASE_CKPT:-ckpts/olmo8-cortex}
if [ "$BASE" = "true" ]; then
    LABEL="base"
    MODEL_DIR="$BASE_CKPT"
else
    [ -n "$RUN" ] || { echo "Set RUN=<run_name> (or BASE=true)"; exit 1; }
    LABEL="$RUN"
    MODEL_DIR="$OUT_ROOT/$RUN/$CKPT_NAME"
    [ -d "$MODEL_DIR" ] || { echo "No such checkpoint: $MODEL_DIR"; exit 1; }
    python tools/prepare_eval_checkpoint.py --base "$PREP_BASE" \
        --dirs "$MODEL_DIR"
fi

OUT="eval_results/basic_${EVAL_TAG}/${LABEL}"
mkdir -p "$OUT"
echo "=== basic eval: $LABEL  ($MODEL_DIR) -> $OUT ==="

T_ARGS=()
T_SUFFIX=""
if [ -n "$T_OVERRIDE" ]; then
    T_ARGS+=(--T "$T_OVERRIDE")
    T_SUFFIX="-T${T_OVERRIDE}"
fi

GSM8K_OUT="$OUT/gsm8k${T_SUFFIX}"
MC_OUT="$OUT/multiple_choice${T_SUFFIX}"
GSM8K_ARGS=()
if [ "$CCOT_PASSES" -gt 0 ]; then
    GSM8K_OUT="$OUT/gsm8k-ccot${CCOT_PASSES}${T_SUFFIX}"
    GSM8K_ARGS+=(--ccot_passes "$CCOT_PASSES")
    SKIP_MC=true    # only GSM8K differs in a CCoT ablation run
fi
if [ "$N_SHOT" != "8" ]; then
    SKIP_MC=true    # the MC suite has no few-shot prompt to vary
fi
# T_OVERRIDE used to force SKIP_MC=true here, on the assumption that the only
# reason to set T was a compute-matched GSM8K control.  That assumption breaks
# on the retrofit line, where T_OVERRIDE=8 is the CORRECT depth rather than an
# ablation (the checkpoint config says 32), and silently dropped five of the
# six benchmarks.  Pass SKIP_MC=true explicitly for a GSM8K-only run.

if [ "$SKIP_MC" != "true" ]; then
    python evals/eval_multiple_choice.py \
        --model_name "$MODEL_DIR" \
        --tasks hellaswag winogrande arc_easy arc_challenge piqa \
        "${T_ARGS[@]}" \
        --out_dir "$MC_OUT"
fi

# GSM8K IS THE EXPENSIVE HALF OF THIS SCRIPT, not the free rider the handoff
# assumed.  Measured on the P0.5 pair: job 13225106 (GSM8K 500 alone) ran
# 3:19:21 and job 13225035 (the same GSM8K plus the whole 5-task MC suite and a
# second checkpoint load) ran 3:42:52 -- so the MC suite is ~21 min for 49,854
# scoring forwards (0.025 s each) and GSM8K is ~3.3 h for 500 256-token decodes
# (23.8 s each).  GSM8K is ~90% of the wall clock and supports ZERO rows of the
# claim-to-experiment map (framework S8).  Pass SKIP_GSM8K=true on any rung
# where you only want the powered half.
if [ "$SKIP_GSM8K" != "true" ]; then
# N_SHOT may be a LIST.  Looping inside one job amortises the checkpoint load
# across the whole sweep instead of paying it five times.
for NS in $N_SHOT; do
    G_OUT="$GSM8K_OUT"
    G_ARGS=("${GSM8K_ARGS[@]}")
    [ "$NS" = "8" ] || { G_OUT="${G_OUT}-s${NS}"; G_ARGS+=(--n_shot "$NS"); }
    if [ "$SCORE_ONLY" = "true" ]; then
        G_OUT="${G_OUT}-nll"
        G_ARGS+=(--score_only)
    fi
    echo "--- gsm8k: n_shot=$NS score_only=$SCORE_ONLY -> $G_OUT ---"
    python evals/eval_gsm8k.py \
        --model_name "$MODEL_DIR" \
        --max_examples $GSM8K_MAX \
        "${T_ARGS[@]}" "${G_ARGS[@]}" \
        --out_dir "$G_OUT"
done
fi
