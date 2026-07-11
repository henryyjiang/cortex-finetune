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

# Long-context eval (BABILong qa1-qa3 + LongMemEval) for ONE model — the
# headline K>0 vs K=0 memory numbers.  One model per job (BLIMP lesson from
# cortex-main: multi-model eval jobs hit the 24 h wall).
#
#   RUN=rung1-k4 EVAL_TAG=20260710 sbatch pace/eval_longcontext.sbatch
#   BASE=true    EVAL_TAG=20260710 sbatch pace/eval_longcontext.sbatch   # unfinetuned base control
#
# Multi-pass ablations (cross-state models only; both default off):
#   PASSES_PER_CHUNK=N  each priming chunk gets N full-model passes, M_cross
#                       carried pass-to-pass (multi-pass buffer fill)
#   CCOT_PASSES=N       N extra silent passes over the final question chunk
#                       before answering (latent CCoT thinking)
#   T_OVERRIDE=N        eval recurrence depth (default: config mean_recurrence=8)
# Any being non-default appends -ppcN-ccotM / -TN to the label so the plain
# run is kept.  Base has no cross state (passes are no-ops) — its compute
# axis is T_OVERRIDE only.
#
# FLOP-matched ladder (per-token layer-apps; base = 8+6r, cortex = p*(8+6T)):
#   k=4 : base T_OVERRIDE=4  <-> cortex T_OVERRIDE=4                (32, exact)
#   k=8 : base T_OVERRIDE=8  <-> cortex default                    (56, exact)
#   k=16: base T_OVERRIDE=16 <-> cortex PASSES_PER_CHUNK=2         (112 vs 104, +8%)
#   k=32: base T_OVERRIDE=32 <-> cortex PASSES_PER_CHUNK=4         (224 vs 200, +12%)
# Keep NUM_CHUNKS=4 fixed across tiers so buffer depth stays at the trained
# regime and only the compute axis varies.
#
# Prereqs (login node): datasets downloaded (evals/download_datasets.py) and
# checkpoints fixed for eval loading (tools/prepare_eval_checkpoint.py) — the
# fix also runs here idempotently as a belt-and-suspenders (pure file ops).
# T is NOT passed: evals read mean_recurrence (8) + memory flags from the
# checkpoint config; memory carry is auto-detected via has_cross_state.

cd $SLURM_SUBMIT_DIR

module load anaconda3
module load cuda/12.1.1
conda activate cortex-retro

export SCRATCH=${SCRATCH:-$HOME/scratch}   # not set by default on PACE Phoenix
export HF_HOME=$SCRATCH/hf_cache
export HF_HUB_OFFLINE=1                 # compute nodes: cache only
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

EVAL_TAG=${EVAL_TAG:-$(date +%Y%m%d)}
PASSES_PER_CHUNK=${PASSES_PER_CHUNK:-1}
CCOT_PASSES=${CCOT_PASSES:-0}
T_OVERRIDE=${T_OVERRIDE:-}
# 1024 matches the training write cadence: train.py splits 4096-token samples
# into cross_chunks=4 sub-windows of 1024, so M_cross was trained on writes
# every 1024 tokens.  (The old eval default of 2048 wrote at 2x the trained
# window size.)
SEQ_LEN=${SEQ_LEN:-1024}
# NUM_CHUNKS=4 reproduces the trained buffer regime exactly: 4 equal
# subwindows -> 3 gated updates before the final read, same as cross_chunks=4
# training.  Window size = ceil(total/4) capped at SEQ_LEN, so examples
# longer than 4*SEQ_LEN tokens fall back to more, SEQ_LEN-sized chunks
# (deeper-than-trained buffer states — unavoidable there without retraining
# at higher cross_chunks).  NUM_CHUNKS=0 restores fixed-size chunking.
NUM_CHUNKS=${NUM_CHUNKS:-4}
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

if [ "$PASSES_PER_CHUNK" -gt 1 ] || [ "$CCOT_PASSES" -gt 0 ]; then
    LABEL="${LABEL}-ppc${PASSES_PER_CHUNK}-ccot${CCOT_PASSES}"
fi
T_ARGS=()
if [ -n "$T_OVERRIDE" ]; then
    LABEL="${LABEL}-T${T_OVERRIDE}"
    T_ARGS=(--T "$T_OVERRIDE")
fi
# Non-default chunking regime goes in the label too (e.g. the no-chunk mode
# NUM_CHUNKS=1 SEQ_LEN=4096 -> "-nc1-sl4096") so it never collides with the
# standard 4-chunk run under the same EVAL_TAG.
[ "$NUM_CHUNKS" -ne 4 ]   && LABEL="${LABEL}-nc${NUM_CHUNKS}"
[ "$SEQ_LEN" -ne 1024 ]   && LABEL="${LABEL}-sl${SEQ_LEN}"

OUT_ROOT="eval_results/longcontext_${EVAL_TAG}/${LABEL}"
mkdir -p "$OUT_ROOT"
echo "=== long-context eval: $LABEL  ($MODEL_DIR) -> $OUT_ROOT ==="

python evals/eval_babilong.py \
    --model_name "$MODEL_DIR" \
    --dataset_path data/BABILong \
    --tasks qa1 qa2 qa3 \
    --seq_len $SEQ_LEN \
    --num_chunks $NUM_CHUNKS \
    --passes_per_chunk $PASSES_PER_CHUNK \
    --ccot_passes $CCOT_PASSES \
    "${T_ARGS[@]}" \
    --out_dir "$OUT_ROOT/babilong"

python evals/eval_longmemeval.py \
    --model_name "$MODEL_DIR" \
    --dataset_path data/LongMemEval \
    --seq_len $SEQ_LEN \
    --num_chunks $NUM_CHUNKS \
    --passes_per_chunk $PASSES_PER_CHUNK \
    --ccot_passes $CCOT_PASSES \
    "${T_ARGS[@]}" \
    --out_dir "$OUT_ROOT/longmemeval"
