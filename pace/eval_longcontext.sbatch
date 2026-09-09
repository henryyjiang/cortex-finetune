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
#   NO_CARRY=true       paired no-memory control: identical chunking, buffer
#                       never built.  This is the ONLY no-carry reference that
#                       holds the weights fixed -- a base-model comparison also
#                       varies the lineage (50B-token conversion vs 5B retrofit),
#                       so it cannot attribute any difference to the memory.
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
#
# B1 / B2 retrofit checkpoints (2026-08-31): OUT_ROOT / CKPT_NAME / PREP_BASE /
# BASE_CKPT, same meaning as in eval_basic.sbatch and the carry-ablation
# sbatch.  Two things do NOT default correctly for a retrofit arm:
#   * T_OVERRIDE.  A retrofit-derived config says mean_recurrence=32 (inherited
#     from the untrained base), so the sentence above — "evals read
#     mean_recurrence (8)" — is a Track-A statement only.  Pass T_OVERRIDE=8.
#   * SEQ_LEN / NUM_CHUNKS.  These must match the arm's TRAINING chunk
#     geometry, i.e. 4096/cross_chunks.  B2's arms train at cross_chunks=8, so
#     SEQ_LEN=512 NUM_CHUNKS=8, not the Track-A 1024/4.
#   OUT_ROOT=cortex-retrofit CKPT_NAME=final_checkpoint \
#       PREP_BASE=ckpts/olmo-retrofit-cortex RUN=retro-b2-acc32-cc8-mr8 \
#       T_OVERRIDE=8 SEQ_LEN=512 NUM_CHUNKS=8 EVAL_TAG=b2-final \
#       sbatch pace/eval_longcontext.sbatch
#
# Powering the carry-on/carry-off contrast (2026-09).  The n=100/bucket tables
# cannot resolve the ~1-point carry effect: pooled over 18 cells the unpaired
# test has an MDE of ~3.1 pt (tools/power_longcontext.py).  Three knobs fix
# that, and all three are on by default now:
#   BABILONG_MAX=250   examples per (task, length) cell.  RMT-team/BABILong
#                      holds only 100 rows per cell -- anything above 100 needs
#                      BABILONG_REPO=RMT-team/babilong-1k-samples (~1000 rows,
#                      qa1-qa5).  The harness WARNS instead of silently capping.
#   records.jsonl      per-example rows, written unconditionally.  Without them
#                      the runs can only be compared unpaired, which throws
#                      away most of the available power; McNemar on the matched
#                      pairs is what tools/analyze_longcontext_pairs.py runs.
#   SCORE_NLL=true     gold-answer NLL alongside 0/1 containment.  Continuous
#                      and paired, so it resolves effects that accuracy cannot
#                      at any n we can afford.  One extra forward per example.
# LME_MAX=0 takes all of longmemeval_s; its 500 questions TOTAL are this eval's
# hard ceiling, so LongMemEval cannot be powered further and BABILong is where
# added examples buy anything.
#
# Sharding: BABILONG_TASKS + BABILONG_OUT_SUFFIX put one task in its own job
# under its own sub-dir (babilong-qa1/, ...); the analysis tool globs and
# merges them.  Needed for the T=32 carry-on cell, which does not fit the wall
# clock at 250/bucket.  Override the wall at submit time: sbatch -t 24:00:00.

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
# Paired carry-off control.  Refused on BASE=true: the base has no cross state,
# so its normal run is ALREADY the no-carry condition and the flag would only
# produce a duplicate under a misleading label.
NO_CARRY=${NO_CARRY:-false}
BASE=${BASE:-false}
# --- power knobs (see the header) ---
BABILONG_MAX=${BABILONG_MAX:-250}
BABILONG_TASKS=${BABILONG_TASKS:-"qa1 qa2 qa3"}
BABILONG_REPO=${BABILONG_REPO:-RMT-team/babilong-1k-samples}
BABILONG_PATH=${BABILONG_PATH:-data/babilong-1k}
BABILONG_OUT_SUFFIX=${BABILONG_OUT_SUFFIX:-}
LME_MAX=${LME_MAX:-0}
SCORE_NLL=${SCORE_NLL:-true}
SKIP_BABILONG=${SKIP_BABILONG:-false}
SKIP_LME=${SKIP_LME:-false}
NLL_ARGS=()
[ "$SCORE_NLL" = "true" ] && NLL_ARGS=(--score_nll)
# ACCUM_MAX overrides the carry buffer's FIFO cap.  The buffer keeps the newest
# ACCUM_MAX/accum_vecs chunks and drops the rest: B2 trained at 256/32 = 8
# chunks = 4096 tokens, and train.py ASSERTS cross_chunks*accum_vecs <=
# accum_max, so the trim branch never fires in training and fires on every eval
# example past 4096 tokens (all of LongMemEval, 3 of 6 BABILong buckets).
# Raising it puts the prefix at an untrained sequence length -- deliberately.
# Cost grows fast: each priming forward attends over n_carried + chunk + slots,
# so 2048 carried vectors is ~3x the sequence length of the 256 default and
# more than 3x the time.  Budget accordingly.
ACCUM_MAX=${ACCUM_MAX:-}
AM_ARGS=()
[ -n "$ACCUM_MAX" ] && AM_ARGS=(--accum_max "$ACCUM_MAX")
# Scoring changes (2026-09-09).  All are metric/record only -- none touches the
# model or the carry, so the carry-on/carry-off contrast stays as fair as it
# was.  RANK_ANSWERS scores the cell's own answer set instead of relying on
# free generation + containment, which is exactly what the raised-P(EOS)
# failure mode attacks; PMI adds the no-story denominator so the answer prior
# drops out; BLANK_CONTEXT is that prior floor on its own (no priming at all --
# minutes, not hours).
RANK_ANSWERS=${RANK_ANSWERS:-true}     # BABILong only: LME answers are free-form
PMI=${PMI:-true}
BLANK_CONTEXT=${BLANK_CONTEXT:-false}
RANK_ARGS=()
[ "$RANK_ANSWERS" = "true" ] && RANK_ARGS+=(--rank_answers)
[ "$PMI" = "true" ]          && RANK_ARGS+=(--pmi)
BLANK_ARGS=()
LABEL_BLANK=""
if [ "$BLANK_CONTEXT" = "true" ]; then
    BLANK_ARGS=(--blank_context)
    LABEL_BLANK="-blank"
fi
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
CARRY_ARGS=()
if [ "$NO_CARRY" = "true" ]; then
    if [ "$BASE" = "true" ]; then
        echo "NO_CARRY=true is a no-op on BASE=true (no cross state to drop)."
        exit 1
    fi
    CARRY_ARGS+=(--no_carry)
    LABEL="${LABEL}-nocarry"
fi
# Buffer capacity goes in the label too: an am-sweep writes several runs of the
# same arm at the same T under one EVAL_TAG, and without this they collide.
# NO_CARRY builds no buffer, so an am label there would be a lie -- refuse it.
if [ -n "$ACCUM_MAX" ]; then
    if [ "$NO_CARRY" = "true" ] || [ "$BASE" = "true" ]; then
        echo "ACCUM_MAX is meaningless without a carry buffer "
        echo "(NO_CARRY=true / BASE=true build none). Drop ACCUM_MAX."
        exit 1
    fi
    LABEL="${LABEL}-am${ACCUM_MAX}"
fi
LABEL="${LABEL}${LABEL_BLANK}"

OUT="eval_results/longcontext_${EVAL_TAG}/${LABEL}"
mkdir -p "$OUT"
echo "=== long-context eval: $LABEL  ($MODEL_DIR) -> $OUT ==="

if [ "$SKIP_BABILONG" != "true" ]; then
python evals/eval_babilong.py \
    --model_name "$MODEL_DIR" \
    --dataset_path "$BABILONG_PATH" \
    --dataset_repo "$BABILONG_REPO" \
    --tasks $BABILONG_TASKS \
    --max_examples $BABILONG_MAX \
    --seq_len $SEQ_LEN \
    --num_chunks $NUM_CHUNKS \
    --passes_per_chunk $PASSES_PER_CHUNK \
    --ccot_passes $CCOT_PASSES \
    "${T_ARGS[@]}" "${CARRY_ARGS[@]}" "${NLL_ARGS[@]}" "${AM_ARGS[@]}" \
    "${RANK_ARGS[@]}" "${BLANK_ARGS[@]}" \
    --out_dir "$OUT/babilong${BABILONG_OUT_SUFFIX}"
fi

if [ "$SKIP_LME" != "true" ]; then
python evals/eval_longmemeval.py \
    --model_name "$MODEL_DIR" \
    --dataset_path data/LongMemEval \
    --max_examples $LME_MAX \
    --seq_len $SEQ_LEN \
    --num_chunks $NUM_CHUNKS \
    --passes_per_chunk $PASSES_PER_CHUNK \
    --ccot_passes $CCOT_PASSES \
    "${T_ARGS[@]}" "${CARRY_ARGS[@]}" "${NLL_ARGS[@]}" "${AM_ARGS[@]}" \
    "${BLANK_ARGS[@]}" \
    --out_dir "$OUT/longmemeval"
fi
