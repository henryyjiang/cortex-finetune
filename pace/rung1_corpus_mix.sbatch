#!/bin/bash
#SBATCH --account=gts-aivanova7-lab
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=64GB
#SBATCH --gres=gpu:A100:1
#SBATCH --constraint=A100-80GB
#SBATCH -t 16:00:00
#SBATCH -q inferno
#SBATCH -o logs/Report-%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=henryyjiang42@gmail.com

# Track A side-run, 2026-08-05 — REPLAY MIX vs CORPUS SWAP.
# Runs alongside B2's healing phase; ~1 A100-day for the pair.
#
# THE QUESTION.  Every Nemotron number on the books is a corpus SWAP.  The
# `-nemo` arms replaced PG-19 outright, and two things moved together: GSM8K
# recovered (acc4v 10.4 -> 14.0, ki4 0.8 -> 39.6) while the write statistics
# changed hard (new-vs-old cos 0.94 -> 0.53, norm 5.78 -> 1.03) and the carry
# delta FELL (0.02370 -> 0.02045, and the slice columns went flat: newest-4
# 0.0079 vs oldest-4 0.0068, i.e. diverse writes bought redundancy not reach).
# So "does replay rescue GSM8K without costing carry?" is unanswered, and B2's
# arm phase has to pick a data recipe in ~2 weeks.  A MIX holds the
# memory-bearing corpus fixed and adds replay on top; a SWAP does not.
#
# TWO ARMS, BOTH REQUIRED (the control is not optional — see MECHANISM below):
#   sbatch pace/rung1_corpus_mix.sbatch                            # PG-19 + Nemotron 50/50
#   DATA_PATH=data/pg19_olmo_len4096 sbatch pace/rung1_corpus_mix.sbatch   # PG-19 pure control
# Prep the mix first, on a LOGIN node:
#   python tools/prepare_corpus_mix.py \
#       --a data/pg19_olmo_len4096 --b data/nemotron_math_olmo_len4096_4b \
#       --out data/pg19_nemo50_olmo_len4096 --ratio 0.5 --max_rows 97656
#   (97,656 rows x 4096 = 400M tokens, matching the Track-A arms' budget.  Run
#    the control with EPOCHS set so it sees the SAME token count, or the
#    comparison is confounded with exposure.)
#
# MECHANISM — why this is NOT directly comparable to 10.4 / 14.0.
# Those arms used the legacy AccumCCoT buffer.  `--cortex.accum_ccot` is now a
# RETIRED FLAG THAT RAISES (bug class #2: silent memory-off runs produce healthy
# loss curves), and the legacy buffer has been superseded by PrefixAccumBuffer,
# which fixes four infidelities to AutoCompressor.  We use the PREFIX buffer
# here on purpose: the whole point is to inform B2, and B2 runs prefix.  The
# cost is that the historical 10.4 / 14.0 pair becomes background context only,
# which is exactly why the PG-19 pure control above has to run too.
#
# GEOMETRY IS HELD AT THE HISTORICAL SETTING ON PURPOSE.
# cross_chunks=4 on 4096-token rows = 1024-token chunks, matching every Track-A
# carry number.  Chunk length is the biggest lever in the project (ceil2: the
# oracle ceiling is 2.2x higher at 512-token chunks), which is precisely why it
# must NOT move in a run whose independent variable is the data.  The cc=8
# question is a separate arm; do not fold it in here.
#
# accum_vecs=32 matches B2, NOT a revival of the acc16v/acc50v arms — those were
# declined because they existed to fix write diversity, and write diversity was
# falsified as the bottleneck at both ends of its range (cos 0.94 -> 0.0005 with
# the delta going +0.024 -> -0.013).  32 here is for B2 parity, nothing else.
#
# READOUTS, in the order they decide anything:
#   1. eval_basic (GSM8K)          — did replay rescue forgetting?
#   2. eval_carry_ablation         — did it cost carry?  (mix vs control, and
#                                    vs the +0.0237 / +0.0205 swap pair)
#   3. diag_accum_buffer           — did the write statistics move, or only GSM8K?
# Do NOT run eval_longcontext on these.  BABILong's full-window baseline is flat
# across a 32x context range (qa1 25/22/23/22/20/22), so it cannot discriminate.

cd $SLURM_SUBMIT_DIR

module load anaconda3
module load cuda/12.1.1
conda activate cortex-retro

export SCRATCH=${SCRATCH:-$HOME/scratch}
export HF_HOME=$SCRATCH/hf_cache
export HF_HUB_OFFLINE=1
export WANDB_DIR=$SCRATCH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MAX_LENGTH=${MAX_LENGTH:-4096}
CROSS_CHUNKS=${CROSS_CHUNKS:-4}          # 1024-token chunks — held fixed, see above
ACCUM_VECS=${ACCUM_VECS:-32}             # PARAMETER SHAPE (summary_emb) — never vary mid-chain
ACCUM_MAX=${ACCUM_MAX:-128}              # 4 x 32 — the assert is exact
CARRY_GRAD_CHUNKS=${CARRY_GRAD_CHUNKS:-2}  # AC's stop-grad after 2 compressions
PREFIX_MODE=${PREFIX_MODE:-accum}        # accum = AutoCompressor append; gated = LM2 merge
PREFIX_EOS_RESET=${PREFIX_EOS_RESET:-false}
PREFIX_POS=${PREFIX_POS:-tail}
RANDOM_SEGMENTS=${RANDOM_SEGMENTS:-true} # AC recipe, matches the -rs arms
EPOCHS=${EPOCHS:-1}
DATA_PATH=${DATA_PATH:-data/pg19_nemo50_olmo_len4096}

# Fail loudly rather than let the FIFO silently trim and break the stop-grad
# slice bookkeeping — same guard as b2_retrofit.sbatch.
if [ $(( CROSS_CHUNKS * ACCUM_VECS )) -gt "$ACCUM_MAX" ]; then
    echo "ERROR: cross_chunks x accum_vecs ($CROSS_CHUNKS x $ACCUM_VECS =" \
         "$(( CROSS_CHUNKS * ACCUM_VECS ))) exceeds accum_max ($ACCUM_MAX)."
    echo "  Raise ACCUM_MAX.  Do NOT lower ACCUM_VECS — it is a parameter shape."
    exit 1
fi
if [ $(( MAX_LENGTH % CROSS_CHUNKS )) -ne 0 ]; then
    echo "ERROR: MAX_LENGTH ($MAX_LENGTH) not divisible by CROSS_CHUNKS" \
         "($CROSS_CHUNKS) — chunks would be ragged."; exit 1
fi

ARM="pfx${PREFIX_MODE}${ACCUM_VECS}-cc${CROSS_CHUNKS}-tb${CARRY_GRAD_CHUNKS}"
[ "$RANDOM_SEGMENTS" = "true" ] && ARM="${ARM}-rs"
case "$DATA_PATH" in
    *nemo*[0-9]*|*mix*) ARM="${ARM}-mix";;
    *nemotron*)         ARM="${ARM}-nemo";;
    *)                  ARM="${ARM}-pg19";;
esac
RUN_NAME=${RUN_NAME:-rung1-${ARM}}

echo "=== $RUN_NAME | data=$DATA_PATH | prefix=$PREFIX_MODE ==="
echo "    geometry: cross_chunks=$CROSS_CHUNKS ($(( MAX_LENGTH / CROSS_CHUNKS ))-tok chunks)" \
     " accum_vecs=$ACCUM_VECS accum_max=$ACCUM_MAX carry_grad_chunks=$CARRY_GRAD_CHUNKS"

python train.py \
    --run_name $RUN_NAME \
    --out_path cortex-retro-ft \
    --model_name ckpts/olmo8-cortex \
    --preprocessed_data_path $DATA_PATH \
    --max_length $MAX_LENGTH \
    --epochs $EPOCHS \
    --micro_batch_size 1 \
    --batch_size 16 \
    --optim_config.lr 1e-4 \
    --muon.use_muon true \
    --muon.lr 0.001 \
    --throttle true \
    --no_amp false \
    --max_grad_norm 1.0 \
    --save_interval 500 \
    --seed 74 \
    --cortex.use_memory true \
    --cortex.memory_lr 5e-4 \
    --cortex.memory_slots 0 \
    --cortex.prefix_memory $PREFIX_MODE \
    --cortex.accum_vecs $ACCUM_VECS \
    --cortex.accum_max $ACCUM_MAX \
    --cortex.carry_grad_chunks $CARRY_GRAD_CHUNKS \
    --cortex.cross_chunks $CROSS_CHUNKS \
    --cortex.eos_from_tokens true \
    --cortex.prefix_eos_reset $PREFIX_EOS_RESET \
    --cortex.prefix_pos $PREFIX_POS \
    --cortex.freeze_loop true \
    --cortex.random_segments $RANDOM_SEGMENTS
