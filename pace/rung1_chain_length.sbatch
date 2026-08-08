#!/bin/bash
#SBATCH --account=gts-aivanova7-lab
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=64GB
#SBATCH --gres=gpu:H200:1
#SBATCH --constraint=H200
#SBATCH -t 36:00:00
#SBATCH -q inferno
#SBATCH -o logs/Report-%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=henryyjiang42@gmail.com

# Track A side-run, 2026-08-05 — CHAIN LENGTH.  Companion to
# pace/rung1_corpus_mix.sbatch; both run alongside B2's healing phase.
#
# THE QUESTION.  Track A trained at cross_chunks=4, so the buffer held at most
# 3 prior writes when it did its last read, and gradient crossed 2 boundaries
# (tb2).  The model got good at exactly two hops — which is precisely the
# BABILong shape (2k fine, 4k+ structural zero) and the non-compounding carry
# delta.  Nobody has ever trained an arm with a longer chain.
#
# WHAT IS ACTUALLY NEW HERE.  Widening the gradient span INSIDE a 4-chunk chain
# is already tested and DEAD: acc4v-tb4 gave +0.02353 vs tb2's +0.02370, -0.1sigma,
# a failed pre-registered prediction.  The eval-time depth lever is also dead
# (T = 4/8/16 -> 32.8/32.5/32.6% recovered; T=8->16 moves loss by 5e-5).  So the
# untested lever is THE CHAIN ITSELF: cross_chunks 4 -> 8.  carry_grad_chunks
# moves 2 -> 4 with it purely to hold the trained fraction at 50%, so the
# comparison isolates chain length rather than re-running the dead tb lever.
#
# PRE-REGISTERED BAR (fraction of ceiling, never raw nats — the 0.1-nat bar was
# retracted 2026-08-04 as being above the measurable ceiling).  Reading the
# EXISTING cc=4 checkpoint off-distribution at nc=8 already gives carry
# +0.04327 against a +0.14855 ceiling = 29.1%.  An arm TRAINED at nc=8 must beat
# that for chain length to be the lever; landing at ~29% means nc=8 is a better
# INSTRUMENT and training on it added nothing.
#
# BUT 29.1% IS NOT A DROP-IN BAR FOR THIS ARM.  It was measured on
# k0-acc4v-tb2-rs-ep3-rcl: LEGACY AccumCCoT, 4 vecs, trained on the recall mix.
# This arm is PrefixAccum, 32 vecs, plain PG-19 — three differences.  The bar
# that actually decides this run is the cc=4 PREFIX control
# (rung1-pfxaccum32-cc4-tb2-rs-pg19, from rung1_corpus_mix.sbatch), read at
# nc=8.  Quote 29.1% as historical context only; if the control is missing,
# this arm has no valid reference and the run cannot conclude anything.
# Secondary: ceil2 found one hint of non-context-substitute behaviour at nc=8 —
# the carry's benefit persisted further into the chunk than the short-k oracle's
# (band1/band0 0.249 vs 0.115).  Re-check that here; it is the only evidence in
# the project pointing at something other than a topic vector.
#
# HONEST COST, STATE IT UP FRONT.  512-token chunks mean less local context, so
# k0 rises 2.865 -> 2.923.  You lose 0.058 nats and the carry hands back 0.043:
# net -0.014 in raw NLL.  This is a better instrument, NOT a free loss win, and
# it must not be presented as one.
#
# THE CONFOUND, DOCUMENTED.  At fixed 4096-token rows, cc=8 changes TWO things:
# chain depth 4->8 AND chunk length 1024->512.  Chunk length is the biggest
# lever in the project (the oracle ceiling is 2.2x higher at 512-token chunks
# and does not collapse), so a win here is not cleanly attributable to depth.
# Disentangling needs 1024-token chunks at depth 8, i.e. 8192-token rows:
#     python tools/prepare_pg19_dataset.py --max_length 8192 \
#         --out data/pg19_olmo_len8192            # LOGIN node
#     MAX_LENGTH=8192 DATA_PATH=data/pg19_olmo_len8192 \
#         sbatch pace/rung1_chain_length.sbatch
# That doubles the retained graph again on top of what already OOMs (below), so
# it is H200-only.  Optional third arm; the cc=8-on-existing-data arm below
# needs no re-prep.
#
# H200, NOT A100 — MEASURED, NOT PRECAUTIONARY (2026-08-05, job 11702365).
# The first submission of this arm died in 64s: torch.OutOfMemoryError, 78.37
# GiB allocated of the A100-80's 79.25.  It is NOT fragmentation — only 28 MiB
# was reserved-but-unallocated and expandable_segments was already on, so there
# is nothing for PYTORCH_CUDA_ALLOC_CONF to reclaim.  The cc4/tb2 sibling peaks
# at 71.8 GiB on the same node type (job 11702363), i.e. ~7 GiB of headroom
# against a change that doubles the retained chunks 2 -> 4.  train.py has no
# --grad_checkpoint (that was cortex-main), so H200 is the only no-code fix.
# Do NOT "fix" this by lowering CARRY_GRAD_CHUNKS to 2: that drops the trained
# fraction from 50% to 25% and confounds chain length with the dead tb lever,
# which is the one thing this arm exists to avoid.
#
# WALLTIME.  36h, up from 16h.  The cc4/tb2 sibling ran 17.4 s/update on A100
# and PG-19 at EPOCHS=1 is 28,592 rows / 16 = 1,787 updates, so the A100 cost
# would have been ~9h — but cc=8 doubles the chain and the H200 rate for this
# geometry has never been measured.  Check Time/step in the first minutes of the
# log and trim on the next link rather than guessing again.
#
# ARMS.  EPOCHS=3 IS REQUIRED, NOT OPTIONAL — a plain `sbatch` of this script is
# EPOCHS=1 and has no valid control (job 11718571, cancelled for exactly that).
#   RUN_NAME=rung1-pfxaccum32-cc8-tb4-rs-pg19-ep3 EPOCHS=3 \
#       sbatch pace/rung1_chain_length.sbatch   # cc=8, tb4, 512-tok chunks
# The CONTROL IS SHARED with the mix run — do not submit it twice, and it must
# carry the SAME EPOCHS or the LR horizons differ (train.py:1198):
#   RUN_NAME=rung1-pfxaccum32-cc4-tb2-rs-pg19-ep3 EPOCHS=3 \
#       DATA_PATH=data/pg19_olmo_len4096 sbatch pace/rung1_corpus_mix.sbatch
# gives pfxaccum32-cc4-tb2-rs-pg19, same mechanism, same data, same budget,
# geometry the only difference.  Three arms total across both scripts.
#
# DATA is plain PG-19, not the recall mix, for two reasons: it shares the
# control above, and the recall-supervised mix already failed its viability gate
# (probe uplift 0.0045 +/- 0.0009 where ~0.011 was predicted, zero recall via
# carry) so it adds a known-dead variable.
#
# EVAL: every downstream eval on this arm needs N_CHUNKS=8, or it will be scored
# at a geometry it was not trained for and silently look worse.
# eval_carry_ablation and eval_context_ceiling are the readouts.  BABILong may
# be run for the paper table vs base, but it cannot measure memory here — its
# full-window baseline is flat across a 32x context range.

cd $SLURM_SUBMIT_DIR

module load anaconda3
module load cuda/12.1.1
conda activate cortex-retro

export SCRATCH=${SCRATCH:-$HOME/scratch}
export HF_HOME=$SCRATCH/hf_cache
export HF_HUB_OFFLINE=1
export WANDB_DIR=$SCRATCH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# TCPStore port for the trivial 1-process group Muon needs (train.py:780).
# train.py derives 20000 + SLURM_JOB_ID % 40000 and installs it with
# os.environ.setdefault, so anything exported here wins.  That derivation was
# itself the fix for shared-node collisions and it STILL collided: job 11702360
# (2026-08-05) died in 29s with EADDRINUSE on 42360.  Probe a free port instead.
# setdefault is why this needs no train.py change.
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=$(python -c "import socket;s=socket.socket();s.bind(('',0));print(s.getsockname()[1]);s.close()")
echo "MASTER_PORT=$MASTER_PORT"

MAX_LENGTH=${MAX_LENGTH:-4096}
CROSS_CHUNKS=${CROSS_CHUNKS:-8}            # THE independent variable (was 4)
ACCUM_VECS=${ACCUM_VECS:-32}               # PARAMETER SHAPE (summary_emb) — never vary mid-chain
ACCUM_MAX=${ACCUM_MAX:-256}                # 8 x 32 — moves WITH cross_chunks
CARRY_GRAD_CHUNKS=${CARRY_GRAD_CHUNKS:-4}  # 4 of 8 = the same 50% as tb2 of cc4
PREFIX_MODE=${PREFIX_MODE:-accum}
PREFIX_EOS_RESET=${PREFIX_EOS_RESET:-false}
PREFIX_POS=${PREFIX_POS:-tail}
RANDOM_SEGMENTS=${RANDOM_SEGMENTS:-true}
EPOCHS=${EPOCHS:-1}
DATA_PATH=${DATA_PATH:-data/pg19_olmo_len4096}

# accum_max must hold the whole chain or the FIFO silently trims and breaks the
# stop-grad slice bookkeeping.  Raise ACCUM_MAX; never lower ACCUM_VECS.
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
if [ $(( CARRY_GRAD_CHUNKS * 2 )) -ne "$CROSS_CHUNKS" ]; then
    echo "WARNING: carry_grad_chunks=$CARRY_GRAD_CHUNKS is not half of" \
         "cross_chunks=$CROSS_CHUNKS — the trained fraction of the chain no" \
         "longer matches the cc4/tb2 control, so a delta here is confounded" \
         "with the (already dead) tb lever."
fi

ARM="pfx${PREFIX_MODE}${ACCUM_VECS}-cc${CROSS_CHUNKS}-tb${CARRY_GRAD_CHUNKS}"
[ "$MAX_LENGTH" -ne 4096 ] && ARM="${ARM}-len${MAX_LENGTH}"
[ "$RANDOM_SEGMENTS" = "true" ] && ARM="${ARM}-rs"
ARM="${ARM}-pg19"
RUN_NAME=${RUN_NAME:-rung1-${ARM}}

# --- resume, for the wallclock case ----------------------------------------
# The H200 rate for cc=8 is unmeasured, so the 36h limit is a guess; restarting
# from 0 on a timeout throws away real GPU-days.  save_interval=500 writes a
# weights-only dir every 500 and a full resumable checkpoint_<step>/chkpt.pt
# every 1,000.  Newest checkpoint, NUMERICALLY (checkpoint_9500 must not
# outrank checkpoint_150000):
#   step=$(ls -d cortex-retro-ft/$RUN_NAME/checkpoint_* | sed 's/.*checkpoint_//' \
#          | sort -n | tail -1)
#   RESUME_PATH=cortex-retro-ft/$RUN_NAME/checkpoint_$step sbatch pace/rung1_chain_length.sbatch
# Pass the SAME geometry env as the original link or the arm identity moves
# under you mid-chain — and the run name will no longer describe the weights.
RESUME_PATH=${RESUME_PATH:-}
CHAIN_ARGS=""
if [ -n "$RESUME_PATH" ]; then
    if [ ! -f "$RESUME_PATH/chkpt.pt" ]; then
        echo "ERROR: no chkpt.pt under RESUME_PATH=$RESUME_PATH"; exit 1
    fi
    CHAIN_ARGS="--resume_path $RESUME_PATH"
    echo "Resuming from $RESUME_PATH"
fi

echo "=== $RUN_NAME | data=$DATA_PATH | prefix=$PREFIX_MODE ==="
echo "    geometry: cross_chunks=$CROSS_CHUNKS ($(( MAX_LENGTH / CROSS_CHUNKS ))-tok chunks)" \
     " accum_vecs=$ACCUM_VECS accum_max=$ACCUM_MAX carry_grad_chunks=$CARRY_GRAD_CHUNKS"
echo "    control to compare against: rung1-pfxaccum32-cc4-tb2-rs-pg19"
echo "    bar to beat: the cc4 PREFIX control read at nc=8 (29.1% is the LEGACY"
echo "                 acc4v checkpoint — historical context, not a valid bar here)"
echo "    REMINDER: every eval on this arm needs N_CHUNKS=8"

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
    --cortex.random_segments $RANDOM_SEGMENTS \
    $CHAIN_ARGS
