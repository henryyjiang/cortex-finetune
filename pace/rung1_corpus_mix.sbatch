#!/bin/bash
#SBATCH --account=gts-aivanova7-lab
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=64GB
#SBATCH --gres=gpu:A100:1
#SBATCH --constraint=A100-80GB
#SBATCH -t 36:00:00
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
#   EPOCHS=3 DATA_PATH=data/pg19_olmo_len4096 sbatch pace/rung1_corpus_mix.sbatch  # PG-19 control
# Prep the mix first, on a LOGIN node:
#   python tools/prepare_corpus_mix.py \
#       --a data/pg19_olmo_len4096 --b data/nemotron_math_olmo_len4096_4b \
#       --out data/pg19_nemo50_olmo_len4096 --ratio 0.5 --max_rows 97656
#   (97,656 rows x 4096 = 400M tokens, matching the Track-A arms' budget.  Run
#    the control with EPOCHS set so it sees the SAME token count, or the
#    comparison is confounded with exposure.)
#
# THE EPOCHS=3 ABOVE IS LOAD-BEARING, AND IT SPLITS THE CONTROL IN TWO.
# EPOCHS defaults to 1, which is correct for the mix (97,656 rows / 16 = 6,103
# updates) and WRONG for the control: pg19_olmo_len4096 is only 28,592 rows =
# 1,787 updates/epoch, so a default-EPOCHS control sees 117M tokens against the
# mix's 400M — a 3.4x exposure gap on top of the corpus difference, i.e. exactly
# the confound the note above warns about.  EPOCHS=3 gives 5,361 updates / 351M
# tokens, the closest integer match.
#   The knock-on: rung1_chain_length.sbatch names pfxaccum32-cc4-tb2-rs-pg19 as
#   ITS control too, so the cc=8 arm has to match whatever EPOCHS this control
#   runs at.  EPOCHS sets the LR horizon directly (train.py:1198,
#   max_training_steps = epochs x updates_per_epoch, since these arms pass no
#   --max_steps), so an EPOCHS=3 control read at step 1,787 is a third of the
#   way through its schedule while an EPOCHS=1 arm is fully cooled down — the
#   per-segment cooldown artifact that got baked into the 2026-06 weights.  An
#   intermediate checkpoint is NOT a valid match for a shorter run.
#
# RESOLVED 2026-08-06: ONE control at EPOCHS=3, and the cc=8 arm runs at
# EPOCHS=3 too.  The alternative (a second EPOCHS=1 control, ~9h) buys nothing
# the longer arm does not, and EPOCHS=3 lands the cc=8 arm on Track A's exact
# 5,361-update / 351M-token budget, so it is cross-referenceable against
# acc4v-tb2-rs-ep3-rcl for free.  3 epochs on PG-19 with random_segments=true
# is what Track A already did — established, not a new risk.
#
# ============================================================================
# SUPERSEDED 2026-08-08.  THE 2026-08-05/06 RUNS ARE VOID — THEY HAD NO MEMORY.
# ============================================================================
# All three arms (-cc4-tb2-rs-pg19-ep3, -cc4-tb2-rs-mix, -cc8-tb4-rs-pg19-ep3)
# trained as plain no-memory baselines.  ckpts/olmo8-cortex held a modeling
# snapshot predating the prefix rewrite, so forward() never called
# prefix_pack/prefix_unpack: the summary slots were never spliced, no gradient
# ever reached cortex.prefix, and summary_emb sat at its N(0, 0.02) init for
# the whole run behind a perfectly healthy loss curve.  Bug class 2, third
# instance.  Confirmed from the checkpoint safetensors (summary_seeded == 0.0,
# summary_emb row-row cosine 0.0001 == the d=2048 random baseline, row norms
# 0.905 == 0.02*sqrt(2048), orthogonal to wte[eos] and 10x smaller).
#   train.py now RAISES at step 0 on a stale snapshot (commit 177181b), so this
#   cannot recur silently.  ckpts/olmo8-cortex was refreshed 2026-08-08 10:40.
#   `ls -l` + `diff` on a snapshot proves nothing about what TRAINING saw — a
#   re-prepare rewrites it.  Compare its mtime against the run's START date.
#
# WHAT SURVIVES: the CORPUS comparison.  Both cc4 arms were no-memory finetunes
# differing ONLY in corpus, so GSM8K 1.6% -> 37.6% with every text column flat
# or up is still clean and controlled — it was never a memory claim.  B2's arm
# corpus is settled on the mix and does not wait on any re-run.
# WHAT DIES: every carry number, and the chain-length arm outright.
#
# RE-RUN PLAN, decided 2026-08-08 — EPOCHS=1, and ONLY the two arms chain
# length needs.  The mix arm is NOT re-run: its question is answered.
#   RUN_NAME=rung1-pfxaccum32-cc4-tb2-rs-pg19-ep1 \
#       DATA_PATH=data/pg19_olmo_len4096 sbatch pace/rung1_corpus_mix.sbatch
#   RUN_NAME=rung1-pfxaccum32-cc8-tb4-rs-pg19-ep1 \
#       sbatch pace/rung1_chain_length.sbatch
# Both default to EPOCHS=1, so pass NOTHING — 1,787 updates / 117M tokens each,
# identical data and tokens-per-step (cross_chunks changes chunk length, not
# row length), so the pair is matched TO EACH OTHER, which is all the
# chain-length comparison requires.  ~1/3 the cost of the EPOCHS=3 pair.
#   The -ep1 suffix is LOAD-BEARING: the dead -ep3 dirs still exist and a name
#   collision would overwrite them, destroying the evidence for the post-mortem
#   above.
#   TWO COSTS, STATE THEM.  (1) At 117M tokens these are NOT cross-referenceable
#   against Track A's 351M-token numbers — do not put them in the same column.
#   (2) A NULL IS AMBIGUOUS: "chain length does not help" and "117M tokens was
#   not enough to learn it" are indistinguishable at this budget.  A positive
#   result is conclusive; a null is not.  EPOCHS=2 (234M) is the compromise if
#   budget appears.
# The old EPOCHS=3 invocation is kept above only as the record of what the void
# runs did.  Do not submit it.
#
# WALLTIME.  36h, up from 16h.  The first mix submission (job 11702363,
# 2026-08-05) hit TIMEOUT at exactly 16:00:09 having reached update 3,311 of
# 6,103 — healthy the whole way (gnorm p99 0.887, clip 0.5%, no NaNs), just
# short of wallclock.  Measured rate 17.4 s/update on A100-80 => the mix needs
# ~29.5h and the EPOCHS=3 control ~25.9h.  16h was never enough for either.
# Peak-Mem there was 71.8 GiB of 79.25 — near the edge, so do not raise
# MAX_LENGTH or CROSS_CHUNKS on this script without moving to H200.
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

# --- resume, for the wallclock case ----------------------------------------
# These arms cost ~26-30h against a 36h limit, so a timeout is a live risk and
# restarting from 0 throws away real GPU-days (job 11702363 lost 3,311 updates
# that way).  save_interval=500 writes a weights-only dir every 500 and a full
# resumable checkpoint_<step>/chkpt.pt every 1,000, so the worst case is 1,000
# steps, not the whole run.  Newest checkpoint, NUMERICALLY (checkpoint_9500
# must not outrank checkpoint_150000):
#   step=$(ls -d cortex-retro-ft/$RUN_NAME/checkpoint_* | sed 's/.*checkpoint_//' \
#          | sort -n | tail -1)
#   RESUME_PATH=cortex-retro-ft/$RUN_NAME/checkpoint_$step sbatch pace/rung1_corpus_mix.sbatch
# Pass the SAME EPOCHS/DATA_PATH/RUN_NAME as the original link or the schedule
# and the arm identity move under you mid-chain.
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
