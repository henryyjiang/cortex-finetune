#!/bin/bash
#SBATCH --account=gts-aivanova7-lab
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=64GB
#SBATCH --gres=gpu:H200:1
#SBATCH --constraint=H200
#SBATCH -t 48:00:00
#SBATCH -q inferno
#SBATCH -o logs/Report-%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=henryyjiang42@gmail.com

# C-CHUNKED — the budget-matched no-memory control at 5B tokens.
# cortex_next_phase_framework.md §1 and §10.  Sibling of b2_retrofit.sbatch.
#
# THE CLAIM THIS RUN EXISTS TO SUPPORT.  paper/ currently states the method
# claim is unsupported because there is no budget-matched no-memory arm at 5B.
# This is that arm.  Every "memory helps by X" number in the paper is measured
# against this run, so the one thing that must be true of it is that it differs
# from the memory model in EXACTLY ONE VARIABLE.
#
# WHY `--cortex.use_memory false` ALONE WAS NOT A CONTROL
# ------------------------------------------------------
# train.py's fwd/bwd dispatch used to read
#     elif cfg.cortex["use_memory"] and int(cfg.cortex["cross_chunks"]) > 1:
# so turning memory off silently turned CHUNKING off too: this run would have
# trained on full 4,096-token windows against the memory model's 8x512 chunks.
# Chunk length is the biggest lever the ceiling probes ever found (nc=8 reaches
# +0.2277 at k512 and is still rising; nc=4 peaks at +0.1036 and then
# COLLAPSES).  A control that differs in both memory and chunk geometry does not
# isolate memory; it is a second experiment.  The dispatch now keys on
# cross_chunks alone, and this script is the only consumer of that change.
#
# So: chunked, no carry, no summary slots, no memory parameters, cross_chunks
# matched to the final model.  Chunks are independent forwards, losses reduced
# token-weighted, one backward.
#
# WHAT IT BUYS BEYOND "A BASELINE"
#   1. The paper's missing control.
#   2. Memory-vs-depth at ZERO extra training cost: T is an inference knob and
#      evals/flops.py is the cost model, so sweeping T on this run against T on
#      the memory model gives accuracy vs analytic FLOPs.  (Caveat to state:
#      depth saturates at T=8 on both mr8 and mr32, so that claim currently
#      rests on a degenerate depth arm.)
#   3. The first carry-off cell that is IN-DISTRIBUTION.  Every carry-off number
#      on record today is the memory model's weights with the buffer yanked out,
#      which is OOD for those weights.  This is the honest zero.
#
# GEOMETRY — D-FALLBACK, locked 2026-09-14 by the OOM gate
# --------------------------------------------------------
# pace/smoke_geometry_oom.sbatch, jobs 13201945 / 13202177 / 13202515 on an
# H200 (138.7 GiB).  Config D (8,192 x cc16) was the framework's locked choice
# on the chain-depth argument and it DOES NOT FIT — nor does 4,096 x cc16 with
# 256-token chunks, at any buffer cap tested.  Measured:
#
#   4096  x cc8  amax 256   FITS  115.44 GiB (83%)  ~1.3 s/micro-step   <- this
#   4096  x cc8  amax 256   FITS  106.25 GiB (76%)  ~1.2 s/micro-step   <- control
#   4096  x cc16 amax 512   OOM
#   4096  x cc16 amax 128   OOM
#   8192  x cc16 amax 512   OOM
#   8192  x cc16 amax 128   OOM
#   16384 x cc16 amax 512   OOM
#
# Calibration: B2 ran 4096 x cc8 for real and peaked at 119.76 GiB against the
# gate's 115.44, so the gate is trustworthy to ~4%.
#
# THE BATCH, AND WHY THE LR IS NOT RESCALED
# -----------------------------------------
# McLeish's retrofit recipe ran at ~1.05M tokens/step (batch_size is PER-RANK;
# 16 x 64 ranks x 1024).  B2 inherited every hyperparameter from it at 16,384
# tokens/step — a 64x smaller global batch — and compensated with 6.1x more
# optimizer steps.  This run raises batch_size to 32 instead: micro_batch_size
# is 1, so batch_size is PURE GRADIENT ACCUMULATION and costs no extra memory
# and no extra wall time (max_steps x batch_size x chunks_per_row is invariant
# at a fixed token budget).  That puts us 8x below the reference rather than
# 64x, at 38,147 optimizer steps against its 50,000.
#
# The quantity that transfers across batch sizes is neither lr nor lr x tokens
# but `lr x optimizer_steps`, the total parameter displacement
# (recipe_sweep_findings.md §11.1): McLeish 2.50 aux / 50.0 Muon, B2 6.1x both,
# this run 0.76x both.  So the batch fix already fixes Muon, and rescaling the
# LR on top would apply the correction twice — sqrt-scaling would land at 0.27x
# the reference's displacement and linear at ~0.1x, badly under-training the
# conversion this run has to accomplish.  KEEP 5e-5 AND 1e-3.  Say in the
# writeup that the batch was raised to close the gap to the recipe the LR came
# from, rather than silently re-tuning the LR, and disclose the residual 8x.
#
# ONE CONTINUOUS RUN, NO BRANCH.  Unlike B2 there is no heal/arm split in the
# weights — the corpus switches at step 11,444 and nothing else does.  That is
# why the corpus-switch link MUST pass EXTRA_ARGS="--reset_dataset_position
# true": data_start_step is the cumulative dataloader-item count since the last
# BRANCH, and there is no branch here, so without the flag the loader
# fast-forwards 11,444 x 32 = 366,208 rows into the mix pack before serving
# anything and the run exhausts at step ~29,375 of 38,147, exiting clean with
# wandb reporting "finished" at 3.85B tokens.  Identical to the B1 incident.
# VERIFY on the switch link's first log line: `[data] fast-forward:` must be
# ABSENT there and present on every later link.
#
# USAGE
#   PHASE=heal sbatch pace/control_retrofit.sbatch
#   PHASE=heal RESUME_PATH=cortex-control/c-chunked/checkpoint_XXXX \
#       sbatch pace/control_retrofit.sbatch
#   # the corpus switch, ONCE, at step 11444:
#   PHASE=mix RESUME_PATH=... EXTRA_ARGS="--reset_dataset_position true" \
#       sbatch pace/control_retrofit.sbatch
#   # every later mix link: same line WITHOUT EXTRA_ARGS
#
# PRE-FLIGHT (do these before the first submit)
#   python tools/check_pack.py --data data/fineweb_edu_olmo_len4096 \
#       --steps 11444 --batch_size 32 --max_length 4096
#   python tools/check_pack.py --data data/pg19_fw50_olmo_len4096 \
#       --steps 26703 --batch_size 32 --max_length 4096
#   grep -m1 '^=== C-chunked' logs/Report-<job>.out    # on EVERY link
#
# Row budgets are DOUBLE the framework's §10 numbers, because §10 computed them
# at batch_size 16 and D-fallback runs 32: heal 11,444 x 32 = 366,208 rows,
# mix 26,703 x 32 = 854,496 (940k with the 1.10 margin).  The token counts are
# unchanged — the rows are half as long.  B2's heal consumed 91,552 x 4 =
# 366,208 rows from this same FineWeb-Edu pack, so the heal leg is known-sized.
#
# CORPUS AS BUILT, 2026-09-14 (all four gates verified before launch)
# -------------------------------------------------------------------
#   heal  data/fineweb_edu_olmo_len4096      488,281 rows / 2.000B  -> 1.33x
#   mix   data/pg19_fw50_olmo_len4096        940,000 rows / 3.850B  -> 1.10x
#         = 470,000 PG-19 + 470,000 FineWeb-Edu
#   PG-19 pack: data/pg19_olmo_len4096_strided, 702,963 rows / 2.88B tokens,
#         24.6 rows per document over 28,602 books, 97.2% full windows.
#
# Epochs: PG-19 0.67 (470,000 of 702,963).  FineWeb-Edu 1.71 (366,208 heal +
# 470,000 mix over 488,281) — a real repeat, disclose it.
#
# THE MIX IS 50/50 BY ROWS AND 49.5/50.5 BY SUPERVISED TOKENS.  PG-19's
# real-token fill is 0.979 against FineWeb-Edu's 1.000, because striding leaves
# one ragged window per book (1 in 24.6 rows).  Quote the TOKEN share.
#
# Ratio chosen at 50/50 and not epoch-balanced on purpose: equalising epochs
# would need ~771k PG-19 rows against the 703k that exist, and the nearest
# feasible version is a 75/25 PG-19-heavy mix that no result supports.  50/50 is
# the only ratio in this project with a measured benchmark behind it — the rung1
# pg19_nemo50 arm, the best arm-benchmark row on record.
#
# The PG-19 leg exists at all because tools/prepare_pg19_dataset.py truncated
# every book to its first window until 2026-09-14: the old pack was 28,602 rows
# / 117M tokens, i.e. 4.1% of the corpus, and every "PG-19 cannot supply the
# volume" decision was downstream of that number.  Both packs are kept on disk;
# data/pg19_olmo_len4096 WITHOUT the _strided suffix is the old one.

cd $SLURM_SUBMIT_DIR

module load anaconda3
module load cuda/12.1.1
conda activate cortex-retro

export SCRATCH=${SCRATCH:-$HOME/scratch}
export HF_HOME=$SCRATCH/hf_cache
export HF_HUB_OFFLINE=1
export WANDB_DIR=$SCRATCH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# See b2_retrofit.sbatch for why the port is probed here rather than left to
# train.py's SLURM_JOB_ID derivation (job 11702360 died in 29s on EADDRINUSE).
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=$(python -c "import socket;s=socket.socket();s.bind(('',0));print(s.getsockname()[1]);s.close()")
echo "MASTER_PORT=$MASTER_PORT"

# --- out_path MUST live on scratch -----------------------------------------
# train.py does wandb.init(project=cfg.out_path, dir=cfg.out_path) and IGNORES
# WANDB_DIR, so a relative name resolves into $HOME.  That is what killed the
# 2026-07-17 round: one checkpoint blew the home quota and took down six
# unrelated jobs, whose Slurm logs also live in home.  Keep the relative name
# (it is the wandb project) and symlink it to scratch.
OUT_DIR=cortex-control
if [ ! -L "$OUT_DIR" ]; then
    if [ -e "$OUT_DIR" ]; then
        echo "ERROR: $OUT_DIR exists in \$HOME but is not a symlink to scratch."
        echo "  Move or delete it (rm -rf $OUT_DIR) and resubmit; this script"
        echo "  will re-create it as a symlink to \$SCRATCH/cortex-control."
        exit 1
    fi
    mkdir -p "$SCRATCH/cortex-control"
    ln -s "$SCRATCH/cortex-control" "$OUT_DIR"
fi
# ---------------------------------------------------------------------------

PHASE=${PHASE:-heal}                       # heal | mix
MAX_STEPS=${MAX_STEPS:-38147}              # 5.000B / 131,072 — FIXED every link
STOP_AT_STEP=${STOP_AT_STEP:-0}            # end a phase early, horizon unchanged
MAX_MEAN_REC=${MAX_MEAN_REC:-8}
RAMP_WARMUP=${RAMP_WARMUP:-0.25}           # full recurrence at step 9,537
MODEL=${MODEL:-ckpts/olmo-retrofit-cortex}
MAX_LENGTH=${MAX_LENGTH:-4096}
CROSS_CHUNKS=${CROSS_CHUNKS:-8}            # 512-token chunks; MATCHES the memory model
BATCH_SIZE=${BATCH_SIZE:-32}               # pure gradient accumulation — free
SAVE_INTERVAL=${SAVE_INTERVAL:-300}        # resumable checkpoint every 600
# LR warmup in STEPS, not as a fraction.  The inherited warmup=0.0025 gave 125
# steps at McLeish's 50,000-step horizon and 764 at B2's 305,176 — but only 96
# here, because the fraction silently tracks a horizon that moved 8x.  Warmup is
# the one schedule term that should NOT scale with the horizon: it exists to let
# Adam's second moment mature and to survive the conversion transient, and both
# are counted in steps.  (OLMo-2 itself warmed up for 4,000.)
WARMUP_STEPS=${WARMUP_STEPS:-96}
RESUME_PATH=${RESUME_PATH:-}
# ONLY on the link that switches corpus — see the header.  Every later link is
# same-corpus and MUST restore its position, or each one replays from the top.
EXTRA_ARGS=${EXTRA_ARGS:-}

HEAL_DATA=${HEAL_DATA:-data/fineweb_edu_olmo_len4096}
MIX_DATA=${MIX_DATA:-data/pg19_fw50_olmo_len4096}

case "$PHASE" in
heal)
    # 30% of the horizon, exactly B2's fraction.  The recurrence ramp reaches
    # full recurrence at 9,537, INSIDE the heal — the property that keeps the
    # comparison clean, since both runs are at fixed T before the corpus moves.
    DATA_PATH=${DATA_PATH:-$HEAL_DATA}
    RUN_NAME=${RUN_NAME:-c-chunked};;
mix)
    DATA_PATH=${DATA_PATH:-$MIX_DATA}
    RUN_NAME=${RUN_NAME:-c-chunked};;
*)
    echo "PHASE must be heal|mix"; exit 1;;
esac

if [ $(( MAX_LENGTH % CROSS_CHUNKS )) -ne 0 ]; then
    echo "ERROR: MAX_LENGTH ($MAX_LENGTH) is not divisible by CROSS_CHUNKS" \
         "($CROSS_CHUNKS) — chunks would be ragged."; exit 1
fi
if [ -n "$RESUME_PATH" ] && [ ! -f "$RESUME_PATH/chkpt.pt" ]; then
    echo "ERROR: no chkpt.pt under RESUME_PATH=$RESUME_PATH"; exit 1
fi
CHAIN_ARGS=""
if [ -n "$RESUME_PATH" ]; then
    CHAIN_ARGS="--resume_path $RESUME_PATH"
    echo "Resuming from $RESUME_PATH"
fi
# There is deliberately no BRANCH_PATH here: this is one continuous run, and a
# branch would reset the cumulative row cursor that --reset_dataset_position
# exists to manage explicitly.

echo "=== C-chunked $RUN_NAME | phase=$PHASE | data=$DATA_PATH | NO MEMORY ==="
echo "    max_steps=$MAX_STEPS (horizon)  stop_at_step=$STOP_AT_STEP"
echo "    geometry: max_length=$MAX_LENGTH cross_chunks=$CROSS_CHUNKS" \
     "($(( MAX_LENGTH / CROSS_CHUNKS ))-tok chunks) batch_size=$BATCH_SIZE" \
     "-> $(( BATCH_SIZE * MAX_LENGTH )) tok/step"
echo "    tokens at horizon: $(( MAX_STEPS * BATCH_SIZE * MAX_LENGTH ))"
echo "    full recurrence $MAX_MEAN_REC at step $(python -c "import math;print(math.ceil($RAMP_WARMUP*$MAX_STEPS))")"
echo "    warmup_steps=$WARMUP_STEPS  extra_args='$EXTRA_ARGS'"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# --cortex.use_memory false is the ONLY mechanism flag set, and train.py's
# step-0 negative guard raises if the graft built cortex anyway.  No
# prefix_memory, no accum_vecs, no memory_lr: those would be dead config here,
# and dead config in a checkpoint's config.json is how a later eval script
# silently reconstructs a mechanism that never trained.
python train.py \
    --run_name $RUN_NAME \
    --out_path cortex-control \
    --model_name $MODEL \
    --preprocessed_data_path $DATA_PATH \
    --max_length $MAX_LENGTH \
    --epochs 1 \
    --max_steps $MAX_STEPS \
    --stop_at_step $STOP_AT_STEP \
    --micro_batch_size 1 \
    --batch_size $BATCH_SIZE \
    --optim_config.lr 5e-5 \
    --scheduler_args.cooldown 0.6 \
    --scheduler_args.warmup_steps $WARMUP_STEPS \
    --muon.use_muon true \
    --muon.lr 0.001 \
    --no_amp false \
    --max_grad_norm 1.0 \
    --compile false \
    --save_interval $SAVE_INTERVAL \
    --save_n_mins_before_timeout 20 \
    --seed 74 \
    --mean_recurrence_schedule.turn_on true \
    --mean_recurrence_schedule.warmup $RAMP_WARMUP \
    --mean_recurrence_schedule.max_mean_rec $MAX_MEAN_REC \
    --mean_recurrence_schedule.warmup_type "1-sqrt" \
    --cortex.use_memory false \
    --cortex.cross_chunks $CROSS_CHUNKS \
    --cortex.eos_from_tokens true \
    $CHAIN_ARGS $EXTRA_ARGS
