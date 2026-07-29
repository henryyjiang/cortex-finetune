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

# B1 — retrofit conversion with memory CO-TRAINED from step 0 (two-track plan
# §B1).  Same McLeish recipe as the B0 parity run, scaled from 1500 to 50000
# steps.  The memory buffer is the only new variable vs his conversion; the
# backbone is fully plastic from the untrained converted checkpoint.
#
# WHY THIS EXISTS: Track A (bolt-on finetuning of the already-converted RDM)
# is closed.  Its dials are measured and small — horizon null (tb2 0.0237 vs
# tb4 0.0235), write-diversity null (nemo cos 0.53 vs pg19 0.94, same delta),
# gated k=32 the only mover at 0.0283 +- 0.0010 (~3 sigma over append-k4).
# The question B1 asks is the one bolt-on cannot: does memory learned DURING
# the recurrence conversion clear the ceiling that memory grafted AFTER it
# could not?
#
# ARMS (decided 2026-07-29): AccumCCoT 4-vec/tb2 + the no-carry control.  The
# `off` control is NOT optional — it is the only way to attribute anything, and
# it runs the same chunked 1024-token windows so data exposure matches exactly.
#
# WHY ACCUM AND NOT GATED k=32 (this was reversed mid-decision, reasoning kept
# so it does not get re-litigated).  gated has the better Track-A carry delta
# (0.0283 +- 0.0010 vs acc4v's 0.0237 +- 0.0012, ~3 sigma) and it was the only
# dial in the whole round that moved.  Three things outweighed that:
#   1. DIAGNOSTICS.  evals/eval_carry_ablation.py --slice_ablate and
#      evals/diag_accum_buffer.py BOTH require cortex.accum (write-once rows)
#      and SystemExit on gated/overwritten states.  A gated B1 yields a single
#      carry-delta number and NOTHING else — no slice ablation, no buffer
#      health at depth.  For the flagship run, whose purpose is to EXPLAIN
#      whether co-training clears the bolt-on ceiling, that is flying blind.
#   2. NARRATIVE CONTINUITY.  Every mechanism result on the books is AccumCCoT
#      (the 0.024 headline, recency-dominance, the redundant-gist finding).  A
#      positive gated result would sit on a mechanism with no diagnostic
#      characterization and no bridge to the Track-A analysis.
#   3. PARITY COVERAGE.  accum ran B0's full 1110-step descent within ~0.007
#      nats of base; gated only reached update 284 before the quota kill.
# And the 3-sigma edge is only 19% relative, measured under frozen-loop bolt-on
# on PG-19-recall data, whereas B1 is plastic-backbone co-training on
# nemotron.  This round is itself the evidence that these rankings are fragile:
# the horizon prediction was pre-registered and failed, the write-diversity
# theory failed, and the one behavioral positive turned out to be training row
# length.  Trading the diagnostic toolkit for a 19% head start on a metric that
# needs +300% is a bad trade.  gated stays wired as MEMORY_MODE=gated and is
# the natural third arm IF accum clears the 20k gate.
#
#   bash pace/submit_retrofit_b1.sh start      # both arms, accum + off
# (use submit_retrofit_b1.sh — it handles the resume chain and the gates)
#
# H200, NOT A100-80GB.  Both B0 arms OOMed at step ~1110, exactly where the
# mean-recurrence 1-sqrt warmup (0.75 * 1500 = 1125) tops out at 8.  The
# `off` arm OOMed too, so this is NOT the carry graph — it is 4 x 1024 chunks
# chained into ONE backward at recurrence 8, i.e. 4x McLeish's per-1024-row
# activation graph.  Genuine capacity OOM (expandable_segments was on, only
# ~12 MiB reserved-unallocated, 78.4 GiB allocated).  B1 spends ~90% of its
# life at max recurrence, so it cannot run on 80GB.  141GB clears it with
# headroom and needs no code change; the alternatives (port cortex-main's
# --grad_checkpoint, or per-chunk backward with carry-only retention) are
# code changes on the critical path and were declined for this round.
#
# STAGED GATES — read before spending the full 50k.  Batch is 4 x 4096 =
# 16,384 tok/step, so:
#     10000 steps = 164M tokens
#     20000 steps = 328M tokens
#     50000 steps = 819M tokens
#
# READ THE RECURRENCE CONFOUND FIRST.  Both schedules key off max_steps
# (train.py:994-1029, max_training_steps = cfg.max_steps), so at MAX_STEPS
# 50000 the mean-recurrence 1-sqrt ramp runs for 0.75 * 50000 = 37500 steps.
# An intermediate checkpoint is therefore at PARTIAL recurrence, while every
# Track-A arm trained at a fixed recurrence 8.  Consequences:
#   * 10k / 20k gates are INTERNAL ONLY — gated-vs-off at the same point in
#     the same recipe.  That is the actual B1 question and it is valid at any
#     recurrence, because both arms sit at the same point on the same ramp.
#   * The CROSS-TRACK comparison to Track A's 0.024-0.0283 @ 350M is clean
#     only at step 50000 (recurrence 8, 819M tok).  Do not read a 20k carry
#     delta against 0.0283 and call co-training a failure — the training
#     recurrence does not match.  (The carry ablation's own eval recurrence is
#     fixed, so the eval is matched; it is the checkpoint that is not.)
# Gate rule: at 10k and 20k, require gated > off with the arms otherwise
# healthy (loss tracking, total_norm bounded, no divergence).  Abort only on
# gated <= off, which would say co-trained memory carries nothing — that
# conclusion does not need the remaining 30k steps.
#
# Do NOT shorten MAX_STEPS to reach full recurrence sooner, and do NOT vary it
# across the resume chain: it is the horizon for the LR schedule AND the
# recurrence ramp, so changing it mid-chain re-runs cooldown on every segment
# (the 2026-06-24 sawtooth artifact, baked into the weights).  Keeping
# warmup=0.75 at 50000 steps is what reproduces McLeish's recipe exactly,
# which is the entire premise of Track B.
#
# NO STABILIZERS (design principle 0.1.3): no throttle, no L2-SP, no spectral
# reg.  B0 needed none — base and accum tracked within ~0.007 nats over the
# whole 10.4 -> 1.06 descent with matched grad norms.  Watch train/total_norm;
# reintroduce one at a time only if instability shows.
#
# DISK: save_checkpoint never rotates.  save_interval writes a weights-only
# HF checkpoint, and every 2 * save_interval writes a full chkpt.pt (model +
# Muon/Adam optimizer + scheduler + dataloader), which for this model is
# ~15-20GB a piece.  At upstream's save_interval=500 a 50k-step run would
# leave 100 weight dirs + 50 full checkpoints ~= 1TB+ per arm.  Hence
# SAVE_INTERVAL=2500 here (20 weight dirs + 10 full ckpts ~= 240GB/arm) plus
# the pre-timeout save below, which is what actually protects the resume
# chain.  Everything lands on $SCRATCH via the symlink guard.
#
# Prep (login node, once — same as B0, skip if already present):
#   python tools/prepare_cortex_checkpoint.py \
#       --src smcleish/Recurrent-OLMo-2-0425-untrained \
#       --dst ckpts/olmo-retrofit-cortex --variant olmo
#   python tools/prepare_packed_dataset.py --tokenizer ckpts/olmo-retrofit-cortex \
#       --out data/nemotron_math_olmo_len4096 --max_length 4096 \
#       --max_tokens 400_000_000
# NOTE: 819M tokens > the 400M-token pack, so the 50k-step run wraps the data
# ~2x.  Repack larger (--max_tokens 900_000_000) if a single-epoch B1 is
# wanted; the gates at 10k/20k are inside the first epoch either way.
#
# Recipe provenance (shells/olmo.sh, unchanged from B0): lr 5e-5, muon lr
# 0.001, cooldown 0.6 / warmup 0.0025, max_grad_norm 1.0,
# mean_recurrence_schedule warmup 0.75 "1-sqrt", max_mean_rec 8.

cd $SLURM_SUBMIT_DIR

module load anaconda3
module load cuda/12.1.1
conda activate cortex-retro

export SCRATCH=${SCRATCH:-$HOME/scratch}
export HF_HOME=$SCRATCH/hf_cache
export HF_HUB_OFFLINE=1
export WANDB_DIR=$SCRATCH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# --- out_path MUST live on scratch -----------------------------------------
# train.py uses out_path as BOTH the wandb project name AND the write dir
# (wandb.init(project=cfg.out_path, dir=cfg.out_path) — WANDB_DIR is ignored),
# and a relative name resolves into $HOME.  That is what killed the 2026-07-17
# round: an update-500 checkpoint blew the home quota and took all three arms
# down (plus, an hour later, all three Track-A arms, whose Slurm logs also
# live in home).  Keep the relative name (preserves the wandb project) and
# symlink it to scratch.
OUT_DIR=cortex-retrofit
if [ ! -L "$OUT_DIR" ]; then
    if [ -e "$OUT_DIR" ]; then
        echo "ERROR: $OUT_DIR exists in \$HOME but is not a symlink to scratch."
        echo "  It is what killed the 2026-07-17 round.  Move or delete it"
        echo "  (rm -rf $OUT_DIR) and resubmit; this script will re-create it"
        echo "  as a symlink to \$SCRATCH/cortex-retrofit."
        exit 1
    fi
    mkdir -p "$SCRATCH/cortex-retrofit"
    ln -s "$SCRATCH/cortex-retrofit" "$OUT_DIR"
fi
# ---------------------------------------------------------------------------

MEMORY_MODE=${MEMORY_MODE:-accum}          # accum | off | gated
MAX_STEPS=${MAX_STEPS:-50000}
MAX_MEAN_REC=${MAX_MEAN_REC:-8}
DATA_PATH=${DATA_PATH:-data/nemotron_math_olmo_len4096}
MODEL=${MODEL:-ckpts/olmo-retrofit-cortex}
CROSS_CHUNKS=${CROSS_CHUNKS:-4}
MAX_LENGTH=${MAX_LENGTH:-4096}
SAVE_INTERVAL=${SAVE_INTERVAL:-2500}
RESUME_PATH=${RESUME_PATH:-}

case "$MEMORY_MODE" in
gated)
    MEMORY_SLOTS=32; ACCUM_CCOT=false; GATED_ACCUM=true;  CARRY_GRAD_CHUNKS=0
    ARM="k32-ga";;
accum)
    MEMORY_SLOTS=0;  ACCUM_CCOT=true;  GATED_ACCUM=false; CARRY_GRAD_CHUNKS=2
    ARM="acc4v-tb2";;
off)
    MEMORY_SLOTS=0;  ACCUM_CCOT=false; GATED_ACCUM=false; CARRY_GRAD_CHUNKS=0
    ARM="base";;
*)
    echo "MEMORY_MODE must be gated|off|accum"; exit 1;;
esac
RUN_NAME=${RUN_NAME:-retro-b1-${ARM}-mr${MAX_MEAN_REC}}

# Resume is opt-in via RESUME_PATH.  train.py restores model + optimizer +
# scheduler + mean-recurrence scheduler + dataloader position + RNG, so a
# chained run is continuous rather than a restart.  Leave ignore_past_* off:
# both the LR schedule and the recurrence ramp must stay on ONE horizon
# across the chain (this is the schedule-sawtooth footgun from 2026-06-24).
RESUME_ARGS=""
if [ -n "$RESUME_PATH" ]; then
    if [ ! -f "$RESUME_PATH/chkpt.pt" ]; then
        echo "ERROR: no chkpt.pt under RESUME_PATH=$RESUME_PATH"
        exit 1
    fi
    RESUME_ARGS="--resume_path $RESUME_PATH"
    echo "Resuming from $RESUME_PATH"
fi

echo "=== B1 $RUN_NAME | mode=$MEMORY_MODE | max_steps=$MAX_STEPS ==="
echo "    tokens at completion: $(( MAX_STEPS * 4 * MAX_LENGTH )) "
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python train.py \
    --run_name $RUN_NAME \
    --out_path cortex-retrofit \
    --model_name $MODEL \
    --preprocessed_data_path $DATA_PATH \
    --max_length $MAX_LENGTH \
    --epochs 1 \
    --max_steps $MAX_STEPS \
    --micro_batch_size 1 \
    --batch_size 4 \
    --optim_config.lr 5e-5 \
    --scheduler_args.cooldown 0.6 \
    --scheduler_args.warmup 0.0025 \
    --muon.use_muon true \
    --muon.lr 0.001 \
    --no_amp false \
    --max_grad_norm 1.0 \
    --compile false \
    --save_interval $SAVE_INTERVAL \
    --save_n_mins_before_timeout 20 \
    --seed 74 \
    --mean_recurrence_schedule.turn_on true \
    --mean_recurrence_schedule.warmup 0.75 \
    --mean_recurrence_schedule.max_mean_rec $MAX_MEAN_REC \
    --mean_recurrence_schedule.warmup_type "1-sqrt" \
    --cortex.use_memory true \
    --cortex.memory_lr 5e-4 \
    --cortex.memory_slots $MEMORY_SLOTS \
    --cortex.accum_ccot $ACCUM_CCOT \
    --cortex.gated_accum $GATED_ACCUM \
    --cortex.carry_grad_chunks $CARRY_GRAD_CHUNKS \
    --cortex.cross_chunks $CROSS_CHUNKS \
    --cortex.eos_from_tokens true \
    --cortex.freeze_loop false \
    $RESUME_ARGS
