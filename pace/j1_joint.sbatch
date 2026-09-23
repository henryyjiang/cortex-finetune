#!/bin/bash
#SBATCH --account=gts-aivanova7-lab
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H200:1
#SBATCH --mem=128GB
#SBATCH -t 8:00:00
#SBATCH -q inferno
#SBATCH -J j1_joint
#SBATCH -o logs/Report-%A_%a.out
#SBATCH --array=0-2
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=henryyjiang42@gmail.com

# J1, THE JOINT RUN -- findings doc, "Attempt 2", Step 2.  The first run where
# the loss reaches Z's WRITE in chunk n through Z's READ in chunk n+1.
#
#   ENCODING=<from Step 0> sbatch pace/j1_joint.sbatch     # [0] real [1] donor [2] noread
#
# THREE LIMBS, identical in everything but the read (same branch, seed, data
# order, schedule, micro-batch):
#   real    LatentRead xattn at read_into on this document's Z
#   donor   the same module on ANOTHER document's Z (batch roll, read side
#           only).  The donor's write still gets read pressure -- from the
#           wrong document -- which is the point (tests/test_j1_joint.py pins it)
#   noread  Z written every chunk at full width, read nowhere
#           (--cortex.latent_write_only true)
#
# THE RECIPE IS THE P2.3 CELL RECIPE (pace/p1_arms.sbatch), on purpose: same
# branch (heal checkpoint_91552_w16), same Muon/Adam LRs, same horizon and
# recurrence ramp, so the real limb is directly comparable to the E-only a3
# cell at the same step.  Everything is trainable: no --train_read_only, no
# freeze_loop.  The branch starts new parameters fresh and does not restore the
# optimizer; the LR schedule and step counter continue (train.py branch_path).
#
# ===== BEFORE THE FIRST LAUNCH ===============================================
#   1. ENCODING comes from Step 0 (pace/z2_diag_content.sbatch).  There is no
#      default on purpose: launching before D1/D2 would pick one silently.
#   2. PRICE THE MEMORY: `ENCODING=... REQUIRE_FIT=1 sbatch pace/j1_price_oom.sbatch`
#      (same geometry, same --window_backward path as this launcher).  MICRO_BS=2
#      (the donor roll needs two rows) doubles activations against the P2.3
#      cells' micro 1, and the read module adds 16.8M parameters with optimizer
#      state.  If it does not fit, the donor limb cannot run at micro 2 and the
#      control needs redesigning -- do NOT drop to micro 1, where the roll
#      returns the same row and the donor limb IS the real limb.
#   3. The data.  Default is the prose + synthetic mix (Step 1,
#      pace/prepare_carry_task.sbatch).  PROSE_ONLY=1 runs on data/pg19_fw50
#      alone, which tests the design but not the positive control.
#   4. Write J1's own pre-registration before reading any number off it.
#
# ===== DECISIONS RECORDED 2026-09-22 (night) =================================
# ENCODING = tokens.  Step 0 (job 13456334, eval_results/z2_content/{a2,a3z}),
#   the rule written before the run: the lowest-D1-ratio Z encoding that also
#   HOLDS CONTENT on D2 bow_next.  Vetoes clear on both arms (50 books by
#   document ends, reseed OK, probe health R^2(E->bow_self) 0.049 / 0.054).
#   D1: every encoding is content-determined (ratios <= 0.002); Z_tok is lowest
#   on both (0.000).  D2 bow_next: Z_tok is the ONLY encoding that holds content
#   beyond E on BOTH arms -- over donor +0.0040 [+0.0030,+0.0055] (a2) /
#   +0.0030 [+0.0022,+0.0042] (a3z), over E +0.0030 [+0.0020,+0.0036] /
#   +0.0018 [+0.0006,+0.0026].  Small: E alone is 0.022-0.025 on the same
#   target.  delta holds on a3z nll_next only; endpoint holds nowhere.
# MEMORY: at micro 2 BOTH limbs OOMed at 138.71 of 139.80 GiB (job 13456835),
#   the no-read limb too.  carry_grad_chunks cannot fix that -- every chunk's
#   loss holds its graph until one backward per row -- so WINDOW_BACKWARD=1
#   (default) backprops each 4-chunk carry window as it closes.  Exact (same
#   gradient to fp summation order, tests/test_window_backward.py), and the
#   windows are the ones carry_grad_chunks 4 already cut, so the recipe is
#   unchanged.  Priced by pace/j1_price_oom.sbatch on the same path.
#
# DATA PROVENANCE, stated so nobody rediscovers it: the heal parent trained on
# FineWeb-Edu, and pg19_fw50's FineWeb half is drawn from the same pack, so some
# prose rows here were in the parent's training.  Harmless for J1's decision
# rules, which are read on HELD-OUT packs (PG-19 validation, the synthetic
# validation pack) and paired across limbs -- not off training curves, where
# the update-400 confound lived.

cd $SLURM_SUBMIT_DIR

module load anaconda3
module load cuda/12.1.1
conda activate cortex-retro

export SCRATCH=${SCRATCH:-$HOME/scratch}
export HF_HOME=$SCRATCH/hf_cache
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODEL=${MODEL:-ckpts/olmo-retrofit-cortex}
BRANCH_PATH=${BRANCH_PATH:-cortex-retrofit/retro-b2-heal/checkpoint_91552_w16}
if [ -n "$PROSE_ONLY" ]; then
    ARM_DATA=${ARM_DATA:-data/pg19_fw50_olmo_len4096}
else
    ARM_DATA=${ARM_DATA:-data/j1_pg19fw50_carry20_len4096}
fi
MAX_LENGTH=${MAX_LENGTH:-4096}
CROSS_CHUNKS=${CROSS_CHUNKS:-8}
ACCUM_VECS=${ACCUM_VECS:-16}
GATE_SLOTS=${GATE_SLOTS:-64}
CARRY_GRAD_CHUNKS=${CARRY_GRAD_CHUNKS:-4}  # 4 of 8, as the P2.3 cells
MAX_MEAN_REC=${MAX_MEAN_REC:-8}
RAMP_WARMUP=${RAMP_WARMUP:-0.25}
MAX_STEPS=${MAX_STEPS:-305176}             # the LR SCHEDULE horizon (P2.3's)
PARENT_STEP=${PARENT_STEP:-91552}
CELL_STEPS=${CELL_STEPS:-2000}             # >= 1000 per the plan; ~33M tokens
STOP_AT_STEP=$(( PARENT_STEP + CELL_STEPS ))
SAVE_INTERVAL=${SAVE_INTERVAL:-1000}
DIAG_INTERVAL=${DIAG_INTERVAL:-25}
MICRO_BS=${MICRO_BS:-2}
BATCH_SIZE=${BATCH_SIZE:-4}
READ_HEADS=${READ_HEADS:-8}
ZNORM=${ZNORM:-rms}                        # encodings differ in content, not scale
ZNORM_TARGET=${ZNORM_TARGET:-3.0}          # the measured delta row norm, P3.0
TOK_POOL=${TOK_POOL:-4}
E_DROPOUT=${E_DROPOUT:-0.0}
WINDOW_BACKWARD=${WINDOW_BACKWARD:-1}      # micro 2 does not fit without it (13456835)
WB_ARGS=""
if [ "$WINDOW_BACKWARD" = "1" ]; then WB_ARGS="--cortex.window_backward true"; fi

if [ -z "$ENCODING" ]; then
    echo "ERROR: ENCODING is unset.  Step 0 (pace/z2_diag_content.sbatch) picks"
    echo "       it: delta | endpoint | tokens.  No default, on purpose."
    exit 1
fi
case "$ENCODING" in delta|endpoint|tokens) ;; *)
    echo "ERROR: ENCODING must be delta|endpoint|tokens; got '$ENCODING'"; exit 1;;
esac
if [ "$MICRO_BS" -lt 2 ]; then
    echo "ERROR: MICRO_BS=$MICRO_BS.  The donor limb rolls Z along the micro-"
    echo "       batch; at 1 the roll returns the same row and the donor limb"
    echo "       IS the real limb.  All three limbs share MICRO_BS so the"
    echo "       recipes stay identical."
    exit 1
fi
[ -f "$BRANCH_PATH/chkpt.pt" ] || { echo "ERROR: no chkpt.pt under BRANCH_PATH=$BRANCH_PATH"; exit 1; }
[ -d "$ARM_DATA" ] || { echo "ERROR: no such pack: $ARM_DATA (build it: pace/prepare_carry_task.sbatch, or PROSE_ONLY=1)"; exit 1; }
if ! grep -q "read_into(x, current_step)" "$MODEL/raven_modeling_minimal_cortex.py" 2>/dev/null; then
    echo "ERROR: $MODEL carries a PRE-P3.0 modeling file.  Refresh it:"
    echo "         python tools/prepare_cortex_checkpoint.py --src $MODEL --dst $MODEL --variant olmo"
    echo "         rm -rf \$SCRATCH/hf_cache/modules/transformers_modules/$(basename $MODEL)"
    exit 1
fi

LIMBS=(real donor noread)
IDX=${SLURM_ARRAY_TASK_ID:-0}
LIMB=${LIMBS[$IDX]}
if [ -z "$LIMB" ]; then echo "no limb at index $IDX (use --array=0-2)"; exit 0; fi

case "$LIMB" in
    real)   READ_ARGS="--cortex.latent_read xattn --cortex.latent_read_heads $READ_HEADS"
            READ_ARGS="$READ_ARGS --cortex.latent_read_scramble false"
            READ_ARGS="$READ_ARGS --cortex.latent_read_znorm $ZNORM --cortex.latent_read_znorm_target $ZNORM_TARGET" ;;
    donor)  READ_ARGS="--cortex.latent_read xattn --cortex.latent_read_heads $READ_HEADS"
            READ_ARGS="$READ_ARGS --cortex.latent_read_scramble true"
            READ_ARGS="$READ_ARGS --cortex.latent_read_znorm $ZNORM --cortex.latent_read_znorm_target $ZNORM_TARGET" ;;
    noread) READ_ARGS="--cortex.latent_read none --cortex.latent_write_only true" ;;
esac

TAG=""
if [ "$E_DROPOUT" != "0.0" ] && [ "$E_DROPOUT" != "0" ]; then TAG="-edrop${E_DROPOUT}"; fi
if [ -n "$PROSE_ONLY" ]; then TAG="$TAG-prose"; fi
# RUN_SUFFIX keeps a smoke (e.g. RUN_SUFFIX=-smoke) out of the real run's dir.
RUN_NAME=j1-a3z-${ENCODING}${TAG}-${LIMB}${RUN_SUFFIX:-}
ACCUM_MAX=$(( ACCUM_VECS * CROSS_CHUNKS ))

echo "=== J1 joint run | limb=$LIMB | encoding=$ENCODING | run=$RUN_NAME ==="
echo "    branch:    $BRANCH_PATH  (step $PARENT_STEP -> stop $STOP_AT_STEP, horizon $MAX_STEPS)"
echo "    data:      $ARM_DATA   (READ THIS LINE -- the only proof the pack arrived)"
echo "    geometry:  gated W=$ACCUM_VECS K=$GATE_SLOTS cc=$CROSS_CHUNKS grad_chunks=$CARRY_GRAD_CHUNKS mr=$MAX_MEAN_REC window_backward=$WINDOW_BACKWARD"
echo "    read:      $READ_ARGS"
echo "    e_dropout: $E_DROPOUT   micro=$MICRO_BS batch=$BATCH_SIZE   diag every $DIAG_INTERVAL"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

python train.py \
    --run_name $RUN_NAME \
    --out_path cortex-retrofit \
    --model_name $MODEL \
    --preprocessed_data_path $ARM_DATA \
    --max_length $MAX_LENGTH \
    --epochs 1 \
    --max_steps $MAX_STEPS \
    --stop_at_step $STOP_AT_STEP \
    --micro_batch_size $MICRO_BS \
    --batch_size $BATCH_SIZE \
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
    --mean_recurrence_schedule.warmup $RAMP_WARMUP \
    --mean_recurrence_schedule.max_mean_rec $MAX_MEAN_REC \
    --mean_recurrence_schedule.warmup_type "1-sqrt" \
    --cortex.use_memory true \
    --cortex.memory_lr 5e-4 \
    --cortex.memory_slots 0 \
    --cortex.prefix_memory gated \
    --cortex.accum_vecs $ACCUM_VECS \
    --cortex.accum_max $ACCUM_MAX \
    --cortex.gate_slots $GATE_SLOTS \
    --cortex.gate_route ring \
    --cortex.gate_norm tanh \
    --cortex.gate_init zero \
    --cortex.gate_fill grow \
    --cortex.latent_carry true \
    --cortex.latent_depth_rule absolute \
    --cortex.latent_depth_lo 2 \
    --cortex.latent_depth_hi 9 \
    --cortex.latent_renorm none \
    --cortex.latent_encoding $ENCODING \
    --cortex.latent_tok_pool $TOK_POOL \
    --cortex.latent_s0_read false \
    $READ_ARGS \
    --cortex.e_dropout $E_DROPOUT \
    --cortex.diag_interval $DIAG_INTERVAL \
    --cortex.carry_grad_chunks $CARRY_GRAD_CHUNKS \
    $WB_ARGS \
    --cortex.cross_chunks $CROSS_CHUNKS \
    --cortex.eos_from_tokens true \
    --cortex.prefix_eos_reset false \
    --cortex.prefix_pos tail \
    --cortex.freeze_loop false \
    --branch_path $BRANCH_PATH
RC=$?

echo ""
if [ $RC -eq 0 ]; then
    echo "J1 $LIMB ($ENCODING) DONE -> cortex-retrofit/$RUN_NAME"
    echo "  In-flight record: cortex-retrofit/$RUN_NAME/cortex_diag.jsonl"
    echo "  (read_gate per limb, write_grad_frac, e_dropped, E rank)."
    echo "  Decision rules are read on HELD-OUT packs, all three limbs paired:"
    echo "    carry task:  eval_carry_2x2 --score carry on the synthetic val pack"
    echo "    prose:       eval_carry_2x2 on data/pg19_olmo_validation_len4096_strided"
else
    echo "J1 $LIMB ($ENCODING) FAILED rc=$RC"
fi
exit $RC
