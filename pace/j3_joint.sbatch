#!/bin/bash
#SBATCH --account=gts-aivanova7-lab
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H200:1
#SBATCH --mem=128GB
#SBATCH -t 8:00:00
#SBATCH -q inferno
#SBATCH -J j3_joint
#SBATCH -o logs/Report-%A_%a.out
#SBATCH --array=0-2
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=henryyjiang42@gmail.com

# J3, THE IN-LOOP SCRATCHPAD -- findings doc, "Attempt 2", design J3, ranked
# first as a final design; j1_prereg.md S7/S10 make it next after J1's NO.
# The design is cortex_memory/scratchpad.py's docstring; in one line: a
# per-position gated latent state written at every loop iteration and read at
# the next, beside the carried Z rows in one softmax, with the scratchpad at
# the chunk's last tokens CARRIED as Z.
#
#   sbatch pace/j3_joint.sbatch                                  # [0] real [1] donor [2] noread
#   E_DROPOUT=0.25 sbatch --array=0,2 pace/j3_joint.sbatch       # the E-dropout pair
#
# THREE LIMBS, identical in everything but the CARRIED rows' read (same branch,
# seed, data order, schedule, micro-batch -- and the same within-chunk
# scratchpad read in all three):
#   real    reads this document's carried Z
#   donor   reads ANOTHER document's carried Z (batch roll, carried rows only;
#           the own scratchpad row is never rolled)
#   noread  carried Z written every chunk, never read
#           (--cortex.latent_carry_read false); the scratchpad read stays
#
# THE RECIPE IS J1's (pace/j1_joint.sbatch), which is the P2.3 cell recipe:
# same branch, LRs, horizon, recurrence ramp, geometry, window_backward, data.
# J3 differs from J1's real limb in THREE things, all in the Z channel:
#   1. the WRITE: the scratchpad m_T, not the loop state s_T (same positions,
#      same pooling -- latent_encoding scratch vs tokens);
#   2. the READ gains one key per position, its own scratchpad row, so the
#      write has a within-chunk consumer on every loop step;
#   3. the read gate's LR is x GATE_LR_MULT (below).
# The three limbs isolate the carry WITHIN J3; they do not isolate these
# three from each other, and J3's pre-registration must say so.
#
# ===== BEFORE THE FIRST LAUNCH ===============================================
#   1. PRICE THE MEMORY: `REQUIRE_FIT=1 sbatch pace/j3_price_oom.sbatch`.  The
#      scratchpad adds one [B, S, D] state per loop iteration to the graph and
#      16.8M gate parameters with Adam state.  J1 peaked at 88.6 GiB; expect a
#      few GiB more.  Do NOT drop to micro 1 (the donor roll needs 2 rows).
#   2. A 10-step smoke: `CELL_STEPS=10 RUN_SUFFIX=-smoke sbatch -t 0:45:00
#      pace/j3_joint.sbatch` -- read Time/step and a FINITE loss; delete the
#      -smoke run dirs after (~17 GB each).
#   3. J3's PRE-REGISTRATION is ../j3_prereg.md (repo root's parent, beside
#      j1_prereg.md), written 2026-09-23 before any J3 run.  Its rules are
#      evals/score_j1.py --experiment j3, including the read-strength audit
#      (READ_COLLAPSE_FRAC on z_read_ratio: with the gate slowed, a collapse
#      can move to out_proj / v_proj).  Run the E-dropout pair: under J1's
#      answer table it is the only thing that can return NO.
#
# ===== DECISIONS RECORDED 2026-09-23 =========================================
# GATE_LR_MULT = 0.1.  J1's read gate fell 0.10 -> 0.00 in ~270 updates on
#   every read limb, LINEARLY, at ~3.9e-4/update = Adam at memory_lr 5e-4
#   moving a sign-stable scalar at full speed (j1_prereg.md S10).  The speed
#   was the optimizer's, and it shut the write's gradient (which scales with
#   the gate) before the write learned.  THE RULE for the value: the gate's
#   FASTEST possible collapse should take about the whole run.  At J1's
#   measured rate the gate moves at most mult x 0.78 over 2,000 updates:
#     1.0   0.78    closes in ~260 updates (J1)
#     0.1   0.078   cannot close before the end (>= ~0.02 left at J1's push),
#                   and can still roughly double if the read turns useful
#     0.01  0.0078  frozen in all but name: no longer a two-sided signal, and
#                   it caps the read (the gate cannot pass ~0.108), so a Z
#                   that became useful would have to open the read through
#                   out_proj instead
#   0.01 was the first draft; 0.1 is the smallest slowdown that meets the
#   rule while keeping the gate readable.  It is a REPARAMETERISATION
#   (LatentRead.gate_lr_mult), not a param group, because a new group breaks
#   the branch's LambdaLR restore.  GATE_LR_MULT=1.0 gives J1's gate exactly.
# E_DROPOUT for the pair = 0.25, J1's value: inside the findings doc's 20-30%
#   range and directly comparable to J1's pair (j3_prereg.md S1).
# ENCODING = scratch, fixed: J3 carries the scratchpad at J1's Step-0 winner's
#   positions (the last 64 real tokens pooled x4 to 16 rows).
# MEMORY: job 13483998 (pace/j3_price_oom.sbatch, micro 2, window_backward):
#   real 92.12 alloc / 92.42 reserved GiB (66%), 1.7 s/micro; noread 91.20 /
#   91.69 (66%), 1.6 s.  FITS.  +3.6 GiB and ~+5% time over J1's 88.48 / 88.89
#   GiB and 1.62 s (job 13459697) -- the scratchpad is in the priced graph.
#   J1 trained ~4% slower than it priced, so expect ~1.77 s/micro, ~2.0 h/limb.

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
CARRY_GRAD_CHUNKS=${CARRY_GRAD_CHUNKS:-4}  # 4 of 8, as J1 and the P2.3 cells
MAX_MEAN_REC=${MAX_MEAN_REC:-8}
RAMP_WARMUP=${RAMP_WARMUP:-0.25}
MAX_STEPS=${MAX_STEPS:-305176}             # the LR SCHEDULE horizon (P2.3's)
PARENT_STEP=${PARENT_STEP:-91552}
CELL_STEPS=${CELL_STEPS:-2000}             # J1's budget, so the two compare
STOP_AT_STEP=$(( PARENT_STEP + CELL_STEPS ))
SAVE_INTERVAL=${SAVE_INTERVAL:-1000}
DIAG_INTERVAL=${DIAG_INTERVAL:-25}
MICRO_BS=${MICRO_BS:-2}
BATCH_SIZE=${BATCH_SIZE:-4}
READ_HEADS=${READ_HEADS:-8}
ZNORM=${ZNORM:-rms}                        # carry AND own rows at one row norm
ZNORM_TARGET=${ZNORM_TARGET:-3.0}
TOK_POOL=${TOK_POOL:-4}
GATE_LR_MULT=${GATE_LR_MULT:-0.1}          # see DECISIONS above
SCRATCH_FB=${SCRATCH_FB:-1.0}              # per-iteration fg 0.731 at init (LM2)
E_DROPOUT=${E_DROPOUT:-0.0}
WINDOW_BACKWARD=${WINDOW_BACKWARD:-1}      # micro 2 does not fit without it (13456835)
WB_ARGS=""
if [ "$WINDOW_BACKWARD" = "1" ]; then WB_ARGS="--cortex.window_backward true"; fi
ENCODING=scratch

if [ "$MICRO_BS" -lt 2 ]; then
    echo "ERROR: MICRO_BS=$MICRO_BS.  The donor limb rolls the carried Z along"
    echo "       the micro-batch; at 1 the roll returns the same row and the"
    echo "       donor limb IS the real limb.  All three limbs share MICRO_BS."
    exit 1
fi
[ -f "$BRANCH_PATH/chkpt.pt" ] || { echo "ERROR: no chkpt.pt under BRANCH_PATH=$BRANCH_PATH"; exit 1; }
[ -d "$ARM_DATA" ] || { echo "ERROR: no such pack: $ARM_DATA (build it: pace/prepare_carry_task.sbatch, or PROSE_ONLY=1)"; exit 1; }
if ! grep -q "read_into(x, current_step)" "$MODEL/raven_modeling_minimal_cortex.py" 2>/dev/null \
   || ! grep -q "iter_write(x, current_step)" "$MODEL/raven_modeling_minimal_cortex.py" 2>/dev/null; then
    echo "ERROR: $MODEL carries a modeling file without the in-loop hooks J3"
    echo "       uses (read_into / iter_write with current_step).  Refresh it:"
    echo "         python tools/prepare_cortex_checkpoint.py --src $MODEL --dst $MODEL --variant olmo"
    echo "         rm -rf \$SCRATCH/hf_cache/modules/transformers_modules/$(basename $MODEL)"
    exit 1
fi

LIMBS=(real donor noread)
IDX=${SLURM_ARRAY_TASK_ID:-0}
LIMB=${LIMBS[$IDX]}
if [ -z "$LIMB" ]; then echo "no limb at index $IDX (use --array=0-2)"; exit 0; fi

# Every limb reads the scratchpad; the limbs differ in the CARRIED rows alone.
READ_ARGS="--cortex.latent_read scratch --cortex.latent_read_heads $READ_HEADS"
READ_ARGS="$READ_ARGS --cortex.latent_read_znorm $ZNORM --cortex.latent_read_znorm_target $ZNORM_TARGET"
READ_ARGS="$READ_ARGS --cortex.latent_read_gate_lr_mult $GATE_LR_MULT"
READ_ARGS="$READ_ARGS --cortex.latent_scratch_forget_bias $SCRATCH_FB"
case "$LIMB" in
    real)   READ_ARGS="$READ_ARGS --cortex.latent_read_scramble false --cortex.latent_carry_read true" ;;
    donor)  READ_ARGS="$READ_ARGS --cortex.latent_read_scramble true --cortex.latent_carry_read true" ;;
    noread) READ_ARGS="$READ_ARGS --cortex.latent_read_scramble false --cortex.latent_carry_read false" ;;
esac

TAG=""
if [ "$E_DROPOUT" != "0.0" ] && [ "$E_DROPOUT" != "0" ]; then TAG="-edrop${E_DROPOUT}"; fi
if [ -n "$PROSE_ONLY" ]; then TAG="$TAG-prose"; fi
# RUN_SUFFIX keeps a smoke (e.g. RUN_SUFFIX=-smoke) out of the real run's dir.
RUN_NAME=j3-a3z-${ENCODING}${TAG}-${LIMB}${RUN_SUFFIX:-}
ACCUM_MAX=$(( ACCUM_VECS * CROSS_CHUNKS ))

echo "=== J3 scratchpad run | limb=$LIMB | run=$RUN_NAME ==="
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
    echo "J3 $LIMB DONE -> cortex-retrofit/$RUN_NAME"
    echo "  In-flight record: cortex-retrofit/$RUN_NAME/cortex_diag.jsonl"
    echo "  (read_gate, read_ratio = the read's effective strength, own_share,"
    echo "   scratch_row_norm, write_grad_frac, e_dropped, E rank)."
    echo "  Read-out: EXPERIMENT=j3 pace/j1_readout.sbatch, then EXPERIMENT=j3"
    echo "  pace/j1_verdict.sbatch -- and only under J3's pre-registration."
else
    echo "J3 $LIMB FAILED rc=$RC"
fi
exit $RC
