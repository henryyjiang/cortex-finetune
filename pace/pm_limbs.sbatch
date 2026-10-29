#!/bin/bash
#SBATCH --account=gts-aivanova7-lab
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128GB
#SBATCH --gres=gpu:H200:1
#SBATCH --constraint=H200
#SBATCH -t 6:00:00
#SBATCH -q inferno
#SBATCH -J pm_limbs
#SBATCH -o logs/Report-%A_%a.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=henryyjiang42@gmail.com

# ============================================================================
# POST-MORTEM LIMBS -- six short trained branches off the finished 5B pair.
#
#   sbatch --array=0-5 pace/pm_limbs.sbatch
#     [0] both  [1] eonly  [2] zonly  [3] edrop  [4] recall  [5] recall_ctrl
#   sbatch --array=6 pace/pm_limbs.sbatch      [6] lownorm (added 2026-10-29)
#
# WHY TRAINED LIMBS AND NOT MORE ABLATIONS (2026-10-28)
# -----------------------------------------------------
# cortex-final trained with e_dropout 0.0 and never saw E switched off, so every
# E-off cell of the 5B 2x2 is out of distribution (score_j5.py:66-73's rule).
# E_main, Z_alone and the interaction -- i.e. the complement-vs-substitute
# question the dual channel exists to answer -- are unanswerable from the two
# trained arms.  J5 answered the same problem by TRAINING one limb per cell.
# This does that again, on the model being presented.
#
#   both        no change.  THE comparison for every other limb: it absorbs the
#               re-warm, the extra 32.8M tokens and the batch-4 noise, so a limb
#               is only ever read against `both`, never against 38,146.
#   eonly       --cortex.latent_carry_read false   (Z's columns are zeros)
#   zonly       --cortex.e_carry_read false        (E's spliced rows are zeros)
#   edrop       --cortex.e_dropout 0.25            (E blanked on 25% of rows)
#               One model in which all four 2x2 cells are in distribution, so
#               E_main / Z_alone / the interaction become readable without a
#               cross-limb comparison.
#   recall      `both` on a recall-supervised mix: a fact 2-7 chunks before a
#               probe at the row's end, so the answer is reachable ONLY through
#               the carry.  The causal test for "is the short memory horizon
#               the OBJECTIVE or the ARCHITECTURE": E's forget gate trained to
#               0.17 and Z's to 0.45 on next-token loss over PG-19/FineWeb,
#               which never pays for retention past ~1 lap.  If the gates move
#               toward retention and BABILong's gold-outside-window subset
#               improves, the ring CAN hold it and the data never asked.
#   recall_ctrl the c-chunked control on the same recall pack.  It cannot see
#               across chunks, so its probe score is the answer-prior floor.
#   lownorm     E AND Z spliced at TOKEN scale: --cortex.e_read_norm_target and
#               latent_read_znorm_target both LOWNORM_TARGET (4.5 = the
#               frequency-weighted ||wte|| row norm on PG-19 tokens, measured
#               2026-10-29) instead of E unscaled at ~84 and Z at 78.9.
#               SandwichBlock reads raw x, so the carry's V is ~18x a token's
#               and the coda puts 57-72% of real-token attention mass on it
#               (local probe, 8 PG-19 test books).  The test of "the chunk-1
#               tax is the weights adapting to a carry 18x too loud": read
#               chunk 1 and MC against `both`.  A continuation, so a null may
#               mean "2,000 updates cannot unlearn it", not "scale is not it".
#
# eonly IS A MEASUREMENT, NOT A SHIP CANDIDATE.  Z stays in cortex-final
# (feedback_z_required); this limb exists to price E alone in distribution.
#
# BRANCH POINT: checkpoint_38146 (Henry, 2026-10-28) -- the model on the slides.
#
# THE LR -- THIS IS THE PART THAT IS EASY TO GET WRONG
# ----------------------------------------------------
# --branch_path restores weights but CONTINUES the parent's LR scheduler and
# step counter (train.py:63-71).  The 5B schedule ended at min_lr_ratio x peak
# = 0.1% of peak, so an inherited schedule trains 2,000 updates at LR ~ 0 and
# returns six copies of 38,146.  So:
#   --ignore_past_scheduler true   a FRESH warmup_stable_decay schedule
#   MAX_STEPS = PARENT + CELL + 1  the horizon the fresh schedule is sized to.
#                                  Decay starts at (1 - cooldown) x MAX_STEPS
#                                  scheduler steps = ~16,000, so the 2,000 run
#                                  here sit in the STABLE phase after a 96-step
#                                  warmup: a constant LR, identical across limbs.
#   + 1                            train.py checks max_steps BEFORE stop_at_step
#                                  and the max_steps exit does NOT save
#                                  (train.py:2540 vs 2549).  With them equal, the
#                                  limb ends with no chkpt.pt and round 2 has
#                                  nothing to load.  One past it, the stop exit
#                                  fires and writes checkpoint_$STOP/chkpt.pt.
#   LR_SCALE 0.1                   x the 5B peaks (lr 5e-5, muon 1e-3, memory
#                                  5e-4).  100x the LR the parent ended at,
#                                  10x under its peak: enough for a one-channel
#                                  limb to adapt in 2,000 updates without
#                                  un-annealing the backbone.  J-arms ran at the
#                                  full peak, but they branched mid-schedule.
# The recurrence ramp is NOT reset (only the LR scheduler is): the restored
# ramp sits past its 0.25 x MAX_STEPS warmup, i.e. at full mean recurrence 8.
#
# BUDGET: 2,000 updates x 4 rows x 4,096 = 32.8M tokens, J-arms' exact budget.
# Measured cortex rate at this geometry 5,077-6,552 tok/s (job 13662406_2), so
# ~1.6-1.8 h of training + ~10 min load/save.  The control is ~25% slower per
# token (it holds 8 chunks to cortex's 4, mbs 1).  6 h wall is 3x margin.
#
# A NULL IS AMBIGUOUS, A POSITIVE IS CONCLUSIVE.  Same caveat as rung1
# EPOCHS=1: 32.8M tokens of adaptation after a finished cosine.  `zonly`
# failing to carry could be "Z cannot" or "Z did not learn to in 2,000 steps".
#
# DATA ORDER.  A branch reads its pack from row 0 under a permutation that is a
# function of (seed, len(dataset)).  At the run's own seed 74 the cortex limbs
# would re-walk EXACTLY the rows the 5B run consumed first after step 11,450.
# SEED 75 draws a fresh permutation of the same pack, identical for every limb,
# so the limbs' training losses still pair step for step
# (paired-update-loss-contrast).
#
# ZNORM_TARGET 78.9 is what the parent trained at from 11,450 on (re-measured
# 2026-10-03, walk 78.9 vs diag 78.4); a branch that does not move the
# lineage reuses it, and the gate below checks ||E|| stayed within 15%.
#
# ROUND 2 reads every limb in ITS OWN cell (j5's rule):
#   both, recall   E1Z1          eonly   E1Z0          zonly   E0Z1
#   edrop          all four cells (that is why it exists)
# ============================================================================

cd $SLURM_SUBMIT_DIR

ARM=cortex                            # final5b_shared reads it; reset per limb
source pace/final_5b_common.sh
OUT_DIR=${OUT_DIR:-cortex-pm}         # its own wandb project and scratch dir
final5b_shared

LIMBS=(both eonly zonly edrop recall recall_ctrl lownorm)
IDX=${SLURM_ARRAY_TASK_ID:-0}
LIMB=${LIMBS[$IDX]}
if [ -z "$LIMB" ]; then echo "no limb at index $IDX (use --array=0-6)"; exit 0; fi

PARENT_STEP=${PARENT_STEP:-38146}
CELL_STEPS=${CELL_STEPS:-2000}
STOP_AT_STEP=$(( PARENT_STEP + CELL_STEPS ))
LIMB_MAX_STEPS=$(( STOP_AT_STEP + 1 ))     # see "+ 1" in the header
LIMB_BATCH=${LIMB_BATCH:-4}
LR_SCALE=${LR_SCALE:-0.1}
SEED=${LIMB_SEED:-75}
ZNORM_TARGET=${ZNORM_TARGET:-78.9}
LOWNORM_TARGET=${LOWNORM_TARGET:-4.5}
# lownorm moves Z's target too, so ARM_ARGS below carries ONE znorm flag.
[ "$LIMB" = "lownorm" ] && ZNORM_TARGET=$LOWNORM_TARGET
RECALL_DATA=${RECALL_DATA:-data/pg19fw50_recall_train_len4096}
DIAG_INTERVAL=${DIAG_INTERVAL:-25}

LR_L=$(python -c "print('%g' % (5e-5 * $LR_SCALE))")
MUON_L=$(python -c "print('%g' % (1e-3 * $LR_SCALE))")
MEM_L=$(python -c "print('%g' % (5e-4 * $LR_SCALE))")

case "$LIMB" in
    recall_ctrl) PARENT_RUN=c-chunked;    MICRO_BS=1 ;;
    *)           PARENT_RUN=cortex-final; MICRO_BS=2 ;;
esac
case "$LIMB" in
    recall|recall_ctrl) DATA_PATH=$RECALL_DATA ;;
    *)                  DATA_PATH=$MIX_DATA ;;
esac
BRANCH_PATH="cortex-5b/$PARENT_RUN/checkpoint_$PARENT_STEP"
RUN_NAME=pm-${LIMB}${RUN_SUFFIX:-}

[ -f "$BRANCH_PATH/chkpt.pt" ] || { echo "ERROR: no chkpt.pt under $BRANCH_PATH"; exit 1; }
[ -d "$DATA_PATH" ] || { echo "ERROR: no such pack: $DATA_PATH"
    [ "$DATA_PATH" = "$RECALL_DATA" ] && echo "  build it first: bash pace/pm_prep_packs.sh"
    exit 1; }
if [ -e "$OUT_DIR/$RUN_NAME/checkpoint_$STOP_AT_STEP/chkpt.pt" ]; then
    echo "ERROR: $OUT_DIR/$RUN_NAME already finished.  Pass RUN_SUFFIX=-b to run"
    echo "  a second copy rather than overwrite the first."; exit 1
fi
# The pack must hold CELL_STEPS x LIMB_BATCH rows or the exhaustion guard
# fires at the end of a 2-hour job.
python tools/check_pack.py --data "$DATA_PATH" --steps "$CELL_STEPS" \
    --batch_size "$LIMB_BATCH" || { echo "REFUSING: pack too small"; exit 1; }

if [ "$LIMB" = "recall_ctrl" ]; then
    ARM_ARGS="--cortex.use_memory false"
else
    # VERBATIM cortex_final_5b.sbatch's ARM_ARGS (tests/test_pm_limbs.py pins
    # it), then ONE per-limb flag appended last so it wins.
    ARM_ARGS="--cortex.use_memory true"
    ARM_ARGS="$ARM_ARGS --cortex.memory_lr $MEM_L --cortex.memory_slots 0"
    ARM_ARGS="$ARM_ARGS --cortex.prefix_memory gated"
    ARM_ARGS="$ARM_ARGS --cortex.accum_vecs 16"
    ARM_ARGS="$ARM_ARGS --cortex.accum_max $(( 16 * CROSS_CHUNKS ))"
    ARM_ARGS="$ARM_ARGS --cortex.gate_slots 64"
    ARM_ARGS="$ARM_ARGS --cortex.gate_route ring --cortex.gate_norm tanh"
    ARM_ARGS="$ARM_ARGS --cortex.gate_init zero --cortex.gate_fill grow"
    ARM_ARGS="$ARM_ARGS --cortex.latent_carry true"
    ARM_ARGS="$ARM_ARGS --cortex.latent_encoding staggered_state"
    ARM_ARGS="$ARM_ARGS --cortex.latent_embed_frozen true"
    ARM_ARGS="$ARM_ARGS --cortex.latent_depth_rule absolute"
    ARM_ARGS="$ARM_ARGS --cortex.latent_depth_lo 1"
    ARM_ARGS="$ARM_ARGS --cortex.latent_depth_hi 9"
    ARM_ARGS="$ARM_ARGS --cortex.latent_renorm none"
    ARM_ARGS="$ARM_ARGS --cortex.latent_s0_read false"
    ARM_ARGS="$ARM_ARGS --cortex.latent_read embeds"
    ARM_ARGS="$ARM_ARGS --cortex.latent_read_znorm rms"
    ARM_ARGS="$ARM_ARGS --cortex.latent_read_znorm_target $ZNORM_TARGET"
    ARM_ARGS="$ARM_ARGS --cortex.latent_read_scramble false"
    ARM_ARGS="$ARM_ARGS --cortex.carry_grad_chunks 4"
    ARM_ARGS="$ARM_ARGS --cortex.window_backward true"
    ARM_ARGS="$ARM_ARGS --cortex.prefix_eos_reset false --cortex.prefix_pos tail"
    ARM_ARGS="$ARM_ARGS --cortex.freeze_loop false"
    case "$LIMB" in
        eonly) READ="--cortex.e_carry_read true  --cortex.latent_carry_read false --cortex.e_dropout 0.0" ;;
        zonly) READ="--cortex.e_carry_read false --cortex.latent_carry_read true  --cortex.e_dropout 0.0" ;;
        edrop) READ="--cortex.e_carry_read true  --cortex.latent_carry_read true  --cortex.e_dropout 0.25" ;;
        lownorm) READ="--cortex.e_carry_read true --cortex.latent_carry_read true --cortex.e_dropout 0.0 --cortex.e_read_norm_target $LOWNORM_TARGET" ;;
        *)     READ="--cortex.e_carry_read true  --cortex.latent_carry_read true  --cortex.e_dropout 0.0" ;;
    esac
    ARM_ARGS="$ARM_ARGS $READ"
fi

echo "=== POST-MORTEM LIMB | limb=$LIMB | run=$OUT_DIR/$RUN_NAME ==="
echo "    branch:   $BRANCH_PATH  (step $PARENT_STEP -> stop $STOP_AT_STEP," \
     "schedule horizon $LIMB_MAX_STEPS, FRESH scheduler)"
echo "    data:     $DATA_PATH   (READ THIS LINE -- the only proof the pack arrived)"
echo "    batch:    $LIMB_BATCH rows x micro $MICRO_BS  ->" \
     "$(( CELL_STEPS * LIMB_BATCH * MAX_LENGTH )) tokens"
echo "    lr:       x$LR_SCALE of the 5B peaks -> lr $LR_L muon $MUON_L memory $MEM_L" \
     "(constant after 96 warmup steps)"
echo "    seed:     $SEED   znorm_target: $ZNORM_TARGET"
echo "    limb:     ${READ:-use_memory false}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
git rev-parse --short HEAD 2>/dev/null | sed 's/^/    commit:   /'

python train.py \
    --run_name $RUN_NAME \
    --out_path $OUT_DIR \
    --model_name $MODEL \
    --preprocessed_data_path $DATA_PATH \
    --max_length $MAX_LENGTH \
    --epochs 1 \
    --max_steps $LIMB_MAX_STEPS \
    --stop_at_step $STOP_AT_STEP \
    --micro_batch_size $MICRO_BS \
    --batch_size $LIMB_BATCH \
    --optim_config.lr $LR_L \
    --scheduler_args.cooldown 0.6 \
    --scheduler_args.warmup_steps 96 \
    --muon.use_muon true \
    --muon.lr $MUON_L \
    --no_amp false \
    --max_grad_norm 1.0 \
    --compile false \
    --save_interval -1 \
    --save_n_mins_before_timeout 20 \
    --seed $SEED \
    --mean_recurrence_schedule.turn_on true \
    --mean_recurrence_schedule.warmup $RAMP_WARMUP \
    --mean_recurrence_schedule.max_mean_rec $MAX_MEAN_REC \
    --mean_recurrence_schedule.warmup_type "1-sqrt" \
    --cortex.cross_chunks $CROSS_CHUNKS \
    --cortex.eos_from_tokens true \
    --cortex.diag_interval $DIAG_INTERVAL \
    $ARM_ARGS \
    --branch_path $BRANCH_PATH \
    --ignore_past_scheduler true
RC=$?
echo ""
echo "=== limb $LIMB finished rc=$RC ==="
[ -f "$OUT_DIR/$RUN_NAME/checkpoint_$STOP_AT_STEP/chkpt.pt" ] \
    && echo "    resumable checkpoint: $OUT_DIR/$RUN_NAME/checkpoint_$STOP_AT_STEP" \
    || { echo "    *** NO chkpt.pt at checkpoint_$STOP_AT_STEP -- round 2 cannot load this limb"; RC=1; }
[ "$LIMB" = "recall_ctrl" ] && exit $RC

# ===== THE GATE: did this limb run ITS OWN condition? ======================
# Reads the MEASURED diagnostics, never the flags (the e_carry_read lesson).
# Every check is j5_joint.sbatch's, re-keyed for these limbs.
python - "$OUT_DIR/$RUN_NAME/cortex_diag.jsonl" "$LIMB" "$ZNORM_TARGET" <<'GATE'
import json, sys
path, limb, target = sys.argv[1], sys.argv[2], float(sys.argv[3])
try:
    rows = [json.loads(l) for l in open(path) if l.strip()]
except FileNotFoundError:
    sys.exit(f"FAIL: no {path}")
if not rows:
    sys.exit("FAIL: cortex_diag.jsonl is empty")
vals = lambda k: [r[k] for r in rows if r.get(k) is not None]
fails, notes = [], []

ecr = {int(v) for v in vals("z_e_carry_read")}
want = int(limb != "zonly")
if ecr != {want}:
    fails.append(f"z_e_carry_read {sorted(ecr)}, expected {{{want}}}")
esn = vals("z_e_spliced_norm")
if not esn:
    fails.append("z_e_spliced_norm absent")
elif limb == "zonly" and max(abs(v) for v in esn) != 0.0:
    fails.append(f"zonly spliced E rows of norm up to {max(esn):.4g}; expected EXACTLY 0")
elif limb == "edrop" and max(esn) <= 0.0:
    fails.append("edrop spliced E norm is 0 on EVERY record: E never reached the splice")
elif limb not in ("zonly", "edrop") and min(esn) <= 0.0:
    # edrop is exempt from the min: at mbs 2 both rows are blanked in ~1/16
    # records, so an exact 0 there is the dropout working (job 14560553_3).
    fails.append(f"{limb} spliced E norm {min(esn):.4g}: E is off on a limb that carries it")
elif limb == "lownorm" and max(abs(v - target) for v in esn) > 0.02 * target:
    fails.append(f"lownorm spliced E norm {min(esn):.3g}..{max(esn):.3g}, expected "
                 f"{target}: e_read_norm_target never reached the graft")
else:
    notes.append(f"z_e_spliced_norm {esn[0]:.2f}..{esn[-1]:.2f}")
er = vals("z_embed_ratio")
if not er:
    fails.append("z_embed_ratio absent")
elif limb == "lownorm":
    # E is CARRIED at ~84 but SPLICED at the target, so the ratio to the
    # carried norm is ~0.05 by design; check Z's spliced norm itself.
    zn = vals("z_embed_norm")
    if not zn or max(abs(v - target) for v in zn) > 0.02 * target:
        fails.append(f"lownorm z_embed_norm {zn[:1]}..{zn[-1:]}, expected {target}")
    else:
        notes.append(f"z_embed_norm {zn[0]:.2f}..{zn[-1]:.2f} (target {target})")
elif limb == "eonly":
    if max(abs(v) for v in er) != 0.0:
        fails.append(f"eonly z_embed_ratio up to {max(er):.4g}; expected EXACTLY 0")
    else:
        notes.append("z_embed_ratio exactly 0 (no Z read)")
else:
    notes.append(f"z_embed_ratio {er[0]:.3f} -> {er[-1]:.3f}")
    if not 0.5 <= er[0] <= 2.0:
        fails.append(f"z_embed_ratio starts at {er[0]:.3f}, outside 0.5..2.0")
if limb == "edrop":
    ed = vals("z_e_dropout")
    if not ed or abs(ed[-1] - 0.25) > 1e-6:
        fails.append(f"z_e_dropout {ed[-1] if ed else None}, expected 0.25 -- the flag never reached the graft")
    else:
        dropped = vals("z_e_dropped")
        # e_dropped is a COUNT of rows blanked in the last forward, so at
        # micro_batch_size 2 the expectation is 0.5, not 0.25.
        m = sum(dropped) / max(len(dropped), 1)
        notes.append(f"z_e_dropout 0.25; rows blanked per forward {m:.3f} "
                     f"(expect ~0.5 at mbs 2) over {len(dropped)} records")
        if dropped and m == 0:
            fails.append("e_dropout configured but no row was ever blanked")
en = vals("z_e_carried_norm")
if en:
    m = sum(en) / len(en)
    notes.append(f"z_e_carried_norm mean {m:.2f} vs target {target:.2f}")
    if limb != "lownorm" and not 0.85 * target <= m <= 1.15 * target:
        fails.append(f"||E|| {m:.2f} is >15% from ZNORM_TARGET {target:.2f}")
for k in ("fg_at_bias", "fg_z_at_bias", "e_eff_rank_pr", "z_eff_rank_pr"):
    v = vals(k)
    if v:
        notes.append(f"{k} {v[0]:.3f} -> {v[-1]:.3f}")
print("\n".join("  ok   " + n for n in notes))
if fails:
    print("\n".join("  FAIL " + f for f in fails))
    sys.exit(1)
print(f"  GATE PASSED for limb {limb}")
GATE
GRC=$?
[ $RC -eq 0 ] && RC=$GRC
exit $RC
