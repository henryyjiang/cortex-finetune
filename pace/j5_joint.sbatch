#!/bin/bash
#SBATCH --account=gts-aivanova7-lab
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H200:1
#SBATCH --mem=128GB
#SBATCH -t 12:00:00
#SBATCH -q inferno
#SBATCH -J j5_joint
#SBATCH -o logs/Report-%A_%a.out
#SBATCH --array=0-2
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=henryyjiang42@gmail.com

# J5 -- WHAT THE BUFFER CARRIES, ON A BASE WHOSE RECURRENCE IS TRAINED.
# Pre-registration: ../j5_prereg.md.  Design: J4's, unchanged (endpoint write at
# the summary columns, read as input_embeds columns by the base model's own
# attention).  What moves is the BASE and the LIMB SET.
#
#   sbatch pace/j5_joint.sbatch        # [0] zonly [1] eonly [2] both
#
# THREE LIMBS, one base, one pack, one budget, one seed, one data order.  They
# differ in WHICH CHANNELS OF THE CARRY BUFFER ARE READ and in nothing else --
# same 656 packed columns, same parameter count (all three build
# latent_embed.proj), same write on both channels every chunk:
#
#   limb    e_carry_read  latent_carry_read   the buffer the model reads
#   zonly   FALSE         true                Z only   -- E's rows are zeros
#   eonly   true          FALSE               E only   -- Z's columns are zeros
#   both    true          true                Z + E    -- J4's design
#
# `e_carry_read=false` (cortex_graft.py, commit c043b56) zeros E's spliced rows
# on EVERY sequence, EVERY chunk, IN TRAINING AND IN EVAL.  It is not
# e_dropout=1.0 -- that bound stays [0, 1) and the two flags refuse each other.
# The buffer still MERGES E, so the write path trains and the ring keeps its
# geometry; only the read is zeroed.  `_e_carried_norm` is taken BEFORE the
# blank, so z_embed_ratio keeps its meaning on exactly the limb that needs it.
#
# ===== WHY THIS ARM, IN THE FEWEST NUMBERS THAT SETTLE IT ===================
# J1-J4 each asked whether a latent channel carries content IN THE RETROFIT and
# each answered NO.  D3 + probe (b) measured WHY, and the reason is structural
# rather than a design fault:
#   * E holds the whole register file: +0.61 (real) / +0.76 (noread) margin over
#     majority on the J4 checkpoints (eval_results/z3_registers/j4-*).
#   * The carried loop state holds ~nothing: Z_end margin +0.007 / +0.015,
#     lambda_rel pinned at the ridge's maximum -- the fit is noise.
#   * Probe (b): everything Z_end adds beyond E is +1.9% of E's margin on the
#     real arm and -0.3% on the noread arm.
#   * J4 then measured what the model does with a channel it does not need:
#     own Z vs a DONOR's, 0.7481 vs 0.7482 nats -- four decimals -- while
#     BLANKING those same columns costs 0.4716.  Load-bearing, content-free.
# So every J arm confounded two things: an UNTRAINED recurrent arrangement, and
# an E that was present and sufficient, so nothing pushed Z to carry.  J5
# removes the first for all three limbs and the second for one of them.
#
# ===== WHAT CHANGES FROM J4, AND EVERY ITEM IS A DECISION ===================
# 1. THE BASE.  ckpts/olmo8-cortex is smcleish/Recurrent-OLMo-2-0425-train-
#    recurrence-8 through the same tools/prepare_cortex_checkpoint.py --variant
#    olmo that built ckpts/olmo-retrofit-cortex, so it is graft-ready and
#    structurally identical -- the difference is that ITS RECURRENCE IS TRAINED.
#    This makes J5 a FINETUNE (Track A), not a retrofit (Track B).  Registered
#    consequence, ../j5_prereg.md S1.2: J5's number is a feasibility statement
#    about the Z design and is never quoted as a retrofit number.
# 2. NO BRANCH_PATH.  The prepared base IS the start, so PARENT_STEP=0 and
#    STOP_AT_STEP=CELL_STEPS.  J4's launcher hardcoded 91552 + 2000.
# 3. THE RECURRENCE RAMP IS OFF, AND THIS IS THE ONE THAT WOULD HAVE RUINED THE
#    RUN SILENTLY.  --branch_path continues the step counter; without it the
#    counter starts at 0, and mean_recurrence_schedule.warmup=0.25 over the
#    305,176-step horizon ramps the depth over 76,294 updates.  At update 4,000
#    the "1-sqrt" ramp is 2.7% of the way up: ceil(0.027 x 8) = 1.  The whole
#    arm would have trained AT DEPTH 1 -- no loop, hence no loop state, hence no
#    Z -- behind a perfectly healthy loss curve.  The ramp exists to ease an
#    UNTRAINED loop into depth; this base's loop is trained at 8, so the ramp is
#    not merely harmful here, it is pointless.  Off, with the depth pinned:
#      --mean_recurrence_schedule.turn_on false
#      --override_mean_recurrence 8  --override_mean_backprop_depth 8
#    which is the (8, 8) J4 actually ran at 91552 once its ramp had finished.
# 4. THE LR WARMUP IS ABSOLUTE, 250 UPDATES.  Same reason: entering the P2.3
#    schedule at step 0 rather than 91552 would spend warmup=0.0025 x 305176 =
#    763 of 4,000 updates below peak LR, a 19% budget tax the J arms never paid.
#    The HORIZON stays 305,176 so the peak LR and its shape are J1/J3/J4's;
#    only the entry point is fixed.  All three limbs share it.
# 5. 4,000 UPDATES, READ AT 2,000 AND 4,000 (SAVE_INTERVAL=2000).  Double J4's,
#    because "Z is slow to learn" is the live hypothesis and answering it at the
#    J arms' own budget leaves the same objection in place.  The 2,000 read-out
#    makes this arm report a TRAJECTORY rather than a point.
# 6. NO TRAINED DONOR LIMB.  J4's donor was a fourth training run; here the
#    donor swap is an EVAL cell (eval_carry_2x2.py --z_null donor), within
#    model, on the same rows -- which is the cell that caught J4's failure and
#    costs no GPU-hours of training.  pace/j5_readout.sbatch runs it.
# 7. THE MEMORY PATH IS FRESH, and this is the honest cost.  The J arms
#    inherited 91,552 steps of summary_emb training from retro-b2-heal; J5
#    inherits none.  `eonly` shares the handicap exactly, which is what makes
#    the deciding comparison internally valid -- and probe (c) measured that
#    the heal phase bought NO register-carrying skill anyway (its carried E
#    decodes the register file at -0.003, CI spanning zero), only adaptation
#    speed.  Whether a fresh write path reaches E's +0.6 in 4,000 updates is
#    exactly what `eonly` measures.  IF `eonly` FAILS task_learned THE WHOLE ARM
#    IS NOT_READ (veto 8): "Z did not learn" and "nothing learns here in 4,000
#    updates" would be one number.
#
# ===== BEFORE THE FIRST LAUNCH, IN ORDER, NONE OPTIONAL =====================
#   1. MEMORY:  MODEL=ckpts/olmo8-cortex REQUIRE_FIT=1 sbatch pace/j4_price_oom.sbatch
#      J4's geometry exactly (656 columns, micro 2, window_backward), ~91 GiB.
#   2. TOKENIZER (runs automatically below, and refuses the launch):
#      the pack was tokenized with ckpts/olmo-retrofit-cortex, not with this
#      base.  A mismatch trains on garbage behind a healthy loss curve.
#   3. MEASURE ||E|| ON THIS BASE -- the one number that is NOT guessable:
#        MEASURE_ENORM=1 CELL_STEPS=10 DIAG_INTERVAL=1 RUN_SUFFIX=-enorm \
#            sbatch -t 0:45:00 pace/j5_joint.sbatch
#        cat logs/j5_enorm/*.txt      # one ZNORM_TARGET for ALL THREE limbs
#      J4's single smoke failure was this and nothing else: registered at 171.0
#      from B2, measured 128.95-142.83 on the heal branch, corrected to 136.0.
#      On this base it is a different number again.  DIAG_INTERVAL=1 is not
#      optional: at any other interval a 10-step run writes no diag rows at all.
#   4. SMOKE GATE, as an exit code so the real limbs can chain on it:
#        ZNORM_TARGET=<step 3's number> CELL_STEPS=10 DIAG_INTERVAL=1 \
#            SMOKE_GATE=1 SMOKE_CLEAN=1 RUN_SUFFIX=-smoke \
#            sbatch -t 0:45:00 pace/j5_joint.sbatch
#      train.py exits 0 on a run whose Z entered at the wrong scale.  The gate
#      below is J4's, extended with vetoes 1-2 of ../j5_prereg.md S3: E's
#      SPLICED norm must be exactly 0 on `zonly` and nonzero on the other two.
#   5. THE REAL LIMBS:
#        J=$(ZNORM_TARGET=<n> sbatch --parsable --dependency=afterok:$SMOKE \
#            pace/j5_joint.sbatch)
#      then pace/j5_readout.sbatch at STEP=2000 and STEP=4000, then
#      pace/j5_verdict.sbatch.

cd $SLURM_SUBMIT_DIR

module load anaconda3
module load cuda/12.1.1
conda activate cortex-retro

export SCRATCH=${SCRATCH:-$HOME/scratch}
export HF_HOME=$SCRATCH/hf_cache
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODEL=${MODEL:-ckpts/olmo8-cortex}          # Track A's base: the TRAINED loop
PACK_TOKENIZER=${PACK_TOKENIZER:-ckpts/olmo-retrofit-cortex}   # what built the pack
ARM_DATA=${ARM_DATA:-data/j1_pg19fw50_carry20_len4096}
MAX_LENGTH=${MAX_LENGTH:-4096}
CROSS_CHUNKS=${CROSS_CHUNKS:-8}
ACCUM_VECS=${ACCUM_VECS:-16}
GATE_SLOTS=${GATE_SLOTS:-64}
CARRY_GRAD_CHUNKS=${CARRY_GRAD_CHUNKS:-4}   # 4 of 8, as J1/J3/J4 and the P2.3 cells
MEAN_REC=${MEAN_REC:-8}                     # pinned, not ramped -- see item 3
BACKPROP_DEPTH=${BACKPROP_DEPTH:-8}
MAX_STEPS=${MAX_STEPS:-305176}              # the LR SCHEDULE horizon (P2.3's)
WARMUP_STEPS=${WARMUP_STEPS:-250}           # ABSOLUTE -- see item 4
PARENT_STEP=0                               # no branch: the base IS the start
CELL_STEPS=${CELL_STEPS:-4000}
STOP_AT_STEP=$(( PARENT_STEP + CELL_STEPS ))
SAVE_INTERVAL=${SAVE_INTERVAL:-2000}        # read-outs at 2000 AND 4000
DIAG_INTERVAL=${DIAG_INTERVAL:-25}
MICRO_BS=${MICRO_BS:-2}
BATCH_SIZE=${BATCH_SIZE:-4}
ZNORM=${ZNORM:-rms}                         # the rescale; the graft refuses 'none'
WINDOW_BACKWARD=${WINDOW_BACKWARD:-1}
WB_ARGS=""
if [ "$WINDOW_BACKWARD" = "1" ]; then WB_ARGS="--cortex.window_backward true"; fi
ENCODING=endpoint                           # D3's vindicated write; the graft
                                            # refuses anything else under embeds

# ZNORM_TARGET HAS NO SAFE DEFAULT ON A NEW BASE.  J4's 136.0 was measured on
# the heal branch and 171.0 before that; a wrong value enters Z at the wrong
# scale behind a healthy loss.  MEASURE_ENORM=1 is the mode that produces it,
# and it is the only mode allowed to run without one.
if [ -n "$MEASURE_ENORM" ]; then
    ZNORM_TARGET=${ZNORM_TARGET:-136.0}     # PLACEHOLDER: e_carried_norm does
                                            # not depend on it.  Do not reuse.
elif [ -z "$ZNORM_TARGET" ]; then
    echo "ERROR: ZNORM_TARGET is not set, and it has no safe default on this"
    echo "       base.  It is ||E||'s MEASURED carried row norm, and J4's"
    echo "       single smoke failure was exactly this number being inherited"
    echo "       from another branch (171.0 -> 136.0).  Measure it first:"
    echo "         MEASURE_ENORM=1 CELL_STEPS=10 DIAG_INTERVAL=1 \\"
    echo "             RUN_SUFFIX=-enorm sbatch -t 0:45:00 pace/j5_joint.sbatch"
    echo "         cat logs/j5_enorm/*.txt"
    echo "       then pass ONE value for all three limbs."
    exit 1
fi

LIMBS=(zonly eonly both)
IDX=${SLURM_ARRAY_TASK_ID:-0}
LIMB=${LIMBS[$IDX]}
if [ -z "$LIMB" ]; then echo "no limb at index $IDX (use --array=0-2)"; exit 0; fi

[ -d "$MODEL" ] || { echo "ERROR: no such base: $MODEL.  Build it with"; \
    echo "  python tools/prepare_cortex_checkpoint.py --src <hf repo> --dst $MODEL --variant olmo"; exit 1; }
[ -d "$ARM_DATA" ] || { echo "ERROR: no such pack: $ARM_DATA (build it: pace/prepare_carry_task.sbatch)"; exit 1; }
if [ "$MICRO_BS" -lt 2 ]; then
    echo "ERROR: MICRO_BS=$MICRO_BS.  The read-out's donor cell pairs rows"
    echo "       across the batch, and every limb must share one micro batch"
    echo "       size or the limbs differ in more than the carry."
    exit 1
fi
# The Z WRITE needs iter_write: without it the endpoint encoding records
# nothing and the arm trains a buffer with a dead Z half.
if ! grep -q "iter_write(x, current_step)" "$MODEL/raven_modeling_minimal_cortex.py" 2>/dev/null; then
    echo "ERROR: $MODEL carries a modeling file without iter_write(x,"
    echo "       current_step) -- the Z write would never fire.  Refresh:"
    echo "         python tools/prepare_cortex_checkpoint.py --src $MODEL --dst $MODEL --variant olmo"
    echo "         rm -rf \$SCRATCH/hf_cache/modules/transformers_modules/$(basename $MODEL)"
    exit 1
fi

# ===== PRE-FLIGHT 2: THE TOKENIZER (../j5_prereg.md S1.4, veto 3) ===========
# The pack was tokenized with PACK_TOKENIZER and this run trains with MODEL.
# Both descend from OLMo-2-0425-1B so the vocabularies are EXPECTED to match --
# which is exactly the kind of expectation that becomes a check here, because a
# mismatch trains on garbage behind a healthy loss curve and the carry task's
# answers are single digit tokens.
python tools/check_tokenizer_match.py --pack_tokenizer "$PACK_TOKENIZER" \
    --model "$MODEL" || {
    echo "REFUSING THE LAUNCH: tokenizer mismatch (../j5_prereg.md veto 3)."
    exit 1; }

# Every limb has BOTH blocks of columns; the limbs differ in what is read.
READ_ARGS="--cortex.latent_read embeds"
READ_ARGS="$READ_ARGS --cortex.latent_read_znorm $ZNORM --cortex.latent_read_znorm_target $ZNORM_TARGET"
READ_ARGS="$READ_ARGS --cortex.latent_read_scramble false"
case "$LIMB" in
    zonly) READ_ARGS="$READ_ARGS --cortex.e_carry_read false --cortex.latent_carry_read true"  ;;
    eonly) READ_ARGS="$READ_ARGS --cortex.e_carry_read true  --cortex.latent_carry_read false" ;;
    both)  READ_ARGS="$READ_ARGS --cortex.e_carry_read true  --cortex.latent_carry_read true"  ;;
esac

RUN_NAME=j5-a3z-${ENCODING}-${LIMB}${RUN_SUFFIX:-}
ACCUM_MAX=$(( ACCUM_VECS * CROSS_CHUNKS ))

echo "=== J5 buffer-contents arm | limb=$LIMB | run=$RUN_NAME ==="
echo "    base:      $MODEL   (NO branch: step $PARENT_STEP -> stop $STOP_AT_STEP, horizon $MAX_STEPS)"
echo "    data:      $ARM_DATA   (READ THIS LINE -- the only proof the pack arrived)"
echo "    geometry:  gated W=$ACCUM_VECS K=$GATE_SLOTS cc=$CROSS_CHUNKS grad_chunks=$CARRY_GRAD_CHUNKS window_backward=$WINDOW_BACKWARD"
echo "    depth:     mean_recurrence $MEAN_REC, backprop depth $BACKPROP_DEPTH, RAMP OFF (see item 3)"
echo "    packed:    64 E + 64 Z + 512 tokens + 16 summary = 656 columns"
echo "    read:      $READ_ARGS"
echo "    schedule:  warmup $WARMUP_STEPS updates (absolute), save every $SAVE_INTERVAL, diag every $DIAG_INTERVAL"
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
    --scheduler_args.warmup_steps $WARMUP_STEPS \
    --muon.use_muon true \
    --muon.lr 0.001 \
    --no_amp false \
    --max_grad_norm 1.0 \
    --compile false \
    --save_interval $SAVE_INTERVAL \
    --save_n_mins_before_timeout 20 \
    --seed 74 \
    --mean_recurrence_schedule.turn_on false \
    --override_mean_recurrence $MEAN_REC \
    --override_mean_backprop_depth $BACKPROP_DEPTH \
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
    --cortex.latent_s0_read false \
    $READ_ARGS \
    --cortex.e_dropout 0.0 \
    --cortex.diag_interval $DIAG_INTERVAL \
    --cortex.carry_grad_chunks $CARRY_GRAD_CHUNKS \
    $WB_ARGS \
    --cortex.cross_chunks $CROSS_CHUNKS \
    --cortex.eos_from_tokens true \
    --cortex.prefix_eos_reset false \
    --cortex.prefix_pos tail \
    --cortex.freeze_loop false
RC=$?

# ===== MEASURE ||E|| AND STOP (pre-flight 3) =================================
# The one number that is not guessable on a new base.  Reported per limb; the
# operator sets ONE ZNORM_TARGET for all three from the mean.
if [ -n "$MEASURE_ENORM" ] && [ $RC -eq 0 ]; then
    echo ""
    echo "=== ||E|| ON $MODEL | limb=$LIMB ==="
    mkdir -p logs/j5_enorm
    python - "cortex-retrofit/$RUN_NAME/cortex_diag.jsonl" "$LIMB" \
        > "logs/j5_enorm/${LIMB}.txt" <<'ENORM'
import json, sys
path, limb = sys.argv[1], sys.argv[2]
try:
    rows = [json.loads(l) for l in open(path) if l.strip()]
except FileNotFoundError:
    sys.exit(f"FAIL: no {path}.  Re-run with DIAG_INTERVAL=1: at any other "
             f"interval a 10-step run writes no diag rows at all.")
vals = [r["z_e_carried_norm"] for r in rows if r.get("z_e_carried_norm") is not None]
if not vals:
    sys.exit(f"FAIL: no z_e_carried_norm in {len(rows)} rows.")
m = sum(vals) / len(vals)
print(f"{limb} n={len(vals)} mean={m:.3f} min={min(vals):.3f} max={max(vals):.3f}")
ENORM
    cat "logs/j5_enorm/${LIMB}.txt"
    echo ""
    echo "  When all three limbs have reported:"
    echo "    cat logs/j5_enorm/*.txt"
    echo "  Take ONE value (the mean across limbs, as J4 did: 128.95/135.77/"
    echo "  142.83 -> 136.0) and pass it as ZNORM_TARGET to every limb.  The"
    echo "  limbs must differ in the carry and in nothing else."
    exit $RC
fi

# ===== THE SMOKE GATE ========================================================
# Only with SMOKE_GATE=1, so a real run is bit-identical without it.  J4's gate,
# extended with ../j5_prereg.md S3 vetoes 1-2: the MEASURED proof that each limb
# ran its own condition.  train.py exits 0 on a run whose Z entered 57x too
# small with a perfectly healthy loss curve; this is the only thing that does
# not.
if [ -n "$SMOKE_GATE" ] && [ $RC -eq 0 ]; then
    echo ""
    echo "=== SMOKE GATE | $RUN_NAME | limb=$LIMB ==="
    python - "cortex-retrofit/$RUN_NAME/cortex_diag.jsonl" "$LIMB" \
             "$ZNORM_TARGET" "$GATE_SLOTS" <<'GATE'
import json, math, sys

path, limb, target, k = sys.argv[1], sys.argv[2], float(sys.argv[3]), int(sys.argv[4])
try:
    rows = [json.loads(l) for l in open(path) if l.strip()]
except FileNotFoundError:
    sys.exit(f"FAIL: no {path}.  A short run at DIAG_INTERVAL > 1 writes no "
             f"rows at all.  Re-run the smoke with DIAG_INTERVAL=1.")
if not rows:
    sys.exit(f"FAIL: {path} is empty.")
fails, notes = [], []


def vals(key):
    return [r[key] for r in rows if r.get(key) is not None]


notes.append(f"{len(rows)} diag rows, steps {rows[0]['step']}..{rows[-1]['step']}")

# 0. ONE CONFIG PER FILE.  cortex_diag.jsonl is opened in APPEND mode, so a
# re-run into a surviving run dir interleaves two configs' rows and every
# number below would mix them.
tg = set(vals("z_znorm_target"))
if len(tg) > 1:
    fails.append(f"this diag holds rows from MORE THAN ONE config "
                 f"(z_znorm_target {sorted(tg)}): the run dir survived an "
                 f"earlier smoke.  rm -rf the -smoke run dirs and re-run.")
elif tg and abs(next(iter(tg)) - target) > 1e-6:
    fails.append(f"the run trained with znorm_target {next(iter(tg))} but this "
                 f"gate was told {target} -- the launcher and the gate disagree.")
elif tg:
    notes.append(f"z_znorm_target {next(iter(tg))} on every row")

# 1. THE LIMB IS THE LIMB (j5_prereg.md S3 vetoes 1-2).  e_carry_read is a
# CONFIG FLAG; e_spliced_norm is the MEASURED half of the same fact, and 0.0 is
# the proof E was off -- the way z_embed_ratio == 0 proved J4's no-read limb
# read nothing.  A limb that did not run its own condition answers nothing.
# cortex_memory/health.training_diag casts every bool to 0/1 so wandb will
# plot it, so this field is an INT in the jsonl and is compared as one.
ecr = {int(v) for v in vals("z_e_carry_read")}
want_ecr = int(limb != "zonly")
if not ecr:
    fails.append("z_e_carry_read absent from every row -- this build predates "
                 "the Z-only arm (cortex_graft.py, commit c043b56)")
elif ecr != {want_ecr}:
    fails.append(f"z_e_carry_read {sorted(ecr)} on limb {limb}, expected "
                 f"{want_ecr}: this limb did not run its own condition.")
else:
    notes.append(f"z_e_carry_read {want_ecr} on every row")

esn = vals("z_e_spliced_norm")
if not esn:
    fails.append("z_e_spliced_norm absent -- the measured proof that E is on "
                 "or off is missing, so vetoes 1-2 cannot be checked")
elif limb == "zonly":
    if max(abs(v) for v in esn) != 0.0:
        fails.append(f"VETO 1: the zonly limb spliced E rows of norm up to "
                     f"{max(esn):.4g}, expected EXACTLY 0.  E's read is not off "
                     f"and this limb is not the Z-only condition.")
    else:
        notes.append("z_e_spliced_norm exactly 0 (E's read is off, measured)")
else:
    if min(esn) <= 0.0:
        fails.append(f"VETO 2: the {limb} limb spliced E rows of norm "
                     f"{min(esn):.4g} -- E is off on a limb that must carry it.")
    else:
        notes.append(f"z_e_spliced_norm {esn[0]:.2f}..{esn[-1]:.2f} (E is on)")

# 2. the Z columns exist at all, on EVERY limb -- that is what makes the limbs
# geometry-matched rather than three different architectures.
zpre = vals("z_n_zpre")
if not zpre:
    fails.append("z_n_zpre absent from every row -- this build predates J4")
elif max(zpre) != k:
    fails.append(f"z_n_zpre max {max(zpre)}, expected {k}: the Z columns were "
                 f"never spliced at full width")
else:
    notes.append(f"z_n_zpre max {max(zpre)} of {k}")

# 3. ZNORM_TARGET is the right number ON THIS BASE.  Measured pre-blank, so it
# is reported on the zonly limb too.
en = vals("z_e_carried_norm")
if not en:
    fails.append("z_e_carried_norm absent -- cannot confirm ZNORM_TARGET")
else:
    m = sum(en) / len(en)
    notes.append(f"z_e_carried_norm mean {m:.2f} vs target {target:.2f} "
                 f"({100 * m / target:.0f}%)")
    if not (0.85 * target <= m <= 1.15 * target):
        fails.append(f"VETO 4: ||E|| measures {m:.2f} but ZNORM_TARGET is "
                     f"{target:.2f} -- more than 15% apart.  Re-measure with "
                     f"MEASURE_ENORM=1 and set ONE value for all three limbs; "
                     f"every limb would otherwise train with Z at the wrong "
                     f"scale, behind a healthy loss curve.")

# 4. Z is entering at E's scale (zonly / both) or not at all (eonly)
er = vals("z_embed_ratio")
if not er:
    fails.append("z_embed_ratio absent -- the only read-strength number here")
elif limb == "eonly":
    if max(abs(v) for v in er) != 0.0:
        fails.append(f"VETO 6: the eonly limb has z_embed_ratio up to "
                     f"{max(er):.4g}, expected EXACTLY 0.  Its Z columns must "
                     f"be zeros.")
    else:
        notes.append("z_embed_ratio exactly 0 (the E-only limb reads no Z)")
else:
    first = er[0]
    notes.append(f"z_embed_ratio {first:.4f} -> {er[-1]:.4f}")
    if not (0.5 <= first <= 2.0):
        fails.append(f"z_embed_ratio starts at {first:.4f}, outside 0.5..2.0.  "
                     f"It is ~1.0 by construction when the rescale runs; ~0.06 "
                     f"means it did not, and Z is entering 17x under E.")

# 5. the design's structural claim: the read has gradient on EVERY batch
rg = vals("z_read_grad_frac")
if not rg:
    fails.append("z_read_grad_frac absent")
elif limb != "eonly" and min(rg) < 1.0:
    fails.append(f"VETO 7: z_read_grad_frac min {min(rg)} on {limb}, expected "
                 f"1.0.  J4's columns live in input_embeds and are re-fed every "
                 f"iteration, so the read is live however long the no-grad "
                 f"prefix is.  Below 1.0 the graft is not reporting the site it "
                 f"ran.")
else:
    notes.append(f"z_read_grad_frac min {min(rg)}")

wg = vals("z_write_grad_frac")
if not wg:
    fails.append("z_write_grad_frac absent")
elif max(wg) <= 0:
    fails.append(f"VETO 7: z_write_grad_frac max {max(wg)}: the write never "
                 f"trained, on a limb where the write is the only thing the "
                 f"loss can improve through the memory")
else:
    notes.append(f"z_write_grad_frac max {max(wg):.3f}")

# 6. the depth is 8, not the ramp's 1.  Item 3 of the header: this is the
# failure that would otherwise be invisible.
nsg = vals("num_steps_with_grad")
nsn = vals("num_steps_no_grad")
if not nsg:
    notes.append("num_steps_with_grad absent from the diag (not fatal)")
else:
    tot = [a + b for a, b in zip(nsn or [0] * len(nsg), nsg)]
    mt = sum(tot) / len(tot)
    notes.append(f"loop iterations {min(tot)}..{max(tot)}, mean {mt:.1f}")
    if mt < 4.0:
        fails.append(f"THE LOOP IS NOT RUNNING AT DEPTH: mean {mt:.1f} "
                     f"iterations against a pinned mean_recurrence of 8.  The "
                     f"recurrence ramp is on, or the overrides did not apply -- "
                     f"see item 3 of pace/j5_joint.sbatch.  With no loop there "
                     f"is no loop state and no Z.")

# 7. and the loss is a number
losses = vals("loss")
if not losses:
    fails.append("no loss in any diag row")
elif not all(math.isfinite(v) for v in losses):
    fails.append("a non-finite loss")
else:
    notes.append(f"loss {losses[0]:.4f} -> {losses[-1]:.4f}, all finite")

for n in notes:
    print(f"  ok   {n}")
for f in fails:
    print(f"  FAIL {f}")
if fails:
    sys.exit(f"\nSMOKE GATE FAILED ({len(fails)}): do NOT launch the real limbs "
             f"until this is understood.  ../j5_prereg.md S3.")
print("\nSMOKE GATE PASSED -- the real limbs may launch.")
GATE
    RC=$?
    if [ $RC -ne 0 ]; then
        echo "J5 $LIMB SMOKE GATE FAILED -- exit $RC, so anything queued"
        echo "  --dependency=afterok on this job will NOT start."
        exit $RC
    fi
    if [ -n "$SMOKE_CLEAN" ]; then
        mkdir -p logs/j5_smoke
        # JOB-SCOPED name: a re-smoke after a fix must not overwrite the diag of
        # the smoke that FOUND the fix.
        cp "cortex-retrofit/$RUN_NAME/cortex_diag.jsonl" \
           "logs/j5_smoke/${RUN_NAME}-${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}.jsonl" 2>/dev/null
        rm -rf cortex-retrofit/$RUN_NAME/checkpoint_*
        echo "  diag kept at logs/j5_smoke/${RUN_NAME}-${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}.jsonl"
        echo "  checkpoints removed; DELETE THE RUN DIR before any re-smoke"
        du -sh "cortex-retrofit/$RUN_NAME" 2>/dev/null
    fi
fi

echo ""
if [ $RC -eq 0 ]; then
    echo "J5 $LIMB DONE -> cortex-retrofit/$RUN_NAME"
    echo "  In-flight record: cortex-retrofit/$RUN_NAME/cortex_diag.jsonl"
    echo "  READ THESE FIRST:"
    echo "    z_e_spliced_norm  0 on zonly, > 0 on eonly/both -- the MEASURED"
    echo "                      proof each limb ran its own condition"
    echo "    z_embed_ratio     ||Z col||/||E col||, ~1.0 at step 0 on"
    echo "                      zonly/both, EXACTLY 0 on eonly; the trajectory"
    echo "                      is the read-strength reading"
    echo "    z_e_carried_norm  confirms ZNORM_TARGET was right on this base"
    echo "    num_steps_*       the loop must be running near 8, not 1"
    echo "    z_write_grad_frac on zonly the write is the ONLY thing the loss"
    echo "                      can improve through the memory"
    echo "  Read-out: STEP=2000 sbatch pace/j5_readout.sbatch, then STEP=4000,"
    echo "  then pace/j5_verdict.sbatch -- under ../j5_prereg.md and nothing else."
else
    echo "J5 $LIMB FAILED rc=$RC"
fi
exit $RC
