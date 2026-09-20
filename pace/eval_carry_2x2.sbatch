#!/bin/bash
#SBATCH --account=gts-aivanova7-lab
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H200:1
#SBATCH --mem=64GB
#SBATCH -t 2:00:00
#SBATCH -q inferno
#SBATCH -J carry2x2
#SBATCH -o logs/Report-%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=henryyjiang42@gmail.com

# THE 2x2 CARRY ABLATION -- E on/off x Z on/off at MATCHED COLUMN COUNT.
# Runs at the END of P2.3, on the cell checkpoints.
#
#   ARM=a1 sbatch pace/eval_carry_2x2.sbatch
#   ARM=a3 sbatch pace/eval_carry_2x2.sbatch
#   ARM=a2 sbatch pace/eval_carry_2x2.sbatch      # a real 2x2: Z is live
#   RUN=<explicit run name> CKPT=<explicit dir> sbatch pace/eval_carry_2x2.sbatch
#
# WRITTEN BEFORE THE CELLS LAND, ON PURPOSE.  `evals/eval_carry_2x2.py` had no
# launcher and no test file -- the exact state `eval_influence_horizon.py` was
# in when RED 9 (a damage path that was a silent no-op on every gated arm) lived
# inside it undetected.  Building the launcher is also what found the hole
# below, at zero GPU cost, instead of after a 32 h cell.
#
# ==== THE HOLE, FOUND 2026-09-17 AND FIXED: RED 11's SHAPE, ONE FILE OVER ====
# Z's null is `_null_latent`, which takes a PER-ELEMENT std.  The tool passed
# `cfg.init_values["std"]` with a 0.02 fallback -- a weight-init number, not
# s0's rms.  And `diag_s0_sensitivity` measured that the substitution site has
# NO GAIN until the injected norm reaches ~175.8 against ||E|| = 171.0, so a
# null at 0.02 sits deep inside the flat region: every Z cell would equal its E
# twin, the interaction term would come out 0.000, and the table would report
# "Z carries nothing" about the INSTRUMENT.  The tool now primes one forward,
# reads the graft's measured `_z_s0_rms`, prints it, and REFUSES on a checkpoint
# too old to record it -- the same guard, in the same words, as the s0 sweep.
#
# WHAT EACH ARM CAN AND CANNOT ANSWER:
#   a1, a3   E-only.  The Z axis does not exist, so the run is an E-on/E-off
#            contrast at matched columns and the tool is told so explicitly with
#            --allow_missing_z.  This is the honest 1x2; it is NOT a 2x2 and the
#            record says so.  It is also what the accum-vs-gated comparison
#            needs today.
#   a2, a3z  Z is live, so the full 2x2 runs and the INTERACTION term -- are E
#            and Z complements or substitutes -- is measurable.  Do NOT pass
#            --allow_missing_z here: the refusal is what stops a missing channel
#            being reported as a null result for Z.
#
# MATCHED COLUMNS IS THE DESIGN, NOT A DETAIL.  Every cell splices the same
# number of columns at the same positions and changes only the CONTENTS; if
# "E off" dropped the columns instead of emptying them, the sequence would
# shorten, initialize_state would draw a different s0, positions would shift,
# and "no information" would be confounded with "shorter sequence".
#
# Matched columns hold WITHIN an arm and NOT across an accum arm and a gated
# one: an append buffer's read block grows while a ring's is fixed at K.  That
# is the mechanism, not a confound -- comparing those two is the influence
# horizon's job, not this one's.

cd $SLURM_SUBMIT_DIR

module load anaconda3
module load cuda/12.1.1
conda activate cortex-retro

export SCRATCH=${SCRATCH:-$HOME/scratch}
export HF_HOME=$SCRATCH/hf_cache
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODEL=${MODEL:-ckpts/olmo-retrofit-cortex}
DATA=${DATA:-data/pg19_olmo_val_len4096}
OUT_ROOT=${OUT_ROOT:-cortex-retrofit}
ARM=${ARM:-a1}
ACCUM_VECS=${ACCUM_VECS:-16}
GATE_SLOTS=${GATE_SLOTS:-64}
CROSS_CHUNKS=${CROSS_CHUNKS:-8}
T=${T:-8}                      # NOT config.mean_recurrence (32 on every
                               # B2-family checkpoint; no arm ran there)
MAX_EXAMPLES=${MAX_EXAMPLES:-100}
BOOT=${BOOT:-2000}
STAMP=${STAMP:-$(date +%Y%m%d)}

# The P2.3 cells save under the p1_arms naming; keep this in step with
# pace/p1_arms.sbatch and pace/eval_deciding.sbatch.  A mismatch here does not
# score a different arm, it scores NOTHING and reports the arm as missing.
if [ -z "$CKPT" ]; then
    # RED 13 (found 2026-09-19, before submit): these defaults carried the
    # `probe-` prefix, because when this launcher was written the ONLY
    # checkpoints on disk were the 400-step architecture probes.  p1_arms.sbatch
    # adds `probe-` only under PROBE=1 (line 324); the P2.3 cells ran without
    # it and saved to the BARE name.  The probe dirs have the same layout, so
    # the old default did not error -- it would have scored a 400-step probe
    # and reported it as the 24,414-step cell.  Keep these in step with
    # p1_arms.sbatch:241/251/269.
    case "$ARM" in
    a1)  RUN=${RUN:-p1-a1-accum-w${ACCUM_VECS}-cc${CROSS_CHUNKS}} ;;
    a2)  RUN=${RUN:-p1-a2-accum-w${ACCUM_VECS}-cc${CROSS_CHUNKS}-z} ;;
    a3)  RUN=${RUN:-p1-a3-gated-w${ACCUM_VECS}k${GATE_SLOTS}-cc${CROSS_CHUNKS}} ;;
    a3z) RUN=${RUN:-p1-a3z-gated-w${ACCUM_VECS}k${GATE_SLOTS}-cc${CROSS_CHUNKS}-z} ;;
    *)   echo "ERROR: unknown ARM=$ARM (expected a1|a2|a3|a3z)"; exit 1 ;;
    esac
    # NUMERIC sort on the step, so checkpoint_91952 does not lose to
    # checkpoint_9999 the way a lexical sort would.
    CKPT=$(ls -d $OUT_ROOT/$RUN/checkpoint_* 2>/dev/null | sort -t_ -k2 -n | tail -1)
fi

# The name guard above is necessary and not sufficient: RUN= can still be
# pointed at a probe by hand, and a cell that died early leaves a real
# checkpoint at the wrong step.  MIN_STEP makes the step number itself the
# check.  The P2.3 cells stop at 115966; the parent is 91552 and every probe
# sits within 400 of it, so any of them trips this.  MIN_STEP=0 disables it.
MIN_STEP=${MIN_STEP:-115966}
if [ "$MIN_STEP" != "0" ] && [ -n "$CKPT" ]; then
    CKPT_STEP=$(basename "$CKPT" | sed 's/^checkpoint_//; s/_.*$//')
    case "$CKPT_STEP" in
    ''|*[!0-9]*)
        echo "ERROR: cannot read a step number out of '$CKPT'."
        echo "       Pass MIN_STEP=0 to skip this guard if that is intended."
        exit 1 ;;
    *)
        if [ "$CKPT_STEP" -lt "$MIN_STEP" ]; then
            echo "ERROR: '$CKPT' is at step $CKPT_STEP, below MIN_STEP=$MIN_STEP."
            echo "       This is a PROBE or an unfinished cell, not the P2.3 cell."
            echo "       Scoring it would report a 400-step run as the 24,414-step"
            echo "       one -- the same shape as REDs 8/9/10.  Check the cell"
            echo "       finished, or pass MIN_STEP=0 deliberately."
            exit 1
        fi ;;
    esac
fi

[ -n "$CKPT" ] && [ -d "$CKPT" ] || {
    echo "ERROR: no checkpoint for arm '$ARM' (looked for '$CKPT')."
    echo "       The P2.3 cells write here; check they have finished."
    exit 1; }
[ -d "$DATA" ] || { echo "ERROR: no such pack: $DATA"; exit 1; }

# The geometry has to be FORCED: --model_name loads the base dir, whose config
# carries NO cortex flags (use_memory is literally '<absent>'), and use_memory
# is the master switch.  Mirrors pace/eval_deciding.sbatch.
ACCUM_MAX=${ACCUM_MAX:-$(( ACCUM_VECS * CROSS_CHUNKS ))}
case "$ARM" in
    a1|a2) PREFIX_MODE=accum ;;
    *)     PREFIX_MODE=gated ;;
esac
case "$ARM" in
    a2|a3z) LATENT=1 ;;
    *)      LATENT=0 ;;
esac

SETS="--set use_memory=true --set memory_slots=0"
SETS="$SETS --set prefix_memory=$PREFIX_MODE"
SETS="$SETS --set accum_vecs=$ACCUM_VECS"
if [ "$PREFIX_MODE" = "accum" ]; then
    SETS="$SETS --set accum_max=$ACCUM_MAX"
else
    SETS="$SETS --set gate_slots=$GATE_SLOTS"
    SETS="$SETS --set gate_route=${GATE_ROUTE:-ring}"
    SETS="$SETS --set gate_norm=${GATE_NORM:-tanh}"
    SETS="$SETS --set gate_init=${GATE_INIT:-zero}"
    SETS="$SETS --set gate_fill=${GATE_FILL:-grow}"
fi
if [ "$LATENT" = "1" ]; then
    SETS="$SETS --set latent_carry=true"
    SETS="$SETS --set latent_depth_rule=${LATENT_RULE:-absolute}"
    SETS="$SETS --set latent_depth_lo=${LATENT_LO:-2}"
    SETS="$SETS --set latent_depth_hi=${LATENT_HI:-9}"
    SETS="$SETS --set latent_renorm=${LATENT_RENORM:-none}"
    ALLOW=""            # a real 2x2; the refusal is the protection
else
    ALLOW="--allow_missing_z"
fi

OUT=${OUT:-eval_results/carry_2x2-$STAMP/$ARM}
echo "=== 2x2 carry ablation | arm=$ARM | T=$T | n_chunks=$CROSS_CHUNKS ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "    checkpoint: $CKPT"
echo "    data:       $DATA"
echo "    geometry:   $PREFIX_MODE  W=$ACCUM_VECS  latent=$LATENT"
echo "    sets:       $SETS $ALLOW"
echo "    out:        $OUT"
if [ "$LATENT" = "1" ]; then
    echo "    Z IS LIVE: look for the '[2x2] Z null at s0's MEASURED per-element"
    echo "    rms ...' line.  If it is absent the null is not at s0's scale and"
    echo "    the Z axis is measuring the instrument."
else
    echo "    E-ONLY ARM: this is a 1x2 (E on/off) and the record says so."
fi
echo ""

python evals/eval_carry_2x2.py \
    --model_name "$MODEL" --checkpoint "$CKPT" --data "$DATA" $SETS $ALLOW \
    --n_chunks "$CROSS_CHUNKS" --T "$T" \
    --max_examples "$MAX_EXAMPLES" --boot "$BOOT" \
    --dtype float32 \
    --out_dir "$OUT"
RC=$?

echo ""
if [ $RC -eq 0 ]; then
    echo "2x2 ($ARM) DONE -> $OUT/results.json"
    echo ""
    echo "READ THE CHUNK-1 SPREAD FIRST.  Every cell runs chunk 1 with no"
    echo "incoming carry, so the four must agree there to floating-point noise."
    echo "A spread that is not ~0 means the cells differ in something other than"
    echo "the carry's contents, and no row below it is interpretable."
elif [ $RC -eq 3 ]; then
    echo "REFUSED: no Z channel on this checkpoint.  That is the guard working"
    echo "-- pass --allow_missing_z (ARM=a1/a3 do it automatically) to run the"
    echo "E axis alone, and do not report the result as a null for Z."
else
    echo "2x2 ($ARM) FAILED rc=$RC"
fi
exit $RC
