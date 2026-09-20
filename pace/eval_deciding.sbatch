#!/bin/bash
#SBATCH --account=gts-aivanova7-lab
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H200:1
#SBATCH --mem=64GB
#SBATCH -t 2:00:00
#SBATCH -q inferno
#SBATCH -J p21_subst
#SBATCH -o logs/Report-%A_%a.out
#SBATCH --array=0-17
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=henryyjiang42@gmail.com

# P2.1 -- THE POWERED CARRY-SUBSTITUTION SWEEP.
# Plan of record: p2_deciding_measurements_handoff.md section 2.
#
#   sbatch pace/eval_deciding.sbatch                      # all 18 cells
#   MAX_EXAMPLES=20 sbatch pace/eval_deciding.sbatch      # a fast shakedown
#
#   PASS --array WHENEVER YOU SUBSET.  The #SBATCH line says 0-17 because that
#   is the full sweep; ARMS/DAMAGES only shorten the CELL LIST, not the array,
#   so the surplus tasks each take an H200 allocation to print "nothing to do"
#   and exit.  Harmless, and 16 wasted slots on a busy partition:
#     ARMS="a1 parent" DAMAGES=donor sbatch --array=0-1 pace/eval_deciding.sbatch
#   The job prints the right --array for your override before it runs anything.
#
# WHAT THIS DECIDES, AND WHY IT RUNS BEFORE ANY CELL.  Two questions, one job:
#
#  1. DOES THE CARRY'S CONTENT HELP.  The P1 probes' donor control came back
#     NEGATIVE on all four arms -- content_delta -0.1234 / -0.1253 / -0.0758 /
#     -0.0540, with `none < donor < real` in loss everywhere -- against +0.0186
#     on the parent at matched geometry.  That is n=1 per arm, one batch, no CI,
#     and P11 section 5 pre-registered "near zero at 400 steps".  If it survives
#     at n=100 with a CI, it is a fact about the DENOMINATOR under every carry
#     number in the project and it outranks both the Z question and the gate
#     question.  The parent is in the sweep for exactly that reason: it is the
#     only row that can say whether 400 steps of branch training did it.
#
#  2. CAN THE INSTRUMENT SEE AT Z'S SCALE.  `--damage random` swaps in noise at
#     the SAME per-row norm.  Z is read by substitution into s0 at 0.90x the
#     trunc_normal_ noise it replaces (P0.1), so "Z moved nothing" and "nothing
#     could move anything at this scale" are the same picture until donor and
#     random are shown to separate.  Random is a CONTROL ON THE INSTRUMENT, not
#     a third result -- do not report it as one.
#
# THE READING, FIXED HERE BEFORE THE NUMBERS EXIST (handoff section 2):
#   donor negative, CI excludes zero, on arms AND parent -> the carry is
#       anti-informative at the endpoint.  STOP the cells; the top line becomes
#       the carry itself and the next question is why, not which buffer.
#   negative on the arms, ~zero on the parent -> 400 steps of branch training
#       did it.  Suspect the W 32->16 slice re-adaptation; re-measure at 2k
#       steps before reading anything into it.
#   ~zero everywhere -> the -0.12 was one-batch noise, P11 section 5 stands,
#       P2.3 proceeds unchanged.
#   donor-Z and random-Z do NOT separate -> the instrument is blind at Z's
#       scale.  Every Z null so far is uninformative REGARDLESS of
#       read_live_frac, and the next Z question is a scale question.
#   they DO separate and real-Z matches both -> Z carries nothing here.  Drop Z
#       from the headline; a2/a3z never need full cells.
#
# --T 8 IS NOT OPTIONAL.  These checkpoints' config.json says mean_recurrence
# 32 -- inherited from the base recipe, never what the arms trained at -- and
# the tool takes the config when --T is unset.  Same trap that put
# read_live_frac 0.0195 in the P1 records for arms that ran at 0.546.
#
# --n_chunks MUST MATCH the arm's cross_chunks (8) or the carry is scored at a
# depth it never trained at.
#
# THE INSTRUMENT WAS FIXED TO RUN THIS.  Until 2026-09-17 `run_chain`'s gated
# branch ignored damage_at entirely, so on A3/A3' every damage mode was a no-op
# and I(d) would have come back exactly 0.0 at every depth -- "the gated carry
# does not matter anywhere", which reads like a result.  The damage now goes in
# before `prefix.merge`, where the header always said it did.  See
# tests/test_influence_horizon.py, which did not exist either.

cd $SLURM_SUBMIT_DIR

module load anaconda3
module load cuda/12.1.1
conda activate cortex-retro

export SCRATCH=${SCRATCH:-$HOME/scratch}
export HF_HOME=$SCRATCH/hf_cache
export HF_HUB_OFFLINE=1

MODEL=${MODEL:-ckpts/olmo-retrofit-cortex}
DATA=${DATA:-data/pg19_olmo_val_len4096}
OUT_ROOT=${OUT_ROOT:-cortex-retrofit}
BRANCH_PATH=${BRANCH_PATH:-cortex-retrofit/retro-b2-heal/checkpoint_91552_w16}
STAMP=${STAMP:-$(date +%Y%m%d)}
OUT_BASE=${OUT_BASE:-eval_results/p21_substitution-$STAMP}

# The geometry that NAMES the runs.  Keep in step with pace/submit_p1_probes.sh
# and pace/p1_arms.sbatch: a mismatch here does not score a different arm, it
# scores NOTHING and reports the arm as missing.
ACCUM_VECS=${ACCUM_VECS:-16}
GATE_SLOTS=${GATE_SLOTS:-64}
CROSS_CHUNKS=${CROSS_CHUNKS:-8}
MAX_MEAN_REC=${MAX_MEAN_REC:-8}
DEPTHS=${DEPTHS:-"1 2 3 4 6"}
MAX_EXAMPLES=${MAX_EXAMPLES:-100}
BOOT=${BOOT:-2000}

ARMS=${ARMS:-"a1 a2 a3 a3z parent"}
DAMAGES=${DAMAGES:-"donor random"}

# RED 13, second site (2026-09-19).  These names were HARDCODED to `probe-`,
# with no override, because P2.1 ran on the 400-step architecture probes and
# they were the only checkpoints that existed.  p1_arms.sbatch adds `probe-`
# only under PROBE=1 (line 324); the P2.3 cells saved to the BARE name.  Run
# this against the cells with the old default and it does not error -- the dirs
# are identical in layout -- it scores a 400-step probe and labels it the cell.
#
#   (default)           the P2.3 cells
#   RUN_PREFIX=probe-   the P1 architecture probes (what P2.1 scored)
RUN_PREFIX=${RUN_PREFIX:-}

arm_ckpt () {
    case "$1" in
    a1)  RUN=${RUN_PREFIX}p1-a1-accum-w${ACCUM_VECS}-cc${CROSS_CHUNKS} ;;
    a2)  RUN=${RUN_PREFIX}p1-a2-accum-w${ACCUM_VECS}-cc${CROSS_CHUNKS}-z ;;
    a3)  RUN=${RUN_PREFIX}p1-a3-gated-w${ACCUM_VECS}k${GATE_SLOTS}-cc${CROSS_CHUNKS} ;;
    a3z) RUN=${RUN_PREFIX}p1-a3z-gated-w${ACCUM_VECS}k${GATE_SLOTS}-cc${CROSS_CHUNKS}-z ;;
    parent) echo "$BRANCH_PATH"; return 0 ;;
    *) return 1 ;;
    esac
    # NUMERIC sort on the step, so checkpoint_91952 does not lose to
    # checkpoint_9999 the way a lexical sort would.
    ls -d $OUT_ROOT/$RUN/checkpoint_* 2>/dev/null | sort -t_ -k2 -n | tail -1
}

# Which arms carry a second channel, and therefore have per-channel cells.  The
# eval REFUSES --damage_channel e|z on an E-only carry (running it as 'both'
# under a per-channel label is how a 1x2 becomes a 2x2), so those cells are not
# generated rather than generated and left to fail.
is_dual () { case "$1" in a2|a3z) return 0 ;; *) return 1 ;; esac; }

CELLS=()
for ARM in $ARMS; do
    for DMG in $DAMAGES; do
        CELLS+=("$ARM $DMG both")
        if is_dual "$ARM"; then
            CELLS+=("$ARM $DMG e")
            CELLS+=("$ARM $DMG z")
        fi
    done
done

IDX=${SLURM_ARRAY_TASK_ID:-0}
echo "=== P2.1 substitution sweep | ${#CELLS[@]} cells | index $IDX ==="
for i in "${!CELLS[@]}"; do echo "    [$i] ${CELLS[$i]}"; done
echo "    for THIS cell list:  sbatch --array=0-$(( ${#CELLS[@]} - 1 ))"
if [ "$IDX" -ge "${#CELLS[@]}" ]; then
    echo "no cell at index $IDX (the array is wider than the cell list, which is"
    echo "what an ARMS/DAMAGES override does).  Nothing to do -- but this task"
    echo "still took a GPU allocation to say so.  Pass --array=0-$(( ${#CELLS[@]} - 1 )) next time."
    exit 0
fi

set -- ${CELLS[$IDX]}
ARM=$1; DMG=$2; CH=$3
CKPT=$(arm_ckpt "$ARM")
if [ -z "$CKPT" ] || [ ! -d "$CKPT" ]; then
    echo "ERROR: no checkpoint for arm '$ARM' (looked for '$CKPT')."
    echo "       Run the P1 probes first: bash pace/submit_p1_probes.sh"
    exit 1
fi

# The name is not enough: RUN_PREFIX can be wrong, and a cell that died early
# leaves a real checkpoint at the wrong step.  Make the step number itself the
# check.  `parent` is EXEMPT -- BRANCH_PATH is checkpoint_91552_w16 by design
# and is supposed to be below the cells.  Auto-off when scoring the probes.
MIN_STEP=${MIN_STEP:-115966}
[ -n "$RUN_PREFIX" ] && MIN_STEP=0
if [ "$MIN_STEP" != "0" ] && [ "$ARM" != "parent" ]; then
    CKPT_STEP=$(basename "$CKPT" | sed 's/^checkpoint_//; s/_.*$//')
    case "$CKPT_STEP" in
    ''|*[!0-9]*)
        echo "ERROR: cannot read a step number out of '$CKPT'. MIN_STEP=0 to skip."
        exit 1 ;;
    *)
        if [ "$CKPT_STEP" -lt "$MIN_STEP" ]; then
            echo "ERROR: arm '$ARM' resolved to '$CKPT', step $CKPT_STEP < MIN_STEP=$MIN_STEP."
            echo "       That is a PROBE or an unfinished cell, not the P2.3 cell."
            echo "       Use RUN_PREFIX=probe- to score the probes deliberately,"
            echo "       or MIN_STEP=0 to override."
            exit 1
        fi ;;
    esac
fi
if [ ! -d "$DATA" ]; then
    echo "ERROR: no such pack: DATA=$DATA"; exit 1
fi

# HOW MANY ROWS DOES THE PACK ACTUALLY HOLD.  --max_examples is a ceiling, not a
# sample size: the P2.1 sweep passed 100 and every one of its 18 cells ran n=50,
# because data/pg19_olmo_val_len4096 holds 50 rows (PG-19 validation is 50 books
# and the old packer kept one truncated row per book).  Half the pre-registered
# power, discovered only afterwards.  Printed here, and the cell says so out
# loud when the flag over-promises.
PACK_ROWS=$(python -c "import sys; from datasets import load_from_disk; print(len(load_from_disk(sys.argv[1])))" "$DATA" 2>/dev/null || echo "?")
echo "    pack rows:  $PACK_ROWS   (--max_examples $MAX_EXAMPLES is a CEILING)"
if [ "$PACK_ROWS" != "?" ] && [ "$PACK_ROWS" -lt "$MAX_EXAMPLES" ] 2>/dev/null; then
    echo "    *** n IS $PACK_ROWS, NOT $MAX_EXAMPLES -- the pack binds, not the flag."
    echo "    *** A bigger validation pack exists for exactly this:"
    echo "    ***   SPLIT=validation sbatch pace/prepare_pg19_pack.sbatch"
    echo "    *** then re-run with DATA=data/pg19_olmo_validation_len4096_strided"
fi

# THE GEOMETRY HAS TO BE FORCED, and the first run of this job died because it
# was not.  --model_name loads ckpts/olmo-retrofit-cortex's config, which
# carries NO cortex flags at all -- `use_memory` is literally '<absent>' there,
# which is why every p1_arms probe log opens with a block of
# "[cortex] OVERRIDE config.use_memory: '<absent>' -> True" lines.  Without
# these the graft builds with no prefix buffer and the eval exits 2 with "this
# checkpoint has no prefix buffer".  Mirrors p1_arms.sbatch's PROBE_SETS,
# which is the construction that is known to load these exact dirs.
ACCUM_MAX=${ACCUM_MAX:-$(( ACCUM_VECS * CROSS_CHUNKS ))}
case "$ARM" in
    a1|a2|parent) PREFIX_MODE=accum ;;
    *)            PREFIX_MODE=gated ;;
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
fi

OUT=$OUT_BASE/$ARM-$DMG-$CH
echo ""
echo "    arm=$ARM damage=$DMG channel=$CH"
echo "    checkpoint: $CKPT"
echo "    data:       $DATA   (READ THIS LINE -- it is the only proof the pack arrived)"
echo "    geometry:   $PREFIX_MODE  W=$ACCUM_VECS  latent=$LATENT"
echo "    sets:       $SETS"
echo "    T=$MAX_MEAN_REC  n_chunks=$CROSS_CHUNKS  depths=$DEPTHS  n=$MAX_EXAMPLES"
echo "    out:        $OUT"
echo ""
echo "    The [cortex] OVERRIDE lines below are the proof the geometry arrived."
echo "    A cell that prints none of them is running on a buffer-less graft."
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

python evals/eval_influence_horizon.py \
    --model_name $MODEL --checkpoint "$CKPT" --data $DATA $SETS \
    --n_chunks $CROSS_CHUNKS --depths $DEPTHS \
    --damage $DMG --damage_channel $CH \
    --T $MAX_MEAN_REC --max_examples $MAX_EXAMPLES --boot $BOOT \
    --dtype float32 \
    --out_dir "$OUT"
RC=$?

echo ""
if [ $RC -eq 0 ]; then
    echo "cell $ARM/$DMG/$CH DONE -> $OUT/results.json"
    echo ""
    echo "This cell is one row of the sweep, not a verdict.  The reading is the"
    echo "TABLE -- arms against the parent for question 1, donor against random"
    echo "for question 2 -- and the responses to each outcome are pre-registered"
    echo "in this file's header and in p2_deciding_measurements_handoff.md."
else
    echo "cell $ARM/$DMG/$CH FAILED rc=$RC"
fi
exit $RC
