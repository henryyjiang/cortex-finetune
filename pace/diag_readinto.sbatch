#!/bin/bash
#SBATCH --account=gts-aivanova7-lab
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H200:1
#SBATCH --mem=64GB
#SBATCH -t 1:00:00
#SBATCH -q inferno
#SBATCH -J readinto_gain
#SBATCH -o logs/Report-%A_%a.out
#SBATCH --array=0-3
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=henryyjiang42@gmail.com

# P3.0 TIERS 0.5 AND 1 -- DOES THE IN-LOOP READ SITE HAVE GAIN?
# No training.  Four cells, ~10 min each, ~0.7 GPU-hours for the lot.
#
#   sbatch pace/diag_readinto.sbatch                      # both tiers, both arms
#   MODULES=xattn sbatch --array=0-1 pace/diag_readinto.sbatch   # tier 1 only
#   ARMS=a3z sbatch --array=0-1 pace/diag_readinto.sbatch        # one arm
#
# WHY IT RUNS BEFORE TIER 1.5.  The oracle read probe costs ~8 GPU-hours and a
# training cell pair costs ~58.  This costs 0.7 and it is the only thing
# standing between here and them.  p30_readinto_prereg.md S3 puts the whole
# ordering on this point: the EXISTENCE question is answered before a single
# training cell is committed.
#
# WHAT IS BEING ASKED.  pace/diag_s0.sbatch established that s0 is NOT a read
# site -- deleting its carried columns outright costs +4.1e-5 / -2.5e-5 nats
# (opposite signs on the two arms, i.e. noise), flat out to 45x ||s0||, turning
# on only at an injected norm of 175.8 against ||E|| = 171.0, where the
# substitution is DROWNING E rather than being read.  P3.0's entire premise is
# that CortexGraft.read_into is different: it fires after the adapter and
# before the core blocks on EVERY iteration of BOTH loops, so the read gradient
# is live by construction, and it injects against ||x|| ~ 10 rather than
# against ||E|| = 171.
#
# THAT PREMISE HAS NEVER BEEN MEASURED.  This measures it.
#
#   tier 1    --module xattn    LatentRead at init.  Sweep its scalar `gate`.
#   tier 0.5  --module refresh  LatentRefresh, Option 0.  Sweep `alpha`.  This is
#             the arm that isolates REPETITION from LEARNED EXTRACTION: it
#             tests whether s0's carried columns are washed because the loop
#             OVERWRITES them at iteration 1, which the s0 sweep cannot rule
#             out and which no amount of renorm would have fixed.
#
# THE AXIS IS A MEASURED ROW NORM, NOT A GATE VALUE.  A gate of 0.1 means
# nothing on its own, so the graft records ||x|| at the injection point, the
# ||Z row|| being read and the ||delta|| actually injected -- all fp32 -- and
# every cell is reported as ||delta||/||x||.  RED 11 was precisely a sweep
# whose labels meant something other than what it injected, off by
# sqrt(D) = 45.25 because a ROW NORM was handed in where a PER-ELEMENT rms
# belonged.  Read the ratio column, not the gate.
#
# HOW TO READ IT -- pre-registered, in evals/diag_readinto_gain.py's docstring.
# The statistic is k in `d_nats ~ k * r^p` with r = ||delta||/||x||, NOT the
# span: job 13420851 read LIVE at one cut of the span and DEAD at every other,
# off the same table, so the span is printed at four cuts and decides nothing.
#
#   SENSITIVE_unlike_s0        k >= 100x s0's 8.6e-06.  Something injected here
#       survives to the loss, which is what s0 failed.  Necessary condition met
#       -> tier 1.5.  Print the row beside the s0 table; the PAIR is the
#       evidence, not either row alone.
#   INERT_like_s0              k at s0's own level.  CONTRADICTS p30's premise;
#       S5 row 3 says stop and audit before spending the 8 GPU-hours.
#   AUDIT_control_not_matched  real and noise separate AT INIT, which a random
#       projection cannot do.  The control is broken, not the site.
#   INVALID_no_axis            no ||delta||/||x|| recorded, so the cells could
#       only be scored on gate values.  Both tier-0.5 cells of job 13420851.
#   INVALID_knob_unproven      nothing moved anywhere, including the oversized
#       cell -- indistinguishable from a knob that was never wired.
#   INVALID_at_chance          RED 10.  The level vetoes the verdict.
#
# p is reported beside k.  p ~ 2 means the response is driven by the SIZE of
# the injection and nothing else.  AT INIT THAT IS EXPECTED -- the module is
# random, so content-specificity is untestable here by construction, and this
# sweep can only establish SENSITIVITY.  Damage vs reading is tier 1.5's job.
#
# THE REAL-vs-NOISE PAIR AT EVERY GATE.  Each cell is scored twice, once on the
# carried Z and once on `latent_read_null` noise at matched scale.  AT INIT THE
# TWO SHOULD AGREE -- the module is untrained, so it cannot prefer real content
# yet, and a separation here is a reason to audit rather than to celebrate.
# What the pair buys is the ability to tell "no gain" from "never landed".
#
# ONLY A READING THAT HOLDS ON BOTH ARMS IS ABOUT THE SITE.  The s0 sweep
# pre-registered this and it earned its place: the flat region agreed on both
# arms to five decimals while the loud region was 7x apart, which is what
# exposed the loud region as a swamp rather than a read.
#
# --T 8 IS NOT OPTIONAL.  config.mean_recurrence is 32 on every B2-family
# checkpoint and is INHERITED from the base recipe; B2's ramp capped
# max_mean_rec at 8 and no arm ever ran at 32.  RED 8 was this exact omission
# reporting read_live 0.0195 for arms that ran 0.546.
#
# THE `probe-` PREFIX BELOW IS DELIBERATE AND IS NOT RED 13, same as
# diag_s0.sbatch: only a2/a3z carry a Z channel and both are the 400-step
# ARCHITECTURE PROBES, so no P2.3 cell exists for either arm (the cells are
# a1/a3, E-only, and cannot host this at all -- they have no Z parameters).
# RUN_PREFIX is exposed anyway so the first trained Z cell does not need this
# file edited, which is the generalisation RED 13 earned.

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
STAMP=${STAMP:-$(date +%Y%m%d)}
OUT_BASE=${OUT_BASE:-eval_results/readinto_gain-$STAMP}
RUN_PREFIX=${RUN_PREFIX-probe-}

ACCUM_VECS=${ACCUM_VECS:-16}
GATE_SLOTS=${GATE_SLOTS:-64}
CROSS_CHUNKS=${CROSS_CHUNKS:-8}
MAX_MEAN_REC=${MAX_MEAN_REC:-8}
MAX_EXAMPLES=${MAX_EXAMPLES:-50}
READ_HEADS=${READ_HEADS:-8}
READ_DEPTH=${READ_DEPTH:-none}

# Gate values, four orders.  0 is the intact control (module present,
# contributing exactly nothing) and 10 is deliberately oversized so the run
# carries its own proof that the knob reaches the model.  KEEP AT LEAST ONE
# CELL ABOVE ||delta|| = ||x|| or the verdict can only ever be INVALID.
GATES=${GATES:-"0 0.001 0.01 0.1 0.3 1 3 10"}

# Only the DUAL arms have a Z to read, and only the xattn/refresh modules have
# a knob to turn.  The tool refuses anything else rather than printing a flat
# table that would read as DEAD by construction.
ARMS=${ARMS:-"a2 a3z"}
MODULES=${MODULES:-"xattn refresh"}
CELLS=()
for mo in $MODULES; do for ar in $ARMS; do CELLS+=("$mo:$ar"); done; done

IDX=${SLURM_ARRAY_TASK_ID:-0}
echo "=== in-loop read gain | ${#CELLS[@]} cells | index $IDX ==="
for i in "${!CELLS[@]}"; do echo "    [$i] ${CELLS[$i]}"; done
echo "    for THIS cell list:  sbatch --array=0-$(( ${#CELLS[@]} - 1 ))"
if [ "$IDX" -ge "${#CELLS[@]}" ]; then
    echo "no cell at index $IDX.  Pass --array=0-$(( ${#CELLS[@]} - 1 ))."
    exit 0
fi

MODULE=${CELLS[$IDX]%%:*}
ARM=${CELLS[$IDX]##*:}
case "$ARM" in
    a2)  RUN=${RUN_PREFIX}p1-a2-accum-w${ACCUM_VECS}-cc${CROSS_CHUNKS}-z
     PREFIX_MODE=accum ;;
a3z) RUN=${RUN_PREFIX}p1-a3z-gated-w${ACCUM_VECS}k${GATE_SLOTS}-cc${CROSS_CHUNKS}-z
     PREFIX_MODE=gated ;;
*)   echo "ERROR: '$ARM' has no Z channel.  Only a2 and a3z carry one."; exit 1;;
esac
RUN=${RUN_OVERRIDE:-$RUN}

# NUMERIC sort on the step, so checkpoint_91952 does not lose to checkpoint_9999.
CKPT=$(ls -d $OUT_ROOT/$RUN/checkpoint_* 2>/dev/null | sort -t_ -k2 -n | tail -1)
if [ -z "$CKPT" ] || [ ! -d "$CKPT" ]; then
    echo "ERROR: no checkpoint for arm '$ARM' (looked under $OUT_ROOT/$RUN)."
    echo "       RUN_PREFIX='$RUN_PREFIX' -- set RUN_PREFIX= (empty) for a"
    echo "       trained cell, or RUN_OVERRIDE=<name> for anything else."
    exit 1
fi
if [ ! -d "$DATA" ]; then echo "ERROR: no such pack: DATA=$DATA"; exit 1; fi

ACCUM_MAX=${ACCUM_MAX:-$(( ACCUM_VECS * CROSS_CHUNKS ))}
SETS="--set use_memory=true --set memory_slots=0"
SETS="$SETS --set prefix_memory=$PREFIX_MODE --set accum_vecs=$ACCUM_VECS"
if [ "$PREFIX_MODE" = "accum" ]; then
    SETS="$SETS --set accum_max=$ACCUM_MAX"
else
    SETS="$SETS --set gate_slots=$GATE_SLOTS --set gate_route=${GATE_ROUTE:-ring}"
    SETS="$SETS --set gate_norm=${GATE_NORM:-tanh} --set gate_init=${GATE_INIT:-zero}"
    SETS="$SETS --set gate_fill=${GATE_FILL:-grow}"
fi
SETS="$SETS --set latent_carry=true"
SETS="$SETS --set latent_depth_rule=${LATENT_RULE:-absolute}"
SETS="$SETS --set latent_depth_lo=${LATENT_LO:-2} --set latent_depth_hi=${LATENT_HI:-9}"
SETS="$SETS --set latent_renorm=${LATENT_RENORM:-none}"
# THE READ SITE.  latent_s0_read=false is the point: the s0 substitution is
# measured dead, and leaving it on would mean the sweep's intact cell already
# carried a second, known-inert read.  One site at a time.
SETS="$SETS --set latent_s0_read=false --set latent_read=$MODULE"
SETS="$SETS --set latent_read_depth=$READ_DEPTH"
SETS="$SETS --set latent_read_heads=$READ_HEADS"

TIER=$([ "$MODULE" = "xattn" ] && echo "1" || echo "0.5")
OUT=$OUT_BASE/$MODULE-$ARM
echo ""
echo "    tier $TIER | module=$MODULE  arm=$ARM"
echo "    checkpoint: $CKPT"
echo "    data:       $DATA   (READ THIS LINE -- the only proof the pack arrived)"
echo "    geometry:   $PREFIX_MODE  W=$ACCUM_VECS  latent=ON  s0_read=OFF"
echo "    T=$MAX_MEAN_REC  n_chunks=$CROSS_CHUNKS  n=$MAX_EXAMPLES"
echo "    gates:      $GATES   (read the |d|/||x|| column, NOT these -- RED 11)"
echo "    out:        $OUT"
echo ""
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

python evals/diag_readinto_gain.py \
    --model_name $MODEL --checkpoint "$CKPT" --data $DATA $SETS \
    --n_chunks $CROSS_CHUNKS --T $MAX_MEAN_REC --module $MODULE \
    --gates $GATES --max_examples $MAX_EXAMPLES \
    --dtype float32 --out_dir "$OUT"
RC=$?

echo ""
if [ $RC -eq 0 ]; then
    echo "cell $MODULE/$ARM DONE -> $OUT/readinto_gain.json"
    echo ""
    echo "READING ORDER, and it is not the order the cells finish in:"
    echo "  1. Is the intact loss below chance?  If not, nothing else in the"
    echo "     table is a result (RED 10)."
    echo "  2. Does the |d|/||x|| column move with the gate?  If not, the axis"
    echo "     is fiction and the verdict is meaningless."
    echo "  3. Do BOTH arms agree in the in-distribution window?  Only then is"
    echo "     the reading about the SITE rather than about a buffer."
    echo "  4. Only then: LIVE -> tier 1.5 (~8 GPU-hours).  DEAD -> audit the"
    echo "     instrument, do NOT spend the 8 hours."
else
    echo "cell $MODULE/$ARM FAILED rc=$RC"
fi
exit $RC
