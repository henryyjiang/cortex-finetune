#!/bin/bash
#SBATCH --account=gts-aivanova7-lab
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H200:1
#SBATCH --mem=64GB
#SBATCH -t 1:00:00
#SBATCH -q inferno
#SBATCH -J s0_sens
#SBATCH -o logs/Report-%A_%a.out
#SBATCH --array=0-1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=henryyjiang42@gmail.com

# IS s0 A READ SITE AT ALL?  No training.  Two cells, minutes each.
#
#   sbatch pace/diag_s0.sbatch                       # a2 and a3z
#   ARMS=a2 sbatch --array=0-0 pace/diag_s0.sbatch   # just the accum arm
#
# WHY THIS RUNS BEFORE ANY Z REDESIGN.  P2.1 put donor-Z and random-Z at 0.0000
# on both dual arms -- 24 numbers, all |.| <= 0.0014 -- while random-E on the
# same rows moved 0.66-1.9 nats, and tests/test_influence_horizon.py pins the z
# limb of the damage path as landing (a091813).  So the Z null is real, and it
# has two explanations that call for OPPOSITE fixes:
#
#   A. the loop is PATH-INDEPENDENT at these columns.  A recurrent-depth model
#      is trained from fresh trunc_normal_ noise at s0 on every forward -- that
#      is the property that makes variable-depth inference work -- so it may
#      wash out anything put there, at any scale.  Then s0 is the wrong read
#      SITE, no renorm helps, and the fix is an in-loop read
#      (CortexGraft.read_into, which fires every iteration and is bypassed
#      entirely in prefix mode, cortex_graft.py:303).
#   B. s0 is live and Z is just too QUIET.  P0.1 measured the staggered delta
#      at 0.90x the noise it displaces.  Then the fix is latent_renorm=s0 plus
#      a scale: one config flag, no parameters, no new module.
#
# The sweep substitutes noise into s0's carried columns at 0, 0.1, 1, 10 and
# 100 x the MEASURED ||s0||, and reads the span.  Flat across three orders of
# magnitude is (A).  Anything else is (B).
#
# THE NO-PARAMETERS RULE IS NOT AT STAKE HERE, and it is worth saying so in the
# file that tests it.  The +0.0044 latent arm that rule comes from ran K=4 AND
# freeze_loop=true AND a zero-init out_proj, three factors at once, and the one
# later isolated was freeze_loop -- B2 flipped it and got x0.90 -> x1.33, the
# largest mechanism effect in the project.  Zero-init per se is used
# deliberately and successfully elsewhere in this codebase (LoRA B=0;
# gate_init=zero on the gated ring P2.1 just measured real I(d) effects on).
# The failure mode is a zero-init read whose GRADIENT PATH IS SEVERED.  So a
# read module is admissible if its read path trains and its gradient survives
# the no-grad prefix -- which is exactly what read_into gives and latent_init
# cannot.
#
# --T 8 IS NOT OPTIONAL.  config.mean_recurrence is 32 on every B2-family
# checkpoint and no arm ran there.
#
# EVAL MODE.  The tool reports its own intact loss (RED 10): the walk and gate
# 4 run inner.train() and score these checkpoints at 11.4-11.97 against
# ln(vocab) 11.5157, while the horizon runs .eval() and gets 3.19.  A cell that
# prints "*** AT CHANCE" is not a result about s0.

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
OUT_BASE=${OUT_BASE:-eval_results/s0_sensitivity-$STAMP}

ACCUM_VECS=${ACCUM_VECS:-16}
GATE_SLOTS=${GATE_SLOTS:-64}
CROSS_CHUNKS=${CROSS_CHUNKS:-8}
MAX_MEAN_REC=${MAX_MEAN_REC:-8}
MAX_EXAMPLES=${MAX_EXAMPLES:-50}
SCALES=${SCALES:-"0 0.1 1 10 100"}

# Only the DUAL arms have a Z to read.  The tool refuses an E-only arm rather
# than printing a flat table that would read as (A) by construction.
ARMS=${ARMS:-"a2 a3z"}
CELLS=($ARMS)

IDX=${SLURM_ARRAY_TASK_ID:-0}
echo "=== s0 read-site sweep | ${#CELLS[@]} cells | index $IDX ==="
for i in "${!CELLS[@]}"; do echo "    [$i] ${CELLS[$i]}"; done
echo "    for THIS cell list:  sbatch --array=0-$(( ${#CELLS[@]} - 1 ))"
if [ "$IDX" -ge "${#CELLS[@]}" ]; then
    echo "no cell at index $IDX.  Pass --array=0-$(( ${#CELLS[@]} - 1 ))."
    exit 0
fi

ARM=${CELLS[$IDX]}
case "$ARM" in
a2)  RUN=probe-p1-a2-accum-w${ACCUM_VECS}-cc${CROSS_CHUNKS}-z
     PREFIX_MODE=accum ;;
a3z) RUN=probe-p1-a3z-gated-w${ACCUM_VECS}k${GATE_SLOTS}-cc${CROSS_CHUNKS}-z
     PREFIX_MODE=gated ;;
*)   echo "ERROR: '$ARM' has no Z channel.  Only a2 and a3z carry one."; exit 1;;
esac

# NUMERIC sort on the step, so checkpoint_91952 does not lose to checkpoint_9999.
CKPT=$(ls -d $OUT_ROOT/$RUN/checkpoint_* 2>/dev/null | sort -t_ -k2 -n | tail -1)
if [ -z "$CKPT" ] || [ ! -d "$CKPT" ]; then
    echo "ERROR: no checkpoint for arm '$ARM' (looked under $OUT_ROOT/$RUN)."
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

OUT=$OUT_BASE/$ARM
echo ""
echo "    arm=$ARM   checkpoint: $CKPT"
echo "    data:       $DATA   (READ THIS LINE -- the only proof the pack arrived)"
echo "    geometry:   $PREFIX_MODE  W=$ACCUM_VECS  latent=ON"
echo "    T=$MAX_MEAN_REC  n_chunks=$CROSS_CHUNKS  n=$MAX_EXAMPLES"
echo "    scales:     $SCALES  (x the measured ||s0||)"
echo "    out:        $OUT"
echo ""
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

python evals/diag_s0_sensitivity.py \
    --model_name $MODEL --checkpoint "$CKPT" --data $DATA $SETS \
    --n_chunks $CROSS_CHUNKS --T $MAX_MEAN_REC \
    --scales $SCALES --max_examples $MAX_EXAMPLES \
    --dtype float32 --out_dir "$OUT"
RC=$?

echo ""
if [ $RC -eq 0 ]; then
    echo "cell $ARM DONE -> $OUT/results.json"
    echo ""
    echo "(A) and (B) send the Z work in opposite directions.  Read BOTH arms"
    echo "before choosing: a2 is accum and a3z is the gated ring, and only a"
    echo "reading that holds on both is about s0 rather than about a buffer."
else
    echo "cell $ARM FAILED rc=$RC"
fi
exit $RC
