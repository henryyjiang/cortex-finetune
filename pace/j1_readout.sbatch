#!/bin/bash
#SBATCH --account=gts-aivanova7-lab
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H200:1
#SBATCH --mem=64GB
#SBATCH -t 1:30:00
#SBATCH -q inferno
#SBATCH -J j1_readout
#SBATCH -o logs/Report-%A_%a.out
#SBATCH --array=0-9
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=henryyjiang42@gmail.com

# J1's PRE-REGISTERED READ-OUT -- ../j1_prereg.md.  One eval_carry_2x2 run per
# array task; evals/score_j1.py pairs them across limbs and applies the rules.
#
#   R=$(ENCODING=tokens sbatch --parsable --dependency=afterok:$J pace/j1_readout.sbatch)
#   RE=$(ENCODING=tokens SET=edrop sbatch --parsable --array=0-3 \
#        --dependency=afterok:$E pace/j1_readout.sbatch)       # the E-dropout pair
#   then pace/j1_verdict.sbatch (CPU) with --dependency=afterany on both.
#
# EACH LIMB IN ITS OWN TRAINED CONDITION, E on (score_j1.py's header):
#   real E1Z1 | donor E1Z0 under --z_null donor | noread E1Z1.
# The main limbs run only E1Z1,E1Z0: they trained with E always on, so their
# E-off cells are an untrained condition.  The E-dropout pair runs all four.
#
# THE TASK TABLE IS evals/score_j1.py's MAIN_TASKS / EDROP_TASKS, row for row
# (tests/test_j1_readout.py parses this file and compares).  Change one, change
# both, and re-run the test before submitting.
#   key  limb  pack  score  z_null  n  cells ('-' = all four)

cd $SLURM_SUBMIT_DIR

module load anaconda3
module load cuda/12.1.1
conda activate cortex-retro

export SCRATCH=${SCRATCH:-$HOME/scratch}
export HF_HOME=$SCRATCH/hf_cache
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MAIN_TASKS=(
    "real_carry_off real carry carry off 300 E1Z1,E1Z0"
    "real_carry_donor real carry carry donor 300 E1Z1,E1Z0"
    "donor_carry_donor donor carry carry donor 300 E1Z1,E1Z0"
    "noread_carry_off noread carry carry off 300 E1Z1,E1Z0"
    "real_local_off real carry local off 100 E1Z1,E1Z0"
    "noread_local_off noread carry local off 100 E1Z1,E1Z0"
    "real_pg19_off real pg19 all off 400 E1Z1,E1Z0"
    "real_pg19_donor real pg19 all donor 400 E1Z1,E1Z0"
    "donor_pg19_donor donor pg19 all donor 400 E1Z1,E1Z0"
    "noread_pg19_off noread pg19 all off 400 E1Z1,E1Z0"
)
EDROP_TASKS=(
    "real_carry_off real carry carry off 300 -"
    "real_carry_donor real carry carry donor 300 -"
    "noread_carry_off noread carry carry off 300 -"
    "real_local_off real carry local off 100 -"
)

MODEL=${MODEL:-ckpts/olmo-retrofit-cortex}
OUT_ROOT=${OUT_ROOT:-cortex-retrofit}
ENCODING=${ENCODING:-tokens}
SET=${SET:-main}
E_DROPOUT=${E_DROPOUT:-0.25}
STEP=${STEP:-93552}               # 91552 + CELL_STEPS 2000: the stop_at_step save
CARRY_DATA=${CARRY_DATA:-data/carry_task_r16_len4096_val}
PG19_DATA=${PG19_DATA:-data/pg19_olmo_validation_len4096_strided}
READ_HEADS=${READ_HEADS:-8}
ZNORM_TARGET=${ZNORM_TARGET:-3.0}
TOK_POOL=${TOK_POOL:-4}
BOOT=${BOOT:-2000}

case "$SET" in
    main)  TASKS=("${MAIN_TASKS[@]}");  TAG="" ;;
    edrop) TASKS=("${EDROP_TASKS[@]}"); TAG="-edrop${E_DROPOUT}" ;;
    *) echo "ERROR: SET must be main|edrop; got '$SET'"; exit 1 ;;
esac
IDX=${SLURM_ARRAY_TASK_ID:-0}
if [ "$IDX" -ge "${#TASKS[@]}" ]; then
    echo "no task at index $IDX in SET=$SET (${#TASKS[@]} tasks: --array=0-$(( ${#TASKS[@]} - 1 )))"
    exit 0
fi
read KEY LIMB PACK SCORE ZNULL N CELLS <<< "${TASKS[$IDX]}"

RUN=j1-a3z-${ENCODING}${TAG}-${LIMB}
CKPT=$OUT_ROOT/$RUN/checkpoint_$STEP
case "$PACK" in
    carry) DATA=$CARRY_DATA ;;
    pg19)  DATA=$PG19_DATA ;;
esac
OUT=eval_results/j1_readout/$RUN/${PACK}-${SCORE}-${ZNULL}

# The graft has to be FORCED to the trained build: --model_name loads the base
# dir's config, which carries no cortex flags at all.  Same geometry as
# pace/j1_joint.sbatch; e_dropout is training-only and is not set.
SETS="--set use_memory=true --set memory_slots=0 --set prefix_memory=gated"
SETS="$SETS --set accum_vecs=16 --set gate_slots=64 --set gate_route=ring"
SETS="$SETS --set gate_norm=tanh --set gate_init=zero --set gate_fill=grow"
SETS="$SETS --set latent_carry=true --set latent_depth_rule=absolute"
SETS="$SETS --set latent_depth_lo=2 --set latent_depth_hi=9 --set latent_renorm=none"
SETS="$SETS --set latent_encoding=$ENCODING --set latent_tok_pool=$TOK_POOL"
SETS="$SETS --set latent_s0_read=false"
ALLOW=""
case "$LIMB" in
    real)   SETS="$SETS --set latent_read=xattn --set latent_read_heads=$READ_HEADS"
            SETS="$SETS --set latent_read_znorm=rms --set latent_read_znorm_target=$ZNORM_TARGET" ;;
    donor)  SETS="$SETS --set latent_read=xattn --set latent_read_heads=$READ_HEADS"
            SETS="$SETS --set latent_read_znorm=rms --set latent_read_znorm_target=$ZNORM_TARGET"
            SETS="$SETS --set latent_read_scramble=true"
            ALLOW="--allow_scrambled" ;;
    noread) SETS="$SETS --set latent_read=none --set latent_write_only=true" ;;
esac
CELLS_ARG=""
if [ "$CELLS" != "-" ]; then CELLS_ARG="--cells $CELLS"; fi

echo "=== J1 read-out | SET=$SET task $IDX = $KEY ==="
echo "    run:        $RUN"
echo "    checkpoint: $CKPT"
echo "    data:       $DATA   (score $SCORE, z_null $ZNULL, n $N, cells ${CELLS})"
echo "    out:        $OUT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

[ -f "$CKPT/chkpt.pt" ] || {
    echo "ERROR: no $CKPT/chkpt.pt.  J1 saves it at stop_at_step $STEP; present:"
    ls -d $OUT_ROOT/$RUN/checkpoint_* 2>/dev/null || echo "  (no checkpoint dirs at all)"
    exit 1; }
[ -d "$DATA" ] || { echo "ERROR: no such pack: $DATA"; exit 1; }
if ! grep -q "read_into(x, current_step)" "$MODEL/raven_modeling_minimal_cortex.py" 2>/dev/null; then
    echo "ERROR: $MODEL carries a PRE-P3.0 modeling file.  Refresh it:"
    echo "         python tools/prepare_cortex_checkpoint.py --src $MODEL --dst $MODEL --variant olmo"
    exit 1
fi

python evals/eval_carry_2x2.py \
    --model_name "$MODEL" --checkpoint "$CKPT" --data "$DATA" $SETS \
    --n_chunks 8 --T 8 --max_examples "$N" --boot "$BOOT" \
    --dtype float32 --z_null "$ZNULL" --score "$SCORE" $CELLS_ARG $ALLOW \
    --out_dir "$OUT"
RC=$?
echo ""
if [ $RC -eq 0 ]; then
    echo "J1 read-out $KEY ($RUN) DONE -> $OUT/results.json"
    echo "  Numbers are READ ONLY through evals/score_j1.py (pace/j1_verdict.sbatch)."
else
    echo "J1 read-out $KEY ($RUN) FAILED rc=$RC"
fi
exit $RC
