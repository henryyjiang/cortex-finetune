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
#   R=$(ENCODING=tokens sbatch --parsable --dependency=afterany:$J pace/j1_readout.sbatch)
#   RE=$(ENCODING=tokens SET=edrop sbatch --parsable --array=0-3 \
#        --dependency=afterany:$E pace/j1_readout.sbatch)      # the E-dropout pair
#   then pace/j1_verdict.sbatch (CPU) with --dependency=afterany on both.
#
# J3 (the in-loop scratchpad, pace/j3_joint.sbatch) is read out by THIS file
# with EXPERIMENT=j3: same limbs, packs, cells and task table, so the decision
# rules stay one piece of code.  ENCODING defaults to scratch there, and the
# limbs get J3's flags (latent_read=scratch; no-read = latent_carry_read=false).
#   R3=$(EXPERIMENT=j3 sbatch --parsable --dependency=afterany:$J3 pace/j1_readout.sbatch)
#   R3E=$(EXPERIMENT=j3 SET=edrop sbatch --parsable --array=0-3 \
#         --dependency=afterany:$J3E pace/j1_readout.sbatch)
# GATE_LR_MULT MUST equal the training run's (the stored gate is gate/mult);
# each task checks it against the run's cortex_diag.jsonl and refuses.
#
# J4 (Z as its own input_embeds columns, pace/j4_joint.sbatch) is read out the
# same way with EXPERIMENT=j4.  ENCODING defaults to endpoint and ZNORM_TARGET
# to 171.0 -- E's MEASURED carried row norm, which is the rescale J4's read
# depends on.  Like GATE_LR_MULT it is a CONFIG FLAG and not a stored
# parameter, so each task checks it against the run's own diag rows and refuses
# on a mismatch: rebuilt at the 3.0 default, Z would enter 57x too small and
# the read-out would score a model nobody trained.
#   R4=$(EXPERIMENT=j4 sbatch --parsable --dependency=afterany:$J4 pace/j1_readout.sbatch)
#   R4E=$(EXPERIMENT=j4 SET=edrop sbatch --parsable --array=0-3 \
#         --dependency=afterany:$J4E pace/j1_readout.sbatch)
#   R4Z=$(EXPERIMENT=j4 SET=edrop E_DROPOUT=0.95 sbatch --parsable --array=0-3 \
#         --dependency=afterany:$J4Z pace/j1_readout.sbatch)   # the Z-only limb
#
# afterany, not afterok: every task refuses on its own when its limb has no
# checkpoint_$STEP, so a limb that crashed costs a one-minute refusal and a
# verdict that names it -- where afterok would hold EVERY read-out, including
# the limbs that finished, behind one failure.
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
EXPERIMENT=${EXPERIMENT:-j1}
case "$EXPERIMENT" in
    j1) ENCODING=${ENCODING:-tokens} ;;
    j3) ENCODING=${ENCODING:-scratch}
        GATE_LR_MULT=${GATE_LR_MULT:-0.1}      # pace/j3_joint.sbatch's default
        SCRATCH_FB=${SCRATCH_FB:-1.0} ;;
    j4) ENCODING=${ENCODING:-endpoint}         # D3's vindicated write
        ZNORM_TARGET=${ZNORM_TARGET:-136.0} ;; # ||E||, pace/j4_joint.sbatch's
    *) echo "ERROR: EXPERIMENT must be j1|j3|j4; got '$EXPERIMENT'"; exit 1 ;;
esac
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

RUN=${EXPERIMENT}-a3z-${ENCODING}${TAG}-${LIMB}
CKPT=$OUT_ROOT/$RUN/checkpoint_$STEP
case "$PACK" in
    carry) DATA=$CARRY_DATA ;;
    pg19)  DATA=$PG19_DATA ;;
esac
OUT=eval_results/${EXPERIMENT}_readout/$RUN/${PACK}-${SCORE}-${ZNULL}

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
if [ "$EXPERIMENT" = "j3" ]; then
    # J3: every limb reads the scratchpad; the limbs differ in the CARRY alone.
    SETS="$SETS --set latent_read=scratch --set latent_read_heads=$READ_HEADS"
    SETS="$SETS --set latent_read_znorm=rms --set latent_read_znorm_target=$ZNORM_TARGET"
    SETS="$SETS --set latent_read_gate_lr_mult=$GATE_LR_MULT"
    SETS="$SETS --set latent_scratch_forget_bias=$SCRATCH_FB"
    case "$LIMB" in
        real)   ;;
        donor)  SETS="$SETS --set latent_read_scramble=true"
                ALLOW="--allow_scrambled" ;;
        noread) SETS="$SETS --set latent_carry_read=false" ;;
    esac
elif [ "$EXPERIMENT" = "j4" ]; then
    # J4: the carried Z is spliced as its own columns of input_embeds.  Every
    # limb has the columns; the limbs differ in what is IN them.
    SETS="$SETS --set latent_read=embeds"
    SETS="$SETS --set latent_read_znorm=rms --set latent_read_znorm_target=$ZNORM_TARGET"
    case "$LIMB" in
        real)   ;;
        donor)  SETS="$SETS --set latent_read_scramble=true"
                ALLOW="--allow_scrambled" ;;
        # NOT latent_write_only: that omits the columns.  The no-read limb keeps
        # them and they are zero, so it differs from the real limb in the carry
        # and not in the geometry (cortex_graft.prefix_pack).
        noread) SETS="$SETS --set latent_carry_read=false" ;;
    esac
else
case "$LIMB" in
    real)   SETS="$SETS --set latent_read=xattn --set latent_read_heads=$READ_HEADS"
            SETS="$SETS --set latent_read_znorm=rms --set latent_read_znorm_target=$ZNORM_TARGET" ;;
    donor)  SETS="$SETS --set latent_read=xattn --set latent_read_heads=$READ_HEADS"
            SETS="$SETS --set latent_read_znorm=rms --set latent_read_znorm_target=$ZNORM_TARGET"
            SETS="$SETS --set latent_read_scramble=true"
            ALLOW="--allow_scrambled" ;;
    noread) SETS="$SETS --set latent_read=none --set latent_write_only=true" ;;
esac
fi
CELLS_ARG=""
if [ "$CELLS" != "-" ]; then CELLS_ARG="--cells $CELLS"; fi

echo "=== $EXPERIMENT read-out | SET=$SET task $IDX = $KEY ==="
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
if [ "$EXPERIMENT" = "j3" ]; then
    # The stored gate parameter is gate/mult: rebuilt at another mult, the
    # eval would score the read at the wrong strength and say nothing.  The
    # run's own diag rows record the mult it trained with.
    DIAG=$OUT_ROOT/$RUN/cortex_diag.jsonl
    [ -f "$DIAG" ] || { echo "ERROR: no $DIAG -- cannot confirm the trained gate_lr_mult"; exit 1; }
    python - "$DIAG" "$GATE_LR_MULT" <<'MULT' || exit 1
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
got = {r.get("z_gate_lr_mult") for r in rows} - {None}
want = float(sys.argv[2])
if got != {want}:
    sys.exit(f"ERROR: the run trained with gate_lr_mult {sorted(got)} but this "
             f"read-out sets {want}.  Pass GATE_LR_MULT=<the run's value>.")
print(f"    gate_lr_mult {want} matches the run's diag rows")
MULT
fi
if [ "$EXPERIMENT" = "j4" ]; then
    # The rescale target is a CONFIG FLAG, not a stored parameter: rebuilt at
    # another value the read enters at another strength and the read-out scores
    # a model nobody trained.  Same failure class as J3's gate_lr_mult, and the
    # gap here is 171 vs the 3.0 default -- a factor of 57.
    DIAG=$OUT_ROOT/$RUN/cortex_diag.jsonl
    [ -f "$DIAG" ] || { echo "ERROR: no $DIAG -- cannot confirm the trained znorm target"; exit 1; }
    python - "$DIAG" "$ZNORM_TARGET" <<'ZNORM' || exit 1
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
got = {r.get("z_znorm_target") for r in rows} - {None}
want = float(sys.argv[2])
if not got:
    sys.exit("ERROR: the run's diag rows carry no z_znorm_target.  That field "
             "was added with J4; a run without it predates the build this "
             "read-out assumes.  Re-check the training run.")
if got != {want}:
    sys.exit(f"ERROR: the run trained with znorm_target {sorted(got)} but this "
             f"read-out sets {want}.  Pass ZNORM_TARGET=<the run's value>.")
print(f"    znorm_target {want} matches the run's diag rows")
ZNORM
fi
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
    echo "$EXPERIMENT read-out $KEY ($RUN) DONE -> $OUT/results.json"
    echo "  Numbers are READ ONLY through evals/score_j1.py (pace/j1_verdict.sbatch)."
else
    echo "$EXPERIMENT read-out $KEY ($RUN) FAILED rc=$RC"
fi
exit $RC
