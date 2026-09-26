#!/bin/bash
#SBATCH --account=gts-aivanova7-lab
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H200:1
#SBATCH --mem=64GB
#SBATCH -t 1:30:00
#SBATCH -q inferno
#SBATCH -J j5_readout
#SBATCH -o logs/Report-%A_%a.out
#SBATCH --array=0-12
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=henryyjiang42@gmail.com

# J5's PRE-REGISTERED READ-OUT -- ../j5_prereg.md S2.  One eval_carry_2x2 run
# per array task; evals/score_j5.py pairs them and applies the rules.
#
#   R2=$(STEP=2000 sbatch --parsable --dependency=afterany:$J pace/j5_readout.sbatch)
#   R4=$(STEP=4000 sbatch --parsable --dependency=afterany:$J pace/j5_readout.sbatch)
#   sbatch --dependency=afterany:$R2:$R4 pace/j5_verdict.sbatch
#
# BOTH STEPS ARE RUN, and that is the design (../j5_prereg.md S1.5): the 2,000
# -> 4,000 DIRECTION is the first evidence this project would have that Z
# improves with budget at all, and it is what a 5B run would be betting on.
# The step lands in the output path, so the two read-outs cannot overwrite each
# other.
#
# afterany, not afterok: every task refuses on its own when its limb has no
# checkpoint_$STEP, so a limb that crashed costs a one-minute refusal and a
# verdict that names it -- where afterok would hold every read-out, including
# the limbs that finished, behind one failure.
#
# ===== EACH LIMB IS SCORED IN ITS OWN TRAINED CONDITION =====================
# This is the whole reason the cell lists differ per limb, and getting it wrong
# reports an out-of-distribution number as the arm's result.
#
#   limb    trained condition        its operating cell   its ablation cells
#   zonly   E's rows are zeros       E0Z1                 E0Z0 (Z off / donor)
#   eonly   Z's columns are zeros    E1Z0                 E0Z0 (E off)
#   both    both channels live       E1Z1                 E1Z0 (Z off / donor)
#
# WHY `zonly` IS NOT REBUILT WITH e_carry_read=false, and why that is exact
# rather than approximate.  The E0 cell zeros the E half of the INCOMING CARRY
# (evals/eval_carry_2x2.null_e), so the E rows spliced into the prefix are
# exactly zeros -- the same tensor training spliced.  It differs from the
# training flag only in what the RING then holds, and the two channels merge
# through SEPARATE gates with no cross-talk (cortex_memory/buffers.gate_pair:
# gate(e_s, e_c) and gate_latent(z_s, z_c)), so nothing that touches Z changes.
# Rebuilding with the flag instead would zero E in EVERY cell and collapse E1
# onto E0, costing the limb's E cells for nothing.
#
# `eonly` IS rebuilt with latent_carry_read=false, because there the flag is
# structural: without it the eval would feed a real Z to a model that never
# read one.  Its E1Z1 and E1Z0 cells must then agree to floating point -- which
# is the instrument check evals/score_j5.py applies (NOREAD_ZMAIN_TOL).
#
# THE DONOR CELL IS WHY THIS FILE HAS TWO PASSES PER LIMB.  Within one run,
# `--z_null donor` makes the Z-OFF cell feed ANOTHER row's carried Z at the
# same chunk index while the Z-ON cell keeps its own; `--z_null off` makes the
# Z-off cell skip the read entirely.  Pairing those two passes row by row gives
# the two numbers that decide (../j5_prereg.md S2):
#   X_off      = NLL(Z skipped)  - NLL(own Z)   -- are the columns load-bearing?
#   X_content  = NLL(donor's Z)  - NLL(own Z)   -- is what they hold CONTENT?
# J4 measured +0.4716 and +0.0001 on that pair.  Load-bearing, content-free.
#
# THE TASK TABLE IS evals/score_j5.py's TASKS, row for row
# (tests/test_j5_arm.py parses this file and compares).  Change one, change
# both, and re-run the test before submitting.
#   key  limb  pack  score  z_null  n  cells

cd $SLURM_SUBMIT_DIR

module load anaconda3
module load cuda/12.1.1
conda activate cortex-retro

export SCRATCH=${SCRATCH:-$HOME/scratch}
export HF_HOME=$SCRATCH/hf_cache
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

TASKS=(
    "zonly_carry_off zonly carry carry off 300 E0Z1,E0Z0"
    "zonly_carry_donor zonly carry carry donor 300 E0Z1,E0Z0"
    "zonly_local_off zonly carry local off 100 E0Z1,E0Z0"
    "zonly_pg19_off zonly pg19 all off 400 E0Z1,E0Z0"
    "zonly_pg19_donor zonly pg19 all donor 400 E0Z1,E0Z0"
    "eonly_carry_off eonly carry carry off 300 E1Z1,E1Z0,E0Z0"
    "eonly_local_off eonly carry local off 100 E1Z1,E1Z0"
    "eonly_pg19_off eonly pg19 all off 400 E1Z0,E0Z0"
    "both_carry_off both carry carry off 300 E1Z1,E1Z0,E0Z1,E0Z0"
    "both_carry_donor both carry carry donor 300 E1Z1,E1Z0,E0Z1,E0Z0"
    "both_local_off both carry local off 100 E1Z1,E1Z0"
    "both_pg19_off both pg19 all off 400 E1Z1,E1Z0"
    "both_pg19_donor both pg19 all donor 400 E1Z1,E1Z0"
)

MODEL=${MODEL:-ckpts/olmo8-cortex}
OUT_ROOT=${OUT_ROOT:-cortex-retrofit}
ENCODING=endpoint
STEP=${STEP:-4000}                 # ../j5_prereg.md S1.5: read at 2000 AND 4000
CARRY_DATA=${CARRY_DATA:-data/carry_task_r16_len4096_val}
PG19_DATA=${PG19_DATA:-data/pg19_olmo_validation_len4096_strided}
TOK_POOL=${TOK_POOL:-4}
BOOT=${BOOT:-2000}

if [ -z "$ZNORM_TARGET" ]; then
    echo "ERROR: ZNORM_TARGET is not set.  It is a CONFIG FLAG and not a stored"
    echo "       parameter, so a read-out rebuilt at the 3.0 default would"
    echo "       enter Z ~45x too small and score a model nobody trained."
    echo "       Pass the value pace/j5_joint.sbatch trained with; each task"
    echo "       checks it against the run's own diag rows and refuses."
    exit 1
fi

IDX=${SLURM_ARRAY_TASK_ID:-0}
if [ "$IDX" -ge "${#TASKS[@]}" ]; then
    echo "no task at index $IDX (${#TASKS[@]} tasks: --array=0-$(( ${#TASKS[@]} - 1 )))"
    exit 0
fi
read KEY LIMB PACK SCORE ZNULL N CELLS <<< "${TASKS[$IDX]}"

RUN=j5-a3z-${ENCODING}-${LIMB}
CKPT=$OUT_ROOT/$RUN/checkpoint_$STEP
case "$PACK" in
    carry) DATA=$CARRY_DATA ;;
    pg19)  DATA=$PG19_DATA ;;
esac
OUT=eval_results/j5_readout/step${STEP}/$RUN/${PACK}-${SCORE}-${ZNULL}

# The graft has to be FORCED to the trained build: --model_name loads the base
# dir's config, which carries no cortex flags at all.  Same geometry as
# pace/j5_joint.sbatch.
SETS="--set use_memory=true --set memory_slots=0 --set prefix_memory=gated"
SETS="$SETS --set accum_vecs=16 --set gate_slots=64 --set gate_route=ring"
SETS="$SETS --set gate_norm=tanh --set gate_init=zero --set gate_fill=grow"
SETS="$SETS --set latent_carry=true --set latent_depth_rule=absolute"
SETS="$SETS --set latent_depth_lo=2 --set latent_depth_hi=9 --set latent_renorm=none"
SETS="$SETS --set latent_encoding=$ENCODING --set latent_tok_pool=$TOK_POOL"
SETS="$SETS --set latent_s0_read=false --set latent_read=embeds"
SETS="$SETS --set latent_read_znorm=rms --set latent_read_znorm_target=$ZNORM_TARGET"
case "$LIMB" in
    # See the header: the E0 cells reproduce zonly's trained splice exactly, so
    # the flag is NOT set here and the limb keeps cells it would otherwise lose.
    zonly) ;;
    # Structural: without it the eval feeds a real Z to a model that never read
    # one.  NOT latent_write_only, which would omit the columns and change the
    # geometry as well as the carry.
    eonly) SETS="$SETS --set latent_carry_read=false" ;;
    both)  ;;
esac

echo "=== J5 read-out | step $STEP | task $IDX = $KEY ==="
echo "    run:        $RUN"
echo "    checkpoint: $CKPT"
echo "    data:       $DATA   (score $SCORE, z_null $ZNULL, n $N, cells ${CELLS})"
echo "    out:        $OUT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

[ -f "$CKPT/chkpt.pt" ] || {
    echo "ERROR: no $CKPT/chkpt.pt.  pace/j5_joint.sbatch saves at"
    echo "  SAVE_INTERVAL 2000 and at stop_at_step 4000; present:"
    ls -d $OUT_ROOT/$RUN/checkpoint_* 2>/dev/null || echo "  (no checkpoint dirs at all)"
    exit 1; }
[ -d "$DATA" ] || { echo "ERROR: no such pack: $DATA"; exit 1; }

# The run's own diag rows are the only record of the two config flags that
# change what this checkpoint IS -- the rescale target, and which channels the
# limb read.  Rebuilt at another value the read-out scores a model nobody
# trained, and says nothing about it.  Same failure class as J3's gate_lr_mult.
DIAG=$OUT_ROOT/$RUN/cortex_diag.jsonl
[ -f "$DIAG" ] || { echo "ERROR: no $DIAG -- cannot confirm the trained config"; exit 1; }
python - "$DIAG" "$ZNORM_TARGET" "$LIMB" <<'CHECK' || exit 1
import json, sys
path, want, limb = sys.argv[1], float(sys.argv[2]), sys.argv[3]
rows = [json.loads(l) for l in open(path) if l.strip()]
got = {r.get("z_znorm_target") for r in rows} - {None}
if not got:
    sys.exit("ERROR: the run's diag rows carry no z_znorm_target -- this run "
             "predates the build this read-out assumes.")
if got != {want}:
    sys.exit(f"ERROR: the run trained with znorm_target {sorted(got)} but this "
             f"read-out sets {want}.  Pass ZNORM_TARGET=<the run's value>.")
print(f"    znorm_target {want} matches the run's diag rows")

# THE LIMB IS THE LIMB, read back off the run rather than trusted from its name
# (../j5_prereg.md S3 vetoes 1-2).  A run dir renamed or reused would otherwise
# be scored under another limb's cells, silently.
ecr = {int(v) for v in (r.get("z_e_carry_read") for r in rows) if v is not None}
cr = {int(v) for v in (r.get("z_carry_read") for r in rows) if v is not None}
want_ecr, want_cr = int(limb != "zonly"), int(limb != "eonly")
if ecr and ecr != {want_ecr}:
    sys.exit(f"ERROR: {limb} trained with e_carry_read {sorted(ecr)}, expected "
             f"{want_ecr}.  This run dir is not the limb its name claims.")
if cr and cr != {want_cr}:
    sys.exit(f"ERROR: {limb} trained with latent_carry_read {sorted(cr)}, "
             f"expected {want_cr}.  This run dir is not the limb its name claims.")
esn = [r["z_e_spliced_norm"] for r in rows if r.get("z_e_spliced_norm") is not None]
if esn:
    if limb == "zonly" and max(abs(v) for v in esn) != 0.0:
        sys.exit(f"ERROR (veto 1): zonly spliced E rows of norm up to "
                 f"{max(esn):.4g} during training -- E's read was not off.")
    if limb != "zonly" and min(esn) <= 0.0:
        sys.exit(f"ERROR (veto 2): {limb} spliced E rows of norm {min(esn):.4g} "
                 f"during training -- E was off on a limb that must carry it.")
print(f"    limb {limb}: e_carry_read {want_ecr}, latent_carry_read {want_cr}, "
      f"confirmed against {len(rows)} diag rows")
CHECK

if ! grep -q "read_into(x, current_step)" "$MODEL/raven_modeling_minimal_cortex.py" 2>/dev/null; then
    echo "ERROR: $MODEL carries a PRE-P3.0 modeling file.  Refresh it:"
    echo "         python tools/prepare_cortex_checkpoint.py --src $MODEL --dst $MODEL --variant olmo"
    exit 1
fi

python evals/eval_carry_2x2.py \
    --model_name "$MODEL" --checkpoint "$CKPT" --data "$DATA" $SETS \
    --n_chunks 8 --T 8 --max_examples "$N" --boot "$BOOT" \
    --dtype float32 --z_null "$ZNULL" --score "$SCORE" --cells "$CELLS" \
    --out_dir "$OUT"
RC=$?
echo ""
if [ $RC -eq 0 ]; then
    echo "J5 read-out $KEY ($RUN, step $STEP) DONE -> $OUT/results.json"
    echo "  Numbers are READ ONLY through evals/score_j5.py (pace/j5_verdict.sbatch)."
else
    echo "J5 read-out $KEY ($RUN, step $STEP) FAILED rc=$RC"
fi
exit $RC
