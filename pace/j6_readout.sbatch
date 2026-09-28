#!/bin/bash
#SBATCH --account=gts-aivanova7-lab
#SBATCH -N 1
#SBATCH --gres=gpu:H200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH -t 3:00:00
#SBATCH -q inferno
#SBATCH -J j6_readout
#SBATCH -o logs/Report-%A_%a.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=henryyjiang42@gmail.com

# ============================================================================
# J6's READ-OUT -- job 2 of 2.
#
#   sbatch --dependency=afterany:<arms jobid> --array=0-3 pace/j6_readout.sbatch
#
# ONE ARRAY TASK PER LIMB, five evals each, run sequentially inside the task.
# J5 split 13 evals across 13 array tasks and then needed a THIRD job to
# score them; grouping by limb keeps the whole arm at two jobs and costs
# nothing, because the five evals of one limb are ~20 min each and the four
# limbs run in parallel.
#
# afterany, not afterok, ON PURPOSE: a limb whose training failed must still
# reach a read-out that NAMES the missing piece, rather than a job that never
# starts.
#
# THERE IS NO VERDICT SCRIPT.  The numbers are dumped raw by
# tools/dump_j6.py, which computes cells and CIs and prints them -- it has no
# thresholds and returns no label.  That is deliberate: J5's scorer headlined
# "NO" while its own fields supported NOT_READ, because the pre-registered
# gate (local_nll) could not detect the failure it existed to catch.  Reading
# four limbs by hand is cheaper than writing a decision table that can be
# wrong in that direction again.
#
# ALL FOUR LIMBS TRAINED WITH BOTH CHANNELS LIVE, so every limb is scored in
# the same cell as J4's real limb: operating E1Z1, Z-ablated E1Z0, plus E0Z0
# for E's own margin.  The limbs differ in how Z was trained, not in what the
# read-out asks of it -- which is what makes them directly comparable.
#
#   carry-carry-off     E1Z1,E1Z0,E0Z0   300   X_off, E_margin, absolute
#   carry-carry-donor   E1Z1,E1Z0        300   X_content  <- THE deciding cell
#   carry-local-off     E1Z1,E1Z0        100   the task's positive control
#   pg19-all-off        E1Z1,E1Z0        400   off-task load
#   pg19-all-donor      E1Z1,E1Z0        400   off-task content
#
# `slow` is the one limb whose read-out could silently differ from its
# training: latent_stride decides WHICH ROWS hold which chunk, so a rebuild
# without it reads the long-horizon limb's Z on the short-horizon schedule.
# It is in prepare_eval_checkpoint.py's CORTEX_FLAGS now, and it is also
# forced here through --set, and the diag check below refuses a mismatch.
# Three places, because this class of bug has cost this project two runs.
# ============================================================================

cd $SLURM_SUBMIT_DIR

module load anaconda3
module load cuda/12.1.1
conda activate cortex-retro

export SCRATCH=${SCRATCH:-$HOME/scratch}
export HF_HOME=$SCRATCH/hf_cache
export HF_HUB_OFFLINE=1

MODEL=${MODEL:-ckpts/olmo-retrofit-cortex}
OUT_ROOT=${OUT_ROOT:-cortex-retrofit}
STEP=${STEP:-93552}                        # 91,552 branch + 2,000
CARRY_DATA=${CARRY_DATA:-data/carry_task_r16_len4096_val}
PG19_DATA=${PG19_DATA:-data/pg19_olmo_validation_len4096_strided}
ZNORM_TARGET=${ZNORM_TARGET:-136.0}
MIX_P=${MIX_P:-0.25}
STRIDE=${STRIDE:-2}
BOOT=${BOOT:-2000}

LIMBS=(ctrl mix stag slow)
IDX=${SLURM_ARRAY_TASK_ID:-0}
LIMB=${LIMBS[$IDX]}
if [ -z "$LIMB" ]; then echo "no limb at index $IDX (use --array=0-3)"; exit 0; fi

RUN=j6-a3z-${LIMB}
CKPT=$OUT_ROOT/$RUN/checkpoint_$STEP
DIAG=$OUT_ROOT/$RUN/cortex_diag.jsonl

echo "=== J6 read-out | limb=$LIMB | step $STEP ==="
echo "    run:        $RUN"
echo "    checkpoint: $CKPT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

[ -f "$CKPT/chkpt.pt" ] || {
    echo "ERROR: no $CKPT/chkpt.pt.  Present:"
    ls -d $OUT_ROOT/$RUN/checkpoint_* 2>/dev/null || echo "  (no checkpoint dirs at all)"
    exit 1; }
[ -f "$DIAG" ] || { echo "ERROR: no $DIAG -- cannot confirm the trained config"; exit 1; }
if ! grep -q "read_into(x, current_step)" "$MODEL/raven_modeling_minimal_cortex.py" 2>/dev/null; then
    echo "ERROR: $MODEL carries a PRE-P3.0 modeling file.  Refresh it."
    exit 1
fi

# ===== the run's own diag is the only record of what this checkpoint IS =====
python - "$DIAG" "$LIMB" "$ZNORM_TARGET" "$MIX_P" "$STRIDE" <<'CHECK' || exit 1
import json, sys
path, limb, want_zn = sys.argv[1], sys.argv[2], float(sys.argv[3])
want_p, want_st = float(sys.argv[4]), int(sys.argv[5])
rows = [json.loads(l) for l in open(path) if l.strip()]
if not rows:
    sys.exit("ERROR: cortex_diag.jsonl is empty")
last = rows[-1]

zn = {r.get("z_znorm_target") for r in rows} - {None}
if zn and abs(next(iter(zn)) - want_zn) > 1e-6:
    sys.exit(f"ERROR: trained at znorm_target {zn}, read-out passing {want_zn}. "
             "The rebuild would score a model nobody trained.")

# DEAD CHECK, KEPT ONLY SO IT IS NOT RE-ADDED AS IF IT WORKED.  training_diag
# emits ints and floats only, so the str field z_encoding never reaches
# cortex_diag.jsonl: `enc` is ALWAYS empty and the `if enc` guard always
# skips.  It has therefore never verified an encoding on any limb of any arm.
# The live proof that `stag` staggered is the measured z_n_depths >= 2 below.
enc = {r.get("z_encoding") for r in rows} - {None}
want_enc = "staggered_state" if limb == "stag" else "endpoint"
if enc and want_enc not in enc:
    sys.exit(f"ERROR: {limb} trained with latent_encoding {sorted(enc)}, "
             f"expected {want_enc}.  This run dir is not the limb its name claims.")

# The measured proofs, re-checked at read-out time.  A limb that did not run
# its own condition during TRAINING cannot be fixed by scoring it correctly.
if limb == "stag":
    # The two things that make this checkpoint the arm its name claims, checked
    # again here because a limb that did not run its own condition during
    # TRAINING cannot be fixed by scoring it correctly.
    nd = last.get("z_n_depths") or 0
    if nd < 2:
        sys.exit(f"ERROR: z_n_depths {nd} -- the staggered write drew ONE "
                 "depth, so this run trained as a single-depth write.")
    # Compare the VALUE, not the identity: training_diag casts bools to int for
    # wandb, so this arrives as 1/0 and `1 is not True` is True -- which is the
    # whole reason 13662415_2 exited in 7s on a checkpoint that HAD frozen its
    # projection.  A missing key is still a hard error; an int 0 is the real
    # failure this check was written for.
    ef = last.get("z_embed_frozen")
    if ef is None:
        sys.exit("ERROR: no z_embed_frozen in the diag -- the flag never "
                 "reached the graft, so the condition is unverifiable.")
    if int(ef) != 1:
        sys.exit("ERROR: z_embed_frozen is 0 -- the projection trained, "
                 "so this is J4's read and not J7's AC condition.")
elif int(last.get("z_embed_frozen") or 0) == 1:
    sys.exit(f"ERROR: {limb} reports z_embed_frozen 1 but only `stag` "
             "freezes the projection.")

mix = last.get("z_mix_frac") or 0.0
st = last.get("z_stride") or 1
if limb == "mix":
    if abs(mix - want_p) > 0.06:
        sys.exit(f"ERROR: mix_frac {mix:.4f} against configured p {want_p} -- "
                 "the roll did not fire, so `mix` trained as `ctrl`.")
elif mix:
    sys.exit(f"ERROR: {limb} reports mix_frac {mix:.4f} but only `mix` mixes.")
if limb == "slow":
    if st != want_st:
        sys.exit(f"ERROR: z_stride {st} != {want_st} -- latent_stride never "
                 "reached the buffer, so `slow` trained as `ctrl`.")
    if not last.get("z_blocks"):
        sys.exit("ERROR: z_blocks 0 -- the ring never filled, the stride never "
                 "engaged, and `slow` trained as `ctrl`.")
elif st != 1:
    sys.exit(f"ERROR: {limb} reports z_stride {st} but only `slow` strides.")

# `want_enc` is what this limb was CONFIGURED to be, and the dead z_encoding
# check above cannot confirm it -- so it is labelled expected, not confirmed.
# What IS measured here is n_depths / embed_frozen / mix_frac / stride.
print(f"    limb {limb}: encoding {want_enc} (expected, not measured); "
      f"MEASURED n_depths {last.get('z_n_depths')}, "
      f"embed_frozen {last.get('z_embed_frozen')}, mix_frac {mix:.4f}, "
      f"stride {st} -- against {len(rows)} diag rows")
CHECK

# The graft must be FORCED to the trained build: --model_name loads the base
# dir's config, which carries no cortex flags at all.
SETS="--set use_memory=true --set memory_slots=0 --set prefix_memory=gated"
SETS="$SETS --set accum_vecs=16 --set gate_slots=64 --set gate_route=ring"
SETS="$SETS --set gate_norm=tanh --set gate_init=zero --set gate_fill=grow"
SETS="$SETS --set latent_carry=true --set latent_depth_rule=absolute"
SETS="$SETS --set latent_depth_lo=2 --set latent_depth_hi=9 --set latent_renorm=none"
SETS="$SETS --set latent_tok_pool=4"
SETS="$SETS --set latent_s0_read=false --set latent_read=embeds"
SETS="$SETS --set latent_read_znorm=rms --set latent_read_znorm_target=$ZNORM_TARGET"
case "$LIMB" in
    # J7.  BOTH flags, or the rebuild reads a rotation the arm never trained and
    # writes a single depth the arm never wrote -- the eval-side twin of the
    # latent_stride failure below.
    stag)  SETS="$SETS --set latent_encoding=staggered_state"
           SETS="$SETS --set latent_embed_frozen=true" ;;
    slow)  SETS="$SETS --set latent_encoding=endpoint --set latent_stride=$STRIDE" ;;
    # `mix` is NOT rebuilt with latent_read_scramble_p: the mix is gated on
    # self.training and is a TRAINING condition, not a read condition.  Its
    # read-out must ask the same question of it that it asks of the control,
    # or the limbs are not comparable.  The donor CELL is how content is
    # measured here, on every limb alike.
    *)     SETS="$SETS --set latent_encoding=endpoint" ;;
esac

# key pack score z_null n cells
TASKS=(
    "carry-carry-off   carry carry off   300 E1Z1,E1Z0,E0Z0"
    "carry-carry-donor carry carry donor 300 E1Z1,E1Z0"
    "carry-local-off   carry local off   100 E1Z1,E1Z0"
    "pg19-all-off      pg19  all   off   400 E1Z1,E1Z0"
    "pg19-all-donor    pg19  all   donor 400 E1Z1,E1Z0"
)

RC_ALL=0
for T in "${TASKS[@]}"; do
    read KEY PACK SCORE ZNULL N CELLS <<< "$T"
    case "$PACK" in
        carry) DATA=$CARRY_DATA ;;
        pg19)  DATA=$PG19_DATA ;;
    esac
    OUT=eval_results/j6_readout/$RUN/$KEY
    echo ""
    echo "--- $LIMB / $KEY  (score $SCORE, z_null $ZNULL, n $N, cells $CELLS)"
    [ -d "$DATA" ] || { echo "ERROR: no such pack: $DATA"; RC_ALL=1; continue; }
    python evals/eval_carry_2x2.py \
        --model_name "$MODEL" --checkpoint "$CKPT" --data "$DATA" $SETS \
        --n_chunks 8 --T 8 --max_examples "$N" --boot "$BOOT" \
        --dtype float32 --z_null "$ZNULL" --score "$SCORE" --cells "$CELLS" \
        --out_dir "$OUT"
    RC=$?
    [ $RC -eq 0 ] || { echo "FAILED rc=$RC -> $KEY"; RC_ALL=$RC; }
done

# The diag reaches git too: the embed-ratio trajectory, mix_frac and z_blocks
# are how each limb proves it was itself, and they live nowhere else.
mkdir -p logs/j6_diag
cp "$DIAG" "logs/j6_diag/${RUN}.jsonl" 2>/dev/null
echo ""
echo "J6 read-out for $LIMB finished (rc=$RC_ALL) -> eval_results/j6_readout/$RUN/"
echo ""
echo "When all four limbs are done, push and dump the RAW numbers locally:"
echo "  git add eval_results/j6_readout logs/j6_diag logs/Report-*.out \\"
echo "    && git commit -m 'j6 readout' && git push"
echo "  python tools/dump_j6.py --root eval_results/j6_readout"
exit $RC_ALL
