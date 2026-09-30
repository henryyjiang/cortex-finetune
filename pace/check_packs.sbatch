#!/bin/bash
#SBATCH --account=gts-aivanova7-lab
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH -t 0:30:00
#SBATCH -q inferno
#SBATCH -J check_packs
#SBATCH -o logs/Report-%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=henryyjiang42@gmail.com

# ============================================================================
# EVERY PACK THE 5B PAIR NEEDS, CHECKED IN ONE JOB, ON A COMPUTE NODE.
#
#   sbatch pace/check_packs.sbatch
#
# WHY THIS IS AN SBATCH AND NOT A LOGIN-NODE ONE-LINER.  tools/check_pack.py's
# own arithmetic is trivial -- len(ds) and one row -- but getting there is not:
# `from datasets import load_from_disk` drags in pyarrow and torch, and
# load_from_disk then stats a multi-shard arrow directory on Lustre, which is
# the random-seek path this repo has been bitten by before (the ~14h
# save_to_disk without num_proc=8).  On a shared, throttled login node that
# enforces its per-user memory cap BY KILLING YOU, that is minutes per pack at
# best and a kill that looks like a failure at worst -- the same trap that made
# tools/check_latent_probe.py read as a probe failure when it was a kill.
# Same reasoning as pace/run_tests.sbatch.
#
# NO GPU IS REQUESTED, on purpose: nothing here touches a device, so this
# schedules on the CPU partitions and usually starts immediately instead of
# queueing behind real training.  The cuda module is loaded only so the
# cortex-retro torch build finds its link-time libraries.
#
# WHAT IT CHECKS, and why each row is the number it is
# ----------------------------------------------------
# Both 5B arms run ONE 38,147-step horizon across TWO corpora, and --steps is
# the steps SERVED FROM THAT PACK, never the horizon:
#
#   heal  data/fineweb_edu_olmo_len4096   steps 0 -> 11,444       = 11,444
#   mix   data/pg19_fw50_olmo_len4096     steps 11,444 -> 38,147  = 26,703
#
# Passing 38,147 for the mix pack would over-state its need by 43%.
#
# batch_size 32, NOT 4.  The row budget is steps x batch_size and is
# independent of micro_batch_size, so this is the same check whether the arm
# runs mbs 1 or 2.  The framework's S10 row counts were computed at 16 and are
# half these.
#
# resume_rows is ZERO for both, and that is a CLAIM about how the run is
# driven, not an omission: neither arm branches, and the corpus-switch link
# passes --reset_dataset_position true, which makes the new pack's skip zero.
# Drop that flag on the switch link and the mix pack must instead hold
# 366,208 + 854,496 rows -- see RESUME_ROWS below, which re-runs the mix check
# that way so the cost of forgetting it is a number in this log rather than a
# surprise at step 29,375.
#
# The two VALIDATION packs are checked for existence only.  They are read by
# pace/midrun_check.sbatch a few hundred rows at a time, so they have no step
# budget to cover -- what matters is that they are there and non-trivial
# BEFORE a mid-run read-out queues behind a 48h link and dies on a missing dir.
# data/fineweb_edu_olmo_val_len4096 does not exist yet; that script's header
# has the build.
# ============================================================================

# This one is also useful by hand (that is the point -- it replaces a
# login-node one-liner), and an unset SLURM_SUBMIT_DIR would silently `cd` to
# $HOME and then report every pack as missing.  Fall back to the repo root.
cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}" || exit 1

module load anaconda3
module load cuda/12.1.1
conda activate cortex-retro

export SCRATCH=${SCRATCH:-$HOME/scratch}
export HF_HOME=$SCRATCH/hf_cache
export HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=1

MAX_LENGTH=${MAX_LENGTH:-4096}
BATCH_SIZE=${BATCH_SIZE:-32}
MAX_STEPS=${MAX_STEPS:-38147}
HEAL_END=${HEAL_END:-11444}
MIX_STEPS=${MIX_STEPS:-$(( MAX_STEPS - HEAL_END ))}
HEAL_DATA=${HEAL_DATA:-data/fineweb_edu_olmo_len4096}
MIX_DATA=${MIX_DATA:-data/pg19_fw50_olmo_len4096}
HEAL_MARGIN=${HEAL_MARGIN:-1.10}
MIX_MARGIN=${MIX_MARGIN:-1.10}
# The mix pack is re-checked as if the corpus-switch link FORGOT
# --reset_dataset_position.  Set 0 to skip that second read.
RESUME_ROWS=${RESUME_ROWS:-$(( HEAL_END * BATCH_SIZE ))}
# Existence-only, space separated.  The FineWeb-Edu val pack is expected to be
# missing until it is built.
VAL_PACKS=${VAL_PACKS:-"data/fineweb_edu_olmo_val_len4096 data/pg19_olmo_validation_len4096_strided data/carry_task_r16_len4096_val"}

echo "=== pack check | $(hostname) ==="
echo "    horizon $MAX_STEPS steps x $BATCH_SIZE seq x $MAX_LENGTH tok" \
     "= $(( MAX_STEPS * BATCH_SIZE * MAX_LENGTH )) tokens"
echo "    heal 0..$HEAL_END ($HEAL_END steps)   mix $HEAL_END..$MAX_STEPS ($MIX_STEPS steps)"
echo ""

RC_ALL=0

echo "--------------------------------------------------------------"
echo "1/2  heal pack"
python tools/check_pack.py --data "$HEAL_DATA" --steps "$HEAL_END" \
    --batch_size "$BATCH_SIZE" --max_length "$MAX_LENGTH" \
    --margin "$HEAL_MARGIN" || RC_ALL=1

echo ""
echo "--------------------------------------------------------------"
echo "2/2  mix pack"
python tools/check_pack.py --data "$MIX_DATA" --steps "$MIX_STEPS" \
    --batch_size "$BATCH_SIZE" --max_length "$MAX_LENGTH" \
    --margin "$MIX_MARGIN" || RC_ALL=1

if [ "$RESUME_ROWS" -gt 0 ]; then
    echo ""
    echo "--------------------------------------------------------------"
    echo "   counterfactual: the mix pack IF the switch link forgets"
    echo "   --reset_dataset_position (skip = $RESUME_ROWS rows)."
    echo "   A FAIL here is EXPECTED and is the point -- it prices the"
    echo "   mistake.  Only the two checks above gate the launch."
    python tools/check_pack.py --data "$MIX_DATA" --steps "$MIX_STEPS" \
        --batch_size "$BATCH_SIZE" --max_length "$MAX_LENGTH" \
        --margin "$MIX_MARGIN" --resume_rows "$RESUME_ROWS" \
        && echo "   (it would have survived it -- still pass the flag)" \
        || echo "   ^ that is what omitting the flag costs."
fi

echo ""
echo "--------------------------------------------------------------"
echo "validation packs (existence only -- read a few hundred rows at a time"
echo "by pace/midrun_check.sbatch, so they have no budget to cover)"
for P in $VAL_PACKS; do
    if [ -d "$P" ]; then
        SZ=$(du -sh "$P" 2>/dev/null | cut -f1)
        echo "  ok      $P  ($SZ)"
    else
        echo "  MISSING $P"
        case "$P" in
            *fineweb_edu_olmo_val*)
                echo "          expected until built -- the build is in the PREP"
                echo "          block of pace/midrun_check.sbatch.  It does NOT"
                echo "          gate a training launch, only the read-outs." ;;
            *) RC_ALL=1 ;;
        esac
    fi
done

echo ""
echo "=============================================================="
if [ $RC_ALL -eq 0 ]; then
    echo "PACKS OK -- both arms can reach the $MAX_STEPS-step horizon."
    echo "  Next: sbatch pace/smoke_control_mbs2.sbatch   (decides MICRO_BS)"
else
    echo "PACK CHECK FAILED.  Do not queue a 48h link against a short pack:"
    echo "  train.py RAISES on exhaustion now rather than exiting clean, but a"
    echo "  failed 48h job is still a failed 48h job.  B1 stopped at 24,414"
    echo "  steps / 400M tokens this way and wandb reported 'finished'."
fi
echo "=============================================================="
exit $RC_ALL
