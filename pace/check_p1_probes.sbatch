#!/bin/bash
#SBATCH --account=gts-aivanova7-lab
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH -t 0:30:00
#SBATCH -q inferno
#SBATCH -o logs/Report-%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=henryyjiang42@gmail.com

# Re-score the P1 architecture probes that have ALREADY trained.  Read-only:
# it loads checkpoints, prints verdicts, and writes nothing under
# cortex-retrofit/.  Re-running it is always safe.
#
#   sbatch pace/check_p1_probes.sbatch                  # all four arms
#   ARMS="a2 a3" sbatch pace/check_p1_probes.sbatch     # just the two that
#                                                       # failed on the old
#                                                       # checker
#   COMPARE=1 sbatch pace/check_p1_probes.sbatch        # + the cross-arm score
#
# WHY THIS IS AN SBATCH AND NOT A LOGIN-NODE ONE-LINER.  This is the same trap
# that killed the 2026-07-17 pack runs and the `datasets` streaming reader:
# THE LOGIN NODE HAS A PER-USER MEMORY CAP AND ENFORCES IT BY KILLING YOU.
# `tools/check_latent_probe.py` is "just a checker", but it torch.loads TWO
# ~1.4B-parameter fp32 checkpoints -- the probe and its parent, ~5.5 GB each --
# so it needs ~11 GB resident before it prints a single line.  There is nothing
# to tune away: the tool has to hold both state dicts to diff them.  It gets
# killed, the kill looks like a crash, and the crash looks like the probe
# failing again.  IT IS NOT.  Run it here.
#
# (It USED to need ~16.5 GB, because it loaded the parent twice -- once for the
# weights and once for `optimizer_step`.  That is fixed; the tool loads it once.
# If a future edit reintroduces a second `_load(args.parent)`, this job starts
# needing 48GB and the login node starts killing it again.)
#
# NO GPU IS REQUESTED, on purpose.  Every check is a CPU tensor diff --
# norms, finiteness, shapes -- with `map_location="cpu"`, so a GPU would sit
# idle for the whole job while queueing behind real training.  That also means
# this job schedules on the CPU partitions and usually starts immediately.
# The cuda module below is loaded only so the cortex-retro torch build finds
# its link-time libraries; nothing in this job touches a device.
#
# WHAT A GREEN RUN HERE DOES AND DOES NOT MEAN.  Same caveat the tool prints:
# it says the machinery is LIVE -- every parameter block that should be
# attached to the loss is attached to it.  It cannot say whether Z helps or
# whether the gate beats accum.  That is the full cell, evals/eval_carry_2x2.py
# and evals/eval_influence_horizon.py.  Do not quote a green probe as a result.
#
# THE TWO CHECKS THAT ARE ARM-SPECIFIC, and why a2/a3 once failed here.  Z is
# PARAMETER-FREE on the accum buffer: PrefixAccumBuffer.merge concatenates E
# and Z on the last dim and owns no gate, so a2 has no `_z` tensors to inspect
# and never will, while PrefixGatedBuffer allocates a second gate under the
# same flag.  The checker now reads the config's `latent_carry` for INTENT and
# only requires `_z` parameters where a GATED arm asked for Z.  For a2,
# liveness is evals/diag_dual_channel_walk.py's question (2D carry width,
# |Z|/s0, z-read), and it already answered it in the training job's own log.

cd $SLURM_SUBMIT_DIR

module load anaconda3
module load cuda/12.1.1
conda activate cortex-retro

export SCRATCH=${SCRATCH:-$HOME/scratch}
export HF_HOME=$SCRATCH/hf_cache
export HF_HUB_OFFLINE=1

# READ-ONLY on the symlink, unlike p1_arms.sbatch which creates it.  If it is
# missing, the probes were never written here and there is nothing to score --
# say that rather than silently creating an empty tree and reporting four
# "no checkpoint" lines.
OUT_ROOT=${OUT_ROOT:-cortex-retrofit}
if [ ! -e "$OUT_ROOT" ]; then
    echo "ERROR: $OUT_ROOT does not exist.  Nothing to score."
    echo "       The probes write there via a symlink to \$SCRATCH that"
    echo "       pace/p1_arms.sbatch creates; run the probes first."
    exit 1
fi

# THE SAME VARIABLES pace/submit_p1_probes.sh USES, and they have to stay in
# step: the run NAME is built from the geometry, so a mismatched ACCUM_VECS
# here does not score a different arm, it scores NOTHING and reports the arm
# as missing.  If you changed a geometry there, change it here.
BRANCH_PATH=${BRANCH_PATH:-cortex-retrofit/retro-b2-heal/checkpoint_91552_w16}
ACCUM_VECS=${ACCUM_VECS:-16}
GATE_SLOTS=${GATE_SLOTS:-64}
CROSS_CHUNKS=${CROSS_CHUNKS:-8}

#
# THE `probe-` PREFIX HERE IS DELIBERATE AND IS NOT RED 13.  This script is
# scoped to the 400-step ARCHITECTURE PROBES by name and by purpose;
# p1_arms.sbatch adds that prefix only under PROBE=1 (line 324).  The P2.3
# CELLS saved to the BARE name, and the launchers that read them
# (eval_carry_2x2, eval_deciding, norm_trace, submit_cell_evals) were fixed
# on 2026-09-19 to match.  Do not 'fix' this one to agree with those.
A1_RUN=probe-p1-a1-accum-w${ACCUM_VECS}-cc${CROSS_CHUNKS}
A2_RUN=probe-p1-a2-accum-w${ACCUM_VECS}-cc${CROSS_CHUNKS}-z
A3_RUN=probe-p1-a3-gated-w${ACCUM_VECS}k${GATE_SLOTS}-cc${CROSS_CHUNKS}
A3Z_RUN=probe-p1-a3z-gated-w${ACCUM_VECS}k${GATE_SLOTS}-cc${CROSS_CHUNKS}-z

ARMS=${ARMS:-"a1 a2 a3 a3z"}

echo "=== P1 probe re-check | parent $BRANCH_PATH ==="
echo "    arms: $ARMS"
echo "    geometry: W=$ACCUM_VECS K=$GATE_SLOTS cc=$CROSS_CHUNKS"
echo "    read-only: no checkpoint is written or modified"
echo ""

if [ ! -d "$BRANCH_PATH" ]; then
    echo "ERROR: parent $BRANCH_PATH not found."
    echo "       Without it the re-seed check cannot run, and that check is the"
    echo "       only thing standing between you and a silent branch-width"
    echo "       failure.  Fix the path; do not drop --parent."
    exit 1
fi

RC=0
SUMMARY=""

for ARM in $ARMS; do
    case "$ARM" in
        a1)  RUN=$A1_RUN ;;
        a2)  RUN=$A2_RUN ;;
        a3)  RUN=$A3_RUN ;;
        a3z) RUN=$A3Z_RUN ;;
        *)   echo "unknown arm '$ARM' (expected a1 a2 a3 a3z)"; RC=1; continue ;;
    esac

    # NUMERIC sort on the step, so checkpoint_91952 does not lose to
    # checkpoint_9999 the way a lexical sort would.  Same idiom as
    # pace/p1_arms.sbatch's post-run block, deliberately.
    LAST=$(ls -d $OUT_ROOT/$RUN/checkpoint_* 2>/dev/null | sort -t_ -k2 -n | tail -1)
    echo ""
    echo "######## $ARM | $RUN ########"
    if [ -z "$LAST" ]; then
        echo "  no checkpoint under $OUT_ROOT/$RUN -- this arm has not run."
        SUMMARY="$SUMMARY
  $ARM  MISSING   (no checkpoint under $OUT_ROOT/$RUN)"
        RC=1
        continue
    fi

    python tools/check_latent_probe.py --probe "$LAST" --parent "$BRANCH_PATH"
    ARM_RC=$?
    if [ $ARM_RC -eq 0 ]; then
        SUMMARY="$SUMMARY
  $ARM  PASS      $(basename $LAST)"
    else
        SUMMARY="$SUMMARY
  $ARM  FAIL      $(basename $LAST)  <- rc=$ARM_RC"
        RC=1
    fi
done

echo ""
echo "=============================================================="
echo "P1 PROBE RE-CHECK SUMMARY"
echo "=============================================================="
echo "$SUMMARY"
echo ""
if [ $RC -eq 0 ]; then
    echo "Every arm is LIVE.  This says the machinery is attached to the loss."
    echo "It says NOTHING about whether Z helps or whether the gate beats"
    echo "accum -- that is the full cell, the 2x2 and the influence horizon."
else
    echo "At least one arm is red.  Before believing it, check whether the"
    echo "CHECK is well-posed for that arm: every red this suite has produced"
    echo "so far has been the instrument encoding an assumption the"
    echo "architecture does not share, not a defect in the architecture."
fi

if [ -n "$COMPARE" ]; then
    # Cheap -- jsonl and json plus a bootstrap, no checkpoints -- so it rides
    # along here rather than earning its own job.  It is also the step that
    # actually answers the round's question, which the per-arm checks do not.
    echo ""
    echo "######## cross-arm comparison ########"
    bash pace/submit_p1_probes.sh --compare || RC=1
fi

echo ""
echo "Next: bash pace/submit_p1_probes.sh --compare   (if you did not set COMPARE=1)"
exit $RC
