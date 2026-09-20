#!/bin/bash
# Launch the four P1 architecture probes, then compare them.
#
#   bash pace/submit_p1_probes.sh                 # submit all four
#   bash pace/submit_p1_probes.sh a1 a3z          # submit a subset
#   bash pace/submit_p1_probes.sh --compare       # score what has finished
#
# THE FOUR ARMS, and what each one isolates
#   a1   accum,      E only   B2's buffer.  THE FAMILY CONTROL, and the only
#                             valid stand-in for "vs B2": same parent, same
#                             corpus, same step window, same seed.
#   a2   accum + Z            does the Z channel exist at all (buffer held)
#   a3   gated ring, E only   the buffer-geometry cell (Z held off)
#   a3z  gated ring + Z       the gate with Z held on
#
# A1 vs A2 moves one variable.  A2 vs A3z moves one variable.  A1 vs A3z moves
# TWO and is not a headline -- say so before a reviewer does.
#
# WHY COMPARING TO B2 ITSELF WOULD BE INVALID.  b2-final is a completed 5B-token
# conversion on a different corpus schedule, a different step count and W=32.
# Every difference against it is confounded four ways.  A1 carries B2's BUFFER
# at this round's recipe, which is the comparison that answers the question.
#
# ---------------------------------------------------------------------------
# THE VARIABLE-PASSING TRAP, and it cost this project 1.7B tokens of training.
#
# A plain assignment inside this script is a SHELL variable, not an environment
# variable, so `sbatch` never sees it and the sbatch's own default wins
# SILENTLY.  That is how the entire B2 accum arm trained on the wrong corpus for
# six weeks.  The working form is the var-assignment PREFIX on the sbatch
# command itself, which is what every submit line below uses.  Do not "tidy"
# them into exports at the top -- read cortex_cluster_ops before changing one.
#
# And `require <path>` succeeding is NOT evidence the run uses that path: the
# check and the consumer were different variables in different processes.  The
# only real check is the BANNER in the job's log.
set -u

SBATCH_FILE=pace/p1_arms.sbatch
BRANCH_PATH=${BRANCH_PATH:-cortex-retrofit/retro-b2-heal/checkpoint_91552_w16}
ARM_DATA=${ARM_DATA:-data/pg19_fw50_olmo_len4096}
ACCUM_VECS=${ACCUM_VECS:-16}
GATE_SLOTS=${GATE_SLOTS:-64}
CROSS_CHUNKS=${CROSS_CHUNKS:-8}
CELL_STEPS=${CELL_STEPS:-400}
# Read by --compare only, and it has to MATCH p1_arms.sbatch's MAX_MEAN_REC:
# it re-anchors read_live_frac onto the sweep row for the depth the arms really
# trained at.  Wrong here does not mis-scale the number, it reports one from an
# operating point no arm ran.
MAX_MEAN_REC=${MAX_MEAN_REC:-8}
DIAG_INTERVAL=${DIAG_INTERVAL:-10}
OUT_ROOT=${OUT_ROOT:-cortex-retrofit}

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

if [ "${1:-}" = "--compare" ]; then
    # Missing arms are reported by compare_arms rather than guessed at; an arm
    # whose step window does not overlap the reference's gets NO delta, because
    # an unpaired one at this horizon is worse than none.
    # ALL FOUR prelaunch records go in, not just the Z arms.  content_delta_e/z
    # only exist on a dual-channel carry, but content_delta_both and
    # column_delta are written for every arm, and the donor control is only
    # readable as a CONTRAST: on 2026-09-16 the two Z arms were passed in alone
    # and the table could not show that all four had gone negative together
    # (a1 -0.1234, a3 -0.0758) against +0.0186 on the parent.  read_eval skips
    # the keys an E-only record does not have.
    python tools/compare_arms.py \
        --arm a1=$OUT_ROOT/$A1_RUN \
        --arm a2=$OUT_ROOT/$A2_RUN \
        --arm a3=$OUT_ROOT/$A3_RUN \
        --arm a3z=$OUT_ROOT/$A3Z_RUN \
        --eval a1=eval_results/probe-$A1_RUN/prelaunch.json \
        --eval a2=eval_results/probe-$A2_RUN/prelaunch.json \
        --eval a3=eval_results/probe-$A3_RUN/prelaunch.json \
        --eval a3z=eval_results/probe-$A3Z_RUN/prelaunch.json \
        --trained_depth $MAX_MEAN_REC \
        --reference a1 \
        --out eval_results/p1_probe_compare.json
    exit $?
fi

ARMS=${@:-a1 a2 a3 a3z}

# Pre-flight that is cheap and catches the expensive mistake: the arms MUST
# share an ancestor, and that ancestor must already be sliced to W.  A branch
# whose summary_emb width does not match drops the trained write path silently
# (strict=False ignores a MISSING key) and re-seeds from wte[eos] behind a
# healthy loss curve.  p1_arms.sbatch re-checks this per job; this is the
# earlier, cheaper copy so a typo does not cost four queue slots.
if [ ! -f "$BRANCH_PATH/chkpt.pt" ]; then
    echo "ERROR: no chkpt.pt under BRANCH_PATH=$BRANCH_PATH"
    echo "  Slice the heal checkpoint first:"
    echo "    python tools/slice_summary_emb.py \\"
    echo "        --src cortex-retrofit/retro-b2-heal/checkpoint_91552 \\"
    echo "        --dst ${BRANCH_PATH} --n_vec ${ACCUM_VECS}"
    exit 1
fi
if [ ! -d "$ARM_DATA" ]; then
    echo "ERROR: no such pack: ARM_DATA=$ARM_DATA"; exit 1
fi

echo "=== P1 probes | branch=$BRANCH_PATH | data=$ARM_DATA ==="
echo "    W=$ACCUM_VECS K=$GATE_SLOTS cc=$CROSS_CHUNKS steps=$CELL_STEPS"
echo "    diag_interval=$DIAG_INTERVAL (the trajectory compare_arms reads back)"
echo "    arms: $ARMS"
echo ""
echo "    READ THE BANNER in each job's log.  'data=' on the === P1 line is the"
echo "    only thing that proves the corpus reached the job."
echo ""

for ARM in $ARMS; do
    case "$ARM" in
    a1|a2|a3|a3z) ;;
    *) echo "skipping unknown arm $ARM"; continue;;
    esac
    # The var-assignment prefix IS the mechanism -- see the header.
    JOB=$(PROBE=1 ARM=$ARM BRANCH_PATH=$BRANCH_PATH ARM_DATA=$ARM_DATA \
          ACCUM_VECS=$ACCUM_VECS GATE_SLOTS=$GATE_SLOTS \
          CROSS_CHUNKS=$CROSS_CHUNKS CELL_STEPS=$CELL_STEPS \
          DIAG_INTERVAL=$DIAG_INTERVAL \
          sbatch --parsable $SBATCH_FILE)
    echo "  $ARM -> job $JOB"
done

echo ""
echo "When they finish:  bash pace/submit_p1_probes.sh --compare"
echo ""
echo "What the comparison CAN decide: whether each arm is worth finishing --"
echo "did either gate leave its exactly-zero init, does Z receive gradient, is"
echo "the carry collapsing, is anything non-finite.  What it CANNOT decide:"
echo "whether Z helps or whether the gate beats accum.  Those need the full"
echo "cells, evals/eval_carry_2x2.py and evals/eval_influence_horizon.py."
