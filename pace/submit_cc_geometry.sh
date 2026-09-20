#!/bin/bash
# JOB 11 -- THE CROSS-CHUNK GEOMETRY CELLS.  The one that can change the 5B spec.
#
#   DRY=1 bash pace/submit_cc_geometry.sh        # print the plan and stop
#   bash pace/submit_cc_geometry.sh              # submit it
#
# READ THE HORIZON FIRST.  This script trains, and each cell is ~29 h on an
# H200.  `pace/submit_cell_evals.sh`'s horizon stage costs minutes and answers
# whether the question is live: if the TRAINED read on the a3 cell still reaches
# only ~2 chunks, cc8 is over-provisioned and these cells are worth the compute.
# If training pushed the horizon out to 6-8, cc8 is justified and this whole job
# is cancelled.  Do not run this on P2.1's 400-step-probe horizon.
#
# THE QUESTION.  P2.1 put the READ horizon at ~2 chunks while the buffer holds
# 8, so a 5B run at cc8 may write six rows per chunk that nothing ever reads.
# `p2_deciding_measurements_handoff.md` S2 flags it as independent of both Z and
# the buffer choice, and says settle it before speccing the run.
#
# ======================= THREE TRAPS, ALL OF THEM REAL =====================
#
# TRAP 1 -- THE RING CONSTRAINT.  `cortex_memory/buffers.py:342` requires
#
#       cross_chunks >= 2 * (n_slots / n_vec)
#
# or the ring never completes a lap during training and the gate receives ZERO
# GRADIENT -- you would have trained an accum buffer with dead gate parameters
# bolted on, and every number it produced would be about a buffer you did not
# mean to build.  `train.py` asserts it, so today it crashes rather than lying;
# that assert is the only thing standing between this experiment and a silent
# dead-gate arm, and it is not this script's to relax.  At W=16 that means
#
#       cc=8  ->  K <= 64          cc=4  ->  K <= 32          cc=2  ->  K <= 16
#
# So a naive "CROSS_CHUNKS=4 with the cells' K=64" is not a geometry sweep, it
# is a crash.  K is DERIVED from cc below, never passed in loose.
#
# TRAP 2 -- VARYING cc VARIES K, UNLESS YOU PIN IT.  The P2.3 a3 cell is
# cc8/K64.  Comparing it to cc4/K32 moves both knobs at once and the result
# would not say which one did it.  Hence PAIR:
#
#   PAIR=cc     (default)  cc4/K32  vs  cc8/K32     K HELD, cc varies.
#                          The clean contrast.  TWO new cells; the existing
#                          cc8/K64 cell does not serve as one of them.
#   PAIR=maxk              cc4/K32  vs  cc8/K64     each at its largest legal K.
#                          ONE new cell, because cc8/K64 is already trained --
#                          but cc and K move together and the contrast is
#                          CONFOUNDED.  Cheap, and honest only if reported that
#                          way.  It is a screen, not a decision.
#
# Running PAIR=cc also hands you a K contrast for free: its cc8/K32 cell against
# the existing cc8/K64 cell is K at fixed cc, on the same parent and budget.
#
# TRAP 3 -- VARYING cc ALSO VARIES CHUNK LENGTH.  chunk_len = MAX_LENGTH / cc,
# so cc4 at the cells' MAX_LENGTH=4096 gives 1024-token chunks against cc8's
# 512.  There is no way to vary the number of chunks, the chunk length and the
# sequence length independently -- two of the three are free.  HOLD says which:
#
#   HOLD=seqlen (default)  MAX_LENGTH=4096 fixed, chunk_len grows (512 -> 1024).
#                          Total context is IDENTICAL across cells, so this asks
#                          the actual question: are fewer, coarser summaries at a
#                          REACHABLE depth better than finer ones at a depth
#                          nothing reads?  This is the 5B decision.
#   HOLD=chunklen          chunk_len=512 fixed, MAX_LENGTH shrinks (4096 ->
#                          2048).  Write granularity is constant and total
#                          context halves.  A different question, and it needs a
#                          PACK AT THAT ROW LENGTH -- the default pack holds
#                          4096-token rows, so this script REFUSES HOLD=chunklen
#                          unless you pass ARM_DATA explicitly.  ARM_DATA
#                          silently not propagating cost this project 1.7B
#                          tokens; it is not being guessed at here.
#
# EVERYTHING ELSE IS HELD.  Same parent (retro-b2-heal/checkpoint_91552_w16),
# same corpus, same seed, same CELL_STEPS=24414 budget, same W=16, same mr8.
# carry_grad_chunks tracks cc at B2's 50% convention (4 of 8 -> 2 of 4), because
# holding it at 4 of 4 would make the cc4 arm fully-backpropagated and the cc8
# arm half-backpropagated, which is a TBPTT contrast wearing a geometry label.
set -u

SUB="sbatch"
[ -n "${DRY:-}" ] && SUB="echo [dry] sbatch"
cd "$(dirname "$0")/.."

ARM=${ARM:-a3}                 # a3 = the gated ring, i.e. the arm P2.3 chose
ACCUM_VECS=${ACCUM_VECS:-16}   # W.  A PARAMETER SHAPE -- never vary mid-chain.
PAIR=${PAIR:-cc}
HOLD=${HOLD:-seqlen}
BASE_LEN=${BASE_LEN:-4096}
CELL_STEPS=${CELL_STEPS:-24414}
BRANCH=${BRANCH:-cortex-retrofit/retro-b2-heal/checkpoint_91552_w16}

case "$ARM" in
a1|a3) ;;
*) echo "ERROR: ARM must be a1 (accum) or a3 (gated ring); got '$ARM'."
   echo "       The Z arms are not part of this question."; exit 1 ;;
esac

# --- the legal K for a given cc, from buffers.py:342 ------------------------
max_k () { echo $(( $1 * ACCUM_VECS / 2 )); }

# CELLS can be overridden as "cc:K cc:K ..." -- it is how the legality guard
# below gets exercised in a test, and how a third point is added without editing
# this file.  The guard applies to an override exactly as to a default.
case "${CELLS:-}" in
"") case "$PAIR" in
    cc)    CELLS="4:32 8:32" ;;
    maxk)  CELLS="4:32" ;;    # cc8/K64 is the existing P2.3 cell
    *)     echo "ERROR: PAIR must be 'cc' or 'maxk'; got '$PAIR'."; exit 1 ;;
    esac ;;
*)  echo "    (CELLS overridden: $CELLS)" ;;
esac

if [ "$HOLD" = "chunklen" ] && [ -z "${ARM_DATA:-}" ]; then
    echo "ERROR: HOLD=chunklen shortens MAX_LENGTH, so it needs a pack whose"
    echo "       rows are that long.  The default pack holds ${BASE_LEN}-token"
    echo "       rows and feeding it to a shorter MAX_LENGTH is the ARM_DATA"
    echo "       trap again.  Pass ARM_DATA=<pack> explicitly, or use the"
    echo "       default HOLD=seqlen."
    exit 1
fi

# Resolved to a CONCRETE value AFTER the guard above (which must see whether the
# user supplied one), and always passed as a LITERAL assignment prefix below --
# never as ${ARM_DATA:+ARM_DATA=...}.  A prefix that arrives by variable
# expansion is not an assignment to bash, it is parsed as a command name, and
# `ARM_DATA=foo: No such file or directory` is what you get.  The handoff's S5
# says the var-assignment prefix IS the mechanism on every sbatch line and that
# the banner in the log is the only proof it arrived; this puts the pack in
# both.  Mirrors p1_arms.sbatch's own default.
ARM_DATA=${ARM_DATA:-data/pg19_fw50_olmo_len4096}

echo "=== job 11: cross-chunk geometry | arm=$ARM | PAIR=$PAIR | HOLD=$HOLD ==="
echo "    parent:  $BRANCH"
echo "    budget:  $CELL_STEPS steps per cell (same as P2.3)"
echo "    W:       $ACCUM_VECS"
echo "    data:    $ARM_DATA"
echo ""
printf "    %-6s%-6s%-11s%-11s%-8s%-7s\n" cc K chunk_len max_length grad legal
for spec in $CELLS; do
    CC=${spec%%:*}; K=${spec##*:}
    if [ "$HOLD" = "seqlen" ]; then
        LEN=$BASE_LEN
    else
        LEN=$(( 512 * CC ))
    fi
    CHUNK=$(( LEN / CC ))
    GRAD=$(( CC / 2 ))
    LEGAL=$(max_k "$CC")
    OK=$([ "$K" -le "$LEGAL" ] && echo "K<=$LEGAL" || echo "ILLEGAL")
    printf "    %-6s%-6s%-11s%-11s%-8s%-7s\n" "$CC" "$K" "$CHUNK" "$LEN" "$GRAD/$CC" "$OK"
done
echo ""
if [ "$PAIR" = "maxk" ]; then
    echo "    + the EXISTING cell p1-a3-gated-w${ACCUM_VECS}k64-cc8 as the cc8 leg."
    echo "      cc AND K both differ across the pair -- CONFOUNDED.  Screen only."
    echo ""
fi

# --- refuse an illegal cell rather than letting train.py's assert find it ---
for spec in $CELLS; do
    CC=${spec%%:*}; K=${spec##*:}
    LEGAL=$(max_k "$CC")
    if [ "$K" -gt "$LEGAL" ]; then
        echo "ERROR: cc=$CC with K=$K violates cross_chunks >= 2*(K/W):"
        echo "       at W=$ACCUM_VECS, cc=$CC allows K <= $LEGAL."
        echo "       The ring would never lap and the gate would get ZERO"
        echo "       gradient.  train.py asserts this; do not work around it."
        exit 1
    fi
done

[ -d "$BRANCH" ] || { echo "ERROR: no parent checkpoint at $BRANCH"; exit 1; }

echo "--- submitting ---"
for spec in $CELLS; do
    CC=${spec%%:*}; K=${spec##*:}
    if [ "$HOLD" = "seqlen" ]; then LEN=$BASE_LEN; else LEN=$(( 512 * CC )); fi
    GRAD=$(( CC / 2 ))
    AMAX=$(( ACCUM_VECS * CC ))     # the FIFO cap, and it is SILENT if wrong:
                                    # p1_arms.sbatch hardcodes 128 (= 16 x 8),
                                    # which at cc4 would leave the buffer twice
                                    # the size the geometry calls for and no
                                    # error anywhere.
    echo "    cc=$CC K=$K len=$LEN grad=$GRAD amax=$AMAX"
    ARM=$ARM \
    CROSS_CHUNKS=$CC \
    GATE_SLOTS=$K \
    ACCUM_VECS=$ACCUM_VECS \
    ACCUM_MAX=$AMAX \
    CARRY_GRAD_CHUNKS=$GRAD \
    MAX_LENGTH=$LEN \
    CELL_STEPS=$CELL_STEPS \
    BRANCH_PATH="$BRANCH" \
    ARM_DATA="$ARM_DATA" \
        $SUB pace/p1_arms.sbatch
done

echo ""
echo "=== submitted.  Compare them PAIRED, the way P2.3 was read. ==="
echo ""
echo "THE READ, and it is not the training loss alone:"
echo "  1. paired delta over the last 500 updates, two non-overlapping windows"
echo "     -- the same stopping rule as p23_cells_prereg.md S1.  Note that rule"
echo "     is still applied BY HAND; compare_arms.py --stopping_rule does not"
echo "     exist yet (open item S7.2.2)."
echo "  2. tok/s in the SAME table.  P2.3's pre-registration predicted a ~6%"
echo "     saving from halving the read columns and measured -0.04%, so do not"
echo "     assume fewer chunks is cheaper -- MEASURE it.  cc4 does fewer, larger"
echo "     forwards; that is not obviously faster or slower."
echo "  3. the influence horizon on the RESULTING checkpoints.  If cc4 reads to"
echo "     depth 2 of 4 and cc8 reads to depth 2 of 8, the buffer is not the"
echo "     thing that binds and the horizon is a property of the READ.  That"
echo "     would be a stronger result than either loss number."
if [ "$HOLD" = "seqlen" ]; then
echo "  4. chunk_len differs across these cells BY CONSTRUCTION (512 vs 1024)."
echo "     Say so in the record.  A cc effect and a chunk-length effect are not"
echo "     separable in this design and no amount of reading makes them so."
fi
