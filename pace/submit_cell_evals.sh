#!/bin/bash
# THE P2.3 CELLS' FOLLOW-UP EVALS -- jobs 7-10 of the 2026-09-19 open list
# (`p2_deciding_measurements_handoff.md` S7.3).  All forward-only.  No training.
#
#   bash pace/submit_cell_evals.sh              # submit everything
#   DRY=1 bash pace/submit_cell_evals.sh        # print the commands and stop
#   ONLY=pack bash pace/submit_cell_evals.sh    # one stage: pack|horizon|write|norm
#
# THE CELLS ARE READ.  P2.3 satisfied its stopping rule on 2026-09-18: the gated
# ring beats accum by -0.0119 (95% CI [-0.01346, -0.01044]) at EQUAL cost
# (-0.04% tok/s, not the ~6% the pre-registration predicted).  That is a paired
# TRAINING-loss result.  Everything here is the deciding read on the checkpoints
# themselves, which is a different instrument and can disagree.
#
# ================= READ THIS BEFORE RUNNING THE PACK STAGE =================
# `data/pg19_olmo_val_len4096` holds FIFTY ROWS.  Every P2.1 cell and the
# submitted 2x2 asked for --max_examples 100 and silently got n=50 -- half the
# pre-registered power, with nothing in the log saying so.  The strided packer
# turns the same 50 books into hundreds of windows at the same row length: more
# samples, same source, no distribution change.
#
# SWITCHING PACKS BREAKS COMPARABILITY with the 18 P2.1 cells, and that is the
# ARM_DATA failure wearing a third hat.  The mitigation is built in, not
# optional: the horizon stage runs ARMS="a1 a3 parent", so the new pack carries
# its OWN parent row and the comparison is internally clean without needing to
# match P2.1's numbers.  SAY IN EVERY RECORD WHICH PACK IT READ.
#
# The pack job is CPU-only and takes ~1 h.  The evals that consume it are
# submitted with a dependency so they cannot start against a half-written pack;
# a killed save leaves shards with no state.json and load_from_disk will happily
# read a truncated pack without complaining.
#
# ============================ WHAT EACH STAGE ASKS ========================
#   pack     the strided validation pack.  Raises n from 50 to ~hundreds.
#   horizon  does the d=4 ring aliasing SURVIVE 24k steps of training?  P2.1
#            measured it on 400-step probes.  If the trained read still reaches
#            only ~2 chunks, cc8 writes six rows per chunk that nothing reads
#            and the cc-geometry cells (pace/submit_cc_geometry.sh) become the
#            next thing to run.
#   write    write capacity on the CELLS.  Job 13329445 scored the PARENT and
#            found the write is not the binding constraint (prev-none +0.0995
#            flat, shuf-none +0.058 -- wrong content is worse than none).  Did
#            24k steps of gated training move that balance?
#   norm     P2.5's E-norm drift curve, reconstructed from the saved
#            checkpoints because DIAG_INTERVAL was 0 for the whole cell.
#
# NOT HERE, DELIBERATELY: the 2x2 (submitted separately 2026-09-19) and anything
# touching Z.  cortex-final is the E-only gated buffer; that is what P2.3
# decided and it launches with or without Z.
set -u

SUB="sbatch"
[ -n "${DRY:-}" ] && SUB="echo [dry] sbatch"
ONLY=${ONLY:-all}
cd "$(dirname "$0")/.."

ACCUM_VECS=${ACCUM_VECS:-16}
GATE_SLOTS=${GATE_SLOTS:-64}
CROSS_CHUNKS=${CROSS_CHUNKS:-8}
STAMP=${STAMP:-$(date +%Y%m%d)}
OUT_ROOT=${OUT_ROOT:-cortex-retrofit}
NEW_PACK=${NEW_PACK:-data/pg19_olmo_validation_len4096_strided}
OLD_PACK=${OLD_PACK:-data/pg19_olmo_val_len4096}

A1_RUN=p1-a1-accum-w${ACCUM_VECS}-cc${CROSS_CHUNKS}
A3_RUN=p1-a3-gated-w${ACCUM_VECS}k${GATE_SLOTS}-cc${CROSS_CHUNKS}

want () { [ "$ONLY" = "all" ] || [ "$ONLY" = "$1" ]; }

# --- a cheap pre-flight, because every stage below assumes these exist -------
echo "=== pre-flight ==="
MISSING=0
for d in "$OUT_ROOT/$A1_RUN" "$OUT_ROOT/$A3_RUN"; do
    if [ -d "$d" ]; then
        LAST=$(ls -d "$d"/checkpoint_* 2>/dev/null | sort -t_ -k2 -n | tail -1)
        echo "    OK   $d -> $(basename "${LAST:-NONE}")"
    else
        echo "    MISS $d"
        MISSING=1
    fi
done
if [ "$MISSING" = "1" ]; then
    echo ""
    echo "ERROR: a P2.3 cell run dir is missing.  Note the names have NO 'probe-'"
    echo "       prefix -- that prefix is the 400-step architecture probe, and"
    echo "       confusing the two is RED 13.  Check the cells finished."
    exit 1
fi
echo ""

PACK_JOB=""

# --------------------------------------------------------------------------
# 7. the strided validation pack.  CPU only, ~1 h.
# --------------------------------------------------------------------------
if want pack; then
    echo "--- 7. strided validation pack -> $NEW_PACK ---"
    if [ -d "$NEW_PACK" ] && [ -f "$NEW_PACK/state.json" ]; then
        echo "    already built and COMPLETE (state.json present); skipping."
        echo "    rows: $(python -c "from datasets import load_from_disk; print(len(load_from_disk('$NEW_PACK')))" 2>/dev/null || echo '?')"
    else
        OUT=$($SUB SPLIT=validation pace/prepare_pg19_pack.sbatch 2>&1) || true
        echo "    $OUT"
        # "Submitted batch job 12345" -> 12345, so the evals can depend on it.
        PACK_JOB=$(echo "$OUT" | grep -oE '[0-9]+$' | tail -1)
        [ -n "$PACK_JOB" ] && echo "    evals will wait on job $PACK_JOB"
    fi
    echo ""
fi

# Which pack the evals read, and what they wait for.  If the pack was just
# submitted the evals take the NEW one behind a dependency; if it already
# exists they take it immediately; if the pack stage was skipped entirely they
# fall back to the old 50-row pack and SAY SO.
if [ -d "$NEW_PACK" ] && [ -f "$NEW_PACK/state.json" ]; then
    EVAL_DATA="$NEW_PACK"; DEP=""
elif [ -n "$PACK_JOB" ]; then
    EVAL_DATA="$NEW_PACK"; DEP="--dependency=afterok:$PACK_JOB"
else
    EVAL_DATA="$OLD_PACK"; DEP=""
    echo "!!! the new pack is neither built nor submitted, so the evals below"
    echo "!!! will read $OLD_PACK and run at n=50.  That is half the"
    echo "!!! pre-registered power.  Run with ONLY=pack first if you want n up."
    echo ""
fi
echo "=== evals will read: $EVAL_DATA ${DEP:+($DEP)} ==="
echo ""

# --------------------------------------------------------------------------
# 8. influence horizon on the cell checkpoints
# --------------------------------------------------------------------------
if want horizon; then
    echo "--- 8. influence horizon on the CELLS (d=4 aliasing after training) ---"
    # ARMS includes `parent` on purpose: it is what makes the new pack's numbers
    # self-contained rather than needing to line up with P2.1's 50-row cells.
    # DAMAGES=donor only -- `random` was P2.1's sensitivity control for Z's
    # SCALE, and Z is not on these arms, so it would buy a row nobody reads.
    # 3 arms x 1 damage x 1 channel = 3 cells.
    ARMS="a1 a3 parent" DAMAGES="donor" \
    DATA="$EVAL_DATA" ACCUM_VECS=$ACCUM_VECS GATE_SLOTS=$GATE_SLOTS \
    CROSS_CHUNKS=$CROSS_CHUNKS STAMP="$STAMP" \
        $SUB $DEP --array=0-2 pace/eval_deciding.sbatch
    echo ""
fi

# --------------------------------------------------------------------------
# 9. write capacity on the cell checkpoints
# --------------------------------------------------------------------------
if want write; then
    echo "--- 9. write capacity on the CELLS (the parent ran as 13329445) ---"
    # eval_write_capacity.sbatch takes RUN + CKPT_NAME rather than an ARM, so
    # the step is resolved here.  N_CHUNKS=8 and T_EVAL=8 match the cells'
    # training geometry; ACCUM_MAX is derived inside the sbatch as
    # N_CHUNKS * ACCUM_VECS, which is the 128 the cells trained with.
    for pair in "a1:$A1_RUN:accum" "a3:$A3_RUN:gated"; do
        ARM=${pair%%:*}; rest=${pair#*:}; RUN=${rest%%:*}; MODE=${rest##*:}
        LAST=$(ls -d $OUT_ROOT/$RUN/checkpoint_* 2>/dev/null \
               | sort -t_ -k2 -n | tail -1)
        CKPT_NAME=$(basename "$LAST")
        STEP=$(echo "$CKPT_NAME" | sed 's/^checkpoint_//; s/_.*$//')
        if [ "$STEP" -lt 115966 ] 2>/dev/null; then
            echo "    REFUSED $ARM: $CKPT_NAME is step $STEP, below the cells'"
            echo "            stop step 115966.  Probe or unfinished cell."
            continue
        fi
        echo "    $ARM -> $RUN/$CKPT_NAME ($MODE)"
        OUT_ROOT="$OUT_ROOT" RUN="$RUN" CKPT_NAME="$CKPT_NAME" \
            BASE=ckpts/olmo-retrofit-cortex EVAL_TAG="cells-$STAMP" \
            PREFIX_MODE="$MODE" ACCUM_VECS=$ACCUM_VECS GATE_SLOTS=$GATE_SLOTS \
            N_CHUNKS=$CROSS_CHUNKS T_EVAL=8 MAX_EXAMPLES=${WC_N:-50} \
            DATA="$EVAL_DATA" \
            $SUB $DEP pace/eval_write_capacity.sbatch
    done
    echo ""
fi

# --------------------------------------------------------------------------
# 10. the E-norm drift curve
# --------------------------------------------------------------------------
if want norm; then
    echo "--- 10. P2.5 E-norm drift, both arms (accum is the control) ---"
    # No dependency: this one reads the OLD pack by design.  The curve is a
    # comparison of six checkpoints against each other, so what matters is that
    # every point reads the SAME text, not that the text is plentiful.
    for ARM in a3 a1; do
        ARM=$ARM ACCUM_VECS=$ACCUM_VECS GATE_SLOTS=$GATE_SLOTS \
        CROSS_CHUNKS=$CROSS_CHUNKS DATA="$OLD_PACK" STAMP="$STAMP" \
            $SUB pace/norm_trace.sbatch
    done
    echo ""
fi

echo "=== submitted.  Read them together, not one at a time. ==="
echo ""
echo "READING ORDER, because two of these can make the third unreadable:"
echo "  1. the horizon's parent row     -- if it disagrees with P2.1's parent,"
echo "                                     the pack changed something and every"
echo "                                     cross-run number needs the caveat"
echo "  2. the horizon's I(d) on a3     -- still ~2 chunks after 24k steps?"
echo "                                     that makes cc8 over-provisioned and"
echo "                                     pace/submit_cc_geometry.sh the next job"
echo "  3. write capacity, cells vs parent  -- did the balance move?"
echo "  4. the norm curves, a3 AGAINST a1   -- a gated drift is only a gate"
echo "                                     result if accum's is flatter"
