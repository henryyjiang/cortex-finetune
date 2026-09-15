#!/bin/bash
# P0.3 — cross-chunk ceiling WITHIN-document vs ACROSS-document, on the actual
# training packs.  2026-09-15.
#
# WHY THIS IS THE LAST PROBE BARRING THE CONTROL LAUNCH.
# cortex_next_phase_framework.md §10 ("Does not block the control") names
# exactly two probes on the control's critical path, P0.3 and P0.4.  P0.4 is
# answered: the strided re-pack yields 702,963 rows / 2.88B tokens and the old
# pack held 4.1% of PG-19.  This is the other one, and it is the last thing that
# can still move the MIX RATIO — which is a RECIPE parameter, which is the only
# thing the control is matched on.  The mix pack is already built at 50/50; if
# this says across-document boundaries carry nothing, the PG-19 share should
# rise, and discovering that after launch costs a rebuilt corpus AND a re-run
# control rather than an afternoon.
#
# THE MEASUREMENT.  prepare_packed_dataset.py joins documents with EOS.  A
# FineWeb-Edu document is ~1.1k tokens, so a 4,096-token row is three to four
# unrelated documents and most of its seven chunk boundaries have nothing worth
# carrying across them.  A carry cannot transfer information that does not
# exist, so those instances sit in the DENOMINATOR of every carry number the
# project has quoted and can never be in the numerator.  --doc_split re-pools
# the instances the ceiling eval already scored, so it is free and the paired
# comparison is untouched.
#
# GEOMETRY IS THE LOCKED CONTROL GEOMETRY, NOT THE EVAL DEFAULT.  N_CHUNKS=8 on
# 4,096-token rows = the 512-token chunks of D-fallback, so the boundaries
# scored are the ones the run will actually see.  CONTEXT_LENS tops out at 512 =
# exactly one chunk; the largest k is also what the 'local' scheme's window
# uses, so the tag covers every condition it labels.
#
# READ IT AS:
#   across ~0, within large  -> the packer is the binding constraint.  The real
#                               cross-chunk signal is only the within-document
#                               share; raise the long-doc leg before launching.
#   both large               -> issue 7 shrinks, 50/50 stands, launch.
#   both ~0 at this geometry -> neither arm can measure memory on this corpus
#                               and that is a pre-registerable fact, not a
#                               post-hoc excuse.
#   PG-19 strided not ~100% within -> the SPLIT is broken, not the corpus.
#                               That pack is doc-per-row by construction; it is
#                               the built-in check and must be read first.
#
# EOS_ID defaults to 100257, OLMo-2's real eos, which is what the packer writes
# (prepare_packed_dataset.py:117 takes tok.eos_token_id).  The 65504/65505/65509
# in every *Recurrent-OLMo* config are HUGINN's ids on a 100,352-vocab OLMo
# tokenizer and decode to ' skating' / ' creek' / ' mandated' — passing one here
# would tag every instance within-document and the probe would read as a clean
# corpus.  See recipe_sweep_findings.md §4.
#
# Forward-only, no training, no backward graph.  Minutes per job on an A100.
#
#   bash pace/submit_p03_doc_split.sh            # all of it
#   ONLY=base bash pace/submit_p03_doc_split.sh  # the data question alone
#   ONLY=arm  bash pace/submit_p03_doc_split.sh  # the dilution of B2's numbers

set -e
cd "$(dirname "$0")/.."

TAG=${TAG:-p03-$(date +%Y%m%d)}
# The DATA question wants converged weights, so the absolute NLL is a scale
# something can be read against.  olmo8-cortex also already has a PG-19 ceiling
# on the books (+0.1036 at k=256, nc=4) measured on these exact weights.  The
# retrofit base is mid-conversion and its NLL is not yet a scale — using it here
# would confound "this pack has no cross-chunk signal" with "this checkpoint
# cannot read yet".
BASE_MODEL=${BASE_MODEL:-ckpts/olmo8-cortex}
ARM=${ARM:-retro-b2-acc32-cc8-mr8}
N=${N:-100}
ONLY=${ONLY:-all}

GEOM="N_CHUNKS=8 CONTEXT_LENS=64 128 256 512"
echo "TAG=$TAG  base=$BASE_MODEL  arm=$ARM  n=$N  geometry: $GEOM"

if [ "$ONLY" != "arm" ]; then

# --- the packs the CONTROL will actually eat -------------------------------
# P3a is the headline: the post-heal mix, 940,000 rows, 70% of the run's tokens.
# P3b is the heal pack, 30% of the tokens, and the pure wrapped case — it is
#     FineWeb-Edu only, so whatever the packer costs shows up undiluted here.
# P3c is the check on the instrument, not on the corpus.  Read it FIRST.
for SPEC in \
    "P3a-mix:pg19_fw50_olmo_len4096" \
    "P3b-heal:fineweb_edu_olmo_len4096" \
    "P3c-pg19strided:pg19_olmo_len4096_strided" ; do
    LABEL=${SPEC%%:*}; PACK=${SPEC#*:}
    if [ ! -d "data/$PACK" ]; then
        echo "SKIP $LABEL: data/$PACK missing"
        continue
    fi
    MODEL=$BASE_MODEL DATA=data/$PACK T_EVAL=8 \
        N_CHUNKS=8 CONTEXT_LENS="64 128 256 512" \
        MAX_EXAMPLES=$N DOC_SPLIT=true POS_BUCKETS=4 \
        EVAL_TAG=${TAG}-${LABEL} sbatch pace/eval_context_ceiling.sbatch
done

# P3d. B2's arm corpus, retrospectively.  Not on the control's path, but it is
#      the same job and it prices the one thing nobody has priced: Nemotron-CC
#      problems are short and EOS-separated, so if its across share is near 100%
#      then B2's carry deltas were measured on a corpus that could not carry.
if [ -d data/nemotron_math_olmo_len4096 ]; then
    MODEL=$BASE_MODEL DATA=data/nemotron_math_olmo_len4096 T_EVAL=8 \
        N_CHUNKS=8 CONTEXT_LENS="64 128 256 512" \
        MAX_EXAMPLES=$N DOC_SPLIT=true POS_BUCKETS=4 \
        EVAL_TAG=${TAG}-P3d-nemotron sbatch pace/eval_context_ceiling.sbatch
fi

fi

if [ "$ONLY" != "base" ]; then

# P3e. The same split ON B2-FINAL, which adds the carry column: the fraction of
#      the ceiling recovered, within- vs across-document.  This is what converts
#      "the packer dilutes the corpus" into "the packer dilutes the MEASUREMENT"
#      — the number to quote if a carry delta is re-reported per-class.
#      T_EVAL=8 matches its training recurrence; do not compare a ceiling at one
#      depth to a carry at another.
OUT_ROOT=cortex-retrofit CKPT_NAME=final_checkpoint \
    BASE=ckpts/olmo-retrofit-cortex RUN=$ARM \
    DATA=data/pg19_fw50_olmo_len4096 T_EVAL=8 \
    N_CHUNKS=8 CONTEXT_LENS="64 128 256 512" \
    MAX_EXAMPLES=$N DOC_SPLIT=true POS_BUCKETS=4 \
    EVAL_TAG=${TAG}-P3e-arm-mix sbatch pace/eval_context_ceiling.sbatch

fi

echo
echo "Submitted. Results -> eval_results/context_ceiling_${TAG}-*/<label>/results.json"
echo "  doc_split.schemes.<local|prefix>.classes.<within|across>"
echo
echo "Read in this order:"
echo "  P3c strided : share_pct within must be ~100 — if not, the SPLIT is wrong"
echo "  P3a mix     : the number that decides whether 50/50 stands"
echo "  P3b heal    : the undiluted cost of wrapped packing"
echo "  P3e arm     : how much of B2's quoted carry was diluted"
echo "  P3d nemotron: retrospective — could the arm phase measure memory at all"
