#!/bin/bash
# Ceiling probes (2026-08-04) — the follow-up to the oracle ceiling measurement.
#
# WHAT THE CEILING RUN LEFT OPEN.  Track A recovers 32.5% of its oracle ceiling
# (0.0237 / 0.0729) on PG-19 at 1024-token chunks.  That is ONE point in the
# space (corpus, chunk length, read depth), and it decomposes into two separate
# questions that need different fixes:
#
#   NUMERATOR — recover more of what is there.  A mechanism/read question.
#   DENOMINATOR — make more of it be there in the first place.  A data and
#                 chunk-geometry question, and nothing about the mechanism
#                 changes it.
#
# Every job here is a forward-only eval on an A100, n=50, minutes not hours.
# None of them touches the B2 heal run.  Run them from a login node at the repo
# root.  Submit as a block; they are independent.
#
#   bash pace/submit_ceiling_probes.sh              # all of it
#   ONLY=denominator bash pace/submit_ceiling_probes.sh
#   ONLY=numerator   bash pace/submit_ceiling_probes.sh
#
# Prereq for probe D3 only: the code pack (login node, ~5 min, needs internet):
#   python tools/prepare_pg19_dataset.py --tokenizer ckpts/olmo8-cortex \
#       --dataset codeparrot/codeparrot-clean-valid --split train \
#       --text_col content --min_tokens 3500 --max_samples 20000 \
#       --out data/code_olmo_val_len4096 --max_length 4096
# It prints "kept N/20000 documents" — if N < 200 the probe is underpowered,
# lower --min_tokens to 3000 and re-run.

set -e
cd "$(dirname "$0")/.."

TAG=${TAG:-ceil2-$(date +%Y%m%d)}
ARM=${ARM:-rung1-k0-acc4v-tb2-rs-ep3-rcl}    # Track A, the 32.5% arm
BASE_MODEL=${BASE_MODEL:-ckpts/olmo8-cortex}  # the arm's own base
ONLY=${ONLY:-all}

echo "TAG=$TAG  ARM=$ARM"

# =====================================================================
# DENOMINATOR — is the data (or the chunk geometry) the binding constraint?
# =====================================================================
if [ "$ONLY" != "numerator" ]; then

# D1. Ceiling on the ARM data.  Nemotron-CC-Math documents are short and
#     EOS-separated, so chunk g is usually a DIFFERENT problem than g-1 and the
#     cross-chunk ceiling there may be ~0 by construction.  If it is, the arm
#     phase cannot measure memory at all, whatever the mechanism does — and
#     that is a launch-blocking fact about B2's second phase, not a nice-to-have.
#     Uses the 400M pack that is too small to train on.
#
#     Run it on olmo8-cortex, NOT on the retrofit base: olmo8 already has a
#     PG-19 ceiling on the books (+0.1036 at k=256) measured on these exact
#     weights, so the comparison is free and needs no control job.  The retrofit
#     base is mid-conversion and its absolute NLL is not a scale anything can be
#     read against yet.
MODEL=$BASE_MODEL DATA=data/nemotron_math_olmo_len4096 \
    T_EVAL=8 EVAL_TAG=${TAG}-D1-nemotron POS_BUCKETS=4 \
    sbatch pace/eval_context_ceiling.sbatch

#     Optional, only if the B2 lineage's own number is wanted: the retrofit base
#     has no published PG-19 ceiling, so it needs its paired control run too.
if [ "${D1_RETROFIT:-false}" = "true" ]; then
    for D in nemotron_math_olmo_len4096 pg19_olmo_val_len4096; do
        MODEL=ckpts/olmo-retrofit-cortex DATA=data/$D T_EVAL=8 \
            EVAL_TAG=${TAG}-D1r-${D%%_*} POS_BUCKETS=4 \
            sbatch pace/eval_context_ceiling.sbatch
    done
fi

# D2. Chunk length as a design lever.  At cross_chunks=4 the carry summarises a
#     1024-token chunk into a 2048-token window, where the base scores -0.977 —
#     we are asking it to compress a span the model cannot use even VERBATIM.
#     At cross_chunks=8 (512-token chunks) the window is 1024, inside the usable
#     range.  Per-boundary ceiling may fall; boundaries double, so read the
#     TOTAL, and read the position bands — a front-loaded ceiling is what makes
#     more boundaries a real win rather than the same nats resliced.
N_CHUNKS=8 CONTEXT_LENS="64 128 256 512" MODEL=$BASE_MODEL \
    DATA=data/pg19_olmo_val_len4096 T_EVAL=8 EVAL_TAG=${TAG}-D2-nc8-base \
    POS_BUCKETS=4 sbatch pace/eval_context_ceiling.sbatch

#     And on the arm, which also gives the fraction recovered at nc=8 — note the
#     arm was TRAINED at cross_chunks=4, so this is an off-distribution read and
#     a low fraction here is not evidence against the lever.  The base row is
#     the one that prices the lever.
N_CHUNKS=8 CONTEXT_LENS="64 128 256 512" RUN=$ARM \
    DATA=data/pg19_olmo_val_len4096 T_EVAL=8 EVAL_TAG=${TAG}-D2-nc8-arm \
    POS_BUCKETS=4 sbatch pace/eval_context_ceiling.sbatch

# D3. A corpus with real long-range structure.  A function defined in chunk 1
#     and called in chunk 4 is the dependency natural prose does not have, and
#     it is semantic-but-not-verbatim — the middle ground a compressed carry
#     could plausibly serve.  --min_tokens keeps only window-spanning files, so
#     this is not the short-document failure mode again.
if [ -d data/code_olmo_val_len4096 ]; then
    MODEL=$BASE_MODEL DATA=data/code_olmo_val_len4096 T_EVAL=8 \
        EVAL_TAG=${TAG}-D3-code POS_BUCKETS=4 \
        sbatch pace/eval_context_ceiling.sbatch
    RUN=$ARM DATA=data/code_olmo_val_len4096 T_EVAL=8 \
        EVAL_TAG=${TAG}-D3-code-arm POS_BUCKETS=4 \
        sbatch pace/eval_context_ceiling.sbatch
else
    echo "SKIP D3: data/code_olmo_val_len4096 missing (see the header for the"
    echo "         one prepare_pg19_dataset.py command that builds it)"
fi

fi

# =====================================================================
# NUMERATOR — can we recover more of the nats already on the table?
# Both probes run on the EXISTING Track A checkpoint.  No training.
# =====================================================================
if [ "$ONLY" != "denominator" ]; then

# N1. Is the carry redundant with context that is free anyway?  The +0.0237 is
#     measured against a k=0 floor no deployed model sits at: at inference the
#     tail of the previous chunk costs nothing to keep.  If carry+128 is no
#     better than 128 alone, the carry is a local-context surrogate, the
#     addressable headroom is (0.0729 - 0.0871-at-k128) rather than the whole
#     ceiling, and "32.5% recovered" is measuring against the wrong denominator.
#     This is the cheapest test that could change the paper's headline number.
RUN=$ARM DATA=data/pg19_olmo_val_len4096 T_EVAL=8 \
    CARRY_PLUS="128 256" POS_BUCKETS=4 EVAL_TAG=${TAG}-N1-residual \
    sbatch pace/eval_context_ceiling.sbatch

# N2. Is the READ depth-limited?  The carry is read by the same loop that does
#     the compressing, so more recurrence at eval time is free extra depth
#     applied to the same fixed carry.  Sweep T with everything else pinned.
#     The ceiling MOVES with T too, so each job reports its own denominator —
#     the quantity to compare across the three is the fraction, never the delta.
for T in 4 8 16; do
    RUN=$ARM DATA=data/pg19_olmo_val_len4096 T_EVAL=$T \
        CARRY_PLUS="128" EVAL_TAG=${TAG}-N2-T$T \
        sbatch pace/eval_context_ceiling.sbatch
done

fi

echo
echo "Submitted. Results -> eval_results/context_ceiling_${TAG}-*/<label>/results.json"
echo "Read in this order:"
echo "  D1 nemotron vs its pg19 control : does the arm phase have a ceiling AT ALL"
echo "  N1 residual_vs_k_alone          : is the carry additive or a local surrogate"
echo "  D2 nc8 vs nc4, position_bands   : does chunk geometry buy addressable nats"
echo "  N2 fraction vs T                : is the read depth-limited"
echo "  D3 code                         : does a structured corpus have a bigger ceiling"
