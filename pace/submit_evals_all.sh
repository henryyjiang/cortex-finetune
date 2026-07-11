#!/bin/bash
# Submit long-context + basic evals for every finished run + the base control,
# all under one shared EVAL_TAG so results land in a single results root.
#
#   bash pace/submit_evals_all.sh                 # everything with a final_checkpoint
#   RUNS="rung1-k4 rung1-k0" bash pace/...        # explicit subset
#   SKIP_BASIC=true bash pace/...                 # long-context only
#   SKIP_NOCHUNK=true bash pace/...               # chunked mode only
#
# Each run gets TWO long-context jobs:
#   chunked  — NUM_CHUNKS=4, 1024-token windows (the trained buffer regime;
#              memory is the only bridge between subwindows)
#   no-chunk — NUM_CHUNKS=1 SEQ_LEN=4096 (full context in one window where it
#              fits; >4k contexts fall back to 4096-token chunking) with
#              CCOT_PASSES=4: four silent full passes over the window, buffer
#              carried pass-to-pass, then generate (~4*56=224 layer-apps/token)
# Base long-context controls: chunked (sees final chunk only), no-chunk T=8
# (full attention, 56/token — per-pass-equal reference), and no-chunk T=32
# (200/token — k-matched vs the 4-pass cortex arm; extra passes are no-ops
# for the base, so its compute axis is T).
#
# Run from the repo root on a login node.  Prereq: eval datasets downloaded
# (python evals/download_datasets.py).

set -e
cd "$(dirname "$0")/.."

EVAL_TAG=${EVAL_TAG:-$(date +%Y%m%d)}
SKIP_BASIC=${SKIP_BASIC:-false}
SKIP_NOCHUNK=${SKIP_NOCHUNK:-false}

if [ -z "$RUNS" ]; then
    RUNS=$(ls -d cortex-retro-ft/*/final_checkpoint 2>/dev/null \
           | sed 's|cortex-retro-ft/||; s|/final_checkpoint||')
fi
[ -n "$RUNS" ] || { echo "No */final_checkpoint under cortex-retro-ft/"; exit 1; }

echo "EVAL_TAG=$EVAL_TAG"
echo "Runs: $RUNS + base"

# Fix all checkpoint dirs for eval loading up front (idempotent).
python tools/prepare_eval_checkpoint.py --base ckpts/olmo8-cortex \
    --runs_root cortex-retro-ft

for RUN in $RUNS; do
    # chunked (trained regime)
    RUN=$RUN EVAL_TAG=$EVAL_TAG sbatch pace/eval_longcontext.sbatch
    # no-chunk: full window, 4 latent passes, then generate
    [ "$SKIP_NOCHUNK" = "true" ] || \
        RUN=$RUN NUM_CHUNKS=1 SEQ_LEN=4096 CCOT_PASSES=4 EVAL_TAG=$EVAL_TAG \
            sbatch pace/eval_longcontext.sbatch
    [ "$SKIP_BASIC" = "true" ] || RUN=$RUN EVAL_TAG=$EVAL_TAG sbatch pace/eval_basic.sbatch
done

# Base controls
BASE=true EVAL_TAG=$EVAL_TAG sbatch pace/eval_longcontext.sbatch
if [ "$SKIP_NOCHUNK" != "true" ]; then
    BASE=true NUM_CHUNKS=1 SEQ_LEN=4096 EVAL_TAG=$EVAL_TAG \
        sbatch pace/eval_longcontext.sbatch                      # full-attn, T=8
    BASE=true NUM_CHUNKS=1 SEQ_LEN=4096 T_OVERRIDE=32 EVAL_TAG=$EVAL_TAG \
        sbatch pace/eval_longcontext.sbatch                      # full-attn, k-matched
fi
[ "$SKIP_BASIC" = "true" ] || BASE=true EVAL_TAG=$EVAL_TAG sbatch pace/eval_basic.sbatch

echo "Submitted. Results -> eval_results/{longcontext,basic}_${EVAL_TAG}/<label>/"
echo "  chunked:   <run>            base"
echo "  no-chunk:  <run>-ppc1-ccot4-nc1-sl4096   base-nc1-sl4096   base-T32-nc1-sl4096"
