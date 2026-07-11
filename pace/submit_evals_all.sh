#!/bin/bash
# Submit long-context + basic evals for every finished run + the base control,
# all under one shared EVAL_TAG so results land in a single results root.
#
#   bash pace/submit_evals_all.sh                 # everything with a final_checkpoint
#   RUNS="rung1-k4 rung1-k0" bash pace/...        # explicit subset
#   SKIP_BASIC=true bash pace/...                 # long-context only
#
# Run from the repo root on a login node.  Prereq: eval datasets downloaded
# (python evals/download_datasets.py).

set -e
cd "$(dirname "$0")/.."

EVAL_TAG=${EVAL_TAG:-$(date +%Y%m%d)}
SKIP_BASIC=${SKIP_BASIC:-false}

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
    RUN=$RUN EVAL_TAG=$EVAL_TAG sbatch pace/eval_longcontext.sbatch
    [ "$SKIP_BASIC" = "true" ] || RUN=$RUN EVAL_TAG=$EVAL_TAG sbatch pace/eval_basic.sbatch
done
BASE=true EVAL_TAG=$EVAL_TAG sbatch pace/eval_longcontext.sbatch
[ "$SKIP_BASIC" = "true" ] || BASE=true EVAL_TAG=$EVAL_TAG sbatch pace/eval_basic.sbatch

echo "Submitted. Results -> eval_results/{longcontext,basic}_${EVAL_TAG}/<run>/"
