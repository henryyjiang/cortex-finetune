#!/bin/bash
# Clean re-run of the rung1 memory-ablation trio (-v2 names, fresh out dirs).
#
# WHY: the original rung1-k0 / rung1-k4 / rung1-k0-ccot trained BEFORE the
# post_init clobber fix — their memory read (out_proj / DirectCCoT.in_proj)
# was RANDOM at step 0, not zero-init, so they were never step-0 == base and
# injected noise from the first step.  rung1b/rung2/rung3 trained after the
# fix and stay valid as-is (rung2-k4-unfreeze500 is evaluated alongside, no
# retrain).
#
# Recipe is byte-identical to the originals (PG-19 4096 x cross_chunks=4,
# frozen loop, same seed) — the ONLY difference is the fixed init, so
# v2-vs-v1 deltas are attributable to the clobber.
#
# Usage (login node, repo root):  bash pace/submit_rung1_v2.sh

set -e
cd "$(dirname "$0")/.."

RUN_NAME=rung1-k4-v2                                        sbatch pace/rung1_frozen_loop.sbatch
RUN_NAME=rung1-k0-v2      MEMORY_SLOTS=0                    sbatch pace/rung1_frozen_loop.sbatch
RUN_NAME=rung1-k0-ccot-v2 MEMORY_SLOTS=0 CCOT_DIRECT=true   sbatch pace/rung1_frozen_loop.sbatch

echo "Submitted rung1 v2 trio: rung1-k4-v2, rung1-k0-v2, rung1-k0-ccot-v2"
echo "When finished, eval with:"
echo '  RUNS="rung1-k4-v2 rung1-k0-v2 rung1-k0-ccot-v2 rung2-k4-unfreeze500" bash pace/submit_evals_all.sh'
