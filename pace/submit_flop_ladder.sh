#!/bin/bash
# FLOP-matched k-ladder: ONE cortex run vs the base RDM at four compute tiers.
# 8 eval jobs total (4 base + 4 cortex), one shared EVAL_TAG.
#
# Per-token layer-apps: base = 8+6r, cortex = passes * (8+6T).
#   k=4 : base T=4  (32)  <->  cortex T=4                (32,  exact)
#   k=8 : base T=8  (56)  <->  cortex default            (56,  exact)
#   k=16: base T=16 (104) <->  cortex passes_per_chunk=2 (112, +8%)
#   k=32: base T=32 (200) <->  cortex passes_per_chunk=4 (224, +12%)
# NUM_CHUNKS stays at the sbatch default (4) across every tier so buffer
# depth is constant (the trained regime) and only compute varies.
#
# Run this AFTER the plain submit_evals_all.sh baseline confirms the fixed
# harness gives non-zero, discriminating numbers — the ladder is 8 GPU jobs.
#
# Usage (login node, repo root):
#   RUN=rung1-k4-v2 bash pace/submit_flop_ladder.sh
#   RUN=... EVAL_TAG=ladder0715 bash pace/submit_flop_ladder.sh

set -e
cd "$(dirname "$0")/.."

[ -n "$RUN" ] || { echo "Set RUN=<cortex run name>"; exit 1; }
EVAL_TAG=${EVAL_TAG:-ladder$(date +%Y%m%d)}
echo "EVAL_TAG=$EVAL_TAG  cortex run: $RUN"

# Base RDM tiers (no cross state -> compute axis is T only)
for T in 4 8 16 32; do
    BASE=true T_OVERRIDE=$T EVAL_TAG=$EVAL_TAG sbatch pace/eval_longcontext.sbatch
done

# Cortex tiers
RUN=$RUN T_OVERRIDE=4      EVAL_TAG=$EVAL_TAG sbatch pace/eval_longcontext.sbatch   # k=4
RUN=$RUN                   EVAL_TAG=$EVAL_TAG sbatch pace/eval_longcontext.sbatch   # k=8
RUN=$RUN PASSES_PER_CHUNK=2 EVAL_TAG=$EVAL_TAG sbatch pace/eval_longcontext.sbatch  # k=16
RUN=$RUN PASSES_PER_CHUNK=4 EVAL_TAG=$EVAL_TAG sbatch pace/eval_longcontext.sbatch  # k=32

echo "Submitted 8 ladder jobs -> eval_results/longcontext_${EVAL_TAG}/"
echo "  base-T4 base-T8 base-T16 base-T32"
echo "  ${RUN}-T4  ${RUN}  ${RUN}-ppc2-ccot0  ${RUN}-ppc4-ccot0"
