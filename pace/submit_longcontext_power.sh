#!/bin/bash
# Submit the powered long-context grid: T in {8,16,32} x {carry, no-carry},
# 250 examples per (task, length) bucket, paired per-example records + gold
# answer NLL.  This is the rerun of the n=100 tables in longcontext_b2-final
# with enough examples for the carry contrast to be testable.
#
#   ./pace/submit_longcontext_power.sh                # submit all six cells
#   DRY=true ./pace/submit_longcontext_power.sh       # print, submit nothing
#
# Design, in one paragraph.  Carry-on and carry-off see a byte-identical final
# window, so every example is a matched pair and the primary endpoint is
# McNemar on the discordant pairs, pooled over the 18 (task, length) cells --
# NOT a per-cell test, which at any affordable n is hopeless (a per-cell MDE
# at 250 is ~8 pt against a ~1 pt effect).  qa1-qa3 x 6 lengths x 250 = 4,500
# pairs per cell of the grid; pooled paired MDE is ~0.9-1.6 pt depending on the
# discordance rate, against an observed +0.89 pt.  That is the honest position:
# this design is powered for ~1.2 pt, so it will confirm the effect only if the
# true effect is at or above what n=100 measured.  The gold-answer NLL is the
# insurance -- continuous, paired, and sensitive well below 1 pt.
#
# Wall clock (measured on the n=100 runs, scaled by examples x chunks x T; see
# the walltimes in logs/Report-126911{477,479}.out):
#   T=8  carry  ~4.5 h    T=8  no-carry ~1.5 h
#   T=16 carry  ~8.5 h    T=16 no-carry ~2.5 h
#   T=32 carry ~16   h    T=32 no-carry ~4.5 h
# The T=32 carry-on cell is sharded one job per task so no single job sits on
# the wall-clock edge; tools/analyze_longcontext_pairs.py merges the shards.
# Total ~40 GPU-h.
#
# Prereqs (login node, once):
#   python evals/download_datasets.py     # now also fetches babilong-1k-samples
#   python tools/preflight_longcontext.py # confirms 250/bucket is actually there

set -euo pipefail
cd "$(dirname "$0")/.."

RUN=${RUN:-retro-b2-acc32-cc8-mr8}
EVAL_TAG=${EVAL_TAG:-b2-power}
OUT_ROOT=${OUT_ROOT:-cortex-retrofit}
CKPT_NAME=${CKPT_NAME:-final_checkpoint}
PREP_BASE=${PREP_BASE:-ckpts/olmo-retrofit-cortex}
DRY=${DRY:-false}

COMMON="OUT_ROOT=$OUT_ROOT CKPT_NAME=$CKPT_NAME PREP_BASE=$PREP_BASE \
RUN=$RUN EVAL_TAG=$EVAL_TAG SEQ_LEN=512 NUM_CHUNKS=8 \
BABILONG_MAX=250 BABILONG_REPO=RMT-team/babilong-1k-samples \
BABILONG_PATH=data/babilong-1k SCORE_NLL=true LME_MAX=0"

submit () {   # submit <wall> <extra env...>
    local wall="$1"; shift
    local cmd="env $COMMON $* sbatch -t $wall pace/eval_longcontext.sbatch"
    echo "  $cmd"
    if [ "$DRY" != "true" ]; then
        eval "$cmd"
    fi
}

echo "=== carry-off cells (cheap: no priming passes at all) ==="
submit 04:00:00 T_OVERRIDE=8  NO_CARRY=true
submit 06:00:00 T_OVERRIDE=16 NO_CARRY=true
submit 08:00:00 T_OVERRIDE=32 NO_CARRY=true

echo "=== carry-on cells ==="
submit 08:00:00 T_OVERRIDE=8
submit 14:00:00 T_OVERRIDE=16

echo "=== carry-on T=32, sharded by task (LongMemEval rides on the qa1 shard) ==="
submit 10:00:00 T_OVERRIDE=32 BABILONG_TASKS=qa1 BABILONG_OUT_SUFFIX=-qa1
submit 10:00:00 T_OVERRIDE=32 BABILONG_TASKS=qa2 BABILONG_OUT_SUFFIX=-qa2 SKIP_LME=true
submit 10:00:00 T_OVERRIDE=32 BABILONG_TASKS=qa3 BABILONG_OUT_SUFFIX=-qa3 SKIP_LME=true

cat <<'NOTE'

Sharded jobs write into the SAME run dir under different sub-dirs
(babilong-qa1/, babilong-qa2/, babilong-qa3/).  results.json / summary.csv are
per-shard; records.jsonl is what the analysis reads, and it globs and merges
them.  Do not run two shards of the same task -- ids would collide silently.

When everything lands:

  python tools/analyze_longcontext_pairs.py --pairs \
      --on  eval_results/longcontext_b2-power/RUN-T8-nc8-sl512 \
      --off eval_results/longcontext_b2-power/RUN-T8-nc8-sl512-nocarry

  python tools/analyze_longcontext_pairs.py --ladder \
      --root eval_results/longcontext_b2-power --arm RUN

The pairs run prints the measured discordance psi.  Feed it back into
tools/power_longcontext.py before designing any further rerun -- psi has been
guessed at up to now because no run ever wrote per-example records.
NOTE
