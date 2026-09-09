#!/bin/bash
# Does the carry buffer's FIFO cap explain the LongMemEval regression?
#
# LongMemEval only, small n, one carry-off control plus a ladder of buffer
# capacities.  This is a BLOW-UP CHECK, not a result: the question is whether
# raising accum_max past its trained value produces a usable model at all, and
# whether the carry starts paying off when it is allowed to keep more than the
# newest 8 chunks.
#
#   ./pace/submit_accum_max_probe.sh                 # submit the ladder
#   DRY=true ./pace/submit_accum_max_probe.sh        # print, submit nothing
#   N=200 ./pace/submit_accum_max_probe.sh           # bigger probe
#
# Why this is the right first experiment.  B2's buffer holds accum_max/accum_vecs
# = 256/32 = 8 chunks.  At LongMemEval's ~224 priming chunks of 512 tokens, the
# carry that reaches the answer summarises the last ~4096 tokens of a ~115,000
# token haystack -- 3.6%, and specifically the 3.6% adjacent to the final
# window, which the ceiling probes showed the carry is nearly a substitute for.
# So the -6.5 pt carry-off advantage may be measuring a buffer that never had
# the answer in it, rather than a memory that cannot use one.
#
# What "blows up" looks like, and where to read it.  Every run writes
# records.jsonl with per-example p_eos_first and entropy_first:
#   * p_eos_first climbing with capacity  -> the carry is pushing the model to
#     stop before it answers; that IS the failure mode behind the -6.5 pt, and
#     more capacity makes it worse.
#   * entropy_first collapsing or exploding -> the OOD prefix length has broken
#     the output distribution; the run is uninterpretable, stop the ladder.
#   * gold_nll_per_tok rising with capacity -> more context is actively hurting.
# Any of those, and the answer is "capacity is not the fix" -- which is worth
# knowing for two GPU-hours before the 40-hour grid.
#
# The ladder is ordered cheapest-first and each rung roughly doubles the
# priming cost, since every chunk's forward attends over n_carried + 512 + 32
# columns.  Watch the first two land before trusting the wall-clock on 2048.
#
# accum_max -> chunks kept -> LongMemEval context retained (~115k tokens):
#     256 (trained)   8 chunks     3.6%
#     512            16 chunks     7.1%
#    1024            32 chunks    14.2%
#    2048            64 chunks    28.5%
# Full retention would need 7168 and is not affordable here; BABILong 32k needs
# only 1984 and is the cheap place to test full retention once this looks sane.

set -euo pipefail
cd "$(dirname "$0")/.."

RUN=${RUN:-retro-b2-acc32-cc8-mr8}
EVAL_TAG=${EVAL_TAG:-b2-ammax}
OUT_ROOT=${OUT_ROOT:-cortex-retrofit}
CKPT_NAME=${CKPT_NAME:-final_checkpoint}
PREP_BASE=${PREP_BASE:-ckpts/olmo-retrofit-cortex}
N=${N:-50}          # per session-depth bucket; 2 buckets are populated -> 2N
DRY=${DRY:-false}

COMMON="OUT_ROOT=$OUT_ROOT CKPT_NAME=$CKPT_NAME PREP_BASE=$PREP_BASE \
RUN=$RUN EVAL_TAG=$EVAL_TAG SEQ_LEN=512 NUM_CHUNKS=8 T_OVERRIDE=8 \
SKIP_BABILONG=true LME_MAX=$N SCORE_NLL=true"

submit () {   # submit <wall> <extra env...>
    local wall="$1"; shift
    local cmd="env $COMMON $* sbatch -t $wall pace/eval_longcontext.sbatch"
    echo "  $cmd"
    [ "$DRY" != "true" ] && eval "$cmd" || true
}

echo "=== control: carry-off (no buffer, so accum_max does not apply) ==="
submit 01:00:00 NO_CARRY=true

echo "=== carry-on ladder ==="
submit 01:00:00 ACCUM_MAX=256      # the trained value: reproduces the -6.5 pt
submit 02:00:00 ACCUM_MAX=512
submit 03:00:00 ACCUM_MAX=1024
submit 06:00:00 ACCUM_MAX=2048

cat <<'NOTE'

The ACCUM_MAX=256 rung is not redundant: it re-runs the trained capacity under
the NEW harness (per-example seeding, records, NLL) so the ladder is internally
paired and is not being compared against the old unseeded numbers.

Read it with:

  for AM in 256 512 1024 2048; do
    python tools/analyze_longcontext_pairs.py --pairs --no_cells \
      --on  eval_results/longcontext_b2-ammax/RUN-T8-nc8-sl512-am$AM \
      --off eval_results/longcontext_b2-ammax/RUN-T8-nc8-sl512-nocarry
  done

and the blow-up diagnostics with:

  python tools/summarize_carry_health.py --root eval_results/longcontext_b2-ammax

Check `[cortex] carry buffer:` in each job's log FIRST.  It prints the max_vecs
the model was actually built with and where the value came from.  If it says
CODE DEFAULT on the ACCUM_MAX=256 rung, the checkpoint's config.json is missing
accum_max and every earlier long-context number ran at a 4-chunk horizon, not 8.
NOTE
