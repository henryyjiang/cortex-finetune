#!/bin/bash
# PASS 1 -- the four non-Z checks, all short, all forward-only or data-free.
#
#   bash pace/submit_pass1.sh            # submit all four
#   DRY=1 bash pace/submit_pass1.sh      # print the commands and stop
#
# WHY THESE FOUR AND WHY NOW.  Each one either decides a full-run parameter or
# repairs an instrument that a later, expensive measurement depends on.  None of
# them trains anything.  Total is well under three GPU-hours, against the ~32 h
# per arm the P2.3 cells (ALREADY RUNNING, launched separately) are spending.
#
#   1  red10_confirm      ~20 min   confirms the label-shift fix; unblocks the
#                                   walk, gate 4 and the width contrast at once
#   2  smoke cc16 x2      ~20 min   reprices the 99% OOM with the one unpriced
#                                   lever; decides K, eviction and the horizon
#   3  gate_geometry_ad   ~20 min   what the d=4 spike IS, at buffer level
#   4  write_capacity     ~40 min   the B2-style whole-carry donor control, on a
#                                   loading path that is not at chance
#
# NOTHING HERE TOUCHES Z.  Every one of these questions survives a Z null, and
# three of them are full-run parameters that the Z work cannot re-decide later.
#
# ORDER.  1 and 2 are independent and go first.  3 and 4 are also independent of
# both -- they are queued together because the queue, not the dependency graph,
# is the constraint.  Nothing in pass 1 depends on another pass-1 result, which
# is the point: one wave, one reading session.
#
# WHAT PASS 2 NEEDS FROM THIS, so the reading has a purpose:
#   red10 clean        -> the width statistic (put PR and entropy on the same
#                         power of s) can be settled off the walk JSONs, no GPU
#   cc16 fits          -> re-spec the full run at K=128/cc16; price the TBPTT
#                         change before committing
#   cc16 OOM at GRAD 1 -> cc8 is forced; close the geometry question
#   A(d) staircase     -> the lap is the ring's signature; publish it
#   A(d) spike         -> forget_bias / K:W is a full-run decision, make it
#   write_capacity     -> the donor/register split on a clean path; this is the
#                         number that licenses memory-on vs memory-off without a
#                         separate no-memory control
set -u
SUB="sbatch"
[ -n "${DRY:-}" ] && SUB="echo [dry] sbatch"

cd "$(dirname "$0")/.."

echo "=== pass 1: four checks, none of them Z ==="
echo ""

echo "--- 1. RED 10 confirmation + the two width walks ---"
$SUB pace/red10_confirm.sbatch

echo ""
echo "--- 2. cc16 repricing, both TBPTT windows ---"
# GEOMETRIES= (empty, and WITHOUT the colon in the script's default expansion)
# means "skip the settled config-D rows".  P1_BS is irrelevant to memory at
# micro_batch_size 1 and is left alone deliberately.
GEOMETRIES= P1=1 P1_CC=16 P1_K=128 P1_AMAX=256 P1_GRAD=2 $SUB pace/smoke_geometry_oom.sbatch
GEOMETRIES= P1=1 P1_CC=16 P1_K=128 P1_AMAX=256 P1_GRAD=1 $SUB pace/smoke_geometry_oom.sbatch

echo ""
echo "--- 3. buffer-level A(d): what the d=4 spike is ---"
$SUB pace/gate_geometry_ad.sbatch

echo ""
echo "--- 4. whole-carry donor control on the parent ---"
# The B2 table this reproduces (write_capacity_b2-final, chunks 2-8):
#   real +0.160 -> +0.211, donor -0.016 -> -0.042, chunk-1 register +0.024.
#   Ordering in loss: real < none < donor.  A donor buffer being WORSE than no
#   buffer is the content-sensitivity result that licenses comparing memory-on
#   against memory-off inside one model -- and it has never been re-measured on
#   a path that was not at chance.  n=50 is the PACK's limit, not the flag's.
OUT_ROOT=cortex-retrofit RUN=retro-b2-heal CKPT_NAME=checkpoint_91552_w16 \
    BASE=ckpts/olmo-retrofit-cortex EVAL_TAG=pass1-parent \
    N_CHUNKS=8 MAX_EXAMPLES=50 T_EVAL=8 \
    $SUB pace/eval_write_capacity.sbatch

echo ""
echo "=== submitted.  Read them together, not one at a time. ==="
echo "The P2.3 cells are running separately and none of this blocks them."
