#!/bin/bash
#SBATCH --account=gts-aivanova7-lab
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=64GB
#SBATCH --gres=gpu:A100:1
#SBATCH -t 1:00:00
#SBATCH -q inferno
#SBATCH -o logs/Report-%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=henryyjiang42@gmail.com

# Pre-submit gate for a memory-arm launch: run the prefix path on the REAL 1B
# checkpoint at the geometry the run will use, and fail loudly on anything that
# a healthy loss curve would otherwise hide.
#
# WHY THIS IS NOT OPTIONAL (2026-08-04).  ckpts/olmo-retrofit-cortex was carrying
# a raven_modeling_minimal_cortex.py that predated the 2026-08-02 prefix rewrite.
# cortex_graft.py is imported LIVE from the repo root, but that modeling file is
# a SNAPSHOT copied into the checkpoint dir — so cortex.prefix existed, every
# --cortex.* flag was correct, the wandb config verified, and forward() never
# called prefix_pack.  The first phase-0 attempt ran 9 hours with NO
# cross-segment memory and a perfectly normal loss curve.  Recurring bug class 2.
# This script is the only check in the pipeline that reads the MODELING FILE
# rather than the config.
#
# BOTH GEOMETRIES, 2026-09-16 (P1.0).  A1 and A3' are different buffers, and the
# accum smoke shares not one assertion with the gated one: the gated carry stops
# growing at K, overwrites rows in place, and "preserved" is SUPPOSED to go
# False.  Running the accum arm and calling A3' covered would be the same
# mistake this script exists to prevent, one level up.  So ARMS defaults to
# both, each at ITS OWN geometry — the one the arm will actually launch on.
#
#   sbatch pace/smoke_prefix_real.sbatch                 # accum + gated
#   ARMS=gated sbatch pace/smoke_prefix_real.sbatch      # A3' only
#   ARMS=accum sbatch pace/smoke_prefix_real.sbatch      # A1 only (old default)
#   ARMS=latent sbatch pace/smoke_prefix_real.sbatch     # the Z channel (a3z)
#   ARMS="accum gated latent" sbatch pace/smoke_prefix_real.sbatch   # all three
#   MODEL=ckpts/olmo8-cortex sbatch pace/smoke_prefix_real.sbatch   # Track-A base
#   FULL=1 sbatch pace/smoke_prefix_real.sbatch          # accum at 4 x 1024, T=8
#
# ACCUM geometry is 2 x 1024 at T=8, which reaches max packed position 1056 —
# the case no toy config ever exercised, and the one the prefix_pos='tail' change
# introduced.  FULL=1 runs the true 4-chunk training shape; that is the backward
# graph that OOMed both B0 arms at 78.4GiB on an A100-80, so it may not fit here.
# It carries no optimizer state, so it is worth trying, but an OOM there is a
# memory limit and not a code failure.
#
# GATED geometry is A3' EXACTLY: W=16, K=64, 8 x 512 tokens, carry_grad_chunks 4.
# Three of those numbers are load-bearing and none of them is a default:
#
#   * cross_chunks 8 is the MINIMUM that fires the gate at K/W=4.  The gate only
#     acts from lap 2 (lap 1 is a plain append under fill="grow"), so a shorter
#     chain trains an accum buffer with 16.8M dead parameters behind a healthy
#     loss curve.  tools/smoke_prefix_real.py refuses the geometry outright
#     rather than reporting a green run; train.py asserts the same bound.
#   * carry_grad_chunks 4, not 1.  merge runs after the chunk's own loss, so the
#     gated state only reaches a loss through the NEXT chunk — and at
#     carry_grad_chunks=1 every chunk detaches its incoming carry, so
#     gate_proj_mem's gradient is None, not small.  The forget gate would never
#     train.  (tests/test_smoke_prefix_real.py pins exactly this.)
#   * accum_vecs 16, not 32.  W is summary_emb's shape and cannot change after
#     training, so the smoke must run the width the arm was BRANCHED at — see
#     tools/slice_summary_emb.py, which produces the [16,D] checkpoint.
#
# THE LATENT (Z) ARM adds the dual-channel carry, and it checks TWO MORE hooks
# that live in the modeling file and nowhere else: `latent_init` (the s0
# substitution, in iterate_forward) and `iter_write(x, current_step)` (the
# trajectory tape, in core_block_forward).  Same bug class, one channel down: a
# stale snapshot would build the dual-channel buffer, allocate its second gate,
# pass every config check, and carry a Z half that is never written or read.
#
# It also reports something no other check can, and that is not in any design
# note before 2026-09-16: A SINGLE NO-GRAD LOOP STEP CUTS Z'S READ GRADIENT TO
# EXACTLY ZERO.  E re-enters the loop through `input_embeds` on every iteration
# and is unaffected; Z enters once, at s0, and the no-grad iterations run FIRST.
# In training n = 0 only when the sampled p <= mean_backprop_depth, so the
# fraction is a TRAINING-RUN AVERAGE -- roughly half the batches at mr8/depth 8,
# essentially none at mr32.  The smoke prints the measured fraction and then
# runs one extra chain at num_steps=[0, T] so the ARCHITECTURE question (is the
# gate wired?) is answered separately from the SCHEDULE question (how often is
# it live?).  Put the fraction in the Z pre-registration: it changes what a null
# result for Z means.
#
# A PASSING GATED RUN PROVES, on real weights and through the packed forward:
# the carry grows by W and stops at K; the ring gates exactly its own W rows and
# passes the other K-W through bit-identical; depth_rows(j) addresses the rows
# the forward actually wrote (the depth-slice ablation's addressing half); and
# both gate projections plus forget_bias are on the loss.

cd $SLURM_SUBMIT_DIR

module load anaconda3
module load cuda/12.1.1
conda activate cortex-retro

export SCRATCH=${SCRATCH:-$HOME/scratch}
export HF_HOME=$SCRATCH/hf_cache
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODEL=${MODEL:-ckpts/olmo-retrofit-cortex}
T_SMOKE=${T_SMOKE:-8}
ARMS=${ARMS:-"accum gated"}

# accum (A1)
CHUNK_LEN=${CHUNK_LEN:-1024}
CROSS_CHUNKS=${CROSS_CHUNKS:-2}
ACCUM_VECS=${ACCUM_VECS:-32}
ACCUM_MAX=${ACCUM_MAX:-128}
if [ -n "$FULL" ]; then CHUNK_LEN=1024; CROSS_CHUNKS=4; fi

# gated (A3'), and every one of these is the arm's own number — see the header
GATE_CHUNK_LEN=${GATE_CHUNK_LEN:-512}
GATE_CROSS_CHUNKS=${GATE_CROSS_CHUNKS:-8}
GATE_VECS=${GATE_VECS:-16}
GATE_SLOTS=${GATE_SLOTS:-64}
GATE_ROUTE=${GATE_ROUTE:-ring}
GATE_NORM=${GATE_NORM:-tanh}
GATE_INIT=${GATE_INIT:-zero}
GATE_FILL=${GATE_FILL:-grow}
GATE_GRAD_CHUNKS=${GATE_GRAD_CHUNKS:-4}

echo "=== prefix smoke: $MODEL | arms: $ARMS | T=$T_SMOKE ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "prefix_pack occurrences in the checkpoint's modeling file:"
grep -c prefix_pack "$MODEL/raven_modeling_minimal_cortex.py" \
    || echo "  (no modeling file at $MODEL — is it graft-prepared?)"

FAILED=0
for ARM in $ARMS; do
    echo ""
    case "$ARM" in
    accum)
        echo "######## A1 / accum: ${CROSS_CHUNKS} x ${CHUNK_LEN} tok," \
             "W=${ACCUM_VECS} cap=${ACCUM_MAX} ########"
        python tools/smoke_prefix_real.py \
            --model_name "$MODEL" \
            --prefix_memory accum \
            --chunk_len "$CHUNK_LEN" \
            --cross_chunks "$CROSS_CHUNKS" \
            --accum_vecs "$ACCUM_VECS" \
            --accum_max "$ACCUM_MAX" \
            --T "$T_SMOKE" \
            --device cuda --dtype bfloat16 || FAILED=1;;
    gated)
        echo "######## A3' / gated ring: ${GATE_CROSS_CHUNKS} x ${GATE_CHUNK_LEN}" \
             "tok, W=${GATE_VECS} K=${GATE_SLOTS} route=${GATE_ROUTE} ########"
        python tools/smoke_prefix_real.py \
            --model_name "$MODEL" \
            --prefix_memory gated \
            --chunk_len "$GATE_CHUNK_LEN" \
            --cross_chunks "$GATE_CROSS_CHUNKS" \
            --accum_vecs "$GATE_VECS" \
            --gate_slots "$GATE_SLOTS" \
            --gate_route "$GATE_ROUTE" \
            --gate_norm "$GATE_NORM" \
            --gate_init "$GATE_INIT" \
            --gate_fill "$GATE_FILL" \
            --carry_grad_chunks "$GATE_GRAD_CHUNKS" \
            --T "$T_SMOKE" \
            --device cuda --dtype bfloat16 || FAILED=1;;
    latent)
        echo "######## a3z / gated ring + Z: ${GATE_CROSS_CHUNKS} x"              "${GATE_CHUNK_LEN} tok, W=${GATE_VECS} K=${GATE_SLOTS} ########"
        python tools/smoke_prefix_real.py             --model_name "$MODEL"             --prefix_memory gated             --chunk_len "$GATE_CHUNK_LEN"             --cross_chunks "$GATE_CROSS_CHUNKS"             --accum_vecs "$GATE_VECS"             --gate_slots "$GATE_SLOTS"             --gate_route "$GATE_ROUTE"             --gate_norm "$GATE_NORM"             --gate_init "$GATE_INIT"             --gate_fill "$GATE_FILL"             --carry_grad_chunks "$GATE_GRAD_CHUNKS"             --latent_carry             --latent_depth_rule "${LATENT_RULE:-absolute}"             --latent_renorm "${LATENT_RENORM:-none}"             --T "$T_SMOKE"             --device cuda --dtype bfloat16 || FAILED=1;;
    *)
        echo "ARMS must be drawn from {accum, gated, latent}; got '$ARM'"
        exit 1;;
    esac
done

echo ""
if [ $FAILED -ne 0 ]; then
    echo "=== SMOKE FAILED — do not launch.  Read the [FAIL] lines above. ==="
    exit 1
fi
echo "=== all requested arms passed ($ARMS) ==="
