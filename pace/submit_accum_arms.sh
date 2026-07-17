#!/bin/bash
# A1 — AccumCCoT memory-recipe arms (two-track plan, 2026-07-17, Track A).
#
# Background: the final bolt-on round closed every other finetuning arm and
# left AccumCCoT as the single live mechanism (carry delta +0.024 nats,
# one-hop 2k recall, zero >= 4k).  These arms change ONE variable each vs the
# acc4v baseline (rung1-k0-acc4v-tb2-rs-ep3-rcl):
#   MEMORY_SLOTS=0 ACCUM_CCOT=true ACCUM_VECS=4 CARRY_GRAD_CHUNKS=2
#   RANDOM_SEGMENTS=true EPOCHS=3 DATA_PATH=data/pg19_olmo_recall25_len4096
#
# `accum_vecs` (per-chunk compression fidelity) and `carry_grad_chunks`
# (gradient horizon) are the two dials pointed at qa3/>=4k.  Run A0 first
# (SLICE_ABLATE=both pace/eval_carry_ablation.sbatch + pace/diag_accum_buffer
# .sbatch) — if the oldest-vector ablation confirms all signal lives in the
# newest vectors, the horizon arms are the priority; if fidelity looks
# saturated, the vec arms are.
#
# Usage (from ~/cortex-finetune on the cluster):
#   bash pace/submit_accum_arms.sh round1    # the 2026-07-17 test round:
#                                            #   acc4v-nemo + k32-ga on pg19 & nemo
#   bash pace/submit_accum_arms.sh vecs      # A1-vec16 + A1-vec50
#   bash pace/submit_accum_arms.sh horizon   # A1-horizon (tb4; cc8-tb8 gated)
#   bash pace/submit_accum_arms.sh nemo      # A1-nemo one-off diagnostic
#   bash pace/submit_accum_arms.sh gated     # gated-accum LM2 k=32 (pg19 + nemo)
#
# Login-node prep (per stage, once):
#   nemo:    python tools/prepare_packed_dataset.py --tokenizer ckpts/olmo8-cortex \
#                --out data/nemotron_math_olmo_len4096 --max_length 4096 \
#                --max_tokens 400_000_000
#   horizon (cc8 arm only):
#            python tools/prepare_pg19_dataset.py --tokenizer ckpts/olmo8-cortex \
#                --out data/pg19_olmo_len8192 --max_length 8192
#            python tools/prepare_recall_mix.py \
#                --data data/pg19_olmo_len8192 --tokenizer ckpts/olmo8-cortex \
#                --out data/pg19_olmo_recall25_len8192 --frac 0.25

set -e
STAGE=${1:-}

RCL4K=data/pg19_olmo_recall25_len4096
NEMO4K=data/nemotron_math_olmo_len4096

submit_nemo_accum() {
    # AccumCCoT × Nemotron-CC — ONE-TIME exploratory diagnostic (design
    # principle 0.1.1): does Nemotron-CC's k=4 stabilizing effect (ki4-nemo
    # GSM8K 39.6 vs ki4-ep4 0.8) carry over to AccumCCoT?  Single run, NOT a
    # data-default change; default data stays the recall mix.
    MEMORY_SLOTS=0 ACCUM_CCOT=true ACCUM_VECS=4 CARRY_GRAD_CHUNKS=2 \
        RANDOM_SEGMENTS=true DATA_PATH=$NEMO4K \
        sbatch --time=48:00:00 pace/rung1_frozen_loop.sbatch
    echo "Submitted: rung1-k0-acc4v-tb2-rs-nemo (one-off diagnostic)"
}

submit_gated32() {
    # Gated-accumulation LM2 variant at k=32 (buffer-choice note): AccumCCoT's
    # extraction write, LM2 gated merge on 32 fixed slots — the
    # append-vs-gated-overwrite comparison at matched write path, and the
    # Track-B gated-candidate shakeout.  Two data arms: recall-mix pg19
    # (Track-A default) + nemotron.  carry_grad_chunks stays 0: the gated
    # state is mixed (rows not separable), so tb2 would whole-detach every 2
    # chunks and starve the memory-feedback gate (needs a >= 3-chunk graph).
    MEMORY_SLOTS=32 GATED_ACCUM=true RANDOM_SEGMENTS=true \
        EPOCHS=3 DATA_PATH=$RCL4K \
        sbatch --time=48:00:00 pace/rung1_frozen_loop.sbatch
    MEMORY_SLOTS=32 GATED_ACCUM=true RANDOM_SEGMENTS=true \
        DATA_PATH=$NEMO4K \
        sbatch --time=48:00:00 pace/rung1_frozen_loop.sbatch
    echo "Submitted: rung1-k32-ga-rs-ep3-rcl | rung1-k32-ga-rs-nemo"
}

case "$STAGE" in
round1)
    # The 2026-07-17 test round (Henry): nemotron×AccumCCoT diagnostic + the
    # k=32 gated-LM2 buffer on both data mixes.  Needs $NEMO4K prepped (see
    # the login-node prep in this file's header); the pg19 recall mix exists
    # from the acc4v round.
    [ -d "$NEMO4K" ] || { echo "missing $NEMO4K — run the login-node prep in this file's header"; exit 1; }
    submit_nemo_accum
    submit_gated32
    ;;
vecs)
    # A1-vec16: 256:1 -> 64:1 per-chunk compression.  ACCUM_MAX raised so the
    # eval-side FIFO cap doesn't bite until 16 accumulated chunks.
    MEMORY_SLOTS=0 ACCUM_CCOT=true ACCUM_VECS=16 ACCUM_MAX=256 \
        CARRY_GRAD_CHUNKS=2 RANDOM_SEGMENTS=true EPOCHS=3 DATA_PATH=$RCL4K \
        sbatch --time=48:00:00 pace/rung1_frozen_loop.sbatch
    # A1-vec50: AutoCompressor's working ratio (~20:1; AC used 50 vectors per
    # 1024-2048-token segment).
    MEMORY_SLOTS=0 ACCUM_CCOT=true ACCUM_VECS=50 ACCUM_MAX=800 \
        CARRY_GRAD_CHUNKS=2 RANDOM_SEGMENTS=true EPOCHS=3 DATA_PATH=$RCL4K \
        sbatch --time=48:00:00 pace/rung1_frozen_loop.sbatch
    echo "Submitted: rung1-k0-acc16v-tb2-rs-ep3-rcl | rung1-k0-acc50v-tb2-rs-ep3-rcl"
    ;;
horizon)
    # A1-horizon step 1 — tb4 on the standard cc4 chain (= full-chain BPTT on
    # 4 chunks; isolates the horizon at fixed chunk count/window size).
    MEMORY_SLOTS=0 ACCUM_CCOT=true ACCUM_VECS=4 CARRY_GRAD_CHUNKS=4 \
        RANDOM_SEGMENTS=true EPOCHS=3 DATA_PATH=$RCL4K \
        sbatch --time=48:00:00 pace/rung1_frozen_loop.sbatch
    echo "Submitted: rung1-k0-acc4v-tb4-rs-ep3-rcl"
    # A1-horizon step 2 — tb8 needs an 8-chunk chain: len8192 data at cc8
    # keeps 1024-token windows (two changed variables vs acc4v — chunk count
    # AND horizon — so read it against tb4, not against acc4v directly).
    if [ -d data/pg19_olmo_recall25_len8192 ]; then
        MEMORY_SLOTS=0 ACCUM_CCOT=true ACCUM_VECS=4 CARRY_GRAD_CHUNKS=8 \
            RANDOM_SEGMENTS=true EPOCHS=3 CROSS_CHUNKS=8 MAX_LENGTH=8192 \
            DATA_PATH=data/pg19_olmo_recall25_len8192 \
            sbatch --time=48:00:00 --gres=gpu:H200:1 --constraint=H200 \
                pace/rung1_frozen_loop.sbatch
        echo "Submitted: rung1-k0-acc4v-cc8-len8192-tb8-rs-ep3-rcl (H200)"
    else
        echo "SKIPPED cc8-tb8 arm: data/pg19_olmo_recall25_len8192 missing"
        echo "(run the login-node prep in this file's header, then resubmit)"
    fi
    ;;
nemo)
    [ -d "$NEMO4K" ] || { echo "missing $NEMO4K — run the login-node prep in this file's header"; exit 1; }
    submit_nemo_accum
    ;;
gated)
    [ -d "$NEMO4K" ] || { echo "missing $NEMO4K — run the login-node prep in this file's header"; exit 1; }
    submit_gated32
    ;;
*)
    echo "usage: bash pace/submit_accum_arms.sh round1|vecs|horizon|nemo|gated"; exit 1;;
esac

echo ""
echo "Follow-up once trained:"
echo "  RUN=<name> SLICE_ABLATE=both sbatch pace/eval_carry_ablation.sbatch"
echo "  RUN=<name> sbatch pace/diag_accum_buffer.sbatch"
echo "  RUNS=\"<names>\" bash pace/submit_evals_all.sh   # chunked BABILong readout"
