#!/bin/bash
#SBATCH --account=gts-aivanova7-lab
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=64GB
#SBATCH --gres=gpu:H200:1
#SBATCH --constraint=H200
#SBATCH -t 48:00:00
#SBATCH -q inferno
#SBATCH -o logs/Report-%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=henryyjiang42@gmail.com

# B2 — retrofit conversion with AutoCompressor-faithful PREFIX memory, run in
# TWO PHASES with a SHARED healing phase.  Supersedes b1_retrofit.sbatch.
#
#   phase 0 (heal)  : FineWeb-Edu, prefix accum, ~1.5B tokens   -- run ONCE
#          |
#          +--> arm A: prefix accum   (Nemotron math, ~3.5B)
#          +--> arm B: prefix gated   (Nemotron math, ~3.5B)
#
# Use pace/submit_retrofit_b2.sh; it drives the resume chain and the branch.
#
# WHY THIS EXISTS — what B1 got wrong
# -----------------------------------
# B1 (accum bolt-on, co-trained from the untrained conversion) measured carry
# = -0.0128 nats (z = -9.3): carrying the buffer was WORSE than zeroing it.
# Three causes were identified afterwards, and this script fixes all three.
#
# 1. THE MECHANISM WAS NOT AUTOCOMPRESSOR.  Reading
#    AutoCompressors-main/auto_compressor.py against ours turned up four
#    divergences (documented at length in cortex_memory/buffers.py):
#    the read was a bolt-on cross-attention with a zero-init out_proj inside
#    the loop (AC PREPENDS its vectors to the input embeddings and lets the
#    model's OWN attention read them); the write was a bolt-on extraction
#    attention (AC APPENDS learned summary tokens and takes their output hidden
#    states); the vectors were tanh(LN)-bounded (AC does not bound them at
#    all); and the summary embeddings were noise (AC seeds them from the EOS
#    embedding).  PrefixAccumBuffer / PrefixGatedBuffer fix all four.  The
#    practical consequence: the carry is LIVE from step 0 instead of being a
#    channel the model has to discover from an exactly-zero injection.
#
# 2. NO HEALING PHASE.  The retrofit paper's own finding: "introducing a
#    'healing' period with minimal distribution shift helps recover basic
#    language modeling performance before switching to task-specific data" —
#    two-phase beats single-phase by 5% Arc-Challenge for the depth-recurrent
#    model (the non-recurrent control shows no significant difference).  B1
#    went straight to Nemotron math from step 0, i.e. MAXIMAL shift exactly
#    where the recipe wants minimal.  Phase 0 heals on general web text.
#
# 3. THE BUDGET.  B1 stopped at 24,414 steps (400M tokens) because the data
#    pack ran dry, not because it finished.  B2 is 5B tokens per arm, and the
#    packs are built past the budget so the loader cannot exhaust (see PREP).
#
# WHY PHASE 0 WAS RESTARTED (2026-08-04)
# --------------------------------------
# The first phase-0 attempt (wandb 4vop2ym8) was cancelled at ~9h / ~24k steps
# after a code audit found two defects in the prefix path.  Both are fixed and
# flag-gated; both were silent, i.e. the loss curve looked perfectly healthy.
#
# A. THE READ WAS OFF FOR MOST CHUNKS.  CortexMemory._carried_state zeroed the
#    WHOLE incoming carry for any lane whose chunk contained an EOS, and
#    write_reset is has_eos for the entire chunk — so one document boundary
#    anywhere killed the carry for every position, including the tokens before
#    it that continue the previous document.  tools/prepare_packed_dataset.py
#    joins documents with EOS separators and this run sets eos_from_tokens=true,
#    so on FineWeb-Edu (~1.1k-token docs, 1024-token chunks) roughly 60% of
#    chunks lost their carry; at 500-token docs, 87%.  Worse, zeroing does not
#    REMOVE the columns — a zero key scores 0, not -inf — so 64-96 dead columns
#    kept absorbing ~3-5% of the softmax mass at every layer.  The per-position
#    mask with the right semantics (cross_read_mask) was computed and then
#    discarded, because only read_into consumes it and that is a no-op in prefix
#    mode; the bolt-on buffers DID use it, so this was a regression introduced
#    by the prefix rewrite.  Now --cortex.prefix_eos_reset false (the default):
#    the carry is read by the whole chunk.  The justification is that the
#    backbone already does exactly this — the modeling file runs with
#    prepared_attn_mask=None and is_causal=True, so base attention is unmasked
#    across document boundaries inside a chunk.  AC and RMT also carry across
#    packed boundaries.  The principled middle option (a per-position block mask
#    so post-boundary tokens cannot attend to the carry) needs the flex_attention
#    path the modeling file leaves disabled, and was not worth the restart delay.
#
# B. THE WRITE READ THE CHUNK AT NEGATIVE ROPE OFFSETS.  prefix_pack put the
#    trailing summary slots at position 0, on the theory that this was our RoPE
#    analog of AutoCompressor's pad-position trick.  It is not: that trick is in
#    OPTLearnedPositionalEmbeddingWithPadding, OPT positions are ADDITIVE, so
#    "no position" means adding a zero vector — and AC's own RoPE model
#    (LlamaAutoCompressorModel) overrides nothing and uses contiguous positions.
#    A slot at position 0 sitting at the END of the sequence queries every real
#    token at a negative relative offset (-1..-1024), which a causal LM never
#    sees (pos_q >= pos_k always).  Measured on a pretrained RoPE LM, that layout
#    also collapses the slots: centred slot-slot cosine 0.94 and effective rank
#    2.1 of 32, vs 0.22 / 4.2 with the tail layout — accum_vecs=32 was buying
#    about two independent directions.  Now --cortex.prefix_pos tail: slots at
#    S+1..S+n_vec.  The CARRY stays at position 0; it sits at the front, so real
#    tokens already query it at positive offsets, and keeping real tokens at 1..S
#    is what lets a cached decode step continue the prefill's numbering.
#
# Set --cortex.prefix_eos_reset true --cortex.prefix_pos zero to reproduce the
# cancelled attempt exactly.  Neither flag existed before 2026-08-04, so a dir
# whose config.json predates it will pick up the NEW defaults — re-prepare eval
# checkpoints rather than assuming they carry their training-time arrangement.
#
# Gate this restart on pace/eval_write_capacity.sbatch rather than on the loss
# curve: it asks whether the carry can reconstruct the chunk it just encoded,
# which is the one question the carry ablation cannot answer.
#
# NOT a from-scratch run, and NOT a finetune.  ckpts/olmo-retrofit-cortex is
# smcleish/Recurrent-OLMo-2-0425-untrained: PRETRAINED OLMo-2-0425-1B weights
# in an untrained (4,6,4)-with-loop-x8 ARRANGEMENT (convert_olmo.py:92-133).
# So this IS the retrofit conversion, with memory co-trained through it.
# From-scratch was considered and rejected: the paper's Figure 9 / Appendix C.1
# puts the random-init vs pretrained-init crossover at ~950B tokens, called an
# underestimate — at any affordable budget from-scratch gives a strictly worse
# backbone, and "backbone too weak" is the leading explanation for B1's null.
# Do NOT switch to Recurrent-OLMo-2-0425-train-recurrence-8 either: that one is
# already retrofitted, which turns the study into finetuning (it is Track A's
# base, ckpts/olmo8-cortex).
#
# ARMS.  accum (append, unbounded width) vs gated (LM2 gate, fixed width) over
# an IDENTICAL write/read path — the only difference is `merge`.  Accum's
# parameters are a strict SUBSET of gated's (summary_emb alone vs summary_emb +
# gate), which is what makes one shared healing phase legitimate for both: it
# trains everything both arms need except the gate.  There is no separate `off`
# control this round; the carry ablation (carried vs zeroed on the SAME
# checkpoint) is the within-arm control and it costs no GPU-hours of training.
#
# ARM SEQUENCING (2026-08-02, Henry's call on compute): run ACCUM FIRST, ALONE.
# gated branches off the same healing checkpoint later if accum earns it.
# Reasoning, recorded so it does not get re-litigated:
#   1. Dropping to one arm REMOVES the between-arm comparison, so all the
#      interpretive weight lands on the within-arm diagnostics — and only accum
#      has any.  The slice ablation and the buffer diagnostic both need
#      write-once rows and refuse a gated state (a gated merge makes the k-th
#      chunk's contribution unrecoverable).  A lone gated arm returns ONE
#      scalar; a lone accum arm returns that scalar plus whether accumulation
#      or recency produced it and whether the writes are distinct.  The
#      constraint argues FOR accum, not against it.
#   2. PrefixAccumBuffer reproduces AutoCompressor's PUBLISHED working config
#      (accumulate_summary=true is their default).  PrefixGatedBuffer is our
#      own variant — LM2's gate on their plumbing — which nobody has shown to
#      work.  When the premise of the round is "we were unfaithful to a working
#      design and it failed", run the faithful version first.
#   3. Every mechanism result on the books (0.024 headline, recency dominance,
#      redundant-gist writes) is accum-lineage on these same instruments, so an
#      accum number is directly comparable to all of it.
# Against: gated had the better Track-A delta (0.0283 +- 0.0010 vs 0.0237 +-
# 0.0012, ~3 sigma) and was the only dial that moved that round.  Outweighed —
# that is +19% relative on a metric needing ~+300%, measured under frozen-loop
# bolt-on on PG-19 recall, in a round that also saw a pre-registered horizon
# prediction and the write-diversity theory both fail.
# The decision is REVERSIBLE AT NO EXTRA HEALING COST: the healing checkpoint
# stays on disk, so `submit_retrofit_b2.sh branch gated` at any later date is
# still a properly controlled comparison against the same ancestor.
#
# FUTURE ARM (Henry, 2026-08-02): the ORIGINAL LM2 buffer deserves its own arm
# alongside these two.  `PrefixGatedBuffer` keeps only LM2's GATE — its
# cross-attention write and read are both gone — so it is no longer an LM2
# reproduction and the two are now entirely different mechanisms.  The faithful
# implementation is still in the tree (`LSTMBuffer`, selected by memory_slots >
# 0) and is NOT wired into this sbatch: it writes/reads through the loop rather
# than the token stream, so it needs the pre-prefix graft path, which the
# 2026-08-02 rewrite left intact.  Wire it as a third PHASE=arm mode when the
# budget allows; it cannot share this healing phase (its parameters are not a
# subset of accum's), so it needs its own healing run.
#
# THE RAMP IS THE REASON B1's GATE WAS UNREADABLE — and it is fixed here.
# The mean-recurrence ramp is a FRACTION of max_steps (train.py:1181-1191,
# max_training_steps = cfg.max_steps).  B1 used McLeish's warmup=0.75, so at
# 50,000 steps full recurrence arrived at step 37,500 and every intermediate
# gate sat at PARTIAL recurrence — not comparable to Track A's fixed
# recurrence 8, which is why the 22.5k read could not be interpreted.  Here
# warmup=0.25 of 305,176 puts full recurrence 8 at step 76,294 (~1.25B
# tokens), INSIDE the healing phase.  Both arms therefore train at constant
# recurrence 8 for their entire lives, and every gate is comparable — to each
# other and to Track A.
#
# MAX_STEPS=305176 IS FIXED FOREVER (5B / 16,384 tok-step).  It is the horizon
# for the LR cosine AND the recurrence ramp, so varying it across links of the
# chain re-runs cooldown per segment — the 2026-06-24 sawtooth artifact, baked
# into the weights.  To end the healing phase early use STOP_AT_STEP, which
# stops the phase WITHOUT moving the horizon.
#
# H200, not A100-80GB.  Both B0 arms OOMed at 78.4GiB at recurrence 8 —
# including the arm that carried nothing, so it is the 4 x 1024-chunk backward
# graph, not the memory.  B2 lives at full recurrence for ~75% of its life.
# Prefix memory adds ~12% more sequence on top (32 carry + 32 summary columns
# per 1024-token chunk at cross_chunks=4).
#
# CHUNK GEOMETRY: THE ARMS RUN AT cross_chunks=8, HEALING STAYS AT 4
# (decided 2026-08-04 on the ceil2 probe round; digits in the numbers table)
#
# The oracle ceiling collapse past a ~1280-token window was a WINDOW-LENGTH
# ARTIFACT, not a property of the model or the data.  Measured on the same base,
# same 50 PG-19 rows, only the chunk size differing:
#
#     k real tokens |  1024-tok chunks  |  512-tok chunks
#         128       |      +0.0871      |     +0.1613
#         256       |    +0.1036 (peak) |     +0.1958
#         512       |      +0.0003      |   +0.2277 (still rising)
#        1024       |      -0.977       |        —
#
# At cross_chunks=4 the carry was being asked to summarise a 1024-token chunk
# into a 2048-token window where the base scores -0.977 — a span the model
# cannot use even VERBATIM.  At cross_chunks=8 the window is 1024, inside the
# usable range, and nothing collapses.  Track A read at 8 chunks (trained at 4,
# so the mismatch penalty is already paid) delivers carry +0.0433 vs +0.0237 —
# 1.83x, and 2.1x more total nats per document.  Nemotron's own ceiling dies
# even faster with distance than PG-19's (-0.048 already at k512), so the arm
# corpus needs this more than the healing corpus does.
#
# Honest cost, do not hide it on a slide: shorter chunks mean less local
# context, so the no-carry baseline RISES 0.058 nats while the carry returns
# 0.043.  Finer chunking is a better INSTRUMENT for the mechanism question, not
# a free win in raw NLL.
#
# Why healing stays at 4: it is mid-flight and restarting costs ~40% of a
# 1.5B-token phase to buy a geometry match that the Track-A measurement shows
# is not needed — that number was itself obtained across exactly this mismatch.
#
# THREE FLAGS MOVE TOGETHER.  cross_chunks=8 alone will NOT start:
#   accum_max 128 -> 256      train.py:306 asserts cross_chunks x accum_vecs <=
#                             accum_max and B2 sits exactly on the boundary
#                             (4x32=128).  It is a plain int in the buffer's
#                             merge(), no parameter tensor, so raising it is
#                             resume-safe.  NEVER satisfy the assert by lowering
#                             accum_vecs — summary_emb is nn.Parameter(n_vec, D)
#                             and changing it re-inits the trained write path.
#   carry_grad_chunks 2 -> 4  keeps the trained fraction of the chain at 50%
#                             (2 of 4 -> 4 of 8) and is MEMORY-NEUTRAL:
#                             4 x 512 = 2 x 1024 = 2048 tokens of backward graph.
# Position budget goes DOWN (chunk 8 packs 512 + 224 carry + 32 = 768 vs 1152),
# so this cannot introduce a new OOM or a RoPE range problem.
# Gate it with:  python tools/smoke_prefix_real.py --cross_chunks 8 --accum_max 256
# Every eval on these arms then needs N_CHUNKS=8, or it reads an 8-trained arm
# at the wrong geometry.
#
# TOKEN MATH (16,384 tok/step: 4 rows x 4096, each row = 4 x 1024 chunks)
#     76,294 steps = 1.25B   full recurrence 8 reached (inside healing)
#     91,552 steps = 1.50B   healing ends, arms branch          <- STOP_AT_STEP
#    305,176 steps = 5.00B   arms finish
# At ~0.8-1.0B tokens per 48h H200 job (B1 ran 8,788 tok/s at recurrence 4;
# recurrence 8 is ~1.75x the layer-applications, 32 -> 56, plus the ~12%
# prefix) that is ~2 jobs for healing and ~4 per arm: ~10 jobs, ~8.5B tokens.
#
# PREP (login node, once).  Build BOTH packs PAST the budget so the loader
# cannot run dry mid-phase — B1's "finished" at 24,414 steps was a loader
# exhaustion, not a completion (train.py:1285 iterates the parquet loader once;
# the only break is optimizer_step >= max_steps, so a dry loader exits clean and
# wandb says "finished" at half the run).  Both packs below are sized ~30% over.
#
#   python tools/prepare_cortex_checkpoint.py \
#       --src smcleish/Recurrent-OLMo-2-0425-untrained \
#       --dst ckpts/olmo-retrofit-cortex --variant olmo
#
#   # healing corpus — general web text, minimal distribution shift, >= 1.5B.
#   # prepare_packed_dataset.py packs from LOCAL shards (the login node's memory
#   # cap kills the streaming reader), so list -> download -> pack:
#   python tools/prepare_packed_dataset.py --list_files \
#       --dataset HuggingFaceFW/fineweb-edu --subset 10BT
#   huggingface-cli download HuggingFaceFW/fineweb-edu --repo-type dataset \
#       --include "sample/10BT/*.parquet"
#   python tools/prepare_packed_dataset.py --tokenizer ckpts/olmo-retrofit-cortex \
#       --dataset HuggingFaceFW/fineweb-edu --subset 10BT \
#       --out data/fineweb_edu_olmo_len4096 --max_length 4096 \
#       --max_tokens 2_000_000_000
#
#   # arm corpus — Nemotron math, >= 3.5B.  ~1B tokens/shard, so download 4-5;
#   # the existing 400M pack (data/nemotron_math_olmo_len4096) is too small and
#   # is NOT reused, because a mid-run wrap would differ between the arms.
#   python tools/prepare_packed_dataset.py --list_files \
#       --dataset nvidia/Nemotron-CC-Math-v1 --subset 4plus
#   huggingface-cli download nvidia/Nemotron-CC-Math-v1 --repo-type dataset \
#       --include "<the first 5 shard paths from the listing>"
#   python tools/prepare_packed_dataset.py --tokenizer ckpts/olmo-retrofit-cortex \
#       --out data/nemotron_math_olmo_len4096_4b --max_length 4096 \
#       --max_tokens 4_000_000_000
#
# Set HF_HOME=$SCRATCH/hf_cache for every one of those commands.  The default
# ~/.cache/huggingface is what blew the home quota on 2026-07-17.
#
# NO STABILIZERS (design principle 0.1.3).  B0 showed memory co-training does
# not perturb the conversion (base vs accum within ~0.007 nats over the whole
# 10.4 -> 1.06 descent, matched grad norms).  Watch train/total_norm: B1's
# accum arm had heavy-tail gradient spikes the control did not (gnorm max 123.7
# vs 8.7), and B2 runs at HIGHER recurrence than B1 ever reached.
#
# DISK.  save_checkpoint never rotates.  save_interval writes a weights-only HF
# dir; every 2 * save_interval writes a full chkpt.pt (model + Muon/Adam +
# scheduler + dataloader), ~15-20GB each.  RESUMABLE checkpoints are therefore
# every 2 * SAVE_INTERVAL, not every SAVE_INTERVAL — B1's crash at 14,579 fell
# back to checkpoint_10000 and burned 4,579 steps.  At SAVE_INTERVAL=2500 a
# crash costs up to 5,000 steps (~2.5h); a 305k-step run leaves 122 weight dirs
# + 61 full checkpoints, so prune old checkpoint_* as the chain advances.

cd $SLURM_SUBMIT_DIR

module load anaconda3
module load cuda/12.1.1
conda activate cortex-retro

export SCRATCH=${SCRATCH:-$HOME/scratch}
export HF_HOME=$SCRATCH/hf_cache
export HF_HUB_OFFLINE=1
export WANDB_DIR=$SCRATCH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# TCPStore port for the trivial 1-process group Muon needs (train.py:780).
# train.py derives 20000 + SLURM_JOB_ID % 40000 and installs it with
# os.environ.setdefault, so anything exported here wins.  That derivation was
# itself the fix for shared-node collisions and it STILL collided on gpu-a100
# (job 11702360, 2026-08-05: EADDRINUSE on 42360, dead in 29s).  No B2 job has
# hit it — gpu-h200 is less crowded — but this is a multi-day job where losing
# the queue slot to a 29-second port clash is the expensive outcome, so probe a
# free port here too.  setdefault is why this needs no train.py change.
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=$(python -c "import socket;s=socket.socket();s.bind(('',0));print(s.getsockname()[1]);s.close()")
echo "MASTER_PORT=$MASTER_PORT"

# --- out_path MUST live on scratch -----------------------------------------
# train.py uses out_path as BOTH the wandb project name AND the write dir
# (wandb.init(project=cfg.out_path, dir=cfg.out_path) — WANDB_DIR is ignored),
# and a relative name resolves into $HOME.  That is what killed the 2026-07-17
# round: an update-500 checkpoint blew the home quota and took down all three
# B0 arms plus, an hour later, all three Track-A arms (whose Slurm logs also
# live in home).  Keep the relative name (preserves the wandb project) and
# symlink it to scratch.
OUT_DIR=cortex-retrofit
if [ ! -L "$OUT_DIR" ]; then
    if [ -e "$OUT_DIR" ]; then
        echo "ERROR: $OUT_DIR exists in \$HOME but is not a symlink to scratch."
        echo "  It is what killed the 2026-07-17 round.  Move or delete it"
        echo "  (rm -rf $OUT_DIR) and resubmit; this script will re-create it"
        echo "  as a symlink to \$SCRATCH/cortex-retrofit."
        exit 1
    fi
    mkdir -p "$SCRATCH/cortex-retrofit"
    ln -s "$SCRATCH/cortex-retrofit" "$OUT_DIR"
fi
# ---------------------------------------------------------------------------

PHASE=${PHASE:-heal}                       # heal | arm
MEMORY_MODE=${MEMORY_MODE:-accum}          # accum | gated   (arms; heal is always accum)
MAX_STEPS=${MAX_STEPS:-305176}             # 5B / 16,384 — FIXED across every link
STOP_AT_STEP=${STOP_AT_STEP:-0}            # end this phase early, horizon unchanged
MAX_MEAN_REC=${MAX_MEAN_REC:-8}
RAMP_WARMUP=${RAMP_WARMUP:-0.25}           # full recurrence at step 76,294
MODEL=${MODEL:-ckpts/olmo-retrofit-cortex}
MAX_LENGTH=${MAX_LENGTH:-4096}
ACCUM_VECS=${ACCUM_VECS:-32}               # a PARAMETER SHAPE (summary_emb) — never vary mid-chain
# CROSS_CHUNKS / ACCUM_MAX / CARRY_GRAD_CHUNKS are defaulted PER PHASE in the
# case block below: healing keeps the 4/128/2 geometry it started on, the arms
# run at 8/256/4.  See the CHUNK GEOMETRY block in the header.  Setting any of
# them in the environment still wins.
# 2026-08-04 restart defaults; set true/zero to reproduce the cancelled 4vop2ym8.
PREFIX_EOS_RESET=${PREFIX_EOS_RESET:-false}  # carry survives a document boundary
PREFIX_POS=${PREFIX_POS:-tail}               # summary slots at S+1..S+n_vec
SAVE_INTERVAL=${SAVE_INTERVAL:-2500}
RESUME_PATH=${RESUME_PATH:-}
BRANCH_PATH=${BRANCH_PATH:-}

HEAL_DATA=${HEAL_DATA:-data/fineweb_edu_olmo_len4096}
ARM_DATA=${ARM_DATA:-data/nemotron_math_olmo_len4096_4b}

case "$PHASE" in
heal)
    # The healing phase is ALWAYS accum: its parameters (summary_emb) are the
    # subset both arms share, so branching into gated only adds the gate.
    # Running it as gated instead would train a gate that arm A must then
    # carry as dead weight, and the arms would no longer share an ancestor.
    PREFIX_MODE=accum
    # The geometry this phase STARTED on.  It is mid-flight; changing it here
    # would make the resume chain internally inconsistent for no measured gain.
    CROSS_CHUNKS=${CROSS_CHUNKS:-4}
    ACCUM_MAX=${ACCUM_MAX:-128}                # 4 x 32 — the assert is exact
    CARRY_GRAD_CHUNKS=${CARRY_GRAD_CHUNKS:-2}  # AC's stop-grad after 2 compressions
    DATA_PATH=${DATA_PATH:-$HEAL_DATA}
    RUN_NAME=${RUN_NAME:-retro-b2-heal};;
arm)
    case "$MEMORY_MODE" in
    accum) PREFIX_MODE=accum; ARM=acc${ACCUM_VECS};;
    gated) PREFIX_MODE=gated; ARM=gate${ACCUM_VECS};;
    *) echo "MEMORY_MODE must be accum|gated"; exit 1;;
    esac
    # 512-token chunks: the ceiling is 2.2x larger there and does not collapse.
    CROSS_CHUNKS=${CROSS_CHUNKS:-8}
    ACCUM_MAX=${ACCUM_MAX:-256}                # 8 x 32 — moves WITH cross_chunks
    CARRY_GRAD_CHUNKS=${CARRY_GRAD_CHUNKS:-4}  # 4 of 8 = the same 50% as 2 of 4
    DATA_PATH=${DATA_PATH:-$ARM_DATA}
    RUN_NAME=${RUN_NAME:-retro-b2-${ARM}-cc${CROSS_CHUNKS}-mr${MAX_MEAN_REC}};;
*)
    echo "PHASE must be heal|arm"; exit 1;;
esac

# Pre-flight the assert train.py:306 makes, so a bad combination dies HERE
# instead of after the scheduler has handed over an H200.
if [ $(( CROSS_CHUNKS * ACCUM_VECS )) -gt "$ACCUM_MAX" ]; then
    echo "ERROR: cross_chunks x accum_vecs ($CROSS_CHUNKS x $ACCUM_VECS =" \
         "$(( CROSS_CHUNKS * ACCUM_VECS ))) exceeds accum_max ($ACCUM_MAX)."
    echo "  train.py asserts this: the FIFO must hold the whole chain or early"
    echo "  chunks are silently trimmed and the stop-grad slice bookkeeping breaks."
    echo "  Raise ACCUM_MAX.  Do NOT lower ACCUM_VECS — it is a parameter shape."
    exit 1
fi
if [ $(( MAX_LENGTH % CROSS_CHUNKS )) -ne 0 ]; then
    echo "ERROR: MAX_LENGTH ($MAX_LENGTH) is not divisible by CROSS_CHUNKS" \
         "($CROSS_CHUNKS) — chunks would be ragged."; exit 1
fi

# Resume continues THIS run (optimizer + scheduler + dataloader + RNG).
# Branch starts an ARM from the healing phase: weights load non-strictly (the
# gated arm adds the gate), the optimizer is NOT restored (its parameter set
# changed) and the dataloader is NOT restored (the corpus changed), while the
# LR schedule, the recurrence ramp and the step counter continue.  Both arms
# branch identically, so the discarded optimizer state cannot favour one.
CHAIN_ARGS=""
if [ -n "$RESUME_PATH" ] && [ -n "$BRANCH_PATH" ]; then
    echo "ERROR: set RESUME_PATH or BRANCH_PATH, not both."; exit 1
fi
if [ -n "$RESUME_PATH" ]; then
    if [ ! -f "$RESUME_PATH/chkpt.pt" ]; then
        echo "ERROR: no chkpt.pt under RESUME_PATH=$RESUME_PATH"; exit 1
    fi
    CHAIN_ARGS="--resume_path $RESUME_PATH"
    echo "Resuming from $RESUME_PATH"
fi
if [ -n "$BRANCH_PATH" ]; then
    if [ ! -f "$BRANCH_PATH/chkpt.pt" ]; then
        echo "ERROR: no chkpt.pt under BRANCH_PATH=$BRANCH_PATH"; exit 1
    fi
    if [ "$PHASE" != "arm" ]; then
        echo "ERROR: BRANCH_PATH only makes sense for PHASE=arm."; exit 1
    fi
    CHAIN_ARGS="--branch_path $BRANCH_PATH"
    echo "Branching from $BRANCH_PATH"
fi

echo "=== B2 $RUN_NAME | phase=$PHASE | prefix=$PREFIX_MODE | data=$DATA_PATH ==="
echo "    max_steps=$MAX_STEPS (horizon)  stop_at_step=$STOP_AT_STEP"
echo "    full recurrence $MAX_MEAN_REC at step $(python -c "import math;print(math.ceil($RAMP_WARMUP*$MAX_STEPS))")"
echo "    tokens at horizon: $(( MAX_STEPS * 4 * MAX_LENGTH ))"
echo "    prefix_pos=$PREFIX_POS  prefix_eos_reset=$PREFIX_EOS_RESET (2026-08-04 fixes)"
echo "    geometry: cross_chunks=$CROSS_CHUNKS ($(( MAX_LENGTH / CROSS_CHUNKS ))-tok chunks)" \
     " accum_vecs=$ACCUM_VECS accum_max=$ACCUM_MAX carry_grad_chunks=$CARRY_GRAD_CHUNKS"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python train.py \
    --run_name $RUN_NAME \
    --out_path cortex-retrofit \
    --model_name $MODEL \
    --preprocessed_data_path $DATA_PATH \
    --max_length $MAX_LENGTH \
    --epochs 1 \
    --max_steps $MAX_STEPS \
    --stop_at_step $STOP_AT_STEP \
    --micro_batch_size 1 \
    --batch_size 4 \
    --optim_config.lr 5e-5 \
    --scheduler_args.cooldown 0.6 \
    --scheduler_args.warmup 0.0025 \
    --muon.use_muon true \
    --muon.lr 0.001 \
    --no_amp false \
    --max_grad_norm 1.0 \
    --compile false \
    --save_interval $SAVE_INTERVAL \
    --save_n_mins_before_timeout 20 \
    --seed 74 \
    --mean_recurrence_schedule.turn_on true \
    --mean_recurrence_schedule.warmup $RAMP_WARMUP \
    --mean_recurrence_schedule.max_mean_rec $MAX_MEAN_REC \
    --mean_recurrence_schedule.warmup_type "1-sqrt" \
    --cortex.use_memory true \
    --cortex.memory_lr 5e-4 \
    --cortex.memory_slots 0 \
    --cortex.prefix_memory $PREFIX_MODE \
    --cortex.accum_vecs $ACCUM_VECS \
    --cortex.accum_max $ACCUM_MAX \
    --cortex.carry_grad_chunks $CARRY_GRAD_CHUNKS \
    --cortex.cross_chunks $CROSS_CHUNKS \
    --cortex.eos_from_tokens true \
    --cortex.prefix_eos_reset $PREFIX_EOS_RESET \
    --cortex.prefix_pos $PREFIX_POS \
    --cortex.freeze_loop false \
    $CHAIN_ARGS
