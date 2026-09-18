#!/bin/bash
#SBATCH --account=gts-aivanova7-lab
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=64GB
#SBATCH --gres=gpu:A100:1
#SBATCH -t 8:00:00
#SBATCH -q inferno
#SBATCH -o logs/Report-%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=henryyjiang42@gmail.com

# Write-capacity diagnostic (2026-08-04) — can the carry reconstruct the chunk
# it just encoded?
#
# WHY THIS EXISTS.  The carry ablation reports one number, and a null in it has
# three incompatible readings that it cannot separate: the write encodes
# nothing, the read cannot use what is there, or natural-text continuation has
# nothing worth carrying.  Three closed lines have now returned deltas between
# -0.013 and +0.028 without the project being able to say which of the three
# was happening.  This asks the easiest possible question instead — score a
# chunk against a carry that contains that chunk's OWN summary — so the answer
# is unambiguous:
#
#   d(recon) large  -> the write works; a flat carry delta is a READ failure
#   d(recon) ~ 0    -> the summary vectors do not contain their own chunk, and
#                      everything downstream of the write is moot
#
# d(info) = shuf - self is the same comparison against a carry from a DIFFERENT
# document.  It is there because the `none` condition removes the prefix columns
# entirely while `self` adds 32-96 of them, so `none - self` confounds
# information transfer with a pure extra-register / attention-sink effect.
# Measured at init on a pretrained RoPE LM, a shuffled carry was
# indistinguishable from a real one — this control is not hypothetical.
#
# Runs on BOTH mechanism generations: prefix models get prefix_write=False,
# bolt-on buffers (B1's AccumCCoT, Track A's LSTMBuffer) just take the state
# through m_cross_in.  So the B1 gate is comparable to anything B2 produces.
#
# Prereq (login node, once — shared with the carry ablation):
#   python tools/prepare_pg19_dataset.py --tokenizer ckpts/olmo8-cortex \
#       --out data/pg19_olmo_val_len4096 --max_length 4096 --split validation
#
# The B1 gate (the checkpoint the -0.01282 carry delta came from):
#   OUT_ROOT=cortex-retrofit CKPT_NAME=model_only_chkpt_22500 \
#       BASE=ckpts/olmo-retrofit-cortex RUN=retro-b1-acc4v-tb2-mr8 \
#       EVAL_TAG=b1-22500 sbatch pace/eval_write_capacity.sbatch
#
# A Track-A arm (bolt-on, for the cross-generation comparison):
#   RUN=rung1-k0-acc4v-tb2-rs-ep3-rcl sbatch pace/eval_write_capacity.sbatch
#
# A B2 gate once the restarted run has one:
#   OUT_ROOT=cortex-retrofit CKPT_NAME=model_only_chkpt_91552 \
#       BASE=ckpts/olmo-retrofit-cortex RUN=retro-b2-heal \
#       EVAL_TAG=b2-heal sbatch pace/eval_write_capacity.sbatch
#
# The P1 parent, i.e. the SLICED w16 branch (overlay form -- no config.json in
# that dir, so BASE loads and the geometry is forced; see the branch below):
#   OUT_ROOT=cortex-retrofit RUN=retro-b2-heal CKPT_NAME=checkpoint_91552_w16 \
#       BASE=ckpts/olmo-retrofit-cortex EVAL_TAG=pass1-parent \
#       N_CHUNKS=8 MAX_EXAMPLES=50 T_EVAL=8 \
#       sbatch pace/eval_write_capacity.sbatch
#
# COST.  Five forwards per chunk (four conditions + one write-chain pass) vs the
# carry ablation's two, i.e. 2.5x per sample.  The first attempt inherited the
# ablation's 4h walltime at n=150 and was killed; 8h and n=60 now.  n=60 is
# ample here — the reconstruction delta is far larger than the ~0.001 nats the
# ablation needs 150 samples to resolve.  The script prints an ETA after the
# first sample, so check the log early rather than at the walltime.
#   MAX_EXAMPLES=20 T_EVAL=8   # a cheap timing probe before committing 8h
# T_EVAL overrides the recurrence; unset means the checkpoint's own
# config.mean_recurrence, which prepare_eval_checkpoint.py prints in the log.
# Verify it matches the arm (B1 is mr8) — the UNTRAINED base config carries 32.
# An A100-40 is enough; this never builds a backward graph.

cd $SLURM_SUBMIT_DIR

module load anaconda3
module load cuda/12.1.1
conda activate cortex-retro

export SCRATCH=${SCRATCH:-$HOME/scratch}
export HF_HOME=$SCRATCH/hf_cache
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

[ -n "$RUN" ] || { echo "Set RUN=<run_name>"; exit 1; }
EVAL_TAG=${EVAL_TAG:-$(date +%Y%m%d)}
DATA=${DATA:-data/pg19_olmo_val_len4096}
OUT_ROOT=${OUT_ROOT:-cortex-retro-ft}
CKPT_NAME=${CKPT_NAME:-final_checkpoint}
BASE=${BASE:-ckpts/olmo8-cortex}
MAX_EXAMPLES=${MAX_EXAMPLES:-60}
N_CHUNKS=${N_CHUNKS:-4}
T_EVAL=${T_EVAL:-}
MODEL_DIR="$OUT_ROOT/$RUN/$CKPT_NAME"

[ -d "$MODEL_DIR" ] || { echo "No such checkpoint: $MODEL_DIR"; exit 1; }

# TWO KINDS OF CHECKPOINT DIR, and they load by DIFFERENT paths.
#
#   save_model_only dirs (final_checkpoint, model_only_chkpt_*) are
#   save_pretrained output: they carry config.json, so they load as
#   --model_name once prepare_eval_checkpoint.py has copied the modeling file
#   in.  This is what the b1-22500 / trackA rows above ran.
#
#   train.py checkpoint_<step> dirs (and the slice_summary_emb _w16 branch
#   dirs cut from them) hold chkpt.pt and NO config.json.  They are OVERLAYS:
#   --model_name is the graft-prepared BASE and the weights go in --checkpoint.
#   The base config carries no cortex flags at all, so the geometry has to be
#   FORCED with --set or the graft builds no buffer.
#
# Job 13307388/13307394 died 12 s in because this script knew only the first
# path and submit_pass1.sh handed it the second: prepare_eval_checkpoint.py
# printed "[skip] ...: no config.json" and RETURNED 0, so the run continued to
# an AutoConfig "Unrecognized model" 200 lines long.  Hence the branch, and
# hence the skip is now fatal here rather than a note in the log.
if [ -f "$MODEL_DIR/config.json" ]; then
    # Re-prepare: the modeling file is COPIED into each prepared dir (unlike
    # cortex_graft.py, which is imported live from the repo root), so a dir
    # prepared before the 2026-08-04 prefix_pos / prefix_eos_reset changes
    # would still be running the old modeling file.
    python tools/prepare_eval_checkpoint.py --base "$BASE" \
        --dirs "$MODEL_DIR" || exit 1
    MODEL_ARGS=(--model_name "$MODEL_DIR")
    echo "    load: prepared HF dir $MODEL_DIR"
else
    [ -f "$MODEL_DIR/chkpt.pt" ] || {
        echo "ERROR: $MODEL_DIR has neither config.json nor chkpt.pt — it is"
        echo "       not a checkpoint dir this eval can load."; exit 1; }
    [ -f "$BASE/config.json" ] || {
        echo "ERROR: BASE=$BASE is not a graft-prepared dir (no config.json)."
        echo "       An overlay checkpoint needs one to load against."; exit 1; }

    # Geometry, forced.  Mirrors pace/eval_deciding.sbatch's SETS block; the
    # defaults are the P1 arm geometry (W=16), NOT the W=32 parent's, because
    # the parent checkpoint reachable from here is the sliced _w16 branch.
    # accum_vecs sizes summary_emb and a wrong value is a load-time size
    # mismatch, not a silent one — but accum_max is the FIFO cap and IS silent,
    # so it is set to cross_chunks * accum_vecs the way training sized it.
    PREFIX_MODE=${PREFIX_MODE:-accum}
    ACCUM_VECS=${ACCUM_VECS:-16}
    ACCUM_MAX=${ACCUM_MAX:-$((N_CHUNKS * ACCUM_VECS))}
    SETS="--set use_memory=true --set memory_slots=0"
    SETS="$SETS --set prefix_memory=$PREFIX_MODE"
    SETS="$SETS --set accum_vecs=$ACCUM_VECS"
    if [ "$PREFIX_MODE" = "accum" ]; then
        SETS="$SETS --set accum_max=$ACCUM_MAX"
    else
        SETS="$SETS --set gate_slots=${GATE_SLOTS:-64}"
        SETS="$SETS --set gate_route=${GATE_ROUTE:-ring}"
        SETS="$SETS --set gate_norm=${GATE_NORM:-tanh}"
        SETS="$SETS --set gate_init=${GATE_INIT:-zero}"
        SETS="$SETS --set gate_fill=${GATE_FILL:-grow}"
    fi
    MODEL_ARGS=(--model_name "$BASE" --checkpoint "$MODEL_DIR" $SETS)
    echo "    load: BASE $BASE + overlay $MODEL_DIR"
    echo "    sets: $SETS"
    echo "    The [cortex] OVERRIDE lines below are the proof it arrived."
fi

OUT="eval_results/write_capacity_${EVAL_TAG}/${RUN}"
mkdir -p "$OUT"
echo "=== write capacity: $RUN -> $OUT ==="

python evals/eval_write_capacity.py \
    "${MODEL_ARGS[@]}" \
    --data "$DATA" \
    --n_chunks "$N_CHUNKS" \
    --max_examples "$MAX_EXAMPLES" \
    ${T_EVAL:+--T "$T_EVAL"} \
    --out_dir "$OUT"
