#!/bin/bash
#SBATCH --account=gts-aivanova7-lab
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=64GB
#SBATCH --gres=gpu:A100:1
#SBATCH -t 4:00:00
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
# Cost is ~4 scoring forwards per chunk plus one write chain, i.e. roughly 2x
# the carry ablation.  An A100-40 is enough; this never builds a backward graph.

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
MAX_EXAMPLES=${MAX_EXAMPLES:-150}
N_CHUNKS=${N_CHUNKS:-4}
MODEL_DIR="$OUT_ROOT/$RUN/$CKPT_NAME"

[ -d "$MODEL_DIR" ] || { echo "No such checkpoint: $MODEL_DIR"; exit 1; }

# Re-prepare: the modeling file is COPIED into each prepared dir (unlike
# cortex_graft.py, which is imported live from the repo root), so a dir prepared
# before the 2026-08-04 prefix_pos / prefix_eos_reset changes would still be
# running the old modeling file.
python tools/prepare_eval_checkpoint.py --base "$BASE" \
    --dirs "$MODEL_DIR"

OUT="eval_results/write_capacity_${EVAL_TAG}/${RUN}"
mkdir -p "$OUT"
echo "=== write capacity: $RUN -> $OUT ==="

python evals/eval_write_capacity.py \
    --model_name "$MODEL_DIR" \
    --data "$DATA" \
    --n_chunks "$N_CHUNKS" \
    --max_examples "$MAX_EXAMPLES" \
    --out_dir "$OUT"
