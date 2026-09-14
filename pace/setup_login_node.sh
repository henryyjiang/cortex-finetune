#!/bin/bash
# One-time PACE Phoenix LOGIN-NODE setup for the cortex finetuning runs.
# Run the steps individually (they are ordered); compute nodes have no internet,
# so everything that downloads must happen here.
#
#   cd ~/retrofitting-recurrence && bash pace/setup_login_node.sh
#
# Assumes the repo was cloned/copied to the home dir. $SCRATCH is PACE scratch.

set -e
REPO=$(cd "$(dirname "$0")/.." && pwd)
cd "$REPO"

# ── 0. conda env (once) ─────────────────────────────────────────────────────
# module load anaconda3
# conda create -n cortex-retro python=3.11 -y
# conda activate cortex-retro
# pip install torch --index-url https://download.pytorch.org/whl/cu121
# pip install -r requirements.txt        # pins transformers==4.51.0
# pip install accelerate safetensors     # from_pretrained init path

# ── 1. HF cache on scratch (home quota is small; ~11 GB PG-19 + 5 GB ckpt) ──
export SCRATCH=${SCRATCH:-$HOME/scratch}   # not set by default on PACE Phoenix
export HF_HOME=$SCRATCH/hf_cache
mkdir -p "$HF_HOME"
echo "HF_HOME=$HF_HOME  (job scripts export the same path + HF_HUB_OFFLINE=1)"

# ── 2. big artifact dirs live on scratch, symlinked into the repo ───────────
mkdir -p "$SCRATCH/cortex_retro/ckpts" "$SCRATCH/cortex_retro/data" \
         "$SCRATCH/cortex_retro/cortex-retro-ft"
[ -L ckpts ]           || ln -s "$SCRATCH/cortex_retro/ckpts" ckpts
[ -L data ]            || ln -s "$SCRATCH/cortex_retro/data" data
[ -L cortex-retro-ft ] || ln -s "$SCRATCH/cortex_retro/cortex-retro-ft" cortex-retro-ft
mkdir -p logs

# ── 3. graft-prepare the base checkpoint (downloads ~5 GB once) ─────────────
# One prepared dir serves every arm — memory flags are passed at train time.
if [ ! -f ckpts/olmo8-cortex/config.json ]; then
    python tools/prepare_cortex_checkpoint.py \
        --src smcleish/Recurrent-OLMo-2-0425-train-recurrence-8 \
        --dst ckpts/olmo8-cortex --variant olmo
fi

# ── 4. tokenize PG-19, one document per row (downloads ~11 GB once) ─────────
if [ ! -d data/pg19_olmo_len4096 ]; then
    python tools/prepare_pg19_dataset.py \
        --tokenizer ckpts/olmo8-cortex \
        --out data/pg19_olmo_len4096 \
        --max_length 4096
fi

# ── 5. eval datasets (BABILong <=128k, LongMemEval, GSM8K, MC) ──────────────
python evals/download_datasets.py

echo "Setup done. Submit runs from the repo root:"
echo "  sbatch pace/rung1_frozen_loop.sbatch                    # K=4"
echo "  MEMORY_SLOTS=0 sbatch pace/rung1_frozen_loop.sbatch     # no-memory control"
echo "  MEMORY_SLOTS=0 CCOT_DIRECT=true sbatch pace/rung1_frozen_loop.sbatch"
echo "  bash pace/submit_retrofit_b2.sh status                  # the live retrofit line"
# Removed 2026-09-14: rung1b_lora_loop / rung2_staged_unfreeze / rung3_l2sp were
# advertised here but the sbatch files had already been deleted, so this block
# printed three commands that could not run.  Their wandb history survives in
# wandb_exports/cortex-retro-ft/ (rung1b-k4-lora16-a32, rung2-k4-unfreeze500,
# rung3-k4-l2sp1e-3); the flags behind the latter two were removed from train.py
# the same day (training-only, so no checkpoint is affected).
