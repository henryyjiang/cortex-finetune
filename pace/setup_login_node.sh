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
# python evals/download_datasets.py

echo "Setup done. Submit runs from the repo root:"
echo "  sbatch pace/rung1_frozen_loop.sbatch                    # K=4"
echo "  MEMORY_SLOTS=0 sbatch pace/rung1_frozen_loop.sbatch     # no-memory control"
echo "  MEMORY_SLOTS=0 CCOT_DIRECT=true sbatch pace/rung1_frozen_loop.sbatch"
echo "  sbatch pace/rung1b_lora_loop.sbatch                     # LoRA-adapted loop"
echo "  sbatch pace/rung2_staged_unfreeze.sbatch"
echo "  sbatch pace/rung3_l2sp.sbatch"
