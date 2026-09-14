#!/bin/bash
#SBATCH --account=gts-aivanova7-lab
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128GB
#SBATCH -t 4:00:00
#SBATCH -q inferno
#SBATCH -o logs/Report-%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=henryyjiang42@gmail.com

# Build the strided PG-19 pack.  CPU ONLY — no --gres line, deliberately.
#
# WHY THIS IS AN SBATCH AND NOT A LOGIN-NODE ONE-LINER
# ----------------------------------------------------
# prepare_pg19_dataset.py's header said "run on a LOGIN node" and that was true
# of the OLD script, which truncated every book to ~32k characters and finished
# in minutes on nothing.  After the 2026-09-14 striding fix it tokenizes ~2.5B
# tokens and writes ~10 GB, and — the part that actually bit — its memory
# profile is spiky.  PG-19's first document is 4.3M characters = 1,153,022
# tokens = 282 rows, and a worker holds the source id list AND every window
# sliced out of it simultaneously, ~100 MB for that one book.  At the original
# --map_batch_size 8 x --num_proc 16 that killed a worker on the login node:
#
#   RuntimeError: One of the subprocesses has abruptly died during map operation
#
# which is SIGKILL from the per-user memory cap, not a bug in the map function.
# --map_batch_size now defaults to 1 and --writer_batch_size to 200; this job
# adds real memory on top so the outlier books have somewhere to live.
#
# It is NOT slow.  2.5B tokens at the measured 527k tok/s is 79 core-minutes,
# ~5 minutes across 16 workers; the 12-hour figure the progress bar showed was
# extrapolation from the first books, which are the largest in the corpus.  The
# 4h limit is for the save, which is the slow half of every pack script here.
#
# PREREQUISITE — the download must already be cached, because compute nodes have
# no internet and this script sets HF_HUB_OFFLINE=1.  Once, on a login node:
#
#   export HF_HOME=$SCRATCH/hf_cache && unset HF_HUB_OFFLINE
#   python -c "from datasets import load_dataset; load_dataset('emozilla/pg19', split='train')"
#
#   sbatch pace/prepare_pg19_pack.sbatch
#   MAX_LENGTH=8192 OUT=data/pg19_olmo_len8192_strided sbatch pace/prepare_pg19_pack.sbatch
#   NUM_PROC=8 sbatch pace/prepare_pg19_pack.sbatch      # if a worker still dies
#
# THIS JOB IS P0.4.  Its output lines are the yield measurement the corpus plan
# has been waiting on — rows per document and total tokens of window.  Expect
# roughly 600k rows / ~2.5B tokens / ~21 rows per book at max_length 4096.  If
# it reports anything near 28,602 rows the striding fix did not take and NOTHING
# downstream of it is safe: every "PG-19 cannot supply the volume" decision in
# the project came from that number.

cd $SLURM_SUBMIT_DIR

module load anaconda3
conda activate cortex-retro

export SCRATCH=${SCRATCH:-$HOME/scratch}
export HF_HOME=$SCRATCH/hf_cache
export HF_HUB_OFFLINE=1
# HF's fast tokenizers use internal thread parallelism; datasets then forks for
# num_proc, and the two together deadlock or warn on every worker.  num_proc is
# where the parallelism should live here — one book per worker is already the
# right granularity.
export TOKENIZERS_PARALLELISM=false

TOKENIZER=${TOKENIZER:-ckpts/olmo-retrofit-cortex}
# NOT data/pg19_olmo_len4096 — that name is TAKEN by the old truncating
# packer's output (28,602 rows / ~117M tokens, the first ~4k tokens of each
# book), which past results were built on.  Two artifacts sharing one name is
# how this project lost six weeks to ARM_DATA and why the two base checkpoints
# carry a standing warning.  The suffix names the mechanism that differs.
OUT=${OUT:-data/pg19_olmo_len4096_strided}
MAX_LENGTH=${MAX_LENGTH:-4096}
NUM_PROC=${NUM_PROC:-16}
MAP_BATCH=${MAP_BATCH:-1}
WRITER_BATCH=${WRITER_BATCH:-200}
DATASET=${DATASET:-emozilla/pg19}
MIN_TOKENS=${MIN_TOKENS:-0}

if [ -e "$OUT" ]; then
    echo "ERROR: $OUT already exists."
    if [ -f "$OUT/state.json" ] && [ -f "$OUT/dataset_info.json" ]; then
        echo "  It is a COMPLETE pack — state.json and dataset_info.json are"
        echo "  both present, and those are written LAST.  So this is not a"
        echo "  killed save; something already built a pack under this name."
        echo "  DO NOT overwrite it.  Check what it holds first:"
        echo "    python -c \"from datasets import load_from_disk;"
        echo "      print(len(load_from_disk('$OUT')))\""
        echo "  ~28,602 rows means the OLD TRUNCATING packer's output, which"
        echo "  past results were built on.  Write the strided pack elsewhere:"
        echo "    OUT=${OUT}_strided sbatch pace/prepare_pg19_pack.sbatch"
    else
        echo "  state.json / dataset_info.json are MISSING, and they are written"
        echo "  last — so this is a killed save and the shards are incomplete."
        echo "  Safe to rm -rf and resubmit."
    fi
    exit 1
fi

echo "=== pg19 pack | $DATASET -> $OUT | max_length=$MAX_LENGTH ==="
echo "    num_proc=$NUM_PROC map_batch=$MAP_BATCH writer_batch=$WRITER_BATCH"
echo "    tokenizer=$TOKENIZER  min_tokens=$MIN_TOKENS"
free -g | head -2

python tools/prepare_pg19_dataset.py \
    --tokenizer "$TOKENIZER" \
    --dataset "$DATASET" \
    --out "$OUT" \
    --max_length "$MAX_LENGTH" \
    --num_proc "$NUM_PROC" \
    --map_batch_size "$MAP_BATCH" \
    --writer_batch_size "$WRITER_BATCH" \
    --min_tokens "$MIN_TOKENS"
RC=$?

echo ""
if [ $RC -ne 0 ]; then
    echo "FAILED (exit $RC).  If a worker died abruptly it is memory: lower"
    echo "NUM_PROC (8, then 4) before raising --mem — peak is num_proc x the"
    echo "largest book, and PG-19's largest are ~1.2M tokens each."
    exit $RC
fi

echo "=== pack written; verifying it is complete ==="
# state.json / dataset_info.json are written LAST.  Their absence is the
# signature of a killed save, and the row-count one-liner cannot see it.
for f in state.json dataset_info.json; do
    if [ ! -f "$OUT/$f" ]; then
        echo "ERROR: $OUT/$f missing — the save did not finish."; exit 1
    fi
done
ls -la "$OUT" | tail -5
echo ""
echo "P0.4 answered.  Next: build the mix, then check_pack both legs —"
echo "  python tools/prepare_corpus_mix.py --a $OUT \\"
echo "      --b data/fineweb_edu_olmo_len4096 \\"
echo "      --out data/pg19_fw50_olmo_len4096 --ratio 0.5 --max_rows 940000"
echo ""
echo "  (--a is $OUT, the STRIDED pack.  data/pg19_olmo_len4096 without the"
echo "   suffix is the old truncated one — do not feed that to the mix.)"
echo "  python tools/check_pack.py --data data/pg19_fw50_olmo_len4096 \\"
echo "      --steps 26703 --batch_size 32 --max_length 4096"
