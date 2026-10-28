#!/bin/bash
# ============================================================================
# POST-MORTEM PACK PREP -- login node, tokenizer only, a few minutes.  Run it
# BEFORE `sbatch --array=0-5 pace/pm_limbs.sbatch` (limbs 4/5 refuse without it).
#
#   bash pace/pm_prep_packs.sh
#
# TRAIN  data/pg19fw50_recall_train_len4096   first 16,000 rows of the 5B mix
#        pack, EVERY row augmented (frac 1.0) with two facts in chunks 0..5 and
#        a probe about one of them at the end of chunk 7.  Fact-to-probe
#        distance is 2..7 chunks = 0.5..1.75 ring laps, so the probe straddles
#        the one-lap mark the trained gates forget at.  The second fact is a
#        same-format distractor (BABILong's shape: the right fact among wrong
#        ones), so "copy the last code word you saw" does not solve it.
#        16,000 rows = 2x the 8,000 a 2,000-update batch-4 limb reads, so the
#        1.10 check_pack margin holds with room.
#        CAVEAT TO STATE WITH ANY NULL: the probe supervises ~1 answer token in
#        4,096 per row, so the retention gradient is sparse.  8,000 probes is
#        the whole signal.
# VAL    data/pg19_val_recall_len4096        400 held-out PG-19 rows, same
#        recipe, for round 2's probe-accuracy-by-distance read-out on every limb
#        (and on 38,146 itself, which never trained on probes: the zero point).
# ============================================================================
set -e
cd ~/cortex-finetune
module load anaconda3
conda activate cortex-retro
export HF_HUB_OFFLINE=1

TOK=ckpts/olmo-retrofit-cortex
MIX=data/pg19_fw50_olmo_len4096
VAL=data/pg19_olmo_validation_len4096_strided
TRAIN_OUT=data/pg19fw50_recall_train_len4096
VAL_OUT=data/pg19_val_recall_len4096
COMMON="--frac 1.0 --n_facts 2 --n_chunks 8 --fact_chunks 0,1,2,3,4,5 --seed 74"

for d in "$MIX" "$VAL" "$TOK"; do
    [ -e "$d" ] || { echo "ERROR: missing $d"; exit 1; }
done

# The limbs' out dir, created HERE and not by the six array tasks: final5b_shared
# makes it on first use, and six tasks starting together can race between its
# `[ ! -L ]` and `[ -e ]` checks -- the loser sees the winner's fresh symlink and
# exits on "exists but is not a symlink".
export SCRATCH=${SCRATCH:-$HOME/scratch}
if [ ! -L cortex-pm ]; then
    [ -e cortex-pm ] && { echo "ERROR: ~/cortex-finetune/cortex-pm exists and is not a symlink"; exit 1; }
    mkdir -p "$SCRATCH/cortex-pm" && ln -s "$SCRATCH/cortex-pm" cortex-pm
fi
echo "limb out dir: cortex-pm -> $(readlink cortex-pm)"

if [ -d "$TRAIN_OUT" ]; then
    echo "exists, skipping: $TRAIN_OUT"
else
    python tools/prepare_recall_mix.py --data "$MIX" --tokenizer "$TOK" \
        --out "$TRAIN_OUT" --max_rows 16000 --num_proc 8 $COMMON
fi
if [ -d "$VAL_OUT" ]; then
    echo "exists, skipping: $VAL_OUT"
else
    python tools/prepare_recall_mix.py --data "$VAL" --tokenizer "$TOK" \
        --out "$VAL_OUT" --max_rows 400 $COMMON
fi

python tools/check_pack.py --data "$TRAIN_OUT" --steps 2000 --batch_size 4
python - "$TRAIN_OUT" "$VAL_OUT" "$TOK" <<'EOF'
import sys
from datasets import load_from_disk
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(sys.argv[3])
for p in sys.argv[1:3]:
    ds = load_from_disk(p)
    n = min(len(ds), 300)
    hit = sum("code word" in tok.decode(ds[i]["input_ids"][-40:]) for i in range(n))
    print(f"{p}: {len(ds)} rows, probe present in {hit}/{n} of the first {n} row tails")
    if hit < 0.9 * n:
        sys.exit(f"ERROR: frac 1.0 should put a probe on nearly every row; "
                 f"{hit}/{n} means the splice skipped rows (padded rows are skipped)")
EOF
echo "packs ready."
