#!/bin/bash
#SBATCH --account=gts-aivanova7-lab
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH -t 2:00:00
#SBATCH -q inferno
#SBATCH -J carry_task
#SBATCH -o logs/Report-%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=henryyjiang42@gmail.com

# Z ATTEMPT 2, STEP 1: the synthetic positive control -- build its packs.
# CPU only.  An sbatch and not a login-node run because it opens the prose
# pack to copy its features and then materialises a mix (the login-node cap
# has killed "just a checker" before; cortex_cluster_ops).
#
#   sbatch pace/prepare_carry_task.sbatch
#
# Writes three packs:
#   $TRAIN   carry task, input_ids + attention_mask cast to the prose pack's
#            features (seed $TRAIN_SEED)
#   $VAL     carry task, + answer_dep, a DIFFERENT seed ($VAL_SEED) so no
#            validation row is a training row
#   $MIX     prose + carry task at $RATIO of rows -- J1's default ARM_DATA
#
# Size check: J1 consumes CELL_STEPS x 4 rows (2000 x 4 = 8,000) and the loader
# does NOT wrap, so the mix must hold more than that.  MIX_ROWS=16000 covers
# CELL_STEPS up to 4000.

cd $SLURM_SUBMIT_DIR

module load anaconda3
conda activate cortex-retro

export SCRATCH=${SCRATCH:-$HOME/scratch}
export HF_HOME=$SCRATCH/hf_cache
export HF_HUB_OFFLINE=1

TOKENIZER=${TOKENIZER:-ckpts/olmo-retrofit-cortex}
PROSE=${PROSE:-data/pg19_fw50_olmo_len4096}
REGISTERS=${REGISTERS:-16}
OPS=${OPS:-1}
TRAIN_ROWS=${TRAIN_ROWS:-4000}
VAL_ROWS=${VAL_ROWS:-1000}
TRAIN_SEED=${TRAIN_SEED:-1}
VAL_SEED=${VAL_SEED:-2}
RATIO=${RATIO:-0.2}
MIX_ROWS=${MIX_ROWS:-16000}
TAG=r${REGISTERS}$( [ "$OPS" != "1" ] && echo "o$OPS" )
TRAIN=${TRAIN:-data/carry_task_${TAG}_len4096}
VAL=${VAL:-data/carry_task_${TAG}_len4096_val}
PCT=$(python -c "print(int(round($RATIO * 100)))")
MIX=${MIX:-data/j1_pg19fw50_carry${PCT}_len4096}

if [ "$TRAIN_SEED" = "$VAL_SEED" ]; then
    echo "ERROR: TRAIN_SEED == VAL_SEED: the validation rows would BE training rows."
    exit 1
fi
[ -d "$PROSE" ] || { echo "ERROR: no prose pack at $PROSE"; exit 1; }
for d in "$TRAIN" "$VAL" "$MIX"; do
    if [ -e "$d" ]; then echo "ERROR: $d exists -- move it aside, not overwriting a pack."; exit 1; fi
done

echo "=== carry task | R=$REGISTERS ops/line=$OPS | train $TRAIN_ROWS (seed $TRAIN_SEED) | val $VAL_ROWS (seed $VAL_SEED) ==="
echo "    tokenizer: $TOKENIZER   features like: $PROSE"
echo "    mix:       $MIX  ($RATIO of $MIX_ROWS rows)"

python tools/prepare_carry_task.py --tokenizer "$TOKENIZER" \
    --features_like "$PROSE" --out "$TRAIN" --rows "$TRAIN_ROWS" \
    --registers "$REGISTERS" --ops_per_line "$OPS" --seed "$TRAIN_SEED" || exit 1
python tools/prepare_carry_task.py --tokenizer "$TOKENIZER" \
    --features_like "$PROSE" --out "$VAL" --rows "$VAL_ROWS" \
    --registers "$REGISTERS" --ops_per_line "$OPS" --seed "$VAL_SEED" \
    --with_answer_dep || exit 1
python tools/prepare_corpus_mix.py --a "$PROSE" --b "$TRAIN" --out "$MIX" \
    --ratio "$RATIO" --max_rows "$MIX_ROWS" || exit 1

echo ""
echo "DONE.  J1 trains on ARM_DATA=$MIX (its default)."
echo "Score any carry model on the task with, e.g.:"
echo "  DATA=$VAL SCORE=carry Z_NULL=donor ... sbatch pace/eval_carry_2x2.sbatch"
echo "and ALWAYS read SCORE=local next to it: it is the task's own positive control."
