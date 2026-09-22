#!/bin/bash
#SBATCH --account=gts-aivanova7-lab
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H200:1
#SBATCH --mem=128GB
#SBATCH -t 3:00:00
#SBATCH -q inferno
#SBATCH -J p30_noread
#SBATCH -o logs/Report-%A_%a.out
#SBATCH --array=0-1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=henryyjiang42@gmail.com

# P3.0 TIER 1.5 READ-OUT: the NO-READ baseline (p30 S5 rows 2 vs 3).
#
#   sbatch pace/p30_noread.sbatch          # [0] a2  [1] a3z
#
# Per arm, two steps on one node:
#   1. tools/oracle_overlay.py -- the oracle's 1000 steps never reached a full
#      chkpt.pt (those save every 2 * save_interval), so build one from the
#      PARENT + the trained latent_reader, after checking frozen tensors match
#      the parent bit-for-bit.  A mismatch aborts the cell.
#   2. pace/eval_carry_2x2.sbatch with Z_NULL=off on the held-out strided
#      validation pack.  Z_main (E1Z0 vs E1Z1) is the trained read on real Z
#      against NO read at all.
#
#   Z_main clearly negative   row 2: the read buys capacity, not content
#   Z_main ~0 or positive     row 3: the trained read does not beat no-read,
#                             which contradicts tier 1 -- audit the instrument

cd $SLURM_SUBMIT_DIR

module load anaconda3
module load cuda/12.1.1
conda activate cortex-retro

OUT_ROOT=${OUT_ROOT:-cortex-retrofit}
ARMS=(a2 a3z)
IDX=${SLURM_ARRAY_TASK_ID:-0}
ARM=${ARMS[$IDX]}
if [ -z "$ARM" ]; then echo "no arm at index $IDX (use --array=0-1)"; exit 0; fi

case "$ARM" in
    a2)  PARENT_RUN=probe-p1-a2-accum-w16-cc8-z ;;
    a3z) PARENT_RUN=probe-p1-a3z-gated-w16k64-cc8-z ;;
esac
PARENT=$(ls -d $OUT_ROOT/$PARENT_RUN/checkpoint_* 2>/dev/null | sort -t_ -k2 -n | tail -1)
ORACLE=$OUT_ROOT/p30-oracle-$ARM-real/final_checkpoint
OVERLAY=$OUT_ROOT/p30-oracle-$ARM-real/eval_overlay

echo "=== p30 no-read | arm=$ARM ==="
echo "    parent:  $PARENT"
echo "    oracle:  $ORACLE"
echo "    overlay: $OVERLAY"
[ -n "$PARENT" ] && [ -d "$PARENT" ] || { echo "ERROR: no parent checkpoint under $OUT_ROOT/$PARENT_RUN"; exit 1; }
[ -d "$ORACLE" ] || { echo "ERROR: no oracle save at $ORACLE"; exit 1; }

python tools/oracle_overlay.py --parent "$PARENT" --oracle "$ORACLE" --out "$OVERLAY"
RC=$?
if [ $RC -ne 0 ]; then echo "overlay ($ARM) FAILED rc=$RC -- eval not run"; exit $RC; fi

# EXPORTED, not prefixed: the child reads these from the environment.
export ARM CKPT=$OVERLAY MIN_STEP=0 Z_NULL=off MAX_EXAMPLES=${MAX_EXAMPLES:-300}
export DATA=${DATA:-data/pg19_olmo_validation_len4096_strided}
export EXTRA_SETS="--set latent_read=xattn --set latent_read_depth=none --set latent_read_heads=8 --set latent_s0_read=false"
export OUT=eval_results/p30_noread/$ARM
bash pace/eval_carry_2x2.sbatch
