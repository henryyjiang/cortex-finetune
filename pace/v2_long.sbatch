#!/bin/bash
#SBATCH --account=gts-aivanova7-lab
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128GB
#SBATCH --gres=gpu:H200:1
#SBATCH --constraint=H200
#SBATCH -t 48:00:00
#SBATCH -q inferno
#SBATCH -J v2_long
#SBATCH -o logs/Report-%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=henryyjiang42@gmail.com

# ============================================================================
# v2capwarm, LONG: the pm_limbs [8] limb (gated_v2 paired + E store cap 82 +
# warm-up carry) continued from checkpoint_38146 for one 48 h wall instead of
# 2,000 updates.  Everything but the length is pm_limbs.sbatch's, run as-is:
# same flags, LR x0.1 constant after 96 warmup steps, seed 75, ZNORM 78.9, the
# same post-run gate.  Differences:
#   batch 32       the 5B recipe's 131,072 tokens/update (the limbs ran 4)
#   no stop step   CELL_STEPS 20,000 is past what 48 h can reach (~7-8.5k
#                  updates at ~5-6.5k tok/s), so the WALL ends it: the timeout
#                  save at 20 min left is the final checkpoint, and pm_limbs
#                  stops python 8 min before the wall so the gate still runs.
#   saves          full resumable checkpoint every 600 updates (~4 h), so a
#                  node fault (links have died ~20 h in) costs <= ~4 h.
#                  NO PRUNING: ~13 GB per resumable + ~5.6 GB per model-only
#                  dir every 300 -> plan on ~250-300 GB under cortex-pm/.
#   no anneal      a branch's fresh schedule counts from 0 against the
#                  absolute max_steps, so the decay can never land inside this
#                  run; the final checkpoint is at constant LR, like the limbs.
#
#   first link:  sbatch pace/v2_long.sbatch
#   next link:   sbatch --export=ALL,RESUME_PATH=cortex-pm/pm-v2capwarm-long/checkpoint_<N> pace/v2_long.sbatch
#                (the end of each link prints this line with <N> filled in;
#                read the next link's "Resumed from" line, not the last logged
#                update -- a timeout save is behind where the link stopped)
# Knobs (via --export): LR_SCALE, WARM_P, E_CAP, V2_AGE_PRIOR, RUN_SUFFIX.
# V2_AGE_PRIOR defaults to 0.25 here (2026-11-09): the setting checked by the
# 2,000-update pm-v2capwarm-stfix limb, with the fixed straight-through
# surrogate (pm_limbs.sbatch header).  Read that limb before launching this.
# ============================================================================

cd $SLURM_SUBMIT_DIR
export SLURM_ARRAY_TASK_ID=8                 # pm_limbs' v2capwarm
export RUN_SUFFIX=${RUN_SUFFIX:--long}
export LIMB_BATCH=${LIMB_BATCH:-32}
export CELL_STEPS=${CELL_STEPS:-20000}
export LIMB_SAVE_INTERVAL=${LIMB_SAVE_INTERVAL:-300}
export DIAG_INTERVAL=${DIAG_INTERVAL:-50}
export V2_AGE_PRIOR=${V2_AGE_PRIOR:-0.25}
export WALL_LIMITED=1
export WALL_SECS=$(( 48 * 3600 ))            # keep equal to #SBATCH -t above
exec bash pace/pm_limbs.sbatch
