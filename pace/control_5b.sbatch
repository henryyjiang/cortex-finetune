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
#SBATCH -J c5b
#SBATCH -o logs/Report-%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=henryyjiang42@gmail.com

# ============================================================================
# C-CHUNKED at 5B -- the budget-matched no-memory control.  ARM 1 OF 2.
#
#   PHASE=heal sbatch pace/control_5b.sbatch
#   PHASE=heal RESUME_PATH=cortex-5b/c-chunked/checkpoint_XXXX sbatch pace/control_5b.sbatch
#   # the corpus switch, ONCE, at step 11,444:
#   PHASE=mix RESUME_PATH=... EXTRA_ARGS="--reset_dataset_position true" \
#       sbatch pace/control_5b.sbatch
#
# Sibling: pace/cortex_final_5b.sbatch.  THIS ARM RUNS FIRST AND ALONE; the
# memory arm launches after it, not beside it.
#
# The shared recipe lives in pace/final_5b_common.sh and is written ONCE, so
# the two arms cannot drift apart.  `diff pace/control_5b.sbatch
# pace/cortex_final_5b.sbatch` is the complete difference between the runs.
#
# Supersedes pace/control_retrofit.sbatch, which never launched (no
# `=== C-chunked` banner exists anywhere in logs/, checked 2026-09-30).  Its
# recipe is carried here unchanged; READ ITS HEADER for the reasoning behind
# the batch size, the LR that was deliberately not re-tuned, the D-fallback
# geometry and the corpus build.  It is the document of record and is kept for
# that; it now refuses to launch.
#
# THE CLAIM THIS RUN EXISTS TO SUPPORT.  paper/ states the method claim is
# unsupported because there is no budget-matched no-memory arm at 5B.  This is
# that arm.  The one thing that must be true of it is that it differs from
# cortex-final in exactly one variable -- which is why the recipe is sourced
# rather than copied.
#
# WHY `--cortex.use_memory false` ALONE WAS NOT A CONTROL.  train.py's fwd/bwd
# dispatch used to read `cfg.cortex["use_memory"] and cross_chunks > 1`, so
# turning memory off silently turned CHUNKING off too and this run would have
# trained on full 4,096-token windows against the memory arm's 8x512.  Chunk
# length is the biggest lever the ceiling probes ever found.  The dispatch now
# keys on cross_chunks ALONE, and this script is the only consumer of that fix.
#
# MICRO_BATCH_SIZE 1, AND IT IS NOT A CONSERVATISM
# ------------------------------------------------
# cortex-final runs 2.  This arm cannot, and the reason inverts the obvious
# guess: `--cortex.window_backward true` is what makes mbs 2 fit, and it is
# REFUSED without memory (train.py:2117, "use_memory is false, so there is no
# carry to window").  The flush inside cortex_fwd_bwd is separately gated on
# `m_cross is not None`, which never holds here.  So THIS arm keeps ALL EIGHT
# chunks' activation graphs alive to one backward while cortex-final keeps
# FOUR.  Column-positions held per micro-step:
#
#     control        8 x 512                 = 4,096
#     cortex-final   4 x (512 + 64 E + 64 Z) = 2,560
#
# The control is the HEAVIER arm by 1.6x, and the 90.84 GiB measured at mbs 2
# (job 13662406_2) is the MEMORY arm's number -- it does not transfer.  The
# only figure ever measured for THIS config is the OOM gate's 106.25 GiB (76%)
# at mbs 1.
#
# TO RAISE IT, MEASURE IT -- and measure it BEFORE the first link, because
# MICRO_BS is pinned for the life of the chain (the resume row cursor is
# data_start_step x micro_batch_size) and cannot be changed on a resume:
#
#     sbatch pace/smoke_control_mbs2.sbatch      # ~10 min of H200, 2h wall cap
#
# If it fits, launch this arm with MICRO_BS=2 from step 0.  It is worth the ten
# minutes: at mbs 1 this arm is the long pole of the whole program.
#
# THE ARMS DIFFERING ON mbs IS NOT A CONFOUND.  Both run 32 rows per optimizer
# step, 38,147 steps, the same LR, the same packs and the same seed; the row
# permutation is a pure function of (seed, len(dataset)) consumed in order, so
# an optimizer step reads THE SAME 32 ROWS at either setting.  What differs is
# reduction order inside a micro-step -- floating-point noise.
#
# PRE-FLIGHT (once, before the first submit)
#   git pull                       # so the run records a known commit
#   python tools/check_pack.py --data data/fineweb_edu_olmo_len4096 \
#       --steps 11444 --batch_size 32 --max_length 4096
#   python tools/check_pack.py --data data/pg19_fw50_olmo_len4096 \
#       --steps 26703 --batch_size 32 --max_length 4096
#   sbatch pace/smoke_control_mbs2.sbatch        # decides MICRO_BS, see above
#
# MONITORING
#   after every link   tools/watch_run.py  (this script calls it on exit)
#   at 4 checkpoints   ARM=control STEP=<n> sbatch pace/midrun_check.sbatch
#   the bars           ../final_5b_launch.md S4
# ============================================================================

cd $SLURM_SUBMIT_DIR

ARM=control
source pace/final_5b_common.sh
final5b_shared

RUN_NAME=${RUN_NAME:-c-chunked}
MICRO_BS_DEFAULT=1                 # see the header -- NOT the memory arm's 2

# --cortex.use_memory false is the ONLY mechanism flag set, and that is
# deliberate.  No prefix_memory, no accum_vecs, no latent_*: those would be
# DEAD CONFIG here, and dead config in a checkpoint's config.json is how a
# later eval script silently reconstructs a mechanism that never trained.
# train.py's step-0 negative guard raises if the graft built cortex anyway.
ARM_ARGS="--cortex.use_memory false"

ARM_BANNER="    mechanism:  NONE (use_memory false).  Chunked, no carry, no summary
                slots, no memory parameters -- the budget-matched control."

final5b_launch
exit $?
