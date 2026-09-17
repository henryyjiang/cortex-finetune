#!/bin/bash
#SBATCH --account=gts-aivanova7-lab
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH -t 0:30:00
#SBATCH -q inferno
#SBATCH -J unit_suite
#SBATCH -o logs/Report-%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=henryyjiang42@gmail.com

# The unit suite, on a compute node.
#
#   sbatch pace/run_tests.sbatch                      # everything under tests/
#   TESTS="tests/test_compare_arms.py tests/test_prelaunch_final.py" \
#       sbatch pace/run_tests.sbatch                  # a subset
#   PYTEST_ARGS="-q -x" sbatch pace/run_tests.sbatch  # stop at the first failure
#
# WHY THIS IS AN SBATCH AND NOT `python -m pytest` ON THE LOGIN NODE.  The suite
# is 40 files that build small Raven models on CPU and run real forward and
# backward passes through them; it is a torch workload, not a lint.  It takes
# ~61s on a compute node (685 passed, 1 skipped, job 13272609) and minutes on
# the login node, which is shared, throttled, and enforces a per-user memory cap
# by KILLING you -- the same trap that made `tools/check_latent_probe.py` look
# like a probe failure when it was a kill.  A slow suite is the polite version
# of that failure; do not wait it out on login.
#
# NO GPU IS REQUESTED, on purpose.  Nothing in tests/ needs a device: the model
# fixtures are CPU float32 by construction, because a test that needed an H200
# could not run in the pre-launch gate either.  That also means this job
# schedules on the CPU partitions and usually starts immediately, instead of
# queueing behind real training.  The cuda module below is loaded only so the
# cortex-retro torch build finds its link-time libraries; nothing here touches a
# device.
#
# The exit code is pytest's, so --mail-type=FAIL means what it says and a red
# suite fails the JOB rather than printing red text into a log nobody reopens.

cd $SLURM_SUBMIT_DIR

module load anaconda3
module load cuda/12.1.1
conda activate cortex-retro

export SCRATCH=${SCRATCH:-$HOME/scratch}
export HF_HOME=$SCRATCH/hf_cache
export HF_HUB_OFFLINE=1

# Threads: the suite is many small tensors, where torch's intra-op pool costs
# more in contention than it buys.  Leave it at 1 per process and let pytest
# walk the files; -n auto is NOT available (pytest-xdist is not in the env).
export OMP_NUM_THREADS=1

TESTS=${TESTS:-tests/}
PYTEST_ARGS=${PYTEST_ARGS:--q}

echo "=== unit suite | $(hostname) ==="
echo "    tests: $TESTS"
echo "    args:  $PYTEST_ARGS"
python -c "import torch, sys; print(f'    torch {torch.__version__}, python {sys.version.split()[0]}')"
echo ""

START=$SECONDS
python -m pytest $TESTS $PYTEST_ARGS
RC=$?
echo ""
echo "=============================================================="
if [ $RC -eq 0 ]; then
    echo "SUITE GREEN in $((SECONDS - START))s"
else
    echo "SUITE RED (rc=$RC) after $((SECONDS - START))s"
    echo "  Before believing a failure in a probe or gate test: every red this"
    echo "  suite has produced so far has been the INSTRUMENT encoding an"
    echo "  assumption the architecture does not share, 8 for 8, with zero"
    echo "  architecture defects.  Check whether the check is well-posed for"
    echo "  the arm it is failing on."
fi
echo "=============================================================="
exit $RC
