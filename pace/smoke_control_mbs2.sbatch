#!/bin/bash
#SBATCH --account=gts-aivanova7-lab
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96GB
#SBATCH --gres=gpu:H200:1
#SBATCH --constraint=H200
#SBATCH -t 1:00:00
#SBATCH -q inferno
#SBATCH -J ombs2
#SBATCH -o logs/Report-%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=henryyjiang42@gmail.com

# ============================================================================
# DOES THE CONTROL FIT AT micro_batch_size 2?  Run this BEFORE the first
# control link -- MICRO_BS is pinned for the life of the chain and cannot be
# raised on a resume.
#
#   sbatch pace/smoke_control_mbs2.sbatch
#
# COST: ~10 minutes of H200 (four short measured micro-steps, no data, no
# pack, no checkpoint).  The 1h wall cap is queue headroom.  ~0.2 GPU-hours to
# decide whether the control is a ~5-link run or a ~10-link one.
#
# THE QUESTION, AND WHY IT IS OPEN
# --------------------------------
# cortex-final runs micro_batch_size 2, MEASURED at 90.84 GiB of 138.7 (job
# 13662406_2).  The obvious inference -- "the control carries nothing, so it
# is lighter" -- IS WRONG.  --cortex.window_backward is what makes mbs 2 fit,
# and it is REFUSED without memory (train.py:2117, "use_memory is false, so
# there is no carry to window"); the flush inside cortex_fwd_bwd is separately
# gated on `m_cross is not None`, which never holds on a no-memory run.  So
# the control keeps ALL EIGHT chunks' activation graphs alive to one backward
# while cortex-final keeps FOUR:
#
#     control        8 x 512                 = 4,096 column-positions
#     cortex-final   4 x (512 + 64 E + 64 Z) = 2,560 column-positions
#
# The control is the HEAVIER arm by 1.6x.  The only figure ever measured for
# its config is the OOM gate's 106.25 GiB (76%) at mbs 1, and a back-of-
# envelope doubling of the activation term does NOT fit in 138.7 -- but that
# envelope is fitted through two points from a different config and it
# over-predicts J7's real measurement by 25 GiB, so it is not trustworthy in
# either direction.  MEASURE IT.  That is what this script is for, and it is
# the project's own rule: gate it with the smoke, not with the arithmetic.
#
# FOUR CELLS, and each one earns its two minutes:
#
#   control mbs1   the reproduction.  If it does not land near the gate's
#                  106.25 GiB then something moved since 2026-09-14 and the
#                  other three rows are not comparable to anything on record.
#   control mbs2   THE QUESTION.
#   cortex  mbs2   the cross-check against a REAL measured run (90.84 GiB on
#                  job 13662406_2).  If this row disagrees with that, the
#                  smoke is mispricing and the control rows inherit the error.
#   cortex  mbs1   the headroom cortex-final is giving up, for the record.
#
# READING IT.  `FITS` is a measured peak, `OOM` is a measured OOM; both are
# answers and neither is a script failure (exit 1 means the script broke).
# Treat anything above ~130 of 138.7 GiB as NOT usable even if it says FITS:
# a 48h run meets ragged batches and allocator fragmentation that a four-step
# smoke does not, and B2 peaked 4% above this gate's own prediction.
#
# IF mbs 2 FITS with room: launch the control with MICRO_BS=2 FROM STEP 0.
# IF IT DOES NOT: launch at 1 and accept ~10 links.  Do not raise it later --
# the resume row cursor is data_start_step x micro_batch_size, so a mid-chain
# change skips the wrong number of rows behind a healthy loss curve.
# ============================================================================

cd $SLURM_SUBMIT_DIR

module load anaconda3
module load cuda/12.1.1
conda activate cortex-retro

export SCRATCH=${SCRATCH:-$HOME/scratch}
export HF_HOME=$SCRATCH/hf_cache
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODEL=${MODEL:-ckpts/olmo-retrofit-cortex}
MAX_LENGTH=${MAX_LENGTH:-4096}
CROSS_CHUNKS=${CROSS_CHUNKS:-8}
BATCH_SIZE=${BATCH_SIZE:-32}
MAX_MEAN_REC=${MAX_MEAN_REC:-8}
JSON=${JSON:-eval_results/oom_control_mbs2.json}
mkdir -p "$(dirname "$JSON")"

echo "=== control micro_batch_size gate | ${MAX_LENGTH} x cc${CROSS_CHUNKS} ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# The one check that reads the MODELING FILE rather than the config -- the
# failure that once ran 9 hours with no cross-segment memory and a healthy
# loss curve.  A stale file here would price the memory cells as controls.
echo "prefix_pack occurrences in the checkpoint's modeling file:"
grep -c prefix_pack "$MODEL/raven_modeling_minimal_cortex.py" \
    || { echo "  (no modeling file at $MODEL -- is it graft-prepared?)"; exit 1; }

# label : micro_bs : use_memory
CELLS=(
    "control-mbs1:1:false"
    "control-mbs2:2:false"
    "cortex-mbs2:2:true"
    "cortex-mbs1:1:true"
)

BROKEN=0
for C in "${CELLS[@]}"; do
    IFS=':' read -r LABEL MBS MEM <<< "$C"
    echo ""
    echo "######## $LABEL (micro_batch_size $MBS, use_memory $MEM) ########"
    # The memory cells carry cortex-final's geometry EXACTLY, window_backward
    # included -- that flag is the whole reason the two arms price differently,
    # so a smoke that omitted it would measure a model nobody is launching.
    if [ "$MEM" = "true" ]; then
        MEM_ARGS="--use_memory true --prefix_memory gated --accum_vecs 16 \
                  --gate_slots 64 --accum_max 128 --gate_route ring \
                  --gate_norm tanh --gate_init zero --gate_fill grow \
                  --latent_carry --carry_grad_chunks 4 --window_backward"
    else
        # No window_backward and no carry_grad_chunks: train.py REFUSES both
        # without memory, so passing them here would price a configuration the
        # control cannot run.
        MEM_ARGS="--use_memory false"
    fi
    python tools/smoke_geometry_oom.py \
        --model_name "$MODEL" \
        --max_length "$MAX_LENGTH" \
        --cross_chunks "$CROSS_CHUNKS" \
        --batch_size "$BATCH_SIZE" \
        --micro_batch_size "$MBS" \
        --mean_recurrence "$MAX_MEAN_REC" \
        --with_optimizer true \
        $MEM_ARGS \
        --json_out "$JSON"
    RC=$?
    if [ $RC -ne 0 ]; then
        echo "!!! script failed (exit $RC) on $LABEL -- this is NOT a measured"
        echo "!!! OOM, it is a broken run.  Read the traceback above."
        BROKEN=1
    fi
done

echo ""
echo "=== summary ==="
python - "$JSON" <<'SUM'
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
print(f"  {'cell':<22} {'status':<6} {'peak':>9} {'of':>7} {'use':>6} {'s/micro':>8}")
for r in rows:
    mem = "gated+Z" if r["use_memory"] else "none"
    lbl = f"{mem} mbs{r['micro_batch_size']}"
    tot = r["total_gib"] or 1
    print(f"  {lbl:<22} {r['status']:<6} {r['peak_res_gib']:>7.2f}G "
          f"{tot:>6.1f}G {r['peak_res_gib']/tot:>5.0%} {r['step_s']:>8.2f}")
ctrl2 = [r for r in rows if not r["use_memory"] and r["micro_batch_size"] == 2]
print()
if not ctrl2:
    print("  the control mbs2 cell did not report -- read the log.")
elif ctrl2[-1]["status"] != "FITS":
    print("  VERDICT: control mbs2 OOMs.  Launch at MICRO_BS=1 and plan ~10")
    print("           links.  Do NOT raise it mid-chain.")
elif ctrl2[-1]["peak_res_gib"] / (ctrl2[-1]["total_gib"] or 1) > 0.94:
    print(f"  VERDICT: control mbs2 FITS at "
          f"{ctrl2[-1]['peak_res_gib']:.1f}G but with under 6% headroom.")
    print("           A 48h run meets ragged batches and fragmentation a")
    print("           four-step smoke does not, and B2 peaked 4% above this")
    print("           gate's own prediction.  Treat as DOES NOT FIT.")
else:
    print(f"  VERDICT: control mbs2 FITS at {ctrl2[-1]['peak_res_gib']:.1f}G.")
    print("           Launch the control with MICRO_BS=2 FROM STEP 0:")
    print("             PHASE=heal MICRO_BS=2 sbatch pace/control_5b.sbatch")
SUM
exit $BROKEN
