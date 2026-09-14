#!/bin/bash
#SBATCH --account=gts-aivanova7-lab
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=96GB
#SBATCH --gres=gpu:H200:1
#SBATCH -t 2:00:00
#SBATCH -q inferno
#SBATCH -o logs/Report-%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=henryyjiang42@gmail.com

# THE LAST GATE BEFORE THE CONTROL LAUNCH (cortex_next_phase_framework.md §10).
#
# The row-length question is not a design question — §10 locks config D on the
# depth argument and the only thing that can veto it is memory.  This job
# answers it in an hour, and it MUST run before any pack is built: at config D
# the FineWeb-Edu pack has to be rebuilt at 8,192, and discovering the OOM after
# that costs a corpus rebuild instead of a job.
#
#   sbatch pace/smoke_geometry_oom.sbatch
#   CONTROL=1 sbatch pace/smoke_geometry_oom.sbatch      # + the no-memory row
#   CAPPED=1  sbatch pace/smoke_geometry_oom.sbatch      # + D with accum_max 128
#   GEOMETRIES="8192:16:16" sbatch pace/smoke_geometry_oom.sbatch
#
# H200, NOT A100.  H100 and A100-80 are both 80GB (cortex-cluster-ops) and B2
# peaked at 119.76 GiB on the H200's 141 at HALF this row length, so an 80GB
# card cannot answer the question the run will actually face.
#
# WHY THE MEMORY MODEL AND NOT THE CONTROL.  The control launches first, but the
# two runs must share one row length, and the memory model's footprint is
# strictly the larger (carry columns in every chunk's attention).  Sizing on the
# control would green-light a geometry the final model cannot run.  CONTROL=1
# adds the no-memory row anyway — it costs four minutes and it is the first
# measurement of what the control itself will use.
#
# Each geometry runs as its OWN python process: peak-memory stats do not reset
# cleanly enough across geometries inside one process, and an OOM in the largest
# must not poison the smaller ones.  The script exits 0 for BOTH fit and OOM
# (a measured OOM is an answer, not a crash) and 1 only when the script itself
# breaks — so a nonzero exit here is a real failure and is reported as one,
# which a blanket `|| true` would have hidden.

cd $SLURM_SUBMIT_DIR

module load anaconda3
module load cuda/12.1.1
conda activate cortex-retro

export SCRATCH=${SCRATCH:-$HOME/scratch}
export HF_HOME=$SCRATCH/hf_cache
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODEL=${MODEL:-ckpts/olmo-retrofit-cortex}
MAX_MEAN_REC=${MAX_MEAN_REC:-8}
ACCUM_VECS=${ACCUM_VECS:-32}
# Both 0 = "let the script apply B2's RULE rather than B2's number":
# accum_max = cross_chunks x accum_vecs (b2_retrofit.sbatch:409, "moves WITH
# cross_chunks") and carry_grad_chunks = cross_chunks / 2 (4 of 8 in the arm).
# Both scale the footprint, and both are larger at cc16 than anything B2 ran —
# 512 carry rows on the last chunk against B2's 256, and 8 chunks of retained
# carry graph against 4.  Pricing them at B2's literal 256/4 would measure a
# geometry nobody intends to run.
ACCUM_MAX=${ACCUM_MAX:-0}
CARRY_GRAD_CHUNKS=${CARRY_GRAD_CHUNKS:-0}
MAX_STEPS=${MAX_STEPS:-38147}

# max_length : cross_chunks : batch_size    (batch_size only scales the wall-clock
# projection — it is pure gradient accumulation and costs no memory).
# Ascending, so the fit/OOM boundary is located before the stretch case.
#   4096:8:32   D-fallback   512-tok chunks, reuses the existing len4096 pack
#   8192:16:16  config D     512-tok chunks, chain depth 16          <- the decision
#   16384:16:8  stretch      1024-tok chunks AND depth 16 at the same tokens/step;
#                            dominates D on both axes if it fits, and it probably
#                            does not (4x B2's per-row activations against D's 2x)
GEOMETRIES=${GEOMETRIES:-"4096:8:32 8192:16:16 16384:16:8"}

JSON=logs/oom_smoke_${SLURM_JOB_ID}.jsonl

echo "=== geometry OOM gate | model=$MODEL | mean_rec=$MAX_MEAN_REC ==="
echo "    geometries: $GEOMETRIES"
echo "    json: $JSON"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# The one check that reads the MODELING FILE and not the config — the failure
# that ran 9 hours with no cross-segment memory and a healthy loss curve
# (smoke_prefix_real.sbatch's header has the full story).  A stale modeling file
# here would silently measure the control's footprint under memory=True.
echo "prefix_pack occurrences in the checkpoint's modeling file:"
grep -c prefix_pack "$MODEL/raven_modeling_minimal_cortex.py" \
    || echo "  (no modeling file at $MODEL — is it graft-prepared?)"

BROKEN=0
for G in $GEOMETRIES; do
    LEN=${G%%:*}; REST=${G#*:}; CC=${REST%%:*}; BS=${REST#*:}
    echo ""
    echo "######## ${LEN} x cc${CC} (bs ${BS}) ########"
    python tools/smoke_geometry_oom.py \
        --model_name "$MODEL" \
        --max_length "$LEN" \
        --cross_chunks "$CC" \
        --batch_size "$BS" \
        --max_steps "$MAX_STEPS" \
        --micro_batch_size 1 \
        --mean_recurrence "$MAX_MEAN_REC" \
        --accum_vecs "$ACCUM_VECS" \
        --accum_max "$ACCUM_MAX" \
        --carry_grad_chunks "$CARRY_GRAD_CHUNKS" \
        --use_memory true \
        --with_optimizer true \
        --json_out "$JSON"
    RC=$?
    if [ $RC -ne 0 ]; then
        echo "!!! script failed (exit $RC) at ${LEN}x cc${CC} — this is NOT a"
        echo "!!! measured OOM, it is a broken run.  Read the traceback above."
        BROKEN=1
    fi
done

if [ -n "$CAPPED" ]; then
    # Config D with the buffer capped at 128 rows instead of 512.  Only worth a
    # run if D came back OOM: it is the one mechanism lever that buys memory
    # back without touching the row length, and unlike D-fallback it KEEPS the
    # chain depth that §10 says is the whole reason for D.
    echo ""
    echo "######## config D with accum_max 128 (capped buffer) ########"
    python tools/smoke_geometry_oom.py \
        --model_name "$MODEL" \
        --max_length 8192 --cross_chunks 16 --batch_size 16 \
        --max_steps "$MAX_STEPS" --micro_batch_size 1 \
        --mean_recurrence "$MAX_MEAN_REC" \
        --accum_vecs "$ACCUM_VECS" --accum_max 128 \
        --use_memory true --with_optimizer true \
        --json_out "$JSON" || BROKEN=1
fi

if [ -n "$CONTROL" ]; then
    LEN=${CONTROL_LEN:-8192}; CC=${CONTROL_CC:-16}
    echo ""
    echo "######## control: ${LEN} x cc${CC}, no memory ########"
    python tools/smoke_geometry_oom.py \
        --model_name "$MODEL" \
        --max_length "$LEN" --cross_chunks "$CC" --batch_size 16 \
        --max_steps "$MAX_STEPS" --micro_batch_size 1 \
        --mean_recurrence "$MAX_MEAN_REC" \
        --use_memory false --with_optimizer true \
        --json_out "$JSON" || BROKEN=1
fi

echo ""
echo "=== summary ==="
python - "$JSON" <<'SUMMARY'
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1])]
print(f"{'row':>7} {'cc':>3} {'chunk':>6} {'mem':>5} {'amax':>5} {'status':>6} "
      f"{'alloc':>8} {'reserved':>9} {'%dev':>5} {'s/step':>7}")
for r in rows:
    pct = 100 * r["peak_res_gib"] / r["total_gib"]
    print(f"{r['max_length']:>7} {r['cross_chunks']:>3} {r['chunk_len']:>6} "
          f"{str(r['use_memory']):>5} {r.get('accum_max', 0):>5} "
          f"{r['status']:>6} "
          f"{r['peak_alloc_gib']:>8.2f} {r['peak_res_gib']:>9.2f} "
          f"{pct:>5.0f} {r['step_s']:>7.1f}")
d = [r for r in rows if r["max_length"] == 8192 and r["cross_chunks"] == 16
     and r["use_memory"] and r.get("accum_max") == 512]
if d:
    fits = d[0]["status"] == "FITS"
    print("")
    print("VERDICT: config D (8192 x cc16) " +
          ("FITS — launch on D as locked." if fits else
           "DOES NOT FIT — trigger D-fallback (4096 x cc8, batch_size 32)."))
    print("Record this number in the sbatch header as the reason, either way.")
else:
    print("\n(no 8192 x cc16 memory row in the json — the decision row did not run)")
SUMMARY

if [ $BROKEN -ne 0 ]; then
    echo ""
    echo "NOTE: at least one geometry failed for a non-OOM reason; the table"
    echo "above is incomplete.  Do not decide the row length off it."
    exit 1
fi
