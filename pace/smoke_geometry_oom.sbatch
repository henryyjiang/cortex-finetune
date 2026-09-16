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
#   P1=1 GEOMETRIES= sbatch pace/smoke_geometry_oom.sbatch   # A1 + A3' ONLY
#   P1=1 P1_Z=1 GEOMETRIES= sbatch pace/smoke_geometry_oom.sbatch  # + A2/A3z
#
# P1_Z=1 adds the two Z arms.  Their memory has NEVER been priced -- every
# number in this gate's history is from an E-only run -- and the second gate
# is ~33.5M more parameters with optimizer state, which is not a rounding
# error.  Run it before any Z cell.
#
# P1=1 (2026-09-16) PRICES THE TWO ARMS THIS GATE WAS NOT WRITTEN FOR.  Both are
# strictly smaller than B2 in sequence terms -- 656 and 592 packed columns
# against B2's 800 -- so on columns alone this is a formality.  It is run anyway
# for two reasons that columns do not cover:
#
#   1. The gate adds ~16.8M buffer parameters, and the optimizer holds TWO
#      moments for each.  That is the one axis on which A3' is the LARGER of the
#      pair, and config D died on a measured OOM after a similar "obviously
#      fits" argument.
#   2. A gated buffer runs a DIFFERENT truncated-BPTT rule: train.py
#      slice-detaches write-once accum rows but detaches the whole gated state
#      every carry_grad_chunks chunks, so the gated arm keeps every chunk since
#      the last detach alive AT ONCE where accum frees them a block at a time.
#      The finding this script already recorded -- that the NUMBER OF RETAINED
#      CHUNK GRAPHS is a first-order cost, 4096 x cc16 amax 128 OOMing with only
#      17% more column-tokens than a geometry that fits -- is exactly the axis
#      that rule moves.  A column-count estimate will under-predict it.
#
# Set GEOMETRIES= (empty) with P1=1 to skip the config-D rows, which are settled.
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

# max_length : cross_chunks : batch_size [: accum_max]
# batch_size only scales the wall-clock projection (pure gradient accumulation,
# no memory cost); accum_max is optional and defaults to $ACCUM_MAX, i.e. to
# B2's rule.  Ascending, so the fit/OOM boundary is located before the stretch.
#
# RESULT 2026-09-14, jobs 13201945 / 13202177 / 13202515, H200 138.7 GiB:
#   4096  x cc8  amax 256   FITS  115.44 GiB reserved (83%), 1.2 s/micro-step
#   4096  x cc16 amax 512   OOM
#   4096  x cc16 amax 128   OOM
#   8192  x cc16 amax 512   OOM
#   8192  x cc16 amax 128   OOM
#   16384 x cc16 amax 512   OOM
#
# Calibration: B2 ran 4096 x cc8 for real and peaked at 119.76 GiB against this
# script's 115.44, so the gate is trustworthy to ~4% and the OOMs are real.
#
# CONFIG D IS DEAD, and so is every attempt to buy chain depth 16 on one H200:
# not at 8,192 tokens, not at 4,096 with 256-token chunks, and not with the
# buffer capped at 128.  D-FALLBACK IS LOCKED: 4096 x cc8, batch_size 32,
# 512-token chunks, 131,072 tok/step, MAX_STEPS 38,147.  The existing
# data/fineweb_edu_olmo_len4096 pack is reusable — no FineWeb-Edu rebuild.
#
# The finding worth keeping: 4096 x cc16 amax 128 has only ~17% more summed
# column-tokens than the geometry that FITS (5,824 vs 4,992) and still OOMs, so
# the NUMBER OF RETAINED CHUNK GRAPHS is a first-order cost in its own right —
# chunk count is not a free axis, and a width-based estimate will under-predict.
#
# Not priced, deliberately: mean_backprop_depth 4 would halve the retained
# recurrence activations and might let cc16 fit, but it changes the TBPTT window
# for BOTH arms, deviates from the one recipe known to convert this model, and
# would buy a 256-token chunk geometry that no ceiling probe has ever measured.
# Two speculative changes stacked to rescue a third.  Don't.
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
    IFS=':' read -r LEN CC BS AMAX <<< "$G"
    AMAX=${AMAX:-$ACCUM_MAX}          # 4th field is optional; 0 = B2's rule
    echo ""
    echo "######## ${LEN} x cc${CC} (bs ${BS}, accum_max ${AMAX}) ########"
    python tools/smoke_geometry_oom.py \
        --model_name "$MODEL" \
        --max_length "$LEN" \
        --cross_chunks "$CC" \
        --batch_size "$BS" \
        --max_steps "$MAX_STEPS" \
        --micro_batch_size 1 \
        --mean_recurrence "$MAX_MEAN_REC" \
        --accum_vecs "$ACCUM_VECS" \
        --accum_max "$AMAX" \
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

if [ -n "$P1" ]; then
    # A1 and A3' at the locked recipe: 4096 x cc8 (512-token chunks), bs 32.
    # Only the buffer moves between the two rows, which is the whole point --
    # they share the row length, the chain, the recurrence and the corpus.
    P1_LEN=${P1_LEN:-4096}; P1_CC=${P1_CC:-8}; P1_BS=${P1_BS:-32}
    P1_W=${P1_W:-16}                      # sliced summary_emb width, both arms
    P1_GRAD=${P1_GRAD:-4}                 # 4 of 8, B2's 50% convention
    echo ""
    echo "######## A1: accum, W ${P1_W}, accum_max 128 (${P1_LEN} x cc${P1_CC}) ########"
    python tools/smoke_geometry_oom.py \
        --model_name "$MODEL" \
        --max_length "$P1_LEN" --cross_chunks "$P1_CC" --batch_size "$P1_BS" \
        --max_steps "$MAX_STEPS" --micro_batch_size 1 \
        --mean_recurrence "$MAX_MEAN_REC" \
        --prefix_memory accum --accum_vecs "$P1_W" --accum_max 128 \
        --carry_grad_chunks "$P1_GRAD" \
        --use_memory true --with_optimizer true \
        --json_out "$JSON" || BROKEN=1
    echo ""
    echo "######## A3': gated ring, W ${P1_W}, K 64 (${P1_LEN} x cc${P1_CC}) ########"
    python tools/smoke_geometry_oom.py \
        --model_name "$MODEL" \
        --max_length "$P1_LEN" --cross_chunks "$P1_CC" --batch_size "$P1_BS" \
        --max_steps "$MAX_STEPS" --micro_batch_size 1 \
        --mean_recurrence "$MAX_MEAN_REC" \
        --prefix_memory gated --accum_vecs "$P1_W" \
        --gate_slots 64 --gate_route ring --gate_norm tanh \
        --gate_init zero --gate_fill grow \
        --carry_grad_chunks "$P1_GRAD" \
        --use_memory true --with_optimizer true \
        --json_out "$JSON" || BROKEN=1
    if [ -n "$P1_Z" ]; then
        # THE Z ARMS.  They were never priced: every number in this gate's
        # history is from an E-only run.  The carry widening to 2D is small and
        # the trajectory tape is ~1 MB, but the SECOND GATE is two more
        # [D, 2D] Linears -- about 33.5M parameters WITH optimizer state -- and
        # a Z arm priced on an E-only row was not priced.
        echo ""
        echo "######## A2: accum + Z, W ${P1_W}, accum_max 128 ########"
        python tools/smoke_geometry_oom.py \
            --model_name "$MODEL" \
            --max_length "$P1_LEN" --cross_chunks "$P1_CC" --batch_size "$P1_BS" \
            --max_steps "$MAX_STEPS" --micro_batch_size 1 \
            --mean_recurrence "$MAX_MEAN_REC" \
            --prefix_memory accum --accum_vecs "$P1_W" --accum_max 128 \
            --latent_carry \
            --carry_grad_chunks "$P1_GRAD" \
            --use_memory true --with_optimizer true \
            --json_out "$JSON" || BROKEN=1
        echo ""
        echo "######## A3z: gated ring + Z, W ${P1_W}, K 64 ########"
        python tools/smoke_geometry_oom.py \
            --model_name "$MODEL" \
            --max_length "$P1_LEN" --cross_chunks "$P1_CC" --batch_size "$P1_BS" \
            --max_steps "$MAX_STEPS" --micro_batch_size 1 \
            --mean_recurrence "$MAX_MEAN_REC" \
            --prefix_memory gated --accum_vecs "$P1_W" \
            --gate_slots 64 --gate_route ring --gate_norm tanh \
            --gate_init zero --gate_fill grow \
            --latent_carry \
            --carry_grad_chunks "$P1_GRAD" \
            --use_memory true --with_optimizer true \
            --json_out "$JSON" || BROKEN=1
    fi
fi

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
    LEN=${CONTROL_LEN:-4096}; CC=${CONTROL_CC:-8}; CBS=${CONTROL_BS:-32}
    echo ""
    echo "######## control: ${LEN} x cc${CC} (bs ${CBS}), no memory ########"
    python tools/smoke_geometry_oom.py \
        --model_name "$MODEL" \
        --max_length "$LEN" --cross_chunks "$CC" --batch_size "$CBS" \
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
print(f"{'row':>7} {'cc':>3} {'chunk':>6} {'buffer':>6} {'W':>3} {'cols':>5} "
      f"{'status':>6} {'alloc':>8} {'reserved':>9} {'%dev':>5} {'s/step':>7}")
for r in rows:
    pct = 100 * r["peak_res_gib"] / r["total_gib"]
    # `cols` is the read block: accum_max for a capped accum, gate_slots for a
    # gated ring.  Printing accum_max for both would report 0 for every gated
    # row and quietly suggest the gated arm carries nothing.
    kind = r.get("buffer", "accum" if r["use_memory"] else "none")
    cols = r.get("gate_slots", 0) if kind == "gated" else r.get("accum_max", 0)
    print(f"{r['max_length']:>7} {r['cross_chunks']:>3} {r['chunk_len']:>6} "
          f"{kind:>6} {r.get('accum_vecs', 0):>3} {cols:>5} "
          f"{r['status']:>6} "
          f"{r['peak_alloc_gib']:>8.2f} {r['peak_res_gib']:>9.2f} "
          f"{pct:>5.0f} {r['step_s']:>7.1f}")
p1 = [r for r in rows if r.get("buffer") in ("accum", "gated")
      and r["cross_chunks"] == 8 and r.get("accum_vecs") == 16]
if len(p1) == 2:
    print("")
    if all(r["status"] == "FITS" for r in p1):
        a = {r["buffer"]: r["peak_res_gib"] for r in p1}
        print(f"P1: A1 (accum) {a.get('accum', 0):.2f} GiB, A3' (gated) "
              f"{a.get('gated', 0):.2f} GiB -- both FIT, launch both.")
        print("Record both numbers in pace/p1_arms.sbatch's header either way.")
    else:
        bad = [r["buffer"] for r in p1 if r["status"] != "FITS"]
        print(f"P1: {bad} DOES NOT FIT.  Do NOT launch it.  A gated OOM is the "
              f"optimizer\nstate on ~16.8M buffer parameters plus the "
              f"whole-state detach window -- lower\ncarry_grad_chunks before "
              f"lowering gate_slots, since K is what the arm is testing.")
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
