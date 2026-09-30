#!/bin/bash
#SBATCH --account=gts-aivanova7-lab
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64GB
#SBATCH --gres=gpu:H200:1
#SBATCH -t 0:30:00
#SBATCH -q inferno
#SBATCH -J znorm
#SBATCH -o logs/Report-%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=henryyjiang42@gmail.com

# ============================================================================
# THE ZNORM_TARGET GATE for pace/cortex_final_5b.sbatch.
#
#   sbatch pace/measure_znorm.sbatch
#
# COST: ~5 minutes of H200 (8 forwards, no backward, one model load).  The 30
# minute wall cap is queue headroom, not an estimate -- Slurm bills what you
# use, so this is ~0.1 GPU-hours.
#
# FIRST RUN OOMED (job 13764409, 48 s).  Fixed here and upstream; see the
# BATCH block below.  If it OOMs again, drop BATCH to 1 before touching
# CHUNKS or CHUNK_LEN -- those two are the RUN'S geometry and changing them
# measures ||E|| under a layout cortex-final does not train.
#
# WHAT IT MEASURES AND WHY IT IS A GATE
# -------------------------------------
# --cortex.latent_read_znorm_target rescales every carried Z row to a FIXED
# norm before the embeds splice, so Z and E enter the same concatenation at
# comparable scale.  The graft REQUIRES it with latent_read=embeds, because
# Z_end rows are ~10 against E's ~136-171: Z entering 17x below the field it
# competes with IS the measured s0 failure (flat to 4e-5 out to 45x ||s0||,
# turning on only at 175.8 vs ||E|| = 171, i.e. when it drowns E rather than
# being read).
#
# The number is E's own carried row norm, and it is LINEAGE-SPECIFIC:
#
#     171.0   B2
#     136.0   the b2 heal branch
#
# -- a 26% spread between two checkpoints of ONE lineage.  J4's single smoke
# failure was exactly this number inherited from a different branch.
# cortex-final descends from neither: it starts fresh from
# ckpts/olmo-retrofit-cortex, where nobody has measured it.  Launching on an
# inherited constant would put Z into the splice at the wrong strength for 5B
# tokens, invisibly, because the loss curve cannot show it.
#
# ON REAL TOKENS, NOT RANDOM IDS.  diag_dual_channel_walk's own header: a
# random-id walk measured loss 11.6-11.9 against ln(vocab) = 11.52, i.e. the
# model was seeing noise, "and every rank number in that table described
# noise".  E is a post-ln_f state and its norm is a property of what the model
# is actually computing, so this walks the corpus the run STARTS on -- the
# heal pack -- at the run's own geometry (8 chunks x 512).
#
# AFTERWARDS, WATCH IT.  The number is a launch-time constant but ||E|| moves:
# over J7's 2,000 steps it went 135.5 -> 144.1 (+6.3%), ending +6.0% from the
# 136.0 that run launched at, and cortex-final is ~19x longer.
# --cortex.diag_interval logs z_e_carried_norm every 50 steps and
# tools/watch_run.py FAILS past +-15%.  A settled norm is a calibration
# constant; a climbing one is an instability with a schedule.  If the band
# breaks, the natural re-targeting point is the corpus switch at step 11,444,
# because the arm already changes there -- and whatever you choose, RECORD IT,
# because changing the constant mid-chain changes the arm.
# ============================================================================

cd $SLURM_SUBMIT_DIR

module load anaconda3
module load cuda/12.1.1
conda activate cortex-retro

export SCRATCH=${SCRATCH:-$HOME/scratch}
export HF_HOME=$SCRATCH/hf_cache
export HF_HUB_OFFLINE=1
# MISSING ON THE FIRST RUN, and the OOM message itself asked for it.  Every
# other GPU sbatch in pace/ exports this; the walk chains 8 chunks, so its
# allocation pattern is exactly the fragmenting one it is for.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODEL=${MODEL:-ckpts/olmo-retrofit-cortex}
CKPT=${CKPT:-}                             # empty = the base, i.e. step 0
DATA=${DATA:-data/fineweb_edu_olmo_len4096}   # the corpus the run STARTS on
CHUNKS=${CHUNKS:-8}                        # the run's cross_chunks
CHUNK_LEN=${CHUNK_LEN:-512}                # 4096 / 8
T=${T:-8}                                  # the run's max_mean_rec
# 2, NOT 4.  The walk's own default, and prelaunch_final.sbatch's.  The first
# run of this script asked for 8 x 512 at batch 4 -- 4x the footprint
# prelaunch walks at (chunk_len 256, batch 2) -- and OOMed an H200 at
# 139.7 GiB in 48 seconds (job 13764409).  Two causes, both now fixed: the
# missing allocator setting above, and `walk()` building the chain's autograd
# graph even under --no_backward (it skipped only the .backward() call; the
# graph is now scoped to backward=True, and
# tests/test_dual_channel_walk.TestTheWalkDoesNotBuildAGraphItWillNotUse
# pins it).  batch 2 over 8 chunks still gives 2 x 64 = 128 carried row norms,
# far more than this needs.
BATCH=${BATCH:-2}
OUT=${OUT:-eval_results/final5b_znorm_base.json}

echo "=== ZNORM_TARGET measurement ==="
echo "    model:  $MODEL ${CKPT:+(checkpoint $CKPT)}"
echo "    data:   $DATA   (real tokens -- random ids measure noise)"
echo "    walk:   $CHUNKS chunks x $CHUNK_LEN tokens, T=$T, batch=$BATCH"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

[ -d "$MODEL" ] || { echo "ERROR: no such base: $MODEL"; exit 1; }
[ -d "$DATA" ]  || { echo "ERROR: no such pack: $DATA"; exit 1; }
if ! grep -q "iter_write(x, current_step)" "$MODEL/raven_modeling_minimal_cortex.py" 2>/dev/null; then
    echo "ERROR: $MODEL carries a modeling file without iter_write(x,"
    echo "       current_step) -- the Z write would never fire, and the carry"
    echo "       this measures would not be the one cortex-final builds."
    echo "       Refresh: python tools/prepare_cortex_checkpoint.py \\"
    echo "           --src $MODEL --dst $MODEL --variant olmo"
    exit 1
fi

# The graft is forced to CORTEX-FINAL's build, not the base dir's config
# (which carries no cortex flags at all).
#
# THE Z-SIDE READ FLAGS ARE OMITTED, AND THE FIRST REASON GIVEN HERE WAS
# WRONG.  It said E's norm "is written before any Z read happens".  It is not:
# the read enters at s0 / every loop iteration, and the E write is post-ln_f at
# the END, so the read is upstream of it.  The real reason is circularity --
# latent_read=embeds REQUIRES latent_read_znorm_target, which is the number
# being measured.
#
# What rescues it is a measurement, not an argument.  Chunk 0 runs with NO
# incoming carry and therefore no Z read at all; chunks 1-7 run with both.
# On job 13765065 those wrote 100.2 against 98.9-100.1 -- a 1.3% spread across
# the whole walk.  So the read's influence on ||E|| is inside 1.3%, an order
# of magnitude under the +-15% band this number is used against.  Read the
# per-chunk write norms the summary prints and re-check that, rather than
# trusting this paragraph.
python evals/diag_dual_channel_walk.py \
    --model_name "$MODEL" ${CKPT:+--checkpoint "$CKPT"} \
    --data "$DATA" \
    --chunks "$CHUNKS" --chunk_len "$CHUNK_LEN" --T "$T" --batch "$BATCH" \
    --dtype float32 --no_backward \
    --set use_memory=true --set memory_slots=0 --set prefix_memory=gated \
    --set accum_vecs=16 --set gate_slots=64 --set gate_route=ring \
    --set gate_norm=tanh --set gate_init=zero --set gate_fill=grow \
    --set latent_carry=true \
    --out "$OUT"
RC=$?
[ $RC -eq 0 ] || { echo "walk failed rc=$RC"; exit $RC; }

echo ""
python - "$OUT" <<'READ'
import json, sys
rec = json.load(open(sys.argv[1]))
c = rec["final_carry"]
e = c.get("e_row_norm")
rows = rec.get("rows") or []
writes = [r.get("e_write_norm") for r in rows if r.get("e_write_norm") is not None]
print("=" * 70)
print(f"  ZNORM_TARGET = {e:.1f}")
print("=" * 70)
print(f"  (carried E row norm, the block the next chunk actually READS)")
if writes:
    print(f"  per-chunk write norms: "
          + "  ".join(f"{w:.1f}" for w in writes))
    print(f"  spread across the walk: {min(writes):.1f} .. {max(writes):.1f} "
          f"({(max(writes)-min(writes))/max(min(writes),1e-9):+.1%})")
print()
print(f"  Launch with:")
print(f"    PHASE=heal ZNORM_TARGET={e:.1f} sbatch pace/cortex_final_5b.sbatch")
print()
print()
print(f"  EXPECT ~98 HERE, NOT 136 OR 171.  This walks the UNTRAINED")
print(f"  conversion -- job 13765065 measured 98.0 at mean NLL 10.27 against")
print(f"  ln(vocab) 11.52, i.e. the arrangement before any training.  136.0")
print(f"  and 171.0 are TRAINED anchors and do not apply at step 0:")
print(f"      ~98   this base, step 0                     (measured)")
print(f"      136   b2 heal branch -- same base, same corpus, same 1.5B heal")
print(f"      171   B2 at 5B")
print(f"  So E's norm is expected to CLIMB over the run, roughly along that")
print(f"  ladder, and a target pinned here falls out of band as it does.")
print(f"  That is the re-target question, not a bad measurement: see")
print(f"  ../final_5b_launch.md S1b and watch cortex/z_embed_ratio.")
print()
print(f"  A reading under ~60 or over ~200 is the one to distrust -- check the")
print(f"  pack arrived and that the graft built (the --set list above).")
READ
