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

MODEL=${MODEL:-ckpts/olmo-retrofit-cortex}
CKPT=${CKPT:-}                             # empty = the base, i.e. step 0
DATA=${DATA:-data/fineweb_edu_olmo_len4096}   # the corpus the run STARTS on
CHUNKS=${CHUNKS:-8}                        # the run's cross_chunks
CHUNK_LEN=${CHUNK_LEN:-512}                # 4096 / 8
T=${T:-8}                                  # the run's max_mean_rec
BATCH=${BATCH:-4}
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
# (which carries no cortex flags at all).  These mirror
# pace/cortex_final_5b.sbatch; the Z-side read flags are deliberately omitted
# because what is being measured is E's carried row norm, which is written
# before any Z read happens.
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
print(f"  Sanity: 171.0 was B2's and 136.0 the b2 heal branch's.  A number far")
print(f"  outside that range means the walk did not measure what it should --")
print(f"  check the pack arrived and that the graft built (the --set list).")
READ
