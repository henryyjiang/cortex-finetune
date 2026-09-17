#!/bin/bash
#SBATCH --account=gts-aivanova7-lab
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H200:1
#SBATCH --mem=64GB
#SBATCH -t 1:00:00
#SBATCH -q inferno
#SBATCH -J red10
#SBATCH -o logs/Report-%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=henryyjiang42@gmail.com

# PASS 1, JOB 1 -- CONFIRM THE RED 10 FIX, AND UNBLOCK THE WIDTH CONTRAST.
#
#   sbatch pace/red10_confirm.sbatch
#
# THE CAUSE WAS FOUND IN THE CODE, NOT ON THE CLUSTER, so this job confirms a
# diagnosis rather than hunting for one.  The modeling file does NOT shift
# labels -- `cross_entropy(logits.view(-1, V), labels.view(-1))` with the
# comment "assuming labels really are labels and not equal to input_ids" -- and
# train.py builds them shifted (train.py:1812).  Two instruments passed the
# UNSHIFTED ids:
#
#     evals/diag_dual_channel_walk.py   walk()            labels=ids
#     tools/prelaunch_final.py:137      _chain_losses     labels=ids
#     tools/prelaunch_final.py:365      gate 4's score()  labels=last
#
# Asking a causal LM to predict token t AT position t is impossible by
# construction, so the loss pins to ln(vocab) = 11.5157 -- which is exactly the
# 11.4-11.97 both instruments reported, in jobs whose training loss was 2.78.
# It also explains why chunk 0 was at chance with carry=None (nothing to do
# with the prefix) and why eval_influence_horizon was clean (it computes its own
# shifted CE).  Fixed via evals/model_utils.shift_labels; regression tests in
# tests/test_dual_channel_walk.py and tests/test_prelaunch_final.py.
#
# THE READING, PRE-REGISTERED:
#   mean NLL ~3.2-3.7, margin ~+8      -> RED 10 closed.  The walk, gate 4 and
#                                         the width contrast are all live again.
#   mean NLL ~11.5, margin ~0          -> the shift was NOT the whole story.
#                                         The walk now prints AT CHANCE four
#                                         times; do not read anything below it.
#                                         Next suspects, in order: inner.train()
#                                         vs .eval(), then chunk_len 256 vs 512.
#
# WHY THE TWO WALKS AND NOT ONE.  P2.2 (the width contrast) needs a walk at
# W=32 and a walk at W=16 off the SAME 91552 checkpoint, and its INVALID verdict
# was never a statistics problem -- rank_stats does centre, and both walks were
# reading a model that predicts nothing.  Re-running both here costs about four
# extra minutes and closes P2.2's dependency in the same job.
#
# The width statistic itself is still open and this job does NOT settle it: PR
# is computed on p ~ s^2 and entropy on p ~ s, so PR << entropy is generic and
# their "disagreement in direction" was never evidence.  Put both on s^2 before
# reading them against each other.  compare_width runs here to produce the
# numbers, not the verdict.

cd $SLURM_SUBMIT_DIR

module load anaconda3
module load cuda/12.1.1
conda activate cortex-retro

export SCRATCH=${SCRATCH:-$HOME/scratch}
export HF_HOME=$SCRATCH/hf_cache
export HF_HUB_OFFLINE=1

MODEL=${MODEL:-ckpts/olmo-retrofit-cortex}
DATA=${DATA:-data/pg19_olmo_val_len4096}
BRANCH=${BRANCH:-cortex-retrofit/retro-b2-heal/checkpoint_91552_w16}
PARENT=${PARENT:-cortex-retrofit/retro-b2-heal/checkpoint_91552}
CHUNKS=${CHUNKS:-8}
CHUNK_LEN=${CHUNK_LEN:-512}      # the geometry the arms trained at, not the
                                 # walk's 256 default
T=${T:-8}                        # NOT config.mean_recurrence (32 on every
                                 # B2-family checkpoint; no arm ran there)
BATCH=${BATCH:-2}
STAMP=${STAMP:-$(date +%Y%m%d)}
OUT=${OUT:-eval_results/red10_confirm-$STAMP}
mkdir -p "$OUT"

for D in "$BRANCH" "$PARENT"; do
    [ -d "$D" ] || { echo "ERROR: no such checkpoint dir: $D"; exit 1; }
done
[ -d "$DATA" ] || { echo "ERROR: no such pack: $DATA"; exit 1; }

echo "=== RED 10 confirmation | $CHUNKS x $CHUNK_LEN tokens | T=$T ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "    branch (W=16): $BRANCH"
echo "    parent (W=32): $PARENT"
echo "    data:          $DATA   <- REAL PROSE.  Random ids score at chance too"
echo "                   (that was RED 3), so the pack line is load-bearing."
echo ""

# --- the branch, W=16, accum ------------------------------------------------
echo "######## walk: branch W=16 ########"
python evals/diag_dual_channel_walk.py \
    --model_name "$MODEL" --checkpoint "$BRANCH" --data "$DATA" \
    --set use_memory=true --set memory_slots=0 \
    --set prefix_memory=accum --set accum_vecs=16 --set accum_max=128 \
    --chunks "$CHUNKS" --chunk_len "$CHUNK_LEN" --batch "$BATCH" --T "$T" \
    --out "$OUT/walk_w16.json"
RC_W16=$?

# --- the parent, W=32, accum ------------------------------------------------
echo ""
echo "######## walk: parent W=32 ########"
python evals/diag_dual_channel_walk.py \
    --model_name "$MODEL" --checkpoint "$PARENT" --data "$DATA" \
    --set use_memory=true --set memory_slots=0 \
    --set prefix_memory=accum --set accum_vecs=32 --set accum_max=256 \
    --chunks "$CHUNKS" --chunk_len "$CHUNK_LEN" --batch "$BATCH" --T "$T" \
    --out "$OUT/walk_w32.json"
RC_W32=$?

# --- the four gates, on the branch -----------------------------------------
# --trained_depth is NOT optional: without it gate 3 anchors on
# config.mean_recurrence (32) and reports read_live_frac 0.0195 for an arm that
# ran at 0.546.  That was RED 8.
echo ""
echo "######## prelaunch gates (gate 4 is the whole-carry donor swap) ########"
python tools/prelaunch_final.py \
    --model_name "$MODEL" --checkpoint "$BRANCH" --data "$DATA" \
    --set use_memory=true --set memory_slots=0 \
    --set prefix_memory=accum --set accum_vecs=16 --set accum_max=128 \
    --chunks "$CHUNKS" --chunk_len "$CHUNK_LEN" --batch "$BATCH" \
    --T "$T" --trained_depth "$T" \
    --out "$OUT/gates_w16.json"
RC_GATE=$?

# --- the width contrast, on two walks that are no longer noise --------------
echo ""
echo "######## width contrast: parent W=32 vs branch W=16 ########"
if [ $RC_W16 -eq 0 ] && [ $RC_W32 -eq 0 ]; then
    python tools/compare_width.py \
        --parent "$OUT/walk_w32.json" --branch "$OUT/walk_w16.json" \
        --out "$OUT/width.json"
    echo "(INVALID here is a REFUSAL, not a verdict -- and PR vs entropy is not"
    echo " a disagreement until both are computed on the same power of s.)"
else
    echo "SKIPPED: a walk failed, so there is nothing comparable to compare."
fi

echo ""
echo "=== summary ==="
python - "$OUT" <<'SUMMARY'
import json, os, sys
out = sys.argv[1]
bad = 0
for name in ("walk_w16.json", "walk_w32.json"):
    p = os.path.join(out, name)
    if not os.path.exists(p):
        print(f"{name:>16}  MISSING"); bad = 1; continue
    h = (json.load(open(p)).get("health") or {})
    if h.get("margin") is None:
        print(f"{name:>16}  no health block -- old code?"); bad = 1; continue
    flag = "AT CHANCE" if h["at_chance"] else "ok"
    print(f"{name:>16}  mean NLL {h['mean_nll']:7.4f}  chance {h['chance']:7.4f}"
          f"  margin {h['margin']:+7.4f}  {flag}")
    bad |= bool(h["at_chance"])
print("")
print("RED 10 CLOSED -- the walk and gate 4 are live instruments again."
      if not bad else
      "STILL AT CHANCE.  The label shift was not the whole story: next suspect\n"
      "is inner.train() (diag_dual_channel_walk.py:472) against the horizon's\n"
      ".eval(), then chunk_len.  Do NOT read the tables above.")
SUMMARY

[ $RC_W16 -eq 0 ] && [ $RC_W32 -eq 0 ] && [ $RC_GATE -eq 0 ]
