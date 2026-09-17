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
#   sbatch pace/red10_confirm.sbatch                    # both sections
#   SKIP_CONFIRM=1 sbatch pace/red10_confirm.sbatch     # width walks only
#   SKIP_WIDTH=1   sbatch pace/red10_confirm.sbatch     # the RED 10 check only
#
# THE CAUSE WAS FOUND IN THE CODE, NOT ON THE CLUSTER.  The modeling file does
# NOT shift labels -- `cross_entropy(logits.view(-1, V), labels.view(-1))` with
# the comment "assuming labels really are labels and not equal to input_ids" --
# and train.py:1812 shifts.  Three call sites passed the raw ids:
#
#     evals/diag_dual_channel_walk.py   walk()            labels=ids
#     tools/prelaunch_final.py:137      _chain_losses     labels=ids
#     tools/prelaunch_final.py:365      gate 4's score()  labels=last
#
# Predicting token t AT position t is impossible for a causal LM, so the loss
# pins to ln(vocab) = 11.5157 -- exactly the 11.4-11.97 both instruments
# reported in jobs whose training loss was 2.78.  Fixed via
# evals/model_utils.shift_labels.
#
# ==== RESULT 2026-09-17, job 13304811 -- RED 10 IS CLOSED ====================
#
#   walk, branch W=16:  mean NLL 3.8197   chance 11.5164   margin +7.6967
#   gate 4 donor:       real 4.5968  none 4.6187  donor_both 4.6674
#                       content delta (donor - real) = +0.0705
#
# The walk scores a MODEL now.  And gate 4's whole-carry swap is POSITIVE and
# agrees in sign with the influence horizon's +0.0595 at d=1, so the -0.1234
# that reordered the whole P2 program is retired and THE SIGN GAP IS CLOSED:
# there was never a coherent-donor-chain-vs-patchwork story, just a model at
# chance.  read_live_frac came back 0.534 at the arms' real operating point.
#
# Two things went wrong in the same job and both are recorded here rather than
# quietly fixed:
#   * THE PARENT WALK (W=32) OOMed at 139.78 GiB.  The walk backprops one summed
#     chain loss, so every chunk's graph is retained at once, and W=32 /
#     accum_max 256 / batch 2 does not fit.  The width contrast does not need
#     gradients at all -- it reads the final carry's rank statistics -- so the
#     WIDTH section below runs --no_backward at batch 1, and runs BOTH sides
#     that way so the two walks stay matched.
#   * THIS SCRIPT'S OWN SUMMARY PRINTED "STILL AT CHANCE" when the branch walk
#     had a +7.70 margin, because a MISSING file and an AT-CHANCE file were
#     folded into one flag.  Same failure class as everything else in this
#     project: a confident wrong verdict, not a loud failure.  The summary now
#     names each file's state separately and only calls chance on a walk that
#     actually ran.
#
# WHAT STAYS OPEN: the width STATISTIC.  PR is computed on p ~ s^2 and entropy
# on p ~ s, so PR << entropy is generic and their "disagreement in direction"
# was never evidence.  This job produces the numbers; it does not settle which
# one to quote.

cd $SLURM_SUBMIT_DIR

module load anaconda3
module load cuda/12.1.1
conda activate cortex-retro

export SCRATCH=${SCRATCH:-$HOME/scratch}
export HF_HOME=$SCRATCH/hf_cache
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

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
WIDTH_BATCH=${WIDTH_BATCH:-1}    # W=32 x accum_max 256 x batch 2 OOMs
STAMP=${STAMP:-$(date +%Y%m%d)}
OUT=${OUT:-eval_results/red10_confirm-$STAMP}
mkdir -p "$OUT"

for D in "$BRANCH" "$PARENT"; do
    [ -d "$D" ] || { echo "ERROR: no such checkpoint dir: $D"; exit 1; }
done
[ -d "$DATA" ] || { echo "ERROR: no such pack: $DATA"; exit 1; }

SETS_W16="--set use_memory=true --set memory_slots=0 --set prefix_memory=accum --set accum_vecs=16 --set accum_max=128"
SETS_W32="--set use_memory=true --set memory_slots=0 --set prefix_memory=accum --set accum_vecs=32 --set accum_max=256"

echo "=== RED 10 confirmation | $CHUNKS x $CHUNK_LEN tokens | T=$T ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "    branch (W=16): $BRANCH"
echo "    parent (W=32): $PARENT"
echo "    data:          $DATA   <- REAL PROSE.  Random ids score at chance too"
echo "                   (that was RED 3), so the pack line is load-bearing."
echo ""

RC_CONFIRM=0
RC_WIDTH=0

# ---- section 1: does the branch walk score a model, and do the gates? -------
if [ -z "$SKIP_CONFIRM" ]; then
    echo "######## walk: branch W=16 (with backward, the full table) ########"
    python evals/diag_dual_channel_walk.py \
        --model_name "$MODEL" --checkpoint "$BRANCH" --data "$DATA" \
        $SETS_W16 \
        --chunks "$CHUNKS" --chunk_len "$CHUNK_LEN" --batch "$BATCH" --T "$T" \
        --out "$OUT/walk_w16.json"
    RC_CONFIRM=$?

    # --trained_depth is NOT optional: without it gate 3 anchors on
    # config.mean_recurrence (32) and reports read_live_frac 0.0195 for an arm
    # that ran at 0.546.  That was RED 8.
    echo ""
    echo "######## prelaunch gates (gate 4 is the whole-carry donor swap) ########"
    python tools/prelaunch_final.py \
        --model_name "$MODEL" --checkpoint "$BRANCH" --data "$DATA" \
        $SETS_W16 \
        --chunks "$CHUNKS" --chunk_len "$CHUNK_LEN" --batch "$BATCH" \
        --T "$T" --trained_depth "$T" \
        --out "$OUT/gates_w16.json"
    [ $? -eq 0 ] || RC_CONFIRM=1
fi

# ---- section 2: the width pair, matched and cheap ---------------------------
# --no_backward on BOTH sides.  The contrast reads final_carry's rank
# statistics; gradients are what made the W=32 side OOM and they are not part
# of the measurement.  Matching matters more than either setting: a rank
# compared across two different walk configurations is not a contrast.
if [ -z "$SKIP_WIDTH" ]; then
    echo ""
    echo "######## width walk: branch W=16 (no backward, batch $WIDTH_BATCH) ########"
    python evals/diag_dual_channel_walk.py \
        --model_name "$MODEL" --checkpoint "$BRANCH" --data "$DATA" \
        $SETS_W16 --no_backward \
        --chunks "$CHUNKS" --chunk_len "$CHUNK_LEN" --batch "$WIDTH_BATCH" --T "$T" \
        --out "$OUT/width_w16.json"
    RC_WIDTH=$?

    echo ""
    echo "######## width walk: parent W=32 (no backward, batch $WIDTH_BATCH) ########"
    python evals/diag_dual_channel_walk.py \
        --model_name "$MODEL" --checkpoint "$PARENT" --data "$DATA" \
        $SETS_W32 --no_backward \
        --chunks "$CHUNKS" --chunk_len "$CHUNK_LEN" --batch "$WIDTH_BATCH" --T "$T" \
        --out "$OUT/width_w32.json"
    [ $? -eq 0 ] || RC_WIDTH=1

    echo ""
    echo "######## width contrast: parent W=32 vs branch W=16 ########"
    if [ $RC_WIDTH -eq 0 ]; then
        python tools/compare_width.py \
            --parent "$OUT/width_w32.json" --branch "$OUT/width_w16.json" \
            --out "$OUT/width.json"
        echo "(INVALID here is a REFUSAL, not a verdict -- and PR vs entropy is not"
        echo " a disagreement until both are computed on the same power of s.)"
    else
        echo "SKIPPED: a width walk failed, so there is nothing comparable to compare."
    fi
fi

echo ""
echo "=== summary ==="
python - "$OUT" <<'SUMMARY'
import json, os, sys
out = sys.argv[1]
ran, at_chance, missing = [], [], []
for name in ("walk_w16.json", "width_w16.json", "width_w32.json"):
    p = os.path.join(out, name)
    if not os.path.exists(p):
        # MISSING IS NOT AT CHANCE.  Folding the two together is what made this
        # script report "STILL AT CHANCE" over a walk with a +7.70 margin.
        missing.append(name)
        print(f"{name:>16}  not produced this run")
        continue
    h = (json.load(open(p)).get("health") or {})
    if h.get("margin") is None:
        missing.append(name)
        print(f"{name:>16}  no health block -- stale code on this node?")
        continue
    flag = "AT CHANCE" if h["at_chance"] else "ok"
    ran.append(name)
    if h["at_chance"]:
        at_chance.append(name)
    print(f"{name:>16}  mean NLL {h['mean_nll']:7.4f}  chance {h['chance']:7.4f}"
          f"  margin {h['margin']:+7.4f}  {flag}")
print("")
if at_chance:
    print("STILL AT CHANCE: " + ", ".join(at_chance))
    print("The label shift was not the whole story for these.  Next suspect is")
    print("inner.train() (diag_dual_channel_walk.py:472) against the horizon's")
    print(".eval(), then chunk_len.  Do NOT read their tables.")
elif ran:
    print(f"Every walk that ran scores a model ({len(ran)} of 3).  RED 10 stays closed.")
    if missing:
        print("Not produced this run (skipped or failed -- read the log above, "
              "and do not\nread this as a chance-level result): "
              + ", ".join(missing))
else:
    print("No walk produced a health block at all -- nothing was measured.")
SUMMARY

[ $RC_CONFIRM -eq 0 ] && [ $RC_WIDTH -eq 0 ]
