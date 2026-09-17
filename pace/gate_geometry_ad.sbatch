#!/bin/bash
#SBATCH --account=gts-aivanova7-lab
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H200:1
#SBATCH --mem=64GB
#SBATCH -t 1:00:00
#SBATCH -q inferno
#SBATCH -J gate_ad
#SBATCH -o logs/Report-%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=henryyjiang42@gmail.com

# PASS 1, JOB 3 -- THE BUFFER-LEVEL A(d), i.e. WHAT THE d=4 SPIKE IS.
#
#   sbatch pace/gate_geometry_ad.sbatch
#
# WHAT IS OPEN.  P2.1 measured, at the LOSS level on the trained arms, that both
# gated arms spike ~70x at exactly d=4 under norm-matched noise (a3 1.1682,
# a3z 1.1302, against ~0.02 at d=3) while being ~20x MORE robust than append
# everywhere else.  K/W = 64/16 = 4, so d=4 is one lap: chunks N and N-4 occupy
# the SAME 16 physical slots.  That is the obvious explanation and it has never
# been checked at the buffer level.
#
# A(d) = || s_N(write_{N-d} := donor) - s_N || / || one chunk's write ||
#
# is the displacement in units of one write, forward-only, no training, and the
# retention model it tests is already validated twice on the B2 checkpoint:
#
#     F(d) = ig * fg ** floor(d / (K/W))      ig=0.500, fg=0.731 at gate_init=zero
#
# predicted 0.195 measured 0.195, predicted 0.143 measured 0.143.
#
# THE READING, PRE-REGISTERED.  F(d) says retention STEPS DOWN at the lap, while
# the loss-level measurement shows influence SPIKING UP there.  Those are not
# the same claim, and which one holds decides whether this is geometry or a
# defect:
#
#   A(d) is a STAIRCASE with its step at d=4, no spike
#       -> the buffer is behaving exactly as designed and the loss spike is a
#          READ effect: noise one lap back survives the gate into the freshest
#          row, which is the row doing the most work.  Geometry, not a defect.
#          Publishable as the ring's signature once A(d) is in the figure.
#   A(d) SPIKES at d=4 too
#       -> the mixing is wrong, not just the reading of it.  Check the slot
#          mapping (depth_rows(j) vs rows_touched) before anything else, then
#          forget_bias: fg=0.731 at init retains 73% of a row per lap, which is
#          what would carry an injected row forward intact.
#   A(d) is FLAT to the FIFO edge
#       -> the cell being scored is an accum row, not the ring.  Read the
#          geometry table it prints and re-check --cell.
#
# DESIGN CONSEQUENCE, which is why this runs in pass 1 rather than at writeup
# time: if the lap is contaminating the newest row, then K/W (laps of
# separation) and gate_init/forget_bias are the levers, and both are full-run
# decisions that the P2.3 cells cannot re-decide afterwards.
#
# NOTE ON WHAT THIS DOES AND DOES NOT USE.  build_cells constructs FRESH buffers
# at gate_init="zero" and replays writes taken from the REAL checkpoint's accum
# carry.  So this is the geometry at init -- the same construction that
# validated F(d) -- and NOT the trained a3 gate.  A trained-gate A(d) needs the
# P2.3 cell checkpoints and belongs in pass 2.

cd $SLURM_SUBMIT_DIR

module load anaconda3
module load cuda/12.1.1
conda activate cortex-retro

export SCRATCH=${SCRATCH:-$HOME/scratch}
export HF_HOME=$SCRATCH/hf_cache
export HF_HUB_OFFLINE=1

MODEL=${MODEL:-ckpts/olmo-retrofit-cortex}
CKPT=${CKPT:-cortex-retrofit/retro-b2-heal/checkpoint_91552_w16}
# Real prose, two DIFFERENT documents: the donor swap is only a content control
# if the donor is content.  Repo-root planning docs, the same source
# diag_depth_band.sbatch uses.
TEXT=${TEXT:-../cortex_next_phase_framework.md}
DONOR=${DONOR:-../control_launch_handoff.md}
CHUNKS=${CHUNKS:-24}        # 6 laps at K/W=4 -- enough to see the staircase
SEQ_LEN=${SEQ_LEN:-512}
BATCH=${BATCH:-1}           # 24 x 512 x 1 = 12,288 tokens per document
STEPS=${STEPS:-8}           # T, the arms' trained depth
STAMP=${STAMP:-$(date +%Y%m%d)}
OUT=${OUT:-eval_results/gate_ad-$STAMP}
mkdir -p "$OUT"

# The three cells that answer the question and nothing else.  The full list is
# ten and each one is a replay over 24 chunks; the locked cell, its cost-matched
# accum control and the dense route are what the d=4 reading needs.
CELLS=${CELLS:-"gated W16 K64 ring|accum W16 K64|gated W16 K64 mix"}

for F in "$TEXT" "$DONOR"; do
    [ -f "$F" ] || { echo "ERROR: no prose at $F"; exit 1; }
done
[ -d "$CKPT" ] || { echo "ERROR: no such checkpoint dir: $CKPT"; exit 1; }

echo "=== gated-ring A(d) | $CHUNKS chunks x $SEQ_LEN | T=$STEPS ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "    checkpoint: $CKPT"
echo "    text:       $TEXT"
echo "    donor:      $DONOR   <- a DIFFERENT document, or A(d) measures nothing"
echo ""

CELL_ARGS=()
IFS='|' read -ra CL <<< "$CELLS"
for c in "${CL[@]}"; do CELL_ARGS+=(--cell "$c"); done

python evals/diag_gate_geometry.py \
    --model_name "$MODEL" --checkpoint "$CKPT" \
    --text_file "$TEXT" --donor_file "$DONOR" \
    --chunks "$CHUNKS" --seq_len "$SEQ_LEN" --batch "$BATCH" \
    --steps "$STEPS" --dtype float32 \
    --tape "$OUT/writes.pt" \
    "${CELL_ARGS[@]}" \
    --out "$OUT/gate_ad.json"
RC=$?

echo ""
if [ $RC -eq 0 ]; then
    echo "A(d) -> $OUT/gate_ad.json"
    echo ""
    echo "Read the A(d) row for 'gated W16 K64 ring' against F(d) = 0.5 *"
    echo "0.731^floor(d/4): 0.500 at d=0-3, 0.366 at d=4-7, 0.267 at d=8-11."
    echo "A STEP at d=4 is the design.  A SPIKE at d=4 is not, and it would make"
    echo "the ring's lap a defect rather than its signature."
else
    echo "FAILED rc=$RC -- and note that a --cell typo exits with the label list."
fi
exit $RC
