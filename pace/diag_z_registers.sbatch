#!/bin/bash
#SBATCH --account=gts-aivanova7-lab
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H200:1
#SBATCH --mem=64GB
#SBATCH -t 1:00:00
#SBATCH -q inferno
#SBATCH -J z3_registers
#SBATCH -o logs/Report-%A_%a.out
#SBATCH --array=0-3
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=henryyjiang42@gmail.com

# D3, THE REGISTER PROBE -- j4_handoff.md step 1.  Forward-only, one trained
# checkpoint per task.  Does the carried Z hold the 16 register values at all?
# E is the positive control.  Decides what J4 WRITES.
#
#   sbatch pace/diag_z_registers.sbatch
#     [0] j3 real            Z_write = scratch (m_T at the last 64 tokens)
#     [1] j1 real            Z_write = tokens  (s_T at the last 64 tokens)
#     [2] j1 noread          tokens, never read -- s_T with no read pressure
#     [3] j3 edrop0.25-real  scratch, trained with E blanked on 25% of rows
# Every task also probes Z_tok and Z_end (s_T at the SUMMARY columns, the
# latent twin of E) and, from chunk 4, the full ring's E and Z halves.
#
# LOOK FOR, per task:  "[z3] ... answer_dep checked" (the register replay agrees
# with the generator), then the PROBE HEALTH line (E must decode), then READING.
# Cross-checkpoint table, after all four:
#   python evals/diag_z_registers.py --summarize eval_results/z3_registers
#
# PROBE (b), j4_prereg.md S9.1's redundancy probe -- the one that bounds how a
# J4 PASS is read.  Same four checkpoints, same capture, one extra flag:
#   INCREMENTAL=1 sbatch pace/diag_z_registers.sbatch
# It reports the INCREMENT (registered), resid_frac and E_pca beside D3's table.
#
# PROBE (c), j4_prereg.md S9.1's heal parent -- the clean NO-J-TRAINING baseline
# for Z_end, the one array task that says whether Z_end's register content (and
# probe (b)'s ~1% increment) is anything J training created:
#   INCREMENTAL=1 sbatch --array=4 pace/diag_z_registers.sbatch
# It probes cortex-retrofit/retro-b2-heal/checkpoint_91552_w16 -- the SLICED w16
# parent every J arm branched from (pace/j1_joint.sbatch BRANCH_PATH), not the
# W=32 checkpoint_91552 beside it -- and lands as a fifth row in --summarize.
#
# The graft is built with EXACTLY the read-out's flags (pace/j1_readout.sbatch):
# the probe reads the writes the trained model makes, so the read that shapes
# the loop state has to be the trained one -- gate_lr_mult included.

cd $SLURM_SUBMIT_DIR

module load anaconda3
module load cuda/12.1.1
conda activate cortex-retro

export SCRATCH=${SCRATCH:-$HOME/scratch}
export HF_HOME=$SCRATCH/hf_cache
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODEL=${MODEL:-ckpts/olmo-retrofit-cortex}
OUT_ROOT=${OUT_ROOT:-cortex-retrofit}
DATA=${DATA:-data/carry_task_r16_len4096_val}
STEP=${STEP:-93552}
MAX_EXAMPLES=${MAX_EXAMPLES:-300}
READ_HEADS=${READ_HEADS:-8}
ZNORM_TARGET=${ZNORM_TARGET:-3.0}
TOK_POOL=${TOK_POOL:-4}
GATE_LR_MULT=${GATE_LR_MULT:-0.1}      # J3's trained value (pace/j3_joint.sbatch)
SCRATCH_FB=${SCRATCH_FB:-1.0}
BOOT=${BOOT:-2000}
# Probe (b), the redundancy probe (j4_prereg.md S4.6/S9.1).  INCREMENTAL=1 adds
# [E, Z_end] vs E alone (paired) and the projected residual, on the SAME capture
# -- no extra forward pass, a few CPU-minutes of probing.  Off by default so a
# D3 re-run reproduces the committed numbers exactly.
INCREMENTAL=${INCREMENTAL:-}
INC_ARGS=""
[ -n "$INCREMENTAL" ] && INC_ARGS="--incremental"

TASKS=(
    "j3 scratch real"
    "j1 tokens real"
    "j1 tokens noread"
    "j3 scratch edrop0.25-real"
    "heal tokens parent"
)
IDX=${SLURM_ARRAY_TASK_ID:-0}
if [ "$IDX" -ge "${#TASKS[@]}" ]; then echo "no task at index $IDX (use --array=0-3)"; exit 0; fi
read EXP ENCODING LIMB <<< "${TASKS[$IDX]}"
if [ "$EXP" = "heal" ]; then
    # PROBE (c), j4_prereg.md S9.1 -- the heal parent, the clean no-J-training
    # baseline for Z_end.  Not an -a3z- run name and not STEP 93552, so both come
    # from their own defaults and THE FOUR REGISTERED TASKS ARE UNTOUCHED: a bare
    # `sbatch pace/diag_z_registers.sbatch` still runs exactly D3 (--array=0-3).
    # This row is a BASELINE, not a fifth registered checkpoint -- its READING
    # line is descriptive and decides nothing on its own.
    RUN=${PARENT_RUN:-retro-b2-heal}
    CKPT=$OUT_ROOT/$RUN/${PARENT_CKPT:-checkpoint_91552_w16}
else
    RUN=${EXP}-a3z-${ENCODING}-${LIMB}
    CKPT=$OUT_ROOT/$RUN/checkpoint_$STEP
fi
OUT=eval_results/z3_registers/$RUN

SETS="--set use_memory=true --set memory_slots=0 --set prefix_memory=gated"
SETS="$SETS --set accum_vecs=16 --set gate_slots=64 --set gate_route=ring"
SETS="$SETS --set gate_norm=tanh --set gate_init=zero --set gate_fill=grow"
SETS="$SETS --set latent_carry=true --set latent_depth_rule=absolute"
SETS="$SETS --set latent_depth_lo=2 --set latent_depth_hi=9 --set latent_renorm=none"
SETS="$SETS --set latent_encoding=$ENCODING --set latent_tok_pool=$TOK_POOL"
SETS="$SETS --set latent_s0_read=false"
case "$EXP:$LIMB" in
    j3:*)       SETS="$SETS --set latent_read=scratch --set latent_read_heads=$READ_HEADS"
                SETS="$SETS --set latent_read_znorm=rms --set latent_read_znorm_target=$ZNORM_TARGET"
                SETS="$SETS --set latent_read_gate_lr_mult=$GATE_LR_MULT"
                SETS="$SETS --set latent_scratch_forget_bias=$SCRATCH_FB" ;;
    j1:real)    SETS="$SETS --set latent_read=xattn --set latent_read_heads=$READ_HEADS"
                SETS="$SETS --set latent_read_znorm=rms --set latent_read_znorm_target=$ZNORM_TARGET" ;;
    j1:noread)  SETS="$SETS --set latent_read=none --set latent_write_only=true" ;;
    heal:parent) # (c): the parent has no trained read, so it must read NOTHING --
                # write_only exercises no untrained module.  Z_end and E are pure
                # forward properties of the branch; the encoding only names the
                # (meaningless here) Z_write column, and `tokens` keeps the
                # write_equals_z_tok self-check live.
                SETS="$SETS --set latent_read=none --set latent_write_only=true" ;;
    *) echo "ERROR: no flag set for $EXP:$LIMB"; exit 1 ;;
esac

echo "=== z3 register probe | task $IDX = $RUN ==="
echo "    checkpoint: $CKPT"
echo "    data:       $DATA (first $MAX_EXAMPLES rows)"
echo "    out:        $OUT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
[ -f "$CKPT/chkpt.pt" ] || { echo "ERROR: no $CKPT/chkpt.pt"; exit 1; }
[ -d "$DATA" ] || { echo "ERROR: no such pack: $DATA"; exit 1; }
if [ "$EXP" = "j3" ]; then
    # Same guard as the read-out: the stored gate is gate/mult.
    DIAG=$OUT_ROOT/$RUN/cortex_diag.jsonl
    [ -f "$DIAG" ] || { echo "ERROR: no $DIAG -- cannot confirm the trained gate_lr_mult"; exit 1; }
    python - "$DIAG" "$GATE_LR_MULT" <<'MULT' || exit 1
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
got = {r.get("z_gate_lr_mult") for r in rows} - {None}
if got != {float(sys.argv[2])}:
    sys.exit(f"ERROR: the run trained with gate_lr_mult {sorted(got)}, not {sys.argv[2]}")
print(f"    gate_lr_mult {sys.argv[2]} matches the run's diag rows")
MULT
fi

python evals/diag_z_registers.py \
    --model_name "$MODEL" --checkpoint "$CKPT" --data "$DATA" $SETS \
    --n_chunks 8 --T 8 --max_examples "$MAX_EXAMPLES" \
    --tok_rows 64 --tok_pool "$TOK_POOL" --boot "$BOOT" \
    --dtype float32 --out_dir "$OUT" $INC_ARGS
RC=$?
echo ""
if [ $RC -eq 0 ]; then
    echo "z3 $RUN DONE -> $OUT/results.json"
else
    echo "z3 $RUN FAILED rc=$RC"
fi
exit $RC
