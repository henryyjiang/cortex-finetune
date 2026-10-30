#!/bin/bash
#SBATCH --account=gts-aivanova7-lab
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96GB
#SBATCH --gres=gpu:H200:1
#SBATCH --constraint=H200
#SBATCH -t 12:00:00
#SBATCH -q inferno
#SBATCH -J pm_probe
#SBATCH -o logs/Report-%A_%a.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=henryyjiang42@gmail.com

# ============================================================================
# POST-MORTEM PROBES -- where the fact dies, and how the model reads memory.
#
#   sbatch --array=0-2 pace/pm_probe.sbatch                              # both parts
#   sbatch --array=0-2 --export=ALL,PARTS=dissect pace/pm_probe.sbatch
#   sbatch --array=0-1 --export=ALL,PARTS=interp  pace/pm_probe.sbatch
#     [0] cortex@38146   [1] cortex@11444   [2] control@38146 (dissection only)
#     [3] limb both      [4] limb recall    [5] limb lownorm  (pm-*/checkpoint_40146)
#
# Task 1 is the heal-end half of the heal-vs-final comparison, as in pm_mech.
#
# LIMB TASKS (2026-10-30), each read against `both`, never against 38,146:
#   sbatch --array=3-4 --export=ALL,PARTS=dissect pace/pm_probe.sbatch
#       does the recall limb's past-the-lap retention (25-31% at 5-6 chunks on
#       its own probe) reach BABILong?  Watch teleport - normal at dist 4-7/8+.
#   sbatch --array=3,5 --export=ALL,PARTS=interp pace/pm_probe.sbatch
#       does reading the carry at token scale move WHERE it is read (coda mass,
#       part B's blocking map) -- the lownorm question the battery left open.
#
#   babilong_dissect/  WRITE -> RETAIN -> READ on BABILong qa1/qa2
#                      (evals/diag_babilong_dissect.py).  normal / nocarry /
#                      teleport (the post-fact carry dropped straight into the
#                      question chunk) / teleport_donor / fact_in_window.
#                        teleport >> normal      retention loses the fact
#                        teleport ~= donor       written-but-unused, or never written
#                        fact_in_window ~ chance the capability floor binds: memory
#                                                cannot win a task the model cannot
#                                                do with the fact in view
#                      The control runs normal + fact_in_window: its in-window
#                      ceiling is the bar cortex's teleport is read against.
#                      NOTE the benchmark's `gold_in_final_window` is NOT "the
#                      fact is visible": in 20 of 23 smoke items it was true only
#                      because the gold word sits in the question's demo suffix.
#                      Use this tool's `distance` field, never that flag.
#                      `normal` reproduced the benchmark's per-item argmin on
#                      40/40 smoke items (residual cand-NLL 0.11: bf16 there).
#   interp/            (cortex + limbs) evals/diag_memory_interp.py:
#                        A read-attention shares per layer x loop iteration x head,
#                          and by ring-row age -> the memory heads, and the loop
#                          depth where memory is consumed
#                        B attention-blocking delta-NLL map per site -> where the
#                          carry's effect actually enters
#                        C write-side attention of the summary slots
#                        D logit lens on E/Z rows vs source / next / donor chunk,
#                          plus (2026-10-30) a RARE-word variant (forms in <10%
#                          of sampled chunks: does a royalty chunk's row decode
#                          to 'king'?) and an INPUT-EMBEDDING lens (cosine to
#                          wte, where E is spliced)
#                        E effective rank + centred cosine of the E and Z ring
#                        F topic steering: a donor book's carry read into a
#                          row -- does it lift THAT book's distinctive words
#                          (vs a third book's)?  steer_lift; + --steer_demo
#                          (royalty vs sea passage into one neutral passage)
#                        G soft carry scaling: real->carried attention x alpha
#                          (0.5 / 0.75 / 1.25).  alpha < 1 improving NLL would
#                          mean the model OVER-attends to memory
#
# COST (H200 fp32; local 5070 Ti smoke / 2.5, plus ~10 min per load):
#   dissection  800 items (qa1/qa2 x 1k/2k/4k/8k x 100): cortex ~1.1 h,
#               control ~25 min
#   interp      24 rows, part B (182 blocked forwards a row) most of it:
#               ~10 min compute + ~10 min load per checkpoint; F+G+the new D
#               lenses add ~12 s a row (smoke: F 9-12, G 23-27, D 5-8 s on
#               the 5070 Ti) = ~5 min
# 12 h wall.
# ============================================================================

cd $SLURM_SUBMIT_DIR
source pace/pm_common.sh
pm_env

CASES=("cortex 38146" "cortex 11444" "control 38146" "limb both" "limb recall" "limb lownorm")
IDX=${SLURM_ARRAY_TASK_ID:-0}
read ARM STEP <<< "${CASES[$IDX]}"
[ -n "$ARM" ] || { echo "no case at index $IDX (use --array=0-5)"; exit 0; }
if [ "$ARM" = "limb" ]; then
    # The limbs branched from 38146 and trained at its 78.9 rebuild flags;
    # lownorm then splices both channels at LOWNORM_TARGET, as its read-out does.
    pm_resolve cortex 38146 >/dev/null || exit 1
    CKPT="cortex-pm/pm-$STEP/checkpoint_40146"
    RES="eval_results/pm/limbs/pm-$STEP"
    [ -f "$CKPT/chkpt.pt" ] || { echo "ERROR: no $CKPT/chkpt.pt"; exit 1; }
    LOWNORM_TARGET=${LOWNORM_TARGET:-4.5}
    [ "$STEP" = "lownorm" ] && SETS="$SETS --set e_read_norm_target=$LOWNORM_TARGET --set latent_read_znorm_target=$LOWNORM_TARGET"
else
    pm_resolve "$ARM" "$STEP" || exit 1
fi
BABILONG_PATH=${BABILONG_PATH:-data/babilong-1k}
PARTS=${PARTS:-dissect,interp}
INTERP_PARTS=${INTERP_PARTS:-A,B,C,D,E,F,G}
# the benchmark's own records, for the per-item reproduction check (38146 only)
case "$ARM@$STEP" in
    cortex@38146)  REC="--records eval_results/longcontext_5b-final/cortex-final-T8-nc8-sl512/babilong" ;;
    control@38146) REC="--records eval_results/longcontext_5b-final/c-chunked-T8-nc8-sl512/babilong" ;;
    *)             REC="" ;;
esac

echo "=== pm_probe | $ARM @ $STEP | parts $PARTS | $CKPT -> $RES ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
git rev-parse --short HEAD 2>/dev/null | sed 's/^/    commit: /'
RC_ALL=0
step() { echo ""; echo "--- [$ARM@$STEP] $1  ($(date +%H:%M:%S))"; }

if [[ ",$PARTS," == *",dissect,"* ]]; then
    step "BABILong write/retain/read dissection"
    python evals/diag_babilong_dissect.py --model_name "$MODEL" --checkpoint "$CKPT" \
        $SETS --dataset_path "$BABILONG_PATH" --tasks qa1 qa2 \
        --lengths 1k 2k 4k 8k --max_items 100 --T "$T" --seq_len 512 \
        --num_chunks 8 --dtype float32 --seed 1234 --boot 2000 $REC \
        --out_dir "$RES/babilong_dissect" || RC_ALL=$?
fi

if [ "$ARM" != "control" ] && [[ ",$PARTS," == *",interp,"* ]]; then
    step "memory interp ($INTERP_PARTS)"
    python evals/diag_memory_interp.py --model_name "$MODEL" --checkpoint "$CKPT" \
        $SETS --data "$PG19_VAL" --n_chunks 8 --T "$T" --max_rows 24 \
        --parts "$INTERP_PARTS" --steer_demo --dtype float32 --seed 1234 --boot 2000 \
        --out_dir "$RES/interp" || RC_ALL=$?
fi

echo ""
echo "=== pm_probe $ARM@$STEP finished rc=$RC_ALL ($(date +%H:%M:%S)) -> $RES ==="
exit $RC_ALL
