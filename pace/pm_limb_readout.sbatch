#!/bin/bash
#SBATCH --account=gts-aivanova7-lab
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96GB
#SBATCH --gres=gpu:H200:1
#SBATCH --constraint=H200
#SBATCH -t 8:00:00
#SBATCH -q inferno
#SBATCH -J pm_read
#SBATCH -o logs/Report-%A_%a.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=henryyjiang42@gmail.com

# ============================================================================
# ROUND 2 -- read every post-mortem limb IN ITS OWN CELL (score_j5.py's rule).
#
#   sbatch --array=0-7 pace/pm_limb_readout.sbatch
#     [0] both [1] eonly [2] zonly [3] edrop [4] recall [5] recall_ctrl
#     [6] cortex-final@38146 recall probe only (the zero point)
#     [7] c-chunked@38146   recall probe only
#     [8] lownorm           (sbatch --array=8; needs pm_limbs --array=6)
#   mc/   both + lownorm: the MC suite, so the benchmark gap is read limb
#         against limb (`both` absorbs the re-warm, as everywhere else).
#
# Run AFTER pm_limbs; each limb task refuses if its checkpoint_40146/chkpt.pt is
# absent.  Read every limb against `both`, never against 38,146: `both` carries
# the same re-warm, tokens and batch-4 noise.
#
#   val_nll_{fwedu,pg19}  per-chunk NLL in the limb's own condition: the
#                 headline cross-limb table, chunk 1 included (does any limb
#                 shed the chunk-1 tax?).
#   recall/       probe answer NLL by fact distance (2..7 chunks): carry vs
#                 donor vs none.  The retention curve the `recall` limb was
#                 trained to bend.
#   z_content/    Z donor vs own Z, in the limb's own E condition (X_content;
#                 for zonly, Z_alone-with-donor).
#   e_content/    E donor vs own E (limbs that read E).
#   edrop_2x2/    edrop only: all four cells, z_null off -- the one model where
#                 E_main, Z_alone and the interaction are in distribution.
#   horizon/      both + recall: I(d) d=1..7, both channels -- did the recall
#                 objective extend the horizon?
#
# COST (H200 fp32, ~0.2 s per chunk forward + ~8 min per checkpoint load):
#   both / recall   val_nll ~21m + recall ~23m + z/e content 2x~32m
#                   + horizon ~24m + loads ~50m                  ~= 3.2 h
#   edrop           + the full 2x2 (~43m) instead of the horizon ~= 3.5 h
#   eonly / zonly   ~2.3 h      recall_ctrl ~1 h      tasks 6/7 ~35 min
# 8 h wall.
# ============================================================================

cd $SLURM_SUBMIT_DIR
source pace/pm_common.sh
pm_env

LIMBS=(both eonly zonly edrop recall recall_ctrl ref_cortex ref_control lownorm)
IDX=${SLURM_ARRAY_TASK_ID:-0}
LIMB=${LIMBS[$IDX]}
[ -n "$LIMB" ] || { echo "no task at index $IDX (use --array=0-8)"; exit 0; }
LOWNORM_TARGET=${LOWNORM_TARGET:-4.5}
STOP=${STOP:-40146}
RECALL_VAL=${RECALL_VAL:-data/pg19_val_recall_len4096}
RC_ALL=0
step() { echo ""; echo "--- [$LIMB] $1  ($(date +%H:%M:%S))"; }

case "$LIMB" in
    ref_cortex)  pm_resolve cortex 38146 || exit 1 ;;
    ref_control) pm_resolve control 38146 || exit 1 ;;
    *)
        CKPT="cortex-pm/pm-$LIMB/checkpoint_$STOP"
        RES="eval_results/pm/limbs/pm-$LIMB"
        mkdir -p "$RES"
        [ -f "$CKPT/chkpt.pt" ] || { echo "ERROR: no $CKPT/chkpt.pt -- did pm_limbs[$IDX] finish?"; exit 1; }
        if [ "$LIMB" = "recall_ctrl" ]; then
            SETS=""
        else
            pm_resolve cortex 38146 >/dev/null || exit 1   # builds the 78.9 SETS
            CKPT="cortex-pm/pm-$LIMB/checkpoint_$STOP"     # pm_resolve overwrote it
            RES="eval_results/pm/limbs/pm-$LIMB"
            case "$LIMB" in
                eonly) SETS="$SETS --set latent_carry_read=false" ;;
                zonly) SETS="$SETS --set e_carry_read=false" ;;
                lownorm)
                    # Later --set wins, so these replace pm_resolve's 78.9.
                    SETS="$SETS --set e_read_norm_target=$LOWNORM_TARGET"
                    SETS="$SETS --set latent_read_znorm_target=$LOWNORM_TARGET"
                    python - "cortex-pm/pm-lownorm/cortex_diag.jsonl" "$LOWNORM_TARGET" <<'CHECK' || exit 1
import json, sys
tg = [json.loads(l).get("z_znorm_target") for l in open(sys.argv[1]) if l.strip()]
tg = [t for t in tg if t is not None]
if not tg or abs(tg[-1] - float(sys.argv[2])) > 1e-6:
    sys.exit(f"ERROR: pm-lownorm trained at znorm_target {tg[-1:]}, rebuilding at {sys.argv[2]}")
print(f"    lownorm znorm_target {sys.argv[2]} confirmed against its diag")
CHECK
                    ;;
            esac
        fi ;;
esac
echo "=== pm_read | $LIMB | $CKPT -> $RES ==="
git rev-parse --short HEAD 2>/dev/null | sed 's/^/    commit: /'

if [ "$LIMB" != "ref_cortex" ] && [ "$LIMB" != "ref_control" ]; then
    EXPECT=""
    [ "$LIMB" != "recall_ctrl" ] && EXPECT="--expect_memory"
    for PAIR in "fwedu:$FWEDU_VAL" "pg19:$PG19_VAL"; do
        KEY=${PAIR%%:*}; DATA=${PAIR#*:}
        step "val NLL $KEY"
        python evals/eval_val_nll.py --model_name "$MODEL" --checkpoint "$CKPT" \
            --data "$DATA" $SETS --n_chunks 8 --T "$T" --max_examples 400 \
            --boot 2000 --dtype float32 $EXPECT --label "pm-$LIMB/$KEY" \
            --out "$RES/val_nll_$KEY.json" || RC_ALL=$?
    done
fi

step "recall probe"
python evals/score_recall_probe.py --model_name "$MODEL" --checkpoint "$CKPT" \
    $SETS --data "$RECALL_VAL" --T "$T" --max_rows 400 --dtype float32 \
    --out_dir "$RES/recall" || RC_ALL=$?

case "$LIMB" in
    both|edrop|recall|lownorm)
        step "Z content (E on)"
        python evals/eval_carry_2x2.py --model_name "$MODEL" --checkpoint "$CKPT" $SETS \
            --data "$PG19_VAL" --n_chunks 8 --T "$T" --max_examples 400 --dtype float32 \
            --z_null donor --cells E1Z1,E1Z0 --out_dir "$RES/z_content" || RC_ALL=$? ;;
    zonly)
        step "Z content (E off: the limb's own cell)"
        python evals/eval_carry_2x2.py --model_name "$MODEL" --checkpoint "$CKPT" $SETS \
            --data "$PG19_VAL" --n_chunks 8 --T "$T" --max_examples 400 --dtype float32 \
            --z_null donor --cells E0Z1,E0Z0 --out_dir "$RES/z_content" || RC_ALL=$? ;;
esac
case "$LIMB" in
    both|eonly|edrop|recall|lownorm)
        step "E content"
        python evals/eval_carry_2x2.py --model_name "$MODEL" --checkpoint "$CKPT" $SETS \
            --data "$PG19_VAL" --n_chunks 8 --T "$T" --max_examples 400 --dtype float32 \
            --e_null donor --cells E1Z1,E0Z1 --out_dir "$RES/e_content" || RC_ALL=$? ;;
esac
if [ "$LIMB" = "edrop" ]; then
    step "full 2x2, in distribution"
    python evals/eval_carry_2x2.py --model_name "$MODEL" --checkpoint "$CKPT" $SETS \
        --data "$PG19_VAL" --n_chunks 8 --T "$T" --max_examples 400 --dtype float32 \
        --z_null off --out_dir "$RES/edrop_2x2" || RC_ALL=$?
fi
if [ "$LIMB" = "both" ] || [ "$LIMB" = "recall" ]; then
    step "influence horizon (both channels)"
    python evals/eval_influence_horizon.py --model_name "$MODEL" --checkpoint "$CKPT" \
        $SETS --data "$PG19_VAL" --n_chunks 8 --depths 1 2 3 4 5 6 7 \
        --damage donor --damage_channel both --T "$T" --max_examples 100 \
        --dtype float32 --out_dir "$RES/horizon" || RC_ALL=$?
fi
if [ "$LIMB" = "both" ] || [ "$LIMB" = "lownorm" ]; then
    step "MC suite (single forward: the chunk-1 tax on benchmarks)"
    python evals/eval_multiple_choice.py --model_name "$MODEL" \
        --checkpoint "$CKPT" $SETS --tasks hellaswag winogrande arc_easy arc_challenge piqa \
        --T "$T" --out_dir "$RES/mc" || RC_ALL=$?
fi
echo ""
echo "=== pm_read $LIMB finished rc=$RC_ALL ($(date +%H:%M:%S)) -> $RES ==="
exit $RC_ALL
