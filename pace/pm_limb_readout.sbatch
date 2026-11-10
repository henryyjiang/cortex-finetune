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
#     [9] carrydrop         (sbatch --array=9; needs pm_limbs --array=7):
#                           val NLL (chunk 1 IS the no-carry mode), recall, MC
#     [10] v2capwarm        (sbatch --array=10 -t 12:00:00; needs pm_limbs --array=8):
#                           rebuilt as gated_v2 (paired) with its E cap, so it is
#                           scored as it trained.  val NLL, recall, MC, plus
#                           depth/ and babilong/ (below), and since 2026-11-09
#                           z_content/ + e_content/.
#         RUN_SUFFIX=-stfix reads pm_limbs' re-run cortex-pm/pm-v2capwarm-stfix
#         into eval_results/pm/limbs/pm-v2capwarm-stfix (any limb, any suffix).
#         CONTENT_ONLY=1 on [10]: ONLY z_content/ + e_content/ (~1.2 h), for
#         the original v2capwarm, whose first read-out ran without them.
#   COMPARE_ONLY=1 with --array=0: ONLY depth/ + babilong/ on pm-both, the
#         baseline v2capwarm is read against (same steps, same flags).
#   depth/    diag_chunk1_drift drift,gates,restart,recency on 202-chunk books:
#             first harmful chunk, benefit at 2/8/32/64/202, gates or v2
#             replace rate, the restart/swap gaps and recency lift.
#   babilong/ qa1 qa2 at 1k-32k in the 5B pair's geometry (8 x 512, T 8,
#             250/cell, ranked + PMI): the cell cortex-final lost from 16k.
#   mc/   both + lownorm + carrydrop + v2capwarm: the MC suite, so the benchmark gap is read limb
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
#   v2capwarm       val_nll ~21m + recall ~23m + MC ~25m + depth ~25m
#                   + z/e content 2x~32m + BABILong ~2-3.5 h + loads ~65m ~= 6-7.5 h
#   both, COMPARE_ONLY=1   depth ~25m + BABILong ~2-3.5 h      ~= 3-4 h
# 8 h wall; submit v2capwarm with -t 12:00:00.
# ============================================================================

cd $SLURM_SUBMIT_DIR
source pace/pm_common.sh
pm_env

LIMBS=(both eonly zonly edrop recall recall_ctrl ref_cortex ref_control lownorm carrydrop v2capwarm)
IDX=${SLURM_ARRAY_TASK_ID:-0}
LIMB=${LIMBS[$IDX]}
[ -n "$LIMB" ] || { echo "no task at index $IDX (use --array=0-10)"; exit 0; }
LOWNORM_TARGET=${LOWNORM_TARGET:-4.5}
E_CAP=${E_CAP:-82}                    # pm_limbs' default: what v2capwarm trained at
BABILONG_PATH=${BABILONG_PATH:-data/babilong-1k}
BABILONG_REPO=${BABILONG_REPO:-RMT-team/babilong-1k-samples}
BABILONG_MAX=${BABILONG_MAX:-250}
STOP=${STOP:-40146}
RECALL_VAL=${RECALL_VAL:-data/pg19_val_recall_len4096}
RC_ALL=0
step() { echo ""; echo "--- [$LIMB] $1  ($(date +%H:%M:%S))"; }

case "$LIMB" in
    ref_cortex)  pm_resolve cortex 38146 || exit 1 ;;
    ref_control) pm_resolve control 38146 || exit 1 ;;
    *)
        RUN="pm-$LIMB${RUN_SUFFIX:-}"         # RUN_SUFFIX: a pm_limbs re-run of this limb
        CKPT="cortex-pm/$RUN/checkpoint_$STOP"
        RES="eval_results/pm/limbs/$RUN"
        mkdir -p "$RES"
        [ -f "$CKPT/chkpt.pt" ] || { echo "ERROR: no $CKPT/chkpt.pt -- did pm_limbs[$IDX] finish?"; exit 1; }
        if [ "$LIMB" = "recall_ctrl" ]; then
            SETS=""
        else
            pm_resolve cortex 38146 >/dev/null || exit 1   # builds the 78.9 SETS
            CKPT="cortex-pm/$RUN/checkpoint_$STOP"     # pm_resolve overwrote it
            RES="eval_results/pm/limbs/$RUN"
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
                v2capwarm)
                    # gated_v2 sets the score heads' shape: rebuilt as gated it
                    # would load with 3 unexpected keys and score the old ring.
                    SETS="$SETS --set prefix_memory=gated_v2 --set v2_pairing=paired"
                    SETS="$SETS --set e_store_norm_target=$E_CAP"
                    python - "cortex-pm/$RUN/cortex_diag.jsonl" "$E_CAP" <<'CHECK' || exit 1
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
if not any(r.get("v2_replace_rate") is not None for r in rows):
    sys.exit("ERROR: pm-v2capwarm's diag has no v2_replace_rate: it did not train gated_v2")
en = [r["z_e_carried_norm"] for r in rows if r.get("z_e_carried_norm") is not None]
cap = float(sys.argv[2])
if not en or max(abs(v - cap) for v in en) > 0.02 * cap:
    sys.exit(f"ERROR: pm-v2capwarm carried E at {en[:1]}..{en[-1:]}, rebuilding at cap {cap}")
print(f"    v2capwarm: gated_v2 and E cap {cap} confirmed against its diag")
CHECK
                    ;;
            esac
        fi ;;
esac
echo "=== pm_read | $LIMB | $CKPT -> $RES ==="
git rev-parse --short HEAD 2>/dev/null | sed 's/^/    commit: /'

# depth/ + babilong/: the Part B read-out (v2_buffer_handoff.md B5), run on
# v2capwarm and, with COMPARE_ONLY=1, on pm-both with the same flags.
depth_babilong() {
    step "depth tests: drift, gates, restart, recency (202-chunk books)"
    python evals/diag_chunk1_drift.py --model_name "$MODEL" --checkpoint "$CKPT" \
        $SETS --data "$PG19_VAL" --T "$T" --parts drift,gates,restart,recency \
        --drift_chunks 202 --drift_streams 6 --depths 8,16,32,64,128 \
        --dtype float32 --out_dir "$RES/depth" || RC_ALL=$?
    step "BABILong qa1 qa2, 1k-32k (8 x 512, T $T)"
    python evals/eval_babilong.py --model_name "$MODEL" --checkpoint "$CKPT" $SETS \
        --dataset_path "$BABILONG_PATH" --dataset_repo "$BABILONG_REPO" \
        --tasks qa1 qa2 --max_examples "$BABILONG_MAX" --seq_len 512 --num_chunks 8 \
        --T "$T" --score_nll --rank_answers --pmi \
        --out_dir "$RES/babilong" || RC_ALL=$?
}
content_tests() {
case "$LIMB" in
    both|edrop|recall|lownorm|v2capwarm)
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
    both|eonly|edrop|recall|lownorm|v2capwarm)
        step "E content"
        python evals/eval_carry_2x2.py --model_name "$MODEL" --checkpoint "$CKPT" $SETS \
            --data "$PG19_VAL" --n_chunks 8 --T "$T" --max_examples 400 --dtype float32 \
            --e_null donor --cells E1Z1,E0Z1 --out_dir "$RES/e_content" || RC_ALL=$? ;;
esac
}
# CONTENT_ONLY=1 on [10]: just z_content/ + e_content/ on a limb already read
# (2026-11-09: v2capwarm's Z rank fell 9.9 -> 3.5; is Z still carrying content?)
if [ "${CONTENT_ONLY:-}" = "1" ]; then
    [ "$LIMB" = "v2capwarm" ] || { echo "CONTENT_ONLY=1 is for v2capwarm (--array=10)"; exit 1; }
    content_tests
    echo ""
    echo "=== pm_read $LIMB (CONTENT_ONLY) finished rc=$RC_ALL ($(date +%H:%M:%S)) -> $RES ==="
    exit $RC_ALL
fi
if [ "${COMPARE_ONLY:-}" = "1" ]; then
    [ "$LIMB" = "both" ] || { echo "COMPARE_ONLY=1 is for the pm-both baseline (--array=0)"; exit 1; }
    depth_babilong
    echo ""
    echo "=== pm_read both (COMPARE_ONLY) finished rc=$RC_ALL ($(date +%H:%M:%S)) -> $RES ==="
    exit $RC_ALL
fi

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

content_tests
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
if [ "$LIMB" = "v2capwarm" ] || [ "$LIMB" = "both" ] || [ "$LIMB" = "lownorm" ] || [ "$LIMB" = "carrydrop" ]; then
    step "MC suite (single forward: the chunk-1 tax on benchmarks)"
    python evals/eval_multiple_choice.py --model_name "$MODEL" \
        --checkpoint "$CKPT" $SETS --tasks hellaswag winogrande arc_easy arc_challenge piqa \
        --T "$T" --out_dir "$RES/mc" || RC_ALL=$?
fi
[ "$LIMB" = "v2capwarm" ] && depth_babilong
echo ""
echo "=== pm_read $LIMB finished rc=$RC_ALL ($(date +%H:%M:%S)) -> $RES ==="
exit $RC_ALL
