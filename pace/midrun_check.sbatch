#!/bin/bash
#SBATCH --account=gts-aivanova7-lab
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --gres=gpu:H200:1
#SBATCH -t 4:00:00
#SBATCH -q inferno
#SBATCH -J midrun
#SBATCH -o logs/Report-%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=henryyjiang42@gmail.com

# ============================================================================
# THE MID-RUN READ-OUT for the 5B pair.  One GPU job, one checkpoint.
#
# Serves both pace/control_5b.sbatch and pace/cortex_final_5b.sbatch.
#
#   ARM=control STEP=12000 sbatch pace/midrun_check.sbatch
#   ARM=cortex  STEP=12000 ZNORM_TARGET=<the launch value> sbatch pace/midrun_check.sbatch
#
# tools/watch_run.py (free, login node, runs after every link) answers "did
# this link run as configured".  THIS job answers the questions that need a
# forward pass on held-out data, and it is the only thing that can:
#
#   control   is the FineWeb-Edu -> mix switch degrading the model, or just
#             moving it?  Two packs scored separately settle it; a training
#             loss cannot, because the corpus changed under it.
#   cortex    the same two numbers, PLUS the three that decide whether the
#             mechanism is doing anything:
#               X_off      = NLL(Z ablated) - NLL(Z on)    are Z's columns
#                            load-bearing at all
#               X_content  = NLL(donor Z)   - NLL(own Z)   do Z's CONTENTS pay,
#                            or are the columns just free capacity
#               E_margin   = NLL(E0Z0)      - NLL(E1Z0)    what E alone buys
#             Every J arm before J7 had X_off strongly positive and X_content
#             at ZERO -- the columns load-bearing, the contents irrelevant.
#             That is the failure this run is trying to beat, and X_content is
#             the only number that can tell you whether it has.
#
# WHEN TO RUN IT.  Not every link.  Four points earn their GPU-hours:
#   * ~step 600      the first resumable checkpoint.  Catches a dead mechanism
#                    before five more links are queued behind it.
#   * ~step 9,600    just past full recurrence (9,537).  The first honest read:
#                    every earlier point is at partial depth.
#   * step 11,444    THE HEAL BOUNDARY.  This one is mandatory on both arms --
#                    it is the baseline every post-switch number is paired
#                    against, and it cannot be recovered later if the
#                    checkpoint is pruned.
#   * every ~6,000 steps through the mix, and at the horizon.
#
# PAIRING.  --baseline pairs this checkpoint against an earlier JSON on the
# SAME pack, row by row, and bootstraps the DIFFERENCES.  That cancels
# between-document variance and is what makes ~0.01 nats resolvable at 400
# rows.  BASELINE_STEP defaults to the heal boundary, which is the comparison
# that answers the mix question.
#
# PREP -- the FineWeb-Edu validation pack does not exist yet.  Build it ONCE
# on a login node.  Two shards, one command each; the SECOND is the one to run.
#
# THE OBVIOUS WAY IS THE EXPENSIVE WAY.  --skip_tokens 2_000_000_000 would be
# disjoint by construction (data/fineweb_edu_olmo_len4096 was built at
# --max_tokens 2_000_000_000, and the mix's FineWeb-Edu leg is drawn from that
# same pack) -- but skip_rows is applied AFTER tokenisation, so it would
# tokenise 2B tokens and throw every one of them away.  Hours, on a shared and
# throttled login node that enforces its memory cap by killing you.
#
# DO THIS INSTEAD.  prepare_packed_dataset consumes shards in SORTED order and
# stopped at 2.0B of a ~10B sample, so any later shard is untouched.  List the
# local shards, then pack the LAST one from the top:
#
#   export HF_HOME=$SCRATCH/hf_cache
#   ls $HF_HOME/hub/datasets--HuggingFaceFW--fineweb-edu/snapshots/*/sample/10BT/*.parquet | sort
#   python tools/prepare_packed_dataset.py \
#       --tokenizer ckpts/olmo-retrofit-cortex \
#       --parquet_glob "<the LAST path from that listing>" \
#       --out data/fineweb_edu_olmo_val_len4096 --max_length 4096 \
#       --max_tokens 20_000_000
#
# --parquet_glob reads local files directly, so it needs no network and works
# under HF_HUB_OFFLINE=1 (--list_files does NOT: it calls list_repo_files).
# 20M tokens = ~4,880 rows, far more than the 400 this job scores, and it packs
# in minutes because nothing is tokenised and discarded.
#
# THE DISJOINTNESS IS AN ASSUMPTION WORTH CHECKING ONCE: it holds only if the
# 2.0B heal pack did not reach the last shard.  If the listing shows fewer than
# three shards, fall back to --skip_tokens and accept the cost.
#
# The other two packs already exist and are the ones the J read-outs used:
#   data/pg19_olmo_validation_len4096_strided   the off-task LM pack
#   data/carry_task_r16_len4096_val             synthetic, with answer_dep, so
#                                               --score carry can select the
#                                               answers only an earlier chunk
#                                               can supply
# ============================================================================

cd $SLURM_SUBMIT_DIR

module load anaconda3
module load cuda/12.1.1
conda activate cortex-retro

export SCRATCH=${SCRATCH:-$HOME/scratch}
export HF_HOME=$SCRATCH/hf_cache
export HF_HUB_OFFLINE=1

ARM=${ARM:?set ARM=control or ARM=cortex}
OUT_ROOT=${OUT_ROOT:-cortex-5b}
MODEL=${MODEL:-ckpts/olmo-retrofit-cortex}
STEP=${STEP:?set STEP=<optimizer step of the checkpoint to score>}
BASELINE_STEP=${BASELINE_STEP:-11444}      # the heal boundary
N_CHUNKS=${N_CHUNKS:-8}                    # MUST match the run's cross_chunks
T=${T:-8}                                  # MANDATORY -- configs say 32
BOOT=${BOOT:-2000}
NLL_N=${NLL_N:-400}
CELL_N=${CELL_N:-300}

FWEDU_VAL=${FWEDU_VAL:-data/fineweb_edu_olmo_val_len4096}
PG19_VAL=${PG19_VAL:-data/pg19_olmo_validation_len4096_strided}
CARRY_VAL=${CARRY_VAL:-data/carry_task_r16_len4096_val}

case "$ARM" in
control) RUN=${RUN:-c-chunked} ;;
cortex)  RUN=${RUN:-cortex-final} ;;
*)       echo "ARM must be control|cortex"; exit 1 ;;
esac

RUN_DIR="$OUT_ROOT/$RUN"
# model_only_chkpt_* lands every save_interval (300); checkpoint_* every 600.
# Prefer the weights-only dir -- it exists at twice the resolution and this
# job needs no optimizer state.
CKPT="$RUN_DIR/model_only_chkpt_$STEP"
[ -d "$CKPT" ] || CKPT="$RUN_DIR/checkpoint_$STEP"
# T is part of the IDENTITY of a read-out, not a knob on one: the slot->depth
# map is span=min(hi,T-1) tiled as j%span (cortex_graft.py:2156), so scoring
# the same checkpoint at another T reads a DIFFERENT map.  Keying the dir on
# STEP alone meant a T=12 run silently overwrote the T=8 JSONs -- and the
# baseline pairing then compared two maps and called it a corpus effect.
RES="eval_results/midrun/$RUN/step_${STEP}_T${T}"
BASE_RES="eval_results/midrun/$RUN/step_${BASELINE_STEP}_T${T}"
mkdir -p "$RES"

echo "=== midrun | arm=$ARM | run=$RUN | step $STEP ==="
echo "    checkpoint: $CKPT"
echo "    baseline:   $BASE_RES  (step $BASELINE_STEP)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

if [ ! -d "$CKPT" ]; then
    echo "ERROR: no checkpoint at step $STEP.  Present:"
    ls -d "$RUN_DIR"/model_only_chkpt_* "$RUN_DIR"/checkpoint_* 2>/dev/null \
        | sed 's/.*\///' | sort -t_ -k3 -n | tail -8
    exit 1
fi

# ===== the graft rebuild, for the cortex arm ONLY ==========================
# --model_name loads the BASE dir's config, which carries NO cortex flags, so
# without these the graft builds with no buffer and every eval dies.  These
# MUST mirror pace/cortex_final_5b.sbatch's flag block exactly -- a rebuild at the
# wrong latent_depth_lo reads a band the arm never wrote, and a rebuild with a
# trainable projection scores J4's read instead of J7's.  This class of
# mismatch has cost this project two runs.
SETS=""
if [ "$ARM" = "cortex" ]; then
    : "${ZNORM_TARGET:?ARM=cortex needs the ZNORM_TARGET the run LAUNCHED with -- a rebuild at another target scores the read at the wrong strength}"
    SETS="--set use_memory=true --set memory_slots=0 --set prefix_memory=gated"
    SETS="$SETS --set accum_vecs=16 --set gate_slots=64 --set gate_route=ring"
    # accum_max explicitly, though the gated ring is fixed at K and does not
    # consume it: model_utils falls back to a CODE DEFAULT of 128 when the key
    # is absent and says so in a line nobody reads.  It happens to equal
    # 16 x 8 today; pinning it means a change to either never becomes a silent
    # geometry mismatch.
    SETS="$SETS --set accum_max=$(( 16 * ${N_CHUNKS:-8} ))"
    # Both read flags, explicitly, for the reason j6_readout gives for forcing
    # latent_stride in three places: a flag that defaults to the right value
    # today is not a flag that was checked.  e_carry_read is the one that
    # parsed, printed and persisted while doing nothing for two runs.
    SETS="$SETS --set e_carry_read=true --set latent_carry_read=true"
    SETS="$SETS --set gate_norm=tanh --set gate_init=zero --set gate_fill=grow"
    SETS="$SETS --set latent_carry=true --set latent_depth_rule=absolute"
    SETS="$SETS --set latent_depth_lo=${LATENT_LO:-1} --set latent_depth_hi=${LATENT_HI:-9}"
    SETS="$SETS --set latent_renorm=none --set latent_tok_pool=4"
    SETS="$SETS --set latent_encoding=staggered_state"
    SETS="$SETS --set latent_embed_frozen=true"
    SETS="$SETS --set latent_s0_read=false --set latent_read=embeds"
    SETS="$SETS --set latent_read_znorm=rms"
    SETS="$SETS --set latent_read_znorm_target=$ZNORM_TARGET"
    # Cross-check the rebuild against what the run ACTUALLY trained with.
    # The flags above are what we BELIEVE; cortex_diag.jsonl is what happened.
    python - "$RUN_DIR/cortex_diag.jsonl" "$ZNORM_TARGET" "$STEP" <<'CHECK' || exit 1
import json, sys
path, want, step = sys.argv[1], float(sys.argv[2]), int(sys.argv[3])
try:
    rows = [json.loads(l) for l in open(path) if l.strip()]
except FileNotFoundError:
    sys.exit(f"ERROR: no {path} -- nothing records what this run trained with.")
if not rows:
    sys.exit("ERROR: cortex_diag.jsonl is empty")
# cortex_diag.jsonl APPENDS ACROSS LINKS, so after a re-target it holds more
# than one znorm_target and more than one architecture state.  Everything below
# must describe the checkpoint being SCORED, i.e. the last record at or before
# $STEP -- not the end of the file, which may be thousands of steps later and
# under a different target.  The previous version took next(iter(seen)) from a
# set, so a re-targeted run either refused a correct rebuild or passed a wrong
# one depending on set iteration order.
at = [r for r in rows if (r.get("step") or 0) <= step]
if not at:
    sys.exit(f"ERROR: no diag record at or before step {step} -- the trace "
             f"starts at {rows[0].get('step')}, so nothing records what this "
             f"checkpoint trained with.")
last = at[-1]
tgt = [r["z_znorm_target"] for r in at if r.get("z_znorm_target") is not None]
if tgt and abs(tgt[-1] - want) > 1e-6:
    allseen = sorted({r.get("z_znorm_target") for r in rows} - {None})
    sys.exit(f"ERROR: at step {step} this run was training at znorm_target "
             f"{tgt[-1]}, and this job passes {want}.  The rebuild would score "
             f"a model nobody trained.  (Targets across the whole chain: "
             f"{allseen} -- pass the one in effect AT the checkpoint.)")
print(f"    scoring step {step} against diag record {last.get('step')} "
      f"(znorm_target {tgt[-1] if tgt else 'n/a'})")
ef = last.get("z_embed_frozen")
if ef is None:
    sys.exit("ERROR: no z_embed_frozen in the diag -- the flag never reached "
             "the graft, so the frozen condition is unverifiable.")
# int, not `is True`: training_diag casts bools to int for wandb, and
# `1 is not True` is True in Python -- the comparison that failed a run whose
# projection WAS frozen (13662406_2).
if int(ef) != 1:
    sys.exit("ERROR: z_embed_frozen is 0 -- the projection TRAINED.  This "
             "checkpoint is J4's read, not the frozen architecture.")
nd = last.get("z_n_depths") or 0
if nd < 2:
    sys.exit(f"ERROR: z_n_depths {nd} -- the staggered write drew ONE depth, "
             "so this trained as a single-depth write.")
if last.get("z_mix_frac") or (last.get("z_stride") or 1) != 1:
    sys.exit("ERROR: this run carries a J-arm treatment flag (mix or stride).")
print(f"    diag OK: {len(rows)} rows, n_depths {nd}, embed_frozen {ef}, "
      f"e_carried_norm {last.get('z_e_carried_norm')}, "
      f"embed_ratio {last.get('z_embed_ratio')}")
CHECK
fi

RC_ALL=0

# ===== 1. held-out NLL, both corpora, BOTH ARMS ============================
# The two packs separate "forgetting the heal corpus" from "learning the mix",
# which one training loss cannot do because the corpus moved under it.
for PAIR in "fwedu:$FWEDU_VAL" "pg19:$PG19_VAL"; do
    KEY=${PAIR%%:*}
    DATA=${PAIR#*:}
    echo ""
    echo "--- val NLL / $KEY  ($DATA)"
    if [ ! -d "$DATA" ]; then
        echo "SKIPPED: no such pack: $DATA"
        [ "$KEY" = "fwedu" ] && echo "  (build it with the PREP block in this script's header)"
        RC_ALL=1
        continue
    fi
    BASE_ARG=""
    if [ -f "$BASE_RES/val_nll_$KEY.json" ] && [ "$STEP" != "$BASELINE_STEP" ]; then
        BASE_ARG="--baseline $BASE_RES/val_nll_$KEY.json"
    fi
    # On the cortex arm a graft that did not build is a FAILED read-out, not a
    # control measurement: without this the eval scores the no-memory path and
    # labels the JSON as the cortex arm.
    EXPECT=""
    [ "$ARM" = "cortex" ] && EXPECT="--expect_memory"
    python evals/eval_val_nll.py \
        --model_name "$MODEL" --checkpoint "$CKPT" --data "$DATA" $SETS \
        --n_chunks "$N_CHUNKS" --T "$T" --max_examples "$NLL_N" \
        --boot "$BOOT" --dtype float32 $EXPECT \
        --label "$RUN@$STEP/$KEY" $BASE_ARG \
        --out "$RES/val_nll_$KEY.json" || RC_ALL=$?
done

# ===== 2. the mechanism cells, CORTEX ARM ONLY =============================
if [ "$ARM" = "cortex" ]; then
    # THE FULL 2x2 ON THE -off TASKS, so the carry DECOMPOSES.
    # Until 2026-10-03 these ran E1Z1,E1Z0,E0Z0 and E1Z1,E1Z0, so the only
    # effect `eval_carry_2x2.py` could compute was Z_main: E_main needs E0Z1
    # and the interaction needs all four, and compute_effects SKIPS any row
    # whose cells are absent.  The step-11,444 read-out therefore measured Z's
    # effect and NOT E's -- E1Z1 vs E0Z0 is the carry E AND Z make TOGETHER
    # (+0.3119 nats there), which is not E's effect and must not be quoted as
    # one.  The interaction is the design's actual hypothesis (negative =
    # E is worth MORE when Z is present) and nothing was reporting it.
    #
    # Each cell is a separate full pass, so this is +1 pass on carry-off and
    # +2 on pg19-off.  CELLS_OFF trims it back if a read-out has to be cheap;
    # the donor tasks stay 2-cell on purpose -- X_content is a Z-only
    # question and E0Z1/E0Z0 add nothing to it.
    CELLS_OFF=${CELLS_OFF:-E1Z1,E1Z0,E0Z1,E0Z0}
    # key                pack   score  z_null  n         cells
    TASKS=(
        "carry-off       carry  carry  off     $CELL_N   $CELLS_OFF"
        "carry-donor     carry  carry  donor   $CELL_N   E1Z1,E1Z0"
        "carry-local     carry  local  off     100       E1Z1,E1Z0"
        "pg19-off        pg19   all    off     $NLL_N    $CELLS_OFF"
        "pg19-donor      pg19   all    donor   $NLL_N    E1Z1,E1Z0"
    )
    for T_ROW in "${TASKS[@]}"; do
        read KEY PACK SCORE ZNULL N CELLS <<< "$T_ROW"
        case "$PACK" in
            carry) DATA=$CARRY_VAL ;;
            pg19)  DATA=$PG19_VAL ;;
        esac
        echo ""
        echo "--- 2x2 / $KEY  (score $SCORE, z_null $ZNULL, n $N, cells $CELLS)"
        [ -d "$DATA" ] || { echo "SKIPPED: no such pack: $DATA"; RC_ALL=1; continue; }
        python evals/eval_carry_2x2.py \
            --model_name "$MODEL" --checkpoint "$CKPT" --data "$DATA" $SETS \
            --n_chunks "$N_CHUNKS" --T "$T" --max_examples "$N" --boot "$BOOT" \
            --dtype float32 --z_null "$ZNULL" --score "$SCORE" --cells "$CELLS" \
            --out_dir "$RES/$KEY" || RC_ALL=$?
    done
fi

# ===== 3. the architecture trace, alongside ================================
mkdir -p logs/final5b_diag
cp "$RUN_DIR/cortex_diag.jsonl" "logs/final5b_diag/${RUN}_at_${STEP}.jsonl" 2>/dev/null
python tools/watch_run.py --run "$RUN_DIR" --arm "$ARM" \
    ${ZNORM_TARGET:+--znorm_target $ZNORM_TARGET} || true

echo ""
echo "=== midrun finished rc=$RC_ALL -> $RES ==="
echo ""
echo "HOW TO READ IT -- the bars, written down before the numbers arrive:"
echo "  val NLL      chunk1 is the 'did it break' number (no carry either arm)."
echo "               Across the switch, expect PG-19 down and FineWeb-Edu up a"
echo "               little.  BOTH up is degradation; PG-19 flat means check"
echo "               the switch link actually read the mix pack."
if [ "$ARM" = "cortex" ]; then
echo "  THE CARRY, DECOMPOSED  the -off tasks now run the full 2x2, so"
echo "               results.json carries four rows, not one:"
echo "                 E_main       E0Z1 - E1Z1.  E's effect, Z held ON."
echo "                 Z_main       E1Z0 - E1Z1.  Z's effect, E held ON  (= X_off)."
echo "                 Z_alone      E0Z0 - E0Z1.  Z's effect with E held OFF."
echo "                 interaction  (E's effect without Z) - (E's effect with Z)."
echo "                              NEGATIVE = COMPLEMENTS: E is worth more when"
echo "                              Z is present.  This is the design's actual"
echo "                              hypothesis and the only sign you cannot read"
echo "                              off a single column."
echo "               E1Z1 vs E0Z0 is E AND Z TOGETHER, not E's effect.  Quote"
echo "               E_main for E.  At step 11,444 the together number was"
echo "               +0.3119 while Z_main was -0.0024, i.e. E was doing"
echo "               essentially all of it -- but that was an inference from"
echo "               Z's null, not a measurement, because E0Z1 never ran."
echo "  X_off        E1Z0 - E1Z1.  Positive on every J arm ever run, including"
echo "               the ones where Z carried nothing.  It is NOT evidence."
echo "  X_content    donor - own.  THE deciding number.  J7 measured +0.2215"
echo "               at latent_depth_lo=2; this run trains lo=1, which re-maps"
echo "               all 16 slots, so that value does not transfer as-is."
echo "               Zero here with X_off positive is the pre-J7 signature:"
echo "               columns load-bearing, contents irrelevant."
echo "  content share  X_content / X_off.  0% on ctrl/mix/slow, 103% on J7."
echo "  chunk1_max_spread  must be ~0 -- the cells must agree where no carry"
echo "               exists, or they differ in something other than the carry."
fi
echo "  chance margin  any cell within 0.25 nats of ln(vocab) is scoring noise"
echo "               and every delta computed from it is a difference of two"
echo "               noise levels."
exit $RC_ALL
