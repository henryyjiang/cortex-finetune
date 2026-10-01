# ============================================================================
# THE SHARED RECIPE for the 5B pair.  Sourced by pace/control_5b.sbatch and
# pace/cortex_final_5b.sbatch.  NOT submittable on its own.
#
# WHY THIS FILE EXISTS
# --------------------
# The control and cortex-final must differ in EXACTLY ONE VARIABLE, because
# every "memory helps by X" number in the paper is their difference.  Two
# sibling sbatch files satisfy that on the day they are written and then drift
# -- someone fixes a warmup in one, bumps a save_interval in the other, and six
# weeks of H200 later the contrast is confounded by a hyperparameter nobody
# remembers changing.  So the two arms get their own launchers (they are
# submitted separately, and deliberately NOT at the same time -- see the
# sequencing note in pace/cortex_final_5b.sbatch), but every value they share
# is written HERE, once.
#
# `git diff` between the two launchers is therefore the complete difference
# between the two runs, and tests/test_final_5b.py enforces that: the shared
# train.py call may set only cross_chunks, eos_from_tokens and diag_interval,
# and everything mechanism-shaped has to come from the arm's $ARM_ARGS.
#
# CONTRACT -- an arm launcher does exactly this, in this order:
#
#     cd $SLURM_SUBMIT_DIR
#     ARM=control                       # or cortex
#     source pace/final_5b_common.sh    # defines the two functions only
#     final5b_shared                    # env, out dir, shared recipe, DATA_PATH
#     MICRO_BS_DEFAULT=1                # the arm's memory profile
#     ARM_ARGS="--cortex.use_memory false"
#     final5b_launch                    # pre-flights, banner, train.py, watch
#
# Between `final5b_shared` and `final5b_launch` the arm may read any shared
# value (CROSS_CHUNKS, MAX_LENGTH, BATCH_SIZE...) and must set MICRO_BS_DEFAULT
# and ARM_ARGS.  It must not change a shared value; that is what this file is
# for.
#
# The reasoning behind the recipe itself -- the batch size, the LR that was
# deliberately not re-tuned, the D-fallback geometry, the corpus build -- is in
# pace/control_retrofit.sbatch's header (superseded as a launcher, kept as the
# document of record) and in ../final_5b_launch.md.
# ============================================================================

final5b_shared() {
    module load anaconda3
    module load cuda/12.1.1
    conda activate cortex-retro

    export SCRATCH=${SCRATCH:-$HOME/scratch}
    export HF_HOME=$SCRATCH/hf_cache
    export HF_HUB_OFFLINE=1
    export WANDB_DIR=$SCRATCH
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

    # See b2_retrofit.sbatch for why the port is probed here rather than left
    # to train.py's SLURM_JOB_ID derivation (job 11702360 died in 29s on
    # EADDRINUSE).  train.py installs its own with os.environ.setdefault, so
    # anything exported here wins.
    export MASTER_ADDR=127.0.0.1
    MASTER_PORT=$(python -c "import socket;s=socket.socket();s.bind(('',0));print(s.getsockname()[1]);s.close()")
    export MASTER_PORT
    echo "MASTER_PORT=$MASTER_PORT"

    # --- out_path is BOTH the wandb project AND the write dir --------------
    # train.py does wandb.init(project=cfg.out_path, dir=cfg.out_path) and
    # IGNORES WANDB_DIR, so a relative name resolves into $HOME.  That is what
    # killed the 2026-07-17 round: one checkpoint blew the home quota and took
    # down six unrelated jobs, whose Slurm logs also live in home.  Keep the
    # relative name (it IS the wandb project) and symlink it to scratch.
    #
    # ONE PROJECT FOR BOTH ARMS, on purpose: the control exists to be overlaid
    # on cortex-final, and wandb overlays within a project for free.  The arms
    # are separate RUNS in separate subdirectories.
    OUT_DIR=${OUT_DIR:-cortex-5b}
    if [ ! -L "$OUT_DIR" ]; then
        if [ -e "$OUT_DIR" ]; then
            echo "ERROR: $OUT_DIR exists in \$HOME but is not a symlink to scratch."
            echo "  Move or delete it (rm -rf $OUT_DIR) and resubmit; this will"
            echo "  re-create it as a symlink to \$SCRATCH/$OUT_DIR."
            exit 1
        fi
        mkdir -p "$SCRATCH/$OUT_DIR"
        ln -s "$SCRATCH/$OUT_DIR" "$OUT_DIR"
    fi

    # ===== THE SHARED RECIPE.  Both arms.  Every line is the control's. =====
    PHASE=${PHASE:-heal}                       # heal | mix
    MODEL=${MODEL:-ckpts/olmo-retrofit-cortex}
    MAX_LENGTH=${MAX_LENGTH:-4096}
    CROSS_CHUNKS=${CROSS_CHUNKS:-8}            # 512-token chunks, D-fallback
    BATCH_SIZE=${BATCH_SIZE:-32}               # rows/step -> 131,072 tokens
    MAX_STEPS=${MAX_STEPS:-38147}              # 5.000B / 131,072 -- FIXED
    HEAL_END=${HEAL_END:-11444}                # 30.0% of the horizon
    # Defaulted PER PHASE in the case block below: heal stops at HEAL_END,
    # mix runs to the horizon.  Setting it in the environment still wins.
    MAX_MEAN_REC=${MAX_MEAN_REC:-8}
    RAMP_WARMUP=${RAMP_WARMUP:-0.25}           # full recurrence at step 9,537
    # LR warmup in STEPS, not as a fraction.  The inherited warmup=0.0025 gave
    # 125 steps at McLeish's 50,000-step horizon and 764 at B2's 305,176 -- but
    # only 96 here, because the fraction silently tracks a horizon that moved
    # 8x.  Warmup is the one schedule term that should NOT scale with the
    # horizon: it exists to let Adam's second moment mature and to survive the
    # conversion transient, and both are counted in steps.  (OLMo-2 itself
    # warmed up for 4,000.)
    WARMUP_STEPS=${WARMUP_STEPS:-96}
    LR=${LR:-5e-5}
    MUON_LR=${MUON_LR:-0.001}
    COOLDOWN=${COOLDOWN:-0.6}
    SEED=${SEED:-74}
    SAVE_INTERVAL=${SAVE_INTERVAL:-300}        # resumable checkpoint every 600
    # The architecture health trace, every N optimizer steps: per-channel carry
    # norms and ranks, whether either gate left its exactly-zero init, the Z
    # read/write gradient fractions, z_e_carried_norm.  One SVD of a [K, D]
    # matrix per call, and it is the ONLY live record of whether the mechanism
    # is attached to the loss.  Emits nothing at all on the control
    # (training_diag returns {} with no cortex), so it is free to leave on for
    # both.  50 gives ~760 points over the horizon.
    DIAG_INTERVAL=${DIAG_INTERVAL:-50}
    RESUME_PATH=${RESUME_PATH:-}
    # ONLY on the link that switches corpus.  Every later link is same-corpus
    # and MUST restore its position, or each one replays the pack from the top.
    EXTRA_ARGS=${EXTRA_ARGS:-}

    HEAL_DATA=${HEAL_DATA:-data/fineweb_edu_olmo_len4096}
    MIX_DATA=${MIX_DATA:-data/pg19_fw50_olmo_len4096}

    case "$PHASE" in
    heal)
        DATA_PATH=${DATA_PATH:-$HEAL_DATA}
        # THE HEAL STOPS AT HEAL_END BY DEFAULT, so no link of the chain has
        # to remember to add it.  Without this the link that crosses 11,444
        # keeps training on FineWeb-Edu -- the pack holds 15,258 steps, so it
        # does not exhaust and nothing complains -- and the arm silently ends
        # up with a longer heal and a shorter mix than its sibling.  That is a
        # corpus-schedule mismatch between the two runs, i.e. the one thing
        # the whole paired design exists to prevent, and it would be invisible
        # in both loss curves.
        #
        # The HORIZON DOES NOT MOVE: stop_at_step ends a phase, max_steps is
        # what the LR cosine and the recurrence ramp are keyed to.  Varying
        # max_steps across links re-runs cooldown per segment (the 2026-06-24
        # sawtooth, baked into the weights).
        STOP_AT_STEP=${STOP_AT_STEP:-$HEAL_END} ;;
    mix)
        DATA_PATH=${DATA_PATH:-$MIX_DATA}
        # 0 = run to the horizon.  NOT optional and not inherited: with the
        # global default removed, leaving this unset emits a bare
        # `--stop_at_step` with no value, and the next flag on the line
        # (--muon.use_muon) becomes its argument.
        STOP_AT_STEP=${STOP_AT_STEP:-0} ;;
    *)    echo "PHASE must be heal|mix"; exit 1 ;;
    esac
}

final5b_launch() {
    : "${ARM:?final5b_launch: ARM is unset -- the launcher must set it}"
    : "${RUN_NAME:?final5b_launch: RUN_NAME is unset}"
    : "${ARM_ARGS:?final5b_launch: ARM_ARGS is unset -- an arm with no flags is not an arm}"
    : "${MICRO_BS_DEFAULT:?final5b_launch: MICRO_BS_DEFAULT is unset -- the arms do not share it}"

    # MICRO_BS is the ONE value the arms are allowed to differ on, because it
    # is the one that changes nothing they are compared on.  Both run
    # batch_size 32 rows per optimizer step, 38,147 steps, the same LR and the
    # same pack, and the row PERMUTATION is a pure function of
    # (seed, len(dataset)) consumed in order -- so at mbs 1 optimizer step k
    # reads items 32k..32k+31 and at mbs 2 it reads items 16k..16k+15, THE SAME
    # 32 ROWS either way.  What differs is the reduction order inside a
    # micro-step, i.e. floating-point noise.
    MICRO_BS=${MICRO_BS:-$MICRO_BS_DEFAULT}

    RUN_DIR="$OUT_DIR/$RUN_NAME"
    mkdir -p "$RUN_DIR"

    # ---- geometry pre-flights, so a bad combination dies HERE -------------
    if [ $(( MAX_LENGTH % CROSS_CHUNKS )) -ne 0 ]; then
        echo "ERROR: MAX_LENGTH ($MAX_LENGTH) is not divisible by CROSS_CHUNKS" \
             "($CROSS_CHUNKS) -- chunks would be ragged."; exit 1
    fi
    if [ $(( BATCH_SIZE % MICRO_BS )) -ne 0 ]; then
        echo "ERROR: BATCH_SIZE ($BATCH_SIZE) is not divisible by MICRO_BS" \
             "($MICRO_BS) -- the last accumulation step would be short and"
        echo "       every optimizer step would see a different token count."
        exit 1
    fi

    # ---- MICRO_BS is pinned for the life of the chain ---------------------
    # train.py:1997 computes the resume row cursor as
    # data_start_step * micro_batch_size, and data_start_step is an ITEM count
    # written under whatever mbs produced it.  Changing mbs between links skips
    # the wrong number of rows, silently, behind a healthy loss curve.  Every
    # cortex run before this one was mbs 1, so this trap is NEW.
    STAMP="$RUN_DIR/.micro_bs"
    if [ -f "$STAMP" ]; then
        WAS=$(cat "$STAMP")
        if [ "$WAS" != "$MICRO_BS" ]; then
            echo "ERROR: this chain was started at MICRO_BS=$WAS and this link"
            echo "  passes $MICRO_BS.  The resume row cursor is data_start_step"
            echo "  x micro_batch_size, so the loader would fast-forward the"
            echo "  wrong number of rows -- silently, with a healthy loss curve."
            echo "  Either pass MICRO_BS=$WAS, or start a NEW run_name."
            exit 1
        fi
    else
        echo "$MICRO_BS" > "$STAMP"
    fi

    # ---- the resume chain -------------------------------------------------
    # There is deliberately no BRANCH_PATH: both arms are ONE CONTINUOUS RUN.
    # The corpus switches at HEAL_END and nothing else does, which is why that
    # link -- and only that link -- passes --reset_dataset_position true.
    CHAIN_ARGS=""
    if [ -n "$RESUME_PATH" ]; then
        if [ ! -f "$RESUME_PATH/chkpt.pt" ]; then
            echo "ERROR: no chkpt.pt under RESUME_PATH=$RESUME_PATH"
            echo "  Resumable checkpoints are every 2 x save_interval =" \
                 "$(( 2 * SAVE_INTERVAL )) steps.  Present:"
            ls -d "$RUN_DIR"/checkpoint_* 2>/dev/null | tail -5 \
                || echo "  (no checkpoint dirs at all)"
            exit 1
        fi
        CHAIN_ARGS="--resume_path $RESUME_PATH"
    fi

    # ---- the corpus-switch guard ------------------------------------------
    # The mistake this catches costs 1.15B tokens and reports "finished".  It
    # is advisory rather than fatal because a re-run of the switch link (after
    # a crash inside it) legitimately resumes mid-mix WITHOUT the flag.
    if [ "$PHASE" = "mix" ] && [ -n "$RESUME_PATH" ]; then
        case "$RESUME_PATH" in
            *_$HEAL_END|*_${HEAL_END}/)
                case "$EXTRA_ARGS" in
                    *reset_dataset_position*) ;;
                    *) echo ""
                       echo "  !!  RESUME_PATH is the heal boundary ($HEAL_END) and"
                       echo "  !!  EXTRA_ARGS does not carry --reset_dataset_position."
                       echo "  !!  Without it the loader fast-forwards 366,208 rows"
                       echo "  !!  into the mix pack and the run exhausts at ~29,375"
                       echo "  !!  of $MAX_STEPS, exiting clean at 3.85B tokens."
                       echo "  !!  Check the first '[data]' line below: on THIS link"
                       echo "  !!  'fast-forward' must be ABSENT."
                       echo "" ;;
                esac ;;
        esac
    fi

    echo "=== 5B $ARM | run=$RUN_NAME | phase=$PHASE | project=$OUT_DIR ==="
    echo "    base:       $MODEL"
    echo "    data:       $DATA_PATH   (READ THIS LINE -- the only proof the pack arrived)"
    echo "    horizon:    max_steps=$MAX_STEPS  stop_at_step=$STOP_AT_STEP  heal_end=$HEAL_END"
    echo "    geometry:   max_length=$MAX_LENGTH cross_chunks=$CROSS_CHUNKS" \
         "($(( MAX_LENGTH / CROSS_CHUNKS ))-tok chunks)"
    echo "    batch:      batch_size=$BATCH_SIZE micro_batch_size=$MICRO_BS" \
         "-> $(( BATCH_SIZE * MAX_LENGTH )) tok/step," \
         "$(( BATCH_SIZE / MICRO_BS )) accumulation micro-steps"
    echo "    tokens at horizon: $(( MAX_STEPS * BATCH_SIZE * MAX_LENGTH ))"
    echo "    schedule:   lr=$LR muon=$MUON_LR warmup_steps=$WARMUP_STEPS cooldown=$COOLDOWN"
    echo "    recurrence: full $MAX_MEAN_REC at step" \
         "$(python -c "import math;print(math.ceil($RAMP_WARMUP*$MAX_STEPS))") (inside the heal)"
    echo "    saves:      every $SAVE_INTERVAL (resumable every $(( 2 * SAVE_INTERVAL )))," \
         "diag every $DIAG_INTERVAL"
    [ -n "${ARM_BANNER:-}" ] && printf '%s\n' "$ARM_BANNER"
    echo "    extra:      '$EXTRA_ARGS'"
    echo "    NOTE an echoed flag is NOT proof it took effect: e_carry_read printed"
    echo "         correctly for two runs while doing nothing.  The measured proof"
    echo "         is cortex_diag.jsonl -- run tools/watch_run.py on this run dir."
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    git rev-parse --short HEAD 2>/dev/null | sed 's/^/    commit:     /'

    python train.py \
        --run_name $RUN_NAME \
        --out_path $OUT_DIR \
        --model_name $MODEL \
        --preprocessed_data_path $DATA_PATH \
        --max_length $MAX_LENGTH \
        --epochs 1 \
        --max_steps $MAX_STEPS \
        --stop_at_step $STOP_AT_STEP \
        --micro_batch_size $MICRO_BS \
        --batch_size $BATCH_SIZE \
        --optim_config.lr $LR \
        --scheduler_args.cooldown $COOLDOWN \
        --scheduler_args.warmup_steps $WARMUP_STEPS \
        --muon.use_muon true \
        --muon.lr $MUON_LR \
        --no_amp false \
        --max_grad_norm 1.0 \
        --compile false \
        --save_interval $SAVE_INTERVAL \
        --save_n_mins_before_timeout 20 \
        --seed $SEED \
        --mean_recurrence_schedule.turn_on true \
        --mean_recurrence_schedule.warmup $RAMP_WARMUP \
        --mean_recurrence_schedule.max_mean_rec $MAX_MEAN_REC \
        --mean_recurrence_schedule.warmup_type "1-sqrt" \
        --cortex.cross_chunks $CROSS_CHUNKS \
        --cortex.eos_from_tokens true \
        --cortex.diag_interval $DIAG_INTERVAL \
        $ARM_ARGS \
        $CHAIN_ARGS $EXTRA_ARGS
    RC=$?

    echo ""
    echo "=== link finished rc=$RC ==="
    python tools/watch_run.py --run "$RUN_DIR" --log "logs/Report-${SLURM_JOB_ID}.out" \
        --arm "$ARM" --max_steps "$MAX_STEPS" --heal_end "$HEAL_END" \
        ${ZNORM_TARGET:+--znorm_target $ZNORM_TARGET} || true

    echo ""
    echo "Next link:"
    LAST=$(ls -d "$RUN_DIR"/checkpoint_* 2>/dev/null | sed 's/.*checkpoint_//' \
           | sort -n | tail -1)
    SELF="pace/$( [ "$ARM" = "cortex" ] && echo cortex_final_5b.sbatch || echo control_5b.sbatch )"
    ZPASS=${ZNORM_TARGET:+ZNORM_TARGET=$ZNORM_TARGET }
    if [ -n "$LAST" ] && [ "$LAST" -ge "$HEAL_END" ] 2>/dev/null; then
        if [ "$LAST" -eq "$HEAL_END" ]; then
            echo "  PHASE=mix RESUME_PATH=$RUN_DIR/checkpoint_$LAST \\"
            echo "    EXTRA_ARGS=\"--reset_dataset_position true\" \\"
            echo "    ${ZPASS}sbatch $SELF          # <- THE SWITCH, ONCE"
        else
            echo "  PHASE=mix RESUME_PATH=$RUN_DIR/checkpoint_$LAST \\"
            echo "    ${ZPASS}sbatch $SELF"
        fi
    elif [ -n "$LAST" ]; then
        echo "  PHASE=heal RESUME_PATH=$RUN_DIR/checkpoint_$LAST ${ZPASS}sbatch $SELF"
        echo "  (stop the heal at $HEAL_END with STOP_AT_STEP=$HEAL_END on the"
        echo "   link that would otherwise run past it -- the horizon does NOT move)"
    else
        echo "  (no checkpoint written -- read the log before resubmitting)"
    fi
    return $RC
}
