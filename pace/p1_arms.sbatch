#!/bin/bash
#SBATCH --account=gts-aivanova7-lab
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=64GB
#SBATCH --gres=gpu:H200:1
#SBATCH --constraint=H200
#SBATCH -t 48:00:00
#SBATCH -q inferno
#SBATCH -o logs/Report-%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=henryyjiang42@gmail.com

# P1.0 — A1 (accum) vs A3' (gated ring), the buffer-geometry pair.
# control_launch_handoff.md §4, p10_gated_prereg.md.  Sibling of
# b2_retrofit.sbatch, which it deliberately does NOT extend: B2's arm phase is
# a 5B-token conversion at W=32 and these are ~400M-token mechanism cells at
# W=16 off a shared, already-healthy parent.  Folding them into one script would
# make every future edit to either one a risk to the other.
#
#   ARM=a1 BRANCH_PATH=<sliced heal ckpt> sbatch pace/p1_arms.sbatch
#   ARM=a3 BRANCH_PATH=<sliced heal ckpt> sbatch pace/p1_arms.sbatch
#   ARM=a1 RESUME_PATH=cortex-retrofit/p1-a1-accum/checkpoint_XXXX sbatch ...
#
#   # the Z channel (dual-channel carry).  PROBE=1 is the short architecture
#   # probe -- see THE Z ARMS below; it is what runs before a full Z launch.
#   PROBE=1 ARM=a2  BRANCH_PATH=<sliced heal ckpt> sbatch pace/p1_arms.sbatch
#   PROBE=1 ARM=a3z BRANCH_PATH=<sliced heal ckpt> sbatch pace/p1_arms.sbatch
#
# THE PAIR
# --------
#   A1   accum,      W 16, accum_max 128  ->  128 read cols, 656 packed
#   A3'  gated ring, W 16, gate_slots 64  ->   64 read cols, 592 packed
#
# Everything else is held: same parent, same corpus, same step window, same
# cross_chunks 8 / 512-token chunks, same carry_grad_chunks 4, same mr8, same
# seed, same LR schedule position.  `fill="grow"` makes A3' BIT-IDENTICAL to A1
# for the first four chunks, so the pair diverges at exactly the chunk where
# accum starts dropping its oldest rows and the ring starts gating them.
#
# THE ONE ASYMMETRY, and it is structural rather than a configuration mistake.
# Two asserts bound K from opposite sides: a no-trim accum needs
# K >= cross_chunks x W, a gated ring needs K <= cross_chunks x W / 2 (it must
# complete two laps or the gate never fires).  THEY NEVER OVERLAP.  A gated ring
# is always at most HALF the read block of a no-trim accum on the same chain.
# So this pair is inherently "cheaper AND?" and never "equal cost AND?", and the
# writeup has to say so before a reviewer asks.  Both numbers above are the
# tightest each assert allows, which is the closest the pair can get.
#
# BOTH ARMS BRANCH FROM A SLICED CHECKPOINT
# -----------------------------------------
# retro-b2-heal/checkpoint_91552 is a cc4/128 run at W=32.  W is summary_emb's
# SHAPE, so W=16 needs tools/slice_summary_emb.py first:
#
#   python tools/slice_summary_emb.py \
#       --src cortex-retrofit/retro-b2-heal/checkpoint_91552 \
#       --dst cortex-retrofit/retro-b2-heal/checkpoint_91552_w16 --n_vec 16
#
# and BOTH arms must branch from the SAME sliced dir, or they no longer share an
# ancestor and the comparison is gone.  The pre-flight below refuses to start if
# BRANCH_PATH is not a sliced checkpoint, because the alternative failure is
# silent: load_state_dict(strict=False) would drop a mismatched summary_emb, the
# graft would re-seed all 16 rows from wte[eos], and the arm would throw away
# 91,552 steps of write-path training behind a perfectly healthy loss curve.
#
# PRE-FLIGHT, in order, and none of them is optional
#   1. sbatch pace/smoke_prefix_real.sbatch          # BOTH arms, modeling file
#   2. P1=1 GEOMETRIES= sbatch pace/smoke_geometry_oom.sbatch
#   3. python tools/slice_summary_emb.py ...         # and READ the rank line
#   4. python tools/check_pack.py --data $ARM_DATA --steps <STOP-91552> \
#          --batch_size 4 --max_length 4096
#
# WHY NOT THE MATH PACK.  B2's arm phase ran Nemotron math and the math arm came
# out net negative; P0.3/P0.4 closed the corpus question in favour of the 50/50
# mix, which is also what the C-chunked control anneals on.  These cells are
# about the BUFFER, so they take the corpus every other current run takes.  The
# one thing that matters is that A1 and A3' take the SAME one — override
# ARM_DATA if you must, but override it for both.
#
# THE Z ARMS (a2 = accum + Z, a3z = gated ring + Z), and what they are NOT
# ------------------------------------------------------------------------
# Z carries the recurrence's TRAJECTORY alongside the token-space summary and
# reads it by substituting into `s0` at the carried columns -- no new parameters
# on the read, no extra sequence length, nothing zero-initialised.  The carried
# tensor widens to 2D (E at [...,:D], Z at [...,D:]); the ROW COUNT and
# therefore the read cost do not move at all.
#
# A1 vs A2 isolates "does Z exist at all" with the buffer held at accum.
# A2 vs A3z isolates the merge rule with Z held on.  A1 vs A3z moves both and is
# not a headline.  RUN a2 BEFORE a3z if the depth-slice ablation matters to you:
# accum rows are write-once, so per-chunk slices stay separable there.  (The
# depth->row map survives gating either way -- that is what `depth_rows` is for,
# and the two slices were conflated in earlier notes.)
#
# TWO THINGS TO SETTLE BEFORE A FULL Z LAUNCH, both cheap and both upstream:
#
#   1. P0.7, `pace/diag_depth_band.sbatch`.  latent_depth_rule defaults to
#      "absolute" because that is what P0.1 measured -- at a SINGLE T, which
#      cannot distinguish "steps 2..9 whatever T is" from "the first quarter of
#      the loop".  The two diverge sharply at T=32.  The flag exists so the
#      answer is a config change; running before P0.7 reports means asserting an
#      answer you do not have, and train.py prints a banner saying so.
#   2. THE GRADIENT WINDOW.  The no-grad loop runs FIRST, so the trainable
#      region is always the LAST mean_backprop_depth (8) iterations.  At mr8 --
#      these checkpoints -- the whole loop is in the gradient and the write band
#      2..9 trains.  At mr32 only steps 25-32 train, which are exactly the
#      saturated ones, and Z becomes a FROZEN FEATURE EXTRACTOR: the model can
#      learn to USE it, never to SHAPE it.  That is the mirror of the frozen-READ
#      failure that cost x0.90 -> x1.33.  The probe reports the measured fraction
#      (latent_write_grad_frac); decide it, do not discover it.
#
# PROBE=1 is the ARCHITECTURE probe, not a cell of the experiment.  It runs a
# few hundred steps and then tools/check_latent_probe.py, which answers the only
# questions a short run CAN answer: did the loss move, did the Z gate leave its
# init, did summary_emb keep training, is anything non-finite.  It cannot say
# whether Z HELPS -- that needs the full cell and the 2x2 -- and the checker says
# so rather than letting a green probe read as a positive result.
#
# BUDGET.  ~400M tokens per cell at 16,384 tokens/step = 24,414 steps, ending at
# 115,966.  MAX_STEPS stays at B2's 305,176 because that is the LR SCHEDULE's
# horizon and a branch continues the schedule; STOP_AT_STEP is what ends the
# cell.  Shortening MAX_STEPS instead would put both arms on a different (much
# steeper) LR decay than the parent they branched from — a second variable.

cd $SLURM_SUBMIT_DIR

module load anaconda3
module load cuda/12.1.1
conda activate cortex-retro

export SCRATCH=${SCRATCH:-$HOME/scratch}
export HF_HOME=$SCRATCH/hf_cache
export HF_HUB_OFFLINE=1
export WANDB_DIR=$SCRATCH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=$(python -c "import socket;s=socket.socket();s.bind(('',0));print(s.getsockname()[1]);s.close()")
echo "MASTER_PORT=$MASTER_PORT"

# out_path MUST live on scratch — a relative name resolves into $HOME and an
# update-500 checkpoint blowing the home quota is what killed the 2026-07-17
# round.  Keep the relative name (it is also the wandb project) and symlink it.
OUT_DIR=cortex-retrofit
if [ ! -L "$OUT_DIR" ]; then
    if [ -e "$OUT_DIR" ]; then
        echo "ERROR: $OUT_DIR exists in \$HOME but is not a symlink to scratch."
        exit 1
    fi
    mkdir -p "$SCRATCH/cortex-retrofit"
    ln -s "$SCRATCH/cortex-retrofit" "$OUT_DIR"
fi

ARM=${ARM:-}
MODEL=${MODEL:-ckpts/olmo-retrofit-cortex}
ARM_DATA=${ARM_DATA:-data/pg19_fw50_olmo_len4096}
MAX_LENGTH=${MAX_LENGTH:-4096}
CROSS_CHUNKS=${CROSS_CHUNKS:-8}            # 512-token chunks
ACCUM_VECS=${ACCUM_VECS:-16}               # W.  A PARAMETER SHAPE — never vary mid-chain.
CARRY_GRAD_CHUNKS=${CARRY_GRAD_CHUNKS:-4}  # 4 of 8, B2's 50% convention
MAX_MEAN_REC=${MAX_MEAN_REC:-8}
RAMP_WARMUP=${RAMP_WARMUP:-0.25}
MAX_STEPS=${MAX_STEPS:-305176}             # the LR SCHEDULE horizon, not the budget
PARENT_STEP=${PARENT_STEP:-91552}
CELL_STEPS=${CELL_STEPS:-24414}            # ~400M tokens at 16,384 tok/step
STOP_AT_STEP=${STOP_AT_STEP:-$(( PARENT_STEP + CELL_STEPS ))}
SAVE_INTERVAL=${SAVE_INTERVAL:-2500}
RESUME_PATH=${RESUME_PATH:-}
BRANCH_PATH=${BRANCH_PATH:-}
EXTRA_ARGS=${EXTRA_ARGS:-}
# The periodic architecture diagnostic (train.py --cortex.diag_interval).  OFF
# for a full cell -- it costs an SVD of the carry per call and every arm on
# record ran without it -- and ON under PROBE, because the whole point of the
# probe is the trajectory of these numbers over the first few hundred steps.
DIAG_INTERVAL=${DIAG_INTERVAL:-0}

LATENT=0
LATENT_ARGS=""
case "$ARM" in
a1|a2)
    PREFIX_MODE=accum
    ACCUM_MAX=${ACCUM_MAX:-128}            # 8 x 16 — the assert is exact, no trim
    GATE_ARGS=""
    RUN_NAME=${RUN_NAME:-p1-${ARM}-accum-w${ACCUM_VECS}-cc${CROSS_CHUNKS}};;
a3|a3z)
    PREFIX_MODE=gated
    GATE_SLOTS=${GATE_SLOTS:-64}
    ACCUM_MAX=${ACCUM_MAX:-128}            # inert for a gated buffer; see below
    GATE_ARGS="--cortex.gate_slots $GATE_SLOTS \
               --cortex.gate_route ${GATE_ROUTE:-ring} \
               --cortex.gate_norm ${GATE_NORM:-tanh} \
               --cortex.gate_init ${GATE_INIT:-zero} \
               --cortex.gate_fill ${GATE_FILL:-grow}"
    RUN_NAME=${RUN_NAME:-p1-${ARM}-gated-w${ACCUM_VECS}k${GATE_SLOTS}-cc${CROSS_CHUNKS}};;
*)
    echo "ERROR: set ARM to one of:"
    echo "    a1   accum, E only        (the family control)"
    echo "    a3   gated ring, E only   (the buffer-geometry cell)"
    echo "    a2   accum + Z            (does Z exist at all)"
    echo "    a3z  gated ring + Z       (accum vs the gate, with Z held on)"
    exit 1;;
esac

case "$ARM" in
a2|a3z)
    LATENT=1
    LATENT_ARGS="--cortex.latent_carry true \
                 --cortex.latent_depth_rule ${LATENT_RULE:-absolute} \
                 --cortex.latent_depth_lo ${LATENT_LO:-2} \
                 --cortex.latent_depth_hi ${LATENT_HI:-9} \
                 --cortex.latent_renorm ${LATENT_RENORM:-none}"
    RUN_NAME="${RUN_NAME}-z";;
esac


# THE ARM'S OWN GEOMETRY, for the post-probe checks.  NOT optional, and the
# failure is not subtle: a probe checkpoint dir holds only chkpt.pt, so the
# config comes from $MODEL -- the BASE, which says W=32, accum, E-only.  Loading
# this arm's [W, D] summary_emb against it is a size mismatch, so the walk and
# the gates would die AFTER the 400 steps were already paid for.  And if the
# widths happened to agree, the checks would rebuild a DIFFERENT buffer from the
# same weights and report on a geometry that never trained -- the exact hazard
# tools/prepare_eval_checkpoint.py's CORTEX_FLAGS list exists for.
#
# Built from the SAME variables the training command uses, immediately after
# them, so the two cannot drift.
# use_memory IS THE MASTER SWITCH (cortex_graft.memory_enabled reads only
# this).  Without it the graft builds NOTHING, the checkpoint's cortex
# tensors load as "unexpected keys" and are dropped, and every check runs
# against a no-memory baseline -- which is how job 13266470 failed all
# three of its post-load steps at once.
PROBE_SETS="--set use_memory=true --set memory_slots=0"
PROBE_SETS="$PROBE_SETS --set prefix_memory=$PREFIX_MODE"
PROBE_SETS="$PROBE_SETS --set accum_vecs=$ACCUM_VECS"
if [ "$PREFIX_MODE" = "accum" ]; then
    PROBE_SETS="$PROBE_SETS --set accum_max=$ACCUM_MAX"
else
    PROBE_SETS="$PROBE_SETS --set gate_slots=$GATE_SLOTS"
    PROBE_SETS="$PROBE_SETS --set gate_route=${GATE_ROUTE:-ring}"
    PROBE_SETS="$PROBE_SETS --set gate_norm=${GATE_NORM:-tanh}"
    PROBE_SETS="$PROBE_SETS --set gate_init=${GATE_INIT:-zero}"
    PROBE_SETS="$PROBE_SETS --set gate_fill=${GATE_FILL:-grow}"
fi
if [ "$LATENT" = "1" ]; then
    PROBE_SETS="$PROBE_SETS --set latent_carry=true"
    PROBE_SETS="$PROBE_SETS --set latent_depth_rule=${LATENT_RULE:-absolute}"
    PROBE_SETS="$PROBE_SETS --set latent_depth_lo=${LATENT_LO:-2}"
    PROBE_SETS="$PROBE_SETS --set latent_depth_hi=${LATENT_HI:-9}"
    PROBE_SETS="$PROBE_SETS --set latent_renorm=${LATENT_RENORM:-none}"
fi

if [ -n "$PROBE" ]; then
    # The architecture probe: short, and named so it can never be mistaken for a
    # cell of the experiment in wandb or on disk.
    CELL_STEPS=${CELL_STEPS:-400}
    STOP_AT_STEP=$(( PARENT_STEP + CELL_STEPS ))
    SAVE_INTERVAL=${SAVE_INTERVAL:-200}
    DIAG_INTERVAL=${DIAG_INTERVAL:-10}
    if [ "$DIAG_INTERVAL" = "0" ]; then DIAG_INTERVAL=10; fi
    RUN_NAME="probe-${RUN_NAME}"
fi

# ---- pre-flight, all of it local and all of it cheap ----------------------
if [ -n "$RESUME_PATH" ] && [ -n "$BRANCH_PATH" ]; then
    echo "ERROR: set RESUME_PATH or BRANCH_PATH, not both."; exit 1
fi
if [ -z "$RESUME_PATH" ] && [ -z "$BRANCH_PATH" ]; then
    echo "ERROR: a P1 cell is never launched fresh — it BRANCHES from the"
    echo "  shared heal checkpoint, or it RESUMES itself.  Starting from the"
    echo "  base would compare two conversions instead of two buffers."
    exit 1
fi

CHAIN_ARGS=""
if [ -n "$RESUME_PATH" ]; then
    if [ ! -f "$RESUME_PATH/chkpt.pt" ]; then
        echo "ERROR: no chkpt.pt under RESUME_PATH=$RESUME_PATH"; exit 1
    fi
    CHAIN_ARGS="--resume_path $RESUME_PATH"
    echo "Resuming $RUN_NAME from $RESUME_PATH"
fi
if [ -n "$BRANCH_PATH" ]; then
    if [ ! -f "$BRANCH_PATH/chkpt.pt" ]; then
        echo "ERROR: no chkpt.pt under BRANCH_PATH=$BRANCH_PATH"; exit 1
    fi
    # THE CHECK THAT SAVES THE ARM.  A branch whose summary_emb width does not
    # match ACCUM_VECS loses the trained write path silently: strict=False drops
    # the mismatched key (a size mismatch raises, but a MISSING key does not),
    # and the graft re-seeds from wte[eos] behind a healthy loss curve.  Read the
    # actual tensor rather than trusting the directory name.
    python - "$BRANCH_PATH" "$ACCUM_VECS" <<'PREFLIGHT' || exit 1
import sys, torch
path, want = sys.argv[1], int(sys.argv[2])
ck = torch.load(f"{path}/chkpt.pt", map_location="cpu", weights_only=False)
w = ck["model"].get("cortex.prefix.summary_emb")
if w is None:
    print("ERROR: branch checkpoint has no cortex.prefix.summary_emb — it is "
          "not a prefix-memory run."); raise SystemExit(1)
if int(w.shape[0]) != want:
    print(f"ERROR: branch summary_emb is [{w.shape[0]}, {w.shape[1]}] but this "
          f"arm runs accum_vecs={want}.  Slice it first:\n"
          f"  python tools/slice_summary_emb.py --src {path} "
          f"--dst {path}_w{want} --n_vec {want}\n"
          f"Launching anyway would DROP the trained write path and re-seed 16 "
          f"rows from wte[eos], with no symptom in the loss curve.")
    raise SystemExit(1)
sl = ck.get("cortex_slice")
print(f"[preflight] summary_emb [{w.shape[0]}, {w.shape[1]}] OK; "
      + (f"sliced {sl['W_old']} -> {sl['W_new']} on {sl['when']} "
         f"(eff. rank {sl['kept']['participation_rank']:.2f} of "
         f"{sl['all']['participation_rank']:.2f})"
         if sl else "not a sliced checkpoint (parent trained at this width)"))
PREFLIGHT
    CHAIN_ARGS="--branch_path $BRANCH_PATH"
    echo "Branching $RUN_NAME from $BRANCH_PATH"
fi

if [ $(( MAX_LENGTH % CROSS_CHUNKS )) -ne 0 ]; then
    echo "ERROR: MAX_LENGTH ($MAX_LENGTH) is not divisible by CROSS_CHUNKS" \
         "($CROSS_CHUNKS) — chunks would be ragged."; exit 1
fi
if [ "$PREFIX_MODE" = "accum" ] && [ $(( CROSS_CHUNKS * ACCUM_VECS )) -gt "$ACCUM_MAX" ]; then
    echo "ERROR: cross_chunks x accum_vecs ($(( CROSS_CHUNKS * ACCUM_VECS )))" \
         "exceeds accum_max ($ACCUM_MAX): early chunks would be silently"
    echo "  trimmed and the stop-grad slice bookkeeping would break."
    echo "  Raise ACCUM_MAX.  Do NOT lower ACCUM_VECS — it is a parameter shape."
    exit 1
fi
if [ "$PREFIX_MODE" = "gated" ]; then
    if [ $(( GATE_SLOTS % ACCUM_VECS )) -ne 0 ]; then
        echo "ERROR: gate_slots ($GATE_SLOTS) must be a multiple of accum_vecs" \
             "($ACCUM_VECS) under gate_route=ring."; exit 1
    fi
    LAP=$(( GATE_SLOTS / ACCUM_VECS ))
    if [ "$CROSS_CHUNKS" -lt $(( 2 * LAP )) ]; then
        echo "ERROR: cross_chunks ($CROSS_CHUNKS) < 2 laps ($(( 2 * LAP )))."
        echo "  The gate first acts on lap 2, so this chain would train an"
        echo "  ACCUM buffer with dead gate parameters attached — behind a"
        echo "  perfectly healthy loss curve.  train.py asserts the same bound."
        exit 1
    fi
fi

echo "=== P1 $RUN_NAME | arm=$ARM | prefix=$PREFIX_MODE | data=$ARM_DATA ==="
echo "    geometry: cross_chunks=$CROSS_CHUNKS ($(( MAX_LENGTH / CROSS_CHUNKS ))-tok chunks)" \
     " W=$ACCUM_VECS carry_grad_chunks=$CARRY_GRAD_CHUNKS"
if [ "$LATENT" = "1" ]; then
    echo "    Z channel ON: carry widens to 2D, rows and read cost UNCHANGED."
    echo "      depth rule=${LATENT_RULE:-absolute} band=${LATENT_LO:-2}..${LATENT_HI:-9}" \
         "renorm=${LATENT_RENORM:-none}"
    echo "      the rule is P0.7's question (pace/diag_depth_band.sbatch)."
    echo "      mr=$MAX_MEAN_REC with mean_backprop_depth 8: the trainable region"
    echo "      is the LAST 8 loop steps, so at mr8 the whole write band trains."
fi
if [ "$PREFIX_MODE" = "gated" ]; then
    echo "    gate: K=$GATE_SLOTS route=${GATE_ROUTE:-ring} norm=${GATE_NORM:-tanh}" \
         "init=${GATE_INIT:-zero} fill=${GATE_FILL:-grow}; lap=$LAP chunks," \
         "gate fires from chunk $LAP"
    echo "    read block 64 cols vs A1's 128 — the pair is 'cheaper AND?',"
    echo "    never 'equal cost AND?'.  See the header."
else
    echo "    buffer: accum_max=$ACCUM_MAX (no trim: $CROSS_CHUNKS x $ACCUM_VECS)"
fi
echo "    steps: parent $PARENT_STEP -> stop $STOP_AT_STEP ($CELL_STEPS steps," \
     "$(( CELL_STEPS * 4 * MAX_LENGTH )) tokens); schedule horizon $MAX_STEPS"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python train.py \
    --run_name $RUN_NAME \
    --out_path cortex-retrofit \
    --model_name $MODEL \
    --preprocessed_data_path $ARM_DATA \
    --max_length $MAX_LENGTH \
    --epochs 1 \
    --max_steps $MAX_STEPS \
    --stop_at_step $STOP_AT_STEP \
    --micro_batch_size 1 \
    --batch_size 4 \
    --optim_config.lr 5e-5 \
    --scheduler_args.cooldown 0.6 \
    --scheduler_args.warmup 0.0025 \
    --muon.use_muon true \
    --muon.lr 0.001 \
    --no_amp false \
    --max_grad_norm 1.0 \
    --compile false \
    --save_interval $SAVE_INTERVAL \
    --save_n_mins_before_timeout 20 \
    --seed 74 \
    --mean_recurrence_schedule.turn_on true \
    --mean_recurrence_schedule.warmup $RAMP_WARMUP \
    --mean_recurrence_schedule.max_mean_rec $MAX_MEAN_REC \
    --mean_recurrence_schedule.warmup_type "1-sqrt" \
    --cortex.use_memory true \
    --cortex.memory_lr 5e-4 \
    --cortex.memory_slots 0 \
    --cortex.prefix_memory $PREFIX_MODE \
    --cortex.accum_vecs $ACCUM_VECS \
    --cortex.accum_max $ACCUM_MAX \
    $GATE_ARGS \
    $LATENT_ARGS \
    --cortex.diag_interval $DIAG_INTERVAL \
    --cortex.carry_grad_chunks $CARRY_GRAD_CHUNKS \
    --cortex.cross_chunks $CROSS_CHUNKS \
    --cortex.eos_from_tokens true \
    --cortex.prefix_eos_reset false \
    --cortex.prefix_pos tail \
    --cortex.freeze_loop false \
    $CHAIN_ARGS $EXTRA_ARGS
RC=$?

if [ -n "$PROBE" ] && [ $RC -eq 0 ]; then
    echo ""
    echo "######## architecture probe: did it actually train? ########"
    # Numerically, never lexically: checkpoint_9500 must not outrank
    # checkpoint_150000.
    LAST=$(ls -d cortex-retrofit/$RUN_NAME/checkpoint_* 2>/dev/null \
           | sort -t_ -k2 -n | tail -1)
    if [ -z "$LAST" ]; then
        echo "no checkpoint written - nothing to check."; exit 1
    fi
    OUT=eval_results/probe-$RUN_NAME
    mkdir -p $OUT

    # 1. DID THE PARAMETER BLOCKS MOVE off their inits, and is summary_emb
    #    CONTINUED rather than re-seeded?  The branch-width failure shows up
    #    here as an enormous parent-to-probe distance, nowhere else.
    python tools/check_latent_probe.py --probe "$LAST" --parent "$BRANCH_PATH" \
        || RC=1

    # 2. THE INSTRUMENTED WALK.  A printed table of what the packed forward did
    #    to each channel on each chunk -- rows touched against the ring rule,
    #    write norms as a ratio to ||s0||, the tape and its depths, whether s0's
    #    head really equals the carried Z, and the per-parameter grad norms.
    #    Read it; it is not a threshold.
    python evals/diag_dual_channel_walk.py --model_name $MODEL $PROBE_SETS \
        --checkpoint "$LAST" --chunks $CROSS_CHUNKS --chunk_len 256 \
        --T $MAX_MEAN_REC --out $OUT/walk.json || RC=1

    # 3. THE PRE-LAUNCH GATES.  Z-off equivalence (the only instrument that can
    #    localise Z perturbing E), the config round trip (bug class 2c, which no
    #    unit test can catch), the MEASURED no-grad split that goes into the Z
    #    pre-registration, and the per-channel donor control (P0.6 -- it never
    #    ran on B2).  Each one can come back negative.
    python tools/prelaunch_final.py --model_name $MODEL $PROBE_SETS \
        --checkpoint "$LAST" --chunks $CROSS_CHUNKS --chunk_len 256 \
        --T $MAX_MEAN_REC --out $OUT/prelaunch.json || RC=1

    echo ""
    echo "When every arm's probe has run, compare them against A1 -- the arm"
    echo "carrying B2's buffer, and the only valid stand-in for 'vs B2' at"
    echo "matched parent, corpus, step window and seed:"
    echo "  bash pace/submit_p1_probes.sh --compare"
fi

exit $RC
