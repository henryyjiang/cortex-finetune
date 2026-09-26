#!/bin/bash
#SBATCH --account=gts-aivanova7-lab
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H200:1
#SBATCH --mem=128GB
#SBATCH -t 8:00:00
#SBATCH -q inferno
#SBATCH -J j4_joint
#SBATCH -o logs/Report-%A_%a.out
#SBATCH --array=0-2
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=henryyjiang42@gmail.com

# J4 -- THE CARRIED Z AS ITS OWN `input_embeds` COLUMNS.  Findings doc,
# "Attempt 2", design J4 (final rank 2); j4_handoff.md step 2; the design is
# cortex_memory/latent_embed.py's docstring.  In one line: the carried Z rows
# get their own K columns beside E's, rescaled to E's measured row norm and
# mapped by one identity-initialised projection, and are read by the BASE
# MODEL's OWN pretrained attention on every one of the T loop iterations.
#
#   sbatch pace/j4_joint.sbatch                                # [0] real [1] donor [2] noread
#   E_DROPOUT=0.25 sbatch --array=0,2 pace/j4_joint.sbatch     # the E-dropout pair
#   E_DROPOUT=0.95 sbatch --array=0,2 pace/j4_joint.sbatch     # the Z-ONLY pair
#
# THREE LIMBS, identical in everything but what is IN the Z columns -- same
# branch, seed, data order, schedule, micro-batch, sequence length, position
# ids and parameter count:
#   real    the columns hold this document's carried Z
#   donor   they hold ANOTHER document's carried Z (batch roll, in
#           _latent_z_rows, so the WRITE is untouched)
#   noread  they hold ZEROS (--cortex.latent_carry_read false); Z is still
#           written to the ring every chunk
# NOT --cortex.latent_write_only (J1's no-read flag): that omits the columns,
# and the limb would then differ from the real one in sequence length, position
# ids and activation memory as well as in the carry.  The graft refuses it.
#
# WHAT J4 CHANGES FROM J3, and it is exactly two things:
#   1. THE WRITE MOVES to the loop endpoint at the SUMMARY columns
#      (latent_encoding=endpoint), which is what D3 vindicated -- see below.
#   2. THE READ IS THE PRETRAINED ATTENTION, not a module.  No gate, no
#      cross-attention, no new reader: 4.19M parameters of projection whose job
#      is coordinates and scale, and whose designed init is the IDENTITY.
# Everything else is J1's recipe, which is the P2.3 cell recipe: same branch,
# LRs, horizon, recurrence ramp, geometry, window_backward, data.
#
# ===== WHY, IN THE FEWEST NUMBERS THAT SETTLE IT ============================
# D3 (evals/diag_z_registers.py, job 13528227, four checkpoints) probed the
# register file out of every candidate write:
#   the write J1 and J3 CARRIED     ~1% of E's margin -- nothing.  Its only
#     (loop state / scratchpad at    flicker (+0.007) is on registers last
#      the chunk's last 64 tokens)   updated INSIDE those last 64 tokens and it
#                                    is flat (-0.001) on anything earlier: a
#                                    recency smear, not a register file.
#   Z_end (s_T at the SUMMARY       CI-backed above chance in all four
#     columns, the pre-coda twin      checkpoints, 22-54% of E's margin.
#     of E)
# So BOTH attempt-2 NOs are explained without blaming the read: there was never
# anything in the channel for a read to find.  THE READ WAS NEVER THE
# BOTTLENECK -- which is why J4 changes the write and then hands it to the best
# reader available rather than building a third one.
#
# WHAT IS STILL OPEN, and it is in j4_prereg.md rather than hidden here: D3
# showed Z_end's register information is PRESENT, never that it is ADDITIONAL
# to E (cortex_graft.py:1231's redundancy objection stands unrefuted).  If
# Z_end is largely a re-coded copy of E it still PASSES the E-off cell -- a copy
# carries the task when the original is blanked -- so a PASS here is read
# against pace/diag_z4_probes.sbatch's incremental probe, not on its own.  A
# NO needs no such qualification.
#
# ===== BEFORE THE FIRST LAUNCH ==============================================
#   1. PRICE THE MEMORY: `REQUIRE_FIT=1 sbatch pace/j4_price_oom.sbatch`.
#      J4 adds 64 packed columns (592 -> 656, +11% sequence) on top of J3's
#      92.4 GiB of 140, and drops J3's scratchpad state and its 16.8M gates.
#      Expect it to land near J1's 88.5 GiB, not above J3's -- but the +11%
#      sequence touches every loop iteration, so do not assume it.
#   2. A 10-step smoke, WITH DIAG_INTERVAL=1 AND SMOKE_GATE=1:
#        CELL_STEPS=10 DIAG_INTERVAL=1 SMOKE_GATE=1 RUN_SUFFIX=-smoke \
#            sbatch -t 0:45:00 pace/j4_joint.sbatch
#
#      DIAG_INTERVAL=1 IS NOT OPTIONAL ON A SHORT RUN, and this is a trap J3's
#      smoke fell into silently.  The diag fires on `optimizer_step % di == 0`
#      against the RESUMED ABSOLUTE counter, and 91552 % 25 = 2 -- so the first
#      row of every 25-interval run lands at 91575, i.e. 23 updates in.  A
#      10-step smoke at the default DIAG_INTERVAL writes NO ROWS AT ALL, and the
#      numbers below cannot be read (`logs/j3_diag/*.jsonl` all start at 91575).
#      At 1 the smoke writes one row per update, for ten SVDs of a [64, 2048]
#      matrix -- free.
#
#      SMOKE_GATE=1 then CHECKS those rows and exits nonzero if any of them is
#      wrong, so this job can gate the real launch with --dependency=afterok.
#      It is the only thing that can: `train.py` exits 0 on a run whose Z
#      entered 57x too small, with a perfectly healthy loss curve.  What it
#      requires, per limb (j4_prereg.md S1.1 and S5):
#        z_n_zpre          64 -- the Z columns were spliced at all
#        z_e_carried_norm  within 15% of ZNORM_TARGET -- S1.1's confirmation
#                          that 171.0 is the right number ON THIS BRANCH
#        z_embed_ratio     real/donor 0.5..2.0 (~1.0 by construction at init;
#                          0.06 would mean the rescale never ran).  noread
#                          EXACTLY 0 -- which is veto 6 measured directly
#        z_read_grad_frac  1.0, the structural claim of the whole design
#        z_write_grad_frac > 0, and a finite loss
#      SMOKE_CLEAN=1 additionally copies each diag to logs/j4_smoke/ and deletes
#      the smoke's CHECKPOINT dirs only (~17 GB each, worthless at 10 steps),
#      leaving the run dir and its diag to read in the morning.
#   3. J4's PRE-REGISTRATION is ../j4_prereg.md, written before any J4 run.
#      Its rules are evals/score_j1.py --experiment j4, including the embed-
#      ratio collapse audit (z_embed_ratio, READ_COLLAPSE_FRAC) and the
#      E-HEALTH check (E_CARRY_MAX) that is new to J4 and exists because this
#      is the first design whose Z write shares E's own columns.
#   4. Run the E-DROPOUT PAIR and the Z-ONLY PAIR.  Under J1's answer table the
#      E-off cell is the only cell that can return NO, and at E_DROPOUT 0.25
#      that pressure reaches only ~5% of rows (0.25 x the 20% carry rows).  The
#      Z-only pair at 0.95 asks whether the pathway can carry anything when it
#      MUST.
#
# ===== DECISIONS RECORDED 2026-09-25 (j4_prereg.md) =========================
# ENCODING = endpoint, fixed, and the graft REFUSES anything else under
#   latent_read=embeds: D3 measured the alternatives at ~1% of E's margin, and
#   reading a dead channel through a better reader would re-run attempt 2.
#   ('delta' at these same columns is the one genuinely unexamined write and is
#   j4_prereg.md S3's named alternative -- a different arm, not a flag to flip.)
# ZNORM_TARGET = 171.0, E's MEASURED carried row norm (fp32, B2).  Z_end rows
#   are ~10-11, so spliced as carried they would enter 17x under the field they
#   compete with -- which IS the s0 site's measured failure (flat out to 45x
#   ||s0||, turning on only at an injected 175.8 against ||E|| = 171, i.e. when
#   it drowns E rather than being read).  The rescale runs BEFORE the
#   projection, so the entering norm is exactly right at step 0 and is then free
#   to MOVE: z_embed_ratio is the readable negative, standing where J1's gate
#   and J3's read_ratio stood.  CONFIRM IT ON THE SMOKE: the diag's
#   z_e_carried_norm is the same quantity measured live, and if it is not ~171
#   on this branch then this target is the wrong number and the run is paused.
# NO GATE, and that is a decision.  J1's gate closed in ~270 updates at Adam's
#   full rate; J3 slowed it and the read stayed ~uniform.  A gate here would
#   only re-add the failure mode the pretrained attention does not have -- it
#   already decides per head and per position how much of E's columns to read,
#   and Z's columns are read by that same softmax.
# INIT = IDENTITY, not zero.  A zero-init read is the x0.90 shape this project
#   has paid for twice.  Identity + rescale means step 0 splices a correctly
#   scaled copy of the pre-coda summary state: a real read from the first batch.
# E_DROPOUT for the pair = 0.25, J1's and J3's value, so the three pairs are
#   directly comparable; the Z-only pair is 0.95.
# MEMORY: <fill in from pace/j4_price_oom.sbatch before the real launch>.

cd $SLURM_SUBMIT_DIR

module load anaconda3
module load cuda/12.1.1
conda activate cortex-retro

export SCRATCH=${SCRATCH:-$HOME/scratch}
export HF_HOME=$SCRATCH/hf_cache
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODEL=${MODEL:-ckpts/olmo-retrofit-cortex}
BRANCH_PATH=${BRANCH_PATH:-cortex-retrofit/retro-b2-heal/checkpoint_91552_w16}
if [ -n "$PROSE_ONLY" ]; then
    ARM_DATA=${ARM_DATA:-data/pg19_fw50_olmo_len4096}
else
    ARM_DATA=${ARM_DATA:-data/j1_pg19fw50_carry20_len4096}
fi
MAX_LENGTH=${MAX_LENGTH:-4096}
CROSS_CHUNKS=${CROSS_CHUNKS:-8}
ACCUM_VECS=${ACCUM_VECS:-16}
GATE_SLOTS=${GATE_SLOTS:-64}
CARRY_GRAD_CHUNKS=${CARRY_GRAD_CHUNKS:-4}  # 4 of 8, as J1 and J3 and the P2.3 cells
MAX_MEAN_REC=${MAX_MEAN_REC:-8}
RAMP_WARMUP=${RAMP_WARMUP:-0.25}
MAX_STEPS=${MAX_STEPS:-305176}             # the LR SCHEDULE horizon (P2.3's)
PARENT_STEP=${PARENT_STEP:-91552}
CELL_STEPS=${CELL_STEPS:-2000}             # J1's and J3's budget, so all three compare
STOP_AT_STEP=$(( PARENT_STEP + CELL_STEPS ))
SAVE_INTERVAL=${SAVE_INTERVAL:-1000}
DIAG_INTERVAL=${DIAG_INTERVAL:-25}
MICRO_BS=${MICRO_BS:-2}
BATCH_SIZE=${BATCH_SIZE:-4}
ZNORM=${ZNORM:-rms}                        # the rescale; the graft refuses 'none'
ZNORM_TARGET=${ZNORM_TARGET:-171.0}        # ||E|| measured -- see DECISIONS
E_DROPOUT=${E_DROPOUT:-0.0}
WINDOW_BACKWARD=${WINDOW_BACKWARD:-1}      # micro 2 did not fit without it (13456835)
WB_ARGS=""
if [ "$WINDOW_BACKWARD" = "1" ]; then WB_ARGS="--cortex.window_backward true"; fi
ENCODING=endpoint

if [ "$MICRO_BS" -lt 2 ]; then
    echo "ERROR: MICRO_BS=$MICRO_BS.  The donor limb rolls the carried Z along"
    echo "       the micro-batch; at 1 the roll returns the same row and the"
    echo "       donor limb IS the real limb.  All three limbs share MICRO_BS."
    exit 1
fi
[ -f "$BRANCH_PATH/chkpt.pt" ] || { echo "ERROR: no chkpt.pt under BRANCH_PATH=$BRANCH_PATH"; exit 1; }
[ -d "$ARM_DATA" ] || { echo "ERROR: no such pack: $ARM_DATA (build it: pace/prepare_carry_task.sbatch, or PROSE_ONLY=1)"; exit 1; }
# J4 needs NO new modeling-file hook: the Z columns are spliced inside
# prefix_pack, and the modeling file round-trips the prepended count into
# prefix_unpack as an opaque number.  The Z WRITE still needs iter_write, which
# is what records the loop state the endpoint encoding reads.
if ! grep -q "iter_write(x, current_step)" "$MODEL/raven_modeling_minimal_cortex.py" 2>/dev/null; then
    echo "ERROR: $MODEL carries a modeling file without iter_write(x,"
    echo "       current_step) -- the Z write would never fire and the run"
    echo "       would train a dual-channel buffer with a dead Z half.  Refresh:"
    echo "         python tools/prepare_cortex_checkpoint.py --src $MODEL --dst $MODEL --variant olmo"
    echo "         rm -rf \$SCRATCH/hf_cache/modules/transformers_modules/$(basename $MODEL)"
    exit 1
fi

LIMBS=(real donor noread)
IDX=${SLURM_ARRAY_TASK_ID:-0}
LIMB=${LIMBS[$IDX]}
if [ -z "$LIMB" ]; then echo "no limb at index $IDX (use --array=0-2)"; exit 0; fi

# Every limb has the Z columns; the limbs differ in what is in them.
READ_ARGS="--cortex.latent_read embeds"
READ_ARGS="$READ_ARGS --cortex.latent_read_znorm $ZNORM --cortex.latent_read_znorm_target $ZNORM_TARGET"
case "$LIMB" in
    real)   READ_ARGS="$READ_ARGS --cortex.latent_read_scramble false --cortex.latent_carry_read true" ;;
    donor)  READ_ARGS="$READ_ARGS --cortex.latent_read_scramble true  --cortex.latent_carry_read true" ;;
    noread) READ_ARGS="$READ_ARGS --cortex.latent_read_scramble false --cortex.latent_carry_read false" ;;
esac

TAG=""
if [ "$E_DROPOUT" != "0.0" ] && [ "$E_DROPOUT" != "0" ]; then TAG="-edrop${E_DROPOUT}"; fi
if [ -n "$PROSE_ONLY" ]; then TAG="$TAG-prose"; fi
# RUN_SUFFIX keeps a smoke (e.g. RUN_SUFFIX=-smoke) out of the real run's dir.
RUN_NAME=j4-a3z-${ENCODING}${TAG}-${LIMB}${RUN_SUFFIX:-}
ACCUM_MAX=$(( ACCUM_VECS * CROSS_CHUNKS ))

echo "=== J4 embeds run | limb=$LIMB | run=$RUN_NAME ==="
echo "    branch:    $BRANCH_PATH  (step $PARENT_STEP -> stop $STOP_AT_STEP, horizon $MAX_STEPS)"
echo "    data:      $ARM_DATA   (READ THIS LINE -- the only proof the pack arrived)"
echo "    geometry:  gated W=$ACCUM_VECS K=$GATE_SLOTS cc=$CROSS_CHUNKS grad_chunks=$CARRY_GRAD_CHUNKS mr=$MAX_MEAN_REC window_backward=$WINDOW_BACKWARD"
echo "    packed:    64 E + 64 Z + 512 tokens + 16 summary = 656 columns (was 592)"
echo "    read:      $READ_ARGS"
echo "    e_dropout: $E_DROPOUT   micro=$MICRO_BS batch=$BATCH_SIZE   diag every $DIAG_INTERVAL"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

python train.py \
    --run_name $RUN_NAME \
    --out_path cortex-retrofit \
    --model_name $MODEL \
    --preprocessed_data_path $ARM_DATA \
    --max_length $MAX_LENGTH \
    --epochs 1 \
    --max_steps $MAX_STEPS \
    --stop_at_step $STOP_AT_STEP \
    --micro_batch_size $MICRO_BS \
    --batch_size $BATCH_SIZE \
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
    --cortex.prefix_memory gated \
    --cortex.accum_vecs $ACCUM_VECS \
    --cortex.accum_max $ACCUM_MAX \
    --cortex.gate_slots $GATE_SLOTS \
    --cortex.gate_route ring \
    --cortex.gate_norm tanh \
    --cortex.gate_init zero \
    --cortex.gate_fill grow \
    --cortex.latent_carry true \
    --cortex.latent_depth_rule absolute \
    --cortex.latent_depth_lo 2 \
    --cortex.latent_depth_hi 9 \
    --cortex.latent_renorm none \
    --cortex.latent_encoding $ENCODING \
    --cortex.latent_s0_read false \
    $READ_ARGS \
    --cortex.e_dropout $E_DROPOUT \
    --cortex.diag_interval $DIAG_INTERVAL \
    --cortex.carry_grad_chunks $CARRY_GRAD_CHUNKS \
    $WB_ARGS \
    --cortex.cross_chunks $CROSS_CHUNKS \
    --cortex.eos_from_tokens true \
    --cortex.prefix_eos_reset false \
    --cortex.prefix_pos tail \
    --cortex.freeze_loop false \
    --branch_path $BRANCH_PATH
RC=$?

# ===== THE SMOKE GATE ========================================================
# Only with SMOKE_GATE=1, so a real run is bit-identical without it.  Turns the
# by-eye go/no-go of j4_prereg.md stage 2 into an exit code, which is what lets
# the real limbs be queued --dependency=afterok on the smoke and left overnight.
if [ -n "$SMOKE_GATE" ] && [ $RC -eq 0 ]; then
    echo ""
    echo "=== SMOKE GATE | $RUN_NAME | limb=$LIMB ==="
    python - "cortex-retrofit/$RUN_NAME/cortex_diag.jsonl" "$LIMB" \
             "$ZNORM_TARGET" "$GATE_SLOTS" <<'GATE'
import json, math, sys

path, limb, target, k = sys.argv[1], sys.argv[2], float(sys.argv[3]), int(sys.argv[4])
try:
    rows = [json.loads(l) for l in open(path) if l.strip()]
except FileNotFoundError:
    sys.exit(f"FAIL: no {path}.  At DIAG_INTERVAL 25 a 10-step run writes no "
             f"rows at all (the counter is absolute: 91552 % 25 = 2, so the "
             f"first row would be 91575).  Re-run the smoke with DIAG_INTERVAL=1.")
if not rows:
    sys.exit(f"FAIL: {path} is empty.")
fails, notes = [], []


def vals(key):
    return [r[key] for r in rows if r.get(key) is not None]


notes.append(f"{len(rows)} diag rows, steps {rows[0]['step']}..{rows[-1]['step']}")

# 1. the columns exist at all
zpre = vals("z_n_zpre")
if not zpre:
    fails.append("z_n_zpre absent from every row -- this build predates J4")
elif max(zpre) != k:
    fails.append(f"z_n_zpre max {max(zpre)}, expected {k}: the Z columns were "
                 f"never spliced at full width")
else:
    notes.append(f"z_n_zpre max {max(zpre)} of {k}")

# 2. ZNORM_TARGET is the right number on THIS branch (j4_prereg.md S1.1)
en = vals("z_e_carried_norm")
if not en:
    fails.append("z_e_carried_norm absent -- cannot confirm ZNORM_TARGET")
else:
    m = sum(en) / len(en)
    notes.append(f"z_e_carried_norm mean {m:.2f} vs target {target:.2f} "
                 f"({100 * m / target:.0f}%)")
    if not (0.85 * target <= m <= 1.15 * target):
        fails.append(f"||E|| measures {m:.2f} but ZNORM_TARGET is {target:.2f} "
                     f"-- more than 15% apart.  j4_prereg.md S1.1 says STOP: "
                     f"the rescale target is the wrong number for this branch, "
                     f"and every limb would train with Z at the wrong scale.")

# 3. Z is entering at E's scale (real/donor) or not at all (noread)
er = vals("z_embed_ratio")
if not er:
    fails.append("z_embed_ratio absent -- J4's only read-strength number")
elif limb == "noread":
    if max(abs(v) for v in er) != 0.0:
        fails.append(f"noread limb has z_embed_ratio up to {max(er):.4g}, "
                     f"expected EXACTLY 0.  Its Z columns must be zeros -- this "
                     f"is veto 6 (the no-read limb reads something).")
    else:
        notes.append("z_embed_ratio exactly 0 (no-read limb reads nothing)")
else:
    first = er[0]
    notes.append(f"z_embed_ratio {first:.4f} -> {er[-1]:.4f}")
    if not (0.5 <= first <= 2.0):
        fails.append(f"z_embed_ratio starts at {first:.4f}, outside 0.5..2.0.  "
                     f"It is ~1.0 by construction when the rescale runs; ~0.06 "
                     f"means it did not, and Z is entering 17x under E.")

# 4. the design's structural claim: the read has gradient on EVERY batch
rg = vals("z_read_grad_frac")
if not rg:
    fails.append("z_read_grad_frac absent")
elif min(rg) < 1.0:
    fails.append(f"z_read_grad_frac min {min(rg)}, expected 1.0.  J4's columns "
                 f"live in input_embeds and are re-fed every iteration, so the "
                 f"read is live however long the no-grad prefix is.  Below 1.0 "
                 f"means the graft is not reporting the site it ran.")
else:
    notes.append("z_read_grad_frac 1.0 (live on every batch, as E is)")

wg = vals("z_write_grad_frac")
if wg and max(wg) <= 0:
    fails.append(f"z_write_grad_frac max {max(wg)}: the write never trained")
elif wg:
    notes.append(f"z_write_grad_frac max {max(wg):.3f}")

# 5. and the loss is a number
losses = vals("loss")
if not losses:
    fails.append("no loss in any diag row")
elif not all(math.isfinite(v) for v in losses):
    fails.append("a non-finite loss")
else:
    notes.append(f"loss {losses[0]:.4f} -> {losses[-1]:.4f}, all finite")

for n in notes:
    print(f"  ok   {n}")
for f in fails:
    print(f"  FAIL {f}")
if fails:
    sys.exit(f"\nSMOKE GATE FAILED ({len(fails)}): do NOT launch the real limbs "
             f"until this is understood.  j4_prereg.md S1.1 / S5.")
print("\nSMOKE GATE PASSED -- the real limbs may launch.")
GATE
    RC=$?
    if [ $RC -ne 0 ]; then
        echo "J4 $LIMB SMOKE GATE FAILED -- exit $RC, so anything queued"
        echo "  --dependency=afterok on this job will NOT start."
        exit $RC
    fi
    if [ -n "$SMOKE_CLEAN" ]; then
        # Keep the evidence, drop the bulk: a 10-step checkpoint is worthless
        # and three of them are ~50 GB of scratch overnight.
        mkdir -p logs/j4_smoke
        cp "cortex-retrofit/$RUN_NAME/cortex_diag.jsonl" \
           "logs/j4_smoke/${RUN_NAME}.jsonl" 2>/dev/null
        rm -rf cortex-retrofit/$RUN_NAME/checkpoint_*
        echo "  diag kept at logs/j4_smoke/${RUN_NAME}.jsonl; checkpoints removed"
        du -sh "cortex-retrofit/$RUN_NAME" 2>/dev/null
    fi
fi

echo ""
if [ $RC -eq 0 ]; then
    echo "J4 $LIMB DONE -> cortex-retrofit/$RUN_NAME"
    echo "  In-flight record: cortex-retrofit/$RUN_NAME/cortex_diag.jsonl"
    echo "  READ THESE FIRST: z_n_zpre (64 = the columns are spliced),"
    echo "  z_embed_ratio (||Z col||/||E col||, ~1.0 at step 0 -- J4's ONLY"
    echo "  read-strength number, and a collapse is an AUDIT beside a positive),"
    echo "  z_e_carried_norm (confirms ZNORM_TARGET was the right number),"
    echo "  z_write_grad_frac, z_read_grad_frac (1.0 by construction here),"
    echo "  e_dropped, E rank."
    echo "  Read-out: EXPERIMENT=j4 pace/j1_readout.sbatch, then EXPERIMENT=j4"
    echo "  pace/j1_verdict.sbatch -- and only under J4's pre-registration."
else
    echo "J4 $LIMB FAILED rc=$RC"
fi
exit $RC
