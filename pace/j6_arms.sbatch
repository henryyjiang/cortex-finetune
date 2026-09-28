#!/bin/bash
#SBATCH --account=gts-aivanova7-lab
#SBATCH -N 1
#SBATCH --gres=gpu:H200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128GB
#SBATCH -t 6:00:00
#SBATCH -q inferno
#SBATCH -J j6_arms
#SBATCH -o logs/Report-%A_%a.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=henryyjiang42@gmail.com

# ============================================================================
# J6 -- the last Z test before the architecture is finalised.
#
#   sbatch --array=0-3 pace/j6_arms.sbatch
#   sbatch --dependency=afterany:<jobid> pace/j6_readout.sbatch
#
# TWO JOBS TOTAL.  No pre-registration and NO AUTOMATED VERDICT: the read-out
# prints raw numbers and they are assessed by hand.  That is a deliberate
# change from J1-J5, where the scorer's own headline ("NO") turned out to be
# the weaker of two readings its own fields supported.
#
# WHAT THIS ARM IS FOR
# --------------------
# J3, J4 and J5 all measured the same signature: Z's columns LOAD-BEARING
# (X_off positive everywhere) and Z's contents IRRELEVANT (X_content +0.0004,
# +0.0001, -0.0000 -- three encodings, two bases, budgets 2,000 and 4,000,
# with E and without).  Reading the code says why, and it is not a learning
# problem:
#
#   E's candidate  = x[:, end:]                 summary cols, POST-coda
#   Z's candidate  = _z_last_x[:, -n_sum:]      summary cols, PRE-coda
#
# Same columns, same forward, one coda apart -- the module's own docstring
# calls Z_end "the pre-coda twin of E".  D3's redundancy numbers agree: Z_end
# adds ~1% over E, with 15-27% of its norm outside E's span.  So Z was never
# given a distinct job, and "use the 64 columns as free capacity, ignore their
# contents" is the CORRECT solution to the problem as posed.  More steps
# optimise that harder, which is what J5's 2,000 -> 4,000 did (X_off +0.0400
# -> +0.0671 while the absolute moved AWAY from chance).
#
# Three limbs, each changing one thing that could make content pay, against a
# J4 control.  On the heal branch, at J4's budget, so the control is directly
# comparable to a known-good number (carry 0.7481, X_content +0.0001).
#
#   ctrl   J4's real limb, replicated.  NOT optional and not replaced by
#          J4's stored numbers: E's own carry skill drifts 0.37-0.85 nats
#          across same-seed runs, which is larger than anything expected
#          here, so the comparison has to live inside one launch.
#   mix    latent_read_scramble_p 0.25.  Capacity-use becomes UNPROFITABLE:
#          on a rolled row the columns lie, so leaning on them costs, and the
#          only strategies that survive are "read the contents" or "stop
#          using Z" -- both readable, unlike today's null.
#          NOTE p=1.0 is J4's `donor` limb and ALREADY RAN (carry 0.4367):
#          at p=1 Z is noise on every row and ignoring it is optimal, which
#          measures nothing about content.  The graft refuses p=1 here.
#   stag   J7, and it REPLACES the `delta` limb (cancelled 2026-09-27 after a
#          6 h node stall).  latent_encoding staggered_state + the projection
#          frozen at identity.  Four changes, stacked deliberately, because the
#          question is FEASIBILITY -- is this channel learnable at all -- and
#          not which single flag matters.  If X_content moves, ablate then; if
#          it does not, the null is the strongest one yet because everything
#          hypothesised to matter was stacked in Z's favour.
#
#          (1) RAW STATES, STAGGERED.  s_{k_j} at slot j, not d_{k_j} and not
#              s_T at every slot.  Staggering and delta-vs-raw are orthogonal
#              concepts that the code COUPLED: latent_write staged the depth
#              map over the delta tape, and `_tape_latent` differenced the raw
#              states away, so `endpoint` (one depth) and `delta` (staggered)
#              were the only two writes expressible.  J7 tapes the states too.
#          (2) WHY THIS BEATS BOTH.  Deltas were chosen by P0.1 on SCALE -- at
#              T=8 a delta is 0.90x the trunc_normal_ noise it replaced in s0,
#              where the endpoint was 28.1x -- but that argument was ONLY ever
#              about the s0 read, and s0 was measured dead (RED 11, flat to
#              4e-5 out to 45x ||s0||, including at exactly zero).  Under the
#              embeds read every Z row is rescaled to ||E|| regardless, so the
#              scale argument for deltas expired and nothing replaced it.
#              D3 never ranked `delta` at all -- its candidates were E, Z_tok,
#              Z_write, Z_end -- so delta-vs-endpoint has never been decided on
#              CONTENT in either direction.  Raw staggered states take delta's
#              actual justification (they sample intermediate depths the coda's
#              output cannot contain; s_2 is six iterations before s_T and is
#              NOT E's pre-coda twin) without delta's scale pathology: across
#              the 2..9 band ||d||/||s0|| spans 12.8 -> 0.90, a 14x range that
#              the per-row rescale then amplifies hardest exactly where P0.1
#              says the deltas are nearest the quantization floor, while
#              ||s_t|| runs 9.2 -> 10.7, a 1.16x range.
#          (3) NO PROJECTION (latent_embed_frozen true).  AutoCompressor has no
#              map anywhere -- auto_compressor.py contains zero nn.Linear, and
#              new_softprompt is last_hidden_state concatenated straight into
#              inputs_embeds -- and E follows it exactly, post-ln_f rows
#              spliced unprojected.  latent_embed is a free DxD map trained on
#              the LM loss, and D3 puts only 15-27% of Z's norm OUTSIDE E's
#              span, so the cheapest thing it can learn is to rotate Z onto the
#              directions the attention already reads well: E's.  If it does, a
#              null cannot distinguish "these depths carry nothing" from "the
#              map laundered them into E".  Frozen at its identity init the
#              module is an exact no-op, so the parameter set, the state_dict
#              and the optimizer count are UNCHANGED -- this arm and J4's
#              differ in requires_grad and nothing else.
#              THE RESCALE STAYS ON.  A scalar per row preserves direction; Z
#              entering at ~10 against E's ~136 inside one concatenation IS the
#              measured s0 failure, and that is a visibility problem, not a
#              coordinate one.
#          (4) READ AND WRITE ALREADY TRAIN TOGETHER, and no read_into module
#              is added.  J6's diag shows z_read_grad_frac and
#              z_write_grad_frac both 1.0 on every record, so J4's embeds read
#              already closed the "never trained together" gap.  And embeds
#              ALREADY READS AT EVERY DEPTH: core_block_forward computes
#              adapter(cat([x, input_embeds])) on every one of the T
#              iterations, so Z's columns are attended at each depth.  A
#              read_into module would not add depth coverage, it would add a
#              second reader -- and both readers that were tried had the model
#              switch them off (J1's gate 0.10 -> 0 in ~270 updates; J3's held
#              open with gate_lr_mult and still 0.014/0.028, read uniform, own
#              row 1/65, no selection learned in 2,000 updates).  The embeds
#              read is the only one whose strength GREW (z_embed_ratio 2.2 ->
#              3.7).  latent_read=embeds is unchanged and is not a variable.
#
#          THE COST, STATED BEFORE THE RUN.  A staggered band is not fully
#          trainable, and endpoint was.  iterate_forward runs num_steps_no_grad
#          FIRST, and at mr8 the loop length is SAMPLED: J6's ctrl diag has
#          z_tape_len from 1 to 26 and num_steps_no_grad 0 on 42/80 records but
#          up to 18 on the rest.  An absolute 2..9 band over 16 slots is then
#          fully trainable on 46/80 records, ENTIRELY frozen on 11/80 (13.8%),
#          and 0.762 of slots on average, against endpoint's 1.0 on every
#          batch.  That is the price of sampling intermediate depths at all.
#          It is REPORTED: latent_write_grad_frac has a per-slot branch for
#          this encoding, because the old last-step shortcut would have printed
#          1.0 over a write whose slots were frozen.  The gate below reads it.
#
#          Also relaxed, in this arm's favour: P0.1's "exclude depth 1, it is
#          23.8x s0" was a DELTA pathology.  Raw s_1 is an ordinary state at
#          norm ~9, so depth 1 is legitimately in range -- which matters,
#          because lo_c = min(lo, hi_c) already pulls the band to 1 on a short
#          loop.
#   slow   latent_stride 2.  The short-term / long-term split: Z's ring
#          pointer advances once per 2 chunks and writes that window's MEAN,
#          so Z reaches further back than E over the same 64 shared rows.
#          E and Z are two halves of ONE [B,K,2D] tensor, so Z cannot be given
#          more rows -- a longer horizon has to be a slower pointer.  The
#          stride cannot engage during a `grow` fill (every chunk must append
#          n_vec rows to keep the tensor rectangular), so at this geometry
#          chunks 1-4 fill and 5-8 ring: E ends holding chunks 5-8, Z ends
#          holding pooled(5,6), pooled(7,8) and the un-overwritten 3-4.  A
#          6-chunk span against E's 4 -- real, but not the clean 2x, and the
#          reading has to say so.
#
# COST: 4 limbs x 2,000 updates at J4's measured 2.04 s/micro ~= 2.3 h/limb,
# ~9 GPU-hours, ~2.5 h wall clock as a 4-task array.  J5 was 14.
#
# WHAT EACH LIMB MUST PROVE ABOUT ITSELF (the e_carry_read lesson)
# ----------------------------------------------------------------
# Every limb here is ONE flag from the control, and every flag defaults to the
# control's value.  e_carry_read parsed, printed, persisted and DID NOTHING for
# two smoke runs because an allowlist did not list it.  So each limb reports a
# MEASURED number the forward pass had to produce, checked by the smoke gate:
#
#   mix    mix_frac ~ 0.25 in cortex_diag.  0.0 means the roll never fired.
#   stag   z_n_depths > 1 (the write drew more than one depth) and
#          z_embed_frozen true.  z_write_grad_frac < 1.0 is EXPECTED here and
#          is not a failure -- see the band arithmetic above.
#   slow   z_stride 2 and z_blocks > 0.  z_blocks 0 means the ring never
#          filled, the stride never engaged, and the limb IS the control.
#   all    z_e_spliced_norm > 0, z_n_zpre 64, z_read_grad_frac 1.0.
# ============================================================================

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
PACK_TOKENIZER=${PACK_TOKENIZER:-ckpts/olmo-retrofit-cortex}
ARM_DATA=${ARM_DATA:-data/j1_pg19fw50_carry20_len4096}
MAX_LENGTH=${MAX_LENGTH:-4096}
CROSS_CHUNKS=${CROSS_CHUNKS:-8}
ACCUM_VECS=${ACCUM_VECS:-16}
GATE_SLOTS=${GATE_SLOTS:-64}
CARRY_GRAD_CHUNKS=${CARRY_GRAD_CHUNKS:-4}
RAMP_WARMUP=${RAMP_WARMUP:-0.25}
MAX_MEAN_REC=${MAX_MEAN_REC:-8}
MAX_STEPS=${MAX_STEPS:-305176}
PARENT_STEP=${PARENT_STEP:-91552}
CELL_STEPS=${CELL_STEPS:-2000}             # J4's budget exactly
STOP_AT_STEP=$(( PARENT_STEP + CELL_STEPS ))
# 1000, so `% (2 * save_interval)` writes the full checkpoint_93552 the
# read-out loads.  At 2000 only save_model_only fires and the read-out has
# nothing to open -- the J5 arithmetic, and a smoke cannot catch it.
SAVE_INTERVAL=${SAVE_INTERVAL:-1000}
DIAG_INTERVAL=${DIAG_INTERVAL:-25}
MICRO_BS=${MICRO_BS:-2}
BATCH_SIZE=${BATCH_SIZE:-4}
ZNORM=${ZNORM:-rms}
# 136.0 is ||E||'s MEASURED carried row norm ON THIS BRANCH (job 13593528:
# 128.95 / 135.77 / 142.83, mean 135.85).  J6 does not re-measure because it
# does not move the branch -- J4's single smoke failure was this number
# inherited from a DIFFERENT branch (B2's 171.0), and staying put is what
# makes reuse legitimate here.
ZNORM_TARGET=${ZNORM_TARGET:-136.0}
WINDOW_BACKWARD=${WINDOW_BACKWARD:-1}
WB_ARGS=""
if [ "$WINDOW_BACKWARD" = "1" ]; then WB_ARGS="--cortex.window_backward true"; fi
MIX_P=${MIX_P:-0.25}
STRIDE=${STRIDE:-2}
# The depth band.  P0.7 ANSWERED the rule this selects under -- job 13273098,
# 2026-09-17, re-run WITH --trained_depth (trained_mean_recurrence 8), verdict
# ABSOLUTE on both rules: magnitude t_hi 8 at T=16/32/64 with cv_absolute 0.0
# against cv_relative 0.53, novelty t_hi 25.  So `absolute` is measured, not
# asserted, and the J6 launch banner's "if P0.7 has not reported" caveat is
# discharged.  2..9 is P0.1's usable band; raising LATENT_LO to 1 is legitimate
# for a RAW-state write (P0.1's d_1 exclusion was a delta-scale pathology) and
# would widen the trainable share -- see the band arithmetic in the header
# before moving it.
LATENT_LO=${LATENT_LO:-2}
LATENT_HI=${LATENT_HI:-9}
SMOKE_GATE=${SMOKE_GATE:-0}

LIMBS=(ctrl mix stag slow)
IDX=${SLURM_ARRAY_TASK_ID:-0}
LIMB=${LIMBS[$IDX]}
if [ -z "$LIMB" ]; then echo "no limb at index $IDX (use --array=0-3)"; exit 0; fi

[ -d "$MODEL" ] || { echo "ERROR: no such base: $MODEL"; exit 1; }
[ -f "$BRANCH_PATH/chkpt.pt" ] || { echo "ERROR: no chkpt.pt under BRANCH_PATH=$BRANCH_PATH"; exit 1; }
[ -d "$ARM_DATA" ] || { echo "ERROR: no such pack: $ARM_DATA"; exit 1; }
if [ "$MICRO_BS" -lt 2 ]; then
    echo "ERROR: MICRO_BS=$MICRO_BS.  The donor cell pairs rows across the"
    echo "       batch AND the \`mix\` limb's roll needs >= 2 rows -- at 1 the"
    echo "       roll returns the same row and \`mix\` silently IS \`ctrl\`."
    exit 1
fi
if ! grep -q "iter_write(x, current_step)" "$MODEL/raven_modeling_minimal_cortex.py" 2>/dev/null; then
    echo "ERROR: $MODEL carries a modeling file without iter_write(x,"
    echo "       current_step) -- the Z write would never fire.  Refresh:"
    echo "         python tools/prepare_cortex_checkpoint.py --src $MODEL --dst $MODEL --variant olmo"
    exit 1
fi
python tools/check_tokenizer_match.py --pack_tokenizer "$PACK_TOKENIZER" \
    --model "$MODEL" || { echo "REFUSING THE LAUNCH: tokenizer mismatch."; exit 1; }

# Every limb: both channels live, both blocks of columns present, Z read
# through the embeds splice at E's measured scale.  The limbs differ in ONE
# flag each and in nothing else.
ENCODING=endpoint
EMBED_FROZEN=false
ARM_ARGS="--cortex.latent_read_scramble false"
ARM_ARGS="$ARM_ARGS --cortex.e_carry_read true --cortex.latent_carry_read true"
case "$LIMB" in
    ctrl)  ;;
    mix)   ARM_ARGS="$ARM_ARGS --cortex.latent_read_scramble_p $MIX_P" ;;
    # J7.  TWO flags, not one, and that is the deliberate exception to this
    # array's one-flag-per-limb rule -- the header says why.  ctrl remains the
    # comparison, and it already ran on this branch at this budget.
    stag)  ENCODING=staggered_state; EMBED_FROZEN=true ;;
    slow)  ARM_ARGS="$ARM_ARGS --cortex.latent_stride $STRIDE" ;;
esac

RUN_NAME=j6-a3z-${LIMB}${RUN_SUFFIX:-}
ACCUM_MAX=$(( ACCUM_VECS * CROSS_CHUNKS ))

echo "=== J6 | limb=$LIMB | run=$RUN_NAME ==="
echo "    base:      $MODEL"
echo "    branch:    $BRANCH_PATH  (step $PARENT_STEP -> stop $STOP_AT_STEP, horizon $MAX_STEPS)"
echo "    data:      $ARM_DATA   (READ THIS LINE -- the only proof the pack arrived)"
echo "    geometry:  gated W=$ACCUM_VECS K=$GATE_SLOTS cc=$CROSS_CHUNKS grad_chunks=$CARRY_GRAD_CHUNKS wb=$WINDOW_BACKWARD"
echo "    encoding:  $ENCODING   znorm $ZNORM -> $ZNORM_TARGET"
echo "    read:      embeds  (every depth: adapter(cat([x, input_embeds])) x T)"
echo "    embed_proj: frozen=$EMBED_FROZEN   band $LATENT_LO..$LATENT_HI rule absolute"
echo "    arm args:  $ARM_ARGS"
echo "    NOTE the echo above is NOT proof a flag took effect -- e_carry_read"
echo "         printed correctly for two runs while doing nothing.  The smoke"
echo "         gate reads cortex_diag; that is the proof."

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
    --cortex.latent_depth_lo $LATENT_LO \
    --cortex.latent_depth_hi $LATENT_HI \
    --cortex.latent_renorm none \
    --cortex.latent_encoding $ENCODING \
    --cortex.latent_s0_read false \
    --cortex.latent_read embeds \
    --cortex.latent_read_znorm $ZNORM \
    --cortex.latent_read_znorm_target $ZNORM_TARGET \
    --cortex.latent_embed_frozen $EMBED_FROZEN \
    $ARM_ARGS \
    --cortex.e_dropout 0.0 \
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

# ===== THE SMOKE GATE: did this limb run ITS OWN condition? =================
# Reads the MEASURED diagnostics, never the flags.  Every check here is
# something that has already gone wrong once in this project.
if [ $RC -eq 0 ]; then
    echo ""
    echo "=== J6 GATE | limb=$LIMB ==="
    python - "cortex-retrofit/$RUN_NAME/cortex_diag.jsonl" "$LIMB" \
             "$MIX_P" "$STRIDE" <<'GATE'
import json, sys
path, limb, mix_p, stride = sys.argv[1], sys.argv[2], float(sys.argv[3]), int(sys.argv[4])
try:
    rows = [json.loads(l) for l in open(path) if l.strip()]
except FileNotFoundError:
    sys.exit(f"FAIL: no {path}")
if not rows:
    sys.exit("FAIL: cortex_diag.jsonl is empty")
last = rows[-1]
def v(k, d=None):
    return last.get(k, d)
fails, notes = [], []

# --- shared: the channel is present and trainable on every limb -------------
if not v("z_e_spliced_norm"):
    fails.append("z_e_spliced_norm is 0/absent: E's rows never spliced")
if v("z_n_pre") != 64:
    fails.append(f"z_n_pre {v('z_n_pre')} != 64: Z's columns were not spliced")
rg = [r.get("z_read_grad_frac") for r in rows if r.get("z_read_grad_frac") is not None]
if not rg or min(rg) < 1.0:
    fails.append(f"z_read_grad_frac min {min(rg) if rg else None}, expected 1.0")
else:
    notes.append("z_read_grad_frac 1.0 (live on every batch)")
wg = [r.get("z_write_grad_frac") for r in rows if r.get("z_write_grad_frac") is not None]
if not wg or max(wg) == 0:
    fails.append("z_write_grad_frac 0: the write never trained")
en = v("z_e_carried_norm")
if en is None or not (0.85 * 136.0 <= en <= 1.15 * 136.0):
    fails.append(f"z_e_carried_norm {en} is >15% from ZNORM_TARGET 136.0 -- "
                 "||E|| moved on this branch and Z entered at the wrong scale")

# --- per limb: the MEASURED proof that this limb is not the control ---------
if limb == "mix":
    got = v("z_mix_frac")
    if got is None:
        fails.append("no mix_frac in the diag: the flag never reached the graft")
    elif abs(got - mix_p) > 0.06:
        fails.append(f"mix_frac {got:.4f} against a configured p of {mix_p} -- "
                     "the roll did not fire at the requested rate")
    else:
        notes.append(f"mix_frac {got:.4f} ~ p {mix_p} (the roll fired)")
elif limb == "stag":
    # NOTE the key is z_tape_len, not tape_len: training_diag PREFIXES every
    # latent_runtime field with z_, so the `delta` branch this replaces read
    # last.get("tape_len") -> None -> 0 and would have FAILED its own limb on a
    # perfectly good run.  It never fired, because delta never reached the gate.
    nd = v("z_n_depths") or 0
    tl = v("z_tape_len") or 0
    if tl < 2:
        fails.append(f"z_tape_len {tl}: the loop tape is empty, so the "
                     "staggered write has no depths to draw from")
    elif nd < 2:
        fails.append(f"z_n_depths {nd}: the write drew ONE depth, so the "
                     "stagger collapsed and this limb IS a single-depth write "
                     "-- exactly the inert-config failure J4/J5 shipped")
    else:
        notes.append(f"z_n_depths {nd} of z_tape_len {tl} "
                     f"(depths {v('z_depths')}) -- the stagger fired")
    # NOT `is not True`: training_diag casts every bool to int so wandb will
    # plot it, so this field arrives as 1/0 and `1 is not True` is True in
    # Python -- which FAILED this limb on 13662406_2, a run whose projection
    # was in fact frozen (the log printed 'latent_embed.[FROZEN at identity]'
    # and z_embed_ratio held at 0.94 while every trainable-proj limb drifted to
    # 3.7-5.3).  Compare the VALUE, and require the key to be present.
    ef = v("z_embed_frozen")
    if ef is None:
        fails.append("no z_embed_frozen in the diag: the flag never reached "
                     "the graft")
    elif int(ef) != 1:
        fails.append("z_embed_frozen is 0: the projection is TRAINING, "
                     "so this is J4's read and not the AC condition")
    else:
        notes.append("z_embed_frozen 1 (projection frozen at identity)")
    # NO z_encoding CHECK HERE.  training_diag keeps only ints and floats, so a
    # str field can never reach cortex_diag.jsonl and `v("z_encoding")` is None
    # on every limb of every arm -- the check it replaced could not pass, and it
    # failed 13662406_2 for that reason alone.  The staggered write's MEASURED
    # proof is z_n_depths >= 2 above, which is the gate's stated principle
    # anyway: read the measurement, never the flag.
    # EXPECTED, and reported rather than gated: a staggered band draws depths
    # that land in the no-grad prefix on a real share of batches (measured
    # 0.762 of slots on J6's ctrl split).  Gate only the floor -- a write that
    # NEVER trained is a dead limb; a write that trained 76% of its slots is
    # the design.
    wgf = [r.get("z_write_grad_frac") for r in rows
           if r.get("z_write_grad_frac") is not None]
    if wgf:
        mean_w = sum(wgf) / len(wgf)
        if max(wgf) == 0:
            fails.append("z_write_grad_frac 0 on every record: the staggered "
                         "write never entered the gradient window")
        else:
            notes.append(f"z_write_grad_frac mean {mean_w:.3f} "
                         f"(min {min(wgf):.3f}) -- <1.0 is expected here")
elif limb == "slow":
    st, blocks = v("z_stride"), v("z_blocks")
    if st != stride:
        fails.append(f"z_stride {st} != {stride}: latent_stride never reached "
                     "the buffer and this limb IS the control")
    elif not blocks:
        fails.append("z_blocks 0: the ring never filled, so the stride never "
                     "engaged and this limb IS the control")
    else:
        notes.append(f"z_stride {st}, z_blocks {blocks} (the stride engaged)")
else:
    if v("z_mix_frac") or (v("z_stride") or 1) != 1 \
            or v("z_embed_frozen") is True or (v("z_n_depths") or 0) > 1:
        fails.append("the CONTROL is carrying a treatment flag")
    notes.append("control: no mix, stride 1, endpoint write, proj trainable")

for n in notes:
    print(f"  ok   {n}")
for f in fails:
    print(f"  FAIL {f}")
print(f"  loss {last.get('loss')}   step {last.get('step')}   "
      f"z_embed_ratio {last.get('z_embed_ratio')}")
sys.exit(1 if fails else 0)
GATE
    GRC=$?
    if [ $GRC -ne 0 ]; then
        echo ""
        echo "GATE FAILED for limb=$LIMB.  This limb did not run its own"
        echo "condition, so it is not comparable to the others.  Do NOT score"
        echo "the array until it is fixed and this limb re-run."
        [ "$SMOKE_GATE" = "1" ] && exit 1
    fi
fi

echo ""
echo "Next (job 2 of 2):"
echo "  sbatch --dependency=afterany:\$SLURM_ARRAY_JOB_ID pace/j6_readout.sbatch"
exit $RC
