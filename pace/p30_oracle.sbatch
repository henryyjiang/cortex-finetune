#!/bin/bash
#SBATCH --account=gts-aivanova7-lab
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H200:1
#SBATCH --mem=96GB
#SBATCH -t 4:00:00
#SBATCH -q inferno
#SBATCH -J p30_oracle
#SBATCH -o logs/Report-%A_%a.out
#SBATCH --array=0-3
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=henryyjiang42@gmail.com

# P3.0 TIER 1.5 -- THE ORACLE READ PROBE.  THE GO/NO-GO.
#
#   sbatch pace/p30_oracle.sbatch                        # 4 cells, ~5-7 GPU-hours
#   ARMS=a2 sbatch --array=0-1 pace/p30_oracle.sbatch    # one arm, both limbs
#
# THE QUESTION, and it is the only one left that Z can be closed on:
#
#   Does the carried recurrence trajectory contain ANYTHING a read can extract,
#   under conditions chosen to be MAXIMALLY FAVOURABLE?
#
# Maximally favourable is the point.  A null here is decisive precisely because
# nothing was held back; a null under ordinary conditions would not be.  Every
# dial is turned in Z's favour:
#
#   * the read is at `read_into`, full-width, DEPTH-AGNOSTIC (maximal access to
#     all carried rows)
#   * EVERYTHING ELSE IS FROZEN -- backbone, buffer, write path, both gates --
#     so nothing can absorb, launder or fight the signal
#   * num_steps is FORCED to [0, T]: zero no-grad prefix, so every iteration
#     carries read gradient.  A deliberate departure from the training
#     distribution.  THIS IS AN EXISTENCE TEST, NOT A CELL, and its loss is not
#     comparable to a P2.3 cell's.
#
# THE CONTROL IS THE WHOLE EXPERIMENT.  The comparison is NOT "does adding a
# read module help" -- a module can help by adding capacity, or by reading
# position and register information that has nothing to do with the previous
# chunk.  It is:
#
#   the IDENTICAL module, IDENTICALLY trained, on REAL Z vs on SHUFFLED Z
#   (rows from a different document's chain).  Same parameters, same capacity,
#   same seed, same data order.  Only the CONTENT CORRESPONDENCE differs.
#
# `--cortex.latent_read_scramble true` rolls the carried Z along the batch on
# the READ side only, so the write still receives this document's own
# trajectory and the two arms differ in exactly one thing.  IT NEEDS BATCH >= 2
# and raises otherwise: a roll of one row returns that same row, and the
# control arm would silently BE the treatment arm.
#
# PRE-REGISTERED READINGS -- p30_readinto_prereg.md S5, fixed before any number:
#
#   A beats B, CI excludes zero   READABLE CARRY EXISTS.  Proceed to tier 2
#       with the real design.  This would be the first positive Z result in the
#       project.
#   A ~ B, both beat no-read      the module buys CAPACITY or REGISTER, not
#       content.  Z's content is inert under maximally favourable conditions.
#       A strong negative, and the negative half of the paper.
#   neither beats no-read         the site is dead even with a trained read.
#       This CONTRADICTS tier 1 (k/k_s0 = 1.3e5, job 13420851) and means the
#       INSTRUMENT is wrong, not the mechanism.  Stop and audit before
#       spending anything further.
#   A beats B but the horizon
#   swap on A moves nothing       incoherent; two instruments disagree.  Audit
#       damage_pair / damage_write for the RED 8/9 class before reporting
#       either.
#
# WHICH ARM DECIDES.  p30 S5 says branch from a2 (accum) because its Z is the
# raw staggered trajectory with no gate in the way -- the cleanest substrate --
# and confirm on a3z after.  That ordering still holds, but note what changed
# since: cortex-final is decided as the GATED RING, so a2 is off the shipping
# path and a3z is the DECISION-RELEVANT arm rather than a confirmation.  Both
# run here; read a2 for the cleanest existence answer and a3z for the one that
# bears on the model that ships.
#
# BOTH Z CHECKPOINTS ARE 400-STEP PROBES.  The P2.3 cells (a1/a3) are E-ONLY
# and cannot host this at all -- they have no Z parameters.  So the write these
# arms read was shaped for 400 steps under NO READ PRESSURE, which is p30 S8
# risk 2: a tier-1.5 positive with a tier-2 null would point at the WRITE, and
# that is a third attempt the S7 budget does not fund.
#
# PRECONDITION, AND IT BIT ON 2026-09-21.  The base checkpoint carries its OWN
# COPY of the grafted modeling file.  Job 13425548 ran against a copy that
# predates P3.0 (`read_into(x)`, one argument) and nothing said so, because
# depth was 'none' and the run never needed current_step.  read_into now raises
# on it.  REFRESH IT FIRST:
#
#     python tools/prepare_cortex_checkpoint.py \
#         --src ckpts/olmo-retrofit-cortex --dst ckpts/olmo-retrofit-cortex \
#         --variant olmo
#     rm -rf $SCRATCH/hf_cache/modules/transformers_modules/olmo-retrofit-cortex
#
# --T 8 IS NOT OPTIONAL.  config.mean_recurrence is 32 on every B2-family
# checkpoint and is INHERITED from the base recipe; the arms trained at 8.

cd $SLURM_SUBMIT_DIR

module load anaconda3
module load cuda/12.1.1
conda activate cortex-retro

export SCRATCH=${SCRATCH:-$HOME/scratch}
export HF_HOME=$SCRATCH/hf_cache
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODEL=${MODEL:-ckpts/olmo-retrofit-cortex}
DATA=${DATA:-data/pg19_fw50_olmo_len4096}
OUT_ROOT=${OUT_ROOT:-cortex-retrofit}
RUN_PREFIX=${RUN_PREFIX-probe-}

ACCUM_VECS=${ACCUM_VECS:-16}
GATE_SLOTS=${GATE_SLOTS:-64}
CROSS_CHUNKS=${CROSS_CHUNKS:-8}
MAX_MEAN_REC=${MAX_MEAN_REC:-8}
READ_HEADS=${READ_HEADS:-8}
READ_DEPTH=${READ_DEPTH:-none}

# MICRO_BS >= 2 IS A HARD REQUIREMENT, not a throughput choice.  The scramble
# rolls the BATCH THAT REACHES THE FORWARD, which is micro_batch_size and not
# batch_size -- the P1 arms ran micro_batch_size 1 with batch_size 4, i.e.
# accumulation, and at micro_batch_size 1 the roll returns the same row and the
# control arm IS the treatment arm.  The graft raises rather than running it;
# this default is what stops the raise ever being reached.
#
# IT COSTS MEMORY, TWICE OVER: micro_batch 2 instead of 1 doubles the backward
# graph, and force_num_steps [0,8] puts all 8 iterations in the gradient where
# the sampler averaged ~4.  Against that, train_read_only means no optimizer
# state for the frozen 99.9% of the model.  The P2.3 cells peaked at 11.9 GB of
# 141, so there is room -- but nobody has measured THIS configuration, so read
# the first cell's peak before assuming the other three fit.
MICRO_BS=${MICRO_BS:-2}
# BATCH_SIZE 4, NOT 8.  The scramble needs micro_batch >= 2; it does NOT need a
# bigger EFFECTIVE batch, and every arm on record ran 4 sequences x 4096 =
# 16,384 tokens per optimizer step (p1_arms: micro 1, batch 4).  Holding that
# constant keeps the token budget comparable and halves the cost; micro 2 with
# batch 4 is simply 2 accumulation steps instead of 4.
BATCH_SIZE=${BATCH_SIZE:-4}
MAX_LENGTH=${MAX_LENGTH:-4096}
if [ "$MICRO_BS" -lt 2 ]; then
    echo "ERROR: MICRO_BS=$MICRO_BS but the scramble control needs >= 2."
    echo "       At micro_batch_size 1 the shuffled limb is identical to the"
    echo "       real limb and the probe would return a null by construction."
    exit 1
fi
if [ $(( MAX_LENGTH % CROSS_CHUNKS )) -ne 0 ]; then
    echo "ERROR: MAX_LENGTH ($MAX_LENGTH) is not divisible by CROSS_CHUNKS"          "($CROSS_CHUNKS) -- chunks would be ragged."; exit 1
fi

# COST, from the P2.3 measurement (3,935 tok/s at micro 1 / batch 4 / 4096,
# sampled num_steps, Muon on, full-model optimizer).  Throughput HERE is
# unmeasured and pulls both ways: the forced [0,T] split puts all 8 iterations
# in the gradient where the sampler averaged fewer, but there is no Muon and
# only FIVE trainable tensors, so the optimizer step is nearly free.  Bracketed
# 2,500-3,900 tok/s:
#
#   1000 steps x 4 x 4096 = 16.4M tok  ->  1.2-1.8 h/cell, 4.7-7.3 GPU-h for 4
#   2000 steps                         ->  2.3-3.6 h/cell, 9.3-14.6 GPU-h
#
# 1000 is the default because only 16.8M parameters train here, at lr 1e-3, and
# they either learn to use Z or they do not -- this is an existence test, not a
# cell, and it is NOT governed by p23's stopping rule.  THE GATE TRAJECTORY IN
# cortex_diag.jsonl IS THE CHECK: if the read gate is still moving at step 1000,
# extend rather than reading a number off a module that had not settled.
PROBE_STEPS=${PROBE_STEPS:-1000}

# DIAG_INTERVAL: p1_arms.sbatch defaults this to 0 and that default is why job
# 10 had to be reconstructed post-hoc from six checkpoints.  The read GATE's
# trajectory is a pre-registered reading here -- "if it collapses toward zero
# the model is telling us it does not want the read, and that is a legitimate
# negative" -- so it must be logged, not inferred.
DIAG_INTERVAL=${DIAG_INTERVAL:-50}

ARMS=${ARMS:-"a2 a3z"}
LIMBS=${LIMBS:-"real shuffled"}
CELLS=()
for ar in $ARMS; do for li in $LIMBS; do CELLS+=("$ar:$li"); done; done

IDX=${SLURM_ARRAY_TASK_ID:-0}
echo "=== P3.0 tier 1.5 oracle probe | ${#CELLS[@]} cells | index $IDX ==="
for i in "${!CELLS[@]}"; do echo "    [$i] ${CELLS[$i]}"; done
echo "    for THIS cell list:  sbatch --array=0-$(( ${#CELLS[@]} - 1 ))"
if [ "$IDX" -ge "${#CELLS[@]}" ]; then
    echo "no cell at index $IDX.  Pass --array=0-$(( ${#CELLS[@]} - 1 ))."
    exit 0
fi

ARM=${CELLS[$IDX]%%:*}
LIMB=${CELLS[$IDX]##*:}
case "$ARM" in
    a2)  BRANCH=${RUN_PREFIX}p1-a2-accum-w${ACCUM_VECS}-cc${CROSS_CHUNKS}-z
     PREFIX_MODE=accum ;;
a3z) BRANCH=${RUN_PREFIX}p1-a3z-gated-w${ACCUM_VECS}k${GATE_SLOTS}-cc${CROSS_CHUNKS}-z
     PREFIX_MODE=gated ;;
*)   echo "ERROR: '$ARM' has no Z channel.  Only a2 and a3z carry one."; exit 1;;
esac
case "$LIMB" in
    real)     SCRAMBLE=false ;;
    shuffled) SCRAMBLE=true ;;
    *) echo "ERROR: limb must be 'real' or 'shuffled'; got '$LIMB'."; exit 1;;
esac

BRANCH_CKPT=$(ls -d $OUT_ROOT/$BRANCH/checkpoint_* 2>/dev/null | sort -t_ -k2 -n | tail -1)
if [ -z "$BRANCH_CKPT" ] || [ ! -d "$BRANCH_CKPT" ]; then
    echo "ERROR: no checkpoint for arm '$ARM' (looked under $OUT_ROOT/$BRANCH)."
    echo "       RUN_PREFIX='$RUN_PREFIX' -- set RUN_PREFIX= for a trained cell."
    exit 1
fi
if [ ! -d "$DATA" ]; then echo "ERROR: no such pack: DATA=$DATA"; exit 1; fi

# PRECONDITION CHECK, cheap and non-negotiable.  A stale modeling file inside
# the base runs a different forward than the source tree and says nothing about
# it -- see the header.  read_into raises on it, but failing HERE costs a
# second instead of a GPU allocation and names the fix.
if ! grep -q "read_into(x, current_step)" "$MODEL/raven_modeling_minimal_cortex.py" 2>/dev/null; then
    echo "ERROR: $MODEL carries a PRE-P3.0 modeling file (read_into takes one"
    echo "       argument).  Refresh it before running anything:"
    echo "         python tools/prepare_cortex_checkpoint.py \\"
    echo "             --src $MODEL --dst $MODEL --variant olmo"
    echo "         rm -rf \$SCRATCH/hf_cache/modules/transformers_modules/$(basename $MODEL)"
    exit 1
fi

RUN=p30-oracle-$ARM-$LIMB
ACCUM_MAX=${ACCUM_MAX:-$(( ACCUM_VECS * CROSS_CHUNKS ))}

CORTEX_ARGS="--cortex.use_memory true --cortex.memory_slots 0"
CORTEX_ARGS="$CORTEX_ARGS --cortex.prefix_memory $PREFIX_MODE"
CORTEX_ARGS="$CORTEX_ARGS --cortex.accum_vecs $ACCUM_VECS"
if [ "$PREFIX_MODE" = "accum" ]; then
    CORTEX_ARGS="$CORTEX_ARGS --cortex.accum_max $ACCUM_MAX"
else
    CORTEX_ARGS="$CORTEX_ARGS --cortex.gate_slots $GATE_SLOTS"
    CORTEX_ARGS="$CORTEX_ARGS --cortex.gate_route ${GATE_ROUTE:-ring}"
    CORTEX_ARGS="$CORTEX_ARGS --cortex.gate_norm ${GATE_NORM:-tanh}"
    CORTEX_ARGS="$CORTEX_ARGS --cortex.gate_init ${GATE_INIT:-zero}"
    CORTEX_ARGS="$CORTEX_ARGS --cortex.gate_fill ${GATE_FILL:-grow}"
fi
CORTEX_ARGS="$CORTEX_ARGS --cortex.latent_carry true"
CORTEX_ARGS="$CORTEX_ARGS --cortex.latent_depth_rule ${LATENT_RULE:-absolute}"
CORTEX_ARGS="$CORTEX_ARGS --cortex.latent_depth_lo ${LATENT_LO:-2}"
CORTEX_ARGS="$CORTEX_ARGS --cortex.latent_depth_hi ${LATENT_HI:-9}"
CORTEX_ARGS="$CORTEX_ARGS --cortex.latent_renorm none"
# s0_read OFF: the site is measured dead, and leaving it on would mean the
# probe trains a read while a second, known-inert read runs beside it.
CORTEX_ARGS="$CORTEX_ARGS --cortex.latent_s0_read false"
CORTEX_ARGS="$CORTEX_ARGS --cortex.latent_read xattn"
CORTEX_ARGS="$CORTEX_ARGS --cortex.latent_read_depth $READ_DEPTH"
CORTEX_ARGS="$CORTEX_ARGS --cortex.latent_read_heads $READ_HEADS"
CORTEX_ARGS="$CORTEX_ARGS --cortex.latent_read_scramble $SCRAMBLE"
CORTEX_ARGS="$CORTEX_ARGS --cortex.cross_chunks $CROSS_CHUNKS"
CORTEX_ARGS="$CORTEX_ARGS --cortex.carry_grad_chunks ${CARRY_GRAD_CHUNKS:-4}"
CORTEX_ARGS="$CORTEX_ARGS --cortex.diag_interval $DIAG_INTERVAL"

echo ""
echo "    arm=$ARM  limb=$LIMB  (scramble=$SCRAMBLE)"
echo "    branch:     $BRANCH_CKPT"
echo "    data:       $DATA   (READ THIS LINE -- the only proof the pack arrived)"
echo "    geometry:   $PREFIX_MODE  W=$ACCUM_VECS  cc=$CROSS_CHUNKS  T=[0,$MAX_MEAN_REC]"
echo "    trainable:  cortex.latent_reader ONLY"
echo "    batch:      micro=$MICRO_BS total=$BATCH_SIZE   (micro >= 2 REQUIRED -- the scramble rolls it)"
echo "    steps:      $PROBE_STEPS   diag every $DIAG_INTERVAL"
echo "    run:        $RUN"
echo ""
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

python train.py     --run_name $RUN     --out_path $OUT_ROOT     --model_name $MODEL     --preprocessed_data_path $DATA     --max_length $MAX_LENGTH     --epochs 1     --max_steps $PROBE_STEPS     --micro_batch_size $MICRO_BS     --batch_size $BATCH_SIZE     --optim_config.lr ${READ_LR:-1e-3}     --scheduler_args.cooldown 0.6     --scheduler_args.warmup 0.02     --muon.use_muon false     --no_amp false     --max_grad_norm 1.0     --compile false     --save_interval $PROBE_STEPS     --seed 74     --train_read_only true     --force_num_steps "0,$MAX_MEAN_REC"     --branch_path "$BRANCH_CKPT"     $CORTEX_ARGS
RC=$?

echo ""
if [ $RC -eq 0 ]; then
    echo "cell $ARM/$LIMB DONE -> $OUT_ROOT/$RUN"
    echo ""
    echo "READING ORDER -- and the pair is the result, never one limb alone:"
    echo "  1. tools/compare_arms.py --stopping_rule on real vs shuffled,"
    echo "     SAME arm.  A paired CI on one side of zero across two"
    echo "     consecutive non-overlapping windows is the claim."
    echo "  2. the read gate's trajectory in cortex_diag.jsonl.  A gate that"
    echo "     collapses toward zero is the model saying it does not want the"
    echo "     read -- pre-registered as a legitimate negative, so do not read"
    echo "     it as a bug."
    echo "  3. eval_influence_horizon --channel z --damage donor on the REAL"
    echo "     limb's checkpoint.  If the trained read does not depend on"
    echo "     content, that contradicts limb 1 and both need auditing."
    echo "  4. a2 answers 'does readable carry exist'.  a3z answers 'does it"
    echo "     exist in the model that ships'.  They can differ."
else
    echo "cell $ARM/$LIMB FAILED rc=$RC"
fi
exit $RC
