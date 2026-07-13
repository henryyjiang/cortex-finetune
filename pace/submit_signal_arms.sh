#!/bin/bash
# Signal/gate round (2026-07-13 plan, step 2+3) — submit helper.
#
# Background: the carry-vs-zeroed ablation (eval_results/carry_ablation_20260713)
# showed the memory pathway is ALIVE (delta 7-13 SE from zero) but tiny:
# +0.0036 (rung2-k4) / +0.0044 (rung1-k4-v2) / +0.0100 (rung1-k0-ccot-v2)
# nats/token — i.e. the zero-init bootstrap DID engage and converged to
# roughly everything the plain PG-19 objective rewards.  The binding
# constraint is the SIGNAL, so these arms raise it (and one opens the gate
# wider) rather than adding tokens.
#
# Usage (from ~/cortex-finetune on the cluster):
#   bash pace/submit_signal_arms.sh probe    # stage 1: teacher-advantage probe
#   bash pace/submit_signal_arms.sh arms     # stage 2: training arms
#
# Stage 1 must be READ before stage 2's distill arm: if the probe shows the
# base RDM's NLL gets WORSE with a 2048 window (LM-loss twin of the ~1.5-2k
# generation cliff), lower DISTILL_WINDOW below the cliff or skip the distill
# arm and lean on the recall mix.
#
# Login-node prep (once, before stage 2):
#   python tools/prepare_recall_mix.py \
#       --data data/pg19_olmo_len4096 --tokenizer ckpts/olmo8-cortex \
#       --out data/pg19_olmo_recall25_len4096 --frac 0.25
#   # optional recall-heavy val set for the follow-up carry ablation:
#   python tools/prepare_recall_mix.py \
#       --data data/pg19_olmo_val_len4096 --tokenizer ckpts/olmo8-cortex \
#       --out data/pg19_olmo_val_recall100_len4096 --frac 1.0

set -e
STAGE=${1:-}

case "$STAGE" in
probe)
    # base RDM: is there distillable signal in a longer window, and where is
    # the LM-loss length cliff?
    sbatch pace/eval_teacher_advantage.sbatch
    echo "Probe submitted. Read eval_results/teacher_advantage_<tag>/base/"
    echo "before submitting the arms (delta_vs_control > 0 at W = distillable)."
    ;;
arms)
    DISTILL_WINDOW=${DISTILL_WINDOW:-2048}
    # Arm 1 (fix #1): full-window-teacher distillation, K=4.  H200 — the
    # teacher (+~2 GB) and per-chunk fp32 logit graphs don't fit A100-80's
    # rung1 margin comfortably.
    DISTILL_COEFF=1.0 DISTILL_WINDOW=$DISTILL_WINDOW \
        sbatch --gres=gpu:H200:1 --constraint=H200 pace/rung1_frozen_loop.sbatch
    # Arm 2 (fix #2): recall-supervised mix, K=4 (plain rung1 memory budget).
    DATA_PATH=data/pg19_olmo_recall25_len4096 \
        sbatch pace/rung1_frozen_loop.sbatch
    # Arm 3 (fix #3): bootstrap-gate — small nonzero read init, K=4.
    READ_INIT=1e-3 \
        sbatch pace/rung1_frozen_loop.sbatch
    # Control for arm 2: K=0 no-carry on the SAME recall data — separates
    # "memory learned recall" from "the model learned the probe format".
    MEMORY_SLOTS=0 DATA_PATH=data/pg19_olmo_recall25_len4096 \
        sbatch pace/rung1_frozen_loop.sbatch
    echo ""
    echo "Submitted: rung1-k4-dstl1.0w${DISTILL_WINDOW} | rung1-k4-rcl |"
    echo "           rung1-k4-ri1e-3 | rung1-k0-rcl (control)"
    echo ""
    echo "Follow-up once trained (headline = recall-probe + carry ablation):"
    echo "  RUNS=\"rung1-k4-dstl1.0w${DISTILL_WINDOW} rung1-k4-rcl rung1-k4-ri1e-3 rung1-k0-rcl\" \\"
    echo "      bash pace/submit_evals_all.sh"
    echo "  RUN=<name> DATA=data/pg19_olmo_val_recall100_len4096 \\"
    echo "      sbatch pace/eval_carry_ablation.sbatch   # carry delta on probe data"
    ;;
*)
    echo "usage: bash pace/submit_signal_arms.sh probe|arms"; exit 1;;
esac
