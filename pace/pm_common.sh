#!/bin/bash
# ============================================================================
# Shared plumbing for the post-mortem EVAL jobs (pm_mech / pm_probe), sourced
# by both so the checkpoint resolution and the cortex rebuild flags exist once.
#
#   pm_env                      modules, conda, offline HF
#   pm_resolve <arm> <step>     sets RUN, CKPT, ZNORM, SETS, RES for
#                               arm in {cortex, control}, step in {38146, 11444}
#
# ZNORM IS PER STEP, NOT PER RUN.  cortex-final trained at znorm_target 98.0
# through the heal (to 11,444) and 78.9 from the switch on (re-measured
# 2026-10-03).  A rebuild at the wrong one scores Z's read at the wrong
# strength, so pm_resolve CHECKS the value against cortex_diag.jsonl at that
# step -- the midrun_check rule -- and refuses on a mismatch.
# ============================================================================

pm_env() {
    module load anaconda3
    module load cuda/12.1.1
    conda activate cortex-retro
    export SCRATCH=${SCRATCH:-$HOME/scratch}
    export HF_HOME=$SCRATCH/hf_cache
    export HF_HUB_OFFLINE=1
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    export PYTHONIOENCODING=utf-8
    MODEL=${MODEL:-ckpts/olmo-retrofit-cortex}
    PG19_VAL=${PG19_VAL:-data/pg19_olmo_validation_len4096_strided}
    FWEDU_VAL=${FWEDU_VAL:-data/fineweb_edu_olmo_val_len4096}
    T=${T:-8}
}

pm_resolve() {
    local arm=$1 step=$2
    case "$arm" in
        cortex)  RUN=cortex-final ;;
        control) RUN=c-chunked ;;
        *) echo "pm_resolve: arm must be cortex|control, got '$arm'"; return 1 ;;
    esac
    CKPT="cortex-5b/$RUN/checkpoint_$step"
    RES="eval_results/pm/$RUN/step_${step}_T${T}"
    mkdir -p "$RES"
    if [ ! -f "$CKPT/chkpt.pt" ]; then
        echo "ERROR: no chkpt.pt at $CKPT.  Present with chkpt.pt:"
        for d in cortex-5b/$RUN/checkpoint_*; do [ -f "$d/chkpt.pt" ] && echo "    ${d##*/}"; done
        return 1
    fi
    SETS=""
    ZNORM=""
    [ "$arm" = "control" ] && return 0
    case "$step" in
        11444) ZNORM=98.0 ;;
        *)     ZNORM=78.9 ;;
    esac
    SETS="--set use_memory=true --set memory_slots=0 --set prefix_memory=gated"
    SETS="$SETS --set accum_vecs=16 --set gate_slots=64 --set gate_route=ring"
    SETS="$SETS --set accum_max=128"
    SETS="$SETS --set e_carry_read=true --set latent_carry_read=true"
    SETS="$SETS --set gate_norm=tanh --set gate_init=zero --set gate_fill=grow"
    SETS="$SETS --set latent_carry=true --set latent_depth_rule=absolute"
    SETS="$SETS --set latent_depth_lo=1 --set latent_depth_hi=9"
    SETS="$SETS --set latent_renorm=none --set latent_tok_pool=4"
    SETS="$SETS --set latent_encoding=staggered_state"
    SETS="$SETS --set latent_embed_frozen=true"
    SETS="$SETS --set latent_s0_read=false --set latent_read=embeds"
    SETS="$SETS --set latent_read_znorm=rms"
    SETS="$SETS --set latent_read_znorm_target=$ZNORM"
    python - "cortex-5b/$RUN/cortex_diag.jsonl" "$ZNORM" "$step" <<'CHECK' || return 1
import json, sys
path, want, step = sys.argv[1], float(sys.argv[2]), int(sys.argv[3])
rows = [json.loads(l) for l in open(path) if l.strip()]
at = [r for r in rows if (r.get("step") or 0) <= step]
tg = [r["z_znorm_target"] for r in at if r.get("z_znorm_target") is not None]
if not tg:
    sys.exit(f"ERROR: no z_znorm_target in {path} at or before step {step}")
if abs(tg[-1] - want) > 1e-6:
    sys.exit(f"ERROR: step {step} trained at znorm_target {tg[-1]}, this job "
             f"rebuilds at {want}.  Fix pm_resolve's table.")
print(f"    znorm_target {want} confirmed against the diag at step {at[-1].get('step')}")
CHECK
}
