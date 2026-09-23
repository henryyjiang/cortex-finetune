#!/bin/bash
#SBATCH --account=gts-aivanova7-lab
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H200:1
#SBATCH --mem=128GB
#SBATCH -t 1:00:00
#SBATCH -q inferno
#SBATCH -J j3_price
#SBATCH -o logs/Report-%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=henryyjiang42@gmail.com

# J3's MEMORY PRICE -- "before the first launch" item 1 in pace/j3_joint.sbatch.
#
#   REQUIRE_FIT=1 sbatch pace/j3_price_oom.sbatch
#
# Two rows at J3's exact geometry (4096 x cc8, gated W=16 K=64, Z on, carry_grad
# _chunks 4, mr8, MICRO_BS=2, worst-case recurrence split, fp32 weights + bf16
# autocast, real Muon/Adam state, --window_backward -- tools/smoke_geometry_oom.py):
#   real    scratchpad read + carried rows: the heaviest limb (the donor limb
#           is the same modules and the same footprint)
#   noread  scratchpad read, carried rows unread
# What J3 adds over J1 (which priced 88.48 / 88.89 GiB, job 13459697): one
# [B, S_packed, D] scratchpad state per loop iteration held in the graph, its
# gate activations, the own-key k/v projections, and 16.8M gate parameters
# with two Adam moments each.  Expect a few GiB over J1, not tens.
#
#   REQUIRE_FIT=1   exit 3 unless the real limb FITS, so this job can gate the
#                   launch:  sbatch --dependency=afterok:<this job> pace/j3_joint.sbatch

cd $SLURM_SUBMIT_DIR

module load anaconda3
module load cuda/12.1.1
conda activate cortex-retro

export SCRATCH=${SCRATCH:-$HOME/scratch}
export HF_HOME=$SCRATCH/hf_cache
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODEL=${MODEL:-ckpts/olmo-retrofit-cortex}
MICRO_BS=${MICRO_BS:-2}
CARRY_GRAD_CHUNKS=${CARRY_GRAD_CHUNKS:-4}
GATE_LR_MULT=${GATE_LR_MULT:-0.1}
JSON=logs/j3_oom_${SLURM_JOB_ID}.jsonl

echo "=== J3 memory price | micro=$MICRO_BS | grad_chunks=$CARRY_GRAD_CHUNKS | window_backward=1 ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
if ! grep -q "iter_write(x, current_step)" "$MODEL/raven_modeling_minimal_cortex.py" 2>/dev/null; then
    echo "ERROR: $MODEL carries a modeling file without iter_write(x, current_step)"
    echo "       -- this would price a model with no scratchpad write.  Refresh it:"
    echo "         python tools/prepare_cortex_checkpoint.py --src $MODEL --dst $MODEL --variant olmo"
    exit 1
fi

COMMON="--model_name $MODEL --max_length 4096 --cross_chunks 8
        --micro_batch_size $MICRO_BS --batch_size 4 --max_steps 2000
        --mean_recurrence 8 --prefix_memory gated --accum_vecs 16
        --gate_slots 64 --gate_route ring --gate_norm tanh --gate_init zero
        --gate_fill grow --latent_carry --carry_grad_chunks $CARRY_GRAD_CHUNKS
        --window_backward
        --use_memory true --with_optimizer true --json_out $JSON
        --set latent_encoding=scratch --set latent_s0_read=false
        --set latent_read=scratch --set latent_read_heads=8
        --set latent_read_znorm=rms --set latent_read_gate_lr_mult=$GATE_LR_MULT"

BROKEN=0
echo ""
echo "######## real limb: scratchpad + carried rows ########"
python tools/smoke_geometry_oom.py $COMMON --set latent_carry_read=true || BROKEN=1
echo ""
echo "######## noread limb: scratchpad, carry unread ########"
python tools/smoke_geometry_oom.py $COMMON --set latent_carry_read=false || BROKEN=1

echo ""
python - "$JSON" <<'SUMMARY'
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1])]
print(f"{'limb':>7} {'micro':>5} {'status':>6} {'alloc':>8} {'reserved':>9} {'%dev':>5} {'s/step':>7}")
for r in rows:
    sets = " ".join(r.get("sets", []))
    limb = "noread" if "latent_carry_read=false" in sets else "real"
    pct = 100 * r["peak_res_gib"] / r["total_gib"]
    print(f"{limb:>7} {r.get('micro_batch_size', '?'):>5} {r['status']:>6} "
          f"{r['peak_alloc_gib']:>8.2f} {r['peak_res_gib']:>9.2f} {pct:>5.0f} "
          f"{r['step_s']:>7.1f}")
real = [r for r in rows if "latent_carry_read=true" in " ".join(r.get("sets", []))]
print("")
print("J1 reference (job 13459697): real 88.48 alloc / 88.89 reserved GiB, 1.62 s/micro.")
if real and real[0]["status"] == "FITS":
    print("J3: the real/donor limb FITS at this micro-batch -- all three limbs can launch.")
    print("    Record the numbers in pace/j3_joint.sbatch's DECISIONS block.")
elif real:
    print("J3: the real/donor limb DOES NOT FIT.  Do NOT drop to micro 1 (the donor")
    print("    roll would return the same row).  The scratchpad is the new term --")
    print("    re-read cortex_memory/scratchpad.py's COST note before cutting anything.")
else:
    print("J3: no real-limb row in the json -- the measurement did not run.")
SUMMARY

if [ $BROKEN -ne 0 ]; then
    echo ""
    echo "NOTE: a row failed for a NON-OOM reason (the script exits 0 on an OOM)."
    echo "      Read the traceback above; the table is incomplete."
    exit 1
fi

if [ -n "$REQUIRE_FIT" ]; then
    if ! python - "$JSON" <<'GATE'
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1])]
real = [r for r in rows if "latent_carry_read=true" in " ".join(r.get("sets", []))]
sys.exit(0 if real and real[0]["status"] == "FITS" else 1)
GATE
    then
        echo ""
        echo "REQUIRE_FIT: the real limb did not fit -- exit 3, so any job queued"
        echo "             --dependency=afterok on this one will NOT start."
        exit 3
    fi
fi
