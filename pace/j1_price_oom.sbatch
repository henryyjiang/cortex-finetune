#!/bin/bash
#SBATCH --account=gts-aivanova7-lab
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H200:1
#SBATCH --mem=64GB
#SBATCH -t 1:00:00
#SBATCH -q inferno
#SBATCH -J j1_oom
#SBATCH -o logs/Report-%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=henryyjiang42@gmail.com

# J1's MEMORY PRICE -- the "before the first launch" item 2 in pace/j1_joint.sbatch.
#
#   sbatch pace/j1_price_oom.sbatch
#
# Two rows, both at J1's exact geometry (4096 x cc8, gated W=16 K=64, Z on,
# carry_grad_chunks 4, mr8, MICRO_BS=2, worst-case recurrence split, fp32
# weights + bf16 autocast, real Muon/Adam state -- tools/smoke_geometry_oom.py):
#   real    the LatentRead xattn read + Z norm: the heaviest limb (the donor
#           limb is the same module and the same footprint)
#   noread  Z written, read nowhere: the lightest
# The encoding barely moves memory (all three write [B, W, D]); ENCODING only
# picks which one is priced.
#
# READ: the status column.  FITS on `real` means all three limbs launch at
# MICRO_BS=2.  An OOM on `real` means the donor control cannot run as designed
# -- do NOT drop to micro 1 (the roll would return the same row); lower
# carry_grad_chunks first and re-price.  Record the numbers in j1_joint.sbatch.

cd $SLURM_SUBMIT_DIR

module load anaconda3
module load cuda/12.1.1
conda activate cortex-retro

export SCRATCH=${SCRATCH:-$HOME/scratch}
export HF_HOME=$SCRATCH/hf_cache
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODEL=${MODEL:-ckpts/olmo-retrofit-cortex}
ENCODING=${ENCODING:-endpoint}
MICRO_BS=${MICRO_BS:-2}
CARRY_GRAD_CHUNKS=${CARRY_GRAD_CHUNKS:-4}
JSON=logs/j1_oom_${SLURM_JOB_ID}.jsonl

echo "=== J1 memory price | micro=$MICRO_BS | encoding=$ENCODING | grad_chunks=$CARRY_GRAD_CHUNKS ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
if ! grep -q "read_into(x, current_step)" "$MODEL/raven_modeling_minimal_cortex.py" 2>/dev/null; then
    echo "ERROR: $MODEL carries a PRE-P3.0 modeling file -- this would price a"
    echo "       model without the in-loop read.  Refresh it first:"
    echo "         python tools/prepare_cortex_checkpoint.py --src $MODEL --dst $MODEL --variant olmo"
    exit 1
fi

COMMON="--model_name $MODEL --max_length 4096 --cross_chunks 8
        --micro_batch_size $MICRO_BS --batch_size 4 --max_steps 2000
        --mean_recurrence 8 --prefix_memory gated --accum_vecs 16
        --gate_slots 64 --gate_route ring --gate_norm tanh --gate_init zero
        --gate_fill grow --latent_carry --carry_grad_chunks $CARRY_GRAD_CHUNKS
        --use_memory true --with_optimizer true --json_out $JSON
        --set latent_encoding=$ENCODING --set latent_s0_read=false"

BROKEN=0
echo ""
echo "######## real limb: LatentRead xattn + znorm ########"
python tools/smoke_geometry_oom.py $COMMON \
    --set latent_read=xattn --set latent_read_heads=8 \
    --set latent_read_znorm=rms || BROKEN=1
echo ""
echo "######## noread limb: write only ########"
python tools/smoke_geometry_oom.py $COMMON \
    --set latent_read=none --set latent_write_only=true || BROKEN=1

echo ""
python - "$JSON" <<'SUMMARY'
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1])]
print(f"{'limb':>7} {'micro':>5} {'status':>6} {'alloc':>8} {'reserved':>9} {'%dev':>5} {'s/step':>7}")
for r in rows:
    sets = " ".join(r.get("sets", []))
    limb = "real" if "latent_read=xattn" in sets else "noread"
    pct = 100 * r["peak_res_gib"] / r["total_gib"]
    print(f"{limb:>7} {r.get('micro_batch_size', '?'):>5} {r['status']:>6} "
          f"{r['peak_alloc_gib']:>8.2f} {r['peak_res_gib']:>9.2f} {pct:>5.0f} "
          f"{r['step_s']:>7.1f}")
real = [r for r in rows if "latent_read=xattn" in " ".join(r.get("sets", []))]
print("")
if real and real[0]["status"] == "FITS":
    print("J1: the real/donor limb FITS at this micro-batch -- all three limbs can launch.")
elif real:
    print("J1: the real/donor limb DOES NOT FIT.  Do NOT drop to micro 1 (the donor")
    print("    roll would return the same row).  Lower CARRY_GRAD_CHUNKS and re-price.")
else:
    print("J1: no real-limb row in the json -- the measurement did not run.")
SUMMARY

if [ $BROKEN -ne 0 ]; then
    echo ""
    echo "NOTE: a row failed for a NON-OOM reason (the script exits 0 on an OOM)."
    echo "      Read the traceback above; the table is incomplete."
    exit 1
fi
