#!/bin/bash
#SBATCH --account=gts-aivanova7-lab
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H200:1
#SBATCH --mem=128GB
#SBATCH -t 1:00:00
#SBATCH -q inferno
#SBATCH -J j4_price
#SBATCH -o logs/Report-%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=henryyjiang42@gmail.com

# J4's MEMORY PRICE -- "before the first launch" item 1 in pace/j4_joint.sbatch.
#
#   REQUIRE_FIT=1 sbatch pace/j4_price_oom.sbatch
#
# Three rows at J4's exact geometry (4096 x cc8, gated W=16 K=64, Z on,
# carry_grad_chunks 4, mr8, MICRO_BS=2, worst-case recurrence split, fp32
# weights + bf16 autocast, real Muon/Adam state, --window_backward --
# tools/smoke_geometry_oom.py):
#   real     the Z columns carry this document's Z: the operating limb
#   noread   the Z columns are ZEROS.  Priced SEPARATELY and not assumed equal,
#            because under J4 the no-read limb still pays for the columns --
#            that is the whole point of blanking rather than omitting them, and
#            this row is what proves the two limbs cost the same.  (Under J3 the
#            two differed in modules; here they differ only in contents, so a
#            gap between these two rows would mean something is wrong.)
#   e-only   latent_carry off entirely: the 592-column baseline, so the +11%
#            sequence has a measured price rather than an estimated one.
#
# WHAT J4 ADDS AND DROPS against J3 (92.12 alloc / 92.42 reserved GiB of 140,
# 1.7 s/micro, job 13483998) and J1 (88.48 / 88.89, 1.62 s, job 13459697):
#   + 64 more packed columns, 592 -> 656.  This is the term to watch: it is
#     ~+11% on the sequence dimension and it is paid on EVERY one of the T loop
#     iterations, in activations and in attention.
#   + one D x D projection, 4.19M parameters, with two Adam moments.
#   - J3's per-iteration [B, S_packed, D] scratchpad state in the graph.
#   - J3's 16.8M scratchpad gate parameters and their Adam state.
#   - J1's and J3's 16.8M-parameter xattn read module (q/k/v/out) entirely.
# So J4 should land NEAR J1 rather than above J3 -- but the +11% sequence is not
# a term either of them paid, so it is measured, not assumed.
#
#   REQUIRE_FIT=1   exit 3 unless the real limb FITS, so this job can gate the
#                   launch:  sbatch --dependency=afterok:<this job> pace/j4_joint.sbatch

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
ZNORM_TARGET=${ZNORM_TARGET:-171.0}
JSON=logs/j4_oom_${SLURM_JOB_ID}.jsonl

echo "=== J4 memory price | micro=$MICRO_BS | grad_chunks=$CARRY_GRAD_CHUNKS | window_backward=1 ==="
echo "    packed: 64 E + 64 Z + 512 tokens + 16 summary = 656 columns (E-only: 592)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
if ! grep -q "iter_write(x, current_step)" "$MODEL/raven_modeling_minimal_cortex.py" 2>/dev/null; then
    echo "ERROR: $MODEL carries a modeling file without iter_write(x, current_step)"
    echo "       -- this would price a model whose Z write never fires.  Refresh it:"
    echo "         python tools/prepare_cortex_checkpoint.py --src $MODEL --dst $MODEL --variant olmo"
    exit 1
fi

BASE="--model_name $MODEL --max_length 4096 --cross_chunks 8
      --micro_batch_size $MICRO_BS --batch_size 4 --max_steps 2000
      --mean_recurrence 8 --prefix_memory gated --accum_vecs 16
      --gate_slots 64 --gate_route ring --gate_norm tanh --gate_init zero
      --gate_fill grow --carry_grad_chunks $CARRY_GRAD_CHUNKS
      --window_backward
      --use_memory true --with_optimizer true --json_out $JSON"
COMMON="$BASE --latent_carry
        --set latent_encoding=endpoint --set latent_s0_read=false
        --set latent_read=embeds
        --set latent_read_znorm=rms --set latent_read_znorm_target=$ZNORM_TARGET"

BROKEN=0
echo ""
echo "######## real limb: 656 columns, Z carried ########"
python tools/smoke_geometry_oom.py $COMMON --set latent_carry_read=true || BROKEN=1
echo ""
echo "######## noread limb: 656 columns, Z columns zero ########"
python tools/smoke_geometry_oom.py $COMMON --set latent_carry_read=false || BROKEN=1
echo ""
echo "######## E-only baseline: 592 columns, no Z channel ########"
python tools/smoke_geometry_oom.py $BASE || BROKEN=1

echo ""
python - "$JSON" <<'SUMMARY'
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1])]
print(f"{'arm':>8} {'micro':>5} {'status':>6} {'alloc':>8} {'reserved':>9} {'%dev':>5} {'s/step':>7}")


def arm_of(r):
    sets = " ".join(r.get("sets", []))
    if "latent_read=embeds" not in sets:
        return "e-only"
    return "noread" if "latent_carry_read=false" in sets else "real"


for r in rows:
    pct = 100 * r["peak_res_gib"] / r["total_gib"]
    print(f"{arm_of(r):>8} {r.get('micro_batch_size', '?'):>5} {r['status']:>6} "
          f"{r['peak_alloc_gib']:>8.2f} {r['peak_res_gib']:>9.2f} {pct:>5.0f} "
          f"{r['step_s']:>7.1f}")
by = {arm_of(r): r for r in rows}
print("")
print("References: J1 88.48 alloc / 88.89 reserved GiB, 1.62 s/micro (job 13459697)")
print("            J3 92.12 / 92.42, 1.70 s (job 13483998).  Device is 140 GiB.")
real, nr, eo = by.get("real"), by.get("noread"), by.get("e-only")
if real and eo:
    print(f"THE +11% SEQUENCE COSTS: {real['peak_res_gib'] - eo['peak_res_gib']:+.2f} GiB "
          f"reserved and {real['step_s'] - eo['step_s']:+.2f} s/micro over the "
          f"592-column E-only arm.")
if real and nr:
    gap = abs(real["peak_res_gib"] - nr["peak_res_gib"])
    print(f"real vs noread: {gap:.2f} GiB apart.  These two limbs differ only in "
          f"the CONTENTS of the Z columns, so anything but a small gap means the "
          f"no-read limb is not the geometric twin it is supposed to be.")
if real and real["status"] == "FITS":
    print("J4: the real/donor limb FITS at this micro-batch -- all three limbs can launch.")
    print("    Record the numbers in pace/j4_joint.sbatch's DECISIONS block (MEMORY).")
elif real:
    print("J4: the real/donor limb DOES NOT FIT.  Do NOT drop to micro 1 (the donor")
    print("    roll would return the same row).  The new term is 64 packed columns")
    print("    on every loop iteration -- the honest cuts are MAX_LENGTH/cross_chunks")
    print("    or carry_grad_chunks, NOT the Z column count (K is the ring's width;")
    print("    shrinking it changes the design, not its price).")
else:
    print("J4: no real-limb row in the json -- the measurement did not run.")
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
real = [r for r in rows
        if "latent_read=embeds" in " ".join(r.get("sets", []))
        and "latent_carry_read=true" in " ".join(r.get("sets", []))]
sys.exit(0 if real and real[0]["status"] == "FITS" else 1)
GATE
    then
        echo ""
        echo "REQUIRE_FIT: the real limb did not fit -- exit 3, so any job queued"
        echo "             --dependency=afterok on this one will NOT start."
        exit 3
    fi
fi
