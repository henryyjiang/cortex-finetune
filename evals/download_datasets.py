"""
Download eval datasets to local disk for use on compute nodes without internet.

Run once on a login node:
    HF_TOKEN=hf_... python evals/download_datasets.py

HF_TOKEN is required if the dataset is gated or rate-limited.
Get one at https://huggingface.co/settings/tokens
"""
import os
from pathlib import Path
from huggingface_hub import snapshot_download

ROOT = Path(__file__).parent.parent / "data"
ROOT.mkdir(exist_ok=True)

token = os.environ.get("HF_TOKEN") or None

if not (ROOT / "LongMemEval" / "longmemeval_oracle").exists():
    print("Downloading xiaowu0162/LongMemEval ...")
    snapshot_download(
        repo_id   = "xiaowu0162/LongMemEval",
        repo_type = "dataset",
        local_dir = str(ROOT / "LongMemEval"),
        token     = token,
    )
    print(f"Saved → {ROOT / 'LongMemEval'}")
else:
    print("xiaowu0162/LongMemEval already present, skipping.")

# Layout (verified): data/<task>/<length>.json.  Restrict to lengths up to
# 128k — the 256k/512k/1M/10M files are multi-GB and not used by the evals;
# downloading them is slow enough that snapshots were left incomplete.
BABILONG_LENGTHS = ("0k", "1k", "2k", "4k", "8k", "16k", "32k", "64k", "128k")

if (ROOT / "BABILong" / "data" / "qa1" / "1k.json").exists():
    print("RMT-team/BABILong already present, skipping.")
else:
    print("Downloading RMT-team/BABILong (lengths up to 128k) ...")
    snapshot_download(
        repo_id        = "RMT-team/BABILong",
        repo_type      = "dataset",
        local_dir      = str(ROOT / "BABILong"),
        allow_patterns = [f"data/*/{length}.json" for length in BABILONG_LENGTHS],
        token          = token,
    )
    print(f"Saved → {ROOT / 'BABILong'}")
