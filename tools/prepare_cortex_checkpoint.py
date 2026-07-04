"""
Prepare a McLeish (or local) raven checkpoint so it loads with the cortex
memory graft active.

Why this is needed
------------------
`from_pretrained(..., trust_remote_code=True)` executes the modeling file that
ships INSIDE the checkpoint snapshot.  McLeish's trained checkpoints carry the
unmodified `raven_modeling_minimal.py`, so to add memory we must point the
checkpoint at OUR grafted modeling file.  This script materialises a local,
ready-to-load model directory:

  1. copies the source checkpoint (HF hub id or local dir) into --dst
  2. copies the grafted convert_pretrained_model/raven_modeling_minimal_<variant>.py
     into --dst as raven_modeling_minimal_cortex.py
  3. patches --dst/config.json:
       - auto_map.AutoModelForCausalLM -> "raven_modeling_minimal_cortex.RavenForCausalLM"
       - memory flags (use_memory / memory_slots / ...) if you pass them here
         (you can also leave them off and pass --cortex.* flags to train.py;
          train.py re-applies its cortex flags onto the config at runtime).

Then point train.py at it:

    python train.py --model_name <dst> \
        --cortex.use_memory true --cortex.memory_slots 4 --cortex.cross_chunks 4 ...

IMPORTANT: run train.py from the retrofitting-recurrence repo root so the
grafted modeling file's `from cortex_graft import ...` resolves (cortex_graft.py
and cortex_memory/ live at the repo root, which is sys.path[0] when you launch
`python train.py`).  The grafted file does NOT need to be copied with its cortex
deps — only the modeling file goes into the snapshot.

Loading note: the checkpoint has no cortex.* weights, so HF reports them as
"newly initialized" — that is correct (zero/identity init → step-0 == base model).

Usage
-----
    python tools/prepare_cortex_checkpoint.py \
        --src tomg-group-umd/Recurrent-OLMo-2-... --dst ./ckpts/olmo-cortex \
        --variant olmo --use_memory --memory_slots 4
"""
from __future__ import annotations

import argparse
import json
import os
import shutil

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _materialise_source(src: str, dst: str) -> None:
    if os.path.isdir(src):
        shutil.copytree(src, dst, dirs_exist_ok=True)
        return
    # treat as an HF hub id
    from huggingface_hub import snapshot_download
    local = snapshot_download(repo_id=src)
    shutil.copytree(local, dst, dirs_exist_ok=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="HF hub id or local checkpoint dir")
    ap.add_argument("--dst", required=True, help="output model dir (created)")
    ap.add_argument("--variant", required=True, choices=["olmo", "llama"])
    # optional: bake memory flags into config.json (else pass --cortex.* to train.py)
    ap.add_argument("--use_memory", action="store_true")
    ap.add_argument("--memory_slots", type=int, default=None)
    ap.add_argument("--memory_slots_iter", type=int, default=None)
    ap.add_argument("--memory_heads", type=int, default=None)
    ap.add_argument("--ccot_direct", action="store_true")
    args = ap.parse_args()

    grafted = os.path.join(REPO, "convert_pretrained_model",
                           f"raven_modeling_minimal_{args.variant}.py")
    if not os.path.exists(grafted):
        raise FileNotFoundError(grafted)

    os.makedirs(args.dst, exist_ok=True)
    print(f"[1/3] materialising {args.src} -> {args.dst}")
    _materialise_source(args.src, args.dst)

    print(f"[2/3] installing grafted modeling file ({args.variant})")
    shutil.copy(grafted, os.path.join(args.dst, "raven_modeling_minimal_cortex.py"))

    print("[3/3] patching config.json (auto_map + memory flags)")
    cfg_path = os.path.join(args.dst, "config.json")
    with open(cfg_path, encoding="utf-8") as f:
        cfg = json.load(f)
    cfg.setdefault("auto_map", {})
    cfg["auto_map"]["AutoModelForCausalLM"] = "raven_modeling_minimal_cortex.RavenForCausalLM"
    if args.use_memory:
        cfg["use_memory"] = True
    for k in ("memory_slots", "memory_slots_iter", "memory_heads"):
        v = getattr(args, k)
        if v is not None:
            cfg[k] = v
    if args.ccot_direct:
        cfg["ccot_direct"] = True
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    print(f"\nDone. Train with (from the repo root):\n"
          f"  python train.py --model_name {args.dst} "
          f"--cortex.use_memory true --cortex.memory_slots 4 --cortex.cross_chunks 4 ...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
