"""
Make train.py-saved checkpoint dirs (final_checkpoint / model_only_chkpt_*)
loadable by the evals via from_pretrained(..., trust_remote_code=True).

Why this is needed
------------------
save_model_only() saves with plain `model.save_pretrained()`, which does NOT
copy the grafted modeling file into the save dir: in transformers 4.51 only
the *config* class gets registered for custom-code copying (AutoConfig calls
`register_for_auto_class()`; the AutoModel trust_remote_code path never sets
`_auto_class` on the model class).  So the saved config.json's auto_map points
at `raven_modeling_minimal_cortex.RavenForCausalLM` but the .py file is
missing from the dir -> from_pretrained fails.

Additionally, save_pretrained serializes `tie_word_embeddings` into
config.json whenever it differs from the transformers default, and
RavenConfig.__init__ ALSO passes tie_word_embeddings explicitly to
super().__init__ while forwarding **kwargs -> duplicate-kwarg TypeError on
reload (same issue tests/smoke_end_to_end.py works around for its tiny base).

Fixes applied per checkpoint dir (idempotent, pure file ops, no GPU/network):
  1. copy raven_modeling_minimal_cortex.py (and raven_config_minimal.py if
     missing) from --base
  2. pop "tie_word_embeddings" from config.json
  3. ensure auto_map.AutoModelForCausalLM -> raven_modeling_minimal_cortex...
  4. print the run's cortex flags so you can sanity-check which arm is which

Usage (login node, from the repo root):
    # fix every run's final_checkpoint under the runs root
    python tools/prepare_eval_checkpoint.py --base ckpts/olmo8-cortex \
        --runs_root cortex-retro-ft

    # or specific checkpoint dirs
    python tools/prepare_eval_checkpoint.py --base ckpts/olmo8-cortex \
        --dirs cortex-retro-ft/rung1-k4/final_checkpoint ...
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil

MODELING_FILE = "raven_modeling_minimal_cortex.py"
CONFIG_FILE = "raven_config_minimal.py"
CORTEX_FLAGS = ("use_memory", "memory_slots", "memory_slots_iter", "memory_heads",
                "ccot_direct", "h_T_proj", "lora_rank", "lora_alpha",
                "mean_recurrence", "mean_backprop_depth")


def fix_checkpoint(ckpt_dir: str, base_dir: str) -> None:
    cfg_path = os.path.join(ckpt_dir, "config.json")
    if not os.path.isfile(cfg_path):
        print(f"[skip] {ckpt_dir}: no config.json")
        return

    actions = []

    # 1. grafted modeling file (+ config file, usually already copied by the
    #    config-side custom_object_save).
    for fname in (MODELING_FILE, CONFIG_FILE):
        src = os.path.join(base_dir, fname)
        dst = os.path.join(ckpt_dir, fname)
        if not os.path.isfile(src):
            raise FileNotFoundError(f"{src} — is --base a graft-prepared dir?")
        if not os.path.isfile(dst):
            shutil.copy(src, dst)
            actions.append(f"copied {fname}")

    # 2 + 3. config.json surgery.
    with open(cfg_path, encoding="utf-8") as f:
        cfg = json.load(f)
    if cfg.pop("tie_word_embeddings", None) is not None:
        actions.append("popped tie_word_embeddings")
    auto_map = cfg.setdefault("auto_map", {})
    want = f"{MODELING_FILE[:-3]}.RavenForCausalLM"
    if auto_map.get("AutoModelForCausalLM") != want:
        auto_map["AutoModelForCausalLM"] = want
        actions.append("fixed auto_map")
    if actions:
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)

    # 4. per-run flag summary.
    flags = {k: cfg[k] for k in CORTEX_FLAGS if k in cfg}
    print(f"[ok]  {ckpt_dir}")
    print(f"      {'; '.join(actions) if actions else 'already prepared'}")
    print(f"      flags: {flags}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True,
                    help="graft-prepared base dir (e.g. ckpts/olmo8-cortex)")
    ap.add_argument("--runs_root", default=None,
                    help="fix <runs_root>/*/final_checkpoint")
    ap.add_argument("--dirs", nargs="+", default=None,
                    help="explicit checkpoint dirs to fix")
    args = ap.parse_args()

    dirs = list(args.dirs or [])
    if args.runs_root:
        dirs += sorted(glob.glob(os.path.join(args.runs_root, "*", "final_checkpoint")))
    if not dirs:
        ap.error("nothing to do — pass --runs_root and/or --dirs")

    for d in dirs:
        fix_checkpoint(d, args.base)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
