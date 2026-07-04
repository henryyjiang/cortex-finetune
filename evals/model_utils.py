"""
Shared model-loading + interface helpers for the cortex evals on the raven
(RavenForCausalLM) model.

These adapt the cortex-main eval scripts (written against CortexGPT) to the
retrofitting-recurrence model with three thin shims:

  load_checkpoint(checkpoint, model_name, memory_slots, dtype, device)
      Load a raven model via from_pretrained(model_name, trust_remote_code).
      `model_name` should be a graft-prepared model dir (see
      tools/prepare_cortex_checkpoint.py) so the grafted modeling file + memory
      flags are active; passing memory_slots forces use_memory on the config.
      `checkpoint` (optional) is a torch .pt saved by train.py whose ["model"]
      state_dict is overlaid with strict=False (finetuned weights).
      Returns (model, config); config.mean_recurrence is the default eval T.

  has_cross_state(model) -> bool
      True if the grafted model carries cross-segment memory (M_cross / DirectCCoT).

  to_num_steps(T) -> Optional[torch.Tensor]
      Eval recurrence depth → raven num_steps. T iterations, all no-grad
      (eval runs under torch.no_grad anyway).  None → model uses its config
      mean_recurrence.

NOTE: run evals from the repo root so the grafted modeling file's
`from cortex_graft import ...` resolves.  Use the cortex-retro env
(transformers ~4.51); the cortex env's transformers 5.x cannot load the base.
"""
from __future__ import annotations

import os
import sys
from typing import Optional

import torch

# Allow importing cortex_graft / cortex_memory from the repo root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _unwrap(model):
    m = model
    if hasattr(m, "module"):
        m = m.module
    if hasattr(m, "_orig_mod"):
        m = m._orig_mod
    return m


def has_cross_state(model) -> bool:
    cortex = getattr(_unwrap(model), "cortex", None)
    return cortex is not None and cortex.has_cross_state


def to_num_steps(T: Optional[int]):
    if T is None:
        return None
    return torch.tensor([int(T), 0])


def load_checkpoint(
    checkpoint: Optional[str],
    model_name: str,
    memory_slots: Optional[int],
    dtype: torch.dtype,
    device: torch.device,
):
    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    if memory_slots is not None:
        # Force the graft on (model_name must use the grafted modeling file).
        config.use_memory = True
        config.memory_slots = memory_slots
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        config=config,
        torch_dtype=dtype,
    )

    # Optional overlay of finetuned weights from a train.py checkpoint.
    if checkpoint and os.path.isfile(checkpoint):
        sd = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if isinstance(sd, dict) and "model" in sd:
            sd = sd["model"]
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"[load] overlaid {checkpoint}: {len(missing)} missing / "
              f"{len(unexpected)} unexpected keys")

    # Fail loud if memory was requested but the graft didn't load: the grafted
    # modeling file falls back to CortexMemory=None when `import cortex_graft`
    # fails (e.g. evals launched from outside the repo root), which would
    # SILENTLY run as a no-memory baseline despite use_memory=True.
    if getattr(config, "use_memory", False) and getattr(_unwrap(model), "cortex", None) is None:
        raise RuntimeError(
            "config.use_memory is set but model.cortex is None — the cortex_graft "
            "import failed (run evals from the repo root) or model_name is not a "
            "graft-prepared dir. Eval would silently run as a no-memory baseline."
        )

    model = model.to(device=device, dtype=dtype).eval()
    return model, config
