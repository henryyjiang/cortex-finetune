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


@torch.no_grad()
def prime_cross_state(model, chunks, num_steps, passes_per_chunk=1):
    """Run priming chunks through the model, carrying M_cross across them.
    passes_per_chunk > 1 runs each chunk through the FULL model that many
    times (M_cross carried pass-to-pass), so the buffer gets multiple writes
    per chunk instead of one — the multi-pass fill the LM2 buffer design
    intends.  Returns the final buffer, or None for models without cross
    state (base / parcae-style) — those see only the final prediction chunk,
    which is exactly the no-memory control condition."""
    if not has_cross_state(model) or not chunks:
        return None
    device = next(model.parameters()).device
    m_cross = None
    for chunk in chunks:
        chunk = chunk.to(device)
        for _ in range(max(passes_per_chunk, 1)):
            out = model(input_ids=chunk, num_steps=num_steps,
                        m_cross_in=m_cross, return_m_cross=True)
            m_cross = out.get("m_cross")
    return m_cross


@torch.no_grad()
def ccot_prime(model, input_ids, num_steps, passes, m_cross_init=None):
    """Mixed CCoT: run `passes` full silent forward passes over the SAME
    tokens, feeding each pass's M_cross write into the next pass's read —
    latent multi-pass 'thinking' before any token is generated.  m_cross_init
    seeds the first pass (e.g. a buffer primed on earlier context chunks).
    Returns the final buffer (m_cross_init unchanged when the model has no
    cross state or passes <= 0)."""
    if passes <= 0 or not has_cross_state(model):
        return m_cross_init
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    m_cross = m_cross_init
    for _ in range(passes):
        out = model(input_ids=input_ids, num_steps=num_steps,
                    m_cross_in=m_cross, return_m_cross=True)
        m_cross = out.get("m_cross")
    return m_cross


@torch.no_grad()
def greedy_generate(model, tokenizer, input_ids, max_new_tokens, num_steps,
                    m_cross=None, stop_on_newline=False):
    """Greedy decoding by full re-forward each step (no KV cache — matches the
    original eval_gsm8k generate).  An optional primed m_cross buffer is held
    fixed as read-only context for every step.  Returns the generated text."""
    device = next(model.parameters()).device
    generated = input_ids.to(device)
    prompt_len = generated.shape[1]
    eos_id = tokenizer.eos_token_id
    for _ in range(max_new_tokens):
        out = model(input_ids=generated, num_steps=num_steps,
                    m_cross_in=m_cross, return_m_cross=False)
        next_tok = out["logits"][0, -1].argmax(dim=-1).view(1, 1)
        generated = torch.cat([generated, next_tok], dim=1)
        if eos_id is not None and next_tok.item() == eos_id:
            break
        if stop_on_newline and "\n" in tokenizer.decode(generated[0, prompt_len:]):
            break
    return tokenizer.decode(generated[0, prompt_len:], skip_special_tokens=True)
