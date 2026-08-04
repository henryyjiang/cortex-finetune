"""
KV-cached decoding vs the re-forward reference, on the REAL raven model.

Not part of the pytest suite, for the same reason as smoke_real_model.py: the
thing under test is the base model's attention + HuginnDynamicCache, which the
fake models in tests/ do not implement.  It runs at toy size, so it needs no
checkpoint and no GPU.

WHY init_scale=0.0 EVERYWHERE.  These two paths can never be bit-identical at
the default init: initialize_state draws s0 ~ trunc_normal(std=sqrt(2/(5*d)))
shaped like whatever it is handed (raven_modeling_minimal_olmo.py:982), so a
forward over one token and a forward over the whole prefix consume different
draws.  init_scale=0.0 sends std to 0 and s0 to a deterministic zero, which
isolates the cache and packing arithmetic — the part that IS supposed to be
exact — from noise that is inherent to the architecture.  A failure here is a
real cache bug; the residual noise at init_scale=1.0 is not.

WHAT WOULD BE SILENT WITHOUT THIS.  Case 3 is the one that matters.  A cached
single-token query runs with is_causal=False (raven_modeling_minimal_olmo.py:450)
and therefore attends to EVERY cached key.  If the summary slots are packed
during a cached prefill they land in the cache and the generated tokens read
them — something the uncached causal mask never permits, because the slots sit
after the real tokens.  The output stays fluent, so nothing downstream would
flag it.  prefix_write=False is what prevents it, and case 3 fails loudly if
that ever regresses.

Run:
    /c/Users/henry/miniconda3/envs/cortex-retro/python.exe tests/smoke_kv_cache.py [olmo|llama]
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys

import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from smoke_real_model import _build_pkg, _DEFAULT_CONFIG_SRC  # noqa: E402

TOL = 2e-4          # bf16-free float32 path; slack for attention re-association


def decode_uncached(model, ids, n_new, num_steps, m_cross=None):
    """Reference: re-forward the whole prefix every step (the original harness).
    Returns the stacked next-token logits, one row per generated step."""
    gen, rows = ids, []
    for _ in range(n_new):
        out = model(input_ids=gen, num_steps=num_steps, m_cross_in=m_cross,
                    return_m_cross=False, init_scale=0.0, prefix_write=False)
        logits = out.logits[:, -1]
        rows.append(logits)
        gen = torch.cat([gen, logits.argmax(-1, keepdim=True)], dim=1)
    return torch.stack(rows)


def decode_cached(model, ids, n_new, num_steps, m_cross=None,
                  prefix_write=False):
    """Prefill once, then one token per step against the cache.  prefix_write
    is a parameter only so case 3 can demonstrate the incorrect setting."""
    S = ids.shape[1]
    pos = torch.arange(S).unsqueeze(0)
    out = model(input_ids=ids, num_steps=num_steps, position_ids=pos,
                m_cross_in=m_cross, return_m_cross=False, init_scale=0.0,
                use_cache=True, prefix_write=prefix_write, prefix_read=True)
    cache = out.past_key_values
    logits = out.logits[:, -1]
    rows = [logits]
    nxt = logits.argmax(-1, keepdim=True)
    for i in range(n_new - 1):
        pos = torch.tensor([[S + i]])
        out = model(input_ids=nxt, num_steps=num_steps, position_ids=pos,
                    m_cross_in=m_cross, return_m_cross=False, init_scale=0.0,
                    past_key_values=cache, use_cache=True,
                    prefix_write=False, prefix_read=False)
        cache = out.past_key_values
        logits = out.logits[:, -1]
        rows.append(logits)
        nxt = logits.argmax(-1, keepdim=True)
    return torch.stack(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("variant", nargs="?", default="olmo", choices=["olmo", "llama"])
    ap.add_argument("--config_dir", default=None)
    args = ap.parse_args()

    config_src = (os.path.join(args.config_dir, "raven_config_minimal.py")
                  if args.config_dir else _DEFAULT_CONFIG_SRC)
    if not os.path.exists(config_src):
        print(f"SKIP: raven_config_minimal.py not found at {config_src}.")
        return 0

    tmp = _build_pkg(args.variant, config_src)
    try:
        from smokepkg.raven_config_minimal import RavenConfig
        from smokepkg.raven_modeling_minimal import RavenForCausalLM
    except Exception as e:
        print(f"SKIP: could not import the raven model (transformers skew — "
              f"target ~4.51): {type(e).__name__}: {e}")
        shutil.rmtree(tmp, ignore_errors=True)
        return 0

    def make_cfg(**flags):
        return RavenConfig(
            n_embd=64, n_heads=4, n_layers=4, block_size=64, vocab_size=128,
            padding_multiple=1, intermediate_size=128, mean_recurrence=4,
            mean_backprop_depth=2, n_layers_in_prelude=1,
            n_layers_in_recurrent_block=2, n_layers_in_coda=1,
            tie_embeddings=False, max_position_embeddings=64,
            rope_theta=10000.0, **flags,
        )

    B, S, NEW, EOS = 1, 12, 6, 127
    D, NV = 64, 4
    ids = torch.randint(0, 128, (B, S))
    num_steps = torch.tensor([0, 2])
    ok = True

    try:
        torch.manual_seed(0)
        base = RavenForCausalLM(make_cfg()).eval()
    except Exception as e:
        print(f"SKIP: base model would not instantiate: {type(e).__name__}: {e}")
        shutil.rmtree(tmp, ignore_errors=True)
        return 0

    # ── 1. no memory: the inherited cache must already be exact ──────────────
    with torch.no_grad():
        a = decode_uncached(base, ids, NEW, num_steps)
        b = decode_cached(base, ids, NEW, num_steps)
    d = (a - b).abs().max().item()
    good = d < TOL
    ok &= good
    print(f"[base]   cached == uncached over {NEW} steps: {good}  (max |d| {d:.2e})")

    # ── 2. prefix memory with a live carry ──────────────────────────────────
    torch.manual_seed(0)
    pm = RavenForCausalLM(make_cfg(use_memory=True, prefix_memory="accum",
                                   accum_vecs=NV, accum_max=32,
                                   eos_token_id=EOS)).eval()
    with torch.no_grad():
        carry = pm(input_ids=ids, num_steps=num_steps, return_m_cross=True,
                   init_scale=0.0).m_cross.detach()
        a = decode_uncached(pm, ids, NEW, num_steps, m_cross=carry)
        b = decode_cached(pm, ids, NEW, num_steps, m_cross=carry)
    d = (a - b).abs().max().item()
    good = d < TOL
    ok &= good
    print(f"[prefix] cached == uncached with a {tuple(carry.shape)} carry: "
          f"{good}  (max |d| {d:.2e})")

    # ── 3. the bug the flag prevents ────────────────────────────────────────
    # Prefill WITH summary slots, then decode against that cache.  The slots are
    # now cached keys, and is_causal=False lets the generated tokens attend to
    # them.  This must differ from the reference — if it ever matches, either
    # the slots stopped being cached or the causal guarantee changed, and the
    # correctness argument for prefix_write=False needs rechecking.
    with torch.no_grad():
        bad = decode_cached(pm, ids, NEW, num_steps, m_cross=carry,
                            prefix_write=True)
    d_bad = (a - bad).abs().max().item()
    detected = d_bad > TOL
    ok &= detected
    print(f"[prefix] slots left in the cache DO corrupt decoding (so the flag "
          f"is load-bearing): {detected}  (max |d| {d_bad:.2e})")

    # ── 4. write=False does not change real-token logits ────────────────────
    # Same claim as the causal-mask argument, but measured end-to-end on the
    # real attention rather than asserted from the layout.
    with torch.no_grad():
        with_slots = pm(input_ids=ids, num_steps=num_steps, m_cross_in=carry,
                        init_scale=0.0, prefix_write=True).logits
        no_slots = pm(input_ids=ids, num_steps=num_steps, m_cross_in=carry,
                      init_scale=0.0, prefix_write=False).logits
    same_shape = with_slots.shape == no_slots.shape == (B, S, 128)
    d = (with_slots - no_slots).abs().max().item()
    good = same_shape and d < TOL
    ok &= good
    print(f"[prefix] dropping summary slots leaves real-token logits intact: "
          f"{good}  (shape {tuple(no_slots.shape)}, max |d| {d:.2e})")

    shutil.rmtree(tmp, ignore_errors=True)
    print("\nPASS" if ok else "\nFAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
