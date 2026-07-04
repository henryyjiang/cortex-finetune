"""
Real-model smoke test for the cortex graft on RavenForCausalLM.

This is NOT part of the pytest suite (the graft *logic* is covered by
tests/test_cortex_graft.py against a faithful fake model).  This script
instantiates the ACTUAL raven model file to verify the graft end-to-end:
self.cortex creation, the forward hooks firing, M_cross carry, DirectCCoT,
M_iter gradient through the cross-segment chain, and EOS handling.

Environment note
----------------
The raven modeling files target transformers ~4.51 (the OLMo2/Llama rotary
embedding API).  Under a much newer transformers (e.g. 5.x) the BASE model's
rotary embedding will fail to instantiate — that is unrelated to the graft.
Run this in an env matching retrofitting-recurrence's deps, e.g.:

    pip install "transformers==4.51.0"
    python tests/smoke_real_model.py [olmo|llama]

It needs a copy of raven_config_minimal.py (ships with every checkpoint
snapshot).  This script borrows the one from a sibling recurrent-pretraining
checkout if present; otherwise pass --config_dir pointing at a snapshot dir.

On a transformers-version mismatch it prints a clear SKIP rather than failing.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile

import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

_DEFAULT_CONFIG_SRC = os.path.join(
    os.path.dirname(REPO), "recurrent-pretraining", "recpre", "raven_config_minimal.py"
)


def _build_pkg(variant: str, config_src: str) -> str:
    tmp = tempfile.mkdtemp(prefix="smoke_raven_")
    pkg = os.path.join(tmp, "smokepkg")
    os.makedirs(pkg)
    open(os.path.join(pkg, "__init__.py"), "w").close()
    shutil.copy(
        os.path.join(REPO, "convert_pretrained_model", f"raven_modeling_minimal_{variant}.py"),
        os.path.join(pkg, "raven_modeling_minimal.py"),
    )
    shutil.copy(config_src, os.path.join(pkg, "raven_config_minimal.py"))
    sys.path.insert(0, tmp)
    return tmp


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("variant", nargs="?", default="olmo", choices=["olmo", "llama"])
    ap.add_argument("--config_dir", default=None,
                    help="dir containing raven_config_minimal.py (a checkpoint snapshot)")
    args = ap.parse_args()

    config_src = (
        os.path.join(args.config_dir, "raven_config_minimal.py")
        if args.config_dir else _DEFAULT_CONFIG_SRC
    )
    if not os.path.exists(config_src):
        print(f"SKIP: raven_config_minimal.py not found at {config_src}. "
              f"Pass --config_dir <snapshot>.")
        return 0

    tmp = _build_pkg(args.variant, config_src)
    try:
        from smokepkg.raven_config_minimal import RavenConfig
        from smokepkg.raven_modeling_minimal import RavenForCausalLM
    except Exception as e:  # transformers version skew on the base model imports
        print(f"SKIP: could not import the raven model (likely a transformers "
              f"version mismatch — target ~4.51): {type(e).__name__}: {e}")
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

    B, S = 2, 16
    ids = torch.randint(0, 128, (B, S))

    def run(m, seed=0, **kw):
        # Re-seed each forward: the base model's initialize_state draws a random
        # h0, so paired comparisons must share a seed to isolate the memory effect.
        torch.manual_seed(seed)
        return m(input_ids=ids, num_steps=torch.tensor([0, 2]), **kw)

    try:
        torch.manual_seed(0)
        m0 = RavenForCausalLM(make_cfg()).eval()
    except Exception as e:
        print(f"SKIP: base model would not instantiate (transformers skew?): "
              f"{type(e).__name__}: {e}")
        shutil.rmtree(tmp, ignore_errors=True)
        return 0

    ok = True
    o = run(m0)
    ok &= (m0.cortex is None) and (o.m_cross is None)
    print(f"[baseline] cortex None & no m_cross: {m0.cortex is None and o.m_cross is None}")

    torch.manual_seed(0)
    m1 = RavenForCausalLM(make_cfg(use_memory=True, memory_slots=4)).eval()
    o1 = run(m1, return_m_cross=True)
    ok &= tuple(o1.m_cross.shape) == (B, 4, 64)
    torch.nn.init.normal_(m1.cortex.m_cross.out_proj.weight, std=0.05)
    mc = o1.m_cross.detach()
    carry = not torch.allclose(run(m1, seed=1, m_cross_in=mc).logits,
                               run(m1, seed=1, m_cross_in=None).logits)
    ok &= carry
    print(f"[M_cross] shape ok & carry changes logits: {carry}")

    torch.manual_seed(0)
    m1b = RavenForCausalLM(make_cfg(use_memory=True, memory_slots=4)).eval()
    noop = torch.allclose(run(m1b, seed=2, m_cross_in=torch.randn(B, 4, 64)).logits,
                          run(m1b, seed=2, m_cross_in=None).logits)
    ok &= noop
    print(f"[M_cross] zero-init read is a no-op at init: {noop}")

    torch.manual_seed(0)
    m2 = RavenForCausalLM(make_cfg(use_memory=True, ccot_direct=True)).eval()
    ds = tuple(run(m2, return_m_cross=True).m_cross.shape)
    ok &= ds == (B, 1, 64)
    print(f"[DirectCCoT] m_cross shape {ds} == (B,1,D): {ds == (B,1,64)}")

    torch.manual_seed(0)
    m3 = RavenForCausalLM(make_cfg(use_memory=True, memory_slots=4, memory_slots_iter=4)).train()
    torch.nn.init.normal_(m3.cortex.m_cross.out_proj.weight, std=0.05)
    labels = torch.randint(0, 128, (B, S))
    out1 = m3(input_ids=ids, num_steps=torch.tensor([0, 3]), return_m_cross=True)
    out2 = m3(input_ids=ids, num_steps=torch.tensor([0, 3]), labels=labels, m_cross_in=out1.m_cross)
    out2.loss.backward()
    g_cross = m3.cortex.m_cross.gate_proj_in.weight.grad is not None
    g_iter = m3.cortex.m_iter.gate_proj_in.weight.grad is not None
    ok &= g_cross and g_iter
    print(f"[grad chain] M_cross grad {g_cross} | M_iter grad {g_iter}")

    torch.manual_seed(0)
    m4 = RavenForCausalLM(make_cfg(use_memory=True, memory_slots=4)).eval()
    eos = torch.zeros(1, S, dtype=torch.bool); eos[0, S - 1] = True
    mc4 = m4(input_ids=ids[:1], num_steps=torch.tensor([0, 2]),
             return_m_cross=True, eos_mask=eos).m_cross
    zero = torch.allclose(mc4, torch.zeros_like(mc4))
    ok &= zero
    print(f"[EOS] eos-at-last carries zero: {zero}")

    # LoRA-on-loop (rung 1b): hooks build on the real loop linears, B zero-init
    # is an exact no-op, and param names dodge the 'adapter'/'core_block'
    # freeze selector while carrying 'cortex' for optimizer routing.
    torch.manual_seed(0)
    m5 = RavenForCausalLM(make_cfg(use_memory=True, memory_slots=4,
                                   lora_rank=4, lora_alpha=8)).eval()
    lora_names = [n for n, _ in m5.named_parameters() if "cortex_lora" in n]
    built = m5.cortex_lora is not None and len(lora_names) > 0
    names_ok = all(("adapter" not in n) and ("core_block" not in n) for n in lora_names)
    torch.manual_seed(0)
    m5_ref = RavenForCausalLM(make_cfg(use_memory=True, memory_slots=4)).eval()
    m5_ref.load_state_dict(m5.state_dict(), strict=False)  # same base weights
    lora_noop = torch.allclose(run(m5, seed=3).logits, run(m5_ref, seed=3).logits)
    with torch.no_grad():
        for _B in m5.cortex_lora.B.values():
            _B.normal_(std=0.05)
    lora_live = not torch.allclose(run(m5, seed=3).logits, run(m5_ref, seed=3).logits)
    ok &= built and names_ok and lora_noop and lora_live
    print(f"[LoRA] built {built} | names dodge freeze {names_ok} | "
          f"zero-init no-op {lora_noop} | nonzero B changes logits {lora_live}")

    shutil.rmtree(tmp, ignore_errors=True)
    print("\n=== REAL-MODEL SMOKE: " + ("PASS ===" if ok else "FAIL ==="))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
