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


def prefix_checks(RavenForCausalLM, make_cfg, ids, B, S, run) -> bool:
    """AutoCompressor-faithful prefix memory, on the REAL model.

    This path has no zero-init read module: the carry is spliced into the token
    stream and consumed by the base model's own attention, so it is LIVE from
    step 0.  Every check below is about the splice being correct — shapes, the
    real-token span, positions, the EOS lanes, and the summary-seeding flag —
    because a wrong splice still produces a perfectly healthy loss curve.
    """
    D, EOS = 64, 127
    ok = True

    # ── accum: shapes, growth, liveness ─────────────────────────────────────
    torch.manual_seed(0)
    ma = RavenForCausalLM(make_cfg(use_memory=True, prefix_memory="accum",
                                   accum_vecs=8, accum_max=24,
                                   eos_token_id=EOS)).eval()
    o1 = run(ma, return_m_cross=True)
    shape1 = tuple(o1.m_cross.shape)
    logits_ok = tuple(o1.logits.shape) == (B, S, 128)
    ok &= shape1 == (B, 8, D) and logits_ok
    print(f"[prefix accum] first carry {shape1} == (B,8,D): {shape1 == (B, 8, D)} | "
          f"logits keep the real-token span {tuple(o1.logits.shape)}: {logits_ok}")

    o2 = run(ma, seed=1, m_cross_in=o1.m_cross.detach(), return_m_cross=True)
    grew = tuple(o2.m_cross.shape) == (B, 16, D)
    live = not torch.allclose(o2.logits, run(ma, seed=1, m_cross_in=None).logits)
    ok &= grew and live
    print(f"[prefix accum] accumulates 8->16: {grew} | carry changes logits at "
          f"init (no zero-init gate to open): {live}")

    # FIFO: 24-vector cap must clamp on the fourth chunk, not silently grow.
    st = o2.m_cross.detach()
    for _ in range(2):
        st = run(ma, seed=2, m_cross_in=st, return_m_cross=True).m_cross.detach()
    capped = tuple(st.shape) == (B, 24, D)
    ok &= capped
    print(f"[prefix accum] FIFO cap holds at accum_max=24: {capped}")

    # ── seeding: summary_emb == wte[EOS], and the flag survives save/load ────
    seeded = torch.allclose(ma.cortex.prefix.summary_emb[0],
                            ma.transformer.wte.weight[EOS])
    all_rows = torch.allclose(ma.cortex.prefix.summary_emb,
                              ma.transformer.wte.weight[EOS].expand(8, D))
    flag = bool(ma.cortex.prefix.summary_seeded)
    ok &= seeded and all_rows and flag
    print(f"[prefix seed] summary_emb == wte[eos] on every row: "
          f"{seeded and all_rows} | seeded flag set: {flag}")

    # A trained summary_emb must NOT be re-seeded when the checkpoint is
    # reloaded — the flag is a persistent buffer precisely so a resume cannot
    # reset the write path back to wte[eos] while the loss curve looks fine.
    with torch.no_grad():
        ma.cortex.prefix.summary_emb.add_(1.0)
    trained = ma.cortex.prefix.summary_emb.clone()
    torch.manual_seed(3)
    reloaded = RavenForCausalLM(make_cfg(use_memory=True, prefix_memory="accum",
                                         accum_vecs=8, accum_max=24,
                                         eos_token_id=EOS)).eval()
    in_sd = "cortex.prefix.summary_seeded" in ma.state_dict()
    reloaded.load_state_dict(ma.state_dict())
    run(reloaded, seed=4)                      # first forward would re-seed
    kept = torch.allclose(reloaded.cortex.prefix.summary_emb, trained)
    ok &= in_sd and kept
    print(f"[prefix seed] flag rides in the state dict: {in_sd} | "
          f"reload+forward keeps the trained summary_emb: {kept}")

    # ── gradient reaches summary_emb through the chunk chain ────────────────
    torch.manual_seed(0)
    mg = RavenForCausalLM(make_cfg(use_memory=True, prefix_memory="accum",
                                   accum_vecs=4, accum_max=32,
                                   eos_token_id=EOS)).train()
    labels = torch.randint(0, 128, (B, S))
    c1 = mg(ids, num_steps=torch.tensor([0, 3]), return_m_cross=True)
    c2 = mg(ids, num_steps=torch.tensor([0, 3]), labels=labels,
            m_cross_in=c1.m_cross)             # un-detached: chunk 2 reads chunk 1
    c2.loss.backward()
    g = mg.cortex.prefix.summary_emb.grad
    grad_ok = g is not None and torch.isfinite(g).all() and g.abs().sum() > 0
    ok &= bool(grad_ok)
    print(f"[prefix grad] chunk-2 loss reaches chunk-1's summary_emb: {bool(grad_ok)}")

    # ── EOS: a lane whose document ended must not carry its vectors forward ──
    torch.manual_seed(0)
    me = RavenForCausalLM(make_cfg(use_memory=True, prefix_memory="accum",
                                   accum_vecs=4, accum_max=32,
                                   eos_token_id=EOS)).eval()
    prev = me(ids[:1], num_steps=torch.tensor([0, 2]), return_m_cross=True).m_cross
    eos = torch.zeros(1, S, dtype=torch.bool)
    eos[0, S - 1] = True                        # doc ends on the last position
    out = me(ids[:1], num_steps=torch.tensor([0, 2]), return_m_cross=True,
             eos_mask=eos, m_cross_in=prev)
    # incoming rows zeroed (ended doc) and this chunk's write zeroed (empty
    # open suffix) -> the whole carry is zero
    zeroed = torch.allclose(out.m_cross, torch.zeros_like(out.m_cross))
    ok &= zeroed
    print(f"[prefix EOS] ended-doc lane carries zero: {zeroed}")

    # ── gated: same splice, constant width ──────────────────────────────────
    torch.manual_seed(0)
    mgt = RavenForCausalLM(make_cfg(use_memory=True, prefix_memory="gated",
                                    accum_vecs=8, eos_token_id=EOS)).eval()
    g1 = run(mgt, return_m_cross=True).m_cross.detach()
    g2 = run(mgt, seed=1, m_cross_in=g1, return_m_cross=True).m_cross.detach()
    fixed_width = tuple(g1.shape) == (B, 8, D) and tuple(g2.shape) == (B, 8, D)
    moved = not torch.allclose(g1, g2)
    ok &= fixed_width and moved
    print(f"[prefix gated] width constant at n_vec across chunks: {fixed_width} "
          f"| gate actually updates the state: {moved}")

    # ── emb_scale: slots enter the network exactly as an EOS token would ─────
    # The toy config has embed_scale = sqrt(n_embd) = 8, so this catches the
    # mismatch that a scale-1 checkpoint would hide.
    scale_ok = abs(float(ma.emb_scale) - 8.0) < 1e-6
    ma.cortex.begin(None, None, S, torch.device("cpu"), torch.float32)  # clear carry
    packed, _, n_pre, n_sum = ma.cortex.prefix_pack(
        torch.zeros(B, S, D), torch.arange(S).unsqueeze(0), ma.emb_scale)
    slot_matches_token = torch.allclose(
        packed[0, -1], ma.cortex.prefix.summary_emb[-1] * ma.emb_scale)
    ok &= scale_ok and slot_matches_token and n_pre == 0 and n_sum == 8
    print(f"[prefix emb_scale] toy scale is {float(ma.emb_scale):.3f} (non-1, so "
          f"the check bites): {scale_ok} | slot enters at emb_scale x "
          f"summary_emb: {slot_matches_token}")

    return ok


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

    ok &= prefix_checks(RavenForCausalLM, make_cfg, ids, B, S, run)

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
