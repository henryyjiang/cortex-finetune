"""
Full end-to-end deployment smoke for the cortex graft on retrofitting-recurrence.

Exercises the REAL pipeline that the unit tests bypass:

    build a tiny base checkpoint  (save_pretrained)
      -> tools/prepare_cortex_checkpoint.py  (install grafted modeling file +
         auto_map redirect + memory flags)
      -> AutoModelForCausalLM.from_pretrained(trust_remote_code=True)  (loads the
         grafted file from the snapshot; its `from cortex_graft import ...`
         resolves via the repo root on sys.path)
      -> forward (M_cross returned)
      -> cross-chunk train step (mirror of train.py cortex_fwd_bwd; M_cross write
         path gets gradient through the un-detached chain)
      -> eval harness (eval_babilong.eval_one with chunked M_cross carry)

This validates auto_map + trust_remote_code loading + the cortex import path,
none of which the temp-package unit tests cover.

Env: cortex-retro (transformers ~4.51, + accelerate, safetensors).  Run from the
repo root so `from cortex_graft import ...` resolves:

    python tests/smoke_end_to_end.py

Skips cleanly if raven_config_minimal.py is unavailable or the base cannot build.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile

import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
_CONFIG_SRC = os.path.join(os.path.dirname(REPO), "recurrent-pretraining",
                           "recpre", "raven_config_minimal.py")
VOCAB = 256


class Tok:
    """Whitespace stub tokenizer (ids in [1, VOCAB))."""
    eos_token_id = 0
    pad_token_id = 0
    eos_token = "<e>"
    pad_token = "<p>"

    class O:
        def __init__(self, ids):
            self.input_ids = ids

    def __call__(self, text, add_special_tokens=False, return_tensors=None):
        ids = [(abs(hash(w)) % (VOCAB - 1)) + 1 for w in text.split()] or [1]
        return Tok.O(torch.tensor([ids]) if return_tensors == "pt" else ids)

    def decode(self, ids, skip_special_tokens=False):
        if isinstance(ids, int):
            ids = [ids]
        return " ".join(str(int(i)) for i in ids)


def main() -> int:
    if not os.path.exists(_CONFIG_SRC):
        print(f"SKIP: {_CONFIG_SRC} not found.")
        return 0

    tmp = tempfile.mkdtemp(prefix="e2e_")
    base = os.path.join(tmp, "base")
    os.makedirs(base)
    bpkg = os.path.join(tmp, "bpkg")
    os.makedirs(bpkg)
    open(os.path.join(bpkg, "__init__.py"), "w").close()
    shutil.copy(os.path.join(REPO, "convert_pretrained_model", "raven_modeling_minimal_olmo.py"),
                os.path.join(bpkg, "raven_modeling_minimal.py"))
    shutil.copy(_CONFIG_SRC, os.path.join(bpkg, "raven_config_minimal.py"))
    sys.path.insert(0, tmp)
    try:
        from bpkg.raven_config_minimal import RavenConfig
        from bpkg.raven_modeling_minimal import RavenForCausalLM
    except Exception as e:
        print(f"SKIP: cannot import raven (transformers skew?): {type(e).__name__}: {e}")
        shutil.rmtree(tmp, ignore_errors=True)
        return 0

    cfg = RavenConfig(
        n_embd=64, n_heads=4, n_layers=4, block_size=64, vocab_size=VOCAB,
        padding_multiple=1, intermediate_size=128, mean_recurrence=4,
        mean_backprop_depth=2, n_layers_in_prelude=1, n_layers_in_recurrent_block=2,
        n_layers_in_coda=1, tie_embeddings=False, max_position_embeddings=128,
        rope_theta=10000.0,
    )
    try:
        RavenForCausalLM(cfg).save_pretrained(base)
    except Exception as e:
        print(f"SKIP: cannot build/save base (transformers skew?): {type(e).__name__}: {e}")
        shutil.rmtree(tmp, ignore_errors=True)
        return 0
    shutil.copy(os.path.join(bpkg, "raven_config_minimal.py"), os.path.join(base, "raven_config_minimal.py"))
    shutil.copy(os.path.join(bpkg, "raven_modeling_minimal.py"), os.path.join(base, "raven_modeling_minimal.py"))
    # save_pretrained injects tie_word_embeddings which RavenConfig also forwards
    # to super() -> duplicate kwarg on reload; real checkpoints don't carry it.
    cj = json.load(open(os.path.join(base, "config.json")))
    cj.pop("tie_word_embeddings", None)
    json.dump(cj, open(os.path.join(base, "config.json"), "w"))
    print("[1] base checkpoint saved")

    dst = os.path.join(tmp, "cortex")
    r = subprocess.run(
        [sys.executable, os.path.join(REPO, "tools", "prepare_cortex_checkpoint.py"),
         "--src", base, "--dst", dst, "--variant", "olmo", "--use_memory", "--memory_slots", "4"],
        capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stderr
    print("[2] prepared with graft (auto_map + memory flags)")

    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(dst, trust_remote_code=True,
                                                 torch_dtype=torch.float32).eval()
    ok = model.cortex is not None and model.cortex.has_cross_state
    print(f"[3] from_pretrained(trust_remote_code) OK | cortex active: {ok}")

    ids = torch.randint(0, VOCAB, (2, 16))
    o1 = model(input_ids=ids, num_steps=torch.tensor([0, 3]), return_m_cross=True)
    ok &= tuple(o1.m_cross.shape) == (2, 4, 64)
    print(f"[4] forward: logits {tuple(o1.logits.shape)} m_cross {tuple(o1.m_cross.shape)}")

    # cross-chunk train step (mirror of train.py cortex_fwd_bwd, incl. .contiguous())
    model.train()
    torch.nn.init.normal_(model.cortex.m_cross.out_proj.weight, std=0.05)
    labels = torch.randint(0, VOCAB, (2, 16))
    xs = [c.contiguous() for c in torch.chunk(ids, 2, 1)]
    ys = [c.contiguous() for c in torch.chunk(labels, 2, 1)]
    mc, losses = None, []
    for xc, yc in zip(xs, ys):
        out = model(input_ids=xc, labels=yc, num_steps=torch.tensor([0, 3]),
                    m_cross_in=mc, return_m_cross=True)
        mc = out["m_cross"]
        losses.append(out["loss"])
    torch.stack(losses).mean().backward()
    grad_ok = model.cortex.m_cross.gate_proj_in.weight.grad is not None
    ok &= grad_ok
    print(f"[5] cross-chunk train step: M_cross write-path grad present: {grad_ok}")

    sys.path.insert(0, os.path.join(REPO, "evals"))
    import eval_babilong as eb
    model.eval()
    ctx = " ".join(f"fact{i} mary office" for i in range(20))
    res = eb.eval_one(model, Tok(), ctx, "where mary", "office", T=3, seq_len=8)
    ok &= isinstance(res, bool)
    print(f"[6] eval_one (chunked M_cross carry) ran end-to-end -> {res}")

    shutil.rmtree(tmp, ignore_errors=True)
    print("\n=== FULL END-TO-END: " + ("PASS ===" if ok else "FAIL ==="))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
