"""
Phase-3 end-to-end eval-harness test.

Runs the ACTUAL ported eval functions (eval_babilong.eval_one,
eval_gsm8k.generate, eval_multiple_choice.score_completion) against a real,
tiny grafted RavenForCausalLM with a stub tokenizer — validating the whole
eval pipeline (chunked M_cross carry, greedy generation, completion scoring)
end-to-end without downloading a 1B checkpoint or any dataset.

Skips cleanly if the raven base cannot be built (transformers version skew) or
raven_config_minimal.py is unavailable.

Run: /c/Users/henry/miniconda3/envs/cortex-retro/python.exe -m pytest tests/test_cortex_eval.py -v
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile

import pytest
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "evals"))

VOCAB = 256
_CONFIG_SRC = os.path.join(os.path.dirname(REPO), "recurrent-pretraining",
                           "recpre", "raven_config_minimal.py")


def _build_raven(**flags):
    """Tiny real RavenForCausalLM (olmo variant) with the cortex graft."""
    if not os.path.exists(_CONFIG_SRC):
        pytest.skip(f"raven_config_minimal.py not found at {_CONFIG_SRC}")
    tmp = tempfile.mkdtemp(prefix="evaltest_")
    pkg = os.path.join(tmp, "evalpkg")
    os.makedirs(pkg)
    open(os.path.join(pkg, "__init__.py"), "w").close()
    shutil.copy(os.path.join(REPO, "convert_pretrained_model", "raven_modeling_minimal_olmo.py"),
                os.path.join(pkg, "raven_modeling_minimal.py"))
    shutil.copy(_CONFIG_SRC, os.path.join(pkg, "raven_config_minimal.py"))
    sys.path.insert(0, tmp)
    try:
        from evalpkg.raven_config_minimal import RavenConfig
        from evalpkg.raven_modeling_minimal import RavenForCausalLM
    except Exception as e:
        shutil.rmtree(tmp, ignore_errors=True)
        pytest.skip(f"cannot import raven model (transformers skew?): {type(e).__name__}: {e}")
    cfg = RavenConfig(
        n_embd=64, n_heads=4, n_layers=4, block_size=64, vocab_size=VOCAB,
        padding_multiple=1, intermediate_size=128, mean_recurrence=4,
        mean_backprop_depth=2, n_layers_in_prelude=1, n_layers_in_recurrent_block=2,
        n_layers_in_coda=1, tie_embeddings=False, max_position_embeddings=128,
        rope_theta=10000.0, **flags,
    )
    try:
        model = RavenForCausalLM(cfg).eval()
    except Exception as e:
        shutil.rmtree(tmp, ignore_errors=True)
        pytest.skip(f"cannot build raven base (transformers skew?): {type(e).__name__}: {e}")
    return model


class StubTok:
    """Whitespace tokenizer mapping words → deterministic ids in [1, VOCAB)."""
    eos_token_id = 0
    pad_token_id = 0
    eos_token = "<eos>"
    pad_token = "<pad>"

    class _Out:
        def __init__(self, ids):
            self.input_ids = ids

    def __call__(self, text, add_special_tokens=False, return_tensors=None):
        ids = [(abs(hash(w)) % (VOCAB - 1)) + 1 for w in text.split()] or [1]
        if return_tensors == "pt":
            return StubTok._Out(torch.tensor([ids], dtype=torch.long))
        return StubTok._Out(ids)

    def decode(self, ids, skip_special_tokens=False):
        if isinstance(ids, int):
            ids = [ids]
        return " ".join(str(int(i)) for i in ids)


@pytest.fixture(scope="module")
def tok():
    return StubTok()


# ---------------------------------------------------------------------------
# has_cross_state / to_num_steps shims
# ---------------------------------------------------------------------------

def test_shims():
    from model_utils import has_cross_state, to_num_steps
    m_k4 = _build_raven(use_memory=True, memory_slots=4)
    m_k0 = _build_raven(use_memory=False)
    assert has_cross_state(m_k4) is True
    assert has_cross_state(m_k0) is False
    assert to_num_steps(None) is None
    ns = to_num_steps(5)
    assert ns.tolist() == [5, 0]


# ---------------------------------------------------------------------------
# BABILong eval_one — the headline K>0 vs K=0 path (chunked M_cross carry)
# ---------------------------------------------------------------------------

class TestBabilong:

    def _ctx(self):
        # long enough that encode_and_chunk (seq_len=8) yields multiple chunks
        return " ".join(f"fact{i} mary went to the office" for i in range(20))

    def test_eval_one_k4_carries_and_returns_bool(self, tok):
        import eval_babilong as eb
        model = _build_raven(use_memory=True, memory_slots=4)
        suffix = eb.build_suffix("qa1", "where is mary")
        prime, final = eb.split_context(tok, self._ctx(), suffix, seq_len=8,
                                        max_new_tokens=2)
        assert len(prime) > 1, "test needs multiple chunks to exercise M_cross carry"
        assert final.shape[1] <= 8 - 2, "final chunk must reserve generation room"
        ok, pred = eb.eval_one(model, tok, self._ctx(), "where is mary", "office",
                               T=3, seq_len=8, max_new_tokens=2, task="qa1")
        assert isinstance(ok, bool) and isinstance(pred, str)

    def test_eval_one_k0_runs(self, tok):
        import eval_babilong as eb
        model = _build_raven(use_memory=False)
        ok, pred = eb.eval_one(model, tok, self._ctx(), "where is mary", "office",
                               T=3, seq_len=8, max_new_tokens=2, task="qa1")
        assert isinstance(ok, bool) and isinstance(pred, str)

    def test_eval_one_directccot_runs(self, tok):
        import eval_babilong as eb
        model = _build_raven(use_memory=True, memory_slots=0, ccot_direct=True)
        ok, pred = eb.eval_one(model, tok, self._ctx(), "where is mary", "office",
                               T=3, seq_len=8, max_new_tokens=2, task="qa1")
        assert isinstance(ok, bool) and isinstance(pred, str)

    def test_contains_answer(self):
        import eval_babilong as eb
        assert eb.contains_answer("the kitchen.", "kitchen")
        assert eb.contains_answer("Kitchen\n", "kitchen")
        assert not eb.contains_answer("kitchenette", "kitchen")
        assert not eb.contains_answer("", "kitchen")


# ---------------------------------------------------------------------------
# GSM8K generate — greedy decode loop through the grafted model
# ---------------------------------------------------------------------------

class TestGSM8K:

    def test_generate_produces_text(self, tok):
        import eval_gsm8k as eg
        model = _build_raven(use_memory=True, memory_slots=4)
        device = torch.device("cpu")
        out = eg.generate(model, tok, "what is two plus two", max_new_tokens=4,
                          T=3, device=device, seq_len=32)
        assert isinstance(out, str)

    def test_generate_with_ccot_passes(self, tok):
        # Mixed CCoT+CoT: latent passes prime M_cross before generation.
        import eval_gsm8k as eg
        model = _build_raven(use_memory=True, memory_slots=4)
        device = torch.device("cpu")
        out = eg.generate(model, tok, "what is two plus two", max_new_tokens=4,
                          T=3, device=device, seq_len=32, ccot_passes=2)
        assert isinstance(out, str)

    def test_ccot_prime_returns_buffer_only_with_cross_state(self, tok):
        from model_utils import ccot_prime, to_num_steps
        ids = tok("some prompt words here", return_tensors="pt").input_ids
        m_k4 = _build_raven(use_memory=True, memory_slots=4)
        buf = ccot_prime(m_k4, ids, to_num_steps(3), passes=2)
        assert buf is not None and buf.shape[1] == 4
        m_k0 = _build_raven(use_memory=False)
        assert ccot_prime(m_k0, ids, to_num_steps(3), passes=2) is None
        assert ccot_prime(m_k4, ids, to_num_steps(3), passes=0) is None

    def test_extract_answer(self):
        import eval_gsm8k as eg
        assert eg.extract_answer("the answer is #### 42") == "42"


# ---------------------------------------------------------------------------
# Multiple choice — completion log-prob scoring
# ---------------------------------------------------------------------------

class TestMultipleChoice:

    def test_score_completion_returns_float(self, tok):
        import eval_multiple_choice as emc
        model = _build_raven(use_memory=True, memory_slots=4)
        ctx_ids = tok("the sky is").input_ids
        comp_ids = tok("blue today").input_ids
        score = emc.log_prob_of_completion(model, ctx_ids, comp_ids, T=3, seq_len=32,
                                           device=torch.device("cpu"))
        assert isinstance(score, float)
