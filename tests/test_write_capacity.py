"""
evals/eval_write_capacity.py against a real tiny grafted raven model.

The diagnostic's whole value is that a null is unambiguous, so the properties
worth pinning are the ones whose violation would silently produce a plausible
number:

  * `self` must score chunk g against the state AFTER g's write and `prev`
    against the state before it — off by one and the headline delta becomes the
    carry ablation with extra steps;
  * scoring must not append a second set of summary slots (prefix_write=False),
    or every condition gains n_vec columns the training forward never had;
  * chunk 1's `prev` and `none` are the same condition by construction, which is
    the built-in control (delta must be exactly 0), mirroring chunk 1 in the
    carry ablation;
  * `shuf` must be shape-compatible with `self` under accumulation, where the
    state grows with the chunk index.

Run: /c/Users/henry/miniconda3/envs/cortex-retro/python.exe -m pytest tests/ -q
"""
from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "evals"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from test_cortex_eval import VOCAB, _build_raven  # noqa: E402

import eval_write_capacity as ewc  # noqa: E402

NV, S, NC = 4, 32, 4
DEV = torch.device("cpu")


def _model(**kw):
    return _build_raven(use_memory=True, memory_slots=0, prefix_memory="accum",
                        accum_vecs=NV, accum_max=64, eos_token_id=VOCAB - 1, **kw)


def _sample():
    torch.manual_seed(0)
    ids = torch.randint(1, VOCAB - 1, (S * NC + 1,))
    return ids[:-1], ids[1:], torch.ones(S * NC)


def _run(model):
    x, y, ym = _sample()
    ns = torch.tensor([2, 0])
    states = ewc.states_for_sample(model, x, NC, ns, 7, DEV)
    shuf = ewc.states_for_sample(model, x.flip(0).contiguous(), NC, ns, 9, DEV)
    rows = ewc.chunk_losses(model, x, y, ym, NC, ns, 7, DEV, True, states, shuf)
    return states, rows


class TestStateChain:

    def test_states_grow_one_write_per_chunk(self):
        m = _model()
        states, _ = _run(m)
        assert len(states) == NC
        for g, st in enumerate(states):
            assert st.shape[1] == (g + 1) * NV, (
                f"chunk {g} state has {st.shape[1]} rows, expected {(g + 1) * NV}")

    def test_self_state_contains_this_chunks_write(self):
        """states[g] must be the POST-write state: one more chunk's worth of
        rows than states[g-1]."""
        m = _model()
        states, _ = _run(m)
        for g in range(1, NC):
            assert states[g].shape[1] - states[g - 1].shape[1] == NV


class TestConditions:

    def test_all_four_conditions_are_finite(self):
        m = _model()
        _, rows = _run(m)
        for cond in ewc.CONDS:
            assert len(rows[cond]) == NC
            for v in rows[cond]:
                assert v is not None and torch.isfinite(torch.tensor(v))

    def test_chunk1_prev_equals_none_exactly(self):
        """The built-in control: before chunk 1 there is no carry, so `prev` and
        `none` are the same forward.  Any drift means the conditions are not
        being seeded identically."""
        m = _model()
        _, rows = _run(m)
        assert rows["prev"][0] == rows["none"][0]

    def test_self_differs_from_prev_after_chunk1(self):
        """Chunk 1's `self` still carries its own write, so it differs from
        `none`; this is the pair the headline delta is built from."""
        m = _model()
        _, rows = _run(m)
        assert rows["self"][0] != rows["none"][0]
        for g in range(1, NC):
            assert rows["self"][g] != rows["prev"][g]

    def test_shuffled_carry_is_a_real_alternative(self):
        """`shuf` must actually change the forward — if it silently fell back to
        None the information control would read as a free win."""
        m = _model()
        _, rows = _run(m)
        for g in range(NC):
            assert rows["shuf"][g] != rows["none"][g]


class TestScoringForward:

    def test_scoring_does_not_append_summary_slots(self):
        """prefix_write=False: the scored forward must return exactly the real
        tokens' logits, with no extra summary columns and no new carry."""
        m = _model()
        x, _, _ = _sample()
        xc = x[:S]
        out = m(input_ids=xc.unsqueeze(0), num_steps=torch.tensor([2, 0]),
                m_cross_in=torch.randn(1, 2 * NV, m.config.n_embd),
                return_m_cross=True, prefix_write=False)
        assert out["logits"].shape == (1, S, VOCAB)
        assert out.get("m_cross") is None

    def test_scoring_is_deterministic_under_a_fixed_seed(self):
        """Conditions are compared per-sample, so the s0 draw must not leak
        between them."""
        m = _model()
        x, y, ym = _sample()
        ns = torch.tensor([2, 0])
        a = ewc.chunk_losses(m, x, y, ym, NC, ns, 7, DEV, True,
                             ewc.states_for_sample(m, x, NC, ns, 7, DEV), None)
        b = ewc.chunk_losses(m, x, y, ym, NC, ns, 7, DEV, True,
                             ewc.states_for_sample(m, x, NC, ns, 7, DEV), None)
        assert a["self"] == b["self"] and a["prev"] == b["prev"]
