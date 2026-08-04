"""
tools/smoke_prefix_real.py's chain logic, on a tiny real raven model.

The smoke itself needs the 1B checkpoint and runs on the cluster; this pins the
part that can be wrong regardless of scale, so a broken smoke is caught here
rather than on a login node at 11pm.  In particular `preserved` is the on-real-
weights version of the Fix A assertion — if it were computed against the
post-detach state instead of the pre-detach one it would read True no matter
what the model did.

Run: /c/Users/henry/miniconda3/envs/cortex-retro/python.exe -m pytest tests/ -q
"""
from __future__ import annotations

import os
import sys

import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "tools"))

from test_cortex_eval import VOCAB, _build_raven  # noqa: E402

from smoke_prefix_real import run_chain  # noqa: E402

NV, CL, NC = 4, 16, 4
EOS = VOCAB - 1
DEV = "cpu"


def _model(**kw):
    return _build_raven(use_memory=True, memory_slots=0, prefix_memory="accum",
                        accum_vecs=NV, accum_max=64, eos_token_id=EOS, **kw).train()


def _batch():
    torch.manual_seed(0)
    ids = torch.randint(0, VOCAB - 1, (1, CL * NC + 1))
    ids[0, CL + CL // 2] = EOS               # boundary inside chunk 2
    ids[0, 2 * CL + 3] = EOS                 # and inside chunk 3
    return ids[:, :-1], ids[:, 1:]


def _run(model, carry_grad_chunks=2):
    x, y = _batch()
    return run_chain(model, x, y, EOS, NC, torch.tensor([1, 1]),
                     carry_grad_chunks, NV, DEV)


class TestChain:

    def test_carry_accumulates_one_write_per_chunk(self):
        m = _model()
        rep = _run(m)
        assert rep["shapes"] == [(1, (g + 1) * NV, m.config.n_embd)
                                 for g in range(NC)]

    def test_carry_survives_document_boundaries(self):
        """The real-weights form of Fix A: earlier chunks' rows must come back
        verbatim from a chunk that contained an EOS."""
        m = _model()
        assert all(_run(m)["preserved"])

    def test_legacy_reset_is_detected_as_not_preserved(self):
        """The check must actually be able to FAIL — under the old policy the
        carry is zeroed, so `preserved` has to go False.  Without this, a
        `preserved` that was trivially True would pass the smoke forever."""
        m = _model(prefix_eos_reset=True)
        assert not all(_run(m)["preserved"])

    def test_losses_finite_and_backward_reaches_the_write(self):
        m = _model()
        rep = _run(m)
        assert all(l == l for l in rep["losses"])
        g = m.cortex.prefix.summary_emb.grad
        assert g is not None and torch.isfinite(g).all() and float(g.norm()) > 0

    def test_loop_params_receive_gradient(self):
        m = _model()
        _run(m)
        loop = [p for n, p in m.named_parameters()
                if "core_block" in n and p.grad is not None]
        assert loop and all(torch.isfinite(p.grad).all() for p in loop)

    def test_stop_gradient_horizon_is_applied(self):
        """carry_grad_chunks must shorten the graph — with a 1-chunk horizon the
        oldest rows are detached, so summary_emb's gradient is strictly smaller
        than under full-chain BPTT."""
        a = _model(); _run(a, carry_grad_chunks=1)
        b = _model(); _run(b, carry_grad_chunks=0)
        assert float(a.cortex.prefix.summary_emb.grad.norm()) \
            < float(b.cortex.prefix.summary_emb.grad.norm())
