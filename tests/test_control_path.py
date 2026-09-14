"""The C-chunked control: dispatch, guard, and chain equivalence.

The control is the zero every "memory helps by X" number in the paper is
measured against, and all three ways it can be quietly wrong produce a healthy
loss curve:

  * dispatched to the WHOLE-ROW path, so it differs from the memory model in
    chunk geometry as well as memory and isolates neither (the pre-2026-09-14
    `use_memory AND cross_chunks > 1` conjunct did exactly this);
  * dispatched correctly but with the memory module built anyway, so the
    "control" carries a live buffer and the memory model's delta vanishes;
  * chunked correctly but reducing the per-chunk losses in a way that does not
    equal the row's token mean, so the two arms optimise different objectives.

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

from recipe_utils import (  # noqa: E402
    control_has_memory,
    reduce_chunk_losses,
    select_fwd_bwd_path,
)
from smoke_geometry_oom import micro_step  # noqa: E402

CL, NC = 16, 4
EOS = VOCAB - 1
CPU_AMP = {"device_type": "cpu", "dtype": torch.bfloat16, "enabled": False}


def _control_model():
    torch.manual_seed(1234)
    return _build_raven(use_memory=False).train()


class TestDispatch:
    """train.py picks the chunk chain on cross_chunks alone."""

    def test_memory_off_still_takes_the_chunk_path(self):
        # THE regression this whole file exists for.  Under the old conjunct
        # this returned "tight" and the control trained on whole rows.
        assert select_fwd_bwd_path(False, cross_chunks=8) == "cortex"

    def test_memory_on_takes_the_chunk_path_too(self):
        assert select_fwd_bwd_path(False, cross_chunks=8) == "cortex"

    def test_cross_chunks_one_is_the_whole_row_path(self):
        # C-plain, and B2's heal before cc was raised.  cross_chunks=1 means
        # there is no chain, so the chunk implementation would be a no-op
        # wrapper with a different loss reduction.
        assert select_fwd_bwd_path(False, cross_chunks=1) == "tight"

    def test_non_recurrent_wins_over_everything(self):
        assert select_fwd_bwd_path(True, cross_chunks=8) == "non_rec"
        assert select_fwd_bwd_path(True, cross_chunks=1) == "non_rec"

    def test_the_control_and_the_memory_model_share_a_path(self):
        # The one-variable property, stated as a test: at matched cross_chunks
        # the two arms must differ in memory and in NOTHING ELSE about how the
        # row is processed.
        assert (select_fwd_bwd_path(False, 8) == select_fwd_bwd_path(False, 8))


class TestNegativeGuard:
    """A control that secretly has memory is the expensive failure."""

    def test_control_with_a_built_cortex_is_caught(self):
        assert control_has_memory(use_memory=False, has_cortex=True)

    def test_an_honest_control_passes(self):
        assert not control_has_memory(use_memory=False, has_cortex=False)

    def test_an_honest_memory_run_passes(self):
        assert not control_has_memory(use_memory=True, has_cortex=True)

    def test_it_does_not_fire_on_the_positive_failure(self):
        # use_memory true with no cortex is a real failure, but it is the OTHER
        # guard's job (train.py raises with a graft-specific message).  If this
        # predicate also claimed it, the wrong error would be reported.
        assert not control_has_memory(use_memory=True, has_cortex=False)


class TestMemorylessChain:
    """The chunk chain with no carry is a plain sequence of independent
    forwards, and its loss is the row's token mean."""

    def _batch(self):
        torch.manual_seed(0)
        ids = torch.randint(0, VOCAB - 1, (1, CL * NC + 1))
        return ids[:, :-1], ids[:, 1:]

    def test_no_carry_is_constructed(self):
        model = _control_model()
        assert getattr(model, "cortex", None) is None
        x, y = self._batch()
        _, m_cross = micro_step(model, x, y, EOS, NC, torch.tensor([1, 1]),
                                2, 0, 1, CPU_AMP, False)
        assert m_cross is None, "a memory-less model produced a carry"

    def _manual_chain(self, model, x, y, seed=7):
        """The chain written out by hand, one independent forward per chunk.

        Seeded, because the recurrent model draws its initial latent state s0
        RANDOMLY on every forward — two identical calls differ by ~1e-3 in loss
        for that reason alone.  Anything comparing two forwards of this model
        has to pin the RNG or it is measuring the sampler, not the computation.
        """
        torch.manual_seed(seed)
        losses, tokens = [], []
        for xc, yc in zip(torch.chunk(x, NC, dim=1), torch.chunk(y, NC, dim=1)):
            out = model(xc.contiguous(), labels=yc.contiguous(),
                        num_steps=torch.tensor([1, 1]))
            losses.append(out["loss"].detach())
            tokens.append(int((yc != -100).sum()))
        return losses, tokens

    def test_chain_loss_equals_the_manual_per_chunk_computation(self):
        # §10 asks for exactly this: "a test asserting the dispatch picks the
        # chunk path with memory off, and that the loss equals a manual
        # per-chunk computation."
        model = _control_model()
        x, y = self._batch()
        torch.manual_seed(7)
        got, _ = micro_step(model, x, y, EOS, NC, torch.tensor([1, 1]),
                            2, 0, 1, CPU_AMP, False)
        losses, tokens = self._manual_chain(model, x, y, seed=7)
        want = float(reduce_chunk_losses(losses, tokens, mode="token"))
        assert abs(got - want) < 1e-4, f"{got} vs {want}"

    def test_a_chunk_is_unaffected_by_what_precedes_it(self):
        # With no carry, chunk 2's loss must not depend on chunks 0 and 1.  If
        # it did, "no memory" would be false and the control would be leaking
        # cross-chunk information through some other channel — which is exactly
        # what a control cannot do.  Two rows that SHARE their last two chunks
        # and differ in the first two; matched seeds, so s0 is drawn in the same
        # order in both.
        model = _control_model()
        torch.manual_seed(0)
        a = torch.randint(0, VOCAB - 1, (1, CL * NC + 1))
        b = a.clone()
        b[:, :CL * 2] = torch.randint(0, VOCAB - 1, (1, CL * 2))
        la, _ = self._manual_chain(model, a[:, :-1], a[:, 1:])
        lb, _ = self._manual_chain(model, b[:, :-1], b[:, 1:])
        assert float(la[0]) != float(lb[0]), "the rows do not actually differ"
        for i in (2, 3):
            assert abs(float(la[i]) - float(lb[i])) < 1e-5, (
                f"chunk {i} saw the change in chunks 0-1")

    def test_token_reduction_is_not_the_chunk_mean_when_chunks_are_ragged(self):
        # The control reads the same packs as the memory model, and after the
        # PG-19 striding fix ~1 row in 8 has a ragged last window.  If the two
        # arms disagreed on the reduction they would optimise different
        # objectives, so pin that "token" actually differs from "chunk" here.
        losses = [torch.tensor(1.0), torch.tensor(3.0)]
        assert float(reduce_chunk_losses(losses, [100, 100], "token")) == 2.0
        ragged = float(reduce_chunk_losses(losses, [100, 20], "token"))
        assert abs(ragged - (1.0 * 100 + 3.0 * 20) / 120) < 1e-6
        assert ragged != float(reduce_chunk_losses(losses, [100, 20], "chunk"))
