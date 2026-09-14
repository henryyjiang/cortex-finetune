"""Recipe fixes from the 2026-09-14 audit (`../recipe_sweep_findings.md`).

Every property here is one that a loss curve cannot show — which is the whole
reason the underlying bugs survived three rounds of training:

  * a resume that fast-forwards into a pack it should have read from row 0 looks
    exactly like a resume that did the right thing (finding 1);
  * a run that exhausts its loader early reports "finished" (finding 1);
  * a chunk-mean row loss and a token-mean row loss differ by ~1% of rows today
    and ~12% after the PG-19 packer fix, which is inside curve noise (finding 8);
  * a warmup that collapsed from 764 steps to 96 when the batch was raised is a
    fraction behaving exactly as written (finding 7).

Run: /c/Users/henry/miniconda3/envs/cortex-retro/python.exe -m pytest tests/ -q
"""
from __future__ import annotations

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from recipe_utils import (
    fast_forward_indices,
    reduce_chunk_losses,
    resolve_warmup_steps,
)


class TestWarmupSteps:
    """Finding 7 — warmup must be settable in steps, not only as a fraction."""

    def test_fraction_is_the_default(self):
        # B2's horizon and McLeish's, with the inherited 0.0025.  763 is
        # corroborated by B2's own logs: the first logged Muon LR is 2.3093e-06,
        # and the WSD lambda at step 1 is (1/763)*(1-0.001)+0.001 = 2.3094e-03.
        assert resolve_warmup_steps({"warmup": 0.0025, "warmup_steps": 0}, 305176) == 763
        assert resolve_warmup_steps({"warmup": 0.0025, "warmup_steps": 0}, 50000) == 125

    def test_the_collapse_this_exists_to_fix(self):
        # Raising the batch 8x shortens warmup 8x, to below the reference's 125.
        assert resolve_warmup_steps({"warmup": 0.0025, "warmup_steps": 0}, 38147) == 96

    def test_absolute_overrides_the_fraction(self):
        assert resolve_warmup_steps({"warmup": 0.0025, "warmup_steps": 600}, 38147) == 600

    def test_absolute_is_clamped_to_the_horizon(self):
        # A warmup longer than the run would leave the LR never reaching peak.
        assert resolve_warmup_steps({"warmup": 0.5, "warmup_steps": 10_000}, 100) == 100

    def test_missing_key_falls_back(self):
        # Configs written before 2026-09-14 have no warmup_steps key at all.
        assert resolve_warmup_steps({"warmup": 0.1}, 1000) == 100


class TestFastForward:
    """Finding 1 — the resume cursor, and what it does to a switched corpus."""

    def test_no_skip_on_a_fresh_run(self):
        assert fast_forward_indices(list(range(100)), 1, 1) is None

    def test_tail_is_exact_at_micro_batch_one(self):
        order = list(range(1000))
        tail = fast_forward_indices(order, 400, 1)
        assert tail == order[400:]
        assert len(tail) == 600

    def test_cursor_counts_micro_batches_not_rows(self):
        # data_start_step is a dataloader-ITEM count; at micro_batch_size 4 each
        # item is 4 rows, so 100 items means 400 rows consumed.
        order = list(range(1000))
        assert fast_forward_indices(order, 100, 4) == order[400:]

    def test_b2_arm_cursor_reproduces_the_logged_value(self):
        # logs/Report-12190106.out: resumed at optimizer step 199,485 having
        # branched at 91,552 with batch_size 4 / micro_batch_size 1, and the
        # first served item is `Step: 431736`.
        cursor = (199_485 - 91_552) * 4
        assert cursor == 431_732
        order = list(range(940_000))            # the mix pack, ~940k rows
        tail = fast_forward_indices(order, cursor, 1)
        assert tail[0] == 431_732               # first row actually served
        assert len(tail) == 508_268
        # ...and it covered the 105,691 remaining optimizer steps, but only just.
        assert len(tail) >= (305_176 - 199_485) * 4

    def test_control_configuration_would_have_run_dry(self):
        # The framework's locked control: heal to 11,444 steps at batch_size 16,
        # then switch corpus on a RESUME into a ~470k-row mix pack.
        cursor = 11_444 * 16
        assert cursor == 183_104
        with pytest.raises(ValueError, match="past the end"):
            # It does not raise at 470k rows — it runs dry mid-flight instead,
            # which is the point: the cursor fits, the BUDGET does not.
            fast_forward_indices(list(range(cursor - 1)), 11_444, 16)
        tail = fast_forward_indices(list(range(470_000)), 11_444, 16)
        served_steps = len(tail) // 16
        assert served_steps == 17_931                       # of 26,703 needed
        assert served_steps < 26_703

    def test_cursor_past_the_end_raises(self):
        with pytest.raises(ValueError, match="past the end"):
            fast_forward_indices(list(range(100)), 200, 1)

    def test_exactly_exhausted_raises(self):
        # Consuming the final row leaves nothing to serve; that is a failure, not
        # an empty-but-valid resume.
        with pytest.raises(ValueError):
            fast_forward_indices(list(range(100)), 100, 1)

    def test_slice_preserves_order_not_just_membership(self):
        # The tail must be the SAME permutation continued, or a resume reshuffles
        # and can repeat rows it already trained on.
        g = torch.Generator().manual_seed(74)
        order = torch.randperm(5000, generator=g).tolist()
        tail = fast_forward_indices(order, 1234, 1)
        assert tail == order[1234:]
        assert set(order[:1234]).isdisjoint(tail)

    def test_permutation_is_reproducible_from_the_seed(self):
        # What makes the slice correct across links: same seed, same length,
        # same order — every time, in a fresh process.
        def perm():
            g = torch.Generator().manual_seed(74)
            return torch.randperm(5000, generator=g).tolist()
        assert perm() == perm()


class TestChunkLossReduction:
    """Finding 8 — mean-of-means vs the row's token mean."""

    def test_identical_when_chunks_are_equal_length(self):
        losses = [torch.tensor(2.0), torch.tensor(3.0), torch.tensor(4.0)]
        tokens = [512, 512, 512]
        tok = reduce_chunk_losses(losses, tokens, mode="token")
        chunk = reduce_chunk_losses(losses, tokens, mode="chunk")
        assert torch.allclose(tok, chunk)
        assert torch.allclose(tok, torch.tensor(3.0))

    def test_they_diverge_on_a_ragged_tail(self):
        # A PG-19 row whose last window is a quarter full — the case the striding
        # packer fix will produce on ~1 row in 8.
        losses = [torch.tensor(2.0), torch.tensor(2.0), torch.tensor(2.0), torch.tensor(6.0)]
        tokens = [512, 512, 512, 128]
        chunk = reduce_chunk_losses(losses, tokens, mode="chunk")
        tok = reduce_chunk_losses(losses, tokens, mode="token")
        assert torch.allclose(chunk, torch.tensor(3.0))          # short chunk gets 1/4
        expected = (2 * 512 * 3 + 6 * 128) / (512 * 3 + 128)
        assert torch.allclose(tok, torch.tensor(expected))
        assert tok < chunk                                        # chunk-mean over-weights it

    def test_token_mode_equals_a_flat_token_mean(self):
        # The property that makes "token" the right default: the row loss is what
        # the model's own loss would have returned over the unsplit row.
        torch.manual_seed(0)
        per_token = torch.rand(1000)
        splits = [(0, 400), (400, 700), (700, 1000)]
        losses = [per_token[a:b].mean() for a, b in splits]
        tokens = [b - a for a, b in splits]
        assert torch.allclose(
            reduce_chunk_losses(losses, tokens, mode="token"), per_token.mean(), atol=1e-6)

    def test_gradients_flow_through_both_modes(self):
        for mode in ("token", "chunk"):
            x = torch.tensor([2.0, 4.0], requires_grad=True)
            reduce_chunk_losses([x[0], x[1]], [300, 100], mode=mode).backward()
            assert x.grad is not None and torch.all(x.grad > 0)

    def test_weights_are_off_the_graph(self):
        # Chunk token counts are data, not parameters: they must not create a
        # gradient path of their own.
        x = torch.tensor([2.0, 4.0], requires_grad=True)
        out = reduce_chunk_losses([x[0], x[1]], [300, 100], mode="token")
        (g,) = torch.autograd.grad(out, x)
        assert torch.allclose(g, torch.tensor([0.75, 0.25]))

    def test_bad_mode_raises(self):
        with pytest.raises(ValueError, match="chunk_loss_reduction"):
            reduce_chunk_losses([torch.tensor(1.0)], [10], mode="mean")

    def test_mismatched_counts_raise(self):
        with pytest.raises(ValueError, match="chunk_tokens"):
            reduce_chunk_losses([torch.tensor(1.0), torch.tensor(2.0)], [10], mode="token")
