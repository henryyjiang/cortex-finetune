"""Job 13434102: the P3.0 oracle probe branched a Muon-trained checkpoint into
an AdamW read-only run.  The parent's LambdaLR state (one lr_lambda per Muon
param group) could not load into the probe's single-group scheduler
(IndexError), and behind that the inherited step counter (91,952) was already
past the probe's max_steps=1000, which would have trained ONE step and exited
clean.  These pins keep both closed."""
import os

import pytest
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _src():
    return open(os.path.join(REPO, "train.py"), encoding="utf-8").read()


def _load_checkpoint_body():
    src = _src()
    i = src.find("def load_checkpoint(state, cfg, device")
    assert i > 0, "load_checkpoint moved or was renamed"
    j = src.find("\ndef ", i + 1)
    return src[i:j]


class TestReadOnlyBranchStartsFresh:
    def test_fresh_is_keyed_on_branch_and_read_only(self):
        assert "fresh = branch and cfg.train_read_only" in _load_checkpoint_body()

    def test_scheduler_load_skipped_when_fresh(self):
        body = _load_checkpoint_body()
        assert "if not cfg.ignore_past_scheduler and not fresh:" in body

    def test_step_counter_reset_when_fresh(self):
        body = _load_checkpoint_body()
        i = body.find("if fresh:\n        agg.update(")
        assert i > 0, "a read-only branch no longer resets optimizer_step"
        assert "optimizer_step=0" in body[i:i + 200]

    def test_group_count_mismatch_named_not_indexerror(self):
        assert "LambdaLR cannot map one onto the other" in _load_checkpoint_body()

    def test_main_refuses_a_counter_past_max_steps(self):
        src = _src()
        i = src.find("agg_dict = load_checkpoint(state, cfg, device")
        assert i > 0
        assert "optimizer_step >= cfg.max_steps" in src[i:i + 1500]


def test_the_failure_mode_is_real():
    """Reproduce the original crash so the guard's premise stays checked:
    a 2-group LambdaLR state does not load into a 1-group scheduler."""
    from functools import partial

    def lam(step, k):
        return 1.0

    p1, p2 = torch.nn.Parameter(torch.zeros(1)), torch.nn.Parameter(torch.zeros(1))
    two = torch.optim.AdamW([{"params": [p1]}, {"params": [p2]}])
    one = torch.optim.AdamW([p1, p2])
    s2 = torch.optim.lr_scheduler.LambdaLR(two, partial(lam, k=1))
    s1 = torch.optim.lr_scheduler.LambdaLR(one, partial(lam, k=1))
    with pytest.raises(IndexError):
        s1.load_state_dict(s2.state_dict())
