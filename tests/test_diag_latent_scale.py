"""
P0.1's trajectory capture, on a tiny real raven model.

The probe hooks `core_block_forward` and `initialize_state` by instance-attribute
shadowing, which is exactly the kind of thing that breaks silently when the
modeling file moves a method or renames a return.  A broken hook does not raise
-- it captures nothing, or captures something adjacent to the trajectory, and
prints a plausible table of wrong numbers.  These tests make that fail here
instead.

Run: /c/Users/henry/miniconda3/envs/cortex-retro/python.exe -m pytest tests/ -q
"""
from __future__ import annotations

import io
import os
import sys

import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from test_cortex_eval import VOCAB, _build_raven  # noqa: E402

from evals.diag_latent_scale import _stats, record_trajectory  # noqa: E402

B, S, T = 2, 16, 5


def _model(**kw):
    torch.manual_seed(1234)
    return _build_raven(**kw).eval()


def _ids(b=B, s=S):
    torch.manual_seed(0)
    return torch.randint(0, VOCAB - 1, (b, s))


class TestTrajectoryCapture:

    def test_captures_exactly_T_steps_plus_s0(self):
        m = _model()
        s0, traj, out = record_trajectory(m, _ids(), T)
        assert s0 is not None, "s0 was never captured"
        assert len(traj) == T, f"captured {len(traj)} steps, asked for {T}"

    def test_latent_states_equals_the_last_hooked_step(self):
        """out.latent_states is x.clone().detach() taken BEFORE the coda, so it
        must equal the final hooked state exactly.  This is the probe's own
        self-check; if the hook ever drifts onto a different tensor (the coda
        output, the post-ln_f state) this is what catches it."""
        m = _model()
        _, traj, out = record_trajectory(m, _ids(), T)
        assert out.latent_states is not None
        assert torch.allclose(out.latent_states.float(), traj[-1].float(),
                              atol=1e-4, rtol=1e-4)

    def test_s0_is_the_init_scale_not_a_loop_state(self):
        """s0 must be the trunc_normal_ draw, orders of magnitude below the
        converged state.  Capturing a loop state as 's0' would silently divide
        every ratio in the report by the wrong number."""
        m = _model()
        s0, traj, _ = record_trajectory(m, _ids(), T)
        assert _stats(s0)["mean"] < _stats(traj[-1])["mean"]

    def test_hooks_are_restored(self):
        """The wrappers are instance attributes on a live model.  Leaking them
        would make every later forward in the same process silently accumulate
        into a stale list."""
        m = _model()
        before_core = type(m).core_block_forward
        before_init = type(m).initialize_state
        record_trajectory(m, _ids(), T)
        assert "core_block_forward" not in vars(m)
        assert "initialize_state" not in vars(m)
        assert type(m).core_block_forward is before_core
        assert type(m).initialize_state is before_init

    def test_hooks_are_restored_even_when_the_forward_raises(self):
        m = _model()
        try:
            record_trajectory(m, torch.full((B, S), VOCAB + 50), T)
        except Exception:
            pass
        assert "core_block_forward" not in vars(m)
        assert "initialize_state" not in vars(m)

    def test_a_moved_entry_point_raises_instead_of_reporting_zeros(self):
        """The failure this script must never have: hook does not fire, report
        prints a table of nothing.  Deleting the method has to raise."""
        m = _model()
        saved = type(m).core_block_forward
        try:
            del type(m).core_block_forward
            try:
                record_trajectory(m, _ids(), T)
            except RuntimeError as e:
                assert "core_block_forward" in str(e)
            else:
                raise AssertionError(
                    "a moved loop entry point must raise, not return empty")
        finally:
            type(m).core_block_forward = saved


class TestQuantizationFloor:
    """The trap that already produced a wrong reading once.

    Late deltas are ~1% of ||s_t||.  bf16 carries ~8 mantissa bits, so
    subtracting two converged bf16 states is substantially rounding, and the
    first bf16 run of this probe reported a delta PLATEAU at 0.37x ||s0|| that
    fp32 put at 0.06x, with cos(d,d-1) -0.62 against fp32's -0.95.  The plateau
    was the quantization floor wearing the shape of a result.
    """

    def test_float32_is_the_default(self):
        """A probe whose default dtype silently fabricates its own plateau is
        worse than no probe."""
        import evals.diag_latent_scale as m
        src = io.open(m.__file__, encoding="utf-8").read()
        assert 'default="float32"' in src
        assert 'default="bfloat16"' not in src

    def test_eps_ordering_is_what_the_check_relies_on(self):
        """bf16's relative spacing must be far coarser than fp32's, or the
        floor check is measuring nothing."""
        assert torch.finfo(torch.bfloat16).eps > 100 * torch.finfo(torch.float32).eps


class TestStats:

    def test_token_norms_are_per_token_not_per_tensor(self):
        """_stats must reduce over the hidden dim only.  A whole-tensor norm
        would scale with batch and sequence length, and every ratio in the
        report would depend on --batch."""
        x = torch.ones(3, 7, 16)
        assert abs(_stats(x)["mean"] - 4.0) < 1e-5     # sqrt(16), not sqrt(336)

    def test_none_is_passed_through(self):
        assert _stats(None) is None
