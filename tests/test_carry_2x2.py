"""The 2x2 carry ablation: E on/off x Z on/off at matched column count.

WHY THIS FILE EXISTS, AND WHY IT EXISTS NOW.  `evals/eval_carry_2x2.py` had no
launcher and no test file -- the exact state `eval_influence_horizon.py` was in
when RED 9 lived inside it: a damage path that was a silent no-op on every gated
arm, which would have reported "the gated carry does not matter anywhere" as a
finding.  This instrument is needed at the END of P2.3, i.e. after ~32 h per arm
has already been spent, so a hole found afterwards is expensive in a way a hole
found now is not.

Writing it found one, in the place RED 11 taught us to look: Z's null was
injected at `cfg.init_values["std"]` with a 0.02 fallback -- a weight-init
number -- where `_null_latent` wants s0's per-element rms.  The s0 sweep
measured that the substitution site has no gain until the injected norm reaches
~175.8 against ||E|| = 171.0, so a null at 0.02 sits deep in the flat region and
every Z cell equals its E twin.  The tool would have reported a confident zero.

Run: /c/Users/henry/miniconda3/envs/cortex-retro/python.exe -m pytest tests/test_carry_2x2.py -q
"""
from __future__ import annotations

import io
import os
import sys

import pytest
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "evals"))

from test_cortex_eval import VOCAB, _build_raven  # noqa: E402

from eval_carry_2x2 import (  # noqa: E402
    CELLS, chain_nll, has_latent_channel, null_e, null_z,
)

NV, K, CL, EOS = 4, 16, 16, VOCAB - 1
NC, T, D = 4, 4, None


def _model(latent=True, gated=False, **kw):
    torch.manual_seed(1234)
    common = dict(use_memory=True, memory_slots=0, accum_vecs=NV,
                  latent_carry=latent, eos_token_id=EOS)
    if gated:
        common.update(prefix_memory="gated", gate_slots=K, gate_route="ring",
                      gate_init="zero", gate_fill="grow")
    else:
        common.update(prefix_memory="accum", accum_max=K * 4)
    common.update(kw)
    return _build_raven(**common).eval()


def _chunks(n=NC, seed=0):
    torch.manual_seed(seed)
    ids = torch.randint(0, VOCAB - 1, (n * CL + 1,))
    x, y = ids[:-1], ids[1:]
    m = torch.ones_like(y, dtype=torch.float32)
    return (list(torch.chunk(x, n)), list(torch.chunk(y, n)),
            list(torch.chunk(m, n)))


class TestTheNullsTouchOnlyTheirOwnChannel:
    """The 2x2 collapses to a 1x2 the moment one null reaches both channels,
    and it collapses SILENTLY -- the cell still produces a number, still gets a
    label, still bootstraps a CI."""

    def test_nulling_E_leaves_Z_byte_identical(self):
        st = torch.randn(1, 8, 2 * 32)
        out = null_e(st, 32)
        assert torch.equal(out[..., 32:], st[..., 32:]), (
            "E's null reached the Z half: E0Z1 is really E0Z0 wearing the "
            "wrong label")
        assert torch.count_nonzero(out[..., :32]) == 0

    def test_nulling_E_on_an_E_only_carry_zeroes_all_of_it(self):
        st = torch.randn(1, 8, 32)
        assert torch.count_nonzero(null_e(st, 32)) == 0

    def test_a_width_that_is_neither_D_nor_2D_raises(self):
        """The wrong hidden_size produces a finite, plausible, WRONG number
        rather than a crash, which is why the tool takes it from the buffer
        instead of inferring it from the tensor."""
        with pytest.raises(ValueError):
            null_e(torch.randn(1, 8, 48), 32)

    def test_every_null_preserves_the_column_count(self):
        """Matched columns IS the design.  If a null dropped columns instead of
        emptying them, the packed sequence would shorten, initialize_state would
        draw a different s0, positions would shift, and 'no information' would
        be confounded with 'shorter sequence'."""
        st = torch.randn(1, 8, 2 * 32)
        assert null_e(st, 32).shape == st.shape
        assert null_z(st, 0.5, 7).shape == st.shape

    def test_Zs_null_is_noise_at_the_scale_it_is_given(self):
        st = torch.zeros(1, 64, 128)
        got = null_z(st, 3.0, 11)
        assert abs(float(got.std()) - 3.0) < 0.2
        assert torch.equal(null_z(st, 3.0, 11), got), "the null must be paired"


class TestTheZAxisActuallyLands:
    """RED 9's property, restated for this instrument: a null the model still
    reads must move the loss.  Without this the Z axis can return 0.000 from a
    knob that reaches nothing, and 0.000 reads as a result."""

    def test_turning_Z_off_changes_the_loss(self):
        m = _model(latent=True)
        xs, ys, ms = _chunks()
        num_steps = torch.tensor([0, T])
        on, _ = chain_nll(m, m.cortex, xs, ys, ms, num_steps,
                          torch.device("cpu"), 0, True, True, 5.0, m.config.n_embd)
        off, _ = chain_nll(m, m.cortex, xs, ys, ms, num_steps,
                           torch.device("cpu"), 0, True, False, 5.0,
                           m.config.n_embd)
        assert on is not None and off is not None
        assert abs(on - off) > 1e-6, (
            "z_on and z_off produced the same loss: latent_read_null is not "
            "reaching the model and every Z row is a statistic of nothing")

    def test_the_hook_is_cleared_afterwards(self):
        """A leaked null silently contaminates whatever runs next in the same
        process -- including the next cell of this very 2x2."""
        m = _model(latent=True)
        xs, ys, ms = _chunks()
        chain_nll(m, m.cortex, xs, ys, ms, torch.tensor([0, T]),
                  torch.device("cpu"), 0, True, False, 5.0, m.config.n_embd)
        assert getattr(m.cortex, "latent_read_null", None) is None

    def test_chunk_one_is_identical_across_cells(self):
        """Chunk 1 has no incoming carry, so every cell must agree there to
        floating-point noise.  This is the instrument's own control, and the
        launcher tells you to read it FIRST: a spread that is not ~0 means the
        cells differ in something other than the carry's contents."""
        m = _model(latent=True)
        xs, ys, ms = _chunks()
        firsts = []
        for _, e_on, z_on in CELLS:
            _, first = chain_nll(m, m.cortex, xs, ys, ms, torch.tensor([0, T]),
                                 torch.device("cpu"), 0, e_on, z_on, 5.0,
                                 m.config.n_embd)
            firsts.append(first)
        assert max(firsts) - min(firsts) < 1e-4


class TestItRefusesRatherThanGuessing:

    def test_an_E_only_model_is_not_a_2x2(self):
        assert not has_latent_channel(_model(latent=False).cortex)
        assert has_latent_channel(_model(latent=True).cortex)

    def test_the_gated_E_only_arm_is_also_refused(self):
        """a3 is gated AND E-only.  Inferring 'has Z' from 'is gated' is how
        reds 6 and 7 happened."""
        assert not has_latent_channel(_model(latent=False, gated=True).cortex)


class TestTheZNullIsAtS0sMeasuredScale:
    """RED 11's shape, one file over.  `_null_latent` takes a PER-ELEMENT std;
    `_z_s0_scale` is a per-token ROW NORM and the two are sqrt(D) = 45.25 apart
    at D=2048.  The tool used to pass a CONFIG value with a 0.02 fallback, which
    is neither."""

    def _src(self):
        return io.open(os.path.join(REPO, "evals", "eval_carry_2x2.py"),
                       encoding="utf-8").read()

    def test_the_config_guess_is_gone(self):
        """The CALL, not the prose: the comment explaining the removal names
        `init_values` too, and a test that matches a comment tests nothing."""
        s = self._src()
        assert 'getattr(cfg, "init_values"' not in s, (
            "the weight-init fallback is back")
        assert "s0_std = float(rms)" in s

    def test_it_reads_the_measured_rms_and_refuses_without_it(self):
        s = self._src()
        assert "_z_s0_rms" in s
        assert "_z_s0_scale" not in s.split("s0_std = float(rms)")[1][:200], (
            "the row norm must not be what gets passed")
        assert "exactly RED 11" in s, (
            "the refusal must say why, or the next person restores the fallback")

    def test_the_scale_it_uses_is_recorded_in_the_json(self):
        """A run whose Z axis is null is only interpretable next to the scale
        the null was injected at."""
        assert '"s0_std": s0_std' in self._src()


class TestTheChainDoesNotBuildAGraph:

    def test_the_forward_runs_under_no_grad(self):
        """`state` carries the graph forward, so without no_grad the chain holds
        every chunk's graph at once.  That is what OOMed the W=32 walk at 139.78
        GiB, and here it would die 32 h into P2.3's payoff rather than at the
        start."""
        s = io.open(os.path.join(REPO, "evals", "eval_carry_2x2.py"),
                    encoding="utf-8").read()
        body = s.split("def chain_nll")[1].split("def paired_ci")[0]
        assert "with torch.no_grad():" in body

    def test_the_returned_losses_are_plain_floats(self):
        m = _model(latent=True)
        xs, ys, ms = _chunks()
        nll, first = chain_nll(m, m.cortex, xs, ys, ms, torch.tensor([0, T]),
                               torch.device("cpu"), 0, True, True, 5.0,
                               m.config.n_embd)
        assert isinstance(nll, float) and isinstance(first, float)


class TestTheLauncher:

    def _sbatch(self):
        return io.open(os.path.join(REPO, "pace", "eval_carry_2x2.sbatch"),
                       encoding="utf-8").read()

    def test_the_E_only_arms_pass_allow_missing_z_and_the_Z_arms_do_not(self):
        s = self._sbatch()
        assert 'ALLOW=""' in s and 'ALLOW="--allow_missing_z"' in s
        body = s.split("cd $SLURM_SUBMIT_DIR")[1]
        assert "LATENT" in body

    def test_it_forces_the_geometry(self):
        """--model_name loads the base dir, whose config carries no cortex
        flags at all; use_memory is the master switch and nothing else is."""
        s = self._sbatch()
        assert "--set use_memory=true" in s
        assert "--set prefix_memory=$PREFIX_MODE" in s

    def test_it_passes_T_explicitly(self):
        """config.mean_recurrence is 32 on every B2-family checkpoint and no
        arm ran there.  That was RED 8."""
        assert '--T "$T"' in self._sbatch()
