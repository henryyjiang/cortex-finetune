"""
The periodic training-time architecture diagnostic.

Every quantity that decides whether a cortex-final arm is worth finishing --
did either gate leave its exactly-zero init, is the carry collapsing towards
rank 4, is the Z channel attached to the loss at all -- moves over the first few
hundred steps and is settled thereafter.  Read off the final checkpoint it
answers too late and shows no trajectory; read off the loss curve it is not
answered at all.

So `training_diag` produces one flat row per interval and train.py writes it to
wandb and to a jsonl.  These tests pin what the row contains, that it is
JSON-safe (a numpy float or a torch tensor in there would crash the write at
step N and take the run's diagnostic with it), and -- by parsing train.py, which
cannot be imported off-cluster -- that the wiring is actually present.

Run: /c/Users/henry/miniconda3/envs/cortex-retro/python.exe -m pytest tests/test_training_diag.py -q
"""
from __future__ import annotations

import ast
import json
import os
import sys

import pytest
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from test_cortex_eval import VOCAB, _build_raven  # noqa: E402

from cortex_memory.health import training_diag  # noqa: E402

NV, K, CL, EOS = 4, 16, 16, VOCAB - 1


def _model(latent=True, gated=True):
    torch.manual_seed(1234)
    common = dict(use_memory=True, memory_slots=0, accum_vecs=NV,
                  latent_carry=latent, eos_token_id=EOS)
    if gated:
        common.update(prefix_memory="gated", gate_slots=K, gate_route="ring",
                      gate_init="zero", gate_fill="grow")
    else:
        common.update(prefix_memory="accum", accum_max=K * 4)
    return _build_raven(**common).train()


def _run(m, n=6):
    torch.manual_seed(0)
    ids = torch.randint(0, VOCAB - 1, (2, CL * n))
    carry = None
    for xc in torch.chunk(ids, n, dim=1):
        out = m(xc.contiguous(), num_steps=torch.tensor([0, 4]),
                m_cross_in=carry, return_m_cross=True)
        carry = out["m_cross"]
    return carry


class TestTheRow:

    def test_it_reports_both_channels_and_both_gates(self):
        m = _model(latent=True)
        row = training_diag(m.cortex, _run(m))
        for key in ("rows", "e_row_norm", "e_eff_rank_pr", "z_row_norm",
                    "z_eff_rank_pr", "z_over_e_norm", "fg_at_bias",
                    "fg_z_at_bias", "gate_left_init", "gate_z_left_init",
                    "z_read_grad_frac", "z_write_grad_frac", "latent_carry"):
            assert key in row, key
        assert row["rows"] == K
        assert row["latent_carry"] == 1

    def test_an_e_only_arm_reports_no_z_keys(self):
        """A dead Z channel and an absent one must not produce the same row --
        a chart of z_row_norm pinned at 0 would read as the first."""
        m = _model(latent=False)
        row = training_diag(m.cortex, _run(m))
        assert row["latent_carry"] == 0
        assert not [k for k in row if k.startswith("z_row")]
        assert "fg_at_bias" in row and "fg_z_at_bias" not in row

    def test_an_accum_arm_reports_no_gate_keys(self):
        m = _model(latent=True, gated=False)
        row = training_diag(m.cortex, _run(m))
        assert "fg_at_bias" not in row
        assert row["rows"] > 0 and row["e_row_norm"] > 0

    def test_every_value_is_json_safe(self):
        """The row is written to a jsonl inside the training loop.  A torch
        tensor or a bool-typed numpy scalar in there raises at step N and takes
        the run's whole diagnostic with it, silently, until someone looks."""
        row = training_diag(_model().cortex, _run(_model()))
        json.loads(json.dumps(row))
        assert all(isinstance(v, (int, float)) for v in row.values()), row

    def test_booleans_are_plottable_integers(self):
        row = training_diag(_model().cortex, _run(_model()))
        assert row["gate_left_init"] in (0, 1)
        assert not isinstance(row["gate_left_init"], bool)

    def test_a_model_without_memory_gives_an_empty_row_rather_than_raising(self):
        assert training_diag(None) == {}

    def test_the_gate_leaving_init_is_visible_in_the_row(self):
        """The whole point of logging it every N steps: at gate_init='zero' the
        weights are EXACTLY zero, so the first non-zero row is the step the
        gradient path proved itself."""
        m = _model()
        row0 = training_diag(m.cortex, _run(m))
        assert row0["gate_left_init"] == 0
        with torch.no_grad():
            m.cortex.prefix.gate_proj_in.weight.add_(1e-3)
        row1 = training_diag(m.cortex, _run(m))
        assert row1["gate_left_init"] == 1


class TestTheMeasuredReadLiveTable:
    """p11_z_probe_prereg.md quotes these numbers.  A pre-registration whose
    figures have quietly drifted from the sampler they came from is worse than
    no pre-registration at all, so the table is pinned against the real sampler
    rather than trusted."""

    def test_the_operating_point_reproduces(self):
        from cortex_memory.health import MEASURED_READ_LIVE, read_live_fraction
        m = _model()
        got = read_live_fraction(m, 20000, 8, 8, seed=7)["read_live_frac"]
        assert got == pytest.approx(MEASURED_READ_LIVE[(8, 8)], abs=0.02)

    def test_deep_recurrence_reproduces(self):
        from cortex_memory.health import MEASURED_READ_LIVE, read_live_fraction
        m = _model()
        got = read_live_fraction(m, 20000, 32, 8, seed=7)["read_live_frac"]
        assert got == pytest.approx(MEASURED_READ_LIVE[(32, 8)], abs=0.01)

    def test_raising_the_backprop_depth_past_the_recurrence_buys_almost_nothing(self):
        """The finding the table adds: ~0.55 is a CEILING.  With
        mean_backprop_depth >= mean_recurrence the Poisson rate is centred on
        the depth and P(Poisson(s) <= s-1) ~ 0.5 for any s, so the extra graph
        memory buys ~3 points and no more."""
        from cortex_memory.health import read_live_fraction
        m = _model()
        at8 = read_live_fraction(m, 20000, 8, 8, seed=3)["read_live_frac"]
        at32 = read_live_fraction(m, 20000, 32, 32, seed=3)["read_live_frac"]
        assert abs(at32 - at8) < 0.08
        assert 0.45 < at8 < 0.65 and 0.45 < at32 < 0.65


class TestTheWiringInTrainPy:
    """train.py cannot be imported off-cluster (device health check), so the
    wiring is checked against the SOURCE -- which is also what runs."""

    @staticmethod
    def _src():
        return open(os.path.join(REPO, "train.py"), encoding="utf-8").read()

    def test_train_py_parses(self):
        ast.parse(self._src())

    def test_diag_interval_defaults_to_off(self):
        """Off by default, so every arm on record keeps its exact behaviour:
        the diagnostic costs an SVD per call and adds keys to the wandb run."""
        src = self._src()
        assert "diag_interval=0," in src

    def test_the_diagnostic_is_imported_and_called(self):
        src = self._src()
        assert "from cortex_memory.health import training_diag" in src
        assert "training_diag(" in src

    def test_it_writes_a_jsonl_beside_the_checkpoints(self):
        """wandb is for watching a trajectory; the jsonl is what
        tools/compare_arms.py reads back, and the only copy that survives the
        run's project being renamed or moved."""
        src = self._src()
        assert "cortex_diag.jsonl" in src

    def test_the_carry_is_detached_before_it_is_held(self):
        """Holding the un-detached carry past the backward would keep the whole
        chunk chain's graph alive -- an OOM that only appears when the
        diagnostic is switched on."""
        src = self._src()
        i = src.index('cortex_diag_state["carry"] =')
        assert ".detach()" in src[i:i + 120]

    def test_the_diagnostic_does_not_ride_to_the_model_config(self):
        """It steers nothing the graft builds, so it must NOT be stamped onto
        config.json -- a training-only flag in a checkpoint's config is a key
        that later reads as architecture."""
        src = self._src()
        start = src.index('for _k in ("use_memory"')
        end = src.index("setattr(config, _k", start)
        assert "diag_interval" not in src[start:end]
