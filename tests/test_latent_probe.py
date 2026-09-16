"""
tools/check_latent_probe.py — the post-run half of the Z architecture probe.

The tool's job is to catch a parameter block that sat inert while the run
trained, which is invisible in a loss curve.  So the tests that matter are the
ones where it must FAIL: a Z gate still at its zero init, a re-seeded
summary_emb, a config/parameter mismatch.  A checker that cannot fail is worse
than no checker, because it launders a broken run as a verified one.

Run: /c/Users/henry/miniconda3/envs/cortex-retro/python.exe -m pytest tests/ -q
"""
from __future__ import annotations

import os
import subprocess
import sys

import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOL = os.path.join(REPO, "tools", "check_latent_probe.py")

D, W = 16, 4


def _write(path, model, step, cortex_cfg=None):
    path.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model,
                "agg_vars_dict": {"optimizer_step": step},
                "cfg": {"cortex": cortex_cfg or {}}},
               path / "chkpt.pt")
    return path


def _parent_model(seed=0):
    torch.manual_seed(seed)
    return {
        "cortex.prefix.summary_emb": torch.randn(W, D),
        "cortex.prefix.gate_proj_in.weight": torch.zeros(2 * D, D),
        "cortex.prefix.gate_proj_mem.weight": torch.zeros(2 * D, D),
    }


def _probe_model(parent, *, z_moved=True, e_moved=True, reseed=False,
                 with_z=True, nonfinite=False):
    m = {k: v.clone() for k, v in parent.items()}
    if e_moved:
        m["cortex.prefix.summary_emb"] += 1e-3
        m["cortex.prefix.gate_proj_in.weight"] += 1e-3
        m["cortex.prefix.gate_proj_mem.weight"] += 1e-3
    if reseed:
        torch.manual_seed(99)
        m["cortex.prefix.summary_emb"] = torch.randn(W, D) * 5
    if with_z:
        z = 1e-3 if z_moved else 0.0
        m["cortex.prefix.gate_proj_in_z.weight"] = torch.full((2 * D, D), z)
        m["cortex.prefix.gate_proj_mem_z.weight"] = torch.full((2 * D, D), z)
        m["cortex.prefix.forget_bias_z"] = torch.tensor([1.0])
        m["cortex.prefix.input_bias_z"] = torch.tensor([0.0])
    if nonfinite:
        m["cortex.prefix.summary_emb"][0, 0] = float("nan")
    return m


def _run(tmp_path, probe_model, *, parent_model=None, cortex_cfg=None,
         step=200, parent_step=100):
    p = _write(tmp_path / "parent", parent_model or _parent_model(),
               parent_step, cortex_cfg)
    q = _write(tmp_path / "probe", probe_model, step,
               cortex_cfg if cortex_cfg is not None
               else {"latent_carry": True, "accum_vecs": W})
    return subprocess.run(
        [sys.executable, TOOL, "--probe", str(q), "--parent", str(p)],
        capture_output=True, text=True, cwd=REPO)


class TestThePassingCase:

    def test_a_live_architecture_passes(self, tmp_path):
        par = _parent_model()
        r = _run(tmp_path, _probe_model(par), parent_model=par)
        assert r.returncode == 0, r.stdout + r.stderr
        assert "PROBE PASSED" in r.stdout

    def test_it_never_claims_a_result_about_z(self, tmp_path):
        """A green probe in a log is exactly the thing that gets quoted later as
        'Z worked'.  The disclaimer has to be in the output, on the pass path."""
        par = _parent_model()
        r = _run(tmp_path, _probe_model(par), parent_model=par)
        assert "NOT A RESULT ABOUT Z" in r.stdout


class TestTheFailingCases:
    """Each one is a run that trains happily with part of itself inert."""

    def test_a_z_gate_still_at_zero_init_fails(self, tmp_path):
        """At gate_init='zero' the projections start EXACTLY at zero, so this is
        not 'a small gradient' -- it is no gradient path at all, and the run
        trained an un-gated latent carry."""
        par = _parent_model()
        r = _run(tmp_path, _probe_model(par, z_moved=False), parent_model=par)
        assert r.returncode == 1
        assert "left its zero init" in r.stdout
        assert "NO GRADIENT PATH" in r.stdout

    def test_a_reseeded_summary_emb_fails(self, tmp_path):
        """The width-mismatch failure: strict=False drops the key, the graft
        re-seeds all W rows from wte[eos], and 91,552 steps of write-path
        training are gone behind a healthy loss curve."""
        par = _parent_model()
        r = _run(tmp_path, _probe_model(par, reseed=True), parent_model=par)
        assert r.returncode == 1
        assert "re-seeded" in r.stdout

    def test_a_frozen_shared_write_path_fails(self, tmp_path):
        par = _parent_model()
        r = _run(tmp_path, _probe_model(par, e_moved=False), parent_model=par)
        assert r.returncode == 1
        assert "summary_emb moved" in r.stdout

    def test_non_finite_parameters_fail(self, tmp_path):
        par = _parent_model()
        r = _run(tmp_path, _probe_model(par, nonfinite=True), parent_model=par)
        assert r.returncode == 1
        assert "every parameter is finite" in r.stdout

    def test_a_config_that_asked_for_z_without_z_parameters_fails(self, tmp_path):
        """--cortex.latent_carry true that never reached the graft.  The config
        says Z, the checkpoint has no Z parameters, and nothing else notices."""
        par = _parent_model()
        r = _run(tmp_path, _probe_model(par, with_z=False), parent_model=par,
                 cortex_cfg={"latent_carry": True, "accum_vecs": W})
        assert r.returncode == 1
        assert "matches the parameters found" in r.stdout

    def test_a_probe_that_did_not_advance_fails(self, tmp_path):
        par = _parent_model()
        r = _run(tmp_path, _probe_model(par), parent_model=par,
                 step=100, parent_step=100)
        assert r.returncode == 1
        assert "advanced past its parent" in r.stdout

    def test_a_width_mismatch_is_reported_as_a_shape_failure(self, tmp_path):
        par = _parent_model()
        m = _probe_model(par)
        m["cortex.prefix.summary_emb"] = torch.randn(W * 2, D)
        r = _run(tmp_path, m, parent_model=par)
        assert r.returncode == 1
        assert "kept its shape" in r.stdout


class TestWithoutAParent:
    """--parent is optional; the zero-init check must still work without it,
    because that is the one that needs no baseline."""

    def test_the_zero_init_check_runs_with_no_parent(self, tmp_path):
        par = _parent_model()
        q = _write(tmp_path / "probe", _probe_model(par, z_moved=False), 200,
                   {"latent_carry": True, "accum_vecs": W})
        r = subprocess.run([sys.executable, TOOL, "--probe", str(q)],
                           capture_output=True, text=True, cwd=REPO)
        assert r.returncode == 1 and "left its zero init" in r.stdout

    def test_a_missing_checkpoint_is_a_clean_failure(self, tmp_path):
        r = subprocess.run(
            [sys.executable, TOOL, "--probe", str(tmp_path / "nope")],
            capture_output=True, text=True, cwd=REPO)
        assert r.returncode != 0
        assert "no chkpt.pt" in r.stdout + r.stderr
