"""
P0.5's ladder discovery, in dry-run mode against synthetic checkpoint trees.

The two things that can go wrong here are both silent:

  * LEXICAL sort.  model_only_chkpt_9500 outranking model_only_chkpt_150000 is
    a documented trap in this project; under it the plotted x-axis is scrambled
    and the curve is meaningless while looking fine.
  * An EMPTY discovery reported as a result.  "no rungs found" and "no rungs
    exist" are different statements, and only one of them is evidence.

Run: /c/Users/henry/miniconda3/envs/cortex-retro/python.exe -m pytest tests/ -q
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(REPO, "pace", "submit_p05_ladder.sh")
BASH = shutil.which("bash")

pytestmark = pytest.mark.skipif(
    BASH is None, reason="no bash on PATH (the script is a cluster shell script)")


def _tree(spec: dict[str, list[int]]) -> str:
    root = tempfile.mkdtemp(prefix="p05_")
    for run, steps in spec.items():
        for s in steps:
            os.makedirs(os.path.join(root, run, f"model_only_chkpt_{s}"))
    return root


def _run(root: str, **env) -> subprocess.CompletedProcess:
    """Dry-run the script against `root`.

    The script hard-codes the two B2 run names on purpose -- it is the B2
    ladder, not a generic checkpoint walker -- so a synthetic tree has to name
    its run through HEAL_RUN/ARM_RUN rather than hoping for a match.
    """
    e = dict(os.environ, OUT_ROOT=root, **{k: str(v) for k, v in env.items()})
    return subprocess.run([BASH, SCRIPT], cwd=REPO, env=e,
                          capture_output=True, text=True)


def _run_single(root: str, run: str = "r", **env) -> subprocess.CompletedProcess:
    """One synthetic run, mapped onto the heal slot."""
    return _run(root, HEAL_RUN=run, ONLY="heal", **env)


def _planned(out: str) -> list[int]:
    """The step numbers the script says it will submit, in printed order."""
    return [int(m.group(1))
            for m in re.finditer(r"step\s+(\d+)\s+=", out)]


class TestLadderDiscovery:

    def test_steps_are_ordered_numerically_not_lexically(self):
        """The trap case: under a lexical sort 150000 sorts before 9500."""
        root = _tree({"r": [500, 9500, 20000, 150000, 305000]})
        got = _planned(_run_single(root, STRIDE=1).stdout)
        assert got == sorted(got), f"not ascending: {got}"
        assert got == [500, 9500, 20000, 150000, 305000]

    def test_stride_subsamples_and_always_keeps_the_last_rung(self):
        """The last rung of each run is a landmark -- the heal's is the 1.5B
        boundary, the arm's is the closest point to b2-final -- so it must
        survive any stride, including one that does not divide evenly."""
        root = _tree({"r": [1000 * i for i in range(1, 12)]})   # 1000..11000
        got = _planned(_run_single(root, STRIDE=4).stdout)
        assert got[0] == 1000
        assert got[-1] == 11000, f"last rung dropped: {got}"
        assert len(got) < 11

    def test_both_runs_appear_in_chain_order(self):
        root = _tree({"retro-b2-heal": [2500, 90000],
                      "retro-b2-acc32-cc8-mr8": [92500, 305000]})
        got = _planned(_run(root, STRIDE=1).stdout)
        assert got == [2500, 90000, 92500, 305000], \
            "the heal half must plot before the arm half -- they are one chain"

    def test_only_arm_skips_the_heal_half(self):
        root = _tree({"retro-b2-heal": [2500, 90000],
                      "retro-b2-acc32-cc8-mr8": [92500, 305000]})
        got = _planned(_run(root, STRIDE=1, ONLY="arm").stdout)
        assert got == [92500, 305000]

    def test_token_axis_uses_b2s_tokens_per_step(self):
        """step x 16,384.  Plotting steps against a 1,048,576-tok/step
        reference would be off by 64x, which is the whole point of the axis."""
        root = _tree({"r": [305176]})
        out = _run_single(root, STRIDE=1).stdout
        assert "5.000B" in out, out

    def test_ignores_non_numeric_and_unrelated_dirs(self):
        root = _tree({"r": [1000, 2000]})
        os.makedirs(os.path.join(root, "r", "final_checkpoint"))
        os.makedirs(os.path.join(root, "r", "checkpoint_1500"))
        os.makedirs(os.path.join(root, "r", "model_only_chkpt_latest"))
        assert _planned(_run_single(root, STRIDE=1).stdout) == [1000, 2000]


class TestRefusals:

    def test_an_empty_tree_refuses_instead_of_submitting_nothing(self):
        root = tempfile.mkdtemp(prefix="p05_empty_")
        r = _run(root, STRIDE=1)
        assert r.returncode == 2
        assert "REFUSING TO SUBMIT" in r.stdout

    def test_dry_run_is_the_default(self):
        """GO=1 is required.  A script that submits ~15 GPU jobs on being run
        with no arguments is one tab-completion away from an accident."""
        root = _tree({"r": [1000]})
        r = _run_single(root, STRIDE=1)
        assert r.returncode == 0
        assert "DRY RUN" in r.stdout
        assert "Submitted" not in r.stdout

    def test_the_mandatory_overrides_are_defaulted_correctly(self):
        """T_OVERRIDE=8 and the RETROFIT prep base.  T unset evaluates an mr8
        arm at T=32 (finding 0c); the wrong prep base merges olmo8-cortex into
        a retrofit checkpoint, which is silent and wrong."""
        root = _tree({"r": [1000]})
        out = _run_single(root, STRIDE=1).stdout
        assert "T=8" in out
        assert "ckpts/olmo-retrofit-cortex" in out
