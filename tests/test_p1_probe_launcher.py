"""
The P1 probe launcher, the sbatch it drives, and the pre-launch job.

These are shell files, so nothing type-checks them and nothing runs them off the
cluster.  What they CAN be checked for is the two failure classes this project
has already paid for in shell:

  * A VARIABLE THAT NEVER REACHED THE JOB.  A plain assignment inside a driver
    is a shell variable, not an environment one, so `sbatch` never sees it and
    the sbatch's own default wins silently.  That is how the whole B2 accum arm
    trained on the wrong corpus for six weeks.  Only the var-assignment PREFIX
    on the sbatch command works.
  * TWO FILES THAT DISAGREE ABOUT A NAME.  The driver hard-codes the run
    directories it will compare; the sbatch builds them from its own variables.
    If they drift, `--compare` silently finds nothing and reports on the arms it
    did find, which reads as "those arms are the experiment".

Run: /c/Users/henry/miniconda3/envs/cortex-retro/python.exe -m pytest tests/test_p1_probe_launcher.py -q
"""
from __future__ import annotations

import os
import re
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

PACE = os.path.join(REPO, "pace")


def _read(name):
    with open(os.path.join(PACE, name), encoding="utf-8") as fh:
        return fh.read()


ARMS = _read("p1_arms.sbatch")
DRIVER = _read("submit_p1_probes.sh")
PRELAUNCH = _read("prelaunch_final.sbatch")


class TestTheVariablesReachTheJob:

    def test_every_submit_uses_the_var_assignment_prefix(self):
        """`VAR=x sbatch ...` puts the value in THAT COMMAND's environment,
        which sbatch --export=ALL then forwards.  An export at the top of the
        driver, or a bare assignment, does not."""
        m = re.search(r"JOB=\$\((.*?)sbatch", DRIVER, re.S)
        assert m, "no sbatch submission found in the driver"
        prefix = m.group(1)
        for var in ("ARM=", "ARM_DATA=", "BRANCH_PATH=", "PROBE=",
                    "DIAG_INTERVAL="):
            assert var in prefix, f"{var} is not on the sbatch command line"

    def test_the_driver_does_not_export_arm_data_instead(self):
        """The failure mode, written down: an `export ARM_DATA=` at the top
        looks equivalent and is, in this repo's history, not."""
        assert not re.search(r"^export ARM_DATA=", DRIVER, re.M)

    def test_the_sbatch_still_has_its_own_default_for_every_forwarded_var(self):
        """Belt and braces: if a var does NOT arrive, the job must start from a
        named default rather than an empty string."""
        for var in ("ARM_DATA", "ACCUM_VECS", "CROSS_CHUNKS", "DIAG_INTERVAL"):
            assert re.search(rf"^{var}=\$\{{{var}:-", ARMS, re.M), var


class TestTheTwoFilesAgreeOnNames:

    def _run_name(self, arm: str) -> str:
        """Rebuild what p1_arms.sbatch's RUN_NAME would be, from its own text."""
        w, k, cc = "16", "64", "8"
        base = {
            "a1": f"p1-a1-accum-w{w}-cc{cc}",
            "a2": f"p1-a2-accum-w{w}-cc{cc}-z",
            "a3": f"p1-a3-gated-w{w}k{k}-cc{cc}",
            "a3z": f"p1-a3z-gated-w{w}k{k}-cc{cc}-z",
        }[arm]
        return "probe-" + base

    def test_the_sbatch_builds_the_names_the_driver_compares(self):
        for arm in ("a1", "a2", "a3", "a3z"):
            name = self._run_name(arm)
            assert name.replace("probe-", "") in DRIVER.replace(
                "${ACCUM_VECS}", "16").replace("${GATE_SLOTS}", "64").replace(
                "${CROSS_CHUNKS}", "8"), f"{arm}: {name} not in the driver"

    def test_the_sbatch_name_template_has_not_changed_shape(self):
        """The driver reproduces the template by hand, so the template itself is
        pinned: a change here has to be made in both files deliberately."""
        assert "RUN_NAME=${RUN_NAME:-p1-${ARM}-accum-w${ACCUM_VECS}-cc${CROSS_CHUNKS}}" in ARMS
        assert "RUN_NAME=${RUN_NAME:-p1-${ARM}-gated-w${ACCUM_VECS}k${GATE_SLOTS}-cc${CROSS_CHUNKS}}" in ARMS
        assert 'RUN_NAME="${RUN_NAME}-z"' in ARMS
        assert 'RUN_NAME="probe-${RUN_NAME}"' in ARMS

    def test_the_reference_arm_is_a1(self):
        """A1 carries B2's buffer at this round's recipe and is the ONLY valid
        stand-in for 'vs B2' -- b2-final differs in corpus, steps and W."""
        assert "--reference a1" in DRIVER


class TestTheProbeRunsTheDiagnostics:

    def test_probe_turns_the_periodic_diagnostic_on(self):
        """A probe without it produces no trajectory, and the trajectory is the
        entire reason for a few-hundred-step run."""
        assert "DIAG_INTERVAL=${DIAG_INTERVAL:-10}" in ARMS
        assert "--cortex.diag_interval $DIAG_INTERVAL" in ARMS

    def test_a_full_cell_leaves_it_off(self):
        """Every arm on record ran without it, and it costs an SVD per call."""
        assert "DIAG_INTERVAL=${DIAG_INTERVAL:-0}" in ARMS

    def test_the_probe_runs_the_walk_and_the_gates(self):
        for tool in ("tools/check_latent_probe.py",
                     "evals/diag_dual_channel_walk.py",
                     "tools/prelaunch_final.py"):
            assert tool in ARMS, tool

    def test_each_post_probe_step_can_fail_the_job(self):
        """A gate whose exit code is discarded is decoration."""
        body = ARMS[ARMS.index('if [ -n "$PROBE" ] && [ $RC -eq 0 ]; then'):]
        for tool in ("check_latent_probe.py", "diag_dual_channel_walk.py",
                     "prelaunch_final.py"):
            i = body.index(tool)
            assert "RC=1" in body[i:i + 400], f"{tool}'s exit code is ignored"


class TestThePreLaunchJob:

    def test_it_runs_the_suite_before_anything_expensive(self):
        # Compare the INVOCATIONS, not the header, which names every tool in
        # its own running order before any of them runs.
        i_tests = PRELAUNCH.index("python -m pytest tests/")
        for later in ("python evals/diag_dual_channel_walk.py",
                      "python tools/prelaunch_final.py"):
            assert PRELAUNCH.index(later) > i_tests

    def test_it_forces_the_z_geometry_on_an_e_only_parent(self):
        """The parent the arms branch from is E-only.  Without --set, these
        gates would test the thing already known to work."""
        assert "--set latent_carry=true" in PRELAUNCH
        assert "--set gate_slots=$GATE_SLOTS" in PRELAUNCH

    def test_it_prices_the_z_arms_memory(self):
        """Every memory number this gate ever produced came from an E-only run,
        and the second gate is ~33.5M more parameters with optimizer state."""
        assert "P1_Z=1" in PRELAUNCH
        assert "P1_Z" in _read("smoke_geometry_oom.sbatch")

    def test_a_failure_says_do_not_launch(self):
        assert "Do NOT launch" in PRELAUNCH


class TestTheCheckpointPathIsNotSilentlyIgnored:

    def test_a_missing_checkpoint_raises_instead_of_using_base_weights(self):
        """`--checkpoint <dir>` used to be silently ignored (isfile is False for
        a directory), so the eval ran on the BASE weights and the log looked
        healthy.  Both new tools pass a dir."""
        sys.path.insert(0, os.path.join(REPO, "evals"))
        from model_utils import load_checkpoint
        with pytest.raises(FileNotFoundError, match="does not exist"):
            load_checkpoint("no/such/checkpoint.pt", "ckpts/does-not-matter",
                            None, None, None)

    def test_a_directory_without_a_chkpt_is_named(self, tmp_path):
        sys.path.insert(0, os.path.join(REPO, "evals"))
        from model_utils import load_checkpoint
        with pytest.raises(FileNotFoundError, match="no chkpt.pt"):
            load_checkpoint(str(tmp_path), "ckpts/does-not-matter",
                            None, None, None)
