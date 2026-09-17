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


class TestThePostProbeChecksRunAtTheArmsGeometry:
    """A probe checkpoint dir holds only chkpt.pt, so the config comes from
    $MODEL -- the BASE, which says W=32, accum, E-only.  Without the arm's own
    flags the post-run checks either die on a summary_emb size mismatch or, if
    the widths happen to agree, rebuild a DIFFERENT buffer from the same weights
    and report on a geometry that never trained.  Either way it happens AFTER
    the 400 steps are paid for."""

    def test_both_post_probe_consumers_get_the_arms_flags(self):
        body = ARMS[ARMS.index('if [ -n "$PROBE" ] && [ $RC -eq 0 ]; then'):]
        for tool in ("diag_dual_channel_walk.py", "prelaunch_final.py"):
            i = body.index(tool)
            line_end = body.index("\n", i)
            assert "$PROBE_SETS" in body[i:line_end], (
                f"{tool} runs against the BASE config, not the arm's")

    def test_probe_sets_is_built_from_the_same_variables_as_the_run(self):
        """Built next to the training command's own GATE_ARGS / LATENT_ARGS so
        the checks and the run cannot describe different geometries."""
        assert "--set prefix_memory=$PREFIX_MODE" in ARMS
        assert "--set accum_vecs=$ACCUM_VECS" in ARMS
        for flag in ("gate_slots=$GATE_SLOTS", "gate_route=", "gate_norm=",
                     "gate_init=", "gate_fill="):
            assert flag in ARMS, flag

    def test_an_accum_arm_does_not_claim_a_gate_it_does_not_have(self):
        i = ARMS.index('PROBE_SETS="--set use_memory=true')
        block = ARMS[i:i + 1600]
        assert 'if [ "$PREFIX_MODE" = "accum" ]; then' in block
        assert "--set accum_max=$ACCUM_MAX" in block

    def test_the_z_flags_are_gated_on_the_arm_actually_carrying_z(self):
        """a1/a3 are E-only: forcing latent_carry on their checks would gate a
        channel those weights never had."""
        i = ARMS.index('PROBE_SETS="--set use_memory=true')
        block = ARMS[i:i + 1800]
        j = block.index("--set latent_carry=true")
        assert 'if [ "$LATENT" = "1" ]; then' in block[:j]


class TestUseMemoryIsTheMasterSwitch:
    """REGRESSION from job 13266470, which failed all three of its post-load
    steps at once with only "no prefix buffer" to show for it.

    `cortex_graft.memory_enabled` reads `use_memory` AND NOTHING ELSE.  A
    graft-prepared BASE dir can carry no cortex flags at all (the log showed
    `config.accum_vecs: '<absent>'`), so overriding prefix_memory / accum_vecs /
    gate_slots without use_memory builds nothing: the checkpoint's cortex
    tensors arrive as UNEXPECTED keys, get dropped, and the run silently becomes
    a no-memory baseline.
    """

    def test_every_set_list_in_the_prelaunch_job_turns_memory_on(self):
        # Scan from each marker to the END of the invocation it introduces,
        # not a fixed window -- the comment blocks are long and a fixed window
        # would pass or fail on prose length rather than on the flag.
        # Steps 3 and 3b reach the switch through $SETS / $ACC_SETS, both of
        # which lead with it; the definitions themselves are checked directly.
        for marker, want in (('echo "######## 3. the 8-chunk', "$SETS"),
                             ('echo "######## 3b. THE WIDTH CONTRAST', "$ACC_SETS"),
                             ('SETS="--set use_memory=true', "--set use_memory=true")):
            i = PRELAUNCH.index(marker)
            j = PRELAUNCH.index("|| RC=1", i) if "########" in marker else i + 400
            assert want in PRELAUNCH[i:j], marker
        assert 'SETS="--set use_memory=true' in PRELAUNCH
        assert 'ACC_SETS="--set use_memory=true' in PRELAUNCH

    def test_the_probe_set_list_turns_memory_on(self):
        assert '--set use_memory=true' in ARMS
        i = ARMS.index('PROBE_SETS="--set use_memory=true')
        assert "--set prefix_memory=$PREFIX_MODE" in ARMS[i:i + 400]

    def test_the_master_switch_is_what_the_graft_actually_reads(self):
        """Pinned against the graft, so this test fails if the switch moves."""
        import os
        src = open(os.path.join(REPO, "cortex_graft.py"), encoding="utf-8").read()
        i = src.index("def memory_enabled(")
        body = src[i:i + 400]
        assert 'getattr(config, "use_memory", False)' in body

    def test_the_failure_diagnoses_itself(self):
        """Three steps reported the same symptom and none named the cause."""
        import sys, os
        sys.path.insert(0, os.path.join(REPO, "evals"))
        from model_utils import explain_missing_cortex

        class Cfg:
            pass
        msg = explain_missing_cortex(Cfg(), {"prefix_memory": "gated",
                                             "accum_vecs": 16})
        assert "use_memory is the master switch" in msg
        assert "--set use_memory=true" in msg
        on = Cfg()
        on.use_memory = True
        assert "master switch" not in explain_missing_cortex(on, {})


class TestAnEmptyGeometryListMeansNone:
    """`${VAR:-default}` falls back on an EMPTY value, so the OOM gate's own
    documented invocation -- `P1=1 GEOMETRIES= sbatch ...` -- ran all three
    config-D rows anyway (job 13266470).  Without the colon, empty means empty.
    """

    def test_geometries_uses_the_non_colon_form(self):
        sb = _read("smoke_geometry_oom.sbatch")
        assert 'GEOMETRIES=${GEOMETRIES-"' in sb
        assert 'GEOMETRIES=${GEOMETRIES:-"' not in sb


class TestThePostProbeChecksHaveAProseSource:
    """Without one the walk now REFUSES to run -- so all four probes would
    finish their 400 steps and then fail their own checks.  And the donor gate
    would measure nothing either way: a carry built from random ids has no
    content to transfer, so its content delta is ~0 by construction."""

    def test_both_post_probe_consumers_get_the_arms_corpus(self):
        body = ARMS[ARMS.index('if [ -n "$PROBE" ] && [ $RC -eq 0 ]; then'):]
        for tool in ("diag_dual_channel_walk.py", "prelaunch_final.py"):
            i = body.index(tool)
            assert "$PROBE_PROSE" in body[i:body.index("\n", i)], tool

    def test_the_corpus_is_the_one_the_arm_trained_on(self):
        assert 'PROBE_PROSE="--data $ARM_DATA"' in ARMS

    def test_the_walk_refuses_rather_than_falling_back_to_noise(self):
        import os
        src = open(os.path.join(REPO, "evals", "diag_dual_channel_walk.py"),
                   encoding="utf-8").read()
        i = src.index("no prose source")
        assert "--random_ids" in src[i:i + 400]

    def test_the_gates_warn_when_they_fall_back_to_noise(self):
        """Gates 1 and 2 are exact-equality checks and stay valid on random ids;
        gate 4 does not, and must say so rather than print a meaningless 0."""
        import os
        src = open(os.path.join(REPO, "tools", "prelaunch_final.py"),
                   encoding="utf-8").read()
        i = src.index("[gates] WARNING")
        block = src[i:i + 700]
        assert "GATE 4" in block and "BY CONSTRUCTION" in block
        assert "Gates 1 and 2" in block

    def test_the_donor_control_draws_a_genuinely_different_document(self):
        """chunks_b must be a different DOCUMENT, not a second random draw --
        otherwise 'donor' and 'real' are the same distribution."""
        import os
        src = open(os.path.join(REPO, "tools", "prelaunch_final.py"),
                   encoding="utf-8").read()
        assert "chunks_a, chunks_b = mk(0), mk(args.batch)" in src
