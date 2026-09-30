"""The 5B pair's launcher: the control and cortex-final must differ in ONE thing.

Every "memory helps by X" number in the paper is the difference between these
two runs, so the property the whole launch rests on is not that each arm is
configured correctly -- it is that they are configured IDENTICALLY except for
the mechanism.  Two sibling scripts satisfy that on the day they are written
and then drift; `pace/final_5b.sbatch` writes every shared value once, above a
single `case "$ARM"` block, and these tests are what keep it that way.

The house rule from tests/test_j6_arms.py applies unchanged: an arm is never
trusted to be itself because a flag was passed.  So the static checks below are
about what the script CANNOT express, and the live checks are about what the
forward pass has to produce.

Run: python -m pytest tests/test_final_5b.py -q
"""
from __future__ import annotations

import os
import re
import subprocess
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

COMMON = os.path.join(REPO, "pace", "final_5b_common.sh")
CONTROL = os.path.join(REPO, "pace", "control_5b.sbatch")
CORTEX = os.path.join(REPO, "pace", "cortex_final_5b.sbatch")
MIDRUN = os.path.join(REPO, "pace", "midrun_check.sbatch")
ZNORM = os.path.join(REPO, "pace", "measure_znorm.sbatch")
OOMGATE = os.path.join(REPO, "pace", "smoke_control_mbs2.sbatch")
WATCH = os.path.join(REPO, "tools", "watch_run.py")


def _read(path):
    with open(path, encoding="utf-8") as fh:
        return fh.read()


def _strip_comments(text):
    """Drop comment lines.  The headers in this repo quote flags at length, so
    a naive grep for `--cortex.latent_carry` finds prose, not configuration."""
    return "\n".join(ln for ln in text.splitlines()
                     if not ln.lstrip().startswith("#"))


def _arm(arm):
    """One arm launcher's body, comments stripped.  The arms are separate
    files now; the shared recipe they source is final_5b_common.sh."""
    return _strip_comments(_read(CONTROL if arm == "control" else CORTEX))


def _train_call():
    """The single `python train.py ...` invocation, which lives in the shared
    file -- there is exactly one for both arms, and that is the point."""
    body = _strip_comments(_read(COMMON))
    m = re.search(r"python train\.py \\\n(.*?)\n\s*RC=\$\?", body, re.S)
    assert m, "no `python train.py` call found in final_5b_common.sh"
    return m.group(1)


class TestTheScriptParses:
    def test_bash_accepts_it(self):
        for path in (COMMON, CONTROL, CORTEX, MIDRUN, ZNORM, OOMGATE):
            r = subprocess.run(["bash", "-n", path], capture_output=True)
            assert r.returncode == 0, f"{path}: {r.stderr.decode()}"

    def test_watch_run_compiles_and_takes_the_flags_the_sbatch_passes(self):
        r = subprocess.run([sys.executable, WATCH, "--help"],
                           capture_output=True)
        assert r.returncode == 0, r.stderr.decode()
        helptext = r.stdout.decode()
        for flag in ("--run", "--log", "--arm", "--max_steps", "--heal_end",
                     "--znorm_target"):
            assert flag in helptext, f"watch_run.py has no {flag}"


class TestTheOneVariable:
    """Everything outside the ARM case block is shared, by construction."""

    def test_no_mechanism_flag_leaks_into_the_shared_train_call(self):
        # The shared call may set cross_chunks (it is the CHUNK GEOMETRY, which
        # the control must match -- that is the whole reason `use_memory false`
        # alone was not a control), eos_from_tokens and diag_interval.  Any
        # OTHER cortex flag out here would silently apply to both arms.
        allowed = {"cross_chunks", "eos_from_tokens", "diag_interval"}
        found = set(re.findall(r"--cortex\.([a-z_0-9]+)", _train_call()))
        assert found == allowed, (
            f"the shared train.py call sets cortex flags {sorted(found)}; "
            f"only {sorted(allowed)} belong outside the ARM block")

    def test_arm_args_is_consumed_exactly_once_and_only_by_train_py(self):
        """Each arm APPENDS to ARM_ARGS freely in its own file; what must be
        true is that it reaches the command line exactly once, in the SHARED
        file, so neither arm can pick up the other's flags."""
        assert len(re.findall(r'ARM_ARGS="\$ARM_ARGS ', _arm("cortex"))) > 10, (
            "the cortex arm should be built by appending")
        body = _strip_comments(_read(COMMON))
        used = [ln for ln in body.splitlines()
                if "$ARM_ARGS" in ln and "ARM_ARGS:?" not in ln]
        assert len(used) == 1, (
            f"ARM_ARGS is interpolated on {len(used)} lines of the shared "
            f"file: {used}")
        assert used[0].strip() == "$ARM_ARGS \\", used[0]

    def test_the_shared_file_sets_no_arm_specific_value(self):
        """MICRO_BS_DEFAULT and RUN_NAME belong to the arms; the shared file
        may only REQUIRE them."""
        body = _strip_comments(_read(COMMON))
        for name in ("MICRO_BS_DEFAULT", "RUN_NAME", "ARM_ARGS", "ARM"):
            assigns = re.findall(rf'^\s*{name}=(?!\$\{{{name}:\?)', body, re.M)
            assert not assigns, (
                f"final_5b_common.sh assigns {name}, which is the arms' to set")

    def test_the_shared_file_is_not_submittable(self):
        """It has no #SBATCH header and no shebang-driven entry point, so a
        stray `sbatch pace/final_5b_common.sh` cannot start a 48h job."""
        raw = _read(COMMON)
        assert "#SBATCH" not in raw
        assert not raw.startswith("#!")

    def test_the_control_carries_use_memory_false_and_nothing_else(self):
        ctrl = _arm("control")
        flags = re.findall(r"--cortex\.([a-z_0-9]+)", ctrl)
        assert flags == ["use_memory"], (
            f"the control sets {flags}.  Dead mechanism config in a "
            "checkpoint's config.json is how a later eval script silently "
            "reconstructs a mechanism that never trained.")
        assert "--cortex.use_memory false" in ctrl

    @pytest.mark.parametrize("flag,value", [
        ("latent_encoding", "staggered_state"),
        ("latent_embed_frozen", "true"),
        ("latent_read", "embeds"),
        ("latent_read_znorm", "rms"),
        ("latent_s0_read", "false"),
        ("latent_depth_rule", "absolute"),
        ("latent_carry", "true"),
        ("latent_carry_read", "true"),
        ("e_carry_read", "true"),
        ("prefix_memory", "gated"),
        ("gate_route", "ring"),
        ("gate_init", "zero"),
        ("gate_fill", "grow"),
        ("window_backward", "true"),
        ("freeze_loop", "false"),
        ("prefix_pos", "tail"),
        ("prefix_eos_reset", "false"),
    ])
    def test_cortex_final_ships_the_frozen_architecture(self, flag, value):
        cortex = _arm("cortex")
        assert f"--cortex.{flag} {value}" in cortex, (
            f"--cortex.{flag} is not pinned to {value}.  The Z architecture "
            "was frozen 2026-09-28; this file is the spec.")

    def test_no_j_arm_treatment_flag_is_wired_in(self):
        """`mix`, `slow` and the Z-only arm are J6 limbs, not the design."""
        cortex = _arm("cortex")
        for leak in ("latent_read_scramble_p", "latent_stride",
                     "latent_write_only", "e_dropout 0.",):
            if leak == "e_dropout 0.":
                assert "--cortex.e_dropout 0.0" in cortex
                continue
            assert leak not in cortex, (
                f"{leak} is a J6 limb's flag and has no place in cortex-final")


class TestTheLaunchGates:
    def test_znorm_target_has_no_default(self):
        """It is E's MEASURED row norm and it is lineage-specific: 171.0 on B2,
        136.0 on the b2 heal branch.  cortex-final descends from neither."""
        body = _arm("cortex")
        assert not re.search(r"ZNORM_TARGET=\$\{ZNORM_TARGET:-", body), (
            "ZNORM_TARGET has a default.  An inherited value puts Z into the "
            "splice at a scale E does not have on this lineage.")
        assert 'if [ -z "${ZNORM_TARGET:-}" ]' in body, (
            "nothing refuses a cortex launch with ZNORM_TARGET unset")

    def test_the_control_does_not_default_to_the_memory_arms_micro_bs(self):
        """window_backward is refused without memory, so the control holds all
        8 chunks' graphs where cortex-final holds 4 -- 4,096 column-positions
        against 2,560.  The 90.84 GiB measured at mbs 2 is the MEMORY arm's."""
        assert "MICRO_BS_DEFAULT=1" in _arm("control")
        assert "MICRO_BS_DEFAULT=2" in _arm("cortex")

    def test_micro_bs_is_pinned_for_the_life_of_the_chain(self):
        """train.py:1997 computes the resume row cursor as data_start_step x
        micro_batch_size, so changing mbs mid-chain skips the wrong rows."""
        body = _strip_comments(_read(COMMON))
        assert ".micro_bs" in body and "MICRO_BS=$WAS" in body, (
            "nothing records or enforces MICRO_BS across the resume chain")

    def test_accum_max_tracks_the_geometry_rather_than_being_a_literal(self):
        cortex = _arm("cortex")
        assert "ACCUM_MAX=$(( ACCUM_VECS * CROSS_CHUNKS ))" in cortex, (
            "accum_max must be derived: train.py asserts "
            "cross_chunks x accum_vecs <= accum_max, and a literal goes stale "
            "the moment either moves")

    def test_the_ring_lap_gate_is_pre_flighted(self):
        """cross_chunks >= 2 x (gate_slots / accum_vecs), or the gate NEVER
        fires in training and the run trains an accum buffer with dead gate
        parameters behind a healthy loss curve.  This geometry sits on the
        exact boundary (8 = 2 x 4), so there is no margin."""
        body = _arm("cortex")
        assert "LAP=$(( GATE_SLOTS / ACCUM_VECS ))" in body
        assert 'CROSS_CHUNKS" -lt $(( 2 * LAP ))' in body

    def test_the_corpus_switch_guard_exists(self):
        """Omitting --reset_dataset_position on the switch link costs 1.15B
        tokens and reports 'finished'.  The B1 incident."""
        body = _strip_comments(_read(COMMON))  # shared: both arms switch
        assert "reset_dataset_position" in body
        assert "HEAL_END" in body


class TestTheSupersededControlRefuses:
    def test_control_retrofit_will_not_launch_without_an_override(self):
        path = os.path.join(REPO, "pace", "control_retrofit.sbatch")
        body = _strip_comments(_read(path))
        assert "I_KNOW_THIS_IS_SUPERSEDED" in body, (
            "the old control script is still launchable, and it would produce "
            "a control at micro_batch_size 1 in another wandb project -- no "
            "longer matched to cortex-final")


class TestTheReadOutMirrorsTheTraining:
    """A rebuild at the wrong flags scores a model nobody trained.  This class
    of mismatch has cost this project two runs, so the midrun read-out's --set
    list is checked against the launcher's, flag by flag."""

    @pytest.mark.parametrize("flag,value", [
        ("latent_encoding", "staggered_state"),
        ("latent_embed_frozen", "true"),
        ("latent_read", "embeds"),
        ("latent_read_znorm", "rms"),
        ("latent_s0_read", "false"),
        ("prefix_memory", "gated"),
        ("gate_route", "ring"),
    ])
    def test_the_midrun_rebuild_matches_the_trained_config(self, flag, value):
        mid = _strip_comments(_read(MIDRUN))
        assert f"--set {flag}={value}" in mid, (
            f"the mid-run read-out does not force {flag}={value}, so it would "
            "rebuild the graft at the BASE config's value")

    def test_the_midrun_depth_band_is_the_one_the_run_trains(self):
        """J6's read-out hardcoded latent_depth_lo=2.  cortex-final trains 1,
        and a rebuild at 2 reads a band the arm never wrote."""
        mid = _strip_comments(_read(MIDRUN))
        assert "latent_depth_lo=${LATENT_LO:-1}" in mid
        assert "LATENT_LO=${LATENT_LO:-1}" in _arm("cortex")

    def test_the_midrun_refuses_a_znorm_target_it_was_not_given(self):
        mid = _strip_comments(_read(MIDRUN))
        assert 'ZNORM_TARGET:?' in mid, (
            "the read-out would rebuild at the graft default (3.0) and score "
            "the read at 45x the wrong strength")

    def test_the_midrun_checks_the_diag_before_scoring(self):
        """The flags above are what we BELIEVE; cortex_diag.jsonl is what
        happened.  e_carry_read printed correctly for two runs while doing
        nothing."""
        # Comments stripped: this file's own header EXPLAINS the `is not
        # True` trap at length, and a naive grep finds the prose.
        mid = _strip_comments(_read(MIDRUN))
        assert "cortex_diag.jsonl" in mid
        assert "z_embed_frozen" in mid and "z_n_depths" in mid
        # int(), not `is True`: training_diag casts bools to int for wandb, and
        # `1 is not True` is True -- the comparison that failed 13662406_2 on a
        # run whose projection WAS frozen.
        assert "int(ef) != 1" in mid
        assert "is not True" not in mid


class TestWatchRunReadsRealDiagnostics:
    """The health check has to survive a real cortex_diag.jsonl, not a mock."""

    DIAG = os.path.join(REPO, "logs", "j6_diag", "j6-a3z-stag.jsonl")

    @pytest.mark.skipif(not os.path.isfile(DIAG), reason="no committed J7 diag")
    def test_it_passes_a_run_that_was_healthy(self, tmp_path):
        import shutil
        run = tmp_path / "cortex-final"
        run.mkdir()
        shutil.copy(self.DIAG, run / "cortex_diag.jsonl")
        r = subprocess.run(
            [sys.executable, WATCH, "--run", str(run), "--arm", "cortex",
             "--znorm_target", "136.0"], capture_output=True)
        out = r.stdout.decode()
        assert "z_embed_ratio" in out and "n_depths" in out
        assert "FAIL" not in out, out

    @pytest.mark.skipif(not os.path.isfile(DIAG), reason="no committed J7 diag")
    def test_it_fails_when_E_has_drifted_out_from_under_the_target(self, tmp_path):
        """||E|| moved 135.5 -> 144.1 over J7's 2,000 steps.  This run is 19x
        longer, so the band is not theoretical."""
        import shutil
        run = tmp_path / "cortex-final"
        run.mkdir()
        shutil.copy(self.DIAG, run / "cortex_diag.jsonl")
        r = subprocess.run(
            [sys.executable, WATCH, "--run", str(run), "--arm", "cortex",
             "--znorm_target", "120.0"], capture_output=True)
        assert r.returncode == 1
        assert "znorm_target" in r.stdout.decode()

    def test_a_control_that_wrote_diag_rows_is_a_failure(self, tmp_path):
        """training_diag returns {} with no cortex, so a non-empty file means
        a graft was built and this is not a no-memory arm."""
        run = tmp_path / "c-chunked"
        run.mkdir()
        (run / "cortex_diag.jsonl").write_text('{"step": 1, "rows": 64}\n')
        r = subprocess.run(
            [sys.executable, WATCH, "--run", str(run), "--arm", "control"],
            capture_output=True)
        assert r.returncode == 1
        assert "should be NONE" in r.stdout.decode()

    def test_a_memory_arm_with_no_diag_at_all_is_a_failure(self, tmp_path):
        run = tmp_path / "cortex-final"
        run.mkdir()
        r = subprocess.run(
            [sys.executable, WATCH, "--run", str(run), "--arm", "cortex"],
            capture_output=True)
        assert r.returncode == 1
        assert "no cortex_diag.jsonl rows" in r.stdout.decode()


class TestThePreFlightGates:
    """Two short GPU jobs stand in front of the two arms, and each has a way
    of silently measuring the wrong thing."""

    def test_the_znorm_walk_uses_real_tokens(self):
        """diag_dual_channel_walk's own header: a random-id walk sat at loss
        11.6-11.9 against ln(vocab) = 11.52 -- the model was seeing noise,
        "and every rank number in that table described noise".  E is a
        post-ln_f state, so its norm has to be measured on real text."""
        body = _strip_comments(_read(ZNORM))
        assert "--random_ids" not in body, (
            "the ZNORM_TARGET walk passes --random_ids, which measures E's "
            "norm on noise")
        assert "--data" in body and "fineweb_edu" in body, (
            "it should walk the corpus the run STARTS on")

    def test_the_znorm_walk_runs_at_the_runs_own_geometry(self):
        body = _strip_comments(_read(ZNORM))
        assert "CHUNKS=${CHUNKS:-8}" in body
        assert "CHUNK_LEN=${CHUNK_LEN:-512}" in body
        assert "T=${T:-8}" in body, (
            "an unset T evaluates a retrofit config at mean_recurrence 32 -- "
            "finding 0c, and it cost the B1 carry headline")

    def test_the_znorm_walk_builds_the_arms_graft(self):
        body = _strip_comments(_read(ZNORM))
        for flag in ("use_memory=true", "prefix_memory=gated", "accum_vecs=16",
                     "gate_slots=64", "latent_carry=true"):
            assert f"--set {flag}" in body, (
                f"the walk does not force {flag}, so it would measure the "
                "BASE config's carry and not cortex-final's")

    def test_the_oom_gate_prices_the_control_without_window_backward(self):
        """train.py REFUSES window_backward and carry_grad_chunks without
        memory, so a smoke that passed them to the control cell would price a
        configuration the control cannot run -- and that is the entire reason
        the two arms differ on micro_batch_size."""
        body = _strip_comments(_read(OOMGATE))
        m = re.search(r'else\n\s*MEM_ARGS="(--use_memory false)"', body)
        assert m, "the no-memory cell should pass use_memory false and nothing else"
        assert "--window_backward" in body, (
            "the MEMORY cells must carry window_backward, or they price a "
            "model nobody is launching")

    def test_the_oom_gate_includes_the_cross_check_against_a_real_run(self):
        """cortex mbs2 has a MEASURED value (90.84 GiB, job 13662406_2).  If
        the smoke disagrees with it, the control rows inherit that error."""
        body = _strip_comments(_read(OOMGATE))
        assert body.count("use_memory true") >= 1
        assert "cortex-mbs2:2:true" in body and "control-mbs1:1:false" in body

    def test_the_oom_gate_does_not_call_a_marginal_fit_a_fit(self):
        """B2 peaked 4% above this gate's own prediction, and a 48h run meets
        fragmentation a four-step smoke does not."""
        assert "0.94" in _strip_comments(_read(OOMGATE))


class TestTheLaunchersPointAtEachOther:
    def test_each_arm_names_its_own_file_in_the_next_link_hint(self):
        body = _strip_comments(_read(COMMON))
        assert "cortex_final_5b.sbatch" in body and "control_5b.sbatch" in body, (
            "the post-link hint must name the arm's OWN launcher, or a resume "
            "is one copy-paste away from starting the other arm")

    def test_the_cortex_arm_warns_when_a_control_link_is_live(self):
        """The plan is control first, memory arm after.  A warning, not a
        refusal: the control's last link draining while this one queues is a
        legitimate overlap."""
        body = _arm("cortex")
        assert "squeue" in body and "-n c5b" in body
        assert "scancel this job" in body
        # and it must NOT exit on it
        assert not re.search(r"squeue[\s\S]{0,900}?exit 1", body), (
            "the sequencing check should warn, never refuse")


class TestThePackCheck:
    """The pack check gates a 48h submit, so its own arithmetic has to be the
    run's -- a check that silently asks a different question is worse than no
    check, because it reports OK."""

    PACKS = os.path.join(REPO, "pace", "check_packs.sbatch")

    def test_the_step_split_is_the_runs_own(self):
        """--steps is the steps SERVED FROM THIS PACK, never the horizon.
        Passing 38,147 for the mix pack over-states its need by 43%."""
        body = _strip_comments(_read(self.PACKS))
        assert "MAX_STEPS=${MAX_STEPS:-38147}" in body
        assert "HEAL_END=${HEAL_END:-11444}" in body
        assert "MIX_STEPS=${MIX_STEPS:-$(( MAX_STEPS - HEAL_END ))}" in body, (
            "the mix budget must be derived from the split, not written twice")

    def test_the_batch_size_matches_the_launchers(self):
        """Row budget is steps x batch_size and is independent of
        micro_batch_size -- so this is one check for both arms.  The
        framework's S10 counts were computed at 16 and are half these."""
        body = _strip_comments(_read(self.PACKS))
        assert "BATCH_SIZE=${BATCH_SIZE:-32}" in body
        assert "BATCH_SIZE=${BATCH_SIZE:-32}" in _strip_comments(_read(COMMON))

    def test_it_checks_the_same_packs_the_launchers_read(self):
        packs = _strip_comments(_read(self.PACKS))
        common = _strip_comments(_read(COMMON))
        for name in ("data/fineweb_edu_olmo_len4096", "data/pg19_fw50_olmo_len4096"):
            assert name in packs and name in common, (
                f"{name} is not shared between the check and the launcher")

    def test_the_forgotten_flag_is_priced_rather_than_hidden(self):
        """resume_rows 0 is a CLAIM about how the run is driven (no branch,
        and --reset_dataset_position on the switch link), so the counterfactual
        is run too -- its FAIL is expected and must not gate the launch."""
        body = _strip_comments(_read(self.PACKS))
        assert "RESUME_ROWS=${RESUME_ROWS:-$(( HEAL_END * BATCH_SIZE ))}" in body
        m = re.search(r'if \[ "\$RESUME_ROWS" -gt 0 \];(.*?)\nfi\n', body, re.S)
        assert m, "the counterfactual block is missing"
        assert "RC_ALL" not in m.group(1), (
            "the counterfactual sets RC_ALL, so an EXPECTED failure would gate "
            "the launch")

    def test_the_missing_val_pack_does_not_gate_a_training_launch(self):
        """fineweb_edu_olmo_val is built later and is read only by the
        mid-run read-out; a training link must not wait on it."""
        body = _strip_comments(_read(self.PACKS))
        m = re.search(r"\*fineweb_edu_olmo_val\*\)(.*?);;", body, re.S)
        assert m, "no special case for the not-yet-built val pack"
        assert "RC_ALL" not in m.group(1)

    def test_it_survives_being_run_by_hand(self):
        """An unset SLURM_SUBMIT_DIR would `cd` to $HOME and then report every
        pack as missing -- a green-looking path to a wrong answer."""
        body = _strip_comments(_read(self.PACKS))
        assert "SLURM_SUBMIT_DIR:-" in body, (
            "bare `cd $SLURM_SUBMIT_DIR` sends a hand-run to $HOME")

    def test_the_znorm_job_exports_the_allocator_setting(self):
        """Its first run OOMed (job 13764409) and the CUDA error asked for
        this by name.  The walk chains 8 chunks, which is exactly the
        fragmenting allocation pattern expandable_segments exists for."""
        assert "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True" in _read(ZNORM)

    def test_the_znorm_walk_batch_matches_the_established_footprint(self):
        """prelaunch_final walks at chunk_len 256, batch 2.  This asks for the
        RUN's geometry (512) so the batch has to come down, or it is 4x a
        footprint nobody has ever had trouble with."""
        assert "BATCH=${BATCH:-2}" in _strip_comments(_read(ZNORM))

    def test_it_does_not_trade_geometry_for_headroom(self):
        """CHUNKS and CHUNK_LEN are the RUN's, so shrinking them to fit would
        measure ||E|| under a layout cortex-final never trains."""
        body = _strip_comments(_read(ZNORM))
        assert "CHUNKS=${CHUNKS:-8}" in body
        assert "CHUNK_LEN=${CHUNK_LEN:-512}" in body
