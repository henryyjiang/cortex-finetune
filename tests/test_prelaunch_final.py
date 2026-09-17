"""
The cortex-final pre-launch gates.

Each gate exists because a failure of its kind has already been paid for once
and was invisible in the loss.  So every test here comes in a pair wherever it
can: the gate passes on a correct model, AND the gate FAILS on a model with the
fault injected.  A gate that has never been seen to fail is not evidence.

Run: /c/Users/henry/miniconda3/envs/cortex-retro/python.exe -m pytest tests/test_prelaunch_final.py -q
"""
from __future__ import annotations

import io
import json
import os
import sys

import pytest
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "evals"))

from test_cortex_eval import VOCAB, _build_raven  # noqa: E402

from tools.prelaunch_final import (  # noqa: E402
    Z_GATE_KEYS, Z_OFF_MAX_DELTA, build_e_twin, check_donor_control,
    check_read_live, check_roundtrip, check_z_off_equivalence, print_report,
)

NV, K, CL, EOS = 4, 16, 16, VOCAB - 1
NC, T = 6, 4


def _model(latent=True, gated=True, **kw):
    torch.manual_seed(1234)
    common = dict(use_memory=True, memory_slots=0, accum_vecs=NV,
                  latent_carry=latent, eos_token_id=EOS)
    if gated:
        common.update(prefix_memory="gated", gate_slots=K, gate_route="ring",
                      gate_init="zero", gate_fill="grow")
    else:
        common.update(prefix_memory="accum", accum_max=K * 4)
    common.update(kw)
    return _build_raven(**common).train()


def _chunks(n=NC, batch=2, seed=0):
    torch.manual_seed(seed)
    ids = torch.randint(0, VOCAB - 1, (batch, CL * n))
    return [c.contiguous() for c in torch.chunk(ids, n, dim=1)]


NUM_STEPS = torch.tensor([0, T])


class TestGate1ZOffEquivalence:

    def test_suppressing_the_z_read_reproduces_the_e_only_loss(self):
        mz = _model(latent=True)
        twin, unexpected = build_e_twin(mz)
        g = check_z_off_equivalence(mz, mz.cortex, twin, _chunks(), NUM_STEPS)
        assert g["passed"], g
        assert g["max_abs_delta"] <= Z_OFF_MAX_DELTA

    def test_the_e_twin_drops_exactly_the_z_gate_and_nothing_else(self):
        """An E-only rebuild that reported NO unexpected keys would mean the Z
        gate was never allocated -- the twin would be comparable for the wrong
        reason."""
        mz = _model(latent=True)
        _, unexpected = build_e_twin(mz)
        assert unexpected, "no Z gate parameters were dropped -- was Z built?"
        assert all(any(k.endswith(z) for z in Z_GATE_KEYS) for k in unexpected)
        assert len(unexpected) == len(Z_GATE_KEYS)

    def test_the_gate_catches_z_leaking_into_e(self):
        """Fault injection: let the Z half contaminate the E half in `merge`.

        With the read suppressed, E's path must be untouched by Z, so this is a
        second path from Z into the loss -- exactly what gate 1 exists to
        localise, and exactly what every other instrument would report only as
        'the Z arm is different'.
        """
        mz = _model(latent=True)
        twin, _ = build_e_twin(mz)
        buf = mz.cortex.prefix
        D = buf.hidden_size
        real_merge = buf.merge

        def leaky(state, new_vecs, new_latent=None):
            out = real_merge(state, new_vecs, new_latent)
            return torch.cat([out[..., :D] + 0.05 * out[..., D:],
                              out[..., D:]], dim=-1)

        buf.merge = leaky
        try:
            g = check_z_off_equivalence(mz, mz.cortex, twin, _chunks(),
                                        NUM_STEPS)
        finally:
            del buf.merge
        assert not g["passed"], g
        assert g["max_abs_delta"] > Z_OFF_MAX_DELTA


class TestGate2CheckpointRoundTrip:

    def test_a_dual_channel_model_survives_the_config_round_trip(self):
        m = _model(latent=True)
        g = check_roundtrip(m, _chunks(n=3), NUM_STEPS)
        assert g["strict_load_error"] is None, g["strict_load_error"]
        assert g["geometry_survived"] and g["both_channels_survived"]
        assert g["carry_width"][0] == g["carry_width"][1]
        assert g["passed"], g

    def test_the_geometry_travels_in_the_json_and_not_in_the_code(self):
        """K, the route and the fill are runtime knobs the graft reads off
        config.json on every load.  A key train.py forgot to persist rebuilds a
        different buffer at resume and at eval, behind a healthy loss curve."""
        m = _model(latent=True)
        d = json.loads(m.config.to_json_string())
        for key in ("gate_slots", "gate_route", "gate_fill", "gate_init",
                    "latent_carry", "accum_vecs"):
            assert key in d, f"{key} does not survive to config.json"
        assert d["gate_slots"] == K and d["latent_carry"] is True

    def test_a_config_that_lost_latent_carry_is_caught_by_the_strict_load(self):
        """Bug class 2c, reproduced.  The rebuild is E-only, so the six Z gate
        tensors are UNEXPECTED keys -- strict=True raises, strict=False would
        drop them and run a half-width carry."""
        m = _model(latent=True)
        d = json.loads(m.config.to_json_string())
        d.pop("tie_word_embeddings", None)     # prepare_eval_checkpoint's pop
        d["latent_carry"] = False
        rebuilt = type(m)(type(m.config).from_dict(d))
        with pytest.raises(RuntimeError, match="[Uu]nexpected key"):
            rebuilt.load_state_dict(m.state_dict(), strict=True)

    def test_the_gate_reports_that_a_raw_saved_config_is_unloadable(self):
        """`save_pretrained` serialises `tie_word_embeddings` whenever it
        differs from the transformers default, and RavenConfig ALSO passes it
        explicitly to super() while forwarding **kwargs -- so a raw saved dir
        raises a duplicate-kwarg TypeError on reload.  That is the whole reason
        tools/prepare_eval_checkpoint.py exists.

        The gate performs the same pop (or it would fail for a reason every
        checkpoint already has, drowning the reason it is here for) and RECORDS
        whether the raw config was loadable, so the dependency on that tool is
        reported rather than assumed.  If transformers ever stops emitting the
        key, this test goes red and the pop can be retired deliberately.
        """
        m = _model(latent=True)          # tie_embeddings=False on this base
        g = check_roundtrip(m, _chunks(n=2), NUM_STEPS)
        assert g["needs_prepare_eval_checkpoint"] is True
        assert "tie_word_embeddings" in g["raw_from_dict_error"]
        assert g["passed"], "the pop is applied, so the gate itself must pass"

    def test_every_graft_building_flag_reached_the_serialised_config(self):
        """The third leg of the flag pin.  test_eval_checkpoint_flags already
        holds train.py's persist list against prepare_eval_checkpoint's; this
        checks the values actually ARRIVE in a config the rebuild reads."""
        m = _model(latent=True)
        g = check_roundtrip(m, _chunks(n=2), NUM_STEPS)
        assert g["flags_missing_from_json"] == [], g["flags_missing_from_json"]
        for key in ("gate_slots", "gate_route", "latent_carry", "accum_vecs"):
            assert key in g["flags_in_json"], key

    def test_an_e_only_model_round_trips_too(self):
        g = check_roundtrip(_model(latent=False), _chunks(n=3), NUM_STEPS)
        assert g["passed"], g
        assert g["z_gate_present"] is None


class TestGate3TheNoGradSplit:

    def test_the_fraction_is_measured_from_the_models_own_sampler(self):
        m = _model()
        g = check_read_live(m, mean_recurrence=8, mean_backprop_depth=8,
                            n_samples=2000)
        a = g["at_run_config"]
        assert 0.0 < a["read_live_frac"] < 1.0
        assert a["mean_recurrence"] == 8 and a["mean_backprop_depth"] == 8
        assert g["passed"]

    def test_the_record_says_which_depth_it_is_anchored_on(self):
        """Without --trained_depth the gate falls back to
        config.mean_recurrence, which on a B2-family checkpoint is the base
        recipe's 32 while the arm trained at 8 -- a 28x swing in the number that
        decides what a null result for Z means.  A downstream reader cannot tell
        the two cases apart from the fraction alone, so the anchor is written
        into the record instead of being left to be inferred."""
        m = _model()
        given = check_read_live(m, mean_recurrence=8, mean_backprop_depth=8,
                                n_samples=500)
        fell_back = check_read_live(m, mean_backprop_depth=8, n_samples=500)
        assert given["anchor"] == "trained_depth"
        assert fell_back["anchor"] == "config.mean_recurrence"
        assert fell_back["at_run_config"]["mean_recurrence"] == \
            m.config.mean_recurrence

    def test_deep_recurrence_starves_the_z_read(self):
        """At mr32 / depth 8 essentially no batch carries a Z read gradient --
        the structural fact that decides how the Z arm must be configured, and
        the reason it is measured rather than asserted."""
        m = _model()
        shallow = check_read_live(m, 8, 8, 2000)["at_run_config"]
        deep = check_read_live(m, 32, 8, 2000)["at_run_config"]
        assert deep["read_live_frac"] < shallow["read_live_frac"]
        assert deep["read_live_frac"] < 0.02

    def test_measuring_it_does_not_disturb_the_model(self):
        """It has to force train mode and move two config fields to sample at
        all; leaving either moved would silently change the run that follows."""
        m = _model()
        m.eval()
        before = (m.training, m.config.mean_recurrence,
                  m.config.mean_backprop_depth)
        check_read_live(m, 32, 4, 200)
        assert (m.training, m.config.mean_recurrence,
                m.config.mean_backprop_depth) == before

    def test_it_is_reproducible(self):
        m = _model()
        a = check_read_live(m, 8, 8, 500)["at_run_config"]["read_live_frac"]
        b = check_read_live(m, 8, 8, 500)["at_run_config"]["read_live_frac"]
        assert a == b


class TestGate4TheDonorControl:

    def test_it_reports_a_per_channel_content_delta(self):
        m = _model(latent=True)
        g = check_donor_control(m, m.cortex, _chunks(seed=1), _chunks(seed=2),
                                NUM_STEPS)
        assert g["passed"] and g["dual_channel"]
        for key in ("content_delta_both", "content_delta_e", "content_delta_z",
                    "column_delta"):
            assert key in g and g[key] == g[key]        # not NaN

    def test_an_e_only_carry_has_no_per_channel_rows(self):
        m = _model(latent=False)
        g = check_donor_control(m, m.cortex, _chunks(seed=1), _chunks(seed=2),
                                NUM_STEPS)
        assert not g["dual_channel"]
        assert "content_delta_e" not in g
        assert "content_delta_both" in g

    def test_the_donor_swap_keeps_the_column_count_and_none_does_not(self):
        """`real - none` drops the columns and so confounds information with a
        register effect; `real - donor` is matched.  That gap is the reason this
        control exists, so the two must not be the same number by accident."""
        m = _model(latent=True)
        g = check_donor_control(m, m.cortex, _chunks(seed=1), _chunks(seed=3),
                                NUM_STEPS)
        assert g["loss"]["donor_both"] != g["loss"]["none"]

    def test_swapping_nothing_changes_nothing(self):
        """Sanity on the swap itself: a donor identical to the real carry must
        score exactly the real carry's loss, or the harness is measuring its own
        substitution machinery."""
        from tools.prelaunch_final import _swap
        D = 8
        real = torch.randn(1, 4, 2 * D)
        for ch in ("e", "z", "both"):
            assert torch.equal(_swap(real, real, D, ch), real)


class TestTheConfigOverride:
    """`--set latent_carry=true` is what lets the gates run on the PARENT, before
    any Z checkpoint exists -- which is the only time a pre-launch gate is worth
    running."""

    def test_booleans_are_parsed_and_not_left_as_strings(self):
        """`latent_carry="false"` is TRUTHY.  Left as a string it would turn an
        intended E-only run into a dual-channel one with no symptom at all."""
        from tools.prelaunch_final import _parse_set
        got = _parse_set(["latent_carry=false", "gate_slots=64",
                          "gate_route=ring"])
        assert got == {"latent_carry": False, "gate_slots": 64,
                       "gate_route": "ring"}
        assert got["latent_carry"] is False
        assert isinstance(got["gate_slots"], int)

    def test_a_malformed_override_stops_the_run(self):
        from tools.prelaunch_final import _parse_set
        with pytest.raises(SystemExit):
            _parse_set(["latent_carry"])


class TestTheReport:

    def test_it_prints_ascii_and_returns_the_verdict(self):
        m = _model(latent=True)
        twin, _ = build_e_twin(m)
        gates = [
            check_z_off_equivalence(m, m.cortex, twin, _chunks(n=3), NUM_STEPS),
            check_roundtrip(m, _chunks(n=2), NUM_STEPS),
            check_read_live(m, 8, 8, 300),
            check_donor_control(m, m.cortex, _chunks(n=3, seed=1),
                                _chunks(n=3, seed=2), NUM_STEPS),
        ]
        buf = io.StringIO()
        ok = print_report(gates, out=buf)
        text = buf.getvalue()
        text.encode("ascii")
        assert ok is True
        assert "CORTEX-FINAL PRE-LAUNCH GATES" in text
        assert "FAIL" not in text
        assert "None of them says whether Z helps" in text

    def test_a_failing_gate_is_reported_as_failing(self):
        buf = io.StringIO()
        ok = print_report([{"gate": "donor_control", "passed": False,
                            "why": "shapes differ"}], out=buf)
        assert ok is False
        assert "[FAIL] donor_control" in buf.getvalue()


class TestTheSharedOverrideParser:
    """`--set` feeds `load_checkpoint(config_overrides=)`, and the walk and the
    gates must parse it identically -- a flag that means one thing to one tool
    and another to the other is the two-files-disagree failure in miniature."""

    def test_both_tools_use_the_same_parser(self):
        from model_utils import parse_config_overrides
        from tools.prelaunch_final import _parse_set
        import evals.diag_dual_channel_walk as walk
        assert _parse_set is parse_config_overrides
        src = open(os.path.join(REPO, "evals", "diag_dual_channel_walk.py"),
                   encoding="utf-8").read()
        assert "parse_config_overrides" in src

    def test_the_walk_accepts_set_because_a_sliced_branch_needs_it(self):
        """summary_emb is a parameter SHAPE.  A [16, D] slice loaded against the
        base dir's config (which still says the PARENT's width) is a size
        mismatch, so without --set the walk cannot open a sliced branch at all.
        """
        src = open(os.path.join(REPO, "evals", "diag_dual_channel_walk.py"),
                   encoding="utf-8").read()
        assert 'p.add_argument("--set"' in src
        assert "config_overrides=overrides" in src

    def test_the_prelaunch_job_passes_accum_vecs_to_the_walk(self):
        """The bug this caught: the walk step was passing the sliced checkpoint
        with no width override, so it would have died before printing a row."""
        sb = open(os.path.join(REPO, "pace", "prelaunch_final.sbatch"),
                  encoding="utf-8").read()
        # Anchor on the INVOCATION, not the header -- the header names every
        # step in running order before any of them runs.
        # Step 3 now takes the arm geometry through $SETS, which carries
        # accum_vecs; 3b sets it per side explicitly.
        i = sb.index('echo "######## 3. the 8-chunk')
        assert "$SETS" in sb[i:i + 900]
        assert "--set accum_vecs=$ACCUM_VECS" in sb

    def test_the_width_contrast_reads_both_sides_from_the_same_step(self):
        """model_only_chkpt_90000 is 1,552 steps short of the parent the slice
        was cut from; using it would confound width with training."""
        sb = open(os.path.join(REPO, "pace", "prelaunch_final.sbatch"),
                  encoding="utf-8").read()
        i = sb.index('echo "######## 3b. THE WIDTH CONTRAST')
        block = sb[i:sb.index("--out $OUT/width_contrast.json")]
        assert "$PARENT_PATH/chkpt.pt" in block
        assert "--set accum_vecs=$PARENT_VECS" in block
        assert "compare_width.py" in block

    def test_the_contrast_is_opt_in_so_the_gate_still_runs_without_a_parent(self):
        sb = open(os.path.join(REPO, "pace", "prelaunch_final.sbatch"),
                  encoding="utf-8").read()
        assert 'if [ -n "$PARENT_PATH" ]; then' in sb
        assert "PARENT_PATH=${PARENT_PATH:-}" in sb


class TestTheWalkRefusesNoise:
    """REGRESSION from the 2026-09-16 19:09 job: the walk ran on RANDOM IDS
    because the sbatch passed no prose source and random was the silent default.
    Its losses sat at 11.6-11.9 against ln(100352) = 11.52 -- the model was
    seeing noise, so every rank number in that table described noise."""

    def test_a_prose_source_is_required(self):
        src = open(os.path.join(REPO, "evals", "diag_dual_channel_walk.py"),
                   encoding="utf-8").read()
        assert 'p.add_argument("--data"' in src
        assert 'p.add_argument("--random_ids"' in src
        i = src.index("no prose source")
        assert "describe noise" in src[i:i + 400]

    def test_random_ids_is_opt_in_and_says_why_it_is_wrong(self):
        src = open(os.path.join(REPO, "evals", "diag_dual_channel_walk.py"),
                   encoding="utf-8").read()
        i = src.index('p.add_argument("--random_ids"')
        assert "nothing to converge on" in src[i:i + 900]

    def test_the_prelaunch_job_passes_a_prose_source_to_every_walk(self):
        sb = open(os.path.join(REPO, "pace", "prelaunch_final.sbatch"),
                  encoding="utf-8").read()
        assert "PROSE=" in sb
        body = sb[sb.index('echo "######## 3. the 8-chunk'):sb.index("compare_width.py")]
        assert body.count("$PROSE") == 3, body.count("$PROSE")

    def test_a_missing_prose_source_stops_the_job(self):
        sb = open(os.path.join(REPO, "pace", "prelaunch_final.sbatch"),
                  encoding="utf-8").read()
        i = sb.index("ERROR: no prose source for the walk")
        assert "exit 1" in sb[i:i + 400]


class TestStepThreeUsesTheArmsGeometry:
    """The 19:09 job ran step 3 as accum / E-only, so the 'dual-channel walk'
    exercised neither the gate nor Z and every Z column printed '-'.  Step 3 is
    the MACHINERY check and must use the arm's geometry; step 3b compares
    WEIGHTS and must use accum on both sides."""

    @staticmethod
    def _sb():
        return open(os.path.join(REPO, "pace", "prelaunch_final.sbatch"),
                    encoding="utf-8").read()

    def test_step_three_walks_the_gate_and_z(self):
        sb = self._sb()
        body = sb[sb.index('echo "######## 3. the 8-chunk'):
                  sb.index('echo "######## 3b.')]
        assert "$SETS" in body, "step 3 must take the arm geometry"
        assert "--set prefix_memory=accum" not in body

    def test_step_three_b_uses_accum_on_both_sides(self):
        sb = self._sb()
        body = sb[sb.index('echo "######## 3b.'):sb.index("compare_width.py")]
        assert body.count("--set prefix_memory=accum") >= 1
        assert "$SETS" not in body, "3b compares weights, not the arm geometry"

    def test_accum_max_is_set_per_width_so_the_chunk_count_matches(self):
        sb = self._sb()
        assert "BRANCH_MAX=$(( CROSS_CHUNKS * ACCUM_VECS ))" in sb
        assert "PARENT_MAX=$(( CROSS_CHUNKS * PARENT_VECS ))" in sb
        body = sb[sb.index('echo "######## 3b.'):sb.index("compare_width.py")]
        assert "--set accum_max=$BRANCH_MAX" in body
        assert "--set accum_max=$PARENT_MAX" in body

    def test_the_branch_side_of_the_contrast_is_its_own_file(self):
        """3b's branch walk is accum, so it cannot reuse step 3's gated walk.json
        -- that is what made the first contrast compare two different buffers."""
        sb = self._sb()
        assert "walk-branch-w$ACCUM_VECS.json" in sb
        assert "--branch $OUT/walk-branch-w$ACCUM_VECS.json" in sb


class TestPostInitClobberIsUndoneOnTheEvalPath:
    """REGRESSION from job 13270491, which went NaN at chunk 5 and took three of
    the four gates with it.

    `from_pretrained` runs post_init on every parameter the checkpoint does not
    supply, and for the cortex gate that is not the designed init -- the walk
    reported fg(bias) = 1.0000 where the design says sigmoid(1.0) = 0.7311, and
    ig_z(bias) = 1.0000 where it says sigmoid(0) = 0.5.  Four chunks ran clean;
    the NaN appeared at chunk 5, the first full SECOND lap, i.e. the first time
    the clobbered forget gate was applied to a row it had already written.

    train.py calls reset_cortex_graft_init for exactly this reason, and so does
    tools/smoke_prefix_real.py -- which is why the smoke passed every equivalent
    check in the same job.  The EVAL path never did.
    """

    @staticmethod
    def _src():
        return open(os.path.join(REPO, "evals", "model_utils.py"),
                    encoding="utf-8").read()

    def test_load_checkpoint_reapplies_the_designed_init(self):
        assert "reset_cortex_graft_init" in self._src()

    def test_it_runs_BEFORE_the_overlay_or_it_would_wipe_the_write_path(self):
        """Resetting re-seeds summary_emb and clears summary_seeded.  After the
        overlay that would destroy a trained write path; before it, every key the
        checkpoint carries is restored on top and only the genuinely missing ones
        keep the designed init."""
        src = self._src()
        assert (src.index("reset_cortex_graft_init(model")
                < src.index("# Optional overlay of finetuned weights"))

    def test_it_is_skipped_when_there_is_no_graft(self):
        src = self._src()
        i = src.index("reset_cortex_graft_init(model")
        head = src[max(0, i - 500):i]
        assert 'getattr(_unwrap(model), "cortex", None) is not None' in head

    def test_a_failure_to_reset_warns_rather_than_killing_the_load(self):
        src = self._src()
        i = src.index("reset_cortex_graft_init(model")
        assert "WARNING" in src[i:i + 600]


class TestGateThreeUsesTheTrainedDepth:
    """Job 13270491 reported read_live_frac at mr=32 (0.019) when the arms run
    mr8 (0.545) -- the inherited-mean_recurrence field again, on the one number
    that decides what a Z null result means."""

    def test_prelaunch_takes_a_trained_depth(self):
        src = open(os.path.join(REPO, "tools", "prelaunch_final.py"),
                   encoding="utf-8").read()
        assert 'p.add_argument("--trained_depth"' in src
        assert "mean_recurrence=args.trained_depth or None" in src

    def test_omitting_it_warns_instead_of_reporting_the_wrong_row(self):
        src = open(os.path.join(REPO, "tools", "prelaunch_final.py"),
                   encoding="utf-8").read()
        i = src.index("[gate 3] WARNING")
        assert "INHERITED" in src[i:i + 500]

    def test_the_sbatch_passes_the_arms_depth(self):
        sb = open(os.path.join(REPO, "pace", "prelaunch_final.sbatch"),
                  encoding="utf-8").read()
        assert "--trained_depth $MAX_MEAN_REC" in sb


class TestTheWidthContrastIsReportedNotGated:
    """It measures W=32 vs W=16 for the P1.0 decision, and p11 section 4 fixes
    the response to every verdict -- none of which changes this round's arms,
    since A1 and A3' both run W=16.  Folding it into RC turns an unresolved
    MEASUREMENT into "the pre-launch failed", which is the same confusion the
    gate-1 NaN caused: an exit code that cannot tell "the claim failed" from
    "this is not the claim being gated"."""

    @staticmethod
    def _sb():
        return open(os.path.join(REPO, "pace", "prelaunch_final.sbatch"),
                    encoding="utf-8").read()

    def test_it_has_its_own_status_variable(self):
        sb = self._sb()
        assert "WIDTH_RC=0" in sb
        assert "--out $OUT/width_contrast.json || WIDTH_RC=1" in sb

    def test_it_does_not_set_the_gate_status(self):
        sb = self._sb()
        i = sb.index("python tools/compare_width.py")
        assert "|| RC=1" not in sb[i:i + 400]

    def test_the_four_real_gates_still_gate(self):
        """The distinction has to be narrow: everything else still fails loudly."""
        sb = self._sb()
        # Anchor on the INVOCATIONS -- the header names every tool first.
        for tool in ("python -m pytest tests/",
                     "python tools/smoke_prefix_real.py",
                     "python evals/diag_dual_channel_walk.py",
                     "python tools/prelaunch_final.py"):
            i = sb.index(tool)
            assert "|| RC=1" in sb[i:i + 700], tool

    def test_an_unresolved_contrast_is_surfaced_and_not_swallowed(self):
        sb = self._sb()
        i = sb.index("width contrast (3b) did not produce a usable verdict")
        assert "REPORTED, not gating" in sb[i:i + 400]
        assert "Read it, do not block on it" in sb[i:i + 500]


class TestGate4RefusesRandomIds:
    """It used to WARN.  The warning printed, the run passed, and the parent's
    +0.0186 (job 13272609) went into the P1 donor table beside the arms'
    real-data -0.12 as though the two were comparable.  A measurement that is
    ~0 by construction is INVALID, not negative, and a gate is where that gets
    said."""

    def test_it_returns_invalid_without_a_prose_source(self):
        m = _model(latent=True)
        g = check_donor_control(m, m.cortex, _chunks(seed=1), _chunks(seed=2),
                                NUM_STEPS, real_data=False)
        assert g["passed"] is False
        assert "loss" not in g
        assert "RANDOM IDS" in g["why"]

    def test_the_default_is_still_a_real_measurement(self):
        m = _model(latent=True)
        g = check_donor_control(m, m.cortex, _chunks(seed=1), _chunks(seed=2),
                                NUM_STEPS)
        assert g["passed"] is True and "loss" in g


class TestRed10TheGateChainIsShifted:
    """RED 10, the other half.  `_chain_losses` feeds every loss-based gate in
    this file, and gate 4's donor swap scores its own final chunk the same way.
    Both passed `labels=ids`, so the -0.1234 content delta that reordered the
    whole P2 program was a ranking of three chance-level numbers
    (none 11.4277 < donor 11.5455 < real 11.6689, all inside 0.25 nats of
    ln(vocab) = 11.5157).  A gate that scores noise cannot fail loudly -- it
    returns a tidy number -- so this is pinned at the call site."""

    def test_chain_losses_passes_next_token_labels(self):
        from tools.prelaunch_final import _chain_losses
        m = _model()
        seen, chunks = [], _chunks()

        class Spy:
            def __getattr__(inner_self, k):
                return getattr(m, k)

            def __call__(inner_self, *a, **kw):
                seen.append((kw.get("input_ids"), kw.get("labels")))
                return m(*a, **kw)

        _chain_losses(Spy(), chunks, torch.tensor([0, T]), seed=0)
        assert len(seen) == len(chunks)
        for ids, y in seen:
            assert not torch.equal(y, ids), (
                "labels == input_ids in the gate chain: RED 10")
            assert torch.equal(y[:, :-1], ids[:, 1:])
            assert (y[:, -1] == -100).all()

    def test_the_donor_gate_scores_its_last_chunk_shifted_too(self):
        """Gate 4 builds its own forward rather than going through
        `_chain_losses`, which is exactly how one call site gets fixed and the
        other does not."""
        m = _model()
        seen = []

        class Spy:
            def __getattr__(inner_self, k):
                return getattr(m, k)

            def __call__(inner_self, *a, **kw):
                seen.append((kw.get("input_ids"), kw.get("labels")))
                return m(*a, **kw)

        chunks_a, chunks_b = _chunks(), _chunks(seed=7)
        check_donor_control(Spy(), m.cortex, chunks_a, chunks_b,
                            num_steps=torch.tensor([0, T]), seed=0)
        scored = [(x, y) for x, y in seen if y is not None]
        assert scored, "gate 4 scored nothing"
        for ids, y in scored:
            assert not torch.equal(y, ids)
            assert torch.equal(y[:, :-1], ids[:, 1:])


class TestTheSummaryDoesNotInventAVerdict:
    """This script's own summary printed "STILL AT CHANCE" over a walk whose
    margin was +7.6967, because a MISSING file and an AT-CHANCE file were folded
    into one flag (job 13304811).  That is the exact failure shape reds 8-11
    share -- a confident wrong verdict, not a loud failure -- committed by the
    instrument built to catch it.  Pinned at the text level, since the summary
    is a heredoc inside the launcher."""

    def _sbatch(self):
        return io.open(os.path.join(REPO, "pace", "red10_confirm.sbatch"),
                       encoding="utf-8").read()

    def test_missing_and_at_chance_are_tracked_separately(self):
        s = self._sbatch()
        assert "missing.append(name)" in s
        assert "at_chance.append(name)" in s
        assert "if at_chance:" in s, "the verdict must key off at_chance alone"

    def test_the_width_walks_do_not_backprop(self):
        """The W=32 walk OOMed at 139.78 GiB because the walk backprops one
        summed chain loss and retains every chunk's graph.  The width contrast
        reads the final carry's rank stats and needs no gradients -- and BOTH
        sides must be run the same way or the contrast is between two different
        measurements."""
        s = self._sbatch()
        # Both command lines, named individually -- counting occurrences would
        # pass on two flags on the same side and a comment.
        assert '$SETS_W16 --no_backward' in s
        assert '$SETS_W32 --no_backward' in s
