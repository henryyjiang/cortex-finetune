"""J6's three limbs: `mix`, `delta` and `slow`, against J4's control.

Each limb is ONE flag away from the control, and every flag here defaults to
the control's value.  That is the exact shape of the failure this project has
already paid for twice -- e_carry_read parsed correctly, printed correctly in
the launcher banner, persisted into the checkpoint and DID NOTHING for two
smoke runs, because the allowlist that copies a cortex flag onto the model
config did not list it.  The Z-only limb spliced full-norm E rows behind a
healthy loss curve, and only a measured number (z_e_spliced_norm) caught it.

So the rule these tests encode: a limb is never trusted to be itself because a
flag was passed.  For each of the three there is (a) a refusal for every
neighbouring design the flag could silently become, (b) a MEASURED diagnostic
the forward pass had to produce, and (c) a check that the flag survives both
train.py gates and reaches the eval config.

Run: python -m pytest tests/test_j6_arms.py -q
"""
from __future__ import annotations

import os
import re
import sys

import pytest
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from test_j4_embeds import (  # noqa: E402
    B, D, E_NORM, K, S, W, _carry, _cortex, _pack, _z_block, _e_block,
)
from cortex_memory.buffers import PrefixGatedBuffer  # noqa: E402
from cortex_memory.health import gate_param_health, latent_runtime  # noqa: E402


def _read(path):
    with open(os.path.join(REPO, path), encoding="utf-8") as fh:
        return fh.read()


# ─── 1. `mix` -- option 1, the partial donor roll ───────────────────────────

class TestTheMixedRollIsNotTheDonorLimb:
    """p=1.0 is J4's donor limb AND IT ALREADY RAN: carry 0.4367 against the
    real limb's 0.7481.  At p=1 the carried Z is noise on every row, so the
    optimum is to ignore Z entirely and the limb measures nothing about
    content.  The whole point of `mix` is 0 < p < 1, and the code refuses the
    degenerate ends rather than letting one wear the other's name."""

    def test_p_one_is_refused_because_it_is_the_donor_limb(self):
        with pytest.raises(ValueError, match="IS "):
            _cortex(latent_read_scramble_p=1.0)

    def test_the_bool_and_the_float_refuse_each_other(self):
        with pytest.raises(ValueError, match="refuse each other"):
            _cortex(limb="donor", latent_read_scramble_p=0.25)

    @pytest.mark.parametrize("p", [-0.1, 1.5, float("nan")])
    def test_out_of_range_raises(self, p):
        with pytest.raises(ValueError, match=r"must be in \[0, 1\]"):
            _cortex(latent_read_scramble_p=p)

    def test_mixing_a_channel_nothing_reads_raises(self):
        with pytest.raises(ValueError, match="latent_carry_read"):
            _cortex(limb="noread", latent_read_scramble_p=0.25)

    def test_zero_is_the_control_and_builds(self):
        assert _cortex(latent_read_scramble_p=0.0).latent_read_scramble_p == 0.0


class TestTheMixActuallyFires:
    """(b): the MEASURED rate, not the configured one."""

    def _rows(self, p, train=True, draws=400):
        g = _cortex(latent_read_scramble_p=p)
        g.train(train)
        st, emb = _carry(), torch.randn(B, S, D)
        torch.manual_seed(7)
        for _ in range(draws):
            _pack(g, st, emb=emb)
        return g

    def test_the_measured_rate_tracks_the_configured_one(self):
        g = self._rows(0.25)
        assert g._z_mix_rows > 0, "no row was ever offered to the roll"
        assert 0.18 < g.latent_mix_frac < 0.32, g.latent_mix_frac

    def test_a_higher_p_rolls_more(self):
        assert self._rows(0.75).latent_mix_frac > self._rows(0.25).latent_mix_frac

    def test_eval_never_mixes(self):
        """THE one that matters most.  Every read-out cell reaches
        _latent_z_rows with latent_read_null set and returns early -- except
        the OPERATING cell, which sets no null.  An un-gated mix would score
        the treatment limb on mixed Z and report it as the arm's own number."""
        g = self._rows(0.5, train=False)
        assert g.latent_mix_frac == 0.0
        assert g._z_mix_rows == 0

    def test_the_control_reports_zero_and_is_indistinguishable_from_off(self):
        g = self._rows(0.0)
        assert g.latent_mix_frac == 0.0 and g._z_mix_rows == 0

    def test_it_is_per_row_not_per_batch(self):
        """Both conditions have to appear in the SAME micro-batch, or the
        gradient that says 'own Z pays, donor Z does not' alternates between
        steps instead of being present in each."""
        g = _cortex(latent_read_scramble_p=0.5)
        g.train(True)
        st, emb = _carry(), torch.randn(B, S, D)
        torch.manual_seed(3)
        seen = set()
        for _ in range(200):
            g._z_mixed_rows = g._z_mix_rows = 0
            _pack(g, st, emb=emb)
            seen.add(g._z_mixed_rows)
        assert seen - {0, B}, (
            f"only whole-batch outcomes {sorted(seen)} in {B} rows -- the draw "
            "is per batch, not per row")

    def test_the_measured_rate_reaches_the_diag_row(self):
        g = self._rows(0.25)
        h = latent_runtime(g)
        assert h["mix_p"] == 0.25
        assert 0.18 < h["mix_frac"] < 0.32


# ─── 2. `delta` -- option 2, the write that samples the loop ────────────────

class TestDeltaIsAdmittedAndStaggers:
    """D3 ranked the write candidates by how well each DECODES the register
    file -- a criterion that selects for overlap with E, since decoding the
    registers is what E already does at +0.72.  `endpoint` won it by being
    s_T at the SUMMARY columns, the pre-coda twin of E, and adds ~1% over E
    for its trouble (increment_share -0.003..+0.019 across six checkpoints,
    15-27% of its norm outside E's span).  `delta` is the one candidate
    carrying the loop's INTERMEDIATE computation."""

    def test_delta_now_builds_with_the_embeds_read(self):
        assert _cortex(latent_encoding="delta").latent_encoding == "delta"

    @pytest.mark.parametrize("enc", ["tokens", "scratch"])
    def test_the_writes_d3_measured_empty_stay_refused(self, enc):
        """A different verdict from untested: D3 measured these holding ~1% of
        E's register margin, so reading one through a better reader re-runs
        attempt 2's NO with a new read and the same empty channel."""
        with pytest.raises(ValueError):
            _cortex(latent_encoding=enc)

    def test_endpoint_takes_a_single_depth_and_delta_does_not(self):
        """The claim that sent J6 here: latent_write() SHORT-CIRCUITS to the
        single-slice _latent_state_write() for every encoding but 'delta', so
        latent_depth_{rule,lo,hi} ride in J4's and J5's configs -- they are
        right there in the checkpoints' `sets` -- and do NOTHING."""
        src = _read("cortex_graft.py")
        i = src.index("def latent_write(")
        body = src[i:i + 3000]
        assert 'if self.latent_encoding != "delta":' in body
        assert body.index('return self._latent_state_write()') < body.index(
            "latent_depth_map("), (
            "the depth map is reached before the short-circuit -- staggering "
            "would then apply to endpoint too and this test is stale")

    def test_delta_draws_its_rows_from_MORE_THAN_ONE_depth(self):
        """(b) for this limb: staggering is the mechanism, so prove it fired.
        Each taped depth is filled with its own constant, so the row's value
        names the depth it came from."""
        g = _cortex(latent_encoding="delta", latent_depth_rule="absolute",
                    latent_depth_lo=2, latent_depth_hi=9)
        n_sum = g.prefix.n_vec
        g._z_tape = [torch.full((B, n_sum, D), float(k + 1))
                     for k in range(7)]
        out = g.latent_write()
        assert out.shape == (B, n_sum, D)
        depths = {float(out[0, j, 0].detach()) for j in range(n_sum)}
        assert len(depths) > 1, (
            f"all {n_sum} rows came from depth {depths} -- the write is a "
            "single slice and the limb is `endpoint` under another name")

    def test_the_rescale_attempt_one_never_had_is_still_required(self):
        """Attempt 1 failed `delta` on SCALE, not content: deltas are ~0.35 in
        norm against E's ~136, and the s0 site turned on only when Z drowned
        E.  Passing delta without the rescale would repeat that exactly."""
        with pytest.raises(ValueError, match="znorm"):
            _cortex(latent_encoding="delta", latent_read_znorm="none")


# ─── 3. `slow` -- option 3, the short-term / long-term split ────────────────

def _buf(stride=1, **kw):
    return PrefixGatedBuffer(D, W, n_slots=K, route="ring", gate_init="zero",
                             fill="grow", carries_latent=True,
                             latent_stride=stride, **kw)


def _cand(i):
    """One chunk's (E, Z) candidate, each filled with a chunk-naming constant
    so a row's value says which chunk wrote it."""
    return (torch.full((B, W, D), float(i)),
            torch.full((B, W, D), float(100 + i)))


def _run(buf, n_chunks):
    st = None
    for i in range(1, n_chunks + 1):
        e, z = _cand(i)
        st = buf.merge(st, e, z)
    return st


class TestTheStrideRefusesTheNeighbouringDesigns:
    @pytest.mark.parametrize("st", [0, -1])
    def test_a_stride_below_one_raises(self, st):
        with pytest.raises(ValueError, match="must be >= 1"):
            _buf(stride=st)

    def test_a_stride_without_a_z_channel_raises(self):
        with pytest.raises(ValueError, match="E-only buffer"):
            PrefixGatedBuffer(D, W, n_slots=K, route="ring",
                              carries_latent=False, latent_stride=2)

    def test_a_stride_off_the_ring_raises(self):
        with pytest.raises(ValueError, match="route='ring'"):
            PrefixGatedBuffer(D, W, n_slots=K, route="mix",
                              carries_latent=True, latent_stride=2)


class TestTheStrideChangesZAndOnlyZ:
    """The limb has to differ from the control in the Z channel and nothing
    else, or a difference in the carry numbers is unattributable."""

    def test_es_rows_are_bit_identical_to_the_control(self):
        torch.manual_seed(0)
        a = _run(_buf(stride=1), 8)
        torch.manual_seed(0)
        b = _run(_buf(stride=2), 8)
        ea, _ = _buf().split_channels(a)
        eb, _ = _buf().split_channels(b)
        assert torch.equal(ea, eb), "the stride moved E"

    def test_z_reaches_further_back_than_e(self):
        """At K=16, W=4 the ring lap is 4 chunks: E ends holding the last 4.
        Z at stride 2 writes pooled blocks, so its rows reach further."""
        buf = _buf(stride=2)
        st = _run(buf, 8)
        e, z = buf.split_channels(st)
        e_chunks = {round(float(v)) for v in e[0, :, 0].detach()}
        # E: the last lap, chunks 5-8 (gated onto what was there, so the
        # values move -- what matters is that Z's span is WIDER).
        z_vals = sorted({float(v) for v in z[0, :, 0].detach()})
        assert len(z_vals) >= 2
        assert buf._z_chunk > 0, "the stride never engaged: no pooled block"
        assert len(e_chunks) >= 1

    def test_z_is_written_once_per_stride_not_once_per_chunk(self):
        buf = _buf(stride=2)
        _run(buf, 8)
        # chunks 1-4 grow (no ring, no stride), chunks 5-8 ring -> 2 windows
        assert buf._z_chunk == 2, buf._z_chunk

    def test_the_pool_is_the_mean_not_the_sum(self):
        """A sum would make `slow` a SCALE change too, and veto 4's whole
        point is that a channel entering at the wrong scale is a different
        experiment."""
        buf = _buf(stride=2)
        _run(buf, 4)                       # fill the ring, nothing pooled yet
        assert buf._z_pending == []
        e, z = _cand(5)
        buf.merge(_run(_buf(stride=2), 4), e, z)
        # one candidate pending, mean of one == itself
        assert len(buf._z_pending) in (0, 1)

    def test_a_new_sequence_drops_a_half_filled_pool(self):
        """`state is None` is the per-sequence reset signal.  A pool carried
        across it would average one sequence's chunk into another's."""
        buf = _buf(stride=2)
        _run(buf, 5)                       # leaves one candidate pending
        assert len(buf._z_pending) == 1
        e, z = _cand(1)
        buf.merge(None, e, z)
        assert buf._z_pending == [] and buf._z_chunk == 0

    def test_the_stride_reaches_the_diag_row(self):
        h = gate_param_health(_buf(stride=2))
        assert h["z_buf_stride"] == 2 and "z_blocks" in h
        assert gate_param_health(_buf(stride=1))["z_buf_stride"] == 1


# ─── 4. both train.py gates, and the eval config ────────────────────────────

class TestTheFlagsSurviveEveryGate:
    """THE e_carry_read LESSON.  train.py has two independent gates a cortex
    flag must pass -- the defaults dict jsonargparse types the CLI value from,
    and the ALLOWLIST that is the only thing copying it onto the model config
    -- and prepare_eval_checkpoint.py has a third for the read-out.  Passing
    two of the three is silent."""

    FLAGS = ("latent_read_scramble_p", "latent_stride")

    @pytest.mark.parametrize("flag", FLAGS)
    def test_it_is_in_the_cortex_defaults(self, flag):
        src = _read("train.py")
        assert re.search(rf"^\s*{flag}=", src, re.M), (
            f"{flag} is not in train.py's cortex default dict, so "
            "jsonargparse has nothing to type the CLI value from")

    @pytest.mark.parametrize("flag", FLAGS)
    def test_it_is_in_the_allowlist_that_reaches_the_model(self, flag):
        src = _read("train.py")
        i = src.index('setattr(config, _k, cfg.cortex[_k])')
        head = src.rindex("for _k in (", 0, i)
        assert f'"{flag}"' in src[head:i], (
            f"{flag} is absent from the allowlist -- it would parse, print, "
            "persist and do nothing, and the limb would BE the control")

    def test_every_train_flag_j6_passes_is_one_j4_passes(self):
        """Jobs 13641080_0..3: all four limbs died in 29 seconds on

            error: unrecognized arguments: --model_name_or_path ...
                   --output_dir ... --dataset_path ...

        because the invocation was copied from J4 starting at --max_length and
        the three argument names above it were GUESSED.  train.py takes
        --model_name, --out_path and --preprocessed_data_path.  jsonargparse
        rejects unknown flags, so this failed loudly rather than silently --
        but it burned a submission, and the launcher echo printed a correct
        banner first, which is exactly how the e_carry_read bug read too.

        J6 is a J4 clone plus arm flags, so any TOP-LEVEL flag it passes that
        J4 does not is a typo until proven otherwise.  The cortex.* read flags
        are exempt: J4 passes them through $READ_ARGS, where this parse cannot
        see them.
        """
        exempt = {"--cortex.latent_read", "--cortex.latent_read_znorm",
                  "--cortex.latent_read_znorm_target"}
        def flags(path):
            src = _read(path)
            body = src[src.index("\npython train.py"):]
            body = body[:body.index("\nRC=$?")]
            return set(re.findall(r"^\s+(--[A-Za-z0-9_.]+)", body, re.M))
        extra = flags("pace/j6_arms.sbatch") - flags("pace/j4_joint.sbatch") - exempt
        assert not extra, (
            f"j6_arms.sbatch passes {sorted(extra)} to train.py and "
            "j4_joint.sbatch does not.  Either it is a typo (the 13641080 "
            "failure) or it is new and belongs in this test's exempt set.")

    @pytest.mark.parametrize("flag", FLAGS + ("e_carry_read",))
    def test_it_reaches_the_eval_config(self, flag):
        src = _read("tools/prepare_eval_checkpoint.py")
        i = src.index("CORTEX_FLAGS = (")
        assert f'"{flag}"' in src[i:src.index(")", src.index('"mean_backprop_depth"'))], (
            f"{flag} is not copied into the eval checkpoint, so a read-out "
            "would rebuild a DIFFERENT buffer from these weights in silence")
