"""
The 8-chunk dual-channel walk, and the health statistics it is built on.

The walk is a printed table rather than a pass/fail, which makes it exactly the
kind of instrument that can rot without anyone noticing -- a column that always
prints "-" looks like a quiet channel, not like a broken probe.  So these tests
pin what each column MEASURES on a toy model whose answers are known by
construction: the ring writes rows congruent to the chunk index, the Z read
delivers the previous chunk's Z, an E-only model has no Z column at all.

Run: /c/Users/henry/miniconda3/envs/cortex-retro/python.exe -m pytest tests/test_dual_channel_walk.py -q
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

from cortex_memory.buffers import PrefixGatedBuffer  # noqa: E402
from cortex_memory.health import (  # noqa: E402
    carry_health, expected_ring_rows, gate_param_health, rank_stats,
    read_live_fraction, split_carry,
)
from evals.diag_dual_channel_walk import WATCHED, print_walk, walk  # noqa: E402

NV, K, CL, EOS = 4, 16, 16, VOCAB - 1
NC = 8                      # K/W = 4, so the chain completes two laps
T = 4


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


def _chunks(n=NC, batch=2):
    torch.manual_seed(0)
    ids = torch.randint(0, VOCAB - 1, (batch, CL * n))
    return [c.contiguous() for c in torch.chunk(ids, n, dim=1)]


def _walk(m, **kw):
    kw.setdefault("num_steps", torch.tensor([0, T]))
    return walk(m, m.cortex, _chunks(), **kw)


class TestTheTableSaysWhatHappened:

    def test_the_ring_writes_the_rows_the_rule_predicts(self):
        """rows_touched is derived from the merge's INPUT vs OUTPUT, and
        rows_expected from the ring rule in cortex_memory.health -- two
        independent derivations, which is the only way this column is evidence.
        """
        rec = _walk(_model(), backward=False)
        checked = [r for r in rec["rows"] if r.get("rows_match") is not None]
        assert len(checked) == NC - K // NV, "the second lap must be checkable"
        assert all(r["rows_match"] for r in checked)
        assert all(len(r["rows_touched"]) == NV for r in checked)

    def test_the_first_lap_grows_and_preserves(self):
        """fill='grow' makes lap 1 bit-identical to an append buffer, which is
        what makes an accum arm and a gated arm diverge at exactly eviction."""
        rec = _walk(_model(), backward=False)
        grew = [r["rows_out"] for r in rec["rows"]]
        assert grew == [NV, 2 * NV, 3 * NV, K, K, K, K, K]

    def test_the_carry_is_two_channels_wide_and_the_rows_do_not_move(self):
        z = _walk(_model(latent=True), backward=False)
        e = _walk(_model(latent=False), backward=False)
        assert all(r["width_is_2d"] for r in z["rows"])
        assert not any(r["width_is_2d"] for r in e["rows"])
        assert ([r["rows_out"] for r in z["rows"]]
                == [r["rows_out"] for r in e["rows"]])

    def test_an_e_only_model_reports_no_z_column_rather_than_a_zero(self):
        """A dead channel and an absent channel must not print the same thing."""
        rec = _walk(_model(latent=False), backward=False)
        assert all(r["z_write_norm"] is None for r in rec["rows"])
        assert all(r["z_over_s0"] is None for r in rec["rows"])
        assert rec["final_carry"]["channels"] == 1
        assert not rec["latent"]["latent_carry"]

    def test_the_z_read_delivers_the_previous_chunks_z(self):
        """THE check the design rests on: the read is a substitution into a
        field that already exists, so if it silently did not fire, Z is a write
        nothing consumes and no loss would notice."""
        rec = _walk(_model(), backward=False)
        errs = [r["z_read_max_abs_err"] for r in rec["rows"]
                if "z_read_max_abs_err" in r]
        assert errs, "no chunk had a carried Z to read"
        assert max(errs) < 1e-5, errs

    def test_read_live_tracks_the_no_grad_split_and_not_the_config(self):
        live = _walk(_model(), num_steps=torch.tensor([0, T]), backward=False)
        dead = _walk(_model(), num_steps=torch.tensor([1, T - 1]), backward=False)
        assert all(r["read_live"] for r in live["rows"][1:])
        assert not any(r["read_live"] for r in dead["rows"][1:])

    def test_the_tape_and_the_depth_map_agree_with_the_loop_length(self):
        rec = _walk(_model(), backward=False)
        for r in rec["rows"]:
            assert r["tape_len"] == T, r
            assert len(r["depths"]) == NV
            assert all(1 <= d <= T - 1 for d in r["depths"]), r["depths"]

    def test_write_norms_are_reported_as_a_ratio_to_s0(self):
        """The ratio is the quantity P0.1 measured and the one that decides
        latent_renorm -- an absolute norm cannot be read against it."""
        rec = _walk(_model(), backward=False)
        r = rec["rows"][-1]
        assert r["s0_scale"] > 0
        assert r["e_over_s0"] == pytest.approx(r["e_write_norm"] / r["s0_scale"])
        assert r["z_over_s0"] == pytest.approx(r["z_write_norm"] / r["s0_scale"])


class TestTheClosingBlock:

    def test_every_watched_parameter_has_a_live_gradient_path(self):
        """A grad of None is 'no path', not 'small'.  gate_proj_mem only
        receives gradient through a chain of >= 3 chunks, which is why the walk
        backwards the SUMMED chain loss rather than one chunk's."""
        rec = _walk(_model())
        assert rec["grad_none"] == [], rec["grad_none"]
        assert rec["grad_zero"] == [], rec["grad_zero"]
        for name in ("prefix.summary_emb", "prefix.gate_proj_in.weight",
                     "prefix.gate_proj_mem.weight",
                     "prefix.gate_proj_in_z.weight",
                     "prefix.gate_proj_mem_z.weight"):
            assert rec["grads"][name] > 0.0, name

    def test_the_z_gate_loses_its_read_gradient_after_one_no_grad_step(self):
        """The E/Z read asymmetry, on the real loop rather than in a comment.

        E's gate keeps a gradient at every split because E re-enters through
        input_embeds on every iteration; Z enters once, at s0, behind the
        no-grad prefix.
        """
        live = _walk(_model(), num_steps=torch.tensor([0, T]))
        dead = _walk(_model(), num_steps=torch.tensor([1, T - 1]))
        assert live["grads"]["prefix.gate_proj_in_z.weight"] > 0.0
        assert dead["grads"]["prefix.gate_proj_in.weight"] > 0.0
        assert (dead["grads"]["prefix.gate_proj_in_z.weight"] == 0.0
                or dead["grads"]["prefix.gate_proj_in_z.weight"] is None), (
            "a no-grad prefix must cut Z's READ gradient; a non-zero here means "
            "the substitution found a second path into the loss")

    def test_the_final_carry_reports_each_channel_separately(self):
        rec = _walk(_model(), backward=False)
        c = rec["final_carry"]
        assert c["channels"] == 2 and c["rows"] == K
        # E rows are post-ln_f states, Z rows are trajectory deltas: pooling the
        # two would report a number about E with a rounding error named Z.
        assert c["e_row_norm"] != c["z_row_norm"]
        for key in ("e_centred_cosine", "e_eff_rank_pr", "z_centred_cosine",
                    "z_eff_rank_pr"):
            assert key in c

    def test_the_gate_is_reported_at_its_init_until_it_trains(self):
        rec = _walk(_model(), backward=False)
        gp = rec["gate_params"]
        assert gp["fg_at_bias"] == pytest.approx(0.7310585786300049, abs=1e-6)
        assert gp["ig_at_bias"] == pytest.approx(0.5, abs=1e-6)
        assert gp["gate_left_init"] is False
        assert gp["gate_z_left_init"] is False


class TestTheHealthStatistics:

    def test_split_carry_does_not_consult_the_buffer(self):
        """The instrument must be able to report a width the model disagrees
        with -- that is what a checkpoint/config mismatch looks like."""
        D = 8
        e, z = split_carry(torch.randn(1, 4, 2 * D), D)
        assert e.shape[-1] == z.shape[-1] == D
        e2, z2 = split_carry(torch.randn(1, 4, D), D)
        assert z2 is None
        with pytest.raises(ValueError, match="neither D"):
            split_carry(torch.randn(1, 4, 3 * D + 1), D)

    def test_rank_stats_is_the_same_statistic_the_geometry_probe_uses(self):
        from evals.diag_gate_geometry import rank_stats as probe_rank_stats
        m = torch.randn(2, 8, 16)
        assert probe_rank_stats(m) == rank_stats(m)

    def test_rank_stats_centres_first(self):
        """K vectors sharing a large common component read as cosine ~1.0
        uncentred however much independent structure sits on top."""
        base = torch.randn(1, 8, 32)
        shifted = base + 50.0 * torch.randn(1, 1, 32)
        assert rank_stats(shifted)["centred_cosine"] == pytest.approx(
            rank_stats(base)["centred_cosine"], abs=1e-4)

    def test_unwritten_z_rows_are_counted_not_averaged_away(self):
        D = 8
        st = torch.randn(1, 6, 2 * D)
        st[:, 3:, D:] = 0.0
        assert carry_health(st, D)["z_zero_rows"] == 3

    def test_expected_ring_rows_matches_the_buffers_own_addressing(self):
        buf = PrefixGatedBuffer(8, n_vec=4, n_slots=16, route="ring")
        for j in range(4):
            rows = buf.depth_rows(j)
            # write index j lands at rows congruent to j (mod W) on every lap
            for c in range(8):
                assert expected_ring_rows(c, 4, 16)[j] in rows


class TestThePrintedTable:

    def test_it_prints_and_stays_ascii(self):
        """The Windows console is cp1252; a non-ASCII character in a probe's
        output crashes the probe rather than the terminal."""
        rec = _walk(_model())
        buf = io.StringIO()
        print_walk(rec, out=buf)
        text = buf.getvalue()
        text.encode("ascii")
        assert "DUAL-CHANNEL WALK" in text
        assert "not about whether Z helps" in text

    def test_a_missing_number_prints_as_a_dash_and_not_as_zero(self):
        rec = _walk(_model(latent=False), backward=False)
        buf = io.StringIO()
        print_walk(rec, out=buf)
        body = [ln for ln in buf.getvalue().splitlines()
                if ln.strip().startswith("0 ")]
        assert body and "-" in body[0]
