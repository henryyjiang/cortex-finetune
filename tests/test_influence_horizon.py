"""
The influence horizon, I(d).

This instrument had no test file until 2026-09-17, and it is the one every
planning doc defers to.  What that cost: `run_chain`'s GATED branch never looked
at `damage_at`, so on A3/A3' every damage mode was a no-op, I(d) came back
exactly 0.0 at every depth, and the reading would have been "the gated carry
does not matter anywhere".  A false negative that clean is indistinguishable
from a result.

So the first test here is the one that would have caught it, and it is written
as a PROPERTY of the instrument rather than of the buffer: damaging a write the
model still reads must move the loss, whatever buffer is underneath.

Run: /c/Users/henry/miniconda3/envs/cortex-retro/python.exe -m pytest tests/test_influence_horizon.py -q
"""
from __future__ import annotations

import os
import sys

import pytest
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "evals"))

from test_cortex_eval import VOCAB, _build_raven  # noqa: E402

from evals.eval_influence_horizon import (  # noqa: E402
    _patched_merge, capture_write, damage_pair, damage_write, intact_health,
    run_chain, scaled_noise,
)

NV, K, CL, EOS = 4, 16, 16, VOCAB - 1
NC, T = 6, 4
D_HIDDEN = None          # read off the built model


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
    return _build_raven(**common).eval()


def _chain_inputs(n=NC, seed=0):
    torch.manual_seed(seed)
    ids = torch.randint(0, VOCAB - 1, (CL * n + 1,))
    x, y = ids[:-1], ids[1:]
    mask = torch.ones_like(y, dtype=torch.float32)
    return (list(torch.chunk(x, n)), list(torch.chunk(y, n)),
            list(torch.chunk(mask, n)))


def _buf(model):
    return model.cortex.prefix


def _last_loss(rows):
    return rows[-1][0]


def _replay(model, damage_at=None, donor=None, donor_z=None, mode="donor",
            channel="both", gen=None, latent=True, gated=True, cached=None):
    m = cached if cached is not None else _model(latent=latent, gated=gated)
    xs, ys, ms = _chain_inputs()
    rows, writes = run_chain(m, _buf(m), xs, ys, ms, T, torch.device("cpu"),
                             7, damage_at, donor, mode, channel,
                             int(_buf(m).hidden_size), gen, donor_z)
    return m, rows, writes


def _donors(model, writes, at):
    """A donor shaped the way THAT buffer's damage path wants it.

    append takes the joined [B, W, 2D] row it slices off the state; gated takes
    the pre-merge (E, Z) pair, because its state is a post-gate mixture.
    """
    e, z = writes[at]
    from evals.eval_influence_horizon import _is_append
    if _is_append(_buf(model)):
        joined = e if z is None else torch.cat([e, z], dim=-1)
        return torch.randn_like(joined), None
    return torch.randn_like(e), (None if z is None else torch.randn_like(z))


class TestDamageActuallyLands:
    """The regression that motivated the file."""

    @pytest.mark.parametrize("gated", [True, False])
    def test_damaging_a_write_the_model_still_reads_moves_the_loss(self, gated):
        m = _model(gated=gated)
        _, intact, writes = _replay(m, cached=m)
        at = NC - 1 - 2                       # d = 2, well inside every buffer
        donor_e, donor_z = _donors(m, writes, at)
        _, damaged, _ = _replay(m, damage_at=at, donor=donor_e,
                                donor_z=donor_z, cached=m)
        moved = abs(_last_loss(damaged) - _last_loss(intact))
        assert moved > 1e-6, (
            f"damage at chunk {at} changed the final loss by {moved:.3e} on a "
            f"{'gated' if gated else 'append'} buffer.  Exactly zero is the "
            f"2026-09-17 bug: the damage never reached the merge.")

    @pytest.mark.parametrize("mode", ["donor", "zero", "random"])
    def test_every_mode_lands_on_the_gated_buffer(self, mode):
        """All three went through the same dead branch, so all three are pinned."""
        m = _model(gated=True)
        _, intact, writes = _replay(m, cached=m)
        at = NC - 1 - 2
        donor_e, donor_z = _donors(m, writes, at)
        gen = torch.Generator().manual_seed(3)
        _, damaged, _ = _replay(m, damage_at=at, donor=donor_e,
                                donor_z=donor_z, mode=mode, gen=gen, cached=m)
        assert abs(_last_loss(damaged) - _last_loss(intact)) > 1e-6

    def test_an_undamaged_replay_is_bit_identical(self):
        """The merge hook is installed on EVERY chunk to capture the pre-merge
        write.  If capturing perturbed the chain, every I(d) would be measuring
        the instrument."""
        m = _model()
        _, a, _ = _replay(m, cached=m)
        _, b, _ = _replay(m, cached=m)
        assert [r[0] for r in a] == [r[0] for r in b]


class TestTheCapturedWrite:

    def test_capture_returns_the_pre_merge_write_not_the_merged_ring(self):
        m = _model(gated=True)
        buf = _buf(m)
        xs, _, _ = _chain_inputs()
        e, z = capture_write(m, buf, xs[0], T, torch.device("cpu"))
        assert e.shape[1] == NV, "the write is W rows, the ring is K"
        assert z is not None and z.shape == e.shape

    def test_patched_merge_leaves_no_residue(self):
        m = _model()
        buf = _buf(m)
        assert "merge" not in buf.__dict__
        with _patched_merge(buf, lambda *a, **k: None):
            assert "merge" in buf.__dict__
        assert "merge" not in buf.__dict__, (
            "a leftover instance attribute would outlive the eval and silently "
            "re-route every later merge")

    def test_it_is_restored_even_when_the_forward_raises(self):
        m = _model()
        buf = _buf(m)
        with pytest.raises(RuntimeError):
            with _patched_merge(buf, lambda *a, **k: None):
                raise RuntimeError("boom")
        assert "merge" not in buf.__dict__


class TestScaledNoise:
    """The Z control is a SCALE argument, so the scale is what gets tested."""

    def test_it_matches_the_per_row_norm_of_what_it_replaces(self):
        ref = torch.randn(1, 5, 8) * torch.tensor([1.0, 10.0, 0.1, 3.0, 7.0]
                                                  ).view(1, 5, 1)
        out = scaled_noise(ref, torch.Generator().manual_seed(0))
        assert torch.allclose(out.norm(dim=-1), ref.norm(dim=-1), atol=1e-5)

    def test_per_row_not_per_tensor(self):
        """A global rescale would let the loudest row set every other row's
        norm, which is exactly the mistake that makes a scale control useless."""
        ref = torch.cat([torch.ones(1, 1, 8) * 100, torch.ones(1, 1, 8)], dim=1)
        out = scaled_noise(ref, torch.Generator().manual_seed(0))
        assert out[0, 1].norm() < out[0, 0].norm() / 10

    def test_the_same_generator_seed_reproduces_it(self):
        ref = torch.randn(1, 4, 8)
        a = scaled_noise(ref, torch.Generator().manual_seed(11))
        b = scaled_noise(ref, torch.Generator().manual_seed(11))
        assert torch.equal(a, b)

    def test_it_does_not_touch_the_global_rng(self):
        """Both chains reseed from --seed and must draw the SAME s0.  A
        randn_like here would consume the stream in the damaged chain only and
        break the pairing the instrument's power rests on."""
        ref = torch.ones(1, 2, 3)
        torch.manual_seed(5)
        untouched = torch.randn(4)
        torch.manual_seed(5)
        scaled_noise(ref, torch.Generator().manual_seed(9))
        after = torch.randn(4)
        assert torch.equal(untouched, after)


class TestChannelSplit:

    def test_damaging_e_leaves_z_exactly_as_it_was(self):
        e, z = torch.randn(1, 4, 8), torch.randn(1, 4, 8)
        de, dz = damage_pair(e, z, None, None, "zero", "e")
        assert torch.equal(dz, z) and de.abs().sum() == 0

    def test_damaging_z_leaves_e_exactly_as_it_was(self):
        e, z = torch.randn(1, 4, 8), torch.randn(1, 4, 8)
        de, dz = damage_pair(e, z, None, None, "zero", "z")
        assert torch.equal(de, e) and dz.abs().sum() == 0

    def test_both_damages_both(self):
        e, z = torch.randn(1, 4, 8), torch.randn(1, 4, 8)
        de, dz = damage_pair(e, z, None, None, "zero", "both")
        assert de.abs().sum() == 0 and dz.abs().sum() == 0

    def test_an_e_only_write_survives_a_z_request(self):
        """`z is None` is an E-only carry; asking for the Z channel must not
        invent one."""
        e = torch.randn(1, 4, 8)
        de, dz = damage_pair(e, None, None, None, "zero", "z")
        assert torch.equal(de, e) and dz is None

    def test_random_needs_a_generator_and_says_so(self):
        with pytest.raises(ValueError, match="generator"):
            damage_pair(torch.randn(1, 2, 3), None, None, None, "random", "e")
        with pytest.raises(ValueError, match="generator"):
            damage_write(torch.randn(1, 2, 3), None, "random", "both", 3)


class TestTheGeometryReachesTheGraft:
    """The first real launch of pace/eval_deciding.sbatch died here.

    `--model_name` loads the BASE dir's config, and ckpts/olmo-retrofit-cortex
    carries no cortex flags at all -- `use_memory` is '<absent>' there, which is
    why every p1_arms probe log opens with a block of "[cortex] OVERRIDE" lines.
    An eval that cannot force them builds a graft with no prefix buffer and
    exits 2.  Both deciding evals now take --set; these pin it, because the
    failure only shows up against a real 1.4B checkpoint that no unit test can
    load.
    """

    @pytest.mark.parametrize("mod", ["eval_influence_horizon", "eval_carry_2x2"])
    def test_the_eval_takes_set_and_parses_it(self, mod, monkeypatch):
        import importlib
        m = importlib.import_module(f"evals.{mod}")
        from model_utils import parse_config_overrides
        monkeypatch.setattr(
            sys, "argv",
            [mod, "--model_name", "x", "--data", "d",
             "--set", "use_memory=true", "--set", "accum_vecs=16"])
        args = m.parse_args()
        assert args.set == ["use_memory=true", "accum_vecs=16"]
        ov = parse_config_overrides(args.set)
        assert ov["use_memory"] is True and ov["accum_vecs"] == 16

    @pytest.mark.parametrize("mod", ["eval_influence_horizon", "eval_carry_2x2"])
    def test_it_actually_forwards_them_to_the_loader(self, mod):
        """Parsing the flag and dropping it would look identical from the CLI
        and identical in the log, right up to the empty buffer."""
        import ast
        src = open(os.path.join(REPO, "evals", f"{mod}.py"),
                   encoding="utf-8").read()
        calls = [n for n in ast.walk(ast.parse(src))
                 if isinstance(n, ast.Call)
                 and getattr(n.func, "id", None) == "load_checkpoint"]
        assert calls, f"{mod} does not call load_checkpoint at all"
        assert all(any(k.arg == "config_overrides" for k in c.keywords)
                   for c in calls), (
            f"{mod} parses --set but never hands it to load_checkpoint")


class TestTheIntactLossIsReported:
    """I(d) is a DIFFERENCE, and a difference of two chance-level losses still
    prints a tidy CI.  On 2026-09-16 the prelaunch walks and gate 4 scored these
    checkpoints at 11.4-11.97 nats -- ln(vocab) is 11.52 -- in the same job
    whose training loss was 2.78, and nothing in the pipeline looked at the
    level.  A ranking of three chance-level numbers became "the carry is
    anti-informative" and ordered a program.  So the level ships with the table.
    """

    def test_a_chance_level_model_is_named_as_one(self):
        vocab = 100278
        h = intact_health([11.67, 11.62, 11.55], vocab)
        assert h["at_chance"] is True
        assert h["margin_below_chance"] < 0
        assert h["n"] == 3

    def test_a_trained_model_is_not(self):
        h = intact_health([2.78, 2.81, 2.75], 100278)
        assert h["at_chance"] is False
        assert h["margin_below_chance"] > 8

    def test_no_samples_reports_nothing_rather_than_zero(self):
        """An empty run must not report 0.0 nats, which reads as a perfect
        model rather than as no measurement."""
        h = intact_health([], 100278)
        assert h["mean_nats"] is None and h["at_chance"] is None


class TestTheZChannelDamageLands:
    """P2.1 came back with donor-Z and random-Z BOTH at 0.0000 on both dual
    arms -- 24 numbers, all |.| <= 0.0014, while random-E on the same rows moved
    0.66-1.9 nats.  The pre-registered reading of that is "the instrument is
    blind at Z's scale", but there is a second explanation with the shape of
    reds 8 and 9: the z limb of the damage path never lands, and an exact zero
    reads as a result.  These pin the limb itself, so the P2.1 Z null can be
    quoted as a statement about SCALE rather than about plumbing.

    The model side is not in doubt: cortex_graft.latent_init substitutes the
    carried Z into s0 on every forward, and the no-grad asymmetry costs
    GRADIENT, not the read.  So if a channel is dead it is dead here.
    """

    @pytest.mark.parametrize("gated", [True, False])
    @pytest.mark.parametrize("mode", ["donor", "random"])
    def test_damaging_only_z_moves_the_loss(self, gated, mode):
        m = _model(latent=True, gated=gated)
        _, intact, writes = _replay(m, cached=m)
        at = NC - 1 - 2
        donor_e, donor_z = _donors(m, writes, at)
        gen = torch.Generator().manual_seed(11)
        _, damaged, _ = _replay(m, damage_at=at, donor=donor_e,
                                donor_z=donor_z, mode=mode, channel="z",
                                gen=gen, cached=m)
        moved = abs(_last_loss(damaged) - _last_loss(intact))
        assert moved > 1e-6, (
            f"--damage_channel z on a {'gated' if gated else 'append'} buffer "
            f"moved the loss by {moved:.3e} in mode {mode}.  Exactly zero here "
            f"means the z limb is a no-op and every P2.1 Z cell measured "
            f"nothing rather than measuring Z.")

    def test_damaging_only_e_leaves_the_carried_z_alone(self):
        """The other side of the 1x2: channel='e' must not perturb Z, or
        'donor e' and 'donor both' agreeing says nothing.  Gated only --
        damage_pair is the pre-merge path, and an append buffer damages the
        joined row through damage_write instead."""
        m = _model(latent=True, gated=True)
        _, _, writes = _replay(m, cached=m)
        at = NC - 1 - 2
        donor_e, donor_z = _donors(m, writes, at)
        e, z = writes[at]
        from evals.eval_influence_horizon import damage_pair
        assert z is not None, "the dual-channel fixture stopped carrying Z"
        out_e, out_z = damage_pair(e, z, donor_e, donor_z, "donor", "e")
        assert torch.equal(out_z, z)
        assert not torch.equal(out_e, e)
