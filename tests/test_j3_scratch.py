"""
J3, the in-loop latent scratchpad (findings doc, "Attempt 2", design J3;
cortex_memory/scratchpad.py): the properties the design rests on, pinned before
any of it trains.

  1. CAUSAL.  The within-chunk read must not leak the future.  The scratchpad
     is per position for exactly this reason, so the test perturbs a token and
     requires every earlier position's logits to be bit-identical.
  2. THE CARRY RE-ENTERS EVERY ITERATION.  The carried Z rows get gradient
     under a no-grad prefix -- the property s0 lacked and J3 is built around.
  3. THE WRITE TRAINS WITHOUT THE CARRY.  On the no-read limb (carry unread)
     the scratchpad's gates still receive gradient, from the within-chunk read.
  4. THE LIMBS DIFFER IN THE CARRY ALONE.  The donor roll, the nulls and the
     no-read flag act on the carried rows; the own row is never touched.
  5. THE GATE'S LR MULTIPLIER is a reparameterisation that moves the gate
     mult x slower under Adam and is bit-identical at 1.0.

Run: /c/Users/henry/miniconda3/envs/cortex-retro/python.exe -m pytest tests/test_j3_scratch.py -q
"""
from __future__ import annotations

import os
import re
import sys
from types import SimpleNamespace

import pytest
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from test_cortex_eval import VOCAB, _build_raven  # noqa: E402

from cortex_graft import CortexMemory, reset_cortex_graft_init  # noqa: E402
from cortex_memory.health import latent_runtime, training_diag  # noqa: E402
from cortex_memory.latent_read import LatentRead, apply_designed_init  # noqa: E402
from cortex_memory.scratchpad import LatentScratchpad  # noqa: E402

NV, K, CL, EOS, D = 4, 16, 16, VOCAB - 1, 64
T = 4
POOL = 2


def _flags(limb="real", **kw):
    f = dict(use_memory=True, memory_slots=0, accum_vecs=NV,
             latent_carry=True, eos_token_id=EOS,
             prefix_memory="gated", gate_slots=K, gate_route="ring",
             gate_init="zero", gate_fill="grow",
             latent_read="scratch", latent_encoding="scratch",
             latent_tok_pool=POOL, latent_s0_read=False, latent_read_heads=4,
             latent_read_znorm="rms", latent_read_znorm_target=3.0,
             latent_read_scramble=(limb == "donor"),
             latent_carry_read=(limb != "noread"))
    f.update(kw)
    return f


def _model(limb="real", **kw):
    torch.manual_seed(1234)
    return _build_raven(**_flags(limb, **kw)).train()


def _ids(n=2, batch=2, seed=0):
    torch.manual_seed(seed)
    return torch.randint(0, VOCAB - 1, (batch, CL * n))


def _chunks(n=2, batch=2, seed=0):
    return [c.contiguous() for c in torch.chunk(_ids(n, batch, seed), n, dim=1)]


def _fwd(m, xc, mc=None, steps=(0, T), seed=None, **kw):
    if seed is not None:
        torch.manual_seed(seed)      # initialize_state draws noise
    return m(xc, num_steps=torch.tensor(list(steps)), m_cross_in=mc,
             return_m_cross=True, **kw)


def _nz(g):
    return g is not None and float(g.abs().sum()) > 0


def _cortex(**kw):
    """The graft alone -- NOT through _build_raven, which turns a ValueError
    into a SKIP (a must-raise test would go green checking nothing)."""
    base = dict(n_embd=D, summary_init_token=EOS)
    base.update(_flags(**kw))
    for k in ("use_memory",):
        base.setdefault(k, True)
    return CortexMemory(SimpleNamespace(**base))


# ─── config validation ──────────────────────────────────────────────────────

class TestTheSwitchesRefuseNonsense:
    @pytest.mark.parametrize("limb", ["real", "donor", "noread"])
    def test_the_three_limbs_build(self, limb):
        c = _cortex(limb=limb)
        assert isinstance(c.latent_scratch, LatentScratchpad)
        assert isinstance(c.latent_reader, LatentRead)
        assert c.latent_carry_read == (limb != "noread")

    def test_defaults_reproduce_every_arm_on_record(self):
        c = _cortex(latent_read="xattn", latent_encoding="tokens",
                    latent_read_scramble=False, latent_carry_read=True)
        assert c.latent_scratch is None
        assert (c.latent_carry_read, c.latent_read_gate_lr_mult,
                c.latent_scratch_forget_bias) == (True, 1.0, 1.0)

    @pytest.mark.parametrize("kw, match", [
        (dict(latent_encoding="tokens"), "pass --cortex.latent_encoding scratch"),
        (dict(latent_read="xattn"), "only exists under"),
        (dict(latent_s0_read=True), "reads in-loop only"),
        (dict(latent_write_only=True), "latent_carry_read false"),
        (dict(latent_read="xattn", latent_encoding="tokens",
              latent_carry_read=False), "needs latent_read='scratch'"),
        (dict(latent_carry_read=False, latent_read_scramble=True),
         "duplicate of the no-read limb"),
        (dict(latent_read_gate_lr_mult=0.0), "gate_lr_mult must be"),
        (dict(latent_read_gate_lr_mult=float("nan")), "gate_lr_mult must be"),
        (dict(latent_read="refresh", latent_encoding="tokens",
              latent_read_znorm="none", latent_read_gate_lr_mult=0.5),
         "reparameterises LatentRead"),
        (dict(latent_read="xattn", latent_encoding="tokens",
              latent_scratch_forget_bias=2.0), "there is no scratchpad"),
    ])
    def test_bad_combinations_raise(self, kw, match):
        with pytest.raises(ValueError, match=match):
            _cortex(**kw)

    def test_znorm_is_allowed_on_the_scratch_read(self):
        assert _cortex().latent_read_znorm == "rms"

    def test_train_py_persists_the_j3_keys(self):
        src = open(os.path.join(REPO, "train.py"), encoding="utf-8").read()
        for k in ("latent_carry_read", "latent_read_gate_lr_mult",
                  "latent_scratch_forget_bias"):
            assert f'"{k}"' in src, f"{k} is not in train.py's persist list"
            assert f"{k}=" in src, f"{k} has no default in CLISettings.cortex"


# ─── the scratchpad module ──────────────────────────────────────────────────

class TestTheScratchpad:
    def test_zero_init_is_the_constant_ema(self):
        sp = LatentScratchpad(D)
        s1, s2 = torch.randn(2, 5, D), torch.randn(2, 5, D)
        m1 = sp(s1)
        m2 = sp(s2, m1)
        fg = torch.sigmoid(torch.tensor(1.0))
        assert torch.allclose(m1, 0.5 * s1, atol=1e-6)
        assert torch.allclose(m2, fg * m1 + 0.5 * s2, atol=1e-6)
        assert abs(sp.ema_forget - float(fg)) < 1e-6

    def test_the_designed_init_survives_a_kaiming_reset(self):
        sp = LatentScratchpad(D, forget_bias_init=2.0)
        for p in (sp.gate_proj_in, sp.gate_proj_mem):
            p.reset_parameters()
        tags = sp.apply_designed_init()
        assert float(sp.gate_proj_in.weight.abs().sum()) == 0.0
        assert float(sp.gate_proj_mem.weight.abs().sum()) == 0.0
        assert float(sp.forget_bias) == 2.0 and tags

    def test_a_state_from_another_layout_raises(self):
        with pytest.raises(ValueError, match="per packed position"):
            LatentScratchpad(D)(torch.randn(2, 5, D), torch.randn(2, 6, D))


# ─── the read with an own key ───────────────────────────────────────────────

class TestTheReadWithAnOwnKey:
    def _reader(self, **kw):
        torch.manual_seed(0)
        return LatentRead(D, n_heads=4, **kw)

    def test_no_live_key_injects_exactly_zero(self):
        r = self._reader()
        x = torch.randn(2, 7, D)
        out = r(x, None, own=torch.zeros_like(x))
        assert torch.equal(out, torch.zeros_like(x))

    def test_own_alone_is_a_gated_projection_of_the_own_row(self):
        r = self._reader()
        x, own = torch.randn(2, 7, D), torch.randn(2, 7, D)
        want = r.gate * r.out_proj(r.v_proj(own))
        assert torch.allclose(r(x, None, own=own), want, atol=1e-5)
        assert abs(r.own_share - 1.0) < 1e-6

    def test_the_eos_mask_removes_the_carry_not_the_read(self):
        r = self._reader()
        x, own = torch.randn(1, 6, D), torch.randn(1, 6, D)
        z = torch.randn(1, 3, D)
        mask = torch.ones(1, 6, 1)
        mask[:, 4:] = 0.0                              # past the first EOS
        both = r(x, z, read_mask=mask, own=own)
        own_only = r(x, None, own=own)
        assert torch.allclose(both[:, 4:], own_only[:, 4:], atol=1e-5)
        assert not torch.allclose(both[:, :4], own_only[:, :4], atol=1e-4)

    def test_the_carry_shares_the_softmax(self):
        r = self._reader()
        x, own = torch.randn(1, 6, D), torch.randn(1, 6, D)
        r(x, torch.randn(1, 3, D), own=own)
        assert 0.0 < r.own_share < 1.0


# ─── the gate's LR multiplier ───────────────────────────────────────────────

class TestTheGateLrMult:
    def test_one_is_the_j1_gate(self):
        r = LatentRead(D, n_heads=4)
        assert r.effective_gate() is r.gate and r.gate_value == pytest.approx(0.1)

    def test_the_effective_gate_starts_at_gate_init(self):
        r = LatentRead(D, n_heads=4, gate_lr_mult=0.01)
        assert r.gate_value == pytest.approx(0.1)
        assert float(r.gate) == pytest.approx(10.0)

    def test_adam_moves_it_mult_times_slower(self):
        """The J1 reading: Adam steps a sign-stable scalar at ~lr per update.
        Reparameterised by 0.01, the SAME optimizer moves the gate 100x less."""
        moved = {}
        for mult in (1.0, 0.01):
            r = LatentRead(D, n_heads=4, gate_lr_mult=mult)
            opt = torch.optim.Adam([r.gate], lr=5e-4)
            for _ in range(10):
                opt.zero_grad()
                r.effective_gate().sum().backward()     # d/dgate = +1: close it
                opt.step()
            moved[mult] = 0.1 - r.gate_value
        assert moved[1.0] == pytest.approx(10 * 5e-4, rel=0.05)
        assert moved[0.01] == pytest.approx(10 * 5e-6, rel=0.05)

    def test_the_designed_init_restores_the_stored_scale(self):
        r = LatentRead(D, n_heads=4, gate_lr_mult=0.01)
        torch.nn.init.constant_(r.gate, 3.0)
        apply_designed_init(r)
        assert r.gate_value == pytest.approx(0.1)


# ─── the model: causality ───────────────────────────────────────────────────

class TestCausality:
    """Perturb token j; every position < j must be BIT-identical."""

    @pytest.mark.parametrize("chunk", [1, 2])
    @pytest.mark.parametrize("j", [3, 9, 15])
    def test_no_earlier_position_sees_a_later_token(self, chunk, j):
        m = _model().eval()
        x1, x2 = _chunks()
        with torch.no_grad():
            mc = _fwd(m, x1, seed=7)["m_cross"] if chunk == 2 else None
            xa = x2 if chunk == 2 else x1
            xb = xa.clone()
            xb[:, j] = (xb[:, j] + 1) % (VOCAB - 1)
            la = _fwd(m, xa, mc=mc, seed=11)["logits"]
            lb = _fwd(m, xb, mc=mc, seed=11)["logits"]
        assert torch.equal(la[:, :j], lb[:, :j])
        assert not torch.equal(la[:, j:], lb[:, j:])     # the perturbation lands

    def test_the_scratchpad_read_actually_ran(self):
        m = _model().eval()
        with torch.no_grad():
            _fwd(m, _chunks()[0], seed=3)
        lat = latent_runtime(m.cortex)
        assert lat["read_calls"] == T - 1           # iterations 2..T read m_{t-1}
        assert lat["own_share"] == pytest.approx(1.0)   # chunk 1: nothing carried
        assert lat["read_ratio"] and lat["read_ratio"] > 0


# ─── the model: gradient paths ──────────────────────────────────────────────

def _carry_grads(limb, steps=(0, T)):
    """Chunk 2's loss with E's half of the carry DETACHED: returns the grad at
    the carried Z rows and the grads of the scratchpad's gate weights."""
    m = _model(limb=limb)
    x1, x2 = _chunks()
    mc = _fwd(m, x1)["m_cross"]
    z1 = mc[..., D:]
    z1.retain_grad()
    mc = torch.cat([mc[..., :D].detach(), z1], dim=-1)
    for p in m.parameters():
        p.grad = None
    _fwd(m, x2, mc=mc, labels=x2, steps=steps)["loss"].backward()
    sp = m.cortex.latent_scratch
    return z1.grad, sp.gate_proj_in.weight.grad, z1.grad_fn is not None


class TestGradientPaths:
    @pytest.mark.parametrize("steps", [(0, T), (1, T - 1), (2, T - 2)])
    def test_the_carry_gets_gradient_under_a_no_grad_prefix(self, steps):
        # s0's read was EXACTLY zero at (1, T-1): J3's re-enters every step.
        gz, _, live = _carry_grads("real", steps)
        assert live and _nz(gz)

    def test_the_no_read_limb_puts_none_on_the_carry(self):
        gz, _, _ = _carry_grads("noread")
        assert not _nz(gz)

    @pytest.mark.parametrize("limb", ["real", "noread"])
    def test_the_write_trains_from_the_within_chunk_read(self, limb):
        _, gw, _ = _carry_grads(limb)
        assert _nz(gw)

    def test_the_donor_limb_trains_the_other_documents_write(self):
        gz, _, _ = _carry_grads("donor")
        assert _nz(gz)


# ─── the model: what is carried, and the limbs ──────────────────────────────

def _capture_writes(m):
    buf, real = m.cortex.prefix, m.cortex.prefix.merge
    got = []

    def wrapped(state, new_vecs, new_latent=None):
        got.append(new_latent)
        return real(state, new_vecs, new_latent)
    buf.merge = wrapped
    return got


class TestTheCarry:
    def test_the_carry_is_the_pooled_scratchpad(self):
        m = _model().eval()
        got = _capture_writes(m)
        with torch.no_grad():
            _fwd(m, _chunks()[0], seed=3)
        sc = m.cortex._scratch
        L = NV * POOL
        tok = sc[:, -NV - L:-NV]
        want = tok.reshape(tok.shape[0], NV, POOL, D).mean(dim=2)
        assert torch.allclose(got[-1], want, atol=1e-6)
        assert not torch.allclose(got[-1], torch.zeros_like(want))

    def test_z_off_keeps_the_within_chunk_read(self):
        m = _model().eval()
        x1, x2 = _chunks()
        with torch.no_grad():
            mc = _fwd(m, x1, seed=3)["m_cross"]
            m.cortex.latent_read_null = ("off",)
            _fwd(m, x2, mc=mc, seed=4)
            m.cortex.latent_read_null = None
        lat = latent_runtime(m.cortex)
        assert lat["read_calls"] == T - 1 and lat["own_share"] == pytest.approx(1.0)

    def test_the_no_read_limb_ignores_the_z_half(self):
        """eval_carry_2x2's veto 6 for J3: E1Z0 must equal E1Z1 on the no-read
        limb, i.e. nothing in the Z half reaches the logits."""
        m = _model(limb="noread").eval()
        x1, x2 = _chunks()
        with torch.no_grad():
            mc = _fwd(m, x1, seed=3)["m_cross"]
            assert float(mc[..., D:].abs().sum()) > 0      # written at full width
            a = _fwd(m, x2, mc=mc, seed=4)["logits"]
            mc2 = torch.cat([mc[..., :D], torch.randn_like(mc[..., D:])], -1)
            b = _fwd(m, x2, mc=mc2, seed=4)["logits"]
        assert torch.equal(a, b)

    def test_the_real_limb_reads_the_z_half(self):
        m = _model().eval()
        x1, x2 = _chunks()
        with torch.no_grad():
            mc = _fwd(m, x1, seed=3)["m_cross"]
            a = _fwd(m, x2, mc=mc, seed=4)["logits"]
            mc2 = torch.cat([mc[..., :D], torch.randn_like(mc[..., D:])], -1)
            b = _fwd(m, x2, mc=mc2, seed=4)["logits"]
        assert not torch.equal(a, b)

    def test_the_donor_roll_never_touches_the_own_row(self):
        """Chunk 1 carries nothing, so the only read is the own row: real and
        donor limbs (same weights, same seed) must agree exactly there."""
        real, donor = _model("real").eval(), _model("donor").eval()
        x1 = _chunks()[0]
        with torch.no_grad():
            a = _fwd(real, x1, seed=5)["logits"]
            b = _fwd(donor, x1, seed=5)["logits"]
        assert torch.equal(a, b)


# ─── the post_init clobber ──────────────────────────────────────────────────

class TestTheInitSurvivesPostInit:
    def test_reset_restores_both_designed_inits(self):
        m = _model(latent_read_gate_lr_mult=0.01)
        c = m.cortex
        torch.nn.init.normal_(c.latent_scratch.gate_proj_in.weight)
        torch.nn.init.constant_(c.latent_reader.gate, 7.0)
        reset_cortex_graft_init(m)
        assert float(c.latent_scratch.gate_proj_in.weight.abs().sum()) == 0.0
        assert c.latent_reader.gate_value == pytest.approx(0.1)


# ─── the diag row ───────────────────────────────────────────────────────────

class TestTheDiagRow:
    def test_the_row_carries_the_read_strength(self):
        m = _model(latent_read_gate_lr_mult=0.01).eval()
        x1, x2 = _chunks()
        with torch.no_grad():
            mc = _fwd(m, x1, seed=3)["m_cross"]
            _fwd(m, x2, mc=mc, seed=4)
        row = training_diag(m.cortex, mc)
        for k in ("z_read_gate", "z_read_ratio", "z_own_share",
                  "z_scratch_row_norm", "z_gate_lr_mult", "z_carry_read"):
            assert k in row, k
        assert row["z_read_gate"] == pytest.approx(0.1)
        assert 0.0 < row["z_own_share"] < 1.0


# ─── the training step: train.py's chunk chain, with window_backward ────────

sys.path.insert(0, os.path.join(REPO, "tools"))
from smoke_geometry_oom import micro_step  # noqa: E402

GNC = 8                                  # cross_chunks: two 4-chunk windows
CPU_AMP = {"device_type": "cpu", "dtype": torch.bfloat16, "enabled": False}


def _batch(batch=2):
    torch.manual_seed(0)
    ids = torch.randint(0, VOCAB - 1, (batch, CL * GNC + 1))
    for gi in range(1, GNC):
        ids[:, gi * CL + CL // (gi + 1)] = EOS      # EOS inside chunks: the mask path
    return ids[:, :-1], ids[:, 1:]


def _step(model, window, grad_chunks=4):
    x, y = _batch()
    torch.manual_seed(7)
    return micro_step(model, x, y, EOS, GNC, torch.tensor([1, T - 1]),
                      grad_chunks, NV, 2, CPU_AMP, True,
                      write_once=False, window_backward=window)


def _grads(model):
    return {n: p.grad.detach().clone() for n, p in model.named_parameters()
            if p.grad is not None}


class TestTheTrainingStep:
    """The mirror of train.py's cortex_fwd_bwd (tools/smoke_geometry_oom.py),
    at a no-grad prefix of 1 and with EOS inside chunks -- J3's real path."""

    @pytest.mark.parametrize("limb", ["real", "donor", "noread"])
    def test_window_backward_is_exact_on_j3(self, limb):
        a = _model(limb)
        loss_a, _ = _step(a, window=False)
        b = _model(limb)
        loss_b, _ = _step(b, window=True)
        assert abs(loss_a - loss_b) < 1e-5 * max(1.0, abs(loss_a))
        ga, gb = _grads(a), _grads(b)
        assert ga.keys() == gb.keys() and ga
        for n in ga:
            scale = float(ga[n].abs().max()) or 1.0
            assert float((ga[n] - gb[n]).abs().max()) <= 1e-5 * scale + 1e-9, n
        assert all(bool(torch.isfinite(g).all()) for g in gb.values())

    @pytest.mark.parametrize("limb", ["real", "noread"])
    def test_every_j3_parameter_trains(self, limb):
        m = _model(limb)
        _step(m, window=True)
        c = m.cortex
        for name, p in (("scratch.gate_proj_in", c.latent_scratch.gate_proj_in.weight),
                        ("scratch.gate_proj_mem", c.latent_scratch.gate_proj_mem.weight),
                        ("reader.gate", c.latent_reader.gate),
                        ("reader.out_proj", c.latent_reader.out_proj.weight)):
            assert _nz(p.grad), f"{limb}: {name} got no gradient"

    def test_the_carry_puts_read_pressure_on_the_z_ring_only_when_read(self):
        """The ring's Z gate is trained ONLY through the carried rows' read, so
        it must get gradient on the real limb and none on the no-read limb --
        the no-read limb is not a second treatment arm."""
        real, noread = _model("real"), _model("noread")
        _step(real, window=True)
        _step(noread, window=True)
        assert _nz(real.cortex.prefix.gate_proj_in_z.weight.grad)
        assert not _nz(noread.cortex.prefix.gate_proj_in_z.weight.grad)


# ─── the launchers ──────────────────────────────────────────────────────────

def _src(path):
    with open(os.path.join(REPO, path), encoding="utf-8") as fh:
        return fh.read()


class TestTheLaunchers:
    def test_each_limb_maps_to_its_flags(self):
        s = _src("pace/j3_joint.sbatch")
        assert "--cortex.latent_read scratch" in s
        assert "--cortex.latent_encoding $ENCODING" in s and "ENCODING=scratch" in s
        assert "latent_read_scramble true --cortex.latent_carry_read true" in s   # donor
        assert "latent_read_scramble false --cortex.latent_carry_read false" in s  # noread
        assert "--cortex.latent_s0_read false" in s

    def test_it_is_j1s_recipe(self):
        j1, j3 = _src("pace/j1_joint.sbatch"), _src("pace/j3_joint.sbatch")
        for line in ("retro-b2-heal/checkpoint_91552_w16", "--seed 74",
                     "--cortex.memory_lr 5e-4", "--muon.lr 0.001",
                     "CELL_STEPS=${CELL_STEPS:-2000}", "MICRO_BS=${MICRO_BS:-2}",
                     "WINDOW_BACKWARD=${WINDOW_BACKWARD:-1}",
                     "data/j1_pg19fw50_carry20_len4096",
                     "CARRY_GRAD_CHUNKS=${CARRY_GRAD_CHUNKS:-4}"):
            assert line in j1 and line in j3, line

    def test_the_gate_lr_mult_is_passed_and_recorded(self):
        s = _src("pace/j3_joint.sbatch")
        # 0.1: the gate's fastest collapse takes about the whole 2,000-update
        # run (the DECISIONS block's rule); 0.01 froze it and capped the read.
        assert "GATE_LR_MULT=${GATE_LR_MULT:-0.1}" in s
        assert "--cortex.latent_read_gate_lr_mult $GATE_LR_MULT" in s
        assert "--cortex.diag_interval" in s

    def test_the_readout_default_matches_the_training_default(self):
        tr = re.search(r"GATE_LR_MULT=\$\{GATE_LR_MULT:-([0-9.]+)\}",
                       _src("pace/j3_joint.sbatch")).group(1)
        ro = re.search(r"GATE_LR_MULT=\$\{GATE_LR_MULT:-([0-9.]+)\}",
                       _src("pace/j1_readout.sbatch")).group(1)
        assert tr == ro
        assert "z_gate_lr_mult" in _src("pace/j1_readout.sbatch")   # the guard

    def test_the_readout_builds_j3s_limbs(self):
        s = _src("pace/j1_readout.sbatch")
        assert "--set latent_read=scratch" in s
        assert "--set latent_carry_read=false" in s
        assert "RUN=${EXPERIMENT}-a3z-${ENCODING}${TAG}-${LIMB}" in s
        assert "eval_results/${EXPERIMENT}_readout" in s

    def test_the_verdict_takes_the_experiment(self):
        s = _src("pace/j1_verdict.sbatch")
        assert '--experiment "$EXPERIMENT"' in s and "j3) ENCODING=${ENCODING:-scratch}" in s

    def test_the_price_measures_j3(self):
        s = _src("pace/j3_price_oom.sbatch")
        assert "--set latent_read=scratch" in s and "--window_backward" in s
        assert "REQUIRE_FIT" in s and "exit 3" in s

    @pytest.mark.parametrize("path", ["pace/j3_joint.sbatch", "pace/j3_price_oom.sbatch"])
    def test_no_var_assignment_prefix_on_the_python_line(self, path):
        for line in _src(path).splitlines():
            if line.strip().startswith("python "):
                assert "=" not in line.split("python")[0]


class TestTheScorerReadsJ3:
    def test_run_names(self):
        sys.path.insert(0, os.path.join(REPO, "evals"))
        import score_j1
        assert score_j1.run_name("real", "tokens") == "j1-a3z-tokens-real"
        assert (score_j1.run_name("noread", "scratch", "0.25", "j3")
                == "j3-a3z-scratch-edrop0.25-noread")
        with pytest.raises(ValueError):
            score_j1.run_name("real", "scratch", "", "j2")

    def test_the_read_trajectory_is_reported(self, tmp_path):
        sys.path.insert(0, os.path.join(REPO, "evals"))
        import json
        import score_j1
        p = tmp_path / "cortex_diag.jsonl"
        # 80 rows (J1/J3's 2,000 updates at diag_interval 25), falling 0.02 ->
        # 0.005 over the run, with one noisy spike in each end window.
        rows = [{"step": 91575 + 25 * i, "z_read_ratio": 0.02 - 0.015 * i / 79}
                for i in range(80)]
        rows[1]["z_read_ratio"] = 0.05
        rows[-2]["z_read_ratio"] = 0.05
        p.write_text("\n".join(json.dumps(r) for r in rows))
        rt = score_j1.read_trajectory(str(p))
        assert rt["window"] == score_j1.READ_WINDOW == 4
        head = [r["z_read_ratio"] for r in rows[:4]]
        tail = [r["z_read_ratio"] for r in rows[-4:]]
        assert rt["first"] == pytest.approx(sum(head) / 4)
        assert rt["last"] == pytest.approx(sum(tail) / 4)
        assert rt["collapsed"] == (rt["last_over_first"] < score_j1.READ_COLLAPSE_FRAC)
        assert score_j1.read_trajectory(str(tmp_path / "missing.jsonl")) is None

    def test_a_steady_read_is_not_collapsed_and_a_dead_one_is(self, tmp_path):
        sys.path.insert(0, os.path.join(REPO, "evals"))
        import json
        import score_j1
        for vals, want in (([0.02] * 40, False), ([0.02] * 20 + [0.001] * 20, True)):
            p = tmp_path / f"d{want}.jsonl"
            p.write_text("\n".join(json.dumps({"step": i, "z_read_ratio": v})
                                   for i, v in enumerate(vals)))
            assert score_j1.read_trajectory(str(p))["collapsed"] is want

    def test_a_positive_beside_a_dead_read_is_an_audit(self):
        sys.path.insert(0, os.path.join(REPO, "evals"))
        import score_j1
        run = "j3-a3z-scratch-edrop0.25-real"
        live_gate = {run: {"collapsed": False, "last": 0.09}}
        dead_read = {run: {"collapsed": True, "last_over_first": 0.1}}
        live_read = {run: {"collapsed": False, "last_over_first": 0.9}}
        assert score_j1.audit_reasons([run], live_gate, dead_read)
        assert not score_j1.audit_reasons([run], live_gate, live_read)
        assert not score_j1.audit_reasons([], live_gate, dead_read)   # no positive: descriptive
        assert score_j1.audit_reasons(
            [run], {run: {"collapsed": True, "last": 0.01}}, live_read)
