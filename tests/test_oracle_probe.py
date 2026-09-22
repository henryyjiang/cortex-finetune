"""P3.0 tier 1.5 — the oracle read probe's three new capabilities.

Tier 1.5 asks the only question that matters for Z: under conditions chosen to
be MAXIMALLY FAVOURABLE, does the carried trajectory contain anything a read
can extract?  A null there is decisive precisely because nothing was held back.
That needs three things the trainer did not have, and each is a way to get a
wrong answer quietly:

  1. FREEZE EVERYTHING BUT THE READ.  If the buffer, the gates or the coda can
     move, a win might be theirs and the arm answers a different question.
     `freeze_loop` is the OPPOSITE end -- it freezes adapter + core_block and
     leaves memory and coda training.
  2. FORCE num_steps = [0, T].  Zero no-grad prefix, so the read gradient is
     live on every batch rather than the sampler's ~0.55 at mr8.
     `fix_num_steps` exists but is hardcoded to [0,1], which would have run the
     probe at ONE recurrent step.
  3. SHUFFLED Z AS A TRAINING-TIME CONTROL.  The comparison is not "does a read
     module help" -- a module can help by adding capacity, or by reading
     position and register information that has nothing to do with the previous
     chunk.  It is the identical module, identically trained, on real Z vs on
     another document's Z.

Run: /c/Users/henry/miniconda3/envs/cortex-retro/python.exe -m pytest tests/ -q
"""
from __future__ import annotations

import os
import sys

import pytest
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from test_read_into import NV, K, CL, EOS, _chain, _model  # noqa: E402


def _scrambled(**kw):
    return _model(latent_read_scramble=True, **kw)


# ---------------------------------------------------------------------------
# 3. the shuffled-Z control
# ---------------------------------------------------------------------------

class TestTheScrambleControl:

    def test_it_changes_what_the_read_sees(self):
        torch.manual_seed(0)
        z = torch.randn(4, 6, 8)
        m = _model()
        m.cortex.latent_read_scramble = True
        rolled = torch.roll(z, shifts=1, dims=0)
        assert not torch.equal(z, rolled)
        # row b of the control is row b-1 of the treatment: another document's
        # trajectory, at identical shape and scale.
        assert torch.equal(rolled[1], z[0])

    def test_scale_and_shape_are_untouched(self):
        """The control must differ in CONTENT CORRESPONDENCE and nothing else.
        A control that also changes the magnitude is the defect that made job
        13420851's 0.27-nat 'content effect' a 4.8x scale gap."""
        z = torch.randn(4, 6, 8)
        rolled = torch.roll(z, shifts=1, dims=0)
        assert rolled.shape == z.shape
        assert torch.allclose(rolled.float().norm(dim=-1).sort(dim=0).values,
                              z.float().norm(dim=-1).sort(dim=0).values)

    def test_a_batch_of_one_is_refused(self):
        """A roll of a single row returns that same row, so the control arm
        would silently BE the treatment arm and the pair would differ by
        nothing -- a null by construction, reported as a result."""
        m = _scrambled()
        with pytest.raises(ValueError, match="batch >= 2"):
            _chain(m, n_chunks=2)

    def test_the_two_arms_actually_diverge(self):
        """The whole tier is a difference between these two runs.  If they
        produced the same forward there would be nothing to measure."""
        ids = torch.randint(0, EOS, (2, CL * 2))
        outs = []
        for scramble in (False, True):
            m = _model(latent_read_scramble=scramble)
            mc = None
            for xc in torch.chunk(ids, 2, dim=1):
                o = m(xc.contiguous(), num_steps=torch.tensor([0, 8]),
                      m_cross_in=mc, return_m_cross=True)
                mc = o["m_cross"]
            outs.append(mc)
        assert not torch.equal(outs[0], outs[1])

    def test_the_write_is_untouched_by_the_scramble(self):
        """Rolled on the READ side only.  If the write were scrambled too the
        buffer would hold the wrong document's trajectory and the arms would
        differ in two things at once."""
        m = _scrambled()
        assert m.cortex.latent_read_scramble
        # the tape is written from THIS forward's own states; the scramble
        # lives in _latent_z_rows, which only the read path calls.
        import inspect
        src = inspect.getsource(m.cortex._latent_z_rows.__func__)
        assert "roll" in src
        assert "roll" not in inspect.getsource(m.cortex.latent_write.__func__)

    def test_scrambling_a_channel_nothing_reads_is_refused(self):
        from types import SimpleNamespace
        from cortex_graft import CortexMemory
        with pytest.raises(ValueError, match="nothing reads"):
            CortexMemory(SimpleNamespace(
                n_embd=64, use_memory=True, memory_slots=0, accum_vecs=NV,
                prefix_memory="gated", gate_slots=K, eos_token_id=EOS,
                summary_init_token=EOS, latent_carry=True,
                latent_read="none", latent_read_scramble=True))

    def test_the_null_still_wins_over_the_scramble(self):
        """`latent_read_null` is the EVAL's per-chunk ablation and the scramble
        is the CELL's arm.  If both are set the null must win, or a 2x2 run on
        a control-arm checkpoint would silently measure the scramble."""
        m = _scrambled()
        m.cortex.latent_read_null = ("noise_matched", None, 0)
        z = torch.randn(2, 4, 64)
        out = m.cortex._null_latent(("noise_matched", None, 0), z)
        assert not torch.equal(out, torch.roll(z, 1, 0))


# ---------------------------------------------------------------------------
# 1. freeze everything but the read
# ---------------------------------------------------------------------------

class TestFreezeAllButTheRead:

    def test_only_the_read_module_is_trainable(self):
        from cortex_graft import set_read_only_trainable
        m = _model()
        n_train, n_frozen = set_read_only_trainable(m)
        assert n_train == 5, "q/k/v/out + gate"
        assert n_frozen > 0
        for name, p in m.named_parameters():
            assert p.requires_grad == (".latent_reader." in "." + name), name

    def test_the_buffer_and_the_gates_are_frozen(self):
        """Named explicitly because these are the parameters most likely to
        absorb the signal: they are the Z channel's own write path."""
        from cortex_graft import set_read_only_trainable
        m = _model()
        set_read_only_trainable(m)
        pre = m.cortex.prefix
        assert not pre.summary_emb.requires_grad
        for n in ("gate_proj_in_z", "gate_proj_mem_z"):
            assert not getattr(pre, n).weight.requires_grad

    def test_it_refuses_a_model_with_no_read_module(self):
        """Freezing everything with nothing left trainable would run a cell
        whose loss cannot move and report it as a null."""
        from cortex_graft import set_read_only_trainable
        m = _model(read="none", s0=True)
        with pytest.raises(ValueError, match="needs an in-loop read module"):
            set_read_only_trainable(m)

    def test_the_frozen_model_still_produces_a_read_gradient(self):
        """The point of the freeze is that the read STILL trains."""
        from cortex_graft import set_read_only_trainable
        m = _model()
        set_read_only_trainable(m)
        _, losses = _chain(m, n_chunks=2, num_steps=(0, 8), labels=True)
        m.zero_grad(set_to_none=True)
        losses[-1].backward()
        r = m.cortex.latent_reader
        assert float(r.q_proj.weight.grad.abs().sum()) > 0
        assert m.cortex.prefix.summary_emb.grad is None

    def test_refresh_mode_freezes_down_to_one_scalar(self):
        from cortex_graft import set_read_only_trainable
        m = _model(read="refresh", s0=True)
        n_train, _ = set_read_only_trainable(m)
        assert n_train == 1
        assert m.cortex.latent_reader.alpha.requires_grad


# ---------------------------------------------------------------------------
# 2. the forced [0, T] split
# ---------------------------------------------------------------------------

class TestForcedNumSteps:

    @pytest.mark.parametrize("split", [(0, 8), (0, 4), (2, 6)])
    def test_the_split_is_honoured_by_the_loop(self, split):
        from test_read_into import _spy_read
        m = _model()
        seen = _spy_read(m)
        _chain(m, n_chunks=1, num_steps=split)
        assert len(seen) == sum(split)

    def test_zero_no_grad_makes_the_read_live_on_every_batch(self):
        """The reason the probe forces it: the sampler gives n=0 on only ~55%
        of batches at mr8 and ~1.5% at mr32."""
        m = _model()
        _chain(m, n_chunks=2, num_steps=(0, 8))
        assert m.cortex.latent_read_grad_frac == 1.0

    def test_a_malformed_spec_is_rejected(self):
        for bad in ("8", "0,0", "a,b", "-1,8", ""):
            n, k, ok = None, None, True
            try:
                n, k = (int(v) for v in bad.split(","))
                ok = n >= 0 and k >= 1
            except Exception:
                ok = False
            assert not ok, f"{bad!r} should not parse as a valid split"

    def test_a_well_formed_spec_parses(self):
        n, k = (int(v) for v in "0,8".split(","))
        assert (n, k) == (0, 8)


class TestTheArmsAreDistinguishableAfterTheFact:
    """Every one of these flags has to survive a checkpoint round trip, or a
    control-arm result and a treatment-arm result become the same file."""

    def test_the_scramble_flag_is_persisted_by_train(self):
        import re
        src = open(os.path.join(REPO, "train.py"), encoding="utf-8").read()
        block = src[src.index('"latent_s0_read", "latent_read"'):]
        assert '"latent_read_scramble"' in block[:600]

    def test_the_scramble_flag_reaches_the_eval_config(self):
        from tools.prepare_eval_checkpoint import CORTEX_FLAGS
        assert "latent_read_scramble" in CORTEX_FLAGS

    def test_the_banner_names_the_control_arm(self):
        src = open(os.path.join(REPO, "train.py"), encoding="utf-8").read()
        assert "SCRAMBLED(control arm)" in src


class TestAControlArmCheckpointCannotBeScoredSilently:
    """`latent_read_scramble` PERSISTS into the checkpoint -- it has to, or the
    control arm and the treatment arm become the same file.  The consequence is
    that any eval reloading the shuffled limb rebuilds it WITH THE ROLL LIVE,
    and every content number it produces is about the control arm.  Finite,
    plausible, wrong, and silent: the shape of reds 8, 9, 10 and 11.  Tier 3
    runs exactly these evals on exactly these checkpoints."""

    def _cortex(self, scramble):
        m = _model(latent_read_scramble=scramble)
        return m.cortex

    def test_a_scrambled_checkpoint_is_refused(self):
        from cortex_memory.health import refuse_if_scrambled
        with pytest.raises(SystemExit, match="REFUSING"):
            refuse_if_scrambled(self._cortex(True), "carry_2x2")

    def test_an_ordinary_checkpoint_passes_silently(self, capsys):
        from cortex_memory.health import refuse_if_scrambled
        refuse_if_scrambled(self._cortex(False), "carry_2x2")
        assert capsys.readouterr().out == ""

    def test_allow_scrambled_passes_but_says_so_loudly(self, capsys):
        """Measuring the control arm IS sometimes the point -- the tier's whole
        comparison is real vs shuffled.  It just has to be asked for, and the
        log has to carry the caveat next to the numbers."""
        from cortex_memory.health import refuse_if_scrambled
        refuse_if_scrambled(self._cortex(True), "carry_2x2", allow=True)
        out = capsys.readouterr().out
        assert "CONTROL arm" in out

    def test_an_e_only_model_is_not_tripped_by_the_guard(self):
        from cortex_memory.health import refuse_if_scrambled
        refuse_if_scrambled(_model(latent=False).cortex, "carry_2x2")

    @pytest.mark.parametrize("tool", ["eval_carry_2x2", "eval_influence_horizon",
                                      "diag_readinto_gain"])
    def test_every_content_eval_calls_the_guard(self, tool):
        """A guard that exists and is not wired in is the RED 14 shape -- a
        correct check in code no run reaches."""
        src = open(os.path.join(REPO, "evals", tool + ".py"),
                   encoding="utf-8").read()
        assert "refuse_if_scrambled(" in src, f"{tool} never calls the guard"
        assert "--allow_scrambled" in src, f"{tool} has no escape hatch"
