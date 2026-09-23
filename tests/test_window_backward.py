"""--cortex.window_backward: one backward per carry window instead of per row.

Why it exists (J1, 2026-09-22): cortex_fwd_bwd keeps every chunk's loss until a
single backward at the end, and each loss holds its chunk's activation graph,
so `carry_grad_chunks` cuts gradient paths and frees nothing.  J1 at micro 2
OOMed on BOTH limbs at 138.71 of 139.80 GiB (job 13456835), and cc16 had OOMed
at carry_grad_chunks 2 and 1 alike.  On the gated buffer the whole carry
detaches every carry_grad_chunks chunks, so the row's graph is a set of
independent windows and each can be backpropagated as it closes.

What has to hold, and what each class pins:
  * the per-chunk weights reproduce reduce_chunk_losses exactly  (the loss)
  * the gradient equals the one-backward gradient                (exactness)
  * the row really is backpropagated once per window             (it happened)
  * the retained graph at the peak shrinks to about one window   (what it is FOR)
  * every geometry where it cannot apply is refused              (no silent no-op)
  * train.py and the launchers carry it                           (the run gets it)

Run: /c/Users/henry/miniconda3/envs/cortex-retro/python.exe -m pytest tests/test_window_backward.py -q
"""
from __future__ import annotations

import os
import random
import sys

import pytest
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "tools"))

from test_cortex_eval import VOCAB, _build_raven  # noqa: E402

from recipe_utils import chunk_loss_weights, reduce_chunk_losses  # noqa: E402
from smoke_geometry_oom import micro_step  # noqa: E402

NV, CL, GK, GNC = 4, 16, 16, 8     # W, chunk_len, K (lap = 4 chunks), cross_chunks
EOS = VOCAB - 1
CPU_AMP = {"device_type": "cpu", "dtype": torch.bfloat16, "enabled": False}


def _read(path):
    with open(os.path.join(REPO, path), encoding="utf-8") as fh:
        return fh.read()


# ─── the weights ────────────────────────────────────────────────────────────

class TestTheWeightsReproduceTheRowLoss:
    @pytest.mark.parametrize("mode", ["token", "chunk"])
    def test_weighted_sum_equals_reduce_chunk_losses(self, mode):
        rng = random.Random(0)
        for _ in range(50):
            n = rng.randint(1, 12)
            tokens = [rng.choice([0, rng.randint(1, 600)]) for _ in range(n)]
            if not any(tokens):
                tokens[0] = 7
            losses = torch.rand(n, dtype=torch.float64) * 5
            w = chunk_loss_weights(tokens, mode)
            kept = [i for i, t in enumerate(tokens) if t]
            want = reduce_chunk_losses([losses[i] for i in kept],
                                       [tokens[i] for i in kept], mode)
            got = sum(wi * float(li) for wi, li in zip(w, losses))
            assert abs(got - float(want)) < 1e-12

    def test_a_fully_masked_chunk_weighs_nothing(self):
        assert chunk_loss_weights([10, 0, 30], "token") == [0.25, 0.0, 0.75]
        assert chunk_loss_weights([10, 0, 30], "chunk") == [0.5, 0.0, 0.5]

    def test_every_chunk_masked_gives_all_zero(self):
        assert chunk_loss_weights([0, 0], "token") == [0.0, 0.0]

    def test_bad_input_raises(self):
        with pytest.raises(ValueError, match="chunk_loss_reduction"):
            chunk_loss_weights([1], "mean")
        with pytest.raises(ValueError, match="negative"):
            chunk_loss_weights([1, -1], "token")


# ─── the J1-shaped model ────────────────────────────────────────────────────

def _j1(limb="real"):
    """Gated W=4/K=16 ring + Z, tokens encoding, xattn read at read_into --
    J1's architecture at test size."""
    torch.manual_seed(1234)
    flags = dict(use_memory=True, memory_slots=0, prefix_memory="gated",
                 accum_vecs=NV, gate_slots=GK, gate_route="ring",
                 gate_init="zero", gate_fill="grow", eos_token_id=EOS,
                 latent_carry=True, latent_encoding="tokens",
                 latent_tok_pool=2, latent_s0_read=False)
    if limb == "noread":
        flags.update(latent_read="none", latent_write_only=True)
    else:
        flags.update(latent_read="xattn", latent_read_heads=4,
                     latent_read_znorm="rms",
                     latent_read_scramble=(limb == "donor"))
    return _build_raven(**flags).train()


def _batch(batch=2):
    torch.manual_seed(0)
    ids = torch.randint(0, VOCAB - 1, (batch, CL * GNC + 1))
    for gi in range(1, GNC):
        ids[:, gi * CL + CL // (gi + 1)] = EOS
    return ids[:, :-1], ids[:, 1:]


def _step(model, grad_chunks, window, batch=2, accumulation_steps=2):
    x, y = _batch(batch)
    torch.manual_seed(7)            # s0 draws: identical order on both paths
    return micro_step(model, x, y, EOS, GNC, torch.tensor([0, 2]),
                      grad_chunks, NV, accumulation_steps, CPU_AMP, True,
                      write_once=False, window_backward=window)


def _grads(model):
    return {n: p.grad.detach().clone() for n, p in model.named_parameters()
            if p.grad is not None}


class TestTheGradientIsTheOneBackwardGradient:
    @pytest.mark.parametrize("limb", ["real", "donor", "noread"])
    @pytest.mark.parametrize("grad_chunks", [4, 2])
    def test_same_loss_and_same_gradient(self, limb, grad_chunks):
        a = _j1(limb)
        loss_a, _ = _step(a, grad_chunks, window=False)
        b = _j1(limb)
        loss_b, _ = _step(b, grad_chunks, window=True)
        assert abs(loss_a - loss_b) < 1e-5 * max(1.0, abs(loss_a))
        ga, gb = _grads(a), _grads(b)
        assert ga.keys() == gb.keys() and ga, "different parameter sets got grad"
        for n in ga:
            scale = float(ga[n].abs().max()) or 1.0
            err = float((ga[n] - gb[n]).abs().max())
            assert err <= 1e-5 * scale + 1e-9, f"{n}: {err} vs scale {scale}"

    def test_the_z_write_still_gets_read_pressure(self):
        """The property J1 exists for survives the split: inside a window the
        read in chunk n+1 still reaches the write in chunk n."""
        m = _j1("real")
        _step(m, 4, window=True)
        g = m.cortex.prefix.gate_proj_in_z.weight.grad
        assert g is not None and float(g.abs().sum()) > 0


class TestTheRowIsSplit:
    @staticmethod
    def _backward_passes(model, grad_chunks, window):
        """How many backward calls reached the embedding (every chunk uses it,
        so every window's backward does).  A leaf's tensor hook fires once
        per backward call, with that call's summed gradient."""
        n = [0]
        emb = model.get_input_embeddings().weight
        h = emb.register_hook(lambda g: n.__setitem__(0, n[0] + 1))
        try:
            _step(model, grad_chunks, window)
        finally:
            h.remove()
        return n[0]

    def test_one_backward_per_row_without_the_flag(self):
        assert self._backward_passes(_j1(), 4, window=False) == 1

    @pytest.mark.parametrize("grad_chunks, windows", [(4, 2), (2, 4), (8, 1)])
    def test_one_backward_per_window_with_it(self, grad_chunks, windows):
        assert self._backward_passes(_j1(), grad_chunks, window=True) == windows


class _Live:
    """saved_tensors_hooks accounting: bytes saved for backward and not yet
    released, and the peak of that.  Counts a tensor once per save, so the
    absolute number over-counts shared storage -- the same way on both paths,
    which is all a ratio needs."""

    def __init__(self):
        self.live = self.peak = 0

    def pack(self, t):
        n = t.numel() * t.element_size()
        self.live += n
        self.peak = max(self.peak, self.live)
        return _Held(t, n, self)

    @staticmethod
    def unpack(h):
        return h.t


class _Held:
    def __init__(self, t, n, acct):
        self.t, self.n, self.acct = t, n, acct

    def __del__(self):
        self.acct.live -= self.n


class TestThePeakShrinks:
    """What the flag is FOR.  Two windows of four at cc8 should hold about
    half the row's saved activations at the peak; the real run's number comes
    from pace/j1_price_oom.sbatch, this pins the direction and the rough size."""

    @staticmethod
    def _peak(grad_chunks, window):
        m = _j1("real")
        acct = _Live()
        with torch.autograd.graph.saved_tensors_hooks(acct.pack, acct.unpack):
            _step(m, grad_chunks, window)
        return acct.peak

    def test_two_windows_hold_about_half(self):
        row = self._peak(4, window=False)
        win = self._peak(4, window=True)
        assert win < 0.65 * row, f"window peak {win} vs row peak {row}"

    def test_carry_grad_chunks_alone_frees_nothing(self):
        """The belief this flag corrects: without it, a shorter horizon does
        NOT lower the peak, because every chunk's loss still holds its graph."""
        four = self._peak(4, window=False)
        two = self._peak(2, window=False)
        assert two > 0.9 * four, f"horizon 2 peak {two} vs horizon 4 peak {four}"


# ─── refusals ───────────────────────────────────────────────────────────────

class TestItRefusesWhereItCannotApply:
    def test_micro_step_refuses_the_accum_slice_detach(self):
        with pytest.raises(ValueError, match="whole-carry detach"):
            micro_step(None, torch.zeros(1, 8, dtype=torch.long),
                       torch.zeros(1, 8, dtype=torch.long), EOS, 2,
                       torch.tensor([0, 1]), 1, NV, 1, CPU_AMP, True,
                       write_once=True, window_backward=True)

    def test_micro_step_refuses_full_chain_bptt(self):
        with pytest.raises(ValueError, match="whole-carry detach"):
            micro_step(None, torch.zeros(1, 8, dtype=torch.long),
                       torch.zeros(1, 8, dtype=torch.long), EOS, 2,
                       torch.tensor([0, 1]), 0, NV, 1, CPU_AMP, True,
                       write_once=False, window_backward=True)

    def test_train_py_refuses_accum_zero_horizon_no_memory_and_ddp(self):
        s = _read("train.py")
        i = s.index('if cfg.cortex["window_backward"]:')
        guard = s[i:i + 2500]
        for needle in ('not cfg.cortex["use_memory"]',
                       'cfg.cortex["prefix_memory"] == "accum"',
                       "_gc <= 0", 'state["distributed"]', "raise ValueError"):
            assert needle in guard, needle


# ─── train.py and the launchers carry it ────────────────────────────────────

class TestTrainPyCarriesIt:
    def test_default_off_so_every_arm_on_record_is_unchanged(self):
        assert "window_backward=False," in _read("train.py")

    def test_the_flush_sits_on_the_whole_carry_detach_and_before_it(self):
        s = _read("train.py")
        i = s.index("def cortex_fwd_bwd(")
        body = s[i:s.index("# NOT `use_memory AND cross_chunks > 1`", i)]
        branch = body[body.index("elif gi % grad_chunks == 0:"):]
        flush = branch.index("if window_bwd and window:")
        detach = branch.index("m_cross = m_cross.detach()")
        assert flush < detach
        # and never on the accum slice-detach branch
        accum = body[body.index("if accum_on:"):body.index("elif gi % grad_chunks == 0:")]
        assert "backward" not in accum

    def test_the_weights_come_from_the_shared_helper(self):
        s = _read("train.py")
        assert "chunk_loss_weights(" in s
        assert "window.append(out[\"loss\"] * chunk_w[gi])" in s

    def test_the_last_window_is_flushed_and_the_total_returned(self):
        s = _read("train.py")
        i = s.index("if window_bwd:\n                        if cfg.cortex[\"diag_interval\"]")
        tail = s[i:i + 1200]
        assert "(part / accumulation_steps).backward()" in tail
        assert "return total, total.exp(), n_ng, n_wg" in tail


class TestTheLaunchersCarryIt:
    def test_j1_turns_it_on_by_default(self):
        s = _read("pace/j1_joint.sbatch")
        assert "WINDOW_BACKWARD=${WINDOW_BACKWARD:-1}" in s
        assert "--cortex.window_backward" in s

    def test_the_price_measures_the_same_path(self):
        s = _read("pace/j1_price_oom.sbatch")
        assert "WINDOW_BACKWARD=${WINDOW_BACKWARD:-1}" in s
        assert "--window_backward" in s

    def test_the_price_can_gate_a_dependency_chain(self):
        # sbatch --dependency=afterok needs a NON-ZERO exit when J1 does not
        # fit; the price used to exit 0 on an OOM by design.
        s = _read("pace/j1_price_oom.sbatch")
        assert "REQUIRE_FIT" in s and "exit 3" in s
