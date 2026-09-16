"""
tools/smoke_prefix_real.py's chain logic, on a tiny real raven model.

The smoke itself needs the 1B checkpoint and runs on the cluster; this pins the
part that can be wrong regardless of scale, so a broken smoke is caught here
rather than on a login node at 11pm.  In particular `preserved` is the on-real-
weights version of the Fix A assertion — if it were computed against the
post-detach state instead of the pre-detach one it would read True no matter
what the model did.

Run: /c/Users/henry/miniconda3/envs/cortex-retro/python.exe -m pytest tests/ -q
"""
from __future__ import annotations

import os
import sys

import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "tools"))

from test_cortex_eval import VOCAB, _build_raven  # noqa: E402

from smoke_prefix_real import run_chain  # noqa: E402

NV, CL, NC = 4, 16, 4
K_SLOTS, NC_GATED = 16, 8      # K/W = 4, so two laps need 8 chunks
EOS = VOCAB - 1
DEV = "cpu"


def _model(**kw):
    # Seed before EVERY build: _build_raven random-inits, so two calls otherwise
    # give different weights and any cross-model comparison (e.g. the
    # stop-gradient test below) is uncontrolled and passes or fails by luck.
    torch.manual_seed(1234)
    kw.setdefault("accum_max", 64)
    return _build_raven(use_memory=True, memory_slots=0, prefix_memory="accum",
                        accum_vecs=NV, eos_token_id=EOS, **kw).train()


def _gated(n_slots=K_SLOTS, **kw):
    """The A3' shape in miniature: W=4, K=16, so a lap is 4 chunks and a
    two-lap chain is 8 -- the same K/W=4 ratio the real arm runs at."""
    torch.manual_seed(1234)
    return _build_raven(use_memory=True, memory_slots=0, prefix_memory="gated",
                        accum_vecs=NV, gate_slots=n_slots, gate_route="ring",
                        gate_init="zero", gate_fill="grow",
                        eos_token_id=EOS, **kw).train()


def _batch(n_chunks=NC):
    """Mirrors main()'s batch construction: one EOS inside every chunk after
    the first, placed relative to n_chunks."""
    torch.manual_seed(0)
    ids = torch.randint(0, VOCAB - 1, (1, CL * n_chunks + 1))
    for gi in range(1, n_chunks):
        ids[0, gi * CL + CL // (gi + 1)] = EOS
    return ids[:, :-1], ids[:, 1:]


def _run(model, carry_grad_chunks=2, n_chunks=NC, write_once=True):
    x, y = _batch(n_chunks)
    return run_chain(model, x, y, EOS, n_chunks, torch.tensor([1, 1]),
                     carry_grad_chunks, NV, DEV, write_once=write_once)


class TestChain:

    def test_carry_accumulates_one_write_per_chunk(self):
        m = _model()
        rep = _run(m)
        assert rep["shapes"] == [(1, (g + 1) * NV, m.config.n_embd)
                                 for g in range(NC)]

    def test_carry_survives_document_boundaries(self):
        """The real-weights form of Fix A: earlier chunks' rows must come back
        verbatim from a chunk that contained an EOS."""
        m = _model()
        assert all(_run(m)["preserved"])

    def test_a_zeroed_carry_is_detected_as_not_preserved(self):
        """The check must actually be able to FAIL, or a `preserved` that was
        trivially True would pass the smoke forever.  This used to be driven by
        prefix_eos_reset=True, whose branch was retired 2026-09-15; zeroing the
        carried state directly exercises the same failure without it, and tests
        the CHECKER rather than a flag."""
        m = _model()
        real = m.cortex._carried_state
        m.cortex._carried_state = lambda: (
            None if real() is None else torch.zeros_like(real()))
        try:
            assert not all(_run(m)["preserved"])
        finally:
            m.cortex._carried_state = real

    def test_losses_finite_and_backward_reaches_the_write(self):
        m = _model()
        rep = _run(m)
        assert all(l == l for l in rep["losses"])
        g = m.cortex.prefix.summary_emb.grad
        assert g is not None and torch.isfinite(g).all() and float(g.norm()) > 0

    def test_loop_params_receive_gradient(self):
        m = _model()
        _run(m)
        loop = [p for n, p in m.named_parameters()
                if "core_block" in n and p.grad is not None]
        assert loop and all(torch.isfinite(p.grad).all() for p in loop)

    def test_two_chunk_chain_works(self):
        """--cross_chunks 2 is the cheapest way to reach the real chunk_len on a
        short GPU session, and it used to crash: the EOS offsets were hardcoded
        for >= 3 chunks and indexed past the end of the sequence."""
        m = _model()
        rep = _run(m, n_chunks=2)
        assert rep["shapes"] == [(1, NV, m.config.n_embd), (1, 2 * NV, m.config.n_embd)]
        assert all(rep["preserved"])

    def test_stop_gradient_horizon_is_applied(self):
        """carry_grad_chunks must shorten the graph — with a 1-chunk horizon the
        oldest rows are detached, so summary_emb's gradient is strictly smaller
        than under full-chain BPTT."""
        a = _model(); _run(a, carry_grad_chunks=1)
        b = _model(); _run(b, carry_grad_chunks=0)
        assert float(a.cortex.prefix.summary_emb.grad.norm()) \
            < float(b.cortex.prefix.summary_emb.grad.norm())


class TestGatedChain:
    """The gated geometry, which shares not one assertion with the accum arm.

    Everything here is a property of the PACKED FORWARD -- the rows the modeling
    file actually spliced and got back -- rather than of the config, which is the
    whole reason the smoke exists.  See tools/smoke_prefix_real.py's header.
    """

    def _rep(self, n_chunks=NC_GATED, **kw):
        return _run(_gated(**kw), n_chunks=n_chunks, write_once=False)

    def test_carry_grows_by_w_then_stops_at_k(self):
        m = _gated()
        rep = _run(m, n_chunks=NC_GATED, write_once=False)
        assert rep["shapes"] == [(1, min((g + 1) * NV, K_SLOTS), m.config.n_embd)
                                 for g in range(NC_GATED)]

    def test_ring_gates_exactly_its_own_rows(self):
        """The sparse ring's defining property: W rows move, K-W pass through.

        A dense route (or a gate applied to the whole state with fg == 1 on the
        untouched rows) would show every row changing, which is the difference
        between the buffer that was designed and the buffer that was built.
        """
        rep = self._rep()
        got = [sorted(c) for c in rep["changed"]]
        want = []
        for g in range(1, NC_GATED):
            want.append([] if g * NV < K_SLOTS
                        else [((g * NV) % K_SLOTS + i) % K_SLOTS
                              for i in range(NV)])
        assert got == want

    def test_first_lap_is_bit_identical_to_accum(self):
        """fill='grow' means lap 1 IS PrefixAccumBuffer, so the two arms diverge
        at EXACTLY the chunk where accum would start dropping rows.  That is the
        contrast the horizon claim rests on; if it ever stops holding, A1 vs A3'
        is no longer a controlled pair."""
        a = _run(_model(accum_max=K_SLOTS), n_chunks=K_SLOTS // NV)
        g = _run(_gated(), n_chunks=K_SLOTS // NV, write_once=False)
        assert a["shapes"] == g["shapes"]
        for la, lg in zip(a["losses"], g["losses"]):
            assert abs(la - lg) < 1e-4

    def test_a_chain_shorter_than_two_laps_never_fires_the_gate(self):
        """The constraint nothing else checks, from the failing side: at one lap
        the run trains an append buffer with the gate's parameters attached and
        the loss curve looks perfectly healthy.  train.py asserts the bound and
        the smoke refuses the geometry -- this pins WHY."""
        rep = self._rep(n_chunks=K_SLOTS // NV)
        assert not any(rep["changed"])

    def test_gate_params_receive_gradient(self):
        m = _gated()
        _run(m, n_chunks=NC_GATED, write_once=False)
        buf = m.cortex.prefix
        for name in ("gate_proj_in", "gate_proj_mem"):
            g = getattr(buf, name).weight.grad
            assert g is not None and torch.isfinite(g).all() and float(g.norm()) > 0
        assert float(buf.forget_bias.grad.norm()) > 0

    def test_carry_grad_chunks_1_leaves_the_forget_gate_untrained(self):
        """`keep carry_grad_chunks >= 2 or gate_proj_mem never trains` is in the
        PrefixGatedBuffer docstring and nothing tested it.  It is exact, not a
        weakening: merge runs AFTER the chunk's own loss, so the gated state
        only ever reaches a loss through the NEXT chunk -- and at
        carry_grad_chunks=1 every chunk detaches its incoming carry, so that
        path never exists and the gradient is None, not small.

        The failure mode is the usual one: a healthy loss curve over a forget
        gate frozen at its init, which is an EMA with millions of dead
        parameters.  run_chain dispatches the horizon exactly as train.py's
        cortex_fwd_bwd does, so this is the run's behaviour and not the smoke's.
        """
        short = _gated(); _run(short, carry_grad_chunks=1, n_chunks=NC_GATED,
                               write_once=False)
        assert short.cortex.prefix.gate_proj_mem.weight.grad is None

        full = _gated(); _run(full, carry_grad_chunks=0, n_chunks=NC_GATED,
                              write_once=False)
        assert float(full.cortex.prefix.gate_proj_mem.weight.grad.norm()) > 0

        arm = _gated(); _run(arm, carry_grad_chunks=4, n_chunks=NC_GATED,
                             write_once=False)      # the A3' setting
        assert float(arm.cortex.prefix.gate_proj_mem.weight.grad.norm()) > 0

    def test_depth_rows_matches_the_rows_the_ring_actually_wrote(self):
        """depth_rows() is the addressing half of the depth-slice ablation.  It
        is checked against the rows the FORWARD moved, not against its own
        formula restated."""
        m = _gated()
        rep = _run(m, n_chunks=NC_GATED, write_once=False)
        buf = m.cortex.prefix
        for chunk_rows in rep["changed"]:
            if not chunk_rows:
                continue
            for r in chunk_rows:
                j = r % NV
                assert r in buf.depth_rows(j)


class TestResetGuard:
    """The smoke applies the designed init, which is right for a graft-prepared
    BASE and destructive for a TRAINED checkpoint.

    The base case is the one the script is normally run on, so it is also the
    one that hides a mistake: resetting a trained buffer would not raise, it
    would just quietly report that a freshly initialised buffer works.
    """

    def test_a_base_checkpoint_is_detected_as_fresh(self):
        from smoke_prefix_real import cortex_weights_are_fresh
        info = {"missing_keys": ["cortex.prefix.summary_emb",
                                 "cortex.prefix.gate_proj_in.weight"],
                "unexpected_keys": []}
        assert cortex_weights_are_fresh(info)

    def test_a_trained_checkpoint_is_not(self):
        from smoke_prefix_real import cortex_weights_are_fresh
        assert not cortex_weights_are_fresh(
            {"missing_keys": [], "unexpected_keys": []})

    def test_missing_non_cortex_keys_do_not_make_it_look_fresh(self):
        """A checkpoint can legitimately be missing tied weights or a buffer
        without its cortex weights being absent."""
        from smoke_prefix_real import cortex_weights_are_fresh
        assert not cortex_weights_are_fresh(
            {"missing_keys": ["lm_head.weight", "transformer.ln_f.bias"]})

    def test_it_is_safe_on_an_empty_or_absent_loading_info(self):
        from smoke_prefix_real import cortex_weights_are_fresh
        assert not cortex_weights_are_fresh({})
        assert not cortex_weights_are_fresh(None)
