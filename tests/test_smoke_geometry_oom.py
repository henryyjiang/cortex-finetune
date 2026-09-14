"""tools/smoke_geometry_oom.py — the parts that can be wrong without a GPU.

The gate itself needs an H200 and the 1B checkpoint.  What it produces is a
NUMBER that decides the row length for both 5B runs, so the ways it can lie
quietly matter more than the ways it can crash:

  * a recurrence split that is not the worst case under-reports the deepest
    graph the run can build, and the run then OOMs at some step in week two;
  * an optimizer grouping that drifts from train.py's prices the wrong optimizer
    state, and Muon vs Adam is a 2x difference per parameter;
  * a missing `/ accumulation_steps` in the mirrored micro-step is invisible in
    a memory measurement and would make the mirror wrong for anything else;
  * `accum_max` / `carry_grad_chunks` resolved to B2's NUMBERS instead of B2's
    RULE measures a geometry nobody intends to run (b2_retrofit.sbatch:409 sets
    accum_max = cross_chunks x accum_vecs, so at cc16 it is 512, not 256).

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

from recipe_utils import carry_rows, worst_case_num_steps  # noqa: E402
from smoke_geometry_oom import build_param_groups, micro_step  # noqa: E402

NV, CL, NC = 4, 16, 4          # accum_vecs, chunk_len, cross_chunks
EOS = VOCAB - 1
# autocast's device_type must match the device; on CPU the gate's bf16 autocast
# is simply disabled, which changes numerics and nothing structural.
CPU_AMP = {"device_type": "cpu", "dtype": torch.bfloat16, "enabled": False}


def _model(use_memory=True, **kw):
    torch.manual_seed(1234)
    if not use_memory:
        return _build_raven(use_memory=False, **kw).train()
    return _build_raven(use_memory=True, memory_slots=0, prefix_memory="accum",
                        accum_vecs=NV, accum_max=NC * NV, eos_token_id=EOS,
                        **kw).train()


def _batch(n_chunks=NC):
    torch.manual_seed(0)
    ids = torch.randint(0, VOCAB - 1, (1, CL * n_chunks + 1))
    for gi in range(1, n_chunks):
        ids[0, gi * CL + CL // (gi + 1)] = EOS
    return ids[:, :-1], ids[:, 1:]


def _run(model, accumulation_steps=1, use_memory=True, carry_grad_chunks=2):
    x, y = _batch()
    return micro_step(model, x, y, EOS, NC, torch.tensor([1, 1]),
                      carry_grad_chunks, NV, accumulation_steps, CPU_AMP,
                      use_memory)


class TestWorstCaseNumSteps:
    """`k` is what costs activation memory, and the sampler caps it at `s`."""

    def test_k_is_the_backprop_depth_not_half_of_it(self):
        # smoke_prefix_real.py pins [T//2, T-T//2] = [4, 4] at T=8, which is the
        # AVERAGE-ish split, not the ceiling.  The gate must use [0, 8].
        assert worst_case_num_steps(8, 8) == (0, 8)

    def test_no_grad_prefix_takes_the_remainder(self):
        assert worst_case_num_steps(8, 2) == (6, 2)
        assert worst_case_num_steps(32, 8) == (24, 8)

    def test_depth_is_clamped_to_the_current_mean_recurrence(self):
        # sheduler_n_k_handler (train.py:1500) does this whenever the ramp has
        # not yet reached the backprop depth; without the clamp the gate would
        # claim a deeper graph than the run can build.
        assert worst_case_num_steps(2, 8) == (0, 2)

    def test_mean_recurrence_floors_at_one(self):
        # "if new_mean_rec <= 0: new_mean_rec = 1" — the schedule starts at 0.
        assert worst_case_num_steps(0, 8) == (0, 1)


class TestCarryRows:

    def test_first_chunk_reads_nothing(self):
        assert carry_rows(0, 32, 512) == 0

    def test_b2s_rule_never_trims(self):
        # accum_max = cross_chunks x accum_vecs, so the last chunk reads every
        # earlier write: 15 x 32 = 480 at cc16, under the 512 cap.
        assert carry_rows(15, 32, 16 * 32) == 480

    def test_a_capped_buffer_trims(self):
        assert carry_rows(15, 32, 128) == 128
        assert carry_rows(3, 32, 128) == 96       # the last chunk before the cap


class TestParamGroups:
    """Mirror of train.py:1078-1132.  A drift here prices the wrong optimizer."""

    def test_every_parameter_lands_in_exactly_one_group(self):
        model = _model()
        groups = build_param_groups(model, 1e-3, 5e-5, 5e-4)
        seen = [id(p) for g in groups for p in g["params"]]
        assert len(seen) == len(set(seen)), "a parameter is in two groups"
        assert set(seen) == {id(p) for p in model.parameters()}

    def test_cortex_params_get_their_own_adam_group_when_memory_lr_is_set(self):
        model = _model()
        groups = build_param_groups(model, 1e-3, 5e-5, 5e-4)
        assert len(groups) == 3
        assert groups[-1]["weight_decay"] == 0.0, (
            "decay on the identity-init projections would pull them to zero")
        assert groups[-1]["lr"] == 5e-4
        cortex_ids = {id(p) for n, p in model.named_parameters() if "cortex" in n}
        assert cortex_ids and {id(p) for p in groups[-1]["params"]} == cortex_ids

    def test_cortex_params_ride_the_aux_group_when_memory_lr_is_zero(self):
        model = _model()
        groups = build_param_groups(model, 1e-3, 5e-5, 0.0)
        assert len(groups) == 2
        cortex_ids = {id(p) for n, p in model.named_parameters() if "cortex" in n}
        assert cortex_ids <= {id(p) for p in groups[1]["params"]}

    def test_no_cortex_parameter_ever_reaches_muon(self):
        # Newton-Schulz orthogonalisation is wrong for zero/identity-init
        # projections; train.py's comment says so and the grouping enforces it.
        model = _model()
        for memory_lr in (0.0, 5e-4):
            groups = build_param_groups(model, 1e-3, 5e-5, memory_lr)
            muon_ids = {id(p) for g in groups if g["use_muon"] for p in g["params"]}
            for n, p in model.named_parameters():
                if "cortex" in n:
                    assert id(p) not in muon_ids

    def test_embeddings_and_norms_stay_off_muon(self):
        model = _model()
        groups = build_param_groups(model, 1e-3, 5e-5, 5e-4)
        muon_ids = {id(p) for g in groups if g["use_muon"] for p in g["params"]}
        for n, p in model.named_parameters():
            if ("wte" in n) or ("lm_head" in n) or ("norm" in n) or ("ln_f" in n):
                assert id(p) not in muon_ids, n

    def test_body_group_is_sorted_largest_first_and_deterministically(self):
        # train.py sorts explicitly because Muon's own init sorting was not
        # deterministic; two builds must give the same order.
        model = _model()
        a = build_param_groups(model, 1e-3, 5e-5, 5e-4)[0]["params"]
        b = build_param_groups(model, 1e-3, 5e-5, 5e-4)[0]["params"]
        assert [id(p) for p in a] == [id(p) for p in b]
        assert [p.numel() for p in a] == sorted(
            [p.numel() for p in a], reverse=True)

    def test_eps_is_live(self):
        # It was omitted until 2026-09-14 and MuonWithAuxAdam filled in 1e-10.
        model = _model()
        for g in build_param_groups(model, 1e-3, 5e-5, 5e-4):
            if not g["use_muon"]:
                assert g["eps"] == 1e-8


class TestMicroStep:
    """The mirror of cortex_fwd_bwd.  Memory is measured around this loop, so it
    has to be the same loop."""

    def test_chain_runs_and_the_carry_accumulates(self):
        model = _model()
        loss, m_cross = _run(model)
        assert loss == loss                                   # not nan
        assert m_cross.shape[1] == NC * NV

    def test_the_write_is_on_the_loss(self):
        model = _model()
        _run(model)
        g = model.cortex.prefix.summary_emb.grad
        assert g is not None and torch.isfinite(g).all() and float(g.norm()) > 0

    def test_accumulation_divisor_is_applied(self):
        # A missing `/ accumulation_steps` cannot show up in a peak-memory
        # number, so nothing else in this file would catch it.
        one = _model()
        _run(one, accumulation_steps=1)
        two = _model()
        _run(two, accumulation_steps=2)
        a = float(one.cortex.prefix.summary_emb.grad.norm())
        b = float(two.cortex.prefix.summary_emb.grad.norm())
        assert abs(a / b - 2.0) < 1e-3, f"{a} vs {b}"

    def test_stop_gradient_horizon_shrinks_the_retained_graph(self):
        # carry_grad_chunks is a memory parameter: fewer retained chunks must
        # mean strictly less gradient reaching the oldest write.
        full = _model()
        _run(full, carry_grad_chunks=0)                       # full-chain BPTT
        short = _model()
        _run(short, carry_grad_chunks=1)
        a = float(full.cortex.prefix.summary_emb.grad.norm())
        b = float(short.cortex.prefix.summary_emb.grad.norm())
        assert a != b

    def test_a_memory_less_model_runs_the_same_chain_and_carries_nothing(self):
        # The control's path: same chunk loop, no carry.  This is the geometry
        # the C-chunked run will use, and `use_memory=False` must NOT raise on
        # the missing m_cross.
        model = _model(use_memory=False)
        assert getattr(model, "cortex", None) is None
        loss, m_cross = _run(model, use_memory=False)
        assert loss == loss
        assert m_cross is None

    def test_a_silently_memory_less_model_is_caught_when_memory_was_asked_for(self):
        # Recurring bug class 2: the run that trained 9 hours with no
        # cross-segment memory and a healthy loss curve.  Here it would measure
        # the CONTROL's footprint and green-light a geometry that cannot fit.
        model = _model(use_memory=False)
        try:
            _run(model, use_memory=True)
        except RuntimeError as e:
            assert "no m_cross" in str(e)
        else:
            raise AssertionError("a memory-less model passed as a memory run")
