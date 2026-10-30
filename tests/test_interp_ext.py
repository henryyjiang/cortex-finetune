"""GPU-free tests for the 2026-10-30 additions to evals/diag_memory_interp.py:
D's rare-form table, the carry-logit bias shared by parts B and G, F's
steer arithmetic and its control-row choice."""
from __future__ import annotations

import math
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evals"))

import diag_memory_interp as mi  # noqa: E402


# -- rare forms ---------------------------------------------------------------

def test_frequent_forms_is_document_frequency():
    # 4 forms over 10 bags: form 0 in all, form 1 in 1, form 2 in 0, form 3 in 2.
    bags = [torch.tensor([True, i == 0, False, i < 2]) for i in range(10)]
    freq = mi.frequent_forms(bags, 0.10)
    assert freq.tolist() == [True, True, False, True]   # 1/10 hits the 10% bar
    assert mi.frequent_forms(bags, 0.25).tolist() == [True, False, False, False]


def test_restrict_forms_drops_frequent_and_keeps_non_alpha_out():
    form_of = torch.tensor([0, 1, -1, 2, 0])            # vocab -> form
    keep = torch.tensor([False, True, True])            # form 0 is frequent
    assert mi.restrict_forms(form_of, keep).tolist() == [-1, 1, -1, 2, -1]


def test_rare_lens_scores_only_rare_hits():
    form_of = torch.tensor([0, 1, 2, -1])
    rare = mi.restrict_forms(form_of, torch.tensor([False, True, True]))
    bow = mi.bag_of_forms(torch.tensor([0, 1]), rare, 3)    # 'the' + 'king'
    assert bow.tolist() == [False, True, False]
    # top-2 [the, king] -> only king counts under the rare table
    assert float(mi.precision_at_k(torch.tensor([[0, 1]]), rare, bow)[0]) == 0.5


# -- the bias parts B and G share ------------------------------------------------

def _causal(S):
    return torch.full((S, S), float("-inf")).triu(1)


def test_carry_logit_bias_zero_is_the_causal_bias():
    c = _causal(6)
    assert torch.equal(mi.carry_logit_bias(c, (2, 6), [(0, 2)], 0.0), c)


def test_carry_logit_bias_minus_inf_is_the_b_block():
    b = mi.carry_logit_bias(_causal(6), (2, 6), [(0, 2)], float("-inf"))
    assert torch.isinf(b[2:, :2]).all() and (b[2:, :2] < 0).all()
    assert torch.equal(b[:2], _causal(6)[:2])     # carried rows untouched
    assert torch.equal(b[2:, 2:], _causal(6)[2:, 2:])


def test_alpha_scales_the_carried_odds_and_tends_to_the_block():
    torch.manual_seed(0)
    s = torch.randn(6, 6)
    c = _causal(6)
    P = torch.softmax(s + c, -1)
    Ph = torch.softmax(s + mi.carry_logit_bias(c, (2, 6), [(0, 2)], math.log(0.5)), -1)
    # odds of carried column 0 vs real column 2, for real query 3, halve
    r = (Ph[3, 0] / Ph[3, 2]) / (P[3, 0] / P[3, 2])
    assert abs(float(r) - 0.5) < 1e-5
    Pb = torch.softmax(s + mi.carry_logit_bias(c, (2, 6), [(0, 2)], float("-inf")), -1)
    P0 = torch.softmax(s + mi.carry_logit_bias(c, (2, 6), [(0, 2)], math.log(1e-12)), -1)
    assert torch.allclose(P0, Pb, atol=1e-9)


# -- F --------------------------------------------------------------------------

def test_steer_deltas_lift_is_the_donor_boost_net_of_control():
    L, V = 5, 8
    own = torch.log_softmax(torch.zeros(L, V), -1)
    boost = torch.zeros(L, V)
    boost[:, [0, 1]] = math.log(4.0)                    # donor carry favours ids 0,1
    don = torch.log_softmax(boost, -1)
    sets = {"don": torch.tensor([0, 1]), "own": torch.tensor([2]),
            "ctl": torch.tensor([3, 4])}
    d = mi.steer_deltas(own, don, sets, torch.ones(L, dtype=torch.bool))
    # P(don set): 2/8 -> 8/14 ; P(ctl set): 2/8 -> 2/14 ; P(own): 1/8 -> 1/14
    assert abs(d["don"] - math.log((8 / 14) / (2 / 8))) < 1e-6
    assert abs(d["ctl"] - math.log((2 / 14) / (2 / 8))) < 1e-6
    assert abs(d["steer_lift"] - math.log(4.0)) < 1e-6    # net of the generic shift
    assert abs(d["own_drop"]) < 1e-6                      # own shifts like control


def test_steer_deltas_identical_carries_give_zero_and_empty_sets_give_none():
    lp = torch.log_softmax(torch.randn(4, 6), -1)
    sets = {"don": torch.tensor([0]), "own": torch.tensor([], dtype=torch.long),
            "ctl": torch.tensor([1])}
    d = mi.steer_deltas(lp, lp, sets, torch.ones(4, dtype=torch.bool))
    assert d["steer_lift"] == 0.0 and d["own"] is None and d["own_drop"] is None


def test_pick_control_avoids_row_and_donor_books():
    usable = list(range(12))
    book = {r: r // 3 for r in usable}                   # 4 books of 3 windows
    for row in usable:
        donor = mi.pick_donor(row, usable, book)
        ctl = mi.pick_control(row, donor, usable, book)
        assert ctl is not None
        assert book[ctl] not in (book[row], book[donor])
    two = {r: r // 6 for r in usable}                    # 2 books: no third
    assert mi.pick_control(0, mi.pick_donor(0, usable, two), usable, two) is None


def test_parts_include_f_and_g():
    assert {"F", "G"} <= set(mi.PARTS)
