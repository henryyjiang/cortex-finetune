"""
Z attempt 2, Step 1: the synthetic positive control, and the two scoring
additions it needs in eval_carry_2x2 (--score, --z_null donor).

Pinned before any model trains on it: that every answer is the right value,
that answer_dep names the token the answer actually depends on, that "carry"
means exactly "only the carry can supply it" at the eval's own chunking, and
that the donor null feeds another row's Z and nothing else.

Run: /c/Users/henry/miniconda3/envs/cortex-retro/python.exe -m pytest tests/test_carry_task.py -q
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
sys.path.insert(0, os.path.join(REPO, "evals"))

from tools.prepare_carry_task import (  # noqa: E402
    NOT_ANSWER, classify, make_row, piece_ids,
)
from evals.eval_carry_2x2 import carried_z, chain_nll, score_mask  # noqa: E402

REG = list(range(100, 116))          # 16 fake register ids
DIG = list(range(200, 210))          # digit d -> 200 + d
PLUS, EQ, NL = 300, 301, 302


def _row(n=4097, seed=0, ops=1, regs=REG):
    return make_row(random.Random(seed), n, regs, DIG, PLUS, EQ, NL, ops)


def _replay(ids, dep, ops=1):
    """Re-derive every answer from the ids alone and check it."""
    value, checked = {}, 0
    line_len = 4 + 2 * ops
    for start in range(0, len(ids) - line_len + 1, line_len):
        line = ids[start:start + line_len]
        r = line[0]
        total = value.get(r, 0)
        for j in range(ops):
            assert line[1 + 2 * j] == PLUS
            total = (total + line[2 + 2 * j] - 200) % 10
        assert line[-3] == EQ and line[-1] == NL
        assert line[-2] == 200 + total, (start, line)
        value[r] = total
        checked += 1
    return checked


class TestTheRow:
    def test_exact_length_and_every_answer_is_right(self):
        ids, dep = _row()
        assert len(ids) == len(dep) == 4097
        assert _replay(ids, dep) > 600

    def test_multi_op_lines_are_right_too(self):
        ids, dep = _row(ops=3)
        assert _replay(ids, dep, ops=3) > 300

    def test_answer_dep_points_at_the_registers_previous_answer(self):
        ids, dep = _row()
        last = {}
        for q, d in enumerate(dep):
            if d == NOT_ANSWER:
                continue
            assert ids[q - 1] == EQ                  # answers follow '='
            r = ids[q - 4]                           # [REG] + d = v
            assert r in REG
            if r in last:
                assert d == last[r]
                assert ids[d - 1] == EQ              # the dep IS an answer
            else:
                assert d == q - 4                    # first update: line start
            last[r] = q

    def test_the_same_seed_is_the_same_row_and_another_is_not(self):
        assert _row(seed=3) == _row(seed=3)
        assert _row(seed=3)[0] != _row(seed=4)[0]


class TestClassify:
    def test_carry_means_the_dependency_is_in_an_earlier_chunk(self):
        ids, dep = _row()
        carry, local = classify(dep, 8)
        L = 4096 // 8
        for p in range(len(carry)):
            d = dep[p + 1]
            if carry[p]:
                assert d // L < p // L
            if local[p]:
                assert d // L == p // L
            assert not (carry[p] and local[p])
            assert (carry[p] or local[p]) == (d != NOT_ANSWER)

    def test_chunk_one_has_no_carry_answers(self):
        _, dep = _row()
        carry, _ = classify(dep, 8)
        assert sum(carry[:512]) == 0

    def test_about_one_carry_answer_per_register_per_later_chunk(self):
        _, dep = _row(seed=7)
        carry, local = classify(dep, 8)
        per_chunk = [sum(carry[i * 512:(i + 1) * 512]) for i in range(1, 8)]
        # 16 registers, ~85 lines a chunk: nearly every register's first update
        # in a chunk is a carry answer (a register missed for a whole chunk
        # still counts, one chunk later).
        assert all(12 <= c <= 17 for c in per_chunk), per_chunk
        assert sum(local) > 4 * sum(carry)

    def test_the_count_follows_the_eval_chunking_not_the_row(self):
        _, dep = _row(seed=7)
        c8 = sum(classify(dep, 8)[0])
        c2 = sum(classify(dep, 2)[0])
        assert c2 < c8        # fewer boundaries, fewer carry-dependent answers


class _StubTok:
    def __init__(self, table):
        self.table = table

    def encode(self, s, add_special_tokens=False):
        return self.table[s]


class TestPieces:
    def test_every_piece_must_be_one_token(self):
        with pytest.raises(ValueError, match="exactly one"):
            piece_ids(_StubTok({"A": [5], "7": [8, 9]}), ["A", "7"])

    def test_two_pieces_may_not_share_an_id(self):
        with pytest.raises(ValueError, match="share"):
            piece_ids(_StubTok({"A": [5], "7": [5]}), ["A", "7"])

    def test_a_clean_table_maps(self):
        assert piece_ids(_StubTok({"A": [5], "7": [8]}), ["A", "7"]) == {"A": 5, "7": 8}


class TestScoreMask:
    def test_all_is_the_old_all_ones_mask(self):
        m = score_mask({"input_ids": [0] * 4097}, "all", 8, 4096)
        assert torch.equal(m, torch.ones(4096))

    def test_carry_and_local_partition_the_answers(self):
        _, dep = _row(seed=2)
        row = {"answer_dep": dep}
        a = score_mask(row, "answers", 8, 4096)
        c = score_mask(row, "carry", 8, 4096)
        l_ = score_mask(row, "local", 8, 4096)
        assert torch.equal(c + l_, a)
        assert float(c.sum()) > 0 and float(l_.sum()) > 0

    def test_a_pack_without_answer_dep_refuses(self):
        with pytest.raises(SystemExit, match="answer_dep"):
            score_mask({"input_ids": [0] * 4097}, "carry", 8, 4096)


# ─── on a tiny real model ───────────────────────────────────────────────────

from test_cortex_eval import VOCAB, _build_raven  # noqa: E402

NV, K, CL, EOS, D, T = 4, 16, 16, VOCAB - 1, 64, 4


def _model():
    torch.manual_seed(1234)
    return _build_raven(use_memory=True, memory_slots=0, accum_vecs=NV,
                        prefix_memory="gated", gate_slots=K, gate_route="ring",
                        gate_init="zero", gate_fill="grow", latent_carry=True,
                        latent_read="xattn", latent_s0_read=False,
                        latent_read_heads=4, eos_token_id=EOS).eval()


def _chunks(seed, n=4):
    torch.manual_seed(seed)
    ids = torch.randint(0, VOCAB - 1, (CL * n + 1,))
    x, y = ids[:-1], ids[1:]
    return (list(torch.chunk(x, n)), list(torch.chunk(y, n)),
            list(torch.chunk(torch.ones(CL * n), n)))


class TestTheDonorNull:
    def _nll(self, m, xs, ys, ms, z_on, donor_z=None, z_null="donor"):
        return chain_nll(m, m.cortex, xs, ys, ms, torch.tensor([T, 0]),
                         torch.device("cpu"), 11, True, z_on, 0.01, D,
                         z_null=z_null, donor_z=donor_z)[0]

    def test_its_own_z_as_donor_reproduces_the_real_cell_exactly(self):
        """The strongest check that the donor path is the real read path with
        only the Z contents swapped: feeding a row its OWN carried Z through
        the donor slot must give the Z-on number to the bit."""
        m = _model()
        xs, ys, ms = _chunks(0)
        own = carried_z(m, m.cortex, xs, torch.tensor([T, 0]),
                        torch.device("cpu"), 11, D)
        real = self._nll(m, xs, ys, ms, z_on=True)
        swapped = self._nll(m, xs, ys, ms, z_on=False, donor_z=own)
        assert real == swapped

    def test_another_rows_z_changes_the_number(self):
        m = _model()
        xs, ys, ms = _chunks(0)
        dxs, _, _ = _chunks(1)
        donor = carried_z(m, m.cortex, dxs, torch.tensor([T, 0]),
                          torch.device("cpu"), 11, D)
        real = self._nll(m, xs, ys, ms, z_on=True)
        other = self._nll(m, xs, ys, ms, z_on=False, donor_z=donor)
        assert real != other

    def test_the_first_chunk_has_no_donor(self):
        m = _model()
        xs, _, _ = _chunks(0)
        z = carried_z(m, m.cortex, xs, torch.tensor([T, 0]),
                      torch.device("cpu"), 11, D)
        assert z[0] is None and all(t is not None for t in z[1:])

    def test_a_donor_of_the_wrong_shape_refuses(self):
        m = _model()
        xs, ys, ms = _chunks(0)
        bad = [None] + [torch.zeros(1, 3, D)] * 3
        with pytest.raises(ValueError, match="donor Z is"):
            self._nll(m, xs, ys, ms, z_on=False, donor_z=bad)

    def test_the_null_is_cleared_after_the_chain(self):
        m = _model()
        xs, ys, ms = _chunks(0)
        own = carried_z(m, m.cortex, xs, torch.tensor([T, 0]),
                        torch.device("cpu"), 11, D)
        self._nll(m, xs, ys, ms, z_on=False, donor_z=own)
        assert m.cortex.latent_read_null is None


class TestChunkOneSanityUnderAMask:
    def test_a_mask_with_nothing_in_chunk_one_still_reports_chunk_one(self):
        """--score carry leaves chunk 1 with ZERO scored tokens by
        construction.  The sanity loss must still be computed there (it is
        over every token), or the chunk-1 veto silently disappears."""
        m = _model()
        xs, ys, ms = _chunks(0)
        ms = [torch.zeros(CL)] + ms[1:]
        nll, first = chain_nll(m, m.cortex, xs, ys, ms, torch.tensor([T, 0]),
                               torch.device("cpu"), 11, True, True, 0.01, D)
        assert first is not None and nll is not None


class TestTheLaunchers:
    def _src(self, name):
        with open(os.path.join(REPO, "pace", name), encoding="utf-8") as fh:
            return fh.read()

    def test_the_val_pack_uses_another_seed_and_carries_answer_dep(self):
        s = self._src("prepare_carry_task.sbatch")
        assert 'TRAIN_SEED" = "$VAL_SEED"' in s
        assert "--with_answer_dep" in s

    def test_the_default_mix_is_j1s_default_arm_data(self):
        # prepare_carry_task names the mix carry<PCT>; RATIO 0.2 -> carry20.
        assert "j1_pg19fw50_carry${PCT}_len4096" in self._src("prepare_carry_task.sbatch")
        assert "data/j1_pg19fw50_carry20_len4096" in self._src("j1_joint.sbatch")

    def test_the_2x2_launcher_passes_the_score(self):
        assert '--score "${SCORE:-all}"' in self._src("eval_carry_2x2.sbatch")
