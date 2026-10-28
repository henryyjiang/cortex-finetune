"""GPU-free tests for the pure helpers in evals/diag_memory_interp.py.

The instrument's column attribution rests on three derivations -- the ring's
writer/age map, the packed column layout under gate_fill=grow, and the share
aggregation -- and each is pinned here against an independent statement of the
same rule, so a drift in one cannot be confirmed by the other.
"""
from __future__ import annotations

import math
import os
import sys

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evals"))

import diag_memory_interp as mi  # noqa: E402
from cortex_memory.buffers import PrefixGatedBuffer  # noqa: E402
from cortex_memory.health import expected_ring_rows  # noqa: E402

W, K = 16, 64


# -- ring writers / ages ------------------------------------------------------

def test_carried_rows_grow_fill():
    assert [mi.carried_rows(q, W, K) for q in range(1, 9)] == \
        [0, 16, 32, 48, 64, 64, 64, 64]


def test_ring_writers_first_lap_is_append_order():
    # Chunk q reads blocks written by chunks 1..q-1 in order during lap 1.
    for q in range(2, 6):
        w = mi.ring_writers(q, W, K)
        assert w == [c for c in range(1, q) for _ in range(W)][:len(w)]


def test_ring_writers_match_the_spec_rule():
    # Spec: chunk c writes rows ((c-1) mod 4)*16 .. +16.
    for q in range(2, 20):
        w = mi.ring_writers(q, W, K)
        for r, c in enumerate(w):
            assert ((c - 1) % (K // W)) * W <= r < ((c - 1) % (K // W)) * W + W
            # and c is the MOST RECENT such chunk before q
            assert c <= q - 1 and c + K // W > q - 1


def test_closed_form_agrees_with_replay():
    for q in range(2, 30):
        w = mi.ring_writers(q, W, K)
        for b in range(len(w) // W):
            assert mi.ring_writer_closed(q, b, W, K) == w[b * W]
        for b in range(len(w) // W, K // W):
            assert mi.ring_writer_closed(q, b, W, K) is None


def test_block_ages():
    assert mi.block_ages(2, W, K) == [1]
    assert mi.block_ages(4, W, K) == [3, 2, 1]
    assert mi.block_ages(5, W, K) == [4, 3, 2, 1]
    # chunk 6: block 0 rewritten by chunk 5 (age 1), block 1 by chunk 2 (age 4)
    assert mi.block_ages(6, W, K) == [1, 4, 3, 2]
    assert mi.block_ages(8, W, K) == [3, 2, 1, 4]
    for q in range(5, 30):
        assert sorted(mi.block_ages(q, W, K)) == [1, 2, 3, 4]


def test_ring_writers_match_the_real_buffer():
    """Drive the REAL PrefixGatedBuffer.merge and tag each write by chunk."""
    D = 4
    buf = PrefixGatedBuffer(D, n_vec=W, n_slots=K, route="ring", fill="grow")
    # Make the gate a pure overwrite (fg=0, ig=1) so a row holds its last
    # writer's tag exactly.
    with torch.no_grad():
        buf.forget_bias.fill_(-1e4)
        buf.input_bias.fill_(1e4)
    state = None
    for c in range(1, 12):
        new = torch.full((1, W, D), float(c))
        if state is not None:
            q = c                     # this state is what chunk c reads
            tags = state[0, :, 0].round().long().tolist()
            assert tags == mi.ring_writers(q, W, K)
        state = buf.merge(state, new)


def test_age_bucket():
    assert [mi.age_bucket(a, 4) for a in (1, 2, 3, 4, 5, 9)] == [0, 1, 2, 3, 3, 3]
    with pytest.raises(ValueError):
        mi.age_bucket(0, 4)


def test_ragged_ring_refused():
    with pytest.raises(ValueError):
        mi.lap_chunks(16, 40)


# -- packed layout -------------------------------------------------------------

def test_column_layout_each_chunk():
    L, S = 512, 16
    for q in range(1, 9):
        lay = mi.column_layout(q, W, K, L, S)
        nE = 16 * min(q - 1, 4)
        assert lay["E"] == (0, nE)
        assert lay["Z"] == (nE, 2 * nE)
        assert lay["real"] == (2 * nE, 2 * nE + L)
        assert lay["summary"] == (2 * nE + L, 2 * nE + L + S)
        assert lay["total"] == 2 * nE + L + S
    assert mi.column_layout(1, W, K, L, S)["real"] == (0, L)


def test_ring_rows_rule_is_contiguous():
    # The spy assumes a chunk's ring rows are one contiguous block.
    for c in range(1, 20):
        rows = expected_ring_rows(c - 1, W, K)
        assert rows == list(range(rows[0], rows[0] + W))


# -- precision@k ----------------------------------------------------------------

def test_precision_at_k():
    vocab = [" The", "the", " cat", ",", " dog", "ing", " Cat", "123"]
    form_of, forms = mi.build_form_table(vocab)
    # " The" and "the" share a form; "," and "123" have none
    assert form_of[0] == form_of[1] and form_of[3] == -1 and form_of[7] == -1
    assert form_of[2] == form_of[6]
    bow = mi.bag_of_forms(torch.tensor([1, 2, 3]), form_of, len(forms))
    top = torch.tensor([[0, 2, 3, 4], [4, 5, 7, 3]])
    p = mi.precision_at_k(top, form_of, bow)
    assert torch.allclose(p, torch.tensor([0.5, 0.0]))


def test_form_table_min_len():
    form_of, forms = mi.build_form_table([" the", " house", " of", "Paris"], 4)
    assert form_of.tolist()[0] == -1 and form_of.tolist()[2] == -1
    assert sorted(forms) == ["house", "paris"]


def test_classify_tokens():
    toks = [" The", " house", " of", " Paris", "ian", ",", " a", "bc", "\n",
            "Hello", " 42"]
    got = mi.classify_tokens(toks)
    # content, content, function, content, continuation, other, function,
    # continuation, other, content (word-initial after a newline), other
    assert got == [0, 0, 1, 0, 2, 3, 1, 2, 3, 0, 3]


# -- aggregation ----------------------------------------------------------------

def _random_causal_P(H, S, seed=0):
    g = torch.Generator().manual_seed(seed)
    s = torch.randn(H, S, S, generator=g)
    s = s + torch.full((S, S), float("-inf")).triu(1)
    return torch.softmax(s, -1)


def test_read_masses_sum_to_one_and_summary_is_zero():
    q, L, nS, H = 6, 40, W, 3
    lay = mi.column_layout(q, W, K, L, nS)
    P = _random_causal_P(H, lay["total"])
    r0, r1 = lay["real"]
    ages = mi.block_ages(q, W, K)
    oh = mi.onehot(mi.position_bins(L, 8), 8)
    res = mi.read_masses(P[:, r0:r1], lay, W, ages, 4, oh)
    assert float(res["sum_err"]) < 1e-5
    assert float(res["summary_max"]) == 0.0
    k = res["kind"]
    # summed over L queries: E+Z+own+summary == L per head
    assert torch.allclose(k[:, :4].sum(-1), torch.full((H,), float(L)), atol=1e-3)
    # age buckets partition E and Z mass exactly
    assert torch.allclose(res["age"][:, 0].sum(-1), k[:, 0], atol=1e-5)
    assert torch.allclose(res["age"][:, 1].sum(-1), k[:, 1], atol=1e-5)
    # position bins partition each kind's mass exactly
    assert torch.allclose(res["pos"].sum(-1), k[:, :3], atol=1e-4)
    # per-column: own normalised by the VISIBLE count i+1
    vis = torch.arange(1, L + 1, dtype=torch.float32)
    own = P[:, r0:r1, r0:r1].sum(-1)
    assert torch.allclose(res["pcol"][:, 2], (own / vis).sum(-1), atol=1e-5)
    assert torch.allclose(res["pcol"][:, 0], k[:, 0] / lay["nE"], atol=1e-6)


def test_read_masses_age_attribution_points_at_the_right_block():
    # All of every query's E mass on the block written 2 chunks ago.
    q, L, H = 6, 16, 1
    lay = mi.column_layout(q, W, K, L, W)
    S = lay["total"]
    ages = mi.block_ages(q, W, K)           # [1, 4, 3, 2]
    b = ages.index(2)
    P = torch.zeros(H, S, S)
    r0, r1 = lay["real"]
    P[0, r0:r1, b * W] = 1.0
    res = mi.read_masses(P[:, r0:r1], lay, W, ages, 4,
                         mi.onehot(mi.position_bins(L, 8), 8))
    assert res["age"][0, 0].tolist() == [0.0, float(L), 0.0, 0.0]
    assert float(res["kind"][0, 1]) == 0.0


def test_write_masses():
    q, L, H = 3, 32, 2
    lay = mi.column_layout(q, W, K, L, W)
    P = _random_causal_P(H, lay["total"], seed=1)
    s0, s1 = lay["summary"]
    cls = torch.tensor([i % 4 for i in range(L)])
    res = mi.write_masses(P[:, s0:s1], lay, mi.onehot(mi.position_bins(L, 8), 8),
                          mi.onehot(cls, 4))
    assert float(res["sum_err"]) < 1e-5
    # renormalised real-token distribution: pos and cls each sum to 1 per slot
    assert torch.allclose(res["pos"].sum(-1), torch.full((H,), float(W)), atol=1e-4)
    assert torch.allclose(res["cls"].sum(-1), torch.full((H,), float(W)), atol=1e-4)
    assert (res["ent"] <= W * math.log(L) + 1e-3).all()
    # removing the first token and renormalising is still a distribution
    assert torch.allclose(res["cls_ex"].sum(-1), torch.full((H,), float(W)), atol=1e-3)
    p = P[:, s0:s1, lay["real"][0]:lay["real"][1]]
    p = p / p.sum(-1, keepdim=True)
    assert torch.allclose(res["first"], p[..., 0].sum(-1), atol=1e-5)


def test_write_cls_ex_removes_a_sink_token():
    # All real-token mass on token 0 (class 'other') except a sliver on a
    # content token: ex-first shares must be 100% content.
    q, L, H = 2, 16, 1
    lay = mi.column_layout(q, W, K, L, W)
    S = lay["total"]
    P = torch.zeros(H, S, S)
    s0, s1 = lay["summary"]
    r0, _ = lay["real"]
    P[0, s0:s1, r0] = 0.9
    P[0, s0:s1, r0 + 5] = 0.1
    cls = torch.full((L,), 3, dtype=torch.long)
    cls[5] = 0
    res = mi.write_masses(P[:, s0:s1], lay, mi.onehot(mi.position_bins(L, 8), 8),
                          mi.onehot(cls, 4))
    assert torch.allclose(res["first"], torch.tensor([0.9 * W]))
    assert torch.allclose(res["cls_ex"][0], torch.tensor([float(W), 0, 0, 0]),
                          atol=1e-5)


def test_write_entropy_uniform_is_log_L():
    q, L, H = 2, 64, 1
    lay = mi.column_layout(q, W, K, L, W)
    S = lay["total"]
    P = torch.zeros(H, S, S)
    s0, s1 = lay["summary"]
    r0, r1 = lay["real"]
    P[0, s0:s1, r0:r1] = 0.5 / L
    P[0, s0:s1, 0] = 0.5
    res = mi.write_masses(P[:, s0:s1], lay, mi.onehot(mi.position_bins(L, 8), 8),
                          mi.onehot(torch.zeros(L, dtype=torch.long), 4))
    assert abs(float(res["ent"][0]) / W - math.log(L)) < 1e-5


def test_accumulator_counts_and_row_means():
    n, L, H, n_blk = 3, 16, 2, 2
    acc = mi.Acc(n_blk, H, n, 4, 8, W, torch.device("cpu"))
    oh = mi.onehot(mi.position_bins(L, 8), 8)
    for q in range(1, n + 1):
        ctx = {"q": q, "L": L, "W": W, "nS": W, "nE": mi.carried_rows(q, W, K),
               "ages": mi.block_ages(q, W, K) if q >= 2 else [],
               "bins_oh": oh, "cls_oh": mi.onehot(torch.zeros(L, dtype=torch.long), 4)}
        acc.begin_chunk(ctx, True, True)
        lay = mi.column_layout(q, W, K, L, W)
        P = _random_causal_P(H, lay["total"], seed=q)
        for blk in range(n_blk):
            acc.add_read(blk, P, lay, ctx)
            acc.add_write(blk, P, lay, ctx)
    acc.end_row()
    assert acc.nq.tolist() == [L, L, L]
    assert acc.n_age_q.tolist() == [2 * L, L, 0, 0]   # q=2: age1; q=3: ages 1,2
    assert acc.nC == 3 * W
    assert len(acc.rows_EZ) == 1 and acc.rows_EZ[0].shape == (n_blk, 2)
    # chunk 1 has no carried columns: no E/Z mass recorded there
    assert float(acc.A_kind[:, :, 0, :2].abs().sum()) == 0.0
    assert float(acc.max_summary) == 0.0


# -- row split / labels -----------------------------------------------------------

def test_split_row_labels_cross_chunk_boundaries():
    ids = torch.arange(4096)
    xs, ys = mi.split_row(ids, 8, 512)
    assert len(xs) == 8 and all(x.numel() == 512 for x in xs)
    assert int(ys[0][-1]) == int(xs[1][0])          # next chunk's first token
    assert int(ys[-1][-1]) == -100                  # nothing past the row
    assert torch.equal(ys[3][:-1], xs[3][1:])
    xs2, ys2 = mi.split_row(torch.arange(4097), 8, 512)
    assert int(ys2[-1][-1]) == 4096
    with pytest.raises(ValueError):
        mi.split_row(torch.arange(100), 8, 512)


# -- rows / books -----------------------------------------------------------------

def _pack_books(lengths, row_len=8, eos=0):
    """Build a strided pack exactly as tools/prepare_pg19_dataset does."""
    sys.path.insert(0, os.path.join(ROOT, "tools"))
    from prepare_pg19_dataset import stride_windows
    rows, masks = [], []
    for n_tok in lengths:
        ids = [eos] + list(range(1, n_tok))          # BOS (== EOS) + body
        for w, m in stride_windows(ids, row_len, eos):
            rows.append(w)
            masks.append(m)
    return rows, masks


def test_detect_books_on_the_real_packer():
    rows, masks = _pack_books([20, 8, 30], row_len=8, eos=0)
    starts = [r[0] == 0 for r in rows]
    tails = [any(t == 0 for t in r[1:]) for r in rows]
    book = mi.detect_books(starts, tails)
    # 20 tok -> 3 windows (last padded); 8 -> exactly ONE full window with no
    # tail marker (ends on the boundary -- caught by the next book's BOS);
    # 30 -> 4 windows (last padded)
    assert book == [0, 0, 0, 1, 2, 2, 2, 2]
    # and with the BOS removed from the third book, the exact-boundary end is
    # undetectable: books 1 and 2 MERGE (the documented, conservative failure)
    starts[4] = False
    assert mi.detect_books(starts, tails) == [0, 0, 0, 1, 1, 1, 1, 1]


def test_select_rows_spread_takes_one_middle_window_per_book():
    book = {r: r // 5 for r in range(30)}           # 6 books x 5 windows
    usable = list(range(30))
    sel = mi.select_rows(usable, book, 6, "spread")
    assert sel == [2, 7, 12, 17, 22, 27]            # middles, one per book
    sel3 = mi.select_rows(usable, book, 3, "spread")
    assert len({book[r] for r in sel3}) == 3        # evenly spaced books
    sel8 = mi.select_rows(usable, book, 8, "spread")
    assert len(sel8) == 8 and len(set(sel8)) == 8
    assert sel8[:6] == [2, 7, 12, 17, 22, 27]       # round 2 only after round 1
    assert mi.select_rows(usable, book, 4, "first") == [0, 1, 2, 3]
    with pytest.raises(ValueError):
        mi.select_rows(usable, book, 4, "bogus")


def test_pick_donor_is_always_another_book():
    book = {r: r // 5 for r in range(30)}
    usable = list(range(30))
    for r in usable:
        d = mi.pick_donor(r, usable, book)
        assert book[d] != book[r]
    assert mi.pick_donor(0, [0, 1, 2], {0: 0, 1: 0, 2: 0}) is None


def test_site_names_and_count():
    class P:
        n_pre, n_core, n_coda, T = 4, 6, 4, 8
    s = mi.build_sites(P)
    assert len(s) == 4 + 6 + 8 + 4 + 3
    names = [mi.site_name(x) for x in s]
    assert len(set(names)) == len(names)
    assert "all_Z_only" in names and "iter.t7@all_core" in names


def test_ring_pointer_guard_restores_and_flags():
    buf = PrefixGatedBuffer(4, n_vec=W, n_slots=K, route="ring", fill="grow")
    buf._chunk = 3
    with mi.ring_pointer_guard(buf):
        pass
    assert buf._chunk == 3
    with pytest.raises(RuntimeError):
        with mi.ring_pointer_guard(buf):
            buf._chunk = 7
    assert buf._chunk == 3
