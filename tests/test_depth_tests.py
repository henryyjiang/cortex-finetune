"""Depth tests (v2_buffer_handoff.md Part A) in evals/diag_chunk1_drift.py:
the gate spy / forced write rules on a bare PrefixGatedBuffer, and the
gates / restart / recency / forced parts end to end on a tiny raven model."""
from __future__ import annotations

import argparse
import os
import string
import sys

import pytest
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "evals"))

from test_cortex_eval import VOCAB, _build_raven  # noqa: E402

from cortex_memory.buffers import PrefixGatedBuffer  # noqa: E402
from evals.diag_chunk1_drift import (check_depths, centred_row_cos,  # noqa: E402
                                     gate_spy, part_depth, part_drift)

D, W, K = 8, 4, 16


def _full_buffer():
    torch.manual_seed(0)
    buf = PrefixGatedBuffer(D, n_vec=W, n_slots=K, route="ring", fill="grow",
                            carries_latent=True)
    for p in buf.parameters():   # spread the dim-mean fg on both sides of 0.5
        torch.nn.init.normal_(p, std=0.5)
    with torch.no_grad():
        buf.forget_bias.zero_()
        buf.forget_bias_z.zero_()
    st = None
    for _ in range(K // W):
        st = buf.merge(st, torch.randn(1, W, D), torch.randn(1, W, D))
    return buf, st


def _ring_idx(buf):
    return [((buf._chunk * W) % K + i) % K for i in range(W)]


# -- pure pieces --------------------------------------------------------------

def test_restart_refuses_depths_that_read_before_chunk_1():
    for c in (1, 4, 7):
        with pytest.raises(ValueError, match="before chunk 1"):
            check_depths([c], 4)
    check_depths([8, 16, 32, 64, 128], 4)


def test_restart_refuses_depths_whose_rings_are_rotated():
    with pytest.raises(ValueError, match="rotated"):
        check_depths([10], 4)


def test_centred_row_cos_is_one_on_itself_per_channel():
    a = torch.randn(1, 6, 2 * D)
    out = centred_row_cos(a, a, D)
    assert out["e_cos"] == pytest.approx(1.0) and out["z_cos"] == pytest.approx(1.0)
    assert set(centred_row_cos(a[..., :D], a[..., :D], D)) == {"e_cos"}


@pytest.mark.parametrize("mode", ["fifo", "hardgate"])
def test_forced_modes_leave_untouched_rows_bit_identical(mode):
    buf, st = _full_buffer()
    kept = replaced = 0
    for _ in range(6):
        idx = _ring_idx(buf)
        nv, nl = torch.randn(1, W, D), torch.randn(1, W, D)
        with gate_spy(buf, mode) as gs:
            out = buf.merge(st, nv, nl)
        assert "gate" not in buf.__dict__ and "gate_latent" not in buf.__dict__
        cand = torch.cat([nv, nl], dim=-1)[0]
        for r in range(K):
            if r not in idx:
                assert torch.equal(out[0, r], st[0, r])
        for j, r in enumerate(idx):
            for sl in (slice(0, D), slice(D, 2 * D)):
                new, old = out[0, r, sl], st[0, r, sl]
                if torch.equal(new, cand[j, sl]):
                    replaced += 1
                else:
                    assert mode == "hardgate" and torch.equal(new, old)
                    kept += 1
        assert {"e_fg", "z_fg", "e_replaced", "z_replaced"} <= set(gs.stats)
        st = out
    if mode == "hardgate":
        assert kept and replaced, "the test buffer must exercise both branches"


def test_trained_spy_is_transparent_and_records_both_channels():
    buf, st = _full_buffer()
    nv, nl = torch.randn(1, W, D), torch.randn(1, W, D)
    ptr = buf._chunk
    ref = buf.merge(st, nv, nl)
    buf._chunk = ptr
    with gate_spy(buf, "trained") as gs:
        out = buf.merge(st, nv, nl)
    assert torch.equal(out, ref)
    for ch in ("e", "z"):
        assert 0 < gs.stats[f"{ch}_fg"] < 1 and gs.stats[f"{ch}_write_over_stored"] > 0
        assert 0 <= gs.stats[f"{ch}_fg_ig_gt1"] <= 1
    assert "e_replaced" not in gs.stats


# -- end to end on a tiny raven -----------------------------------------------

NV, KS, CL, NCH = 4, 16, 16, 8
CHUNKS = 14


def _model():
    torch.manual_seed(1234)
    return _build_raven(use_memory=True, memory_slots=0, accum_vecs=NV,
                        latent_carry=True, eos_token_id=VOCAB - 1,
                        prefix_memory="gated", gate_slots=KS, gate_route="ring",
                        gate_init="zero", gate_fill="grow").eval()


def _ds(n_books=3, rows_per_book=4):
    from datasets import Dataset
    g = torch.Generator().manual_seed(0)
    rows = [torch.randint(0, VOCAB - 2, (NCH * CL + 1,), generator=g).tolist()
            for _ in range(n_books * rows_per_book)]
    mask = [[1] * len(r) for r in rows]
    for b in range(n_books):      # a padded last row ends each book
        mask[b * rows_per_book + rows_per_book - 1][-1] = 0
    return Dataset.from_dict({"input_ids": rows, "attention_mask": mask})


def _args(**kw):
    a = dict(n_chunks=NCH, drift_chunks=CHUNKS, drift_streams=3, seed=7, boot=50,
             depths="8,12", data="tiny")
    a.update(kw)
    return argparse.Namespace(**a)


def _vocab_text():
    letters = string.ascii_lowercase
    return [letters[i // 26 % 26] + letters[i % 26] + "x" for i in range(VOCAB)]


@pytest.fixture(scope="module")
def tiny():
    m = _model()
    return m, m.cortex.prefix, _ds(), torch.tensor([4, 0]), torch.device("cpu")


def test_drift_output_is_unchanged_when_the_new_parts_are_off(tiny):
    m, buf, ds, ns, dev = tiny
    plain = part_drift(m, buf, ds, _args(), ns, dev, buf.hidden_size)
    spied = part_drift(m, buf, ds, _args(), ns, dev, buf.hidden_size, "trained")
    assert "gate_mode" not in plain
    for p, s in zip(plain["per_chunk"], spied["per_chunk"]):
        assert not any(k.endswith(("_fg", "_ig", "_write_over_stored")) for k in p)
        assert {k: v for k, v in s.items() if k in p} == p
    assert any("z_fg" in r for r in spied["per_chunk"][4:])


@pytest.mark.parametrize("mode", ["fifo", "hardgate"])
def test_forced_drift_runs_and_leaves_nocarry_paired(tiny, mode):
    m, buf, ds, ns, dev = tiny
    plain = part_drift(m, buf, ds, _args(), ns, dev, buf.hidden_size)
    forced = part_drift(m, buf, ds, _args(), ns, dev, buf.hidden_size, mode)
    for p, f in zip(plain["per_chunk"], forced["per_chunk"]):
        assert f["nll_nocarry"] == p["nll_nocarry"]
        if p["chunk"] <= KS // NV + 1:      # the ring fills before any gate acts
            assert f["nll_carry"] == p["nll_carry"]
    assert forced["per_chunk"][-1]["e_replaced"] >= 0


def test_restart_and_recency_pair_with_the_drift(tiny):
    m, buf, ds, ns, dev = tiny
    plain = part_drift(m, buf, ds, _args(), ns, dev, buf.hidden_size)
    rs, rc = part_depth(m, buf, ds, _args(), ns, dev, buf.hidden_size,
                        True, True, _vocab_text())
    for c in ("8", "12"):
        d = plain["per_chunk"][int(c) - 1]
        assert rs["depths"][c]["nll_long"]["mean"] == pytest.approx(d["nll_carry"], abs=1e-6)
        assert rs["depths"][c]["nll_nocarry"]["mean"] == pytest.approx(d["nll_nocarry"], abs=1e-6)
    # at c = 8 the fresh memory reads exactly the long one's chunks
    d8 = rs["depths"]["8"]
    for key in ("fresh_minus_long_gain", "fresh_z_recovers", "fresh_e_recovers"):
        assert d8[key]["mean"] == pytest.approx(0.0, abs=1e-6)
    assert d8["e_cos"]["mean"] == pytest.approx(1.0, abs=1e-5)
    for c in ("8", "12"):
        assert rc["depths"][c]["n_recent"] > 0 and rc["depths"][c]["n_old"] > 0
    assert rc["depths"]["12"]["n_ctl"] > 0
    assert rc["depths"]["12"]["recent_lift"]["n"] == 3


def test_restart_refuses_c_le_7_on_the_model(tiny):
    m, buf, ds, ns, dev = tiny
    with pytest.raises(ValueError, match="before chunk 1"):
        part_depth(m, buf, ds, _args(depths="4,8"), ns, dev, buf.hidden_size,
                   True, False)


def test_contents_saves_and_decodes_every_e_row(tiny, tmp_path):
    from evals.diag_chunk1_drift import part_contents
    m, buf, ds, ns, dev = tiny
    plain = part_drift(m, buf, ds, _args(), ns, dev, buf.hidden_size)
    out = part_contents(m, m, buf, ds, _args(content_depths="2,9,14",
                                             out_dir=str(tmp_path)),
                        ns, dev, buf.hidden_size, _vocab_text())
    saved = torch.load(tmp_path / "contents_e.pt")
    assert saved["E"]["0_2"].shape == (NV, buf.hidden_size)      # one chunk written
    assert saved["E"]["0_14"].shape == (KS, buf.hidden_size)     # full ring
    text = (tmp_path / "contents_e.txt").read_text(encoding="utf-8")
    assert text.count("\n  row ") == 3 * (NV + 2 * KS)
    d14 = out["depths"]["14"]
    assert 1 <= d14["top1_distinct"] <= KS and 0 <= d14["rare_own_p10"] <= 1
    assert 0 < d14["attractor_share"] <= 1 and 0 <= d14["xbook_jaccard"] <= 1
    assert d14["e_centred_cosine"] is not None
    # same chain as the drift: its e_row_norm at chunk 14 is these rows' mean norm
    assert d14["row_norm"] == pytest.approx(plain["per_chunk"][13]["e_row_norm"], rel=1e-5)
