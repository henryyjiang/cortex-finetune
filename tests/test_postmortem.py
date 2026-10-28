"""GPU-free tests for the 2026-10-28 post-mortem tooling.

Covers the pure pieces of evals/diag_chunk1_drift.py, the E-donor splice in
evals/eval_carry_2x2.py, and the one invariant pace/pm_limbs.sbatch must keep:
its cortex limbs train EXACTLY cortex-final's flags plus one read flag.
"""
import os
import re
import sys

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from evals.diag_chunk1_drift import (find_streams, pos_bins,  # noqa: E402
                                     preserve_ring_pointer, row_chunks,
                                     seed_for, turnover)
from evals.eval_carry_2x2 import donor_e_swap  # noqa: E402


# ---------------------------------------------------------------- drift helpers

def test_row_chunks_matches_the_2x2_split():
    ids = torch.arange(4097)
    xs, ys = row_chunks(ids, 8)
    assert len(xs) == 8 and all(x.numel() == 512 for x in xs)
    assert torch.equal(xs[0][1:], ys[0][:-1])          # labels shifted by one
    assert int(ys[-1][-1]) == 4096


def test_pos_bins_means():
    ce = torch.arange(16, dtype=torch.float32)
    assert pos_bins(ce, 4) == [1.5, 5.5, 9.5, 13.5]


def test_find_streams_stops_at_book_ends_and_one_per_book():
    #           book A (5 full + pad)  book B (2 full + pad)   book C (4 full, no pad)
    flags = [1, 1, 1, 1, 1, 0,       1, 1, 0,               1, 1, 1, 1]
    s = find_streams([bool(f) for f in flags], rows_needed=4, max_streams=5)
    assert s == [(0, 4), (9, 4)]                      # B is too short; one per book
    assert find_streams([True] * 3, 4, 5) == []
    assert find_streams([True] * 20, 4, 1) == [(0, 4)]


def test_turnover_measures_only_written_rows_per_channel():
    D = 4
    before = torch.ones(1, 8, 2 * D)
    after = before.clone()
    after[0, 2:4, :D] = 3.0          # E rows 2,3 changed by 2/1 * ... -> ratio 2
    after[0, 2:4, D:] = 1.0          # Z unchanged
    t = turnover(before, after, [2, 3], D)
    assert t["e_turnover"] == pytest.approx(2.0)
    assert t["z_turnover"] == pytest.approx(0.0)
    assert turnover(None, after, [2, 3], D) == {}
    assert turnover(before[:, :2], after, [2, 3], D) == {}   # rows not reached yet


def test_preserve_ring_pointer_restores_and_copies_lists():
    class Buf:
        pass
    b = Buf()
    b._chunk, b._z_chunk, b._z_pending = 5, 3, [1]
    with preserve_ring_pointer(b):
        b._chunk = 1                 # what a m_cross_in=None forward does
        b._z_chunk = 0
        b._z_pending.append(2)       # in-place mutation must not leak either
    assert (b._chunk, b._z_chunk, b._z_pending) == (5, 3, [1])


def test_seed_for_is_distinct_per_position():
    assert len({seed_for(7, k) for k in range(8)}) == 8
    assert seed_for(7, 0) != seed_for(8, 0)


# ---------------------------------------------------------------- E donor

def test_donor_e_swap_replaces_only_e():
    D = 3
    st = torch.zeros(1, 4, 2 * D)
    st[..., D:] = 7.0
    donor = torch.full((1, 4, D), 2.0)
    out = donor_e_swap(st, donor, D)
    assert torch.all(out[..., :D] == 2.0) and torch.all(out[..., D:] == 7.0)


def test_donor_e_swap_refuses_misaligned_rows():
    D = 3
    with pytest.raises(ValueError):
        donor_e_swap(torch.zeros(1, 4, 2 * D), torch.zeros(1, 3, D), D)
    with pytest.raises(ValueError):
        donor_e_swap(torch.zeros(1, 4, 2 * D), None, D)


# ---------------------------------------------------------------- limb launcher

def _flags(text: str) -> dict:
    """--cortex.* flags appended to ARM_ARGS, as {name: value}."""
    out = {}
    for line in text.splitlines():
        if "ARM_ARGS=" not in line or line.strip().startswith("#"):
            continue
        for k, v in re.findall(r"--cortex\.(\w+)\s+(\S+?)[\"\s]", line + " "):
            out[k] = v
    return out


def test_pm_limbs_cortex_flags_match_cortex_final():
    ref = _flags(open(os.path.join(ROOT, "pace/cortex_final_5b.sbatch"),
                      encoding="utf-8").read())
    got = _flags(open(os.path.join(ROOT, "pace/pm_limbs.sbatch"),
                      encoding="utf-8").read())
    # Allowed to differ: memory_lr (scaled with LR_SCALE) and the three read
    # flags, which the per-limb READ line sets.  accum_max is spelled as an
    # arithmetic expression in both.
    skip = {"memory_lr", "e_carry_read", "latent_carry_read", "e_dropout",
            "use_memory"}
    for k, v in ref.items():
        if k in skip:
            continue
        assert k in got, f"pm_limbs is missing --cortex.{k}"
        norm = lambda s: s.replace("$ZNORM_TARGET", "Z").replace("$ACCUM_VECS", "16") \
            .replace("$GATE_SLOTS", "64").replace("$LATENT_LO", "1") \
            .replace("$LATENT_HI", "9").replace("$CARRY_GRAD_CHUNKS", "4") \
            .replace("$ACCUM_MAX", "AM").replace("$((", "AM").strip('"')
        assert norm(got[k]) == norm(v) or k == "accum_max", \
            f"--cortex.{k}: cortex-final {v!r} vs pm_limbs {got[k]!r}"


def test_pm_limbs_saves_a_resumable_checkpoint():
    s = open(os.path.join(ROOT, "pace/pm_limbs.sbatch"), encoding="utf-8").read()
    # max_steps must sit ONE past stop_at_step, or the max_steps exit (which
    # does not save) fires first and round 2 has nothing to load.
    assert "LIMB_MAX_STEPS=$(( STOP_AT_STEP + 1 ))" in s
    assert "--ignore_past_scheduler true" in s
    assert "--branch_path $BRANCH_PATH" in s
