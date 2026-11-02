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


def test_lownorm_moves_both_channels_and_is_read_as_itself():
    """A lownorm limb that lost either flag would train (or be scored as) a
    half-rescaled carry under the limb's name."""
    s = open(os.path.join(ROOT, "pace/pm_limbs.sbatch"), encoding="utf-8").read()
    assert "LIMBS=(both eonly zonly edrop recall recall_ctrl lownorm carrydrop v2capwarm)" in s
    assert '[ "$LIMB" = "lownorm" ] && ZNORM_TARGET=$LOWNORM_TARGET' in s
    assert "--cortex.e_read_norm_target $LOWNORM_TARGET" in s
    # the switch must precede the ARM_ARGS line that consumes ZNORM_TARGET
    assert s.index("ZNORM_TARGET=$LOWNORM_TARGET") < \
        s.index("--cortex.latent_read_znorm_target $ZNORM_TARGET")
    r = open(os.path.join(ROOT, "pace/pm_limb_readout.sbatch"), encoding="utf-8").read()
    assert r.count("lownorm") >= 4
    assert "--set e_read_norm_target=$LOWNORM_TARGET" in r
    assert "--set latent_read_znorm_target=$LOWNORM_TARGET" in r
    # same default target on both sides
    pat = r"LOWNORM_TARGET=\$\{LOWNORM_TARGET:-([0-9.]+)\}"
    assert re.findall(pat, s) == re.findall(pat, r) != []


def test_carrydrop_trains_its_flag_and_is_read_against_both():
    """carrydrop must train with carry_dropout (and nothing else changed), the
    gate must check the measured drop, and round 2 must run the MC suite on it
    -- the benchmark gap is the whole point of the limb."""
    s = open(os.path.join(ROOT, "pace/pm_limbs.sbatch"), encoding="utf-8").read()
    assert "--cortex.carry_dropout $CARRYDROP_P" in s
    assert "CARRYDROP_P=${CARRYDROP_P:-0.5}" in s
    assert 'vals("z_carry_dropped")' in s
    r = open(os.path.join(ROOT, "pace/pm_limb_readout.sbatch"), encoding="utf-8").read()
    limbs = re.search(r"^LIMBS=\((.*)\)$", r, re.M).group(1).split()
    assert limbs.index("carrydrop") == 9 and limbs.index("both") == 0
    assert '[ "$LIMB" = "carrydrop" ]; then\n    step "MC suite' in r


def test_v2capwarm_trains_all_three_changes_and_its_gate_checks_them():
    """v2capwarm = gated_v2 (paired) + the E store cap + the warm-up, appended
    on its READ line so they win over the shared ARM_ARGS; the gate must read
    each one back from the measured diagnostics."""
    s = open(os.path.join(ROOT, "pace/pm_limbs.sbatch"), encoding="utf-8").read()
    line = re.search(r"^\s*v2capwarm\) READ=\"(.*)\" ;;$", s, re.M).group(1)
    for flag in ("--cortex.prefix_memory gated_v2", "--cortex.v2_pairing paired",
                 "--cortex.e_store_norm_target $E_CAP", "--cortex.warm_p $WARM_P",
                 "--cortex.warm_max_chunks $WARM_MAX"):
        assert flag in line, flag
    assert "E_CAP=${E_CAP:-82}" in s and "WARM_P=${WARM_P:-0.75}" in s
    for key in ('vals("v2_replace_rate")', 'vals("warm_share")', "e_store_norm_target never"):
        assert key in s, key


def test_v2_long_reuses_limb_8_and_stops_itself_after_the_timeout_save():
    w = open(os.path.join(ROOT, "pace/v2_long.sbatch"), encoding="utf-8").read()
    s = open(os.path.join(ROOT, "pace/pm_limbs.sbatch"), encoding="utf-8").read()
    limbs = re.search(r"^LIMBS=\((.*)\)$", s, re.M).group(1).split()
    idx = int(re.search(r"SLURM_ARRAY_TASK_ID=(\d+)", w).group(1))
    assert limbs[idx] == "v2capwarm"
    hours = int(re.search(r"#SBATCH -t (\d+):00:00", w).group(1))
    assert f"WALL_SECS=$(( {hours} * 3600 ))" in w
    assert "exec bash pace/pm_limbs.sbatch" in w and "WALL_LIMITED=1" in w
    # a limb-only name: final5b_shared defaults SAVE_INTERVAL to 300
    assert "--save_interval $LIMB_SAVE_INTERVAL" in s
    assert "LIMB_SAVE_INTERVAL=${LIMB_SAVE_INTERVAL:--1}" in s
    assert "LIMB_SAVE_INTERVAL" in w and "export SAVE_INTERVAL" not in w
    # python must stop after the 20-min timeout save and before the wall
    assert "--save_n_mins_before_timeout 20" in s
    assert "$(( WALL_SECS - SECONDS - 480 ))" in s
    assert '[ -n "${WALL_LIMITED:-}" ] && [ $RC -eq 124 ] && RC=0' in s
    assert 'CHAIN="--resume_path $RESUME_PATH"' in s


def test_the_readout_scores_v2capwarm_as_it_trained_and_pm_both_the_same_way():
    r = open(os.path.join(ROOT, "pace/pm_limb_readout.sbatch"), encoding="utf-8").read()
    s = open(os.path.join(ROOT, "pace/pm_limbs.sbatch"), encoding="utf-8").read()
    limbs = re.search(r"^LIMBS=\((.*)\)$", r, re.M).group(1).split()
    assert limbs.index("v2capwarm") == 10 and limbs.index("both") == 0
    assert '--set prefix_memory=gated_v2 --set v2_pairing=paired' in r
    assert '--set e_store_norm_target=$E_CAP' in r
    pat = r"E_CAP=\$\{E_CAP:-([0-9.]+)\}"
    assert re.findall(pat, r) == re.findall(pat, s) != []
    # depth + BABILong run on v2capwarm, and on pm-both via COMPARE_ONLY
    assert "--parts drift,gates,restart,recency" in r
    assert "--tasks qa1 qa2" in r and "--seq_len 512 --num_chunks 8" in r
    assert '[ "$LIMB" = "v2capwarm" ] && depth_babilong' in r
    assert 'if [ "${COMPARE_ONLY:-}" = "1" ]; then' in r
    assert r.index("depth_babilong() {") < r.index('if [ "${COMPARE_ONLY:-}" = "1" ]; then')
