"""
P0.7 — the depth-band invariance probe (`evals/diag_depth_band.py`).

This probe's whole output is a VERDICT, and a verdict is worth exactly as much
as the scoring rule behind it.  So what is pinned here is the scoring: that a
synthetic trajectory with a known answer gets that answer, that a genuinely
ambiguous one is reported as INCONCLUSIVE instead of a coin flip, and that the
band criteria reproduce P0.1's published 2..9 rather than having been tuned to
whatever this checkpoint happens to do.

Run: /c/Users/henry/miniconda3/envs/cortex-retro/python.exe -m pytest tests/ -q
"""
from __future__ import annotations

import math
import os
import sys

import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from test_cortex_eval import VOCAB, _build_raven  # noqa: E402

from evals.diag_depth_band import (  # noqa: E402
    BAND_LO, MAG_FLOOR, MARGIN, NOVELTY_FLOOR, band_edges, cv, score_rules,
    slot_depth_map, trajectory_stats)


def _rows(edges_at, T, cos_at=None):
    """A synthetic per-step table whose magnitude band ends at `edges_at`."""
    out = []
    for t in range(1, T + 1):
        mag = 1.0 if t <= edges_at else 0.1          # straddles MAG_FLOOR = 0.5
        cos = 0.0 if (cos_at is None or t <= cos_at) else -0.95
        out.append({"t": t, "d_over_s0": mag,
                    "cos_delta": None if t == 1 else cos, "s_norm": 1.0})
    return out


class TestBandCriteriaReproduceP01:
    """The criteria were fixed to reproduce the published band, not tuned."""

    def test_magnitude_floor_puts_the_edge_at_9_on_p01s_numbers(self):
        # P0.1, fp32, T=32: d/s0 by t.  0.90 at t=8, 0.45 at t=10 -- so a floor
        # of 0.5 lands the edge at 9, which is the published band's top.
        p01 = {1: 23.8, 2: 12.8, 3: 7.5, 4: 4.6, 5: 3.0, 6: 1.9, 7: 1.3,
               8: 0.90, 9: 0.62, 10: 0.45, 12: 0.20, 16: 0.11, 24: 0.07,
               32: 0.06}
        rows = [{"t": t, "d_over_s0": v, "cos_delta": 0.0, "s_norm": 1.0}
                for t, v in sorted(p01.items())]
        assert band_edges(rows)["t_hi_magnitude"] == 9

    def test_novelty_floor_sits_between_p01s_measured_neighbours(self):
        """-0.63 at t=10, -0.87 at t=16.  The floor has to separate them or it
        is not measuring the transition it claims to."""
        assert -0.87 < NOVELTY_FLOOR < -0.63

    def test_d1_is_excluded_by_construction(self):
        """d_1 was 23.8x ||s0|| -- a different regime.  If the lower edge ever
        stopped being fixed at 2, every band would start at 1 and the verdict
        would be measuring the init transient."""
        assert BAND_LO == 2
        rows = _rows(edges_at=1, T=8)            # only t=1 is above the floor
        assert band_edges(rows)["t_hi_magnitude"] is None


class TestCensoring:
    """An edge that runs past the measured range is a lower bound, and saying
    so is the difference between a result and an artefact of the sweep."""

    def test_an_uncrossed_criterion_reports_none_not_T(self):
        rows = _rows(edges_at=0, T=16)
        e = band_edges(rows)
        assert e["t_hi_magnitude"] is None
        assert not e["magnitude_censored"]

    def test_a_band_still_open_at_T_is_flagged_censored(self):
        rows = _rows(edges_at=16, T=16)
        e = band_edges(rows)
        assert e["t_hi_magnitude"] == 16 and e["magnitude_censored"]


class TestScoring:

    def _per_T(self, edges):
        """`edges` maps T -> t_hi.  An edge equal to T means the band was still
        open when the loop ran out, which `band_edges` flags as censored -- so
        the helper derives the flag the same way rather than letting a test
        smuggle in an uncensored short row that cannot occur in real data."""
        return {T: {"t_hi_magnitude": v, "magnitude_censored": v == T}
                for T, v in edges.items()}

    def test_a_fixed_edge_scores_absolute(self):
        sc = score_rules(self._per_T({4: 4, 8: 9, 16: 9, 32: 9}),
                         "t_hi_magnitude")
        assert sc["censored_T"] == [4]           # the loop ran out, not the band
        assert sc["verdict"] == "ABSOLUTE"
        assert sc["cv_absolute"] < sc["cv_relative"]

    def test_a_censored_short_T_does_not_decide_the_verdict(self):
        """THE bias this exclusion exists for.  A truly absolute band of 9 reads
        as 4 at T=4 because the loop ended first.  Counting that row makes a
        constant edge look like a moving one -- and it can only ever hurt the
        ABSOLUTE rule, since a relative edge is never truncated by definition.

        Same data, censoring honoured vs ignored: the verdict flips.
        """
        honoured = score_rules(self._per_T({4: 4, 8: 9, 16: 9, 32: 9}),
                               "t_hi_magnitude")
        ignored = score_rules(
            {T: {"t_hi_magnitude": v, "magnitude_censored": False}
             for T, v in {4: 4, 8: 9, 16: 9, 32: 9}.items()},
            "t_hi_magnitude")
        assert honoured["verdict"] == "ABSOLUTE"
        assert ignored["verdict"] != "ABSOLUTE"
        assert honoured["cv_absolute"] < ignored["cv_absolute"]

    def test_an_all_censored_sweep_gives_no_verdict(self):
        """Every T ended before its band did.  There is nothing to compare, and
        saying so beats reporting the CV of a one-point set."""
        sc = score_rules(self._per_T({4: 4, 8: 8}), "t_hi_magnitude")
        assert sc["T"] == [] and sc["verdict"].startswith("NO VERDICT")

    def test_an_edge_that_tracks_t_over_T_scores_relative(self):
        sc = score_rules(self._per_T({4: 1, 8: 2, 16: 4, 32: 8}),
                         "t_hi_magnitude")
        assert sc["verdict"] == "RELATIVE"

    def test_an_ambiguous_sweep_is_inconclusive_not_a_coin_flip(self):
        """Neither rule wins by the required margin.  Reporting a winner here
        would be the probe inventing the finding it was built to measure."""
        sc = score_rules(self._per_T({4: 3, 8: 5, 16: 7, 32: 9}),
                         "t_hi_magnitude")
        assert sc["verdict"] == "INCONCLUSIVE"

    def test_the_margin_is_what_makes_inconclusive_reachable(self):
        assert MARGIN > 1.0

    def test_missing_edges_are_dropped_rather_than_counted_as_zero(self):
        """A T whose criterion was never crossed contributes NOTHING.  Counting
        it as 0 would drag the absolute CV up and hand the verdict to
        `relative`."""
        sc = score_rules(self._per_T({4: None, 8: 9, 16: 9, 32: 9}),
                         "t_hi_magnitude")
        assert sc["T"] == [8, 16, 32] and sc["verdict"] == "ABSOLUTE"

    def test_a_rule_with_no_points_can_never_win(self):
        assert cv([]) == float("inf")
        assert cv([5.0]) == float("inf")

    def test_the_discriminating_pair_is_reported_separately(self):
        """At T=4 and T=8 a clamped-absolute band and a relative band pick
        nearly the same depths, so the full-sweep CV flatters both.  16 vs 32 is
        the row that actually separates them."""
        sc = score_rules(self._per_T({4: 4, 8: 9, 16: 9, 32: 9}),
                         "t_hi_magnitude")
        p = sc["discriminating_pair"]
        assert p is not None
        assert abs(p["absolute_ratio"] - 1.0) < 1e-9      # absolute: 9 -> 9
        assert abs(p["relative_ratio"] - 0.5) < 1e-9      # not relative

    def test_no_pair_when_the_sweep_omits_16_or_32(self):
        sc = score_rules(self._per_T({4: 4, 8: 9}), "t_hi_magnitude")
        assert sc["discriminating_pair"] is None

    def test_no_pair_when_one_of_16_or_32_is_censored(self):
        """A censored T16 makes the ratio meaningless, and the ratio is the row
        the header tells the reader to trust most."""
        sc = score_rules(self._per_T({8: 7, 16: 16, 32: 9}), "t_hi_magnitude")
        assert sc["discriminating_pair"] is None


class TestSlotDepthMap:
    """What the verdict buys: the map is the deliverable, not the CV."""

    def test_absolute_map_stays_inside_the_band_and_tiles_it(self):
        m = slot_depth_map("absolute", 32, 16, BAND_LO, 9)
        assert min(m) == BAND_LO and max(m) == 9
        assert set(m) == set(range(BAND_LO, 10))

    def test_absolute_map_clamps_when_the_loop_is_shorter_than_the_band(self):
        """The rule-A failure mode named in the header: at T=8 the top of the
        band does not exist, so the map must lump rather than request a depth
        the forward never reaches."""
        m = slot_depth_map("absolute", 8, 16, BAND_LO, 9)
        assert max(m) <= 7 and min(m) >= BAND_LO

    def test_absolute_map_is_identical_at_T16_and_T32(self):
        """This is rule A's cost, stated as a test: the loop's extra depth at
        T=32 goes unused.  If that ever stops being true, rule A has silently
        become rule B."""
        assert (slot_depth_map("absolute", 16, 16, BAND_LO, 9)
                == slot_depth_map("absolute", 32, 16, BAND_LO, 9))

    def test_relative_map_spreads_with_T(self):
        """And rule B's cost: at T=32 it reaches into the region P0.1 calls
        saturated."""
        a = slot_depth_map("relative", 16, 16, BAND_LO, 9)
        b = slot_depth_map("relative", 32, 16, BAND_LO, 9)
        assert max(b) > max(a)

    def test_every_depth_is_reachable_by_the_forward(self):
        for rule in ("absolute", "relative"):
            for T in (2, 4, 8, 16, 32):
                m = slot_depth_map(rule, T, 16, BAND_LO, 9)
                assert all(1 <= k <= T - 1 for k in m), (rule, T, m)

    def test_repeated_depths_are_allowed_and_expected(self):
        """16 slots over an 8-wide band means two columns per depth.  They are
        independent views of that depth -- different sequence columns, different
        trajectories -- not duplicates, and the design says to sample the band
        twice rather than spend rows on saturated deltas."""
        m = slot_depth_map("absolute", 32, 16, BAND_LO, 9)
        assert len(m) == 16 and len(set(m)) == 8


class TestTrajectoryStatsOnARealModel:
    """The stats feed the verdict, so they have to come off the real loop."""

    def test_stats_line_up_with_the_captured_trajectory(self):
        from evals.diag_depth_band import trajectory_stats as ts
        from evals.diag_latent_scale import record_trajectory
        torch.manual_seed(1234)
        m = _build_raven().eval()
        torch.manual_seed(0)
        ids = torch.randint(0, VOCAB - 1, (2, 16))
        s0, traj, _ = record_trajectory(m, ids, 6)
        rows = ts(s0, traj)
        assert [r["t"] for r in rows] == [1, 2, 3, 4, 5, 6]
        assert rows[0]["cos_delta"] is None      # no previous delta at t=1
        assert all(r["cos_delta"] is not None for r in rows[1:])
        assert all(math.isfinite(r["d_over_s0"]) and r["d_over_s0"] >= 0
                   for r in rows)

    def test_deltas_are_measured_against_s0_not_the_running_state(self):
        """d/s0 is the ratio the renorm decision is made on -- against the noise
        Z would REPLACE, not against whatever the state has grown to."""
        s0 = torch.zeros(1, 4, 8)
        s0[..., 0] = 2.0                          # ||s0|| = 2 per token
        traj = [s0 + 1.0 * torch.nn.functional.one_hot(
            torch.tensor(1), 8).float(), ]
        rows = trajectory_stats(s0, traj)
        assert abs(rows[0]["d_over_s0"] - 0.5) < 1e-5
