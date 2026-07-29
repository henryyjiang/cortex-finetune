"""
Tests for the walltime-left probe behind --save_n_mins_before_timeout.

Upstream's `get_flux_timeleft` shells out to `flux job timeleft` (LLNL Flux).
On PACE there is no `flux` binary, so `subprocess.run(..., check=True)` raised
FileNotFoundError and killed the run at the first step that checked — the flag
was unusable here.  train.py now dispatches on $SLURM_JOB_ID and parses
`squeue -h -j <id> -o %L`.  B1 is a resume chain across 48h walltimes, so a
pre-timeout checkpoint save is what keeps a killed job from losing every step
since the last periodic save.

`parse_slurm_timeleft` is MIRRORED below: train.py imports torch/lightning and
will not import off-cluster, so per repo convention (see test_signal_fixes.py,
test_final_arms.py) the pure function is duplicated here.  KEEP IN SYNC with
train.py.

Run: /c/Users/henry/miniconda3/envs/cortex/python.exe -m pytest tests/test_timeleft.py -v
"""
from __future__ import annotations

import pytest


# --- mirror of train.py:parse_slurm_timeleft -------------------------------
def parse_slurm_timeleft(raw: str) -> int:
    """Seconds from a Slurm `squeue -o %L` duration: [[D-]HH:]MM:SS."""
    raw = raw.strip()
    days, _, rest = raw.rpartition("-")
    parts = [int(p) for p in rest.split(":")]
    while len(parts) < 3:
        parts.insert(0, 0)
    h, m, s = parts
    return ((int(days) if days else 0) * 24 + h) * 3600 + m * 60 + s
# ---------------------------------------------------------------------------


class TestParseSlurmTimeleft:
    """Every duration shape `squeue -o %L` emits."""

    @pytest.mark.parametrize("raw,expected", [
        ("0:30", 30),                      # MM:SS  (under a minute left)
        ("12:00", 720),                    # MM:SS
        ("59:59", 3599),                   # MM:SS, max before rolling to HH
        ("1:00:00", 3600),                 # HH:MM:SS
        ("47:59:59", 172799),              # HH:MM:SS, just under a 48h wall
        ("1-00:00:00", 86400),             # D-HH:MM:SS
        ("2-13:04:05", 219845),            # D-HH:MM:SS
    ])
    def test_shapes(self, raw, expected):
        assert parse_slurm_timeleft(raw) == expected

    def test_strips_whitespace(self):
        assert parse_slurm_timeleft("  1:00:00\n") == 3600

    def test_monotonic_across_shape_boundary(self):
        """A shorter shape must never outrank a longer one."""
        assert (parse_slurm_timeleft("59:59")
                < parse_slurm_timeleft("1:00:00")
                < parse_slurm_timeleft("1-00:00:00"))


class TestSaveTrigger:
    """The comparison check_if_save() makes, at the B1 setting (20 min)."""

    THRESHOLD_S = 20 * 60

    def test_fires_inside_window(self):
        assert self.THRESHOLD_S > parse_slurm_timeleft("19:59")

    def test_silent_outside_window(self):
        assert not self.THRESHOLD_S > parse_slurm_timeleft("20:01")

    def test_silent_early_in_a_48h_job(self):
        assert not self.THRESHOLD_S > parse_slurm_timeleft("47:30:00")
