"""
Tests for the A0.1 oldest-vs-newest slice ablation (two-track plan):

  ablate_vec_slice      — pure helper in cortex_memory/chunking.py
  eval_carry_ablation   — the eval's chunk_losses reconstruction logic run
                          against the FakeRaven accum harness (real code)

Run: /c/Users/henry/miniconda3/envs/cortex/python.exe -m pytest tests/test_slice_ablation.py -v
"""
from __future__ import annotations

import os
import sys

import pytest
import torch
import torch.nn as nn

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "evals"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cortex_memory.chunking import ablate_vec_slice
from test_cortex_graft import B, H, S, FakeRaven, _ids

NV = 3


# ---------------------------------------------------------------------------
# ablate_vec_slice (pure)
# ---------------------------------------------------------------------------

class TestAblateVecSlice:

    def _state(self, n_rows=9):
        # rows numbered 1..n so provenance is checkable (and no row is 0)
        return torch.arange(1, n_rows + 1, dtype=torch.float32) \
                    .view(1, n_rows, 1).expand(B, n_rows, H).clone()

    def test_drop_oldest(self):
        out = ablate_vec_slice(self._state(), NV, "oldest", "drop")
        assert out.shape == (B, 6, H)
        assert float(out[0, 0, 0]) == NV + 1      # rows [NV:] survive

    def test_drop_newest(self):
        out = ablate_vec_slice(self._state(), NV, "newest", "drop")
        assert out.shape == (B, 6, H)
        assert float(out[0, -1, 0]) == 6          # rows [:-NV] survive

    def test_zero_oldest_keeps_shape(self):
        out = ablate_vec_slice(self._state(), NV, "oldest", "zero")
        assert out.shape == (B, 9, H)
        assert torch.all(out[:, :NV] == 0)
        assert torch.all(out[:, NV:] == self._state()[:, NV:])

    def test_zero_newest_keeps_shape(self):
        out = ablate_vec_slice(self._state(), NV, "newest", "zero")
        assert out.shape == (B, 9, H)
        assert torch.all(out[:, -NV:] == 0)

    def test_n_zero_is_identity(self):
        s = self._state()
        assert ablate_vec_slice(s, 0, "oldest", "drop") is s

    def test_none_state_passthrough(self):
        assert ablate_vec_slice(None, NV, "oldest", "drop") is None

    def test_drop_everything_returns_empty(self):
        out = ablate_vec_slice(self._state(NV), NV, "oldest", "drop")
        assert out.shape == (B, 0, H)

    def test_zero_everything_returns_zeros(self):
        out = ablate_vec_slice(self._state(NV), 2 * NV, "newest", "zero")
        assert out.shape == (B, NV, H) and torch.all(out == 0)

    def test_original_not_mutated_by_zero(self):
        s = self._state()
        ablate_vec_slice(s, NV, "oldest", "zero")
        assert torch.all(s == self._state())      # helper clones before zeroing


# ---------------------------------------------------------------------------
# eval chunk_losses reconstruction (real eval code on FakeRaven)
# ---------------------------------------------------------------------------

def _accum_model(active_read=True):
    m = FakeRaven(use_memory=True, memory_slots=0, accum_ccot=True,
                  accum_vecs=NV, accum_max=24)
    if active_read:
        with torch.no_grad():
            nn.init.normal_(m.cortex.accum.out_proj.weight, std=0.05)
    return m


def _sample(n_chunks=4):
    ids = _ids(b=1, s=S * n_chunks)[0]
    x, y = ids[:-1], ids[1:]
    # pad x/y to a chunkable length: chunk_losses torch.chunk's them
    x = torch.cat([x, x[:1]])
    y = torch.cat([y, y[:1]])
    ymask = torch.ones_like(y, dtype=torch.float)
    return x, y, ymask


class TestEvalSliceReconstruction:

    def setup_method(self, _):
        torch.manual_seed(0)
        self.sample = _sample()           # ONE sample, shared across conditions

    def _losses(self, model, cond, slice_n, seed=7, n_chunks=4):
        from eval_carry_ablation import chunk_losses
        x, y, ym = self.sample
        return chunk_losses(model, x, y, ym, n_chunks, (1, 1), cond, seed,
                            torch.device("cpu"), n_vec=NV, slice_n=slice_n)

    def test_slice_n_zero_equals_carried(self):
        """slice_n=0 makes the ablation a no-op, so the ablated condition must
        reproduce carried EXACTLY — this exercises the true-state
        reconstruction path (re-appending the returned state's new rows): any
        reconstruction bug diverges the later chunks."""
        model = _accum_model()
        torch.manual_seed(0)
        rc = self._losses(model, "carried", 0)
        ro = self._losses(model, "oldest", 0)
        rn = self._losses(model, "newest", 0)
        for (lc, _), (lo, _), (ln, _) in zip(rc, ro, rn):
            assert lc == pytest.approx(lo, abs=1e-6)
            assert lc == pytest.approx(ln, abs=1e-6)

    def test_chunk2_oldest_full_drop_equals_zeroed(self):
        """At chunk 2 the state holds exactly one chunk's vectors, so dropping
        the oldest NV == dropping everything == the zeroed condition."""
        model = _accum_model()
        rz = self._losses(model, "zeroed", NV)
        ro = self._losses(model, "oldest", NV)
        assert ro[1][0] == pytest.approx(rz[1][0], abs=1e-6)

    def test_chunk1_identical_across_conditions(self):
        model = _accum_model()
        rc = self._losses(model, "carried", NV)
        rz = self._losses(model, "zeroed", NV)
        ro = self._losses(model, "oldest", NV)
        assert rc[0][0] == pytest.approx(rz[0][0], abs=1e-6)
        assert rc[0][0] == pytest.approx(ro[0][0], abs=1e-6)

    def test_ablation_changes_later_chunks_when_read_active(self):
        """With an active read, dropping the newest vectors must change
        chunk >= 3 losses vs carried (the ablated view differs)."""
        model = _accum_model()
        rc = self._losses(model, "carried", NV)
        rn = self._losses(model, "newest", NV)
        diffs = [abs(rc[i][0] - rn[i][0]) for i in (2, 3)]
        assert max(diffs) > 1e-8
