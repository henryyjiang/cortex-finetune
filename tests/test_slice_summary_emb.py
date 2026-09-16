"""
tools/slice_summary_emb.py — the W=32 -> W=16 branch step.

The tool runs once, on the cluster, against a 91,552-step checkpoint, and the
thing it is defending against is SILENT: a dropped key re-seeds summary_emb from
wte[eos] and the arm trains on with a perfectly healthy loss curve.  So the
refusals matter as much as the happy path, and both are pinned here.

Run: /c/Users/henry/miniconda3/envs/cortex-retro/python.exe -m pytest tests/ -q
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "tools"))

W_OLD, D = 32, 48


def _ckpt(tmp_path, *, seeded=True, distinct=True, extra=None):
    """A minimal chkpt.pt with the two keys the tool reads."""
    torch.manual_seed(0)
    w = (torch.randn(W_OLD, D) if distinct
         else torch.randn(1, D).expand(W_OLD, D).contiguous())
    model = {
        "transformer.wte.weight": torch.randn(100, D),
        "cortex.prefix.summary_emb": w,
        "cortex.prefix.summary_seeded": torch.tensor(bool(seeded)),
    }
    model.update(extra or {})
    src = tmp_path / "checkpoint_91552"
    src.mkdir()
    torch.save({"model": model, "optimizer": {}, "scheduler": {},
                "agg_vars_dict": {"optimizer_step": 91552}}, src / "chkpt.pt")
    (src / "config.json").write_text("{}")
    return src


def _run(src, dst, n_vec=16, *extra):
    return subprocess.run(
        [sys.executable, os.path.join(REPO, "tools", "slice_summary_emb.py"),
         "--src", str(src), "--dst", str(dst), "--n_vec", str(n_vec), *extra],
        capture_output=True, text=True, cwd=REPO)


class TestSlice:

    def test_keeps_the_first_rows_verbatim(self, tmp_path):
        """The rows must be the PARENT's, bit for bit.  A tool that re-seeded or
        re-normalised them would pass every shape check and throw away 91,552
        steps of write-path training."""
        src = _ckpt(tmp_path)
        dst = tmp_path / "w16"
        r = _run(src, dst)
        assert r.returncode == 0, r.stdout + r.stderr
        before = torch.load(src / "chkpt.pt", weights_only=False)["model"]
        after = torch.load(dst / "chkpt.pt", weights_only=False)["model"]
        w_new = after["cortex.prefix.summary_emb"]
        assert w_new.shape == (16, D)
        assert torch.equal(w_new, before["cortex.prefix.summary_emb"][:16])

    def test_the_source_checkpoint_is_untouched(self, tmp_path):
        src = _ckpt(tmp_path)
        _run(src, tmp_path / "w16")
        w = torch.load(src / "chkpt.pt", weights_only=False)[
            "model"]["cortex.prefix.summary_emb"]
        assert w.shape == (W_OLD, D)

    def test_it_logs_the_slice(self, tmp_path):
        """'and LOG IT' is half the task: the run log must carry the rank cost,
        because that is P1.0's bet being tested on trained weights."""
        src = _ckpt(tmp_path)
        dst = tmp_path / "w16"
        r = _run(src, dst)
        assert "effective rank retained" in r.stdout
        rec = json.loads((dst / "cortex_slice.json").read_text())
        assert rec["W_old"] == W_OLD and rec["W_new"] == 16
        assert rec["kept"]["rows"] == 16 and rec["dropped"]["rows"] == 16
        assert 0.0 < rec["kept"]["participation_rank"] <= 16.0
        # and it rides on the checkpoint itself, which is what train.py reads
        assert torch.load(dst / "chkpt.pt",
                          weights_only=False)["cortex_slice"]["W_new"] == 16

    def test_sidecar_files_come_along(self, tmp_path):
        """--dst has to be a drop-in --branch_path."""
        src = _ckpt(tmp_path)
        dst = tmp_path / "w16"
        _run(src, dst)
        assert (dst / "config.json").exists()

    def test_refuses_an_unseeded_checkpoint(self, tmp_path):
        """summary_seeded False means the rows are post_init noise.  Slicing
        noise produces a perfectly valid file and an arm with no write path."""
        src = _ckpt(tmp_path, seeded=False)
        r = _run(src, tmp_path / "w16")
        assert r.returncode == 3
        assert not (tmp_path / "w16").exists()

    def test_refuses_when_every_row_is_still_the_eos_seed(self, tmp_path):
        src = _ckpt(tmp_path, distinct=False)
        r = _run(src, tmp_path / "w16")
        assert r.returncode == 3
        assert "never trained" in r.stdout

    def test_refuses_when_another_tensor_carries_the_write_width(self, tmp_path):
        """A future per-write parameter (slot_init under gate_fill='init' is the
        near case) would be left at the parent's width, and the arm would build
        a buffer whose halves disagree."""
        src = _ckpt(tmp_path, extra={
            "cortex.prefix.some_future_per_write_param": torch.randn(W_OLD, D)})
        r = _run(src, tmp_path / "w16")
        assert r.returncode == 3
        assert "some_future_per_write_param" in r.stdout

    def test_refuses_to_widen(self, tmp_path):
        src = _ckpt(tmp_path)
        r = _run(src, tmp_path / "w64", 64)
        assert r.returncode == 2

    def test_refuses_in_place(self, tmp_path):
        src = _ckpt(tmp_path)
        r = _run(src, src)
        assert r.returncode == 2
        w = torch.load(src / "chkpt.pt", weights_only=False)[
            "model"]["cortex.prefix.summary_emb"]
        assert w.shape == (W_OLD, D)

    def test_force_overrides_the_diagnostics(self, tmp_path):
        src = _ckpt(tmp_path, seeded=False)
        dst = tmp_path / "w16"
        r = _run(src, dst, 16, "--force")
        assert r.returncode == 0
        assert json.loads((dst / "cortex_slice.json").read_text())["forced"]


class TestTrainGuard:
    """train.py must refuse a RESUME off a sliced checkpoint: the optimizer
    state in it still carries the parent's [W_old, D] moments."""

    def test_the_marker_is_what_train_py_keys_on(self, tmp_path):
        src = _ckpt(tmp_path)
        dst = tmp_path / "w16"
        _run(src, dst)
        ck = torch.load(dst / "chkpt.pt", weights_only=False)
        assert "cortex_slice" in ck
        assert torch.load(src / "chkpt.pt",
                          weights_only=False).get("cortex_slice") is None

    def test_train_py_reads_the_marker_on_both_paths(self):
        """Cheap source-level pin: the guard must branch on `branch`, or it
        would either block the intended use or wave through the unintended one."""
        src = open(os.path.join(REPO, "train.py"), encoding="utf-8").read()
        i = src.find('sliced = ckpt.get("cortex_slice")')
        assert i > 0, "the slice guard is gone from load_checkpoint"
        block = src[i:i + 1600]
        assert "if not branch:" in block and "raise RuntimeError" in block
