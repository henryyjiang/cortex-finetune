"""tools/oracle_overlay.py: parent weights + the oracle's trained reader, and a
refusal when the freeze did not hold."""
import os
import sys

import pytest
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "tools"))

from oracle_overlay import build  # noqa: E402

R = "cortex.latent_reader."


def _parent():
    return {"a.w": torch.ones(3), "cortex.prefix.x": torch.zeros(2),
            R + "gate": torch.tensor([0.1]), R + "q_proj.weight": torch.zeros(2, 2)}


def test_reader_replaced_rest_kept():
    reader = {R + "gate": torch.tensor([0.014]), R + "q_proj.weight": torch.ones(2, 2)}
    out = build(_parent(), reader, {"a.w": torch.ones(3)})
    assert float(out[R + "gate"]) == pytest.approx(0.014)
    assert torch.equal(out[R + "q_proj.weight"], torch.ones(2, 2))
    assert torch.equal(out["a.w"], torch.ones(3))


def test_moved_frozen_tensor_refused():
    with pytest.raises(RuntimeError, match="FROZEN TENSOR MOVED"):
        build(_parent(), {R + "gate": torch.tensor([0.0])},
              {"a.w": torch.full((3,), 2.0)})


def test_no_reader_refused():
    with pytest.raises(RuntimeError, match="latent_reader"):
        build(_parent(), {}, {"a.w": torch.ones(3)})


def test_shape_mismatch_refused():
    with pytest.raises(RuntimeError, match="shape"):
        build(_parent(), {R + "q_proj.weight": torch.ones(3, 3)},
              {"a.w": torch.ones(3)})


def test_end_to_end_files(tmp_path):
    import subprocess
    from safetensors.torch import save_file
    parent = tmp_path / "parent"; parent.mkdir()
    torch.save({"model": _parent(), "optimizer": {}}, parent / "chkpt.pt")
    run = tmp_path / "run"; oracle = run / "final_checkpoint"; oracle.mkdir(parents=True)
    sd = {k: v.clone() for k, v in _parent().items()}
    sd[R + "gate"] = torch.tensor([0.014])
    save_file(sd, str(oracle / "model.safetensors"))
    r = subprocess.run([sys.executable, os.path.join(REPO, "tools", "oracle_overlay.py"),
                        "--parent", str(parent), "--oracle", str(oracle)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    out = torch.load(run / "eval_overlay" / "chkpt.pt", weights_only=False)
    assert float(out["model"][R + "gate"]) == pytest.approx(0.014)
    assert out["oracle_overlay"]["reader_gate"] == pytest.approx(0.014)
