"""
Build an eval-loadable `chkpt.pt` for a P3.0 tier-1.5 oracle run.

WHY THIS EXISTS
---------------
The oracle ran 1000 steps with `--save_interval 1000`.  train.py writes a full
`checkpoint_<step>/chkpt.pt` only every 2 * save_interval, so the runs left only
HF-format model-only saves (`final_checkpoint/`, `model_only_chkpt_1000/`).
evals/model_utils.load_checkpoint overlays a `chkpt.pt` and nothing else.

The oracle froze EVERYTHING except `cortex.latent_reader`, so the correct model
is exactly: the PARENT's chkpt.pt model weights + the oracle's trained reader.
This writes that, and before writing it CHECKS the premise -- a sample of
frozen tensors in the oracle's save must equal the parent's bit for bit.  If
they differ, the freeze did not hold and the overlay would silently discard
whatever else trained.

The output is `<oracle_run>/eval_overlay/chkpt.pt` holding {"model": ...,
"oracle_overlay": provenance}.  No optimizer state: it is not resumable and
not named `checkpoint_*`, so no launcher's auto-discovery picks it up.

    python tools/oracle_overlay.py \\
        --parent cortex-retrofit/probe-p1-a2-accum-w16-cc8-z/checkpoint_91952 \\
        --oracle cortex-retrofit/p30-oracle-a2-real/final_checkpoint
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from datetime import datetime

import torch

READER = "cortex.latent_reader."


def _safetensor_files(d: str) -> list[str]:
    files = sorted(glob.glob(os.path.join(d, "*.safetensors")))
    if not files:
        raise FileNotFoundError(f"no *.safetensors under {d}")
    return files


def read_oracle(d: str, frozen_sample: list[str]) -> tuple[dict, dict]:
    """-> (reader tensors, the requested frozen tensors), read lazily."""
    from safetensors import safe_open
    reader, frozen, want = {}, {}, set(frozen_sample)
    for f in _safetensor_files(d):
        with safe_open(f, framework="pt", device="cpu") as h:
            for k in h.keys():
                if k.startswith(READER):
                    reader[k] = h.get_tensor(k)
                elif k in want:
                    frozen[k] = h.get_tensor(k)
    return reader, frozen


def build(parent_sd: dict, reader: dict, frozen: dict) -> dict:
    """Parent weights + trained reader.  Raises if the premise fails."""
    if not reader:
        raise RuntimeError(f"the oracle save has no {READER}* keys -- this is "
                           f"not a tier-1.5 run (or the reader was never built)")
    if not frozen:
        raise RuntimeError("none of the sampled frozen keys were found in the "
                           "oracle save; the freeze cannot be checked")
    for k, v in frozen.items():
        if k not in parent_sd:
            raise RuntimeError(f"{k} is in the oracle save but not the parent")
        if not torch.equal(parent_sd[k].cpu().to(v.dtype), v):
            raise RuntimeError(
                f"FROZEN TENSOR MOVED: {k} differs between parent and oracle.  "
                f"The freeze did not hold, and overlaying only the reader "
                f"would discard what else trained.")
    out = dict(parent_sd)
    for k, v in reader.items():
        if k in out and out[k].shape != v.shape:
            raise RuntimeError(f"{k}: shape {tuple(v.shape)} vs parent "
                               f"{tuple(out[k].shape)}")
        out[k] = v
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--parent", required=True,
                   help="the branch_path the oracle ran from (dir with chkpt.pt)")
    p.add_argument("--oracle", required=True,
                   help="the oracle's HF save dir (final_checkpoint)")
    p.add_argument("--out", default=None,
                   help="default: <oracle run dir>/eval_overlay")
    p.add_argument("--n_check", type=int, default=12,
                   help="frozen tensors compared bit-for-bit before writing")
    a = p.parse_args()

    ck = torch.load(os.path.join(a.parent, "chkpt.pt"), map_location="cpu",
                    weights_only=False, mmap=True)
    parent_sd = ck["model"]
    keys = [k for k in parent_sd if not k.startswith(READER)]
    stride = max(1, len(keys) // a.n_check)
    sample = keys[::stride][:a.n_check]
    sample += [k for k in keys if k.startswith("cortex.")][:4]

    reader, frozen = read_oracle(a.oracle, sample)
    sd = build(parent_sd, reader, frozen)

    gate = reader.get(READER + "gate")
    out = a.out or os.path.join(os.path.dirname(os.path.normpath(a.oracle)),
                                "eval_overlay")
    os.makedirs(out, exist_ok=True)
    prov = dict(parent=a.parent, oracle=a.oracle,
                reader_keys=sorted(reader), frozen_checked=sorted(frozen),
                reader_gate=None if gate is None else float(gate.flatten()[0]),
                when=datetime.now().isoformat(timespec="seconds"))
    torch.save({"model": sd, "oracle_overlay": prov},
               os.path.join(out, "chkpt.pt"))
    print(json.dumps(prov, indent=1))
    print(f"[overlay] {len(reader)} reader tensors on the parent; "
          f"{len(frozen)} frozen tensors matched bit-for-bit -> {out}/chkpt.pt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
