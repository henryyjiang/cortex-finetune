"""Every eval that can be handed a checkpoint must be able to load an OVERLAY.

THE DEFECT CLASS THIS EXISTS TO PREVENT (RED 12, 2026-09-18).
------------------------------------------------------------
There are two kinds of checkpoint directory in this project and they load by
different paths:

  save_model_only dirs (`final_checkpoint`, `model_only_chkpt_*`) carry
  config.json and load as `--model_name`.

  train.py `checkpoint_<step>` dirs -- and the `_w16` branch dirs sliced from
  them -- hold `chkpt.pt` and NO config.json.  They are OVERLAYS: `--model_name`
  is the graft-prepared BASE, the weights arrive via `--checkpoint`, and the
  BASE config carries NO cortex flags at all (`use_memory` is literally
  '<absent>' on ckpts/olmo-retrofit-cortex).  So the geometry has to be FORCED
  with `--set`, or the graft builds with no prefix buffer.

`pace/eval_write_capacity.sbatch` was the one eval never given `--set`, and it
died 12 s into a real job when submit_pass1.sh handed it an overlay.  That fix
went into ONE file.  Every P2.3 cell checkpoint is an overlay, so any eval
pointed at a cell needs the same plumbing -- and a sweep on 2026-09-19 found
eleven more evals without it.

This test is a STATIC check over the eval sources, so it needs no GPU, no
checkpoint and no torch, and it fails the moment a twelfth eval is written
without the plumbing rather than when someone points it at a cell.

It deliberately checks the WIRING, not the behaviour: that the flag exists,
that the parser value reaches `parse_config_overrides`, and that the result
reaches `load_checkpoint`.  A test of the behaviour would need a model.
"""
from __future__ import annotations

import ast
import os

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVALS = os.path.join(REPO, "evals")

#: Files that legitimately never load a checkpoint of their own.
EXEMPT = {
    "model_utils.py",        # the loader itself
    "download_datasets.py",  # no model
    "flops.py",              # arithmetic only
}


def _sources():
    out = []
    for fn in sorted(os.listdir(EVALS)):
        if not fn.endswith(".py") or fn in EXEMPT or fn.startswith("_"):
            continue
        path = os.path.join(EVALS, fn)
        with open(path, encoding="utf-8") as fh:
            src = fh.read()
        out.append((fn, src, ast.parse(src)))
    return out


def _calls(tree, name):
    return [n for n in ast.walk(tree)
            if isinstance(n, ast.Call)
            and (getattr(n.func, "id", None) == name
                 or getattr(n.func, "attr", None) == name)]


def _loads_a_checkpoint(tree):
    return bool(_calls(tree, "load_checkpoint"))


LOADERS = [(fn, src, tree) for fn, src, tree in _sources()
           if _loads_a_checkpoint(tree)]


def _ids(v):
    return [f[0] for f in v]


class TestEveryLoaderCanTakeAnOverlay:

    def test_there_are_loaders_to_check(self):
        # If this fires, the discovery above broke and every test below is
        # passing vacuously -- which is worse than failing.
        assert len(LOADERS) >= 10, _ids(LOADERS)

    @pytest.mark.parametrize("fn,src,tree", LOADERS, ids=_ids(LOADERS))
    def test_it_registers_a_set_flag(self, fn, src, tree):
        flags = [c.args[0].value for c in _calls(tree, "add_argument")
                 if c.args and isinstance(c.args[0], ast.Constant)]
        assert "--set" in flags, (
            f"{fn} can be handed a checkpoint but has no --set, so it cannot "
            f"force the geometry on an overlay.  See this module's docstring.")

    @pytest.mark.parametrize("fn,src,tree", LOADERS, ids=_ids(LOADERS))
    def test_it_parses_the_overrides_with_the_shared_helper(self, fn, src, tree):
        # Not a dict comprehension at the call site: parse_config_overrides
        # TYPES the values, and a config flag arriving as the string "false"
        # is TRUTHY -- `latent_carry=false` would silently turn an E-only run
        # into a dual-channel one.
        assert _calls(tree, "parse_config_overrides"), (
            f"{fn} has --set but never calls parse_config_overrides, so the "
            f"flag is accepted and ignored -- worse than not having it.")

    @pytest.mark.parametrize("fn,src,tree", LOADERS, ids=_ids(LOADERS))
    def test_the_overrides_reach_load_checkpoint(self, fn, src, tree):
        for call in _calls(tree, "load_checkpoint"):
            kwargs = {k.arg for k in call.keywords}
            assert "config_overrides" in kwargs, (
                f"{fn} parses --set but does not pass config_overrides to "
                f"load_checkpoint, so the geometry is never forced.")

    @pytest.mark.parametrize("fn,src,tree", LOADERS, ids=_ids(LOADERS))
    def test_the_flag_value_is_what_gets_parsed(self, fn, src, tree):
        # Guards the seam between the two: a file could parse something else
        # entirely and still satisfy both checks above.
        for call in _calls(tree, "parse_config_overrides"):
            arg = ast.unparse(call.args[0]) if call.args else ""
            assert arg.endswith(".set"), (
                f"{fn} calls parse_config_overrides({arg}) -- expected the "
                f"parser's own --set value (args.set).")
