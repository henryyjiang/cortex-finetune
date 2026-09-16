"""
The eval-side config flags must cover every flag that changes what the graft
BUILDS.

WHY THIS FILE EXISTS.  train.py keeps a persist list -- the cortex keys it
stamps onto config.json so a resume or an eval rebuilds the same buffer.
tools/prepare_eval_checkpoint.py keeps a SECOND list for the eval path, and
until the 2026-09-16 audit the two had silently diverged by ten keys.

The failure is specific and produces no error.  Under `gate_route="ring"` NO
parameter carries a `gate_slots` dimension, so an A3' checkpoint (W=16 / K=64)
loads its gate weights perfectly into a buffer rebuilt at the `gate_slots=0`
default -- K = W = 16, a QUARTER of the read block it trained with.  Every eval
number would then describe a geometry that never existed, and the loss curve
that produced the checkpoint would look fine.  One channel down, `latent_carry`
defaulting to False rebuilds an E-only buffer, drops the Z gate's weights as
unexpected keys, and runs the carry at half width.

So the lists are pinned against each other HERE rather than by inspection.
train.py cannot be imported (it needs wandb, and this suite runs anywhere), so
its list is read out of the source -- which is also the honest thing to check,
since the source is what runs.

Run: /c/Users/henry/miniconda3/envs/cortex-retro/python.exe -m pytest tests/ -q
"""
from __future__ import annotations

import ast
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "tools"))

from prepare_eval_checkpoint import CORTEX_FLAGS  # noqa: E402

#: Keys that only steer TRAINING and have no effect on what the graft builds,
#: so the eval side is right not to carry them.  Listed explicitly: an
#: unexplained absence is how the last divergence survived.
TRAIN_ONLY = {
    "ccot_direct",     # legacy selector, load-path only
    "accum_ccot",      # legacy selector, load-path only
    "gated_accum",     # legacy selector, load-path only
}


def _train_persist_list() -> set[str]:
    """The tuple of keys train.py stamps onto the config, read from source.

    Located structurally (the `for _k in (...)` whose body sets attributes on
    `config`) rather than by line number, so an edit above it does not silently
    make this test pass against the wrong tuple.
    """
    src = open(os.path.join(REPO, "train.py"), encoding="utf-8").read()
    tree = ast.parse(src)
    found: list[set[str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.For) or node.target.__class__ is not ast.Name:
            continue
        if node.target.id != "_k" or not isinstance(node.iter, ast.Tuple):
            continue
        keys = {e.value for e in node.iter.elts
                if isinstance(e, ast.Constant) and isinstance(e.value, str)}
        if "prefix_memory" in keys:
            found.append(keys)
    assert len(found) == 1, (
        f"expected exactly one cortex persist loop in train.py, found "
        f"{len(found)} -- the locator needs updating before this test means "
        f"anything")
    return found[0]


class TestTheTwoListsAgree:

    def test_every_persisted_build_flag_reaches_the_eval_config(self):
        train = _train_persist_list()
        missing = sorted(train - set(CORTEX_FLAGS) - TRAIN_ONLY)
        assert not missing, (
            f"train.py persists {missing} but prepare_eval_checkpoint.py does "
            f"not copy them.  An eval would rebuild a DIFFERENT buffer from "
            f"these weights and say nothing.")

    def test_the_train_only_exemptions_are_really_in_the_train_list(self):
        """An exemption for a key train.py does not persist is a stale
        exemption, and a stale exemption hides the next divergence."""
        train = _train_persist_list()
        stale = sorted(TRAIN_ONLY - train)
        assert not stale, f"TRAIN_ONLY names keys train.py no longer persists: {stale}"

    def test_the_geometry_keys_are_present_by_name(self):
        """Belt and braces on the two families that actually bit: naming them
        explicitly means a future refactor of the locator cannot quietly turn
        this whole file into a no-op."""
        for k in ("gate_slots", "gate_route", "gate_norm", "gate_init",
                  "gate_fill"):
            assert k in CORTEX_FLAGS, f"P1.0 gate key {k} missing"
        for k in ("latent_carry", "latent_depth_rule", "latent_depth_lo",
                  "latent_depth_hi", "latent_renorm"):
            assert k in CORTEX_FLAGS, f"Z channel key {k} missing"

    def test_gate_slots_default_is_the_trap_this_guards(self):
        """The reason a MISSING key is worse than a wrong one: the graft's
        fallback for gate_slots is 0, which means 'K = W' -- a valid buffer,
        just not the trained one."""
        from cortex_graft import CortexMemory  # noqa: F401
        import inspect
        src = inspect.getsource(CortexMemory.__init__)
        assert 'getattr(config, "gate_slots", 0)' in src
        assert 'getattr(config, "latent_carry", False)' in src

    def test_no_duplicate_entries(self):
        assert len(CORTEX_FLAGS) == len(set(CORTEX_FLAGS))
