#!/usr/bin/env python
"""
Standalone runner for tests/test_b0_retrofit.py -- no pytest required.

WHY THIS EXISTS: the cortex-retro env on PACE has no pytest, and the B0/B1
parity tests are the ones that CANNOT run in the local `cortex` env (they need
the real raven modeling file, which skips off-cluster on transformers skew).
So the one place they can actually run is the one place pytest is missing.
Installing pytest into a working training env right before a ~96 GPU-hour run
is a risk with no upside; this script is the zero-risk path.

It also does something pytest does NOT, which matters here: **a skip is a
FAILURE**.  Under pytest, `pytest.skip("cannot build raven base ...")` prints
a dot and passes the suite -- exactly the false confidence you do not want
before committing the compute.  Here the stub turns every skip into a loud
error, so "cannot build the model" cannot masquerade as "parity verified".

Usage (login node, cortex-retro env, from ~/cortex-finetune):
    python tools/check_b0_parity.py

Exit 0 = every parity invariant held.  Anything else = do not submit B1.
"""
from __future__ import annotations

import importlib.machinery
import importlib.util
import os
import sys
import traceback
import types

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TESTS = os.path.join(REPO, "tests")


class SkipIsFailure(RuntimeError):
    """Raised in place of pytest.skip -- see the module docstring."""


def _install_pytest_stub() -> None:
    """Minimal pytest surface for the two test modules involved.

    test_b0_retrofit uses @pytest.fixture(scope=...) and pytest.skip;
    test_cortex_eval's _build_raven uses pytest.skip.  Nothing else is
    referenced, so the stub stays this small.  If a future test reaches for
    pytest.mark/parametrize/raises, this will AttributeError loudly rather
    than silently skip it -- which is the behaviour we want.

    __spec__ MUST be a real ModuleSpec, not the None that types.ModuleType
    defaults to.  transformers' optional-dependency probing calls
    importlib.util.find_spec("pytest"), and find_spec raises
    "ValueError: pytest.__spec__ is None" for any module already in
    sys.modules whose spec is None -- which broke the raven import on the
    cluster (2026-07-29).  With a real spec, find_spec succeeds, the follow-up
    importlib.metadata.version("pytest") raises PackageNotFoundError, and the
    caller correctly concludes pytest is unavailable.

    We always stub, even if real pytest happens to be installed: under real
    pytest, @fixture turns `base` into a fixture object that cannot be called
    directly, so mod.base() would break.  Keeping one code path makes this
    runner behave identically everywhere.
    """
    stub = types.ModuleType("pytest")
    stub.__spec__ = importlib.machinery.ModuleSpec("pytest", loader=None)
    stub.__version__ = "0.0.0+cortex-stub"

    def fixture(*args, **kwargs):
        # Support both @fixture and @fixture(scope="module").
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]
        return lambda fn: fn

    def skip(reason: str = "", **kwargs):
        raise SkipIsFailure(reason)

    stub.fixture = fixture
    stub.skip = skip
    sys.modules["pytest"] = stub


def main() -> int:
    _install_pytest_stub()
    for p in (REPO, TESTS):
        if p not in sys.path:
            sys.path.insert(0, p)

    import test_b0_retrofit as mod
    import test_cortex_eval

    # Report WHICH raven config was picked up.  It is not in this repo -- it
    # comes from the sibling recurrent-pretraining checkout locally, or from a
    # prepared checkpoint snapshot on the cluster -- so naming the resolved
    # path is the difference between a diagnosable failure and a guess.
    print(f"raven config: {test_cortex_eval._CONFIG_SRC or 'NOT FOUND'}")

    # The `base` fixture is a plain function under the stub; build it once, the
    # same way scope="module" would.
    try:
        base = mod.base()
    except SkipIsFailure as e:
        print(f"FAIL  could not build the raven base model: {e}")
        print("\nThis is the check skipping silently under pytest.  The graft"
              "\ncannot be verified in this env -- do NOT submit B1.")
        return 2

    classes = [getattr(mod, n) for n in dir(mod)
               if n.startswith("Test") and isinstance(getattr(mod, n), type)]
    passed, failed = [], []

    for cls in sorted(classes, key=lambda c: c.__name__):
        inst = cls()
        methods = sorted(n for n in dir(cls) if n.startswith("test_"))
        print(f"\n{cls.__name__}")
        for name in methods:
            label = f"  {name}"
            try:
                getattr(inst, name)(base)
            except SkipIsFailure as e:
                print(f"{label:<62} SKIPPED->FAIL ({e})")
                failed.append((cls.__name__, name, f"skipped: {e}"))
            except Exception:
                print(f"{label:<62} FAIL")
                failed.append((cls.__name__, name, traceback.format_exc()))
            else:
                print(f"{label:<62} ok")
                passed.append(name)

    print("\n" + "=" * 72)
    print(f"{len(passed)} passed, {len(failed)} failed")
    if failed:
        for cname, name, tb in failed:
            print(f"\n--- {cname}.{name} ---\n{tb}")
        print("Do NOT submit B1 until these pass -- they are the step-0 parity")
        print("mechanism (memory-ON == memory-OFF bitwise) and the co-training")
        print("gradient path.  A wiring bug here surfaces as a diverging loss")
        print("curve hours into a 48h job.")
        return 1

    print("All B0/B1 parity invariants hold.  Safe to submit:")
    print("  bash pace/submit_retrofit_b1.sh start")
    return 0


if __name__ == "__main__":
    sys.exit(main())
