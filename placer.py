"""
Canonical entry point for the HAPpy Placer submission.

The competition eval expects a `Placer` class with a `place(benchmark)` method.
This thin shim re-exports the full pipeline implemented in
`submissions/hybridv2.HybridAnalyticalPlacerV2`.

Run via:
    uv run evaluate placer.py --all
"""

import importlib.util
import os
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# Load the implementation file by path (the eval harness loads this `placer.py`
# standalone, so `from submissions.hybridv2 import ...` doesn't resolve).
_HV2_PATH = os.path.join(
    _ROOT,
    "submissions",
    "hybridv2.py",
)
_spec = importlib.util.spec_from_file_location("_hv2_impl", _HV2_PATH)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

HybridAnalyticalPlacerV2 = _mod.HybridAnalyticalPlacerV2


def _env_enabled(name: str, default: str = "") -> bool:
    return os.environ.get(name, default).strip().lower() not in ("", "0", "false", "no", "off")


if _env_enabled("HAP_MULTI", "1"):
    _MULTI_PATH = os.path.join(
        _ROOT,
        "submissions",
        "hybridv2_multi.py",
    )
    _multi_spec = importlib.util.spec_from_file_location("_hv2_multi_impl", _MULTI_PATH)
    _multi_mod = importlib.util.module_from_spec(_multi_spec)
    _multi_spec.loader.exec_module(_multi_mod)

    HybridAnalyticalPlacerV2Multi = _multi_mod.HybridAnalyticalPlacerV2Multi

    class Placer(HybridAnalyticalPlacerV2Multi):
        """HAPpy Placer multi-start wrapper; set HAP_MULTI=0 to disable."""

        def __init__(self):
            super().__init__(
                num_steps=int(os.environ.get("HAP_MULTI_NUM_STEPS", "50000")),
                threads_per_worker=int(os.environ.get("HAP_MULTI_THREADS", "3")),
            )

else:

    class Placer(HybridAnalyticalPlacerV2):
        """HAPpy Placer — pipeline detailed in README.md."""
        pass
