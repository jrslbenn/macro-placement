"""
Canonical entry point for the SPIRAL Placer submission.

The competition eval expects a `Placer` class with a `place(benchmark)` method.
This thin shim re-exports the full pipeline implemented in
`submissions/hybridv2.HybridAnalyticalPlacerV2`.

Run via:
    uv run evaluate placer.py --all
"""

import importlib.util
import os

# Load the implementation file by path (the eval harness loads this `placer.py`
# standalone, so `from submissions.hybridv2 import ...` doesn't resolve).
_HV2_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "submissions",
    "hybridv2.py",
)
_spec = importlib.util.spec_from_file_location("_hv2_impl", _HV2_PATH)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

HybridAnalyticalPlacerV2 = _mod.HybridAnalyticalPlacerV2


class Placer(HybridAnalyticalPlacerV2):
    """SPIRAL Placer — pipeline detailed in README.md."""
    pass
