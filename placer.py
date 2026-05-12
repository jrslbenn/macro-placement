"""
Canonical entry point for the SPIRAL Placer submission.

The competition eval expects a `Placer` class with a `place(benchmark)` method.
This thin shim re-exports the full pipeline implemented in
`submissions/hybridv2.HybridAnalyticalPlacerV2`.

Run via:
    uv run evaluate placer.py --all
"""

from submissions.hybridv2 import HybridAnalyticalPlacerV2


class Placer(HybridAnalyticalPlacerV2):
    """SPIRAL Placer — pipeline detailed in README.md."""
    pass
