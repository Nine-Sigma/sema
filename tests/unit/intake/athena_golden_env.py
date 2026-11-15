"""Shared setup for test_athena_golden.py and the tests that reuse it."""

from __future__ import annotations

from pathlib import Path

GOLDEN = Path(__file__).parents[2] / "fixtures" / "intake" / "athena" / "golden_export.csv"
