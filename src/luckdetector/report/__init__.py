"""Turning statistics into an answer: verdict aggregation, and later plots and HTML."""

from __future__ import annotations

from .verdict import (
    SELECTION_AWARE_TESTS,
    TEST_ORDER,
    assess,
)

__all__ = [
    "SELECTION_AWARE_TESTS",
    "TEST_ORDER",
    "assess",
]
