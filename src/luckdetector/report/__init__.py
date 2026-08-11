"""Turning statistics into an answer: verdict aggregation, plots, HTML and the demo."""

from __future__ import annotations

from .analysis import Analysis, analyse, analyse_mined
from .demo import DemoResult, run_demo
from .html import render_report, write_report
from .plots import cumulative_return_figure, figure_to_base64, trial_sharpe_figure
from .verdict import (
    SELECTION_AWARE_TESTS,
    TEST_ORDER,
    assess,
)

__all__ = [
    "SELECTION_AWARE_TESTS",
    "TEST_ORDER",
    "Analysis",
    "DemoResult",
    "analyse",
    "analyse_mined",
    "assess",
    "cumulative_return_figure",
    "figure_to_base64",
    "render_report",
    "run_demo",
    "trial_sharpe_figure",
    "write_report",
]
