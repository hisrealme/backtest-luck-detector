"""Exception hierarchy for luckdetector.

Every failure mode gets a named exception so callers (and the CLI) can react
differently to "your data is malformed" versus "this statistic is undefined for
your data".
"""

from __future__ import annotations


class LuckDetectorError(Exception):
    """Base class for all errors raised by this package."""


class DataValidationError(LuckDetectorError):
    """Input data is malformed: wrong shape, NaNs, impossible values."""


class DegenerateSeriesError(LuckDetectorError):
    """A statistic is undefined for this series (e.g. zero volatility)."""


class InsufficientDataError(LuckDetectorError):
    """Not enough observations or trials to compute the requested statistic."""
