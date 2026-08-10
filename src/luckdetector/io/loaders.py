"""Load return series and trial matrices from CSV/parquet, with strict validation.

Design stance: **fail loudly and early**. A silently misread CSV — prices treated
as returns, percentages treated as fractions, a stray NaN filled with zero —
produces a report that is confidently wrong, which is worse than no report. Every
loader here validates and explains rather than coercing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from ..exceptions import DataValidationError
from ..types import FloatArray, ReturnSeries, TrialMatrix

__all__ = [
    "infer_periods_per_year",
    "load_returns_csv",
    "load_trials_csv",
    "returns_from_prices",
    "trial_matrix_from_frame",
]

NanPolicy = Literal["raise", "drop"]

#: Median calendar spacing (days) → periods per year.
_SPACING_TO_PPY: tuple[tuple[float, int], ...] = (
    (2.0, 252),  # daily / business-daily
    (10.0, 52),  # weekly
    (45.0, 12),  # monthly
    (135.0, 4),  # quarterly
    (400.0, 1),  # annual
)


def infer_periods_per_year(index: pd.DatetimeIndex) -> int:
    """Guess the observation frequency from a datetime index.

    Uses the *median* spacing so that weekends, holidays and the odd data gap
    don't shift the answer.
    """
    if len(index) < 3:
        raise DataValidationError("Need at least 3 timestamps to infer a frequency.")
    spacing_days = float(np.median(np.diff(index.values).astype("timedelta64[s]").astype(float)))
    spacing_days /= 86_400.0
    for threshold, ppy in _SPACING_TO_PPY:
        if spacing_days <= threshold:
            return ppy
    raise DataValidationError(
        f"Median spacing of {spacing_days:.1f} days is coarser than annual; "
        "pass periods_per_year explicitly."
    )


def returns_from_prices(
    prices: pd.Series | pd.DataFrame | FloatArray,
    *,
    log: bool = False,
) -> FloatArray:
    """Convert a price level series to simple (or log) returns.

    The output is one observation shorter than the input.
    """
    arr = np.asarray(prices, dtype=np.float64)
    if np.any(arr <= 0.0):
        raise DataValidationError(
            "Price series contains non-positive values; returns are undefined there."
        )
    if arr.ndim == 1:
        ratio = arr[1:] / arr[:-1]
    elif arr.ndim == 2:
        ratio = arr[1:, :] / arr[:-1, :]
    else:
        raise DataValidationError(f"Prices must be 1-D or 2-D, got shape {arr.shape}.")
    return np.log(ratio) if log else ratio - 1.0


def _handle_nans(frame: pd.DataFrame, nan_policy: NanPolicy, source: str) -> pd.DataFrame:
    n_nan = int(frame.isna().to_numpy().sum())
    if n_nan == 0:
        return frame
    if nan_policy == "raise":
        raise DataValidationError(
            f"{source} contains {n_nan} missing value(s). Fix the data, or pass "
            'nan_policy="drop" if dropping those rows is genuinely correct.'
        )
    # Annotated local rather than a bare return: pandas-stubs 3.x types ``dropna``
    # as returning Any, which trips mypy's no-any-return under --strict. A cast()
    # would be flagged as redundant by the 2.x stubs, so this is the form that
    # satisfies both.
    cleaned: pd.DataFrame = frame.dropna()
    return cleaned


def _read_table(path: str | Path, date_column: str | None) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise DataValidationError(f"No such file: {path}")
    if path.suffix.lower() in {".parquet", ".pq"}:
        frame = pd.read_parquet(path)
    else:
        frame = pd.read_csv(path)
    if date_column is not None:
        if date_column not in frame.columns:
            raise DataValidationError(
                f"date_column {date_column!r} not found; columns are {list(frame.columns)}."
            )
        frame[date_column] = pd.to_datetime(frame[date_column])
        frame = frame.set_index(date_column).sort_index()
    return frame


def load_returns_csv(
    path: str | Path,
    *,
    column: str | None = None,
    date_column: str | None = None,
    periods_per_year: int | None = None,
    are_prices: bool = False,
    nan_policy: NanPolicy = "raise",
    name: str | None = None,
) -> ReturnSeries:
    """Load a single strategy's track record.

    Parameters
    ----------
    column:
        Which column holds the data. Required if the file has more than one
        non-index column — guessing would be a silent-error factory.
    date_column:
        If given, parsed as the index and used to infer ``periods_per_year``.
    are_prices:
        Set ``True`` if the column contains price levels rather than returns.
    nan_policy:
        ``"raise"`` (default) or ``"drop"``.
    """
    frame = _read_table(path, date_column)
    frame = _handle_nans(frame, nan_policy, f"{path}")

    numeric = frame.select_dtypes(include=[np.number])
    if column is not None:
        if column not in numeric.columns:
            raise DataValidationError(
                f"Column {column!r} not found or not numeric; numeric columns are "
                f"{list(numeric.columns)}."
            )
        picked = column
    elif numeric.shape[1] == 1:
        picked = str(numeric.columns[0])
    else:
        raise DataValidationError(
            f"File has {numeric.shape[1]} numeric columns {list(numeric.columns)}; "
            "pass column= to say which one holds the track record."
        )

    values: FloatArray = np.asarray(numeric[picked].to_numpy(), dtype=np.float64)
    if are_prices:
        values = returns_from_prices(values)

    if periods_per_year is None:
        index = frame.index
        if isinstance(index, pd.DatetimeIndex):
            periods_per_year = infer_periods_per_year(index)
        else:
            periods_per_year = 252

    series = ReturnSeries(
        values=values,
        periods_per_year=periods_per_year,
        name=name or picked,
    )
    if series.looks_like_percentages():
        raise DataValidationError(
            f"Column {picked!r} has a per-period standard deviation of "
            f"{float(np.std(series.values)):.2f}. That is almost certainly percentages "
            "(1.5 meaning 1.5%) rather than fractions (0.015). Divide by 100, or pass "
            "are_prices=True if these are price levels."
        )
    return series


def trial_matrix_from_frame(
    frame: pd.DataFrame,
    *,
    periods_per_year: int | None = None,
    orient: Literal["columns", "rows"] = "columns",
) -> TrialMatrix:
    """Build a :class:`TrialMatrix` from a DataFrame.

    ``orient="columns"`` (the natural layout: one column per strategy, one row
    per date) or ``"rows"`` (one row per strategy).
    """
    numeric = frame.select_dtypes(include=[np.number])
    if numeric.empty:
        raise DataValidationError("No numeric columns found for the trial matrix.")

    if orient == "columns":
        values = numeric.to_numpy(dtype=np.float64).T
        labels = [str(c) for c in numeric.columns]
        index = frame.index
    else:
        values = numeric.to_numpy(dtype=np.float64)
        labels = [str(i) for i in numeric.index]
        index = numeric.columns

    if periods_per_year is None:
        periods_per_year = (
            infer_periods_per_year(index) if isinstance(index, pd.DatetimeIndex) else 252
        )

    return TrialMatrix(values=values, periods_per_year=periods_per_year, labels=labels)


def load_trials_csv(
    path: str | Path,
    *,
    date_column: str | None = None,
    periods_per_year: int | None = None,
    orient: Literal["columns", "rows"] = "columns",
    nan_policy: NanPolicy = "raise",
) -> TrialMatrix:
    """Load every strategy variant that was tried, aligned on a common calendar.

    This is the input the strong tests (PBO, Reality Check, SPA) need. If you only
    kept the winner, you have thrown away the evidence required to judge it.
    """
    frame = _read_table(path, date_column)
    frame = _handle_nans(frame, nan_policy, f"{path}")
    return trial_matrix_from_frame(frame, periods_per_year=periods_per_year, orient=orient)
