"""
Data cleaning and date-range utilities for navpy.

clean_nav()   : remove bad records from a raw NAV series
parse_dates() : resolve flexible date inputs to (start, end) Timestamps
"""

from __future__ import annotations
from typing import Optional, Tuple, List, Dict
import pandas as pd
from datetime import datetime
from .exceptions import InvalidDateError

MAX_SINGLE_DAY_MOVE = 0.50


def _resample_freq(freq: str) -> str:
    """Return pandas-version-safe resample frequency string."""
    mapping = {"ME": "ME", "YE": "YE", "W": "W"}
    try:
        import pandas as pd
        major = int(pd.__version__.split(".")[0])
        minor = int(pd.__version__.split(".")[1])
        if major < 2 or (major == 2 and minor < 2):
            mapping = {"ME": "M", "YE": "Y", "W": "W"}
    except Exception:
        pass
    return mapping.get(freq, freq)


def raw_to_dataframe(data: List[Dict]) -> pd.DataFrame:
    """
    Convert mfapi.in raw data list to a clean dated DataFrame.

    Parameters
    ----------
    data : list of {"date": "DD-MM-YYYY", "nav": "123.45"}

    Returns
    -------
    pd.DataFrame with columns: date (Timestamp), nav (float)
    Sorted ascending by date.
    """
    if not data:
        return pd.DataFrame(columns=["date", "nav"])

    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["date"], format="%d-%m-%Y", errors="coerce")
    df["nav"] = pd.to_numeric(df["nav"], errors="coerce")
    df = df.dropna(subset=["date", "nav"])
    df = df.sort_values("date").reset_index(drop=True)
    return df[["date", "nav"]]


def clean_nav(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove data quality issues from a NAV DataFrame.

    Rules applied in order:
    1. Drop rows where nav <= 0
    2. Drop rows where single-day return exceeds +/-50%
    3. Drop duplicate dates (keep last)
    4. Re-sort by date ascending
    """
    if df.empty:
        return df.copy()

    df = df.copy()
    df = df[df["nav"] > 0]

    if len(df) > 1:
        daily_ret = df["nav"].pct_change().abs()
        bad_idx = daily_ret[daily_ret > MAX_SINGLE_DAY_MOVE].index
        df = df.drop(bad_idx)

    df = df.drop_duplicates(subset="date", keep="last")
    df = df.sort_values("date").reset_index(drop=True)
    return df


def apply_date_filter(
    df: pd.DataFrame,
    start: Optional[pd.Timestamp],
    end: Optional[pd.Timestamp],
) -> pd.DataFrame:
    """Filter a NAV DataFrame to [start, end] inclusive."""
    if df.empty:
        return df
    if start is not None:
        df = df[df["date"] >= start]
    if end is not None:
        df = df[df["date"] <= end]
    return df.reset_index(drop=True)


def parse_dates(
    start: Optional[str],
    end: Optional[str],
) -> Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]:
    """
    Resolve flexible date inputs to (start_ts, end_ts) Timestamps.

    Accepted formats:
        YYYY-MM-DD, DD-MM-YYYY, DD/MM/YYYY
        'max', 'all', 'full' -> None
        'ytd'  -> Jan 1 of current year
        '1y', '3y', '5y', '6m', '1m' -> relative to today
        None   -> None
    """
    today = pd.Timestamp.today().normalize()
    return _parse_one(start, today, is_start=True), _parse_one(end, today, is_start=False)


def _parse_one(
    value: Optional[str],
    today: pd.Timestamp,
    is_start: bool,
) -> Optional[pd.Timestamp]:
    if value is None:
        return None

    v = str(value).strip().lower()

    if v in ("max", "all", "full"):
        return None

    if v == "ytd":
        return pd.Timestamp(today.year, 1, 1) if is_start else today

    if len(v) >= 2 and v[-1] in ("y", "m") and v[:-1].isdigit():
        n = int(v[:-1])
        unit = v[-1]
        if unit == "y":
            ts = today - pd.DateOffset(years=n)
        else:
            ts = today - pd.DateOffset(months=n)
        return ts.normalize()

    for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y", "%Y/%m/%d"):
        try:
            return pd.Timestamp(datetime.strptime(v, fmt))
        except ValueError:
            continue

    try:
        return pd.Timestamp(v)
    except Exception:
        raise InvalidDateError(value)
