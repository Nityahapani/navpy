"""
Plan A splice logic.

Combines Regular Plan (pre-Jan 2013) and Direct Plan (post-Jan 2013) NAV
into a single continuous series, rebased at the splice point.
"""

from __future__ import annotations
from typing import Tuple
import pandas as pd
from . import fetch as _fetch
from . import clean as _clean

SPLICE_DATE = pd.Timestamp("2013-01-01")


def _load_and_clean(scheme_code: str, force_refresh: bool = False) -> pd.DataFrame:
    """Fetch raw NAV for a code and return a cleaned DataFrame."""
    try:
        payload = _fetch.fetch_scheme(scheme_code, force_refresh=force_refresh)
        df = _clean.raw_to_dataframe(payload.get("data", []))
        return _clean.clean_nav(df)
    except Exception:
        return pd.DataFrame(columns=["date", "nav"])


def get_direct(scheme_code: str, force_refresh: bool = False) -> pd.DataFrame:
    """Return cleaned NAV for a Direct Plan scheme code."""
    return _load_and_clean(scheme_code, force_refresh)


def get_regular(scheme_code: str, force_refresh: bool = False) -> pd.DataFrame:
    """Return cleaned NAV for a Regular Plan scheme code."""
    return _load_and_clean(scheme_code, force_refresh)


def splice(
    regular_code: str,
    direct_code: str,
    force_refresh: bool = False,
) -> Tuple[pd.DataFrame, str]:
    """
    Build a Plan A spliced NAV series.

    Uses Regular plan pre-Jan 2013, Direct plan from Jan 2013 onward.
    Scales Regular at the splice point to eliminate discontinuity.

    Returns
    -------
    (df, plan_used) where plan_used is 'splice', 'direct', or 'regular'
    """
    reg_df = _load_and_clean(regular_code, force_refresh)
    dir_df = _load_and_clean(direct_code, force_refresh)

    if reg_df.empty and dir_df.empty:
        return pd.DataFrame(columns=["date", "nav"]), "none"

    if reg_df.empty:
        return dir_df.copy(), "direct"

    if dir_df.empty:
        return reg_df.copy(), "regular"

    reg_pre = reg_df[reg_df["date"] < SPLICE_DATE].copy()
    dir_post = dir_df[dir_df["date"] >= SPLICE_DATE].copy()

    if reg_pre.empty:
        return dir_post.copy(), "direct"

    if dir_post.empty:
        return reg_pre.copy(), "regular"

    dir_first_date = dir_post["date"].iloc[0]
    reg_at_splice = reg_df[reg_df["date"] <= dir_first_date]["nav"]

    if not reg_at_splice.empty and dir_post["nav"].iloc[0] > 0:
        scale = dir_post["nav"].iloc[0] / reg_at_splice.iloc[-1]
        reg_pre["nav"] = reg_pre["nav"] * scale

    combined = pd.concat([reg_pre, dir_post], ignore_index=True)
    combined = combined.drop_duplicates(subset="date", keep="last")
    combined = combined.sort_values("date").reset_index(drop=True)

    return combined, "splice"


def auto_splice(
    query: str,
    force_refresh: bool = False,
) -> Tuple[pd.DataFrame, str, str, str]:
    """
    Automatically find Regular and Direct codes for a fund name and splice them.

    Returns
    -------
    (df, plan_used, reg_code, dir_code)
    """
    from .search import resolve_pair
    reg_info, dir_info = resolve_pair(query)

    reg_code = reg_info.scheme_code if reg_info else ""
    dir_code = dir_info.scheme_code if dir_info else ""

    if not reg_code and not dir_code:
        return pd.DataFrame(columns=["date", "nav"]), "none", "", ""

    if not reg_code:
        df = _load_and_clean(dir_code, force_refresh)
        return df, "direct", "", dir_code

    if not dir_code:
        df = _load_and_clean(reg_code, force_refresh)
        return df, "regular", reg_code, ""

    df, plan_used = splice(reg_code, dir_code, force_refresh)
    return df, plan_used, reg_code, dir_code
