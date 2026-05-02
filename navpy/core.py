"""
Core navpy API — the main get() entry point.
"""

from __future__ import annotations
from typing import Optional

from .models import NAVResult
from .exceptions import NoDataError
from . import fetch as _fetch
from . import clean as _clean
from . import splice as _splice
from .search import resolve


def get(
    query: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
    plan: str = "splice",
    option: str = "growth",
    force_refresh: bool = False,
    interactive: bool = True,
) -> NAVResult:
    """
    Fetch day-wise NAV for a mutual fund scheme.

    Parameters
    ----------
    query : str
        Fund name (full or partial), keyword, or numeric AMFI scheme code.

    start : str, optional
        Start date. Accepts 'YYYY-MM-DD', '1y', '3y', '5y', '6m',
        'ytd', 'max', or None (full history).

    end : str, optional
        End date. Same formats as start. Defaults to today.

    plan : str, optional
        'splice' (default), 'direct', or 'regular'.

    option : str, optional
        'growth' (default) or 'idcw'.

    force_refresh : bool, optional
        Bypass local cache. Default False.

    interactive : bool, optional
        Prompt when multiple schemes match. Default True.

    Returns
    -------
    NAVResult

    Raises
    ------
    SchemeNotFoundError, NoDataError, InvalidDateError
    """
    from .clean import parse_dates

    query = str(query).strip()
    start_ts, end_ts = parse_dates(start, end)

    plan_lower = plan.lower()
    option_lower = option.lower()

    if plan_lower in ("direct", "regular"):
        prefer_kw = f"{plan_lower} {option_lower}"
    else:
        prefer_kw = option_lower

    scheme = resolve(
        query,
        prefer=prefer_kw,
        plan=plan_lower,
        interactive=interactive,
    )

    reg_code = ""
    dir_code = ""

    if plan_lower == "splice":
        dir_code = scheme.scheme_code
        from .search import resolve_pair
        reg_info, dir_info = resolve_pair(query)

        if reg_info:
            reg_code = reg_info.scheme_code
        if dir_info and not dir_code:
            dir_code = dir_info.scheme_code

        if reg_code and dir_code:
            df, plan_used = _splice.splice(reg_code, dir_code, force_refresh)
        elif dir_code:
            raw = _fetch.fetch_scheme(dir_code, force_refresh)
            df = _clean.raw_to_dataframe(raw.get("data", []))
            df = _clean.clean_nav(df)
            plan_used = "direct"
        else:
            raw = _fetch.fetch_scheme(reg_code, force_refresh)
            df = _clean.raw_to_dataframe(raw.get("data", []))
            df = _clean.clean_nav(df)
            plan_used = "regular"

    elif plan_lower == "direct":
        dir_code = scheme.scheme_code
        raw = _fetch.fetch_scheme(dir_code, force_refresh)
        df = _clean.raw_to_dataframe(raw.get("data", []))
        df = _clean.clean_nav(df)
        plan_used = "direct"

    elif plan_lower == "regular":
        reg_code = scheme.scheme_code
        raw = _fetch.fetch_scheme(reg_code, force_refresh)
        df = _clean.raw_to_dataframe(raw.get("data", []))
        df = _clean.clean_nav(df)
        plan_used = "regular"

    else:
        raise ValueError(
            f"Unknown plan '{plan}'. Choose from: 'splice', 'direct', 'regular'."
        )

    df = _clean.apply_date_filter(df, start_ts, end_ts)

    if df.empty:
        raise NoDataError(
            scheme.scheme_code,
            start=str(start_ts.date()) if start_ts else None,
            end=str(end_ts.date()) if end_ts else None,
        )

    return NAVResult(
        scheme_code=scheme.scheme_code,
        scheme_name=scheme.scheme_name,
        plan=plan_used,
        start_date=str(df["date"].iloc[0].date()),
        end_date=str(df["date"].iloc[-1].date()),
        data=df.reset_index(drop=True),
        _reg_code=reg_code,
        _dir_code=dir_code,
    )
