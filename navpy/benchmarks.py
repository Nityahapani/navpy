"""
Benchmark data for Indian mutual fund performance comparison.
"""

from __future__ import annotations
from typing import Optional, Dict
import pandas as pd

BENCHMARKS: Dict[str, dict] = {
    "nifty50": {
        "name": "Nifty 50",
        "ticker": "^NSEI",
        "description": "NSE large-cap index of 50 stocks — primary Indian equity benchmark",
        "asset_class": "equity",
        "category": "broad_market",
    },
    "sensex": {
        "name": "BSE Sensex",
        "ticker": "^BSESN",
        "description": "BSE flagship index of 30 large-cap stocks",
        "asset_class": "equity",
        "category": "broad_market",
    },
    "nifty500": {
        "name": "Nifty 500",
        "ticker": "^CRSLDX",
        "description": "Top 500 companies by market cap — broadest NSE index",
        "asset_class": "equity",
        "category": "broad_market",
    },
    "nifty100": {
        "name": "Nifty 100",
        "ticker": "^CNX100",
        "description": "Top 100 companies by market cap on NSE",
        "asset_class": "equity",
        "category": "large_cap",
    },
    "nifty_largecap50": {
        "name": "Nifty Largecat 250",
        "ticker": "^NSEI",
        "description": "NSE large-cap proxy (Nifty 50 used as standard large-cap benchmark)",
        "asset_class": "equity",
        "category": "large_cap",
    },
    "nifty_midcap50": {
        "name": "Nifty Midcap 50",
        "ticker": "^NSEMDCP50",
        "description": "NSE mid-cap index of 50 stocks",
        "asset_class": "equity",
        "category": "mid_cap",
    },
    "nifty_midcap100": {
        "name": "Nifty Midcap 100",
        "ticker": "NIFTYMIDCAP100.NS",
        "description": "NSE mid-cap index of 100 stocks",
        "asset_class": "equity",
        "category": "mid_cap",
    },
    "nifty_midcap150": {
        "name": "Nifty Midcap 150",
        "ticker": "^NSEMDCP150",
        "description": "Broader NSE mid-cap benchmark",
        "asset_class": "equity",
        "category": "mid_cap",
    },
    "nifty_smallcap50": {
        "name": "Nifty Smallcap 50",
        "ticker": "^CNXSC",
        "description": "NSE small-cap index of 50 stocks",
        "asset_class": "equity",
        "category": "small_cap",
    },
    "nifty_smallcap100": {
        "name": "Nifty Smallcap 100",
        "ticker": "NIFTYSMLCAP100.NS",
        "description": "NSE small-cap index of 100 stocks",
        "asset_class": "equity",
        "category": "small_cap",
    },
    "nifty_smallcap250": {
        "name": "Nifty Smallcap 250",
        "ticker": "NIFTYSMLCAP250.NS",
        "description": "Broadest NSE small-cap benchmark",
        "asset_class": "equity",
        "category": "small_cap",
    },
    "nifty_largemidcap250": {
        "name": "Nifty LargeMidcap 250",
        "ticker": "NIFTYLARGEMID250.NS",
        "description": "Combined large + mid cap",
        "asset_class": "equity",
        "category": "large_mid_cap",
    },
    "nifty_multicap50_25_25": {
        "name": "Nifty Multicap 50:25:25",
        "ticker": "NIFTYMCAP50_25_25.NS",
        "description": "Equal-weight large/mid/small",
        "asset_class": "equity",
        "category": "multi_cap",
    },
    "nifty_bank": {
        "name": "Nifty Bank",
        "ticker": "^NSEBANK",
        "description": "NSE banking sector index",
        "asset_class": "equity",
        "category": "sectoral",
    },
    "nifty_it": {
        "name": "Nifty IT",
        "ticker": "^CNXIT",
        "description": "NSE information technology sector index",
        "asset_class": "equity",
        "category": "sectoral",
    },
    "nifty_fmcg": {
        "name": "Nifty FMCG",
        "ticker": "NIFTYFMCG.NS",
        "description": "NSE fast-moving consumer goods sector index",
        "asset_class": "equity",
        "category": "sectoral",
    },
    "nifty_pharma": {
        "name": "Nifty Pharma",
        "ticker": "NIFTYPHARMA.NS",
        "description": "NSE pharmaceutical sector index",
        "asset_class": "equity",
        "category": "sectoral",
    },
    "nifty_infra": {
        "name": "Nifty Infrastructure",
        "ticker": "NIFTYINFRA.NS",
        "description": "NSE infrastructure sector index",
        "asset_class": "equity",
        "category": "sectoral",
    },
    "nifty_hybrid_composite": {
        "name": "Nifty 50 Hybrid Composite Debt 65:35",
        "ticker": "^NSEI",
        "description": "65% Nifty 50 + 35% CRISIL Composite Bond (proxy: Nifty 50)",
        "asset_class": "hybrid",
        "category": "hybrid",
    },
    "sp500": {
        "name": "S&P 500",
        "ticker": "^GSPC",
        "description": "US large-cap benchmark — for global context",
        "asset_class": "equity",
        "category": "international",
    },
    "msci_em": {
        "name": "MSCI Emerging Markets (USD)",
        "ticker": "EEM",
        "description": "iShares MSCI EM ETF — emerging market comparison",
        "asset_class": "equity",
        "category": "international",
    },
}

_ALIAS_MAP: Dict[str, str] = {
    "nifty": "nifty50",
    "nifty 50": "nifty50",
    "bse": "sensex",
    "bse sensex": "sensex",
    "midcap": "nifty_midcap150",
    "midcap150": "nifty_midcap150",
    "midcap50": "nifty_midcap50",
    "smallcap": "nifty_smallcap100",
    "smallcap100": "nifty_smallcap100",
    "bank": "nifty_bank",
    "it": "nifty_it",
    "fmcg": "nifty_fmcg",
    "pharma": "nifty_pharma",
    "sp500": "sp500",
    "s&p500": "sp500",
    "s&p 500": "sp500",
    "em": "msci_em",
    "msci em": "msci_em",
}


def list_benchmarks(category: Optional[str] = None) -> pd.DataFrame:
    """
    Return a DataFrame of all available benchmarks.

    Parameters
    ----------
    category : str, optional
        Filter by category. Options: broad_market, large_cap, mid_cap,
        small_cap, large_mid_cap, multi_cap, sectoral, hybrid, international
    """
    rows = []
    for alias, meta in BENCHMARKS.items():
        rows.append({
            "alias": alias,
            "name": meta["name"],
            "ticker": meta["ticker"],
            "asset_class": meta["asset_class"],
            "category": meta["category"],
            "description": meta["description"],
        })
    df = pd.DataFrame(rows)
    if category:
        df = df[df["category"] == category]
    return df.reset_index(drop=True)


def resolve_benchmark(identifier: str) -> dict:
    """
    Resolve a benchmark identifier to its metadata dict.

    Accepts: alias key, shorthand, Yahoo Finance ticker, or full name.

    Raises
    ------
    ValueError if no match is found
    """
    key = identifier.strip().lower()

    if key in BENCHMARKS:
        return {"alias": key, **BENCHMARKS[key]}

    if key in _ALIAS_MAP:
        resolved = _ALIAS_MAP[key]
        return {"alias": resolved, **BENCHMARKS[resolved]}

    for alias, meta in BENCHMARKS.items():
        if meta["ticker"].lower() == key:
            return {"alias": alias, **meta}

    for alias, meta in BENCHMARKS.items():
        if key in meta["name"].lower():
            return {"alias": alias, **meta}

    available = ", ".join(sorted(BENCHMARKS.keys()))
    raise ValueError(
        f"Unknown benchmark '{identifier}'. "
        f"Available aliases:\n  {available}\n"
        "Use navpy.list_benchmarks() for a full table."
    )


def get_benchmark(
    identifier: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
    force_refresh: bool = False,
) -> pd.Series:
    """
    Fetch benchmark index data as a date-indexed pandas Series.

    Parameters
    ----------
    identifier    : str  alias, shorthand, ticker, or name
    start         : str  date string or shorthand ('3y', 'ytd', etc.)
    end           : str  date string or shorthand (default: today)
    force_refresh : bool bypass yfinance cache

    Returns
    -------
    pd.Series  date-indexed, name = benchmark name
    """
    try:
        import yfinance as yf
    except ImportError:
        raise ImportError(
            "yfinance is required for benchmark data. "
            "Install it with: pip install yfinance"
        )

    from .clean import parse_dates
    meta = resolve_benchmark(identifier)
    ticker = meta["ticker"]

    start_ts, end_ts = parse_dates(start, end)

    yf_start = start_ts.strftime("%Y-%m-%d") if start_ts else "2000-01-01"
    yf_end = (end_ts + pd.Timedelta(days=1)).strftime("%Y-%m-%d") if end_ts else None

    try:
        raw = yf.download(
            ticker,
            start=yf_start,
            end=yf_end,
            progress=False,
            auto_adjust=True,
        )
    except Exception as e:
        raise ValueError(f"Could not fetch benchmark '{identifier}' ({ticker}): {e}")

    if raw is None or raw.empty:
        raise ValueError(
            f"No data returned for benchmark '{identifier}' ({ticker}). "
            "The ticker may be unavailable on Yahoo Finance."
        )

    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"].iloc[:, 0]
    elif "Close" in raw.columns:
        close = raw["Close"]
    else:
        close = raw.iloc[:, 0]

    series = close.squeeze().dropna()
    series.index = pd.to_datetime(series.index)
    series.name = meta["name"]
    return series
