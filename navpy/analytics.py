"""
Advanced analytics for navpy NAVResult objects.

All functions operate on a pd.Series (date-indexed NAV).
"""

from __future__ import annotations
from typing import Optional, Dict, List
import pandas as pd
import numpy as np
from scipy import stats

TRADING_DAYS = 252


def _daily_ret(s: pd.Series) -> pd.Series:
    return s.pct_change().dropna()


def _resample_freq(freq: str) -> str:
    """Return pandas-version-safe resample frequency string."""
    try:
        major = int(pd.__version__.split(".")[0])
        minor = int(pd.__version__.split(".")[1])
        if major < 2 or (major == 2 and minor < 2):
            return {"ME": "M", "YE": "Y", "W": "W"}.get(freq, freq)
    except Exception:
        pass
    return freq


# ── Rolling metrics ─────────────────────────────────────────────────────────

def rolling_returns(
    nav: pd.Series,
    window: int = 252,
    annualise: bool = True,
) -> pd.Series:
    """Rolling point-to-point return over a sliding window."""
    roll = (1 + _daily_ret(nav)).rolling(window).apply(np.prod, raw=True) - 1
    if annualise:
        years = window / TRADING_DAYS
        roll = (1 + roll) ** (1 / years) - 1
    return roll.rename(f"rolling_return_{window}d")


def rolling_alpha(
    nav: pd.Series,
    benchmark: pd.Series,
    window: int = 252,
    rf: float = 0.065,
) -> pd.Series:
    """Rolling Jensen's alpha (annualised) over a sliding window."""
    fund_ret = _daily_ret(nav)
    bm_ret = _daily_ret(benchmark)
    aligned = pd.concat([fund_ret, bm_ret], axis=1).dropna()
    aligned.columns = ["fund", "bm"]

    rf_daily = rf / TRADING_DAYS
    ex_fund = aligned["fund"] - rf_daily
    ex_bm = aligned["bm"] - rf_daily

    alphas: List[float] = []
    dates = []
    for i in range(window - 1, len(aligned)):
        f = ex_fund.iloc[i - window + 1:i + 1].values
        b = ex_bm.iloc[i - window + 1:i + 1].values
        if len(f) < 10:
            alphas.append(np.nan)
        else:
            _, intercept, _, _, _ = stats.linregress(b, f)
            alphas.append(intercept * TRADING_DAYS)
        dates.append(aligned.index[i])

    return pd.Series(alphas, index=dates, name=f"rolling_alpha_{window}d")


def rolling_beta(
    nav: pd.Series,
    benchmark: pd.Series,
    window: int = 252,
) -> pd.Series:
    """Rolling beta (sensitivity to benchmark) over a sliding window."""
    fund_ret = _daily_ret(nav)
    bm_ret = _daily_ret(benchmark)
    aligned = pd.concat([fund_ret, bm_ret], axis=1).dropna()
    aligned.columns = ["fund", "bm"]

    betas: List[float] = []
    dates = []
    for i in range(window - 1, len(aligned)):
        f = aligned["fund"].iloc[i - window + 1:i + 1].values
        b = aligned["bm"].iloc[i - window + 1:i + 1].values
        if len(f) < 10:
            betas.append(np.nan)
        else:
            slope, _, _, _, _ = stats.linregress(b, f)
            betas.append(slope)
        dates.append(aligned.index[i])

    return pd.Series(betas, index=dates, name=f"rolling_beta_{window}d")


def rolling_sharpe(
    nav: pd.Series,
    window: int = 252,
    rf: float = 0.065,
) -> pd.Series:
    """Rolling Sharpe ratio over a sliding window."""
    dr = _daily_ret(nav)
    rf_daily = rf / TRADING_DAYS
    excess = dr - rf_daily
    roll_mean = excess.rolling(window).mean() * TRADING_DAYS
    roll_std = dr.rolling(window).std() * np.sqrt(TRADING_DAYS)
    sharpe = roll_mean / roll_std
    return sharpe.rename(f"rolling_sharpe_{window}d")


def rolling_sortino(
    nav: pd.Series,
    window: int = 252,
    rf: float = 0.065,
) -> pd.Series:
    """Rolling Sortino ratio over a sliding window."""
    dr = _daily_ret(nav)

    sortinos: List[float] = []
    dates = []
    for i in range(window - 1, len(dr)):
        seg = dr.iloc[i - window + 1:i + 1]
        ann_ret = ((1 + seg).prod() ** (TRADING_DAYS / window)) - 1
        downside = seg[seg < 0].std() * np.sqrt(TRADING_DAYS)
        if downside == 0 or np.isnan(downside):
            sortinos.append(np.nan)
        else:
            sortinos.append((ann_ret - rf) / downside)
        dates.append(dr.index[i])

    return pd.Series(sortinos, index=dates, name=f"rolling_sortino_{window}d")


def rolling_drawdown(nav: pd.Series) -> pd.Series:
    """Running drawdown from the most recent peak at every date."""
    roll_max = nav.cummax()
    dd = (nav - roll_max) / roll_max
    return dd.rename("drawdown")


def rolling_volatility(
    nav: pd.Series,
    window: int = 63,
) -> pd.Series:
    """Rolling annualised volatility (std of daily returns)."""
    dr = _daily_ret(nav)
    vol = dr.rolling(window).std() * np.sqrt(TRADING_DAYS)
    return vol.rename(f"rolling_vol_{window}d")


# ── Point-in-time metrics ────────────────────────────────────────────────────

def alpha_beta(
    nav: pd.Series,
    benchmark: pd.Series,
    rf: float = 0.065,
) -> Dict[str, float]:
    """Jensen's alpha and beta via full-period OLS regression."""
    fund_ret = _daily_ret(nav)
    bm_ret = _daily_ret(benchmark)
    aligned = pd.concat([fund_ret, bm_ret], axis=1).dropna()
    aligned.columns = ["fund", "bm"]

    if len(aligned) < 30:
        return {"alpha": np.nan, "beta": np.nan, "r_squared": np.nan,
                "p_value": np.nan, "std_error": np.nan}

    rf_daily = rf / TRADING_DAYS
    ex_fund = aligned["fund"] - rf_daily
    ex_bm = aligned["bm"] - rf_daily

    slope, intercept, r, p, se = stats.linregress(ex_bm.values, ex_fund.values)
    return {
        "alpha": float(intercept * TRADING_DAYS),
        "beta": float(slope),
        "r_squared": float(r ** 2),
        "p_value": float(p),
        "std_error": float(se),
    }


def information_ratio(nav: pd.Series, benchmark: pd.Series) -> float:
    """Information ratio: annualised active return / tracking error."""
    fund_ret = _daily_ret(nav)
    bm_ret = _daily_ret(benchmark)
    aligned = pd.concat([fund_ret, bm_ret], axis=1).dropna()
    aligned.columns = ["fund", "bm"]
    active = aligned["fund"] - aligned["bm"]
    te = active.std() * np.sqrt(TRADING_DAYS)
    if te == 0:
        return np.nan
    return float((active.mean() * TRADING_DAYS) / te)


def updown_capture(
    nav: pd.Series,
    benchmark: pd.Series,
) -> Dict[str, float]:
    """Upside and downside capture ratios."""
    fund_ret = _daily_ret(nav)
    bm_ret = _daily_ret(benchmark)
    aligned = pd.concat([fund_ret, bm_ret], axis=1).dropna()
    aligned.columns = ["fund", "bm"]

    up = aligned[aligned["bm"] > 0]
    down = aligned[aligned["bm"] < 0]

    uc = (up["fund"].mean() / up["bm"].mean() * 100) if len(up) > 5 else np.nan
    dc = (down["fund"].mean() / down["bm"].mean() * 100) if len(down) > 5 else np.nan
    cr = (uc / dc) if (not np.isnan(uc) and not np.isnan(dc) and dc != 0) else np.nan

    def _r(v: float) -> float:
        return round(float(v), 2) if not np.isnan(v) else np.nan

    return {
        "upside_capture": _r(uc),
        "downside_capture": _r(dc),
        "capture_ratio": _r(cr),
    }


def calmar_ratio(nav: pd.Series) -> float:
    """Calmar ratio: annualised CAGR / absolute max drawdown."""
    if len(nav) < 2:
        return np.nan
    years = (nav.index[-1] - nav.index[0]).days / 365.25
    if years <= 0:
        return np.nan
    cagr = (nav.iloc[-1] / nav.iloc[0]) ** (1 / years) - 1
    mdd = abs((nav - nav.cummax()).div(nav.cummax()).min())
    return float(cagr / mdd) if mdd > 0 else np.nan


def omega_ratio(nav: pd.Series, threshold: float = 0.0) -> float:
    """Omega ratio: probability-weighted ratio of gains to losses."""
    dr = _daily_ret(nav)
    wins = dr[dr > threshold] - threshold
    loss = threshold - dr[dr <= threshold]
    if loss.sum() == 0:
        return np.nan
    return float(wins.sum() / loss.sum())


def pain_index(nav: pd.Series) -> float:
    """Pain index: mean of all drawdown values (absolute). Lower is better."""
    dd = rolling_drawdown(nav)
    return float(abs(dd.mean()) * 100)


def ulcer_index(nav: pd.Series) -> float:
    """Ulcer index: RMS of drawdowns. Lower is better."""
    dd = rolling_drawdown(nav) * 100
    return float(np.sqrt((dd ** 2).mean()))


def var_cvar(
    nav: pd.Series,
    confidence: float = 0.95,
) -> Dict[str, float]:
    """Value at Risk and Conditional VaR (Expected Shortfall) at daily level."""
    dr = _daily_ret(nav) * 100
    var = float(np.percentile(dr, (1 - confidence) * 100))
    cvar = float(dr[dr <= var].mean())
    return {
        "confidence": confidence,
        "var_pct": round(abs(var), 4),
        "cvar_pct": round(abs(cvar), 4),
    }


def drawdown_table(nav: pd.Series, top_n: int = 10) -> pd.DataFrame:
    """Table of the top N worst drawdown episodes with recovery info."""
    roll_max = nav.cummax()
    dd = (nav - roll_max) / roll_max

    in_dd = False
    episodes = []
    peak_date = trough_date = recovery_date = None
    trough_val = 0.0

    for date, val in dd.items():
        if not in_dd:
            if val < 0:
                in_dd = True
                loc = nav.index.get_loc(date)
                peak_date = nav.index[loc - 1] if loc > 0 else date
                trough_date = date
                trough_val = val
        else:
            if val < trough_val:
                trough_date = date
                trough_val = val
            if val >= 0:
                recovery_date = date
                episodes.append({
                    "peak_date": peak_date,
                    "trough_date": trough_date,
                    "recovery_date": recovery_date,
                    "drawdown_pct": round(trough_val * 100, 2),
                    "duration_days": (trough_date - peak_date).days,
                    "recovery_days": (recovery_date - trough_date).days,
                })
                in_dd = False

    if in_dd:
        episodes.append({
            "peak_date": peak_date,
            "trough_date": trough_date,
            "recovery_date": None,
            "drawdown_pct": round(trough_val * 100, 2),
            "duration_days": (trough_date - peak_date).days,
            "recovery_days": None,
        })

    df = pd.DataFrame(episodes)
    if df.empty:
        return df
    return df.nsmallest(top_n, "drawdown_pct").reset_index(drop=True)


def period_returns(nav: pd.Series) -> pd.DataFrame:
    """Standard period returns: 1M, 3M, 6M, 1Y, 2Y, 3Y, 5Y, 10Y, inception."""
    today = nav.index[-1]
    periods = {
        "1 Month": today - pd.DateOffset(months=1),
        "3 Months": today - pd.DateOffset(months=3),
        "6 Months": today - pd.DateOffset(months=6),
        "1 Year": today - pd.DateOffset(years=1),
        "2 Years": today - pd.DateOffset(years=2),
        "3 Years": today - pd.DateOffset(years=3),
        "5 Years": today - pd.DateOffset(years=5),
        "10 Years": today - pd.DateOffset(years=10),
        "Since Inception": nav.index[0],
    }

    rows = []
    for label, start in periods.items():
        seg = nav[nav.index >= start]
        if len(seg) < 2:
            continue
        abs_ret = (seg.iloc[-1] / seg.iloc[0] - 1) * 100
        years = (seg.index[-1] - seg.index[0]).days / 365.25
        cagr = ((seg.iloc[-1] / seg.iloc[0]) ** (1 / years) - 1) * 100 if years > 0 else np.nan
        rows.append({
            "period": label,
            "return_pct": round(abs_ret, 2),
            "annualised_pct": round(cagr, 2) if years >= 1 else None,
            "trading_days": len(seg),
        })

    return pd.DataFrame(rows)


def monthly_returns_table(nav: pd.Series) -> pd.DataFrame:
    """Monthly returns matrix — rows = years, columns = months."""
    freq = _resample_freq("ME")
    monthly = nav.resample(freq).last().pct_change() * 100
    monthly = monthly.dropna()
    monthly.index = pd.to_datetime(monthly.index)

    df = monthly.to_frame("return")
    df["year"] = df.index.year
    df["month"] = df.index.month

    pivot = df.pivot(index="year", columns="month", values="return")
    month_names = {
        1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr",
        5: "May", 6: "Jun", 7: "Jul", 8: "Aug",
        9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec",
    }
    pivot.columns = [month_names.get(c, c) for c in pivot.columns]

    freq_y = _resample_freq("YE")
    annual = nav.resample(freq_y).last().pct_change() * 100
    annual.index = annual.index.year
    pivot["Annual"] = annual
    pivot.index.name = "year"

    return pivot.round(2)


def regime_returns(
    nav: pd.Series,
    benchmark: pd.Series,
    bull_threshold: float = -0.10,
    bear_threshold: float = -0.20,
) -> pd.DataFrame:
    """Break down fund and benchmark returns by market regime."""
    fund_ret = _daily_ret(nav)
    bm_ret = _daily_ret(benchmark)
    bm_dd = rolling_drawdown(benchmark)
    aligned = pd.concat([fund_ret, bm_ret, bm_dd], axis=1, sort=False).dropna()
    aligned.columns = ["fund", "bm", "dd"]

    conditions = [
        aligned["dd"] >= bull_threshold,
        (aligned["dd"] >= bear_threshold) & (aligned["dd"] < bull_threshold),
        aligned["dd"] < bear_threshold,
    ]
    aligned["regime"] = np.select(conditions, ["Bull", "Correction", "Bear"], default="Bull")

    rows = []
    for regime in ["Bull", "Correction", "Bear"]:
        seg = aligned[aligned["regime"] == regime]
        if len(seg) < 5:
            continue
        fund_cum = (1 + seg["fund"]).prod()
        bm_cum = (1 + seg["bm"]).prod()
        years = len(seg) / TRADING_DAYS
        fund_cagr = (fund_cum ** (1 / years) - 1) * 100 if years > 0.1 else np.nan
        bm_cagr = (bm_cum ** (1 / years) - 1) * 100 if years > 0.1 else np.nan
        hit_rate = (seg["fund"] > seg["bm"]).mean() * 100
        active = (fund_cagr - bm_cagr) if not (np.isnan(fund_cagr) or np.isnan(bm_cagr)) else np.nan
        rows.append({
            "regime": regime,
            "fund_cagr": round(fund_cagr, 2),
            "bm_cagr": round(bm_cagr, 2),
            "active_return": round(active, 2) if not np.isnan(active) else np.nan,
            "days": len(seg),
            "hit_rate_pct": round(hit_rate, 1),
        })

    return pd.DataFrame(rows)


def full_analytics(
    nav: pd.Series,
    benchmark: Optional[pd.Series] = None,
    rf: float = 0.065,
    rolling_windows: Optional[List[int]] = None,
) -> dict:
    """
    Compute all analytics at once and return as a structured dict.

    Parameters
    ----------
    nav             : pd.Series  date-indexed NAV
    benchmark       : pd.Series, optional
    rf              : float      annual risk-free rate
    rolling_windows : list of int  default [63, 126, 252]
    """
    if rolling_windows is None:
        rolling_windows = [63, 126, 252]

    result = {
        "period_returns": period_returns(nav),
        "drawdown_table": drawdown_table(nav),
        "var_cvar": var_cvar(nav),
        "pain_index": pain_index(nav),
        "ulcer_index": ulcer_index(nav),
        "omega_ratio": omega_ratio(nav),
        "calmar_ratio": calmar_ratio(nav),
        "rolling_returns": {w: rolling_returns(nav, window=w) for w in rolling_windows},
        "rolling_sharpe": {w: rolling_sharpe(nav, window=w, rf=rf) for w in rolling_windows},
        "rolling_vol": {w: rolling_volatility(nav, window=w) for w in rolling_windows},
        "monthly_returns": monthly_returns_table(nav),
    }

    if benchmark is not None:
        result.update({
            "alpha_beta": alpha_beta(nav, benchmark, rf),
            "information_ratio": information_ratio(nav, benchmark),
            "updown_capture": updown_capture(nav, benchmark),
            "regime_returns": regime_returns(nav, benchmark),
            "rolling_alpha": {
                w: rolling_alpha(nav, benchmark, window=w, rf=rf) for w in rolling_windows
            },
            "rolling_beta": {
                w: rolling_beta(nav, benchmark, window=w) for w in rolling_windows
            },
        })

    return result
