"""
navpy.compare — Compare a NAVResult against one or more benchmarks.

Usage
-----
>>> import navpy
>>> result = navpy.get("Mirae Asset Large Cap", start="5y", plan="direct")
>>> cmp = navpy.compare(result, benchmarks=["nifty50", "nifty_midcap150"])
>>> cmp.print_summary()
>>> cmp.to_dataframe()
"""

from __future__ import annotations
from typing import Optional, Union
import pandas as pd
import numpy as np

from .models import NAVResult
from .benchmarks import get_benchmark, resolve_benchmark
from .analytics import (
    alpha_beta, information_ratio, updown_capture,
    rolling_returns, rolling_alpha,
    period_returns, regime_returns,
)
from .clean import parse_dates

TRADING_DAYS = 252


class ComparisonResult:
    """
    Result of comparing a NAVResult against one or more benchmarks.

    Attributes
    ----------
    nav_result      : NAVResult
    benchmarks      : dict  {alias: pd.Series}
    metrics_table   : pd.DataFrame  one row per series (fund + each benchmark)
    """

    def __init__(
        self,
        nav_result: NAVResult,
        benchmarks: dict,
        metrics: pd.DataFrame,
        period_rets: dict,
        regime_rets: dict,
        rolling_rets: dict,
        rolling_alphas: dict,
    ):
        self.nav_result = nav_result
        self.benchmarks = benchmarks
        self.metrics_table = metrics
        self._period_rets = period_rets
        self._regime_rets = regime_rets
        self._rolling_rets = rolling_rets
        self._rolling_alphas = rolling_alphas

    # ── Display ───────────────────────────────────────────────────────────────

    def print_summary(self) -> None:
        """Print a formatted comparison table to the terminal."""
        print(f"\n{'BENCHMARK COMPARISON':^70}")
        print("─" * 70)
        print(f"  Fund   : {self.nav_result.scheme_name}")
        print(f"  Period : {self.nav_result.start_date}  →  {self.nav_result.end_date}")
        print("─" * 70)
        print(self.metrics_table.to_string())
        print("─" * 70)

        if self._period_rets:
            print("\n  PERIOD RETURNS")
            print("─" * 70)
            combined = None
            for name, df in self._period_rets.items():
                df2 = df.set_index("period")[["return_pct"]].rename(
                    columns={"return_pct": name[:25]}
                )
                combined = df2 if combined is None else combined.join(df2, how="outer")
            if combined is not None:
                print(combined.to_string())
            print("─" * 70)

        if self._regime_rets:
            print("\n  REGIME PERFORMANCE (Active Return %)")
            print("─" * 70)
            for bm_name, df in self._regime_rets.items():
                print(f"\n  vs {bm_name}")
                if df is not None and not df.empty:
                    print(df[["regime", "fund_cagr", "bm_cagr",
                              "active_return", "hit_rate_pct"]].to_string(index=False))
            print("─" * 70)

        print()

    # ── Accessors ─────────────────────────────────────────────────────────────

    def to_dataframe(self) -> pd.DataFrame:
        """Return the main metrics comparison as a DataFrame."""
        return self.metrics_table.copy()

    def period_returns(self) -> dict:
        """Return period returns dict {series_name: DataFrame}."""
        return self._period_rets

    def regime_analysis(self) -> dict:
        """Return regime returns dict {benchmark_name: DataFrame}."""
        return self._regime_rets

    def rolling_returns(self, window: int = 252) -> pd.DataFrame:
        """
        Return rolling returns for fund + all benchmarks aligned on one DataFrame.

        Parameters
        ----------
        window : int  trading days (default 252 = 1 year)

        Returns
        -------
        pd.DataFrame  date-indexed, one column per series
        """
        data = {}
        for name, series in self._rolling_rets.items():
            if window in series:
                data[name] = series[window]
        if not data:
            return pd.DataFrame()
        return pd.concat(data, axis=1)

    def rolling_alpha(self, window: int = 252) -> pd.DataFrame:
        """
        Return rolling alpha for fund vs each benchmark.

        Returns
        -------
        pd.DataFrame  date-indexed, one column per benchmark
        """
        data = {}
        for bm_name, windows in self._rolling_alphas.items():
            if window in windows:
                data[f"alpha_vs_{bm_name[:20]}"] = windows[window]
        if not data:
            return pd.DataFrame()
        return pd.concat(data, axis=1)

    def to_csv(self, path: str) -> None:
        """Save main metrics table to CSV."""
        self.metrics_table.to_csv(path)
        print(f"Saved comparison to {path}")


def compare(
    result: NAVResult,
    benchmarks: Optional[Union[str, list]] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    rf: float = 0.065,
    rolling_windows: Optional[list] = None,
) -> ComparisonResult:
    """
    Compare a NAVResult against one or more benchmarks.

    Parameters
    ----------
    result : NAVResult
        The fund NAV result to compare (from navpy.get()).

    benchmarks : str or list of str, optional
        Benchmark identifiers. Each can be an alias, shorthand, ticker, or name.
        Examples:
            "nifty50"
            ["nifty50", "nifty_midcap150"]
            ["nifty50", "sensex", "sp500"]
        Default: ["nifty50"] (Nifty 50)
        Use navpy.list_benchmarks() to see all available.

    start : str, optional
        Override start date for comparison period.
        Defaults to result's own start date.

    end : str, optional
        Override end date for comparison period.
        Defaults to result's own end date.

    rf : float, optional
        Annual risk-free rate for alpha/Sharpe calculation (default 6.5%).

    rolling_windows : list of int, optional
        Windows for rolling metrics in trading days.
        Default: [126, 252]  (6 months, 1 year)

    Returns
    -------
    ComparisonResult
        with attributes:
          .metrics_table        pd.DataFrame  main comparison metrics
          .print_summary()      formatted terminal output
          .to_dataframe()       returns metrics_table
          .rolling_returns(w)   rolling returns DataFrame
          .rolling_alpha(w)     rolling alpha vs each benchmark
          .period_returns()     dict of period return tables
          .regime_analysis()    dict of regime performance tables
          .to_csv(path)         save metrics to CSV

    Examples
    --------
    >>> import navpy
    >>> result = navpy.get("Mirae Asset Large Cap", start="5y", plan="direct")
    >>>
    >>> # Single benchmark
    >>> cmp = navpy.compare(result, "nifty50")
    >>> cmp.print_summary()
    >>>
    >>> # Multiple benchmarks
    >>> cmp = navpy.compare(result, ["nifty50", "nifty_midcap150", "sensex"])
    >>> df = cmp.to_dataframe()
    >>>
    >>> # Rolling alpha chart data
    >>> ra = cmp.rolling_alpha(window=252)
    >>> ra.plot()
    """
    if rolling_windows is None:
        rolling_windows = [126, 252]

    if benchmarks is None:
        benchmarks = ["nifty50"]
    elif isinstance(benchmarks, str):
        benchmarks = [benchmarks]

    # ── Determine comparison date range ───────────────────────────────────────
    start_ts, end_ts = parse_dates(start, end)
    if start_ts is None:
        start_ts = result.data["date"].iloc[0]
    if end_ts is None:
        end_ts = result.data["date"].iloc[-1]

    # ── Clip fund NAV to comparison window ───────────────────────────────────
    fund_nav = result.to_series()
    fund_nav = fund_nav[(fund_nav.index >= start_ts) & (fund_nav.index <= end_ts)]

    if fund_nav.empty:
        raise ValueError("No fund data in the specified comparison period.")

    # ── Fetch benchmark series ────────────────────────────────────────────────
    bm_series = {}
    failed = []
    for bm_id in benchmarks:
        try:
            bm = get_benchmark(bm_id, start=str(start_ts.date()), end=str(end_ts.date()))
            bm_series[resolve_benchmark(bm_id)["name"]] = bm
        except Exception as e:
            failed.append(f"{bm_id}: {e}")

    if failed:
        print(f"  ⚠ Could not load benchmark(s): {'; '.join(failed)}")

    if not bm_series:
        raise ValueError("No benchmarks could be loaded. Check identifiers or network.")

    # ── Compute metrics for fund and each benchmark ───────────────────────────
    def _scalar_metrics(nav_s: pd.Series, rf: float) -> dict:
        """Compute standard scalar metrics for a NAV series."""
        dr = nav_s.pct_change().dropna()
        years = (nav_s.index[-1] - nav_s.index[0]).days / 365.25
        cagr = ((nav_s.iloc[-1] / nav_s.iloc[0]) ** (1 / years) - 1) * 100 if years > 0 else np.nan
        vol = dr.std() * np.sqrt(TRADING_DAYS) * 100
        mdd = ((nav_s - nav_s.cummax()) / nav_s.cummax()).min() * 100
        sh = (cagr / 100 - rf) / (vol / 100) if vol > 0 else np.nan
        so_dn = dr[dr < 0].std() * np.sqrt(TRADING_DAYS) * 100
        so = (cagr / 100 - rf) / (so_dn / 100) if so_dn > 0 else np.nan
        calmar = (cagr / 100) / abs(mdd / 100) if mdd != 0 else np.nan
        return {
            "CAGR %": round(cagr, 2),
            "Volatility %": round(vol, 2),
            "Sharpe": round(sh, 3) if not np.isnan(sh) else np.nan,
            "Sortino": round(so, 3) if not np.isnan(so) else np.nan,
            "Max DD %": round(mdd, 2),
            "Calmar": round(calmar, 3) if not np.isnan(calmar) else np.nan,
        }

    rows = {}
    fund_label = result.scheme_name[:40]
    rows[fund_label] = _scalar_metrics(fund_nav, rf)

    for bm_name, bm_s in bm_series.items():
        # Align benchmark to fund dates
        bm_aligned = bm_s.reindex(fund_nav.index, method="ffill").dropna()
        if len(bm_aligned) < 30:
            continue
        rows[bm_name[:40]] = _scalar_metrics(bm_aligned, rf)

    metrics_df = pd.DataFrame(rows).T

    # Add benchmark-relative metrics for the fund row
    for bm_name, bm_s in bm_series.items():
        bm_aligned = bm_s.reindex(fund_nav.index, method="ffill").dropna()
        if len(bm_aligned) < 30:
            continue
        ab = alpha_beta(fund_nav, bm_aligned, rf)
        ir = information_ratio(fund_nav, bm_aligned)
        ud = updown_capture(fund_nav, bm_aligned)
        col = bm_name[:20]
        metrics_df.loc[fund_label, f"Alpha({col})"] = round(ab["alpha"] * 100, 2)
        metrics_df.loc[fund_label, f"Beta({col})"] = round(ab["beta"], 3)
        metrics_df.loc[fund_label, f"IR({col})"] = round(ir, 3) if not np.isnan(ir) else np.nan
        metrics_df.loc[fund_label, f"UpCap({col})"] = ud["upside_capture"]
        metrics_df.loc[fund_label, f"DnCap({col})"] = ud["downside_capture"]

    # ── Period returns ────────────────────────────────────────────────────────
    period_rets = {fund_label: period_returns(fund_nav)}
    for bm_name, bm_s in bm_series.items():
        bm_aligned = bm_s.reindex(fund_nav.index, method="ffill").dropna()
        if len(bm_aligned) > 30:
            period_rets[bm_name[:40]] = period_returns(bm_aligned)

    # ── Regime returns ────────────────────────────────────────────────────────
    regime_rets = {}
    for bm_name, bm_s in bm_series.items():
        bm_aligned = bm_s.reindex(fund_nav.index, method="ffill").dropna()
        if len(bm_aligned) > 60:
            try:
                regime_rets[bm_name] = regime_returns(fund_nav, bm_aligned)
            except Exception:
                regime_rets[bm_name] = None

    # ── Rolling metrics ───────────────────────────────────────────────────────
    rolling_rets = {}
    # Fund
    fund_roll = {}
    for w in rolling_windows:
        fund_roll[w] = rolling_returns(fund_nav, window=w)
    rolling_rets[fund_label] = fund_roll

    # Benchmarks
    for bm_name, bm_s in bm_series.items():
        bm_aligned = bm_s.reindex(fund_nav.index, method="ffill").dropna()
        if len(bm_aligned) > max(rolling_windows):
            bm_roll = {}
            for w in rolling_windows:
                bm_roll[w] = rolling_returns(bm_aligned, window=w)
            rolling_rets[bm_name[:40]] = bm_roll

    # Rolling alphas
    rolling_alphas = {}
    for bm_name, bm_s in bm_series.items():
        bm_aligned = bm_s.reindex(fund_nav.index, method="ffill").dropna()
        if len(bm_aligned) > max(rolling_windows):
            bm_alphas = {}
            for w in rolling_windows:
                if len(bm_aligned) > w:
                    try:
                        bm_alphas[w] = rolling_alpha(fund_nav, bm_aligned, window=w, rf=rf)
                    except Exception:
                        pass
            rolling_alphas[bm_name[:40]] = bm_alphas

    return ComparisonResult(
        nav_result=result,
        benchmarks=bm_series,
        metrics=metrics_df,
        period_rets=period_rets,
        regime_rets=regime_rets,
        rolling_rets=rolling_rets,
        rolling_alphas=rolling_alphas,
    )
