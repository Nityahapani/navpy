"""Data models for navpy."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import pandas as pd
import numpy as np


@dataclass
class SchemeInfo:
    """Metadata for a single mutual fund scheme."""

    scheme_code: str
    scheme_name: str
    fund_house: str = ""
    scheme_type: str = ""
    scheme_category: str = ""
    scheme_nav_name: str = ""
    isin_growth: str = ""
    isin_div_reinvestment: str = ""

    def __str__(self) -> str:
        return f"{self.scheme_code}  |  {self.scheme_name}"

    def __repr__(self) -> str:
        return f"SchemeInfo(code={self.scheme_code!r}, name={self.scheme_name!r})"


@dataclass
class NAVResult:
    """
    Result object returned by navpy.get().

    Contains the day-wise NAV DataFrame and convenience analytics methods.
    """

    scheme_code: str
    scheme_name: str
    plan: str
    start_date: Optional[str]
    end_date: Optional[str]
    data: pd.DataFrame
    _reg_code: str = field(default="", repr=False)
    _dir_code: str = field(default="", repr=False)

    def __len__(self) -> int:
        return len(self.data)

    def __repr__(self) -> str:
        if self.data.empty:
            return f"NAVResult(scheme={self.scheme_name!r}, records=0)"
        s = self.data["date"].iloc[0].strftime("%Y-%m-%d")
        e = self.data["date"].iloc[-1].strftime("%Y-%m-%d")
        return (
            f"NAVResult(\n"
            f"  scheme     = {self.scheme_name!r}\n"
            f"  code       = {self.scheme_code}\n"
            f"  plan       = {self.plan}\n"
            f"  period     = {s}  ->  {e}\n"
            f"  records    = {len(self.data)}\n"
            f"  nav_range  = {self.data['nav'].min():.4f}  ->  {self.data['nav'].max():.4f}\n"
            f")"
        )

    # ── Export ────────────────────────────────────────────────────────────────

    def to_csv(self, path: str) -> None:
        """Save NAV data to CSV."""
        self.data.to_csv(path, index=False)
        print(f"Saved {len(self.data)} records to {path}")

    def to_json(self, path: Optional[str] = None) -> Optional[str]:
        """Return NAV data as JSON string, or save to file."""
        json_str = self.data.to_json(orient="records", date_format="iso", indent=2)
        if path:
            with open(path, "w") as f:
                f.write(json_str)
            print(f"Saved {len(self.data)} records to {path}")
            return None
        return json_str

    def to_series(self) -> pd.Series:
        """Return NAV as a date-indexed pandas Series."""
        return self.data.set_index("date")["nav"].rename(self.scheme_name)

    # ── Basic analytics ───────────────────────────────────────────────────────

    def returns(self, period: str = "daily") -> pd.DataFrame:
        """
        Compute returns over a given period.

        Parameters
        ----------
        period : str  'daily', 'weekly', 'monthly', or 'yearly'
        """
        if self.data.empty:
            return pd.DataFrame(columns=["date", "nav", "return_pct"])

        from .clean import _resample_freq
        df = self.data.set_index("date")["nav"].copy()

        if period == "daily":
            ret = df.pct_change() * 100
        elif period == "weekly":
            df = df.resample("W").last()
            ret = df.pct_change() * 100
        elif period == "monthly":
            freq = _resample_freq("ME")
            df = df.resample(freq).last()
            ret = df.pct_change() * 100
        elif period == "yearly":
            freq = _resample_freq("YE")
            df = df.resample(freq).last()
            ret = df.pct_change() * 100
        else:
            raise ValueError(f"Unknown period '{period}'. Use: daily, weekly, monthly, yearly.")

        out = pd.DataFrame({"nav": df, "return_pct": ret}).dropna().reset_index()
        out.rename(columns={"index": "date"}, inplace=True)
        return out

    def cagr(self, start: Optional[str] = None, end: Optional[str] = None) -> float:
        """CAGR % over the full period or a sub-range."""
        df = self._slice(start, end)
        if df.empty or len(df) < 2:
            return float("nan")
        years = (df["date"].iloc[-1] - df["date"].iloc[0]).days / 365.25
        if years <= 0:
            return float("nan")
        c = (df["nav"].iloc[-1] / df["nav"].iloc[0]) ** (1 / years) - 1
        return round(c * 100, 4)

    def abs_return(self, start: Optional[str] = None, end: Optional[str] = None) -> float:
        """Absolute point-to-point return %."""
        df = self._slice(start, end)
        if df.empty or len(df) < 2:
            return float("nan")
        return round((df["nav"].iloc[-1] / df["nav"].iloc[0] - 1) * 100, 4)

    def nav_on(self, date: str) -> float:
        """Return NAV on a specific date (or nearest prior available date)."""
        target = pd.Timestamp(date)
        idx_series = self.data.set_index("date")["nav"]
        if target in idx_series.index:
            return float(idx_series[target])
        prior = idx_series[idx_series.index <= target]
        if prior.empty:
            raise ValueError(f"No NAV available on or before {date}.")
        return float(prior.iloc[-1])

    def max_drawdown(self) -> float:
        """Maximum peak-to-trough drawdown %."""
        if self.data.empty:
            return float("nan")
        nav = self.data.set_index("date")["nav"]
        roll_max = nav.cummax()
        dd = (nav - roll_max) / roll_max * 100
        return round(float(dd.min()), 4)

    def volatility(self, annualised: bool = True) -> float:
        """Annualised standard deviation of daily returns %."""
        if self.data.empty or len(self.data) < 2:
            return float("nan")
        dr = self.data.set_index("date")["nav"].pct_change().dropna() * 100
        vol = dr.std()
        if annualised:
            vol *= np.sqrt(252)
        return round(float(vol), 4)

    def summary(self) -> Dict[str, Any]:
        """Return a summary dictionary of key statistics."""
        if self.data.empty:
            return {"error": "No data available."}
        return {
            "scheme_name": self.scheme_name,
            "scheme_code": self.scheme_code,
            "plan": self.plan,
            "start": str(self.data["date"].iloc[0].date()),
            "end": str(self.data["date"].iloc[-1].date()),
            "records": len(self.data),
            "start_nav": round(float(self.data["nav"].iloc[0]), 4),
            "end_nav": round(float(self.data["nav"].iloc[-1]), 4),
            "cagr_pct": self.cagr(),
            "abs_return_pct": self.abs_return(),
            "max_drawdown_pct": self.max_drawdown(),
            "volatility_pct": self.volatility(),
        }

    def print_summary(self) -> None:
        """Print a formatted summary to the terminal."""
        s = self.summary()
        if "error" in s:
            print(s["error"])
            return
        w = 52
        sep = "-" * w
        print(f"\n{'NAV SUMMARY':^{w}}")
        print(sep)
        print(f"  Scheme      : {s['scheme_name']}")
        print(f"  Code        : {s['scheme_code']}   Plan: {s['plan']}")
        print(f"  Period      : {s['start']}  ->  {s['end']}")
        print(f"  Records     : {s['records']} trading days")
        print(sep)
        print(f"  Start NAV   : Rs {s['start_nav']:.4f}")
        print(f"  End NAV     : Rs {s['end_nav']:.4f}")
        print(sep)
        print(f"  CAGR        : {s['cagr_pct']:>8.2f} %")
        print(f"  Total Ret   : {s['abs_return_pct']:>8.2f} %")
        print(f"  Max DD      : {s['max_drawdown_pct']:>8.2f} %")
        print(f"  Volatility  : {s['volatility_pct']:>8.2f} % p.a.")
        print(sep + "\n")

    # ── Analytics delegation ──────────────────────────────────────────────────

    def rolling_returns(self, window: int = 252, annualise: bool = True) -> pd.Series:
        """Rolling returns. See navpy.analytics.rolling_returns for full docs."""
        from .analytics import rolling_returns as _rr
        return _rr(self.to_series(), window=window, annualise=annualise)

    def rolling_alpha(self, benchmark: Any, window: int = 252, rf: float = 0.065) -> pd.Series:
        """Rolling Jensen's alpha. benchmark = pd.Series or benchmark alias string."""
        from .analytics import rolling_alpha as _ra
        bm = self._resolve_bm(benchmark)
        return _ra(self.to_series(), bm, window=window, rf=rf)

    def rolling_beta(self, benchmark: Any, window: int = 252) -> pd.Series:
        """Rolling beta."""
        from .analytics import rolling_beta as _rb
        bm = self._resolve_bm(benchmark)
        return _rb(self.to_series(), bm, window=window)

    def rolling_sharpe(self, window: int = 252, rf: float = 0.065) -> pd.Series:
        """Rolling Sharpe ratio."""
        from .analytics import rolling_sharpe as _rs
        return _rs(self.to_series(), window=window, rf=rf)

    def rolling_sortino(self, window: int = 252, rf: float = 0.065) -> pd.Series:
        """Rolling Sortino ratio."""
        from .analytics import rolling_sortino as _rso
        return _rso(self.to_series(), window=window, rf=rf)

    def rolling_volatility(self, window: int = 63) -> pd.Series:
        """Rolling annualised volatility."""
        from .analytics import rolling_volatility as _rv
        return _rv(self.to_series(), window=window)

    def drawdown_table(self, top_n: int = 10) -> pd.DataFrame:
        """Table of top N worst drawdown episodes."""
        from .analytics import drawdown_table as _dt
        return _dt(self.to_series(), top_n=top_n)

    def period_returns(self) -> pd.DataFrame:
        """Standard period returns: 1M, 3M, 6M, 1Y, 3Y, 5Y, inception."""
        from .analytics import period_returns as _pr
        return _pr(self.to_series())

    def monthly_returns_table(self) -> pd.DataFrame:
        """Monthly returns matrix (rows=year, cols=month)."""
        from .analytics import monthly_returns_table as _mrt
        return _mrt(self.to_series())

    def var_cvar(self, confidence: float = 0.95) -> Dict[str, float]:
        """Value at Risk and Conditional VaR."""
        from .analytics import var_cvar as _vc
        return _vc(self.to_series(), confidence=confidence)

    def omega_ratio(self, threshold: float = 0.0) -> float:
        """Omega ratio."""
        from .analytics import omega_ratio as _om
        return _om(self.to_series(), threshold=threshold)

    def ulcer_index(self) -> float:
        """Ulcer index."""
        from .analytics import ulcer_index as _ui
        return _ui(self.to_series())

    def pain_index(self) -> float:
        """Pain index (mean drawdown)."""
        from .analytics import pain_index as _pi
        return _pi(self.to_series())

    def regime_returns(self, benchmark: Any) -> pd.DataFrame:
        """Returns breakdown by Bull/Correction/Bear regime."""
        from .analytics import regime_returns as _rr
        bm = self._resolve_bm(benchmark)
        return _rr(self.to_series(), bm)

    def alpha_beta(self, benchmark: Any, rf: float = 0.065) -> Dict[str, float]:
        """Full-period Jensen's alpha and beta."""
        from .analytics import alpha_beta as _ab
        bm = self._resolve_bm(benchmark)
        return _ab(self.to_series(), bm, rf=rf)

    def information_ratio(self, benchmark: Any) -> float:
        """Information ratio vs benchmark."""
        from .analytics import information_ratio as _ir
        bm = self._resolve_bm(benchmark)
        return _ir(self.to_series(), bm)

    def updown_capture(self, benchmark: Any) -> Dict[str, float]:
        """Upside and downside capture ratios."""
        from .analytics import updown_capture as _ud
        bm = self._resolve_bm(benchmark)
        return _ud(self.to_series(), bm)

    def full_analytics(self, benchmark: Any = None, rf: float = 0.065) -> dict:
        """All analytics in one dict. Pass benchmark alias or Series."""
        from .analytics import full_analytics as _fa
        bm = self._resolve_bm(benchmark) if benchmark is not None else None
        return _fa(self.to_series(), benchmark=bm, rf=rf)

    def compare(self, benchmarks: Any = None, **kwargs: Any) -> Any:
        """Compare this result against benchmarks. See navpy.compare() for docs."""
        from .compare import compare as _cmp
        return _cmp(self, benchmarks=benchmarks, **kwargs)

    def _resolve_bm(self, benchmark: Any) -> pd.Series:
        """Resolve benchmark to a pd.Series — accepts alias string or Series."""
        if isinstance(benchmark, pd.Series):
            return benchmark
        if isinstance(benchmark, str):
            from .benchmarks import get_benchmark
            return get_benchmark(
                benchmark,
                start=self.start_date,
                end=self.end_date,
            )
        raise TypeError(
            f"benchmark must be a string alias or pd.Series, got {type(benchmark)}"
        )

    def _slice(self, start: Optional[str] = None, end: Optional[str] = None) -> pd.DataFrame:
        df = self.data.copy()
        if start:
            df = df[df["date"] >= pd.Timestamp(start)]
        if end:
            df = df[df["date"] <= pd.Timestamp(end)]
        return df.reset_index(drop=True)
