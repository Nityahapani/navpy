"""
navpy — Indian Mutual Fund NAV fetcher & analyser
===================================================

Fetch day-wise NAV data for any Indian mutual fund from AMFI,
compare against benchmarks, and run advanced rolling analytics.

Quick start
-----------
>>> import navpy
>>>
>>> # Fetch by fund name
>>> result = navpy.get("Mirae Asset Large Cap", start="3y")
>>> result.print_summary()
>>> df = result.data          # pandas DataFrame: date, nav
>>>
>>> # Fetch by AMFI code
>>> result = navpy.get("107578", start="2018-01-01", end="2023-12-31")
>>>
>>> # Direct plan only
>>> result = navpy.get("HDFC Flexi Cap", plan="direct", start="5y")
>>>
>>> # Search for funds
>>> results = navpy.search("mirae asset")
>>> for r in results:
...     print(r)
>>>
>>> # Compare against benchmarks
>>> cmp = navpy.compare(result, benchmarks=["nifty50", "nifty_midcap150"])
>>> cmp.print_summary()
>>>
>>> # See all available benchmarks
>>> navpy.list_benchmarks()
>>>
>>> # Analytics
>>> result.cagr()
>>> result.rolling_returns(window=252)
>>> result.rolling_alpha("nifty50", window=252)
>>> result.drawdown_table()
>>> result.period_returns()
>>> result.regime_returns("nifty50")
"""

from .core import get
from .search import search
from .compare import compare
from .models import NAVResult, SchemeInfo
from .benchmarks import get_benchmark, list_benchmarks, BENCHMARKS
from .exceptions import (
    NavpyError,
    SchemeNotFoundError,
    AmbiguousSchemeError,
    FetchError,
    NoDataError,
    InvalidDateError,
)
from . import cache
from . import analytics

__version__ = "1.0.0"
__author__ = "navpy"
__all__ = [
    "get",
    "search",
    "compare",
    "NAVResult",
    "SchemeInfo",
    "get_benchmark",
    "list_benchmarks",
    "BENCHMARKS",
    "cache",
    "analytics",
    "NavpyError",
    "SchemeNotFoundError",
    "AmbiguousSchemeError",
    "FetchError",
    "NoDataError",
    "InvalidDateError",
]
