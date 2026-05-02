"""
navpy test suite.

Tests are split into:
  - Unit tests (no network) — mock all HTTP calls
  - Integration tests (live network) — marked with @pytest.mark.live
    Run with: pytest tests/ -m live
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from datetime import datetime

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


# ── Fixtures & mock data ──────────────────────────────────────────────────────

def make_nav_data(n=500, start="2020-01-01", start_nav=100.0):
    """Generate synthetic NAV data."""
    dates = pd.date_range(start, periods=n, freq='B')
    np.random.seed(42)
    returns = np.random.normal(0.0004, 0.008, n)
    navs    = start_nav * np.cumprod(1 + returns)
    return [
        {"date": d.strftime("%d-%m-%Y"), "nav": f"{v:.4f}"}
        for d, v in zip(dates, navs)
    ]


MOCK_META = {
    "fund_house":         "Test Fund House",
    "scheme_type":        "Open Ended Schemes",
    "scheme_category":    "Equity Scheme - Flexi Cap Fund",
    "scheme_name":        "Test Flexi Cap Fund - Direct Plan - Growth",
    "scheme_nav_name":    "Test Flexi Cap Fund - Direct Plan - Growth Option",
    "scheme_code":        "999999",
    "isin_growth":        "INF123456789",
    "isin_div_reinvestment": "",
}

MOCK_PAYLOAD = {
    "meta": MOCK_META,
    "data": make_nav_data(500),
    "status": "SUCCESS",
}

MOCK_SEARCH_RESULTS = [
    {"schemeCode": 999999, "schemeName": "Test Flexi Cap Fund - Direct Plan - Growth"},
    {"schemeCode": 999998, "schemeName": "Test Flexi Cap Fund - Regular Plan - Growth"},
    {"schemeCode": 999997, "schemeName": "Test Flexi Cap Fund - Direct Plan - IDCW"},
]


# ── clean.py tests ────────────────────────────────────────────────────────────

class TestClean:

    def test_raw_to_dataframe_basic(self):
        from navpy.clean import raw_to_dataframe
        data = [
            {"date": "01-01-2023", "nav": "150.50"},
            {"date": "02-01-2023", "nav": "151.20"},
        ]
        df = raw_to_dataframe(data)
        assert len(df) == 2
        assert list(df.columns) == ['date', 'nav']
        assert df['nav'].iloc[0] == 150.50
        assert df['date'].iloc[0] == pd.Timestamp("2023-01-01")

    def test_raw_to_dataframe_empty(self):
        from navpy.clean import raw_to_dataframe
        df = raw_to_dataframe([])
        assert df.empty

    def test_raw_to_dataframe_sorted(self):
        from navpy.clean import raw_to_dataframe
        data = [
            {"date": "05-01-2023", "nav": "155.0"},
            {"date": "01-01-2023", "nav": "150.0"},
            {"date": "03-01-2023", "nav": "152.0"},
        ]
        df = raw_to_dataframe(data)
        assert df['date'].is_monotonic_increasing

    def test_clean_nav_removes_zeros(self):
        from navpy.clean import clean_nav
        df = pd.DataFrame({
            'date': pd.date_range("2023-01-01", periods=5, freq='B'),
            'nav':  [100.0, 0.0, 102.0, 103.0, 104.0],
        })
        cleaned = clean_nav(df)
        assert 0.0 not in cleaned['nav'].values
        assert len(cleaned) == 4

    def test_clean_nav_removes_negatives(self):
        from navpy.clean import clean_nav
        df = pd.DataFrame({
            'date': pd.date_range("2023-01-01", periods=4, freq='B'),
            'nav':  [100.0, -5.0, 102.0, 103.0],
        })
        cleaned = clean_nav(df)
        assert all(cleaned['nav'] > 0)

    def test_clean_nav_removes_outliers(self):
        from navpy.clean import clean_nav
        # 200% single day move should be removed
        df = pd.DataFrame({
            'date': pd.date_range("2023-01-01", periods=4, freq='B'),
            'nav':  [100.0, 101.0, 305.0, 103.0],
        })
        cleaned = clean_nav(df)
        assert 305.0 not in cleaned['nav'].values

    def test_clean_nav_keeps_normal_moves(self):
        from navpy.clean import clean_nav
        df = pd.DataFrame({
            'date': pd.date_range("2023-01-01", periods=4, freq='B'),
            'nav':  [100.0, 102.0, 104.0, 103.5],
        })
        cleaned = clean_nav(df)
        assert len(cleaned) == 4

    def test_parse_dates_literal(self):
        from navpy.clean import parse_dates
        start, end = parse_dates("2020-01-01", "2023-12-31")
        assert start == pd.Timestamp("2020-01-01")
        assert end   == pd.Timestamp("2023-12-31")

    def test_parse_dates_none(self):
        from navpy.clean import parse_dates
        start, end = parse_dates(None, None)
        assert start is None
        assert end   is None

    def test_parse_dates_max(self):
        from navpy.clean import parse_dates
        start, _ = parse_dates("max", None)
        assert start is None

    def test_parse_dates_relative_year(self):
        from navpy.clean import parse_dates
        start, _ = parse_dates("3y", None)
        today = pd.Timestamp.today().normalize()
        expected = today - pd.DateOffset(years=3)
        # Allow 2-day tolerance for date arithmetic
        assert abs((start - expected).days) <= 2

    def test_parse_dates_relative_month(self):
        from navpy.clean import parse_dates
        start, _ = parse_dates("6m", None)
        today = pd.Timestamp.today().normalize()
        expected = today - pd.DateOffset(months=6)
        assert abs((start - expected).days) <= 2

    def test_parse_dates_ytd(self):
        from navpy.clean import parse_dates
        start, _ = parse_dates("ytd", None)
        today = pd.Timestamp.today()
        assert start == pd.Timestamp(today.year, 1, 1)

    def test_parse_dates_invalid_raises(self):
        from navpy.clean import parse_dates
        from navpy.exceptions import InvalidDateError
        with pytest.raises(InvalidDateError):
            parse_dates("not-a-date", None)

    def test_parse_dates_alternate_formats(self):
        from navpy.clean import parse_dates
        s1, _ = parse_dates("01-06-2022", None)
        s2, _ = parse_dates("01/06/2022", None)
        assert s1 == pd.Timestamp("2022-06-01")
        assert s2 == pd.Timestamp("2022-06-01")

    def test_apply_date_filter(self):
        from navpy.clean import apply_date_filter
        df = pd.DataFrame({
            'date': pd.date_range("2020-01-01", periods=100, freq='B'),
            'nav':  np.linspace(100, 200, 100),
        })
        filtered = apply_date_filter(
            df,
            pd.Timestamp("2020-03-01"),
            pd.Timestamp("2020-06-30"),
        )
        assert filtered['date'].min() >= pd.Timestamp("2020-03-01")
        assert filtered['date'].max() <= pd.Timestamp("2020-06-30")


# ── cache.py tests ────────────────────────────────────────────────────────────

class TestCache:

    def test_set_and_get(self, tmp_path):
        import navpy.cache as c
        original_dir = c.CACHE_DIR
        c.CACHE_DIR = tmp_path / "cache"
        try:
            c.set("TEST001", {"meta": {}, "data": [{"date": "01-01-2023", "nav": "100"}]})
            result = c.get("TEST001")
            assert result is not None
            assert result["data"][0]["nav"] == "100"
        finally:
            c.CACHE_DIR = original_dir

    def test_get_missing_returns_none(self, tmp_path):
        import navpy.cache as c
        original_dir = c.CACHE_DIR
        c.CACHE_DIR = tmp_path / "cache"
        try:
            result = c.get("NONEXISTENT")
            assert result is None
        finally:
            c.CACHE_DIR = original_dir

    def test_invalidate(self, tmp_path):
        import navpy.cache as c
        original_dir = c.CACHE_DIR
        c.CACHE_DIR = tmp_path / "cache"
        try:
            c.set("TEST002", {"meta": {}, "data": []})
            assert c.get("TEST002") is not None
            deleted = c.invalidate("TEST002")
            assert deleted is True
            assert c.get("TEST002") is None
        finally:
            c.CACHE_DIR = original_dir

    def test_clear_all(self, tmp_path):
        import navpy.cache as c
        original_dir = c.CACHE_DIR
        c.CACHE_DIR = tmp_path / "cache"
        try:
            for i in range(3):
                c.set(f"TEST{i:03d}", {"meta": {}, "data": []})
            n = c.clear_all()
            assert n == 3
            info = c.cache_info()
            assert info['file_count'] == 0
        finally:
            c.CACHE_DIR = original_dir

    def test_ttl_stale(self, tmp_path):
        import navpy.cache as c
        import time
        original_dir = c.CACHE_DIR
        c.CACHE_DIR = tmp_path / "cache"
        try:
            c.set("TEST003", {"meta": {}, "data": []})
            # TTL of 0 means always stale
            result = c.get("TEST003", ttl=0)
            assert result is None
        finally:
            c.CACHE_DIR = original_dir


# ── models.py tests ───────────────────────────────────────────────────────────

class TestNAVResult:

    def _make_result(self, n=500):
        from navpy.models import NAVResult
        dates = pd.date_range("2020-01-01", periods=n, freq='B')
        np.random.seed(0)
        navs  = 100 * np.cumprod(1 + np.random.normal(0.0004, 0.008, n))
        df    = pd.DataFrame({'date': dates, 'nav': navs})
        return NAVResult(
            scheme_code="999999",
            scheme_name="Test Fund",
            plan="direct",
            start_date=str(dates[0].date()),
            end_date=str(dates[-1].date()),
            data=df,
        )

    def test_len(self):
        r = self._make_result(100)
        assert len(r) == 100

    def test_cagr_reasonable(self):
        r = self._make_result(500)
        c = r.cagr()
        assert isinstance(c, float)
        assert -50 < c < 200   # sanity bounds

    def test_abs_return(self):
        from navpy.models import NAVResult
        dates = pd.date_range("2023-01-01", periods=2, freq='B')
        df    = pd.DataFrame({'date': dates, 'nav': [100.0, 110.0]})
        r     = NAVResult("c", "n", "direct", None, None, df)
        assert abs(r.abs_return() - 10.0) < 0.01

    def test_max_drawdown_negative(self):
        r = self._make_result(500)
        dd = r.max_drawdown()
        assert dd <= 0

    def test_volatility_positive(self):
        r = self._make_result(500)
        v = r.volatility()
        assert v > 0

    def test_nav_on_exact(self):
        from navpy.models import NAVResult
        # Use explicit dates to avoid freq='B' ambiguity around weekends
        dates = pd.to_datetime(["2023-01-02", "2023-01-03", "2023-01-04", "2023-01-05", "2023-01-06"])
        df    = pd.DataFrame({'date': dates, 'nav': [100.0, 101.0, 102.0, 103.0, 104.0]})
        r     = NAVResult("c", "n", "direct", None, None, df)
        assert r.nav_on("2023-01-04") == 102.0

    def test_nav_on_nearest_prior(self):
        from navpy.models import NAVResult
        dates = pd.date_range("2023-01-02", periods=3, freq='B')  # Mon/Tue/Wed
        df    = pd.DataFrame({'date': dates, 'nav': [100.0, 101.0, 102.0]})
        r     = NAVResult("c", "n", "direct", None, None, df)
        # Weekend date → should return last Friday's value
        val = r.nav_on("2023-01-07")   # Saturday
        assert val == 102.0            # last available

    def test_returns_daily(self):
        r  = self._make_result(100)
        dr = r.returns("daily")
        assert 'return_pct' in dr.columns
        assert len(dr) == 99   # n-1 for daily

    def test_returns_monthly(self):
        r  = self._make_result(500)
        mr = r.returns("monthly")
        assert 'return_pct' in mr.columns
        assert len(mr) > 0

    def test_returns_invalid_period(self):
        r = self._make_result(100)
        with pytest.raises(ValueError):
            r.returns("quarterly")

    def test_to_series(self):
        r  = self._make_result(100)
        s  = r.to_series()
        assert isinstance(s, pd.Series)
        assert s.index.name == 'date'

    def test_summary_keys(self):
        r = self._make_result(200)
        s = r.summary()
        expected_keys = {
            'scheme_name', 'scheme_code', 'plan', 'start', 'end',
            'records', 'start_nav', 'end_nav', 'cagr_pct',
            'abs_return_pct', 'max_drawdown_pct', 'volatility_pct',
        }
        assert expected_keys.issubset(set(s.keys()))

    def test_to_csv(self, tmp_path):
        r    = self._make_result(50)
        path = str(tmp_path / "test_nav.csv")
        r.to_csv(path)
        loaded = pd.read_csv(path)
        assert len(loaded) == 50
        assert 'nav' in loaded.columns

    def test_to_json_string(self):
        r    = self._make_result(10)
        j    = r.to_json()
        import json
        data = json.loads(j)
        assert isinstance(data, list)
        assert len(data) == 10

    def test_empty_result_summary(self):
        from navpy.models import NAVResult
        r = NAVResult("c", "n", "direct", None, None, pd.DataFrame(columns=['date','nav']))
        s = r.summary()
        assert "error" in s


# ── exceptions.py tests ───────────────────────────────────────────────────────

class TestExceptions:

    def test_scheme_not_found(self):
        from navpy.exceptions import SchemeNotFoundError
        e = SchemeNotFoundError("bad query")
        assert "bad query" in str(e)

    def test_no_data_error(self):
        from navpy.exceptions import NoDataError
        e = NoDataError("12345", start="2020-01-01", end="2020-12-31")
        assert "12345" in str(e)

    def test_invalid_date_error(self):
        from navpy.exceptions import InvalidDateError
        e = InvalidDateError("notadate")
        assert "notadate" in str(e)

    def test_fetch_error(self):
        from navpy.exceptions import FetchError
        e = FetchError("99999", "timeout")
        assert "99999" in str(e)


# ── splice.py tests ───────────────────────────────────────────────────────────

class TestSplice:

    def _make_df(self, start, end):
        dates = pd.date_range(start, end, freq='B')
        navs  = np.linspace(100, 200, len(dates))
        return pd.DataFrame({'date': dates, 'nav': navs})

    @patch('navpy.splice._load_and_clean')
    def test_splice_both_present(self, mock_load):
        from navpy.splice import splice

        reg_df = self._make_df("2010-01-01", "2013-01-31")
        dir_df = self._make_df("2013-01-01", "2023-01-01")

        def side_effect(code, force_refresh=False):
            return reg_df if code == "REG" else dir_df

        mock_load.side_effect = side_effect

        df, plan = splice("REG", "DIR")
        assert plan == "splice"
        assert len(df) > 0
        assert df['date'].is_monotonic_increasing

    @patch('navpy.splice._load_and_clean')
    def test_splice_only_direct(self, mock_load):
        from navpy.splice import splice
        dir_df = self._make_df("2015-01-01", "2023-01-01")
        mock_load.side_effect = lambda code, force_refresh=False: (
            pd.DataFrame(columns=['date', 'nav']) if code == "REG" else dir_df
        )
        df, plan = splice("REG", "DIR")
        assert plan == "direct"

    @patch('navpy.splice._load_and_clean')
    def test_splice_neither(self, mock_load):
        from navpy.splice import splice
        mock_load.return_value = pd.DataFrame(columns=['date', 'nav'])
        df, plan = splice("REG", "DIR")
        assert plan == "none"
        assert df.empty

    @patch('navpy.splice._load_and_clean')
    def test_splice_continuity(self, mock_load):
        """Splice point should not create a jump discontinuity."""
        from navpy.splice import splice

        reg_df = self._make_df("2010-01-01", "2012-12-31")
        # Direct starts at 200, regular ends at ~150 — after scaling they should match
        dir_dates = pd.date_range("2013-01-01", "2023-01-01", freq='B')
        dir_navs  = np.linspace(200, 400, len(dir_dates))
        dir_df = pd.DataFrame({'date': dir_dates, 'nav': dir_navs})

        mock_load.side_effect = lambda code, force_refresh=False: reg_df if code == "REG" else dir_df

        df, plan = splice("REG", "DIR")
        # Last regular value and first direct value should be very close
        reg_end = df[df['date'] < pd.Timestamp("2013-01-01")]['nav'].iloc[-1]
        dir_start = df[df['date'] >= pd.Timestamp("2013-01-01")]['nav'].iloc[0]
        assert abs(reg_end - dir_start) / dir_start < 0.05   # within 5%


# ── search.py tests ───────────────────────────────────────────────────────────

class TestSearch:

    @patch('navpy.search._fetch.search_schemes')
    def test_search_returns_scheme_info(self, mock_search):
        from navpy.search import search
        mock_search.return_value = [
            {"schemeCode": "107578", "schemeName": "Mirae Asset Large Cap Fund - Growth Plan"},
        ]
        results = search("mirae large cap")
        assert len(results) == 1
        assert results[0].scheme_code == "107578"

    @patch('navpy.search._fetch.search_schemes')
    def test_search_empty(self, mock_search):
        from navpy.search import search
        mock_search.return_value = []
        results = search("xyznotexist")
        assert results == []

    def test_is_scheme_code(self):
        from navpy.search import is_scheme_code
        assert is_scheme_code("107578") is True
        assert is_scheme_code("mirae asset") is False
        assert is_scheme_code("12abc") is False

    @patch('navpy.search._fetch.fetch_meta')
    @patch('navpy.search._fetch.search_schemes')
    def test_resolve_by_code(self, mock_search, mock_meta):
        from navpy.search import resolve
        mock_meta.return_value = {"scheme_name": "Test Fund", "fund_house": "Test AMC"}
        info = resolve("107578")
        assert info.scheme_code == "107578"
        mock_search.assert_not_called()

    @patch('navpy.search._fetch.fetch_meta')
    @patch('navpy.search._fetch.search_schemes')
    def test_resolve_single_result(self, mock_search, mock_meta):
        from navpy.search import resolve
        mock_search.return_value = [
            {"schemeCode": "107578", "schemeName": "Mirae Asset Large Cap - Direct Plan - Growth"},
        ]
        mock_meta.return_value = {"scheme_name": "Mirae Asset Large Cap - Direct Plan - Growth"}
        info = resolve("mirae large cap direct growth")
        assert info.scheme_code == "107578"

    @patch('navpy.search._fetch.search_schemes')
    def test_resolve_not_found_raises(self, mock_search):
        from navpy.search import resolve
        from navpy.exceptions import SchemeNotFoundError
        mock_search.return_value = []
        with pytest.raises(SchemeNotFoundError):
            resolve("xyznotexist", interactive=False)

    @patch('navpy.search._fetch.fetch_meta')
    @patch('navpy.search._fetch.search_schemes')
    def test_resolve_ambiguous_noninteractive_raises(self, mock_search, mock_meta):
        from navpy.search import resolve
        from navpy.exceptions import AmbiguousSchemeError
        mock_meta.return_value = {}
        mock_search.return_value = [
            {"schemeCode": "111111", "schemeName": "Test Fund Direct Growth"},
            {"schemeCode": "111112", "schemeName": "Test Fund Regular Growth"},
        ]
        with pytest.raises(AmbiguousSchemeError):
            resolve("test fund", interactive=False, plan="splice")


# ── fetch.py tests ────────────────────────────────────────────────────────────

class TestFetch:

    @patch('navpy.fetch.requests.get')
    def test_fetch_scheme_success(self, mock_get, tmp_path):
        import navpy.cache as c
        c.CACHE_DIR = tmp_path / "cache"

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = MOCK_PAYLOAD
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        from navpy.fetch import fetch_scheme
        result = fetch_scheme("999999", force_refresh=True)
        assert result["meta"]["scheme_code"] == "999999"
        assert len(result["data"]) == 500

    @patch('navpy.fetch.requests.get')
    def test_fetch_uses_cache(self, mock_get, tmp_path):
        import navpy.cache as c
        c.CACHE_DIR = tmp_path / "cache"

        # Prime the cache
        c.set("999998", MOCK_PAYLOAD)

        from navpy.fetch import fetch_scheme
        result = fetch_scheme("999998", force_refresh=False)
        # Should not have made an HTTP call
        mock_get.assert_not_called()
        assert result is not None

    @patch('navpy.fetch.requests.get')
    def test_search_schemes_normalises_codes(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = MOCK_SEARCH_RESULTS
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        from navpy.fetch import search_schemes
        results = search_schemes("test fund")
        # All codes should be strings
        for r in results:
            assert isinstance(r["schemeCode"], str)


# ── Integration tests (live network) ─────────────────────────────────────────

@pytest.mark.live
class TestLiveIntegration:
    """
    These tests hit the real mfapi.in API.
    Run with: pytest tests/test_navpy.py -m live
    """

    def test_search_mirae(self):
        from navpy.search import search
        results = search("mirae asset large cap", max_results=10)
        assert len(results) > 0
        codes = [r.scheme_code for r in results]
        assert "107578" in codes or any("mirae" in r.scheme_name.lower() for r in results)

    def test_get_by_code(self):
        import navpy
        result = navpy.get("107578", start="3y", plan="direct", interactive=False)
        assert not result.data.empty
        assert result.cagr() != float('nan')
        assert len(result) > 100

    def test_get_by_name(self):
        import navpy
        result = navpy.get(
            "Mirae Asset Large Cap Fund",
            start="2y",
            plan="direct",
            interactive=False,
        )
        assert not result.data.empty

    def test_get_returns_daily(self):
        import navpy
        result = navpy.get("107578", start="1y", plan="direct", interactive=False)
        dr = result.returns("daily")
        assert len(dr) > 0
        assert 'return_pct' in dr.columns

    def test_get_returns_monthly(self):
        import navpy
        result = navpy.get("107578", start="2y", plan="direct", interactive=False)
        mr = result.returns("monthly")
        assert len(mr) >= 20

    def test_nav_on_specific_date(self):
        import navpy
        result = navpy.get("107578", start="5y", plan="direct", interactive=False)
        val = result.nav_on("2022-06-15")
        assert val > 0

    def test_summary_completeness(self):
        import navpy
        result = navpy.get("107578", start="3y", plan="direct", interactive=False)
        s = result.summary()
        assert s['records'] > 0
        assert s['cagr_pct'] != float('nan')

    def test_to_csv(self, tmp_path):
        import navpy
        result = navpy.get("107578", start="1y", plan="direct", interactive=False)
        path   = str(tmp_path / "test.csv")
        result.to_csv(path)
        loaded = pd.read_csv(path)
        assert len(loaded) == len(result)

    def test_splice_longer_than_direct(self):
        import navpy
        direct = navpy.get("107578", plan="direct",  interactive=False)
        spliced = navpy.get("107578", plan="splice",  interactive=False)
        # Splice should have more records (includes pre-2013 regular plan data)
        assert len(spliced) >= len(direct)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-m", "not live"])


# ── analytics.py tests ────────────────────────────────────────────────────────

class TestAnalytics:

    def _nav(self, n=600, seed=42):
        np.random.seed(seed)
        dates = pd.date_range("2018-01-01", periods=n, freq="B")
        vals  = 100 * np.cumprod(1 + np.random.normal(0.0004, 0.008, n))
        return pd.Series(vals, index=dates, name="fund")

    def _bm(self, n=600, seed=99):
        np.random.seed(seed)
        dates = pd.date_range("2018-01-01", periods=n, freq="B")
        vals  = 100 * np.cumprod(1 + np.random.normal(0.0003, 0.007, n))
        return pd.Series(vals, index=dates, name="bm")

    def test_rolling_returns_length(self):
        from navpy.analytics import rolling_returns
        nav = self._nav(400)
        rr  = rolling_returns(nav, window=252)
        assert rr.dropna().shape[0] == 400 - 252

    def test_rolling_returns_annualised(self):
        from navpy.analytics import rolling_returns
        nav = self._nav(400)
        rr  = rolling_returns(nav, window=252, annualise=True)
        # Annualised values should be in reasonable range (-1 to +5)
        assert rr.dropna().between(-1, 5).all()

    def test_rolling_alpha_output(self):
        from navpy.analytics import rolling_alpha
        nav = self._nav(400)
        bm  = self._bm(400)
        ra  = rolling_alpha(nav, bm, window=252)
        assert len(ra.dropna()) > 0
        assert ra.name == "rolling_alpha_252d"

    def test_rolling_beta_output(self):
        from navpy.analytics import rolling_beta
        nav = self._nav(400)
        bm  = self._bm(400)
        rb  = rolling_beta(nav, bm, window=252)
        assert len(rb.dropna()) > 0

    def test_rolling_sharpe_output(self):
        from navpy.analytics import rolling_sharpe
        nav = self._nav(400)
        rs  = rolling_sharpe(nav, window=252)
        assert len(rs.dropna()) > 0

    def test_rolling_sortino_output(self):
        from navpy.analytics import rolling_sortino
        nav = self._nav(400)
        rs  = rolling_sortino(nav, window=252)
        assert len(rs.dropna()) > 0

    def test_rolling_volatility_positive(self):
        from navpy.analytics import rolling_volatility
        nav = self._nav(300)
        rv  = rolling_volatility(nav, window=63)
        assert (rv.dropna() > 0).all()

    def test_rolling_drawdown_non_positive(self):
        from navpy.analytics import rolling_drawdown
        nav = self._nav(300)
        dd  = rolling_drawdown(nav)
        assert (dd <= 0).all()

    def test_alpha_beta_keys(self):
        from navpy.analytics import alpha_beta
        nav = self._nav(400)
        bm  = self._bm(400)
        ab  = alpha_beta(nav, bm)
        for key in ("alpha", "beta", "r_squared", "p_value"):
            assert key in ab

    def test_alpha_beta_r_squared_bounds(self):
        from navpy.analytics import alpha_beta
        nav = self._nav(400)
        bm  = self._bm(400)
        ab  = alpha_beta(nav, bm)
        assert 0 <= ab["r_squared"] <= 1

    def test_information_ratio_finite(self):
        from navpy.analytics import information_ratio
        nav = self._nav(400)
        bm  = self._bm(400)
        ir  = information_ratio(nav, bm)
        assert np.isfinite(ir)

    def test_updown_capture_keys(self):
        from navpy.analytics import updown_capture
        nav = self._nav(400)
        bm  = self._bm(400)
        ud  = updown_capture(nav, bm)
        assert "upside_capture" in ud
        assert "downside_capture" in ud
        assert "capture_ratio" in ud

    def test_calmar_positive_for_uptrend(self):
        from navpy.analytics import calmar_ratio
        # Fund with strong positive drift and some drawdowns → positive calmar
        np.random.seed(1)
        dates = pd.date_range("2018-01-01", periods=500, freq="B")
        nav   = pd.Series(
            100 * np.cumprod(1 + np.random.normal(0.0008, 0.006, 500)),
            index=dates,
        )
        c = calmar_ratio(nav)
        assert not np.isnan(c)
        assert c > 0

    def test_omega_ratio_above_one_for_good_fund(self):
        from navpy.analytics import omega_ratio
        # Fund with strong positive drift should have omega > 1
        np.random.seed(0)
        dates = pd.date_range("2018-01-01", periods=500, freq="B")
        nav   = pd.Series(100 * np.cumprod(1 + np.random.normal(0.001, 0.005, 500)),
                          index=dates)
        o = omega_ratio(nav)
        assert o > 1

    def test_pain_index_non_negative(self):
        from navpy.analytics import pain_index
        nav = self._nav(300)
        assert pain_index(nav) >= 0

    def test_ulcer_index_non_negative(self):
        from navpy.analytics import ulcer_index
        nav = self._nav(300)
        assert ulcer_index(nav) >= 0

    def test_var_cvar_keys(self):
        from navpy.analytics import var_cvar
        nav = self._nav(300)
        vc  = var_cvar(nav, confidence=0.95)
        assert "var_pct" in vc and "cvar_pct" in vc
        # CVaR >= VaR (expected shortfall is at least as bad as VaR)
        assert vc["cvar_pct"] >= vc["var_pct"] - 1e-6

    def test_drawdown_table_columns(self):
        from navpy.analytics import drawdown_table
        nav = self._nav(500)
        dt  = drawdown_table(nav, top_n=5)
        for col in ("peak_date", "trough_date", "drawdown_pct", "duration_days"):
            assert col in dt.columns

    def test_drawdown_table_sorted(self):
        from navpy.analytics import drawdown_table
        nav = self._nav(500)
        dt  = drawdown_table(nav, top_n=5)
        if len(dt) > 1:
            assert dt["drawdown_pct"].is_monotonic_increasing

    def test_period_returns_columns(self):
        from navpy.analytics import period_returns
        nav = self._nav(600)
        pr  = period_returns(nav)
        assert "period" in pr.columns
        assert "return_pct" in pr.columns

    def test_monthly_returns_table_shape(self):
        from navpy.analytics import monthly_returns_table
        nav = self._nav(600)
        mr  = monthly_returns_table(nav)
        assert "Annual" in mr.columns
        assert mr.index.name == "year"

    def test_regime_returns_regimes(self):
        from navpy.analytics import regime_returns
        nav = self._nav(800)
        bm  = self._bm(800)
        rr  = regime_returns(nav, bm)
        assert "regime" in rr.columns
        assert set(rr["regime"]).issubset({"Bull", "Correction", "Bear"})

    def test_full_analytics_all_keys(self):
        from navpy.analytics import full_analytics
        nav = self._nav(600)
        bm  = self._bm(600)
        fa  = full_analytics(nav, bm)
        for key in ("period_returns", "drawdown_table", "alpha_beta",
                    "regime_returns", "rolling_returns", "rolling_alpha",
                    "monthly_returns", "var_cvar"):
            assert key in fa, f"Missing key: {key}"

    def test_full_analytics_no_benchmark(self):
        from navpy.analytics import full_analytics
        nav = self._nav(400)
        fa  = full_analytics(nav, benchmark=None)
        assert "period_returns" in fa
        assert "alpha_beta" not in fa


# ── benchmarks.py tests ───────────────────────────────────────────────────────

class TestBenchmarks:

    def test_list_benchmarks_returns_dataframe(self):
        from navpy.benchmarks import list_benchmarks
        df = list_benchmarks()
        assert isinstance(df, pd.DataFrame)
        assert "alias" in df.columns
        assert "ticker" in df.columns
        assert len(df) > 10

    def test_list_benchmarks_category_filter(self):
        from navpy.benchmarks import list_benchmarks
        df = list_benchmarks(category="mid_cap")
        assert len(df) > 0
        assert (df["category"] == "mid_cap").all()

    def test_resolve_direct_alias(self):
        from navpy.benchmarks import resolve_benchmark
        meta = resolve_benchmark("nifty50")
        assert meta["name"] == "Nifty 50"
        assert meta["ticker"] == "^NSEI"

    def test_resolve_shorthand_alias(self):
        from navpy.benchmarks import resolve_benchmark
        meta = resolve_benchmark("nifty")
        assert meta["alias"] == "nifty50"

    def test_resolve_by_ticker(self):
        from navpy.benchmarks import resolve_benchmark
        meta = resolve_benchmark("^BSESN")
        assert meta["alias"] == "sensex"

    def test_resolve_by_partial_name(self):
        from navpy.benchmarks import resolve_benchmark
        meta = resolve_benchmark("Nifty Bank")
        assert meta["alias"] == "nifty_bank"

    def test_resolve_unknown_raises(self):
        from navpy.benchmarks import resolve_benchmark
        with pytest.raises(ValueError, match="Unknown benchmark"):
            resolve_benchmark("xyznotabenchmark12345")

    def test_all_benchmarks_have_required_keys(self):
        from navpy.benchmarks import BENCHMARKS
        for alias, meta in BENCHMARKS.items():
            for key in ("name", "ticker", "description", "asset_class", "category"):
                assert key in meta, f"Missing key '{key}' in benchmark '{alias}'"

    def test_all_aliases_in_alias_map_point_to_valid_keys(self):
        from navpy.benchmarks import _ALIAS_MAP, BENCHMARKS
        for alias, target in _ALIAS_MAP.items():
            assert target in BENCHMARKS, \
                f"Alias '{alias}' points to unknown key '{target}'"


# ── compare.py tests ──────────────────────────────────────────────────────────

class TestCompare:

    def _make_result(self, n=800, seed=42):
        from navpy.models import NAVResult
        np.random.seed(seed)
        dates = pd.date_range("2018-01-01", periods=n, freq="B")
        vals  = 100 * np.cumprod(1 + np.random.normal(0.0004, 0.008, n))
        df    = pd.DataFrame({"date": dates, "nav": vals})
        return NAVResult("999", "Test Fund", "direct",
                         str(dates[0].date()), str(dates[-1].date()), df)

    def _bm_series(self, n=800, seed=99):
        np.random.seed(seed)
        dates = pd.date_range("2018-01-01", periods=n, freq="B")
        vals  = 100 * np.cumprod(1 + np.random.normal(0.0003, 0.007, n))
        return pd.Series(vals, index=dates, name="Nifty 50")

    @patch("navpy.compare.get_benchmark")
    def test_compare_returns_comparison_result(self, mock_bm):
        from navpy.compare import compare, ComparisonResult
        mock_bm.return_value = self._bm_series()
        r   = self._make_result()
        cmp = compare(r, benchmarks=["nifty50"])
        assert isinstance(cmp, ComparisonResult)

    @patch("navpy.compare.get_benchmark")
    def test_metrics_table_has_fund_and_bm_rows(self, mock_bm):
        from navpy.compare import compare
        mock_bm.return_value = self._bm_series()
        r   = self._make_result()
        cmp = compare(r, benchmarks=["nifty50"])
        assert len(cmp.metrics_table) == 2

    @patch("navpy.compare.get_benchmark")
    def test_metrics_table_has_core_columns(self, mock_bm):
        from navpy.compare import compare
        mock_bm.return_value = self._bm_series()
        r   = self._make_result()
        cmp = compare(r, benchmarks=["nifty50"])
        for col in ("CAGR %", "Volatility %", "Sharpe", "Max DD %"):
            assert col in cmp.metrics_table.columns

    @patch("navpy.compare.get_benchmark")
    def test_benchmark_relative_cols_on_fund_row(self, mock_bm):
        from navpy.compare import compare
        mock_bm.return_value = self._bm_series()
        r   = self._make_result()
        cmp = compare(r, benchmarks=["nifty50"])
        fund_row = cmp.metrics_table.iloc[0]
        assert any("Alpha" in c for c in cmp.metrics_table.columns)
        assert any("Beta"  in c for c in cmp.metrics_table.columns)

    @patch("navpy.compare.get_benchmark")
    def test_multiple_benchmarks(self, mock_bm):
        from navpy.compare import compare
        mock_bm.side_effect = [self._bm_series(seed=99),
                                self._bm_series(seed=77)]
        r   = self._make_result()
        cmp = compare(r, benchmarks=["nifty50", "sensex"])
        # Fund + 2 benchmarks = 3 rows
        assert len(cmp.metrics_table) == 3

    @patch("navpy.compare.get_benchmark")
    def test_period_returns_has_fund_key(self, mock_bm):
        from navpy.compare import compare
        mock_bm.return_value = self._bm_series()
        r   = self._make_result()
        cmp = compare(r, benchmarks=["nifty50"])
        pr  = cmp.period_returns()
        assert isinstance(pr, dict)
        assert len(pr) > 0

    @patch("navpy.compare.get_benchmark")
    def test_rolling_returns_dataframe(self, mock_bm):
        from navpy.compare import compare
        mock_bm.return_value = self._bm_series()
        r   = self._make_result()
        cmp = compare(r, benchmarks=["nifty50"], rolling_windows=[252])
        rr  = cmp.rolling_returns(window=252)
        assert isinstance(rr, pd.DataFrame)
        assert len(rr.columns) >= 1

    @patch("navpy.compare.get_benchmark")
    def test_rolling_alpha_dataframe(self, mock_bm):
        from navpy.compare import compare
        mock_bm.return_value = self._bm_series()
        r   = self._make_result()
        cmp = compare(r, benchmarks=["nifty50"], rolling_windows=[252])
        ra  = cmp.rolling_alpha(window=252)
        assert isinstance(ra, pd.DataFrame)

    @patch("navpy.compare.get_benchmark")
    def test_to_dataframe(self, mock_bm):
        from navpy.compare import compare
        mock_bm.return_value = self._bm_series()
        r   = self._make_result()
        cmp = compare(r, benchmarks=["nifty50"])
        df  = cmp.to_dataframe()
        assert isinstance(df, pd.DataFrame)

    @patch("navpy.compare.get_benchmark")
    def test_to_csv(self, mock_bm, tmp_path):
        from navpy.compare import compare
        mock_bm.return_value = self._bm_series()
        r    = self._make_result()
        cmp  = compare(r, benchmarks=["nifty50"])
        path = str(tmp_path / "cmp.csv")
        cmp.to_csv(path)
        loaded = pd.read_csv(path)
        assert len(loaded) == 2

    @patch("navpy.compare.get_benchmark")
    def test_no_benchmarks_raises(self, mock_bm):
        from navpy.compare import compare
        mock_bm.side_effect = ValueError("cannot load")
        r = self._make_result()
        with pytest.raises(ValueError, match="No benchmarks could be loaded"):
            compare(r, benchmarks=["bad_bm"])


# ── NAVResult analytics method delegation tests ───────────────────────────────

class TestNAVResultAnalyticsMethods:
    """Ensure every analytics method on NAVResult delegates correctly."""

    def _result(self, n=600):
        from navpy.models import NAVResult
        np.random.seed(5)
        dates = pd.date_range("2018-01-01", periods=n, freq="B")
        vals  = 100 * np.cumprod(1 + np.random.normal(0.0004, 0.008, n))
        df    = pd.DataFrame({"date": dates, "nav": vals})
        return NAVResult("T", "T Fund", "direct",
                         str(dates[0].date()), str(dates[-1].date()), df)

    def _bm(self, n=600):
        np.random.seed(9)
        dates = pd.date_range("2018-01-01", periods=n, freq="B")
        vals  = 100 * np.cumprod(1 + np.random.normal(0.0003, 0.007, n))
        return pd.Series(vals, index=dates)

    def test_rolling_returns_method(self):
        r  = self._result()
        rr = r.rolling_returns(252)
        assert isinstance(rr, pd.Series)

    def test_rolling_sharpe_method(self):
        r  = self._result()
        rs = r.rolling_sharpe(252)
        assert isinstance(rs, pd.Series)

    def test_rolling_sortino_method(self):
        r  = self._result()
        rs = r.rolling_sortino(252)
        assert isinstance(rs, pd.Series)

    def test_rolling_volatility_method(self):
        r  = self._result()
        rv = r.rolling_volatility(63)
        assert isinstance(rv, pd.Series)

    def test_rolling_alpha_with_series(self):
        r  = self._result()
        bm = self._bm()
        ra = r.rolling_alpha(bm, 252)
        assert isinstance(ra, pd.Series)

    def test_rolling_beta_with_series(self):
        r  = self._result()
        bm = self._bm()
        rb = r.rolling_beta(bm, 252)
        assert isinstance(rb, pd.Series)

    def test_drawdown_table_method(self):
        r  = self._result()
        dt = r.drawdown_table(5)
        assert isinstance(dt, pd.DataFrame)

    def test_period_returns_method(self):
        r  = self._result()
        pr = r.period_returns()
        assert isinstance(pr, pd.DataFrame)

    def test_monthly_returns_table_method(self):
        r  = self._result()
        mr = r.monthly_returns_table()
        assert isinstance(mr, pd.DataFrame)

    def test_var_cvar_method(self):
        r  = self._result()
        vc = r.var_cvar()
        assert "var_pct" in vc

    def test_omega_ratio_method(self):
        r  = self._result()
        o  = r.omega_ratio()
        assert isinstance(o, float)

    def test_ulcer_index_method(self):
        r  = self._result()
        u  = r.ulcer_index()
        assert u >= 0

    def test_pain_index_method(self):
        r  = self._result()
        p  = r.pain_index()
        assert p >= 0

    def test_alpha_beta_method_with_series(self):
        r  = self._result()
        bm = self._bm()
        ab = r.alpha_beta(bm)
        assert "alpha" in ab and "beta" in ab

    def test_information_ratio_method(self):
        r  = self._result()
        bm = self._bm()
        ir = r.information_ratio(bm)
        assert np.isfinite(ir)

    def test_updown_capture_method(self):
        r  = self._result()
        bm = self._bm()
        ud = r.updown_capture(bm)
        assert "upside_capture" in ud

    def test_regime_returns_method(self):
        r  = self._result()
        bm = self._bm()
        rr = r.regime_returns(bm)
        assert isinstance(rr, pd.DataFrame)

    def test_full_analytics_method(self):
        r  = self._result()
        bm = self._bm()
        fa = r.full_analytics(benchmark=bm)
        assert "alpha_beta" in fa

    def test_resolve_bm_type_error(self):
        r = self._result()
        with pytest.raises(TypeError):
            r._resolve_bm(12345)


