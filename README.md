# navpy

Fetch, analyse, and benchmark Indian mutual fund NAV data from AMFI.

## Install

```bash
pip install navpy
```

Or from source:

```bash
git clone https://github.com/Nityahapani/navpy
cd navpy
pip install -e .
```

---

## Python API

```python
import navpy

# By fund name — last 3 years, Plan A splice (default)
result = navpy.get("Mirae Asset Large Cap", start="3y")
result.print_summary()

# By AMFI scheme code — specific date range
result = navpy.get("107578", start="2018-01-01", end="2023-12-31")

# Direct plan only
result = navpy.get("HDFC Flexi Cap", plan="direct", start="5y")

# Regular plan only
result = navpy.get("DSP Flexi Cap", plan="regular", start="2016-01-01")

# Access the raw data
df = result.data           # pandas DataFrame: date, nav
print(df.head())
```

### Basic analytics

```python
result.cagr()                    # CAGR % over full period
result.cagr("2020-01-01")        # CAGR from a specific start
result.abs_return()              # total point-to-point return %
result.max_drawdown()            # peak-to-trough drawdown %
result.volatility()              # annualised daily vol %
result.nav_on("2022-06-15")      # NAV on a specific date

result.returns("daily")          # day-on-day returns DataFrame
result.returns("monthly")        # month-on-month returns
result.returns("yearly")         # year-on-year returns

result.summary()                 # dict of all key stats
result.print_summary()           # formatted terminal output
```

### Export

```python
result.to_csv("output.csv")      # save to CSV
result.to_json("output.json")    # save to JSON
result.to_json()                 # get JSON string
result.to_series()               # date-indexed pandas Series
```

### Search

```python
results = navpy.search("mirae asset")
for r in results:
    print(r.scheme_code, r.scheme_name)
```

### Date shorthands

| Input            | Meaning                      |
|------------------|------------------------------|
| `"1y"`           | 1 year ago from today        |
| `"3y"`           | 3 years ago                  |
| `"5y"`           | 5 years ago                  |
| `"6m"`           | 6 months ago                 |
| `"ytd"`          | Jan 1 of current year        |
| `"max"` / `None` | Full available history       |
| `"YYYY-MM-DD"`   | Literal date                 |

### Plan options

| Plan        | Behaviour                                                     |
|-------------|---------------------------------------------------------------|
| `"splice"`  | Regular plan pre-Jan 2013, Direct plan from Jan 2013 onward   |
| `"direct"`  | Direct plan only                                              |
| `"regular"` | Regular plan only                                             |

---

## Fund Manager Career Pipeline

Use navpy to reconstruct a fund manager's full NAV history across multiple funds
and tenures, export a unified CSV, and run comparative analytics.

```python
import navpy
import pandas as pd

# 1. Initialize result list
career_results = []

# 2. Map Fund Manager's schemes
res1 = navpy.get("EXACT_SCHEME_NAME", plan="direct", start="YYYY-MM-DD", end="YYYY-MM-DD")
career_results.append(res1)
# ... repeat for all schemes ...

# 3. Print summaries
print("="*60)
for res in career_results:
    res.print_summary()
print("="*60)

# 4. Export to unified CSV
all_series = []
for i, res in enumerate(career_results):
    s = res.to_series()
    s.name = f"{res.scheme_name}_{i}"
    all_series.append(s)

final_df = pd.concat(all_series, axis=1)
final_df.to_csv("manager_career_navs.csv")
print("Successfully saved unified NAV data to 'manager_career_navs.csv'")
```

### Real example — Taher Badshah career reconstruction

```python
import navpy
import pandas as pd

career = []

# Motilal Oswal Flexi Cap (Multicap 35) — Apr 2014 to Dec 2016
career.append(navpy.get("129046", plan="direct",
                         start="2014-04-28", end="2016-12-09",
                         interactive=False))

# Invesco India Contra Fund — Jan 2017 to present
career.append(navpy.get("120348", plan="direct",
                         start="2017-01-13",
                         interactive=False))

# Invesco India Smallcap Fund — Oct 2018 to present
career.append(navpy.get("145137", plan="direct",
                         start="2018-10-30",
                         interactive=False))

print("="*60)
for res in career:
    res.print_summary()
print("="*60)

# Export unified CSV
all_series = [res.to_series().rename(res.scheme_name) for res in career]
pd.concat(all_series, axis=1).to_csv("taher_badshah_career.csv")
print("Saved to taher_badshah_career.csv")

# Run rolling analytics on each stint
for res in career:
    print(f"\n{res.scheme_name}")
    print(f"  CAGR        : {res.cagr():.2f}%")
    print(f"  Sharpe      : {res.rolling_sharpe(252).mean():.3f}")
    print(f"  Max DD      : {res.max_drawdown():.2f}%")
    pr = res.period_returns()
    print(f"  Period rets :")
    print(pr[["period", "return_pct"]].to_string(index=False))
```

---

## Benchmark Comparison

Compare any fund against one or more Indian market benchmarks.

```python
import navpy

result = navpy.get("Mirae Asset Large Cap", start="5y", plan="direct",
                   interactive=False)

# Single benchmark
cmp = navpy.compare(result, "nifty50")
cmp.print_summary()

# Multiple benchmarks
cmp = navpy.compare(result, ["nifty50", "nifty_midcap150", "sensex"])

# Get metrics table as a DataFrame
df = cmp.to_dataframe()
print(df)

# Period returns vs all benchmarks
for name, df in cmp.period_returns().items():
    print(f"\n{name}:")
    print(df[["period", "return_pct", "annualised_pct"]])

# Regime analysis (Bull / Correction / Bear)
for bm_name, df in cmp.regime_analysis().items():
    print(f"\nRegime vs {bm_name}:")
    print(df[["regime", "fund_cagr", "bm_cagr", "active_return", "hit_rate_pct"]])

# Rolling returns for fund + all benchmarks (1-year window)
roll_df = cmp.rolling_returns(window=252)
roll_df.plot(title="1Y Rolling Returns")

# Rolling alpha vs all benchmarks
alpha_df = cmp.rolling_alpha(window=252)
alpha_df.plot(title="1Y Rolling Alpha")

# Save to CSV
cmp.to_csv("mirae_vs_benchmarks.csv")
```

### Available benchmarks

```python
# See full list
navpy.list_benchmarks()

# Filter by category
navpy.list_benchmarks(category="mid_cap")
navpy.list_benchmarks(category="large_cap")
navpy.list_benchmarks(category="small_cap")
navpy.list_benchmarks(category="sectoral")
navpy.list_benchmarks(category="international")
```

| Alias                    | Name                        | Category      |
|--------------------------|-----------------------------|---------------|
| `nifty50`                | Nifty 50                    | broad_market  |
| `sensex`                 | BSE Sensex                  | broad_market  |
| `nifty500`               | Nifty 500                   | broad_market  |
| `nifty100`               | Nifty 100                   | large_cap     |
| `nifty_midcap50`         | Nifty Midcap 50             | mid_cap       |
| `nifty_midcap150`        | Nifty Midcap 150            | mid_cap       |
| `nifty_smallcap100`      | Nifty Smallcap 100          | small_cap     |
| `nifty_smallcap250`      | Nifty Smallcap 250          | small_cap     |
| `nifty_largemidcap250`   | Nifty LargeMidcap 250       | large_mid_cap |
| `nifty_multicap50_25_25` | Nifty Multicap 50:25:25     | multi_cap     |
| `nifty_bank`             | Nifty Bank                  | sectoral      |
| `nifty_it`               | Nifty IT                    | sectoral      |
| `nifty_fmcg`             | Nifty FMCG                  | sectoral      |
| `nifty_pharma`           | Nifty Pharma                | sectoral      |
| `sp500`                  | S&P 500                     | international |
| `msci_em`                | MSCI Emerging Markets       | international |

You can also pass any Yahoo Finance ticker directly:

```python
cmp = navpy.compare(result, ["^NSEI", "^NSEBANK"])
```

You can also fetch a benchmark series directly:

```python
bm = navpy.get_benchmark("nifty50", start="5y")   # returns pd.Series
```

---

## Advanced Analytics

All analytics are available as methods on `NAVResult`, or via `navpy.analytics`.

### Rolling metrics

```python
result = navpy.get("107578", start="5y", plan="direct", interactive=False)

# Rolling 1-year returns (annualised CAGR)
roll_ret  = result.rolling_returns(window=252)

# Rolling 6-month returns
roll_6m   = result.rolling_returns(window=126)

# Rolling Sharpe ratio (1-year window)
roll_sh   = result.rolling_sharpe(window=252)

# Rolling Sortino ratio
roll_so   = result.rolling_sortino(window=252)

# Rolling volatility (3-month window)
roll_vol  = result.rolling_volatility(window=63)

# Rolling alpha vs Nifty 50 (accepts alias string or pd.Series)
roll_alpha = result.rolling_alpha("nifty50", window=252)

# Rolling beta vs Nifty 50
roll_beta  = result.rolling_beta("nifty50", window=252)
```

### Drawdown analysis

```python
# Table of worst drawdown episodes with recovery info
dd_table = result.drawdown_table(top_n=10)
print(dd_table)
# Columns: peak_date, trough_date, recovery_date,
#          drawdown_pct, duration_days, recovery_days

# Ulcer index (penalises deep + prolonged drawdowns)
ui = result.ulcer_index()

# Pain index (mean of all drawdown values)
pi = result.pain_index()
```

### Risk metrics

```python
# Value at Risk and Conditional VaR (95% confidence)
vc = result.var_cvar(confidence=0.95)
print(f"Daily VaR:  {vc['var_pct']:.2f}%")
print(f"Daily CVaR: {vc['cvar_pct']:.2f}%")

# Omega ratio (probability-weighted gains / losses, > 1 is good)
omega = result.omega_ratio()

# Calmar ratio (CAGR / max drawdown)
from navpy.analytics import calmar_ratio
calmar = calmar_ratio(result.to_series())
```

### Benchmark-relative metrics

```python
# Full-period Jensen's alpha and beta
ab = result.alpha_beta("nifty50")
print(f"Alpha: {ab['alpha']*100:.2f}%")
print(f"Beta:  {ab['beta']:.3f}")
print(f"R²:    {ab['r_squared']:.3f}")

# Information ratio
ir = result.information_ratio("nifty50")

# Upside / downside capture ratios
ud = result.updown_capture("nifty50")
print(f"Upside capture:   {ud['upside_capture']:.1f}%")
print(f"Downside capture: {ud['downside_capture']:.1f}%")
print(f"Capture ratio:    {ud['capture_ratio']:.2f}")
```

### Regime analysis

```python
# Returns breakdown by Bull / Correction / Bear market
# (Bull = Nifty within 10% of ATH, Correction = -10% to -20%, Bear = below -20%)
regime = result.regime_returns("nifty50")
print(regime)
# Columns: regime, fund_cagr, bm_cagr, active_return, days, hit_rate_pct
```

### Period and monthly returns

```python
# Standard period returns table
pr = result.period_returns()
print(pr)
# Periods: 1M, 3M, 6M, 1Y, 2Y, 3Y, 5Y, 10Y, Since Inception

# Monthly returns heatmap (rows=year, cols=month+Annual)
mr = result.monthly_returns_table()
print(mr)
```

### All analytics at once

```python
# Get everything in one call
full = result.full_analytics(benchmark="nifty50")

# Access any sub-result
print(full["period_returns"])
print(full["drawdown_table"])
print(full["alpha_beta"])
print(full["regime_returns"])
print(full["rolling_returns"][252])   # 1Y rolling returns Series
print(full["rolling_alpha"][252])     # 1Y rolling alpha Series
print(full["monthly_returns"])        # monthly heatmap DataFrame
print(full["var_cvar"])
print(full["omega_ratio"])
print(full["ulcer_index"])
```

### Using the analytics module directly

```python
import navpy
from navpy import analytics

nav = navpy.get("107578", start="5y", plan="direct", interactive=False).to_series()
bm  = navpy.get_benchmark("nifty50", start="5y")

# All functions accept plain pd.Series
ra   = analytics.rolling_alpha(nav, bm, window=252)
rb   = analytics.rolling_beta(nav, bm, window=252)
dd   = analytics.drawdown_table(nav, top_n=5)
rr   = analytics.regime_returns(nav, bm)
mr   = analytics.monthly_returns_table(nav)
ab   = analytics.alpha_beta(nav, bm)
vc   = analytics.var_cvar(nav, confidence=0.95)
fa   = analytics.full_analytics(nav, benchmark=bm)
```

---

## Cache management

NAV data is cached locally at `~/.navpy/cache/` with a 24-hour TTL.
Repeat calls are instant. Force a fresh fetch with:

```python
navpy.get("107578", force_refresh=True)

# Programmatic cache management
navpy.cache.cache_info()           # show cache stats
navpy.cache.invalidate("107578")   # delete one entry
navpy.cache.clear_all()            # wipe entire cache
```

---

## CLI

```bash
# Basic fetch — last 3 years
navpy "Mirae Asset Large Cap" --start 3y

# By AMFI code
navpy "107578" --start 2018-01-01 --end 2023-12-31

# Direct plan, save to CSV
navpy "DSP Flexi Cap" --plan direct --output nav.csv

# Save as JSON
navpy "HDFC Flexi Cap" --start 5y --output hdfc.json

# Force refresh (bypass cache)
navpy "107578" --start 5y --force-refresh

# Search
navpy search "mirae asset"
navpy search "hdfc" --limit 30

# Cache management
navpy cache info
navpy cache clear
navpy cache invalidate 107578
```

---

## Running tests

```bash
pip install pytest
# Unit tests only (no network required) — runs in ~15 seconds
pytest tests/ -m "not live" -v

# All tests including live API calls (~2 minutes)
pytest tests/ -m live -v
```

---

## GitHub Actions / CI

Three workflows are included:

| Workflow         | Trigger                  | What it does                             |
|------------------|--------------------------|------------------------------------------|
| `ci.yml`         | Every push + PR to main  | Tests (Py 3.11–3.12), lint, build check   |
| `publish.yml`    | Tag push (`v*.*.*`)      | Test → Build → TestPyPI → PyPI → Release |
| `live_tests.yml` | Nightly 2 AM UTC         | Live API integration tests               |

### Publishing a release

```bash
# 1. Bump version in pyproject.toml
# 2. Commit and tag
git tag v1.0.1
git push origin v1.0.1
# The publish workflow handles the rest automatically:
#   runs tests → builds wheel+sdist → publishes TestPyPI →
#   smoke tests the install → publishes PyPI → creates GitHub Release
```

Requires two GitHub Environment secrets with PyPI Trusted Publisher (OIDC) configured:
- `pypi` environment → linked to pypi.org
- `testpypi` environment → linked to test.pypi.org

No API token storage needed — OIDC handles authentication automatically.

---

## Data sources

- **NAV data** — [api.mfapi.in](https://api.mfapi.in) aggregates AMFI's public NAV data. No API key required.
- **Benchmark data** — [Yahoo Finance](https://finance.yahoo.com) via [yfinance](https://github.com/ranaroussi/yfinance). No API key required.
