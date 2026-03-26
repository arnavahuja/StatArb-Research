# StatArb-Research

Reproduction of the **Avellaneda & Lee (2010)** statistical arbitrage framework for market-neutral equity trading using PCA-based factor decomposition and mean-reverting residual models.

**Course:** IEOR 4733 - Algorithmic Trading, Columbia University

**Team:** Arnav Ahuja, Sahethi DG, Aryamaan Srivastava, Aaditya Pai

## Overview

The core idea is to decompose stock returns into systematic factors and idiosyncratic residuals, model the residuals as mean-reverting Ornstein-Uhlenbeck processes, and trade stocks whose residuals deviate significantly from equilibrium.

### Factor Models
- **PCA (Eigenportfolios):** Extract risk factors from the correlation matrix of returns (Section 2.1)
- **Sector ETF Regression:** Regress each stock on its sector ETF to isolate idiosyncratic returns (Section 2.3)
- **Combined (SPY + ETF + PCA):** 3-stage decomposition removing market, sector, then PCA factors

### Signal Generation
- Estimate residuals as an Ornstein-Uhlenbeck process via AR(1) regression (Appendix A)
- Compute s-scores: `s = (X(t) - m_centered) / sigma_eq`
- Entry at `|s| > 1.25`, exit long at `s > -0.50`, exit short at `s < 0.75`
- Optional volume-adjusted trading time (Section 6)

### Backtesting
- Day-by-day rolling re-estimation with 60-day OU window and 252-day correlation window
- Bang-bang position sizing with 2+2 leverage (2x long, 2x short)
- 5 bps per-side transaction costs
- Beta-neutral hedging via SPY or sector ETFs

## Project Structure

```
StatArb-Research/
├── config.py                        # All configuration (Python dataclasses)
├── requirements.txt
├── .env.example                     # WRDS/CRSP credentials template
│
├── statarb/                         # Core backend
│   ├── data/                        # Data layer (yfinance, CRSP, extensible)
│   │   ├── base.py                  # Abstract DataSource interface
│   │   ├── yfinance_source.py
│   │   ├── crsp_source.py
│   │   └── universe.py              # Sector mapping & data source factory
│   ├── factors/                     # Factor models
│   │   ├── base.py                  # FactorModel ABC & FactorResult
│   │   ├── pca.py                   # PCA eigenportfolio model
│   │   ├── etf.py                   # Sector ETF regression model
│   │   ├── combined.py              # SPY + ETF + PCA 3-stage model
│   │   └── registry.py              # Factory to build model from config
│   ├── signals/                     # Signal generation
│   │   ├── ou_estimator.py          # AR(1) → OU parameter estimation
│   │   ├── sscore.py                # S-score computation & mean centering
│   │   ├── volume_time.py           # Volume-adjusted trading time
│   │   └── filters.py              # Kappa eligibility filter
│   ├── backtest/                    # Backtesting engine
│   │   ├── engine.py                # Day-by-day backtest orchestrator
│   │   ├── portfolio.py             # Position tracking & leverage
│   │   ├── costs.py                 # Transaction cost model
│   │   └── metrics.py              # Sharpe, drawdown, win rate, etc.
│   └── extensions/                  # Future enhancements (stubs)
│       ├── hmm_regime.py            # HMM regime detection
│       ├── almgren_chriss.py        # Optimal execution
│       └── vol_targeting.py         # Volatility-targeted sizing
│
├── app/                             # Streamlit frontend
│   ├── Home.py                      # Main dashboard
│   ├── pages/
│   │   ├── 1_Factor_Diagnostics.py  # Correlation, eigenvalues, loadings
│   │   └── 2_Trade_Analytics.py     # Trade KPIs, PnL, sector breakdown
│   ├── components/
│   │   ├── sidebar.py               # Configuration sidebar builder
│   │   ├── kpi_cards.py             # Reusable KPI card row
│   │   └── charts.py               # Reusable Plotly chart functions
│   └── state.py                     # Session state helpers
│
└── tests/                           # Unit tests (28 tests)
    ├── conftest.py                  # Shared synthetic data fixtures
    ├── test_data.py
    ├── test_factors.py
    ├── test_signals.py
    └── test_backtest.py
```

## Setup

```bash
# Activate the conda environment
conda activate statarb

# Install dependencies
pip install -r requirements.txt

# (Optional) For CRSP data, copy and fill in credentials
cp .env.example .env
```

## Usage

### Run the Dashboard

```bash
streamlit run app/Home.py
```

The dashboard has three pages:
1. **Home** - Configure parameters, run backtest, view equity curve, s-scores, per-ticker drill-down
2. **Factor Diagnostics** - Correlation heatmap, eigenvalue spectrum, PCA loadings, factor returns
3. **Trade Analytics** - Trade-level KPIs, PnL distributions, long/short breakdown, sector Sharpes

### Run Tests

```bash
python -m pytest tests/ -v
```

## Configuration

All parameters are in `config.py` as Python dataclasses. Key defaults from the paper:

| Parameter | Default | Paper Reference |
|-----------|---------|-----------------|
| PCA correlation window | 252 days | Section 2.1 |
| OU estimation window | 60 days | Section 3 |
| Entry threshold (s_bo) | 1.25 | Section 4 |
| Exit long (s_sc) | 0.50 | Section 4 |
| Exit short (s_bc) | 0.75 | Section 4 |
| Kappa filter | > 8.4 | Section 4 |
| Transaction cost | 5 bps/side | Section 5 |
| Leverage | 2+2 | Section 5 |
| Volume trailing window | 10 days | Section 6 |

All parameters are configurable from the Streamlit sidebar at runtime.

## Data Sources

| Source | Credentials | Notes |
|--------|-------------|-------|
| **yfinance** | None required | Default. Free Yahoo Finance data |
| **CRSP** | WRDS account (`.env`) | Academic database via WRDS |

To add a new data source, subclass `statarb.data.base.DataSource` and register it in `statarb/data/universe.py`.

## Module Ownership

Each backend module is designed for independent development:

| Module | Responsibility |
|--------|---------------|
| `statarb/data/` | Data fetching, caching, new source integrations |
| `statarb/factors/` | Factor model implementations, new decomposition methods |
| `statarb/signals/` | OU estimation, s-scores, signal enhancements |
| `statarb/backtest/` | Backtest engine, portfolio management, cost models |

## Planned Extensions

These are stubbed out in `statarb/extensions/` with planned interfaces:

- **HMM Regime Detection** - Gate entries based on market regime (Hamilton 1989)
- **Almgren-Chriss Execution** - Replace bang-bang with optimal trade scheduling
- **Volatility-Targeted Sizing** - Risk-parity position sizing instead of equal-notional

## Reference

Avellaneda, M., & Lee, J.-H. (2010). Statistical arbitrage in the US equities market. *Quantitative Finance*, 10(7), 761-782.
