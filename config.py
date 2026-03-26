"""
Central configuration for the StatArb system.
All parameters from Avellaneda & Lee (2010) with sensible defaults.
"""
from dataclasses import dataclass, field
from typing import List, Optional


# ── Sector ETFs from Paper (Table 3) ─────────────────────
SECTOR_ETFS: List[str] = [
    "XLE", "XLF", "XLI", "XLK", "XLP", "XLV", "XLY",
    "IYR", "IYT", "OIH", "SMH", "RTH", "RKH", "UTH",
]

MARKET_ETF: str = "SPY"

DEFAULT_TICKERS: List[str] = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META",
    "JPM", "BAC", "GS", "WFC", "C",
    "XOM", "CVX", "COP", "SLB", "EOG",
    "JNJ", "PFE", "UNH", "ABT", "MRK",
    "HD", "PG", "KO", "PEP", "WMT",
    "NEE", "DUK", "SO", "D", "AEP",
    "UNP", "UPS", "FDX", "CSX", "NSC",
    "NVDA", "INTC", "AVGO", "TXN", "QCOM",
]

# Mapping: sector name → ETF ticker (from paper Table 3)
SECTOR_TO_ETF_MAP = {
    "Technology": "XLK",
    "Financial Services": "XLF",
    "Financials": "XLF",
    "Energy": "XLE",
    "Healthcare": "XLV",
    "Consumer Cyclical": "XLY",
    "Consumer Defensive": "XLP",
    "Industrials": "XLI",
    "Real Estate": "IYR",
    "Utilities": "UTH",
    "Basic Materials": "XLI",
    "Communication Services": "XLK",
    "Consumer Discretionary": "XLY",
    "Consumer Staples": "XLP",
    "Information Technology": "XLK",
    "Health Care": "XLV",
    "Materials": "XLI",
}

# Available data sources
DATA_SOURCES = ["yfinance", "crsp"]


@dataclass
class FactorConfig:
    """Factor model configuration."""
    model_type: str = "pca"                     # "pca" | "etf" | "combined"
    pca_lookback: int = 252                     # correlation matrix window (days)
    pca_n_components: Optional[int] = 15        # None → use explained_variance_threshold
    explained_variance_threshold: float = 0.55
    use_ledoit_wolf: bool = True
    beta_rolling_window: int = 252              # for ETF/combined rolling beta


@dataclass
class OUConfig:
    """Ornstein-Uhlenbeck estimation configuration."""
    estimation_window: int = 60                 # days for AR(1) fit
    kappa_min: float = 8.4                      # annualized mean-reversion speed filter
    mean_center: bool = True                    # cross-sectional mean centering (Eq. 18)


@dataclass
class SignalConfig:
    """Trading signal thresholds (Paper Section 4)."""
    s_bo: float = 1.25                          # buy-to-open: long if s < -s_bo
    s_so: float = 1.25                          # sell-to-open: short if s > s_so
    s_sc: float = 0.50                          # sell-to-close: exit long if s > -s_sc
    s_bc: float = 0.75                          # buy-to-close: exit short if s < s_bc
    s_limit: float = 4.0                        # forced exit threshold


@dataclass
class VolumeConfig:
    """Volume-adjusted trading time (Paper Section 6)."""
    enabled: bool = False
    trailing_window: int = 10                   # days for avg volume


@dataclass
class BacktestConfig:
    """Backtesting parameters (Paper Section 5)."""
    initial_equity: float = 1_000_000.0
    leverage_long: float = 2.0                  # 2x long
    leverage_short: float = 2.0                 # 2x short
    tc_bps: float = 5.0                         # transaction cost per side (basis points)
    hedge_instrument: str = "SPY"               # "SPY" | "sector_etf" | "none"
    risk_free_rate: float = 0.02                # for Sharpe ratio calculation
    dt: float = 1.0 / 252.0                     # annualization factor


@dataclass
class Config:
    """Master configuration combining all sub-configs."""
    factor: FactorConfig = field(default_factory=FactorConfig)
    ou: OUConfig = field(default_factory=OUConfig)
    signal: SignalConfig = field(default_factory=SignalConfig)
    volume: VolumeConfig = field(default_factory=VolumeConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    data_source: str = "yfinance"
    start_date: str = "2015-01-01"
    end_date: str = "2023-12-31"
    tickers: List[str] = field(default_factory=lambda: DEFAULT_TICKERS.copy())
