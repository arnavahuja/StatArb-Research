"""Shared test fixtures: synthetic price/volume data."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_tickers():
    return ["AAPL", "MSFT", "GOOGL", "JPM", "XOM",
            "JNJ", "HD", "NEE", "UNP", "NVDA"]


@pytest.fixture
def sample_dates():
    return pd.bdate_range("2020-01-01", periods=500)


@pytest.fixture
def sample_prices(sample_tickers, sample_dates):
    """Generate synthetic price data with realistic properties."""
    np.random.seed(42)
    n_dates = len(sample_dates)
    n_tickers = len(sample_tickers)

    # Geometric Brownian Motion
    mu = 0.0001  # daily drift
    sigma = 0.02  # daily vol
    returns = np.random.normal(mu, sigma, (n_dates, n_tickers))

    # Add some correlation structure
    corr_factor = np.random.normal(0, 0.01, n_dates)
    returns += corr_factor[:, np.newaxis] * 0.5

    prices = 100 * np.exp(np.cumsum(returns, axis=0))

    return pd.DataFrame(prices, index=sample_dates, columns=sample_tickers)


@pytest.fixture
def sample_volume(sample_tickers, sample_dates):
    """Generate synthetic volume data."""
    np.random.seed(123)
    n_dates = len(sample_dates)
    n_tickers = len(sample_tickers)

    volume = np.random.lognormal(mean=15, sigma=0.5, size=(n_dates, n_tickers))
    return pd.DataFrame(volume.astype(int), index=sample_dates, columns=sample_tickers)


@pytest.fixture
def sample_returns(sample_prices):
    """Log returns from synthetic prices."""
    return np.log(sample_prices / sample_prices.shift(1)).dropna()


@pytest.fixture
def sample_sector_mapping(sample_tickers):
    """Simple sector mapping for tests."""
    mapping = {
        "AAPL": "XLK", "MSFT": "XLK", "GOOGL": "XLK",
        "JPM": "XLF", "XOM": "XLE", "JNJ": "XLV",
        "HD": "XLY", "NEE": "UTH", "UNP": "XLI", "NVDA": "SMH",
    }
    return {t: mapping.get(t, "XLK") for t in sample_tickers}


@pytest.fixture
def sample_etf_returns(sample_dates):
    """Synthetic ETF returns."""
    np.random.seed(99)
    etfs = ["XLK", "XLF", "XLE", "XLV", "XLY", "UTH", "XLI", "SMH"]
    n_dates = len(sample_dates) - 1  # match returns (one less than prices)
    returns = np.random.normal(0.0001, 0.015, (n_dates, len(etfs)))
    return pd.DataFrame(returns, index=sample_dates[1:], columns=etfs)
