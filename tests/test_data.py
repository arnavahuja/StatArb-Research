"""Tests for the data layer."""
import pandas as pd
import numpy as np
from statarb.data.base import DataSource
from statarb.data.universe import get_data_source


class MockDataSource(DataSource):
    """Concrete implementation for testing."""
    def __init__(self, prices, volume):
        self._prices = prices
        self._volume = volume

    def fetch_prices(self, tickers, start, end):
        return self._prices[tickers]

    def fetch_volume(self, tickers, start, end):
        return self._volume[tickers]


def test_datasource_returns(sample_prices, sample_volume, sample_tickers):
    """Test that fetch_returns computes log returns correctly."""
    ds = MockDataSource(sample_prices, sample_volume)
    returns = ds.fetch_returns(sample_tickers, "2020-01-01", "2021-12-31")

    assert isinstance(returns, pd.DataFrame)
    assert len(returns) == len(sample_prices) - 1
    assert list(returns.columns) == sample_tickers

    # Verify log return formula
    expected = np.log(sample_prices / sample_prices.shift(1)).dropna(how="all")
    pd.testing.assert_frame_equal(returns, expected)


def test_get_data_source_yfinance():
    """Test factory creates YFinanceSource."""
    ds = get_data_source("yfinance")
    from statarb.data.yfinance_source import YFinanceSource
    assert isinstance(ds, YFinanceSource)


def test_get_data_source_invalid():
    """Test factory raises on unknown source."""
    import pytest
    with pytest.raises(ValueError, match="Unknown data source"):
        get_data_source("bloomberg")
