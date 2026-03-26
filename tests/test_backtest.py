"""Tests for the backtest engine."""
import numpy as np
import pandas as pd
import pytest

from statarb.backtest.portfolio import PortfolioManager, Position
from statarb.backtest.costs import compute_transaction_cost
from statarb.backtest.metrics import compute_metrics, compute_drawdown_series


class TestTransactionCost:
    def test_basic_cost(self):
        cost = compute_transaction_cost(10_000, tc_bps=5.0)
        assert cost == pytest.approx(5.0)

    def test_zero_cost(self):
        assert compute_transaction_cost(10_000, tc_bps=0.0) == 0.0

    def test_negative_notional(self):
        cost = compute_transaction_cost(-10_000, tc_bps=5.0)
        assert cost == pytest.approx(5.0)


class TestPortfolioManager:
    def test_initial_state(self):
        pm = PortfolioManager(initial_equity=1_000_000)
        assert pm.equity == 1_000_000
        assert pm.gross_exposure == 0
        assert len(pm.positions) == 0

    def test_open_and_close_position(self):
        pm = PortfolioManager(initial_equity=1_000_000, tc_bps=0)
        date = pd.Timestamp("2020-01-01")

        # Open long
        pos = pm.open_position("AAPL", 1, 100.0, date, 10_000)
        assert pos is not None
        assert pos.quantity == 100.0
        assert "AAPL" in pm.positions

        # Close at higher price
        pnl = pm.close_position("AAPL", 110.0, date)
        assert pnl == pytest.approx(1000.0)  # 100 shares * $10 gain
        assert "AAPL" not in pm.positions

    def test_leverage_limit(self):
        pm = PortfolioManager(
            initial_equity=100_000, leverage_long=2.0, tc_bps=0
        )
        date = pd.Timestamp("2020-01-01")

        # Should be able to open up to 200K long
        pm.open_position("A", 1, 100.0, date, 100_000)
        pm.open_position("B", 1, 100.0, date, 100_000)

        # Third should fail (exceeds 2x leverage)
        pos = pm.open_position("C", 1, 100.0, date, 100_000)
        assert pos is None

    def test_no_duplicate_positions(self):
        pm = PortfolioManager(initial_equity=1_000_000, tc_bps=0)
        date = pd.Timestamp("2020-01-01")
        pm.open_position("AAPL", 1, 100.0, date, 10_000)
        dup = pm.open_position("AAPL", -1, 100.0, date, 10_000)
        assert dup is None


class TestMetrics:
    def test_basic_metrics(self):
        dates = pd.bdate_range("2020-01-01", periods=252)
        equity = pd.Series(
            1_000_000 * np.exp(np.cumsum(np.random.normal(0.0004, 0.01, 252))),
            index=dates,
        )
        trades = pd.DataFrame({
            "ticker": ["AAPL"] * 10,
            "direction": [1, -1] * 5,
            "entry_date": dates[:10],
            "exit_date": dates[10:20],
            "pnl": np.random.normal(100, 500, 10),
            "notional": [10000] * 10,
            "entry_price": [100] * 10,
            "exit_price": [101] * 10,
        })
        metrics = compute_metrics(equity, trades)

        assert metrics.num_trades == 10
        assert -1 <= metrics.max_drawdown <= 0
        assert 0 <= metrics.win_rate <= 1
        assert metrics.annualized_vol > 0

    def test_drawdown_series(self):
        equity = pd.Series([100, 110, 105, 120, 90, 130])
        dd = compute_drawdown_series(equity)
        assert dd.iloc[0] == 0.0
        assert dd.iloc[2] < 0  # 105 < 110 peak
        assert dd.iloc[4] < dd.iloc[2]  # 90 < 120 peak is worse
