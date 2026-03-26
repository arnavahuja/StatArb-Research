"""
Performance metrics for backtesting evaluation.

Computes standard risk/return metrics including Sharpe ratio,
maximum drawdown, win rate, and per-sector breakdowns.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics for a backtest."""
    total_return: float
    annualized_return: float
    annualized_vol: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    max_drawdown_duration: int          # trading days
    win_rate: float                     # fraction of winning days
    profit_factor: float
    avg_holding_period: float           # trading days
    num_trades: int
    total_costs: float
    turnover: float                     # annualized


def compute_metrics(
    equity_curve: pd.Series,
    trades: pd.DataFrame,
    risk_free_rate: float = 0.02,
    total_costs: float = 0.0,
) -> PerformanceMetrics:
    """
    Compute performance metrics from an equity curve and trade log.

    Args:
        equity_curve: Series of daily portfolio equity values.
        trades: DataFrame with columns: ticker, direction, entry_date,
            exit_date, entry_price, exit_price, pnl, notional.
        risk_free_rate: Annualized risk-free rate for Sharpe calculation.
        total_costs: Total transaction costs incurred.

    Returns:
        PerformanceMetrics dataclass.
    """
    if len(equity_curve) < 2:
        return PerformanceMetrics(
            total_return=0, annualized_return=0, annualized_vol=0,
            sharpe_ratio=0, sortino_ratio=0, max_drawdown=0,
            max_drawdown_duration=0, win_rate=0, profit_factor=0,
            avg_holding_period=0, num_trades=0, total_costs=total_costs,
            turnover=0,
        )

    # Daily returns
    daily_returns = equity_curve.pct_change().dropna()
    n_days = len(daily_returns)
    n_years = n_days / 252.0

    # Total and annualized return
    total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1.0
    annualized_return = (1 + total_return) ** (1.0 / max(n_years, 0.01)) - 1.0

    # Volatility
    annualized_vol = daily_returns.std() * np.sqrt(252)

    # Sharpe ratio
    daily_rf = risk_free_rate / 252.0
    excess_returns = daily_returns - daily_rf
    sharpe = (
        excess_returns.mean() / (excess_returns.std() + 1e-10) * np.sqrt(252)
    )

    # Sortino ratio
    downside = daily_returns[daily_returns < daily_rf] - daily_rf
    downside_std = np.sqrt((downside ** 2).mean()) if len(downside) > 0 else 1e-10
    sortino = excess_returns.mean() / (downside_std + 1e-10) * np.sqrt(252)

    # Drawdown
    dd_series = compute_drawdown_series(equity_curve)
    max_drawdown = dd_series.min()

    # Max drawdown duration
    is_underwater = dd_series < 0
    if is_underwater.any():
        underwater_groups = (~is_underwater).cumsum()
        underwater_groups[~is_underwater] = np.nan
        durations = underwater_groups.groupby(underwater_groups).transform("count")
        max_dd_duration = int(durations.max()) if not durations.isna().all() else 0
    else:
        max_dd_duration = 0

    # Win rate (daily)
    win_rate = (daily_returns > 0).mean()

    # Profit factor
    gross_profit = daily_returns[daily_returns > 0].sum()
    gross_loss = abs(daily_returns[daily_returns < 0].sum())
    profit_factor = gross_profit / (gross_loss + 1e-10)

    # Trade-level metrics
    num_trades = len(trades) if trades is not None and not trades.empty else 0
    avg_holding = 0.0
    if num_trades > 0 and "entry_date" in trades.columns and "exit_date" in trades.columns:
        holding_days = (
            pd.to_datetime(trades["exit_date"]) - pd.to_datetime(trades["entry_date"])
        ).dt.days
        avg_holding = holding_days.mean()

    # Turnover
    turnover = 0.0
    if num_trades > 0 and "notional" in trades.columns:
        total_traded = trades["notional"].sum() * 2  # round-trip
        avg_equity = equity_curve.mean()
        turnover = total_traded / (avg_equity * n_years + 1e-10)

    return PerformanceMetrics(
        total_return=total_return,
        annualized_return=annualized_return,
        annualized_vol=annualized_vol,
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
        max_drawdown=max_drawdown,
        max_drawdown_duration=max_dd_duration,
        win_rate=win_rate,
        profit_factor=profit_factor,
        avg_holding_period=avg_holding,
        num_trades=num_trades,
        total_costs=total_costs,
        turnover=turnover,
    )


def compute_drawdown_series(equity_curve: pd.Series) -> pd.Series:
    """
    Compute drawdown series: (equity - running_max) / running_max.

    Returns a Series of drawdowns (negative values).
    """
    running_max = equity_curve.cummax()
    drawdown = (equity_curve - running_max) / running_max
    return drawdown
