"""Reusable Plotly chart builders for the dashboard."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from config import SignalConfig


def plot_equity_curve(equity: pd.Series) -> go.Figure:
    """Plot the portfolio equity curve."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=equity.index, y=equity.values,
        mode="lines", name="Equity",
        line=dict(color="#1f77b4", width=2),
    ))
    fig.update_layout(
        title="Portfolio Equity Curve",
        xaxis_title="Date",
        yaxis_title="Equity ($)",
        hovermode="x unified",
        template="plotly_white",
    )
    return fig


def plot_drawdown(equity: pd.Series) -> go.Figure:
    """Plot the drawdown series."""
    running_max = equity.cummax()
    drawdown = (equity - running_max) / running_max

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=drawdown.index, y=drawdown.values,
        fill="tozeroy", name="Drawdown",
        line=dict(color="#d62728", width=1),
        fillcolor="rgba(214, 39, 40, 0.3)",
    ))
    fig.update_layout(
        title="Drawdown",
        xaxis_title="Date",
        yaxis_title="Drawdown",
        yaxis_tickformat=".1%",
        hovermode="x unified",
        template="plotly_white",
    )
    return fig


def plot_gross_exposure(daily_positions: pd.DataFrame, equity: pd.Series) -> go.Figure:
    """Plot gross exposure over time."""
    if daily_positions.empty:
        fig = go.Figure()
        fig.update_layout(title="Gross Exposure")
        return fig

    exposure = daily_positions.groupby("date")["notional"].sum()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=exposure.index, y=exposure.values,
        fill="tozeroy", name="Gross Exposure",
        line=dict(color="#2ca02c", width=1),
        fillcolor="rgba(44, 160, 44, 0.3)",
    ))
    fig.update_layout(
        title="Gross Exposure Over Time",
        xaxis_title="Date",
        yaxis_title="Exposure ($)",
        hovermode="x unified",
        template="plotly_white",
    )
    return fig


def plot_correlation_heatmap(corr: pd.DataFrame) -> go.Figure:
    """Plot a correlation heatmap."""
    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.columns.tolist(),
        y=corr.index.tolist(),
        colorscale="RdBu_r",
        zmid=0,
        zmin=-1,
        zmax=1,
    ))
    fig.update_layout(
        title="Return Correlation Matrix",
        template="plotly_white",
        height=600,
    )
    return fig


def plot_eigenvalue_spectrum(eigenvalues: np.ndarray) -> go.Figure:
    """Plot eigenvalue spectrum (scree plot) with cumulative variance."""
    total = eigenvalues.sum()
    pct = eigenvalues / total * 100
    cumulative = np.cumsum(pct)
    n = min(len(eigenvalues), 50)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=list(range(1, n + 1)),
        y=pct[:n],
        name="% Variance",
        marker_color="#1f77b4",
    ))
    fig.add_trace(go.Scatter(
        x=list(range(1, n + 1)),
        y=cumulative[:n],
        name="Cumulative %",
        yaxis="y2",
        line=dict(color="#d62728", width=2),
    ))
    fig.update_layout(
        title="Eigenvalue Spectrum",
        xaxis_title="Component",
        yaxis_title="% Variance Explained",
        yaxis2=dict(
            title="Cumulative %",
            overlaying="y",
            side="right",
            range=[0, 100],
        ),
        template="plotly_white",
        legend=dict(x=0.7, y=0.5),
    )
    return fig


def plot_sscore_timeseries(
    sscores: pd.Series, ticker: str, signal_cfg: SignalConfig
) -> go.Figure:
    """Plot s-score time series with entry/exit threshold bands."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=sscores.index, y=sscores.values,
        mode="lines", name="s-score",
        line=dict(color="#1f77b4", width=1.5),
    ))

    # Threshold bands
    for val, name, color, dash in [
        (signal_cfg.s_bo, "Entry Short", "#d62728", "dash"),
        (-signal_cfg.s_bo, "Entry Long", "#2ca02c", "dash"),
        (signal_cfg.s_bc, "Exit Short", "#d62728", "dot"),
        (-signal_cfg.s_sc, "Exit Long", "#2ca02c", "dot"),
    ]:
        fig.add_hline(
            y=val, line_dash=dash, line_color=color,
            annotation_text=name, annotation_position="right",
        )

    fig.add_hline(y=0, line_dash="solid", line_color="gray", opacity=0.5)

    fig.update_layout(
        title=f"S-Score: {ticker}",
        xaxis_title="Date",
        yaxis_title="s-score",
        hovermode="x unified",
        template="plotly_white",
    )
    return fig


def plot_pnl_histogram(trades: pd.DataFrame) -> go.Figure:
    """Plot PnL distribution histogram."""
    if trades.empty or "pnl" not in trades.columns:
        fig = go.Figure()
        fig.update_layout(title="PnL Distribution")
        return fig

    longs = trades[trades["direction"] == 1]["pnl"]
    shorts = trades[trades["direction"] == -1]["pnl"]

    fig = go.Figure()
    if not longs.empty:
        fig.add_trace(go.Histogram(
            x=longs, name="Long", marker_color="#2ca02c", opacity=0.7,
        ))
    if not shorts.empty:
        fig.add_trace(go.Histogram(
            x=shorts, name="Short", marker_color="#d62728", opacity=0.7,
        ))
    fig.update_layout(
        title="Trade PnL Distribution",
        xaxis_title="PnL ($)",
        yaxis_title="Count",
        barmode="overlay",
        template="plotly_white",
    )
    return fig


def plot_cumulative_pnl(trades: pd.DataFrame) -> go.Figure:
    """Plot cumulative realized PnL over time."""
    if trades.empty or "pnl" not in trades.columns:
        fig = go.Figure()
        fig.update_layout(title="Cumulative PnL")
        return fig

    sorted_trades = trades.sort_values("exit_date")
    cum_pnl = sorted_trades["pnl"].cumsum()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=sorted_trades["exit_date"],
        y=cum_pnl.values,
        mode="lines",
        name="Cumulative PnL",
        line=dict(color="#1f77b4", width=2),
    ))
    fig.update_layout(
        title="Cumulative Realized PnL",
        xaxis_title="Date",
        yaxis_title="PnL ($)",
        template="plotly_white",
    )
    return fig


def plot_regime_proba(
    regime_proba: pd.Series,
    equity: pd.Series,
    entry_threshold: float = 0.5,
) -> go.Figure:
    """
    Plot HMM P(favorable regime) alongside the equity curve.

    Shades periods where the model considers the regime unfavorable
    (P < entry_threshold) so you can see whether the filter fires at
    the right times (e.g. during drawdowns).
    """
    fig = go.Figure()

    # Equity on secondary y-axis
    fig.add_trace(go.Scatter(
        x=equity.index, y=equity.values,
        mode="lines", name="Equity",
        line=dict(color="#1f77b4", width=1.5, dash="dot"),
        yaxis="y2",
        opacity=0.6,
    ))

    # Regime probability
    fig.add_trace(go.Scatter(
        x=regime_proba.index, y=regime_proba.values,
        mode="lines", name="P(Favorable)",
        line=dict(color="#2ca02c", width=2),
        fill="tozeroy",
        fillcolor="rgba(44, 160, 44, 0.15)",
    ))

    # Threshold line
    fig.add_hline(
        y=entry_threshold,
        line_dash="dash",
        line_color="#d62728",
        annotation_text=f"Entry threshold ({entry_threshold})",
        annotation_position="right",
    )

    # Shade unfavorable periods
    blocked = regime_proba < entry_threshold
    in_block = False
    block_start = None
    for date, is_blocked in blocked.items():
        if is_blocked and not in_block:
            block_start = date
            in_block = True
        elif not is_blocked and in_block:
            fig.add_vrect(
                x0=block_start, x1=date,
                fillcolor="rgba(214, 39, 40, 0.10)",
                layer="below", line_width=0,
            )
            in_block = False
    if in_block and block_start is not None:
        fig.add_vrect(
            x0=block_start, x1=regime_proba.index[-1],
            fillcolor="rgba(214, 39, 40, 0.10)",
            layer="below", line_width=0,
        )

    fig.update_layout(
        title="HMM Regime Probability — P(Favorable / Mean-Reverting)",
        xaxis_title="Date",
        yaxis=dict(title="P(Favorable)", range=[0, 1.05]),
        yaxis2=dict(
            title="Equity ($)",
            overlaying="y",
            side="right",
            showgrid=False,
        ),
        hovermode="x unified",
        template="plotly_white",
        legend=dict(x=0.01, y=0.99),
    )
    return fig


def plot_notional_distribution(daily_positions: pd.DataFrame) -> go.Figure:
    """
    Histogram of position notionals — shows whether vol-targeting is
    creating a spread of sizes (good) or a spike at one value (equal-notional).
    """
    if daily_positions.empty or "notional" not in daily_positions.columns:
        fig = go.Figure()
        fig.update_layout(title="Position Notional Distribution")
        return fig

    notionals = daily_positions["notional"].dropna()
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=notionals,
        nbinsx=40,
        marker_color="#ff7f0e",
        opacity=0.8,
        name="Notional",
    ))
    fig.update_layout(
        title="Position Notional Distribution (Vol-Targeting Diagnostic)",
        xaxis_title="Notional ($)",
        yaxis_title="Count",
        template="plotly_white",
    )
    return fig


def plot_sector_sharpes(trades: pd.DataFrame, sector_mapping: dict) -> go.Figure:
    """Plot per-sector Sharpe ratios."""
    if trades.empty or "pnl" not in trades.columns:
        fig = go.Figure()
        fig.update_layout(title="Sector Sharpe Ratios")
        return fig

    trades = trades.copy()
    trades["sector_etf"] = trades["ticker"].map(sector_mapping).fillna("Unknown")

    sector_sharpes = {}
    for sector, group in trades.groupby("sector_etf"):
        if len(group) < 5:
            continue
        mean_pnl = group["pnl"].mean()
        std_pnl = group["pnl"].std()
        if std_pnl > 0:
            sector_sharpes[sector] = mean_pnl / std_pnl * np.sqrt(252)

    if not sector_sharpes:
        fig = go.Figure()
        fig.update_layout(title="Sector Sharpe Ratios")
        return fig

    sectors = list(sector_sharpes.keys())
    values = list(sector_sharpes.values())
    colors = ["#2ca02c" if v > 0 else "#d62728" for v in values]

    fig = go.Figure(go.Bar(
        x=sectors, y=values,
        marker_color=colors,
    ))
    fig.update_layout(
        title="Per-Sector Sharpe Ratios",
        xaxis_title="Sector ETF",
        yaxis_title="Sharpe Ratio",
        template="plotly_white",
    )
    return fig
