"""
Trade Analytics Page

Trade-level performance analysis:
- KPI cards (PnL, win rate, profit factor, holding period)
- PnL distribution histograms
- Cumulative PnL
- Long vs Short breakdown
- Holding period analysis
- Sector Sharpe breakdown
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from app.state import get_backtest_result, get_config, has_backtest_result
from app.components.charts import plot_pnl_histogram, plot_cumulative_pnl, plot_sector_sharpes
from config import SECTOR_TO_ETF_MAP


st.set_page_config(page_title="Trade Analytics", layout="wide")
st.title("Trade Analytics")

if not has_backtest_result():
    st.warning("Run a backtest on the Home page first.")
    st.stop()

result = get_backtest_result()
config = get_config()
trades = result.trades

if trades.empty:
    st.warning("No trades were executed in this backtest.")
    st.stop()

# ── Trade KPI Cards ──
st.subheader("Trade-Level KPIs")
cols = st.columns(6)

total_pnl = trades["pnl"].sum()
win_trades = trades[trades["pnl"] > 0]
lose_trades = trades[trades["pnl"] <= 0]
win_rate = len(win_trades) / len(trades) if len(trades) > 0 else 0
gross_profit = win_trades["pnl"].sum()
gross_loss = abs(lose_trades["pnl"].sum())
profit_factor = gross_profit / (gross_loss + 1e-10)

if "entry_date" in trades.columns and "exit_date" in trades.columns:
    holding = (
        pd.to_datetime(trades["exit_date"]) - pd.to_datetime(trades["entry_date"])
    ).dt.days
    avg_hold = holding.mean()
else:
    avg_hold = 0
    holding = pd.Series(dtype=float)

cols[0].metric("Total PnL", f"${total_pnl:,.0f}")
cols[1].metric("Win Rate", f"{win_rate:.1%}")
cols[2].metric("Profit Factor", f"{profit_factor:.2f}")
cols[3].metric("Avg Holding (days)", f"{avg_hold:.1f}")
cols[4].metric("Best Trade", f"${trades['pnl'].max():,.0f}")
cols[5].metric("Worst Trade", f"${trades['pnl'].min():,.0f}")

# ── Additional Risk Metrics ──
st.subheader("Risk Metrics")
cols = st.columns(4)

long_trades = trades[trades["direction"] == 1]
short_trades = trades[trades["direction"] == -1]

long_win_rate = (
    len(long_trades[long_trades["pnl"] > 0]) / len(long_trades)
    if len(long_trades) > 0 else 0
)
short_win_rate = (
    len(short_trades[short_trades["pnl"] > 0]) / len(short_trades)
    if len(short_trades) > 0 else 0
)

cols[0].metric("Long Win Rate", f"{long_win_rate:.1%}")
cols[1].metric("Short Win Rate", f"{short_win_rate:.1%}")
cols[2].metric("Long Trades", f"{len(long_trades):,}")
cols[3].metric("Short Trades", f"{len(short_trades):,}")

# ── PnL Histogram ──
st.subheader("PnL Distribution")
st.plotly_chart(plot_pnl_histogram(trades), use_container_width=True)

# ── Cumulative PnL ──
st.subheader("Cumulative Realized PnL")
st.plotly_chart(plot_cumulative_pnl(trades), use_container_width=True)

# ── Long vs Short Cumulative PnL ──
st.subheader("Long vs Short Performance")
if not long_trades.empty or not short_trades.empty:
    fig = go.Figure()
    if not long_trades.empty:
        lt_sorted = long_trades.sort_values("exit_date")
        fig.add_trace(go.Scatter(
            x=lt_sorted["exit_date"],
            y=lt_sorted["pnl"].cumsum(),
            mode="lines", name="Long",
            line=dict(color="#2ca02c", width=2),
        ))
    if not short_trades.empty:
        st_sorted = short_trades.sort_values("exit_date")
        fig.add_trace(go.Scatter(
            x=st_sorted["exit_date"],
            y=st_sorted["pnl"].cumsum(),
            mode="lines", name="Short",
            line=dict(color="#d62728", width=2),
        ))
    fig.update_layout(
        xaxis_title="Date", yaxis_title="Cumulative PnL ($)",
        template="plotly_white", hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)

# ── Holding Period vs Return Scatter ──
st.subheader("Holding Period vs PnL")
if not holding.empty:
    fig = go.Figure()
    colors = ["#2ca02c" if d == 1 else "#d62728" for d in trades["direction"]]
    fig.add_trace(go.Scatter(
        x=holding,
        y=trades["pnl"],
        mode="markers",
        marker=dict(color=colors, opacity=0.6, size=6),
        text=trades["ticker"],
        hovertemplate="Ticker: %{text}<br>Hold: %{x} days<br>PnL: $%{y:,.0f}",
    ))
    fig.update_layout(
        xaxis_title="Holding Period (days)",
        yaxis_title="PnL ($)",
        template="plotly_white",
    )
    st.plotly_chart(fig, use_container_width=True)

# ── Ticker Contribution ──
st.subheader("PnL by Ticker")
ticker_pnl = trades.groupby("ticker")["pnl"].sum().sort_values(ascending=False)
colors = ["#2ca02c" if v > 0 else "#d62728" for v in ticker_pnl.values]
fig = go.Figure(go.Bar(
    x=ticker_pnl.index.tolist(),
    y=ticker_pnl.values,
    marker_color=colors,
))
fig.update_layout(
    xaxis_title="Ticker", yaxis_title="Total PnL ($)",
    template="plotly_white",
)
st.plotly_chart(fig, use_container_width=True)

# ── Sector Sharpe Breakdown ──
st.subheader("Per-Sector Sharpe Ratios")
sector_mapping = result.factor_result.metadata.get("sector_mapping", {})
if not sector_mapping:
    # Build from config
    from statarb.data.universe import get_sector_mapping
    sector_mapping = {t: SECTOR_TO_ETF_MAP.get(t, "XLK") for t in trades["ticker"].unique()}
st.plotly_chart(
    plot_sector_sharpes(trades, sector_mapping),
    use_container_width=True,
)
