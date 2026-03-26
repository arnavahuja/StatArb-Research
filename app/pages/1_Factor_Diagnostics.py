"""
Factor Diagnostics Page

Visualizes factor model internals:
- Return correlation heatmap
- Eigenvalue spectrum (PCA)
- Beta loadings heatmap
- Factor return time series
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from app.state import get_backtest_result, get_config, has_backtest_result
from app.components.charts import plot_correlation_heatmap, plot_eigenvalue_spectrum


st.set_page_config(page_title="Factor Diagnostics", layout="wide")
st.title("Factor Diagnostics")

if not has_backtest_result():
    st.warning("Run a backtest on the Home page first.")
    st.stop()

result = get_backtest_result()
config = get_config()
factor_result = result.factor_result

# ── Return Correlation Heatmap ──
st.subheader("Return Correlation Matrix")
if not factor_result.residuals.empty:
    returns_for_corr = factor_result.residuals.dropna(how="all")
    if len(returns_for_corr) > 30:
        corr = returns_for_corr.corr()
        st.plotly_chart(
            plot_correlation_heatmap(corr),
            use_container_width=True,
        )

# ── Eigenvalue Spectrum (PCA only) ──
if "all_eigenvalues" in factor_result.metadata:
    st.subheader("Eigenvalue Spectrum")
    eigenvalues = factor_result.metadata["all_eigenvalues"]
    st.plotly_chart(
        plot_eigenvalue_spectrum(eigenvalues),
        use_container_width=True,
    )

    n_components = factor_result.metadata.get("n_components", 0)
    explained = factor_result.metadata.get("explained_variance_ratio", 0)
    st.info(
        f"Using **{n_components}** components explaining "
        f"**{explained:.1%}** of total variance."
    )

# ── Beta Loadings Heatmap ──
st.subheader("Factor Loadings (Betas)")
if not factor_result.betas.empty:
    betas = factor_result.betas
    # Show top components only if PCA
    if betas.shape[1] > 10:
        betas = betas.iloc[:, :10]

    fig = go.Figure(data=go.Heatmap(
        z=betas.values,
        x=betas.columns.tolist(),
        y=betas.index.tolist(),
        colorscale="RdBu_r",
        zmid=0,
    ))
    fig.update_layout(
        title="Factor Loadings Heatmap",
        xaxis_title="Factor",
        yaxis_title="Ticker",
        height=max(400, len(betas) * 15),
        template="plotly_white",
    )
    st.plotly_chart(fig, use_container_width=True)

# ── Factor Returns Timeline ──
st.subheader("Factor Returns")
if not factor_result.factor_returns.empty:
    fr = factor_result.factor_returns.dropna(how="all")
    # Select which factors to display
    factor_cols = fr.columns.tolist()
    display_cols = factor_cols[:min(8, len(factor_cols))]

    selected = st.multiselect(
        "Select factors to plot",
        factor_cols,
        default=display_cols,
    )

    if selected:
        cum_returns = fr[selected].cumsum()
        fig = go.Figure()
        for col in selected:
            fig.add_trace(go.Scatter(
                x=cum_returns.index,
                y=cum_returns[col],
                mode="lines",
                name=col,
            ))
        fig.update_layout(
            title="Cumulative Factor Returns",
            xaxis_title="Date",
            yaxis_title="Cumulative Return",
            hovermode="x unified",
            template="plotly_white",
        )
        st.plotly_chart(fig, use_container_width=True)

# ── R-squared per Stock ──
r_squared = factor_result.metadata.get("r_squared", {})
if r_squared:
    st.subheader("R-squared per Stock")
    r2_df = pd.Series(r_squared).sort_values(ascending=False)
    fig = go.Figure(go.Bar(
        x=r2_df.index.tolist(),
        y=r2_df.values,
        marker_color="#1f77b4",
    ))
    fig.update_layout(
        title="Variance Explained by Factor Model (per Stock)",
        xaxis_title="Ticker",
        yaxis_title="R-squared",
        template="plotly_white",
    )
    st.plotly_chart(fig, use_container_width=True)
