"""Session state helpers for the Streamlit app."""
import streamlit as st

from config import Config
from statarb.backtest.engine import BacktestResult


def get_config() -> Config | None:
    """Retrieve the Config from session state."""
    return st.session_state.get("config")


def set_config(config: Config):
    """Store the Config in session state."""
    st.session_state["config"] = config


def get_backtest_result() -> BacktestResult | None:
    """Retrieve the BacktestResult from session state."""
    return st.session_state.get("backtest_result")


def set_backtest_result(result: BacktestResult):
    """Store the BacktestResult in session state."""
    st.session_state["backtest_result"] = result


def has_backtest_result() -> bool:
    """Check if a backtest result exists in session state."""
    return "backtest_result" in st.session_state
