"""
Backtest engine: day-by-day simulation (Paper Section 5).

Orchestrates the full pipeline:
    1. (Optional) Fit HMM on training window; compute causal regime probs
    2. For each day, re-estimate OU parameters on trailing residuals
    3. Compute s-scores
    4. Gate entries by HMM regime probability (if enabled)
    5. Generate entry/exit signals
    6. Size positions with vol-targeting or equal-notional (if enabled)
    7. Execute trades via PortfolioManager
    8. Record equity, positions, and trades
"""
from dataclasses import dataclass, field
import logging

import numpy as np
import pandas as pd

from config import Config
from statarb.factors.base import FactorResult
from statarb.signals.ou_estimator import estimate_ou_params, OUParams
from statarb.signals.sscore import compute_sscores
from statarb.signals.filters import filter_eligible
from statarb.signals.volume_time import compute_volume_adjusted_returns
from statarb.extensions.vol_targeting import VolTargetedSizer
from statarb.extensions.hmm_regime import HMMRegimeDetector
from .portfolio import PortfolioManager
from .metrics import compute_metrics, PerformanceMetrics

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    """Complete output from a backtest run."""
    equity_curve: pd.Series
    trades: pd.DataFrame
    daily_positions: pd.DataFrame
    daily_sscores: pd.DataFrame
    metrics: PerformanceMetrics
    factor_result: FactorResult
    daily_ou_params: dict = field(default_factory=dict)
    # Regime probabilities indexed by date (None if HMM disabled)
    regime_proba: pd.Series | None = None


def _setup_hmm(
    config: Config, residuals: pd.DataFrame
) -> tuple[np.ndarray | None, HMMRegimeDetector | None]:
    """
    Build HMM features, fit the model on the training window, and return
    the full causal probability array (one value per day in residuals).

    Returns (favorable_proba_array, detector) or (None, None) if disabled.
    """
    if not config.hmm.enabled:
        return None, None

    detector = HMMRegimeDetector(
        n_states=config.hmm.n_states,
        training_window=config.hmm.training_window,
        feature_window=config.hmm.feature_window,
    )

    features = detector.build_features(residuals)

    try:
        detector.fit(features)
    except ValueError as exc:
        logger.warning("HMM fit failed (%s) — regime gating disabled.", exc)
        return None, None

    proba = detector.predict_proba_causal(features)
    favorable = detector.get_favorable_proba(proba)
    logger.info(
        "HMM fitted. Favorable state: %d. Mean P(favorable)=%.3f",
        detector.favorable_state,
        np.nanmean(favorable),
    )
    return favorable, detector


def run_backtest(
    config: Config,
    prices: pd.DataFrame,
    volume: pd.DataFrame,
    factor_result: FactorResult,
) -> BacktestResult:
    """
    Run the full day-by-day backtest.

    Args:
        config: Master configuration object.
        prices: Adjusted close prices (dates x tickers).
        volume: Daily volume (dates x tickers).
        factor_result: Pre-computed factor model output with residuals.

    Returns:
        BacktestResult with equity curve, trades, diagnostics, and (if
        enabled) regime probability series.
    """
    residuals = factor_result.residuals
    tickers = [t for t in residuals.columns if t in prices.columns]
    residuals = residuals[tickers]
    prices = prices[tickers]

    # ── Volume adjustment ──────────────────────────────────────────────
    if config.volume.enabled:
        vol_tickers = [t for t in tickers if t in volume.columns]
        if vol_tickers:
            adjusted = compute_volume_adjusted_returns(
                residuals[vol_tickers],
                volume[vol_tickers],
                trailing_window=config.volume.trailing_window,
            )
            residuals = residuals.copy()
            residuals[vol_tickers] = adjusted

    dates = residuals.index
    ou_window = config.ou.estimation_window
    min_start = ou_window + 10  # need enough history for reliable OU estimates

    # ── HMM setup (pre-loop, causal) ──────────────────────────────────
    favorable_proba, hmm_detector = _setup_hmm(config, residuals)

    # ── Vol-targeting sizer ───────────────────────────────────────────
    vol_sizer = VolTargetedSizer(
        floor_multiplier=config.vol_target.floor_multiplier,
        cap_multiplier=config.vol_target.cap_multiplier,
    ) if config.vol_target.enabled else None

    # ── Portfolio ─────────────────────────────────────────────────────
    portfolio = PortfolioManager(
        initial_equity=config.backtest.initial_equity,
        leverage_long=config.backtest.leverage_long,
        leverage_short=config.backtest.leverage_short,
        tc_bps=config.backtest.tc_bps,
    )

    # ── Storage ───────────────────────────────────────────────────────
    equity_values: list[float] = []
    equity_dates: list = []
    trade_records: list[dict] = []
    position_records: list[dict] = []
    sscore_records: dict = {}
    daily_ou_params: dict = {}
    regime_records: dict = {}

    n_target = max(len(tickers) // 2, 10)  # expected positions for sizing

    for i in range(min_start, len(dates)):
        date = dates[i]
        date_ts = pd.Timestamp(date)

        current_prices = prices.iloc[i]
        price_dict = current_prices.to_dict()

        # ── Step 1: Estimate OU parameters ────────────────────────────
        trailing_residuals = residuals.iloc[max(0, i - ou_window) : i]
        ou_params: dict[str, OUParams] = {}
        for ticker in tickers:
            if ticker not in trailing_residuals.columns:
                continue
            series = trailing_residuals[ticker]
            params = estimate_ou_params(
                series, window=ou_window, dt=config.backtest.dt
            )
            if params is not None:
                ou_params[ticker] = params

        daily_ou_params[str(date)[:10]] = ou_params

        # ── Step 2: Filter by kappa ───────────────────────────────────
        eligible = filter_eligible(ou_params, kappa_min=config.ou.kappa_min)

        # ── Step 3: Compute s-scores ──────────────────────────────────
        eligible_params = {t: ou_params[t] for t in eligible}
        if not eligible_params:
            portfolio.mark_to_market(price_dict)
            equity_values.append(portfolio.equity)
            equity_dates.append(date)
            continue

        sscores = compute_sscores(
            residuals.iloc[:i],
            eligible_params,
            mean_center=config.ou.mean_center,
        )
        sscore_records[date] = sscores

        # ── Step 4: HMM regime probability ───────────────────────────
        in_favorable_regime = True
        if favorable_proba is not None:
            p_fav = float(favorable_proba[i]) if not np.isnan(favorable_proba[i]) else 1.0
            in_favorable_regime = p_fav >= config.hmm.entry_threshold
            regime_records[date] = p_fav

        # ── Step 5: Check exits ───────────────────────────────────────
        # Exits are NOT gated by the HMM — we always honour close signals
        # and close positions that lose eligibility.
        tickers_to_close: list[str] = []
        for ticker in list(portfolio.positions.keys()):
            if ticker not in sscores.index:
                tickers_to_close.append(ticker)
                continue

            s = sscores[ticker]
            pos = portfolio.positions[ticker]

            should_close = False
            if pos.direction == 1:   # long: close when s reverts above -s_sc
                if s >= -config.signal.s_sc:
                    should_close = True
            elif pos.direction == -1:  # short: close when s reverts below +s_bc
                if s <= config.signal.s_bc:
                    should_close = True

            if abs(s) >= config.signal.s_limit:
                should_close = True

            if should_close:
                tickers_to_close.append(ticker)

        for ticker in tickers_to_close:
            if ticker not in price_dict or not np.isfinite(price_dict[ticker]):
                continue
            pos = portfolio.positions.get(ticker)
            if pos is None:
                continue
            pnl = portfolio.close_position(ticker, price_dict[ticker], date_ts)
            trade_records.append({
                "ticker": ticker,
                "direction": pos.direction,
                "entry_date": pos.entry_date,
                "exit_date": date_ts,
                "entry_price": pos.entry_price,
                "exit_price": price_dict[ticker],
                "pnl": pnl,
                "notional": pos.notional,
            })

        # ── Step 6: Check entries ─────────────────────────────────────
        # Skip new entries if HMM says we are in an unfavorable regime.
        if in_favorable_regime:
            base_notional = portfolio.compute_notional_per_position(n_target)

            # Vol-targeting: compute cross-sectional target sigma once per day
            target_sigma: float | None = None
            if vol_sizer is not None:
                target_sigma = vol_sizer.compute_target_sigma(eligible_params)

            for ticker in eligible:
                if ticker in portfolio.positions:
                    continue
                if ticker not in sscores.index:
                    continue
                if ticker not in price_dict or not np.isfinite(price_dict[ticker]):
                    continue

                s = sscores[ticker]
                direction: int | None = None
                if s <= -config.signal.s_bo:
                    direction = 1    # buy to open
                elif s >= config.signal.s_so:
                    direction = -1   # sell to open

                if direction is None:
                    continue

                # Compute final notional (vol-targeted or equal-notional)
                if vol_sizer is not None and target_sigma is not None:
                    notional = vol_sizer.scale_notional(
                        base_notional,
                        ou_params[ticker].sigma_eq,
                        target_sigma,
                    )
                else:
                    notional = base_notional

                portfolio.open_position(
                    ticker=ticker,
                    direction=direction,
                    price=price_dict[ticker],
                    date=date_ts,
                    notional=notional,
                )

        # ── Step 7: Mark to market ────────────────────────────────────
        portfolio.mark_to_market(price_dict)
        equity_values.append(portfolio.equity)
        equity_dates.append(date)

        for ticker, pos in portfolio.positions.items():
            position_records.append({
                "date": date,
                "ticker": ticker,
                "direction": pos.direction,
                "notional": pos.notional,
                "entry_price": pos.entry_price,
                "current_price": price_dict.get(ticker, np.nan),
            })

    # ── Assemble outputs ──────────────────────────────────────────────
    equity_curve = pd.Series(equity_values, index=pd.DatetimeIndex(equity_dates))

    trades = pd.DataFrame(trade_records) if trade_records else pd.DataFrame(
        columns=["ticker", "direction", "entry_date", "exit_date",
                 "entry_price", "exit_price", "pnl", "notional"]
    )

    daily_positions = pd.DataFrame(position_records) if position_records else pd.DataFrame(
        columns=["date", "ticker", "direction", "notional",
                 "entry_price", "current_price"]
    )

    daily_sscores = pd.DataFrame(sscore_records).T
    daily_sscores.index = pd.DatetimeIndex(daily_sscores.index)

    regime_proba: pd.Series | None = None
    if regime_records:
        regime_proba = pd.Series(regime_records, name="p_favorable")
        regime_proba.index = pd.DatetimeIndex(regime_proba.index)

    metrics = compute_metrics(
        equity_curve, trades,
        risk_free_rate=config.backtest.risk_free_rate,
        total_costs=portfolio.total_costs,
    )

    return BacktestResult(
        equity_curve=equity_curve,
        trades=trades,
        daily_positions=daily_positions,
        daily_sscores=daily_sscores,
        metrics=metrics,
        factor_result=factor_result,
        daily_ou_params=daily_ou_params,
        regime_proba=regime_proba,
    )
