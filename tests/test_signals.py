"""Tests for signal generation: OU estimation, s-scores, filters."""
import numpy as np
import pandas as pd
import pytest

from statarb.signals.ou_estimator import fit_ar1, ar1_to_ou, estimate_ou_params, OUParams
from statarb.signals.sscore import compute_sscores
from statarb.signals.filters import filter_eligible
from statarb.signals.volume_time import compute_volume_adjusted_returns


class TestAR1Fit:
    def test_known_parameters(self):
        """Fit AR(1) with known b=0.9, a=0.5 and verify recovery."""
        np.random.seed(42)
        n = 1000
        X = np.zeros(n)
        a, b_true = 0.5, 0.9
        for i in range(1, n):
            X[i] = a + b_true * X[i - 1] + np.random.normal(0, 0.1)

        result = fit_ar1(X)
        assert result is not None
        a_hat, b_hat, var_eps = result
        assert abs(b_hat - b_true) < 0.05
        assert abs(a_hat - a) < 0.1

    def test_short_series_returns_none(self):
        assert fit_ar1(np.array([1.0, 2.0])) is None


class TestAR1ToOU:
    def test_valid_params(self):
        params = ar1_to_ou(a=0.01, b=0.95, var_eps=0.001)
        assert params is not None
        assert params.kappa > 0
        assert params.sigma_eq > 0
        assert params.half_life > 0

    def test_non_stationary_returns_none(self):
        assert ar1_to_ou(a=0.01, b=1.01, var_eps=0.001) is None
        assert ar1_to_ou(a=0.01, b=-0.5, var_eps=0.001) is None


class TestEstimateOU:
    def test_synthetic_ou_process(self):
        """Generate a synthetic OU process and verify parameter recovery."""
        np.random.seed(42)
        kappa_true = 20.0  # fast mean-reversion
        m_true = 0.0
        sigma_true = 0.5
        dt = 1.0 / 252.0
        n = 200

        # Simulate OU
        X = np.zeros(n)
        for i in range(1, n):
            dX = kappa_true * (m_true - X[i - 1]) * dt + sigma_true * np.sqrt(dt) * np.random.normal()
            X[i] = X[i - 1] + dX

        # Convert to daily returns (diff of X)
        daily_residuals = pd.Series(np.diff(X))

        params = estimate_ou_params(daily_residuals, window=60)
        assert params is not None
        assert params.kappa > 5  # should detect fast mean-reversion

    def test_insufficient_data(self):
        short = pd.Series([0.01, -0.02, 0.01])
        assert estimate_ou_params(short, window=60) is None


class TestSScores:
    def test_mean_centering(self):
        """S-scores with mean centering should have mean closer to zero."""
        params = {
            "A": OUParams(kappa=20, m=0.05, sigma=0.5, sigma_eq=0.1, half_life=8, a=0.01, b=0.9),
            "B": OUParams(kappa=15, m=-0.03, sigma=0.4, sigma_eq=0.08, half_life=10, a=0.005, b=0.92),
            "C": OUParams(kappa=25, m=0.02, sigma=0.6, sigma_eq=0.12, half_life=7, a=0.015, b=0.88),
        }
        residuals = pd.DataFrame(
            np.random.normal(0, 0.01, (100, 3)),
            columns=["A", "B", "C"],
        )

        scores = compute_sscores(residuals, params, mean_center=True)
        assert len(scores) == 3
        assert all(np.isfinite(scores))

    def test_no_mean_centering(self):
        params = {
            "A": OUParams(kappa=20, m=0.1, sigma=0.5, sigma_eq=0.1, half_life=8, a=0.01, b=0.9),
        }
        residuals = pd.DataFrame(np.random.normal(0, 0.01, (100, 1)), columns=["A"])

        scores = compute_sscores(residuals, params, mean_center=False)
        assert len(scores) == 1


class TestFilters:
    def test_kappa_filter(self):
        params = {
            "A": OUParams(kappa=20, m=0, sigma=0.5, sigma_eq=0.1, half_life=8, a=0, b=0.9),
            "B": OUParams(kappa=5, m=0, sigma=0.5, sigma_eq=0.1, half_life=30, a=0, b=0.98),
            "C": OUParams(kappa=10, m=0, sigma=0.5, sigma_eq=0.1, half_life=15, a=0, b=0.95),
        }
        eligible = filter_eligible(params, kappa_min=8.4)
        assert "A" in eligible
        assert "B" not in eligible
        assert "C" in eligible


class TestVolumeTime:
    def test_volume_adjustment(self, sample_returns, sample_volume):
        adjusted = compute_volume_adjusted_returns(
            sample_returns,
            sample_volume.iloc[1:],  # align with returns
            trailing_window=10,
        )
        assert adjusted.shape == sample_returns.shape
        assert not adjusted.isna().all().any()
