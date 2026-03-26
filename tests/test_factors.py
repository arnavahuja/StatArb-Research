"""Tests for factor models."""
import numpy as np
import pandas as pd
import pytest

from statarb.factors.pca import PCAFactorModel
from statarb.factors.etf import ETFFactorModel
from statarb.factors.base import FactorResult


class TestPCAFactorModel:
    def test_fit_returns_factor_result(self, sample_returns):
        model = PCAFactorModel(n_components=5, lookback=252)
        result = model.fit(sample_returns)

        assert isinstance(result, FactorResult)
        assert result.residuals.shape == sample_returns.shape
        assert result.factor_returns.shape[0] == sample_returns.shape[0]
        assert result.factor_returns.shape[1] == 5
        assert result.betas.shape == (sample_returns.shape[1], 5)

    def test_residuals_orthogonal_to_factors(self, sample_returns):
        model = PCAFactorModel(n_components=5, lookback=252)
        result = model.fit(sample_returns)

        # Residuals should be approximately uncorrelated with factors
        for col in result.factor_returns.columns:
            for ticker in result.residuals.columns:
                corr = result.residuals[ticker].corr(result.factor_returns[col])
                assert abs(corr) < 0.3, (
                    f"Residual {ticker} correlated with {col}: {corr:.3f}"
                )

    def test_explained_variance_in_metadata(self, sample_returns):
        model = PCAFactorModel(n_components=5, lookback=252)
        result = model.fit(sample_returns)

        assert "explained_variance_ratio" in result.metadata
        assert "eigenvalues" in result.metadata
        assert "n_components" in result.metadata
        assert result.metadata["n_components"] == 5

    def test_variable_components(self, sample_returns):
        model = PCAFactorModel(
            n_components=None,
            explained_variance_threshold=0.55,
            lookback=252,
        )
        result = model.fit(sample_returns)

        assert result.metadata["n_components"] > 0
        assert result.metadata["explained_variance_ratio"] >= 0.50


class TestETFFactorModel:
    def test_fit_returns_factor_result(
        self, sample_returns, sample_sector_mapping, sample_etf_returns
    ):
        model = ETFFactorModel(
            sector_mapping=sample_sector_mapping,
            rolling_window=60,
        )
        result = model.fit(sample_returns, etf_returns=sample_etf_returns)

        assert isinstance(result, FactorResult)
        assert result.residuals.shape == sample_returns.shape

    def test_requires_etf_returns(self, sample_returns, sample_sector_mapping):
        model = ETFFactorModel(sector_mapping=sample_sector_mapping)
        with pytest.raises(ValueError, match="etf_returns is required"):
            model.fit(sample_returns)
