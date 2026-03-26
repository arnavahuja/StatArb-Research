"""Factory registry to build factor models from configuration."""
from config import FactorConfig
from .base import FactorModel
from .pca import PCAFactorModel
from .etf import ETFFactorModel
from .combined import CombinedFactorModel


def build_factor_model(
    cfg: FactorConfig, sector_mapping: dict[str, str]
) -> FactorModel:
    """
    Instantiate the correct FactorModel from configuration.

    Args:
        cfg: FactorConfig with model_type and parameters.
        sector_mapping: Dict mapping ticker -> sector ETF symbol.

    Returns:
        An instance of the appropriate FactorModel subclass.
    """
    if cfg.model_type == "pca":
        return PCAFactorModel(
            n_components=cfg.pca_n_components,
            explained_variance_threshold=cfg.explained_variance_threshold,
            use_ledoit_wolf=cfg.use_ledoit_wolf,
            lookback=cfg.pca_lookback,
        )
    elif cfg.model_type == "etf":
        return ETFFactorModel(
            sector_mapping=sector_mapping,
            rolling_window=cfg.beta_rolling_window,
        )
    elif cfg.model_type == "combined":
        return CombinedFactorModel(
            sector_mapping=sector_mapping,
            rolling_window=cfg.beta_rolling_window,
            pca_n_components=cfg.pca_n_components,
            pca_lookback=cfg.pca_lookback,
            use_ledoit_wolf=cfg.use_ledoit_wolf,
        )
    else:
        raise ValueError(
            f"Unknown factor model type: '{cfg.model_type}'. "
            f"Choose from: pca, etf, combined"
        )
