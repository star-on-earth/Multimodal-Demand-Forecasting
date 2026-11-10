"""Central configuration and the rules for shrinking it to fit short datasets."""
from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class Config:
    # Forecast problem
    horizon: int = 14            # days predicted at once (direct multi-horizon)
    window: int = 56             # days of sales history fed to the models
    text_window: int = 14        # days of review features fed to the text branch
    test_days: int = 120         # chronological hold-out (target days)
    # Modalities
    img_size: int = 32
    max_img_age: int = 21        # an image older than this counts as "no image"
    lsa_dim: int = 16            # review embedding size (TF-IDF -> SVD)
    min_modality_coverage: float = 0.02
    # Training
    epochs: int = 40
    batch_size: int = 128
    patience: int = 6
    learning_rate: float = 2e-3
    modality_dropout: float = 0.25
    val_frac: float = 0.15
    enc_frac: float = 0.60       # staged fusion: share of train origins used to fit encoders
    seeds: tuple = (0,)
    # Ingestion
    max_products: int = 12
    min_review_count: int = 50

    def with_(self, **kw) -> "Config":
        return replace(self, **kw)


MIN_WINDOW = 35   # need 28 days of lags plus a week of slack for the weekday features
MIN_TRAIN_ORIGINS = 150


def effective_config(cfg: Config, n_days: int, n_products: int = 1) -> Config:
    """Shrink window / hold-out so short uploads still run, or explain why they cannot.

    Raises ValueError with the minimum length needed when even the shrunken
    settings leave too little training data.
    """
    h = cfg.horizon
    window = max(MIN_WINDOW, min(cfg.window, n_days // 4))
    test_days = max(h + 7, min(cfg.test_days, n_days // 4))
    new = cfg.with_(window=window, test_days=test_days,
                    text_window=min(cfg.text_window, window))
    n_train_origins = (n_days - test_days - h) - (window - 1)
    if n_train_origins * max(1, n_products) < MIN_TRAIN_ORIGINS:
        # smallest series that would leave MIN_TRAIN_ORIGINS for this many products
        per = -(-MIN_TRAIN_ORIGINS // max(1, n_products))
        need = per + (window - 1) + h + (h + 7)
        raise ValueError(
            f"Not enough history: {n_days} days across {n_products} product(s) leaves "
            f"{max(0, n_train_origins * n_products)} training windows (need at least "
            f"{MIN_TRAIN_ORIGINS}). Provide roughly {need}+ days of daily data per product."
        )
    return new
