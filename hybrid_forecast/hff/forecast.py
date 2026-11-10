"""Fit on all available history and forecast the next H days from the last day."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import Config, effective_config
from .fusion import JointPredictor, StagedBackbone, StagedPredictor
from .ingest import Bundle
from .windows import Prepared


@dataclass
class FinalForecast:
    path: str
    modalities: str
    products: list
    dates: pd.DatetimeIndex                 # the H future days
    forecast: np.ndarray                    # (P, H) raw units
    hidden: dict                            # 'text' / 'image' / 'text+image' -> (P, H) with that input removed
    recent: np.ndarray                      # (P,) mean of the last 14 observed days
    prep: Prepared = field(repr=False, default=None)
    cfg: Config = field(repr=False, default=None)

    def table(self) -> pd.DataFrame:
        rows = []
        for p, name in enumerate(self.products):
            for h, d in enumerate(self.dates):
                rows.append({"product": name, "date": d, "forecast": float(self.forecast[p, h])})
        return pd.DataFrame(rows)

    def sensitivity(self) -> pd.DataFrame:
        """Change in the mean forecast (as % of the recent level) when an input is removed."""
        rows = []
        for p, name in enumerate(self.products):
            row = {"product": name}
            for k, v in self.hidden.items():
                row[f"removing {k}"] = (self.forecast[p].mean() - v[p].mean()) / max(self.recent[p], 1e-9)
            rows.append(row)
        return pd.DataFrame(rows)


def fit_forecast(bundle: Bundle, cfg: Config | None = None, path: str = "B", seed: int = 0,
                 progress=None) -> FinalForecast:
    """Train on every origin whose targets are known, then predict from the final day.

    path "B": joint network; path "A": staged encoders + LightGBM. Uses every modality the data supports.
    """
    cfg = effective_config(cfg or Config(), bundle.n_days, len(bundle.products))
    say = progress or (lambda f, m: None)
    D, H, W = bundle.n_days, cfg.horizon, cfg.window
    prep = Prepared(bundle, cfg, D - 1)
    S_all = prep.samples(np.arange(W - 1, D - H))
    S_now = prep.samples([D - 1])
    ut, ui = prep.use_text, prep.use_image
    dims = dict(W=W, H=H, Tw=cfg.text_window, Ct=prep.text_dim, S=prep.img_size)
    mods = "+".join(["sales"] + (["text"] if ut else []) + (["image"] if ui else []))

    say(0.1, "training")
    if path == "B":
        model = JointPredictor(dims, ut, ui, cfg, seed).fit(S_all)
    else:
        bb = StagedBackbone(dims, cfg, ut, ui, seed).fit(S_all)
        model = StagedPredictor(bb, ut, ui, cfg, seed).fit(S_all)
    say(0.8, "forecasting")

    scale = S_now.scale[:, None]
    fc = np.maximum(model.predict(S_now) * scale, 0.0)
    hidden = {}
    if ut:
        hidden["text"] = np.maximum(model.predict(S_now, ("text",)) * scale, 0.0)
    if ui:
        hidden["image"] = np.maximum(model.predict(S_now, ("image",)) * scale, 0.0)
    recent = bundle.sales[:, -14:].mean(axis=1)
    dates = pd.date_range(bundle.dates[-1] + pd.Timedelta(days=1), periods=H, freq="D")
    say(1.0, "done")
    return FinalForecast(path="Path B (joint)" if path == "B" else "Path A (staged)", modalities=mods,
                         products=bundle.products, dates=dates, forecast=fc, hidden=hidden, recent=recent,
                         prep=prep, cfg=cfg)
