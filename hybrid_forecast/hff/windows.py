"""Turn a ``Bundle`` into model-ready windows.

Everything at a forecast origin ``T`` uses only information from days ``<= T``:

* sales window  : days T-W+1 .. T          (+ observed/imputed flag, calendar)
* text window   : daily review embeddings for days T-Tw+1 .. T
* image         : the latest QC photo taken on or before T (and how old it is)
* targets       : days T+1 .. T+H  (direct multi-horizon, no recursion)

Text featurisation (TF-IDF + SVD) is fitted on reviews dated <= ``train_end_idx``
only, so no vocabulary or statistics leak from the hold-out period.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from .config import Config
from .ingest import Bundle

CAL_DIM = 4
FLAT_NAMES = ["last", "lag7", "lag14", "lag28", "mean7", "mean14", "mean28", "std7",
              "trend", "obs_frac28"]


def calendar_features(dates: pd.DatetimeIndex) -> np.ndarray:
    dow, doy = dates.dayofweek.values, dates.dayofyear.values
    return np.stack([np.sin(2 * np.pi * dow / 7), np.cos(2 * np.pi * dow / 7),
                     np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)],
                    axis=1).astype(np.float32)


def split_origins(n_days: int, cfg: Config):
    """Chronological split of forecast origins. No train target overlaps the hold-out."""
    W, H, T = cfg.window, cfg.horizon, cfg.test_days
    test_start = n_days - T
    train = np.arange(W - 1, test_start - H)          # targets end at test_start-1
    test = np.arange(test_start - 1, n_days - H)      # targets start at test_start
    return train, test, test_start


@dataclass
class TextTools:
    vec: TfidfVectorizer
    svd: TruncatedSVD
    mu: np.ndarray
    sd: np.ndarray


def build_text_daily(b: Bundle, cfg: Config, train_end_idx: int):
    """Daily per-product review features. Returns (array (P, D, k+3) | None, TextTools | None, reason)."""
    rv = b.reviews
    if rv is None:
        return None, None, "no reviews"
    tr = rv[rv["day_idx"] <= train_end_idx]
    if len(tr) < 30:
        return None, None, "too few reviews before the hold-out period"
    vec = TfidfVectorizer(min_df=2, max_df=0.9, stop_words="english", max_features=4000, sublinear_tf=True)
    try:
        Xtr = vec.fit_transform(tr["text"].values)
    except ValueError:
        return None, None, "reviews contain no usable words"
    if Xtr.shape[1] < 4:
        return None, None, "review vocabulary too small"
    k = int(min(cfg.lsa_dim, Xtr.shape[1] - 1, Xtr.shape[0] - 1))
    svd = TruncatedSVD(n_components=k, random_state=0).fit(Xtr)
    Z = svd.transform(vec.transform(rv["text"].values))
    P, D = b.sales.shape
    p, d = rv["p_idx"].values, rv["day_idx"].values
    sum_z, cnt = np.zeros((P, D, k)), np.zeros((P, D))
    np.add.at(sum_z, (p, d), Z)
    np.add.at(cnt, (p, d), 1)
    has = cnt > 0
    mean_z = sum_z / np.maximum(cnt, 1)[..., None]
    sel = has & (np.arange(D) <= train_end_idx)[None, :]
    if sel.sum() < 5:
        return None, None, "too few review-days before the hold-out period"
    mu, sd = mean_z[sel].mean(0), mean_z[sel].std(0) + 1e-6
    zs = np.where(has[..., None], (mean_z - mu) / sd, 0.0)
    rating = np.zeros((P, D))
    if b.has_rating:
        r = rv["rating"].values
        ok = np.isfinite(r)
        sum_r, cnt_r = np.zeros((P, D)), np.zeros((P, D))
        np.add.at(sum_r, (p[ok], d[ok]), (r[ok] - 3.0) / 2.0)
        np.add.at(cnt_r, (p[ok], d[ok]), 1)
        rating = np.where(cnt_r > 0, sum_r / np.maximum(cnt_r, 1), 0.0)
    logc = np.where(has, np.log1p(cnt) / 2.0, 0.0)
    daily = np.concatenate([zs, rating[..., None], logc[..., None], has[..., None].astype(float)], axis=2)
    return daily.astype(np.float32), TextTools(vec, svd, mu, sd), "ok"


@dataclass
class Samples:
    meta: pd.DataFrame            # p_idx, product, origin, date
    inputs: dict                  # name -> np.ndarray (N, ...)
    y: np.ndarray                 # (N, H) scaled targets (NaN where unknown)
    scale: np.ndarray             # (N,) per-product scale to return to raw units
    flat: np.ndarray              # (N, F) hand-crafted features
    snaive: np.ndarray            # (N, H) same-weekday value last week
    dow4: np.ndarray              # (N, H) mean of the same weekday over 4 weeks
    cal_f: np.ndarray             # (N, H, CAL_DIM) calendar of target days

    def __len__(self):
        return len(self.meta)

    def take(self, idx) -> "Samples":
        idx = np.asarray(idx)
        return Samples(self.meta.iloc[idx].reset_index(drop=True),
                       {k: v[idx] for k, v in self.inputs.items()},
                       self.y[idx], self.scale[idx], self.flat[idx], self.snaive[idx],
                       self.dow4[idx], self.cal_f[idx])

    @property
    def level(self) -> np.ndarray:
        return self.inputs["level"]


def hide_modalities(inputs: dict, hide=()) -> dict:
    """Return a copy of ``inputs`` with the named modalities zeroed and their masks cleared."""
    out = dict(inputs)
    if "text" in hide and "text" in out:
        out["text"] = np.zeros_like(out["text"])
        out["text_mask"] = np.zeros_like(out["text_mask"])
    if "image" in hide and "image" in out:
        out["image"] = np.zeros_like(out["image"])
        out["img_extra"] = np.zeros_like(out["img_extra"])
        out["img_mask"] = np.zeros_like(out["img_mask"])
    return out


class Prepared:
    """Scaling, text features and image look-ups fitted on data up to ``train_end_idx``."""

    def __init__(self, bundle: Bundle, cfg: Config, train_end_idx: int):
        self.b, self.cfg, self.train_end_idx = bundle, cfg, train_end_idx
        P, D = bundle.sales.shape
        H = cfg.horizon
        in_train = np.arange(D) <= train_end_idx
        w = bundle.observed & in_train[None, :]
        denom = w.sum(1)
        scale = np.where(denom > 0, (bundle.sales * w).sum(1) / np.maximum(denom, 1), 1.0)
        self.scale = np.where(scale > 1e-6, scale, 1.0)
        self.S = (bundle.sales / self.scale[:, None]).astype(np.float32)
        self.OBS = bundle.observed.astype(np.float32)
        self.cal_ext = calendar_features(pd.date_range(bundle.dates[0], periods=D + H, freq="D"))

        self.text_daily, self.text_tools, self.text_reason = None, None, "unavailable"
        if bundle.text_available:
            self.text_daily, self.text_tools, self.text_reason = build_text_daily(bundle, cfg, train_end_idx)
        self.use_text = self.text_daily is not None

        self.use_image = bool(bundle.image_available and bundle.images)
        self.img_latest = {}
        self.img_size = cfg.img_size
        for p, iset in bundle.images.items():
            self.img_latest[p] = np.searchsorted(iset.day_idx, np.arange(D), side="right") - 1
            self.img_size = iset.arr.shape[1]

    # -- helpers ---------------------------------------------------------------
    @property
    def text_dim(self) -> int:
        return 0 if self.text_daily is None else self.text_daily.shape[2]

    def samples(self, origins) -> Samples:
        cfg, b = self.cfg, self.b
        W, H, Tw = cfg.window, cfg.horizon, cfg.text_window
        origins = np.asarray(origins, dtype=int)
        parts = {k: [] for k in ["sales", "level", "cal", "text", "text_mask", "image", "img_extra", "img_mask",
                                 "y", "scale", "flat", "snaive", "dow4", "cal_f"]}
        metas = []
        hh = np.arange(1, H + 1)
        kk = np.ceil(hh / 7).astype(int)
        pos = (W - 1) + hh - 7 * kk
        for p, name in enumerate(b.products):
            n = len(origins)
            wi = origins[:, None] + np.arange(-W + 1, 1)[None, :]
            s = self.S[p][wi]
            ob = self.OBS[p][wi]
            x_sales = np.concatenate([s[..., None], ob[..., None], self.cal_ext[wi]], axis=2)
            fut = origins[:, None] + hh[None, :]
            s_pad = np.concatenate([self.S[p], np.full(H, np.nan, dtype=np.float32)])
            parts["sales"].append(x_sales)
            parts["level"].append(s[:, -7:].mean(1, keepdims=True))
            parts["cal_f"].append(self.cal_ext[fut])
            parts["cal"].append(self.cal_ext[fut].reshape(n, -1))
            parts["y"].append(s_pad[fut])
            parts["scale"].append(np.full(n, self.scale[p]))
            parts["flat"].append(np.stack([
                s[:, -1], s[:, -7], s[:, -14], s[:, -28], s[:, -7:].mean(1), s[:, -14:].mean(1),
                s[:, -28:].mean(1), s[:, -7:].std(1), s[:, -7:].mean(1) - s[:, -28:].mean(1),
                ob[:, -28:].mean(1)], axis=1))
            parts["snaive"].append(s[:, pos])
            parts["dow4"].append(np.mean([s[:, pos - 7 * j] for j in range(4)], axis=0))
            if self.use_text:
                tw = origins[:, None] + np.arange(-Tw + 1, 1)[None, :]
                tx = self.text_daily[p][tw]
                parts["text"].append(tx)
                parts["text_mask"].append((tx[..., -1].sum(1, keepdims=True) > 0).astype(np.float32))
            if self.use_image:
                S_ = self.img_size
                img = np.zeros((n, S_, S_, 3), dtype=np.float32)
                extra = np.zeros((n, 2), dtype=np.float32)
                mask = np.zeros((n, 1), dtype=np.float32)
                if p in b.images:
                    iset = b.images[p]
                    idx = self.img_latest[p][origins]
                    age = origins - np.where(idx >= 0, iset.day_idx[np.maximum(idx, 0)], -10**9)
                    ok = (idx >= 0) & (age <= cfg.max_img_age)
                    img[ok] = iset.arr[idx[ok]]
                    extra[ok, 0] = age[ok] / cfg.max_img_age
                    extra[ok, 1] = 1.0
                    mask[ok, 0] = 1.0
                parts["image"].append(img)
                parts["img_extra"].append(extra)
                parts["img_mask"].append(mask)
            metas.append(pd.DataFrame({"p_idx": p, "product": name, "origin": origins,
                                       "date": b.dates[0] + pd.to_timedelta(origins, unit="D")}))
        cat = lambda k: np.concatenate(parts[k], axis=0)
        inputs = {"sales": cat("sales").astype(np.float32), "level": cat("level").astype(np.float32),
                  "cal": cat("cal").astype(np.float32)}
        if self.use_text:
            inputs["text"] = cat("text")
            inputs["text_mask"] = cat("text_mask")
        if self.use_image:
            inputs["image"] = cat("image")
            inputs["img_extra"] = cat("img_extra")
            inputs["img_mask"] = cat("img_mask")
        return Samples(pd.concat(metas, ignore_index=True), inputs, cat("y").astype(np.float32),
                       cat("scale"), cat("flat").astype(np.float32), cat("snaive").astype(np.float32),
                       cat("dow4").astype(np.float32), cat("cal_f").astype(np.float32))
