"""The two ways of combining modalities, plus a lags-only LightGBM baseline.

Path A - staged fusion
    1. Train a sales encoder (GRU + 1D-CNN), a text encoder and an image CNN
       *separately*, on the first ``enc_frac`` of the training origins.
    2. Freeze them, take their embeddings, and train a LightGBM stacker on the
       remaining (later) origins, with an H-day embargo so the stacker never sees
       an embedding produced by an encoder that was trained on the same labels.
    Cost of this design: the stacker only sees part of the training data.

Path B - joint fusion
    One network contains all branches; a single loss trains them together on all
    training origins. Whole modalities are randomly dropped while training so the
    network still works when text or images are missing.

All predictors return *scaled* forecasts of shape (N, H); multiply by
``Samples.scale`` for raw units.
"""
from __future__ import annotations

import lightgbm as lgb
import numpy as np

from . import nets
from .config import Config
from .windows import Samples, hide_modalities

LGB_PARAMS = dict(n_estimators=250, learning_rate=0.04, num_leaves=15, min_child_samples=30,
                  subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=1.0,
                  objective="regression_l1", verbose=-1, n_jobs=2)


def _stack(S: Samples, blocks) -> np.ndarray:
    """One row per (sample, horizon): lags/rolling stats, h, same-weekday values, target calendar, embeddings."""
    N, H = S.y.shape
    cols = [np.repeat(S.flat, H, axis=0),
            np.tile(np.arange(1, H + 1), N)[:, None],
            S.snaive.reshape(-1, 1), S.dow4.reshape(-1, 1), S.cal_f.reshape(N * H, -1)]
    cols += [np.repeat(b, H, axis=0) for b in blocks]
    return np.concatenate(cols, axis=1).astype(np.float32)


def _target(S: Samples) -> np.ndarray:
    return (S.y - S.level).reshape(-1)


def _fit_lgb(X, y, seed):
    return lgb.LGBMRegressor(random_state=seed, **LGB_PARAMS).fit(X, y)


class FlatLGBPredictor:
    """LightGBM on hand-built lag features only (the classical baseline)."""

    def __init__(self, seed: int = 0):
        self.seed = seed

    def fit(self, S: Samples):
        self.m = _fit_lgb(_stack(S, []), _target(S), self.seed)
        return self

    def predict(self, S: Samples, hide=()):
        return self.m.predict(_stack(S, [])).reshape(len(S), -1) + S.level


class JointPredictor:
    """Path B: end-to-end network over the chosen modalities."""

    def __init__(self, dims: dict, use_text: bool, use_image: bool, cfg: Config, seed: int = 0):
        self.dims, self.use_text, self.use_image, self.cfg, self.seed = dims, use_text, use_image, cfg, seed

    def fit(self, S: Samples):
        self.model = nets.build_joint_model(self.dims, self.use_text, self.use_image,
                                            self.cfg.modality_dropout, self.seed)
        itr, iva = nets.time_split(S.meta, self.cfg.horizon, self.cfg.val_frac)
        self.history = nets.fit_model(self.model, S, itr, iva, self.cfg)
        self.epochs = len(self.history.history["loss"])
        return self

    def predict(self, S: Samples, hide=()):
        return nets.predict(self.model, hide_modalities(S.inputs, hide))


class StagedBackbone:
    """Path A, steps 1-2: separately trained encoders, shared by every stacker variant."""

    def __init__(self, dims: dict, cfg: Config, use_text: bool, use_image: bool, seed: int = 0):
        self.dims, self.cfg, self.use_text, self.use_image, self.seed = dims, cfg, use_text, use_image, seed

    def fit(self, S: Samples):
        cfg, H = self.cfg, self.cfg.horizon
        origins = S.meta["origin"].values
        cut = np.quantile(origins, cfg.enc_frac)
        enc_idx = np.where(origins <= cut)[0]
        self.stack_idx = np.where(origins > cut + H)[0]          # embargo of H days
        if len(self.stack_idx) < 30:
            raise ValueError("Too little data left to train the staged stacker; use a longer series.")
        Se = S.take(enc_idx)
        itr, iva = nets.time_split(Se.meta, H, cfg.val_frac)
        self.sales = nets.build_sales_model(self.dims, self.seed)
        nets.fit_model(self.sales, Se, itr, iva, cfg)
        self.emb = {"sales": nets.embedding_model(self.sales, "sales_emb")}
        if self.use_text:
            m = nets.build_text_model(self.dims, self.seed + 1)
            nets.fit_model(m, Se, itr, iva, cfg)
            self.emb["text"] = nets.embedding_model(m, "text_emb")
        if self.use_image:
            m = nets.build_image_model(self.dims, self.seed + 2)
            nets.fit_model(m, Se, itr, iva, cfg)
            self.emb["image"] = nets.embedding_model(m, "image_emb")
        return self

    def blocks(self, S: Samples, use_text: bool, use_image: bool, hide=(), drop=0.0, rng=None):
        out = [nets.predict(self.emb["sales"], S.inputs)]
        for name, use, mask_key in (("text", use_text, "text_mask"), ("image", use_image, "img_mask")):
            if not use:
                continue
            e = nets.predict(self.emb[name], S.inputs).astype(np.float32)
            gone = (S.inputs[mask_key][:, 0] <= 0) | (name in hide)
            if drop > 0 and rng is not None:
                gone = gone | (rng.random(len(e)) < drop)
            e[gone] = np.nan                                      # LightGBM treats NaN as "missing"
            out.append(e)
        return out


class StagedPredictor:
    """Path A, step 3: LightGBM stacker over frozen embeddings for one modality subset."""

    def __init__(self, backbone: StagedBackbone, use_text: bool, use_image: bool, cfg: Config, seed: int = 0):
        self.bb, self.use_text, self.use_image, self.cfg, self.seed = backbone, use_text, use_image, cfg, seed

    def fit(self, S_all: Samples):
        Ss = S_all.take(self.bb.stack_idx)
        rng = np.random.default_rng(self.seed)
        blocks = self.bb.blocks(Ss, self.use_text, self.use_image, drop=self.cfg.modality_dropout, rng=rng)
        self.m = _fit_lgb(_stack(Ss, blocks), _target(Ss), self.seed)
        return self

    def predict(self, S: Samples, hide=()):
        blocks = self.bb.blocks(S, self.use_text, self.use_image, hide=hide)
        return self.m.predict(_stack(S, blocks)).reshape(len(S), -1) + S.level
