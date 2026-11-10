"""Keras models: GRU + 1D-CNN for sales, a text sequence encoder, an image CNN,
and the joint (end-to-end) fusion network.

Every model predicts ``H`` days at once and adds its output to the recent level,
so it only has to learn the *change* from "about what it was last week".
"""
from __future__ import annotations

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import keras  # noqa: E402
import numpy as np  # noqa: E402
from keras import layers, ops  # noqa: E402

from .config import Config  # noqa: E402

SALES_CH = 6   # scaled sales, observed flag, 4 calendar channels


class ModalityGate(layers.Layer):
    """Multiply an embedding by its presence mask and randomly drop whole modalities
    while training, so the network learns to work when a modality is absent.

    Output = concat(embedding * gate, gate), where gate is 0 or 1 per sample.
    """

    def __init__(self, drop_prob: float = 0.0, seed: int = 0, **kw):
        super().__init__(**kw)
        self.drop_prob = float(drop_prob)
        self.seed_gen = keras.random.SeedGenerator(seed)

    def call(self, inputs, training=False):
        emb, mask = inputs
        gate = mask
        if training and self.drop_prob > 0:
            keep = ops.cast(keras.random.uniform(ops.shape(mask), seed=self.seed_gen) >= self.drop_prob, mask.dtype)
            gate = mask * keep
        return ops.concatenate([emb * gate, gate], axis=-1)

    def compute_output_shape(self, input_shape):
        emb_shape, _ = input_shape
        return (emb_shape[0], emb_shape[-1] + 1)


# ---- encoders ---------------------------------------------------------------
def sales_branch(x, emb: int = 32):
    g = layers.GRU(32)(x)                                           # order of days
    c = layers.Conv1D(16, 7, padding="same", activation="relu")(x)  # local / weekly shapes
    c = layers.Conv1D(16, 5, padding="same", activation="relu")(c)
    c = layers.Concatenate()([layers.GlobalAveragePooling1D()(c), layers.GlobalMaxPooling1D()(c)])
    return layers.Dense(emb, activation="relu", name="sales_emb")(layers.Concatenate()([g, c]))


def text_branch(x, emb: int = 16):
    c = layers.Conv1D(16, 3, padding="same", activation="relu")(x)
    g = layers.GRU(16)(c)
    return layers.Dense(emb, activation="relu", name="text_emb")(g)


def image_branch(img, extra, emb: int = 16):
    h = layers.Conv2D(8, 3, padding="same", activation="relu")(img)
    h = layers.MaxPooling2D()(h)
    h = layers.Conv2D(16, 3, padding="same", activation="relu")(h)
    h = layers.MaxPooling2D()(h)
    h = layers.Conv2D(16, 3, padding="same", activation="relu")(h)
    h = layers.Concatenate()([layers.GlobalAveragePooling2D()(h), layers.GlobalMaxPooling2D()(h), extra])
    return layers.Dense(emb, activation="relu", name="image_emb")(h)


def _head(z, cal, level, H: int):
    z = layers.Concatenate()([z, cal])
    z = layers.Dense(64, activation="relu")(z)
    z = layers.Dropout(0.1)(z)
    delta = layers.Dense(H)(z)
    lvl = layers.Concatenate()([level] * H)
    return layers.Add(name="forecast")([delta, lvl])


def _inputs(dims: dict, names):
    spec = {"sales": (dims["W"], SALES_CH), "level": (1,), "cal": (dims["H"] * 4,),
            "text": (dims.get("Tw", 1), dims.get("Ct", 1)), "text_mask": (1,),
            "image": (dims.get("S", 1), dims.get("S", 1), 3), "img_extra": (2,), "img_mask": (1,)}
    return {n: keras.Input(shape=spec[n], name=n) for n in names}


def build_sales_model(dims: dict, seed: int = 0):
    """GRU + 1D-CNN sales-only forecaster. ``sales_emb`` layer = the sales embedding."""
    keras.utils.set_random_seed(seed)
    i = _inputs(dims, ["sales", "level", "cal"])
    out = _head(sales_branch(i["sales"]), i["cal"], i["level"], dims["H"])
    return keras.Model(list(i.values()), out, name="sales_gru_cnn")


def build_text_model(dims: dict, seed: int = 0):
    """Text-only encoder trained to forecast (given level + calendar)."""
    keras.utils.set_random_seed(seed)
    i = _inputs(dims, ["text", "text_mask", "level", "cal"])
    z = layers.Concatenate()([text_branch(i["text"]), i["text_mask"]])
    return keras.Model(list(i.values()), _head(z, i["cal"], i["level"], dims["H"]), name="text_encoder")


def build_image_model(dims: dict, seed: int = 0):
    """Image CNN trained to forecast (given level + calendar)."""
    keras.utils.set_random_seed(seed)
    i = _inputs(dims, ["image", "img_extra", "img_mask", "level", "cal"])
    z = layers.Concatenate()([image_branch(i["image"], i["img_extra"]), i["img_mask"]])
    return keras.Model(list(i.values()), _head(z, i["cal"], i["level"], dims["H"]), name="image_cnn")


def build_joint_model(dims: dict, use_text: bool, use_image: bool, drop_prob: float, seed: int = 0):
    """One network, one loss: all branches are trained together (end-to-end fusion)."""
    keras.utils.set_random_seed(seed)
    names = ["sales", "level", "cal"] + (["text", "text_mask"] if use_text else []) \
        + (["image", "img_extra", "img_mask"] if use_image else [])
    i = _inputs(dims, names)
    parts = [sales_branch(i["sales"])]
    if use_text:
        parts.append(ModalityGate(drop_prob, seed + 1, name="text_gate")([text_branch(i["text"]), i["text_mask"]]))
    if use_image:
        parts.append(ModalityGate(drop_prob, seed + 2, name="image_gate")(
            [image_branch(i["image"], i["img_extra"]), i["img_mask"]]))
    z = parts[0] if len(parts) == 1 else layers.Concatenate()(parts)
    tag = "_".join(["sales"] + (["text"] if use_text else []) + (["image"] if use_image else []))
    return keras.Model(list(i.values()), _head(z, i["cal"], i["level"], dims["H"]), name=f"joint_{tag}")


def embedding_model(model, layer_name: str):
    return keras.Model(model.inputs, model.get_layer(layer_name).output)


# ---- training ---------------------------------------------------------------
def time_split(meta, H: int, frac: float):
    """Chronological train/validation split with an H-day embargo between them."""
    origins = meta["origin"].values
    cut = np.quantile(origins, 1.0 - frac)
    tr = np.where(origins <= cut - H)[0]
    va = np.where(origins > cut)[0]
    return tr, va


def fit_model(model, S, idx_train, idx_val, cfg: Config):
    names = [t.name.split(":")[0] for t in model.inputs]
    x_tr = {k: S.inputs[k][idx_train] for k in names}
    x_va = {k: S.inputs[k][idx_val] for k in names}
    model.compile(optimizer=keras.optimizers.Adam(cfg.learning_rate), loss="mae")
    cb = [keras.callbacks.EarlyStopping(monitor="val_loss", patience=cfg.patience, restore_best_weights=True)]
    hist = model.fit(x_tr, S.y[idx_train], validation_data=(x_va, S.y[idx_val]), epochs=cfg.epochs,
                     batch_size=cfg.batch_size, callbacks=cb, verbose=0, shuffle=True)
    return hist


def predict(model, inputs: dict, batch_size: int = 512) -> np.ndarray:
    names = [t.name.split(":")[0] for t in model.inputs]
    return np.asarray(model.predict({k: inputs[k] for k in names}, batch_size=batch_size, verbose=0))
