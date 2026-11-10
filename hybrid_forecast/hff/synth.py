"""Synthetic multimodal retail data: sales, review text and product images.

All three modalities are driven by shared latent processes, so the dependency
between them is known exactly:

* ``q_demand`` : product quality as it actually affects demand (lag ``LAG`` days).
* ``q_text``   : quality as reflected in review sentiment.
* ``q_image``  : quality as reflected in defects visible in QC photos.

``signal=True``  -> q_text and q_image are the same process as q_demand, so reviews
                    and photos *lead* demand by about ``LAG`` days.
``signal=False`` -> q_text / q_image come from independent processes with the same
                    statistics. The modalities look identical but carry no information
                    about demand. This is the control that shows a model does not
                    invent gains from noise.

Any improvement measured on ``signal=True`` data therefore reflects how the
generator was written, not a real-world claim. Use the control to sanity-check.
"""
from __future__ import annotations

import colorsys
import io
import os
import zipfile
from dataclasses import dataclass

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

LAG = 6                       # days between a quality change and its effect on demand
GAMMA = 1.8                   # strength of the quality -> demand effect
DOW_EFFECT = np.array([0.95, 0.95, 0.98, 1.0, 1.10, 1.25, 1.15])   # Mon..Sun

PRODUCT_NAMES = ["kettle", "lamp", "backpack", "headphones", "blender", "keyboard",
                 "speaker", "drone", "mixer", "monitor", "router", "camera"]

POS_OPEN = ["Absolutely love it.", "Great purchase.", "Very happy with this.",
            "Works like a charm.", "Exceeded my expectations.", "Fantastic product.",
            "Really pleased overall.", "Would buy again."]
NEU_OPEN = ["It is okay.", "Does the job.", "Average product.", "Nothing special.",
            "Fine for the price.", "Decent but not amazing."]
NEG_OPEN = ["Very disappointed.", "Not worth the money.", "Would not recommend.",
            "Terrible experience.", "Really unhappy with this.", "Waste of money.",
            "Regret buying this."]
POS_QUAL = ["The build quality is excellent.", "It works perfectly.",
            "Looks flawless and feels solid.", "Finish is smooth and sturdy.",
            "No issues at all, runs great.", "Materials feel premium."]
NEU_QUAL = ["Build quality is acceptable.", "Works most of the time.",
            "Finish is a bit plain.", "A few minor marks but usable."]
NEG_QUAL = ["It arrived cracked and scratched.", "Stopped working after a week.",
            "The surface is covered in scratches.", "Broke on the first day.",
            "Visible defects and a flimsy build.", "Parts feel cheap and it keeps failing.",
            "Came damaged with dents."]
FILLER = ["Delivery was on time.", "Shipping took longer than expected.",
          "Packaging was fine.", "The box was a bit squashed.", "Price is reasonable.",
          "Customer service answered quickly.", "Color matches the photos.",
          "Size is as described.", "Instructions were clear."]


@dataclass
class RawData:
    sales: pd.DataFrame            # date, product, sales
    reviews: pd.DataFrame          # date, product, review_text, rating
    images: pd.DataFrame           # date, product, image_id
    image_arrays: dict             # image_id -> uint8 array (S, S, 3)
    latent: pd.DataFrame           # date, product, q_demand, q_text, q_image
    signal: bool


def _quality_path(rng: np.random.Generator, n: int, mu: float) -> np.ndarray:
    q = np.empty(n)
    x = 0.0
    for t in range(n):
        x = 0.97 * x + rng.normal(0, 0.03)
        q[t] = mu + x
    t = int(rng.integers(30, 90))
    while t < n:                                   # defect episodes: trapezoid dips
        dur = int(rng.integers(14, 36))
        depth = rng.uniform(0.25, 0.5)
        k = np.arange(dur)
        ramp = np.minimum(1.0, np.minimum(k + 1, dur - k) / 4.0)
        end = min(n, t + dur)
        q[t:end] -= depth * ramp[: end - t]
        t += dur + int(rng.integers(50, 130))
    return np.clip(q, 0.02, 0.98)


def _render_image(rng: np.random.Generator, rgb: tuple, q: float, size: int) -> np.ndarray:
    S = 64
    bg = 235 + rng.normal(0, 3, (S, S, 3))
    img = Image.fromarray(np.clip(bg, 0, 255).astype(np.uint8))
    d = ImageDraw.Draw(img)
    cx, cy = 32 + int(rng.integers(-3, 4)), 32 + int(rng.integers(-3, 4))
    d.ellipse([cx - 20, cy - 20, cx + 20, cy + 20], fill=rgb)
    n_def = int(np.clip(round((1 - q) * 14 + rng.normal(0, 1.0)), 0, 20))
    for _ in range(n_def):
        ang, r = rng.uniform(0, 2 * np.pi), rng.uniform(0, 17)
        x0, y0 = cx + r * np.cos(ang), cy + r * np.sin(ang)
        length, a2 = rng.uniform(4, 10), rng.uniform(0, 2 * np.pi)
        d.line([x0, y0, x0 + length * np.cos(a2), y0 + length * np.sin(a2)],
               fill=(30, 30, 30), width=int(rng.integers(1, 3)))
    return np.asarray(img.resize((size, size), Image.BILINEAR)).copy()


def _review_text(rng: np.random.Generator, s: float, name: str) -> tuple[str, int]:
    if s < 0.45:
        opening, qual = NEG_OPEN, NEG_QUAL
    elif s < 0.65:
        opening, qual = NEU_OPEN, NEU_QUAL
    else:
        opening, qual = POS_OPEN, POS_QUAL
    parts = [str(rng.choice(opening))]
    if rng.random() < 0.8:
        parts.append(str(rng.choice(qual)))
    for _ in range(int(rng.integers(0, 3))):
        parts.append(str(rng.choice(FILLER)))
    parts.append(f"This {name} is what I got.")
    rating = int(np.clip(round(1 + 4 * s + rng.normal(0, 0.3)), 1, 5))
    return " ".join(parts), rating


def generate(n_products: int = 6, n_days: int = 900, seed: int = 0, signal: bool = True,
             start: str = "2022-01-01", img_size: int = 32, image_every: int = 7,
             review_rate: float = 0.05) -> RawData:
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, periods=n_days, freq="D")
    doy = dates.dayofyear.values
    dow = dates.dayofweek.values
    t_idx = np.arange(n_days)

    sales_rows, review_rows, image_rows, latent_rows = [], [], [], []
    image_arrays: dict = {}
    img_id = 0

    for p in range(n_products):
        name = f"{PRODUCT_NAMES[p % len(PRODUCT_NAMES)]}_{p + 1}"
        mu = float(np.clip(rng.normal(0.75, 0.06), 0.55, 0.9))
        q_d = _quality_path(rng, n_days, mu)
        q_t, q_i = (q_d, q_d) if signal else (_quality_path(rng, n_days, mu),
                                              _quality_path(rng, n_days, mu))

        base = rng.uniform(30, 120)
        phase = rng.uniform(0, 2 * np.pi)
        trend = 1 + rng.uniform(-0.1, 0.25) * t_idx / n_days
        yearly = 1 + 0.2 * np.sin(2 * np.pi * doy / 365.25 + phase)
        promo = np.ones(n_days)
        t = 0
        while t < n_days:
            if rng.random() < 0.02:
                promo[t:t + 3] = 1.25
                t += 3
            t += 1
        q_lag = np.concatenate([np.full(LAG, q_d[0]), q_d[:-LAG]])
        lam = base * trend * DOW_EFFECT[dow] * yearly * promo * np.exp(GAMMA * (q_lag - mu))
        sales = rng.poisson(lam * rng.lognormal(0, 0.08, n_days))

        for d in range(n_days):
            sales_rows.append((dates[d], name, int(sales[d])))
            latent_rows.append((dates[d], name, q_d[d], q_t[d], q_i[d]))

        # reviews: volume follows (past) sales; sentiment follows q_text
        for d in range(n_days):
            prev = sales[max(0, d - 3)]
            for _ in range(int(rng.poisson(review_rate * prev))):
                s = float(np.clip(q_t[d] + rng.normal(0, 0.15), 0, 1))
                text, rating = _review_text(rng, s, name.split("_")[0])
                review_rows.append((dates[d], name, text, rating))

        # weekly QC photo reflecting q_image
        rgb = tuple(int(255 * c) for c in colorsys.hsv_to_rgb((p * 0.17) % 1.0, 0.6, 0.75))
        offset = int(rng.integers(0, image_every))
        for d in range(offset, n_days, image_every):
            iid = f"img_{img_id:06d}"
            img_id += 1
            image_arrays[iid] = _render_image(rng, rgb, float(q_i[d]), img_size)
            image_rows.append((dates[d], name, iid))

    return RawData(
        sales=pd.DataFrame(sales_rows, columns=["date", "product", "sales"]),
        reviews=pd.DataFrame(review_rows, columns=["date", "product", "review_text", "rating"]),
        images=pd.DataFrame(image_rows, columns=["date", "product", "image_id"]),
        image_arrays=image_arrays,
        latent=pd.DataFrame(latent_rows, columns=["date", "product", "q_demand", "q_text", "q_image"]),
        signal=signal,
    )


def images_to_zip_bytes(raw: RawData) -> bytes:
    """Package the QC photos as a zip with manifest.csv (date, product, file) + PNGs."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        rows = []
        for r in raw.images.itertuples(index=False):
            fname = f"{r.image_id}.png"
            b = io.BytesIO()
            Image.fromarray(raw.image_arrays[r.image_id]).save(b, format="PNG")
            z.writestr(fname, b.getvalue())
            rows.append((r.date.strftime("%Y-%m-%d"), r.product, fname))
        z.writestr("manifest.csv", pd.DataFrame(rows, columns=["date", "product", "file"]).to_csv(index=False))
    return buf.getvalue()


def write_files(raw: RawData, folder: str) -> dict:
    """Write sales.csv, reviews.csv and images.zip - the same formats the app accepts."""
    os.makedirs(folder, exist_ok=True)
    paths = {"sales": os.path.join(folder, "sales.csv"),
             "reviews": os.path.join(folder, "reviews.csv"),
             "images": os.path.join(folder, "images.zip")}
    raw.sales.to_csv(paths["sales"], index=False)
    raw.reviews.to_csv(paths["reviews"], index=False)
    with open(paths["images"], "wb") as f:
        f.write(images_to_zip_bytes(raw))
    return paths
