"""Turn whatever the user uploads into a clean ``Bundle``.

Design rule: nothing optional may stop the pipeline. Every missing or unusable
piece is turned into (a) a status entry explaining why, and (b) a masked
modality the models can run without.

Only a dataset with *no* usable demand signal at all, or too little history,
raises ``ValueError`` - and the message says exactly what is needed.
"""
from __future__ import annotations

import io
import re
import warnings
import zipfile
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from PIL import Image

from .config import Config, effective_config

DATE_ALIASES = ["date", "order_date", "timestamp", "datetime", "day", "review_date",
                "posted", "created_at", "time", "ds"]
DEMAND_ALIASES = ["sales", "demand", "quantity", "units_sold", "units", "qty", "orders",
                  "volume", "sold"]
TEXT_ALIASES = ["review_text", "review", "text", "comment", "comments", "body", "content",
                "feedback", "review_body"]
RATING_ALIASES = ["rating", "stars", "star_rating", "score", "overall"]
GROUP_HINTS = ("store", "item", "product", "category", "sku", "series")
IMAGE_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".webp")
MAX_IMAGES = 20000
MAX_IMAGE_BYTES = 5_000_000


@dataclass
class ModalityStatus:
    name: str
    available: bool
    reason: str
    n_items: int = 0
    coverage: float = 0.0


@dataclass
class ImageSet:
    day_idx: np.ndarray            # sorted day indices
    arr: np.ndarray                # (n, S, S, 3) float32 in [0, 1]


@dataclass
class Bundle:
    dates: pd.DatetimeIndex
    products: list
    sales: np.ndarray              # (P, D) float64, gaps imputed
    observed: np.ndarray           # (P, D) bool, False where imputed
    reviews: pd.DataFrame | None   # p_idx, day_idx, text, rating (NaN if unknown)
    has_rating: bool
    images: dict                   # p_idx -> ImageSet
    status: dict                   # name -> ModalityStatus
    notes: list = field(default_factory=list)
    demand_is_proxy: bool = False
    synthetic_dates: bool = False

    @property
    def n_days(self) -> int:
        return len(self.dates)

    @property
    def text_available(self) -> bool:
        return self.status["text"].available

    @property
    def image_available(self) -> bool:
        return self.status["image"].available


# --------------------------------------------------------------------------- helpers
def _find(cols, aliases, exclude=()):
    low = {c.lower().strip(): c for c in cols if c not in exclude}
    for a in aliases:
        if a in low:
            return low[a]
    for a in aliases:
        if len(a) >= 4:
            for k, c in low.items():
                if a in k:
                    return c
    return None


def _parse_dates(s: pd.Series) -> pd.Series:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = pd.to_datetime(s, errors="coerce")
        if out.isna().mean() > 0.2:
            alt = pd.to_datetime(s, errors="coerce", format="mixed", dayfirst=False)
            if alt.isna().mean() < out.isna().mean():
                out = alt
    if getattr(out.dt, "tz", None) is not None:
        out = out.dt.tz_localize(None)
    return out.dt.normalize()


def _group_cols(df: pd.DataFrame, exclude) -> list:
    return [c for c in df.columns if c not in exclude and any(h in c.lower() for h in GROUP_HINTS)]


def _keys(df: pd.DataFrame, cols: list) -> pd.Series:
    if not cols:
        return pd.Series("ALL", index=df.index)
    s = df[cols[0]].astype(str).str.strip()
    for c in cols[1:]:
        s = s + " | " + df[c].astype(str).str.strip()
    return s


def _map_keys(other: pd.Series, known: list, notes: list, what: str):
    """Match another table's product ids onto the sales product keys."""
    known_set = set(known)
    if set(other.unique()) & known_set:
        keep = other.isin(known_set)
        if (~keep).any():
            notes.append(f"{what}: dropped {int((~keep).sum())} rows for products not in the sales data.")
        return other, keep
    if len(known) == 1:
        notes.append(f"{what}: no product column matched; assigning everything to '{known[0]}'.")
        return pd.Series(known[0], index=other.index), pd.Series(True, index=other.index)
    return None, None


def read_image_zip(data, size: int = 32):
    """Read a zip of images (+ optional manifest.csv) -> (manifest_df, {file: float32 array}, notes).

    Manifest columns: date, product (optional), file. Without a manifest, dates are
    parsed from file names such as ``kettle_2024-03-05.png``.
    """
    notes: list = []
    z = zipfile.ZipFile(io.BytesIO(data) if isinstance(data, (bytes, bytearray)) else data)
    infos = [i for i in z.infolist() if not i.is_dir()]
    arrays, bad = {}, 0
    for info in infos:
        base = info.filename.replace("\\", "/").split("/")[-1]
        if not base.lower().endswith(IMAGE_EXT) or info.file_size > MAX_IMAGE_BYTES:
            continue
        if len(arrays) >= MAX_IMAGES:
            notes.append(f"Images: stopped after {MAX_IMAGES} files.")
            break
        try:
            img = Image.open(io.BytesIO(z.read(info))).convert("RGB").resize((size, size), Image.BILINEAR)
            arrays[base] = np.asarray(img, dtype=np.float32) / 255.0
        except Exception:
            bad += 1
    if bad:
        notes.append(f"Images: skipped {bad} unreadable image file(s).")
    manifest = None
    for info in infos:
        if info.filename.lower().endswith("manifest.csv"):
            try:
                manifest = pd.read_csv(io.BytesIO(z.read(info)))
            except Exception:
                notes.append("Images: manifest.csv could not be parsed; falling back to file names.")
            break
    if manifest is None:
        rows = []
        for base in arrays:
            m = re.search(r"(\d{4}-\d{2}-\d{2})", base)
            if m:
                prod = re.sub(r"[_\-\s]*$", "", re.sub(r"^[_\-\s]*", "", base[: m.start()]))
                rows.append((m.group(1), prod or None, base))
        manifest = pd.DataFrame(rows, columns=["date", "product", "file"])
        if len(manifest):
            notes.append("Images: no manifest.csv found; dates read from file names.")
    return manifest, arrays, notes


def images_from_raw(raw, size: int = 32):
    """Adapter so synthetic data goes through the same path as an uploaded zip."""
    arrays = {}
    for k, a in raw.image_arrays.items():
        im = a if a.shape[0] == size else np.asarray(Image.fromarray(a).resize((size, size), Image.BILINEAR))
        arrays[k] = im.astype(np.float32) / 255.0
    manifest = raw.images.rename(columns={"image_id": "file"})[["date", "product", "file"]]
    return manifest, arrays


# --------------------------------------------------------------------------- main
def build_bundle(sales_df: pd.DataFrame | None = None,
                 reviews_df: pd.DataFrame | None = None,
                 images=None,
                 cfg: Config | None = None) -> Bundle:
    cfg = cfg or Config()
    notes: list = []
    status = {
        "sales": ModalityStatus("sales", False, "not provided"),
        "text": ModalityStatus("text", False, "no review file provided"),
        "image": ModalityStatus("image", False, "no image file provided"),
    }

    # ---- reviews: parse first, because they may have to stand in for demand
    rev = None
    if reviews_df is not None and len(reviews_df):
        rev = _prepare_reviews(reviews_df, status, notes)

    # ---- sales
    have_sales = False
    demand_is_proxy = False
    synthetic_dates = False
    series = None            # DataFrame: key, date, y
    transactional = False
    if sales_df is not None and len(sales_df):
        series, transactional = _prepare_sales(sales_df, status, notes)
        have_sales = series is not None
        synthetic_dates = status["sales"].reason.startswith("no date column")
    if not have_sales:
        if rev is not None and len(rev):
            counts = rev.groupby(["_key", "_date"]).size().rename("y").reset_index()
            counts.columns = ["key", "date", "y"]
            series, transactional = counts, True
            demand_is_proxy = True
            status["sales"] = ModalityStatus("sales", True, "no demand column: daily review count used as a proxy for demand",
                                             len(counts))
            notes.append("No usable demand column was found, so the number of reviews per day is used as a "
                         "proxy for demand. Treat the forecast as 'review volume', not sales.")
        else:
            raise ValueError("No usable demand data. Upload a sales file with a date column and a numeric demand "
                             "column (e.g. date, product, sales), or a reviews file with dates so review volume "
                             "can stand in for demand.")

    # ---- grid and per-product arrays
    series = series.dropna(subset=["date", "y"])
    keys_by_volume = series.groupby("key")["y"].sum().sort_values(ascending=False)
    keep_keys = list(keys_by_volume.index[: cfg.max_products])
    if len(keys_by_volume) > cfg.max_products:
        notes.append(f"Using the {cfg.max_products} highest-volume series out of {len(keys_by_volume)} "
                     f"(raise Config.max_products to include more).")
    series = series[series["key"].isin(keep_keys)]
    d0, d1 = series["date"].min(), series["date"].max()
    dates = pd.date_range(d0, d1, freq="D")
    piv = (series.groupby(["date", "key"])["y"].sum().unstack("key")
           .reindex(dates)[keep_keys])
    if transactional:
        observed = np.ones(piv.shape, dtype=bool)
        filled = piv.fillna(0.0)
    else:
        observed = piv.notna().values
        filled = piv.interpolate(limit=7, limit_area="inside")
        med = piv.median()
        filled = filled.fillna(med).fillna(0.0)
        n_imputed = int((~observed).sum())
        if n_imputed:
            notes.append(f"{n_imputed} missing product-days were filled (linear interpolation up to 7 days, "
                         f"otherwise the product median) and flagged to the model as imputed.")
    sales = filled.values.T.astype(np.float64)          # (P, D)
    observed = observed.T
    neg = int((sales < 0).sum())
    if neg:
        sales = np.clip(sales, 0, None)
        notes.append(f"{neg} negative demand values were clipped to 0.")

    # drop products with almost no real observations
    n_obs = observed.sum(axis=1)
    ok = n_obs >= 60
    if not ok.all():
        notes.append(f"Dropped {int((~ok).sum())} product(s) with fewer than 60 observed days.")
    if not ok.any():
        raise ValueError(f"No product has at least 60 days of data (longest has {int(n_obs.max())}).")
    products = [k for k, o in zip(keep_keys, ok) if o]
    sales, observed = sales[ok], observed[ok]
    if status["sales"].reason == "not provided":
        status["sales"] = ModalityStatus("sales", True, "ok", int(observed.sum()), float(observed.mean()))
    else:
        status["sales"].n_items = int(observed.sum())
        status["sales"].coverage = float(observed.mean())

    effective_config(cfg, len(dates), len(products))      # fail fast with a clear message

    bundle = Bundle(dates=dates, products=products, sales=sales, observed=observed,
                    reviews=None, has_rating=False, images={}, status=status, notes=notes,
                    demand_is_proxy=demand_is_proxy, synthetic_dates=synthetic_dates)

    _attach_reviews(bundle, rev, cfg)
    _attach_images(bundle, images, sales_df if have_sales and not demand_is_proxy else None, cfg)
    return bundle


# --------------------------------------------------------------------------- sales
def _prepare_sales(df: pd.DataFrame, status: dict, notes: list):
    df = df.copy()
    date_col = _find(df.columns, DATE_ALIASES)
    demand_col = _find(df.columns, DEMAND_ALIASES, exclude=[date_col] if date_col else [])
    if demand_col is None:
        num = [c for c in df.columns if c != date_col and pd.api.types.is_numeric_dtype(df[c])
               and not any(h in c.lower() for h in GROUP_HINTS) and not c.lower().endswith("id")]
        if len(num) == 1:
            demand_col = num[0]
            notes.append(f"No standard demand column name found; using the only numeric column '{demand_col}'.")
    if demand_col is None:
        status["sales"] = ModalityStatus("sales", False, "no demand column found")
        return None, False
    if date_col is None:
        df["_date"] = pd.date_range("2020-01-01", periods=len(df), freq="D")
        status["sales"] = ModalityStatus("sales", True, "no date column: rows treated as consecutive days")
        notes.append("No date column was found, so rows are treated as consecutive days starting 2020-01-01 "
                     "(one product only). Calendar effects will be meaningless.")
        gcols = []
    else:
        df["_date"] = _parse_dates(df[date_col])
        gcols = _group_cols(df, exclude=[date_col, demand_col])
    df["_y"] = pd.to_numeric(df[demand_col], errors="coerce")
    bad_date, bad_y = int(df["_date"].isna().sum()), int(df["_y"].isna().sum())
    if bad_date:
        notes.append(f"{bad_date} rows with unreadable dates were dropped.")
    if bad_y:
        notes.append(f"{bad_y} rows with non-numeric demand were dropped.")
    df["_key"] = _keys(df, gcols)
    df = df.dropna(subset=["_date", "_y"])
    if df.empty:
        status["sales"] = ModalityStatus("sales", False, "no valid rows after cleaning")
        return None, False
    per_day = df.groupby(["_key", "_date"]).size()
    transactional = bool((per_day > 1).mean() > 0.2)
    out = df.groupby(["_key", "_date"])["_y"].sum().reset_index()
    out.columns = ["key", "date", "y"]
    return out, transactional


# --------------------------------------------------------------------------- reviews
def _prepare_reviews(df: pd.DataFrame, status: dict, notes: list):
    df = df.copy()
    text_col = _find(df.columns, TEXT_ALIASES)
    date_col = _find(df.columns, DATE_ALIASES)
    if text_col is None:
        status["text"] = ModalityStatus("text", False, "reviews file has no text column")
        return None
    if date_col is None:
        status["text"] = ModalityStatus("text", False, "reviews file has no date column (cannot align to sales)")
        return None
    rating_col = _find(df.columns, RATING_ALIASES, exclude=[text_col, date_col])
    gcols = _group_cols(df, exclude=[text_col, date_col] + ([rating_col] if rating_col else []))
    out = pd.DataFrame({
        "_date": _parse_dates(df[date_col]),
        "_text": df[text_col].astype(str).str.strip(),
        "_rating": pd.to_numeric(df[rating_col], errors="coerce") if rating_col else np.nan,
        "_key": _keys(df, gcols),
    })
    n0 = len(out)
    out = out[out["_date"].notna() & (out["_text"].str.len() > 0) & (out["_text"].str.lower() != "nan")]
    if len(out) < n0:
        notes.append(f"Reviews: {n0 - len(out)} rows with missing text or unreadable dates were dropped.")
    if out.empty:
        status["text"] = ModalityStatus("text", False, "no usable review rows")
        return None
    return out


def _attach_reviews(b: Bundle, rev: pd.DataFrame | None, cfg: Config) -> None:
    if rev is None:
        return
    keys, keep = _map_keys(rev["_key"], b.products, b.notes, "Reviews")
    if keys is None:
        b.status["text"] = ModalityStatus("text", False,
                                          "could not match review products to the sales products "
                                          "(use the same product column values)")
        return
    rev = rev[keep.values].copy()
    rev["_key"] = keys[keep].values
    pidx = {k: i for i, k in enumerate(b.products)}
    rev["p_idx"] = rev["_key"].map(pidx)
    rev["day_idx"] = (rev["_date"] - b.dates[0]).dt.days
    rev = rev[(rev["day_idx"] >= 0) & (rev["day_idx"] < b.n_days)]
    if len(rev) < cfg.min_review_count:
        b.status["text"] = ModalityStatus("text", False,
                                          f"only {len(rev)} usable reviews (need {cfg.min_review_count})", len(rev))
        return
    counts = np.zeros((len(b.products), b.n_days))
    np.add.at(counts, (rev["p_idx"].values, rev["day_idx"].values), 1)
    cov = float((counts > 0).mean())
    if cov < cfg.min_modality_coverage:
        b.status["text"] = ModalityStatus("text", False,
                                          f"reviews cover only {cov:.1%} of product-days", len(rev), cov)
        return
    b.reviews = pd.DataFrame({"p_idx": rev["p_idx"].astype(int).values,
                              "day_idx": rev["day_idx"].astype(int).values,
                              "text": rev["_text"].values,
                              "rating": rev["_rating"].values})
    b.has_rating = bool(np.isfinite(b.reviews["rating"]).mean() > 0.5)
    why = "ok" if b.has_rating else "ok (no rating column: text only)"
    b.status["text"] = ModalityStatus("text", True, why, len(rev), cov)


# --------------------------------------------------------------------------- images
def _attach_images(b: Bundle, images, sales_df, cfg: Config) -> None:
    if images is None:
        return
    manifest, arrays = images[0], images[1]
    if manifest is None or not len(manifest) or not arrays:
        b.status["image"] = ModalityStatus("image", False, "no readable images found")
        return
    date_col = _find(manifest.columns, DATE_ALIASES)
    file_col = _find(manifest.columns, ["file", "filename", "image", "image_id", "path"])
    if date_col is None or file_col is None:
        b.status["image"] = ModalityStatus("image", False, "image manifest needs date and file columns")
        return
    gcols = _group_cols(manifest, exclude=[date_col, file_col])
    m = pd.DataFrame({"_date": _parse_dates(manifest[date_col]),
                      "_file": manifest[file_col].astype(str).str.split("/").str[-1],
                      "_key": _keys(manifest, gcols)})
    m = m.dropna(subset=["_date"])
    m = m[m["_file"].isin(arrays.keys())]
    if m.empty:
        b.status["image"] = ModalityStatus("image", False, "manifest rows do not match any image file")
        return
    keys, keep = _map_keys(m["_key"], b.products, b.notes, "Images")
    if keys is None:
        b.status["image"] = ModalityStatus("image", False,
                                           "could not match image products to the sales products "
                                           "(add a product column to the manifest)")
        return
    m = m[keep.values].copy()
    m["_key"] = keys[keep].values
    pidx = {k: i for i, k in enumerate(b.products)}
    m["p_idx"] = m["_key"].map(pidx)
    m["day_idx"] = (m["_date"] - b.dates[0]).dt.days
    m = m[(m["day_idx"] >= 0) & (m["day_idx"] < b.n_days)].sort_values("day_idx")
    covered = np.zeros((len(b.products), b.n_days), dtype=bool)
    for p, g in m.groupby("p_idx"):
        d = g["day_idx"].values
        b.images[int(p)] = ImageSet(day_idx=d.astype(int),
                                    arr=np.stack([arrays[f] for f in g["_file"]]).astype(np.float32))
        last = -10**9
        di = 0
        for day in range(b.n_days):
            while di < len(d) and d[di] <= day:
                last = d[di]
                di += 1
            covered[int(p), day] = (day - last) <= cfg.max_img_age
    cov = float(covered.mean())
    n_img = int(len(m))
    if cov < cfg.min_modality_coverage or n_img < 3:
        b.images = {}
        b.status["image"] = ModalityStatus("image", False,
                                           f"images cover only {cov:.1%} of product-days", n_img, cov)
        return
    b.status["image"] = ModalityStatus("image", True, "ok", n_img, cov)
