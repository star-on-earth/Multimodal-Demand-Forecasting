import io
import zipfile

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from hff import ingest, synth
from hff.config import Config, effective_config
from .conftest import TINY


def avail(b):
    return {k: v.available for k, v in b.status.items()}


def test_all_three(full_bundle):
    assert avail(full_bundle) == {"sales": True, "text": True, "image": True}
    assert full_bundle.sales.shape == (3, 380)


def test_sales_only(sales_bundle):
    a = avail(sales_bundle)
    assert a["sales"] and not a["text"] and not a["image"]


def test_sales_plus_reviews_only(raw):
    b = ingest.build_bundle(raw.sales, raw.reviews, None, TINY)
    assert avail(b) == {"sales": True, "text": True, "image": False}


def test_sales_plus_images_only(raw):
    b = ingest.build_bundle(raw.sales, None, ingest.images_from_raw(raw, 32), TINY)
    assert avail(b) == {"sales": True, "text": False, "image": True}


def test_reviews_only_uses_proxy(raw):
    b = ingest.build_bundle(None, raw.reviews, None, TINY)
    assert b.demand_is_proxy and b.status["sales"].available
    assert any("proxy" in n.lower() or "stand" in n.lower() or "review count" in n.lower() for n in b.notes)


def test_nothing_gives_clear_error():
    with pytest.raises((ValueError, Exception)) as e:
        ingest.build_bundle(None, None, None, TINY)
    assert str(e.value)


def test_messy_column_names(raw):
    s = raw.sales.rename(columns={"date": "Order Date", "product": "Item", "sales": "Units Sold"})
    r = raw.reviews.rename(columns={"date": "Review Date", "product": "Item", "review_text": "Comments", "rating": "Stars"})
    b = ingest.build_bundle(s, r, None, TINY)
    assert b.status["text"].available and len(b.products) == 3


def test_reviews_without_rating_or_product(raw):
    r = raw.reviews.drop(columns=["rating", "product"])
    b = ingest.build_bundle(raw.sales[raw.sales["product"] == "kettle_1"], r, None, TINY)
    assert b.status["sales"].available
    assert not b.has_rating


def test_reviews_for_some_products_only(raw):
    r = raw.reviews[raw.reviews["product"] == raw.sales["product"].iloc[0]]
    b = ingest.build_bundle(raw.sales, r, None, TINY)
    assert b.status["sales"].available     # must not crash


def test_missing_days_are_imputed_and_flagged(raw):
    s = raw.sales.drop(raw.sales.sample(frac=0.1, random_state=0).index)
    b = ingest.build_bundle(s, None, None, TINY)
    assert not b.observed.all() and np.isfinite(b.sales).all()


def test_no_date_column(raw):
    s = raw.sales[raw.sales["product"] == "kettle_1"][["sales"]].reset_index(drop=True)
    b = ingest.build_bundle(s, None, None, TINY)
    assert b.synthetic_dates and b.n_days == len(s)


def test_too_short_history_explains():
    with pytest.raises(ValueError, match="Not enough history"):
        effective_config(Config(), 60, 1)


def _zip(files: dict) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for k, v in files.items():
            z.writestr(k, v)
    return buf.getvalue()


def _png(color=(200, 10, 10)):
    b = io.BytesIO()
    Image.new("RGB", (40, 40), color).save(b, format="PNG")
    return b.getvalue()


def test_zip_with_corrupt_file_and_filename_dates():
    data = _zip({"kettle_1_2022-01-05.png": _png(), "kettle_1_2022-01-12.png": _png(),
                 "broken_2022-01-20.png": b"not an image", "readme.txt": b"hi"})
    man, arr, notes = ingest.read_image_zip(data, 32)
    assert len(arr) == 2 and len(man) == 2 and any("unreadable" in n for n in notes)


def test_zip_without_dates_is_handled_not_crashing(raw):
    data = _zip({"a.png": _png(), "b.png": _png()})
    man, arr, notes = ingest.read_image_zip(data, 32)
    b = ingest.build_bundle(raw.sales, None, (man, arr), TINY)
    assert not b.status["image"].available and b.status["sales"].available


def test_images_too_sparse_are_dropped(raw):
    man, arr = ingest.images_from_raw(raw, 32)
    keep = man.iloc[:2]
    b = ingest.build_bundle(raw.sales, None, (keep, {k: arr[k] for k in keep["file"]}), TINY)
    assert not b.status["image"].available and b.status["image"].reason
