import numpy as np
import pandas as pd

from hff import synth


def test_deterministic():
    a = synth.generate(n_products=2, n_days=100, seed=3)
    b = synth.generate(n_products=2, n_days=100, seed=3)
    pd.testing.assert_frame_equal(a.sales, b.sales)
    pd.testing.assert_frame_equal(a.reviews, b.reviews)


def _lead_corr(raw, lag=synth.LAG):
    """Correlation between review rating on day t and demand residual on day t+lag."""
    cs = []
    for prod, g in raw.sales.groupby("product"):
        s = g.set_index("date")["sales"].astype(float)
        resid = s - s.rolling(28, center=True, min_periods=7).mean()
        r = raw.reviews[raw.reviews["product"] == prod].groupby("date")["rating"].mean().reindex(s.index)
        cs.append(r.corr(resid.shift(-lag)))
    return float(np.nanmean(cs))


def test_signal_vs_control():
    sig = synth.generate(n_products=4, n_days=700, seed=2, signal=True)
    ctl = synth.generate(n_products=4, n_days=700, seed=2, signal=False)
    cs, cc = _lead_corr(sig), _lead_corr(ctl)
    assert cs > 0.15, cs
    assert abs(cc) < 0.1, cc
    assert cs > cc + 0.1


def test_images_and_reviews_exist(raw):
    assert len(raw.reviews) > 500 and len(raw.images) > 30
    assert set(raw.images["image_id"]) == set(raw.image_arrays)
    assert (raw.sales["sales"] >= 0).all()
