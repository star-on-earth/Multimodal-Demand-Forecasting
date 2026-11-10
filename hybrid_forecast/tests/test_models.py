import numpy as np
import pytest

from hff.benchmark import run_benchmark, wape
from hff.config import effective_config
from hff.explain import ReviewImpactModel, explain
from hff.forecast import fit_forecast
from hff.fusion import JointPredictor
from hff.windows import Prepared, split_origins
from .conftest import TINY


def test_joint_model_masked_inference(full_bundle):
    cfg = effective_config(TINY, full_bundle.n_days, 3)
    tr, te, ts = split_origins(full_bundle.n_days, cfg)
    prep = Prepared(full_bundle, cfg, ts - 1)
    S_tr, S_te = prep.samples(tr), prep.samples(te)
    dims = dict(W=cfg.window, H=cfg.horizon, Tw=cfg.text_window, Ct=prep.text_dim, S=prep.img_size)
    jp = JointPredictor(dims, True, True, cfg, 0).fit(S_tr)
    full = jp.predict(S_te)
    no_text = jp.predict(S_te, ("text",))
    no_both = jp.predict(S_te, ("text", "image"))
    for p in (full, no_text, no_both):
        assert p.shape == S_te.y.shape and np.isfinite(p).all()
    assert not np.allclose(full, no_text)       # the model actually uses the text input


@pytest.mark.parametrize("which", ["full", "sales"])
def test_benchmark_runs(which, full_bundle, sales_bundle):
    b = full_bundle if which == "full" else sales_bundle
    res = run_benchmark(b, TINY)
    names = list(res.table["approach"])
    assert "Seasonal-naive" in names and "LightGBM lags only" in names
    assert any(n.startswith("Path A") for n in names) and any(n.startswith("Path B") for n in names)
    if which == "sales":
        assert not any("text" in n or "image" in n for n in names)
    else:
        assert "Path B joint | sales+text+image" in names and "Path A staged | sales+text+image" in names
        assert len(res.robustness) >= 6
    assert res.table["WAPE"].between(0, 5).all() and res.verdict
    assert wape(res.y_raw, res.preds["Seasonal-naive"][0]) == pytest.approx(
        res.table.set_index("approach").loc["Seasonal-naive", "WAPE"])


@pytest.mark.parametrize("path", ["B", "A"])
def test_forecast_and_explain(path, full_bundle):
    ff = fit_forecast(full_bundle, TINY, path=path)
    assert ff.forecast.shape == (3, 14) and (ff.forecast >= 0).all() and np.isfinite(ff.forecast).all()
    assert set(ff.hidden) == {"text", "image"}
    impact = ReviewImpactModel.fit(full_bundle, ff.prep, ff.prep.train_end_idx)
    ex = explain(ff, full_bundle, impact, ff.products[0])
    assert ex["summary"] and "template" in ex["summary"]
    assert len(ex["evidence"]) <= 3
    assert not ff.sensitivity().empty


def test_forecast_sales_only(sales_bundle):
    ff = fit_forecast(sales_bundle, TINY, path="A")
    assert ff.hidden == {} and ff.forecast.shape[1] == 14
    ex = explain(ff, sales_bundle, None, ff.products[0])
    assert "Only sales history" in ex["summary"] and ex["evidence"].empty
