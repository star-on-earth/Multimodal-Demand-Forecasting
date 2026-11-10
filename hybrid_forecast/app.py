"""Streamlit app: multimodal demand forecasting with two fusion strategies.

    streamlit run app.py

Every input except sales is optional. Whatever is missing is reported on the
"Data check" tab and the models run on what is there.
"""
from __future__ import annotations

import io
import os
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from hff import ingest, synth  # noqa: E402
from hff.benchmark import run_benchmark  # noqa: E402
from hff.config import Config  # noqa: E402
from hff.explain import ReviewImpactModel, explain  # noqa: E402
from hff.forecast import fit_forecast  # noqa: E402

st.set_page_config(page_title="Multimodal demand forecasting", layout="wide")

SPEEDS = {
    "Quick (about 2-4 min)": dict(epochs=12, seeds=(0,)),
    "Standard (about 8-15 min)": dict(epochs=40, seeds=(0,)),
    "Thorough, 3 seeds (30+ min)": dict(epochs=40, seeds=(0, 1, 2)),
}
FAMILY_COLOR = {"Baseline": "#9aa0a6", "Classical ML": "#5f7fa8", "Path A (staged)": "#d98e04", "Path B (joint)": "#1b8a6b"}


# --------------------------------------------------------------------------- data loading
def load_bundle(source: str, cfg: Config, n_products: int, n_days: int, files: dict):
    if source.startswith("Synthetic"):
        raw = synth.generate(n_products=n_products, n_days=n_days, seed=1, signal=source.endswith("signal"))
        return ingest.build_bundle(raw.sales, raw.reviews, ingest.images_from_raw(raw, cfg.img_size), cfg)
    sales = pd.read_csv(files["sales"]) if files["sales"] else None
    reviews = pd.read_csv(files["reviews"]) if files["reviews"] else None
    images = None
    notes = []
    if files["images"]:
        man, arr, notes = ingest.read_image_zip(files["images"].getvalue(), cfg.img_size)
        images = (man, arr)
    b = ingest.build_bundle(sales, reviews, images, cfg)
    b.notes = list(notes) + list(b.notes)
    return b


def status_table(bundle) -> pd.DataFrame:
    rows = []
    for k in ("sales", "text", "image"):
        s = bundle.status[k]
        rows.append({"input": {"sales": "Sales history", "text": "Review text", "image": "Product images"}[k],
                     "usable": "yes" if s.available else "no", "items": s.n_items,
                     "coverage": f"{s.coverage:.0%}" if s.available and k != "sales" else "",
                     "detail": s.reason})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- charts
def comparison_chart(table: pd.DataFrame):
    t = table.sort_values("WAPE", ascending=False)
    fig = go.Figure(go.Bar(
        x=t["WAPE"], y=t["approach"], orientation="h",
        marker_color=[FAMILY_COLOR.get(f, "#888") for f in t["family"]],
        error_x=dict(type="data", array=t["WAPE_std"].fillna(0).values, visible=bool(t["WAPE_std"].notna().any())),
        text=[f"{v:.3f}" for v in t["WAPE"]], textposition="outside"))
    fig.update_layout(height=60 + 28 * len(t), margin=dict(l=0, r=30, t=10, b=30),
                      xaxis_title="WAPE on the hold-out period (lower is better)", yaxis_title=None)
    return fig


def horizon_chart(by_h: pd.DataFrame, names: list):
    fig = go.Figure()
    for n in names:
        if n in by_h.index:
            fig.add_scatter(x=list(range(1, by_h.shape[1] + 1)), y=by_h.loc[n].values, mode="lines+markers", name=n)
    fig.update_layout(height=340, margin=dict(l=0, r=0, t=10, b=30),
                      xaxis_title="days ahead", yaxis_title="WAPE", legend=dict(orientation="h", y=-0.3))
    return fig


def forecast_chart(bundle, ff, product: str, hist_days: int = 90):
    p = ff.products.index(product)
    fig = go.Figure()
    fig.add_scatter(x=bundle.dates[-hist_days:], y=bundle.sales[p, -hist_days:], name="history", line=dict(color="#5f6368"))
    fig.add_scatter(x=ff.dates, y=ff.forecast[p], name=f"forecast ({ff.path})", line=dict(color="#1b8a6b", width=3))
    for k, v in ff.hidden.items():
        fig.add_scatter(x=ff.dates, y=v[p], name=f"forecast without {k}", line=dict(dash="dot"))
    fig.update_layout(height=380, margin=dict(l=0, r=0, t=10, b=30), yaxis_title="units per day",
                      legend=dict(orientation="h", y=-0.25))
    return fig


# --------------------------------------------------------------------------- sidebar
st.title("Multimodal demand forecasting")
st.caption("Sales history + review text + product images, fused two ways (staged and joint) and compared on a held-out period.")

with st.sidebar:
    st.header("1. Data")
    source = st.radio("Source", ["Synthetic - signal", "Synthetic - control (no signal in text/images)", "Upload my own"],
                      help="Synthetic data has a known structure. 'Control' has reviews and images that carry no information about demand, "
                           "so a sound model should show no gain from them.")
    files = {"sales": None, "reviews": None, "images": None}
    n_products, n_days = 6, 900
    if source.startswith("Synthetic"):
        n_products = st.slider("Products", 2, 12, 6)
        n_days = st.slider("Days of history", 400, 1200, 900, step=50)
        raw_demo = synth.generate(n_products=2, n_days=120, seed=3)
        st.download_button("Download a small example sales.csv", raw_demo.sales.to_csv(index=False), "sales.csv", "text/csv")
    else:
        st.caption("Only sales is needed. Reviews and images are optional, and may cover only some products or dates.")
        files["sales"] = st.file_uploader("Sales CSV (date, sales, optional product/store/item columns)", type="csv")
        files["reviews"] = st.file_uploader("Reviews CSV (date, review text, optional rating/product)", type="csv")
        files["images"] = st.file_uploader("Images ZIP (optional manifest.csv with date, product, file)", type="zip")
    st.header("2. Speed")
    speed = st.selectbox("Training effort", list(SPEEDS))
    go_btn = st.button("Load data and run comparison", type="primary")

_spd = dict(SPEEDS[speed])
if os.environ.get("HFF_EPOCHS"):              # lets automated tests run the app quickly
    _spd["epochs"] = int(os.environ["HFF_EPOCHS"])
cfg = Config(**_spd)

if go_btn:
    for k in ("bundle", "result", "forecast", "impact"):
        st.session_state.pop(k, None)
    try:
        with st.spinner("Reading and checking the data"):
            st.session_state["bundle"] = load_bundle(source, cfg, n_products, n_days, files)
        bundle = st.session_state["bundle"]
        bar = st.progress(0.0, text="Starting")
        st.session_state["result"] = run_benchmark(bundle, cfg, lambda f, m: bar.progress(min(f, 1.0), text=m))
        bar.empty()
    except ValueError as e:
        st.session_state.pop("result", None)
        st.error(str(e))

bundle = st.session_state.get("bundle")
result = st.session_state.get("result")
if bundle is None:
    st.info("Pick a data source on the left and press **Load data and run comparison**. "
            "The synthetic option needs no files.")
    st.stop()

tab1, tab2, tab3 = st.tabs(["Data check", "Comparison", "Forecast and explanation"])

# --------------------------------------------------------------------------- tab 1
with tab1:
    st.subheader("What the models can use")
    st.dataframe(status_table(bundle), hide_index=True, use_container_width=True)
    for n in bundle.notes:
        st.write("- " + n)
    if bundle.demand_is_proxy:
        st.warning("No sales were provided. Daily review count is being used as a stand-in for demand, so results describe review volume.")
    st.caption(f"{len(bundle.products)} product(s), {bundle.n_days} days: {bundle.dates[0].date()} to {bundle.dates[-1].date()}.")

# --------------------------------------------------------------------------- tab 2
with tab2:
    if result is None:
        st.warning("The comparison could not run; see the message above.")
    else:
        for n in result.notes:
            st.write("- " + n)
        st.subheader("Verdict")
        for v in result.verdict:
            st.write("- " + v)
        st.subheader("Error on the hold-out period")
        st.plotly_chart(comparison_chart(result.table), use_container_width=True)
        show = result.table.copy()
        show["vs_seasonal_naive"] = show["vs_seasonal_naive"].map("{:+.1%}".format)
        st.dataframe(show, hide_index=True, use_container_width=True)
        st.caption("WAPE = total absolute error / total actual sales. MASE below 1 beats a naive 'same weekday last week' forecast "
                   "scaled per product. 'vs_seasonal_naive' is the WAPE improvement over that naive forecast.")
        st.subheader("Error by forecast distance")
        paths = result.table[result.table["family"].str.startswith("Path")]
        longest = max(paths["modalities"], key=len) if len(paths) else ""
        default = list(paths.loc[paths["modalities"] == longest, "approach"])
        names = st.multiselect("Approaches", list(result.by_horizon.index), default=default or list(result.by_horizon.index)[:2])
        st.plotly_chart(horizon_chart(result.by_horizon, names), use_container_width=True)
        if len(result.robustness):
            st.subheader("What if an input goes missing at forecast time?")
            st.caption("Models trained with all inputs, then tested with some inputs removed. Small increases mean graceful degradation.")
            st.dataframe(result.robustness.assign(WAPE=result.robustness["WAPE"].round(3)).drop(columns=["WAPE_std"]),
                         hide_index=True, use_container_width=True)
        if source.startswith("Synthetic"):
            st.caption("Synthetic data: gains from text or images in 'signal' mode come from how the generator was written. "
                       "Run the 'control' source to confirm no gain appears when there is no signal.")

# --------------------------------------------------------------------------- tab 3
with tab3:
    if result is None:
        st.warning("Run the comparison first.")
    else:
        c1, c2 = st.columns([1, 2])
        path = c1.radio("Fusion path", ["B", "A"], format_func=lambda x: {"B": "Path B - joint network", "A": "Path A - staged encoders + LightGBM"}[x])
        if c1.button("Train on all history and forecast the next 14 days"):
            bar = st.progress(0.0, text="Training")
            try:
                ff = fit_forecast(bundle, cfg, path=path, progress=lambda f, m: bar.progress(min(f, 1.0), text=m))
                st.session_state["forecast"] = ff
                st.session_state["impact"] = ReviewImpactModel.fit(bundle, ff.prep, ff.prep.train_end_idx)
            except ValueError as e:
                st.error(str(e))
            bar.empty()
        ff = st.session_state.get("forecast")
        if ff is not None:
            product = c2.selectbox("Product", ff.products)
            st.plotly_chart(forecast_chart(bundle, ff, product), use_container_width=True)
            ex = explain(ff, bundle, st.session_state.get("impact"), product)
            st.subheader("Why this forecast")
            for line in ex["lines"]:
                st.write(line)
            if len(ex["evidence"]):
                st.markdown("**Recent reviews most associated with the forecast direction**")
                st.dataframe(ex["evidence"], hide_index=True, use_container_width=True)
            if len(ff.hidden):
                st.markdown("**Sensitivity of every product's forecast to missing inputs**")
                st.dataframe(ff.sensitivity().style.format({c: "{:+.1%}" for c in ff.sensitivity().columns if c != "product"}),
                             hide_index=True, use_container_width=True)
            buf = io.StringIO()
            ff.table().to_csv(buf, index=False)
            st.download_button("Download forecasts (CSV)", buf.getvalue(), "forecast.csv", "text/csv")
