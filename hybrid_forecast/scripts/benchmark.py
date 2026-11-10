"""Command-line benchmark.

    python scripts/benchmark.py --mode signal --products 6 --days 900 --seeds 3 --out results/signal
    python scripts/benchmark.py --mode control ...
    python scripts/benchmark.py --sales my_sales.csv --reviews my_reviews.csv --images my_images.zip

Writes <out>.csv (comparison table), <out>_robustness.csv and <out>.md (readable summary).
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")

import pandas as pd  # noqa: E402

from hff import ingest, synth  # noqa: E402
from hff.benchmark import run_benchmark  # noqa: E402
from hff.config import Config  # noqa: E402


def to_markdown(res, title: str) -> str:
    t = res.table.copy()
    t["WAPE"] = t.apply(lambda r: f"{r.WAPE:.3f}" + (f" ± {r.WAPE_std:.3f}" if pd.notna(r.WAPE_std) else ""), axis=1)
    t["MASE"] = t["MASE"].map("{:.3f}".format)
    t["vs_seasonal_naive"] = t["vs_seasonal_naive"].map("{:+.1%}".format)
    t = t.drop(columns=["WAPE_std"])
    lines = [f"# {title}", "", f"Runtime: {res.runtime_s / 60:.1f} min. Seeds: {list(res.cfg.seeds)}.", ""]
    lines += [f"- {n}" for n in res.notes] + ["", "## Comparison (lower WAPE is better)", ""]
    lines += [t.to_markdown(index=False)] if hasattr(t, "to_markdown") else [t.to_string(index=False)]
    r = res.robustness.copy()
    if len(r):
        r["WAPE"] = r.apply(lambda x: f"{x.WAPE:.3f}" + (f" ± {x.WAPE_std:.3f}" if pd.notna(x.WAPE_std) else ""), axis=1)
        r = r.drop(columns=["WAPE_std"])
        lines += ["", "## Robustness: model trained on all inputs, tested with inputs removed", ""]
        lines += [r.to_markdown(index=False)] if hasattr(r, "to_markdown") else [r.to_string(index=False)]
    lines += ["", "## Verdict", ""] + [f"- {v}" for v in res.verdict]
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["signal", "control"], default="signal")
    ap.add_argument("--products", type=int, default=6)
    ap.add_argument("--days", type=int, default=900)
    ap.add_argument("--seed-data", type=int, default=1)
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--test-days", type=int, default=120)
    ap.add_argument("--horizon", type=int, default=14)
    ap.add_argument("--sales"), ap.add_argument("--reviews"), ap.add_argument("--images")
    ap.add_argument("--out", default="results/run")
    a = ap.parse_args()

    cfg = Config(epochs=a.epochs, seeds=tuple(range(a.seeds)), test_days=a.test_days, horizon=a.horizon)
    if a.sales or a.reviews or a.images:
        sales = pd.read_csv(a.sales) if a.sales else None
        reviews = pd.read_csv(a.reviews) if a.reviews else None
        images = None
        if a.images:
            man, arr, _ = ingest.read_image_zip(open(a.images, "rb").read(), cfg.img_size)
            images = (man, arr)
        bundle = ingest.build_bundle(sales, reviews, images, cfg)
        title = "Uploaded data"
    else:
        raw = synth.generate(n_products=a.products, n_days=a.days, seed=a.seed_data, signal=(a.mode == "signal"))
        bundle = ingest.build_bundle(raw.sales, raw.reviews, ingest.images_from_raw(raw, cfg.img_size), cfg)
        title = f"Synthetic data ({a.mode}): {a.products} products x {a.days} days"

    last = [-1]

    def progress(f, m):
        pct = int(f * 100)
        if pct != last[0]:
            last[0] = pct
            print(f"[{pct:3d}%] {m}", flush=True)

    res = run_benchmark(bundle, cfg, progress)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    res.table.to_csv(a.out + ".csv", index=False)
    res.robustness.to_csv(a.out + "_robustness.csv", index=False)
    with open(a.out + ".md", "w") as f:
        f.write(to_markdown(res, title))
    print(open(a.out + ".md").read())


if __name__ == "__main__":
    main()
