"""Run every approach on the same chronological hold-out and compare them.

Rows of the result table:
  baselines          mean, last-7-day mean, seasonal-naive
  classical ML       LightGBM on lag features
  Path A (staged)    frozen encoders -> LightGBM, for each modality subset
  Path B (joint)     one end-to-end network, for each modality subset
Modality subsets that the data cannot support are simply not run.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import Config, effective_config
from .fusion import FlatLGBPredictor, JointPredictor, StagedBackbone, StagedPredictor
from .ingest import Bundle
from .windows import Prepared, split_origins

MOD_LABEL = {(False, False): "sales", (True, False): "sales+text",
             (False, True): "sales+image", (True, True): "sales+text+image"}


# ---- metrics ---------------------------------------------------------------
def wape(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.abs(y - p).sum() / max(np.abs(y).sum(), 1e-9))


def wape_by_horizon(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    return np.abs(y - p).sum(0) / np.maximum(np.abs(y).sum(0), 1e-9)


def mase(y: np.ndarray, p: np.ndarray, p_idx: np.ndarray, naive_scale: np.ndarray) -> float:
    """Per-product MAE divided by that product's in-sample weekly-naive MAE, averaged over products."""
    vals = []
    for k in np.unique(p_idx):
        m = p_idx == k
        vals.append(np.abs(y[m] - p[m]).mean() / max(naive_scale[k], 1e-9))
    return float(np.mean(vals))


@dataclass
class BenchmarkResult:
    table: pd.DataFrame
    by_horizon: pd.DataFrame
    robustness: pd.DataFrame
    verdict: list
    notes: list
    cfg: Config
    meta: pd.DataFrame
    y_raw: np.ndarray
    preds: dict
    status: dict
    runtime_s: float
    epochs: dict = field(default_factory=dict)


def run_benchmark(bundle: Bundle, cfg: Config | None = None, progress=None) -> BenchmarkResult:
    t0 = time.time()
    cfg = effective_config(cfg or Config(), bundle.n_days, len(bundle.products))
    H, D = cfg.horizon, bundle.n_days
    say = progress or (lambda f, m: None)

    train_o, test_o, test_start = split_origins(D, cfg)
    prep = Prepared(bundle, cfg, test_start - 1)
    S_tr, S_te = prep.samples(train_o), prep.samples(test_o)
    use_text, use_image = prep.use_text, prep.use_image
    notes = list(bundle.notes)
    if bundle.text_available and not use_text:
        notes.append(f"Text modality switched off: {prep.text_reason}.")
    if not use_text and not use_image:
        notes.append("No review or image data is usable, so both fusion paths reduce to sales-only models.")
    notes.append(f"Hold-out: the last {cfg.test_days} days; {len(S_te)} forecast origins x {H} horizons; "
                 f"{len(S_tr)} training windows.")

    dims = dict(W=cfg.window, H=H, Tw=cfg.text_window, Ct=prep.text_dim, S=prep.img_size)
    scale = S_te.scale[:, None]
    y_raw = S_te.y * scale
    p_idx = S_te.meta["p_idx"].values
    naive_scale = np.array([np.abs(bundle.sales[p, 7:test_start] - bundle.sales[p, :test_start - 7]).mean()
                            for p in range(len(bundle.products))])

    results: dict = {}

    def add(name, family, mods, pred_scaled):
        r = results.setdefault(name, {"family": family, "mods": mods, "preds": []})
        r["preds"].append(np.asarray(pred_scaled) * scale)

    # baselines (deterministic)
    add("Mean baseline", "Baseline", "sales", np.ones_like(S_te.y))
    add("Last-7-day mean", "Baseline", "sales", np.repeat(S_te.level, H, axis=1))
    add("Seasonal-naive", "Baseline", "sales", S_te.snaive)

    subsets = [(False, False)]
    if use_text:
        subsets.append((True, False))
    if use_image:
        subsets.append((False, True))
    if use_text and use_image:
        subsets.append((True, True))
    full = (use_text, use_image)

    scenarios = [("all inputs", ())]
    if use_text:
        scenarios.append(("text hidden", ("text",)))
    if use_image:
        scenarios.append(("images hidden", ("image",)))
    if use_text and use_image:
        scenarios.append(("text and images hidden", ("text", "image")))
    robust = {}   # (path, scenario) -> [wape per seed]
    epochs = {}

    seeds = list(cfg.seeds)
    steps_per_seed = 1 + len(subsets) + 3 + len(subsets)
    total, done = steps_per_seed * len(seeds), 0

    def tick(msg):
        nonlocal done
        done += 1
        say(min(done / total, 0.99), msg)

    import keras
    for seed in seeds:
        flat = FlatLGBPredictor(seed).fit(S_tr)
        add("LightGBM lags only", "Classical ML", "sales", flat.predict(S_te))
        tick(f"seed {seed}: LightGBM on lags")

        for ut, ui in subsets:                                    # Path B
            jp = JointPredictor(dims, ut, ui, cfg, seed).fit(S_tr)
            name = f"Path B joint | {MOD_LABEL[(ut, ui)]}"
            add(name, "Path B (joint)", MOD_LABEL[(ut, ui)], jp.predict(S_te))
            epochs.setdefault(name, []).append(jp.epochs)
            if (ut, ui) == full and (ut or ui):
                for sc, hide in scenarios:
                    robust.setdefault(("Path B (joint)", sc), []).append(
                        wape(y_raw, jp.predict(S_te, hide) * scale))
            tick(f"seed {seed}: joint network ({MOD_LABEL[(ut, ui)]})")
            keras.backend.clear_session()

        bb = StagedBackbone(dims, cfg, use_text, use_image, seed)    # Path A
        bb.fit(S_tr)
        tick(f"seed {seed}: staged encoders")
        for ut, ui in subsets:
            sp = StagedPredictor(bb, ut, ui, cfg, seed).fit(S_tr)
            add(f"Path A staged | {MOD_LABEL[(ut, ui)]}", "Path A (staged)", MOD_LABEL[(ut, ui)], sp.predict(S_te))
            if (ut, ui) == full and (ut or ui):
                for sc, hide in scenarios:
                    robust.setdefault(("Path A (staged)", sc), []).append(
                        wape(y_raw, sp.predict(S_te, hide) * scale))
            tick(f"seed {seed}: stacker ({MOD_LABEL[(ut, ui)]})")
        keras.backend.clear_session()

    # ---- tables
    rows, hz = [], {}
    sn_w = wape(y_raw, results["Seasonal-naive"]["preds"][0])
    for name, r in results.items():
        ws = [wape(y_raw, p) for p in r["preds"]]
        ms = [mase(y_raw, p, p_idx, naive_scale) for p in r["preds"]]
        rows.append({"approach": name, "family": r["family"], "modalities": r["mods"],
                     "WAPE": float(np.mean(ws)), "WAPE_std": float(np.std(ws)) if len(ws) > 1 else np.nan,
                     "MASE": float(np.mean(ms)), "vs_seasonal_naive": 1 - float(np.mean(ws)) / sn_w})
        hz[name] = np.mean([wape_by_horizon(y_raw, p) for p in r["preds"]], axis=0)
    table = pd.DataFrame(rows).sort_values("WAPE").reset_index(drop=True)
    by_h = pd.DataFrame(hz, index=range(1, H + 1)).T
    by_h.columns = [f"h{h}" for h in by_h.columns]

    rob_rows = []
    for (path, sc), ws in robust.items():
        rob_rows.append({"path": path, "test inputs": sc, "WAPE": float(np.mean(ws)),
                         "WAPE_std": float(np.std(ws)) if len(ws) > 1 else np.nan})
    robustness = pd.DataFrame(rob_rows)

    res = BenchmarkResult(table=table, by_horizon=by_h, robustness=robustness, verdict=[], notes=notes, cfg=cfg,
                          meta=S_te.meta, y_raw=y_raw,
                          preds={k: np.stack(v["preds"]) for k, v in results.items()},
                          status=bundle.status, runtime_s=time.time() - t0, epochs=epochs)
    res.verdict = make_verdict(res, full)
    say(1.0, "done")
    return res


def make_verdict(res: BenchmarkResult, full: tuple) -> list:
    """Plain-language reading of the table. Refuses to call a winner inside the noise."""
    t = res.table.set_index("approach")
    n_seeds = len(res.cfg.seeds)
    out = []
    best = t["WAPE"].idxmin()
    out.append(f"Lowest error: {best} (WAPE {t.loc[best, 'WAPE']:.3f}, "
               f"{t.loc[best, 'vs_seasonal_naive']:+.1%} vs seasonal-naive).")

    def noise(*names):
        s = [t.loc[n, "WAPE_std"] for n in names if n in t.index and np.isfinite(t.loc[n, "WAPE_std"])]
        return float(np.sqrt(np.sum(np.square(s)))) if s else None

    def compare(a, b, what):
        if a not in t.index or b not in t.index:
            return
        d = t.loc[b, "WAPE"] - t.loc[a, "WAPE"]          # positive: a is better
        sd = noise(a, b)
        floor = 2 * sd if sd is not None else 0.01
        if abs(d) <= floor:
            how = "across seeds" if sd is not None else "(single seed; run more seeds to judge)"
            out.append(f"{what}: no clear difference {how} ({a}: {t.loc[a, 'WAPE']:.3f}, {b}: {t.loc[b, 'WAPE']:.3f}).")
        else:
            w, l = (a, b) if d > 0 else (b, a)
            out.append(f"{what}: {w} is better by {abs(d):.3f} WAPE ({t.loc[w, 'WAPE']:.3f} vs {t.loc[l, 'WAPE']:.3f}).")

    mods = MOD_LABEL[full]
    if full != (False, False):
        compare(f"Path B joint | {mods}", f"Path A staged | {mods}", "Joint vs staged fusion")
        compare(f"Path B joint | {mods}", "Path B joint | sales", "Do text/images help the joint network?")
        compare(f"Path A staged | {mods}", "Path A staged | sales", "Do text/images help the staged model?")
    compare("Path B joint | sales", "LightGBM lags only", "Neural (GRU+CNN) vs LightGBM on sales alone")
    if n_seeds == 1:
        out.append("Only one seed was run, so small gaps may be training noise.")
    return out
