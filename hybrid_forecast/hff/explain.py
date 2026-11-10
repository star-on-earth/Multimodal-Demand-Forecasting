"""Retrieval-based explanation of a forecast.

This is retrieval-augmented, but the "generation" step is a plain template: no
language model is called, so it works offline and cannot invent facts. Everything
in the text is a number computed from the data or the model.

* Review impact model: a ridge regression from TF-IDF review text to the demand
  change that followed (next 7 days vs previous 7 days), fitted on reviews dated
  before the hold-out only. Its coefficients say which words precede drops or rises.
* Retrieval: the recent reviews for a product are ranked by that impact score, and
  the ones pointing the same way as the forecast are returned as evidence.
* Sensitivity: how much the forecast moves when the text or image input is removed.
  This is model sensitivity, not proof of cause.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from .forecast import FinalForecast
from .ingest import Bundle


@dataclass
class ReviewImpactModel:
    vec: object
    coef: np.ndarray
    terms: np.ndarray

    @classmethod
    def fit(cls, bundle: Bundle, prep, train_end_idx: int | None = None):
        """Returns None when there is too little text to learn from."""
        if prep.text_tools is None or bundle.reviews is None:
            return None
        end = prep.train_end_idx if train_end_idx is None else train_end_idx
        rv = bundle.reviews
        rv = rv[(rv["day_idx"] >= 6) & (rv["day_idx"] + 7 <= end)]
        if len(rv) < 100:
            return None
        S = prep.S
        p, d = rv["p_idx"].values, rv["day_idx"].values
        ahead = np.array([S[a, b + 1:b + 8].mean() for a, b in zip(p, d)])
        past = np.array([S[a, b - 6:b + 1].mean() for a, b in zip(p, d)])
        y = ahead - past
        y = (y - y.mean()) / (y.std() + 1e-9)
        vec = prep.text_tools.vec
        X = vec.transform(rv["text"].values)
        ridge = Ridge(alpha=5.0).fit(X, y)
        return cls(vec, ridge.coef_, vec.get_feature_names_out())

    def score(self, texts) -> np.ndarray:
        return self.vec.transform(list(texts)) @ self.coef

    def top_terms(self, k: int = 6):
        order = np.argsort(self.coef)
        return list(self.terms[order[:k]]), list(self.terms[order[::-1][:k]])


def explain(ff: FinalForecast, bundle: Bundle, impact: ReviewImpactModel | None, product: str, k: int = 3) -> dict:
    p = ff.products.index(product)
    H = len(ff.dates)
    fc_mean, recent = float(ff.forecast[p].mean()), float(ff.recent[p])
    change = (fc_mean - recent) / max(recent, 1e-9)
    unit = "review volume" if bundle.demand_is_proxy else "demand"
    lines = [f"{product}: the {ff.path} model expects {fc_mean:.1f} units/day of {unit} over the next {H} days, "
             f"{change:+.1%} versus the last 14 days ({recent:.1f}/day)."]

    sens = {}
    for name, hid in ff.hidden.items():
        sens[name] = (fc_mean - float(hid[p].mean())) / max(recent, 1e-9)
    if sens:
        parts = [f"removing the {n} input would shift the forecast by {v:+.1%}" for n, v in sens.items()]
        lines.append("Model sensitivity: " + "; ".join(parts) + ".")
    else:
        lines.append("Only sales history was available, so the forecast rests on sales patterns alone.")

    evidence = pd.DataFrame(columns=["date", "rating", "impact score", "review"])
    if impact is not None and bundle.reviews is not None:
        rv = bundle.reviews
        end = len(bundle.dates) - 1
        win = rv[(rv["p_idx"] == p) & (rv["day_idx"] > end - 14)]
        if len(win):
            sc = impact.score(win["text"].values)
            order = np.argsort(sc) if change < 0 else np.argsort(-sc)
            pick = win.iloc[order[:k]]
            evidence = pd.DataFrame({
                "date": bundle.dates[pick["day_idx"].values].date,
                "rating": pick["rating"].values if bundle.has_rating else np.nan,
                "impact score": sc[order[:k]],
                "review": pick["text"].values})
            direction = "downward" if change < 0 else "upward"
            lines.append(f"Retrieved {len(evidence)} of the last 14 days' reviews that the learned impact model "
                         f"associates most strongly with {direction} demand movement.")
        neg, pos = impact.top_terms()
        lines.append("Words that most often preceded a demand drop: " + ", ".join(neg) +
                     ". Words that preceded a rise: " + ", ".join(pos) + ".")
    elif bundle.text_available:
        lines.append("Not enough review history to learn which words matter, so no evidence was retrieved.")
    lines.append("This explanation is template-generated from the numbers above; it is not written by a language model.")
    return {"summary": " ".join(lines), "lines": lines, "evidence": evidence, "sensitivity": sens}
