# Synthetic data (control): 6 products x 900 days

Runtime: 5.2 min. Seeds: [0].

- Hold-out: the last 120 days; 642 forecast origins x 14 horizons; 4266 training windows.

## Comparison (lower WAPE is better)

| approach                         | family          | modalities       |   WAPE |   MASE | vs_seasonal_naive   |
|:---------------------------------|:----------------|:-----------------|-------:|-------:|:--------------------|
| LightGBM lags only               | Classical ML    | sales            |  0.234 |  1.121 | +15.7%              |
| Path A staged | sales+text       | Path A (staged) | sales+text       |  0.236 |  1.125 | +14.9%              |
| Path A staged | sales            | Path A (staged) | sales            |  0.236 |  1.125 | +14.9%              |
| Path A staged | sales+image      | Path A (staged) | sales+image      |  0.237 |  1.127 | +14.8%              |
| Path A staged | sales+text+image | Path A (staged) | sales+text+image |  0.239 |  1.14  | +13.9%              |
| Path B joint | sales+image       | Path B (joint)  | sales+image      |  0.245 |  1.175 | +11.9%              |
| Path B joint | sales             | Path B (joint)  | sales            |  0.247 |  1.18  | +11.1%              |
| Path B joint | sales+text        | Path B (joint)  | sales+text       |  0.249 |  1.186 | +10.2%              |
| Path B joint | sales+text+image  | Path B (joint)  | sales+text+image |  0.257 |  1.224 | +7.3%               |
| Last-7-day mean                  | Baseline        | sales            |  0.266 |  1.265 | +4.3%               |
| Seasonal-naive                   | Baseline        | sales            |  0.278 |  1.328 | +0.0%               |
| Mean baseline                    | Baseline        | sales            |  0.34  |  1.648 | -22.6%              |

## Robustness: model trained on all inputs, tested with inputs removed

| path            | test inputs            |   WAPE |
|:----------------|:-----------------------|-------:|
| Path B (joint)  | all inputs             |  0.257 |
| Path B (joint)  | text hidden            |  0.239 |
| Path B (joint)  | images hidden          |  0.259 |
| Path B (joint)  | text and images hidden |  0.24  |
| Path A (staged) | all inputs             |  0.239 |
| Path A (staged) | text hidden            |  0.236 |
| Path A (staged) | images hidden          |  0.239 |
| Path A (staged) | text and images hidden |  0.235 |

## Verdict

- Lowest error: LightGBM lags only (WAPE 0.234, +15.7% vs seasonal-naive).
- Joint vs staged fusion: Path A staged | sales+text+image is better by 0.018 WAPE (0.239 vs 0.257).
- Do text/images help the joint network?: Path B joint | sales is better by 0.010 WAPE (0.247 vs 0.257).
- Do text/images help the staged model?: no clear difference (single seed; run more seeds to judge) (Path A staged | sales+text+image: 0.239, Path A staged | sales: 0.236).
- Neural (GRU+CNN) vs LightGBM on sales alone: LightGBM lags only is better by 0.013 WAPE (0.234 vs 0.247).
- Only one seed was run, so small gaps may be training noise.
