# Synthetic data (signal): 6 products x 900 days

Runtime: 2.4 min. Seeds: [0].

- Hold-out: the last 120 days; 642 forecast origins x 14 horizons; 4266 training windows.

## Comparison (lower WAPE is better)

| approach                         | family          | modalities       |   WAPE |   MASE | vs_seasonal_naive   |
|:---------------------------------|:----------------|:-----------------|-------:|-------:|:--------------------|
| Path B joint | sales+text+image  | Path B (joint)  | sales+text+image |  0.181 |  0.874 | +22.7%              |
| Path A staged | sales+text       | Path A (staged) | sales+text       |  0.183 |  0.881 | +22.0%              |
| Path B joint | sales+text        | Path B (joint)  | sales+text       |  0.183 |  0.884 | +21.9%              |
| Path A staged | sales+text+image | Path A (staged) | sales+text+image |  0.184 |  0.883 | +21.6%              |
| Path B joint | sales             | Path B (joint)  | sales            |  0.196 |  0.953 | +16.1%              |
| Path B joint | sales+image       | Path B (joint)  | sales+image      |  0.197 |  0.962 | +15.6%              |
| Path A staged | sales            | Path A (staged) | sales            |  0.2   |  0.967 | +14.5%              |
| Path A staged | sales+image      | Path A (staged) | sales+image      |  0.203 |  0.982 | +13.1%              |
| Last-7-day mean                  | Baseline        | sales            |  0.209 |  1.016 | +10.8%              |
| LightGBM lags only               | Classical ML    | sales            |  0.21  |  0.994 | +10.2%              |
| Seasonal-naive                   | Baseline        | sales            |  0.234 |  1.134 | +0.0%               |
| Mean baseline                    | Baseline        | sales            |  0.284 |  1.344 | -21.2%              |

## Robustness: model trained on all inputs, tested with inputs removed

| path            | test inputs            |   WAPE |
|:----------------|:-----------------------|-------:|
| Path B (joint)  | all inputs             |  0.181 |
| Path B (joint)  | text hidden            |  0.205 |
| Path B (joint)  | images hidden          |  0.181 |
| Path B (joint)  | text and images hidden |  0.205 |
| Path A (staged) | all inputs             |  0.184 |
| Path A (staged) | text hidden            |  0.2   |
| Path A (staged) | images hidden          |  0.184 |
| Path A (staged) | text and images hidden |  0.2   |

## Verdict

- Lowest error: Path B joint | sales+text+image (WAPE 0.181, +22.7% vs seasonal-naive).
- Joint vs staged fusion: no clear difference (single seed; run more seeds to judge) (Path B joint | sales+text+image: 0.181, Path A staged | sales+text+image: 0.184).
- Do text/images help the joint network?: Path B joint | sales+text+image is better by 0.015 WAPE (0.181 vs 0.196).
- Do text/images help the staged model?: Path A staged | sales+text+image is better by 0.017 WAPE (0.184 vs 0.200).
- Neural (GRU+CNN) vs LightGBM on sales alone: Path B joint | sales is better by 0.014 WAPE (0.196 vs 0.210).
- Only one seed was run, so small gaps may be training noise.
