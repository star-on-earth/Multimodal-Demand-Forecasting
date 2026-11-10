# Synthetic data (signal): 6 products x 900 days

Runtime: 8.6 min. Seeds: [0, 1, 2].

- Hold-out: the last 120 days; 642 forecast origins x 14 horizons; 4266 training windows.

## Comparison (lower WAPE is better)

| approach                         | family          | modalities       | WAPE          |   MASE | vs_seasonal_naive   |
|:---------------------------------|:----------------|:-----------------|:--------------|-------:|:--------------------|
| Path A staged | sales+text+image | Path A (staged) | sales+text+image | 0.179 ± 0.003 |  0.869 | +23.3%              |
| Path B joint | sales+text        | Path B (joint)  | sales+text       | 0.181 ± 0.002 |  0.876 | +22.5%              |
| Path B joint | sales+text+image  | Path B (joint)  | sales+text+image | 0.184 ± 0.005 |  0.89  | +21.3%              |
| Path A staged | sales+text       | Path A (staged) | sales+text       | 0.185 ± 0.003 |  0.893 | +20.8%              |
| Path A staged | sales+image      | Path A (staged) | sales+image      | 0.202 ± 0.001 |  0.975 | +13.8%              |
| Path A staged | sales            | Path A (staged) | sales            | 0.202 ± 0.002 |  0.976 | +13.5%              |
| Path B joint | sales+image       | Path B (joint)  | sales+image      | 0.204 ± 0.007 |  0.981 | +12.8%              |
| Path B joint | sales             | Path B (joint)  | sales            | 0.208 ± 0.009 |  0.997 | +11.3%              |
| Last-7-day mean                  | Baseline        | sales            | 0.209         |  1.016 | +10.8%              |
| LightGBM lags only               | Classical ML    | sales            | 0.212 ± 0.003 |  1.003 | +9.3%               |
| Seasonal-naive                   | Baseline        | sales            | 0.234         |  1.134 | +0.0%               |
| Mean baseline                    | Baseline        | sales            | 0.284         |  1.344 | -21.2%              |

## Robustness: model trained on all inputs, tested with inputs removed

| path            | test inputs            | WAPE          |
|:----------------|:-----------------------|:--------------|
| Path B (joint)  | all inputs             | 0.184 ± 0.005 |
| Path B (joint)  | text hidden            | 0.202 ± 0.004 |
| Path B (joint)  | images hidden          | 0.184 ± 0.007 |
| Path B (joint)  | text and images hidden | 0.203 ± 0.005 |
| Path A (staged) | all inputs             | 0.179 ± 0.003 |
| Path A (staged) | text hidden            | 0.198 ± 0.001 |
| Path A (staged) | images hidden          | 0.184 ± 0.001 |
| Path A (staged) | text and images hidden | 0.202 ± 0.001 |

## Verdict

- Lowest error: Path A staged | sales+text+image (WAPE 0.179, +23.3% vs seasonal-naive).
- Joint vs staged fusion: no clear difference across seeds (Path B joint | sales+text+image: 0.184, Path A staged | sales+text+image: 0.179).
- Do text/images help the joint network?: Path B joint | sales+text+image is better by 0.023 WAPE (0.184 vs 0.208).
- Do text/images help the staged model?: Path A staged | sales+text+image is better by 0.023 WAPE (0.179 vs 0.202).
- Neural (GRU+CNN) vs LightGBM on sales alone: no clear difference across seeds (Path B joint | sales: 0.208, LightGBM lags only: 0.212).
