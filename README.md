# Multimodal demand forecasting: CNN + GRU + NLP + RAG, two fusion strategies

Forecasts the next 14 days of demand per product from three kinds of input, and compares two ways of combining them.

| Input | Model | Required? |
|---|---|---|
| Sales history | GRU + 1D-CNN (and LightGBM lag features) | **Yes** (or reviews alone, see below) |
| Review text | TF-IDF -> SVD (LSA) embedding, daily per product | No |
| Product images | small CNN trained from scratch on 32x32 images | No |
| Explanation | RAG-style retrieval of recent reviews + template text | Only if reviews exist |

## Run it

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py                                    # the app
python scripts/benchmark.py --mode signal --seeds 3 --out results/my_run   # same comparison, no UI
python -m pytest tests -q                               # tests (about 5 min on CPU)
```

In the app choose **Synthetic - signal** first (no files needed), press *Load data and run comparison*, then use the *Forecast and explanation* tab.
The "Quick" speed setting takes a few minutes on a laptop CPU.

## The two fusion strategies being compared

* **Path A, staged:** a sales encoder, a text encoder and an image CNN are trained separately on the first 60% of training windows and frozen. A LightGBM model on their embeddings plus lag features is then trained on the remaining windows (with a 14-day gap so no target leaks). Random whole-modality dropout teaches the stacker to cope with missing inputs.
* **Path B, joint:** one network with three branches and one loss, trained end to end. A gate layer randomly switches off whole modalities during training for the same reason.

Both predict all 14 days at once (direct multi-horizon), because reviews and images for future days are unknown. Everything is measured on the same chronological hold-out (the last 120 days) against baselines: mean, last-7-day mean, seasonal-naive and LightGBM on lags only. Metrics: WAPE (total absolute error / total actual) and MASE.

## Uploading your own data (anything may be missing)

| You provide | What happens |
|---|---|
| Sales CSV only | sales-only models run; text and image rows are skipped and the app says why |
| Sales + reviews | text is used; images skipped |
| Sales + images ZIP | images used; text skipped |
| Reviews only | daily review count stands in for demand (clearly flagged) |
| Reviews or images for only some products or dates | covered rows use them; the rest are treated as missing; if coverage is below 2% the modality is switched off |
| Gaps in sales dates | filled and flagged as imputed |
| Unusual column names | detected from aliases (date, sales/units/quantity, product/store/item, review text, rating) |
| Too little history | a clear message states how many days are needed |

Formats: sales `date, sales[, product ...]`; reviews `date, review text[, rating, product]`; images ZIP with images named like `kettle_2024-03-05.png`, or any file names plus a `manifest.csv` with `date, product, file`. Examples are in `sample_data/`.

A model trained with all inputs is also *tested* with text or images removed (the robustness table), so you can see how much it leans on each.

## Results on the synthetic data (6 products x 900 days, 3 seeds, 120-day hold-out)

Lower is better. Full tables: `results/signal_3seeds.md`, `results/control.md`.

| Approach | WAPE (signal data) | WAPE (control: no signal in text/images) |
|---|---|---|
| Seasonal-naive | 0.234 | 0.278 |
| LightGBM lags only | 0.212 | 0.234 (1 seed) |
| Path B joint, sales only | 0.208 | 0.247 |
| Path A staged, sales only | 0.202 | 0.236 |
| Path B joint, sales+text | 0.181 | 0.249 |
| Path B joint, all inputs | 0.184 | 0.257 |
| Path A staged, sales+text | 0.185 | 0.236 |
| Path A staged, all inputs | **0.179** | 0.239 |

What this does and does not show:

* On the signal data, review text lowers error by about 0.02 WAPE for both paths. Images add a small further gain for Path A and none measurable for Path B.
* **Joint vs staged: no clear winner** across 3 seeds (0.184 vs 0.179, seed spread 0.003-0.005).
* On the **control** data, where reviews and images carry no information, neither path gains from them. Path B is slightly *worse* with them (0.257 vs 0.247 on one seed): the joint network overfits noise inputs more than the staged stacker does. That is the one consistent difference between the two paths.
* **These gains reflect how the generator was written** (reviews and photos lead demand by 6 days by construction). They show the architectures can find a signal that exists; they are not evidence about real retail data. Use your own data for that.

## Honest limitations

* The image CNN is trained from scratch on 32x32 synthetic images (no pretrained backbone), so real-photo performance is untested.
* The "RAG" generation step is a template, not a language model: it retrieves recent reviews using a learned review-impact score and fills in numbers. It cannot invent facts, but it also does not write fluent prose. Modality sensitivity in the explanation is model sensitivity, not proof of cause.
* Framework: Keras 3 on TensorFlow (CPU). Everything runs without a GPU.
* The text encoder is LSA (TF-IDF + SVD), not a transformer, to keep training fast and offline.

## Layout

```
app.py              Streamlit app
hff/ingest.py       reads/validates uploads, handles missing modalities
hff/synth.py        synthetic sales + reviews + images (signal and control modes)
hff/windows.py      leak-free windows, splits, features
hff/nets.py         GRU/CNN branches, modality gate, joint model
hff/fusion.py       Path A (staged) and Path B (joint) predictors
hff/benchmark.py    comparison, robustness, verdict
hff/forecast.py     final forecast on all history
hff/explain.py      retrieval-based explanation
scripts/benchmark.py  command-line comparison
tests/              pytest suite (leakage, ingestion matrix, models, app)
```
