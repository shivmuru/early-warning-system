# FuelSignal

An early-warning system that estimates whether a US region's gasoline prices will reach distress
levels within the next 1-3 months. The aim is to give energy assistance programs and community
organizations time to prepare.

**Main result:** simple oil-price and local-price signals beat the macro factor models in a
2012-2025 walk-forward backtest. The best model, using recent local and crude price moves, reached
a PR-AUC of 0.43 compared with 0.17 for a calendar-only baseline. Adding the macro factors did not
improve on it.

## What counts as distress

A region-month is labeled 1 when, within the next three months, the inflation-adjusted price:

1. rises at least 15% from the current month, and
2. ends above its average for the 36 months ending in the current month.

The second condition keeps rebounds from a crash, such as April 2020, from counting as distress.

| Example | Peak real rise within 3 months | Above 36-month average? | Label |
| --- | ---: | --- | ---: |
| Midwest, Dec 2021 | +29.9% | Yes | 1 |
| US, Apr 2020 | +20.4% | No | 0 |
| Midwest, Jun 2022 | -12.1% | No rise | 0 |

## Data

The panel runs monthly from January 2006 to November 2025 across 7 non-overlapping EIA regions,
with 17 predictors:

| Group | Series |
| --- | --- |
| Energy prices | Brent, WTI, Henry Hub |
| Macro and financial | Real broad dollar, fed funds rate, 10Y-2Y curve, VIX |
| US inventories | Commercial crude, SPR, gasoline, distillate, jet fuel, residual fuel |
| US supply | Crude production, total imports, Persian Gulf imports, Canadian imports |

Raw files are not stored in this public repo. `data/MANIFEST.csv` lists each source and the SHA-256
of the file used.

## Preventing lookahead

- **Hard cutoff:** nothing released after November 30, 2025 can enter any output. A test corrupts
  all later values and confirms that the outputs do not change.
- **Release lags:** each value counts only once it would have been public: 1 day for daily data and
  retail prices, 5 days for weekly EIA data, 1 month for the dollar and fed funds, and 3 months for
  production and imports.
- **Walk-forward training:** models are refit each January. Training labels must have complete
  outcomes before the refit date.
- **Filtered estimates:** DFM factors and HMM regime probabilities use filtered values, not smoothed
  ones. Their parameters come only from the training window.
- **Local features:** model inputs use nominal prices because same-month CPI was not yet public.

## Results

Out-of-sample test period: January 2012 to August 2025. It covers 1,148 region-months, 89 positives
(7.8%), and 10 separate episodes across regions (40 regional episodes). Alerts use a probability
threshold of 0.30, set before the backtest.

| Model | PR-AUC | ROC-AUC | Brier | Alert precision | Alert recall | Regional episodes warned early |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Calendar and region | 0.169 | 0.776 | 0.078 | 19% | 44% | 26/40 |
| Calendar + crude | 0.411 | 0.841 | 0.064 | 38% | 60% | 28/40 |
| Calendar + local prices | 0.327 | 0.839 | 0.067 | 33% | 56% | 30/40 |
| **Calendar + local prices + crude** | **0.433** | **0.844** | **0.065** | **35%** | **58%** | **30/40** |
| Local + energy-only DFM + regime | 0.195 | 0.778 | 0.080 | 26% | 40% | 19/40 |
| Local + all-series DFM + regime | 0.225 | 0.784 | 0.080 | 25% | 48% | 26/40 |
| Local + all-series PCA + regime | 0.184 | 0.784 | 0.098 | 21% | 38% | 18/40 |
| Local + all-series DFM + regime, boosted | 0.142 | 0.671 | 0.086 | 14% | 23% | 15/40 |

PR-AUC gaps, with 90% intervals from resampling whole test years:

| Comparison | Gap | 90% interval |
| --- | ---: | --- |
| Crude vs. calendar | +0.242 | +0.039 to +0.403 |
| Local vs. calendar | +0.158 | +0.022 to +0.378 |
| All-series DFM vs. calendar | +0.056 | -0.032 to +0.232 |
| All-series DFM vs. local + crude | -0.208 | -0.302 to -0.052 |
| All-series DFM vs. energy-only DFM | +0.030 | -0.062 to +0.155 |

### What this means

- **Oil and local prices carry the usable signal.** Both clearly beat the seasonal baseline.
- **The factors did not earn their complexity.** The all-series DFM model was worse than the simple
  local-and-crude model, and this data does not show a clear benefit from macro and financial series.
- **The factor count is not settled.** The Bai-Ng criterion chose 8 factors, its maximum, on the
  pre-2012 data. I fixed k = 4 in advance; k = 3 and k = 5 scored 0.294 and 0.208. These are
  sensitivity checks and were not used to select a model.
- **Seasonality matters.** Most labels fall from December to February, so any model has to beat the
  calendar baseline.

### Grouped PCA and the 34-series extension

A later test added PCA directly to the best model, with no regime signal. This is a new hypothesis
checked on a test period already seen, so treat it as exploratory.

| Model | Series | PR-AUC | Gap vs. local + crude | 90% interval |
| --- | ---: | ---: | ---: | --- |
| Local + crude | 0 factors | 0.433 | | |
| Local + crude + 1 PCA factor per group | 17 | 0.262 | -0.171 | -0.298 to +0.018 |
| Local + crude + 4 PCA factors | 17 | 0.236 | -0.197 | -0.313 to +0.010 |

`scripts/fetch_series.py` downloads 17 more pre-specified series. `make expanded` then runs the same
comparison on all 34. The expanded run has not been executed, and its series IDs and download URLs
need to be confirmed by that first run.

## Limitations

- The best model was picked by comparing results on the same test period, so its score is likely
  somewhat optimistic. It needs confirmation on later data.
- Ten separate episodes is a small sample, and regions move together.
- Files contain current revised data, not the values first published.
- October 2025 CPI was never published, and September 2025 is carried forward for October and
  November.
- Prices are the only outcome. The label does not measure income or household hardship.
- Data after the cutoff has not been scored.

## Run it

Put the files listed in `data/MANIFEST.csv` in `data/raw/`, then:

```bash
make install
make test
make data
make backtest
```

Outputs go to `data/processed/`. Metrics are in `reports/metrics.json`, and full-sample PCA loadings
are in `reports/pca_loadings.csv`.

## Layout

```text
src/fuelsignal/
  config.py    Cutoff, label, and backtest settings
  data.py      Point-in-time predictors and prices
  labels.py    Distress labels and episodes
  factors.py   PCA and Dynamic Factor Model
  regimes.py   Gaussian HMM with a forward filter
  backtest.py  Models, walk-forward testing, and metrics
  cli.py       Command line entry point
tests/         Unit tests and real-data leakage checks
```
