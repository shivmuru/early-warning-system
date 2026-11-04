# State unemployment distress

This project predicts whether a US state will enter unemployment distress within 6 months.

**Main result:** a state's own unemployment trend plus the national trend reached a PR-AUC of 0.570,
compared with a 7.0% base rate. No pre-specified factor model beat it. A 3-factor PCA run scored
0.614 in a later diagnostic, but that gain is not statistically clear and needs fresh data.

## Definition

A state is in distress when its 3-month average unemployment rate is at least 1 percentage point
above its low over the prior 12 months. Each warning row is a state that is not yet in distress. It
is labeled 1 if the state enters distress within 6 months, and rows with unknown outcomes are dropped.

Full sample: 25,251 rows, 1,839 positives, and 313 state episodes.

## Data and timing

| Source | Contents | Timing used |
| --- | --- | --- |
| FRED-MD, October 2026 file | 126 national series | Public 1 month later, or 2 months for slower series |
| FRED-SD, April 2026 file | State unemployment, jobs, sector jobs, claims | Public 1 month later |

Rows are indexed by the newest state unemployment month, k. The forecast is treated as made at the
end of month k + 1. Nothing public after November 30, 2025 is used. State unemployment in the file
stops in August 2025, so the last labeled row is February 2025.

## Backtest

Models are refit each year and tested from January 2000 to February 2025: 13,294 rows, 929 positives
(7.0%), and 157 state episodes. National series are transformed with FRED-MD's codes and limited to 5
training standard deviations. Alerts fire at a probability of 0.25, set in advance.

### Pre-specified models

| Model | Inputs | PR-AUC | PR-AUC without COVID | Alert precision | Alert recall | Episodes warned |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| State + national trend | State and US unemployment trends | **0.570** | **0.688** | 51% | 56% | 139/157 |
| State trend | State unemployment trend | 0.546 | 0.660 | 49% | 55% | 131/157 |
| Boosted PCA | Local inputs + 5 PCA factors | 0.541 | 0.654 | 62% | 48% | 139/157 |
| PCA, 5 factors | Local inputs + 5 PCA factors | 0.511 | 0.577 | 45% | 50% | 148/157 |
| PCA + exposure | PCA with factor x industry-share terms | 0.495 | 0.557 | 41% | 49% | 146/157 |
| State local | Trends + state jobs, claims, and industry shares | 0.484 | 0.540 | 47% | 46% | 140/157 |
| PCA + regime | PCA + filtered stress regime | 0.470 | 0.525 | 43% | 46% | 143/157 |
| DFM, 5 factors | Local inputs + 5 DFM factors | 0.403 | 0.433 | 31% | 44% | 123/157 |
| DFM + exposure | DFM with factor x industry-share terms | 0.400 | 0.433 | 29% | 45% | 130/157 |
| Hand-picked macro | Local inputs + curve, credit, VIX, stocks, oil | 0.399 | 0.466 | 35% | 46% | 121/157 |

The 5-factor choice was set before testing, and Bai-Ng also selected 5 on the pre-2000 data.
Sensitivity runs on local inputs scored 0.561 with 3 factors and 0.462 with 8.

### Diagnostics added after the first results

State job data starts in 1990, so models using it train on a shorter history. These runs separate
that effect from the effect of the factors. Intervals are 90%, from resampling whole test years.

| Comparison | PR-AUC gap | 90% interval |
| --- | ---: | --- |
| Trend model trained from 1991 vs. from 1977 | -0.077 | -0.156 to -0.005 |
| Adding state jobs, same 1991 history | -0.009 | -0.037 to +0.024 |
| Adding jobs and 5 PCA factors, same 1991 history | +0.018 | -0.088 to +0.082 |
| Adding 5 PCA factors, full history | -0.011 | -0.127 to +0.057 |
| Adding 3 PCA factors, full history | +0.044 | -0.027 to +0.082 |
| Adding 5 DFM factors, full history | -0.126 | -0.332 to +0.019 |

### Away from recessions

These rows fall outside the span from 3 months before each recession to 12 months after it. They hold
165 positives and test whether the models catch state-specific shocks.

| Model | PR-AUC |
| --- | ---: |
| State trend | 0.314 |
| State + national trend | 0.291 |
| 3 PCA factors + trends, diagnostic | 0.265 |
| PCA, 5 factors | 0.236 |
| DFM, 5 factors | 0.218 |

## What this means

- **Longer history helped more than extra inputs.** Cutting the trend model's training start from
  1977 to 1991 cost more than state jobs or PCA factors added.
- **PCA is competitive, but not clearly better.** It is close to the baseline on equal history. The
  3-factor run is promising, but it was singled out after seeing results.
- **The DFM scored lower here.** This version is simplified, as described below.
- **National signals do not help with state-specific shocks.** The state's own trend did best there.
- **Exposure terms and regimes did not help.**

## Limitations

- **Revised data:** the files contain later revisions, so results may be optimistic. A true
  point-in-time test needs archived vintages.
- **Simplified DFM:** it uses independent noise and at most 100 EM iterations, because the default
  version exceeded 8 GB of memory.
- **Clustered events:** most positives sit in 2001, 2008, and 2020, so intervals are wide.
- **Diagnostics:** they were added after the results and should be confirmed on later data.
- **Timing:** release delays are simple rules, not exact historical dates.
- **Lead time:** it counts from the newest data month, so usable warning is about 1 month shorter.

## Run it

Put `2026-10-MD.csv` and `State_2026-04.xlsx` in `data/raw/`, then run `make state-backtest`.
Metrics are saved to `reports/state_distress/metrics.json`.
