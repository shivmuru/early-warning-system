# State Unemployment Distress: Early-Warning System

This project estimates whether a US state will enter unemployment distress within the next 6 months.
It covers all 50 states and DC, using walk-forward tests from January 2000 to February 2025.

**Main result:** the state's own unemployment trend plus the national trend reached a PR-AUC of 0.570,
compared with a 7.0% base rate. No pre-specified PCA, DFM, or hand-picked macro model beat it. A
3-factor PCA run added afterward scored 0.614, but that gain is not statistically clear and needs
testing on new data.

## What counts as distress

A state is in distress when its 3-month average unemployment rate is at least 1 percentage point
above its low over the prior 12 months.

Each warning row is a state not yet in distress. It is labeled 1 if the state enters distress within 6
months. Rows with unknown outcomes are dropped.

The full sample has 25,251 rows, 1,839 positives, and 313 state episodes.

## Data

| Source | Contents | Assumed release delay |
| --- | --- | --- |
| [FRED-MD](https://www.stlouisfed.org/research/economists/mccracken/fred-databases), October 2026 file | 126 national series | 1 month, or 2 months for slower series |
| [FRED-SD](https://www.stlouisfed.org/research/economists/owyang/fred-sd), April 2026 file | State unemployment, jobs, sector jobs, and claims | 1 month |

Raw files are not stored in this repo. `data/MANIFEST.csv` lists each source and the SHA-256 of the file
used.

## Preventing lookahead

- **Cutoff:** nothing public after November 30, 2025 is used.
- **Release delays:** each value counts only after its assumed release delay.
- **Walk-forward training:** models are refit yearly, and training labels must have known outcomes.
- **Factors:** PCA and DFM parameters come only from training data. The DFM and regime models use
  filtered values, and outlier limits are also set on training data.
- **Tests:** one test changes later data and confirms that earlier predictions do not change.

## Results

The test period has 13,294 rows, 929 positives (7.0%), and 157 state episodes. Alerts fire at a
probability of 0.25, set in advance.

| Model | PR-AUC | PR-AUC without COVID | Alert precision | Alert recall | Episodes warned |
| --- | ---: | ---: | ---: | ---: | ---: |
| **State + national trend** | **0.570** | **0.688** | **51%** | **56%** | **139/157** |
| State trend | 0.546 | 0.660 | 49% | 55% | 131/157 |
| Boosted PCA | 0.541 | 0.654 | 62% | 48% | 139/157 |
| PCA, 5 factors | 0.511 | 0.577 | 45% | 50% | 148/157 |
| PCA + industry exposure | 0.495 | 0.557 | 41% | 49% | 146/157 |
| Trends + state jobs, claims, and industry mix | 0.484 | 0.540 | 47% | 46% | 140/157 |
| PCA + regime | 0.470 | 0.525 | 43% | 46% | 143/157 |
| DFM, 5 factors | 0.403 | 0.433 | 31% | 44% | 123/157 |
| DFM + industry exposure | 0.400 | 0.433 | 29% | 45% | 130/157 |
| Trends and state data + hand-picked macro | 0.399 | 0.466 | 35% | 46% | 121/157 |

All PCA and DFM rows above also include the trends and state data. Bai-Ng selected 5 factors on
pre-2000 data, matching the number fixed in advance.

### Diagnostics

Job data starts in 1990, so models using it train on less history. These runs were added after the
first results to separate that effect from the effect of the factors. Intervals are 90%, from
resampling whole test years.

| Comparison | PR-AUC gap | 90% interval |
| --- | ---: | --- |
| Trend model trained from 1991 vs. from 1977 | -0.077 | -0.156 to -0.005 |
| Adding state jobs, same 1991 history | -0.009 | -0.037 to +0.024 |
| Adding 5 PCA factors, full history | -0.011 | -0.127 to +0.057 |
| Adding 3 PCA factors, full history | +0.044 | -0.027 to +0.082 |
| Adding 5 DFM factors, full history | -0.126 | -0.332 to +0.019 |

### What this means

- **Longer history helped more than extra inputs.**
- **PCA is competitive, but not clearly better.**
- **State-specific shocks are hard to predict.** Away from recessions, the state trend alone did best,
  with a PR-AUC of 0.314.
- **Regimes and industry-exposure terms did not help.**

## Limitations

- **Revised data:** the files contain later revisions, so results may be optimistic.
- **Simplified DFM:** it uses independent noise and at most 100 iterations to stay within memory.
- **Clustered events:** most positives fall around 2001, 2008, and 2020.
- **Diagnostics:** these were added after seeing results and need confirmation on later data.
- **Approximate timing:** release delays are simple rules, not exact historical dates.

## Run it

Put `2026-10-MD.csv` and `State_2026-04.xlsx` in `data/raw/`, then run:

```bash
make install
make test
make state-backtest
```

Metrics are saved to `reports/state_distress/metrics.json`. The full methodology is in
[`docs/state-distress.md`](docs/state-distress.md).
