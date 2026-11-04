"""Run with ``statedistress data`` then ``statedistress backtest``."""

from __future__ import annotations

import argparse
import json

import pandas as pd

from fuelsignal.factors import bai_ng_icp2, pca_factors
from statedistress import config
from statedistress.backtest import (
    Spec,
    bootstrap_gap,
    episode_summary,
    factors_for,
    score,
    subsets,
    walk_forward,
    windows,
)
from statedistress.data import load_state_panel, national_levels, national_predictors, read_md
from statedistress.features import national_features, state_features
from statedistress.labels import build_labels, distress_state

COMPARISONS = [
    ("state_national", "state_trend"),
    ("state_local", "state_national"),
    ("hand_macro", "state_local"),
    ("pca", "state_local"),
    ("pca", "hand_macro"),
    ("pca_exposure", "pca"),
    ("pca_regime", "pca"),
    ("dfm", "pca"),
    ("dfm", "state_local"),
    ("dfm_exposure", "dfm"),
]
LEVELS = ["UNRATE", "T10YFFM", "BAAFFM", "VIXCLSx", "S&P 500", "OILPRICEx"]


def prepare():
    levels, codes = read_md()
    predictors = national_predictors(levels, codes)
    panel = load_state_panel()
    labels = build_labels(panel["UR"])
    design = labels.merge(state_features(panel), on=["month", "state"], how="left")
    macro = national_features(national_levels(levels, LEVELS))
    design = design.merge(macro, left_on="month", right_index=True, how="left")
    return predictors, panel, labels, design


def run_data() -> None:
    predictors, panel, labels, design = prepare()
    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    predictors.to_csv(config.CACHE_DIR / "national_predictors.csv")
    design.to_csv(config.CACHE_DIR / "design.csv", index=False)
    print(f"{len(design)} rows, {int(labels['distress_label'].sum())} positive")


def run_backtest() -> None:
    predictors, panel, labels, design = prepare()
    results = walk_forward(design, predictors)
    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(config.CACHE_DIR / "predictions.csv", index=False)
    pre_test = predictors.loc[: pd.Period(f"{config.FIRST_TEST_YEAR - 1}-11", "M")]
    pre_test = pre_test.dropna(axis=1, thresh=int(0.9 * len(pre_test)))
    _, pca = pca_factors(predictors.loc["1960":], predictors.loc["1960":], 8, config.SEED)
    distress = distress_state(panel["UR"])
    onsets = ((distress == 1) & (distress.shift(1) == 0)).sum().sum()
    summary = {
        "cutoff": str(config.CUTOFF.date()),
        "label_rows": len(labels),
        "label_positives": int(labels["distress_label"].sum()),
        "state_episodes_full_sample": int(onsets),
        "national_series": predictors.shape[1],
        "bai_ng_factors_pre_test": bai_ng_icp2(pre_test.loc["1960":], k_max=15),
        "pca_variance_share_full_sample": [float(v) for v in pca.explained_variance_ratio_],
        "models": {
            name: {
                subset: {**score(part), **episode_summary(part)}
                for subset, part in subsets(group).items()
            }
            for name, group in results.groupby("model", sort=False)
        },
        "comparisons": {
            subset: {f"{a}_vs_{b}": bootstrap_gap(part, a, b) for a, b in COMPARISONS}
            for subset, part in subsets(results).items()
        },
    }
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = config.OUTPUT_DIR / "metrics.json"
    path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Wrote {path}")


def run_dfm_year(year: int) -> None:
    """Fit one year's DFM and save its factors. Separate processes keep memory from building up."""
    predictors, _, _, design = prepare()
    months = pd.PeriodIndex(design["month"].unique(), freq="M")
    for data_end, _, test_start, test_end in windows(months):
        if test_start.year == year:
            factors_for(predictors, Spec("dfm", engine="dfm"), data_end, test_end)
            print(f"dfm {year} done", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(prog="statedistress")
    parser.add_argument("command", choices=["data", "backtest", "dfm-year"])
    parser.add_argument("--year", type=int)
    args = parser.parse_args()
    if args.command == "dfm-year":
        run_dfm_year(args.year)
        return
    {"data": run_data, "backtest": run_backtest}[args.command]()


if __name__ == "__main__":
    main()
