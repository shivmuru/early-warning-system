"""Command line entry point: ``fuelsignal data`` then ``fuelsignal backtest``."""

from __future__ import annotations

import argparse
import json

import pandas as pd

from fuelsignal import config
from fuelsignal.backtest import bootstrap_gap, build_design, episode_summary, score, walk_forward
from fuelsignal.data import (
    build_cpi,
    build_levels,
    build_predictors,
    build_prices,
    data_dictionary,
    load_raw,
)
from fuelsignal.factors import bai_ng_icp2, pca_factors
from fuelsignal.labels import build_panel

COMPARISONS = [
    ("local", "calendar"),
    ("crude", "calendar"),
    ("all_factors", "calendar"),
    ("all_factors", "local"),
    ("all_factors", "local_crude"),
    ("all_factors", "energy_factors"),
]


def prepare():
    raw = load_raw()
    levels = build_levels(raw)
    predictors = build_predictors(levels)
    prices = build_prices(raw)
    panel = build_panel(prices, build_cpi(raw))
    return levels, predictors, prices, panel


def run_data() -> None:
    levels, predictors, _, panel = prepare()
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    levels.loc[config.MODEL_START :].to_csv(config.PROCESSED_DIR / "predictors_levels.csv")
    predictors.to_csv(config.PROCESSED_DIR / "predictors_model.csv")
    panel.to_csv(config.PROCESSED_DIR / "targets_panel.csv", index=False)
    data_dictionary().to_csv(config.PROCESSED_DIR / "data_dictionary.csv", index=False)
    print(f"Wrote {len(predictors)} months and {len(panel)} target rows")


def run_backtest() -> None:
    levels, predictors, prices, panel = prepare()
    design, calendar = build_design(panel, prices, levels)
    results = walk_forward(design, calendar, predictors)
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(config.PROCESSED_DIR / "predictions.csv", index=False)

    metrics = {
        name: {**score(group), **episode_summary(group)}
        for name, group in results.groupby("model", sort=False)
    }
    comparisons = {f"{a}_vs_{b}": bootstrap_gap(results, a, b) for a, b in COMPARISONS}
    first = results[results["model"] == "calendar"]
    pre_test = predictors.loc[: pd.Period(f"{config.FIRST_TEST_YEAR - 1}-12", "M")]
    _, pca = pca_factors(predictors, predictors, 5, config.SEED)
    loadings = pd.DataFrame(
        pca.components_.T[:, :config.N_FACTORS],
        index=predictors.columns,
        columns=[f"factor_{i + 1}" for i in range(config.N_FACTORS)],
    )
    summary = {
        "cutoff": str(config.CUTOFF.date()),
        "test_months": [str(first["month"].min()), str(first["month"].max())],
        "test_rows": len(first),
        "test_positive_rows": int(first["distress_label"].sum()),
        "prevalence": float(first["distress_label"].mean()),
        "bai_ng_factors_pre_test": bai_ng_icp2(pre_test),
        "pca_variance_share_full_sample": [float(v) for v in pca.explained_variance_ratio_],
        "models": metrics,
        "comparisons": comparisons,
    }
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (config.REPORTS_DIR / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
    loadings.round(3).to_csv(config.REPORTS_DIR / "pca_loadings.csv")
    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(prog="fuelsignal")
    parser.add_argument("command", choices=["data", "backtest"])
    args = parser.parse_args()
    {"data": run_data, "backtest": run_backtest}[args.command]()


if __name__ == "__main__":
    main()
