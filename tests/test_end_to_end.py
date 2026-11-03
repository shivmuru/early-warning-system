"""Runs against the real raw files when present; CI has none and skips these."""

import numpy as np
import pandas as pd
import pytest

from fuelsignal import config
from fuelsignal.backtest import ModelSpec, build_design, walk_forward
from fuelsignal.data import build_cpi, build_levels, build_predictors, build_prices, load_raw
from fuelsignal.labels import build_panel

pytestmark = pytest.mark.skipif(
    not (config.RAW_DIR / "RBRTEw.xls").exists(), reason="raw data not available"
)


def outputs(raw):
    levels = build_levels(raw)
    predictors = build_predictors(levels)
    prices = build_prices(raw)
    panel = build_panel(prices, build_cpi(raw))
    return levels, predictors, prices, panel


def corrupt_after(raw, date):
    changed = {}
    for name, series in raw.items():
        series = series.copy()
        series[series.index > date] = series[series.index > date] * 50 + 1000
        changed[name] = series
    return changed


def test_post_cutoff_values_cannot_change_any_output():
    raw = load_raw()
    base = outputs(raw)
    changed = outputs(corrupt_after(raw, config.CUTOFF))
    for a, b in zip(base, changed, strict=True):
        pd.testing.assert_frame_equal(a, b)


def test_test_period_predictions_ignore_later_data():
    raw = load_raw()
    specs = [
        ModelSpec("crude", ("calendar", "crude")),
        ModelSpec("all_factors", ("calendar", "local", "factors"), "dfm"),
    ]
    levels, predictors, prices, panel = outputs(raw)
    design, calendar = build_design(panel, prices, levels)
    base = walk_forward(design, calendar, predictors, specs)

    # Change every predictor and price feature after Jun 2018 and every label after that date.
    moved_predictors = predictors.copy()
    moved_levels = levels.copy()
    moved_prices = prices.copy()
    moved_predictors.loc["2018-07":] += 100
    moved_levels.loc["2018-07":] *= 50
    moved_prices.loc["2018-07":] *= 50
    moved_panel = panel.copy()
    moved_panel.loc[moved_panel["month"] > "2018-06", "distress_label"] = 1.0
    design2, calendar2 = build_design(moved_panel, moved_prices, moved_levels)
    moved = walk_forward(design2, calendar2, moved_predictors, specs)

    keys = ["model", "region", "month"]
    limit = pd.Period("2018-06", "M")
    a = base[base["month"] <= limit].sort_values(keys).reset_index(drop=True)
    b = moved[moved["month"] <= limit].sort_values(keys).reset_index(drop=True)
    np.testing.assert_allclose(a["probability"], b["probability"])
