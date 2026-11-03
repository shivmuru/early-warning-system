import numpy as np
import pandas as pd

from fuelsignal.labels import distress_labels, episode_ids


def make(values):
    index = pd.period_range("2000-01", periods=len(values), freq="M")
    return pd.Series(values, index=index, dtype=float), pd.Series(100.0, index=index)


def test_spike_to_high_price_is_distress():
    prices, cpi = make([3.0] * 40 + [3.6, 3.6, 3.6, 3.6])
    out = distress_labels(prices, cpi)
    assert out["distress_label"].iloc[39] == 1
    assert out["months_to_distress"].iloc[39] == 1


def test_rebound_below_recent_average_is_not_distress():
    prices, cpi = make([4.0] * 40 + [2.0, 2.5, 2.5, 2.5, 2.5])
    out = distress_labels(prices, cpi)
    assert out["fwd_peak_real_rise_pct"].iloc[40] == 25
    assert out["distress_label"].iloc[40] == 0


def test_inflation_can_remove_a_nominal_spike():
    prices, cpi = make([3.0] * 40 + [3.5] * 4)
    cpi.iloc[40:] = 120.0
    assert distress_labels(prices, cpi)["distress_label"].iloc[39] == 0


def test_unknown_outcomes_stay_missing():
    prices, cpi = make([3.0] * 44)
    assert distress_labels(prices, cpi)["distress_label"].iloc[-3:].isna().all()


def test_label_ignores_rows_without_full_lookback():
    prices, cpi = make([3.0] * 10 + [4.0] * 4)
    assert distress_labels(prices, cpi)["distress_label"].isna().all()


def test_episode_gap_merging():
    label = pd.Series([1, 0, 0, 1, 0, 0, 0, 0, 1], dtype=float)
    assert np.array_equal(episode_ids(label).dropna().values, [1, 1, 2])
