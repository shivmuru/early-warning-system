import numpy as np
import pandas as pd

from fuelsignal.backtest import training_windows
from fuelsignal.factors import dfm_factors, pca_factors
from fuelsignal.regimes import forward_filter


def panel(rows=80, cols=6, seed=1):
    rng = np.random.default_rng(seed)
    index = pd.period_range("2006-01", periods=rows, freq="M")
    common = rng.normal(size=(rows, 1))
    data = common + rng.normal(scale=0.5, size=(rows, cols))
    return pd.DataFrame(data, index=index, columns=[f"s{i}" for i in range(cols)])


def test_pca_scores_do_not_depend_on_later_rows():
    x = panel()
    changed = x.copy()
    changed.iloc[60:] += 100
    a, _ = pca_factors(x.iloc[:60], x, 2)
    b, _ = pca_factors(changed.iloc[:60], changed, 2)
    pd.testing.assert_frame_equal(a.iloc[:60], b.iloc[:60])


def test_dfm_filtered_values_do_not_depend_on_later_rows():
    x = panel()
    changed = x.copy()
    changed.iloc[70:] += 100
    a = dfm_factors(x.iloc[:60], x, 1)
    b = dfm_factors(x.iloc[:60], changed, 1)
    np.testing.assert_allclose(a.iloc[:70].values, b.iloc[:70].values)
    assert not np.allclose(a.iloc[70:].values, b.iloc[70:].values)


def test_forward_filter_ignores_future_observations():
    obs = np.random.default_rng(2).normal(size=(30, 2))
    args = (
        np.array([0.5, 0.5]),
        np.array([[0.9, 0.1], [0.2, 0.8]]),
        np.array([[0.0, 0.0], [1.0, 1.0]]),
        np.array([[1.0, 1.0], [4.0, 4.0]]),
    )
    full = forward_filter(obs, *args)
    changed = obs.copy()
    changed[20:] = 50.0
    np.testing.assert_allclose(full[:20], forward_filter(changed, *args)[:20])
    np.testing.assert_allclose(full.sum(axis=1), 1.0)


def test_training_labels_are_known_before_test_year():
    months = pd.period_range("2006-01", "2025-08", freq="M")
    for data_end, label_end, test_start, test_end in training_windows(months):
        assert label_end + 3 == data_end
        assert data_end + 1 == test_start
        assert test_end <= months.max()
