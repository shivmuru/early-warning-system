import numpy as np
import pandas as pd
import pytest

from statedistress import config
from statedistress.backtest import Spec, clip_outliers, episode_summary, walk_forward, windows
from statedistress.cli import prepare
from statedistress.data import align, apply_tcode, release_lag
from statedistress.labels import build_labels, distress_state


def periods(n, start="2000-01"):
    return pd.period_range(start, periods=n, freq="M")


def test_tcodes_match_fred_md_definitions():
    s = pd.Series([1.0, 2.0, 4.0, 8.0], index=periods(4))
    assert apply_tcode(s, 1).equals(s)
    assert apply_tcode(s, 2).iloc[-1] == 4
    assert apply_tcode(s, 3).iloc[-1] == 2
    assert apply_tcode(s, 4).iloc[-1] == pytest.approx(np.log(8))
    assert apply_tcode(s, 5).iloc[-1] == pytest.approx(np.log(2))
    assert apply_tcode(s, 6).iloc[-1] == pytest.approx(0)
    assert apply_tcode(s, 7).iloc[-1] == pytest.approx(0)


def test_alignment_respects_release_lag_and_cutoff():
    s = pd.Series(1.0, index=pd.period_range("2025-07", "2026-03", freq="M"))
    assert align(s, 1).index.max() == pd.Period("2025-10", "M")
    # A two-month series enters a row one month after its date, and its last public date is Sep.
    assert align(s, 2).index.max() == pd.Period("2025-10", "M")
    assert release_lag("PCEPI") == 2 and release_lag("UNRATE") == 1


def state_ur(values):
    return pd.DataFrame({"AA": values}, index=periods(len(values)), dtype=float)


def test_distress_needs_full_history_and_a_one_point_rise():
    flagged = distress_state(state_ur([4.0] * 20 + [6.0] * 6))["AA"]
    assert flagged.iloc[:14].isna().all()
    assert flagged.iloc[19] == 0
    assert flagged.iloc[20] == 0  # 3-month average is 4.67, up 0.67
    assert flagged.iloc[21] == 1


def test_labels_warn_before_entry_and_skip_rows_in_distress():
    labels = build_labels(state_ur([4.0] * 30 + [7.0] * 10 + [7.0] * 8), horizon=6)
    onset = labels.dropna(subset=["onset"])["onset"].iloc[0]
    positive = labels[labels["distress_label"] == 1]
    assert positive["month"].max() == onset - 1
    assert positive["month"].min() == onset - 6
    assert not (labels["month"] >= onset).any() or labels[labels["month"] >= onset].empty


def test_incomplete_outcome_windows_are_dropped():
    labels = build_labels(state_ur([4.0] * 40), horizon=6)
    assert labels["month"].max() == periods(40)[-7]


def test_training_labels_are_known_at_fit_date():
    months = pd.period_range("1990-01", "2025-02", freq="M")
    for data_end, label_end, test_start, test_end in windows(months):
        fit_date = test_start - 1
        assert data_end == fit_date - 1
        # Row label_end + horizon is public one month later, at the fit date.
        assert label_end + config.HORIZON_MONTHS + 1 == fit_date
        assert test_end <= months.max()


def test_clip_uses_training_limits_only():
    x = pd.DataFrame({"a": np.arange(100.0)}, index=periods(100))
    spiked = x.copy()
    spiked.iloc[-1, 0] = 1e9
    train, full = clip_outliers(x.iloc[:50], spiked)
    limit = x["a"].iloc[:50].mean() + config.OUTLIER_SD * x["a"].iloc[:50].std()
    assert full["a"].iloc[-1] == pytest.approx(limit)
    assert train["a"].max() == 49


def test_episode_summary_counts_alert_before_entry():
    onset = pd.Period("2001-07", "M")
    frame = pd.DataFrame(
        {
            "state": ["AA", "AA", "BB"],
            "month": pd.PeriodIndex(["2001-03", "2001-05", "2001-05"], freq="M"),
            "onset": [onset, onset, onset],
            "probability": [0.5, 0.1, 0.1],
        }
    )
    assert episode_summary(frame) == {
        "episodes": 2, "warned": 1, "episode_recall": 0.5, "median_lead": 4.0,
    }


real_data = pytest.mark.skipif(
    not (config.MD_FILE.exists() and config.SD_FILE.exists()), reason="raw data not available"
)


@real_data
def test_real_labels_and_ranges():
    predictors, panel, labels, design = prepare()
    assert predictors.index.max() == pd.Period("2025-10", "M")
    assert design["month"].max() == pd.Period("2025-02", "M")
    assert panel["NA"].notna().any().all()
    assert design.duplicated(["month", "state"]).sum() == 0


@real_data
def test_real_predictions_ignore_later_data():
    predictors, _, _, design = prepare()
    specs = [Spec("pca", ("ur_gap", "us_gap"), "pca"), Spec("base", ("ur_gap",))]
    first = design[design["month"] <= pd.Period("2003-12", "M")]
    base = walk_forward(first, predictors.loc[:"2003-12"], specs)
    limit = pd.Period("2002-06", "M")
    moved = predictors.loc[:"2003-12"].copy()
    moved.loc["2002-07":] += 100
    changed = first.copy()
    changed.loc[changed["month"] > limit, "distress_label"] = 1.0
    changed.loc[changed["month"] > limit, ["ur_gap", "us_gap"]] = 99.0
    other = walk_forward(changed, moved, specs)
    keys = ["model", "state", "month"]
    a = base[base["month"] <= limit].sort_values(keys)
    b = other[other["month"] <= limit].sort_values(keys)
    np.testing.assert_allclose(a["probability"], b["probability"])
