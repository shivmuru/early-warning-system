import numpy as np
import pandas as pd

from fuelsignal import config
from fuelsignal.data import as_of_monthly, build_cpi


def weekly(values, start="2025-10-03"):
    index = pd.date_range(start, periods=len(values), freq="7D")
    return pd.Series(values, index=index, dtype=float)


def test_weekly_value_waits_for_release():
    # The Friday Nov 28 observation is released Dec 3, so November must use Nov 21.
    series = weekly(np.arange(10.0))
    november = as_of_monthly(series, "days", 5).loc["2025-11"]
    assert series.loc["2025-11-21"] == november


def test_monthly_value_uses_publication_lag():
    series = pd.Series(
        [1.0, 2.0, 3.0], index=pd.to_datetime(["2025-08-01", "2025-09-01", "2025-10-01"])
    )
    result = as_of_monthly(series, "months", 1)
    assert result.loc["2025-09"] == 1.0
    assert result.loc["2025-11"] == 3.0


def test_values_after_cutoff_never_change_output():
    series = weekly(np.arange(30.0), start="2025-06-06")
    changed = series.copy()
    changed[changed.index > config.CUTOFF - pd.Timedelta(days=5)] = 1e9
    pd.testing.assert_series_equal(
        as_of_monthly(series, "days", 5), as_of_monthly(changed, "days", 5)
    )


def test_output_stops_at_last_month():
    result = as_of_monthly(weekly(np.arange(30.0), start="2025-06-06"), "days", 5)
    assert result.index[-1] == config.LAST_MONTH


def test_cpi_carries_last_public_print_forward():
    index = pd.date_range("1993-01-01", "2025-12-01", freq="MS")
    values = pd.Series(np.arange(len(index), dtype=float), index=index)
    values.loc["2025-10-01"] = np.nan
    cpi = build_cpi({"cpi": values})
    assert cpi.loc["2025-11"] == cpi.loc["2025-09"] == values.loc["2025-09-01"]
