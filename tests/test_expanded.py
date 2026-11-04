import numpy as np
import pandas as pd

from fuelsignal.backtest import ModelSpec, _factor_features
from fuelsignal.data import SERIES
from fuelsignal.expanded import NEW_SERIES, group_map


def test_expansion_has_unique_names_and_reaches_34_series():
    names = [s.name for s in (*SERIES, *NEW_SERIES)]
    assert len(names) == len(set(names)) == 34
    assert sum(len(v) for v in group_map(True).values()) == 34


def test_group_factors_do_not_depend_on_later_rows():
    rng = np.random.default_rng(3)
    index = pd.period_range("2006-01", periods=80, freq="M")
    x = pd.DataFrame(rng.normal(size=(80, 4)), index=index, columns=list("abcd"))
    spec = ModelSpec(
        "t", ("factors",), "group_pca", regime=False,
        groups=(("one", ("a", "b")), ("two", ("c", "d"))),
    )
    end, stop = pd.Period("2010-12", "M"), pd.Period("2012-08", "M")
    changed = x.copy()
    changed.loc["2011-07":] += 50
    a = _factor_features(x, spec, end, stop)
    b = _factor_features(changed, spec, end, stop)
    assert list(a.columns) == ["group_one", "group_two"]
    pd.testing.assert_frame_equal(a.loc[:"2011-06"], b.loc[:"2011-06"])
