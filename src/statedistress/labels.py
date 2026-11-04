"""State distress: a sharp rise in unemployment from its recent low."""

from __future__ import annotations

import numpy as np
import pandas as pd

from statedistress import config


def distress_state(ur: pd.DataFrame, rise: float = config.RISE_POINTS) -> pd.DataFrame:
    """1 when the 3-month average is ``rise`` points above its low over the prior 12 months."""
    average = ur.rolling(3).mean()
    low = average.shift(1).rolling(12).min()
    flagged = ((average - low) >= rise - 1e-9).astype(float)
    return flagged.where(average.notna() & low.notna())


def build_labels(
    ur: pd.DataFrame, rise: float = config.RISE_POINTS, horizon: int = config.HORIZON_MONTHS
) -> pd.DataFrame:
    """Long table of warning rows.

    A row exists only when a state is not in distress and its full outcome window is known.
    Label 1 means it enters distress within ``horizon`` months. ``onset`` is the entry month.
    """
    distress = distress_state(ur, rise)
    rows = []
    for state in ur.columns:
        d = distress[state]
        for i in range(len(d) - horizon):
            future = d.iloc[i + 1 : i + 1 + horizon]
            if d.iloc[i] != 0 or future.isna().any():
                continue
            hits = np.flatnonzero(future.values == 1)
            rows.append(
                {
                    "month": d.index[i],
                    "state": state,
                    "distress_label": float(len(hits) > 0),
                    "onset": future.index[hits[0]] if len(hits) else pd.NaT,
                }
            )
    return pd.DataFrame(rows)


def recession_window(index: pd.PeriodIndex, before: int = 3, after: int = 12) -> pd.Series:
    mask = pd.Series(False, index=index)
    for start, end in config.RECESSIONS:
        low, high = pd.Period(start, "M") - before, pd.Period(end, "M") + after
        mask[(index >= low) & (index <= high)] = True
    return mask
