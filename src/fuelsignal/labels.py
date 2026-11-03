"""The distress label: a real price spike, to a price that is high by recent standards."""

from __future__ import annotations

import numpy as np
import pandas as pd

from fuelsignal import config


def distress_labels(
    nominal: pd.Series,
    cpi: pd.Series,
    rise: float = config.RISE,
    lookback: int = config.LOOKBACK_MONTHS,
    horizon: int = config.HORIZON_MONTHS,
) -> pd.DataFrame:
    """Label each month from prices in the following ``horizon`` months.

    A month is 1 when, within the horizon, the real price is at least ``rise`` above this month
    and also above its average over the ``lookback`` months ending this month. Rows whose outcome
    window runs past the data are left missing, never assumed to be 0.
    """
    real = nominal * cpi.dropna().iloc[-1] / cpi
    average = real.rolling(lookback, min_periods=lookback).mean()
    out = pd.DataFrame(
        {
            "nominal_price": nominal,
            "real_price": real,
            "real_avg_prior": average,
            "fwd_peak_real_rise_pct": np.nan,
            "distress_label": np.nan,
            "months_to_distress": np.nan,
        }
    )
    for i in range(len(real) - horizon):
        if np.isnan(real.iloc[i]) or np.isnan(average.iloc[i]):
            continue
        future = real.iloc[i + 1 : i + 1 + horizon]
        if future.isna().any():
            continue
        hit = (future / real.iloc[i] - 1 >= rise) & (future > average.iloc[i])
        out.iloc[i, out.columns.get_loc("fwd_peak_real_rise_pct")] = 100 * (
            future.max() / real.iloc[i] - 1
        )
        out.iloc[i, out.columns.get_loc("distress_label")] = float(hit.any())
        if hit.any():
            first_hit = float(np.argmax(hit.values) + 1)
            out.iloc[i, out.columns.get_loc("months_to_distress")] = first_hit
    out["episode_id"] = episode_ids(out["distress_label"], gap=horizon)
    return out


def episode_ids(label: pd.Series, gap: int = config.HORIZON_MONTHS) -> pd.Series:
    """Number runs of positive labels, merging runs separated by at most ``gap`` months."""
    ids = pd.Series(np.nan, index=label.index)
    last, current = None, 0
    for i, value in enumerate(label.values):
        if value == 1:
            if last is None or i - last > gap:
                current += 1
            ids.iloc[i] = current
            last = i
    return ids


def build_panel(prices: pd.DataFrame, cpi: pd.Series) -> pd.DataFrame:
    frames = []
    for region in prices.columns:
        frame = distress_labels(prices[region], cpi)
        frame.insert(0, "region", region)
        frame.insert(0, "month", frame.index.astype(str))
        frames.append(frame.loc[config.MODEL_START :])
    return pd.concat(frames, ignore_index=True)
