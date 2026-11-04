"""Row-level features, using only state data through month k."""

from __future__ import annotations

import numpy as np
import pandas as pd

UR_FEATURES = ["ur", "ur_gap", "ur_chg_3m", "ur_chg_12m"]
NATIONAL_TREND = ["us_gap", "us_chg_3m"]
STATE_JOBS = ["jobs_3m", "jobs_12m", "claims_3m", "mfg_share", "mining_share"]
HAND_MACRO = ["curve", "credit", "log_vix", "stocks_6m", "oil_12m"]
EXPOSURE = ["mfg_share", "mining_share"]


def _stack(frame: pd.DataFrame, name: str) -> pd.Series:
    return frame.stack(future_stack=True).rename(name)


def state_features(panel: dict[str, pd.DataFrame]) -> pd.DataFrame:
    ur = panel["UR"]
    average = ur.rolling(3).mean()
    jobs = panel["NA"]
    parts = {
        "ur": ur,
        "ur_gap": average - average.shift(1).rolling(12).min(),
        "ur_chg_3m": ur - ur.shift(3),
        "ur_chg_12m": ur - ur.shift(12),
        "jobs_3m": 100 * np.log(jobs).diff(3),
        "jobs_12m": 100 * np.log(jobs).diff(12),
        "claims_3m": 100 * np.log(panel["ICLAIMS"]).diff(3),
        "mfg_share": 100 * panel["MFG"] / jobs,
        "mining_share": (100 * panel["MINNG"] / jobs).fillna(0.0),
    }
    frame = pd.concat([_stack(v, k) for k, v in parts.items()], axis=1)
    frame.index.names = ["month", "state"]
    return frame.reset_index()


def national_features(levels: pd.DataFrame) -> pd.DataFrame:
    us = levels["UNRATE"]
    average = us.rolling(3).mean()
    return pd.DataFrame(
        {
            "us_gap": average - average.shift(1).rolling(12).min(),
            "us_chg_3m": us - us.shift(3),
            "curve": levels["T10YFFM"],
            "credit": levels["BAAFFM"],
            "log_vix": np.log(levels["VIXCLSx"]),
            "stocks_6m": 100 * np.log(levels["S&P 500"]).diff(6),
            "oil_12m": 100 * np.log(levels["OILPRICEx"]).diff(12),
        }
    )
