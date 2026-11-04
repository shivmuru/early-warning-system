"""Point-in-time national (FRED-MD) and state (FRED-SD) data.

Each row is indexed by month k, the newest state unemployment month in hand. That figure is public
by the end of month k + 1, so the forecast is made then, using only what was public at that date.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from statedistress import config

# Published about two months after the reference month. Everything else counts after one month.
SLOW_SERIES = frozenset(
    {
        "RPI", "W875RX1", "DPCERA3M086SBEA", "CMRMTSPLx", "BUSINVx", "ISRATIOx", "ACOGNO",
        "AMDMNOx", "ANDENOx", "AMDMUOx", "NONREVSL", "CONSPI", "DTCOLNVHFNM", "DTCTHFNM", "INVEST",
        "PCEPI", "DDURRG3M086SBEA", "DNDGRG3M086SBEA", "DSERRG3M086SBEA", "M1SL", "M2SL", "M2REAL",
        "BOGMBASE", "TOTRESNS", "NONBORRES", "BUSLOANS", "REALLN",
    }
)  # fmt: skip
STATE_VARS = ("UR", "NA", "MFG", "MINNG", "ICLAIMS")


def apply_tcode(series: pd.Series, code: int) -> pd.Series:
    """FRED-MD transformation codes."""
    if code == 1:
        return series
    if code == 2:
        return series.diff()
    if code == 3:
        return series.diff().diff()
    if code == 4:
        return np.log(series)
    if code == 5:
        return np.log(series).diff()
    if code == 6:
        return np.log(series).diff().diff()
    if code == 7:
        return series.pct_change(fill_method=None).diff()
    raise ValueError(f"unknown transformation code: {code}")


def release_lag(name: str) -> int:
    return 2 if name in SLOW_SERIES else 1


def align(series: pd.Series, lag: int, last: pd.Period = config.LAST_MONTH) -> pd.Series:
    """Drop observations not public by ``last``, then index by forecast row k.

    An observation public ``lag`` months after its date enters row k = date + lag - 1.
    """
    public = series[series.index + lag <= last].copy()
    public.index = public.index + (lag - 1)
    return public


def read_md(path: Path = config.MD_FILE) -> tuple[pd.DataFrame, pd.Series]:
    raw = pd.read_csv(path)
    codes = raw.iloc[0, 1:].astype(float).astype(int)
    body = raw.iloc[1:].dropna(subset=["sasdate"]).copy()
    body.index = pd.PeriodIndex(pd.to_datetime(body.pop("sasdate"), format="%m/%d/%Y"), freq="M")
    return body.apply(pd.to_numeric, errors="coerce"), codes


def national_predictors(levels: pd.DataFrame, codes: pd.Series) -> pd.DataFrame:
    frames = {}
    for name in levels.columns:
        frames[name] = align(apply_tcode(levels[name], int(codes[name])), release_lag(name))
    return pd.DataFrame(frames).sort_index()


def national_levels(levels: pd.DataFrame, names: list[str]) -> pd.DataFrame:
    return pd.DataFrame({n: align(levels[n], release_lag(n)) for n in names}).sort_index()


def read_state_panel(path: Path = config.SD_FILE) -> dict[str, pd.DataFrame]:
    """Return one frame per variable. ``NA`` is a real FRED-SD column, not a missing value."""
    sheets = pd.read_excel(
        path, sheet_name=None, keep_default_na=False, na_values=[""], index_col=0
    )
    output: dict[str, dict[str, pd.Series]] = {v: {} for v in STATE_VARS}
    for state, frame in sheets.items():
        frame.index = pd.PeriodIndex(pd.to_datetime(frame.index, format="%m/%d/%Y"), freq="M")
        for variable in STATE_VARS:
            output[variable][state] = pd.to_numeric(frame[variable], errors="coerce")
    return {
        v: pd.DataFrame(states).loc[: config.LAST_MONTH - 1] for v, states in output.items()
    }


def load_state_panel() -> dict[str, pd.DataFrame]:
    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = config.CACHE_DIR / "state_panel.pkl"
    if cache.exists() and cache.stat().st_mtime >= config.SD_FILE.stat().st_mtime:
        return pd.read_pickle(cache)
    panel = read_state_panel()
    pd.to_pickle(panel, cache)
    return panel
