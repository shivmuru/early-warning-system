"""Load raw files and build point-in-time monthly predictors and regional prices."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import xlrd

from fuelsignal import config

STOCK = "Weekly U.S. Ending Stocks "
IMPORT = "U.S. Imports {} of Crude Oil and Petroleum Products (Thousand Barrels)"
RETAIL = "PET_PRI_GND_A_EPM0_PTE_DPGAL_W.xls"
STOCKS = "PET_STOC_WSTK_DCU_NUS_W.xls"
IMPORTS = "PET_MOVE_IMPCUS_A2_NUS_EP00_IM0_MBBL_M.xls"
PRODUCTION = "PET_CRD_CRPDN_ADC_MBBL_M.xls"


@dataclass(frozen=True)
class Series:
    """One predictor and the rule for when each observation became public."""

    name: str
    group: str
    file: str
    column: str | None
    lag_unit: str
    lag: int
    transform: str


SERIES = (
    Series("brent_spot", "energy_price", "RBRTEw.xls", None, "days", 5, "dlog"),
    Series("wti_spot", "energy_price", "RWTCw.xls", None, "days", 5, "dlog"),
    Series("henry_hub_spot", "energy_price", "RNGWHHDw.xls", None, "days", 5, "dlog"),
    Series("real_broad_dollar", "macro_financial", "RTWEXBGS.csv", None, "months", 1, "dlog"),
    Series("fed_funds", "macro_financial", "FEDFUNDS.csv", None, "months", 1, "diff"),
    Series("curve_10y2y", "macro_financial", "T10Y2Y.csv", None, "days", 1, "level"),
    Series("vix", "macro_financial", "VIXCLS.csv", None, "days", 1, "log"),
    Series(
        "stocks_crude_ex_spr", "inventory", STOCKS,
        STOCK + "excluding SPR of Crude Oil  (Thousand Barrels)", "days", 5, "yoy",
    ),
    Series(
        "stocks_spr", "inventory", STOCKS,
        STOCK + "of Crude Oil in SPR  (Thousand Barrels)", "days", 5, "yoy",
    ),
    Series(
        "stocks_gasoline", "inventory", STOCKS,
        STOCK + "of Total Gasoline  (Thousand Barrels)", "days", 5, "yoy",
    ),
    Series(
        "stocks_distillate", "inventory", STOCKS,
        STOCK + "of Distillate Fuel Oil  (Thousand Barrels)", "days", 5, "yoy",
    ),
    Series(
        "stocks_jet", "inventory", STOCKS,
        STOCK + "of Kerosene-Type Jet Fuel  (Thousand Barrels)", "days", 5, "yoy",
    ),
    Series(
        "stocks_residual", "inventory", STOCKS,
        STOCK + "of Residual Fuel Oil  (Thousand Barrels)", "days", 5, "yoy",
    ),
    Series(
        "us_crude_production", "supply", PRODUCTION,
        "U.S. Field Production of Crude Oil (Thousand Barrels)", "months", 3, "yoy",
    ),
    Series("imports_total", "supply", IMPORTS, IMPORT.format("").replace("  ", " "),
           "months", 3, "yoy"),
    Series("imports_persian_gulf", "supply", IMPORTS,
           IMPORT.format("from Persian Gulf Countries"), "months", 3, "yoy"),
    Series("imports_canada", "supply", IMPORTS, IMPORT.format("from Canada"), "months", 3, "yoy"),
)

REGIONS = {
    "new_england": "New England (PADD 1A)",
    "central_atlantic": "Central Atlantic (PADD 1B)",
    "lower_atlantic": "Lower Atlantic (PADD 1C)",
    "midwest": "Midwest",
    "gulf_coast": "Gulf Coast",
    "rocky_mountain": "Rocky Mountain",
    "west_coast": "West Coast All",
}

TRANSFORMS: dict[str, Callable[[pd.Series], pd.Series]] = {
    "dlog": lambda s: 100 * np.log(s).diff(),
    "diff": lambda s: s.diff(),
    "level": lambda s: s,
    "log": np.log,
    "yoy": lambda s: 100 * np.log(s).diff(12),
}


def _find(raw_dir: Path, name: str) -> Path:
    """Match a file by source name, allowing a prefix added on download."""
    matches = sorted(raw_dir.glob(f"*{name}"))
    if not matches:
        raise FileNotFoundError(f"{name} not found in {raw_dir}; see data/MANIFEST.csv")
    return matches[0]


def read_eia(path: Path, sheet: str = "Data 1") -> pd.DataFrame:
    book = xlrd.open_workbook(path).sheet_by_name(sheet)
    frame = pd.DataFrame(
        [book.row_values(r) for r in range(3, book.nrows)], columns=book.row_values(2)
    )
    frame.index = pd.to_datetime([xlrd.xldate_as_datetime(v, 0) for v in frame.iloc[:, 0]])
    return frame.iloc[:, 1:].apply(pd.to_numeric, errors="coerce")


def read_fred(path: Path) -> pd.Series:
    frame = pd.read_csv(path, parse_dates=["observation_date"], index_col=0)
    return pd.to_numeric(frame.iloc[:, 0], errors="coerce")


def load_raw(raw_dir: Path = config.RAW_DIR) -> dict[str, pd.Series]:
    """Return every raw series used by the pipeline, keyed by a stable name."""
    raw: dict[str, pd.Series] = {}
    for spec in SERIES:
        path = _find(raw_dir, spec.file)
        if spec.file.endswith(".csv"):
            raw[spec.name] = read_fred(path)
        else:
            frame = read_eia(path)
            raw[spec.name] = frame.iloc[:, 0] if spec.column is None else frame[spec.column]
    retail = _find(raw_dir, RETAIL)
    regions = read_eia(retail, "Data 2")
    for region, key in REGIONS.items():
        column = next(c for c in regions.columns if c.startswith(f"Weekly {key}"))
        raw[f"price_{region}"] = regions[column]
    raw["price_us_national"] = read_eia(retail, "Data 1").iloc[:, 0]
    raw["cpi"] = read_fred(_find(raw_dir, "CPIAUCSL.csv"))
    return raw


def months_index() -> pd.PeriodIndex:
    return pd.period_range(config.HISTORY_START, config.LAST_MONTH, freq="M")


def as_of_monthly(
    series: pd.Series, lag_unit: str, lag: int, cutoff: pd.Timestamp = config.CUTOFF
) -> pd.Series:
    """Latest value public at each month-end, dropping anything released after the cutoff."""
    months = months_index()
    series = series.dropna().sort_index()
    if lag_unit == "days":
        public = series.copy()
        public.index = public.index + pd.Timedelta(days=lag)
        public = public[public.index <= cutoff]
        last = public.groupby(public.index.to_period("M")).last()
        return last.reindex(months).ffill(limit=1)
    if lag_unit == "months":
        public = series.copy()
        public.index = public.index.to_period("M") + lag
        return public[public.index <= cutoff.to_period("M")].reindex(months)
    raise ValueError(f"unknown lag unit: {lag_unit}")


def build_levels(raw: dict[str, pd.Series]) -> pd.DataFrame:
    return pd.DataFrame(
        {s.name: as_of_monthly(raw[s.name], s.lag_unit, s.lag) for s in SERIES}
    )


def build_predictors(levels: pd.DataFrame) -> pd.DataFrame:
    frame = pd.DataFrame({s.name: TRANSFORMS[s.transform](levels[s.name]) for s in SERIES})
    return frame.loc[config.MODEL_START :]


def build_prices(raw: dict[str, pd.Series]) -> pd.DataFrame:
    """Retail prices are published about a day after their Monday survey date."""
    names = [*REGIONS, "us_national"]
    return pd.DataFrame({r: as_of_monthly(raw[f"price_{r}"], "days", 1) for r in names})


def build_cpi(raw: dict[str, pd.Series]) -> pd.Series:
    """CPI for deflating outcomes. September 2025 was the last print public at the cutoff."""
    cpi = raw["cpi"].copy()
    cpi.index = cpi.index.to_period("M")
    return cpi[cpi.index <= pd.Period("2025-09", "M")].reindex(months_index()).ffill()


def data_dictionary() -> pd.DataFrame:
    labels = {
        "dlog": "100 x monthly log change",
        "diff": "monthly change",
        "level": "level",
        "log": "log level",
        "yoy": "100 x 12-month log change",
    }
    return pd.DataFrame(
        [
            {
                "series": s.name,
                "group": s.group,
                "source_file": s.file,
                "public_after": f"{s.lag} {s.lag_unit}",
                "transform": labels[s.transform],
            }
            for s in SERIES
        ]
    )
