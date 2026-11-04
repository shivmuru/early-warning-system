"""Pre-specified expansion from 17 to 34 predictors, with PCA run separately per group."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from fuelsignal import config
from fuelsignal.data import SERIES, TRANSFORMS, Series, as_of_monthly, read_eia, read_fred


def _fred(name: str, group: str, unit: str, lag: int, transform: str) -> Series:
    return Series(name, group, f"{name}.csv", None, unit, lag, transform)


def _eia(name: str, group: str, transform: str) -> Series:
    return Series(name, group, f"{name}w.xls", None, "days", 5, transform)


NEW_SERIES = (
    _fred("NFCI", "macro_financial", "days", 5, "level"),
    _fred("NFCICREDIT", "macro_financial", "days", 5, "level"),
    _fred("WALCL", "macro_financial", "days", 2, "yoy"),
    _fred("M2SL", "macro_financial", "months", 2, "yoy"),
    _fred("T10Y3M", "macro_financial", "days", 1, "level"),
    _fred("DTWEXEMEGS", "macro_financial", "days", 7, "dlog"),
    _fred("PCOPPUSDM", "global_demand", "months", 2, "dlog"),
    _fred("GSCPI", "global_demand", "months", 1, "level"),
    _fred("INDPRO", "global_demand", "months", 1, "yoy"),
    _fred("UMCSENT", "global_demand", "months", 1, "diff"),
    _eia("WPULEUS3", "supply", "yoy"),
    _eia("WCRFPUS2", "supply", "yoy"),
    _eia("WGFUPUS2", "supply", "yoy"),
    _eia("WDIUPUS2", "supply", "yoy"),
    _eia("WCRIMUS2", "supply", "yoy"),
    _eia("EER_EPMRU_PF4_Y35NY_DPG", "energy_price", "dlog"),
    _eia("EER_EPD2DXL0_PF4_Y35NY_DPG", "energy_price", "dlog"),
)


def missing_files(raw_dir: Path = config.RAW_DIR) -> list[str]:
    return [s.file for s in NEW_SERIES if not list(raw_dir.glob(f"*{s.file}"))]


def load_new_predictors(raw_dir: Path = config.RAW_DIR) -> pd.DataFrame:
    """Transformed new predictors. The usual release rules and cutoff apply."""
    output = {}
    for spec in NEW_SERIES:
        path = next(raw_dir.glob(f"*{spec.file}"))
        series = read_fred(path) if spec.file.endswith(".csv") else read_eia(path).iloc[:, 0]
        output[spec.name] = TRANSFORMS[spec.transform](
            as_of_monthly(series, spec.lag_unit, spec.lag)
        )
    return pd.DataFrame(output).loc[config.MODEL_START :]


def group_map(extended: bool) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    for spec in (*SERIES, *NEW_SERIES) if extended else SERIES:
        groups.setdefault(spec.group, []).append(spec.name)
    return groups
