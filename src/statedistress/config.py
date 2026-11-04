"""Settings fixed before any state-distress backtest result was seen."""

from __future__ import annotations

import pandas as pd

from fuelsignal.config import PROCESSED_DIR, RAW_DIR, REPORTS_DIR

MD_FILE = RAW_DIR / "2026-10-MD.csv"
SD_FILE = RAW_DIR / "State_2026-04.xlsx"
CACHE_DIR = PROCESSED_DIR / "state_distress"
OUTPUT_DIR = REPORTS_DIR / "state_distress"

# Nothing public after this date may enter an output.
CUTOFF = pd.Timestamp("2025-11-30")
LAST_MONTH = pd.Period("2025-11", "M")
UR_START = pd.Period("1976-01", "M")

# Label
RISE_POINTS = 1.0
HORIZON_MONTHS = 6

# Backtest
FIRST_TEST_YEAR = 2000
N_FACTORS = 5
SENSITIVITY_FACTORS = (3, 8)
N_REGIMES = 2
ALERT_THRESHOLD = 0.25
OUTLIER_SD = 5.0
DFM_MAX_ITER = 100
SEED = 7

# Used only to evaluate results, never as a model input.
RECESSIONS = (
    ("1980-01", "1980-07"),
    ("1981-07", "1982-11"),
    ("1990-07", "1991-03"),
    ("2001-03", "2001-11"),
    ("2007-12", "2009-06"),
    ("2020-02", "2020-04"),
)
