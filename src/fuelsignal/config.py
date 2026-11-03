"""Settings fixed before any backtest result was seen."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
REPORTS_DIR = ROOT / "reports"

# Hard information cutoff: nothing released after this date may enter any output.
CUTOFF = pd.Timestamp("2025-11-30")
HISTORY_START = pd.Period("1993-01", "M")
LAST_MONTH = pd.Period("2025-11", "M")
MODEL_START = pd.Period("2006-01", "M")

# Distress label
RISE = 0.15
LOOKBACK_MONTHS = 36
HORIZON_MONTHS = 3

# Backtest
FIRST_TEST_YEAR = 2012
N_FACTORS = 4
SENSITIVITY_FACTORS = (3, 5)
N_REGIMES = 2
ALERT_THRESHOLD = 0.30
SEED = 7
