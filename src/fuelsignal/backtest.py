"""Walk-forward comparison of models, with every estimate limited to data public at the time."""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from fuelsignal import config
from fuelsignal.data import SERIES
from fuelsignal.factors import dfm_factors, pca_factors
from fuelsignal.regimes import stress_probability

ENERGY_ONLY = [s.name for s in SERIES if s.group != "macro_financial"]
LOCAL = ["price_chg_1m", "price_chg_3m", "price_vs_avg"]
CRUDE = ["brent_chg_1m", "brent_chg_3m", "brent_vs_avg", "wti_chg_3m"]


@dataclass(frozen=True)
class ModelSpec:
    name: str
    features: tuple[str, ...]
    factor_engine: str | None = None
    columns: tuple[str, ...] | None = None
    k: int = config.N_FACTORS
    classifier: str = "logit"


def model_specs() -> list[ModelSpec]:
    energy = tuple(ENERGY_ONLY)
    return [
        ModelSpec("calendar", ("calendar",)),
        ModelSpec("crude", ("calendar", "crude")),
        ModelSpec("local", ("calendar", "local")),
        ModelSpec("local_crude", ("calendar", "local", "crude")),
        ModelSpec("energy_factors", ("calendar", "local", "factors"), "dfm", energy),
        ModelSpec("all_factors", ("calendar", "local", "factors"), "dfm"),
        ModelSpec("all_factors_pca", ("calendar", "local", "factors"), "pca"),
        ModelSpec("all_factors_boosted", ("calendar", "local", "factors"), "dfm", None,
                  config.N_FACTORS, "boost"),
        *[
            ModelSpec(f"all_factors_k{k}", ("calendar", "local", "factors"), "dfm", None, k)
            for k in config.SENSITIVITY_FACTORS
        ],
    ]


def local_features(prices: pd.DataFrame, regions: list[str]) -> pd.DataFrame:
    """Price signals in nominal terms: same-month CPI was not yet public at the forecast date."""
    frames = []
    for region in regions:
        log_price = np.log(prices[region])
        frames.append(
            pd.DataFrame(
                {
                    "month": prices.index,
                    "region": region,
                    "price_chg_1m": 100 * log_price.diff(1).values,
                    "price_chg_3m": 100 * log_price.diff(3).values,
                    "price_vs_avg": 100
                    * (log_price - log_price.rolling(config.LOOKBACK_MONTHS).mean()).values,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def crude_features(levels: pd.DataFrame) -> pd.DataFrame:
    brent, wti = np.log(levels["brent_spot"]), np.log(levels["wti_spot"])
    return pd.DataFrame(
        {
            "brent_chg_1m": 100 * brent.diff(1),
            "brent_chg_3m": 100 * brent.diff(3),
            "brent_vs_avg": 100 * (brent - brent.rolling(config.LOOKBACK_MONTHS).mean()),
            "wti_chg_3m": 100 * wti.diff(3),
        }
    )


def build_design(
    panel: pd.DataFrame, prices: pd.DataFrame, levels: pd.DataFrame
) -> tuple[pd.DataFrame, list[str]]:
    regions = sorted(r for r in panel["region"].unique() if r != "us_national")
    frame = panel[panel["region"].isin(regions)].copy()
    frame["month"] = pd.PeriodIndex(frame["month"], freq="M")
    frame = frame.merge(local_features(prices, regions), on=["month", "region"], how="left")
    frame = frame.merge(crude_features(levels), left_on="month", right_index=True, how="left")
    angle = 2 * np.pi * frame["month"].dt.month / 12
    frame["season_sin"], frame["season_cos"] = np.sin(angle), np.cos(angle)
    calendar = ["season_sin", "season_cos"]
    for region in regions:
        frame[f"region_{region}"] = (frame["region"] == region).astype(float)
        calendar.append(f"region_{region}")
    return frame.reset_index(drop=True), calendar


def training_windows(
    months: pd.PeriodIndex, first_year: int = config.FIRST_TEST_YEAR
) -> list[tuple[pd.Period, pd.Period, pd.Period, pd.Period]]:
    """Yearly refits. Returns (data_end, label_end, test_start, test_end).

    A model used for year Y is fit at the end of Y-1. A training label needs its full outcome
    window, so the last usable training row is three months before that date.
    """
    windows = []
    last = months.max()
    for year in range(first_year, last.year + 1):
        test_start = pd.Period(f"{year}-01", "M")
        data_end = test_start - 1
        label_end = data_end - config.HORIZON_MONTHS
        test_end = min(pd.Period(f"{year}-12", "M"), last)
        windows.append((data_end, label_end, test_start, test_end))
    return windows


def _factor_features(
    predictors: pd.DataFrame, spec: ModelSpec, data_end: pd.Period, test_end: pd.Period
) -> pd.DataFrame:
    columns = list(spec.columns) if spec.columns else list(predictors.columns)
    train = predictors.loc[: data_end, columns]
    full = predictors.loc[: test_end, columns]
    if spec.factor_engine == "dfm":
        factors = dfm_factors(train, full, spec.k)
    else:
        factors, _ = pca_factors(train, full, spec.k, config.SEED)
    stress = stress_probability(
        factors.loc[: data_end], factors, config.N_REGIMES, config.SEED
    )
    return factors.assign(stress_prob=stress)


def _classifier(kind: str):
    if kind == "logit":
        return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))
    return HistGradientBoostingClassifier(
        max_depth=2, learning_rate=0.05, max_iter=100, random_state=config.SEED
    )


def _resolve_columns(spec: ModelSpec, calendar: list[str], factor_cols: list[str]) -> list[str]:
    groups = {"calendar": calendar, "local": LOCAL, "crude": CRUDE, "factors": factor_cols}
    return [column for group in spec.features for column in groups[group]]


def walk_forward(design: pd.DataFrame, calendar: list[str], predictors: pd.DataFrame,
                 specs: list[ModelSpec] | None = None) -> pd.DataFrame:
    """Out-of-sample probabilities for every model and labeled test row."""
    specs = specs or model_specs()
    labeled = design.dropna(subset=["distress_label"])
    outputs = []
    months = pd.PeriodIndex(labeled["month"].unique(), freq="M")
    for data_end, label_end, test_start, test_end in training_windows(months):
        factor_cache: dict[tuple, pd.DataFrame] = {}
        for spec in specs:
            frame = labeled
            factor_cols: list[str] = []
            if spec.factor_engine:
                key = (spec.factor_engine, spec.columns, spec.k)
                if key not in factor_cache:
                    factor_cache[key] = _factor_features(predictors, spec, data_end, test_end)
                factors = factor_cache[key]
                factor_cols = list(factors.columns)
                frame = labeled.merge(factors, left_on="month", right_index=True, how="left")
            columns = _resolve_columns(spec, calendar, factor_cols)
            train = frame[frame["month"] <= label_end].dropna(subset=columns)
            test = frame[frame["month"].between(test_start, test_end)]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = _classifier(spec.classifier).fit(
                    train[columns], train["distress_label"].astype(int)
                )
            result = test[["month", "region", "distress_label", "months_to_distress",
                           "episode_id"]].copy()
            result["model"] = spec.name
            result["probability"] = model.predict_proba(test[columns])[:, 1]
            outputs.append(result)
    return pd.concat(outputs, ignore_index=True)


def score(frame: pd.DataFrame) -> dict[str, float]:
    y, p = frame["distress_label"].astype(int), frame["probability"]
    alerts = p >= config.ALERT_THRESHOLD
    return {
        "pr_auc": average_precision_score(y, p),
        "roc_auc": roc_auc_score(y, p),
        "brier": brier_score_loss(y, p),
        "alert_rate": float(alerts.mean()),
        "precision_at_threshold": float(y[alerts].mean()) if alerts.any() else float("nan"),
        "recall_at_threshold": float(alerts[y == 1].mean()),
    }


def episode_summary(frame: pd.DataFrame) -> dict[str, float]:
    """Did an alert come before each regional episode's first qualifying price?"""
    leads, total = [], 0
    for _, group in frame.dropna(subset=["episode_id"]).groupby(["region", "episode_id"]):
        group = group.sort_values("month")
        total += 1
        first = group.iloc[0]
        first_distress = first["month"] + int(first["months_to_distress"])
        early = group[(group["probability"] >= config.ALERT_THRESHOLD)
                      & (group["month"] < first_distress)]
        if len(early):
            leads.append((first_distress - early.iloc[0]["month"]).n)
    return {
        "region_episodes": total,
        "episodes_warned_early": len(leads),
        "episode_recall": len(leads) / total if total else float("nan"),
        "median_lead_months": float(np.median(leads)) if leads else float("nan"),
    }


def bootstrap_gap(
    frame: pd.DataFrame, model: str, baseline: str, draws: int = 2000, seed: int = config.SEED
) -> dict[str, float]:
    """PR-AUC gap with a 90% interval, resampling whole test years because regions co-move."""
    a = frame[frame["model"] == model].sort_values(["month", "region"]).reset_index(drop=True)
    b = frame[frame["model"] == baseline].sort_values(["month", "region"]).reset_index(drop=True)
    years = a["month"].dt.year.values
    y = a["distress_label"].astype(int).values
    p, q = a["probability"].values, b["probability"].values
    unique = np.unique(years)
    positions = {year: np.flatnonzero(years == year) for year in unique}
    rng = np.random.default_rng(seed)
    gaps = []
    for _ in range(draws):
        rows = np.concatenate([positions[v] for v in rng.choice(unique, len(unique))])
        if y[rows].sum() == 0:
            continue
        gaps.append(average_precision_score(y[rows], p[rows])
                    - average_precision_score(y[rows], q[rows]))
    return {
        "gap": average_precision_score(y, p) - average_precision_score(y, q),
        "ci_low": float(np.quantile(gaps, 0.05)),
        "ci_high": float(np.quantile(gaps, 0.95)),
    }
