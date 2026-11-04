"""Walk-forward comparison of state distress models."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from fuelsignal.factors import Scaler, dfm_factors, pca_factors
from fuelsignal.regimes import stress_probability
from statedistress import config
from statedistress.features import EXPOSURE, HAND_MACRO, NATIONAL_TREND, STATE_JOBS, UR_FEATURES
from statedistress.labels import recession_window


@dataclass(frozen=True)
class Spec:
    name: str
    features: tuple[str, ...] = field(default_factory=tuple)
    engine: str | None = None
    k: int = config.N_FACTORS
    regime: bool = False
    interact: bool = False
    classifier: str = "logit"


BASE = tuple(UR_FEATURES)
TREND = (*BASE, *NATIONAL_TREND)
LOCAL = (*TREND, *STATE_JOBS)


def model_specs() -> list[Spec]:
    return [
        Spec("state_trend", BASE),
        Spec("state_national", TREND),
        Spec("state_local", LOCAL),
        Spec("hand_macro", (*LOCAL, *HAND_MACRO)),
        Spec("pca", LOCAL, "pca"),
        Spec("pca_exposure", LOCAL, "pca", interact=True),
        Spec("pca_regime", LOCAL, "pca", regime=True),
        Spec("dfm", LOCAL, "dfm"),
        Spec("dfm_exposure", LOCAL, "dfm", interact=True),
        Spec("pca_boosted", LOCAL, "pca", classifier="boost"),
        *[Spec(f"pca_k{k}", LOCAL, "pca", k=k) for k in config.SENSITIVITY_FACTORS],
    ]


def windows(months: pd.PeriodIndex) -> list[tuple[pd.Period, pd.Period, pd.Period, pd.Period]]:
    """Yearly refits: (data_end, label_end, test_start, test_end).

    The model for year Y is fit at the end of December Y-1, when row November Y-1 is the newest
    public. A label is known once row k + horizon is public, so training ends at
    December Y-1 minus horizon minus one month.
    """
    out = []
    last = months.max()
    for year in range(config.FIRST_TEST_YEAR, last.year + 1):
        fit_date = pd.Period(f"{year - 1}-12", "M")
        out.append(
            (
                fit_date - 1,
                fit_date - 1 - config.HORIZON_MONTHS,
                pd.Period(f"{year}-01", "M"),
                min(pd.Period(f"{year}-12", "M"), last),
            )
        )
    return out


def clip_outliers(train: pd.DataFrame, full: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Limit each series to +/- OUTLIER_SD training standard deviations."""
    scaler = Scaler.fit(train)
    low = scaler.mean - config.OUTLIER_SD * scaler.std
    high = scaler.mean + config.OUTLIER_SD * scaler.std
    return train.clip(low, high, axis=1), full.clip(low, high, axis=1)


def factors_for(
    predictors: pd.DataFrame, spec: Spec, data_end: pd.Period, test_end: pd.Period
) -> pd.DataFrame:
    train, full = clip_outliers(predictors.loc[:data_end], predictors.loc[:test_end])
    if spec.engine == "dfm":
        return dfm_factors(train, full, spec.k, config.DFM_MAX_ITER)
    factors, _ = pca_factors(train, full, spec.k, config.SEED)
    return factors


def _classifier(kind: str):
    if kind == "logit":
        return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=3000))
    return HistGradientBoostingClassifier(
        max_depth=2, learning_rate=0.05, max_iter=150, random_state=config.SEED
    )


def walk_forward(
    design: pd.DataFrame, predictors: pd.DataFrame, specs: list[Spec] | None = None
) -> pd.DataFrame:
    specs = specs or model_specs()
    months = pd.PeriodIndex(design["month"].unique(), freq="M")
    outputs = []
    keep = ["month", "state", "distress_label", "onset"]
    for data_end, label_end, test_start, test_end in windows(months):
        cache: dict[tuple, pd.DataFrame] = {}
        for spec in specs:
            frame = design
            columns = list(spec.features)
            if spec.engine:
                key = (spec.engine, spec.k)
                if key not in cache:
                    cache[key] = factors_for(predictors, spec, data_end, test_end)
                factors = cache[key]
                extra = list(factors.columns)
                if spec.regime:
                    stress = stress_probability(
                        factors.loc[:data_end], factors, config.N_REGIMES, config.SEED
                    )
                    factors = factors.assign(stress_prob=stress)
                    extra.append("stress_prob")
                frame = design.merge(factors, left_on="month", right_index=True, how="left")
                columns += extra
                if spec.interact:
                    products = {
                        f"{f}_x_{e}": frame[f] * frame[e]
                        for f in factors.columns
                        if f.startswith("factor")
                        for e in EXPOSURE
                    }
                    frame = frame.assign(**products)
                    columns += list(products)
            train = frame[frame["month"] <= label_end].dropna(subset=columns)
            test = frame[frame["month"].between(test_start, test_end)].dropna(subset=columns)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = _classifier(spec.classifier).fit(
                    train[columns], train["distress_label"].astype(int)
                )
            result = test[keep].copy()
            result["model"] = spec.name
            result["probability"] = model.predict_proba(test[columns])[:, 1]
            outputs.append(result)
        print(f"finished {test_start.year}", flush=True)
    return pd.concat(outputs, ignore_index=True)


def score(frame: pd.DataFrame) -> dict[str, float]:
    y, p = frame["distress_label"].astype(int), frame["probability"]
    alerts = p >= config.ALERT_THRESHOLD
    return {
        "rows": len(frame),
        "positives": int(y.sum()),
        "pr_auc": average_precision_score(y, p),
        "roc_auc": roc_auc_score(y, p),
        "brier": brier_score_loss(y, p),
        "precision": float(y[alerts].mean()) if alerts.any() else float("nan"),
        "recall": float(alerts[y == 1].mean()),
    }


def episode_summary(frame: pd.DataFrame) -> dict[str, float]:
    """Share of state episodes with an alert before entry, and the median lead in data months."""
    leads, total = [], 0
    for (_, onset), group in frame.dropna(subset=["onset"]).groupby(["state", "onset"]):
        total += 1
        alerted = group[group["probability"] >= config.ALERT_THRESHOLD].sort_values("month")
        if len(alerted):
            leads.append((onset - alerted.iloc[0]["month"]).n)
    return {
        "episodes": total,
        "warned": len(leads),
        "episode_recall": len(leads) / total if total else float("nan"),
        "median_lead": float(np.median(leads)) if leads else float("nan"),
    }


def subsets(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    month = pd.PeriodIndex(frame["month"], freq="M")
    covid = (month >= pd.Period("2019-10", "M")) & (month <= pd.Period("2020-06", "M"))
    away = ~recession_window(month).values
    return {"all": frame, "excluding_covid": frame[~covid], "away_from_recessions": frame[away]}


def bootstrap_gap(
    frame: pd.DataFrame, model: str, baseline: str, draws: int = 1000, seed: int = config.SEED
) -> dict[str, float]:
    """PR-AUC gap with a 90% interval, resampling whole test years because states co-move."""
    keys = ["month", "state"]
    a = frame[frame["model"] == model].sort_values(keys).reset_index(drop=True)
    b = frame[frame["model"] == baseline].sort_values(keys).reset_index(drop=True)
    merged = a[[*keys, "distress_label", "probability"]].merge(
        b[[*keys, "probability"]], on=keys, suffixes=("_a", "_b")
    )
    years = pd.PeriodIndex(merged["month"], freq="M").year.values
    y = merged["distress_label"].astype(int).values
    p, q = merged["probability_a"].values, merged["probability_b"].values
    unique = np.unique(years)
    rows_by_year = {year: np.flatnonzero(years == year) for year in unique}
    rng = np.random.default_rng(seed)
    gaps = []
    for _ in range(draws):
        rows = np.concatenate([rows_by_year[v] for v in rng.choice(unique, len(unique))])
        if y[rows].sum() == 0:
            continue
        gaps.append(
            average_precision_score(y[rows], p[rows]) - average_precision_score(y[rows], q[rows])
        )
    return {
        "gap": average_precision_score(y, p) - average_precision_score(y, q),
        "ci_low": float(np.quantile(gaps, 0.05)),
        "ci_high": float(np.quantile(gaps, 0.95)),
    }
