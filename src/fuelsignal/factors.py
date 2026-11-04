"""PCA and Dynamic Factor Model factors, always estimated on a training window only."""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from statsmodels.tsa.statespace.dynamic_factor_mq import DynamicFactorMQ


@dataclass
class Scaler:
    mean: pd.Series
    std: pd.Series

    @classmethod
    def fit(cls, frame: pd.DataFrame) -> Scaler:
        return cls(frame.mean(), frame.std().replace(0, 1))

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        return (frame - self.mean) / self.std


def pca_factors(
    train: pd.DataFrame, full: pd.DataFrame, k: int, seed: int = 0
) -> tuple[pd.DataFrame, PCA]:
    """Fit on ``train`` and project every row of ``full`` using those fixed weights."""
    scaler = Scaler.fit(train)
    model = PCA(n_components=k, random_state=seed).fit(scaler.transform(train).fillna(0.0))
    scores = model.transform(scaler.transform(full).fillna(0.0))
    columns = [f"factor_{i + 1}" for i in range(k)]
    return pd.DataFrame(scores, index=full.index, columns=columns), model


def dfm_factors(
    train: pd.DataFrame, full: pd.DataFrame, k: int, maxiter: int = 200
) -> pd.DataFrame:
    """Fit a DFM on ``train``, then return filtered factors for ``full``.

    Filtered values use only observations up to each row, unlike smoothed values, which would
    leak future data into the features.
    """
    scaler = Scaler.fit(train)
    train_z, full_z = scaler.transform(train), scaler.transform(full)
    train_z.index = train_z.index.to_timestamp()
    full_z.index = full_z.index.to_timestamp()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = DynamicFactorMQ(train_z, factors=k, factor_orders=1, standardize=False)
        result = model.fit(disp=False, maxiter=maxiter)
        filtered = result.apply(full_z).factors.filtered
    columns = [f"factor_{i + 1}" for i in range(k)]
    return pd.DataFrame(filtered.values, index=full.index, columns=columns)


def bai_ng_icp2(frame: pd.DataFrame, k_max: int = 8) -> int:
    """Bai-Ng IC_p2 estimate of the number of factors."""
    x = Scaler.fit(frame).transform(frame).fillna(0.0).values
    t, n = x.shape
    u, s, vt = np.linalg.svd(x, full_matrices=False)
    best_k, best_ic = 0, np.inf
    for k in range(1, k_max + 1):
        fitted = (u[:, :k] * s[:k]) @ vt[:k]
        penalty = k * (n + t) / (n * t) * np.log(min(n, t))
        criterion = np.log(((x - fitted) ** 2).mean()) + penalty
        if criterion < best_ic:
            best_k, best_ic = k, criterion
    return best_k
