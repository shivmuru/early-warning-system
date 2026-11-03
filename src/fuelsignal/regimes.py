"""Gaussian HMM regimes, read with a forward filter so probabilities never use future data."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from scipy.special import logsumexp
from scipy.stats import norm


def forward_filter(
    obs: np.ndarray,
    start: np.ndarray,
    transition: np.ndarray,
    means: np.ndarray,
    variances: np.ndarray,
) -> np.ndarray:
    """P(state at t | observations through t) for a diagonal Gaussian HMM."""
    log_density = norm.logpdf(obs[:, None, :], means[None], np.sqrt(variances)[None]).sum(axis=2)
    out = np.zeros((len(obs), len(start)))
    predicted = np.log(start + 1e-300)
    for t in range(len(obs)):
        joint = predicted + log_density[t]
        filtered = joint - logsumexp(joint)
        out[t] = np.exp(filtered)
        predicted = logsumexp(filtered[:, None] + np.log(transition + 1e-300), axis=0)
    return out


def stress_probability(
    train: pd.DataFrame, full: pd.DataFrame, n_states: int = 2, seed: int = 0
) -> pd.Series:
    """Filtered probability of the highest-variance state, fit on ``train`` only."""
    model = GaussianHMM(
        n_components=n_states, covariance_type="diag", n_iter=200, random_state=seed
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(train.values)
    variances = np.array([np.diag(c) for c in model.covars_])
    probabilities = forward_filter(
        full.values, model.startprob_, model.transmat_, model.means_, variances
    )
    stress_state = int(variances.mean(axis=1).argmax())
    return pd.Series(probabilities[:, stress_state], index=full.index, name="stress_prob")
