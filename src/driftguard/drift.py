"""Distribution-drift detection: PSI, KS, and per-feature drift reports.

This module is the *detector* half of the drift → retrain loop. The star of the
project is **prediction/concept drift**: on the bike-sharing data the model is
trained on 2011 and monitored on 2012, where demand grew ~63% (prediction
PSI ≈ 0.24, significant) while the weather features barely move (PSI ≈ 0.04).
Feature-only monitoring would miss that shift, so :class:`DriftReport` treats a
significant shift in the model's *predictions* as drift in its own right.

Everything here is pure numpy/pandas/scipy — no MLflow, no I/O.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

# Industry convention: PSI > 0.2 signals a significant population shift.
# (0.1 to 0.2 is a "moderate" warning band; below 0.1 is considered stable.)
PSI_THRESHOLD = 0.2

# Floor applied to bin proportions so empty bins never yield ln(0) = -inf or a
# 0/0 = nan when computing the PSI term.
_EPS = 1e-6


def psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index between a reference and a current sample.

    Bin edges are the ``bins``-quantiles of ``reference`` with the outer edges
    pushed to ``±inf`` so every ``current`` value lands in a bin (no
    out-of-range points). Bin proportions are clipped to a small epsilon so
    empty bins contribute a finite term instead of ``inf``/``nan``.

    Formula: ``PSI = sum((cur_prop - ref_prop) * ln(cur_prop / ref_prop))``.

    Parameters
    ----------
    reference:
        Baseline sample defining the bin edges (e.g. training-period values).
    current:
        Sample to compare against the baseline (e.g. production values).
    bins:
        Number of quantile bins to build from ``reference`` (default 10).

    Returns
    -------
    float
        A non-negative PSI value. ``0.0`` means the two samples fall
        identically across the reference quantile bins.
    """
    ref = np.asarray(reference, dtype=float).ravel()
    cur = np.asarray(current, dtype=float).ravel()

    if ref.size == 0 or cur.size == 0:
        return 0.0

    # Quantile edges from the reference. Duplicate edges (from ties / low
    # cardinality) collapse via np.unique; the outer edges become ±inf so the
    # full real line is covered.
    quantiles = np.linspace(0.0, 1.0, bins + 1)
    edges = np.unique(np.quantile(ref, quantiles))
    if edges.size < 2:
        # Reference is effectively constant -> a single bin, no structure to
        # compare against, so there is nothing to flag as drift.
        return 0.0
    edges[0] = -np.inf
    edges[-1] = np.inf

    ref_counts, _ = np.histogram(ref, bins=edges)
    cur_counts, _ = np.histogram(cur, bins=edges)

    ref_prop = ref_counts / ref_counts.sum()
    cur_prop = cur_counts / cur_counts.sum()

    ref_prop = np.clip(ref_prop, _EPS, None)
    cur_prop = np.clip(cur_prop, _EPS, None)

    value = float(np.sum((cur_prop - ref_prop) * np.log(cur_prop / ref_prop)))
    # Clipping can leave a tiny negative residual from floating-point noise;
    # PSI is defined as non-negative.
    return max(value, 0.0)


def ks(reference: np.ndarray, current: np.ndarray) -> tuple[float, float]:
    """Two-sample Kolmogorov-Smirnov test.

    Returns
    -------
    tuple[float, float]
        ``(statistic, p_value)`` from :func:`scipy.stats.ks_2samp`. A small
        p-value means the two samples are unlikely to share a distribution.
    """
    ref = np.asarray(reference, dtype=float).ravel()
    cur = np.asarray(current, dtype=float).ravel()
    if ref.size == 0 or cur.size == 0:
        return 0.0, 1.0
    result = ks_2samp(ref, cur)
    return float(result.statistic), float(result.pvalue)


@dataclass
class FeatureDrift:
    """Drift diagnostics for a single feature column."""

    feature: str
    psi: float
    ks_stat: float
    ks_pvalue: float
    drifted: bool

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "feature": self.feature,
            "psi": self.psi,
            "ks_stat": self.ks_stat,
            "ks_pvalue": self.ks_pvalue,
            "drifted": self.drifted,
        }


@dataclass
class DriftReport:
    """Aggregate drift report over features plus (optionally) predictions."""

    features: list[FeatureDrift]
    prediction_psi: float | None
    n_reference: int
    n_current: int
    psi_threshold: float

    @property
    def drifted(self) -> bool:
        """True if any feature drifted, or the prediction PSI is significant.

        The second clause is the concept-drift signal: even when every input
        feature looks stable, a significant shift in the model's predictions
        (``prediction_psi > psi_threshold``) counts as drift.
        """
        if any(feature.drifted for feature in self.features):
            return True
        return self.prediction_psi is not None and self.prediction_psi > self.psi_threshold

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation (for artifacts + /drift)."""
        return {
            "drifted": self.drifted,
            "features": [feature.to_dict() for feature in self.features],
            "prediction_psi": self.prediction_psi,
            "n_reference": self.n_reference,
            "n_current": self.n_current,
            "psi_threshold": self.psi_threshold,
        }


def feature_drift_report(
    reference_X: pd.DataFrame,
    current_X: pd.DataFrame,
    features: list[str],
    reference_pred: np.ndarray | None = None,
    current_pred: np.ndarray | None = None,
    psi_threshold: float = PSI_THRESHOLD,
) -> DriftReport:
    """Compute a :class:`DriftReport` comparing reference vs current data.

    For each column in ``features`` the PSI and KS statistics are computed; the
    feature is flagged ``drifted`` when its ``psi > psi_threshold``. When both
    ``reference_pred`` and ``current_pred`` are provided, ``prediction_psi`` is
    the PSI between them (the concept-drift signal); otherwise it is ``None``.

    Parameters
    ----------
    reference_X, current_X:
        Baseline and monitored feature frames (must contain ``features``).
    features:
        Column names to evaluate, in report order.
    reference_pred, current_pred:
        Optional model predictions on the reference and current data.
    psi_threshold:
        PSI above which a feature (or the prediction distribution) is drifted.
    """
    feature_reports: list[FeatureDrift] = []
    for feature in features:
        ref_values = reference_X[feature].to_numpy(dtype=float)
        cur_values = current_X[feature].to_numpy(dtype=float)
        feature_psi = psi(ref_values, cur_values)
        ks_stat, ks_pvalue = ks(ref_values, cur_values)
        feature_reports.append(
            FeatureDrift(
                feature=feature,
                psi=feature_psi,
                ks_stat=ks_stat,
                ks_pvalue=ks_pvalue,
                drifted=feature_psi > psi_threshold,
            )
        )

    prediction_psi: float | None = None
    if reference_pred is not None and current_pred is not None:
        prediction_psi = psi(np.asarray(reference_pred), np.asarray(current_pred))

    return DriftReport(
        features=feature_reports,
        prediction_psi=prediction_psi,
        n_reference=len(reference_X),
        n_current=len(current_X),
        psi_threshold=psi_threshold,
    )
