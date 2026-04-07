"""Tests for driftguard.drift: PSI, KS, and the concept-drift report."""

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from driftguard import schema
from driftguard.drift import (
    PSI_THRESHOLD,
    DriftReport,
    FeatureDrift,
    feature_drift_report,
    ks,
    psi,
)


def test_psi_zero_for_identical_samples() -> None:
    rng = np.random.default_rng(0)
    sample = rng.normal(size=2000)
    # Same array compared to itself -> perfectly stable populations.
    assert psi(sample, sample) == 0.0


def test_psi_near_zero_for_same_distribution() -> None:
    rng = np.random.default_rng(1)
    ref = rng.normal(size=5000)
    cur = rng.normal(size=5000)
    # Independent draws from the same distribution: PSI should be well below
    # the "moderate" (0.1) band.
    assert psi(ref, cur) < 0.1


def test_psi_large_for_shifted_distribution() -> None:
    rng = np.random.default_rng(2)
    ref = rng.normal(loc=0.0, size=5000)
    cur = rng.normal(loc=3.0, size=5000)
    # A three-sigma mean shift is a big, unmistakable population move.
    assert psi(ref, cur) > PSI_THRESHOLD


def test_psi_is_finite_with_empty_bins() -> None:
    # current lives entirely inside one reference bin -> every other bin is
    # empty. Epsilon clipping must keep the result finite (no inf/nan).
    ref = np.linspace(0.0, 100.0, 1000)
    cur = np.full(500, 50.0)
    value = psi(ref, cur)
    assert math.isfinite(value)
    assert value >= 0.0


def test_psi_finite_when_current_outside_reference_range() -> None:
    # ±inf outer edges must catch current values beyond the reference span.
    ref = np.linspace(0.0, 1.0, 1000)
    cur = np.linspace(5.0, 10.0, 1000)
    value = psi(ref, cur)
    assert math.isfinite(value)
    assert value > PSI_THRESHOLD


def test_psi_constant_reference_returns_zero() -> None:
    ref = np.full(100, 7.0)
    cur = np.arange(100, dtype=float)
    # No structure in a constant reference -> nothing to compare, and no crash.
    assert psi(ref, cur) == 0.0


def test_psi_empty_input_returns_zero() -> None:
    assert psi(np.array([]), np.array([1.0, 2.0])) == 0.0
    assert psi(np.array([1.0, 2.0]), np.array([])) == 0.0


def test_psi_non_negative_and_symmetric_magnitude() -> None:
    rng = np.random.default_rng(3)
    ref = rng.normal(size=3000)
    cur = rng.normal(loc=1.0, size=3000)
    assert psi(ref, cur) >= 0.0


def test_ks_matches_scipy() -> None:
    rng = np.random.default_rng(4)
    ref = rng.normal(size=500)
    cur = rng.normal(loc=1.0, size=500)
    stat, pvalue = ks(ref, cur)
    expected = ks_2samp(ref, cur)
    assert stat == expected.statistic
    assert pvalue == expected.pvalue


def test_ks_detects_shift() -> None:
    rng = np.random.default_rng(5)
    ref = rng.normal(size=1000)
    cur = rng.normal(loc=2.0, size=1000)
    stat, pvalue = ks(ref, cur)
    assert stat > 0.5
    assert pvalue < 0.01


def test_ks_empty_input_is_safe() -> None:
    stat, pvalue = ks(np.array([]), np.array([1.0, 2.0]))
    assert stat == 0.0
    assert pvalue == 1.0


def _identical_frame(rng: np.random.Generator, n: int = 500) -> pd.DataFrame:
    return pd.DataFrame({feature: rng.normal(size=n) for feature in schema.FEATURES})


def test_report_flags_feature_drift() -> None:
    rng = np.random.default_rng(6)
    ref = _identical_frame(rng)
    cur = ref.copy()
    # Push one feature far away; it should be flagged, and so the report drifts.
    cur["temp"] = cur["temp"] + 5.0
    report = feature_drift_report(ref, cur, schema.FEATURES)
    assert report.drifted is True
    temp = next(f for f in report.features if f.feature == "temp")
    assert temp.drifted is True
    assert temp.psi > PSI_THRESHOLD


def test_report_stable_when_nothing_moves() -> None:
    rng = np.random.default_rng(7)
    ref = _identical_frame(rng)
    cur = ref.copy()
    report = feature_drift_report(ref, cur, schema.FEATURES)
    assert report.drifted is False
    assert all(not f.drifted for f in report.features)
    assert report.prediction_psi is None


def test_concept_drift_predictions_shift_features_identical() -> None:
    # THE core scenario: features are byte-for-byte identical (no feature
    # drifts) but the model's predictions shift a lot -> the report must still
    # report drift via prediction_psi.
    rng = np.random.default_rng(8)
    ref = _identical_frame(rng)
    cur = ref.copy()  # identical features

    reference_pred = rng.normal(loc=100.0, scale=10.0, size=len(ref))
    current_pred = reference_pred + 80.0  # ~63%-style demand growth

    report = feature_drift_report(
        ref,
        cur,
        schema.FEATURES,
        reference_pred=reference_pred,
        current_pred=current_pred,
    )

    assert all(not f.drifted for f in report.features)  # no feature drift
    assert report.prediction_psi is not None
    assert report.prediction_psi > PSI_THRESHOLD
    assert report.drifted is True  # caught purely by prediction drift


def test_report_counts_and_threshold_recorded() -> None:
    rng = np.random.default_rng(9)
    ref = _identical_frame(rng, n=300)
    cur = _identical_frame(rng, n=120)
    report = feature_drift_report(ref, cur, schema.FEATURES, psi_threshold=0.25)
    assert report.n_reference == 300
    assert report.n_current == 120
    assert report.psi_threshold == 0.25


def test_to_dict_round_trips_through_json() -> None:
    rng = np.random.default_rng(10)
    ref = _identical_frame(rng)
    cur = ref.copy()
    cur["hum"] = cur["hum"] + 4.0
    report = feature_drift_report(
        ref,
        cur,
        schema.FEATURES,
        reference_pred=rng.normal(size=len(ref)),
        current_pred=rng.normal(loc=5.0, size=len(cur)),
    )
    payload = report.to_dict()
    restored = json.loads(json.dumps(payload))
    assert restored == payload
    assert restored["drifted"] is True
    assert len(restored["features"]) == len(schema.FEATURES)
    assert restored["features"][0]["feature"] == schema.FEATURES[0]


def test_dataclass_shapes() -> None:
    fd = FeatureDrift(feature="temp", psi=0.3, ks_stat=0.4, ks_pvalue=0.01, drifted=True)
    assert fd.to_dict()["feature"] == "temp"
    report = DriftReport(
        features=[fd],
        prediction_psi=None,
        n_reference=10,
        n_current=5,
        psi_threshold=PSI_THRESHOLD,
    )
    # Any drifted feature makes the whole report drift.
    assert report.drifted is True
