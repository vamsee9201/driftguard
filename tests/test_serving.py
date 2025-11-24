"""Tests for the FastAPI serving layer.

These exercise the real serving stack against a real MLflow registry backed by a
temporary sqlite database and a tiny trained+promoted model. No network access.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from driftguard.model import ModelRegistry, train
from driftguard.schema import FEATURES
from driftguard.serving import create_app


def _reference_frame(n: int = 200, seed: int = 0) -> pd.DataFrame:
    """Build a synthetic feature frame with the contract's exact columns."""
    rng = np.random.default_rng(seed)
    hour = rng.integers(0, 24, size=n).astype(float)
    frame = pd.DataFrame(
        {
            "hour": hour,
            "dayofweek": rng.integers(0, 7, size=n).astype(float),
            "month": rng.integers(1, 13, size=n).astype(float),
            "workingday": rng.integers(0, 2, size=n).astype(float),
            "holiday": np.zeros(n),
            "season": rng.integers(1, 5, size=n).astype(float),
            "hour_sin": np.sin(2 * np.pi * hour / 24.0),
            "hour_cos": np.cos(2 * np.pi * hour / 24.0),
            "temp": rng.uniform(0.2, 0.5, size=n),
            "hum": rng.uniform(0.3, 0.7, size=n),
            "windspeed": rng.uniform(0.0, 0.3, size=n),
            "weathersit": rng.integers(1, 4, size=n).astype(float),
        }
    )
    return frame[FEATURES]


def _target(frame: pd.DataFrame, seed: int = 1) -> pd.Series:
    """A learnable target: demand rises with the hour-of-day signal."""
    rng = np.random.default_rng(seed)
    base = 50.0 + 100.0 * (frame["hour_sin"].to_numpy() + 1.0)
    noise = rng.normal(0, 5, size=len(frame))
    return pd.Series(base + noise, name="cnt")


@pytest.fixture
def client(tmp_path) -> TestClient:
    """A TestClient wired to a registry with one promoted model."""
    reference_X = _reference_frame()
    y = _target(reference_X)
    pipeline = train(reference_X, y)

    registry = ModelRegistry(f"sqlite:///{tmp_path}/mlflow.db")
    version = registry.log_and_register(
        pipeline, {"n_train": len(reference_X)}, {"mae": 0.0}
    )
    registry.promote(version)

    app = create_app(registry, reference_X)
    return TestClient(app)


def test_health_reports_promoted_version(client: TestClient) -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["model_version"] == 1


def test_predict_shape_and_version(client: TestClient) -> None:
    records = _reference_frame(n=5, seed=99).to_dict(orient="records")
    resp = client.post("/predict", json={"records": records})
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["predictions"]) == 5
    assert all(isinstance(p, float) for p in body["predictions"])
    assert body["model_version"] == 1


def test_drift_empty_buffer_reports_no_traffic(client: TestClient) -> None:
    resp = client.get("/drift")
    assert resp.status_code == 200
    body = resp.json()
    assert body["drifted"] is False
    assert body["detail"] == "no traffic yet"


def test_drift_detects_shifted_traffic(client: TestClient) -> None:
    # Reference temp lives in [0.2, 0.5]; push served rows far outside it.
    shifted = _reference_frame(n=80, seed=7)
    shifted["temp"] = np.linspace(0.85, 0.99, len(shifted))
    resp = client.post("/predict", json={"records": shifted.to_dict(orient="records")})
    assert resp.status_code == 200

    report = client.get("/drift").json()
    assert report["drifted"] is True
    assert report["n_current"] == 80
    temp_entry = next(f for f in report["features"] if f["feature"] == "temp")
    assert temp_entry["psi"] > 0.2
    assert temp_entry["drifted"] is True


def test_reload_returns_current_version(client: TestClient) -> None:
    resp = client.post("/reload")
    assert resp.status_code == 200
    assert resp.json()["model_version"] == 1
