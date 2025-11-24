"""Tests for the model training + MLflow registry boundary (``model.py``).

These exercise the *real* MLflow tracking + registry against a temporary sqlite
store (no network). Real logging is a few seconds per model, so the data is kept
tiny and the number of round-trips small.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from driftguard import model as model_mod
from driftguard.model import ModelRegistry, build_pipeline, train
from driftguard.schema import FEATURES


def _tiny_xy(n: int = 40, seed: int = 0) -> tuple[pd.DataFrame, pd.Series]:
    """Build a small, well-formed (X, y) with the contract's feature columns."""
    rng = np.random.default_rng(seed)
    data = {col: rng.normal(size=n) for col in FEATURES}
    X = pd.DataFrame(data, columns=FEATURES)
    # A learnable signal so the fitted model predicts something non-degenerate.
    y = pd.Series(3.0 * X["temp"] + X["hour"] + rng.normal(scale=0.1, size=n), name="cnt")
    return X, y


def _registry(tmp_path) -> ModelRegistry:
    uri = f"sqlite:///{tmp_path}/mlflow.db"
    return ModelRegistry(tracking_uri=uri, model_name="test_bike_demand")


def test_build_pipeline_is_unfitted_pipeline() -> None:
    pipe = build_pipeline(random_state=0)
    assert isinstance(pipe, Pipeline)
    # An unfitted pipeline must not be predictable yet.
    with pytest.raises(Exception):  # noqa: B017 - sklearn NotFittedError family
        pipe.predict(_tiny_xy(5)[0])


def test_train_fits_and_predicts() -> None:
    X, y = _tiny_xy()
    pipe = train(X, y, random_state=0)
    preds = pipe.predict(X)
    assert preds.shape == (len(X),)
    assert np.all(np.isfinite(preds))


def test_train_is_deterministic() -> None:
    X, y = _tiny_xy()
    a = train(X, y, random_state=0).predict(X)
    b = train(X, y, random_state=0).predict(X)
    np.testing.assert_allclose(a, b)


def test_log_and_register_returns_increasing_versions(tmp_path) -> None:
    reg = _registry(tmp_path)
    X, y = _tiny_xy()
    pipe = train(X, y)

    v1 = reg.log_and_register(pipe, params={"random_state": 0}, metrics={"mae": 1.0})
    v2 = reg.log_and_register(pipe, params={"random_state": 0}, metrics={"mae": 0.5})

    assert v1 == 1
    assert v2 == 2


def test_promote_and_production_version_round_trip(tmp_path) -> None:
    reg = _registry(tmp_path)
    X, y = _tiny_xy()
    pipe = train(X, y)

    assert reg.production_version() is None  # nothing promoted yet

    v1 = reg.log_and_register(pipe, params={}, metrics={})
    v2 = reg.log_and_register(pipe, params={}, metrics={})

    reg.promote(v1)
    assert reg.production_version() == v1

    reg.promote(v2)
    assert reg.production_version() == v2


def test_load_production_raises_before_promote(tmp_path) -> None:
    reg = _registry(tmp_path)
    X, y = _tiny_xy()
    reg.log_and_register(train(X, y), params={}, metrics={})

    with pytest.raises(RuntimeError):
        reg.load_production()


def test_load_production_returns_working_model(tmp_path) -> None:
    reg = _registry(tmp_path)
    X, y = _tiny_xy()
    pipe = train(X, y)
    v1 = reg.log_and_register(pipe, params={}, metrics={})
    reg.promote(v1)

    loaded = reg.load_production()
    preds = loaded.predict(X)
    assert preds.shape == (len(X),)
    # Loaded model reproduces the in-memory model's predictions.
    np.testing.assert_allclose(preds, pipe.predict(X), rtol=1e-5, atol=1e-5)


def test_load_version_loads_specific_version(tmp_path) -> None:
    reg = _registry(tmp_path)
    X, y = _tiny_xy()
    pipe = train(X, y)
    v1 = reg.log_and_register(pipe, params={}, metrics={})

    loaded = reg.load_version(v1)
    np.testing.assert_allclose(loaded.predict(X), pipe.predict(X), rtol=1e-5, atol=1e-5)


def test_production_alias_constant() -> None:
    assert model_mod.PRODUCTION_ALIAS == "production"
