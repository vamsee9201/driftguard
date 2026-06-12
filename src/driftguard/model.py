"""Model training and the MLflow tracking + registry boundary.

This is the ONLY module in driftguard that imports ``mlflow``. Everything else
depends solely on the small, MLflow-agnostic interfaces defined here
(:func:`build_pipeline`, :func:`train`, and :class:`ModelRegistry`).

The registry uses MLflow **aliases** (the modern replacement for the deprecated
stages API): a fitted pipeline is logged and registered as a new integer version,
and the ``production`` alias is moved to point at whichever version is live.
Tracking and the model registry share a single **sqlite** URI — the local file
store cannot back the registry, so sqlite is required.
"""

from __future__ import annotations

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.exceptions import MlflowException, RestException
from mlflow.tracking import MlflowClient
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline

from driftguard.schema import EXPERIMENT, MODEL_NAME

#: Alias that always points at the currently-serving model version.
PRODUCTION_ALIAS = "production"

#: Sub-path under an MLflow run where the sklearn model artifact is stored.
_ARTIFACT_NAME = "model"


def build_pipeline(random_state: int = 0) -> Pipeline:
    """Build the (unfitted) demand-forecasting pipeline.

    A single-step sklearn :class:`~sklearn.pipeline.Pipeline` wrapping a
    :class:`~sklearn.ensemble.HistGradientBoostingRegressor` with a
    ``squared_error`` loss. The gradient-boosting model handles the raw
    numeric feature matrix directly, so no scaling/encoding step is needed.

    Args:
        random_state: Seed for deterministic fits.

    Returns:
        An unfitted pipeline ready for ``.fit(X, y)``.
    """
    regressor = HistGradientBoostingRegressor(
        loss="squared_error",
        random_state=random_state,
    )
    return Pipeline([("regressor", regressor)])


def train(X: pd.DataFrame, y: pd.Series, random_state: int = 0) -> Pipeline:
    """Fit :func:`build_pipeline` on ``(X, y)`` and return the fitted pipeline.

    Args:
        X: Feature matrix (columns are :data:`driftguard.schema.FEATURES`).
        y: Target vector (bike demand ``cnt``).
        random_state: Seed forwarded to :func:`build_pipeline`.

    Returns:
        The fitted pipeline.
    """
    pipeline = build_pipeline(random_state=random_state)
    pipeline.fit(X, y)
    return pipeline


class ModelRegistry:
    """Thin façade over MLflow tracking + the model registry.

    All MLflow calls in the project funnel through this class. It logs fitted
    pipelines as registered model versions and manages the ``production`` alias
    so that serving/retraining code can load "the live model" without knowing
    anything about MLflow.

    Args:
        tracking_uri: A sqlite URI such as ``"sqlite:////abs/path/mlflow.db"``.
            Both tracking and the registry are backed by this store.
        model_name: Registered-model name. Defaults to
            :data:`driftguard.schema.MODEL_NAME`.
        experiment: Experiment name; created if it does not exist. Defaults to
            :data:`driftguard.schema.EXPERIMENT`.
    """

    def __init__(
        self,
        tracking_uri: str,
        model_name: str = MODEL_NAME,
        experiment: str = EXPERIMENT,
    ) -> None:
        self.tracking_uri = tracking_uri
        self.model_name = model_name
        self.experiment = experiment

        mlflow.set_tracking_uri(tracking_uri)
        # Registry URI is the same sqlite store.
        mlflow.set_registry_uri(tracking_uri)
        self._client = MlflowClient(
            tracking_uri=tracking_uri,
            registry_uri=tracking_uri,
        )
        # Create the experiment on first use; set_experiment is idempotent.
        mlflow.set_experiment(experiment)

    def log_and_register(
        self,
        pipeline: Pipeline,
        params: dict,
        metrics: dict,
    ) -> int:
        """Log a fitted pipeline as a new registered model version.

        Opens a run, records ``params``/``metrics``, logs the sklearn model, and
        registers it under :attr:`model_name`, creating a fresh integer version.

        Args:
            pipeline: A fitted pipeline.
            params: Parameters to record on the run (``mlflow.log_params``).
            metrics: Metrics to record on the run (``mlflow.log_metrics``).

        Returns:
            The integer version number just created.
        """
        with mlflow.start_run(experiment_id=self._experiment_id()):
            if params:
                mlflow.log_params(params)
            if metrics:
                mlflow.log_metrics(metrics)
            model_info = mlflow.sklearn.log_model(
                sk_model=pipeline,
                name=_ARTIFACT_NAME,
                registered_model_name=self.model_name,
            )
        version = model_info.registered_model_version
        if version is None:  # pragma: no cover - defensive; registration always sets it
            raise RuntimeError("MLflow did not return a registered model version")
        return int(version)

    def promote(self, version: int) -> None:
        """Point the ``production`` alias at ``version``.

        Args:
            version: The registered model version to promote.
        """
        self._client.set_registered_model_alias(
            name=self.model_name,
            alias=PRODUCTION_ALIAS,
            version=str(version),
        )

    def production_version(self) -> int | None:
        """Return the version behind the ``production`` alias, or ``None``.

        Returns:
            The integer version currently aliased to ``production``, or ``None``
            when the alias (or the registered model) does not yet exist.
        """
        try:
            mv = self._client.get_model_version_by_alias(
                name=self.model_name,
                alias=PRODUCTION_ALIAS,
            )
        except (MlflowException, RestException):
            return None
        return int(mv.version)

    def load_production(self) -> Pipeline:
        """Load the model behind the ``production`` alias.

        Returns:
            The fitted production pipeline.

        Raises:
            RuntimeError: If no version is aliased to ``production``.
        """
        if self.production_version() is None:
            raise RuntimeError(
                f"No '{PRODUCTION_ALIAS}' alias set for model '{self.model_name}'"
            )
        uri = f"models:/{self.model_name}@{PRODUCTION_ALIAS}"
        return mlflow.sklearn.load_model(uri)

    def load_version(self, version: int) -> Pipeline:
        """Load a specific registered model version.

        Args:
            version: The registered model version to load.

        Returns:
            The fitted pipeline for that version.
        """
        uri = f"models:/{self.model_name}/{version}"
        return mlflow.sklearn.load_model(uri)

    def _experiment_id(self) -> str:
        """Return the experiment id, creating the experiment if needed."""
        exp = self._client.get_experiment_by_name(self.experiment)
        if exp is None:
            return self._client.create_experiment(self.experiment)
        return exp.experiment_id
