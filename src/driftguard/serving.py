"""FastAPI serving layer for the bike-demand model.

Exposes prediction, health, on-line drift monitoring, and hot model reload.
The production model is loaded once at application startup and swapped in place
whenever ``POST /reload`` is called. Every request served is appended to a
bounded in-memory buffer so ``GET /drift`` can compare live traffic against the
training reference distribution without any external store.
"""

from __future__ import annotations

import json
import logging
import time
from collections import deque
from collections.abc import Awaitable, Callable
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from starlette.requests import Request
from starlette.responses import Response

from .drift import feature_drift_report
from .model import ModelRegistry
from .schema import FEATURES

logger = logging.getLogger("driftguard.serving")

#: Maximum number of served records retained for on-line drift analysis.
BUFFER_MAXLEN = 5000


class PredictRequest(BaseModel):
    """Request body for ``POST /predict``.

    ``records`` is a list of feature dictionaries; each dictionary should carry
    the keys in :data:`driftguard.schema.FEATURES`. Missing keys become NaN and
    are handled by the model pipeline.
    """

    records: list[dict[str, float]]


class _ServingState:
    """Mutable per-application state shared across request handlers."""

    def __init__(self, registry: ModelRegistry, reference_X: pd.DataFrame) -> None:
        self.registry = registry
        self.reference_X = reference_X.reset_index(drop=True)
        self.model: Any | None = None
        self.model_version: int | None = None
        # Each entry is (feature_record, prediction).
        self.buffer: deque[tuple[dict[str, float], float]] = deque(maxlen=BUFFER_MAXLEN)
        self._load()

    def _load(self) -> None:
        """Load the production model, tolerating the no-model-yet case."""
        try:
            self.model = self.registry.load_production()
            self.model_version = self.registry.production_version()
        except RuntimeError:
            self.model = None
            self.model_version = None


def create_app(registry: ModelRegistry, reference_X: pd.DataFrame) -> FastAPI:
    """Build the serving application.

    Args:
        registry: Model registry used to load and reload the production model.
        reference_X: Training-time feature frame used as the drift reference.

    Returns:
        A configured :class:`fastapi.FastAPI` instance.
    """
    app = FastAPI(title="driftguard", version="0.1.0")
    state = _ServingState(registry, reference_X)
    app.state.dg = state

    @app.middleware("http")
    async def access_log(
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        """Emit one structured JSON log line per request."""
        start = time.perf_counter()
        response = await call_next(request)
        duration_ms = (time.perf_counter() - start) * 1000.0
        logger.info(
            json.dumps(
                {
                    "method": request.method,
                    "path": request.url.path,
                    "status": response.status_code,
                    "duration_ms": round(duration_ms, 2),
                }
            )
        )
        return response

    @app.get("/health")
    def health() -> dict[str, Any]:
        """Liveness probe reporting the currently loaded model version."""
        return {"status": "ok", "model_version": state.model_version}

    @app.post("/predict")
    def predict(req: PredictRequest) -> dict[str, Any]:
        """Score a batch of feature records and buffer them for drift analysis."""
        if state.model is None:
            raise HTTPException(status_code=503, detail="no production model loaded")
        if not req.records:
            return {"predictions": [], "model_version": state.model_version}
        frame = pd.DataFrame(req.records).reindex(columns=FEATURES)
        preds = np.asarray(state.model.predict(frame), dtype=float)
        for record, pred in zip(req.records, preds, strict=True):
            state.buffer.append((record, float(pred)))
        return {
            "predictions": [float(p) for p in preds],
            "model_version": state.model_version,
        }

    @app.get("/drift")
    def drift() -> dict[str, Any]:
        """Compare buffered live traffic against the training reference."""
        if not state.buffer:
            return {"drifted": False, "detail": "no traffic yet"}
        records = [record for record, _ in state.buffer]
        current_pred = np.asarray([pred for _, pred in state.buffer], dtype=float)
        current_X = pd.DataFrame(records).reindex(columns=FEATURES)
        reference_pred: np.ndarray | None = None
        if state.model is not None:
            reference_pred = np.asarray(
                state.model.predict(state.reference_X), dtype=float
            )
        report = feature_drift_report(
            state.reference_X,
            current_X,
            FEATURES,
            reference_pred=reference_pred,
            current_pred=current_pred,
        )
        return report.to_dict()

    @app.post("/reload")
    def reload() -> dict[str, Any]:
        """Reload the production model into the running application."""
        state.model = state.registry.load_production()
        state.model_version = state.registry.production_version()
        return {"model_version": state.model_version}

    return app
