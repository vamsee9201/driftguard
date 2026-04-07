# driftguard

An end-to-end ML lifecycle that watches itself: train -> track and register (MLflow) -> serve (FastAPI) -> detect drift -> auto-retrain. Demonstrated on a real distribution shift, evaluated honestly on held-out data.

> Production ML doesn't fail at model.fit() -- it fails three months later when the world moves and nobody notices. The interesting part of this project isn't the model; it's the closed loop that detects the model going stale and repairs it.

## The lifecycle, on real data

`driftguard simulate` runs the whole loop and writes `artifacts/lifecycle.json`. Actual output:

```
driftguard lifecycle simulation (held-out 2012 evaluation)
  v1 (2011 only)         -> version 1, held-out MAE = 87.31
  drift on held-out      -> True (prediction PSI = 0.223)
  v2 (2011+obs-2012)     -> version 2, held-out MAE = 48.35
  drift after retrain    -> 0.067
  MAE reduction          -> 44.6%
```

A model trained on 2011 bike-share demand degrades on 2012 (rentals grew ~63%). The monitor detects the drift (prediction PSI 0.223 > 0.2), auto-retrains and promotes v2, and drift falls back to 0.067 -- a 44.6% MAE reduction on data neither model was trained on.

## Why this drift is the interesting kind

The 2011-2012 shift is a demand surge with near-stationary input features -- weather and calendar distributions barely move (feature PSI ~ 0.04). A drift monitor watching only feature distributions would see nothing and let the model rot. This is concept/label drift, catchable only by comparing predictions against observed outcomes. driftguard monitors exactly that, which is why the retrain trigger needs labels. Distinguishing feature drift from concept drift is the difference between a monitoring dashboard and a monitoring system.

## Honest evaluation (the part most demos get wrong)

Retraining on the shifted data and then scoring on that same data is train-on-test -- it manufactures an improvement. driftguard splits 2012 into an observed slice (80%, used to detect drift and retrain) and a held-out slice (20%, never used to train either model). Both models are scored only on the held-out slice. v2's win is real generalization, not leakage. (The model card in docs/model_card.md documents the split and an honest limitation: tree models cannot extrapolate a trend beyond their training range.)

## Architecture

```
                    +------------ MLflow (sqlite) ------------+
 train -> pipeline ->  tracking (params/metrics) + registry   |
                    |  model versions, "production" alias      |
                    +---------------+-------------------------+
                                    | load @production
                     FastAPI serving  +-- POST /predict  (buffers inputs+preds)
                                    |  +-- GET  /drift    (buffer vs reference)
                                    |  +-- POST /reload   (hot-swap model)
                                    |  +-- GET  /health
                                    v
              monitor: drift(predictions, actuals) > 0.2 ?
                                    | yes
                     retrain -> register new version -> promote -> reload
```

Modules: `data` (load + year split + features) * `model` (the only MLflow boundary -- training + registry with aliases) * `drift` (PSI + KS, feature and concept drift) * `serving` (FastAPI) * `monitor` (the retrain loop) * `cli`.

## Run it

```bash
git clone https://github.com/vamseekrishna9201/driftguard && cd driftguard
uv sync
uv run driftguard train      --tracking sqlite:///$PWD/mlflow.db   # train 2011, promote v1
uv run driftguard simulate   --tracking sqlite:///$PWD/mlflow.db   # the full drift->retrain demo
uv run driftguard serve      --tracking sqlite:///$PWD/mlflow.db   # FastAPI on :8000
```

Or the containerized topology -- an MLflow tracking server and the API as separate services:

```bash
docker compose up --build     # mlflow on :5000, api on :8000
```

Serving endpoints: `POST /predict`, `GET /drift`, `POST /reload`, `GET /health`.

## Drift detection

- PSI (Population Stability Index) with reference-quantile bins and epsilon-clipped proportions -- numerically safe on empty/out-of-range bins (no inf/nan). Threshold 0.2 = significant (industry convention).
- KS two-sample test (scipy.stats.ks_2samp) per feature.
- A `DriftReport` flags drift when any feature drifts OR prediction/concept drift crosses threshold, and serializes to JSON for the /drift endpoint and artifacts.

## Tests and CI

```bash
uv run pytest -q     # 39 tests: data, MLflow registry (real, sqlite), PSI/KS math, serving, retrain loop
uv run ruff check src tests && uv run mypy src
```

CI runs lint, types, the full suite, and a Docker build on every push. Registry tests hit real MLflow against a temp sqlite store; nothing touches the network.

## Development and Integration

This project was developed using an agent-orchestrated workflow against docs/CONTRACT.md (all MLflow use isolated to one module; aliases mandated over MLflow 3.x's deprecated stages). During integration, I identified and corrected a methodology flaw that originated in the contract itself -- the demo evaluated the retrained model on its own training data -- replacing it with the held-out evaluation above. Details in docs/REVIEW.md.

## Future Roadmap

- Terraform to stand up the MLflow server + API on ECS/Fargate.
- Scheduled monitoring (cron/Airflow) that runs the drift check on a rolling window and opens a PR / pages on drift.
- A trend-aware model so the retrain helps on chronological-future data, not only the observed regime.
- Prometheus metrics + a Grafana panel for live drift and latency.

## Maintainer
Vamsee Krishna Kotha
Data Scientist - AI
Email: vamseekrishna9201@gmail.com
Key Skills: Python, SQL, R, Go, TypeScript, LangGraph, ADK
Experience: 3+ years of professional experience in AI and Data Science.