# driftguard — module contract (source of truth for parallel implementation)

An end-to-end ML lifecycle for hourly bike-demand: train → track & register
(MLflow) → serve (FastAPI) → detect drift → auto-retrain. The star is the
**drift → retrain loop**, demonstrated on a REAL distribution shift: the model
trains on 2011 and is monitored on 2012, where demand grew ~63% (prediction
PSI ≈ 0.24, significant) while weather features barely move (PSI ≈ 0.04). So
feature-only monitoring would MISS the drift; prediction/concept-drift
monitoring catches it. Keep that story intact.

Conventions: Python 3.12, ruff line-length 100 (`E,F,I,UP,B,SIM,RUF`), mypy
`check_untyped_defs`, plain pytest (asyncio_mode=auto). Type public functions.
Deterministic: models take `random_state: int = 0`.

## src/driftguard/schema.py  (ALREADY WRITTEN — do not modify)

Feature-column constants + target name. Everyone imports from here.

## src/driftguard/data.py  (Agent A)

```python
DATA_URL = "https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip"

def download_raw(dest_dir: Path) -> Path
    # Download+extract hour.csv to dest_dir/hour.csv; skip if present. httpx.

def load_split(csv_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]
    # Read hour.csv; return (df_2011, df_2012) split on the `yr` column (0/1),
    # each sorted by (dteday, hr). These are the reference (train) and the
    # monitored (production) periods.

def make_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]
    # Point-regression features (NO lags — this project is about lifecycle, not
    # forecast sophistication). X columns = exactly schema.FEATURES:
    #   calendar: hour, dayofweek, month, workingday, holiday, season,
    #             hour_sin, hour_cos  (derive dayofweek/month from dteday)
    #   weather:  temp, hum, windspeed, weathersit
    # y = df["cnt"] (float). X.index = df.index reset to RangeIndex; X and y aligned.
```

Tests `tests/test_data.py`: synthetic hour.csv fixture (no network); assert the
year split partitions rows, feature columns == schema.FEATURES in order, no NaN,
hour_sin/cos correct for a known hour.

## src/driftguard/model.py  (Agent B — THE ONLY MODULE THAT IMPORTS mlflow)

Isolate ALL MLflow use here so the rest of the system depends only on these
interfaces. Use MLflow **aliases**, NOT the deprecated stages API. Tracking +
registry both use a **sqlite** URI (file store does not support the registry).

```python
def build_pipeline(random_state: int = 0) -> Pipeline
    # sklearn Pipeline wrapping HistGradientBoostingRegressor (squared_error).
def train(X: pd.DataFrame, y: pd.Series, random_state: int = 0) -> Pipeline
    # build_pipeline().fit(X, y); returns the fitted pipeline.

PRODUCTION_ALIAS = "production"

class ModelRegistry:
    def __init__(self, tracking_uri: str, model_name: str = "bike_demand",
                 experiment: str = "driftguard")
        # calls mlflow.set_tracking_uri(tracking_uri); tracking_uri is like
        # "sqlite:////abs/path/mlflow.db". Creates the experiment if needed.
    def log_and_register(self, pipeline: Pipeline, params: dict, metrics: dict) -> int
        # start_run -> log_params, log_metrics, sklearn.log_model(registered_model_name=
        # model_name) -> return the integer version just created.
    def promote(self, version: int) -> None
        # set the "production" alias to point at `version`
        # (MlflowClient.set_registered_model_alias).
    def production_version(self) -> int | None
        # version behind the "production" alias, or None if unset.
    def load_production(self) -> Pipeline
        # load models:/<model_name>@production ; raise RuntimeError if no alias.
    def load_version(self, version: int) -> Pipeline
```

Tests `tests/test_model.py`: use `tracking_uri=f"sqlite:///{tmp_path}/mlflow.db"`,
tiny data. Assert: train fits & predicts; log_and_register returns increasing
versions; promote + production_version round-trip; load_production returns a
working model; load_production raises before any promote. (These hit real MLflow
and are a bit slow — keep data tiny; a handful of tests is fine.)

## src/driftguard/drift.py  (Agent C)

```python
PSI_THRESHOLD = 0.2   # >0.2 = significant drift (industry convention)

def psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float
    # Population Stability Index. Bin edges = quantiles of REFERENCE (outer edges
    # -> +-inf so all current values fall in a bin). Proportions clipped to a
    # small epsilon (e.g. 1e-6) so empty bins don't produce inf/nan.
    # PSI = sum((cur_prop - ref_prop) * ln(cur_prop / ref_prop)). Returns >= 0.

def ks(reference: np.ndarray, current: np.ndarray) -> tuple[float, float]
    # (statistic, p_value) via scipy.stats.ks_2samp.

@dataclass
class FeatureDrift:
    feature: str; psi: float; ks_stat: float; ks_pvalue: float; drifted: bool

@dataclass
class DriftReport:
    features: list[FeatureDrift]
    prediction_psi: float | None
    n_reference: int
    n_current: int
    psi_threshold: float
    @property
    def drifted(self) -> bool
        # True if ANY feature.drifted OR (prediction_psi is not None and > threshold)
    def to_dict(self) -> dict           # JSON-serializable (for artifacts + /drift)

def feature_drift_report(
    reference_X: pd.DataFrame, current_X: pd.DataFrame, features: list[str],
    reference_pred: np.ndarray | None = None, current_pred: np.ndarray | None = None,
    psi_threshold: float = PSI_THRESHOLD,
) -> DriftReport
    # per-feature psi + ks; a feature is drifted if its psi > psi_threshold.
    # prediction_psi = psi(reference_pred, current_pred) if both given else None.
```

Tests `tests/test_drift.py`: psi ≈ 0 for identical samples, large for shifted;
psi handles empty bins without inf/nan; ks matches scipy on a known case;
report.drifted true when predictions shift even if features are identical (the
core concept-drift scenario); to_dict round-trips through json.

## src/driftguard/serving.py + monitor.py + cli.py  (Agent D)

**serving.py** — `create_app(registry: ModelRegistry, reference_X: pd.DataFrame) -> FastAPI`:
- `POST /predict` body `{"records": [ {feature: value, ...}, ... ]}` -> `{"predictions": [...], "model_version": N}`. Appends the input rows + predictions to an in-memory buffer (bounded, e.g. deque maxlen 5000).
- `GET /health` -> `{"status":"ok","model_version":N|null}`.
- `GET /drift` -> the `DriftReport.to_dict()` comparing the buffered inputs+preds
  against `reference_X` (+ reference predictions from the production model). If the
  buffer is empty -> `{"drifted": false, "detail": "no traffic yet"}`.
- `POST /reload` -> reload the production model into the app; returns new version.
- Structured access logging; model loaded once at startup, reloaded on /reload.

**monitor.py**:
```python
@dataclass
class RetrainDecision:
    drifted: bool; retrained: bool
    old_version: int | None; new_version: int | None
    drift_before: DriftReport; drift_after: DriftReport | None

def check_and_maybe_retrain(
    registry: ModelRegistry, reference_X, current_X, current_y,
    retrain_X, retrain_y, psi_threshold=drift.PSI_THRESHOLD,
) -> RetrainDecision
    # 1. load production model; predict on reference_X and current_X.
    # 2. drift_before = feature_drift_report(...) with those predictions.
    # 3. if not drifted -> return (retrained=False).
    # 4. else RETRAIN on (retrain_X, retrain_y), log_and_register -> new version,
    #    promote it, then recompute drift_after using the NEW model's predictions.
    #    Return the full decision. This IS the auto-retrain loop.
```

**cli.py** (`main(argv=None) -> int`, argparse):
```
driftguard train    [--data-dir data] [--tracking sqlite:///.../mlflow.db]
    # download, features(2011), train, log_and_register v1, promote it. Print version.
driftguard simulate [--data-dir data] [--tracking ...] [--out artifacts]
    # THE DEMO: train on 2011 -> v1; evaluate v1 on 2012 (MAE) and compute drift
    # (2011 ref vs 2012 current); run check_and_maybe_retrain (retrain on 2011+2012)
    # -> v2; evaluate v2 on 2012. Write artifacts/lifecycle.json with: v1_mae_2012,
    # v2_mae_2012, prediction_psi_before, prediction_psi_after, versions, drifted
    # flags. Print a summary. These are the numbers the README/model card quote.
driftguard serve    [--tracking ...] [--host] [--port]
```

Tests `tests/test_serving.py` + `tests/test_cli.py`: use FastAPI TestClient with a
ModelRegistry on a tmp sqlite URI and a tiny trained+promoted model; /health,
/predict (shape + version), /drift (buffer empty -> not drifted; after feeding
shifted rows -> drifted), /reload. CLI `train` then `simulate` on a tiny cached
CSV writes lifecycle.json with all keys and shows v2 improving on v1. No network.

## Boundaries
- Agent A: data.py, tests/test_data.py
- Agent B: model.py, tests/test_model.py  (ONLY Agent B imports mlflow)
- Agent C: drift.py, tests/test_drift.py
- Agent D: serving.py, monitor.py, cli.py, tests/test_serving.py, tests/test_cli.py
- Nobody edits pyproject.toml or schema.py or another agent's files.
