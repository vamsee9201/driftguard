# Model card — driftguard bike-demand model

All numbers come from `artifacts/lifecycle.json`, produced by
`driftguard simulate`. Nothing here is hand-typed.

## Model

- **Task:** predict hourly bike rentals (`cnt`) from calendar + weather features.
- **Estimator:** scikit-learn `HistGradientBoostingRegressor` (squared error) in a Pipeline.
- **Features:** hour (+ cyclical sin/cos), day-of-week, month, working-day, holiday, season, temperature, humidity, windspeed, weather situation. **No target lags** — this project is about the *lifecycle*, not forecast sophistication.
- **Tracking & registry:** MLflow (sqlite backend), promotion via the modern **alias** API (`production`), not deprecated stages.

## The lifecycle it demonstrates

| stage | model | trained on | held-out 2012 MAE |
|---|---|---|---|
| initial | v1 | 2011 only | **87.31** |
| after auto-retrain | v2 | 2011 + observed 2012 | **48.35** (−44.6%) |

Prediction drift (PSI of predictions vs. actual demand on the held-out slice): **0.223 → 0.067** (crosses back below the 0.2 significance threshold after retraining).

## Honest evaluation (read this)

The 2012 data is split into an **observed** slice (80%, used to detect drift and
retrain) and a **held-out** slice (20%, never used for training *either* model).
Both v1 and v2 are scored only on the held-out slice, so v2's improvement is
generalization to genuinely unseen data — not a train-on-test artifact. The
split is random (not chronological) so it measures "does retraining on the new
regime help predict the new regime" without confounding by within-2012
seasonality.

## Why this drift is interesting

The 2011→2012 shift is a **~63% demand increase** with **near-stationary input
features** (weather PSI ≈ 0.04). So a drift monitor watching only feature
distributions would see nothing. The shift is **concept/label drift**, visible
only by comparing the model's predictions against observed outcomes — which is
exactly what the monitor does, and why `current_y` (labels) is required.

## Limitations

- A tree model cannot extrapolate a trend beyond its training range: on a
  *chronological* future slice (later than all training data) retraining helps
  far less, because 2012-H2 demand sits above anything the model has seen. The
  random-split evaluation here answers a narrower, honest question (predicting
  the observed regime), not "forecast an accelerating trend." A trend/level
  feature or a time-aware model would be the next step.
- Single model family, default hyperparameters — the point is the lifecycle, not
  squeezing the last MAE point.
- `simulate` runs the whole loop in-process for reproducibility; the serving
  path (`driftguard serve`) and `docker-compose` show the same pieces wired as
  separate services.
