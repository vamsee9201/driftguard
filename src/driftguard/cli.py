"""Command-line entry point for the driftguard lifecycle.

Subcommands
-----------
``train``     Download data, train on 2011, register + promote v1.
``simulate``  The full demo, evaluated honestly. 2012 is split into an
              *observed* slice (80%, monitored + used for retraining) and a
              *held-out* slice (20%, never trained on). v1 trains on 2011;
              drift is detected on the held-out slice; v2 retrains on
              2011 + observed-2012 (excluding the held-out slice); both models
              are scored on the held-out slice, so the improvement is real, not
              a train-on-test artifact. Writes ``lifecycle.json`` — the single
              source of truth for the numbers the README/model card quote.
``serve``     Launch the FastAPI serving app against the production model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

from . import data, monitor, serving
from .model import ModelRegistry, train


def _default_tracking() -> str:
    """Absolute sqlite tracking URI under the current working directory."""
    return f"sqlite:///{Path.cwd() / 'mlflow.db'}"


def _mae(model: object, X: pd.DataFrame, y: pd.Series) -> float:
    """Mean absolute error of ``model`` on ``(X, y)``."""
    preds = model.predict(X)  # type: ignore[attr-defined]
    return float(mean_absolute_error(y, preds))


def cmd_train(args: argparse.Namespace) -> int:
    """Train on 2011, register, and promote the first production model."""
    csv_path = data.download_raw(Path(args.data_dir))
    df_2011, _ = data.load_split(csv_path)
    X, y = data.make_features(df_2011)

    pipeline = train(X, y)
    registry = ModelRegistry(args.tracking)
    version = registry.log_and_register(
        pipeline,
        {"n_train": len(X), "period": "2011"},
        {"mae_train": _mae(pipeline, X, y)},
    )
    registry.promote(version)
    print(f"Trained and promoted model version {version} (n_train={len(X)}).")
    return 0


def _split_observed_heldout(
    X: pd.DataFrame, y: pd.Series, heldout_fraction: float = 0.2, random_state: int = 0
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Randomly partition (X, y) into (observed, held-out).

    A *random* (not chronological) split so the held-out slice is drawn from the
    same 2012 regime as the observed slice — isolating "does retraining on the
    new distribution help predict the new distribution" from seasonality.
    """
    rng = np.random.default_rng(random_state)
    order = rng.permutation(len(X))
    n_obs = round((1.0 - heldout_fraction) * len(X))
    obs_idx, held_idx = order[:n_obs], order[n_obs:]
    return (
        X.iloc[obs_idx], y.iloc[obs_idx],
        X.iloc[held_idx], y.iloc[held_idx],
    )


def cmd_simulate(args: argparse.Namespace) -> int:
    """Run the drift -> retrain demo (held-out-honest) and persist ``lifecycle.json``."""
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = data.download_raw(Path(args.data_dir))
    df_2011, df_2012 = data.load_split(csv_path)
    X_2011, y_2011 = data.make_features(df_2011)
    X_2012, y_2012 = data.make_features(df_2012)

    # 2012 -> observed (monitored + retrain pool) and held-out (fair eval, never trained on)
    X_obs, y_obs, X_held, y_held = _split_observed_heldout(X_2012, y_2012)

    registry = ModelRegistry(args.tracking)

    # --- v1: train on 2011 only, promote -------------------------------------
    v1_model = train(X_2011, y_2011)
    v1_mae_heldout = _mae(v1_model, X_held, y_held)
    v1 = registry.log_and_register(
        v1_model,
        {"n_train": len(X_2011), "period": "2011"},
        {"mae_heldout_2012": v1_mae_heldout},
    )
    registry.promote(v1)

    # --- drift on the HELD-OUT slice + auto-retrain on 2011 + observed-2012 ---
    # The retrain pool excludes X_held, so scoring v2 on X_held is leakage-free.
    retrain_X = pd.concat([X_2011, X_obs], ignore_index=True)
    retrain_y = pd.concat([y_2011, y_obs], ignore_index=True)
    decision = monitor.check_and_maybe_retrain(
        registry, X_2011, X_held, y_held, retrain_X, retrain_y
    )

    v2 = decision.new_version
    if v2 is not None:
        v2_model = registry.load_version(v2)
        v2_mae_heldout = _mae(v2_model, X_held, y_held)
    else:
        v2_mae_heldout = v1_mae_heldout

    lifecycle = {
        "versions": {"v1": v1, "v2": v2},
        "eval": "held-out 2012 slice (never used for training)",
        "n_train_v1": len(X_2011),
        "n_train_v2": len(retrain_X),
        "n_heldout": len(X_held),
        "v1_mae_heldout": v1_mae_heldout,
        "v2_mae_heldout": v2_mae_heldout,
        "mae_reduction_pct": (
            round(100 * (v1_mae_heldout - v2_mae_heldout) / v1_mae_heldout, 1)
            if v1_mae_heldout
            else None
        ),
        "prediction_psi_before": decision.drift_before.prediction_psi,
        "prediction_psi_after": (
            decision.drift_after.prediction_psi if decision.drift_after else None
        ),
        "drifted_before": decision.drift_before.drifted,
        "drifted_after": (
            decision.drift_after.drifted if decision.drift_after else None
        ),
        "retrained": decision.retrained,
        "improved": v2_mae_heldout < v1_mae_heldout,
    }
    out_path = out_dir / "lifecycle.json"
    out_path.write_text(json.dumps(lifecycle, indent=2))

    print("driftguard lifecycle simulation (held-out 2012 evaluation)")
    print(f"  v1 (2011 only)         -> version {v1}, held-out MAE = {v1_mae_heldout:.2f}")
    print(f"  drift on held-out      -> {decision.drift_before.drifted} "
          f"(prediction PSI = {decision.drift_before.prediction_psi:.3f})")
    print(f"  v2 (2011+obs-2012)     -> version {v2}, held-out MAE = {v2_mae_heldout:.2f}")
    psi_after = decision.drift_after.prediction_psi if decision.drift_after else None
    print(f"  drift after retrain    -> {psi_after:.3f}" if psi_after is not None else
          "  drift after retrain    -> n/a")
    print(f"  MAE reduction          -> {lifecycle['mae_reduction_pct']}%")
    print(f"  wrote {out_path}")
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    """Serve the production model over HTTP with drift monitoring."""
    import uvicorn

    csv_path = data.download_raw(Path(args.data_dir))
    df_2011, _ = data.load_split(csv_path)
    reference_X, _ = data.make_features(df_2011)

    registry = ModelRegistry(args.tracking)
    app = serving.create_app(registry, reference_X)
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Construct the argparse parser with all subcommands."""
    parser = argparse.ArgumentParser(prog="driftguard", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="train on 2011 and promote v1")
    p_train.add_argument("--data-dir", default="data")
    p_train.add_argument("--tracking", default=_default_tracking())
    p_train.set_defaults(func=cmd_train)

    p_sim = sub.add_parser("simulate", help="run the drift -> retrain demo")
    p_sim.add_argument("--data-dir", default="data")
    p_sim.add_argument("--tracking", default=_default_tracking())
    p_sim.add_argument("--out", default="artifacts")
    p_sim.set_defaults(func=cmd_simulate)

    p_serve = sub.add_parser("serve", help="serve the production model")
    p_serve.add_argument("--data-dir", default="data")
    p_serve.add_argument("--tracking", default=_default_tracking())
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8000)
    p_serve.set_defaults(func=cmd_serve)

    return parser


def main(argv: list[str] | None = None) -> int:
    """Parse ``argv`` and dispatch to the selected subcommand."""
    parser = build_parser()
    args = parser.parse_args(argv)
    func = args.func
    result: int = func(args)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
