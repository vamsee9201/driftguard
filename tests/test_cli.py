"""Tests for the driftguard CLI.

``train`` then ``simulate`` run end-to-end against a synthetic, cached ``hour.csv``
so ``download_raw`` short-circuits and no network access occurs. The synthetic
data bakes in the real story: 2012 demand is much higher than 2011 (a shift the
2011-only model cannot capture) plus a warmer-temperature shift that trips drift
detection — so v2 (retrained on 2011+2012) must beat v1 on 2012 MAE.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from driftguard.cli import main


def _write_synthetic_csv(path: Path, days: int = 20, seed: int = 0) -> None:
    """Write a UCI-shaped hour.csv with a 2011->2012 demand + temperature shift."""
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    instant = 1
    # Weather stays stable year-over-year (feature PSI ~0) — exactly the real
    # story: the 2011->2012 shift lives in the *target* (a demand jump the
    # 2011-only model cannot see), caught only by prediction/concept drift.
    for yr, start in ((0, "2011-01-01"), (1, "2012-01-01")):
        dates = pd.date_range(start, periods=days, freq="D")
        temp_lo, temp_hi = 0.20, 0.50
        for date in dates:
            for hr in range(24):
                hour_signal = np.sin(2 * np.pi * hr / 24.0) + 1.0
                cnt = 50.0 + 40.0 * hour_signal + 200.0 * yr + rng.normal(0, 3)
                rows.append(
                    {
                        "instant": instant,
                        "dteday": date.strftime("%Y-%m-%d"),
                        "season": 1,
                        "yr": yr,
                        "mnth": date.month,
                        "hr": hr,
                        "holiday": 0,
                        "weekday": int(date.dayofweek),
                        "workingday": 1,
                        "weathersit": 1,
                        "temp": float(rng.uniform(temp_lo, temp_hi)),
                        "atemp": 0.3,
                        "hum": float(rng.uniform(0.3, 0.7)),
                        "windspeed": float(rng.uniform(0.0, 0.3)),
                        "casual": 1,
                        "registered": 1,
                        "cnt": max(0.0, round(cnt)),
                    }
                )
                instant += 1
    pd.DataFrame(rows).to_csv(path, index=False)


def test_train_then_simulate_writes_lifecycle(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _write_synthetic_csv(data_dir / "hour.csv")

    rc = main(
        [
            "train",
            "--data-dir",
            str(data_dir),
            "--tracking",
            f"sqlite:///{tmp_path}/train.db",
        ]
    )
    assert rc == 0

    out_dir = tmp_path / "artifacts"
    rc = main(
        [
            "simulate",
            "--data-dir",
            str(data_dir),
            "--tracking",
            f"sqlite:///{tmp_path}/sim.db",
            "--out",
            str(out_dir),
        ]
    )
    assert rc == 0

    lifecycle = json.loads((out_dir / "lifecycle.json").read_text())
    expected_keys = {
        "versions",
        "v1_mae_heldout",
        "v2_mae_heldout",
        "n_heldout",
        "prediction_psi_before",
        "prediction_psi_after",
        "drifted_before",
        "drifted_after",
        "retrained",
        "improved",
    }
    assert expected_keys <= lifecycle.keys()

    # Drift must have been detected and a v2 retrained + promoted.
    assert lifecycle["drifted_before"] is True
    assert lifecycle["retrained"] is True
    assert lifecycle["versions"]["v1"] is not None
    assert lifecycle["versions"]["v2"] is not None

    # The whole point, evaluated honestly: retraining on 2011 + OBSERVED-2012
    # beats the 2011-only model on a HELD-OUT 2012 slice neither model trained on.
    assert lifecycle["improved"] is True
    assert lifecycle["v2_mae_heldout"] < lifecycle["v1_mae_heldout"]
