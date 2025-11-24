"""Data acquisition and feature engineering for the bike-demand lifecycle.

This module owns the *reference* (2011) vs *monitored* (2012) split that drives
the whole drift story, and turns the raw UCI hourly bike-sharing records into
the point-regression feature matrix defined by :mod:`driftguard.schema`.

No MLflow, no model logic — just deterministic, side-effect-light data prep.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

from driftguard import schema

DATA_URL = "https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip"

# Name of the hourly CSV inside the UCI zip archive.
_HOUR_CSV_NAME = "hour.csv"


def download_raw(dest_dir: Path) -> Path:
    """Download and extract ``hour.csv`` from the UCI archive into ``dest_dir``.

    Idempotent: if ``dest_dir/hour.csv`` already exists it is returned as-is and
    no network request is made.

    Args:
        dest_dir: Directory to hold the extracted ``hour.csv``. Created if
            missing.

    Returns:
        Path to the extracted ``hour.csv``.

    Raises:
        FileNotFoundError: If the downloaded archive does not contain
            ``hour.csv``.
    """
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    csv_path = dest_dir / _HOUR_CSV_NAME
    if csv_path.exists():
        return csv_path

    response = httpx.get(DATA_URL, follow_redirects=True, timeout=60.0)
    response.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        member = next(
            (n for n in archive.namelist() if Path(n).name == _HOUR_CSV_NAME),
            None,
        )
        if member is None:
            raise FileNotFoundError(
                f"{_HOUR_CSV_NAME!r} not found in archive at {DATA_URL}"
            )
        with archive.open(member) as src:
            csv_path.write_bytes(src.read())

    return csv_path


def load_split(csv_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read ``hour.csv`` and split it into the 2011 and 2012 periods.

    The raw dataset encodes the year in the ``yr`` column (``0`` = 2011,
    ``1`` = 2012). 2011 is the reference/training period and 2012 is the
    monitored/production period where demand grew ~63%.

    Args:
        csv_path: Path to the raw ``hour.csv``.

    Returns:
        ``(df_2011, df_2012)``, each parsed and sorted by ``(dteday, hr)`` with
        ``dteday`` coerced to ``datetime64``. Indexes are reset to a contiguous
        ``RangeIndex``.
    """
    df = pd.read_csv(csv_path, parse_dates=["dteday"])
    df = df.sort_values(["dteday", "hr"], kind="stable")

    df_2011 = df[df["yr"] == 0].reset_index(drop=True)
    df_2012 = df[df["yr"] == 1].reset_index(drop=True)
    return df_2011, df_2012


def make_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Build the point-regression feature matrix and target from raw records.

    Produces exactly the columns in :data:`driftguard.schema.FEATURES`, in that
    order. There are deliberately no lag features — this project is about the
    lifecycle, not forecast sophistication.

    ``dayofweek`` and ``month`` are derived from ``dteday`` (not the raw
    ``weekday``/``mnth`` columns), and ``hour_sin``/``hour_cos`` give the hour a
    cyclic encoding so 23:00 sits next to 00:00.

    Args:
        df: A raw slice of ``hour.csv`` (e.g. one year from :func:`load_split`).

    Returns:
        ``(X, y)`` where ``X`` has columns == ``schema.FEATURES`` (float) and
        ``y`` is ``df["cnt"]`` as float. Both are reindexed to a shared
        ``RangeIndex`` and are row-aligned.
    """
    dteday = pd.to_datetime(df["dteday"])
    hour = df["hr"].astype(float)
    radians = 2.0 * np.pi * hour / 24.0

    features = pd.DataFrame(
        {
            "hour": hour,
            "dayofweek": dteday.dt.dayofweek.astype(float),
            "month": dteday.dt.month.astype(float),
            "workingday": df["workingday"].astype(float),
            "holiday": df["holiday"].astype(float),
            "season": df["season"].astype(float),
            "hour_sin": np.sin(radians),
            "hour_cos": np.cos(radians),
            "temp": df["temp"].astype(float),
            "hum": df["hum"].astype(float),
            "windspeed": df["windspeed"].astype(float),
            "weathersit": df["weathersit"].astype(float),
        }
    )

    # Enforce the canonical column order and reset the index for clean alignment.
    x = features[schema.FEATURES].reset_index(drop=True)
    y = df[schema.TARGET].astype(float).reset_index(drop=True)
    y.name = schema.TARGET
    return x, y
