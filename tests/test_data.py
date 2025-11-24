"""Tests for :mod:`driftguard.data` — no network; a synthetic ``hour.csv``."""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import pytest

from driftguard import data, schema

# Raw column layout of the UCI hour.csv, used to build synthetic fixtures.
_RAW_COLUMNS = [
    "instant",
    "dteday",
    "season",
    "yr",
    "mnth",
    "hr",
    "holiday",
    "weekday",
    "workingday",
    "weathersit",
    "temp",
    "atemp",
    "hum",
    "windspeed",
    "casual",
    "registered",
    "cnt",
]


def _synthetic_raw() -> pd.DataFrame:
    """A tiny two-year raw frame: 24 hours/day across a few days per year."""
    rows: list[dict[str, object]] = []
    instant = 1
    # yr=0 -> 2011 (3 days), yr=1 -> 2012 (2 days): differing row counts on purpose.
    plan = [(0, "2011-01-01", 3), (1, "2012-06-01", 2)]
    for yr, start, n_days in plan:
        base = pd.Timestamp(start)
        for day in range(n_days):
            dteday = base + pd.DateOffset(days=day)
            for hr in range(24):
                rows.append(
                    {
                        "instant": instant,
                        "dteday": dteday.strftime("%Y-%m-%d"),
                        "season": 1 + (dteday.month % 4),
                        "yr": yr,
                        "mnth": dteday.month,
                        "hr": hr,
                        "holiday": 0,
                        "weekday": (dteday.dayofweek + 1) % 7,
                        "workingday": int(dteday.dayofweek < 5),
                        "weathersit": 1 + (hr % 3),
                        "temp": 0.2 + 0.5 * (hr / 24.0) + 0.3 * yr,
                        "atemp": 0.25 + 0.5 * (hr / 24.0),
                        "hum": 0.5 + 0.2 * math.sin(hr),
                        "windspeed": 0.1 + 0.05 * (hr % 5),
                        "casual": hr,
                        "registered": 10 * hr + 100 * yr,
                        "cnt": 11 * hr + 100 * yr + 1,
                    }
                )
                instant += 1
    return pd.DataFrame(rows, columns=_RAW_COLUMNS)


@pytest.fixture
def csv_path(tmp_path: Path) -> Path:
    path = tmp_path / "hour.csv"
    _synthetic_raw().to_csv(path, index=False)
    return path


def test_load_split_partitions_rows(csv_path: Path) -> None:
    raw = pd.read_csv(csv_path)
    df_2011, df_2012 = data.load_split(csv_path)

    # Split is a clean partition: no overlap, nothing dropped or duplicated.
    assert (df_2011["yr"] == 0).all()
    assert (df_2012["yr"] == 1).all()
    assert len(df_2011) + len(df_2012) == len(raw)
    assert len(df_2011) == 3 * 24
    assert len(df_2012) == 2 * 24
    # Instants (unique ids) are disjoint across the two periods.
    assert set(df_2011["instant"]).isdisjoint(set(df_2012["instant"]))


def test_load_split_sorted_and_reindexed(csv_path: Path) -> None:
    df_2011, df_2012 = data.load_split(csv_path)
    for df in (df_2011, df_2012):
        assert list(df.index) == list(range(len(df)))
        ordering = df[["dteday", "hr"]].copy()
        ordering["dteday"] = pd.to_datetime(ordering["dteday"])
        assert ordering.equals(ordering.sort_values(["dteday", "hr"]).reset_index(drop=True))


def test_make_features_columns_exact_order(csv_path: Path) -> None:
    df_2011, _ = data.load_split(csv_path)
    x, y = data.make_features(df_2011)

    assert list(x.columns) == schema.FEATURES
    assert y.name == schema.TARGET
    assert len(x) == len(y) == len(df_2011)
    assert list(x.index) == list(range(len(x)))
    assert list(y.index) == list(range(len(y)))


def test_make_features_no_nan_and_float(csv_path: Path) -> None:
    df_2011, _ = data.load_split(csv_path)
    x, y = data.make_features(df_2011)

    assert not x.isna().any().any()
    assert not y.isna().any()
    assert all(pd.api.types.is_float_dtype(dt) for dt in x.dtypes)
    assert pd.api.types.is_float_dtype(y.dtype)


def test_make_features_hour_cyclic_encoding(csv_path: Path) -> None:
    df_2011, _ = data.load_split(csv_path)
    x, _ = data.make_features(df_2011)

    # Row 0 is hour 0 after sorting: sin(0)=0, cos(0)=1.
    assert x.loc[0, "hour"] == pytest.approx(0.0)
    assert x.loc[0, "hour_sin"] == pytest.approx(0.0)
    assert x.loc[0, "hour_cos"] == pytest.approx(1.0)

    # Row 6 is hour 6: sin(pi/2)=1, cos(pi/2)=0.
    assert x.loc[6, "hour"] == pytest.approx(6.0)
    assert x.loc[6, "hour_sin"] == pytest.approx(1.0)
    assert x.loc[6, "hour_cos"] == pytest.approx(0.0, abs=1e-12)


def test_make_features_derives_calendar_from_dteday(csv_path: Path) -> None:
    df_2011, _ = data.load_split(csv_path)
    x, _ = data.make_features(df_2011)

    # 2011-01-01 is a Saturday -> dayofweek == 5, month == 1.
    assert x.loc[0, "dayofweek"] == pytest.approx(5.0)
    assert x.loc[0, "month"] == pytest.approx(1.0)


def test_target_matches_cnt(csv_path: Path) -> None:
    df_2011, _ = data.load_split(csv_path)
    _, y = data.make_features(df_2011)
    assert y.tolist() == df_2011["cnt"].astype(float).tolist()
