"""Shared feature/target names — the contract between modules. Keep tiny/stable."""

TARGET = "cnt"

CALENDAR_FEATURES = [
    "hour",
    "dayofweek",
    "month",
    "workingday",
    "holiday",
    "season",
    "hour_sin",
    "hour_cos",
]
WEATHER_FEATURES = ["temp", "hum", "windspeed", "weathersit"]

# Stable order used everywhere X is built or consumed.
FEATURES = CALENDAR_FEATURES + WEATHER_FEATURES

# Registry / experiment defaults
MODEL_NAME = "bike_demand"
EXPERIMENT = "driftguard"
