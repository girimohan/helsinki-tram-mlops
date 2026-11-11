"""Feature and target construction for delay forecasting.

Task: given a tram's current observation, predict its delay ``horizon_s`` seconds
later on the same journey. Features only use the current and earlier observations.
"""

from __future__ import annotations

from typing import Literal

import pandas as pd

CATEGORICAL = ["line", "direction"]
NUMERIC = [
    "delay_s",
    "delay_change_60s",
    "journey_elapsed_s",
    "speed_mps",
    "doors_open",
    "at_stop",
    "occupancy",
    "lat",
    "lon",
    "hour",
    "weekday",
    "minute_of_day",
]
FEATURES = CATEGORICAL + NUMERIC
TARGET = "future_delay_s"


def _trip_key(df: pd.DataFrame) -> pd.Series:
    return df["vehicle"].astype(str) + "#" + df["journey_id"]


def _lookup(df: pd.DataFrame, offset_s: float, tolerance_s: float,
            direction: Literal["backward", "forward", "nearest"]) -> pd.Series:
    """Delay of the same trip at the observation nearest ``timestamp + offset_s``."""
    left = pd.DataFrame({
        "key": _trip_key(df),
        "t": df["timestamp"] + pd.Timedelta(seconds=offset_s),
        "row": df.index,
    }).sort_values("t")
    right = pd.DataFrame({
        "key": _trip_key(df),
        "t": df["timestamp"],
        "matched_delay": df["delay_s"],
    }).sort_values("t")
    merged = pd.merge_asof(
        left, right, on="t", by="key", direction=direction,
        tolerance=pd.Timedelta(seconds=tolerance_s),
    )
    return merged.set_index("row")["matched_delay"].reindex(df.index)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add model features to a cleaned positions table (see process.clean)."""
    out = df.copy()
    local = out["local_time"]
    out["minute_of_day"] = local.dt.hour * 60 + local.dt.minute
    out["at_stop"] = out["at_stop"].astype(int)

    # Scheduled start is "HH:MM" local time on a 24-hour clock. An operating day runs until about
    # 04:30 the next morning, so a trip on operating day D that starts at 01:50 starts on D + 1.
    day_start = pd.to_datetime(out["operating_day"], errors="coerce").dt.tz_localize(
        local.dt.tz, ambiguous="NaT", nonexistent="shift_forward")
    start_offset = pd.to_timedelta(out["start_time"].astype(str) + ":00", errors="coerce")
    elapsed = (local - (day_start + start_offset)).dt.total_seconds()
    # No tram journey lasts 12 hours, so a value that large means the start was after midnight.
    out["journey_elapsed_s"] = elapsed.where(elapsed < 12 * 3600, elapsed - 86400)

    earlier = _lookup(out, -60, 20, "nearest")
    out["delay_change_60s"] = out["delay_s"] - earlier
    for col in CATEGORICAL:
        out[col] = out[col].astype("string").fillna("unknown").astype(str)
    return out


def future_delay(feats: pd.DataFrame, horizon_s: float, tolerance_s: float) -> pd.Series:
    """The target: each row's trip delay at the first observation ``horizon_s`` (+ ``tolerance_s``) later.

    Training (``build_training_set``) and live scoring (``forecaster``) both use this, so the
    live "true answer" is defined exactly as the model was trained on.
    """
    return _lookup(feats, horizon_s, tolerance_s, "forward")


def build_training_set(df: pd.DataFrame, horizon_s: float, tolerance_s: float) -> pd.DataFrame:
    """Features plus the delay observed ``horizon_s`` later; rows without a future match are dropped."""
    if df.empty:
        return pd.DataFrame(columns=FEATURES + [TARGET, "timestamp"])
    out = add_features(df)
    out[TARGET] = future_delay(out, horizon_s, tolerance_s)
    out = out.dropna(subset=[TARGET])
    return out[FEATURES + [TARGET, "timestamp", "vehicle"]].reset_index(drop=True)
