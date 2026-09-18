from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from conftest import make_vp
from tram_mlops.features import FEATURES, TARGET, build_training_set
from tram_mlops.process import clean
from tram_mlops.train import InsufficientDataError, load_latest, predict_delay, save, time_split, train

T0 = datetime(2026, 2, 4, 10, 0, tzinfo=timezone.utc)


def test_target_is_future_delay_of_same_trip():
    raw = [make_vp(veh=1, t=T0 + timedelta(seconds=60 * i), dl=-10 * i) for i in range(10)]
    # A different vehicle at the target time must not be used.
    raw.append(make_vp(veh=2, t=T0 + timedelta(seconds=300), dl=-999))
    data = build_training_set(clean(pd.DataFrame(raw)), horizon_s=300, tolerance_s=30)
    data = data[data["vehicle"] == 1]
    assert data["delay_s"].tolist() == [0, 10, 20, 30, 40]
    assert data[TARGET].tolist() == [50, 60, 70, 80, 90]


def test_target_not_taken_from_a_different_journey():
    raw = [make_vp(t=T0, dl=0, start="12:00"), make_vp(t=T0 + timedelta(seconds=300), dl=-500, start="12:30")]
    assert build_training_set(clean(pd.DataFrame(raw)), 300, 30).empty


def test_delay_change_only_looks_backwards():
    raw = [make_vp(t=T0 + timedelta(seconds=60 * i), dl=-10 * i) for i in range(3)]
    feats = build_training_set(clean(pd.DataFrame(raw)), 60, 10)
    assert feats["delay_change_60s"].isna().iloc[0]
    assert feats["delay_change_60s"].iloc[1] == 10


def test_train_beats_or_matches_persistence_and_roundtrips(synthetic_raw, tmp_path):
    df = clean(pd.DataFrame(synthetic_raw))
    model, metrics = train(df, horizon_s=300, tolerance_s=30, min_rows=100)
    assert metrics["test_rows"] > 0 and metrics["train_rows"] > metrics["test_rows"]
    # Lateness drifts steadily, so a model using the recent trend should beat persistence.
    assert metrics["model"]["mae_s"] < metrics["persistence_baseline"]["mae_s"]

    save(model, metrics, tmp_path)
    loaded_model, loaded_metrics = load_latest(tmp_path)
    assert loaded_metrics["features"] == FEATURES
    preds = predict_delay(loaded_model, df.head(50))
    assert len(preds) == 50 and preds.notna().all()


def test_train_refuses_too_little_data():
    raw = [make_vp(t=T0 + timedelta(seconds=i)) for i in range(60)]
    with pytest.raises(InsufficientDataError):
        train(clean(pd.DataFrame(raw)), horizon_s=300, tolerance_s=30)


def test_load_latest_without_model(tmp_path):
    assert load_latest(tmp_path) is None


def test_time_split_purges_targets_that_reach_into_test_period():
    ts = pd.date_range("2026-02-04 10:00", periods=100, freq="10s", tz="UTC")
    train_df, test_df = time_split(pd.DataFrame({"timestamp": ts}), test_fraction=0.25, horizon_s=300)
    cutoff = test_df["timestamp"].min()
    assert (train_df["timestamp"] + pd.Timedelta(seconds=300) < cutoff).all()
    assert len(train_df) + len(test_df) < 100


def test_metrics_report_segments(synthetic_raw):
    _, metrics = train(clean(pd.DataFrame(synthetic_raw)), horizon_s=300, tolerance_s=30, min_rows=100)
    segments = metrics["segments"]
    assert sum(s["rows"] for s in segments.values()) == metrics["test_rows"]
    assert all("mae_s" in s["model"] and "mae_s" in s["persistence_baseline"] for s in segments.values())
