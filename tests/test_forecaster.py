import json
from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest

from tram_mlops import forecaster as fc
from tram_mlops import monitoring
from tram_mlops.process import clean
from tram_mlops.train import save, train

T0 = datetime(2026, 2, 4, 8, 0, tzinfo=UTC)  # synthetic_raw starts here, one record per tram every 10 s


@pytest.fixture
def setup(tmp_path, synthetic_raw):
    raw_dir, models_dir, out_dir = tmp_path / "raw", tmp_path / "models", tmp_path / "predictions"
    raw_dir.mkdir()
    (raw_dir / "tram_2026-02-04.jsonl").write_text("".join(json.dumps(r) + "\n" for r in synthetic_raw))
    model, metrics = train(clean(pd.DataFrame(synthetic_raw)), horizon_s=300, tolerance_s=60, min_rows=100)
    save(model, metrics, models_dir)
    f = fc.Forecaster(raw_dir, models_dir, out_dir, horizon_s=300, tolerance_s=60, window_min=15, log_every_s=30)
    return f, synthetic_raw, out_dir


def at(minutes, seconds=0):
    return pd.Timestamp(T0 + timedelta(minutes=minutes, seconds=seconds))


def test_forecasts_are_published_logged_once_and_scored_like_training(setup):
    f, raw, out_dir = setup
    first = f.step(at(30))
    assert first == {"forecasts": 8, "logged": 8, "scored": 0}
    latest = fc.read_latest(out_dir)
    assert latest["model_version"] == f.model_version and len(latest["forecasts"]) == 8

    # 20 s later: forecasts refresh, but each tram is logged at most every 30 s.
    assert f.step(at(30, 20))["logged"] == 0

    # Once horizon + tolerance has passed, the 8 forecasts are scored.
    assert f.step(at(36, 10))["scored"] == 8
    scored = monitoring.read_scored(out_dir, T0.date(), T0.date())
    assert len(scored) == 8 and scored["actual"].notna().all()

    # The actual is the delay the same tram reported 300 s after the forecast's observation.
    truth = {(r["veh"], pd.Timestamp(r["tst"])): -r["dl"] for r in raw}
    for _, row in scored.iterrows():
        assert row["actual"] == truth[(row["vehicle"], row["observed_at"] + pd.Timedelta(seconds=300))]
        assert row["delay_now"] == truth[(row["vehicle"], row["observed_at"])]
    assert set(scored["segment"]) == {"en_route"}


def test_live_metrics_compare_model_and_persistence_on_the_same_rows(setup):
    f, _, out_dir = setup
    for minute in range(30, 50):
        f.step(at(minute))
    scored = monitoring.read_scored(out_dir, T0.date(), T0.date())
    metrics = monitoring.live_metrics(scored, now=at(50))
    hour = metrics["1h"]
    assert hour["forecasts"] == len(scored) and hour["matched_pct"] == 100.0
    assert hour["overall"]["rows"] == len(scored)
    assert hour["segments"]["en_route"]["rows"] == len(scored)
    # Lateness drifts steadily in the synthetic data, so the model should beat persistence.
    assert hour["overall"]["model_mae_s"] < hour["overall"]["persistence_mae_s"]
    assert hour["model_versions"] == [f.model_version]
    over_time = monitoring.mae_over_time(scored, freq="5min")
    assert list(over_time.columns) == ["Model", "Persistence"] and len(over_time) >= 3


def test_no_model_means_no_forecasts_until_one_is_saved(tmp_path, synthetic_raw):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "tram_2026-02-04.jsonl").write_text("".join(json.dumps(r) + "\n" for r in synthetic_raw))
    f = fc.Forecaster(raw_dir, tmp_path / "models", tmp_path / "out", horizon_s=300, tolerance_s=60)
    assert f.step(at(30)) == {"forecasts": 0, "logged": 0, "scored": 0}
    assert fc.read_latest(tmp_path / "out") is None

    model, metrics = train(clean(pd.DataFrame(synthetic_raw)), horizon_s=300, tolerance_s=60, min_rows=100)
    save(model, metrics, tmp_path / "models")
    assert f.step(at(31))["forecasts"] == 8


def test_forecasts_older_than_the_window_expire_instead_of_counting_as_misses(setup):
    f, _, out_dir = setup
    f.step(at(30))
    # The service was down for 20 minutes: the observations behind these forecasts are gone.
    assert f.step(at(50))["scored"] == 0
    assert f.stats["expired"] == 8 and f.stats["unmatched"] == 0
    assert not list(out_dir.glob("forecasts_*"))


def test_stale_trams_are_not_forecast(setup):
    f, _, _ = setup
    # The data ends at 10:00 UTC; at 10:05 every tram's newest observation is 5 min old.
    assert f.step(at(125))["forecasts"] == 0
