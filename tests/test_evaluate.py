from datetime import UTC, date, datetime, timedelta

import numpy as np
import pandas as pd
import pytest
from conftest import make_vp

from tram_mlops import evaluate, storage
from tram_mlops.config import load_config
from tram_mlops.features import TARGET
from tram_mlops.process import clean

DAY1, DAY2 = date(2026, 2, 4), date(2026, 2, 5)


def two_days(n_trams=6):
    """Trams doing 30-minute trips 22:00-02:00 UTC across midnight, one record every 10 s.

    Each tram's lateness drifts at its own rate within a trip, so recent change predicts the next 5 min.
    """
    rng = np.random.default_rng(1)
    t0 = datetime(2026, 2, 4, 22, 0, tzinfo=UTC)  # 00:00 Helsinki on 5 Feb
    records = []
    for veh in range(1, n_trams + 1):
        drift = rng.normal(0.3, 0.2)
        for trip in range(8):
            start = t0 + timedelta(minutes=30 * trip)
            local_start = start + timedelta(hours=2)
            delay = 0.0
            for step in range(180):
                delay += drift + rng.normal(0, 2)
                records.append(make_vp(veh=veh, t=start + timedelta(seconds=10 * step), dl=-int(delay),
                                       desi=str(veh % 2 + 1), start=f"{local_start:%H:%M}", oday="2026-02-05"))
    return clean(pd.DataFrame(records))


@pytest.fixture
def partitions(tmp_path, monkeypatch):
    df = two_days()
    root = tmp_path / "processed"
    for day, part in df.groupby(df["timestamp"].dt.date):
        storage.write_partition(root, day, part.reset_index(drop=True))
    monkeypatch.setenv("TRAM_PATHS_PROCESSED_DIR", str(root))
    monkeypatch.setenv("TRAM_EVALUATE_MIN_TEST_ROWS", "100")
    load_config.cache_clear()
    yield root, df
    load_config.cache_clear()


def test_backtest_day_trains_only_on_targets_before_the_test_day(partitions):
    _, df = partitions
    pred = evaluate.backtest_day(df, DAY2, horizon_s=300, tolerance_s=60)
    start = pd.Timestamp(DAY2, tz="UTC")
    assert (pred["timestamp"] >= start).all()
    # Two hours before midnight, minus the purged last 5 minutes, with a matching target.
    assert 0 < pred["train_rows"].iloc[0] < 6 * 720
    assert set(evaluate.METHODS) <= set(pred.columns)
    # A day with nothing before it can't be tested.
    assert evaluate.backtest_day(df[df["timestamp"] >= start], DAY2, 300, 60) is None


def test_line_hour_baseline_uses_training_means_with_fallback():
    def rows(line, hour, delay, target, elapsed=100):
        return {"line": line, "hour": hour, "delay_s": delay, TARGET: target, "journey_elapsed_s": elapsed}
    train = pd.DataFrame([rows("1", 8, 0, 10)] * 40 + [rows("2", 8, 0, -20)] * 5)
    test = pd.DataFrame([rows("1", 8, 100, 0), rows("2", 8, 100, 0), rows("3", 9, 100, 0)])
    # Line 1 learned +10; line 2 has too few rows and line 3 none, so both use the segment mean.
    fallback = (40 * 10 + 5 * -20) / 45
    expected = [110.0, round(100 + fallback, 4), round(100 + fallback, 4)]
    assert evaluate.line_hour_baseline(train, test).round(4).tolist() == expected


def test_bootstrap_interval_brackets_the_point_estimate():
    rng = np.random.default_rng(0)
    n = 4000
    truth = rng.normal(0, 60, n)
    pred = pd.DataFrame({
        "vehicle": rng.integers(0, 50, n), "journey_id": rng.integers(0, 20, n).astype(str), TARGET: truth,
        "model": truth + rng.normal(0, 30, n), "persistence": truth + rng.normal(0, 40, n),
        "line_hour": truth + rng.normal(0, 40, n),
    })
    result = evaluate.bootstrap_differences(pred, reps=300)["persistence"]
    assert result["ci_low_s"] < result["diff_s"] < result["ci_high_s"] < 0


def test_run_writes_report_and_readme_block(partitions, tmp_path):
    report = evaluate.run(train_days=1, reps=50, out_dir=tmp_path / "reports")
    assert report["test_days"] == [DAY2.isoformat()]
    assert report["pooled"]["rows"] > 1000
    en_route = report["pooled"]["segments"]["en_route"]
    # Every method is scored on the same rows. (Accuracy itself is tested on real data in
    # test_pipeline.py; this synthetic night is too short for a fair model-vs-baseline contest.)
    assert all(en_route[m] > 0 for m in evaluate.METHODS) and en_route["rows"] <= report["pooled"]["rows"]
    diffs = report["pooled"]["en_route_vs_baselines"]["persistence"]
    assert diffs["diff_s"] == pytest.approx(en_route["model"] - en_route["persistence"], abs=0.02)
    assert (tmp_path / "reports" / f"{report['run']}.json").exists()
    markdown = (tmp_path / "reports" / "latest.md").read_text(encoding="utf-8")
    assert "En route" in markdown and "95% interval" in markdown

    readme = tmp_path / "README.md"
    readme.write_text(f"# x\n{evaluate.README_START}\nold\n{evaluate.README_END}\ntail\n", encoding="utf-8")
    evaluate.update_readme(markdown, readme)
    text = readme.read_text(encoding="utf-8")
    assert "old" not in text and markdown in text and text.endswith("tail\n")
    plain = tmp_path / "plain.md"
    plain.write_text("no markers", encoding="utf-8")
    with pytest.raises(ValueError):
        evaluate.update_readme(markdown, plain)
