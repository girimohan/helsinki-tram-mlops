import json
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from conftest import make_vp
from tram_mlops.config import PROJECT_ROOT, load_config
from tram_mlops.process import clean
from tram_mlops.train import save, train

DASHBOARD = str(PROJECT_ROOT / "src" / "tram_mlops" / "dashboard.py")


@pytest.fixture
def isolated_paths(tmp_path, monkeypatch):
    monkeypatch.setenv("TRAM_PATHS_RAW_DIR", str(tmp_path / "raw"))
    monkeypatch.setenv("TRAM_PATHS_PROCESSED_FILE", str(tmp_path / "positions.parquet"))
    monkeypatch.setenv("TRAM_PATHS_MODELS_DIR", str(tmp_path / "models"))
    load_config.cache_clear()
    yield tmp_path
    load_config.cache_clear()


def test_dashboard_renders_from_sample_without_model(isolated_paths):
    app = AppTest.from_file(DASHBOARD, default_timeout=60).run()
    assert not app.exception
    assert (isolated_paths / "positions.parquet").exists()
    assert any("No trained model" in i.value for i in app.info)


def test_dashboard_shows_forecasts_with_model(isolated_paths, synthetic_raw):
    df = clean(pd.DataFrame(synthetic_raw))
    df.to_parquet(isolated_paths / "positions.parquet")
    model, metrics = train(df, horizon_s=300, tolerance_s=30, min_rows=100)
    save(model, metrics, isolated_paths / "models")

    app = AppTest.from_file(DASHBOARD, default_timeout=60).run()
    assert not app.exception
    labels = [m.label for m in app.metric]
    assert "Persistence baseline MAE" in labels


def test_dashboard_live_mode_reads_fresh_raw_data(isolated_paths):
    raw_dir = isolated_paths / "raw"
    raw_dir.mkdir()
    now = datetime.now(timezone.utc)
    # Trams 1-5 run 50-250 s late; the ingester writes records in time order.
    records = [make_vp(veh=v, t=now - timedelta(seconds=s), dl=-50 * v, desi=str(v))
               for s in range(110, -1, -10) for v in range(1, 6)]
    day = now.strftime("%Y-%m-%d")
    (raw_dir / f"tram_{day}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))

    app = AppTest.from_file(DASHBOARD, default_timeout=60).run()
    assert not app.exception
    assert app.sidebar.radio[0].value == "Live"
    metrics = {m.label: m.value for m in app.metric}
    assert metrics["Trams in service"] == "5"
    assert metrics["More than 3 min late"] == "2"  # trams 4 and 5
