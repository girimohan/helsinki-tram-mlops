"""End to end on the bundled sample: what a fresh clone (and CI) runs."""

import json

import pytest

from tram_mlops import process, train
from tram_mlops.config import load_config


@pytest.fixture
def isolated_paths(tmp_path, monkeypatch):
    # No collected data, so process falls back to the bundled sample.
    monkeypatch.setenv("TRAM_PATHS_RAW_DIR", str(tmp_path / "raw"))
    monkeypatch.setenv("TRAM_PATHS_PROCESSED_DIR", str(tmp_path / "processed"))
    monkeypatch.setenv("TRAM_PATHS_MODELS_DIR", str(tmp_path / "models"))
    load_config.cache_clear()
    yield tmp_path
    load_config.cache_clear()


def test_sample_processes_and_trains(isolated_paths):
    process.run()
    metrics = train.run()

    assert metrics["test_rows"] > 1000
    # The sample is 45 min of morning rush; the model should clearly beat persistence overall
    # and must not lose to it on trams already en route.
    assert metrics["model"]["mae_s"] < 0.8 * metrics["persistence_baseline"]["mae_s"]
    en_route = metrics["segments"]["en_route"]
    assert en_route["model"]["mae_s"] <= en_route["persistence_baseline"]["mae_s"]

    pointer = json.loads((isolated_paths / "models" / "latest.json").read_text())
    assert (isolated_paths / "models" / pointer["version"] / "model.joblib").exists()
