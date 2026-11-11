import gzip
import json
import os
from datetime import UTC, date, datetime, timedelta

import pandas as pd
from conftest import make_vp

from tram_mlops import status, storage

TODAY = date(2026, 11, 11)


def test_hfp_source_freshness_and_counts(tmp_path):
    now = datetime.now(UTC)
    records = [make_vp(veh=v, t=now - timedelta(seconds=30 - v)) for v in range(3)]
    path = tmp_path / f"tram_{TODAY}.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in records) + '{"veh": 1, "tst": "2026')  # mid-write

    source = status.hfp_source("Positions", "MQTT", tmp_path, "tram", TODAY, stale_after_s=120)
    assert source.records_today == 3
    assert source.newest_record == pd.Timestamp(records[-1]["tst"])
    assert source.size_bytes == path.stat().st_size
    assert source.state(pd.Timestamp(now)) == status.FRESH
    assert source.state(pd.Timestamp(now) + pd.Timedelta(minutes=10)) == status.STALE

    # Before today's file exists, the newest earlier file still tells when data last arrived.
    path.rename(tmp_path / "tram_2026-11-10.jsonl")
    source = status.hfp_source("Positions", "MQTT", tmp_path, "tram", TODAY, stale_after_s=120)
    assert source.path is None and source.records_today == 0 and source.last_write is not None

    missing = status.hfp_source("Events", "MQTT", tmp_path, "events", TODAY, stale_after_s=300)
    assert missing.state(pd.Timestamp(now)) == status.MISSING and missing.size_bytes == 0


def test_gtfsrt_sources(tmp_path):
    assert status.alerts_source(tmp_path, TODAY).state(pd.Timestamp.now(tz="UTC")) == status.MISSING
    snapshot = {"fetched_at": "2026-11-11T08:16:54+00:00", "count": 22, "alerts": []}
    (tmp_path / "alerts_2026-11-11.jsonl.gz").write_bytes(gzip.compress((json.dumps(snapshot) + "\n").encode()))
    alerts = status.alerts_source(tmp_path, TODAY)
    assert alerts.last_write == pd.Timestamp("2026-11-11T08:16:54Z")
    assert alerts.note == "22 alerts in the feed (all modes)"
    # Saved only on change, so an old snapshot is not a stale poller.
    assert alerts.state(pd.Timestamp("2026-11-12T08:00Z")) == status.FRESH

    path = tmp_path / "trip_updates_2026-11-11.jsonl.gz"
    path.write_bytes(gzip.compress(b"{}\n"))
    old = datetime(2026, 11, 11, 8, 0, tzinfo=UTC).timestamp()
    os.utime(path, (old, old))
    trips = status.trip_updates_source(tmp_path, TODAY, stale_after_s=120)
    assert trips.state(pd.Timestamp("2026-11-11T08:01Z")) == status.FRESH
    assert trips.state(pd.Timestamp("2026-11-11T08:05Z")) == status.STALE


def test_manifest_table(tmp_path):
    assert status.manifest_table(tmp_path).empty
    storage.save_manifest(tmp_path, {"partitions": {
        "2026-11-10": {"source": "tram_2026-11-10.jsonl", "raw_rows": 200, "rows": 150, "processing_version": 2,
                       "processed_at": "2026-11-11T06:33:13+00:00",
                       "dropped": {"missing_required": 40, "duplicate": 10}},
        "2026-11-11": {"source": "tram_2026-11-11.jsonl", "raw_rows": 0, "rows": 0},
    }})
    table = status.manifest_table(tmp_path)
    assert table["day"].tolist() == ["2026-11-11", "2026-11-10"]  # newest first
    row = table.iloc[1]
    assert (row["kept_pct"], row["missing_required"], row["duplicate"], row["implausible_delay"]) == (75.0, 40, 10, 0)
    assert row["processed_at"] == pd.Timestamp("2026-11-11T06:33:13Z")
    assert pd.isna(table.iloc[0]["kept_pct"]) and pd.isna(table.iloc[0]["processed_at"])


def test_model_info_reads_metrics_without_the_model(tmp_path):
    assert status.model_info(tmp_path) is None
    (tmp_path / "latest.json").write_text(json.dumps({"version": "20260918T091855Z"}))
    assert status.model_info(tmp_path) is None  # pointer to a missing version
    (tmp_path / "20260918T091855Z").mkdir()
    (tmp_path / "20260918T091855Z" / "metrics.json").write_text(json.dumps({"version": "20260918T091855Z",
                                                                           "train_rows": 10}))
    assert status.model_info(tmp_path) == {"version": "20260918T091855Z", "train_rows": 10}


def test_disk_usage_and_last_record_time(tmp_path):
    (tmp_path / "tram_a.jsonl").write_bytes(b"x" * 10)
    (tmp_path / "tram_b.jsonl").write_bytes(b"x" * 5)
    usage = status.disk_usage({"Positions": (tmp_path, "tram_*"), "Nothing": (tmp_path, "none_*")})
    assert usage.set_index("data").to_dict("index") == {"Positions": {"files": 2, "bytes": 15},
                                                         "Nothing": {"files": 0, "bytes": 0}}
    assert status.last_record_time(tmp_path / "tram_a.jsonl") is None
