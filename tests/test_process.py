import json
from datetime import datetime, timedelta, timezone
from functools import partial
from unittest.mock import patch

import pandas as pd
import pytest

from conftest import make_vp
from tram_mlops.config import load_config, resolve_path
from tram_mlops import process
from tram_mlops.process import clean, load_raw, load_recent, summarize


def test_delay_sign_is_positive_when_behind_schedule():
    # HFP dl = -120 means 120 s behind schedule.
    df = clean(pd.DataFrame([make_vp(dl=-120), make_vp(veh=2, dl=30)]))
    assert df.set_index("vehicle")["delay_s"].to_dict() == {1: 120, 2: -30}


def test_duplicates_are_removed():
    df = clean(pd.DataFrame([make_vp(), make_vp(), make_vp(veh=2)]))
    assert len(df) == 2


def test_rows_missing_delay_or_timestamp_are_dropped():
    raw = pd.DataFrame([make_vp(), make_vp(veh=2, dl=None), make_vp(veh=3, tst="garbage")])
    assert clean(raw)["vehicle"].tolist() == [1]


def test_local_hour_uses_helsinki_time():
    raw = pd.DataFrame([make_vp(t=datetime(2026, 2, 4, 10, 30, tzinfo=timezone.utc))])
    assert clean(raw)["hour"].iloc[0] == 12  # UTC+2 in winter


def test_on_time_window():
    raw = pd.DataFrame([make_vp(veh=1, dl=61), make_vp(veh=2, dl=0), make_vp(veh=3, dl=-181)])
    on_time = clean(raw, on_time_window=(-60, 180)).set_index("vehicle")["on_time"]
    assert on_time.to_dict() == {1: False, 2: True, 3: False}


def test_missing_required_field_raises():
    with pytest.raises(ValueError, match="dl"):
        clean(pd.DataFrame([{"veh": 1, "tst": "2026-02-04T10:00:00Z"}]))


def test_load_raw_skips_malformed_lines(tmp_path):
    path = tmp_path / "raw.jsonl"
    path.write_text(json.dumps(make_vp()) + "\n{not json\n[1, 2]\n" + json.dumps(make_vp(veh=2)) + "\n")
    assert len(load_raw([path])) == 2


def test_bundled_sample_processes():
    sample = resolve_path(load_config()["paths"]["sample_file"])
    df = clean(load_raw([sample]))
    stats = summarize(df)
    # 13,200 lines: every record appears twice, and 181 unique records have no delay.
    assert stats["rows"] == 6419
    assert stats["vehicles"] > 100
    assert df["delay_s"].between(-3600, 3600).all()


def test_implausible_delays_are_dropped():
    raw = pd.DataFrame([make_vp(veh=1, dl=-1700), make_vp(veh=2, dl=-6000), make_vp(veh=3, dl=5000)])
    assert clean(raw, max_abs_delay_s=1800)["vehicle"].tolist() == [1]


def test_excluded_lines_are_dropped():
    raw = pd.DataFrame([make_vp(veh=1, desi="000"), make_vp(veh=2, desi="4")])
    assert clean(raw, exclude_lines=["000"])["line"].tolist() == ["4"]


def _write_jsonl(path, records, trailing=""):
    path.write_text("".join(json.dumps(r) + "\n" for r in records) + trailing, encoding="utf-8")


def test_load_recent_reads_only_the_window(tmp_path):
    t0 = datetime(2026, 2, 4, 10, 0, tzinfo=timezone.utc)
    records = [make_vp(veh=i % 5, t=t0 + timedelta(seconds=10 * i)) for i in range(600)]
    _write_jsonl(tmp_path / "tram_2026-02-04.jsonl", records, trailing='{"partial": ')
    now = pd.Timestamp(t0 + timedelta(seconds=10 * 599))
    # A tiny chunk size forces several backwards reads.
    with patch.object(process, "_tail_records", partial(process._tail_records, chunk_size=512)):
        raw = load_recent(tmp_path, pd.Timedelta(minutes=5), now=now)
    assert len(raw) == 31  # 300 s at one record per 10 s, inclusive of both ends
    assert raw["tst"].is_monotonic_increasing


def test_load_recent_spans_utc_midnight(tmp_path):
    before = datetime(2026, 2, 4, 23, 59, 30, tzinfo=timezone.utc)
    after = datetime(2026, 2, 5, 0, 0, 30, tzinfo=timezone.utc)
    _write_jsonl(tmp_path / "tram_2026-02-04.jsonl", [make_vp(t=before - timedelta(hours=1)), make_vp(t=before)])
    _write_jsonl(tmp_path / "tram_2026-02-05.jsonl", [make_vp(t=after)])
    raw = load_recent(tmp_path, pd.Timedelta(minutes=5), now=pd.Timestamp(after))
    assert len(raw) == 2


def test_load_recent_without_files(tmp_path):
    assert load_recent(tmp_path, pd.Timedelta(minutes=5)).empty
