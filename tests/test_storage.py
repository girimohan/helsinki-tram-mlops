import gzip
import json
import os
from datetime import UTC, date, datetime, timedelta

import pandas as pd
import pyarrow.parquet as pq
import pytest
from conftest import make_vp

from tram_mlops import process, storage
from tram_mlops.config import load_config


def _write_raw(path, records):
    text = "".join(json.dumps(r) + "\n" for r in records)
    if path.suffix == ".gz":
        with gzip.open(path, "wt", encoding="utf-8") as f:
            f.write(text)
    else:
        path.write_text(text, encoding="utf-8")


def _day_records(day, n=30, veh=1):
    t0 = datetime(day.year, day.month, day.day, 8, 0, tzinfo=UTC)
    return [make_vp(veh=veh, t=t0 + timedelta(seconds=10 * i), dl=-i, oday=day.isoformat()) for i in range(n)]


@pytest.fixture
def paths(tmp_path, monkeypatch):
    monkeypatch.setenv("TRAM_PATHS_RAW_DIR", str(tmp_path / "raw"))
    monkeypatch.setenv("TRAM_PATHS_PROCESSED_DIR", str(tmp_path / "processed"))
    load_config.cache_clear()
    (tmp_path / "raw").mkdir()
    yield tmp_path / "raw", tmp_path / "processed"
    load_config.cache_clear()


D1, D2, D3 = date(2026, 11, 9), date(2026, 11, 10), date(2026, 11, 11)


def test_partitions_share_one_schema_and_read_by_range(paths):
    raw, root = paths
    _write_raw(raw / f"tram_{D1}.jsonl", _day_records(D1))
    # A day where optional fields are missing must still get the same column types.
    sparse = [{k: v for k, v in r.items() if k not in ("stop", "occu", "drst")} for r in _day_records(D2, veh=2)]
    _write_raw(raw / f"tram_{D2}.jsonl", sparse)
    process.run()

    assert storage.list_partitions(root) == [D1, D2]
    for day in (D1, D2):
        assert pq.read_schema(storage.partition_path(root, day)) == storage.POSITIONS_SCHEMA
    assert storage.read_positions(root)["vehicle"].unique().tolist() == [1, 2]
    only_d2 = storage.read_positions(root, start=D2, columns=["vehicle", "delay_s"])
    assert list(only_d2.columns) == ["vehicle", "delay_s"] and set(only_d2["vehicle"]) == {2}
    assert storage.read_positions(root, start=D3).empty


def test_unchanged_days_are_skipped_and_changed_days_reprocessed(paths):
    raw, root = paths
    _write_raw(raw / f"tram_{D1}.jsonl", _day_records(D1))
    _write_raw(raw / f"tram_{D2}.jsonl", _day_records(D2))
    assert process.run()["processed"] == [str(D1), str(D2)]
    assert process.run() == {"processed": [], "skipped": [str(D1), str(D2)]}

    # The collector appends to today's file.
    with open(raw / f"tram_{D2}.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(make_vp(veh=9, t=datetime(2026, 11, 10, 9, tzinfo=UTC))) + "\n")
    assert process.run() == {"processed": [str(D2)], "skipped": [str(D1)]}
    assert 9 in set(storage.read_positions(root, D2, D2)["vehicle"])

    manifest = storage.load_manifest(root)["partitions"][str(D2)]
    assert manifest["raw_rows"] == 31 and manifest["rows"] == 31
    assert manifest["processing_version"] == process.PROCESSING_VERSION


def test_since_rebuild_and_version_bump_force_reprocessing(paths, monkeypatch):
    raw, _ = paths
    for day in (D1, D2):
        _write_raw(raw / f"tram_{day}.jsonl", _day_records(day))
    process.run()
    assert process.run(since=D2)["processed"] == [str(D2)]
    assert process.run(rebuild=True)["processed"] == [str(D1), str(D2)]
    monkeypatch.setattr(process, "PROCESSING_VERSION", process.PROCESSING_VERSION + 1)
    assert process.run()["processed"] == [str(D1), str(D2)]


def test_compaction_keeps_recent_days_and_does_not_trigger_reprocessing(paths):
    raw, root = paths
    for day in (D1, D2, D3):
        _write_raw(raw / f"tram_{day}.jsonl", _day_records(day))
    process.run()
    before = storage.read_positions(root, D1, D1)

    result = process.run(compact=True, today=D3)
    # Only D1 is older than yesterday; the live reader still needs D2 and D3 uncompressed.
    assert sorted(p.name for p in raw.iterdir()) == [
        f"tram_{D1}.jsonl.gz", f"tram_{D2}.jsonl", f"tram_{D3}.jsonl"]
    # Same content under a new name: recognised by hash, not reprocessed.
    assert result == {"processed": [], "skipped": [str(D1), str(D2), str(D3)]}
    assert storage.load_manifest(root)["partitions"][str(D1)]["source"] == f"tram_{D1}.jsonl.gz"
    pd.testing.assert_frame_equal(storage.read_positions(root, D1, D1), before)


def test_compaction_keeps_original_if_archive_is_incomplete(tmp_path):
    src = tmp_path / f"tram_{D1}.jsonl"
    _write_raw(src, _day_records(D1))
    # A leftover archive from an interrupted run that doesn't hold the whole file.
    with gzip.open(tmp_path / f"tram_{D1}.jsonl.gz", "wt") as f:
        f.write("{}\n")
    assert storage.compact_raw(tmp_path, today=D3) == []
    assert src.exists()


def test_failed_write_leaves_previous_partition_intact(tmp_path, monkeypatch):
    df = process.clean(pd.DataFrame(_day_records(D1)))
    path = storage.write_partition(tmp_path, D1, df)

    def boom(*args, **kwargs):
        raise OSError("disk full")
    monkeypatch.setattr(storage.pq, "write_table", boom)
    with pytest.raises(OSError):
        storage.write_partition(tmp_path, D1, df.head(1))
    assert len(pq.read_table(path)) == len(df)
    assert [p.name for p in path.parent.iterdir()] == [storage.PARTITION_FILE]


def test_gzipped_and_plain_files_hash_the_same(tmp_path):
    plain, packed = tmp_path / "a.jsonl", tmp_path / "a.jsonl.gz"
    _write_raw(plain, _day_records(D1))
    _write_raw(packed, _day_records(D1))
    assert storage.file_sha256(plain) == storage.file_sha256(packed)
    os.utime(plain, (0, 0))
    assert storage.file_sha256(plain) == storage.file_sha256(packed)


def test_chunked_cleaning_matches_cleaning_everything_at_once(synthetic_raw, tmp_path):
    path = tmp_path / "tram_2026-02-04.jsonl"
    # Duplicates across chunk boundaries must still be removed.
    _write_raw(path, synthetic_raw + synthetic_raw[:500])
    chunked, counts = process.clean_files([path], chunk_size=777)
    whole = process.clean_with_config(process.load_raw([path]))
    pd.testing.assert_frame_equal(chunked, whole)
    assert counts["duplicate"] == 500 and counts["raw_rows"] == len(synthetic_raw) + 500


@pytest.mark.skipif(os.name != "posix", reason="POSIX file modes")
def test_written_files_are_readable_by_others(paths):
    raw, root = paths
    _write_raw(raw / f"tram_{D1}.jsonl", _day_records(D1))
    process.run()
    for path in (storage.partition_path(root, D1), root / storage.MANIFEST):
        assert path.stat().st_mode & 0o777 == storage.FILE_MODE
