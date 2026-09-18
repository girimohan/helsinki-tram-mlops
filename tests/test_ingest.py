import json

from conftest import make_vp
from tram_mlops.ingest import RawWriter, parse_message

DAY_1 = 1770200000.0  # 2026-02-04 UTC


def test_parse_message_extracts_vp():
    vp = make_vp()
    assert parse_message(json.dumps({"VP": vp}).encode()) == vp


def test_parse_message_ignores_other_events():
    assert parse_message(json.dumps({"DOO": {}}).encode()) is None
    assert parse_message(b"\xff\xfe not json") is None
    assert parse_message(b"[1, 2]") is None


def test_writer_downsamples_per_vehicle(tmp_path):
    writer = RawWriter(tmp_path, sample_interval_s=10)
    kept = [
        writer.write(make_vp(veh=1), DAY_1),
        writer.write(make_vp(veh=1), DAY_1 + 5),   # too soon
        writer.write(make_vp(veh=2), DAY_1 + 5),   # other vehicle
        writer.write(make_vp(veh=1), DAY_1 + 10),
    ]
    writer.close()
    assert kept == [True, False, True, True]
    lines = (tmp_path / "tram_2026-02-04.jsonl").read_text().splitlines()
    assert [json.loads(l)["veh"] for l in lines] == [1, 2, 1]


def test_writer_rolls_over_daily(tmp_path):
    writer = RawWriter(tmp_path)
    writer.write(make_vp(), DAY_1)
    writer.write(make_vp(), DAY_1 + 86400)
    writer.close()
    assert sorted(p.name for p in tmp_path.iterdir()) == ["tram_2026-02-04.jsonl", "tram_2026-02-05.jsonl"]
