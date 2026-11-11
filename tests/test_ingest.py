import json

from conftest import make_vp

from tram_mlops.ingest import EventWriter, RawWriter, build_client, event_topics, parse_event, parse_message

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
    assert [json.loads(line)["veh"] for line in lines] == [1, 2, 1]


def test_writer_rolls_over_daily(tmp_path):
    writer = RawWriter(tmp_path)
    writer.write(make_vp(), DAY_1)
    writer.write(make_vp(), DAY_1 + 86400)
    writer.close()
    assert sorted(p.name for p in tmp_path.iterdir()) == ["tram_2026-02-04.jsonl", "tram_2026-02-05.jsonl"]


def test_downsampling_is_per_operator_and_vehicle(tmp_path):
    writer = RawWriter(tmp_path, sample_interval_s=10)
    kept = [writer.write(make_vp(veh=1, oper=40), DAY_1), writer.write(make_vp(veh=1, oper=22), DAY_1 + 1)]
    writer.close()
    assert kept == [True, True]


def test_parse_event_tags_any_single_event():
    assert parse_event(json.dumps({"ARS": {"veh": 1, "stop": 123}}).encode()) == ("ARS", {"veh": 1, "stop": 123})
    assert parse_event(json.dumps({"VP": {}, "ARS": {}}).encode()) is None
    assert parse_event(json.dumps({"ARS": 5}).encode()) is None


def test_event_topics_swap_the_event_segment():
    assert event_topics("/hfp/v2/journey/ongoing/vp/tram/#", ["ars", "DEP"]) == [
        "/hfp/v2/journey/ongoing/ars/tram/#", "/hfp/v2/journey/ongoing/dep/tram/#"]


def test_events_are_kept_in_full_in_their_own_file(tmp_path):
    events = EventWriter(tmp_path)
    for i in range(3):  # within one second: positions would be downsampled, events are not
        events.write("ARS", {"veh": 1, "stop": 100 + i}, DAY_1 + i * 0.1)
    events.close()
    lines = [json.loads(line) for line in (tmp_path / "events_2026-02-04.jsonl").read_text().splitlines()]
    assert [(r["event"], r["stop"]) for r in lines] == [("ARS", 100), ("ARS", 101), ("ARS", 102)]


def test_messages_are_routed_by_event_type(tmp_path):
    positions, events = RawWriter(tmp_path, 10), EventWriter(tmp_path)
    client = build_client(["/t"], positions, events)

    class Msg:
        def __init__(self, payload):
            self.topic, self.payload = "/t", json.dumps(payload).encode()

    client.on_message(client, None, Msg({"VP": make_vp()}))
    client.on_message(client, None, Msg({"DEP": {"veh": 1}}))
    client.on_message(client, None, Msg({"nonsense": 1}))
    assert (positions.written, events.written) == (1, 1)
