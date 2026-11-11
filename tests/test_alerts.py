import gzip
import json

import pandas as pd
import pytest

from tram_mlops import alerts

NOW = pd.Timestamp("2026-11-11T10:00:00Z")
HOUR = 3600


def raw_alert(id_, entities, start=-HOUR, end=HOUR, severity="INFO", effect="OTHER_EFFECT", **extra):
    epoch = int(NOW.timestamp())
    period = {}
    if start is not None:
        period["start"] = str(epoch + start)
    if end is not None:
        period["end"] = str(epoch + end)
    return {
        "id": id_, "active_period": [period], "informed_entity": entities, "cause": "CONSTRUCTION",
        "effect": effect, "severity_level": severity,
        "header_text": {"translation": [{"text": f"Otsikko {id_}", "language": "fi"},
                                        {"text": f"Header {id_}", "language": "en"}]},
        **extra,
    }


def write_snapshots(path, *snapshots, truncate_last=False):
    data = b"".join(gzip.compress((json.dumps(s) + "\n").encode()) for s in snapshots)
    if truncate_last:
        data += gzip.compress(b'{"fetched_at": "later", "alerts": []}\n')[:-12]  # a member still being written
    path.write_bytes(data)


def test_latest_snapshot_reads_last_complete_member_and_falls_back(tmp_path):
    assert alerts.latest_snapshot(tmp_path) is None
    write_snapshots(tmp_path / "alerts_2026-11-10.jsonl.gz", {"fetched_at": "a", "alerts": []},
                    {"fetched_at": "b", "alerts": [raw_alert("1", [])]})
    assert alerts.latest_snapshot(tmp_path)["fetched_at"] == "b"

    write_snapshots(tmp_path / "alerts_2026-11-11.jsonl.gz", {"fetched_at": "c", "alerts": []}, truncate_last=True)
    assert alerts.latest_snapshot(tmp_path)["fetched_at"] == "c"

    # Today's file has no complete snapshot yet: the newest earlier one still holds.
    (tmp_path / "alerts_2026-11-11.jsonl.gz").write_bytes(b"\x1f\x8b\x08")
    assert alerts.latest_snapshot(tmp_path)["fetched_at"] == "b"


def test_parse_prefers_english_and_reads_periods():
    [alert] = alerts.parse({"alerts": [raw_alert("7", [{"route_id": "1007"}, {"stop_id": "1"}],
                                                 url={"translation": [{"text": "https://x", "language": "fi"}]})]})
    assert alert.header == "Header 7" and alert.url == "https://x" and alert.description == ""
    assert alert.routes == ("1007",) and alert.stops == ("1",)
    assert alert.periods == ((NOW - pd.Timedelta(hours=1), NOW + pd.Timedelta(hours=1)),)
    assert alerts.parse(None) == []


@pytest.mark.parametrize(("start", "end", "active"), [
    (-HOUR, HOUR, True), (HOUR, 2 * HOUR, False), (-2 * HOUR, -HOUR, False),
    (None, HOUR, True), (-HOUR, None, True),
])
def test_active_periods(start, end, active):
    [alert] = alerts.parse({"alerts": [raw_alert("1", [], start=start, end=end)]})
    assert alert.active(NOW) is active
    assert alerts.active([alert], NOW) == ([alert] if active else [])


def test_alert_without_periods_is_always_active():
    [alert] = alerts.parse({"alerts": [{"id": "1", "informed_entity": []}]})
    assert alert.active(NOW) and alert.ends(NOW) is None
    assert alert.effect == "UNKNOWN_EFFECT" and alert.header == ""


def test_for_trams_matches_routes_variants_and_stop_only_alerts():
    route_lines = {"1009": "9", "2015": "15", "1001": "1"}
    stop_lines = {"1140453": ["1", "8"], "1100159": ["3"]}
    feed = alerts.parse({"alerts": [
        raw_alert("bus", [{"route_id": "1059"}, {"stop_id": "1100159"}]),  # bus alert at a shared stop
        raw_alert("variant", [{"route_id": "1009H"}, {"route_id": "1009"}]),
        raw_alert("lr", [{"route_id": "2015"}], severity="WARNING", effect="DETOUR"),
        raw_alert("stops", [{"stop_id": "1140453"}, {"stop_id": "999"}]),
        raw_alert("metro", [{"stop_id": "999"}]),
    ]})
    trams = alerts.for_trams(feed, route_lines, stop_lines)
    assert [(a.id, a.lines) for a in trams] == [
        ("lr", ("15",)), ("stops", ("1", "8")), ("variant", ("9",))]  # disruptive first
    assert trams[0].disruptive and not trams[1].disruptive
    assert alerts.per_line(trams) == {"15": 1, "1": 1, "8": 1, "9": 1}


def test_helpers():
    assert alerts.line_for_route("1013E", {"1013": "13"}) == "13"
    assert alerts.line_for_route("1059", {"1013": "13"}) is None
    assert alerts.label("SIGNIFICANT_DELAYS") == "Significant delays"
    assert sorted(["10", "1T", "2", "1", "H"], key=alerts.line_sort_key) == ["1", "1T", "2", "10", "H"]
    [alert] = alerts.parse({"alerts": [raw_alert("1", [], start=-HOUR, end=HOUR)]})
    assert alert.ends(NOW) == NOW + pd.Timedelta(hours=1)
