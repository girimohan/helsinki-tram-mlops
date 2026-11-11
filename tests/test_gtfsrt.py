import gzip
import json
import urllib.error
import zipfile

import pytest
from google.transit import gtfs_realtime_pb2 as rt

from tram_mlops import gtfsrt, storage

TRAMS = {"1009", "2015"}


def feed(trips=(), alerts=(), ts=1794300000):
    message = rt.FeedMessage()
    message.header.gtfs_realtime_version = "2.0"
    message.header.timestamp = ts
    for i, (route, direction, start, stops) in enumerate(trips):
        update = message.entity.add(id=f"t{i}").trip_update
        update.trip.route_id, update.trip.direction_id = route, direction
        update.trip.start_date, update.trip.start_time = "20261111", start
        for stop_id, arrival, uncertainty in stops:
            stu = update.stop_time_update.add(stop_id=stop_id)
            stu.arrival.time = arrival
            if uncertainty is not None:
                stu.arrival.uncertainty = uncertainty
    for i, text in enumerate(alerts):
        alert = message.entity.add(id=f"a{i}").alert
        alert.header_text.translation.add(text=text, language="en")
        alert.informed_entity.add(route_id="1009")
    return message


def test_tram_routes_include_light_rail(tmp_path):
    path = tmp_path / "gtfs.zip"
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("routes.txt", "﻿route_id,route_type\n1009,0\n2015,900\n1020,3\n")
    assert gtfsrt.tram_routes_from_gtfs(path) == TRAMS


@pytest.mark.parametrize("route,expected", [
    ("2015", True), ("1013E", True), ("1009TX", True),   # light rail and lettered variants
    ("1020", False), ("10133", False),                   # buses
])
def test_is_tram(route, expected):
    assert gtfsrt.is_tram(route, {"1009", "1013", "2015"}) is expected


def test_trip_updates_archive_only_changed_tram_trips():
    archive = gtfsrt.TripUpdateArchive()
    tram = ("1009", 1, "25:10:00", [("1", 100, 0), ("2", 200, None)])
    bus = ("1020", 0, "10:00:00", [("9", 100, None)])
    first = archive.changes(feed([tram, bus]), TRAMS, "t1")
    assert [line["key"] for line in first] == ["1009|1|20261111|25:10:00"]
    assert first[0]["stops"] == [["1", 100, None, 0, "SCHEDULED"], ["2", 200, None, None, "SCHEDULED"]]
    assert first[0]["fetched_at"] == "t1" and first[0]["header_ts"] == 1794300000

    assert archive.changes(feed([tram, bus]), TRAMS, "t2") == []  # unchanged
    later = ("1009", 1, "25:10:00", [("1", 100, 0), ("2", 260, None)])
    assert [line["stops"][1][1] for line in archive.changes(feed([later]), TRAMS, "t3")] == [260]
    assert archive.changes(feed([]), TRAMS, "t4") == [
        {"fetched_at": "t4", "header_ts": 1794300000, "key": "1009|1|20261111|25:10:00", "removed": True}]


def test_alerts_are_kept_only_when_their_content_changes():
    archive = gtfsrt.AlertArchive()
    first = archive.changes(feed(alerts=["Track works"]), "t1")
    assert first[0]["count"] == 1 and first[0]["alerts"][0]["informed_entity"] == [{"route_id": "1009"}]
    # HSL re-publishes the same alerts with a new header timestamp: not a change.
    assert archive.changes(feed(alerts=["Track works"], ts=1794300300), "t2") == []
    assert archive.changes(feed(alerts=["Track works", "Detour"]), "t3")[0]["count"] == 2


def test_gzip_members_append_and_read_back(tmp_path):
    path = tmp_path / "trip_updates_2026-11-11.jsonl.gz"
    storage.append_gzip_jsonl(path, [{"a": 1}])
    storage.append_gzip_jsonl(path, [])
    storage.append_gzip_jsonl(path, [{"a": 2}, {"a": 3}])
    with gzip.open(path, "rt") as f:
        assert [json.loads(line)["a"] for line in f] == [1, 2, 3]
    # A crash mid-write leaves a truncated member; the reader keeps everything before it.
    with open(path, "ab") as f:
        f.write(gzip.compress(b'{"a": 4}\n')[:15])
    assert [r["a"] for r in storage.read_gzip_jsonl(path)] == [1, 2, 3]


def test_fetch_treats_304_as_unchanged(monkeypatch):
    def not_modified(request, timeout):
        assert request.get_header("If-none-match") == '"abc"'
        raise urllib.error.HTTPError(request.full_url, 304, "Not Modified", {}, None)
    monkeypatch.setattr(gtfsrt.urllib.request, "urlopen", not_modified)
    result = gtfsrt.fetch("https://example.invalid/alerts", etag='"abc"')
    assert result.status == 304 and result.body == b"" and result.etag == '"abc"'


def test_poller_keeps_archiving_alerts_when_static_gtfs_fails(tmp_path, monkeypatch):
    cfg = {"trip_updates_url": "tu", "alerts_url": "al", "static_gtfs_url": "zip", "static_refresh_h": 24,
           "tram_route_types": ["0", "900"], "trip_updates_interval_s": 30, "alerts_interval_s": 60}
    bodies = {"tu": feed([("1009", 0, "10:00:00", [("1", 100, None)])]).SerializeToString(),
              "al": feed(alerts=["x"]).SerializeToString()}
    monkeypatch.setattr(gtfsrt, "fetch", lambda url, *a, **k: gtfsrt.Fetched(200, bodies[url], '"e"'))
    downloads = []

    def failing_download(url, types):
        downloads.append(url)
        raise OSError("static GTFS unavailable")
    monkeypatch.setattr(gtfsrt, "download_tram_routes", failing_download)

    poller = gtfsrt.Poller(cfg, tmp_path)
    poller.run(duration_s=0.01)
    # Trip updates wait for the tram route list; alerts are archived regardless.
    assert [p.name.split("_")[0] for p in tmp_path.iterdir()] == ["alerts"]
    # The ~90 MB download isn't retried until the backoff has passed.
    poller.refresh_routes(poller.routes_retry_at - 1)
    assert len(downloads) == 1

    monkeypatch.setattr(gtfsrt, "download_tram_routes", lambda url, types: set(TRAMS))
    poller.refresh_routes(poller.routes_retry_at + 1)
    poller.poll_trip_updates()
    assert sorted(p.name.split("_")[0] for p in tmp_path.iterdir()) == ["alerts", "trip"]
