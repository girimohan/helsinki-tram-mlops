import gzip
import json
from datetime import UTC, date, datetime, timedelta

import pandas as pd
from conftest import make_vp

from tram_mlops import feeds

NOW = datetime(2026, 11, 11, 10, 0, tzinfo=UTC)


def append_members(path, *batches):
    """Write each batch as its own gzip member, as the poller does."""
    with open(path, "ab") as f:
        for batch in batches:
            f.write(gzip.compress("".join(json.dumps(r) + "\n" for r in batch).encode()))


def trip(route, start, status, fetched_at, day="20261111"):
    return {"fetched_at": fetched_at, "key": f"{route}|0|{day}|{start}", "route_id": route, "start_date": day,
            "start_time": start, "schedule_relationship": status, "stops": [["1", 1, 2, 0, "SCHEDULED"]]}


def test_service_day_runs_past_midnight():
    assert feeds.service_day(pd.Timestamp("2026-11-11T10:00Z")) == date(2026, 11, 11)
    # 01:30 Helsinki time on the 12th still belongs to the 11th's service.
    assert feeds.service_day(pd.Timestamp("2026-11-11T23:30Z")) == date(2026, 11, 11)
    assert feeds.service_day(pd.Timestamp("2026-11-12T03:00Z")) == date(2026, 11, 12)


def test_trips_by_route_uses_each_trips_latest_status(tmp_path):
    assert feeds.trips_by_route(tmp_path, date(2026, 11, 11)).empty
    append_members(
        tmp_path / "trip_updates_2026-11-11.jsonl.gz",
        [trip("1009", "10:00:00", "SCHEDULED", "2026-11-11T08:00:00+00:00"),
         trip("1009", "10:10:00", "CANCELED", "2026-11-11T08:00:00+00:00"),
         trip("2015", "10:00:00", "CANCELED", "2026-11-11T08:00:00+00:00")],
        # Later polls: the line 15 trip is reinstated, another line 9 trip is cancelled, one trip ends.
        [trip("2015", "10:00:00", "SCHEDULED", "2026-11-11T08:00:30+00:00"),
         trip("1009", "10:20:00", "CANCELED", "2026-11-11T08:00:30+00:00"),
         {"fetched_at": "2026-11-11T08:01:00+00:00", "key": "1009|0|20261111|10:00:00", "removed": True}],
    )
    # Trips of the next service day are fetched on the same UTC day; they're not counted.
    append_members(tmp_path / "trip_updates_2026-11-12.jsonl.gz",
                   [trip("1009", "05:00:00", "CANCELED", "2026-11-12T03:00:00+00:00", day="20261112")])
    got = feeds.trips_by_route(tmp_path, date(2026, 11, 11)).set_index("route_id")
    assert got.to_dict("index") == {"1009": {"trips": 3, "cancelled": 2}, "2015": {"trips": 1, "cancelled": 0}}


def write_events(raw_dir, day, events):
    raw_dir.mkdir(exist_ok=True)
    (raw_dir / f"events_{day}.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events) + '{"partial":')


def ars(veh, minutes_ago, delay_s, stop, line="9", event="ARS"):
    t = NOW - timedelta(minutes=minutes_ago)
    return make_vp(veh=veh, t=t, desi=line, event=event, stop=stop, lat=60.1 + stop / 1e7, long=24.9,
                   ttarr=(t - timedelta(seconds=delay_s)).strftime("%Y-%m-%dT%H:%M:%S.000Z"))


def test_stop_arrivals_aggregates_ars_events(tmp_path):
    assert feeds.stop_arrivals(tmp_path, pd.Timestamp(NOW)).empty
    write_events(tmp_path, "2026-11-10", [ars(1, 60 * 20, 100, 11)])  # yesterday, too old for the window
    write_events(tmp_path, "2026-11-11", [
        ars(1, 5, 0, 11), ars(2, 10, 120, 11), ars(3, 15, 600, 11, line="15"),
        ars(1, 5, 30, 22), ars(2, 6, 3600, 22),  # a stale journey, ignored
        ars(1, 5, 999, 33, event="ARR"),  # not an arrival at the stop
    ])
    got = feeds.stop_arrivals(tmp_path, pd.Timestamp(NOW) - pd.Timedelta(hours=1)).set_index("stop")
    assert list(got.index) == ["11", "22"]
    assert got.loc["11", "arrivals"] == 3 and got.loc["11", "mean_delay_s"] == 240
    assert got.loc["11", "median_delay_s"] == 120 and round(got.loc["11", "on_time_pct"]) == 67
    assert got.loc["11", "lines"] == "15, 9"
    assert got.loc["22", "arrivals"] == 1

    only_15 = feeds.stop_arrivals(tmp_path, pd.Timestamp(NOW) - pd.Timedelta(hours=1), lines=["15"])
    assert only_15["stop"].tolist() == ["11"] and only_15["arrivals"].tolist() == [1]
    wider = feeds.stop_arrivals(tmp_path, pd.Timestamp(NOW) - pd.Timedelta(days=1)).set_index("stop")
    assert wider.loc["11", "arrivals"] == 4


def test_event_counts(tmp_path):
    assert feeds.event_counts(tmp_path / "events_2026-11-11.jsonl").empty
    write_events(tmp_path, "2026-11-11", [ars(1, 1, 0, 1), ars(1, 1, 0, 1, event="TLR"), ars(2, 1, 0, 1)])
    counts = feeds.event_counts(tmp_path / "events_2026-11-11.jsonl")
    assert counts.set_index("event")["count"].to_dict() == {"ARS": 2, "TLR": 1}
