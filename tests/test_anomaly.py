import json
from datetime import UTC, date, datetime, timedelta

import pandas as pd
import pytest
from conftest import make_vp

from tram_mlops import anomaly, gtfsrt, storage
from tram_mlops.features import add_features
from tram_mlops.network import Network
from tram_mlops.process import clean

T0 = datetime(2026, 11, 10, 7, 0, tzinfo=UTC)  # 09:00 in Helsinki; trips started at 08:00
S = anomaly.Settings(baseline_min_rows=1)
TERMINUS = (60.20, 24.90)
NETWORK = Network(
    routes={"1003": {"line": "3", "name": ""}, "1006": {"line": "6", "name": ""}},
    stops={"10": ["Eiran sairaala", "H0010", "", 60.1581, 24.9414, "0", ""],
           "20": ["Kallio", "H0020", "", 60.1840, 24.9500, "0", ""]})


def track(veh, line, route, minutes, *, speed=5.0, still=(), start_delay=0, lat0=60.150, lon0=24.940,
          step_lat=0.0003, delay_per_still_step=10, jumps=None, crawl=None):
    """One record per tram every 10 s. During ``still`` (minutes from T0) it stands still and loses
    ``delay_per_still_step`` s per record; ``crawl`` minutes it moves at 1 m/s and loses 8 s per record;
    ``jumps`` adds a sudden step to the delay value at a minute."""
    records, lat, delay = [], lat0, start_delay
    for i in range(minutes * 6):
        minute = i / 6
        stopped = still and still[0] <= minute < still[1]
        crawling = crawl and crawl[0] <= minute < crawl[1]
        if stopped:
            delay += delay_per_still_step
        elif crawling:
            delay += 8
            lat += step_lat / 5
        else:
            lat += step_lat
        if jumps and abs(minute - jumps[0]) < 1e-9:
            delay += jumps[1]
        records.append(make_vp(veh=veh, t=T0 + timedelta(seconds=10 * i), dl=-delay, desi=line, route=route,
                               dir="1", start="08:00", oday="2026-11-10", spd=0.0 if stopped else
                               (1.0 if crawling else speed), lat=lat, long=lon0))
    return records


def scene():
    raw = (
        # Two trams stuck within ~100 m of each other mid-route (lines 3 and 6).
        track(1, "3", "1003", 30, still=(10, 18), lat0=60.155, step_lat=0.0)
        + track(2, "6", "1006", 30, still=(11, 17), lat0=60.1555, step_lat=0.0)
        # A tram waiting at its terminus: not an incident.
        + track(4, "4", "1004", 30, still=(5, 20), lat0=TERMINUS[0], lon0=TERMINUS[1], step_lat=0.0)
        # Moving normally, far away.
        + track(3, "9", "1009", 30, lat0=60.19)
        # The delay value jumps by 25 min at once: a data glitch, not a tram losing time.
        + track(5, "7", "1007", 30, lat0=60.17, jumps=(15, 1500))
        # Crawling at 1 m/s for 12 min, losing 8 s every 10 s: a delay jump without stopping.
        + track(6, "1", "1001", 30, lat0=60.21, lon0=25.0, crawl=(8, 20))
    )
    return add_features(clean(pd.DataFrame(raw)))


@pytest.fixture
def feats():
    return scene()


def baseline():
    return anomaly.Baseline(termini={"1004|1": [list(TERMINUS)]})


def test_threshold_uses_line_hour_then_line_then_default_with_a_floor():
    b = anomaly.Baseline(change={"3|8": [10, 50, 500], "3": [0, 20, 900]}, default=[10, 47, 0])
    assert b.threshold("3", 8, S) == pytest.approx(10 + 6 * 1.4826 * 50)
    assert b.threshold("3", 9, S) == S.jump_min_s  # 0 + 6 x 1.4826 x 20 = 178 s is below the floor
    assert b.threshold("9", 8, S) == pytest.approx(10 + 6 * 1.4826 * 47)


def test_stopped_trams_are_flagged_except_at_a_terminus(feats):
    runs = anomaly.still_runs(feats, S)
    events = anomaly.stopped_events(runs, baseline(), S, ongoing_only=False)
    assert sorted(e["vehicle"] for e in events) == [1, 2]
    first = next(e for e in events if e["vehicle"] == 1)
    assert first["delay_gain_s"] >= 470 and first["kinds"] == ["stopped"]
    # Without the terminus in the baseline, the waiting tram would be flagged too.
    assert 4 in {e["vehicle"] for e in anomaly.stopped_events(runs, anomaly.Baseline(), S, ongoing_only=False)}


def test_delay_jumps_skip_impossible_glitches(feats):
    events = anomaly.jump_events(feats, baseline(), S, latest_only=False)
    vehicles = {e["vehicle"] for e in events}
    assert 6 in vehicles  # crawling: lost ~8 min in 10 without ever stopping
    assert 5 not in vehicles  # a 25-min step in 10 s cannot be real
    assert anomaly.jump_rows(feats, baseline(), S).attrs["glitches"] > 0


def test_nearby_anomalies_form_one_incident_named_after_the_nearest_stop(feats):
    events, found = anomaly.detect_day(feats, baseline(), S, NETWORK)
    assert {e["vehicle"] for e in events} == {1, 2, 6}
    disruption = found[0]
    assert disruption["trams"] == 2 and disruption["lines"] == ["3", "6"]
    assert disruption["near"] == "Eiran sairaala"
    assert [i["trams"] for i in found] == [2, 1]


def test_detector_tracks_live_events_and_logs_them_when_they_end(feats, tmp_path):
    (tmp_path / gtfsrt.NETWORK).write_text(json.dumps({"routes": NETWORK.routes, "stops": NETWORK.stops}))
    baseline().save(tmp_path / anomaly.BASELINE)
    detector = anomaly.Detector(tmp_path, tmp_path / gtfsrt.NETWORK, S)

    def at(minute):
        now = pd.Timestamp(T0 + timedelta(minutes=minute))
        return detector.step(feats[feats["timestamp"] <= now], now)

    # Trams 1 and 2 stop at 10 and 11 and are flagged after 5 minutes still.
    assert [at(m)["active"] for m in range(9, 17)] == [0, 0, 0, 0, 0, 0, 1, 2]
    latest = anomaly.read_latest(tmp_path)
    assert latest["incidents"][0]["trams"] == 2 and latest["incidents"][0]["near"] == "Eiran sairaala"
    assert {e["near"] for e in latest["active"]} == {"Eiran sairaala"}
    # Once their condition stops holding and close_after_s (2 min) has passed, the events are logged:
    # tram 2 first, tram 1 later (its 10-minute delay gain stays high for a while after it moves on),
    # and tram 6, whose crawl is flagged as a delay jump from minute 17.
    assert sum(at(m)["closed"] for m in range(17, 24)) == 3
    logged = anomaly.read_events(tmp_path, T0.date(), T0.date())
    assert set(logged["vehicle"]) == {1, 2, 6}
    assert logged.loc[logged["vehicle"] == 1, "start"].iloc[0] == pd.Timestamp(T0 + timedelta(minutes=10))


def test_baseline_from_history_round_trips(feats, tmp_path):
    sample, points = anomaly.baseline_sample(feats, S, every=1)
    b = anomaly.combine_baseline([sample], [points, points], S, ["2026-11-10"])
    assert "3" in b.change and "3|9" in b.change
    assert b.at_terminus("1004", "1", *TERMINUS, radius_m=50)
    b.save(tmp_path / "b.json")
    assert anomaly.Baseline.load(tmp_path / "b.json") == b


def write_alert(gtfsrt_dir, seen, start, route="1003", effect="REDUCED_SERVICE"):
    alert = {"id": f"alert-{route}", "effect": effect, "informed_entity": [{"route_id": route}],
             "active_period": [{"start": str(int(start.timestamp())), "end": str(int(start.timestamp()) + 3600)}],
             "header_text": {"translation": [{"text": "Line 3 disruption", "language": "en"}]}}
    storage.append_gzip_jsonl(gtfsrt_dir / f"alerts_{seen:%Y-%m-%d}.jsonl.gz",
                              [{"fetched_at": seen.isoformat(), "count": 1, "alerts": [alert]}])


def test_reactive_alerts_are_weak_labels_and_scored(tmp_path, feats):
    start = pd.Timestamp(T0 + timedelta(minutes=12))
    write_alert(tmp_path, start + timedelta(minutes=3), start)
    write_alert(tmp_path, start - timedelta(days=2), start, route="1006")  # planned: announced days before
    labels = anomaly.reactive_alerts(tmp_path, NETWORK)
    assert [lbl["lines"] for lbl in labels] == [["3"]]
    _, found = anomaly.detect_day(feats, baseline(), S, NETWORK)
    scored = anomaly.evaluate(found, labels)
    assert scored["caught"] == 1
    assert scored["details"][0]["lead_min"] == pytest.approx(5.0)  # detected at 10:00, HSL said so at 15:00


def test_replay_judges_each_day_against_the_others(tmp_path):
    processed, gtfsrt_dir = tmp_path / "processed", tmp_path / "gtfsrt"
    gtfsrt_dir.mkdir()
    (gtfsrt_dir / gtfsrt.NETWORK).write_text(json.dumps({"routes": NETWORK.routes, "stops": NETWORK.stops}))
    # The tram moves, then stands still 10-18 min at Eiran sairaala (60 steps of 0.0003 from lat0).
    df = clean(pd.DataFrame([r for rows in (track(1, "3", "1003", 30, still=(10, 18), lat0=60.1401),
                                            track(3, "9", "1009", 30, lat0=60.19)) for r in rows]))
    storage.write_partition(processed, date(2026, 11, 10), df)
    later = df.assign(timestamp=df["timestamp"] + pd.Timedelta(days=1),
                      local_time=df["local_time"] + pd.Timedelta(days=1))
    storage.write_partition(processed, date(2026, 11, 11), later)
    report = anomaly.replay(processed, gtfsrt_dir, S, tmp_path / "reports")
    assert [d["stopped"] for d in report["days"]] == [1, 1]
    assert report["top_incidents"][0]["near"] == "Eiran sairaala"
    assert "Eiran sairaala" in (tmp_path / "reports" / "latest.md").read_text(encoding="utf-8")
