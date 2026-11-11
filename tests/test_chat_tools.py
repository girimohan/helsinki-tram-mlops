import json
from datetime import UTC, date, datetime, timedelta

import pandas as pd
import pytest
from conftest import make_vp

from tram_mlops import chat_tools, gtfsrt, process, storage
from tram_mlops.chat_tools import Network, TramData, call_tool, normalize, normalize_line

NOW = pd.Timestamp("2026-11-11T10:00:00Z")  # 12:00 in Helsinki
T = int(NOW.timestamp())

NETWORK = {
    "routes": {"1004": {"line": "4", "name": "Katajanokka - Munkkiniemi"}, "1009": {"line": "9", "name": "x"}},
    "headsigns": {"1004|0": "Munkkiniemi", "1004|1": "Katajanokka", "1009|0": "Ilmala"},
    "stops": {
        "1020443": ["Lasipalatsi", "H0102", "Mannerheimintie", 60.17, 24.93, "0", "52"],
        "1020444": ["Lasipalatsi", "H0101", "Mannerheimintie", 60.17, 24.93, "0", "51"],
        "1020174": ["Lasipalatsi", "H1903", "Mannerheimintie 9", 60.17, 24.93, "3", "53"],  # bus platform
        "1100": ["Töölön tulli", "H0200", "", 60.19, 24.92, "0", ""],
        "1200": ["Keilaniemi", "E1234", "", 60.17, 24.83, "3", ""],  # typed as a bus stop, used by trams
        "1300": ["Munkkiniemi", "H0300", "", 60.19, 24.88, "0", ""],
        "1400": ["Kauppatori", "H0400", "", 60.16, 24.95, "3", ""],  # buses only
    },
}


def stop(stop_id, at, passed=False):
    return [stop_id, at, at + 20, 0 if passed else None, "SCHEDULED"]


def trip(route, direction, start, stops, status="SCHEDULED", day="20261111"):
    return {"key": f"{route}|{direction}|{day}|{start}", "route_id": route, "direction_id": direction,
            "start_date": day, "start_time": start, "schedule_relationship": status, "stops": stops}


TRIPS = [
    trip("1004", 0, "11:55:00", [stop("1020444", T + 180), stop("1100", T + 400), stop("1300", T + 900)]),
    trip("1004", 1, "11:40:00", [stop("1300", T - 600, passed=True), stop("1020443", T + 60)]),  # ends here
    trip("1009", 0, "12:10:00", [stop("1020443", T + 840), stop("1200", T + 1500)], status="CANCELED"),
    trip("1004", 0, "11:30:00", [stop("1020444", T - 120, passed=True), stop("1300", T + 300)]),
]


def write_alerts(gtfsrt_dir, alerts, fetched_at=NOW):
    storage.append_gzip_jsonl(gtfsrt_dir / "alerts_2026-11-11.jsonl.gz",
                              [{"fetched_at": fetched_at.isoformat(), "count": len(alerts), "alerts": alerts}])


@pytest.fixture
def data(tmp_path):
    gtfsrt_dir = tmp_path / "raw" / "gtfsrt"
    gtfsrt_dir.mkdir(parents=True)
    (gtfsrt_dir / gtfsrt.NETWORK).write_text(json.dumps(NETWORK), encoding="utf-8")
    storage.write_json_atomic(gtfsrt_dir / gtfsrt.CURRENT_TRIPS,
                              {"fetched_at": (NOW - pd.Timedelta(seconds=20)).isoformat(), "trips": TRIPS})
    t = NOW.to_pydatetime() - timedelta(seconds=5)
    positions = [
        make_vp(veh=401, t=t, dl=-120, desi="4", route="1004", dir="1", start="11:55", oday="2026-11-11"),
        make_vp(veh=402, t=t, dl=30, desi="4", route="1004", dir="2", start="11:40", oday="2026-11-11"),
        make_vp(veh=901, t=t, dl=0, desi="9", route="1009", dir="1", start="11:30", oday="2026-11-11"),
    ]
    (tmp_path / "raw" / "tram_2026-11-11.jsonl").write_text(
        "".join(json.dumps(p) + "\n" for p in positions), encoding="utf-8")
    storage.write_json_atomic(tmp_path / "predictions" / "latest.json", {
        "issued_at": (NOW - pd.Timedelta(seconds=10)).isoformat(), "forecasts": [{"vehicle": 401, "predicted": 150.0}]})
    return TramData(gtfsrt_dir=gtfsrt_dir, raw_dir=tmp_path / "raw", processed_dir=tmp_path / "processed",
                    predictions_dir=tmp_path / "predictions", clock=lambda: NOW)


def test_names_are_matched_without_case_or_accents():
    assert normalize("Töölön  Tulli!") == "toolon tulli"
    assert [normalize_line(x) for x in ("Tram 4", "line 4", " 6t ", "#9", "", None)] == ["4", "4", "6T", "9",
                                                                                         None, None]


def test_find_stops_by_name_code_typo_and_only_tram_platforms():
    network = Network(**NETWORK)
    trams = network.tram_stop_ids({"1200"})
    assert "1020174" not in trams and "1400" not in trams and "1200" in trams

    def best(query):
        groups = network.find_stops(query, trams)
        return (groups[0].name, groups[0].ids) if groups else None

    assert best("lasipalatsi") == ("Lasipalatsi", ["1020443", "1020444"])  # not the bus platform
    assert best("H0102") == ("Lasipalatsi", ["1020443", "1020444"])
    assert best("toolon tulli")[0] == "Töölön tulli"
    assert best("Lasipalatsy")[0] == "Lasipalatsi"
    assert best("Keilaniemi")[0] == "Keilaniemi"
    assert best("Kauppatori") is None


def test_next_trams_lists_upcoming_trams_soonest_first(data):
    out = call_tool(data, "next_trams", {"stop": "lasipalatsi"})
    assert out["stop"] == "Lasipalatsi" and out["platforms"] == ["H0101", "H0102"] and out["now"] == "12:00"
    assert "warning" not in out
    arrivals = out["arrivals"]
    # The 11:30 trip has already passed; the cancelled line 9 trip is still listed, marked.
    assert [(a["line"], a["towards"], a["time"], a["in_min"]) for a in arrivals] == [
        ("4", "Katajanokka", "12:01", 1), ("4", "Munkkiniemi", "12:03", 3), ("9", "Ilmala", "12:14", 14)]
    assert arrivals[0]["terminates_here"] and arrivals[0]["platform"] == "H0102"
    assert arrivals[2]["cancelled"]
    # Trams being tracked carry their delay now (HFP direction 1 = GTFS direction 0).
    assert arrivals[0]["tram_now"] == {"vehicle": 402, "delay_min": -0.5}
    assert arrivals[1]["tram_now"] == {"vehicle": 401, "delay_min": 2.0}
    assert "tram_now" not in arrivals[2]

    only_9 = call_tool(data, "next_trams", {"stop": "Lasipalatsi", "line": "Tram 9", "limit": 1})
    assert [a["line"] for a in only_9["arrivals"]] == ["9"] and only_9["line_filter"] == "9"


def test_next_trams_explains_unknown_stops_and_stale_or_missing_data(data):
    unknown = call_tool(data, "next_trams", {"stop": "Lasipalatsit"})
    assert unknown["stop"] == "Lasipalatsi"  # close enough
    missing = call_tool(data, "next_trams", {"stop": "Kauppatori"})
    assert "error" in missing and "similar_names" in missing

    data.clock = lambda: NOW + pd.Timedelta(minutes=10)
    assert "warning" in call_tool(data, "next_trams", {"stop": "Lasipalatsi"})

    (data.gtfsrt_dir / gtfsrt.CURRENT_TRIPS).unlink()
    assert "error" in call_tool(data, "next_trams", {"stop": "Lasipalatsi"})


def test_find_stop_lists_platforms_and_lines(data):
    out = call_tool(data, "find_stop", {"name": "Lasipalatsi"})
    platforms = out["matches"][0]["platforms"]
    assert sorted(p["code"] for p in platforms) == ["H0101", "H0102"]
    assert {line for p in platforms for line in p["lines_now"]} == {"4", "9"}


def test_line_status_overall_and_for_one_line(data):
    overall = call_tool(data, "line_status", {})
    assert overall["trams_in_service"] == 3
    assert [(x["line"], x["trams"]) for x in overall["lines"]] == [("4", 2), ("9", 1)]

    line4 = call_tool(data, "line_status", {"line": "4"})
    assert [t["vehicle"] for t in line4["trams"]] == [401, 402]  # most late first
    first = line4["trams"][0]
    assert first == {"vehicle": 401, "delay_min": 2.0, "towards": "Munkkiniemi", "next_stop": "Lasipalatsi",
                     "forecast_delay_in_5_min": 2.5}
    assert call_tool(data, "line_status", {"line": "7"})["trams_in_service"] == 0


def test_service_alerts_only_for_trams(data):
    in_force = {"start": T - 3600, "end": T + 3600}
    write_alerts(data.gtfsrt_dir, [
        {"id": "a", "header_text": {"translation": [{"text": "Line 4 detour", "language": "en"}]},
         "informed_entity": [{"route_id": "1004"}], "active_period": [in_force], "effect": "DETOUR"},
        {"id": "b", "header_text": {"translation": [{"text": "Bus 20 detour", "language": "en"}]},
         "informed_entity": [{"route_id": "1020"}], "active_period": [in_force]},
        {"id": "c", "header_text": {"translation": [{"text": "Stop closed", "language": "en"}]},
         "informed_entity": [{"stop_id": "1200"}]},
    ])
    out = call_tool(data, "service_alerts", {})
    assert [(a["header"], a["lines"]) for a in out["alerts"]] == [("Line 4 detour", ["4"]), ("Stop closed", ["9"])]
    assert out["alerts"][0]["effect"] == "Detour" and out["alerts"][0]["until"] == "Wed 11 Nov 13:00"
    assert [a["header"] for a in call_tool(data, "service_alerts", {"line": "9"})["alerts"]] == ["Stop closed"]


def test_punctuality_from_processed_history(data):
    assert "error" in call_tool(data, "punctuality", {})
    for day in (date(2026, 11, 9), date(2026, 11, 10)):
        t0 = datetime(day.year, day.month, day.day, 6, 0, tzinfo=UTC)
        raw = [make_vp(veh=v, t=t0 + timedelta(seconds=30 * i), dl=-60 * v, desi=str(v), oday=day.isoformat())
               for v in (1, 4) for i in range(100)]
        storage.write_partition(data.processed_dir, day, process.clean(pd.DataFrame(raw)))
    by_line = call_tool(data, "punctuality", {"days": 7})
    # Line 4 runs 4 min late (not on time), line 1 runs 1 min late (on time).
    assert [(x["line"], x["on_time_pct"], x["mean_delay_min"]) for x in by_line["by_line"]] == [
        ("4", 0.0, 4.0), ("1", 100.0, 1.0)]
    by_day = call_tool(data, "punctuality", {"line": "1", "group_by": "day"})
    assert [d["day"] for d in by_day["by_day"]] == ["Mon 2026-11-09", "Tue 2026-11-10"]
    assert call_tool(data, "punctuality", {"line": "4", "group_by": "hour"})["by_hour"][0]["hour"] == 8


def test_call_tool_reports_problems_instead_of_raising(data, monkeypatch):
    assert "error" in call_tool(data, "drop_tables", {})
    assert "error" in call_tool(data, "next_trams", {})
    # Unknown arguments are dropped rather than failing the call.
    assert call_tool(data, "next_trams", {"stop": "Lasipalatsi", "radius": 5})["stop"] == "Lasipalatsi"
    assert "error" in call_tool(data, "next_trams", {"stop": "Lasipalatsi", "limit": "many"})

    def broken(**kwargs):
        raise OSError("disk")
    monkeypatch.setattr(data, "service_alerts", broken)
    assert call_tool(data, "service_alerts", {}) == {"error": "service_alerts failed: OSError"}


def test_tool_schemas_match_the_methods():
    for tool in chat_tools.TOOLS:
        spec = tool["function"]
        assert callable(getattr(TramData, spec["name"]))
        assert set(spec["parameters"].get("required", [])) <= set(spec["parameters"]["properties"])
