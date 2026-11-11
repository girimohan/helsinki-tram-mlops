import gzip
import json
from datetime import UTC, date, datetime, timedelta

import pandas as pd
import pytest
from conftest import make_vp
from streamlit.testing.v1 import AppTest
from streamlit.util import calc_md5

from tram_mlops import feeds, storage
from tram_mlops.config import PROJECT_ROOT, load_config
from tram_mlops.process import clean
from tram_mlops.train import save, train

DASHBOARD = str(PROJECT_ROOT / "src" / "tram_mlops" / "dashboard.py")


@pytest.fixture
def isolated_paths(tmp_path, monkeypatch):
    monkeypatch.setenv("TRAM_PATHS_RAW_DIR", str(tmp_path / "raw"))
    monkeypatch.setenv("TRAM_PATHS_GTFSRT_DIR", str(tmp_path / "raw" / "gtfsrt"))
    monkeypatch.setenv("TRAM_PATHS_PROCESSED_DIR", str(tmp_path / "processed"))
    monkeypatch.setenv("TRAM_PATHS_MODELS_DIR", str(tmp_path / "models"))
    load_config.cache_clear()
    yield tmp_path
    load_config.cache_clear()


def open_page(app: AppTest, url_path: str) -> AppTest:
    """Select a page by its URL path, as a browser visiting /<url_path> would.

    AppTest.switch_page only knows file-based pages; ``st.Page`` callables are keyed by
    the md5 of their URL path.
    """
    app._page_hash = calc_md5(url_path)
    # AppTest reads a single-select segmented control's value back as a string but re-sends it as
    # a list of options, so pass the current selection as a list before rerunning.
    for group in app.button_group:
        if isinstance(group.value, str):
            group.set_value([group.value])
    return app.run()


def show_section(app: AppTest, section: str) -> AppTest:
    """Open a History section. (After ``st.switch_page`` AppTest stays on the original URL, so name the page.)"""
    app.button_group(key="history_section").set_value([section])
    return open_page(app, "history")


def write_live_raw(raw_dir, now):
    """Two minutes of positions for trams 1-5 on lines 1-5, running 50-250 s late."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    records = [make_vp(veh=v, t=now - timedelta(seconds=s), dl=-50 * v, desi=str(v), route=f"100{v}")
               for s in range(110, -1, -10) for v in range(1, 6)]
    (raw_dir / f"tram_{now:%Y-%m-%d}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))


def write_gtfsrt(gtfsrt_dir, now):
    """An alert snapshot with a detour on line 3 and a bus alert, and a cancelled trip on line 4."""
    gtfsrt_dir.mkdir(parents=True, exist_ok=True)
    start, end = str(int((now - timedelta(hours=1)).timestamp())), str(int((now + timedelta(hours=1)).timestamp()))
    snapshot = {"fetched_at": now.isoformat(timespec="seconds"), "count": 2, "alerts": [
        {"id": "a1", "effect": "DETOUR", "severity_level": "WARNING", "cause": "CONSTRUCTION",
         "active_period": [{"start": start, "end": end}], "informed_entity": [{"route_id": "1003"}],
         "header_text": {"translation": [{"text": "Tram 3 detour via Kamppi", "language": "en"}]}},
        {"id": "a2", "effect": "DETOUR", "severity_level": "WARNING",
         "active_period": [{"start": start, "end": end}], "informed_entity": [{"route_id": "1059"}],
         "header_text": {"translation": [{"text": "Bus 59 detour", "language": "en"}]}},
    ]}
    with gzip.open(gtfsrt_dir / f"alerts_{now:%Y-%m-%d}.jsonl.gz", "wt", encoding="utf-8") as f:
        f.write(json.dumps(snapshot) + "\n")
    day = feeds.service_day(pd.Timestamp(now)).strftime("%Y%m%d")
    trips = [{"fetched_at": now.isoformat(timespec="seconds"), "key": f"1004|0|{day}|10:00:00", "route_id": "1004",
              "start_date": day, "start_time": "10:00:00", "schedule_relationship": "CANCELED", "stops": []},
             {"fetched_at": now.isoformat(timespec="seconds"), "key": f"1001|0|{day}|10:00:00", "route_id": "1001",
              "start_date": day, "start_time": "10:00:00", "schedule_relationship": "SCHEDULED", "stops": []}]
    with gzip.open(gtfsrt_dir / f"trip_updates_{now:%Y-%m-%d}.jsonl.gz", "wt", encoding="utf-8") as f:
        f.write("".join(json.dumps(t) + "\n" for t in trips))


def test_dashboard_opens_history_from_sample_without_model(isolated_paths):
    app = AppTest.from_file(DASHBOARD, default_timeout=60).run()
    assert not app.exception
    assert storage.list_partitions(isolated_paths / "processed") == [date(2026, 11, 10)]
    # No collector running: the first page is History, with its day picker.
    assert len(app.sidebar.date_input) == 1
    sections = app.button_group(key="history_section")
    assert sections.value == "Overview"
    assert "How late trams run" in {s.value for s in app.subheader}

    expected = {"Lines": "Reliability by line", "Time of day": "On time (%)", "Trams": "Most delayed trams"}
    for section, heading in expected.items():
        show_section(app, section)
        assert not app.exception
        assert heading in {s.value for s in app.subheader}, section
    assert any("No trained model" in i.value for i in app.info)  # Trams has the forecast panel
    show_section(app, "Stops")
    assert any("No stop events" in i.value for i in app.info)


def test_dashboard_never_processes_when_on_demand_is_off(isolated_paths, monkeypatch):
    monkeypatch.setenv("TRAM_DASHBOARD_PROCESS_ON_DEMAND", "false")
    load_config.cache_clear()
    app = AppTest.from_file(DASHBOARD, default_timeout=60).run()
    assert not app.exception
    assert not (isolated_paths / "processed").exists()
    assert any("No processed data" in w.value for w in app.warning)


def test_dashboard_shows_forecasts_with_model(isolated_paths, synthetic_raw):
    df = clean(pd.DataFrame(synthetic_raw))
    storage.write_partition(isolated_paths / "processed", df["timestamp"].dt.date.iloc[0], df)
    model, metrics = train(df, horizon_s=300, tolerance_s=30, min_rows=100)
    save(model, metrics, isolated_paths / "models")

    app = AppTest.from_file(DASHBOARD, default_timeout=60).run()
    assert not app.exception
    show_section(app, "Trams")
    labels = [m.label for m in app.metric]
    assert "Persistence baseline MAE" in labels

    app = open_page(app, "system")
    assert "Training rows" in [m.label for m in app.metric]


def test_dashboard_live_mode_reads_fresh_raw_data(isolated_paths):
    now = datetime.now(UTC)
    write_live_raw(isolated_paths / "raw", now)

    app = AppTest.from_file(DASHBOARD, default_timeout=60).run()
    assert not app.exception
    metrics = {m.label: m.value for m in app.metric}
    assert metrics["Trams in service"] == "5"
    assert metrics["More than 3 min late"] == "2"  # trams 4 and 5
    assert not app.sidebar.date_input  # the Live page has no day picker


def test_live_view_shows_tram_alerts_and_cancellations(isolated_paths):
    now = datetime.now(UTC)
    write_live_raw(isolated_paths / "raw", now)
    write_gtfsrt(isolated_paths / "raw" / "gtfsrt", now)

    app = AppTest.from_file(DASHBOARD, default_timeout=60).run()
    assert not app.exception
    # The tram detour is shown; the bus alert is not.
    assert [w.value for w in app.warning] == ["**Line 3** · Tram 3 detour via Kamppi"]
    assert [e.label for e in app.expander if "alert" in e.label] == ["1 HSL service alert for trams in force"]
    board = app.dataframe[0].value.set_index("line")
    assert board.loc["3", "disruptions"] == "⚠️ 1"
    assert board.loc["4", "disruptions"] == "1 cancelled"
    assert board.loc["1", "disruptions"] == ""


def test_live_page_without_data_explains_itself(isolated_paths):
    app = AppTest.from_file(DASHBOARD, default_timeout=60).run()  # lands on History
    open_page(app, "live")
    assert not app.exception
    assert any("No tram data" in w.value for w in app.warning)


def test_system_page_reports_sources_and_processing(isolated_paths):
    now = datetime.now(UTC)
    write_live_raw(isolated_paths / "raw", now)
    write_gtfsrt(isolated_paths / "raw" / "gtfsrt", now)
    df = clean(pd.DataFrame([make_vp(veh=1, t=now - timedelta(minutes=5), dl=-30)]))
    storage.write_partition(isolated_paths / "processed", now.date(), df)
    storage.save_manifest(isolated_paths / "processed", {"partitions": {now.date().isoformat(): {
        "source": f"tram_{now:%Y-%m-%d}.jsonl", "raw_rows": 3, "rows": 1, "processing_version": 2,
        "processed_at": now.isoformat(timespec="seconds"), "dropped": {"implausible_delay": 2}}}})

    app = open_page(AppTest.from_file(DASHBOARD, default_timeout=60), "system")
    assert not app.exception
    assert {"Data sources", "Processing runs", "Deployed model"} <= {s.value for s in app.subheader}
    sources = app.dataframe[0].value.set_index("source")
    assert sources.loc["Vehicle positions", "status"] == "🟢 Live"
    assert sources.loc["Vehicle positions", "today"] == "60 records"
    assert sources.loc["Stop and journey events", "status"] == "⚪ No data"
    assert sources.loc["Trip updates", "today"] == "2 tram trips today, 1 cancelled"
    runs = app.dataframe[1].value
    assert runs.loc[0, "implausible_delay"] == 2 and runs.loc[0, "kept_pct"] == pytest.approx(33.3)
    assert any("No model deployed" in i.value for i in app.info)


def test_history_stops_tab_uses_stop_events(isolated_paths):
    now = datetime.now(UTC)
    raw_dir = isolated_paths / "raw"
    raw_dir.mkdir()
    events = [make_vp(veh=v, t=now - timedelta(minutes=m), event="ARS", stop=1000 + v, desi=str(v),
                      ttarr=(now - timedelta(minutes=m, seconds=30 * v)).strftime("%Y-%m-%dT%H:%M:%S.000Z"))
              for v in range(1, 4) for m in range(10, 70, 10)]
    (raw_dir / f"events_{now:%Y-%m-%d}.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))

    app = open_page(AppTest.from_file(DASHBOARD, default_timeout=60), "history")
    assert not app.exception
    show_section(app, "Stops")
    metrics = {m.label: m.value for m in app.metric}
    assert metrics["Arrivals"] == "18" and metrics["Stops"] == "3"
