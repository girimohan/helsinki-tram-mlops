from datetime import UTC, date, datetime, timedelta

import pandas as pd
import pytest
from conftest import make_vp

from tram_mlops import process, storage
from tram_mlops.history import History

D1, D2 = date(2026, 11, 7), date(2026, 11, 9)  # a Saturday and a Monday


@pytest.fixture
def root(tmp_path):
    frames = {}
    for day in (D1, D2):
        t0 = datetime(day.year, day.month, day.day, 6, 0, tzinfo=UTC)
        raw = [make_vp(veh=v, t=t0 + timedelta(seconds=30 * i), dl=-(v * 10 + i), desi=str(v % 3 + 1),
                       oday=day.isoformat())
               for v in range(1, 6) for i in range(240)]
        frames[day] = process.clean(pd.DataFrame(raw))
        storage.write_partition(tmp_path, day, frames[day])
    return tmp_path, frames


def test_summary_and_aggregates_match_pandas(root):
    path, frames = root
    df = pd.concat(frames.values())
    h = History(path, D1, D2)
    summary = h.summary()
    expected = process.summarize(df)
    for key in ("rows", "vehicles", "journeys", "mean_delay_s", "median_delay_s", "on_time_pct"):
        assert summary[key] == expected[key], key
    assert summary["from"] == df["local_time"].min()

    by_line = h.by_line(late_s=180).set_index("line").sort_index()
    grouped = df.groupby("line")
    pd.testing.assert_series_equal(by_line["mean_delay_s"], grouped["delay_s"].mean(),
                                   check_index_type=False, check_names=False)
    pd.testing.assert_series_equal(by_line["on_time_pct"], 100 * grouped["on_time"].mean(),
                                   check_index_type=False, check_names=False)
    pd.testing.assert_series_equal(by_line["p90_delay_s"], grouped["delay_s"].quantile(0.9),
                                   check_index_type=False, check_names=False)
    assert by_line["late_pct"].to_dict() == pytest.approx(
        (100 * (df["delay_s"] > 180).groupby(df["line"]).mean()).to_dict())
    assert by_line["rows"].sum() == len(df)
    by_hour = h.by_hour()
    assert set(by_hour["day_type"]) == {"Weekend", "Weekday"}
    assert by_hour["rows"].sum() == len(df)
    assert h.over_time(3600).sum() == pytest.approx(
        df.set_index("local_time")["delay_s"].resample("1h").mean().dropna().round().sum())


def test_line_filter_and_day_range(root):
    path, frames = root
    h = History(path, D2, D2, lines=["2"])
    expected = frames[D2][frames[D2]["line"] == "2"]
    assert h.summary()["rows"] == len(expected)
    assert set(h.most_delayed()["line"]) == {"2"}
    # Lines for the sidebar ignore the line filter.
    assert h.lines_seen() and set(h.lines_seen()) == {"1", "2", "3"}
    assert History(path, date(2026, 12, 1), date(2026, 12, 2)).empty


def test_latest_and_recent_come_from_the_last_day(root):
    path, frames = root
    h = History(path, D1, D2)
    latest = h.latest_positions()
    assert len(latest) == 5 and (latest["timestamp"].dt.date == D2).all()
    recent = h.recent(minutes=1)
    newest = frames[D2]["timestamp"].max()
    assert (recent["timestamp"] >= newest - pd.Timedelta(minutes=1)).all()
    assert str(recent["local_time"].dt.tz) == "Europe/Helsinki"


def test_line_hour_day_and_histogram(root):
    path, frames = root
    df = pd.concat(frames.values())
    h = History(path, D1, D2)

    line_hour = h.line_hour()
    assert line_hour["rows"].sum() == len(df)
    expected = 100 * df.groupby(["line", "hour"])["on_time"].mean()
    got = line_hour.set_index(["line", "hour"])["on_time_pct"]
    assert got.to_dict() == pytest.approx(expected.to_dict())

    by_day = h.by_day()
    assert list(by_day["day"].dt.date) == [D1, D2]  # service days, from operating_day
    assert by_day["rows"].tolist() == [len(frames[D1]), len(frames[D2])]

    hist = h.delay_histogram(bin_s=60, lo=-120, hi=120)
    assert hist["rows"].sum() == len(df)
    assert hist["bin"].min() >= -120 and hist["bin"].max() <= 60  # out-of-range delays land in the end bins
    clipped = df["delay_s"].clip(-120, 119) // 60 * 60
    assert hist.set_index("bin")["rows"].to_dict() == clipped.value_counts().to_dict()


def test_network_maps_routes_and_stops_to_lines(tmp_path):
    t0 = datetime(2026, 11, 9, 6, 0, tzinfo=UTC)
    raw = [make_vp(veh=1, t=t0, desi="9", route="1009", stop=1234567),
           make_vp(veh=2, t=t0, desi="15", route="2015", stop=1234567),
           make_vp(veh=3, t=t0, desi="15", route="2015")]
    storage.write_partition(tmp_path, D2, process.clean(pd.DataFrame(raw)))
    route_lines, stop_lines = History(tmp_path, D2, D2, lines=["9"]).network()
    assert route_lines == {"1009": "9", "2015": "15"}  # ignores the line filter
    assert stop_lines == {"1234567": ["15", "9"]}
    assert History(tmp_path, D1, D1).network() == ({}, {})
