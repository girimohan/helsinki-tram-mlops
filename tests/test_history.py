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
        raw = [make_vp(veh=v, t=t0 + timedelta(seconds=30 * i), dl=-(v * 10 + i), desi=str(v % 3 + 1))
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

    pd.testing.assert_series_equal(
        h.by_line().sort_index(), df.groupby("line")["delay_s"].mean().round().rename("mean_delay_s"),
        check_index_type=False, check_names=False)
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
