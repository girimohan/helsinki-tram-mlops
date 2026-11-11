"""Dashboard queries over the raw E16 feeds, computed in DuckDB.

- trip updates (``gtfsrt/trip_updates_YYYY-MM-DD.jsonl.gz``): HSL's cancellations per route
- stop events (``events_YYYY-MM-DD.jsonl``): arrival punctuality at each stop

These files are raw until E16 phase 2 turns them into partitions, so queries read only
the columns they need and return aggregates, never the rows themselves.
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import duckdb
import pandas as pd

# The dashboard shares a 1 GB container; keep each scan well inside it.
DUCKDB_SETTINGS: dict[str, str | int] = {"threads": 2, "memory_limit": "256MB"}
# HSL's service day runs past midnight: night trips belong to the previous day.
SERVICE_DAY_START_H = 4.5

TRIP_UPDATE_COLUMNS = ("{fetched_at: 'VARCHAR', key: 'VARCHAR', route_id: 'VARCHAR', start_date: 'VARCHAR', "
                       "schedule_relationship: 'VARCHAR', removed: 'BOOLEAN'}")
EVENT_COLUMNS = ("{event: 'VARCHAR', desi: 'VARCHAR', route: 'VARCHAR', veh: 'BIGINT', stop: 'BIGINT', "
                 "tst: 'TIMESTAMPTZ', ttarr: 'TIMESTAMPTZ', lat: 'DOUBLE', long: 'DOUBLE'}")


def connect() -> duckdb.DuckDBPyConnection:
    return duckdb.connect(config=dict(DUCKDB_SETTINGS))


def _file_list(files: list[Path]) -> str:
    return "[" + ", ".join("'" + f.as_posix().replace("'", "''") + "'" for f in files) + "]"


def service_day(now: pd.Timestamp, tz: str = "Europe/Helsinki") -> date:
    """The HSL service day running at ``now`` (trips after midnight count to the day before)."""
    return (now.tz_convert(tz) - pd.Timedelta(hours=SERVICE_DAY_START_H)).date()


def trip_update_files(gtfsrt_dir: Path, day: date) -> list[Path]:
    """Archive files that can hold trips of service day ``day`` (fetched on that UTC day or the next)."""
    names = [f"trip_updates_{d.isoformat()}.jsonl.gz" for d in (day, day + timedelta(days=1))]
    return [gtfsrt_dir / n for n in names if (gtfsrt_dir / n).exists()]


def trips_by_route(gtfsrt_dir: Path, day: date) -> pd.DataFrame:
    """Per GTFS route: tram trips HSL reported for service day ``day`` and how many it has cancelled.

    A trip counts as cancelled when its latest update says so (HSL also reinstates trips).
    Columns: ``route_id``, ``trips``, ``cancelled``.
    """
    files = trip_update_files(gtfsrt_dir, day)
    if not files:
        return pd.DataFrame(columns=["route_id", "trips", "cancelled"])
    sql = f"""
        WITH latest AS (
            SELECT key, any_value(route_id) AS route_id,
                   arg_max(schedule_relationship, fetched_at) AS status
            FROM read_json({_file_list(files)}, format = 'newline_delimited', compression = 'gzip',
                           columns = {TRIP_UPDATE_COLUMNS}, ignore_errors = true)
            WHERE start_date = $day AND coalesce(NOT removed, true) AND route_id IS NOT NULL
            GROUP BY key)
        SELECT route_id, count(*) AS trips, count(*) FILTER (WHERE status = 'CANCELED') AS cancelled
        FROM latest GROUP BY route_id ORDER BY route_id"""
    with connect() as con:
        return con.execute(sql, {"day": day.strftime("%Y%m%d")}).df()


def event_files(raw_dir: Path, days: int = 2) -> list[Path]:
    """The newest ``days`` daily stop-event files still being kept uncompressed."""
    return sorted(raw_dir.glob("events_*.jsonl"))[-days:]


def event_counts(path: Path) -> pd.DataFrame:
    """Events per type in one stop-event file. Columns: ``event``, ``count``."""
    if not path.exists():
        return pd.DataFrame(columns=["event", "count"])
    sql = f"""
        SELECT event, count(*) AS count
        FROM read_json({_file_list([path])}, format = 'newline_delimited', columns = {EVENT_COLUMNS},
                       ignore_errors = true)
        WHERE event IS NOT NULL GROUP BY event ORDER BY count DESC"""
    with connect() as con:
        return con.execute(sql).df()


def stop_arrivals(raw_dir: Path, since: pd.Timestamp, lines: list[str] | None = None,
                  on_time_window: tuple[int, int] = (-60, 180)) -> pd.DataFrame:
    """Arrival punctuality at each stop since ``since``, from HFP ``ARS`` (arrived at stop) events.

    Delay is the arrival time minus the scheduled arrival ``ttarr`` (positive = late).
    Columns: ``stop``, ``arrivals``, ``mean_delay_s``, ``median_delay_s``, ``on_time_pct``,
    ``lines`` (comma-separated), ``lat``, ``lon``; most arrivals first.
    """
    files = event_files(raw_dir)
    if not files:
        return pd.DataFrame(columns=["stop", "arrivals", "mean_delay_s", "median_delay_s", "on_time_pct",
                                     "lines", "lat", "lon"])
    lo, hi = on_time_window
    params: dict[str, object] = {"since": since.to_pydatetime(), "lo": lo, "hi": hi}
    line_filter = ""
    if lines:
        line_filter = "AND list_contains($lines, desi)"
        params["lines"] = list(lines)
    sql = f"""
        WITH arrivals AS (
            SELECT stop, desi, lat, long, epoch(tst) - epoch(ttarr) AS delay_s
            FROM read_json({_file_list(files)}, format = 'newline_delimited', columns = {EVENT_COLUMNS},
                           ignore_errors = true)
            WHERE event = 'ARS' AND stop IS NOT NULL AND ttarr IS NOT NULL AND tst >= $since {line_filter})
        SELECT stop, count(*) AS arrivals, avg(delay_s) AS mean_delay_s, median(delay_s) AS median_delay_s,
               100 * avg((delay_s BETWEEN $lo AND $hi)::INT) AS on_time_pct,
               string_agg(DISTINCT desi, ', ' ORDER BY desi) AS lines,
               median(lat) AS lat, median(long) AS lon
        FROM arrivals
        WHERE abs(delay_s) <= 1800  -- stale journeys, as in process.clean
        GROUP BY stop ORDER BY arrivals DESC, stop"""
    with connect() as con:
        df = con.execute(sql, params).df()
    df["stop"] = df["stop"].astype("int64").astype(str)
    return df
