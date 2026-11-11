"""Aggregates over the processed partitions, computed in DuckDB.

The dashboard's History view uses these instead of loading positions into pandas, so
its memory use doesn't grow with the number of days collected. Only aggregated
results, the latest position per tram and a few minutes of recent rows come back.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

from tram_mlops import storage


def available_days(root: Path) -> list[date]:
    return storage.list_partitions(root)


class History:
    """Positions from the UTC days ``start``..``end``, optionally limited to some lines."""

    def __init__(self, root: Path, start: date, end: date, lines: list[str] | None = None,
                 tz: str = "Europe/Helsinki"):
        days = [d for d in available_days(root) if start <= d <= end]
        self.files = [storage.partition_path(root, d).as_posix() for d in days]
        self.lines = list(lines or [])
        self.tz = tz
        self._con = duckdb.connect()

    @property
    def empty(self) -> bool:
        return not self.files

    def _query(self, sql: str, files: list[str] | None = None, lines: list[str] | None = None,
               **params: Any) -> pd.DataFrame:
        """Run ``sql`` with ``{src}`` standing for the selected positions.

        ``files`` and ``lines`` default to this view's; pass them to query a subset.
        """
        files = self.files if files is None else files
        lines = self.lines if lines is None else lines
        quoted = ", ".join("'" + f.replace("'", "''") + "'" for f in files)
        where = ""
        if lines:
            where = "WHERE list_contains($lines, line)"
            params["lines"] = lines
        src = f"(SELECT * FROM read_parquet([{quoted}], hive_partitioning = false) {where})"
        return self._con.execute(sql.format(src=src), params).df()

    def _localise(self, df: pd.DataFrame) -> pd.DataFrame:
        # DuckDB returns timestamps in the session's time zone; normalise to UTC + Helsinki.
        if "timestamp" in df:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).astype("datetime64[ns, UTC]")
            df["local_time"] = df["timestamp"].dt.tz_convert(self.tz)
        return df

    def summary(self) -> dict[str, Any]:
        if self.empty:
            return {"rows": 0}
        row = self._query("""
            SELECT count(*) AS rows, count(DISTINCT vehicle) AS vehicles,
                   count(DISTINCT journey_id) AS journeys,
                   min(timestamp) AS first, max(timestamp) AS last,
                   avg(delay_s) AS mean_delay_s, median(delay_s) AS median_delay_s,
                   avg(on_time::INT) AS on_time
            FROM {src}""").iloc[0]
        if not row["rows"]:
            return {"rows": 0}
        return {
            "rows": int(row["rows"]),
            "vehicles": int(row["vehicles"]),
            "journeys": int(row["journeys"]),
            "from": pd.Timestamp(row["first"]).tz_convert(self.tz),
            "to": pd.Timestamp(row["last"]).tz_convert(self.tz),
            "mean_delay_s": round(float(row["mean_delay_s"]), 1),
            "median_delay_s": float(row["median_delay_s"]),
            "on_time_pct": round(100 * float(row["on_time"]), 1),
        }

    def by_line(self) -> pd.Series:
        df = self._query("SELECT line, avg(delay_s) AS mean_delay_s FROM {src} GROUP BY line")
        return df.set_index("line")["mean_delay_s"].round()

    def over_time(self, bucket_s: int) -> pd.Series:
        """Mean delay per time bucket, indexed by Helsinki wall-clock time."""
        df = self._query("""
            SELECT (epoch(timestamp)::BIGINT // $bucket) * $bucket AS bucket, avg(delay_s) AS mean_delay_s
            FROM {src} GROUP BY bucket ORDER BY bucket""", bucket=bucket_s)
        index = pd.to_datetime(df["bucket"], unit="s", utc=True).dt.tz_convert(self.tz).dt.tz_localize(None)
        return pd.Series(df["mean_delay_s"].round().to_numpy(), index=pd.DatetimeIndex(index), name="mean_delay_s")

    def by_hour(self) -> pd.DataFrame:
        """Mean delay and on-time share by Helsinki hour of day, weekdays vs weekends."""
        df = self._query("""
            SELECT hour, CASE WHEN weekday >= 5 THEN 'Weekend' ELSE 'Weekday' END AS day_type,
                   avg(delay_s) AS mean_delay_s, 100 * avg(on_time::INT) AS on_time_pct, count(*) AS rows
            FROM {src} GROUP BY ALL ORDER BY hour""")
        return df

    def latest_positions(self) -> pd.DataFrame:
        """Each tram's last observation on the most recent day in range."""
        if self.empty:
            return pd.DataFrame()
        return self._localise(self._query("""
            SELECT vehicle, line, delay_s, lat, lon, timestamp FROM {src}
            QUALIFY row_number() OVER (PARTITION BY vehicle ORDER BY timestamp DESC) = 1""",
            files=self.files[-1:]))

    def recent(self, minutes: float) -> pd.DataFrame:
        """All columns for the last ``minutes`` of data in range, e.g. to make forecasts from."""
        if self.empty:
            return pd.DataFrame()
        df = self._query("""
            SELECT * FROM {src}
            WHERE timestamp >= (SELECT max(timestamp) FROM {src}) - to_microseconds($us)""",
            files=self.files[-1:], us=int(minutes * 60e6))
        df = self._localise(df)
        for col in ("line", "route", "direction", "operating_day", "start_time"):
            df[col] = df[col].astype("string")
        return df.sort_values(["vehicle", "timestamp"]).reset_index(drop=True)

    def most_delayed(self, n: int = 20) -> pd.DataFrame:
        return self._query("""
            SELECT vehicle, line, round(avg(delay_s))::INT AS mean, max(delay_s) AS max, count(*) AS count
            FROM {src} GROUP BY vehicle, line ORDER BY mean DESC LIMIT $n""", n=n)

    def lines_seen(self) -> list[str]:
        if self.empty:
            return []
        return [x for x in self._query("SELECT DISTINCT line FROM {src}", lines=[])["line"] if x is not None]
