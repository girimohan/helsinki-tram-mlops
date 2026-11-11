"""Archive HSL's GTFS-RT service alerts and tram trip updates.

Run with ``python -m tram_mlops.gtfsrt``. Nobody publishes an archive of these feeds,
so history exists only if it is recorded as it happens. Files go to
``paths.gtfsrt_dir``, one per UTC day:

- ``alerts_YYYY-MM-DD.jsonl.gz``: the full alert list, written whenever its content changes
  (HSL re-publishes unchanged alerts every few minutes, so ETags alone aren't enough)
- ``trip_updates_YYYY-MM-DD.jsonl.gz``: one line per tram trip whose update changed since
  the previous poll, and a ``removed`` line when a trip leaves the feed

and two files that are replaced rather than appended to:

- ``trip_updates_current.json``: every tram trip in the latest poll, so "when is the next
  tram" needs no scan of the day's archive
- ``tram_network.json``: tram lines, their destinations and every stop's name, from the
  static GTFS (refreshed daily). The realtime feeds name stops only by id.

Each poll is appended as its own gzip member, so a crash loses at most that poll and
the files stay readable with ``gzip.open``. Every line carries ``fetched_at``: joins
must only use updates fetched before the moment being predicted.

Trips count as trams when their route is GTFS route type 0 (tram) or 900 (light rail,
line 15) in HSL's static GTFS, or is a lettered variant of one (e.g. ``1013E``).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import logging
import shutil
import tempfile
import time
import urllib.error
import urllib.request
import zipfile
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from google.protobuf.json_format import MessageToDict
from google.transit import gtfs_realtime_pb2

from tram_mlops.config import load_config, resolve_path
from tram_mlops.storage import append_gzip_jsonl, write_json_atomic

log = logging.getLogger(__name__)

USER_AGENT = "tram-mlops (+https://github.com/girimohan/helsinki-tram-mlops)"
CURRENT_TRIPS = "trip_updates_current.json"
NETWORK = "tram_network.json"


# --- HTTP ------------------------------------------------------------------------------

@dataclass
class Fetched:
    status: int            # 200, or 304 when unchanged since the validators sent
    body: bytes = b""
    etag: str | None = None
    last_modified: str | None = None


def fetch(url: str, etag: str | None = None, last_modified: str | None = None,
          timeout: float = 30) -> Fetched:
    headers = {"User-Agent": USER_AGENT}
    if etag:
        headers["If-None-Match"] = etag
    if last_modified:
        headers["If-Modified-Since"] = last_modified
    request = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return Fetched(response.status, response.read(), response.headers.get("ETag"),
                           response.headers.get("Last-Modified"))
    except urllib.error.HTTPError as err:
        if err.code == 304:
            return Fetched(304, etag=etag, last_modified=last_modified)
        raise


def parse_feed(body: bytes) -> gtfs_realtime_pb2.FeedMessage:
    feed = gtfs_realtime_pb2.FeedMessage()
    feed.ParseFromString(body)
    return feed


# --- which routes are trams ------------------------------------------------------------

def tram_routes_from_gtfs(zip_path: Path, route_types: Iterable[str] = ("0", "900")) -> set[str]:
    """Route ids of the given GTFS route types, read from a static GTFS zip's routes.txt."""
    wanted = set(route_types)
    with zipfile.ZipFile(zip_path) as archive, archive.open("routes.txt") as f:
        reader = csv.DictReader(io.TextIOWrapper(f, encoding="utf-8-sig"))
        return {row["route_id"] for row in reader if row.get("route_type") in wanted}


def _csv_rows(archive: zipfile.ZipFile, name: str) -> Iterator[dict[str, str]]:
    with archive.open(name) as f:
        yield from csv.DictReader(io.TextIOWrapper(f, encoding="utf-8-sig"))


def network_from_gtfs(zip_path: Path, routes: set[str]) -> dict[str, Any]:
    """Names for what the realtime feeds give only as ids: tram lines, destinations, stops.

    - ``routes``: tram route id -> ``{"line", "name"}`` (``1004`` -> line ``4``)
    - ``headsigns``: ``"route|direction"`` -> the destination its trips usually show
    - ``stops``: stop id -> ``[name, code, description, lat, lon, vehicle_type, platform]``
      for every stop, since a few line 15 stops are typed as bus stops
    """
    with zipfile.ZipFile(zip_path) as archive:
        lines = {row["route_id"]: {"line": row["route_short_name"].strip(), "name": row["route_long_name"].strip()}
                 for row in _csv_rows(archive, "routes.txt") if row["route_id"] in routes}
        headsigns: defaultdict[str, Counter[str]] = defaultdict(Counter)
        for row in _csv_rows(archive, "trips.txt"):
            if row["route_id"] in routes and row.get("trip_headsign"):
                headsigns[f"{row['route_id']}|{row['direction_id']}"][row["trip_headsign"].strip()] += 1
        stops = {}
        for row in _csv_rows(archive, "stops.txt"):
            if (row.get("location_type") or "0").strip() not in ("", "0"):
                continue  # stations and entrances; vehicles stop at their platforms
            try:
                lat, lon = round(float(row["stop_lat"]), 6), round(float(row["stop_lon"]), 6)
            except (KeyError, ValueError):
                continue
            stops[row["stop_id"]] = [row["stop_name"].strip(), (row.get("stop_code") or "").strip(),
                                     (row.get("stop_desc") or "").strip(), lat, lon,
                                     (row.get("vehicle_type") or "").strip(), (row.get("platform_code") or "").strip()]
    return {"routes": lines, "headsigns": {k: c.most_common(1)[0][0] for k, c in sorted(headsigns.items())},
            "stops": stops}


def download_static(url: str, route_types: Iterable[str], network_path: Path | None = None) -> set[str]:
    """Download static GTFS (~90 MB) to a temporary file and return its tram route ids.

    With ``network_path``, also save the tram network (``network_from_gtfs``) there.
    """
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "gtfs.zip"
        with urllib.request.urlopen(request, timeout=300) as response, open(path, "wb") as out:
            shutil.copyfileobj(response, out, 1 << 20)
        routes = tram_routes_from_gtfs(path, route_types)
        if network_path is not None and routes:
            network = network_from_gtfs(path, routes)
            network["downloaded_at"] = datetime.now(UTC).isoformat(timespec="seconds")
            write_json_atomic(network_path, network)
        return routes


def is_tram(route_id: str, routes: set[str]) -> bool:
    """A tram route, or a lettered variant of one that static GTFS doesn't list (``1013E``)."""
    if route_id in routes:
        return True
    base, suffix = route_id[:4], route_id[4:]
    return bool(suffix) and suffix.isalpha() and base in routes


# --- trip updates ----------------------------------------------------------------------

def trip_key(trip: gtfs_realtime_pb2.TripDescriptor) -> str:
    """HSL has no stable trip ids; a trip is route + direction + start date + start time."""
    direction = trip.direction_id if trip.HasField("direction_id") else ""
    return f"{trip.route_id}|{direction}|{trip.start_date}|{trip.start_time}"


def _opt(message: Any, field: str) -> Any:
    return getattr(message, field) if message.HasField(field) else None


def trip_record(entity: gtfs_realtime_pb2.FeedEntity) -> dict[str, Any]:
    """A compact, JSON-ready view of one trip update."""
    update = entity.trip_update
    trip = update.trip
    rel = gtfs_realtime_pb2.TripDescriptor.ScheduleRelationship.Name
    stop_rel = gtfs_realtime_pb2.TripUpdate.StopTimeUpdate.ScheduleRelationship.Name
    stops = []
    for stu in update.stop_time_update:
        arrival = _opt(stu.arrival, "time") if stu.HasField("arrival") else None
        departure = _opt(stu.departure, "time") if stu.HasField("departure") else None
        # uncertainty 0 marks a stop already passed: the time is observed, not predicted.
        timed = stu.arrival if stu.HasField("arrival") else stu.departure
        uncertainty = _opt(timed, "uncertainty")
        stops.append([stu.stop_id, arrival, departure, uncertainty, stop_rel(stu.schedule_relationship)])
    record: dict[str, Any] = {
        "key": trip_key(trip),
        "route_id": trip.route_id,
        "direction_id": _opt(trip, "direction_id"),
        "start_date": trip.start_date,
        "start_time": trip.start_time,
        "schedule_relationship": rel(trip.schedule_relationship),
        "timestamp": _opt(update, "timestamp"),
        "stops": stops,
    }
    if update.HasField("vehicle") and update.vehicle.id:
        record["vehicle"] = update.vehicle.id
    return record


class TripUpdateArchive:
    """Turns successive trip-update snapshots into lines for the trips that changed."""

    def __init__(self) -> None:
        self._seen: dict[str, str] = {}
        self.current: list[dict[str, Any]] = []  # every tram trip in the latest snapshot

    def changes(self, feed: gtfs_realtime_pb2.FeedMessage, routes: set[str],
                fetched_at: str) -> list[dict[str, Any]]:
        current: dict[str, str] = {}
        lines = []
        self.current = []
        header_ts = feed.header.timestamp or None
        for entity in feed.entity:
            if not entity.HasField("trip_update") or not is_tram(entity.trip_update.trip.route_id, routes):
                continue
            record = trip_record(entity)
            self.current.append(record)
            digest = hashlib.sha1(json.dumps(record, sort_keys=True).encode()).hexdigest()
            current[record["key"]] = digest
            if self._seen.get(record["key"]) != digest:
                lines.append({"fetched_at": fetched_at, "header_ts": header_ts, **record})
        for key in self._seen.keys() - current.keys():
            lines.append({"fetched_at": fetched_at, "header_ts": header_ts, "key": key, "removed": True})
        self._seen = current
        return lines


# --- alerts ----------------------------------------------------------------------------

class AlertArchive:
    """Keeps the alert list only when its content changes, ignoring re-publication."""

    def __init__(self) -> None:
        self._digest: str | None = None

    def changes(self, feed: gtfs_realtime_pb2.FeedMessage, fetched_at: str) -> list[dict[str, Any]]:
        alerts = [{"id": e.id, **MessageToDict(e.alert, preserving_proto_field_name=True)}
                  for e in feed.entity if e.HasField("alert")]
        digest = hashlib.sha1(json.dumps(alerts, sort_keys=True).encode()).hexdigest()
        if digest == self._digest:
            return []
        self._digest = digest
        return [{"fetched_at": fetched_at, "header_ts": feed.header.timestamp or None,
                 "count": len(alerts), "alerts": alerts}]


# --- storage ---------------------------------------------------------------------------

def daily_path(directory: Path, prefix: str, fetched_at: datetime) -> Path:
    return directory / f"{prefix}_{fetched_at:%Y-%m-%d}.jsonl.gz"


# --- poller ----------------------------------------------------------------------------

class Poller:
    def __init__(self, cfg: dict[str, Any], out_dir: Path):
        self.cfg = cfg
        self.out_dir = out_dir
        self.trips = TripUpdateArchive()
        self.alerts = AlertArchive()
        self.routes: set[str] = set()
        self.routes_loaded_at = 0.0
        self.routes_retry_at = 0.0
        self.alert_validators: tuple[str | None, str | None] = (None, None)
        self.stats = {"trip_lines": 0, "alert_snapshots": 0, "errors": 0}

    def refresh_routes(self, now: float) -> None:
        if now < self.routes_retry_at or (
                self.routes and now - self.routes_loaded_at < self.cfg["static_refresh_h"] * 3600):
            return
        # Whatever happens, don't download ~90 MB again for a while.
        self.routes_retry_at = now + 600
        routes = download_static(self.cfg["static_gtfs_url"], self.cfg["tram_route_types"],
                                 self.out_dir / NETWORK)
        if not routes:
            raise ValueError("static GTFS lists no tram routes")
        self.routes, self.routes_loaded_at = routes, now
        log.info("Loaded %d tram routes from static GTFS", len(routes))

    def poll_trip_updates(self) -> None:
        if not self.routes:
            return  # can't tell trams from buses yet
        now = datetime.now(UTC)
        result = fetch(self.cfg["trip_updates_url"])
        lines = self.trips.changes(parse_feed(result.body), self.routes, now.isoformat(timespec="seconds"))
        append_gzip_jsonl(daily_path(self.out_dir, "trip_updates", now), lines)
        write_json_atomic(self.out_dir / CURRENT_TRIPS,
                          {"fetched_at": now.isoformat(timespec="seconds"), "trips": self.trips.current})
        self.stats["trip_lines"] += len(lines)

    def poll_alerts(self) -> None:
        now = datetime.now(UTC)
        result = fetch(self.cfg["alerts_url"], *self.alert_validators)
        if result.status == 304:
            return
        self.alert_validators = (result.etag, result.last_modified)
        lines = self.alerts.changes(parse_feed(result.body), now.isoformat(timespec="seconds"))
        append_gzip_jsonl(daily_path(self.out_dir, "alerts", now), lines)
        self.stats["alert_snapshots"] += len(lines)

    def run(self, duration_s: float | None = None) -> None:
        tasks = [
            (self.poll_alerts, self.cfg["alerts_interval_s"]),
            (self.poll_trip_updates, self.cfg["trip_updates_interval_s"]),
        ]
        due = [0.0] * len(tasks)
        started = last_log = time.monotonic()
        while duration_s is None or time.monotonic() - started < duration_s:
            now = time.monotonic()
            try:
                self.refresh_routes(time.time())
            except Exception as exc:  # keep polling alerts even if static GTFS is unavailable
                self.stats["errors"] += 1
                log.warning("Could not load tram routes from static GTFS: %s", exc)
            for i, (task, interval) in enumerate(tasks):
                if now >= due[i]:
                    due[i] = now + interval
                    try:
                        task()
                    except Exception as exc:  # one failed poll must not stop the archive
                        self.stats["errors"] += 1
                        log.warning("%s failed: %s", task.__name__, exc)
            if now - last_log >= 600:
                log.info("Archived %(trip_lines)d trip-update lines and %(alert_snapshots)d alert snapshots "
                         "(%(errors)d errors)", self.stats)
                last_log = now
            wait = min(due) - time.monotonic()
            if duration_s is not None:
                wait = min(wait, started + duration_s - time.monotonic())
            time.sleep(max(0.0, wait))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--duration", type=float, help="stop after this many seconds (default: run forever)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config()
    Poller(cfg["gtfsrt"], resolve_path(cfg["paths"]["gtfsrt_dir"])).run(args.duration)


if __name__ == "__main__":
    main()
