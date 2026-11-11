"""Health of the running system, for the dashboard's System page.

Everything here is read from files the services write: how recently each source was
written, how much arrived today, what the processing runs kept and dropped, and which
model is deployed. Nothing is loaded in bulk; line counts stream through the files.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from tram_mlops import alerts, storage

FRESH, STALE, MISSING = "fresh", "stale", "missing"
BADGE = {FRESH: "🟢 Live", STALE: "🟠 Stale", MISSING: "⚪ No data"}


@dataclass
class Source:
    name: str
    feed: str                     # where it comes from and how
    path: Path | None             # today's file, if any
    last_write: pd.Timestamp | None
    newest_record: pd.Timestamp | None = None
    records_today: int | None = None
    stale_after_s: float | None = None  # None: written only on change, so age says nothing about health
    note: str = ""

    def state(self, now: pd.Timestamp) -> str:
        if self.last_write is None:
            return MISSING
        if self.stale_after_s is None:
            return FRESH
        return FRESH if (now - self.last_write).total_seconds() <= self.stale_after_s else STALE

    @property
    def size_bytes(self) -> int:
        return self.path.stat().st_size if self.path is not None and self.path.exists() else 0


def mtime(path: Path | None) -> pd.Timestamp | None:
    if path is None or not path.exists():
        return None
    return pd.Timestamp(path.stat().st_mtime, unit="s", tz="UTC")


def count_lines(path: Path, chunk_size: int = 1 << 22) -> int:
    """Lines in a plain-text file, streamed in chunks."""
    n = 0
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            n += chunk.count(b"\n")
    return n


def last_record_time(path: Path, tail_bytes: int = 1 << 16) -> pd.Timestamp | None:
    """``tst`` of the newest complete JSON line at the end of a JSONL file."""
    with open(path, "rb") as f:
        size = f.seek(0, os.SEEK_END)
        f.seek(max(0, size - tail_bytes))
        lines = f.read().split(b"\n")
    for line in reversed(lines):
        try:
            record = json.loads(line)
            return pd.Timestamp(record["tst"]).tz_convert("UTC")
        except (ValueError, KeyError, TypeError, AttributeError):
            continue
    return None


def _today_file(directory: Path, pattern: str, today: date) -> Path | None:
    path = directory / pattern.format(day=today.isoformat())
    return path if path.exists() else None


def _newest(directory: Path, glob: str) -> Path | None:
    files = sorted(directory.glob(glob))
    return files[-1] if files else None


def hfp_source(name: str, feed: str, raw_dir: Path, prefix: str, today: date, stale_after_s: float) -> Source:
    """A daily JSONL file the MQTT collector appends to (positions or stop events)."""
    path = _today_file(raw_dir, prefix + "_{day}.jsonl", today)
    newest = path or _newest(raw_dir, f"{prefix}_*.jsonl")
    return Source(
        name=name, feed=feed, path=path, last_write=mtime(newest),
        newest_record=last_record_time(newest) if newest else None,
        records_today=count_lines(path) if path else 0, stale_after_s=stale_after_s,
    )


def trip_updates_source(gtfsrt_dir: Path, today: date, stale_after_s: float) -> Source:
    path = _today_file(gtfsrt_dir, "trip_updates_{day}.jsonl.gz", today)
    return Source(
        name="Trip updates", feed="HSL GTFS-RT, polled every 30 s; changed tram trips only",
        path=path, last_write=mtime(path or _newest(gtfsrt_dir, "trip_updates_*.jsonl.gz")),
        stale_after_s=stale_after_s,
    )


def alerts_source(gtfsrt_dir: Path, today: date) -> Source:
    path = _today_file(gtfsrt_dir, "alerts_{day}.jsonl.gz", today)
    snapshot = alerts.latest_snapshot(gtfsrt_dir)
    fetched = pd.Timestamp(snapshot["fetched_at"]).tz_convert("UTC") if snapshot else None
    return Source(
        name="Service alerts", feed="HSL GTFS-RT, polled every 60 s; saved when the alert list changes",
        path=path, last_write=fetched, newest_record=fetched,
        note=f"{snapshot.get('count', 0)} alerts in the feed (all modes)" if snapshot else "",
    )


def manifest_table(root: Path) -> pd.DataFrame:
    """One row per processed day: raw rows read, rows kept, rows dropped per cleaning rule."""
    partitions = storage.load_manifest(root).get("partitions", {})
    rows = []
    for day, entry in sorted(partitions.items(), reverse=True):
        dropped = entry.get("dropped", {})
        raw_rows = entry.get("raw_rows") or 0
        rows.append({
            "day": day,
            "source": entry.get("source", ""),
            "raw_rows": raw_rows,
            "rows": entry.get("rows", 0),
            "kept_pct": round(100 * entry.get("rows", 0) / raw_rows, 1) if raw_rows else None,
            "missing_required": dropped.get("missing_required", 0),
            "implausible_delay": dropped.get("implausible_delay", 0),
            "excluded_line": dropped.get("excluded_line", 0),
            "duplicate": dropped.get("duplicate", 0),
            "malformed": dropped.get("malformed", 0),
            "processed_at": pd.Timestamp(entry["processed_at"]) if entry.get("processed_at") else pd.NaT,
            "version": entry.get("processing_version"),
        })
    return pd.DataFrame(rows)


def model_info(models_dir: Path) -> dict[str, Any] | None:
    """The deployed model's metrics record, without loading the model itself."""
    pointer = models_dir / "latest.json"
    if not pointer.exists():
        return None
    try:
        version = json.loads(pointer.read_text())["version"]
        return json.loads((models_dir / version / "metrics.json").read_text())
    except (OSError, ValueError, KeyError):
        return None


def disk_usage(paths: dict[str, tuple[Path, str]]) -> pd.DataFrame:
    """Bytes and file count per labelled glob, e.g. ``{"Positions": (raw_dir, "tram_*")}``."""
    rows = []
    for label, (directory, glob) in paths.items():
        files = [f for f in directory.glob(glob) if f.is_file()]
        rows.append({"data": label, "files": len(files), "bytes": sum(f.stat().st_size for f in files)})
    return pd.DataFrame(rows)
