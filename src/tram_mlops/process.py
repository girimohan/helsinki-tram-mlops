"""Turn raw HFP JSONL files into clean, deduplicated, day-partitioned Parquet.

Run with ``python -m tram_mlops.process``. It reads ``paths.raw_dir`` (falling back
to the bundled sample) and writes one partition per UTC day under
``paths.processed_dir`` (see ``storage``). Days whose raw file hasn't changed since
the last run are skipped, so it is cheap to run every hour. Each day is parsed on its
own, in chunks, so memory use is bounded by one day of cleaned data.

    --since DATE   reprocess days from DATE on, changed or not (a backfill)
    --rebuild      reprocess every day
    --compact      first gzip raw days older than yesterday
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
from collections import Counter
from collections.abc import Iterable, Iterator
from datetime import UTC, date, datetime
from itertools import islice
from pathlib import Path
from typing import IO, Any, Final

import pandas as pd

from tram_mlops import storage
from tram_mlops.config import load_config, resolve_path

log = logging.getLogger(__name__)

# HFP field -> output column. See https://digitransit.fi/en/developers/apis/5-realtime-api/vehicle-positions/
COLUMNS = {
    "veh": "vehicle",
    "oper": "operator",
    "desi": "line",
    "route": "route",
    "dir": "direction",
    "oday": "operating_day",
    "start": "start_time",
    "spd": "speed_mps",
    "hdg": "heading",
    "lat": "lat",
    "long": "lon",
    "odo": "odometer_m",
    "drst": "doors_open",
    "stop": "stop",
    "occu": "occupancy",
    "loc": "location_source",
}
REQUIRED = ["veh", "tst", "dl"]
# Raise when cleaning rules change, so the next run rebuilds every partition.
PROCESSING_VERSION = 2
# Arrow-backed strings take a fraction of the memory of Python string objects.
STRING: Final = "string[pyarrow]"
STRING_COLUMNS = ("line", "route", "direction", "operating_day", "start_time", "location_source")
# Vehicle positions; the collector's other files (events_*, gtfsrt/) are processed elsewhere.
RAW_PATTERNS = ("tram_*.jsonl", "tram_*.jsonl.gz")


def find_raw_files(inputs: Iterable[Path]) -> list[Path]:
    files: list[Path] = []
    for path in inputs:
        if path.is_dir():
            files.extend(sorted(f for pattern in RAW_PATTERNS for f in path.glob(pattern)))  # day order
        elif path.exists():
            files.append(path)
    return files


def _open_text(file: Path) -> IO[str]:
    """Open a raw file, transparently decompressing finished days stored as ``.gz``."""
    if file.suffix == ".gz":
        return gzip.open(file, "rt", encoding="utf-8")
    return open(file, encoding="utf-8")


def _parse_lines(lines: Iterable[str | bytes]) -> tuple[list[dict], int]:
    records, bad = [], 0
    for line in lines:
        try:
            record = json.loads(line)
        except (json.JSONDecodeError, UnicodeDecodeError):
            bad += 1
            continue
        if isinstance(record, dict):
            records.append(record)
        else:
            bad += 1
    return records, bad


def iter_raw_chunks(files: Iterable[Path], chunk_size: int = 50_000,
                    counts: Counter[str] | None = None) -> Iterator[pd.DataFrame]:
    """Raw records, at most ``chunk_size`` per DataFrame, skipping lines that aren't JSON objects."""
    counts = Counter() if counts is None else counts
    for file in files:
        with _open_text(file) as f:
            while lines := list(islice(f, chunk_size)):
                records, bad = _parse_lines(lines)
                counts["raw_rows"] += len(lines)
                counts["malformed"] += bad
                if records:
                    yield pd.DataFrame.from_records(records)


def load_raw(files: Iterable[Path]) -> pd.DataFrame:
    """Read JSONL files into one DataFrame, skipping lines that aren't valid JSON objects."""
    counts: Counter[str] = Counter()
    chunks = list(iter_raw_chunks(files, counts=counts))
    if counts["malformed"]:
        log.warning("Skipped %d malformed lines", counts["malformed"])
    return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()


def _tail_records(file: Path, cutoff: pd.Timestamp, chunk_size: int = 1 << 20) -> tuple[list[dict], bool]:
    """Records at the end of a JSONL file, reading backwards until one is older than ``cutoff``.

    Returns the records and whether the cutoff was reached inside this file.
    """
    with open(file, "rb") as f:
        pos = f.seek(0, 2)
        data = b""
        while pos > 0:
            step = min(chunk_size, pos)
            pos -= step
            f.seek(pos)
            data = f.read(step) + data
            # The first line of a mid-file chunk may be partial; check the first complete one.
            lines = data.split(b"\n")
            first = lines[1] if pos > 0 and len(lines) > 1 else lines[0]
            records, _ = _parse_lines([first])
            if records and pd.Timestamp(records[0].get("tst", "")) < cutoff:
                break
    lines = data.split(b"\n")
    if pos > 0:
        lines = lines[1:]
    # The writer may be mid-line at the end of the file; _parse_lines skips that line.
    records, _ = _parse_lines(line for line in lines if line.strip())
    return records, pos > 0


def load_recent(raw_dir: Path, window: pd.Timedelta, now: pd.Timestamp | None = None) -> pd.DataFrame:
    """Raw records from the last ``window`` of the daily files the ingester is writing."""
    now = now or pd.Timestamp.now(tz="UTC")
    cutoff = now - window
    records: list[dict] = []
    # Daily files are named by UTC date, so a window can span the last two.
    for file in reversed(sorted(raw_dir.glob("tram_*.jsonl"))[-2:]):
        file_records, reached = _tail_records(file, cutoff)
        records = file_records + records
        if reached:
            break
    raw = pd.DataFrame.from_records(records)
    if raw.empty or "tst" not in raw:
        return raw
    tst = pd.to_datetime(raw["tst"], utc=True, errors="coerce", format="ISO8601")
    return raw[tst >= cutoff].reset_index(drop=True)


def _clean_settings() -> dict[str, Any]:
    proc = load_config()["process"]
    return {
        "tz": proc["timezone"],
        "on_time_window": (proc["on_time_min_s"], proc["on_time_max_s"]),
        "max_abs_delay_s": proc["max_abs_delay_s"],
        "exclude_lines": proc.get("exclude_lines", ()),
    }


def clean_with_config(raw: pd.DataFrame) -> pd.DataFrame:
    """``clean`` with the settings from configs/config.toml."""
    return clean(raw, **_clean_settings())


def clean(raw: pd.DataFrame, tz: str = "Europe/Helsinki",
          on_time_window: tuple[int, int] = (-60, 180), max_abs_delay_s: int = 1800,
          exclude_lines: Iterable[str] = (), counts: Counter[str] | None = None) -> pd.DataFrame:
    """Normalise raw HFP records.

    ``delay_s`` is positive when a tram runs *behind* schedule. HFP's ``dl`` uses
    the opposite sign (positive = ahead), so it is negated here. ``counts``, if given,
    collects how many rows each rule dropped.
    """
    return _finish(_normalise(raw, tz, on_time_window, max_abs_delay_s, exclude_lines, counts), counts)


def clean_files(files: Iterable[Path], chunk_size: int = 50_000) -> tuple[pd.DataFrame, Counter[str]]:
    """``clean_with_config`` over raw files, one chunk at a time, plus rows dropped per rule.

    Only the cleaned, compact columns of each chunk are kept, so peak memory stays far
    below loading every raw record at once.
    """
    settings = _clean_settings()
    counts: Counter[str] = Counter()
    parts = [_normalise(chunk, counts=counts, **settings)
             for chunk in iter_raw_chunks(files, chunk_size, counts)]
    parts = [p for p in parts if not p.empty]
    combined = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    return _finish(combined, counts), counts


def _finish(df: pd.DataFrame, counts: Counter[str] | None = None) -> pd.DataFrame:
    if df.empty:
        return df
    # The broker can deliver the same message more than once.
    deduped = df.drop_duplicates(subset=["vehicle", "timestamp"])
    if counts is not None:
        counts["duplicate"] += len(df) - len(deduped)
    return deduped.sort_values(["vehicle", "timestamp"]).reset_index(drop=True)


def _normalise(raw: pd.DataFrame, tz: str = "Europe/Helsinki",
               on_time_window: tuple[int, int] = (-60, 180), max_abs_delay_s: int = 1800,
               exclude_lines: Iterable[str] = (), counts: Counter[str] | None = None) -> pd.DataFrame:
    """Everything ``clean`` does except removing duplicates and sorting."""
    counts = Counter() if counts is None else counts
    if raw.empty:
        return pd.DataFrame()
    missing = [c for c in REQUIRED if c not in raw.columns]
    if missing:
        raise ValueError(f"raw data is missing required fields: {missing}")

    df = raw.copy()
    df["timestamp"] = pd.to_datetime(df["tst"], utc=True, errors="coerce", format="ISO8601")
    df["dl"] = pd.to_numeric(df["dl"], errors="coerce")
    df["veh"] = pd.to_numeric(df["veh"], errors="coerce")
    before = len(df)
    df = df.dropna(subset=["timestamp", "dl", "veh"])
    counts["missing_required"] += before - len(df)
    # Stale journey sign-ins: trams still signed into a journey that ended hours ago.
    implausible = df["dl"].abs() > max_abs_delay_s
    counts["implausible_delay"] += int(implausible.sum())
    df = df[~implausible]
    if exclude_lines and "desi" in df.columns:
        excluded = df["desi"].astype(str).isin(list(exclude_lines))
        counts["excluded_line"] += int(excluded.sum())
        df = df[~excluded]

    out = df[[c for c in COLUMNS if c in df.columns]].rename(columns=COLUMNS)
    out["timestamp"] = df["timestamp"]
    out["delay_s"] = -df["dl"].astype(int)
    out["vehicle"] = out["vehicle"].astype(int)

    for col in STRING_COLUMNS:
        if col in out:
            out[col] = out[col].astype(STRING)
    for col in ("speed_mps", "heading", "lat", "lon", "odometer_m", "doors_open", "stop", "occupancy"):
        if col in out:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    local = out["timestamp"].dt.tz_convert(tz)
    out["local_time"] = local
    out["hour"] = local.dt.hour
    out["weekday"] = local.dt.weekday
    out["at_stop"] = out["stop"].notna() if "stop" in out else False
    lo, hi = on_time_window
    out["on_time"] = out["delay_s"].between(lo, hi)
    # A journey is one scheduled run: HSL identifies it by day, route, direction and start time.
    out["journey_id"] = (
        out["operating_day"].fillna("") + "|" + out["route"].fillna("") + "|"
        + out["direction"].fillna("") + "|" + out["start_time"].fillna("")
    )
    return out.sort_values(["vehicle", "timestamp"]).reset_index(drop=True)


def summarize(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"rows": 0}
    return {
        "rows": len(df),
        "vehicles": int(df["vehicle"].nunique()),
        "journeys": int(df["journey_id"].nunique()),
        "from": str(df["local_time"].min()),
        "to": str(df["local_time"].max()),
        "mean_delay_s": round(float(df["delay_s"].mean()), 1),
        "median_delay_s": float(df["delay_s"].median()),
        "on_time_pct": round(100 * float(df["on_time"].mean()), 1),
    }


def _fingerprint(file: Path) -> dict[str, Any]:
    stat = file.stat()
    return {"source": file.name, "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _is_current(entry: dict[str, Any] | None, file: Path, root: Path, day: date) -> bool:
    """Whether ``day``'s partition was already built from exactly this file's content."""
    if not entry or entry.get("processing_version") != PROCESSING_VERSION:
        return False
    if not storage.partition_path(root, day).exists():
        return False
    fingerprint = _fingerprint(file)
    if all(entry.get(k) == v for k, v in fingerprint.items()):
        return True
    # Name, size or mtime changed (e.g. the day was gzipped): compare the content itself.
    if storage.file_sha256(file) == entry.get("sha256"):
        entry.update(fingerprint)
        return True
    return False


def _files_by_day(files: list[Path]) -> tuple[dict[date, Path], list[Path]]:
    """Dated raw files keyed by day (preferring the uncompressed one), and undated files."""
    dated: dict[date, Path] = {}
    undated = []
    for file in files:
        day = storage.raw_day(file)
        if day is None:
            undated.append(file)
        elif day not in dated or file.suffix != ".gz":
            dated[day] = file
    return dated, undated


def _write_day(root: Path, day: date, df: pd.DataFrame, file: Path, counts: Counter[str],
               manifest: dict[str, Any]) -> None:
    storage.write_partition(root, day, df)
    dropped = {k: v for k, v in sorted(counts.items()) if k != "raw_rows" and v}
    manifest["partitions"][day.isoformat()] = {
        **_fingerprint(file),
        "sha256": storage.file_sha256(file),
        "processing_version": PROCESSING_VERSION,
        "processed_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "raw_rows": counts["raw_rows"],
        "rows": len(df),
        "dropped": dropped,
    }
    storage.save_manifest(root, manifest)  # after every day, so an interrupted run keeps its progress
    log.info("%s: %d raw -> %d clean rows from %s (dropped: %s)",
             day, counts["raw_rows"], len(df), file.name, dropped or "none")


def run(inputs: list[Path] | None = None, since: date | None = None, rebuild: bool = False,
        compact: bool = False, today: date | None = None) -> dict[str, list[str]]:
    """Bring the processed partitions up to date; returns the days processed and skipped."""
    cfg = load_config()
    paths = cfg["paths"]
    root = resolve_path(paths["processed_dir"])
    raw_dir = resolve_path(paths["raw_dir"])
    if compact and raw_dir.exists():
        storage.compact_raw(raw_dir, today or datetime.now(UTC).date())

    if inputs:
        files = find_raw_files(inputs)
    else:
        files = find_raw_files([raw_dir])
        if not files:
            log.info("No collected data in %s; using the bundled sample", paths["raw_dir"])
            files = find_raw_files([resolve_path(paths["sample_file"])])
    if not files:
        raise FileNotFoundError("no raw JSONL files found")

    manifest = storage.load_manifest(root)
    result: dict[str, list[str]] = {"processed": [], "skipped": []}
    dated, undated = _files_by_day(files)
    for day, file in sorted(dated.items()):
        forced = rebuild or (since is not None and day >= since)
        if not forced and _is_current(manifest["partitions"].get(day.isoformat()), file, root, day):
            result["skipped"].append(day.isoformat())
            continue
        df, counts = clean_files([file])
        _write_day(root, day, df, file, counts, manifest)
        result["processed"].append(day.isoformat())

    # Files without a date in their name (e.g. the bundled sample) are split by UTC day.
    for file in undated:
        df, counts = clean_files([file])
        if df.empty:
            continue
        for day, part in df.groupby(df["timestamp"].dt.date):
            _write_day(root, day, part.reset_index(drop=True), file, counts, manifest)
            result["processed"].append(day.isoformat())

    if result["skipped"]:
        log.info("%d day(s) unchanged since the last run", len(result["skipped"]))
    storage.save_manifest(root, manifest)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, nargs="*", help="JSONL files or directories")
    parser.add_argument("--since", type=date.fromisoformat, help="reprocess days from this date (YYYY-MM-DD)")
    parser.add_argument("--rebuild", action="store_true", help="reprocess every day")
    parser.add_argument("--compact", action="store_true", help="gzip raw days older than yesterday first")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run(args.input, since=args.since, rebuild=args.rebuild, compact=args.compact)


if __name__ == "__main__":
    main()
