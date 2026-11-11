"""Day-partitioned storage for processed positions, plus raw-file housekeeping.

Layout under ``paths.processed_dir``::

    date=YYYY-MM-DD/part.parquet   one partition per UTC collection day
    _manifest.json                 what each partition was built from

Every partition is written with the same Arrow schema, so readers can scan any
range of days as one table. Writes go to a temporary file first and are moved
into place with ``os.replace``, so a crash never leaves a half-written file.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import logging
import os
import re
import shutil
import tempfile
from collections.abc import Iterable
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

log = logging.getLogger(__name__)

PARTITION_FILE = "part.parquet"
MANIFEST = "_manifest.json"
RAW_NAME = re.compile(r"(?:tram|events)_(\d{4}-\d{2}-\d{2})\.jsonl(\.gz)?$")
# Daily raw files the collector writes as plain JSONL (positions and stop events).
RAW_PREFIXES = ("tram", "events")
TZ = "Europe/Helsinki"
# Temporary files are created owner-only; finished files get normal read permissions.
FILE_MODE = 0o644

# The processed table's contract: column order and types are fixed for every partition.
POSITIONS_SCHEMA = pa.schema([
    ("vehicle", pa.int64()),
    ("operator", pa.int64()),
    ("line", pa.string()),
    ("route", pa.string()),
    ("direction", pa.string()),
    ("operating_day", pa.string()),
    ("start_time", pa.string()),
    ("speed_mps", pa.float64()),
    ("heading", pa.float64()),
    ("lat", pa.float64()),
    ("lon", pa.float64()),
    ("odometer_m", pa.float64()),
    ("doors_open", pa.float64()),
    ("stop", pa.float64()),
    ("occupancy", pa.float64()),
    ("location_source", pa.string()),
    ("timestamp", pa.timestamp("ns", tz="UTC")),
    ("delay_s", pa.int64()),
    ("local_time", pa.timestamp("ns", tz=TZ)),
    ("hour", pa.int64()),
    ("weekday", pa.int64()),
    ("at_stop", pa.bool_()),
    ("on_time", pa.bool_()),
    ("journey_id", pa.string()),
])
COLUMNS = POSITIONS_SCHEMA.names


# --- raw files -------------------------------------------------------------------------

def raw_day(path: Path) -> date | None:
    """The UTC collection day in a raw file's name (``tram_`` or ``events_YYYY-MM-DD.jsonl[.gz]``)."""
    match = RAW_NAME.search(path.name)
    return date.fromisoformat(match.group(1)) if match else None


def file_sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    """Hash of a raw file's *uncompressed* content, so gzipping a day doesn't change it."""
    digest = hashlib.sha256()
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as f:
        while chunk := f.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def compact_raw(raw_dir: Path, today: date, keep_days: int = 2) -> list[Path]:
    """Gzip finished raw days older than ``keep_days`` and remove the originals.

    The live reader (``process.load_recent``) only reads uncompressed files, and only
    the newest two, so the default keeps today and yesterday as they are.
    """
    compacted = []
    for src in sorted(f for prefix in RAW_PREFIXES for f in raw_dir.glob(f"{prefix}_*.jsonl")):
        day = raw_day(src)
        if day is None or (today - day).days < keep_days:
            continue
        dest = src.with_name(src.name + ".gz")
        size = src.stat().st_size
        if not dest.exists():
            fd, tmp_name = tempfile.mkstemp(dir=raw_dir, prefix=".compact-", suffix=".gz")
            tmp = Path(tmp_name)
            try:
                with os.fdopen(fd, "wb") as raw_out, \
                        gzip.GzipFile(filename=src.name, mode="wb", fileobj=raw_out, mtime=0) as out, \
                        open(src, "rb") as f:
                    shutil.copyfileobj(f, out, 1 << 20)
                shutil.copymode(src, tmp)
                os.replace(tmp, dest)
            finally:
                tmp.unlink(missing_ok=True)
        # Only delete the original once the archive provably holds all of it.
        if _gzip_size(dest) != size % (1 << 32):
            log.error("%s does not match %s; keeping the original", dest.name, src.name)
            continue
        src.unlink()
        compacted.append(dest)
        log.info("Compressed %s (%.0f MB -> %.0f MB)", src.name, size / 1e6, dest.stat().st_size / 1e6)
    return compacted


def _gzip_size(path: Path) -> int:
    """Uncompressed size modulo 2**32, from the gzip trailer."""
    with open(path, "rb") as f:
        f.seek(-4, os.SEEK_END)
        return int.from_bytes(f.read(4), "little")


# --- partitions ------------------------------------------------------------------------

def partition_path(root: Path, day: date) -> Path:
    return root / f"date={day.isoformat()}" / PARTITION_FILE


def list_partitions(root: Path) -> list[date]:
    days = []
    for path in root.glob(f"date=*/{PARTITION_FILE}"):
        try:
            days.append(date.fromisoformat(path.parent.name.removeprefix("date=")))
        except ValueError:
            continue
    return sorted(days)


def to_table(df: pd.DataFrame) -> pa.Table:
    """Conform a cleaned positions frame to ``POSITIONS_SCHEMA`` (missing columns become null)."""
    arrays = [
        pa.array(df[field.name], type=field.type, from_pandas=True) if field.name in df
        else pa.nulls(len(df), type=field.type)
        for field in POSITIONS_SCHEMA
    ]
    return pa.Table.from_arrays(arrays, schema=POSITIONS_SCHEMA)


def write_partition(root: Path, day: date, df: pd.DataFrame) -> Path:
    path = partition_path(root, day)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=".part-", suffix=".parquet")
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        pq.write_table(to_table(df), tmp, compression="zstd")
        tmp.chmod(FILE_MODE)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
    return path


def read_positions(root: Path, start: date | None = None, end: date | None = None,
                   columns: Iterable[str] | None = None) -> pd.DataFrame:
    """Processed positions for the UTC days ``start``..``end`` (inclusive; open-ended if None).

    Only the matching partitions are opened, and only the requested columns are read.
    """
    days = [d for d in list_partitions(root)
            if (start is None or d >= start) and (end is None or d <= end)]
    cols = list(columns) if columns is not None else COLUMNS
    if not days:
        return to_table(pd.DataFrame()).select(cols).to_pandas()
    dataset = ds.dataset([str(partition_path(root, d)) for d in days], schema=POSITIONS_SCHEMA, format="parquet")
    df = dataset.to_table(columns=cols).to_pandas()
    for col in ("line", "route", "direction", "operating_day", "start_time"):
        if col in df:
            df[col] = df[col].astype("string")
    return df


def dataset_glob(root: Path) -> str:
    """Glob over all partitions, for DuckDB's ``read_parquet``."""
    return (root / "date=*" / PARTITION_FILE).as_posix()


def write_json_atomic(path: Path, data: Any) -> None:
    """Replace ``path`` with ``data`` as JSON, so readers see the old file or the new one, never half."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.stem}-", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, separators=(",", ":"), ensure_ascii=False)
        os.chmod(tmp, FILE_MODE)
        os.replace(tmp, path)
    finally:
        Path(tmp).unlink(missing_ok=True)


# --- append-only logs ------------------------------------------------------------------

def append_gzip_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    """Append records as one gzip member.

    Concatenated members read back as one stream with ``gzip.open``, so a crash loses at
    most the batch being written.
    """
    if not records:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    data = "".join(json.dumps(r, separators=(",", ":"), default=str) + "\n" for r in records)
    with open(path, "ab") as f:
        f.write(gzip.compress(data.encode("utf-8"), compresslevel=6, mtime=0))


def read_gzip_jsonl(path: Path) -> list[dict[str, Any]]:
    """Records from an append-only gzip JSONL log, ignoring a truncated final member or line."""
    records = []
    try:
        with gzip.open(path, "rt", encoding="utf-8") as f:
            for line in f:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except (EOFError, gzip.BadGzipFile, OSError):
        pass  # the writer was interrupted mid-member; keep what was read before it
    return records


# --- manifest --------------------------------------------------------------------------

def load_manifest(root: Path) -> dict[str, Any]:
    path = root / MANIFEST
    if not path.exists():
        return {"partitions": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def save_manifest(root: Path, manifest: dict[str, Any]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=root, prefix=".manifest-", suffix=".json")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    os.chmod(tmp_name, FILE_MODE)
    os.replace(tmp_name, root / MANIFEST)
