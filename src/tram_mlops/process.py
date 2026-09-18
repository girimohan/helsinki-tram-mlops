"""Turn raw HFP JSONL files into a clean, deduplicated Parquet table.

Run with ``python -m tram_mlops.process [--input FILE_OR_DIR ...]``. Without
``--input`` it reads ``paths.raw_dir`` and falls back to the bundled sample.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Iterable

import pandas as pd

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


def find_raw_files(inputs: Iterable[Path]) -> list[Path]:
    files: list[Path] = []
    for path in inputs:
        if path.is_dir():
            files.extend(sorted(path.glob("*.jsonl")))
        elif path.exists():
            files.append(path)
    return files


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


def load_raw(files: Iterable[Path]) -> pd.DataFrame:
    """Read JSONL files into one DataFrame, skipping lines that aren't valid JSON objects."""
    records, bad = [], 0
    for file in files:
        with open(file, encoding="utf-8") as f:
            file_records, file_bad = _parse_lines(f)
        records.extend(file_records)
        bad += file_bad
    if bad:
        log.warning("Skipped %d malformed lines", bad)
    return pd.DataFrame.from_records(records)


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


def clean_with_config(raw: pd.DataFrame) -> pd.DataFrame:
    """``clean`` with the settings from configs/config.toml."""
    proc = load_config()["process"]
    return clean(raw, proc["timezone"], (proc["on_time_min_s"], proc["on_time_max_s"]),
                 proc["max_abs_delay_s"], proc.get("exclude_lines", ()))


def clean(raw: pd.DataFrame, tz: str = "Europe/Helsinki",
          on_time_window: tuple[int, int] = (-60, 180), max_abs_delay_s: int = 1800,
          exclude_lines: Iterable[str] = ()) -> pd.DataFrame:
    """Normalise raw HFP records.

    ``delay_s`` is positive when a tram runs *behind* schedule. HFP's ``dl`` uses
    the opposite sign (positive = ahead), so it is negated here.
    """
    if raw.empty:
        return pd.DataFrame()
    missing = [c for c in REQUIRED if c not in raw.columns]
    if missing:
        raise ValueError(f"raw data is missing required fields: {missing}")

    df = raw.copy()
    df["timestamp"] = pd.to_datetime(df["tst"], utc=True, errors="coerce", format="ISO8601")
    df["dl"] = pd.to_numeric(df["dl"], errors="coerce")
    df["veh"] = pd.to_numeric(df["veh"], errors="coerce")
    df = df.dropna(subset=["timestamp", "dl", "veh"])
    # The broker can deliver the same message more than once.
    df = df.drop_duplicates(subset=["veh", "timestamp"])
    implausible = df["dl"].abs() > max_abs_delay_s
    if implausible.any():
        log.info("Dropping %d rows with |delay| > %d s (stale journey sign-ins)",
                 implausible.sum(), max_abs_delay_s)
        df = df[~implausible]
    if exclude_lines and "desi" in df.columns:
        df = df[~df["desi"].astype(str).isin(list(exclude_lines))]

    out = df[[c for c in COLUMNS if c in df.columns]].rename(columns=COLUMNS)
    out["timestamp"] = df["timestamp"]
    out["delay_s"] = -df["dl"].astype(int)
    out["vehicle"] = out["vehicle"].astype(int)

    for col in ("line", "route", "direction", "operating_day", "start_time"):
        if col in out:
            out[col] = out[col].astype("string")
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


def run(inputs: list[Path] | None = None, output: Path | None = None) -> pd.DataFrame:
    cfg = load_config()
    paths = cfg["paths"]
    if inputs:
        files = find_raw_files(inputs)
    else:
        files = find_raw_files([resolve_path(paths["raw_dir"])])
        if not files:
            log.info("No collected data in %s; using the bundled sample", paths["raw_dir"])
            files = find_raw_files([resolve_path(paths["sample_file"])])
    if not files:
        raise FileNotFoundError("no raw JSONL files found")

    log.info("Loading %d file(s)", len(files))
    raw = load_raw(files)
    df = clean_with_config(raw)
    log.info("%d raw rows -> %d clean rows", len(raw), len(df))

    output = output or resolve_path(paths["processed_file"])
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output, index=False)
    log.info("Wrote %s", output)
    for key, value in summarize(df).items():
        log.info("  %-15s %s", key, value)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, nargs="*", help="JSONL files or directories")
    parser.add_argument("--output", type=Path, help="Parquet output path")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run(args.input, args.output)


if __name__ == "__main__":
    main()
