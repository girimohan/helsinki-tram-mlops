"""Cut a continuous time slice from a collected raw day into the bundled sample.

    python scripts/make_sample.py data/raw/tram_2026-11-10.jsonl \
        --start 2026-11-10T05:30:00Z --minutes 45

Keeps every record whose ``tst`` falls in [start, start + minutes), in the order the
collector wrote them, so the sample has the same quirks as live data (duplicates,
stale sign-ins). The slice must be continuous and longer than the forecast horizon,
or ``train`` finds no 5-minute-ahead targets. Uses only the standard library so it
can run on the server next to the raw files. The output is gzipped with a fixed
mtime, so re-running on the same input gives byte-identical files.
"""

from __future__ import annotations

import argparse
import gzip
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "data" / "sample" / "tram_sample.jsonl.gz"


def _parse_ts(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC)


def _open(path: Path):
    return gzip.open(path, "rt", encoding="utf-8") if path.suffix == ".gz" else open(path, encoding="utf-8")


def cut(source: Path, output: Path, start: datetime, minutes: float) -> dict:
    end = start + timedelta(minutes=minutes)
    kept, vehicles = 0, set()
    output.parent.mkdir(parents=True, exist_ok=True)
    with _open(source) as src, open(output, "wb") as raw_out, \
            gzip.GzipFile(filename="", mode="wb", fileobj=raw_out, mtime=0, compresslevel=9) as out:
        for line in src:
            try:
                tst = _parse_ts(json.loads(line)["tst"])
            except (ValueError, KeyError, TypeError):
                continue
            if start <= tst < end:
                out.write(line.encode("utf-8") if line.endswith("\n") else (line + "\n").encode("utf-8"))
                kept += 1
                vehicles.add(json.loads(line).get("veh"))
    return {"records": kept, "vehicles": len(vehicles), "bytes": output.stat().st_size}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", type=Path, help="raw daily file (.jsonl or .jsonl.gz)")
    parser.add_argument("--start", required=True, help="slice start, ISO 8601 (e.g. 2026-11-10T05:30:00Z)")
    parser.add_argument("--minutes", type=float, default=45)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    stats = cut(args.source, args.output, _parse_ts(args.start), args.minutes)
    if not stats["records"]:
        raise SystemExit(f"no records between {args.start} and +{args.minutes:g} min in {args.source}")
    print(f"{args.output}: {stats['records']:,} records, {stats['vehicles']} trams, "
          f"{stats['bytes'] / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
