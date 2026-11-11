"""Live accuracy from the forecaster's scored forecasts (roadmap E6).

``python -m tram_mlops.monitoring`` prints the rolling metrics. The dashboard uses
``live_metrics`` and ``mae_over_time``. Every forecast is compared with persistence
(the delay stays as it is) on the same rows, so the two are always comparable.
"""

from __future__ import annotations

import argparse
import json
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from tram_mlops import storage
from tram_mlops.config import load_config, resolve_path

WINDOWS = {"1h": pd.Timedelta(hours=1), "24h": pd.Timedelta(hours=24)}


def read_scored(pred_dir: Path, start: date, end: date) -> pd.DataFrame:
    """Scored forecasts logged on the UTC days ``start``..``end``."""
    records: list[dict[str, Any]] = []
    day = start
    while day <= end:
        path = pred_dir / f"forecasts_{day:%Y-%m-%d}.jsonl.gz"
        if path.exists():
            records.extend(storage.read_gzip_jsonl(path))
        day += timedelta(days=1)
    df = pd.DataFrame.from_records(records)
    if df.empty:
        return df
    for col in ("observed_at", "issued_at", "scored_at"):
        df[col] = pd.to_datetime(df[col], utc=True, format="ISO8601")
    return df


def _scores(df: pd.DataFrame) -> dict[str, Any]:
    matched = df.dropna(subset=["actual"])
    if matched.empty:
        return {"rows": 0}
    model = (matched["predicted"] - matched["actual"]).abs().mean()
    persistence = (matched["delay_now"] - matched["actual"]).abs().mean()
    return {
        "rows": len(matched),
        "model_mae_s": round(float(model), 1),
        "persistence_mae_s": round(float(persistence), 1),
        "improvement_pct": round(100 * (1 - model / persistence), 1) if persistence else None,
    }


def live_metrics(scored: pd.DataFrame, now: pd.Timestamp | None = None) -> dict[str, Any]:
    """Rolling MAE of the model and persistence, overall and by segment, per window."""
    now = now or pd.Timestamp.now(tz="UTC")
    out: dict[str, Any] = {"as_of": now.isoformat(timespec="seconds")}
    for name, width in WINDOWS.items():
        window = scored[scored["observed_at"] >= now - width] if not scored.empty else scored
        entry: dict[str, Any] = {"forecasts": len(window)}
        if window.empty:
            out[name] = entry
            continue
        # Unmatched forecasts: the trip ended or the data had a gap; no "true answer" exists.
        entry["matched_pct"] = round(100 * float(window["actual"].notna().mean()), 1)
        entry["overall"] = _scores(window)
        entry["segments"] = {seg: _scores(rows) for seg, rows in window.groupby("segment")}
        entry["model_versions"] = sorted(window["model_version"].dropna().unique().tolist())
        out[name] = entry
    return out


def mae_over_time(scored: pd.DataFrame, freq: str = "15min", segment: str | None = "en_route") -> pd.DataFrame:
    """Model and persistence MAE per time bucket (by observation time), for charts."""
    df = scored.dropna(subset=["actual"])
    if segment is not None and not df.empty:
        df = df[df["segment"] == segment]
    if df.empty:
        return pd.DataFrame(columns=["Model", "Persistence"])
    errors = pd.DataFrame({
        "Model": (df["predicted"] - df["actual"]).abs().to_numpy(),
        "Persistence": (df["delay_now"] - df["actual"]).abs().to_numpy(),
    }, index=pd.DatetimeIndex(df["observed_at"]))
    return errors.resample(freq).mean().dropna(how="all").round(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, default=2, help="how many recent UTC days of forecasts to read")
    args = parser.parse_args()
    cfg = load_config()
    today = pd.Timestamp.now(tz="UTC").date()
    scored = read_scored(resolve_path(cfg["paths"]["predictions_dir"]), today - timedelta(days=args.days - 1), today)
    print(json.dumps(live_metrics(scored), indent=2))


if __name__ == "__main__":
    main()
