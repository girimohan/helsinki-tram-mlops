"""Live forecasts, logged and scored against what happened (roadmap E6).

Run with ``python -m tram_mlops.forecaster``. Every ``forecast.interval_s`` seconds it:

1. reads the last ``forecast.window_min`` minutes the collector wrote, cleans them and
   builds features exactly as training does
2. forecasts each tram's delay ``model.horizon_s`` ahead from its newest observation and
   writes them to ``latest.json`` (what the dashboard shows)
3. logs at most one forecast per tram every ``forecast.log_every_s`` seconds, and once
   the horizon has passed, scores it: the actual delay is ``features.future_delay``, the
   same definition the model was trained on. Persistence (delay stays as it is) is
   scored on the same rows.

Scored forecasts are appended to ``paths.predictions_dir/forecasts_YYYY-MM-DD.jsonl.gz``
(one gzip member per batch); ``monitoring`` turns them into live accuracy.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from tram_mlops import process, storage
from tram_mlops.config import load_config, resolve_path
from tram_mlops.features import add_features, future_delay
from tram_mlops.train import load_latest, predict_from_features, segment_masks

log = logging.getLogger(__name__)
LATEST = "latest.json"
CLOCK_SLACK = pd.Timedelta(seconds=30)


def _iso(ts: pd.Timestamp) -> str:
    return ts.isoformat(timespec="milliseconds")


@dataclass
class Forecaster:
    raw_dir: Path
    models_dir: Path
    out_dir: Path
    horizon_s: float
    tolerance_s: float
    window_min: float = 15
    log_every_s: float = 30
    max_age_s: float = 120
    model: Any = None
    model_version: str | None = None
    _model_mtime: float = -1.0
    # Logged forecasts waiting for their horizon to pass, keyed by (vehicle, observed_at).
    pending: dict[tuple[int, pd.Timestamp], dict[str, Any]] = field(default_factory=dict)
    _last_logged: dict[int, pd.Timestamp] = field(default_factory=dict)
    stats: dict[str, int] = field(default_factory=lambda: {"forecasts": 0, "scored": 0, "unmatched": 0, "expired": 0})

    def reload_model(self) -> None:
        pointer = self.models_dir / "latest.json"
        mtime = pointer.stat().st_mtime if pointer.exists() else 0.0
        if mtime == self._model_mtime:
            return
        loaded = load_latest(self.models_dir)
        self._model_mtime = mtime
        if loaded is None:
            self.model, self.model_version = None, None
            log.warning("No trained model in %s; forecasts paused", self.models_dir)
            return
        self.model, metrics = loaded
        self.model_version = metrics["version"]
        if metrics.get("horizon_s") not in (None, self.horizon_s):
            log.warning("Model %s was trained for %ss ahead, not %ss", self.model_version,
                        metrics["horizon_s"], self.horizon_s)
        log.info("Using model %s", self.model_version)

    def recent_features(self, now: pd.Timestamp) -> pd.DataFrame:
        raw = process.load_recent(self.raw_dir, pd.Timedelta(minutes=self.window_min), now=now)
        df = process.clean_with_config(raw)
        if df.empty:
            return df
        # Never use data from after `now` (a little slack for vehicle clocks running ahead).
        df = df[df["timestamp"] <= now + CLOCK_SLACK].reset_index(drop=True)
        return add_features(df) if not df.empty else df

    def step(self, now: pd.Timestamp | None = None) -> dict[str, int]:
        """One cycle: forecast, log, score. Returns counts for this cycle."""
        now = now or pd.Timestamp.now(tz="UTC")
        self.reload_model()
        feats = self.recent_features(now)
        counts = {"forecasts": 0, "logged": 0, "scored": 0}
        if feats.empty:
            return counts
        counts["scored"] = len(self._score(feats))
        if self.model is not None:
            latest = feats.sort_values("timestamp").groupby("vehicle").tail(1)
            latest = latest[latest["timestamp"] >= now - pd.Timedelta(seconds=self.max_age_s)]
            if not latest.empty:
                rows = self._forecast_rows(latest)
                self._write_latest(rows, now)
                counts["forecasts"] = len(rows)
                counts["logged"] = self._log(rows, now)
        self.stats["forecasts"] += counts["forecasts"]
        return counts

    def _forecast_rows(self, latest: pd.DataFrame) -> list[dict[str, Any]]:
        predicted = predict_from_features(self.model, latest).to_numpy()
        en_route = segment_masks(latest)["en_route"].to_numpy()
        return [
            {"vehicle": int(v), "observed_at": t, "journey_id": j, "line": line, "direction": d,
             "delay_now": int(delay), "predicted": round(float(p), 1),
             "segment": "en_route" if moving else "before_departure"}
            for v, t, j, line, d, delay, p, moving in zip(
                latest["vehicle"], latest["timestamp"], latest["journey_id"], latest["line"],
                latest["direction"], latest["delay_s"], predicted, en_route, strict=True)
        ]

    def _log(self, rows: list[dict[str, Any]], now: pd.Timestamp) -> int:
        logged = 0
        for row in rows:
            vehicle, observed_at = row["vehicle"], row["observed_at"]
            last = self._last_logged.get(vehicle)
            if last is not None and observed_at - last < pd.Timedelta(seconds=self.log_every_s):
                continue
            self._last_logged[vehicle] = observed_at
            self.pending[(vehicle, observed_at)] = {
                "vehicle": vehicle,
                "trip": f"{vehicle}#{row['journey_id']}",
                "line": row["line"],
                "direction": row["direction"],
                "observed_at": _iso(observed_at),
                "issued_at": _iso(now),
                "horizon_s": self.horizon_s,
                "delay_now": row["delay_now"],
                "predicted": row["predicted"],
                "segment": row["segment"],
                "model_version": self.model_version,
            }
            logged += 1
        return logged

    def _score(self, feats: pd.DataFrame) -> list[dict[str, Any]]:
        """Score pending forecasts whose horizon (plus tolerance) is covered by the data."""
        if not self.pending:
            return []
        newest, oldest = feats["timestamp"].max(), feats["timestamp"].min()
        actual = future_delay(feats, self.horizon_s, self.tolerance_s)
        answers = dict(zip(zip(feats["vehicle"], feats["timestamp"], strict=True), actual, strict=True))
        ready = [k for k in self.pending
                 if k[1] + pd.Timedelta(seconds=self.horizon_s + self.tolerance_s) <= newest]
        lines = []
        for key in ready:
            record = self.pending.pop(key)
            if key[1] < oldest:
                # The service stalled past the window: the answer can't be looked up any more.
                self.stats["expired"] += 1
                continue
            value = answers.get(key)
            matched = value is not None and bool(pd.notna(value))
            record.update(actual=int(value) if value is not None and matched else None, scored_at=_iso(newest))
            self.stats["scored" if matched else "unmatched"] += 1
            lines.append(record)
        if lines:
            # Files are named by the UTC day of the data, like the raw files.
            storage.append_gzip_jsonl(self.out_dir / f"forecasts_{newest:%Y-%m-%d}.jsonl.gz", lines)
        return lines

    def _write_latest(self, rows: list[dict[str, Any]], now: pd.Timestamp) -> None:
        snapshot = {
            "issued_at": _iso(now),
            "model_version": self.model_version,
            "horizon_s": self.horizon_s,
            "forecasts": [
                {k: (_iso(v) if k == "observed_at" else v) for k, v in row.items() if k != "journey_id"}
                for row in rows
            ],
        }
        write_json_atomic(self.out_dir / LATEST, snapshot)

    def run(self, interval_s: float, duration_s: float | None = None) -> None:
        started = last_log = time.monotonic()
        while duration_s is None or time.monotonic() - started < duration_s:
            cycle = time.monotonic()
            try:
                self.step()
            except Exception:  # a bad cycle must not stop the service
                log.exception("Forecast cycle failed")
            if cycle - last_log >= 600:
                log.info("Forecasts made %(forecasts)d, scored %(scored)d, unmatched %(unmatched)d", self.stats)
                last_log = cycle
            time.sleep(max(0.0, interval_s - (time.monotonic() - cycle)))


def write_json_atomic(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".latest-", suffix=".json")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(data, f)
    os.chmod(tmp, storage.FILE_MODE)
    os.replace(tmp, path)


def read_latest(out_dir: Path) -> dict[str, Any] | None:
    """The forecaster's current forecasts, or None if it hasn't written any."""
    path = out_dir / LATEST
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def from_config() -> tuple[Forecaster, float]:
    cfg = load_config()
    paths, model, fc = cfg["paths"], cfg["model"], cfg["forecast"]
    forecaster = Forecaster(
        raw_dir=resolve_path(paths["raw_dir"]),
        models_dir=resolve_path(paths["models_dir"]),
        out_dir=resolve_path(paths["predictions_dir"]),
        horizon_s=model["horizon_s"],
        tolerance_s=model["horizon_tolerance_s"],
        window_min=fc["window_min"],
        log_every_s=fc["log_every_s"],
        max_age_s=fc["max_age_s"],
    )
    return forecaster, fc["interval_s"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--duration", type=float, help="stop after this many seconds (default: run forever)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    forecaster, interval_s = from_config()
    forecaster.run(interval_s, args.duration)


if __name__ == "__main__":
    main()
