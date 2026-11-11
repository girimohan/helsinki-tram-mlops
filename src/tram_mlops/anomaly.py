"""Anomaly detection: trams stuck en route, sudden delay jumps, and incidents (roadmap E15).

Delays are forecast by the model; this module answers a different question: is something
unusual happening right now? Events are rare and unlabelled, so it uses robust statistics
from the collected history rather than a trained model:

- **stopped**: a tram en route has stood still (speed < ``still_speed_mps``) for at least
  ``stopped_min_s``. In normal service 99.9% of such stops are under 3 minutes. Stops at
  a route's termini don't count: trams wait there by design.
- **delay jump**: a tram's delay grew over the last ``jump_window_s`` by more than its line
  usually allows at that hour: median + ``jump_z`` robust standard deviations (1.4826 x MAD),
  and at least ``jump_min_s``.
- **incident**: anomalous trams within ``cluster_radius_m`` of each other, named after the
  nearest tram stop. A blockage affects the few trams near it, so a line-wide average
  hides it; grouping by place shows it.

Run modes (``python -m tram_mlops.anomaly ...``):

- ``baseline``: per line x hour delay-change statistics and termini positions from the
  last ``baseline_days`` of processed data, saved to ``anomalies_dir/baseline.json``
- ``replay``: run the detector over the collected history, each day with a baseline built
  from the *other* days, and write ``reports/anomalies/<run>.json`` and ``latest.md``

Live detection runs inside the forecaster (``Detector.step``) every cycle and writes
``anomalies_dir/latest.json`` (active events and incidents) and, as events end,
``anomalies_dir/events_YYYY-MM-DD.jsonl.gz``.
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import os
import sys
import time
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from tram_mlops import gtfsrt, storage
from tram_mlops.config import PROJECT_ROOT, load_config, resolve_path
from tram_mlops.features import _lookup, add_features
from tram_mlops.network import Network, distance_m

log = logging.getLogger(__name__)
BASELINE = "baseline.json"
LATEST = "latest.json"
MAD_TO_SD = 1.4826
POSITION_COLUMNS = ["vehicle", "line", "route", "direction", "operating_day", "start_time", "speed_mps", "lat",
                    "lon", "doors_open", "stop", "timestamp", "delay_s", "local_time", "hour", "weekday",
                    "at_stop", "on_time", "journey_id"]


@dataclass(frozen=True)
class Settings:
    still_speed_mps: float = 0.5
    stopped_min_s: float = 300
    max_gap_s: float = 60
    jump_window_s: float = 600
    jump_tolerance_s: float = 60
    jump_min_s: float = 300
    jump_z: float = 6.0
    cluster_radius_m: float = 500
    terminus_radius_m: float = 150
    close_after_s: float = 120
    max_age_s: float = 120
    baseline_days: int = 14
    baseline_min_rows: int = 200
    baseline_refresh_h: float = 24

    @classmethod
    def from_config(cls) -> Settings:
        return cls(**load_config()["anomaly"])


# --- baseline --------------------------------------------------------------------------

@dataclass
class Baseline:
    """What normal looks like: delay change per line and hour, and where routes start and end."""
    change: dict[str, list[float]] = field(default_factory=dict)  # "line|hour" or "line" -> [median, mad, rows]
    default: list[float] = field(default_factory=lambda: [10.0, 47.0, 0.0])  # all lines, from 10-11 Nov
    termini: dict[str, list[list[float]]] = field(default_factory=dict)  # "route|direction" -> [[lat, lon], ...]
    built_at: str = ""
    days: list[str] = field(default_factory=list)

    def threshold(self, line: str, hour: int, s: Settings) -> float:
        """Delay gain over ``jump_window_s`` that counts as a jump for this line and hour."""
        median, mad, _ = self.change.get(f"{line}|{hour}") or self.change.get(line) or self.default
        return max(s.jump_min_s, median + s.jump_z * MAD_TO_SD * max(mad, 5.0))

    def at_terminus(self, route: str, direction: str, lat: float, lon: float, radius_m: float) -> bool:
        return any(distance_m(lat, lon, p[0], p[1]) <= radius_m
                   for p in self.termini.get(f"{route}|{direction}", []))

    @classmethod
    def load(cls, path: Path) -> Baseline:
        return cls(**json.loads(path.read_text(encoding="utf-8")))

    def save(self, path: Path) -> None:
        storage.write_json_atomic(path, asdict(self))


def delay_change(feats: pd.DataFrame, s: Settings) -> pd.Series:
    """Each row's delay minus the same trip's delay ``jump_window_s`` earlier (NaN without history)."""
    return feats["delay_s"] - _lookup(feats, -s.jump_window_s, s.jump_tolerance_s, "nearest")


def baseline_sample(feats: pd.DataFrame, s: Settings, every: int = 6) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One day's contribution: en-route delay changes (every ``every``-th row; rows 10 s apart are
    near-duplicates) and each journey's first and last en-route position."""
    en = feats[feats["journey_elapsed_s"] >= 0]
    change = delay_change(en, s)
    sample = pd.DataFrame({"line": en["line"].astype(str), "hour": en["hour"].astype(int),
                           "change": change.astype("float32")}).dropna().iloc[::every]
    ordered = en.sort_values("timestamp")
    ends = pd.concat([ordered.groupby(["vehicle", "journey_id"]).head(1),
                      ordered.groupby(["vehicle", "journey_id"]).tail(1)])
    points = pd.DataFrame({"key": ends["route"].astype(str) + "|" + ends["direction"].astype(str),
                           "lat": ends["lat"], "lon": ends["lon"]}).dropna()
    return sample, points


def _termini(points: pd.DataFrame, min_trips: int = 2, merge_m: float = 100) -> dict[str, list[list[float]]]:
    """Places where trips of a route and direction usually start or end: positions rounded to
    about 100 m, kept where at least ``min_trips`` journeys start or end."""
    out: dict[str, list[list[float]]] = {}
    for key, group in points.groupby("key"):
        cells = group.assign(cell_lat=(group["lat"] / 0.001).round(), cell_lon=(group["lon"] / 0.002).round())
        counts = cells.groupby(["cell_lat", "cell_lon"]).agg(n=("lat", "size"), lat=("lat", "median"),
                                                            lon=("lon", "median"))
        kept: list[list[float]] = []
        for row in counts[counts["n"] >= min_trips].sort_values("n", ascending=False).to_dict("records"):
            lat, lon = float(row["lat"]), float(row["lon"])
            if all(distance_m(lat, lon, p[0], p[1]) > merge_m for p in kept):
                kept.append([round(lat, 6), round(lon, 6)])
        if kept:
            out[str(key)] = kept
    return out


def combine_baseline(samples: Iterable[pd.DataFrame], points: Iterable[pd.DataFrame], s: Settings,
                     days: list[str]) -> Baseline:
    sample = pd.concat(list(samples), ignore_index=True)
    pts = pd.concat(list(points), ignore_index=True)
    stats: dict[str, list[float]] = {}

    def robust(values: pd.Series) -> list[float]:
        median = float(values.median())
        return [round(median, 1), round(float((values - median).abs().median()), 1), float(len(values))]

    if not sample.empty:
        for line, group in sample.groupby("line"):
            if len(group) >= s.baseline_min_rows:
                stats[str(line)] = robust(group["change"])
            for hour, cell in group.groupby("hour"):
                if len(cell) >= s.baseline_min_rows:
                    stats[f"{line}|{hour}"] = robust(cell["change"])
    default = robust(sample["change"]) if len(sample) >= s.baseline_min_rows else Baseline().default
    return Baseline(change=stats, default=default, termini=_termini(pts) if not pts.empty else {},
                    built_at=datetime.now(UTC).isoformat(timespec="seconds"), days=days)


def day_features(root: Path, day: date) -> pd.DataFrame:
    df = storage.read_positions(root, day, day, columns=POSITION_COLUMNS)
    return add_features(df) if not df.empty else df


def build_baseline(root: Path, s: Settings, days: list[date] | None = None) -> Baseline:
    days = days if days is not None else storage.list_partitions(root)[-s.baseline_days:]
    samples, points = [], []
    for day in days:
        feats = day_features(root, day)
        if feats.empty:
            continue
        sample, pts = baseline_sample(feats, s)
        samples.append(sample)
        points.append(pts)
        del feats
    if not samples:
        return Baseline()
    return combine_baseline(samples, points, s, [d.isoformat() for d in days])


# --- detection -------------------------------------------------------------------------

def _gaps_s(timestamps: pd.Series) -> pd.Series:
    """Seconds since the previous row (infinite for the first)."""
    return (timestamps.astype("int64").diff() / 1e9).fillna(np.inf)


def still_runs(feats: pd.DataFrame, s: Settings) -> pd.DataFrame:
    """Spells of standing still en route, one row each, with where and how long."""
    en = feats[feats["journey_elapsed_s"] >= 0].sort_values(["vehicle", "timestamp"])
    if en.empty:
        return pd.DataFrame()
    still = (en["speed_mps"] < s.still_speed_mps).fillna(False)
    trip = en["vehicle"].astype(str) + "#" + en["journey_id"].astype(str)
    gap = _gaps_s(en["timestamp"])
    run = ((still != still.shift()) | (trip != trip.shift()) | (gap > s.max_gap_s)).cumsum()
    last_of_trip = en["timestamp"] == en.groupby(trip)["timestamp"].transform("max")
    rows = en.assign(run=run, trip=trip, last_of_trip=last_of_trip)[still]
    if rows.empty:
        return pd.DataFrame()
    runs = rows.groupby("run").agg(
        vehicle=("vehicle", "first"), journey_id=("journey_id", "first"), line=("line", "first"),
        route=("route", "first"), direction=("direction", "first"), hour=("hour", "first"),
        start=("timestamp", "min"), end=("timestamp", "max"), lat=("lat", "median"), lon=("lon", "median"),
        at_stop=("at_stop", "any"), doors_open=("doors_open", "max"), delay_start=("delay_s", "first"),
        delay_end=("delay_s", "last"), ongoing=("last_of_trip", "any"))
    runs["duration_s"] = (runs["end"] - runs["start"]).dt.total_seconds()
    return runs.reset_index(drop=True)


def _event(kind: str, row: Any, start: pd.Timestamp, end: pd.Timestamp, gain: float) -> dict[str, Any]:
    return {
        "kinds": [kind], "vehicle": int(row["vehicle"]), "line": str(row["line"]),
        "journey_id": str(row["journey_id"]), "start": start, "end": end,
        "lat": round(float(row["lat"]), 6), "lon": round(float(row["lon"]), 6),
        "delay_s": int(row["delay_end"]), "delay_gain_s": int(round(gain)),
        "at_stop": bool(row.get("at_stop", False)), "doors_open": bool(row.get("doors_open", 0) == 1),
    }


def stopped_events(runs: pd.DataFrame, baseline: Baseline, s: Settings, ongoing_only: bool) -> list[dict[str, Any]]:
    if runs.empty:
        return []
    long = runs[runs["duration_s"] >= s.stopped_min_s]
    if ongoing_only:
        long = long[long["ongoing"]]
    events = []
    for row in long.to_dict("records"):
        if baseline.at_terminus(str(row["route"]), str(row["direction"]), row["lat"], row["lon"],
                                s.terminus_radius_m):
            continue
        events.append(_event("stopped", row, row["start"], row["end"], row["delay_end"] - row["delay_start"]))
    return events


def jump_rows(feats: pd.DataFrame, baseline: Baseline, s: Settings) -> pd.DataFrame:
    """En-route rows whose delay gain over ``jump_window_s`` exceeds the line's threshold."""
    en = feats[feats["journey_elapsed_s"] >= 0]
    if en.empty:
        return en
    change = delay_change(en, s)
    cache: dict[tuple[str, int], float] = {}
    limits = []
    for line, hour in zip(en["line"].astype(str), en["hour"].astype(int), strict=True):
        key = (line, hour)
        if key not in cache:
            cache[key] = baseline.threshold(line, hour, s)
        limits.append(cache[key])
    # Standing still for the whole window adds at most the window to the delay. A bigger gain is the
    # delay value itself jumping (e.g. a journey re-timed in HFP), a data glitch, not an incident.
    plausible = change <= s.jump_window_s + s.jump_tolerance_s
    flagged = (change > np.asarray(limits)) & plausible
    out = en[flagged.to_numpy()].copy()
    # A tram waiting at a terminus gains delay by leaving late; that is not an incident.
    at_end = [baseline.at_terminus(str(r), str(d), la, lo, s.terminus_radius_m)
              for r, d, la, lo in zip(out["route"], out["direction"], out["lat"], out["lon"], strict=True)]
    out = out[~np.asarray(at_end, dtype=bool)] if len(out) else out
    out["delay_gain_s"] = change.reindex(out.index)
    out.attrs["glitches"] = int(((change > np.asarray(limits)) & ~plausible).sum())
    return out


def jump_events(feats: pd.DataFrame, baseline: Baseline, s: Settings, latest_only: bool) -> list[dict[str, Any]]:
    rows = jump_rows(feats, baseline, s)
    if rows.empty:
        return []
    rows = rows.sort_values(["vehicle", "timestamp"])
    if latest_only:
        newest = feats.groupby("vehicle")["timestamp"].max()
        rows = rows[rows["timestamp"].to_numpy() == newest.reindex(rows["vehicle"]).to_numpy()]
        return [_event("delay_jump", {**r, "delay_end": r["delay_s"]}, r["timestamp"], r["timestamp"],
                       r["delay_gain_s"]) for r in rows.to_dict("records")]
    trip = rows["vehicle"].astype(str) + "#" + rows["journey_id"].astype(str)
    gap = _gaps_s(rows["timestamp"])
    spell = ((trip != trip.shift()) | (gap > 2 * s.max_gap_s)).cumsum()
    events = []
    for _, group in rows.groupby(spell):
        worst = group.loc[group["delay_gain_s"].idxmax()]
        row = {**worst.to_dict(), "lat": group["lat"].median(), "lon": group["lon"].median(),
               "delay_end": group["delay_s"].iloc[-1]}
        events.append(_event("delay_jump", row, group["timestamp"].min(), group["timestamp"].max(),
                             group["delay_gain_s"].max()))
    return events


def merge_by_trip(events: list[dict[str, Any]], max_gap_s: float = 300) -> list[dict[str, Any]]:
    """One event per trip spell: a stopped tram usually also shows a delay jump."""
    merged: list[dict[str, Any]] = []
    for ev in sorted(events, key=lambda e: (e["vehicle"], e["journey_id"], e["start"])):
        last = merged[-1] if merged else None
        if (last and last["vehicle"] == ev["vehicle"] and last["journey_id"] == ev["journey_id"]
                and ev["start"] <= last["end"] + pd.Timedelta(seconds=max_gap_s)):
            last["kinds"] = sorted(set(last["kinds"]) | set(ev["kinds"]))
            last["end"] = max(last["end"], ev["end"])
            last["delay_gain_s"] = max(last["delay_gain_s"], ev["delay_gain_s"])
            last["delay_s"] = max(last["delay_s"], ev["delay_s"])
            last["at_stop"] = last["at_stop"] or ev["at_stop"]
            last["doors_open"] = last["doors_open"] or ev["doors_open"]
        else:
            merged.append(dict(ev))
    return merged


def incidents(events: list[dict[str, Any]], s: Settings, network: Network | None,
              same_time: bool = False) -> list[dict[str, Any]]:
    """Group events that are close in space (and, with ``same_time``, overlap in time)."""
    n = len(events)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        for j in range(i + 1, n):
            a, b = events[i], events[j]
            if same_time and (a["start"] > b["end"] or b["start"] > a["end"]):
                continue
            if distance_m(a["lat"], a["lon"], b["lat"], b["lon"]) <= s.cluster_radius_m:
                parent[find(i)] = find(j)
    groups: dict[int, list[dict[str, Any]]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(events[i])
    out = []
    for members in groups.values():
        lat = float(np.mean([e["lat"] for e in members]))
        lon = float(np.mean([e["lon"] for e in members]))
        near = network.nearest_stop(lat, lon) if network else None
        out.append({
            "trams": len({e["vehicle"] for e in members}),
            "lines": sorted({e["line"] for e in members}, key=lambda x: (len(x), x)),
            "kinds": sorted({k for e in members for k in e["kinds"]}),
            "near": near[0] if near else None,
            "lat": round(lat, 6), "lon": round(lon, 6),
            "start": min(e["start"] for e in members), "end": max(e["end"] for e in members),
            "max_delay_gain_s": max(e["delay_gain_s"] for e in members),
            "vehicles": sorted({e["vehicle"] for e in members}),
        })
    return sorted(out, key=lambda x: (-x["trams"], -x["max_delay_gain_s"]))


def _jsonable(record: dict[str, Any]) -> dict[str, Any]:
    return {k: (v.isoformat(timespec="seconds") if isinstance(v, pd.Timestamp) else v) for k, v in record.items()}


class Detector:
    """Live detection, one step per forecaster cycle. Events stay open while their condition
    holds and close ``close_after_s`` after it last held; closed events are logged."""

    def __init__(self, out_dir: Path, network_path: Path | None, s: Settings):
        self.out_dir, self.network_path, self.s = out_dir, network_path, s
        self.open: dict[tuple[int, str], dict[str, Any]] = {}
        self._baseline: tuple[float, Baseline] = (-1.0, Baseline())
        self._network: tuple[float, Network | None] = (-1.0, None)

    def baseline(self) -> Baseline:
        path = self.out_dir / BASELINE
        mtime = path.stat().st_mtime if path.exists() else 0.0
        if mtime != self._baseline[0]:
            self._baseline = (mtime, Baseline.load(path) if mtime else Baseline())
        return self._baseline[1]

    def network(self) -> Network | None:
        if self.network_path is None or not self.network_path.exists():
            return None
        mtime = self.network_path.stat().st_mtime
        if mtime != self._network[0]:
            self._network = (mtime, Network.load(self.network_path))
        return self._network[1]

    def step(self, feats: pd.DataFrame, now: pd.Timestamp) -> dict[str, int]:
        baseline, network = self.baseline(), self.network()
        fresh = feats[feats.groupby("vehicle")["timestamp"].transform("max")
                      >= now - pd.Timedelta(seconds=self.s.max_age_s)]
        current = merge_by_trip(stopped_events(still_runs(fresh, self.s), baseline, self.s, ongoing_only=True)
                                + jump_events(fresh, baseline, self.s, latest_only=True))
        for ev in current:
            key = (ev["vehicle"], ev["journey_id"])
            if key in self.open:
                known = self.open[key]
                known.update(end=ev["end"], lat=ev["lat"], lon=ev["lon"], delay_s=ev["delay_s"],
                             kinds=sorted(set(known["kinds"]) | set(ev["kinds"])),
                             delay_gain_s=max(known["delay_gain_s"], ev["delay_gain_s"]))
                known["last_seen"] = now
            else:
                near = network.nearest_stop(ev["lat"], ev["lon"]) if network else None
                self.open[key] = {**ev, "detected_at": now, "last_seen": now, "near": near[0] if near else None}
        closed = [self.open.pop(k) for k, ev in list(self.open.items())
                  if ev["last_seen"] < now - pd.Timedelta(seconds=self.s.close_after_s)]
        if closed:
            storage.append_gzip_jsonl(self.out_dir / f"events_{now:%Y-%m-%d}.jsonl.gz",
                                      [_jsonable(ev) for ev in closed])
        active = [ev for ev in self.open.values() if ev["last_seen"] == now]
        storage.write_json_atomic(self.out_dir / LATEST, {
            "issued_at": now.isoformat(timespec="seconds"),
            "active": [_jsonable(ev) for ev in active],
            "incidents": [_jsonable(i) for i in incidents(active, self.s, network)],
        })
        return {"active": len(active), "closed": len(closed)}


def read_latest(out_dir: Path) -> dict[str, Any] | None:
    try:
        return json.loads((out_dir / LATEST).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def read_events(out_dir: Path, start: date, end: date) -> pd.DataFrame:
    """Logged (ended) events of the UTC days ``start``..``end``."""
    records: list[dict[str, Any]] = []
    day = start
    while day <= end:
        path = out_dir / f"events_{day:%Y-%m-%d}.jsonl.gz"
        if path.exists():
            records.extend(storage.read_gzip_jsonl(path))
        day += timedelta(days=1)
    df = pd.DataFrame.from_records(records)
    for col in ("start", "end", "detected_at", "last_seen"):
        if col in df:
            df[col] = pd.to_datetime(df[col], utc=True, format="ISO8601")
    return df


# --- replay and evaluation -------------------------------------------------------------

def detect_day(feats: pd.DataFrame, baseline: Baseline, s: Settings,
               network: Network | None) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """All events of a day of history, and the incidents they form."""
    events = merge_by_trip(stopped_events(still_runs(feats, s), baseline, s, ongoing_only=False)
                           + jump_events(feats, baseline, s, latest_only=False))
    return events, incidents(events, s, network, same_time=True)


def reactive_alerts(gtfsrt_dir: Path, network: Network | None, lead_s: float = 900) -> list[dict[str, Any]]:
    """Weak labels: tram alerts first seen within ``lead_s`` of their start (planned works are
    announced long before), with an effect other than NO_EFFECT."""
    routes = {r for r in (network.routes if network else {})}
    first: dict[str, dict[str, Any]] = {}
    for path in sorted(gtfsrt_dir.glob("alerts_*.jsonl.gz")):
        try:
            with gzip.open(path, "rt", encoding="utf-8") as f:
                for line in f:
                    try:
                        snapshot = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    for alert in snapshot.get("alerts", []):
                        first.setdefault(alert["id"], {"seen": snapshot["fetched_at"], "alert": alert})
        except (EOFError, OSError):
            continue
    labels = []
    for item in first.values():
        alert, seen = item["alert"], pd.Timestamp(item["seen"]).tz_convert("UTC")
        lines = sorted({network.line_of(e["route_id"]) or "" for e in alert.get("informed_entity", [])
                        if network and e.get("route_id") and (e["route_id"] in routes
                                                              or e["route_id"][:4] in routes)} - {""})
        if not lines or alert.get("effect") == "NO_EFFECT":
            continue
        for period in alert.get("active_period", [{}]) or [{}]:
            start = pd.Timestamp(int(period["start"]), unit="s", tz="UTC") if period.get("start") else seen
            end = pd.Timestamp(int(period["end"]), unit="s", tz="UTC") if period.get("end") else seen + pd.Timedelta(
                hours=1)
            if abs((seen - start).total_seconds()) <= lead_s:
                text = next((t.get("text", "") for t in alert.get("header_text", {}).get("translation", [])
                             if t.get("language") == "en"), "")
                labels.append({"id": alert["id"], "lines": lines, "start": start, "end": end, "seen": seen,
                               "effect": alert.get("effect"), "header": text[:120]})
    return labels


def evaluate(incident_list: list[dict[str, Any]], labels: list[dict[str, Any]],
             before_s: float = 1800) -> dict[str, Any]:
    """Recall of reactive alerts: an alert counts as caught if an incident on one of its lines
    overlaps from ``before_s`` before its start to its end."""
    caught = []
    for label in labels:
        hits = [i for i in incident_list if set(i["lines"]) & set(label["lines"])
                and i["end"] >= label["start"] - pd.Timedelta(seconds=before_s) and i["start"] <= label["end"]]
        lead = None
        if hits:
            first_hit = min(h["start"] for h in hits)
            lead = round((label["seen"] - first_hit).total_seconds() / 60, 1)
        caught.append({**_jsonable(label), "caught": bool(hits), "lead_min": lead})
    return {"alerts": len(labels), "caught": sum(c["caught"] for c in caught), "details": caught}


def replay(root: Path, gtfsrt_dir: Path, s: Settings, out_dir: Path | None = None) -> dict[str, Any]:
    days = storage.list_partitions(root)
    if not days:
        raise ValueError(f"no processed data in {root}")
    network_path = gtfsrt_dir / gtfsrt.NETWORK
    network = Network.load(network_path) if network_path.exists() else None
    per_day_feats = {}
    samples, points = {}, {}
    for day in days:
        feats = day_features(root, day)
        if feats.empty:
            continue
        per_day_feats[day] = feats
        samples[day], points[day] = baseline_sample(feats, s)

    labels = reactive_alerts(gtfsrt_dir, network)
    report_days, all_incidents = [], []
    for day, feats in per_day_feats.items():
        others = [d for d in per_day_feats if d != day]
        # Each day is judged against what the other days say is normal, not against itself.
        baseline = (combine_baseline([samples[d] for d in others], [points[d] for d in others], s,
                                     [d.isoformat() for d in others]) if others else Baseline())
        events, found = detect_day(feats, baseline, s, network)
        all_incidents.extend(found)
        en_route_tram_hours = feats[feats["journey_elapsed_s"] >= 0].groupby("vehicle")["timestamp"].agg(
            lambda t: (t.max() - t.min()).total_seconds() / 3600).sum()
        report_days.append({
            "day": day.isoformat(), "events": len(events),
            "stopped": sum("stopped" in e["kinds"] for e in events),
            "delay_jump": sum("delay_jump" in e["kinds"] for e in events),
            "incidents": len(found), "multi_tram_incidents": sum(i["trams"] >= 2 for i in found),
            "tram_hours": round(float(en_route_tram_hours), 1),
        })
        log.info("%s: %d events, %d incidents", day, len(events), len(found))

    covered = [lbl for lbl in labels if any(lbl["start"].date() == d for d in per_day_feats)]
    report = {
        "run": datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ"),
        "settings": asdict(s),
        "days": report_days,
        "top_incidents": [_jsonable(i) for i in sorted(all_incidents, key=lambda i: (-i["trams"],
                                                                                     -i["max_delay_gain_s"]))[:20]],
        "labels": evaluate(all_incidents, covered),
    }
    out_dir = out_dir or PROJECT_ROOT / "reports" / "anomalies"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{report['run']}.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    (out_dir / "latest.md").write_text(render_markdown(report), encoding="utf-8")
    return report


def render_markdown(report: dict[str, Any], tz: str = "Europe/Helsinki") -> str:
    s = report["settings"]
    lines = [
        f"Anomaly replay `{report['run']}`: each day checked against a baseline built from the other days. "
        f"Stopped = standing still en route for {s['stopped_min_s'] / 60:.0f}+ min away from termini; "
        f"delay jump = delay grew more than the line's usual spread allows over {s['jump_window_s'] / 60:.0f} min; "
        f"incident = such trams within {s['cluster_radius_m']:.0f} m at the same time. "
        "Generated by `python -m tram_mlops.anomaly replay`.",
        "",
        "| Day | Tram-hours en route | Events | Stopped | Delay jump | Incidents | 2+ trams |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for d in report["days"]:
        lines.append(f"| {d['day']} | {d['tram_hours']:,.0f} | {d['events']} | {d['stopped']} | {d['delay_jump']} | "
                     f"{d['incidents']} | {d['multi_tram_incidents']} |")
    lines += ["", "Largest incidents:", "", "| Start (Helsinki) | Duration | Trams | Lines | Near | Max delay gain |",
              "|---|---:|---:|---|---|---:|"]
    for i in report["top_incidents"][:10]:
        start, end = pd.Timestamp(i["start"]).tz_convert(tz), pd.Timestamp(i["end"]).tz_convert(tz)
        lines.append(f"| {start:%a %d %b %H:%M} | {(end - start).total_seconds() / 60:.0f} min | {i['trams']} | "
                     f"{', '.join(i['lines'])} | {i['near'] or '?'} | {i['max_delay_gain_s'] / 60:.1f} min |")
    lab = report["labels"]
    lines += ["", f"Reactive HSL tram alerts in the replayed days (weak labels): {lab['alerts']}, "
                  f"caught {lab['caught']}."]
    for d in lab["details"]:
        lines.append(f"- {pd.Timestamp(d['start']).tz_convert(tz):%a %d %b %H:%M}, lines {', '.join(d['lines'])}: "
                     f"{d['header']} — {'caught' if d['caught'] else 'missed'}"
                     + (f", {d['lead_min']} min before HSL's alert" if d["lead_min"] is not None else ""))
    return "\n".join(lines) + "\n"


# --- command line ----------------------------------------------------------------------

def run_baseline(force: bool = False) -> Baseline | None:
    cfg = load_config()
    s = Settings.from_config()
    out_dir = resolve_path(cfg["paths"]["anomalies_dir"])
    path = out_dir / BASELINE
    if not force and path.exists() and time.time() - os.path.getmtime(path) < s.baseline_refresh_h * 3600:
        log.info("Baseline is less than %s h old; skipping (--force to rebuild)", s.baseline_refresh_h)
        return None
    baseline = build_baseline(resolve_path(cfg["paths"]["processed_dir"]), s)
    baseline.save(path)
    log.info("Baseline from %d days: %d line/hour cells, %d routes with termini", len(baseline.days),
             len(baseline.change), len(baseline.termini))
    return baseline


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=["baseline", "replay"])
    parser.add_argument("--force", action="store_true", help="baseline: rebuild even if recent")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.mode == "baseline":
        run_baseline(args.force)
        return
    cfg = load_config()
    report = replay(resolve_path(cfg["paths"]["processed_dir"]), resolve_path(cfg["paths"]["gtfsrt_dir"]),
                    Settings.from_config())
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    print(render_markdown(report))


if __name__ == "__main__":
    main()
