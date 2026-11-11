"""Read-only lookups the chat assistant calls to answer questions (roadmap E18).

Each tool is a method of ``TramData`` that takes a few simple arguments and returns a
small JSON-ready dict, so an LLM can call it through function calling and quote it.
None of them runs a query the model writes, and none of them changes anything.

- ``next_trams``: upcoming trams at a stop, from HSL's realtime predictions
  (``gtfsrt.CURRENT_TRIPS``), with each tram's delay now when it is being tracked
- ``find_stop``: tram stops matching a name, with the lines that serve them
- ``line_status``: trams in service now per line, or each tram on one line, with the
  model's 5-minute delay forecast
- ``service_alerts``: HSL's service alerts for trams in force now
- ``punctuality``: on-time share and delay over past days, by line, hour or day
- ``forecast_accuracy``: how the delay model is doing against persistence

Stop names come from static GTFS (``gtfsrt.NETWORK``); the realtime feeds use ids only.
"""

from __future__ import annotations

import difflib
import json
import os
import re
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from tram_mlops import alerts, forecaster, gtfsrt, monitoring, process
from tram_mlops.config import load_config, resolve_path
from tram_mlops.history import History, available_days

TRAM_VEHICLE_TYPES = {"0", "900"}
# Older trip predictions than this are flagged as stale (the poller fetches every 30 s).
STALE_TRIPS_S = 180
FORECAST_FRESH_S = 120


# --- names -----------------------------------------------------------------------------

def normalize(text: str) -> str:
    """Lower case, no accents, single spaces: ``"Töölön tulli"`` -> ``"toolon tulli"``."""
    decomposed = unicodedata.normalize("NFKD", text)
    plain = "".join(ch for ch in decomposed if not unicodedata.combining(ch)).casefold()
    return " ".join(re.sub(r"[^0-9a-z]+", " ", plain).split())


def normalize_line(line: str | None) -> str | None:
    """``"Tram 4"``, ``"line 4"``, ``" 4t "`` -> ``"4"``, ``"4"``, ``"4T"``; empty -> None."""
    if line is None:
        return None
    text = re.sub(r"(?i)\b(tram|line|raitiovaunu|linja|no\.?|number)\b", " ", str(line))
    text = text.replace("#", " ").strip().upper()
    return text.replace(" ", "") or None


@dataclass
class StopGroup:
    """All tram platforms sharing one name, e.g. both directions of Lasipalatsi."""
    name: str
    ids: list[str]
    score: float


@dataclass
class Network:
    routes: dict[str, dict[str, str]] = field(default_factory=dict)
    headsigns: dict[str, str] = field(default_factory=dict)
    stops: dict[str, list[Any]] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path) -> Network:
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(data.get("routes", {}), data.get("headsigns", {}), data.get("stops", {}))

    def line_of(self, route: str) -> str | None:
        """Line name of a GTFS route, also for lettered variants static GTFS doesn't list."""
        route_lines = {r: v["line"] for r, v in self.routes.items()}
        return alerts.line_for_route(route, route_lines)

    def stop_name(self, stop_id: str) -> str:
        stop = self.stops.get(stop_id)
        return str(stop[0]) if stop else stop_id

    def destination(self, route: str, direction: Any, stops: list[list[Any]]) -> str:
        """Where a trip is heading: its usual headsign, else the name of its last stop."""
        headsign = self.headsigns.get(f"{route}|{direction}")
        if headsign:
            return headsign
        return self.stop_name(str(stops[-1][0])) if stops else "?"

    def tram_stop_ids(self, extra: set[str] | None = None) -> set[str]:
        """Stops typed as tram stops, plus those trams are scheduled at now (``extra``)."""
        typed = {sid for sid, s in self.stops.items() if len(s) > 5 and s[5] in TRAM_VEHICLE_TYPES}
        return typed | ((extra or set()) & self.stops.keys())

    def find_stops(self, query: str, tram_ids: set[str]) -> list[StopGroup]:
        """Tram stops matching ``query`` by name, stop code (``H0102``) or id, best match first."""
        q = normalize(query)
        if not q:
            return []
        by_name: dict[str, list[str]] = {}
        for sid in tram_ids:
            by_name.setdefault(self.stop_name(sid), []).append(sid)
        scores: dict[str, float] = {}
        for name, ids in by_name.items():
            n = normalize(name)
            codes = {normalize(str(self.stops[s][1])) for s in ids} | {normalize(s) for s in ids}
            if n == q or q in codes:
                score = 3.0
            elif n.startswith(q) or q.startswith(n + " "):
                score = 2.0
            elif q in n or any(word == q for word in n.split()):
                score = 1.0
            else:
                continue
            scores[name] = score
        if not scores:
            # Typos: "lasipalatsi" for "Lasipalatsi" is caught above, "lasipalatsy" here.
            names = {normalize(name): name for name in by_name}
            for close in difflib.get_close_matches(q, list(names), n=3, cutoff=0.75):
                scores[names[close]] = 0.5
        groups = [StopGroup(name, sorted(by_name[name]), score) for name, score in scores.items()]
        return sorted(groups, key=lambda g: (-g.score, len(g.name), g.name))

    def similar_stop_names(self, query: str, tram_ids: set[str], n: int = 5) -> list[str]:
        names = {normalize(self.stop_name(s)): self.stop_name(s) for s in tram_ids}
        return [names[m] for m in difflib.get_close_matches(normalize(query), list(names), n=n, cutoff=0.5)]


def _journey(route: str, direction: Any, day: str, start: str) -> tuple[str, str, str, str]:
    """A trip's identity in both feeds: route, HFP direction (1/2), service day, HH:MM.

    GTFS-RT counts directions 0/1 and writes trips after midnight as ``25:10:00``;
    HFP counts directions 1/2 and wraps the hour.
    """
    hh, _, rest = str(start).partition(":")
    try:
        hour = int(hh) % 24
        direction_hfp = str(int(direction) + 1)
    except (TypeError, ValueError):
        return (route, "", "", "")
    day_iso = f"{day[:4]}-{day[4:6]}-{day[6:8]}" if len(day) == 8 and "-" not in day else day
    return (route, direction_hfp, day_iso, f"{hour:02d}:{rest[:2]}")


def _minutes(seconds: float) -> float:
    return round(float(seconds) / 60, 1)


# --- tools -----------------------------------------------------------------------------

@dataclass
class TramData:
    gtfsrt_dir: Path
    raw_dir: Path
    processed_dir: Path
    predictions_dir: Path
    tz: str = "Europe/Helsinki"
    on_time_window: tuple[int, int] = (-60, 180)
    in_service_s: float = 120
    live_window_min: float = 15
    clock: Callable[[], pd.Timestamp] = field(default=lambda: pd.Timestamp.now(tz="UTC"))
    _cache: dict[str, tuple[float, Any]] = field(default_factory=dict, repr=False)

    @classmethod
    def from_config(cls) -> TramData:
        cfg = load_config()
        paths, proc, dash = cfg["paths"], cfg["process"], cfg["dashboard"]
        return cls(
            gtfsrt_dir=resolve_path(paths["gtfsrt_dir"]), raw_dir=resolve_path(paths["raw_dir"]),
            processed_dir=resolve_path(paths["processed_dir"]),
            predictions_dir=resolve_path(paths["predictions_dir"]), tz=proc["timezone"],
            on_time_window=(proc["on_time_min_s"], proc["on_time_max_s"]),
            in_service_s=dash["in_service_s"], live_window_min=dash["live_window_min"],
        )

    # -- cached sources --

    def _by_mtime(self, key: str, path: Path, load: Callable[[Path], Any]) -> Any:
        """``load(path)``, reloaded only when the file changes; None when it doesn't exist."""
        try:
            mtime = os.stat(path).st_mtime
        except OSError:
            return None
        cached = self._cache.get(key)
        if cached is None or cached[0] != mtime:
            cached = (mtime, load(path))
            self._cache[key] = cached
        return cached[1]

    def network(self) -> Network | None:
        return self._by_mtime("network", self.gtfsrt_dir / gtfsrt.NETWORK, Network.load)

    def current_trips(self) -> dict[str, Any] | None:
        return self._by_mtime("trips", self.gtfsrt_dir / gtfsrt.CURRENT_TRIPS,
                              lambda p: json.loads(p.read_text(encoding="utf-8")))

    def live(self) -> pd.DataFrame:
        """Each tram's newest observation, for trams seen in the last ``in_service_s``."""
        now = self.clock()
        cached = self._cache.get("live")
        if cached is not None and now.timestamp() - cached[0] < 10:
            return cached[1]
        raw = process.load_recent(self.raw_dir, pd.Timedelta(minutes=self.live_window_min), now=now)
        df = process.clean_with_config(raw) if not raw.empty else pd.DataFrame()
        if not df.empty:
            newest = df["timestamp"].max()
            df = df[df["timestamp"] >= newest - pd.Timedelta(seconds=self.in_service_s)]
            df = df.sort_values("timestamp").groupby("vehicle").tail(1).reset_index(drop=True)
        self._cache["live"] = (now.timestamp(), df)
        return df

    def _local(self, ts: pd.Timestamp) -> pd.Timestamp:
        return ts.tz_convert(self.tz)

    def _trips_or_error(self) -> tuple[Network, dict[str, Any]] | dict[str, Any]:
        network, trips = self.network(), self.current_trips()
        if network is None:
            return {"error": "Stop names are not available yet (static GTFS has not been downloaded)."}
        if trips is None:
            return {"error": "HSL's realtime trip predictions are not being collected right now."}
        return network, trips

    def _freshness(self, trips: dict[str, Any], now: pd.Timestamp) -> dict[str, Any]:
        fetched = pd.Timestamp(trips["fetched_at"]).tz_convert("UTC")
        age = (now - fetched).total_seconds()
        out: dict[str, Any] = {"predictions_fetched_at": f"{self._local(fetched):%H:%M:%S}"}
        if age > STALE_TRIPS_S:
            out["warning"] = f"Predictions are {age / 60:.0f} min old; the feed may have stopped."
        return out

    def _tracked(self) -> dict[tuple[str, str, str, str], dict[str, Any]]:
        """Trams on a journey now, keyed like ``_journey``: their delay and vehicle number."""
        live = self.live()
        if live.empty:
            return {}
        out = {}
        for row in live.to_dict("records"):
            out[self._live_key(row)] = {"vehicle": int(row["vehicle"]), "delay_s": int(row["delay_s"])}
        return out

    @staticmethod
    def _live_key(row: dict[Any, Any]) -> tuple[str, str, str, str]:
        return (str(row["route"]), str(row["direction"]), str(row["operating_day"]), str(row["start_time"])[:5])

    @staticmethod
    def _stop_lines(network: Network, trips: dict[str, Any]) -> dict[str, set[str]]:
        lines: dict[str, set[str]] = {}
        for trip in trips.get("trips", []):
            line = network.line_of(str(trip.get("route_id", "")))
            if line:
                for stop in trip.get("stops", []):
                    lines.setdefault(str(stop[0]), set()).add(line)
        return lines

    def find_stop(self, name: str) -> dict[str, Any]:
        loaded = self._trips_or_error()
        if isinstance(loaded, dict):
            return loaded
        network, trips = loaded
        stop_lines = self._stop_lines(network, trips)
        tram_ids = network.tram_stop_ids(set(stop_lines))
        groups = network.find_stops(name, tram_ids)
        if not groups:
            return {"error": f"No tram stop matches {name!r}.",
                    "similar_names": network.similar_stop_names(name, tram_ids)}
        out = []
        for group in groups[:6]:
            platforms = []
            for sid in group.ids:
                stop = network.stops[sid]
                platforms.append({"code": stop[1], "street": stop[2],
                                  "lines_now": sorted(stop_lines.get(sid, ()), key=alerts.line_sort_key)})
            out.append({"name": group.name, "platforms": platforms})
        return {"matches": out}

    def next_trams(self, stop: str, line: str | None = None, limit: int = 8) -> dict[str, Any]:
        loaded = self._trips_or_error()
        if isinstance(loaded, dict):
            return loaded
        network, trips = loaded
        now = self.clock()
        wanted = normalize_line(line)
        stop_lines = self._stop_lines(network, trips)
        tram_ids = network.tram_stop_ids(set(stop_lines))
        groups = network.find_stops(stop, tram_ids)
        if not groups:
            return {"error": f"No tram stop matches {stop!r}.",
                    "similar_names": network.similar_stop_names(stop, tram_ids)}
        group = groups[0]
        ids = set(group.ids)
        tracked = self._tracked()
        limit = max(1, min(int(limit), 15))

        arrivals = []
        for trip in trips.get("trips", []):
            route = str(trip.get("route_id", ""))
            trip_line = network.line_of(route) or route
            if wanted and trip_line.upper() != wanted:
                continue
            stops = trip.get("stops", [])
            for i, (sid, arrival, departure, _uncertainty, relationship) in enumerate(stops):
                if sid not in ids:
                    continue
                at = arrival or departure
                if at is None or at < now.timestamp() - 30:
                    continue
                when = pd.Timestamp(int(at), unit="s", tz="UTC")
                item: dict[str, Any] = {
                    "line": trip_line,
                    "towards": network.destination(route, trip.get("direction_id"), stops),
                    "platform": network.stops.get(sid, ["", ""])[1],
                    "time": f"{self._local(when):%H:%M}",
                    "in_min": max(0, int((at - now.timestamp()) // 60)),
                    "_at": at,
                }
                if i == len(stops) - 1:
                    item["terminates_here"] = True
                if trip.get("schedule_relationship") == "CANCELED" or relationship == "SKIPPED":
                    item["cancelled"] = True
                tram = tracked.get(_journey(route, trip.get("direction_id"), str(trip.get("start_date", "")),
                                            str(trip.get("start_time", ""))))
                if tram:
                    item["tram_now"] = {"vehicle": tram["vehicle"], "delay_min": _minutes(tram["delay_s"])}
                arrivals.append(item)
        arrivals.sort(key=lambda a: a["_at"])
        for a in arrivals:
            del a["_at"]
        out: dict[str, Any] = {
            "stop": group.name,
            "platforms": sorted({network.stops[s][1] for s in ids if network.stops[s][1]}),
            "now": f"{self._local(now):%H:%M}",
            "arrivals": arrivals[:limit],
            "source": "HSL realtime predictions; tram_now is the tram's current delay from its live position",
            **self._freshness(trips, now),
        }
        if wanted:
            out["line_filter"] = wanted
        if not arrivals:
            out["note"] = "No trams predicted at this stop in the trip feed now (HSL lists trips about 1-2 h ahead)."
        if len(groups) > 1:
            out["other_matching_stops"] = [g.name for g in groups[1:5]]
        return out

    def line_status(self, line: str | None = None) -> dict[str, Any]:
        live = self.live()
        if live.empty:
            return {"error": "No live tram positions from the last few minutes; the collector may be down."}
        now = self.clock()
        newest = live["timestamp"].max()
        lo, hi = self.on_time_window
        wanted = normalize_line(line)
        base: dict[str, Any] = {"as_of": f"{self._local(newest):%H:%M:%S}",
                                "on_time_means": f"between {-lo // 60} min early and {hi // 60} min late"}
        if not wanted:
            table = live.groupby("line").agg(trams=("vehicle", "nunique"), mean_delay_s=("delay_s", "mean"),
                                             on_time=("on_time", "mean"), most_late_s=("delay_s", "max"))
            per_line = sorted(table.reset_index().to_dict("records"),
                              key=lambda r: alerts.line_sort_key(str(r["line"])))
            lines = [{"line": str(r["line"]), "trams": int(r["trams"]), "mean_delay_min": _minutes(r["mean_delay_s"]),
                      "on_time_pct": round(100 * float(r["on_time"])), "most_late_min": _minutes(r["most_late_s"])}
                     for r in per_line]
            return {**base, "trams_in_service": int(live["vehicle"].nunique()),
                    "on_time_pct": round(100 * float(live["on_time"].mean())),
                    "mean_delay_min": _minutes(live["delay_s"].mean()), "lines": lines}

        trams = live[live["line"].astype(str).str.upper() == wanted]
        if trams.empty:
            return {**base, "line": wanted, "trams_in_service": 0,
                    "note": f"No line {wanted} trams in service in the last few minutes.",
                    "lines_in_service": sorted({str(x) for x in live["line"]}, key=alerts.line_sort_key)}
        forecasts = self._forecasts(now)
        network = self.network()
        trips = self.current_trips() or {}
        by_journey = {_journey(str(t.get("route_id", "")), t.get("direction_id"), str(t.get("start_date", "")),
                               str(t.get("start_time", ""))): t for t in trips.get("trips", [])}
        rows = []
        for r in trams.sort_values("delay_s", ascending=False).to_dict("records"):
            vehicle = int(r["vehicle"])
            item: dict[str, Any] = {"vehicle": vehicle, "delay_min": _minutes(r["delay_s"])}
            trip = by_journey.get(self._live_key(r))
            if trip and network:
                stops = trip.get("stops", [])
                item["towards"] = network.destination(str(trip["route_id"]), trip.get("direction_id"), stops)
                upcoming = [s for s in stops if (s[1] or s[2] or 0) >= now.timestamp()]
                if upcoming:
                    item["next_stop"] = network.stop_name(str(upcoming[0][0]))
            if vehicle in forecasts:
                item["forecast_delay_in_5_min"] = _minutes(forecasts[vehicle])
            rows.append(item)
        cancelled = sum(1 for t in trips.get("trips", []) if t.get("schedule_relationship") == "CANCELED"
                        and network is not None and network.line_of(str(t.get("route_id", ""))) == wanted)
        out = {**base, "line": wanted, "trams_in_service": len(rows),
               "on_time_pct": round(100 * float(trams["on_time"].mean())),
               "mean_delay_min": _minutes(trams["delay_s"].mean()), "trams": rows[:20]}
        if cancelled:
            out["cancelled_trips_in_feed"] = cancelled
        if forecasts:
            out["forecast_source"] = "this project's model, forecasting each tram's delay 5 minutes ahead"
        return out

    def _forecasts(self, now: pd.Timestamp) -> dict[int, float]:
        latest = forecaster.read_latest(self.predictions_dir)
        if not latest or not latest.get("issued_at"):
            return {}
        if (now - pd.Timestamp(latest["issued_at"]).tz_convert("UTC")).total_seconds() > FORECAST_FRESH_S:
            return {}
        return {int(f["vehicle"]): float(f["predicted"]) for f in latest.get("forecasts", [])}

    def service_alerts(self, line: str | None = None) -> dict[str, Any]:
        snapshot = alerts.latest_snapshot(self.gtfsrt_dir)
        if snapshot is None:
            return {"error": "HSL's service alerts are not being collected right now."}
        network = self.network()
        trips = self.current_trips() or {}
        if network is None:
            return {"error": "Stop names are not available yet (static GTFS has not been downloaded)."}
        now = self.clock()
        route_lines = {r: v["line"] for r, v in network.routes.items()}
        found = alerts.active(alerts.for_trams(alerts.parse(snapshot), route_lines,
                                               self._stop_lines(network, trips)), now)
        wanted = normalize_line(line)
        if wanted:
            found = [a for a in found if wanted in {x.upper() for x in a.lines}]
        items = []
        for a in found[:10]:
            item: dict[str, Any] = {"lines": list(a.lines), "header": a.header,
                                    "effect": alerts.label(a.effect), "cause": alerts.label(a.cause)}
            if a.description and a.description != a.header:
                item["description"] = a.description[:500]
            end = a.ends(now)
            if end is not None:
                item["until"] = f"{self._local(end):%a %d %b %H:%M}"
            if a.url:
                item["url"] = a.url
            items.append(item)
        fetched = pd.Timestamp(snapshot["fetched_at"]).tz_convert("UTC")
        # Snapshots are saved when the alert list changes, so this is when it last changed, not the last check.
        return {"alert_list_changed_at": f"{self._local(fetched):%a %H:%M}", "alerts_in_force": len(found),
                "alerts": items}

    def punctuality(self, line: str | None = None, days: int = 7, group_by: str = "line") -> dict[str, Any]:
        collected = available_days(self.processed_dir)
        if not collected:
            return {"error": "No processed history yet."}
        days = max(1, min(int(days), 60))
        end = collected[-1]
        start = end - timedelta(days=days - 1)
        wanted = normalize_line(line)
        h = History(self.processed_dir, start, end, [wanted] if wanted else None, self.tz)
        summary = h.summary()
        if not summary["rows"]:
            return {"error": f"No history for line {wanted} in those days." if wanted else "No history yet."}
        out: dict[str, Any] = {
            "from": f"{summary['from']:%Y-%m-%d %H:%M}", "to": f"{summary['to']:%Y-%m-%d %H:%M}",
            "line": wanted or "all", "on_time_pct": round(summary["on_time_pct"], 1),
            "mean_delay_min": _minutes(summary["mean_delay_s"]),
            "on_time_means": f"between {-self.on_time_window[0] // 60} min early and "
                             f"{self.on_time_window[1] // 60} min late",
            "measured_from": "every tram's position every 10 s, cleaned",
        }
        if group_by == "hour":
            df = h.by_hour()
            out["by_hour"] = [{"hour": int(r["hour"]), "day_type": r["day_type"],
                               "on_time_pct": round(float(r["on_time_pct"]), 1),
                               "mean_delay_min": _minutes(r["mean_delay_s"])} for r in df.to_dict("records")]
        elif group_by == "day":
            df = h.by_day()
            out["by_day"] = [{"day": f"{r['day']:%a %Y-%m-%d}", "on_time_pct": round(float(r["on_time_pct"]), 1),
                              "mean_delay_min": _minutes(r["mean_delay_s"])} for r in df.to_dict("records")]
        else:
            df = h.by_line(self.on_time_window[1])
            rows: list[dict[str, Any]] = [
                {"line": str(r["line"]), "on_time_pct": round(float(r["on_time_pct"]), 1),
                 "mean_delay_min": _minutes(r["mean_delay_s"]), "p90_delay_min": _minutes(r["p90_delay_s"]),
                 "observations": int(r["rows"])}
                for r in df.to_dict("records")]
            # Lettered lines (3H, 7B, 1T) are short or occasional variants with few trips. Ranked with
            # the main lines they top every list on a handful of runs, and the model ranks them even
            # when they are listed apart, so they are named only (ask for one with ``line``).
            shown = [r for r in rows if wanted or r["line"].isdigit()]
            out["by_line"] = sorted(shown, key=lambda x: x["on_time_pct"])
            variants = sorted((r["line"] for r in rows if r not in shown), key=alerts.line_sort_key)
            if variants:
                out["variants_left_out"] = {"lines": variants, "why": "short or occasional variants of a main "
                                            "line with few trips; ask for one by its line to see it"}
        return out

    def forecast_accuracy(self) -> dict[str, Any]:
        now = self.clock()
        scored = monitoring.read_scored(self.predictions_dir, (now - pd.Timedelta(days=1)).date(), now.date())
        if scored.empty:
            return {"error": "No scored forecasts yet."}
        metrics = monitoring.live_metrics(scored, now)
        out: dict[str, Any] = {
            "what": "Mean absolute error of the 5-minute-ahead delay forecast, scored against what happened, "
                    "vs persistence ('the delay stays the same'). Lower is better.",
        }
        for window in ("1h", "24h"):
            entry = metrics.get(window, {})
            overall = entry.get("overall", {})
            if overall.get("rows"):
                out[window] = {
                    "forecasts_scored": overall["rows"],
                    "model_mae_s": overall["model_mae_s"], "persistence_mae_s": overall["persistence_mae_s"],
                    "improvement_pct": overall["improvement_pct"],
                    "en_route": {k: entry["segments"].get("en_route", {}).get(k)
                                 for k in ("model_mae_s", "persistence_mae_s")},
                    "model_versions": entry.get("model_versions", []),
                }
        return out


# --- tool definitions ------------------------------------------------------------------

_LINE = {"type": "string", "description": "Tram line, e.g. '4' or '6T'. Omit for all lines."}
TOOLS: list[dict[str, Any]] = [
    {"type": "function", "function": {
        "name": "next_trams",
        "description": "Upcoming trams at a stop, soonest first, from HSL's realtime predictions: line, "
                       "destination, platform code, clock time, minutes away, and the tram's current delay "
                       "when it is being tracked. Use for 'when is the next tram at X'.",
        "parameters": {"type": "object", "properties": {
            "stop": {"type": "string", "description": "Stop name as HSL writes it in Finnish (e.g. "
                     "'Lasipalatsi', 'Rautatieasema'), or a stop code like 'H0102'."},
            "line": _LINE,
            "limit": {"type": "integer", "description": "How many trams to list (default 8, at most 15)."},
        }, "required": ["stop"]}}},
    {"type": "function", "function": {
        "name": "find_stop",
        "description": "Tram stops matching a name, with their platform codes, streets and the lines "
                       "stopping there now. Use for 'which trams stop at X' or when a stop name is unclear.",
        "parameters": {"type": "object", "properties": {
            "name": {"type": "string", "description": "Full or partial stop name, or a stop code."},
        }, "required": ["name"]}}},
    {"type": "function", "function": {
        "name": "line_status",
        "description": "Trams in service right now. Without a line: per-line count, mean delay and on-time "
                       "share. With a line: each tram's delay, destination, next stop and this project's "
                       "forecast of its delay 5 minutes ahead.",
        "parameters": {"type": "object", "properties": {"line": _LINE}}}},
    {"type": "function", "function": {
        "name": "service_alerts",
        "description": "HSL's service alerts for trams in force now (disruptions, detours, track works).",
        "parameters": {"type": "object", "properties": {"line": _LINE}}}},
    {"type": "function", "function": {
        "name": "punctuality",
        "description": "Past punctuality from the collected history: on-time share and mean delay over the "
                       "last N days, grouped by line, by hour of day (weekday vs weekend) or by day.",
        "parameters": {"type": "object", "properties": {
            "line": _LINE,
            "days": {"type": "integer", "description": "How many recent days (default 7, at most 60)."},
            "group_by": {"type": "string", "enum": ["line", "hour", "day"],
                         "description": "Breakdown to return (default 'line')."},
        }}}},
    {"type": "function", "function": {
        "name": "forecast_accuracy",
        "description": "How accurate this project's 5-minute delay forecasts have been in the last hour and "
                       "24 hours, compared with assuming the delay stays the same.",
        "parameters": {"type": "object", "properties": {}}}},
]
TOOL_NAMES = {t["function"]["name"] for t in TOOLS}


def call_tool(data: TramData, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Run one tool by name; errors come back as ``{"error": ...}`` for the model to read."""
    if name not in TOOL_NAMES:
        return {"error": f"Unknown tool {name!r}."}
    spec = next(t["function"] for t in TOOLS if t["function"]["name"] == name)
    allowed = spec["parameters"]["properties"].keys()
    kwargs = {k: v for k, v in arguments.items() if k in allowed and v not in (None, "")}
    missing = [k for k in spec["parameters"].get("required", []) if k not in kwargs]
    if missing:
        return {"error": f"Missing argument(s): {', '.join(missing)}."}
    try:
        return getattr(data, name)(**kwargs)
    except (TypeError, ValueError) as exc:
        return {"error": f"Bad arguments for {name}: {exc}"}
    except Exception as exc:  # a broken data file must not end the conversation
        return {"error": f"{name} failed: {type(exc).__name__}"}
