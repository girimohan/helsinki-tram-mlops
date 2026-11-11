"""The tram network's names and places: lines, destinations and stops (from static GTFS).

``gtfsrt`` saves ``tram_network.json`` daily; the realtime feeds name stops and routes only
by id, so the chat assistant (``chat_tools``) and the anomaly detector (``anomaly``) look
names up here.
"""

from __future__ import annotations

import difflib
import json
import math
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tram_mlops import alerts

TRAM_VEHICLE_TYPES = {"0", "900"}
EARTH_RADIUS_M = 6_371_000


def normalize(text: str) -> str:
    """Lower case, no accents, single spaces: ``"Töölön tulli"`` -> ``"toolon tulli"``."""
    decomposed = unicodedata.normalize("NFKD", text)
    plain = "".join(ch for ch in decomposed if not unicodedata.combining(ch)).casefold()
    return " ".join(re.sub(r"[^0-9a-z]+", " ", plain).split())


def distance_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in metres."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * EARTH_RADIUS_M * math.asin(math.sqrt(a))


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

    def nearest_stop(self, lat: float, lon: float, stop_ids: set[str] | None = None) -> tuple[str, float] | None:
        """Name of the closest stop (tram stops by default) and its distance in metres."""
        best: tuple[str, float] | None = None
        for sid in stop_ids if stop_ids is not None else self.tram_stop_ids():
            stop = self.stops.get(sid)
            if not stop:
                continue
            d = distance_m(lat, lon, float(stop[3]), float(stop[4]))
            if best is None or d < best[1]:
                best = (str(stop[0]), d)
        return best

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
