"""HSL service alerts for the dashboard, read from the GTFS-RT archive (see ``gtfsrt``).

The poller saves the full alert list whenever its content changes, one snapshot per
line in ``alerts_YYYY-MM-DD.jsonl.gz``, so the newest snapshot holds every alert HSL
is publishing now. HSL's feed covers all modes; an alert concerns trams when it names
a tram route (or a lettered variant such as ``1011H``) or a stop that trams serve.
"""

from __future__ import annotations

import gzip
import json
import zlib
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import pandas as pd

LANGUAGES = ("en", "fi", "sv")
SEVERITY_RANK = {"UNKNOWN_SEVERITY": 0, "INFO": 1, "WARNING": 2, "SEVERE": 3}
# Effects that change how trams run, as opposed to information or moved stops.
DISRUPTIVE_EFFECTS = {"NO_SERVICE", "REDUCED_SERVICE", "SIGNIFICANT_DELAYS", "DETOUR", "MODIFIED_SERVICE"}


@dataclass(frozen=True)
class Alert:
    id: str
    header: str
    description: str
    url: str
    cause: str
    effect: str
    severity: str
    periods: tuple[tuple[pd.Timestamp | None, pd.Timestamp | None], ...]
    routes: tuple[str, ...]
    stops: tuple[str, ...]
    lines: tuple[str, ...] = ()  # tram lines it names, filled in by ``for_trams``

    @property
    def disruptive(self) -> bool:
        return SEVERITY_RANK.get(self.severity, 0) >= 2 or self.effect in DISRUPTIVE_EFFECTS

    def active(self, now: pd.Timestamp) -> bool:
        """In force at ``now``; an alert without periods is always in force (GTFS-RT)."""
        if not self.periods:
            return True
        return any((start is None or start <= now) and (end is None or now < end) for start, end in self.periods)

    def ends(self, now: pd.Timestamp) -> pd.Timestamp | None:
        """End of the period in force at ``now``, if it has one."""
        for start, end in self.periods:
            if (start is None or start <= now) and (end is None or now < end):
                return end
        return None


def snapshot_files(gtfsrt_dir: Path) -> list[Path]:
    """Daily alert archives, newest first."""
    return sorted(gtfsrt_dir.glob("alerts_*.jsonl.gz"), reverse=True)


def read_last_snapshot(path: Path) -> dict[str, Any] | None:
    """The newest complete snapshot in one archive file.

    The poller appends each snapshot as its own gzip member, so a reader can catch the
    last one half written; everything before it is still readable.
    """
    last = None
    try:
        with gzip.open(path, "rt", encoding="utf-8") as f:
            for line in f:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(record, dict) and isinstance(record.get("alerts"), list):
                    last = record
    except (EOFError, OSError, zlib.error):
        pass
    return last


def latest_snapshot(gtfsrt_dir: Path) -> dict[str, Any] | None:
    """The alert list as last fetched; falls back to earlier days if today's file has none yet."""
    for path in snapshot_files(gtfsrt_dir):
        snapshot = read_last_snapshot(path)
        if snapshot is not None:
            return snapshot
    return None


def _text(translated: Any) -> str:
    """English if available, then Finnish, then Swedish, then whatever there is."""
    if not isinstance(translated, dict):
        return ""
    texts = {t.get("language"): t.get("text", "") for t in translated.get("translation", []) if isinstance(t, dict)}
    for lang in LANGUAGES:
        if texts.get(lang):
            return str(texts[lang])
    return str(next(iter(texts.values()), ""))


def _epoch(value: Any) -> pd.Timestamp | None:
    try:
        return pd.Timestamp(int(value), unit="s", tz="UTC")
    except (TypeError, ValueError):
        return None


def parse(snapshot: Mapping[str, Any] | None) -> list[Alert]:
    """Alerts from a snapshot (as written by ``gtfsrt.AlertArchive``)."""
    alerts = []
    for raw in (snapshot or {}).get("alerts", []):
        entities = [e for e in raw.get("informed_entity", []) if isinstance(e, dict)]
        alerts.append(Alert(
            id=str(raw.get("id", "")),
            header=_text(raw.get("header_text")),
            description=_text(raw.get("description_text")),
            url=_text(raw.get("url")),
            cause=str(raw.get("cause", "UNKNOWN_CAUSE")),
            effect=str(raw.get("effect", "UNKNOWN_EFFECT")),
            severity=str(raw.get("severity_level", "UNKNOWN_SEVERITY")),
            periods=tuple((_epoch(p.get("start")), _epoch(p.get("end")))
                          for p in raw.get("active_period", []) if isinstance(p, dict)),
            routes=tuple(dict.fromkeys(str(e["route_id"]) for e in entities if e.get("route_id"))),
            stops=tuple(dict.fromkeys(str(e["stop_id"]) for e in entities if e.get("stop_id"))),
        ))
    return alerts


def line_for_route(route: str, route_lines: Mapping[str, str]) -> str | None:
    """The tram line a GTFS route id belongs to, also for lettered variants (``1011H`` -> line of ``1011``)."""
    if route in route_lines:
        return route_lines[route]
    base = route.rstrip("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    if base != route and base in route_lines:
        return route_lines[base]
    return None


def for_trams(alerts: Iterable[Alert], route_lines: Mapping[str, str],
              stop_lines: Mapping[str, Iterable[str]]) -> list[Alert]:
    """Alerts naming a tram route or tram stop, with the tram lines they concern filled in.

    ``route_lines`` maps tram route ids to line names (e.g. ``"1009" -> "9"``) and
    ``stop_lines`` maps the stops trams serve to the lines seen there; both come from the
    positions data, the ground truth for what counts as a tram.

    An alert naming routes concerns trams only if one of them is a tram route: a bus
    alert may also list a stop that trams share. An alert naming only stops concerns the
    lines seen at those stops.
    """
    out = []
    for alert in alerts:
        if alert.routes:
            lines = {line for line in (line_for_route(r, route_lines) for r in alert.routes) if line}
        else:
            lines = {line for s in alert.stops for line in stop_lines.get(s, ())}
        if lines:
            out.append(replace(alert, lines=tuple(sorted(lines, key=line_sort_key))))
    return sorted(out, key=lambda a: (-SEVERITY_RANK.get(a.severity, 0), not a.disruptive, a.lines, a.id))


def active(alerts: Iterable[Alert], now: pd.Timestamp) -> list[Alert]:
    return [a for a in alerts if a.active(now)]


def per_line(alerts: Iterable[Alert]) -> dict[str, int]:
    """Number of alerts naming each line."""
    counts: dict[str, int] = {}
    for alert in alerts:
        for line in alert.lines:
            counts[line] = counts.get(line, 0) + 1
    return counts


def label(code: str) -> str:
    """``SIGNIFICANT_DELAYS`` -> ``Significant delays``."""
    return code.replace("_", " ").capitalize()


def line_sort_key(line: str) -> tuple[int, str]:
    """Order lines as people read them: 1, 2, ..., 10, 1T."""
    digits = "".join(ch for ch in line if ch.isdigit())
    return (int(digits) if digits else 10_000, line)
