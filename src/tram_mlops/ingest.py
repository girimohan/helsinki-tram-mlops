"""Collect live tram data from the HSL high-frequency positioning (HFP) MQTT feed.

Two kinds of message arrive on one connection:

- vehicle positions (``vp``), downsampled to one per tram per ``sample_interval_s`` and
  appended to ``tram_YYYY-MM-DD.jsonl``
- stop and journey events (``ingest.events``: arrivals, departures, journey sign-in and
  sign-off, traffic-light priority), kept in full in ``events_YYYY-MM-DD.jsonl``

Files go to ``paths.raw_dir``, one per UTC day. Run with
``python -m tram_mlops.ingest [--duration SECONDS]``.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import IO, Any

import paho.mqtt.client as mqtt

from tram_mlops.config import load_config, resolve_path

log = logging.getLogger(__name__)


class DailyJsonlWriter:
    """Append JSON records to ``<prefix>_YYYY-MM-DD.jsonl`` (UTC day), keeping one handle open."""

    def __init__(self, raw_dir: Path, prefix: str):
        self.raw_dir = raw_dir
        self.prefix = prefix
        self._day: str | None = None
        self._handle: IO[str] | None = None
        self.written = 0

    def _file_for(self, day: str) -> IO[str]:
        if day != self._day:
            self.close()
            self.raw_dir.mkdir(parents=True, exist_ok=True)
            path = self.raw_dir / f"{self.prefix}_{day}.jsonl"
            self._handle = open(path, "a", encoding="utf-8")  # noqa: SIM115 -- kept open across writes; close() releases it
            self._day = day
            log.info("Writing to %s", path)
        assert self._handle is not None
        return self._handle

    def append(self, record: dict[str, Any], received_at: float) -> None:
        day = datetime.fromtimestamp(received_at, tz=UTC).strftime("%Y-%m-%d")
        handle = self._file_for(day)
        handle.write(json.dumps(record) + "\n")
        handle.flush()
        self.written += 1

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None
            self._day = None


class RawWriter(DailyJsonlWriter):
    """Vehicle positions, keeping at most one per vehicle every ``sample_interval_s``."""

    def __init__(self, raw_dir: Path, sample_interval_s: float = 0):
        super().__init__(raw_dir, "tram")
        self.sample_interval_s = sample_interval_s
        # Keyed by (operator, vehicle): vehicle numbers are only unique within an operator.
        # A record without them is still stored and dropped later by process.clean.
        self._last_kept: dict[tuple[Any, Any], float] = {}

    def write(self, vp: dict, received_at: float | None = None) -> bool:
        """Store a record unless its vehicle was stored less than sample_interval_s ago."""
        received_at = time.time() if received_at is None else received_at
        key = (vp.get("oper"), vp.get("veh"))
        last = self._last_kept.get(key)
        if last is not None and received_at - last < self.sample_interval_s:
            return False
        self._last_kept[key] = received_at
        self.append(vp, received_at)
        return True


class EventWriter(DailyJsonlWriter):
    """Stop and journey events, all of them, each tagged with its event type."""

    def __init__(self, raw_dir: Path):
        super().__init__(raw_dir, "events")

    def write(self, event: str, payload: dict, received_at: float | None = None) -> None:
        received_at = time.time() if received_at is None else received_at
        self.append({"event": event, **payload}, received_at)


def parse_event(payload: bytes) -> tuple[str, dict] | None:
    """``(event type, record)`` from an HFP payload such as ``{"ARS": {...}}``, or None."""
    try:
        message = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(message, dict) or len(message) != 1:
        return None
    ((event, record),) = message.items()
    return (event.upper(), record) if isinstance(record, dict) else None


def parse_message(payload: bytes) -> dict | None:
    """Return the vehicle-position record from an HFP payload, or None if it isn't one."""
    parsed = parse_event(payload)
    return parsed[1] if parsed and parsed[0] == "VP" else None


def event_topics(vp_topic: str, events: list[str]) -> list[str]:
    """MQTT topics for ``events``, built by swapping the event segment of the ``vp`` topic."""
    parts = vp_topic.split("/")
    if "vp" not in parts:
        raise ValueError(f"expected a vp topic, got {vp_topic!r}")
    i = parts.index("vp")
    return ["/".join([*parts[:i], event.lower(), *parts[i + 1:]]) for event in events]


def build_client(topics: list[str], positions: RawWriter, events: EventWriter | None = None,
                 tls: bool = False) -> mqtt.Client:
    client = mqtt.Client(callback_api_version=mqtt.CallbackAPIVersion.VERSION2)
    client.reconnect_delay_set(min_delay=1, max_delay=60)
    if tls:
        client.tls_set()  # the system's CA certificates

    def on_connect(client, userdata, flags, reason_code, properties=None):
        if reason_code.is_failure:
            log.error("Connection refused: %s", reason_code)
            return
        # Subscribing here means the subscriptions are restored after every reconnect.
        client.subscribe([(topic, 0) for topic in topics])
        log.info("Connected, subscribed to %d topic(s): %s", len(topics), ", ".join(topics))

    def on_disconnect(client, userdata, flags, reason_code, properties=None):
        log.warning("Disconnected (%s); paho will reconnect", reason_code)

    def on_message(client, userdata, msg):
        parsed = parse_event(msg.payload)
        if parsed is None:
            log.debug("Skipping unparseable message on %s", msg.topic)
            return
        event, record = parsed
        if event == "VP":
            positions.write(record)
        elif events is not None:
            events.write(event, record)

    client.on_connect = on_connect
    client.on_disconnect = on_disconnect
    client.on_message = on_message
    return client


def run(duration_s: float | None = None) -> int:
    cfg = load_config()
    ingest_cfg = cfg["ingest"]
    raw_dir = resolve_path(cfg["paths"]["raw_dir"])
    positions = RawWriter(raw_dir, ingest_cfg["sample_interval_s"])
    events = EventWriter(raw_dir)
    topics = [ingest_cfg["topic"], *event_topics(ingest_cfg["topic"], ingest_cfg.get("events", []))]
    client = build_client(topics, positions, events, tls=ingest_cfg.get("tls", False))

    log.info("Connecting to %s:%s", ingest_cfg["host"], ingest_cfg["port"])
    client.connect(ingest_cfg["host"], ingest_cfg["port"], keepalive=60)
    client.loop_start()
    started = time.monotonic()
    last_log = 0.0
    try:
        while duration_s is None or time.monotonic() - started < duration_s:
            time.sleep(10)
            if time.monotonic() - last_log >= 60:
                log.info("%d positions and %d events written", positions.written, events.written)
                last_log = time.monotonic()
    except KeyboardInterrupt:
        log.info("Stopping data collection")
    finally:
        client.disconnect()
        client.loop_stop()
        positions.close()
        events.close()
    log.info("Done: %d positions and %d events written", positions.written, events.written)
    return positions.written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--duration", type=float, help="stop after this many seconds (default: run forever)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run(args.duration)


if __name__ == "__main__":
    main()
