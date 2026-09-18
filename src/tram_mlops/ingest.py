"""Collect live tram positions from the HSL high-frequency positioning (HFP) MQTT feed.

Messages are appended to one JSONL file per UTC day under ``paths.raw_dir``.
Run with ``python -m tram_mlops.ingest [--duration SECONDS]``.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import IO

import paho.mqtt.client as mqtt

from tram_mlops.config import load_config, resolve_path

log = logging.getLogger(__name__)


class RawWriter:
    """Append vehicle-position records to daily JSONL files, keeping one handle open."""

    def __init__(self, raw_dir: Path, sample_interval_s: float = 0):
        self.raw_dir = raw_dir
        self.sample_interval_s = sample_interval_s
        self._last_kept: dict[int, float] = {}
        self._day: str | None = None
        self._handle: IO[str] | None = None
        self.written = 0

    def _file_for(self, day: str) -> IO[str]:
        if day != self._day:
            self.close()
            self.raw_dir.mkdir(parents=True, exist_ok=True)
            path = self.raw_dir / f"tram_{day}.jsonl"
            self._handle = open(path, "a", encoding="utf-8")
            self._day = day
            log.info("Writing to %s", path)
        return self._handle

    def write(self, vp: dict, received_at: float | None = None) -> bool:
        """Store a record unless its vehicle was stored less than sample_interval_s ago."""
        received_at = time.time() if received_at is None else received_at
        veh = vp.get("veh")
        last = self._last_kept.get(veh)
        if last is not None and received_at - last < self.sample_interval_s:
            return False
        self._last_kept[veh] = received_at

        day = datetime.fromtimestamp(received_at, tz=timezone.utc).strftime("%Y-%m-%d")
        handle = self._file_for(day)
        handle.write(json.dumps(vp) + "\n")
        handle.flush()
        self.written += 1
        return True

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None
            self._day = None


def parse_message(payload: bytes) -> dict | None:
    """Return the vehicle-position record from an HFP payload, or None if it isn't one."""
    try:
        message = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    vp = message.get("VP") if isinstance(message, dict) else None
    return vp if isinstance(vp, dict) else None


def build_client(topic: str, writer: RawWriter) -> mqtt.Client:
    client = mqtt.Client(callback_api_version=mqtt.CallbackAPIVersion.VERSION2)
    client.reconnect_delay_set(min_delay=1, max_delay=60)

    def on_connect(client, userdata, flags, reason_code, properties=None):
        if reason_code.is_failure:
            log.error("Connection refused: %s", reason_code)
            return
        # Subscribing here means the subscription is restored after every reconnect.
        client.subscribe(topic)
        log.info("Connected, subscribed to %s", topic)

    def on_disconnect(client, userdata, flags, reason_code, properties=None):
        log.warning("Disconnected (%s); paho will reconnect", reason_code)

    def on_message(client, userdata, msg):
        vp = parse_message(msg.payload)
        if vp is None:
            log.debug("Skipping non-VP message on %s", msg.topic)
            return
        writer.write(vp)

    client.on_connect = on_connect
    client.on_disconnect = on_disconnect
    client.on_message = on_message
    return client


def run(duration_s: float | None = None) -> int:
    cfg = load_config()
    ingest_cfg = cfg["ingest"]
    writer = RawWriter(resolve_path(cfg["paths"]["raw_dir"]), ingest_cfg["sample_interval_s"])
    client = build_client(ingest_cfg["topic"], writer)

    log.info("Connecting to %s:%s", ingest_cfg["host"], ingest_cfg["port"])
    client.connect(ingest_cfg["host"], ingest_cfg["port"], keepalive=60)
    client.loop_start()
    started = time.monotonic()
    try:
        while duration_s is None or time.monotonic() - started < duration_s:
            time.sleep(10)
            log.info("%d records written", writer.written)
    except KeyboardInterrupt:
        log.info("Stopping data collection")
    finally:
        client.disconnect()
        client.loop_stop()
        writer.close()
    log.info("Done: %d records written", writer.written)
    return writer.written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, help="stop after this many seconds (default: run forever)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run(args.duration)


if __name__ == "__main__":
    main()
