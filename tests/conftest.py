from datetime import datetime, timedelta, timezone

import numpy as np
import pytest


def make_vp(veh=1, t=None, dl=0, desi="4", route="1004", dir="1", start="12:00", **extra):
    t = t or datetime(2026, 2, 4, 10, 0, tzinfo=timezone.utc)
    record = {
        "desi": desi, "dir": dir, "oper": 40, "veh": veh,
        "tst": t.strftime("%Y-%m-%dT%H:%M:%S.000Z"), "tsi": int(t.timestamp()),
        "spd": 5.0, "hdg": 90, "lat": 60.17, "long": 24.94, "acc": 0.0, "dl": dl,
        "odo": 1000, "drst": 0, "oday": "2026-02-04", "jrn": 1, "line": 30,
        "start": start, "loc": "GPS", "stop": None, "route": route, "occu": 0,
    }
    record.update(extra)
    return record


@pytest.fixture
def synthetic_raw():
    """Two hours of 10 s observations for several trams whose lateness drifts over time."""
    rng = np.random.default_rng(0)
    t0 = datetime(2026, 2, 4, 8, 0, tzinfo=timezone.utc)
    records = []
    for veh in range(1, 9):
        delay = 0.0  # seconds late
        drift = rng.normal(0.3, 0.2)
        for step in range(720):
            delay += drift + rng.normal(0, 2)
            records.append(make_vp(
                veh=veh, t=t0 + timedelta(seconds=10 * step), dl=-int(delay),
                desi=str(veh % 3 + 1), start="10:00",
            ))
    return records
