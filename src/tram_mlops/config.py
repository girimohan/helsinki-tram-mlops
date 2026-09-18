"""Load pipeline settings from configs/config.toml with env-var overrides."""

from __future__ import annotations

import os
import tomllib
from functools import lru_cache
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "config.toml"


def _coerce(raw: str, current: Any) -> Any:
    if isinstance(current, bool):
        return raw.lower() in {"1", "true", "yes"}
    if isinstance(current, int):
        return int(raw)
    if isinstance(current, float):
        return float(raw)
    return raw


@lru_cache(maxsize=1)
def load_config(path: str | os.PathLike | None = None) -> dict[str, dict[str, Any]]:
    config_path = Path(path or os.environ.get("TRAM_CONFIG", DEFAULT_CONFIG))
    with open(config_path, "rb") as f:
        config = tomllib.load(f)

    for section, values in config.items():
        for key, current in values.items():
            env_value = os.environ.get(f"TRAM_{section}_{key}".upper())
            if env_value is not None:
                values[key] = _coerce(env_value, current)
    return config


def resolve_path(value: str | os.PathLike) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path
