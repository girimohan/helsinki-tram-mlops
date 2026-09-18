"""Train, evaluate and version the delay-forecasting model.

Run with ``python -m tram_mlops.train``. Each run writes ``models/<version>/``
(model.joblib + metrics.json) and points ``models/latest.json`` at it.
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from tram_mlops.config import load_config, resolve_path
from tram_mlops.features import CATEGORICAL, FEATURES, TARGET, add_features, build_training_set

log = logging.getLogger(__name__)


class InsufficientDataError(RuntimeError):
    pass


def make_model(random_state: int = 42) -> Pipeline:
    encode = ColumnTransformer(
        [("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan,
                                encoded_missing_value=np.nan), CATEGORICAL)],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )
    # The regressor predicts the *change* in delay; predict_delay adds it back.
    regressor = HistGradientBoostingRegressor(
        categorical_features=list(range(len(CATEGORICAL))),
        max_iter=300,
        learning_rate=0.05,
        early_stopping=True,
        random_state=random_state,
    )
    return Pipeline([("encode", encode), ("regressor", regressor)])


def time_split(data: pd.DataFrame, test_fraction: float,
               horizon_s: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train on the earliest observations and test on the latest.

    Training rows whose target (observed ``horizon_s`` later) falls after the cutoff
    are dropped, so no test-period information leaks into training.
    """
    cutoff = data["timestamp"].quantile(1 - test_fraction)
    horizon = pd.Timedelta(seconds=horizon_s)
    return data[data["timestamp"] + horizon < cutoff], data[data["timestamp"] >= cutoff]


def _scores(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "mae_s": round(float(mean_absolute_error(y_true, y_pred)), 2),
        "rmse_s": round(float(root_mean_squared_error(y_true, y_pred)), 2),
    }


def train(df: pd.DataFrame, horizon_s: float, tolerance_s: float, test_fraction: float = 0.25,
          min_rows: int = 200, random_state: int = 42) -> tuple[Pipeline, dict]:
    data = build_training_set(df, horizon_s, tolerance_s)
    if len(data) < min_rows:
        raise InsufficientDataError(
            f"only {len(data)} observations have a matching delay {horizon_s:.0f}s later "
            f"(need {min_rows}); collect more continuous data with `python -m tram_mlops.ingest`"
        )
    train_df, test_df = time_split(data, test_fraction, horizon_s)
    if train_df.empty or test_df.empty:
        raise InsufficientDataError("time split produced an empty train or test set")
    empty = [c for c in FEATURES if train_df[c].isna().all()]
    if empty:
        raise InsufficientDataError(
            f"features {empty} have no values in the training data; observations are too sparse or "
            f"short-lived (e.g. delay_change_60s needs each trip to be seen for over a minute)"
        )

    model = make_model(random_state)
    model.fit(train_df[FEATURES], train_df[TARGET] - train_df["delay_s"])
    predicted = test_df["delay_s"] + model.predict(test_df[FEATURES])

    metrics = {
        "horizon_s": horizon_s,
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "train_vehicles": int(train_df["vehicle"].nunique()),
        "test_from": str(test_df["timestamp"].min()),
        "test_to": str(test_df["timestamp"].max()),
        "model": _scores(test_df[TARGET], predicted),
        "persistence_baseline": _scores(test_df[TARGET], test_df["delay_s"]),
    }
    base_mae = metrics["persistence_baseline"]["mae_s"]
    metrics["mae_improvement_pct"] = (
        round(100 * (1 - metrics["model"]["mae_s"] / base_mae), 1) if base_mae else None
    )
    # Trams waiting at the terminus report a misleading "early" delay until they depart, which
    # persistence handles badly; report that segment separately so it can't flatter the headline.
    metrics["segments"] = {}
    for name, mask in segment_masks(test_df).items():
        if mask.any():
            metrics["segments"][name] = {
                "rows": int(mask.sum()),
                "model": _scores(test_df.loc[mask, TARGET], predicted[mask]),
                "persistence_baseline": _scores(test_df.loc[mask, TARGET], test_df.loc[mask, "delay_s"]),
            }
    return model, metrics


def segment_masks(data: pd.DataFrame) -> dict[str, pd.Series]:
    before = data["journey_elapsed_s"] < 0
    return {"en_route": ~before, "before_departure": before}


def save(model: Pipeline, metrics: dict, models_dir: Path) -> Path:
    version = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = models_dir / version
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out / "model.joblib")
    record = {"version": version, "features": FEATURES, **metrics}
    (out / "metrics.json").write_text(json.dumps(record, indent=2))
    (models_dir / "latest.json").write_text(json.dumps({"version": version}, indent=2))
    return out


def load_latest(models_dir: Path) -> tuple[Pipeline, dict] | None:
    pointer = models_dir / "latest.json"
    if not pointer.exists():
        return None
    version_dir = models_dir / json.loads(pointer.read_text())["version"]
    model = joblib.load(version_dir / "model.joblib")
    metrics = json.loads((version_dir / "metrics.json").read_text())
    return model, metrics


def predict_delay(model: Pipeline, df: pd.DataFrame) -> pd.Series:
    """Forecast delay_s at the model's horizon for each row of a cleaned positions table."""
    feats = add_features(df)
    return pd.Series(feats["delay_s"].to_numpy() + model.predict(feats[FEATURES]), index=df.index)


def run(horizon_s: float | None = None) -> dict:
    cfg = load_config()
    paths, model_cfg = cfg["paths"], cfg["model"]
    processed = resolve_path(paths["processed_file"])
    if not processed.exists():
        raise FileNotFoundError(f"{processed} not found; run `python -m tram_mlops.process` first")
    df = pd.read_parquet(processed)

    model, metrics = train(
        df,
        horizon_s=horizon_s or model_cfg["horizon_s"],
        tolerance_s=model_cfg["horizon_tolerance_s"],
        test_fraction=model_cfg["test_fraction"],
        min_rows=model_cfg["min_training_rows"],
        random_state=model_cfg["random_state"],
    )
    out = save(model, metrics, resolve_path(paths["models_dir"]))
    log.info("Saved model to %s", out)
    log.info("Model MAE %.1fs vs persistence baseline %.1fs (%s%% better)",
             metrics["model"]["mae_s"], metrics["persistence_baseline"]["mae_s"],
             metrics["mae_improvement_pct"])
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--horizon", type=float, help="forecast horizon in seconds (overrides config)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        run(args.horizon)
    except InsufficientDataError as exc:
        log.error("Not enough data to train: %s", exc)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
