"""Rolling-origin backtest (roadmap E5).

Run with ``python -m tram_mlops.evaluate [--train-days K] [--update-readme]``. For each
collected UTC day D that has data before it:

- features are built over days D-K..D together, so trips crossing midnight keep their history
- the model trains on rows whose 5-minute target falls before D starts (purged) and is
  tested on every row of D
- two baselines are scored on the same rows: *persistence* (the delay stays as it is) and
  *line x hour*: the mean delay change for the row's line, hour of day and segment in the
  training rows, a much stronger reference than persistence

The primary metric is MAE for trams already en route. Differences between the model and
each baseline get 95% intervals from a bootstrap that resamples whole trips, since rows
from one trip are strongly correlated. Results go to ``reports/backtest/<run>.json`` and a
Markdown summary; ``--update-readme`` writes the summary into README.md.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from tram_mlops import storage
from tram_mlops.config import PROJECT_ROOT, load_config, resolve_path
from tram_mlops.features import FEATURES, TARGET, build_training_set
from tram_mlops.train import DEFAULT_LOSS, fit, segment_masks

log = logging.getLogger(__name__)

METHODS = ("model", "persistence", "line_hour")
LABELS = {"model": "Model", "persistence": "Persistence", "line_hour": "Line × hour"}
DELAY_BINS = [-np.inf, -60, 0, 60, 180, 300, np.inf]
DELAY_LABELS = ["> 1 min early", "0–1 min early", "0–1 min late", "1–3 min late", "3–5 min late", "> 5 min late"]
README_START, README_END = "<!-- backtest:start -->", "<!-- backtest:end -->"


# --- baselines -------------------------------------------------------------------------

def _segment(data: pd.DataFrame) -> pd.Series:
    return pd.Series(np.where(segment_masks(data)["en_route"], "en_route", "before_departure"), index=data.index)


def line_hour_baseline(train: pd.DataFrame, test: pd.DataFrame, min_rows: int = 30) -> pd.Series:
    """Delay now + the mean delay change seen in training for the same line, hour and segment.

    Groups with fewer than ``min_rows`` training rows fall back to the segment's mean change.
    """
    keys = ["line", "hour", "segment"]
    tr = train.assign(segment=_segment(train), change=train[TARGET] - train["delay_s"])
    stats = tr.groupby(keys)["change"].agg(["mean", "size"])
    group_change: pd.Series = stats.loc[stats["size"] >= min_rows, "mean"].rename("group_change")
    fallback = tr.groupby("segment")["change"].mean()
    te = test.assign(segment=_segment(test))
    change = te.join(group_change, on=keys)["group_change"]
    change = change.fillna(te["segment"].map(fallback)).fillna(0.0)
    return test["delay_s"] + change


# --- one test day ----------------------------------------------------------------------

def backtest_day(positions: pd.DataFrame, day: date, horizon_s: float, tolerance_s: float,
                 random_state: int = 42, loss: str = DEFAULT_LOSS) -> pd.DataFrame | None:
    """Predictions of every method for every testable row of ``day``, or None if untestable."""
    data = build_training_set(positions, horizon_s, tolerance_s)
    start = pd.Timestamp(day, tz="UTC")
    end = start + pd.Timedelta(days=1)
    train = data[data["timestamp"] + pd.Timedelta(seconds=horizon_s) < start]
    test = data[(data["timestamp"] >= start) & (data["timestamp"] < end)]
    if train.empty or test.empty:
        return None
    model = fit(train, random_state, loss)
    out = test[["timestamp", "vehicle", "journey_id", "line", "hour", "weekday", "delay_s", TARGET]].copy()
    out["segment"] = _segment(test)
    out["model"] = test["delay_s"] + model.predict(test[FEATURES])
    out["persistence"] = test["delay_s"].astype(float)
    out["line_hour"] = line_hour_baseline(train, test)
    out["test_day"] = day.isoformat()
    out["train_rows"] = len(train)
    return out


# --- scoring ---------------------------------------------------------------------------

def mae(pred: pd.DataFrame) -> dict[str, float]:
    return {m: round(float((pred[m] - pred[TARGET]).abs().mean()), 2) for m in METHODS}


def bootstrap_differences(pred: pd.DataFrame, reps: int = 1000, seed: int = 0) -> dict[str, dict[str, float]]:
    """95% intervals for MAE(model) - MAE(baseline), resampling whole trips (Poisson bootstrap)."""
    trips = pred.assign(trip=pred["vehicle"].astype(str) + "#" + pred["journey_id"].astype(str))
    errors = {m: (trips[m] - trips[TARGET]).abs() for m in METHODS}
    per_trip = pd.DataFrame({"n": trips.groupby("trip").size(),
                             **{m: errors[m].groupby(trips["trip"]).sum() for m in METHODS}})
    weights = np.random.default_rng(seed).poisson(1.0, size=(reps, len(per_trip))).astype(np.float32)
    n = weights @ per_trip["n"].to_numpy(np.float32)
    boot = {m: (weights @ per_trip[m].to_numpy(np.float32)) / n for m in METHODS}
    point = mae(pred)
    out = {}
    for baseline in ("persistence", "line_hour"):
        diff = boot["model"] - boot[baseline]
        out[baseline] = {
            "diff_s": round(point["model"] - point[baseline], 2),
            "ci_low_s": round(float(np.percentile(diff, 2.5)), 2),
            "ci_high_s": round(float(np.percentile(diff, 97.5)), 2),
            "trips": len(per_trip),
        }
    return out


def breakdown(pred: pd.DataFrame, by: pd.Series | np.ndarray, min_rows: int = 200) -> list[dict[str, Any]]:
    rows = []
    for key, group in pred.groupby(by, observed=True):
        if len(group) >= min_rows:
            rows.append({"group": str(key), "rows": len(group), **mae(group)})
    return rows


def summarize(pred: pd.DataFrame, reps: int = 1000) -> dict[str, Any]:
    en_route = pred[pred["segment"] == "en_route"]
    summary: dict[str, Any] = {
        "rows": len(pred),
        "trips": int((pred["vehicle"].astype(str) + pred["journey_id"].astype(str)).nunique()),
        "overall": mae(pred),
        "segments": {seg: {"rows": len(g), **mae(g)} for seg, g in pred.groupby("segment")},
    }
    if not en_route.empty:
        summary["en_route_vs_baselines"] = bootstrap_differences(en_route, reps)
        summary["breakdowns"] = {
            "hour": breakdown(en_route, en_route["hour"]),
            "line": breakdown(en_route, en_route["line"]),
            "day_type": breakdown(en_route, np.where(en_route["weekday"] >= 5, "weekend", "weekday")),
            "delay_now": breakdown(en_route, pd.cut(en_route["delay_s"], DELAY_BINS, labels=DELAY_LABELS)),
        }
    return summary


# --- run -------------------------------------------------------------------------------

def run(train_days: int | None = None, reps: int = 1000, out_dir: Path | None = None) -> dict[str, Any]:
    cfg = load_config()
    model_cfg, eval_cfg = cfg["model"], cfg["evaluate"]
    root = resolve_path(cfg["paths"]["processed_dir"])
    train_days = train_days or eval_cfg["train_days"]
    days = storage.list_partitions(root)
    if len(days) < 2:
        raise ValueError(f"need at least two days of processed data in {root}, found {len(days)}")

    per_day, predictions = [], []
    for day in days[1:]:
        positions = storage.read_positions(root, day - timedelta(days=train_days), day)
        pred = backtest_day(positions, day, model_cfg["horizon_s"], model_cfg["horizon_tolerance_s"],
                            model_cfg["random_state"], model_cfg.get("loss", DEFAULT_LOSS))
        del positions
        if pred is None or len(pred) < eval_cfg["min_test_rows"]:
            log.info("%s: not enough data to test", day)
            continue
        scores = summarize(pred, reps=200)
        per_day.append({"day": day.isoformat(), "train_rows": int(pred["train_rows"].iloc[0]),
                        "test_rows": len(pred), "segments": scores["segments"],
                        "en_route_vs_baselines": scores.get("en_route_vs_baselines")})
        log.info("%s: %d test rows, en-route MAE %s", day, len(pred), scores["segments"].get("en_route"))
        keep = ["timestamp", "vehicle", "journey_id", "line", "hour", "weekday", "delay_s", TARGET,
                "segment", *METHODS]
        predictions.append(pred[keep])
    if not predictions:
        raise ValueError("no day had enough data to test")

    pooled = pd.concat(predictions, ignore_index=True)
    report = {
        "run": datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ"),
        "horizon_s": model_cfg["horizon_s"],
        "loss": model_cfg.get("loss", DEFAULT_LOSS),
        "train_days": train_days,
        "test_days": [d["day"] for d in per_day],
        "data_from": days[0].isoformat(),
        "data_to": days[-1].isoformat(),
        "pooled": summarize(pooled, reps),
        "per_day": per_day,
    }
    out_dir = out_dir or PROJECT_ROOT / "reports" / "backtest"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{report['run']}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out_dir / "latest.md").write_text(render_markdown(report), encoding="utf-8")
    log.info("Wrote %s", out_dir / f"{report['run']}.json")
    return report


# --- report ----------------------------------------------------------------------------

def _ci(entry: dict[str, float]) -> str:
    return f"{entry['diff_s']:+.1f} s [{entry['ci_low_s']:+.1f}, {entry['ci_high_s']:+.1f}]"


def render_markdown(report: dict[str, Any]) -> str:
    pooled = report["pooled"]
    seg = pooled["segments"]
    lines = [
        f"Backtest `{report['run']}`: {len(report['test_days'])} test day(s) "
        f"({', '.join(report['test_days'])}), each trained on up to {report['train_days']} earlier day(s); "
        f"{pooled['rows']:,} test rows from {pooled['trips']:,} trips. Generated by `python -m tram_mlops.evaluate`.",
        "",
        "| 5‑min‑ahead MAE | Rows | Persistence | Line × hour | **Model** |",
        "|---|---:|---:|---:|---:|",
        f"| All observations | {pooled['rows']:,} | {pooled['overall']['persistence']:.1f} s | "
        f"{pooled['overall']['line_hour']:.1f} s | **{pooled['overall']['model']:.1f} s** |",
    ]
    for key, label in (("en_route", "En route (tram has departed)"),
                       ("before_departure", "Before departure (waiting at terminus)")):
        if key in seg:
            s = seg[key]
            lines.append(f"| {label} | {s['rows']:,} | {s['persistence']:.1f} s | {s['line_hour']:.1f} s | "
                         f"**{s['model']:.1f} s** |")
    if "en_route_vs_baselines" in pooled:
        diffs = pooled["en_route_vs_baselines"]
        lines += [
            "",
            "En route, model minus baseline with a 95% interval (bootstrap over "
            f"{diffs['persistence']['trips']:,} trips; negative means the model is better):",
            f"vs persistence **{_ci(diffs['persistence'])}**, vs line × hour **{_ci(diffs['line_hour'])}**.",
        ]
    if len(report["per_day"]) > 1:
        lines += ["", "| Test day | Train rows | Test rows | En-route MAE: persistence / line × hour / model |",
                  "|---|---:|---:|---|"]
        for d in report["per_day"]:
            e = d["segments"].get("en_route", {})
            lines.append(f"| {d['day']} | {d['train_rows']:,} | {d['test_rows']:,} | "
                         f"{e.get('persistence', float('nan')):.1f} / {e.get('line_hour', float('nan')):.1f} / "
                         f"{e.get('model', float('nan')):.1f} s |")
    return "\n".join(lines) + "\n"


def update_readme(markdown: str, readme: Path | None = None) -> None:
    readme = readme or PROJECT_ROOT / "README.md"
    text = readme.read_text(encoding="utf-8")
    pattern = re.compile(re.escape(README_START) + r".*?" + re.escape(README_END), re.S)
    if not pattern.search(text):
        raise ValueError(f"{readme} has no {README_START} ... {README_END} block")
    readme.write_text(pattern.sub(lambda _: f"{README_START}\n{markdown}{README_END}", text), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train-days", type=int, help="train on up to this many days before each test day")
    parser.add_argument("--reps", type=int, default=1000, help="bootstrap repetitions")
    parser.add_argument("--update-readme", action="store_true", help="write the summary into README.md")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    report = run(args.train_days, args.reps)
    markdown = render_markdown(report)
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]  # Windows consoles default to cp1252
    print(markdown)
    if args.update_readme:
        update_readme(markdown)


if __name__ == "__main__":
    main()
