"""Render the README figures from the processed data and the latest model.

Usage: python scripts/make_figures.py   (after `process` and `train`)
Writes PNGs to docs/images/.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tram_mlops.config import PROJECT_ROOT, load_config, resolve_path
from tram_mlops.features import FEATURES, TARGET, build_training_set
from tram_mlops.train import load_latest, time_split

OUT = PROJECT_ROOT / "docs" / "images"

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e4e3df"
BLUE = "#2a78d6"
BLUE_LIGHT = "#cde2fb"
NEUTRAL = "#a3a29c"

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID,
    "axes.labelcolor": TEXT_2,
    "axes.titlecolor": TEXT,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "axes.titlepad": 12,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "xtick.color": TEXT_2,
    "ytick.color": TEXT_2,
    "font.size": 10.5,
    "font.family": "sans-serif",
})


def _save(fig, name: str) -> None:
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=160)
    plt.close(fig)
    print(f"wrote docs/images/{name}")


def delay_distribution(df: pd.DataFrame, on_time: tuple[int, int]) -> None:
    minutes = df["delay_s"] / 60
    fig, ax = plt.subplots(figsize=(8, 3.8))
    lo, hi = on_time[0] / 60, on_time[1] / 60
    ax.axvspan(lo, hi, color=BLUE_LIGHT, alpha=0.6, lw=0, zorder=0)
    left = np.floor(minutes.quantile(0.005))
    right = np.ceil(minutes.quantile(0.995))
    ax.hist(minutes.clip(left, right), bins=np.arange(left, right + 0.5, 0.5), color=BLUE,
            edgecolor=SURFACE, linewidth=1, zorder=2)
    ax.set_xlim(left, right)
    share = df["delay_s"].between(*on_time).mean()
    ax.text(lo - 0.3, ax.get_ylim()[1] * 0.9, f"On-time window\n{share:.0%} of observations",
            color=TEXT_2, va="top", ha="right", fontsize=9.5)
    ax.set_title("How late are Helsinki trams?")
    ax.set_xlabel("Delay (minutes, positive = behind schedule)")
    ax.set_ylabel("Observations")
    ax.grid(axis="x", visible=False)
    _save(fig, "delay_distribution.png")


def delay_by_line(df: pd.DataFrame) -> None:
    by_line = df.groupby("line")["delay_s"].agg(["mean", "size"])
    by_line = by_line[by_line["size"] >= 50].sort_values("mean")
    fig, ax = plt.subplots(figsize=(8, 0.34 * len(by_line) + 1.2))
    ax.barh(by_line.index, by_line["mean"] / 60, color=BLUE, height=0.65, zorder=2)
    ax.axvline(0, color=TEXT_2, lw=1, zorder=3)
    for y, value in enumerate(by_line["mean"] / 60):
        ax.text(value + (0.05 if value >= 0 else -0.05), y, f"{value:+.1f}",
                va="center", ha="left" if value >= 0 else "right", color=TEXT_2, fontsize=9)
    ax.set_title("Mean delay by tram line")
    ax.set_xlabel("Mean delay (minutes, positive = behind schedule)")
    ax.set_ylabel("Line")
    ax.grid(axis="y", visible=False)
    ax.margins(x=0.15)
    _save(fig, "delay_by_line.png")


def model_evaluation(df: pd.DataFrame, model, metrics: dict, test_fraction: float) -> None:
    horizon = metrics["horizon_s"]
    data = build_training_set(df, horizon, load_config()["model"]["horizon_tolerance_s"])
    _, test = time_split(data, test_fraction, horizon)
    predicted = test["delay_s"] + model.predict(test[FEATURES])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2), gridspec_kw={"width_ratios": [1, 1.25]})

    # Error per segment: trams already moving vs. trams waiting at the terminus.
    names = {"en_route": "En route", "before_departure": "Before departure"}
    segments = [(names[k], v) for k, v in metrics["segments"].items()]
    x = np.arange(len(segments))
    width = 0.36
    series = [("Persistence baseline", "persistence_baseline", NEUTRAL, -1),
              ("Gradient boosting", "model", BLUE, 1)]
    top_value = 0
    for label, key, color, side in series:
        values = [seg[key]["mae_s"] for _, seg in segments]
        top_value = max(top_value, *values)
        bars = ax1.bar(x + side * (width / 2 + 0.01), values, width, color=color, label=label, zorder=2)
        for bar, value in zip(bars, values):
            ax1.text(bar.get_x() + bar.get_width() / 2, value + 3, f"{value:.0f} s",
                     ha="center", va="bottom", color=TEXT, fontsize=9.5)
    ax1.set_xticks(x, [f"{name}\n({seg['rows']:,} test rows)" for name, seg in segments])
    ax1.set_title(f"Forecast error, {horizon / 60:g} min ahead")
    ax1.set_ylabel("Mean absolute error (s)")
    ax1.grid(axis="x", visible=False)
    ax1.set_ylim(0, top_value * 1.2)
    ax1.legend(frameon=False, loc="upper left", fontsize=9, labelcolor=TEXT_2)

    actual_min, predicted_min = test[TARGET] / 60, predicted / 60
    lim = np.percentile(np.abs(np.concatenate([actual_min, predicted_min])), 99.5) * 1.05
    ax2.plot([-lim, lim], [-lim, lim], color=TEXT_2, lw=1, ls="--", zorder=1, label="Perfect forecast")
    ax2.scatter(actual_min, predicted_min, s=10, color=BLUE, alpha=0.35, lw=0, zorder=2,
                label="Test observation")
    ax2.set_xlim(-lim, lim)
    ax2.set_ylim(-lim, lim)
    ax2.set_aspect("equal")
    ax2.set_title("Forecast vs. actual delay")
    ax2.set_xlabel("Actual delay (min)")
    ax2.set_ylabel("Forecast delay (min)")
    ax2.legend(frameon=False, loc="upper left", fontsize=9, labelcolor=TEXT_2)
    _save(fig, "model_evaluation.png")


def main() -> None:
    cfg = load_config()
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(resolve_path(cfg["paths"]["processed_file"]))
    delay_distribution(df, (cfg["process"]["on_time_min_s"], cfg["process"]["on_time_max_s"]))
    delay_by_line(df)
    loaded = load_latest(resolve_path(cfg["paths"]["models_dir"]))
    if loaded is None:
        print("no trained model; skipping model_evaluation.png")
        return
    model_evaluation(df, *loaded, test_fraction=cfg["model"]["test_fraction"])


if __name__ == "__main__":
    main()
