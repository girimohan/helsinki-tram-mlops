"""Streamlit dashboard: ``streamlit run src/tram_mlops/dashboard.py``.

Live mode reads the newest raw records the ingester is writing and refreshes itself;
History mode aggregates the processed partitions with DuckDB (see ``history``).
"""

from __future__ import annotations

import time
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import pydeck as pdk
import streamlit as st

from tram_mlops import process, storage
from tram_mlops.config import load_config, resolve_path
from tram_mlops.history import History, available_days
from tram_mlops.train import load_latest, predict_delay

cfg = load_config()
PROCESSED_DIR = resolve_path(cfg["paths"]["processed_dir"])
MODELS_DIR = resolve_path(cfg["paths"]["models_dir"])
RAW_DIR = resolve_path(cfg["paths"]["raw_dir"])
LIVE = cfg["dashboard"]
ON_TIME_MIN, ON_TIME_MAX = cfg["process"]["on_time_min_s"], cfg["process"]["on_time_max_s"]

LATE = [227, 73, 72]
EARLY = [42, 120, 214]
ON_TIME = [163, 162, 156]


def _mtime(path: Path) -> float:
    return path.stat().st_mtime if path.exists() else 0.0


def ensure_processed() -> list[date]:
    """Days with processed data, building them from raw data (or the sample) if allowed and none exist."""
    days = available_days(PROCESSED_DIR)
    if not days and LIVE["process_on_demand"]:
        _process_once()
        days = available_days(PROCESSED_DIR)
    return days


@st.cache_resource
def _process_once() -> None:
    process.run()


# Cache keys include the manifest's mtime, so a processing run shows up immediately.
@st.cache_data(max_entries=32)
def history_data(start: date, end: date, lines: tuple[str, ...], version: float) -> dict[str, Any]:
    h = History(PROCESSED_DIR, start, end, list(lines), cfg["process"]["timezone"])
    summary = h.summary()
    if not summary["rows"]:
        return {"summary": summary}
    span = summary["to"] - summary["from"]
    bucket_s = 300 if span <= pd.Timedelta(hours=6) else 3600
    return {
        "summary": summary,
        "by_line": h.by_line(),
        "bucket_s": bucket_s,
        "over_time": h.over_time(bucket_s),
        "by_hour": h.by_hour(),
        "latest": h.latest_positions(),
        "recent": h.recent(6),
        "most_delayed": h.most_delayed(20),
    }


@st.cache_data(max_entries=4)
def lines_seen(start: date, end: date, version: float) -> list[str]:
    return History(PROCESSED_DIR, start, end).lines_seen()


@st.cache_resource
def load_model(models_dir: Path, mtime: float):
    return load_latest(models_dir)


def load_live() -> pd.DataFrame:
    raw = process.load_recent(RAW_DIR, pd.Timedelta(minutes=LIVE["live_window_min"]))
    return process.clean_with_config(raw)


def filter_lines(df: pd.DataFrame, lines: list[str]) -> pd.DataFrame:
    return df[df["line"].isin(lines)] if lines and not df.empty else df


def latest_per_tram(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values("timestamp").groupby("vehicle").tail(1)


def render_map(latest: pd.DataFrame, key: str) -> None:
    latest = latest.dropna(subset=["lat", "lon"]).copy()
    latest["color"] = [LATE if d > ON_TIME_MAX else EARLY if d < ON_TIME_MIN else ON_TIME
                       for d in latest["delay_s"]]
    latest["seen"] = latest["local_time"].dt.strftime("%H:%M:%S")
    st.caption(f":red[●] more than {ON_TIME_MAX // 60} min late · :gray[●] on time · "
               f":blue[●] more than {-ON_TIME_MIN // 60} min early — hover a tram for details")
    st.pydeck_chart(pdk.Deck(
        map_style=None,
        initial_view_state=pdk.ViewState(latitude=60.180, longitude=24.935, zoom=11.6),
        layers=[pdk.Layer(
            "ScatterplotLayer", latest, get_position=["lon", "lat"], get_fill_color="color",
            get_radius=60, radius_min_pixels=5, radius_max_pixels=12,
            stroked=True, get_line_color=[255, 255, 255], line_width_min_pixels=1, pickable=True,
        )],
        tooltip={"text": "Tram {vehicle} · line {line}\nDelay: {delay_s} s\nSeen: {seen}"},
    ), height=480, key=key)


def render_forecast(df: pd.DataFrame, limit: int | None = None) -> None:
    loaded = load_model(MODELS_DIR, _mtime(MODELS_DIR / "latest.json"))
    if loaded is None:
        st.info("No trained model yet. Run `python -m tram_mlops.train` once enough data is collected.")
        return
    model, metrics = loaded
    horizon_min = metrics["horizon_s"] / 60
    m1, m2, m3 = st.columns(3)
    m1.metric(f"Model MAE ({horizon_min:g} min ahead)", f"{metrics['model']['mae_s']:.0f} s")
    m2.metric("Persistence baseline MAE", f"{metrics['persistence_baseline']['mae_s']:.0f} s",
              help="Error from assuming the delay stays as it is now")
    m3.metric("Improvement", f"{metrics['mae_improvement_pct']} %")
    st.caption(f"Model {metrics['version']} · trained on {metrics['train_rows']:,} rows, "
               f"tested on {metrics['test_rows']:,}")

    # Features look back 60 s, so predict from each tram's recent history.
    recent = df[df["timestamp"] >= df["timestamp"].max() - pd.Timedelta(minutes=5)].copy()
    recent["forecast_delay_s"] = predict_delay(model, recent).round().astype(int)
    table = latest_per_tram(recent)
    table = table.assign(seen=table["local_time"].dt.strftime("%H:%M:%S"))
    table = table.sort_values("forecast_delay_s", ascending=False)
    if limit is not None:
        table = table.head(limit)
    st.dataframe(
        table[["vehicle", "line", "direction", "delay_s", "forecast_delay_s", "seen"]]
        .rename(columns={"vehicle": "Tram", "line": "Line", "direction": "Direction",
                         "delay_s": "Delay now (s)",
                         "forecast_delay_s": f"Forecast in {horizon_min:g} min (s)",
                         "seen": "Last seen"}),
        hide_index=True,
        width="stretch",
    )


def line_status(now: pd.DataFrame) -> pd.DataFrame:
    status = now.groupby("line").agg(
        trams=("vehicle", "nunique"),
        mean_delay=("delay_s", "mean"),
        on_time=("on_time", "mean"),
        worst=("delay_s", "max"),
    )
    status["mean_delay"] = (status["mean_delay"] / 60).round(1)
    status["worst"] = (status["worst"] / 60).round(1)
    status["on_time"] = (status["on_time"] * 100).round()
    return status.sort_values("mean_delay", ascending=False).reset_index()


def render_live(lines: list[str]) -> None:
    @st.fragment(run_every=f"{LIVE['refresh_s']}s")
    def live_panel() -> None:
        df = filter_lines(load_live(), lines)
        if df.empty:
            st.warning(f"No tram data from the last {LIVE['live_window_min']} min. "
                       "Start the collector with `python -m tram_mlops.ingest`.")
            return
        newest = df["timestamp"].max()
        age_s = (pd.Timestamp.now(tz="UTC") - newest).total_seconds()
        now = latest_per_tram(df[df["timestamp"] >= newest - pd.Timedelta(seconds=LIVE["in_service_s"])])

        st.caption(f"Updates every {LIVE['refresh_s']} s · newest record "
                   f"{newest.tz_convert(cfg['process']['timezone']):%H:%M:%S} ({age_s:.0f} s ago)")
        if age_s > LIVE["stale_after_s"]:
            st.warning(f"No new data for {age_s / 60:.0f} min — is the collector still running?")

        late = int((now["delay_s"] > ON_TIME_MAX).sum())
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Trams in service", len(now))
        c2.metric("On time now", f"{100 * now['on_time'].mean():.0f} %",
                  help=f"Between {-ON_TIME_MIN} s early and {ON_TIME_MAX} s late")
        c3.metric("Mean delay now", f"{now['delay_s'].mean():.0f} s", help="Positive = behind schedule")
        c4.metric(f"More than {ON_TIME_MAX // 60} min late", late)

        left, right = st.columns([3, 2])
        with left:
            st.subheader("Where the trams are now")
            render_map(now, key="live_map")
        with right:
            st.subheader("Line status")
            status = line_status(now)
            st.dataframe(
                status,
                hide_index=True,
                width="stretch",
                height=35 * (len(status) + 1) + 3,  # show every line without scrolling
                column_config={
                    "line": st.column_config.TextColumn("Line", width="small"),
                    "trams": st.column_config.NumberColumn("Trams", width="small"),
                    "mean_delay": st.column_config.NumberColumn("Mean delay", format="%.1f min", width="small"),
                    "on_time": st.column_config.ProgressColumn("On time", min_value=0, max_value=100,
                                                               format="%d%%", width="small"),
                    "worst": st.column_config.NumberColumn("Most late", format="%.1f min", width="small"),
                },
            )

        st.subheader("Delay forecast")
        render_forecast(df, limit=15)

    live_panel()


def render_history(lines: list[str], days: tuple[date, date] | None) -> None:
    if days is None:
        st.warning("No processed data yet. Run `python -m tram_mlops.process`.")
        return
    data = history_data(days[0], days[1], tuple(lines), _mtime(PROCESSED_DIR / storage.MANIFEST))
    summary = data["summary"]
    if not summary["rows"]:
        st.warning("No observations for these days and lines.")
        return

    st.caption(f"{summary['rows']:,} observations · {summary['from']:%Y-%m-%d %H:%M} → "
               f"{summary['to']:%Y-%m-%d %H:%M} (Helsinki time)")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Mean delay", f"{summary['mean_delay_s']:.0f} s", help="Positive = behind schedule")
    c2.metric("Median delay", f"{summary['median_delay_s']:.0f} s")
    c3.metric("On time", f"{summary['on_time_pct']:.0f} %",
              help=f"Between {-ON_TIME_MIN} s early and {ON_TIME_MAX} s late")
    c4.metric("Trams", summary["vehicles"])

    left, right = st.columns(2)
    with left:
        st.subheader("Mean delay by line (s)")
        st.bar_chart(data["by_line"], horizontal=True)
    with right:
        st.subheader("Mean delay over time (s)")
        st.line_chart(data["over_time"])
        st.caption(f"Averaged over {data['bucket_s'] // 60} min buckets, Helsinki time")

    st.subheader("By hour of day")
    by_hour = data["by_hour"]
    h_left, h_right = st.columns(2)
    with h_left:
        st.caption("Mean delay (s)")
        st.line_chart(by_hour.pivot(index="hour", columns="day_type", values="mean_delay_s").round())
    with h_right:
        st.caption(f"On time (%), between {-ON_TIME_MIN} s early and {ON_TIME_MAX} s late")
        st.line_chart(by_hour.pivot(index="hour", columns="day_type", values="on_time_pct").round(1))

    st.subheader("Latest positions")
    render_map(data["latest"], key="history_map")

    st.subheader("Delay forecast")
    if data["recent"].empty:
        st.info("No recent observations to forecast from.")
    else:
        render_forecast(data["recent"])

    st.subheader("Most delayed trams")
    st.dataframe(
        data["most_delayed"].rename(columns={"vehicle": "Tram", "line": "Line", "mean": "Mean delay (s)",
                                             "max": "Max delay (s)", "count": "Observations"}),
        hide_index=True,
        width="stretch",
    )


def pick_days(days: list[date]) -> tuple[date, date] | None:
    """Sidebar control for the History view's UTC day range."""
    if not days:
        return None
    default_start = max(days[0], days[-1] - timedelta(days=LIVE["history_days"] - 1))
    picked = st.sidebar.date_input("Days", value=(default_start, days[-1]), min_value=days[0],
                                   max_value=days[-1], help="Collection days (UTC)")
    # While a range is being picked, Streamlit returns only its first day.
    if isinstance(picked, tuple | list):
        return (picked[0], picked[-1]) if picked else (default_start, days[-1])
    return (picked, picked) if isinstance(picked, date) else (default_start, days[-1])


st.set_page_config(page_title="Helsinki Tram Delays", page_icon="🚋", layout="wide")
st.title("🚋 Helsinki Tram Delays")

raw_files = sorted(RAW_DIR.glob("tram_*.jsonl"))
live_available = bool(raw_files) and time.time() - _mtime(raw_files[-1]) < LIVE["stale_after_s"]
mode = st.sidebar.radio("View", ["Live", "History"], index=0 if live_available else 1,
                        help="Live reads the collector's newest data; History shows the processed dataset.")
processed_days = ensure_processed() if mode == "History" else available_days(PROCESSED_DIR)
day_range = pick_days(processed_days) if mode == "History" else None
known_lines: set[str] = set()
if processed_days:
    known_lines |= set(lines_seen(processed_days[-1] - timedelta(days=LIVE["history_days"]), processed_days[-1],
                                  _mtime(PROCESSED_DIR / storage.MANIFEST)))
if live_available:
    known_lines |= set(load_live().get("line", pd.Series(dtype="string")).dropna())
lines = st.sidebar.multiselect("Lines", sorted(known_lines, key=lambda s: (len(s), s)),
                               placeholder="All lines")

if mode == "Live":
    render_live(lines)
else:
    render_history(lines, day_range)
