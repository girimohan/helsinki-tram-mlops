"""Streamlit dashboard: ``streamlit run src/tram_mlops/dashboard.py``.

Live mode reads the newest raw records the ingester is writing and refreshes itself;
History mode shows the processed dataset.
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import pydeck as pdk
import streamlit as st

from tram_mlops import process
from tram_mlops.config import load_config, resolve_path
from tram_mlops.train import load_latest, predict_delay

cfg = load_config()
PROCESSED = resolve_path(cfg["paths"]["processed_file"])
MODELS_DIR = resolve_path(cfg["paths"]["models_dir"])
RAW_DIR = resolve_path(cfg["paths"]["raw_dir"])
LIVE = cfg["dashboard"]
ON_TIME_MIN, ON_TIME_MAX = cfg["process"]["on_time_min_s"], cfg["process"]["on_time_max_s"]

LATE = [227, 73, 72]
EARLY = [42, 120, 214]
ON_TIME = [163, 162, 156]


def _mtime(path: Path) -> float:
    return path.stat().st_mtime if path.exists() else 0.0


# Cache keys include file mtimes so freshly processed data or a new model shows up immediately.
@st.cache_data
def load_positions(path: Path, mtime: float) -> pd.DataFrame:
    if not path.exists():
        return process.run(output=path)
    return pd.read_parquet(path)


@st.cache_resource
def load_model(models_dir: Path, mtime: float):
    return load_latest(models_dir)


def load_live() -> pd.DataFrame:
    raw = process.load_recent(RAW_DIR, pd.Timedelta(minutes=LIVE["live_window_min"]))
    return process.clean_with_config(raw)


def filter_lines(df: pd.DataFrame, lines: list[str]) -> pd.DataFrame:
    return df[df["line"].isin(lines)] if lines else df


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
    st.dataframe(
        table[["vehicle", "line", "direction", "delay_s", "forecast_delay_s", "seen"]].head(limit)
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


def render_history(lines: list[str]) -> None:
    df = load_positions(PROCESSED, _mtime(PROCESSED))
    view = filter_lines(df, lines)
    if view.empty:
        st.warning("No processed data yet. Run `python -m tram_mlops.process`.")
        return

    summary = process.summarize(view)
    st.caption(f"{summary['rows']:,} observations · {summary['from'][:16]} → {summary['to'][:16]} "
               "(Helsinki time)")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Mean delay", f"{summary['mean_delay_s']:.0f} s", help="Positive = behind schedule")
    c2.metric("Median delay", f"{summary['median_delay_s']:.0f} s")
    c3.metric("On time", f"{summary['on_time_pct']:.0f} %",
              help=f"Between {-ON_TIME_MIN} s early and {ON_TIME_MAX} s late")
    c4.metric("Trams", summary["vehicles"])

    left, right = st.columns(2)
    with left:
        st.subheader("Mean delay by line (s)")
        st.bar_chart(view.groupby("line")["delay_s"].mean().round(), horizontal=True)
    with right:
        span = view["local_time"].max() - view["local_time"].min()
        freq = "5min" if span <= pd.Timedelta(hours=6) else "1h"
        st.subheader("Mean delay over time (s)")
        over_time = view.set_index("local_time")["delay_s"].resample(freq).mean().round()
        over_time.index = over_time.index.tz_localize(None)  # plot in Helsinki wall-clock time
        st.line_chart(over_time)
        st.caption(f"Averaged over {freq} buckets, Helsinki time")

    st.subheader("Latest positions")
    render_map(latest_per_tram(view), key="history_map")

    st.subheader("Delay forecast")
    render_forecast(view)

    st.subheader("Most delayed trams")
    st.dataframe(
        view.groupby(["vehicle", "line"])["delay_s"].agg(["mean", "max", "count"]).round(0).astype(int)
        .sort_values("mean", ascending=False).head(20).reset_index()
        .rename(columns={"vehicle": "Tram", "line": "Line", "mean": "Mean delay (s)",
                         "max": "Max delay (s)", "count": "Observations"}),
        hide_index=True,
        width="stretch",
    )


st.set_page_config(page_title="Helsinki Tram Delays", page_icon="🚋", layout="wide")
st.title("🚋 Helsinki Tram Delays")

raw_files = sorted(RAW_DIR.glob("tram_*.jsonl"))
live_available = bool(raw_files) and time.time() - _mtime(raw_files[-1]) < LIVE["stale_after_s"]
mode = st.sidebar.radio("View", ["Live", "History"], index=0 if live_available else 1,
                        help="Live reads the collector's newest data; History shows the processed dataset.")
known_lines: set[str] = set()
if PROCESSED.exists():
    known_lines |= set(load_positions(PROCESSED, _mtime(PROCESSED))["line"].dropna())
if live_available:
    known_lines |= set(load_live().get("line", pd.Series(dtype="string")).dropna())
lines = st.sidebar.multiselect("Lines", sorted(known_lines, key=lambda s: (len(s), s)),
                               placeholder="All lines")

if mode == "Live":
    render_live(lines)
else:
    render_history(lines)
