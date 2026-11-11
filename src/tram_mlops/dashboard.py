"""Streamlit dashboard: ``streamlit run src/tram_mlops/dashboard.py``.

Pages (``st.navigation`` at the top; each is one ``*_page`` function, and only the open
one runs):

- **Live** reads the newest raw records the collector is writing and refreshes itself,
  with HSL's service alerts and cancellations for each line.
- **History** aggregates the processed partitions with DuckDB (see ``history``).
- **Model health** shows live accuracy: every forecast the forecaster logs is scored
  against what happened 5 minutes later, next to persistence (see ``monitoring``).
- **System** shows how fresh and complete every data source is, what each processing
  run kept and dropped, and which model is deployed (see ``status``).

To add a page, write a ``*_page()`` function and add it to ``pages()``.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import altair as alt
import pandas as pd
import pydeck as pdk
import streamlit as st

from tram_mlops import alerts, feeds, forecaster, monitoring, process, status, storage
from tram_mlops.config import load_config, resolve_path
from tram_mlops.history import History, available_days
from tram_mlops.train import load_latest, predict_delay

cfg = load_config()
PROCESSED_DIR = resolve_path(cfg["paths"]["processed_dir"])
MODELS_DIR = resolve_path(cfg["paths"]["models_dir"])
RAW_DIR = resolve_path(cfg["paths"]["raw_dir"])
GTFSRT_DIR = resolve_path(cfg["paths"]["gtfsrt_dir"])
PREDICTIONS_DIR = resolve_path(cfg["paths"]["predictions_dir"])
# The forecaster's snapshot counts as current if it is at most this old.
FORECAST_FRESH_S = 120
LIVE = cfg["dashboard"]
TZ = cfg["process"]["timezone"]
ON_TIME_MIN, ON_TIME_MAX = cfg["process"]["on_time_min_s"], cfg["process"]["on_time_max_s"]
REPO_URL = "https://github.com/girimohan/helsinki-tram-mlops"

# Punctuality colours, used by the maps and the charts alike (a blue-red diverging pair with a neutral midpoint).
LATE = [227, 73, 72]
EARLY = [42, 120, 214]
ON_TIME = [163, 162, 156]
PUNCTUALITY = {"Early": "#2a78d6", "On time": "#a3a29c", "Late": "#e34948"}
# Sequential blue ramp, light (low) to dark (high), for magnitudes such as on-time share.
SEQUENTIAL = ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]

EVENT_TYPES = {
    "ARR": "Arrived within a stop's radius",
    "ARS": "Arrived at a stop (stopped)",
    "PDE": "Ready to depart",
    "DEP": "Departed a stop",
    "PAS": "Passed a stop without stopping",
    "VJA": "Signed in to a journey",
    "VJOUT": "Signed out of a journey",
    "TLR": "Traffic-light priority request",
    "TLA": "Traffic-light priority response",
}


def _mtime(path: Path) -> float:
    return path.stat().st_mtime if path.exists() else 0.0


def manifest_version() -> tuple[str, float]:
    """Cache key for anything read from the processed partitions: changes with every processing run."""
    return str(PROCESSED_DIR), _mtime(PROCESSED_DIR / storage.MANIFEST)


def helsinki(ts: pd.Timestamp) -> pd.Timestamp:
    return ts.tz_convert(TZ)


def ago(ts: pd.Timestamp | None, now: pd.Timestamp) -> str:
    if ts is None:
        return "never"
    s = max(0.0, (now - ts).total_seconds())
    if s < 120:
        return f"{s:.0f} s ago"
    if s < 7200:
        return f"{s / 60:.0f} min ago"
    if s < 172800:
        return f"{s / 3600:.0f} h ago"
    return f"{s / 86400:.0f} d ago"


def punctuality(delay_s: float) -> str:
    return "Late" if delay_s > ON_TIME_MAX else "Early" if delay_s < ON_TIME_MIN else "On time"


def color_of(delay_s: float) -> list[int]:
    return {"Late": LATE, "Early": EARLY, "On time": ON_TIME}[punctuality(delay_s)]


# --- data access (cached) --------------------------------------------------------------

def ensure_processed() -> list[date]:
    """Days with processed data, building them from raw data (or the sample) if allowed and none exist."""
    days = available_days(PROCESSED_DIR)
    if not days and LIVE["process_on_demand"]:
        _process_once(str(PROCESSED_DIR))
        days = available_days(PROCESSED_DIR)
    return days


@st.cache_resource
def _process_once(processed_dir: str) -> None:
    """Build the processed dataset at most once per process and directory."""
    process.run()


# Cache keys include the manifest's mtime, so a processing run shows up immediately.
@st.cache_data(max_entries=32)
def history_data(start: date, end: date, lines: tuple[str, ...], version: tuple[str, float]) -> dict[str, Any]:
    h = History(PROCESSED_DIR, start, end, list(lines), TZ)
    summary = h.summary()
    if not summary["rows"]:
        return {"summary": summary}
    span = summary["to"] - summary["from"]
    bucket_s = 300 if span <= pd.Timedelta(hours=6) else 3600
    return {
        "summary": summary,
        "by_line": h.by_line(ON_TIME_MAX),
        "line_hour": h.line_hour(),
        "by_day": h.by_day(),
        "histogram": h.delay_histogram(HISTOGRAM_BIN_S),
        "bucket_s": bucket_s,
        "over_time": h.over_time(bucket_s),
        "by_hour": h.by_hour(),
        "latest": h.latest_positions(),
        "recent": h.recent(6),
        "most_delayed": h.most_delayed(20),
    }


@st.cache_data(max_entries=4)
def lines_seen(start: date, end: date, version: tuple[str, float]) -> list[str]:
    return History(PROCESSED_DIR, start, end).lines_seen()


@st.cache_data(max_entries=2)
def network(version: tuple[str, float]) -> tuple[dict[str, str], dict[str, list[str]]]:
    """Tram route -> line and stop -> lines, from the newest processed day."""
    days = available_days(PROCESSED_DIR)
    return History(PROCESSED_DIR, days[-1], days[-1]).network() if days else ({}, {})


def alerts_version() -> tuple[str, float]:
    files = alerts.snapshot_files(GTFSRT_DIR)
    return (str(files[0]), _mtime(files[0])) if files else (str(GTFSRT_DIR), 0.0)


@st.cache_data(max_entries=2)
def all_alerts(version: tuple[str, float]) -> tuple[list[alerts.Alert], str | None]:
    """Every alert in HSL's feed (all modes) and when it was fetched."""
    snapshot = alerts.latest_snapshot(GTFSRT_DIR)
    return alerts.parse(snapshot), (snapshot or {}).get("fetched_at")


# HSL's trip updates change every poll; a few minutes' delay is fine for a daily count.
@st.cache_data(ttl=300, max_entries=2)
def trips_by_route(gtfsrt_dir: Path, day: date) -> pd.DataFrame:
    return feeds.trips_by_route(gtfsrt_dir, day)


@st.cache_data(ttl=300, max_entries=8)
def stop_arrivals(raw_dir: Path, lines: tuple[str, ...], hours: int) -> pd.DataFrame:
    since = pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=hours)
    return feeds.stop_arrivals(raw_dir, since, list(lines), (ON_TIME_MIN, ON_TIME_MAX))


@st.cache_data(ttl=30, max_entries=2)
def source_statuses(raw_dir: Path, gtfsrt_dir: Path, today: date) -> list[status.Source]:
    trip_updates = status.trip_updates_source(gtfsrt_dir, today, LIVE["trip_updates_stale_after_s"])
    trips = trips_by_route(gtfsrt_dir, feeds.service_day(pd.Timestamp.now(tz="UTC"), TZ))
    if not trips.empty:
        trip_updates.note = (f"{int(trips['trips'].sum()):,} tram trips today, "
                             f"{int(trips['cancelled'].sum())} cancelled")
    return [
        status.hfp_source("Vehicle positions", "HSL HFP over MQTT; one record per tram every 10 s",
                          raw_dir, "tram", today, LIVE["stale_after_s"]),
        status.hfp_source("Stop and journey events", "HSL HFP over MQTT; every event, unsampled",
                          raw_dir, "events", today, LIVE["events_stale_after_s"]),
        trip_updates,
        status.alerts_source(gtfsrt_dir, today),
    ]


@st.cache_data(ttl=60, max_entries=2)
def event_counts(path: Path) -> pd.DataFrame:
    return status.event_type_counts(path)


@st.cache_resource
def load_model(models_dir: Path, mtime: float):
    return load_latest(models_dir)


def load_live() -> pd.DataFrame:
    raw = process.load_recent(RAW_DIR, pd.Timedelta(minutes=LIVE["live_window_min"]))
    return process.clean_with_config(raw)


def live_available() -> bool:
    raw_files = sorted(RAW_DIR.glob("tram_*.jsonl"))
    return bool(raw_files) and time.time() - _mtime(raw_files[-1]) < LIVE["stale_after_s"]


def filter_lines(df: pd.DataFrame, lines: list[str]) -> pd.DataFrame:
    return df[df["line"].isin(lines)] if lines and not df.empty else df


def latest_per_tram(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values("timestamp").groupby("vehicle").tail(1)


def tram_network(live: pd.DataFrame) -> tuple[dict[str, str], dict[str, set[str]]]:
    """Route -> line and stop -> lines, from the newest processed day plus what is running now."""
    route_lines, stop_lines_seen = network(manifest_version())
    route_lines = dict(route_lines)
    stop_lines = {stop: set(lines) for stop, lines in stop_lines_seen.items()}
    if live.empty or "line" not in live.columns:
        return route_lines, stop_lines
    if "route" in live.columns:
        pairs = live.dropna(subset=["route", "line"]).drop_duplicates("route", keep="last")
        route_lines.update(zip(pairs["route"].astype(str), pairs["line"].astype(str), strict=True))
    if "stop" in live.columns:
        at_stops = live.dropna(subset=["stop", "line"]).drop_duplicates(["stop", "line"])
        for stop, line in zip(at_stops["stop"], at_stops["line"], strict=True):
            stop_lines.setdefault(str(int(stop)), set()).add(str(line))
    return route_lines, stop_lines


# --- shared widgets --------------------------------------------------------------------

def pick_lines(known: set[str]) -> list[str]:
    """Sidebar line filter, kept while moving between pages."""
    saved = st.session_state.get("lines", [])
    if "lines_widget" not in st.session_state:
        st.session_state["lines_widget"] = saved
    options = sorted(known | set(saved), key=alerts.line_sort_key)
    picked = st.sidebar.multiselect("Lines", options, key="lines_widget", placeholder="All lines",
                                    help="Show only these tram lines")
    st.session_state["lines"] = picked
    return picked


def known_lines(processed_days: list[date], live: bool) -> set[str]:
    lines: set[str] = set()
    if processed_days:
        lines |= set(lines_seen(processed_days[-1] - timedelta(days=LIVE["history_days"]), processed_days[-1],
                                manifest_version()))
    if live:
        lines |= set(load_live().get("line", pd.Series(dtype="string")).dropna())
    return lines


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


def render_deck(df: pd.DataFrame, key: str, tooltip: str, radius: int = 60) -> None:
    """Points on a Helsinki map; ``df`` needs ``lat``, ``lon`` and an RGB ``color`` per row."""
    st.pydeck_chart(pdk.Deck(
        map_style=None,
        initial_view_state=pdk.ViewState(latitude=60.180, longitude=24.935, zoom=11.6),
        layers=[pdk.Layer(
            "ScatterplotLayer", df, get_position=["lon", "lat"], get_fill_color="color",
            get_radius=radius, radius_min_pixels=4, radius_max_pixels=12,
            stroked=True, get_line_color=[255, 255, 255], line_width_min_pixels=1, pickable=True,
        )],
        tooltip={"text": tooltip},
    ), height=480, key=key)


def legend_caption(what: str) -> None:
    st.caption(f":red[●] more than {ON_TIME_MAX // 60} min late · :gray[●] on time · "
               f":blue[●] more than {-ON_TIME_MIN // 60} min early — {what}")


def render_map(latest: pd.DataFrame, key: str) -> None:
    latest = latest.dropna(subset=["lat", "lon"]).copy()
    latest["color"] = [color_of(d) for d in latest["delay_s"]]
    latest["seen"] = latest["local_time"].dt.strftime("%H:%M:%S")
    legend_caption("hover a tram for details")
    render_deck(latest, key, "Tram {vehicle} · line {line}\nDelay: {delay_s} s\nSeen: {seen}")


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


@st.cache_data(ttl=60, max_entries=2)
def scored_forecasts(predictions_dir: Path, today: date) -> pd.DataFrame:
    """The forecaster's scored forecasts from yesterday and today (UTC)."""
    return monitoring.read_scored(predictions_dir, today - timedelta(days=1), today)


def render_live_forecast(lines: list[str], now: pd.Timestamp) -> None:
    """The forecaster service's current forecasts, with its live accuracy over the last hour.

    Without a running forecaster (e.g. a local checkout), the dashboard forecasts itself.
    """
    snapshot = forecaster.read_latest(PREDICTIONS_DIR)
    issued = pd.Timestamp(snapshot["issued_at"]) if snapshot else None
    if snapshot is None or issued is None or (now - issued).total_seconds() > FORECAST_FRESH_S:
        df = filter_lines(load_live(), lines)
        if not df.empty:
            render_forecast(df, limit=15)
        return

    horizon_min = snapshot["horizon_s"] / 60
    hour = monitoring.live_metrics(scored_forecasts(PREDICTIONS_DIR, now.date()), now)["1h"]
    en_route = hour.get("segments", {}).get("en_route", {})
    if en_route.get("rows"):
        m1, m2, m3 = st.columns(3)
        m1.metric(f"Live error, last hour ({horizon_min:g} min ahead)", f"{en_route['model_mae_s']:.0f} s",
                  help="Mean absolute error of forecasts for trams already en route, scored against "
                       "the delay they actually had 5 minutes later")
        m2.metric("Persistence, same forecasts", f"{en_route['persistence_mae_s']:.0f} s",
                  help="Error from assuming the delay stays as it is now")
        m3.metric("Improvement", f"{en_route['improvement_pct']} %")

    table = pd.DataFrame(snapshot["forecasts"])
    if lines and not table.empty:
        table = table[table["line"].isin(lines)]
    if table.empty:
        st.info("No forecasts for these lines right now.")
        return
    table["seen"] = pd.to_datetime(table["observed_at"], utc=True).dt.tz_convert(TZ).dt.strftime("%H:%M:%S")
    table["change"] = table["predicted"] - table["delay_now"]
    table = table.sort_values("predicted", ascending=False).head(15)
    st.dataframe(
        table[["vehicle", "line", "direction", "delay_now", "predicted", "change", "seen"]],
        hide_index=True,
        width="stretch",
        column_config={
            "vehicle": "Tram", "line": "Line", "direction": "Direction",
            "delay_now": st.column_config.NumberColumn("Delay now", format="%d s"),
            "predicted": st.column_config.NumberColumn(f"Forecast in {horizon_min:g} min", format="%.0f s"),
            "change": st.column_config.NumberColumn("Change", format="%+.0f s",
                                                    help="Forecast minus delay now"),
            "seen": "Last seen",
        },
    )
    st.caption(f"Model {snapshot['model_version']} · forecasts issued {helsinki(issued):%H:%M:%S} by the "
               "forecaster service, which scores every one of them 5 minutes later "
               "(see Model health)")


# --- Live ------------------------------------------------------------------------------

def line_status(now: pd.DataFrame, alert_counts: dict[str, int] | None = None,
                cancelled: dict[str, int] | None = None) -> pd.DataFrame:
    status_ = now.groupby("line").agg(
        trams=("vehicle", "nunique"),
        mean_delay=("delay_s", "mean"),
        on_time=("on_time", "mean"),
        worst=("delay_s", "max"),
    )
    status_["mean_delay"] = (status_["mean_delay"] / 60).round(1)
    status_["worst"] = (status_["worst"] / 60).round(1)
    status_["on_time"] = (status_["on_time"] * 100).round()
    status_ = status_.sort_values("mean_delay", ascending=False).reset_index()
    if alert_counts is not None or cancelled is not None:
        status_["disruptions"] = [disruptions(line, alert_counts or {}, cancelled or {}) for line in status_["line"]]
    return status_


def disruptions(line: str, alert_counts: dict[str, int], cancelled: dict[str, int]) -> str:
    """``⚠️ 1 · 2 cancelled`` (alerts, cancelled trips) for the line status board; empty when there are none."""
    parts = []
    if alert_counts.get(line):
        parts.append(f"⚠️ {alert_counts[line]}")
    if cancelled.get(line):
        parts.append(f"{cancelled[line]} cancelled")
    return " · ".join(parts)


def cancelled_by_line(route_lines: dict[str, str], now: pd.Timestamp) -> dict[str, int] | None:
    """Trips HSL has cancelled today, per line; None when no trip updates are being collected."""
    trips = trips_by_route(GTFSRT_DIR, feeds.service_day(now, TZ))
    if trips.empty:
        return None
    counts: dict[str, int] = {}
    for route, n in zip(trips["route_id"], trips["cancelled"], strict=True):
        line = alerts.line_for_route(str(route), route_lines)
        if line and n:
            counts[line] = counts.get(line, 0) + int(n)
    return counts


def alert_scope(alert: alerts.Alert) -> str:
    where = ("Line " if len(alert.lines) == 1 else "Lines ") + ", ".join(alert.lines)
    return where if alert.routes else f"Stops on {where[0].lower()}{where[1:]}"


def render_alerts(active: list[alerts.Alert], fetched_at: str | None, now: pd.Timestamp) -> None:
    if not active:
        if fetched_at:
            st.caption(f"No HSL service alerts for trams right now (checked "
                       f"{helsinki(pd.Timestamp(fetched_at)):%H:%M}).")
        return
    disruptive = [a for a in active if a.disruptive]
    for alert in disruptive[:3]:
        st.warning(f"**{alert_scope(alert)}** · {alert.header}", icon="⚠️")
    title = f"{len(active)} HSL service alert{'s' if len(active) > 1 else ''} for trams in force"
    with st.expander(title, expanded=False):
        for alert in active:
            end = alert.ends(now)
            until = f" · until {helsinki(end):%a %d %b %H:%M}" if end is not None else ""
            more = f" · [details]({alert.url})" if alert.url else ""
            st.markdown(f"**{alert_scope(alert)}** — {alert.header}  \n"
                        f":gray[{alerts.label(alert.effect)} · {alerts.label(alert.cause)}{until}{more}]")
            if alert.description and alert.description != alert.header:
                st.caption(alert.description)


def render_line_status(table: pd.DataFrame) -> None:
    columns = {
        "line": st.column_config.TextColumn("Line", width="small"),
        "trams": st.column_config.NumberColumn("Trams", width="small"),
        "mean_delay": st.column_config.NumberColumn("Mean delay", format="%.1f min", width="small"),
        "on_time": st.column_config.ProgressColumn("On time", min_value=0, max_value=100,
                                                   format="%d%%", width="small"),
        "worst": st.column_config.NumberColumn("Most late", format="%.1f min", width="small"),
        "disruptions": st.column_config.TextColumn(
            "Disruptions", width="medium",
            help="⚠️ HSL service alerts for this line in force now (listed above the map), "
                 "and trips HSL has cancelled today"),
    }
    st.dataframe(
        table,
        hide_index=True,
        width="stretch",
        height=35 * (len(table) + 1) + 3,  # show every line without scrolling
        column_config={k: v for k, v in columns.items() if k in table.columns},
    )


def render_live(lines: list[str]) -> None:
    @st.fragment(run_every=f"{LIVE['refresh_s']}s")
    def live_panel() -> None:
        live = load_live()
        df = filter_lines(live, lines)
        if df.empty:
            st.warning(f"No tram data from the last {LIVE['live_window_min']} min. "
                       "Start the collector with `python -m tram_mlops.ingest`.")
            return
        newest = df["timestamp"].max()
        clock = pd.Timestamp.now(tz="UTC")
        age_s = (clock - newest).total_seconds()
        now = latest_per_tram(df[df["timestamp"] >= newest - pd.Timedelta(seconds=LIVE["in_service_s"])])

        st.caption(f"Updates every {LIVE['refresh_s']} s · data received {helsinki(newest):%H:%M:%S} "
                   f"Helsinki time ({age_s:.0f} s ago) from HSL via Digitransit, CC BY 4.0")
        if age_s > LIVE["stale_after_s"]:
            st.warning(f"No new data for {age_s / 60:.0f} min — is the collector still running?")

        late = int((now["delay_s"] > ON_TIME_MAX).sum())
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Trams in service", len(now))
        c2.metric("On time now", f"{100 * now['on_time'].mean():.0f} %",
                  help=f"Between {-ON_TIME_MIN} s early and {ON_TIME_MAX} s late")
        c3.metric("Mean delay now", f"{now['delay_s'].mean():.0f} s", help="Positive = behind schedule")
        c4.metric(f"More than {ON_TIME_MAX // 60} min late", late)

        route_lines, stop_lines = tram_network(live)
        feed, fetched_at = all_alerts(alerts_version())
        active = alerts.active(alerts.for_trams(feed, route_lines, stop_lines), clock)
        if lines:
            active = [a for a in active if set(a.lines) & set(lines)]
        render_alerts(active, fetched_at, clock)

        left, right = st.columns([1, 1])
        with left:
            st.subheader("Where the trams are now")
            render_map(now, key="live_map")
        with right:
            st.subheader("Line status")
            alert_counts = alerts.per_line(active) if fetched_at else None
            render_line_status(line_status(now, alert_counts, cancelled_by_line(route_lines, clock)))

        st.subheader("Delay forecast")
        render_live_forecast(lines, clock)

    live_panel()


def live_page() -> None:
    lines = pick_lines(known_lines(available_days(PROCESSED_DIR), live_available()))
    render_live(lines)


# --- History ---------------------------------------------------------------------------

def heatmap(df: pd.DataFrame, min_rows: int = 30) -> alt.Chart:
    """On-time share by line and hour of day; cells with fewer than ``min_rows`` observations are left out."""
    df = df[df["rows"] >= min_rows]
    lines = sorted(df["line"].unique(), key=alerts.line_sort_key)
    # Start the colour scale near the worst cells, so differences among the usual 60-90 % show.
    low = max(0, int(df["on_time_pct"].quantile(0.05) // 10 * 10)) if not df.empty else 0
    return alt.Chart(df).mark_rect(stroke="white", strokeWidth=1, cornerRadius=2).encode(
        x=alt.X("hour:O", title="Hour of day (Helsinki)", axis=alt.Axis(labelAngle=0)),
        y=alt.Y("line:N", title="Line", sort=lines, axis=alt.Axis(labelOverlap=False)),
        color=alt.Color("on_time_pct:Q", title="On time %",
                        scale=alt.Scale(domain=[low, 100], range=SEQUENTIAL, clamp=True)),
        tooltip=[alt.Tooltip("line:N", title="Line"), alt.Tooltip("hour:O", title="Hour"),
                 alt.Tooltip("on_time_pct:Q", title="On time %", format=".0f"),
                 alt.Tooltip("mean_delay_s:Q", title="Mean delay (s)", format=".0f"),
                 alt.Tooltip("rows:Q", title="Observations", format=",")],
    ).properties(height=alt.Step(22))


HISTOGRAM_BIN_S = 60  # HFP delays cluster on whole minutes, so finer bins zig-zag


def distribution_chart(hist: pd.DataFrame, bin_s: int = HISTOGRAM_BIN_S) -> alt.Chart:
    total = hist["rows"].sum()
    df = hist.assign(
        start=hist["bin"] / 60, end=(hist["bin"] + bin_s) / 60,
        share=100 * hist["rows"] / total if total else 0.0,
        punctuality=[punctuality(b + bin_s / 2) for b in hist["bin"]],
    )
    return alt.Chart(df).mark_bar(stroke="white", strokeWidth=1).encode(
        x=alt.X("start:Q", title="Delay (min, positive = late; ends include everything beyond)"),
        x2="end:Q",
        y=alt.Y("share:Q", title="Share of observations (%)"),
        y2=alt.datum(0),
        color=alt.Color("punctuality:N", title=None, legend=alt.Legend(orient="top"),
                        scale=alt.Scale(domain=list(PUNCTUALITY), range=list(PUNCTUALITY.values()))),
        tooltip=[alt.Tooltip("start:Q", title="From (min)", format=".1f"),
                 alt.Tooltip("end:Q", title="To (min)", format=".1f"),
                 alt.Tooltip("share:Q", title="Share (%)", format=".1f"),
                 alt.Tooltip("rows:Q", title="Observations", format=",")],
    ).properties(height=280)


def daily_chart(by_day: pd.DataFrame, field: str, title: str, fmt: str) -> alt.Chart:
    df = by_day.assign(label=by_day["day"].dt.strftime("%a %d %b"))
    return alt.Chart(df).mark_bar(color=SEQUENTIAL[2], cornerRadius=3, width=alt.RelativeBandSize(0.7)).encode(
        x=alt.X("label:O", title="Service day", sort=list(df["label"]), axis=alt.Axis(labelAngle=0)),
        y=alt.Y(f"{field}:Q", title=title),
        tooltip=[alt.Tooltip("label:O", title="Day"),
                 alt.Tooltip(f"{field}:Q", title=title, format=fmt),
                 alt.Tooltip("rows:Q", title="Observations", format=","),
                 alt.Tooltip("trams:Q", title="Trams")],
    ).properties(height=220)


def render_overview(data: dict[str, Any]) -> None:
    by_day = data["by_day"]
    if len(by_day) > 1:
        st.subheader("Day by day")
        left, right = st.columns(2)
        left.altair_chart(daily_chart(by_day, "on_time_pct", "On time (%)", ".1f"), width="stretch")
        right.altair_chart(daily_chart(by_day, "mean_delay_s", "Mean delay (s)", ".0f"), width="stretch")

    left, right = st.columns(2)
    with left:
        st.subheader("Mean delay over time (s)")
        st.line_chart(data["over_time"])
        st.caption(f"Averaged over {data['bucket_s'] // 60} min buckets, Helsinki time")
    with right:
        st.subheader("How late trams run")
        st.altair_chart(distribution_chart(data["histogram"]), width="stretch")
        st.caption(f"On time = between {-ON_TIME_MIN} s early and {ON_TIME_MAX} s late, HSL's definition")


def render_lines(data: dict[str, Any]) -> None:
    st.subheader("On time by line and hour of day")
    st.altair_chart(heatmap(data["line_hour"]), width="stretch")
    st.caption("Share of observations on time; darker is better. Hours with fewer than 30 observations are blank.")

    st.subheader("Reliability by line")
    by_line = data["by_line"].sort_values("line", key=lambda s: s.map(alerts.line_sort_key))
    for col in ("mean_delay_s", "p90_delay_s"):
        by_line[col] = (by_line[col] / 60).round(1)
    st.caption("Lines ending in H are runs to or from the depot. Click a column header to sort.")
    st.dataframe(
        by_line, hide_index=True, width="stretch", height=35 * (len(by_line) + 1) + 3,
        column_config={
            "line": st.column_config.TextColumn("Line", width="small"),
            "trams": st.column_config.NumberColumn("Trams", width="small"),
            "rows": st.column_config.NumberColumn("Observations", format="localized"),
            "on_time_pct": st.column_config.ProgressColumn("On time", min_value=0, max_value=100, format="%.0f%%"),
            "mean_delay_s": st.column_config.NumberColumn("Mean delay", format="%.1f min"),
            "p90_delay_s": st.column_config.NumberColumn("90th percentile", format="%.1f min",
                                                         help="9 in 10 observations are less late than this"),
            "late_pct": st.column_config.NumberColumn(f"> {ON_TIME_MAX // 60} min late", format="%.0f%%"),
        },
    )


def render_time_of_day(data: dict[str, Any]) -> None:
    by_hour = data["by_hour"]
    h_left, h_right = st.columns(2)
    with h_left:
        st.subheader("Mean delay (s)")
        st.line_chart(by_hour.pivot(index="hour", columns="day_type", values="mean_delay_s").round())
    with h_right:
        st.subheader("On time (%)")
        st.line_chart(by_hour.pivot(index="hour", columns="day_type", values="on_time_pct").round(1))
    st.caption(f"By Helsinki hour of day. On time = between {-ON_TIME_MIN} s early and {ON_TIME_MAX} s late.")


def render_trams(data: dict[str, Any]) -> None:
    st.subheader("Latest positions")
    render_map(data["latest"], key="history_map")

    st.subheader("Most delayed trams")
    st.dataframe(
        data["most_delayed"].rename(columns={"vehicle": "Tram", "line": "Line", "mean": "Mean delay (s)",
                                             "max": "Max delay (s)", "count": "Observations"}),
        hide_index=True,
        width="stretch",
    )

    st.subheader("Delay forecast")
    if data["recent"].empty:
        st.info("No recent observations to forecast from.")
    else:
        render_forecast(data["recent"])


def render_stops(lines: list[str]) -> None:
    hours = LIVE["stop_events_hours"]
    st.caption(f"Arrival delay at each stop over the last {hours} h: when the tram stopped (HFP `ARS` event) "
               "minus its scheduled arrival. From the raw stop events, independent of the days picked above.")
    arrivals = stop_arrivals(RAW_DIR, tuple(lines), hours)
    arrivals = arrivals[arrivals["arrivals"] >= LIVE["min_stop_arrivals"]]
    if arrivals.empty:
        st.info("No stop events collected in this window yet.")
        return
    c1, c2, c3 = st.columns(3)
    total = int(arrivals["arrivals"].sum())
    c1.metric("Arrivals", f"{total:,}")
    c2.metric("Stops", len(arrivals))
    c3.metric("Arrived on time",
              f"{(arrivals['on_time_pct'] * arrivals['arrivals']).sum() / total:.0f} %")

    shown = arrivals.dropna(subset=["lat", "lon"]).copy()
    shown["color"] = [color_of(d) for d in shown["mean_delay_s"]]
    shown["delay"] = shown["mean_delay_s"].round().astype(int)
    shown["on_time"] = shown["on_time_pct"].round().astype(int)
    legend_caption("by mean arrival delay; hover a stop for details")
    render_deck(shown, "stops_map", "Stop {stop} · lines {lines}\nMean arrival delay: {delay} s\n"
                                    "On time: {on_time} % of {arrivals} arrivals", radius=40)

    st.subheader("Stops where trams arrive latest")
    worst = arrivals.sort_values("mean_delay_s", ascending=False).head(15)
    st.dataframe(
        worst[["stop", "lines", "arrivals", "mean_delay_s", "median_delay_s", "on_time_pct"]],
        hide_index=True, width="stretch",
        column_config={
            "stop": st.column_config.TextColumn("Stop id"),
            "lines": st.column_config.TextColumn("Lines"),
            "arrivals": st.column_config.NumberColumn("Arrivals"),
            "mean_delay_s": st.column_config.NumberColumn("Mean delay", format="%.0f s"),
            "median_delay_s": st.column_config.NumberColumn("Median delay", format="%.0f s"),
            "on_time_pct": st.column_config.ProgressColumn("On time", min_value=0, max_value=100, format="%.0f%%"),
        },
    )


def render_history(lines: list[str], days: tuple[date, date] | None) -> None:
    if days is None:
        st.warning("No processed data yet. Run `python -m tram_mlops.process`.")
        return
    data = history_data(days[0], days[1], tuple(lines), manifest_version())
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

    # Unlike tabs, only the chosen section runs, so maps and the stop-event scan cost nothing until opened.
    sections: dict[str, Callable[[], None]] = {
        "Overview": lambda: render_overview(data),
        "Lines": lambda: render_lines(data),
        "Time of day": lambda: render_time_of_day(data),
        "Trams": lambda: render_trams(data),
        "Stops": lambda: render_stops(lines),
    }
    chosen = st.segmented_control("Section", list(sections), default="Overview", key="history_section",
                                  label_visibility="collapsed")
    sections[chosen or "Overview"]()


def history_page() -> None:
    processed_days = ensure_processed()
    day_range = pick_days(processed_days)
    lines = pick_lines(known_lines(processed_days, live_available()))
    render_history(lines, day_range)


# --- System ----------------------------------------------------------------------------

PIPELINE = """
digraph {
  rankdir=LR; bgcolor="transparent"; nodesep=0.25; ranksep=0.35;
  node [shape=box, style="rounded,filled", fillcolor="#f0efec", color="#a3a29c", fontname="sans-serif",
        fontsize=11, fontcolor="#2b2b2b"];
  edge [color="#8a8984", arrowsize=0.6];
  hfp [label="HSL HFP\\n(MQTT stream)", fillcolor="#cde2fb"];
  rt [label="HSL GTFS-RT\\n(alerts, trip updates)", fillcolor="#cde2fb"];
  ingest [label="ingest\\ncollector"]; poller [label="gtfsrt\\npoller"];
  raw [label="raw JSONL\\nper UTC day"];
  process [label="process\\nhourly"]; parts [label="Parquet partitions\\n+ manifest"];
  train [label="train"]; model [label="versioned\\nmodel"];
  dash [label="dashboard", fillcolor="#86b6ef"]; backup [label="Allas backup\\nhourly"];
  hfp -> ingest -> raw; rt -> poller -> raw;
  raw -> process -> parts -> train -> model -> dash;
  parts -> dash; raw -> dash [label="live", fontsize=9, fontname="sans-serif"];
  raw -> backup [style=dashed]; parts -> backup [style=dashed];
}
"""


def render_sources(sources: list[status.Source], now: pd.Timestamp) -> None:
    rows = []
    for s in sources:
        rows.append({
            "status": status.BADGE[s.state(now)],
            "source": s.name,
            "last_update": ago(s.last_write, now),
            "newest": f"{helsinki(s.newest_record):%d %b %H:%M:%S}" if s.newest_record is not None else "",
            "today": s.note or (f"{s.records_today:,} records" if s.records_today is not None else ""),
            "size_mb": s.size_bytes / 1e6,
            "feed": s.feed,
        })
    st.dataframe(
        pd.DataFrame(rows), hide_index=True, width="stretch",
        column_config={
            "status": st.column_config.TextColumn("Status", width="small"),
            "source": st.column_config.TextColumn("Source"),
            "last_update": st.column_config.TextColumn("Last write", help="When the file was last written"),
            "newest": st.column_config.TextColumn("Newest record", help="Helsinki time"),
            "today": st.column_config.TextColumn("Today", help="What arrived today (UTC day)"),
            "size_mb": st.column_config.NumberColumn("Today's file", format="%.1f MB"),
            "feed": st.column_config.TextColumn("How it arrives", width="large"),
        },
    )
    st.caption("Stale = not written within "
               f"{LIVE['stale_after_s']} s (positions), {LIVE['events_stale_after_s']} s (stop events) or "
               f"{LIVE['trip_updates_stale_after_s']} s (trip updates). Trams don't run from about 2 to 5 am, "
               "so overnight staleness is expected. Alerts are saved only when they change, so their age says "
               "nothing about the poller's health.")


def render_processing() -> None:
    table = status.manifest_table(PROCESSED_DIR)
    if table.empty:
        st.info("Nothing processed yet. Run `python -m tram_mlops.process`.")
        return
    c1, c2, c3 = st.columns(3)
    c1.metric("Days processed", len(table))
    c2.metric("Rows kept", f"{int(table['rows'].sum()):,}")
    last = table["processed_at"].max()
    c3.metric("Last processing run", f"{helsinki(last):%d %b %H:%M}" if pd.notna(last) else "—")
    table["processed_at"] = [f"{helsinki(t):%d %b %H:%M}" if pd.notna(t) else "" for t in table["processed_at"]]
    count = st.column_config.NumberColumn
    st.dataframe(
        table, hide_index=True, width="stretch",
        column_config={
            "day": st.column_config.TextColumn("Day (UTC)"),
            "source": st.column_config.TextColumn("Raw file"),
            "raw_rows": count("Raw rows", format="localized"),
            "rows": count("Kept", format="localized"),
            "kept_pct": count("Kept %", format="%.1f%%"),
            "missing_required": count("No delay/time", format="localized",
                                      help="Dropped: missing vehicle, timestamp or delay"),
            "implausible_delay": count("Stale journey", format="localized",
                                       help="Dropped: |delay| over the limit, from trams still signed in "
                                            "to a journey that ended long ago"),
            "excluded_line": count("Not in service", format="localized",
                                   help="Dropped: lines that are not public service (e.g. 000)"),
            "duplicate": count("Duplicate", format="localized", help="Dropped: the broker sent it twice"),
            "malformed": count("Malformed", format="localized", help="Dropped: not valid JSON"),
            "processed_at": st.column_config.TextColumn("Processed", help="Helsinki time"),
            "version": count("Rules version", help="Processing rules version; a change rebuilds every day"),
        },
    )


def render_model_summary() -> None:
    info = status.model_info(MODELS_DIR)
    if info is None:
        st.info("No model deployed yet. Run `python -m tram_mlops.train`.")
        return
    trained = pd.to_datetime(info.get("version", ""), format="%Y%m%dT%H%M%SZ", utc=True, errors="coerce")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Model version", info.get("version", "?"))
    c2.metric("Trained", f"{helsinki(trained):%d %b %Y}" if pd.notna(trained) else "?")
    c3.metric("Test MAE", f"{info['model']['mae_s']:.0f} s",
              delta=f"{-info['mae_improvement_pct']:.1f} % vs baseline" if info.get("mae_improvement_pct") else None,
              delta_color="inverse", help="Offline test-set error; lower is better")
    c4.metric("Training rows", f"{info.get('train_rows', 0):,}")


def status_page() -> None:
    now = pd.Timestamp.now(tz="UTC")
    st.caption("What the pipeline is doing right now: how fresh each data source is, what each processing run "
               "kept and dropped, and which model is deployed.")
    st.graphviz_chart(PIPELINE)

    st.subheader("Data sources")
    sources = source_statuses(RAW_DIR, GTFSRT_DIR, now.date())
    render_sources(sources, now)

    events = next((s for s in sources if s.name == "Stop and journey events"), None)
    if events is not None and events.path is not None:
        counts = event_counts(events.path)
        if not counts.empty:
            with st.expander("Stop and journey events today, by type"):
                counts["meaning"] = counts["event"].map(EVENT_TYPES).fillna("")
                st.dataframe(counts[["event", "meaning", "count"]], hide_index=True, width="stretch",
                             column_config={"event": "Event", "meaning": "Meaning",
                                            "count": st.column_config.NumberColumn("Today", format="localized")})

    st.subheader("Processing runs")
    st.caption("Each UTC day of raw positions becomes one Parquet partition. Unchanged days are skipped, "
               "and every dropped row is counted by the rule that dropped it.")
    render_processing()

    st.subheader("Deployed model")
    render_model_summary()

    usage = status.disk_usage({
        "Raw positions": (RAW_DIR, "tram_*"),
        "Raw stop events": (RAW_DIR, "events_*"),
        "GTFS-RT archive": (GTFSRT_DIR, "*.jsonl.gz"),
        "Processed partitions": (PROCESSED_DIR, "date=*/*.parquet"),
    })
    with st.expander(f"Data on disk: {usage['bytes'].sum() / 1e9:.2f} GB"):
        usage["mb"] = usage["bytes"] / 1e6
        st.dataframe(usage[["data", "files", "mb"]], hide_index=True, width="stretch",
                     column_config={"data": "Data", "files": "Files",
                                    "mb": st.column_config.NumberColumn("Size", format="%.1f MB")})


# --- Model health ----------------------------------------------------------------------

def health_metrics(window: dict[str, Any], label: str) -> None:
    """One row of KPIs for a rolling window of scored forecasts."""
    en_route = window.get("segments", {}).get("en_route", {})
    overall = window.get("overall", {})
    c1, c2, c3, c4 = st.columns(4)
    if en_route.get("rows"):
        c1.metric(f"En-route error, {label}", f"{en_route['model_mae_s']:.1f} s",
                  delta=f"{en_route['model_mae_s'] - en_route['persistence_mae_s']:+.1f} s vs persistence",
                  delta_color="inverse",
                  help="Trams already moving: the hard case, where the delay changes on the way")
        c2.metric("Persistence, same forecasts", f"{en_route['persistence_mae_s']:.1f} s")
    if overall.get("rows"):
        c3.metric(f"All forecasts, {label}", f"{overall['model_mae_s']:.1f} s",
                  delta=f"{overall['improvement_pct']:+.0f} % better" if overall.get("improvement_pct") else None,
                  help="Includes trams waiting to depart, where persistence is badly wrong")
    c4.metric("Forecasts scored", f"{overall.get('rows', 0):,}",
              help=f"{window.get('matched_pct', 0):.0f}% of logged forecasts had a true answer; the rest "
                   "belong to trips that ended within 5 minutes")


def mae_chart(errors: pd.DataFrame) -> alt.Chart:
    long = errors.rename_axis("time").reset_index().melt("time", var_name="Forecast", value_name="MAE (s)")
    long["time"] = long["time"].dt.tz_convert(TZ).dt.tz_localize(None)
    return alt.Chart(long).mark_line(point=True).encode(
        x=alt.X("time:T", title="Helsinki time"),
        y=alt.Y("MAE (s):Q", title="Mean absolute error (s)"),
        color=alt.Color("Forecast:N", scale=alt.Scale(domain=["Model", "Persistence"],
                                                      range=[SEQUENTIAL[3], PUNCTUALITY["On time"]])),
        tooltip=["time:T", "Forecast:N", alt.Tooltip("MAE (s):Q", format=".1f")],
    ).properties(height=280)


def line_errors(scored: pd.DataFrame) -> pd.DataFrame:
    """En-route model and persistence error per line, for lines with enough scored forecasts."""
    df = scored.dropna(subset=["actual"])
    df = df[df["segment"] == "en_route"]
    if df.empty:
        return pd.DataFrame()
    df = df.assign(model=(df["predicted"] - df["actual"]).abs(), persistence=(df["delay_now"] - df["actual"]).abs())
    table = df.groupby("line").agg(forecasts=("model", "size"), model=("model", "mean"),
                                   persistence=("persistence", "mean"))
    table = table[table["forecasts"] >= 20]
    table["improvement"] = 100 * (1 - table["model"] / table["persistence"])
    return table.sort_values("model", ascending=False).reset_index()


def model_health_page() -> None:
    now = pd.Timestamp.now(tz="UTC")
    st.caption("Every 15 s the forecaster service forecasts each tram's delay 5 minutes ahead. One forecast per "
               "tram every 30 s is logged and, 5 minutes later, scored against the delay the tram actually "
               "had, using the same definition the model was trained on. Persistence (\"the delay stays as "
               "it is\") is scored on the same forecasts.")
    scored = scored_forecasts(PREDICTIONS_DIR, now.date())
    if scored.empty:
        st.info("No scored forecasts yet. Start the forecaster with `python -m tram_mlops.forecaster`; "
                "the first scores appear about 6 minutes later.")
        render_model_summary()
        return

    metrics = monitoring.live_metrics(scored, now)
    health_metrics(metrics["1h"], "last hour")
    health_metrics(metrics["24h"], "last 24 h")

    st.subheader("Error over time")
    segment = st.segmented_control("Trams", ["En route", "Waiting to depart"], default="En route",
                                   key="health_segment")
    errors = monitoring.mae_over_time(
        scored[scored["observed_at"] >= now - pd.Timedelta(hours=24)], freq="15min",
        segment="before_departure" if segment == "Waiting to depart" else "en_route")
    if errors.empty:
        st.info("No scored forecasts for these trams in the last 24 hours.")
    else:
        st.altair_chart(mae_chart(errors), width="stretch")
        st.caption("Mean absolute error per 15-minute window, by when the forecast was made.")

    st.subheader("By line, last 24 h")
    by_line = line_errors(scored[scored["observed_at"] >= now - pd.Timedelta(hours=24)])
    if by_line.empty:
        st.info("Not enough scored en-route forecasts per line yet.")
    else:
        st.dataframe(by_line, hide_index=True, width="stretch", column_config={
            "line": "Line",
            "forecasts": st.column_config.NumberColumn("Scored", format="localized"),
            "model": st.column_config.NumberColumn("Model error", format="%.1f s"),
            "persistence": st.column_config.NumberColumn("Persistence error", format="%.1f s"),
            "improvement": st.column_config.NumberColumn("Improvement", format="%.0f %%"),
        })
        st.caption("Trams en route only, lines with at least 20 scored forecasts. Worst first.")

    st.subheader("Live vs offline test")
    st.caption("The deployed model's error on its held-out test set when it was trained. A live error well "
               "above it means the world has changed since training (see roadmap E11, drift).")
    render_model_summary()
    versions = metrics["24h"].get("model_versions", [])
    if len(versions) > 1:
        st.caption(f"Models live in the last 24 h: {', '.join(versions)}")


# --- navigation ------------------------------------------------------------------------

def pages() -> dict[str, Any]:
    """The dashboard's pages by URL path, in menu order. Add a page by adding an entry here.

    URLs stay the same whatever data is available (``/``, ``/history``, ``/system``), so
    links to a page always work.
    """
    entries: list[tuple[Callable[[], None], str, str, str]] = [
        (live_page, "Live", "live", ":material/radar:"),
        (history_page, "History", "history", ":material/bar_chart:"),
        (model_health_page, "Model health", "model-health", ":material/query_stats:"),
        (status_page, "System", "system", ":material/monitor_heart:"),
    ]
    return {path: st.Page(fn, title=title, url_path=path, icon=icon, default=path == "live")
            for fn, title, path, icon in entries}


def render_about() -> None:
    with st.sidebar:
        st.divider()
        with st.expander("About this dashboard"):
            st.markdown(
                "An end-to-end MLOps project on a real stream: a collector subscribes to every Helsinki "
                "tram's position and stop events, a poller archives HSL's service alerts and trip updates, "
                "an hourly job cleans the data into day partitions, and a model forecasts each tram's delay "
                "5 minutes ahead. Everything runs on one small VM on CSC cPouta.\n\n"
                f"[Source code and design notes]({REPO_URL})")
        st.caption("Data: [HSL](https://www.hsl.fi/en) open data via "
                   "[Digitransit](https://digitransit.fi/en/developers/), "
                   "[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). "
                   "Maps © [CARTO](https://carto.com/attributions), "
                   "© [OpenStreetMap](https://www.openstreetmap.org/copyright) contributors.")


st.set_page_config(page_title="Helsinki Tram Delays", page_icon="🚋", layout="wide")
all_pages = pages()
page = st.navigation(list(all_pages.values()), position="top")
# Without a running collector the Live page is empty, so a visitor's first page is History instead.
if page.url_path == "" and not st.session_state.get("landed") and not live_available():
    st.session_state["landed"] = True
    st.switch_page(all_pages["history"])
st.session_state["landed"] = True
st.title("🚋 Helsinki Tram Delays")
st.caption("Every Helsinki tram, tracked live from HSL's open data, cleaned into a dataset and forecast "
           "5 minutes ahead by a continuously running pipeline.")
page.run()
render_about()
