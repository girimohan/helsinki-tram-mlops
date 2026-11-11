# Engineering roadmap

This plan turns the current pipeline into a system that runs on its own:
- it collects data continuously
- it retrains on a schedule, but only promotes a new model when it beats the current one
- it measures its own live accuracy

It builds on the existing code instead of replacing it. `process.clean()`, `features.add_features()`, the purged time split and the Streamlit dashboard all stay.

## Deliverables

The goal is an end-to-end production MLOps system on a real-world streaming source. Each deliverable maps to roadmap items:

| Deliverable | Today | Delivered by |
|---|---|---|
| Live tram tracking | Done: live map and line status, refreshed every 15 s | E3 (always on, public) |
| Historical analysis | History view over any range of days (DuckDB), by line and hour of day | E4 (quality reports) |
| Delay prediction | Gradient boosting, 5 min ahead, against a persistence baseline | E5 (trustworthy evaluation), E16 (HSL's own predictions as a benchmark), E13, E14 |
| Anomaly detection | Not started | E15, with labels from E16 (alerts, cancellations, skipped stops) |
| Automated retraining | Manual `train` | E7, E8 |
| Model monitoring | Offline test metrics only | E6 (live accuracy), E10 (alerts), E11 (drift), E16 (feed freshness) |

## Principles

1. **Collection time is the bottleneck, not code.** The model's weakest segment (trams already en route) needs weeks of data covering rush hours, weekends and weather. The collector has to run permanently first; most later items depend on the data it gathers.
2. **The true answer arrives 5 minutes after each forecast.** So live accuracy can be measured directly instead of being inferred from data drift. Monitoring is designed around this.
3. **Add a tool only when the project has the problem it solves.** See [Deliberately not planned](#deliberately-not-planned).

## Status

Ordered by value per day of effort, adjusted for what each item unblocks.

| # | Enhancement | Effort | Needs | Status |
|---|---|---|---|---|
| 1 | [E1 Reproducible pipeline and CI](#e1-reproducible-pipeline-and-ci) | 1–1.5 d | none | **Done** 2026-11-10 |
| 2 | [E2 Incremental, day-partitioned storage](#e2-incremental-day-partitioned-storage) | 3–4 d | none | **Done** 2026-11-11 |
| 3 | [E3 Always-on host on CSC cPouta](#e3-always-on-host-on-csc-cpouta) | 2–3 d | none | **Phases 1–2 done**: collecting since 2026-11-09, public HTTPS dashboard, deploy on green CI, hourly Allas backups ([setup guide](../deploy/cpouta/README.md)). Phase 3 (Terraform) planned |
| 4 | [E16 HSL realtime feeds](#e16-hsl-realtime-feeds-alerts-trip-updates-and-stop-events) | 1–1.5 d to start archiving; about 9 d in all | E3 | **Phase 0 done** 2026-11-11: archiving alerts, tram trip updates and stop events |
| 5 | [E4 Data contracts and quality reports](#e4-data-contracts-and-quality-reports) | 2 d | E2 | Planned |
| 6 | [E6 Prediction logging and live accuracy](#e6-prediction-logging-and-live-accuracy) | 3 d | E3 | Planned |
| 7 | [E5 Backtesting](#e5-backtesting) | 2–3 d | E2, ~1 week of data | Planned |
| 8 | [E7 Promotion gate and scheduled retraining](#e7-promotion-gate-and-scheduled-retraining) | 3 d | E2, E5 | Planned |
| 9 | [E15 Anomaly detection](#e15-anomaly-detection) | 3–4 d | E2, E6, E16, ~2 weeks of data | Planned |
| 10 | [E12 Packaging and container hardening](#e12-packaging-and-container-hardening) | 1.5–2 d | none | Planned |
| 11 | [E8 Experiment tracking and model registry](#e8-experiment-tracking-and-model-registry) | 2–3 d | E7 | Planned |
| 12 | [E10 Metrics, logs and alerts](#e10-metrics-logs-and-alerts) | 3 d | E3 | Planned |
| 13 | [E11 Drift detection](#e11-drift-detection) | 2 d | E6 | Planned |
| 14 | [E9 Inference API](#e9-inference-api) | 3–4 d | E6 | Planned |
| 15 | [E13 Prediction intervals](#e13-prediction-intervals) | 3–4 d | E5, ~2 weeks of data | Planned |
| 16 | [E14 Network-aware features](#e14-network-aware-features) | 1.5–2 wk | E5, ~3 weeks of data | Planned |

Items 1–8 complete the core loop. Item 4 comes early only because its data can't be backfilled. Item 9 completes the six deliverables. The rest adds depth.

---

## E1. Reproducible pipeline and CI

**Why.** The bundled sample is two one-minute bursts 55 minutes apart, so `train` on a fresh clone finds no 5-minute targets and stops. CI runs `process` but never `train`.

**Design.**
- `scripts/make_sample.py` cuts a continuous 45-minute slice (all trams, one record per 10 s, about 1.5 MB gzipped) from a real collection day.
- `load_raw` reads `.gz` files.
- CI matrix on Python 3.11 and 3.13: ruff, mypy on `src/`, pytest with coverage, a full `process` then `train` run, and `docker build`.

**Changes.** `scripts/make_sample.py`, `data/sample/tram_sample.jsonl.gz`, `.pre-commit-config.yaml`, `Makefile`, ruff and mypy settings in `pyproject.toml`, `ci.yml`.

**Done 2026-11-10.**
- **Sample:** 07:30–08:15 Helsinki time on Tuesday 10 Nov 2026: 30,135 records, 119 trams, 1.2 MB gzipped. The cut is deterministic, and the script streams the raw file so it runs on the VM.
- **Fresh clone trains:** on the sample, the model scores 35.6 s MAE against 63.0 s for persistence, and 37.7 s against 42.1 s en route.
- **Tests:** an end-to-end test runs process → train on the sample and fails if the model loses to persistence en route.
- **CI jobs:**
  - lint: ruff and mypy, with stubs that match pandas 2.3
  - test: Python 3.11 and 3.13, coverage gate at 80%, then the command-line pipeline
  - docker: builds the image and processes the sample inside it
- **Dependencies:** dev tools are pinned in `requirements-dev.txt`, and `pytest` is out of the runtime image.

## E2. Incremental, day-partitioned storage

**Why.** `process` loads all raw data into memory and rebuilds everything on each run. Raw data grows at about 12 MB per hour, so memory runs out after a few weeks of collection.

**Design.**
- **Layout:** `data/processed/positions/date=YYYY-MM-DD/part.parquet`. Finished raw days are gzipped.
- **Manifest:** `_manifest.json` records each source file's size, hash, row counts and `PROCESSING_VERSION`.
  - Unchanged files are skipped, so re-running is safe.
  - Raising `PROCESSING_VERSION` reprocesses everything.
  - `--since` and `--rebuild` run backfills by hand.
- **Memory:** each day is parsed on its own, so memory use is bounded by one day.
- **Safe writes:** write to a temp file, then `os.replace`.
- **Reads:** `storage.read_positions(start, end, columns)` opens only the needed partitions (via `pyarrow.dataset`). The History view uses DuckDB aggregates.
- **Midnight:** trips can cross UTC midnight, so features for day D also read the last 10 minutes of day D−1.

**Changes.** New `storage.py`; `process.py` updated; `[paths] processed_dir`; `tests/test_storage.py`.

**Done 2026-11-11.**
- **Memory:** a full day (696k raw records) used to peak at 2.2 GB, more than the VM has. It now peaks at 0.8 GB:
  - each 50k-record chunk is cleaned as it is read, and duplicates are removed once at the end
  - text columns are Arrow strings
  - the cleaned day takes 140 MB instead of 325 MB, with identical rows
- **Disk:**
  - a day's partition is 17 MB (zstd)
  - gzipped raw days are 8× smaller (234 MB → 29 MB)
  - compaction deletes the original only after checking the archive's size, and the content hash means a gzipped day isn't reprocessed
- **Schema:** every partition is written with one fixed Arrow schema (`storage.POSITIONS_SCHEMA`), even on days with missing fields. This is a first data contract ahead of E4.
- **Manifest:** records each day's rows and the rows dropped per rule, an early version of E4's drop counts.
- **History view:** DuckDB queries take 0.01–0.13 s on a full day, and only aggregates reach the dashboard. New: a day-range picker, and delay and on-time share by hour of day, weekday vs weekend.
- **Scheduling:** `tram-process.timer` runs the incremental job hourly on the VM, with its own memory cap. `train` reads the last `train_days` days.
- **Midnight:** trips crossing UTC midnight are handled because `train` reads a continuous range of days. Per-day feature builds (E5) will need the D−1 tail.

## E3. Always-on host on CSC cPouta

**Why.** The collector has to run around the clock to build a multi-week dataset. A server also makes a live demo possible.

**Design.**
- **Phase 1 (done 2026-11-09):**
  - one `standard.small` VM (Ubuntu 24.04) on cPouta, project 2022568
  - raw data on a separate persistent volume, so it outlives the VM
  - the collector and dashboard run with Docker Compose
  - the dashboard is only reachable through an SSH tunnel
  - Guide: [deploy/cpouta/README.md](../deploy/cpouta/README.md).
- **Phase 2:**
  - **Done 2026-11-09:** Caddy with an automatic Let's Encrypt certificate serves the public dashboard. Only port 443 is open, and the dashboard is memory-capped so it can't starve the collector.
  - **Done 2026-11-09:** pull-based deploy. A systemd timer on the VM fast-forwards to new commits on `main` once all their GitHub checks have passed, so the VM needs no inbound access or deploy keys.
  - **Done 2026-11-09:** hourly `rclone copy` of raw data and models to the CSC Allas (S3) bucket `tram-mlops-2022568`. It never deletes anything, and it uploads a snapshot of the file the collector is still writing.
- **Phase 3:** Terraform with the OpenStack provider (cPouta is OpenStack) and application credentials. It defines the VM, volume, security groups and floating IP as code.

**Changes.** `deploy/cpouta/`, environment-driven paths in `docker-compose.yml`. Later: `infra/terraform/`, `deploy/Caddyfile`, `.github/workflows/release.yml`.

## E4. Data contracts and quality reports

**Why.** The cleaning rules handle known problems in the feed, but their effect is only logged. An unannounced change in the feed is the biggest data risk (the delay field's sign was already inverted once).

**Design.**
- **Schemas:** Pandera schemas for raw records (types, plus plausible ranges such as lat 59.9–60.4, lon 24.6–25.3, speed 0–30 m/s) and for the processed table.
- **Drop counts:** `clean_with_report()` returns rows dropped per rule; `clean()` stays as a wrapper.
- **Per-partition report:**
  - rows in and out
  - rows dropped per rule
  - null rates
  - vehicle count
  - records per vehicle per minute (shows gaps in collection)
  - delay quantiles
- **Thresholds:** a `[quality]` config section sets warn and fail levels. A partition that fails moves to `data/quarantine/` and training skips it.

**Changes.** `schemas.py`, `quality.py`, `data/processed/_quality/date=…/report.json`, a data-quality tab in the dashboard.

## E5. Backtesting

**Why.** Today's results come from a single 6-minute test window. The en-route gain (41.0 s vs 43.4 s MAE) can't be told apart from noise.

**Design.**
- **Rolling-origin backtest:** for each day D, train on [D−k, D) with the existing purge and test on D.
- **Primary metric:** en-route MAE.
- **Baselines:** persistence, plus a stronger one: the mean delay change by line × hour.
- **Uncertainty:** a block bootstrap that resamples whole trips gives 95% intervals on the MAE difference.
- **Breakdowns:** by hour, line, weekday vs weekend, and current delay.
- The README results are generated from `reports/backtest/<run>.json`.

**Changes.** `evaluate.py`; `time_split` and `train()` take explicit time windows.

## E6. Prediction logging and live accuracy

**Why.** The real delay is known 5 minutes after each forecast, so the system can report its own live accuracy against the baseline.

**Design.**
- **Forecaster service:** every 15 s it runs `load_recent`, then `clean_with_config`, then `add_features`, then predicts. Each prediction is appended to `data/predictions/date=…/` with `vehicle`, `trip_key`, `issued_at`, `horizon_s`, `delay_now`, `predicted` and `model_version`.
- **Scoring job:** joins predictions to what happened, using the same `features._lookup` as the training target, so the live "true answer" matches training exactly.
- **Live metrics:** rolling 1-hour and 24-hour MAE for the model and for persistence, split by segment.
- **Dashboard:** a "Model health" tab. The live forecasts come from the forecaster, so the model no longer runs inside the dashboard.

**Changes.** `forecaster.py`, `monitoring.py`, a compose service, a dashboard tab.

## E7. Promotion gate and scheduled retraining

**Why.** `train.save()` points `latest.json` at every new model without checking it. One bad day of data would put a worse model live.

**Design.**
- **Pipeline:** `pipeline.py` runs `compact → process → validate → train → evaluate → promote`. Every step is safe to re-run.
- **The gate:** the new model and the current one are scored on the same purged holdout day. The new model is promoted only if:
  - en-route MAE improves by at least X%
  - the lower end of the bootstrap interval is above 0
  - no breakdown slice gets worse by more than Y%
  - the training data passed its quality checks
- Rejected models are kept, with the reason recorded.
- **Rollback:** `pipeline promote --version`.
- **Scheduling:** a supercronic `scheduler` service. Retraining runs daily; scoring runs every 10 minutes.

**Changes.** `pipeline.py`, `promotion.py`, `deploy/crontab`, a compose service.

## E8. Experiment tracking and model registry

**Why.** `metrics.json` doesn't record the training data window, data hash, git commit, config or library versions.

**Design.**
- **MLflow** runs in compose, with a SQLite backend.
- **Each run logs:** parameters, overall and segment metrics, backtest results, the git SHA, the manifest hash and the model with its signature.
- **Registry aliases:** `champion` and `challenger`.
- **`registry.py`** has two backends: `file` (today's behaviour) and `mlflow`. Tests and offline use keep working without a server.

## E9. Inference API

**Design.**
- **Service:** a FastAPI app that uses the forecaster's code and refreshes its state in the background.
- **Endpoints:**
  - `/v1/trams/{vehicle}/forecast`
  - `/v1/lines/{line}/forecasts`
  - `/v1/forecasts`
  - `POST /v1/predict`
  - `/healthz`
  - `/readyz` (model loaded and data fresh)
  - `/metrics`
- **Responses** include `model_version`, `issued_at` and `data_age_s`.
- **Model reload:** when the champion changes.
- **Load test:** k6, with a p95 target under 50 ms.

## E10. Metrics, logs and alerts

**Design.**
- **Prometheus metrics:**
  - ingester: messages received and kept, parse errors, reconnects, last message time
  - API: response latency
  - batch jobs: time of last success
  - live accuracy: MAE gauges
- **Grafana:** configured as code.
- **Alerts:**
  - collection stalled for more than 2 minutes
  - no trams between 06:00 and 23:00
  - no successful pipeline run in 26 hours
  - live en-route MAE worse than persistence for an hour
  - a failed quality check
- **Logs:** JSON format, with `run_id` and `model_version`.

## E11. Drift detection

**Design.**
- **Comparisons:** daily, the latest feature distributions against a reference sample saved at training time:
  - PSI for numeric features
  - Jensen–Shannon divergence for categorical ones
  - prediction drift
  - target drift
- **Seasonality:** compare against the same hour and day type, or every rush hour looks like drift.
- **Reports:** Evidently produces the report; thresholds use the project's own PSI values. This ranks below E6 because live accuracy is measured directly.

## E12. Packaging and container hardening

**Design.**
- **Dependencies:** a `uv.lock` with separate runtime and dev groups.
- **Declarations:** `pydeck` and `numpy` are declared.
- **Dockerfile:** a multi-stage build that installs a wheel, which requires a `TRAM_HOME` setting in place of `PROJECT_ROOT`. The base image is pinned by digest and the image has a `HEALTHCHECK`.
- **Compose:** read-only filesystem, all capabilities dropped, `no-new-privileges`, memory limits.
- **CI:** a Trivy image scan, an SBOM, Dependabot, actions pinned to SHAs, and a least-privilege `permissions:` block.
- **Models:** a model file's checksum is verified before `joblib.load`.

## E13. Prediction intervals

**Design.** Quantile models at 0.1, 0.5 and 0.9, calibrated on a separate day so the stated coverage holds.
- The backtest reports pinball loss and actual coverage.
- The API returns p10, p50 and p90.
- E6 tracks live coverage.
- This makes the "When should I leave?" feature possible.

## E14. Network-aware features

**Design.**
- **Headway:** the gap to the previous tram on the same line and direction, which shows bunching.
- **Historical delay patterns:** the mean delay change by line × direction × stop × hour × day type, computed only from data before the training cutoff and versioned as `features/historical/asof=…/`.
- **Weather:** Finnish Meteorological Institute open data.
- **Ablation:** the backtest shows what each feature group adds.
- **Code layout:** `features.py` becomes a package; `__init__` re-exports the current names, so existing imports keep working.

## E15. Anomaly detection

**Why.** Delay forecasting answers "how late will this tram be?". Operators also need to know when something unusual is happening right now: a breakdown, a blocked track, a line-wide disruption. These events are rare, unlabelled and costly, so they need a different method from the regression model.

**Design.**
- **Two levels:**
  - **Vehicle:** a tram is stationary away from a stop for several minutes while in service, or its delay grows much faster than is usual for its line at that hour.
  - **Line:** many trams on one line lose time at once. The line's delay change over the last 10 minutes is an outlier for that line × hour × day type.
- **Method:**
  - Robust statistics first (median and MAD baselines computed from historical partitions, robust z-scores). They are explainable and need no labels.
  - Isolation Forest on per-line 10-minute aggregates only if the simple rules miss events.
- **Model-aware signal:** once E6 logs predictions, a burst of large forecast errors on one line means a disruption the model didn't see coming.
- **Evaluation:** weak labels from E16:
  - *reactive* service alerts, i.e. those first seen close to their start (planned works are excluded)
  - trip cancellations and skipped stops from trip updates

  Precision and recall are measured against these labels. Lead time counts from when we first *saw* an alert, not from its stated start, which can be backdated.
- **Outputs:**
  - events in `data/anomalies/date=…/` with type, line, vehicle, start and end time, score and the evidence behind them
  - an "Incidents" panel in the Live view
  - an alert through E10

**Changes.** `anomaly.py`, a scoring step in the forecaster loop (E6), a dashboard panel, and `tests/test_anomaly.py` with replayed real incidents as fixtures.

## E16. HSL realtime feeds: alerts, trip updates and stop events

**Why.** The HFP `vp` events we collect show where each tram is and how late it is. HSL publishes three more things the platform needs:
- **Service alerts:** what is disrupted, and why.
- **Trip updates:** HSL's own predicted arrival at every remaining stop.
- **HFP stop events:** exact arrival and departure at each stop, with the scheduled time.

No public archive of alerts or trip updates exists. History only exists if we record it ourselves, so phase 0 should start early ([Principle 1](#principles)).

Sources:
- [HSL GTFS-RT](https://hsldevcom.github.io/gtfs_rt/)
- [HFP event types](https://digitransit.fi/en/developers/apis/5-realtime-api/vehicle-positions/high-frequency-positioning/)
- [GTFS-RT reference](https://gtfs.org/documentation/realtime/reference/)
- [HSL's transitdata pipeline](https://github.com/HSLdevcom/transitdata)

### What the feeds hold

Measured on 11 Nov 2026, 09:11–09:30 Helsinki time (morning peak).

| Source | Per fetch | Trams | Updates | Verdict |
|---|---|---|---|---|
| [Trip updates](https://realtime.hsl.fi/realtime/trip-updates/v2/hsl) | 1.22 MB, never gzipped, always changed | ~234 of 1312 trips | every 15 s | **Collect**, tram trips only, changed trips only: ~18 MB/day as Parquet |
| [Service alerts](https://realtime.hsl.fi/realtime/service-alerts/v2/hsl) | 27 kB; ETag returns 304 when unchanged | 3 of 23 alerts | every 5 min, and on new alerts | **Collect**, versioned, < 1 MB/day |
| HFP stop events (`ars`, `arr`, `dep`, `pde`, `pas`, `vja`, `vjout`, `tlr`, `tla`) | same MQTT connection as today | ~530 events/min at peak (2.2 MB per 10 min); traffic-light priority is 59% of the bytes | per event | **Collect** raw: ~250 MB/day, ~30 MB/day after compaction. Stop events carry the scheduled time (`ttarr`/`ttdep`); `tlr` carries the junction id (`sid`) |
| [Static GTFS](https://infopalvelut.storage.hsldev.com/gtfs/hsl.zip) | 88 MB | 33 tram routes | regenerated daily | **Collect** once a day, as a tram subset |
| [GTFS-RT vehicle positions](https://realtime.hsl.fi/realtime/vehicle-positions/v2/hsl) | 130 kB | ~117 trams | every 1 s | **Skip.** It is built from HFP, so it adds only the next-stop id (100% vs 39% in HFP) and occupancy (37%). Use it only as a Live-view fallback |
| Digitransit Routing API (GraphQL) | — | — | — | **Skip.** It needs a subscription key and repeats what the feeds above provide. Keep it for debugging |

### Findings that shape the design

- **HSL's 5-minute predictions are no better than persistence.** They have MAE 54 s against 51 s for "the delay stays the same", and they run about 45 s early. That makes them a fair external baseline, and "beats HSL" a meaningful claim. This rests on one 13-minute peak window; E5 confirms it on weeks of data. No published benchmark against HSL's predictions was found.
- **The two delay definitions differ.** HSL's predicted arrival delay at the next stop is about 50 s lower than HFP's delay (median). Settle whether we predict arrival or departure delay before comparing.
- **Trip updates have no delay field.** They give only absolute arrival and departure times, so delay is the predicted time minus the scheduled time from static GTFS. That makes static GTFS a dependency, not an extra.
- **Predictions change only at stop events.** 93.6% of predictions are identical between 15-s polls. Polling every 30 s and storing only changed trips loses little: about 3.5 GB/day downloaded instead of 7 GB, and ~18 MB/day stored.
- **Trams include route type 900.** Line 15 is light rail (route `2015`, type 900, not 0) and makes up 18% of tram records. Filtering on type 0 alone, or on a `10xx` route pattern, silently drops it.
- **Today's disruptions appeared in trip updates, not alerts.** Trip updates showed 2 cancelled trips and 100 planned night trips with skipped stops. All 3 tram alerts were informational. Disruption labels need both sources.
- **Alert quirks:**
  - Some alerts name only stop ids, with the routes only in the text, so match on tram stop ids too.
  - Alerts are re-published unchanged every ~5.5 min with a new ETag, so deduplicate on a content hash.
  - Alerts are revised and withdrawn in place, so keep every version with `first_seen` and `last_seen`.
- **Bug found and fixed (2026-11-11):** time since scheduled departure (`journey_elapsed_s`, the model's most important feature) was off by 24 h for trips that start after midnight. That was 18,008 rows, 2.7%, on 10 Nov. HFP start times use a 24-hour clock; GTFS-RT uses times past 24:00.

### Joining HFP and GTFS-RT

117 of 118 live trams matched. The one miss, `1013E`, is an extra trip that is not in static GTFS, though it is in trip updates.

```python
def gtfs_trip_key(route, hfp_dir, oday, start, local_time):
    h, m = map(int, start.split(":"))              # HFP start: 24-hour local clock
    if local_time.date().isoformat() != oday and h < 12:
        h += 24                                    # GTFS-RT: 25:10 for 01:10 on the next morning
    return f"{route}|{int(hfp_dir) - 1}|{oday.replace('-', '')}|{h:02d}:{m:02d}:00"
# trip updates: f"{route_id}|{direction_id}|{start_date}|{start_time}"; vehicle id: f"{oper}/{veh}"
```

- HFP `route` equals GTFS `route_id`, and HFP direction 1/2 is GTFS 0/1. With the wrong direction mapping, 8 trips still "matched", so it fails quietly.
- About 30 keys a day belong to two trips, so the vehicle is part of the key wherever one is available.
- No trip IDs: HSL changes them whenever static GTFS is regenerated.

**Leakage rule.** Every join with trip updates or alerts is *as of the time we fetched them*: take the latest snapshot fetched at or before the forecast time, never the final version. Backtests replay stored snapshots.

### Design

- **Stop events:** `ingest.py` subscribes to the stop and journey events on its existing MQTT connection. They aren't downsampled and are written to `data/raw/events_YYYY-MM-DD.jsonl`. At the same time:
  - move to TLS on port 8883, as HSL recommends
  - key downsampling on (operator, vehicle) instead of vehicle alone
- **Poller:** a new compose service, a small Python loop capped at 256 MB:
  - alerts every 60 s with ETag
  - trip updates every 30 s, tram routes only, changed trips only
- **Static GTFS:** `tram-gtfs.timer` fetches it daily with `If-Modified-Since` and keeps the tram subset as `data/static/gtfs/asof=YYYY-MM-DD/`.
- **Storage:** day partitions with fixed Arrow schemas and the E2 manifest. All of it is covered by the hourly Allas backup.
  - `stop_visits/date=…`: per vehicle and stop, scheduled, arrival and departure times, dwell and delay
  - `trip_updates/date=…`: fetch time, trip key, stop, predicted times, schedule relationship
  - `alerts/alert_versions.parquet`: id, content hash, first and last seen, cause, effect, severity, periods, routes, stops, text in fi/sv/en

### Phases

| Phase | Goal | Changes | Effort | Needs |
|---|---|---|---|---|
| 0. Start archiving | Begin the history that can't be backfilled | Poller service (raw alerts and tram trip updates); stop-event subscriptions in the ingester; a 1-hour probe of event volume first | 1–1.5 d | E3 |
| 1. Static GTFS, versioned | Join keys, stop sequences, scheduled headways | `gtfs_static.py`, the daily timer, a trip-key helper with tests (after midnight, DST, duplicate keys) | 1 d | 0 |
| 2. Stop visits | Exact arrival, dwell and segment-time truth | `process` handles event partitions; `vja`/`vjout` journey boundaries cross-check the current \|delay\| > 30 min stale-journey rule | 1.5–2 d | 0, E2 |
| 3. Alerts in the dashboard | Explain *why* a line is late | `alerts.py` versioning (planned vs reactive); alert badges on the Live line board; alert timeline in History | 1.5 d | 0 |
| 4. Feed freshness and cross-checks | Detect a stale or disagreeing feed | Last-change time per source; stale after 90 s (trip updates) or 10 min (alerts), per [GTFS-RT best practices](https://gtfs.org/documentation/realtime/realtime-best-practices/); HFP vs trip-update disagreement; Live view falls back to GTFS-RT vehicle positions when our MQTT link is stale | 1–1.5 d | 0; feeds E4, E10 |
| 5. HSL as an external baseline | "Does the model beat HSL?" | As-of join; HSL's predicted delay at t + 5 min, interpolated between stops; scored with persistence and our model on rows where all three exist; HSL's coverage reported separately; added to the backtest, the Model health tab and the E7 gate | 1.5–2 d | 1; E5, E6; ~1 week of trip updates |
| 6. Features and labels | Use the new data in the models | E14: segment times, stop headways, HSL's predicted delay, an "alert active on this line" flag, all as-of. E15: reactive alerts, cancellations and skipped stops as weak labels | +1 d each on E14 and E15 | 2, 3, 5; 2–3 weeks of data |
| 7. Traffic-light priority (optional) | Signal-priority denial rate per junction and hour | DuckDB aggregate and dashboard panel, if trams do send `tlr`/`tla` | 0.5–1 d | 2 |

**Phase 0 done 2026-11-11.**
- **New `gtfsrt` service:**
  - polls alerts every 60 s (with ETag) and trip updates every 30 s
  - keeps tram trips only (33 routes from static GTFS, refreshed daily, plus lettered variants such as `1013E`)
  - writes a line only when a trip changes, and saves the alert list only when its content changes
  - stores one gzip member per poll in `data/raw/gtfsrt/`
  - a live 150-s run averaged ~15 kB per poll, about 40 MB/day
- **Collector:**
  - stop and journey events go to `events_YYYY-MM-DD.jsonl`, unsampled
  - TLS on port 8883
  - downsampling keyed on (operator, vehicle)
- **Backups and compaction:** the backup now snapshots every file still being written today, and compaction also gzips old event files.
- **Disk:** about 120 MB/day in all, so the 25 GB volume lasts about 6 months. Phase 2, which turns events into Parquet, is the place to cut that.
- **Dependency pin:** `gtfs-realtime-bindings` is held at 2.x because 3.x needs protobuf 7, which Streamlit 1.54 doesn't allow.

**Licence.** Digitransit data is CC BY 4.0. The attribution must name Digitransit and give the time the data was received ([terms](https://digitransit.fi/en/developers/apis/7-terms-of-use/)).

---

## Deliberately not planned

| Tool | Reason | When it would make sense |
|---|---|---|
| Kafka | HSL's MQTT broker already acts as the stream; there is one producer at about 10 messages/s after sampling | When 3 or more independent consumers need to replay the stream |
| Kubernetes | A handful of containers on one VM | When several nodes or autoscaling are needed |
| Airflow, Spark | Three scheduled jobs and gigabytes, not terabytes | When there are more than 10 dependent tasks or terabytes of data |
| Feast | E14 builds the point-in-time feature table directly | When a second model shares the features |
