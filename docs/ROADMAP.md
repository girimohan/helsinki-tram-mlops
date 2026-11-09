# Engineering roadmap

This plan turns the current pipeline into a system that runs on its own:
- it collects data continuously
- it retrains on a schedule, but only promotes a new model when it beats the current one
- it measures its own live accuracy

It builds on the existing code instead of replacing it. `process.clean()`, `features.add_features()`, the purged time split and the Streamlit dashboard all stay.

## Principles

1. **Collection time is the bottleneck, not code.** The model's weakest segment (trams already en route) needs weeks of data covering rush hours, weekends and weather. The collector has to run permanently first; most later items depend on the data it gathers.
2. **The true answer arrives 5 minutes after each forecast.** So live accuracy can be measured directly instead of being inferred from data drift. Monitoring is designed around this.
3. **Add a tool only when the project has the problem it solves.** See [Deliberately not planned](#deliberately-not-planned).

## Status

Ordered by value per day of effort, adjusted for what each item unblocks.

| # | Enhancement | Effort | Needs | Status |
|---|---|---|---|---|
| 1 | [E1 Reproducible pipeline and CI](#e1-reproducible-pipeline-and-ci) | 1–1.5 d | none | Planned |
| 2 | [E2 Incremental, day-partitioned storage](#e2-incremental-day-partitioned-storage) | 3–4 d | none | Planned |
| 3 | [E3 Always-on host on CSC cPouta](#e3-always-on-host-on-csc-cpouta) | 2–3 d | none | **In progress**: [setup guide](../deploy/cpouta/README.md) |
| 4 | [E4 Data contracts and quality reports](#e4-data-contracts-and-quality-reports) | 2 d | E2 | Planned |
| 5 | [E6 Prediction logging and live accuracy](#e6-prediction-logging-and-live-accuracy) | 3 d | E3 | Planned |
| 6 | [E5 Backtesting](#e5-backtesting) | 2–3 d | E2, ~1 week of data | Planned |
| 7 | [E7 Promotion gate and scheduled retraining](#e7-promotion-gate-and-scheduled-retraining) | 3 d | E2, E5 | Planned |
| 8 | [E12 Packaging and container hardening](#e12-packaging-and-container-hardening) | 1.5–2 d | none | Planned |
| 9 | [E8 Experiment tracking and model registry](#e8-experiment-tracking-and-model-registry) | 2–3 d | E7 | Planned |
| 10 | [E10 Metrics, logs and alerts](#e10-metrics-logs-and-alerts) | 3 d | E3 | Planned |
| 11 | [E11 Drift detection](#e11-drift-detection) | 2 d | E6 | Planned |
| 12 | [E9 Inference API](#e9-inference-api) | 3–4 d | E6 | Planned |
| 13 | [E13 Prediction intervals](#e13-prediction-intervals) | 3–4 d | E5, ~2 weeks of data | Planned |
| 14 | [E14 Network-aware features](#e14-network-aware-features) | 1.5–2 wk | E5, ~3 weeks of data | Planned |

Items 1–7 (about 18 working days) complete the core loop. The rest adds depth.

---

## E1. Reproducible pipeline and CI

**Why.** The bundled sample is two one-minute bursts 55 minutes apart, so `train` on a fresh clone finds no 5-minute targets and stops. CI runs `process` but never `train`.

**Design.**
- `scripts/make_sample.py` cuts a continuous 45-minute slice (all trams, one record per 10 s, about 1.5 MB gzipped) from a real collection day.
- `load_raw` reads `.gz` files.
- CI matrix on Python 3.11 and 3.13: ruff, mypy on `src/`, pytest with coverage, a full `process` then `train` run, and `docker build`.

**Changes.** `scripts/make_sample.py`, `data/sample/tram_sample.jsonl.gz`, `.pre-commit-config.yaml`, `Makefile`, ruff and mypy settings in `pyproject.toml`, `ci.yml`.

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

## E3. Always-on host on CSC cPouta

**Why.** The collector has to run around the clock to build a multi-week dataset. A server also makes a live demo possible.

**Design.**
- **Phase 1 (current):**
  - one `standard.small` VM (Ubuntu 24.04) on cPouta, project 2022568
  - raw data on a separate persistent volume, so it outlives the VM
  - the collector and dashboard run with Docker Compose
  - the dashboard is only reachable through an SSH tunnel
  - Guide: [deploy/cpouta/README.md](../deploy/cpouta/README.md).
- **Phase 2:**
  - nightly backup of finished raw days to CSC Allas (S3-compatible object storage)
  - Caddy with TLS for a public dashboard URL
  - a release workflow that builds and pushes the image to GHCR and deploys over SSH
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

---

## Deliberately not planned

| Tool | Reason | When it would make sense |
|---|---|---|
| Kafka | HSL's MQTT broker already acts as the stream; there is one producer at about 10 messages/s after sampling | When 3 or more independent consumers need to replay the stream |
| Kubernetes | A handful of containers on one VM | When several nodes or autoscaling are needed |
| Airflow, Spark | Three scheduled jobs and gigabytes, not terabytes | When there are more than 10 dependent tasks or terabytes of data |
| Feast | E14 builds the point-in-time feature table directly | When a second model shares the features |
