# Development log

How this project was built, in order: what each stage set out to do, the decisions behind it, what it achieved, and what went wrong on the way. For the plan and the full design of each item, see [ROADMAP.md](ROADMAP.md). For how to run things, see the [README](../README.md).

**Keeping it current:** add an entry whenever a stage is finished, using the same headings. Numbers come from the stage's own reports and commits, which are linked.

## Timeline

| Date | Stage | Outcome | Commits |
|---|---|---|---|
| Feb–Sep 2026 | [Prototype](#prototype-febsep-2026) | Collector, cleaning, 5-min delay model, dashboard, Docker, CI | `2515711`…`154cc00` |
| 9 Nov | [E3 Always-on host](#e3-always-on-host-on-csc-cpouta-9-nov) | 24/7 collection on CSC cPouta, public HTTPS dashboard, deploy on green CI, backups | `f0ed115`…`2ce46cf` |
| 10–11 Nov | [E1 Reproducible pipeline](#e1-reproducible-pipeline-and-ci-1011-nov) | A fresh clone trains; CI runs the whole pipeline | `33aaace` |
| 11 Nov | [E2 Day-partitioned storage](#e2-incremental-day-partitioned-storage-11-nov) | Processing runs on the server: 2.2 GB → 0.8 GB peak | `eca1b61`, `8086d9a`, `7c2d0f4` |
| 11 Nov | [E16 phase 0 HSL realtime feeds](#e16-phase-0-archive-hsls-realtime-feeds-11-nov) | Archiving service alerts, trip predictions and stop events | `2fcaae4`, `4524954`, `97b2a99` |
| 11 Nov | [E6 Live accuracy](#e6-forecaster-service-and-live-accuracy-11-nov) | The model's live forecasts are scored 5 minutes later | `36fdd06` |
| 11 Nov | [Dashboard pages](#dashboard-live-history-model-health-system-11-nov) | Live, History, Model health and System pages | `8874ca4`…`26960d2` |
| 11 Nov | [E5 Backtesting](#e5-backtesting-11-nov) | Day-by-day backtest with confidence intervals; found the late-tram weakness | `6fdc0f8` |
| 11 Nov | [E18 Chat assistant](#e18-chat-assistant-11-nov) | Ask page: an LLM answers tram questions through read-only tools | `1312484`…`b87e2cd` |
| 11 Nov | [E17 Late-tram weakness](#e17-late-tram-weakness-11-nov) | Absolute-error loss; the model now beats persistence at every delay level | `5afe34b` |

---

## Prototype (Feb–Sep 2026)

**Goal.** Collect live tram positions, clean them, forecast delays and show the result.

**Built.**
- An MQTT collector for HSL's high-frequency positioning feed, keeping one record per tram every 10 s
- Cleaning into Parquet, using Helsinki local time
- A gradient-boosting model that predicts each tram's delay 5 minutes ahead
- A Streamlit dashboard, Docker setup and CI

**Key decisions.**
- **Predict the change in delay, not the delay.** The model adds its prediction to the delay now, so "the delay stays the same" (persistence) is the natural baseline to beat.
- **Time-based split with purging.** Training rows whose 5-minute target falls in the test period are dropped. Without this, the improvement looked like 45% instead of 32%.
- **Report results by segment.** Trams waiting at their terminus are scored apart from trams en route, because the waiting ones flatter the headline number.

**Data-quality findings** (each fixed and covered by a test; full table in the README):
- HSL's delay field `dl` is positive when a tram is *ahead* of schedule, so every delay had the wrong sign.
- Trams left signed into finished journeys report fake delays of 1–2.5 h, which pushed the mean delay to +237 s instead of −3 s.
- The broker sometimes delivers a message twice.
- Line "000" is not a public line.
- Timestamps are UTC.

---

## E3 Always-on host on CSC cPouta (9 Nov)

**Goal.** The model needs weeks of data covering rush hours and weekends, so collection must run around the clock.

**Built.**
- A `standard.small` VM (Ubuntu 24.04) on CSC cPouta, OpenStack project 2022568
- A separate 25 GB data volume, so the data outlives the VM
- Docker Compose for all services
- Caddy serving the dashboard over HTTPS with a Let's Encrypt certificate
- A pull-based deploy: a systemd timer fast-forwards to new commits on `main` once every GitHub check has passed
- Hourly backups to CSC Allas, an S3-compatible store

**Key decisions.**
- **Pull, don't push, deploys.** The server only reads from GitHub, so it needs no inbound SSH from CI and no deploy keys.
- **Memory caps per container.** The dashboard is capped at 1 GB and the collector is the last thing the kernel kills, so a traffic spike can't stop data collection.
- **Backups never delete.** They copy only, and upload a snapshot of the file being written, because rclone aborts if a file grows during upload.

**Problems.**
- The OpenStack application credential could not create S3 keys (Keystone 403). The fix was CSC's own `allas_conf` tool, with `enable-backups.sh` reading the keys it writes.

**Guide.** [deploy/cpouta/README.md](../deploy/cpouta/README.md).

---

## E1 Reproducible pipeline and CI (10–11 Nov)

**Goal.** A fresh clone should run the whole pipeline. The old sample was two one-minute bursts, too short to train on.

**Built.**
- `scripts/make_sample.py` cuts a continuous 45-minute sample: 119 trams, 1.2 MB.
- CI runs ruff and mypy, pytest on Python 3.11 and 3.13 with an 80% coverage gate, then the process → train pipeline from the command line, then a Docker build.

**Result.** On the sample, the model scores 35.6 s MAE against 63.0 s for persistence. An end-to-end test fails if the model loses to persistence en route.

---

## E2 Incremental, day-partitioned storage (11 Nov)

**Goal.** Processing loaded all raw data at once. A full day peaked at 2.2 GB of memory, more than the server has.

**Built.**
- One Parquet partition per day (`date=YYYY-MM-DD/part.parquet`), always written with the same fixed schema
- A manifest of what was processed, keyed by file hash, so unchanged days are skipped
- Finished raw days are gzipped
- An hourly processing timer on the server

**Results.**
- Peak memory: 2.2 GB → 0.8 GB, by cleaning in 50k-record chunks and storing text as Arrow strings.
- Size on disk: a day's partition is 17 MB; gzipped raw days are 8× smaller.
- Speed: History queries in DuckDB take 0.01–0.13 s per day.

**Problems.**
- **Trips starting after midnight.** HSL writes start times on a 24-hour clock, so a trip starting at 01:50 on operating day D actually starts on D + 1. The "time since departure" feature came out as about 24 h for 2.7% of rows on 10 Nov (fixed in `7c2d0f4`).

---

## E16 phase 0: archive HSL's realtime feeds (11 Nov)

**Goal.** Nobody archives HSL's service alerts or arrival predictions, so their history only exists if it is recorded as it happens. It can't be backfilled later.

**Built.**
- A `gtfsrt` service that polls trip updates every 30 s and alerts every 60 s, keeping tram trips only and writing only what changed
- Unsampled stop and journey events from the collector
- The collector moved to TLS on port 8883

**Results.** About 120 MB of data a day in total, so the 25 GB volume lasts about 6 months.

**Problems.**
- **Stale job image.** `docker compose up --build` only rebuilt running services, so the hourly processing job kept an old image. That image read the new events file as positions and overwrote the 11 Nov partition. The fix: every service shares one image tag, so every deploy updates the jobs too (`97b2a99`).

---

## E6 Forecaster service and live accuracy (11 Nov)

**Goal.** The true answer to every forecast arrives 5 minutes later, so the system can measure its own live accuracy instead of guessing at drift.

**Built.**
- A forecaster service that forecasts every tram every 15 s (about 0.5 s per cycle for ~115 trams)
- Logging of one forecast per tram every 30 s
- Scoring of each logged forecast 5 minutes later against persistence, using `features.future_delay`, the same function that builds the training target

**Results.** On the last 12 hours of data: en-route MAE 36.3 s against 43.9 s for persistence.

**Problems.**
- Training peaked at 1.2 GB, too much for the server, so models are trained on the laptop and copied to the server. Fixing this is part of E7.

---

## Dashboard: Live, History, Model health, System (11 Nov)

**Built.** Four pages:
- **Live:** map, line status board, HSL alerts and cancelled trips
- **History:** DuckDB over the day partitions, by line, hour, tram and stop
- **Model health:** live error of the model against persistence
- **System:** how fresh each data source is, processing runs, the deployed model and disk use

**Key decision.** Every query is an aggregate computed in DuckDB, so dashboard memory doesn't grow with the number of days collected.

---

## E5 Backtesting (11 Nov)

**Goal.** One test window can't separate a few seconds of gain from noise.

**Built.** `python -m tram_mlops.evaluate` runs a rolling-origin backtest:
- Each day is predicted by a model trained only on earlier days.
- Two baselines: persistence, and a line × hour average of the change in delay.
- A bootstrap that resamples whole trips gives 95% intervals.
- Breakdowns by hour, line, weekday vs weekend and current delay.
- `--update-readme` writes the results into the README.

**Results.**
- En route, the model was 4.1 s better than persistence, 95% CI [−4.5, −3.8], so the gain is real.
- The line × hour baseline is no better than persistence: delays don't drift predictably by line or hour.

**Finding.** The model was *worse* than persistence for trams already more than 3 minutes late: 115.9 s against 82.5 s above 5 minutes. This became E17.

---

## E18 Chat assistant (11 Nov)

**Goal.** Let anyone ask questions in plain language, such as "When is the next tram at Lasipalatsi?", answered from the project's own data.

**Built.**
- An **Ask** page on the dashboard, with example questions and the lookups behind each answer
- `chat.py`: a tool-calling loop over any OpenAI-compatible API. The default is Groq's free tier, with `openai/gpt-oss-120b` and `gpt-oss-20b` as fallback.
- `chat_tools.py`: six read-only tools:
  - next trams at a stop
  - stop search
  - live line status, with the model's forecasts
  - service alerts
  - punctuality history
  - forecast accuracy

**Key decisions.**
- **Tools, not text-to-SQL.** The model picks a function and simple arguments; it can't run its own queries or change anything.
- **A "current trips" snapshot.** Scanning the day's archive took 19 s per question. The archiver now also writes the latest poll to one small file, so lookups take milliseconds.
- **Stop names from static GTFS.** The realtime feeds name stops only by id. Some line 15 stops are typed as bus stops, so a stop counts as a tram stop when a tram trip uses it.
- **A shared quota.** Visitors share one free quota, so questions are capped per minute, per day and per visit.

**Problems found in the first live run, all fixed:**
- English questions got Finnish answers, because the tool data is in Finnish.
- Delays came out as signed numbers ("–2 min") instead of words.
- A rarely run variant line (3H) topped the punctuality ranking even when listed separately. Variants are now named, not ranked.

---

## E17 Late-tram weakness (11 Nov)

**Goal.** Make the model at least as good as persistence at every level of current delay. This is also the condition E7 will check before promoting a model.

**Diagnosis.**
- For trams more than 5 minutes late, the model forecast a 40–67 s recovery; they actually lost about 10 s more.
- 10 of 295 trips caused 57% of the excess error, mostly line 3 on 10 Nov, with delays up to 29 min.
- The training data didn't show a recovery on average. Squared-error loss was being pulled by the rare trams whose delay suddenly dropped a lot.

**Candidates tried** (test day 11 Nov, MAE for trams more than 5 min late; persistence 59.8 s):

| Variant | MAE |
|---|---:|
| Squared error (before) | 69.5 s |
| **Absolute error** | **53.5 s** |
| Late rows weighted ×5 | 80.1 s |
| Persistence above 5 min | 59.8 s |

**Change.** `[model] loss = "absolute_error"`. The score is MAE, which absolute error optimizes directly. A regression test checks that absolute error beats persistence on synthetic late trams, and that squared error fails, so the test is known to catch the bug.

**Results** (backtest of 10–11 Nov):
- More than 5 min late: 68.7 s against 74.1 s for persistence (was 115.9 s).
- All trams en route: 37.6 s, 5.8 s better than persistence, 95% CI [−6.0, −5.7].

**Deployed.** Model `20261111T185951Z` went live on 11 Nov. Rollback: point `/srv/tram/models/latest.json` back at `20261111T084019Z`.

---

## Lessons across stages

- **Measure on the server's limits, not the laptop's.** Memory drove E2's design, and it's still open for training.
- **Every new baseline or breakdown found something.** Segments exposed the waiting-at-terminus effect, the backtest exposed late trams, and the live chatbot run exposed language and ranking problems. Check slices before trusting a headline number.
- **Deploys need the same care as code.** The stale-image incident cost a day's partition. Shared images, deploy only on green CI, and backups that never delete all came from problems like it.
- **Archive what can't be backfilled early.** HSL's predictions and alerts are now usable for the chatbot, and later for a "beats HSL" benchmark, only because archiving started before there was a use for them.

## Current state and next steps (11 Nov 2026)

**Running.** On one cPouta VM:
- **Collection:** the collector and the GTFS-RT archiver
- **Forecasting:** the forecaster, using the E17 model
- **Dashboard:** five pages, behind Caddy
- **Scheduled jobs:** hourly processing and backups, and deploys on green CI

**Data.** Collected since 9 Nov, weekdays only so far.

**Skipped for now.** E7 (scheduled retraining with a promotion gate). It first needs training to fit in the server's memory.

**Other open items.** See the roadmap's [Status table](ROADMAP.md#status). Next in order: E4 data-quality reports and E15 anomaly detection, which needs about 2 weeks of data. Also: rerun the E5 backtest as weekends and more weeks arrive.
