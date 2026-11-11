# NusaWatt intelligence integration

This folder is the clean GitHub-ready package for the NusaWatt forecasting
and dashboard integration. It separates deployable runtime code from
experiment evidence and does not contain raw meter data, secrets, model
weights, or the local Python environment.

## Folder layout

```text
intelligence_github/
├── app/
│   ├── forecast_job.py       # train, forecast, and evaluate scheduled jobs
│   ├── forecast_api.py       # thin FastAPI service for the dashboard
│   ├── nusawatt_lstm_v2.py   # current electricity forecasting implementation
│   ├── nusawatt_common.py    # shared cleaning and forecasting helpers
│   └── nusawatt_services.py  # anomaly and peak-load analysis
├── tests/
├── docs/evidence/
│   ├── configs/              # frozen run configurations
│   ├── figures/              # selected final plots and service figures
│   └── *.md                  # experiment summaries
├── .env.example
├── .gitignore
└── requirements.txt
```

## Runtime architecture

```text
Supabase/PostgreSQL
        │
        ▼
app/forecast_job.py train
        │
        ▼
forecast_store/models/
        │
        ▼
app/forecast_job.py forecast
        │
        ▼
forecast_store/runs/
        │
        ▼
app/forecast_api.py
        │
        ▼
dashboard
```

The API reads stored forecast files for forecasts and reads the Supabase catalog
for dashboard unit/tower identifiers. It does not load TensorFlow. The
dashboard should call `/api/forecast` and `/api/catalog`, not read experiment
CSV files directly.

## Setup

From this folder on Windows:

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Edit `.env` locally. Never commit `.env`. The required database value is a
read-only PostgreSQL connection string:

```text
SUPABASE_DB_URL=postgresql://readonly_user:password@host:5432/postgres?sslmode=require
```

The job currently reads `SUPABASE_DB_URL` from the process environment. If the
deployment does not load `.env` automatically, configure the variables in the
service or scheduler before running the job.

## Train and forecast

Run these commands from the `app` directory, or set `PYTHONPATH` to it:

```powershell
Set-Location .\app

..\.venv\Scripts\python.exe forecast_job.py train `
  --rusun 3 --utility ELECTRICITY --start-date 2025-09-01 `
  --horizon 7 --store ..\forecast_store

..\.venv\Scripts\python.exe forecast_job.py forecast `
  --rusun 3 --utility ELECTRICITY --horizon 7 `
  --store ..\forecast_store
```

For water, use the same commands with `--utility WATER`. Water is currently
stored at tower level and uses the `avg4` fallback unless a suitable model
forecast is available.

## Start the API

Set the store and optional authentication variables, then start FastAPI:

```powershell
$env:FORECAST_STORE = ".\forecast_store"
$env:API_KEY = "set-a-local-secret"
$env:ALLOWED_ORIGINS = "http://localhost:3000"
.\.venv\Scripts\python.exe -m uvicorn app.forecast_api:app --host 127.0.0.1 --port 8000
```

Health check:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/api/health
```

The analytics frontend uses `VITE_INTELLIGENCE_API_URL` (see its
`.env.example`) and must be able to reach this API. Keep `SUPABASE_DB_URL` only
in the intelligence service environment; never put it in the frontend.

The dashboard sends a `POST` request to `/api/forecast` with `unitIds`,
`horizonDays`, and `metric`. Supported metrics are `electricity` and
`water`. Cost forecasting is intentionally not implemented because tariff
rules have not been confirmed.

## Tests

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The tests cover cleaning, reset handling, missingness, leakage prevention,
forecast windows, eligibility, and API identifier resolution. A full live
Supabase train-to-API test still requires a configured database and should be
performed in a controlled environment.

## Evidence included

The `docs/evidence` directory contains the artifacts behind the figures and numbers below:

| Artifact | Path |
|---|---|
| Rusun 3 electricity — 7-day frozen config | `docs/evidence/configs/r3_elec_h7.json` |
| Rusun 3 electricity — 30-day frozen config | `docs/evidence/configs/r3_elec_h30.json` |
| Rusun 3 electricity — Ridge frozen config | `docs/evidence/configs/r3_elec_h7_ridge.json` |
| Rusun 3 water frozen config (tower aggregated) | `docs/evidence/configs/r3_water.json` |
| Rusun 3 water Ridge frozen config | `docs/evidence/configs/r3_water_ridge.json` |
| Forecast model figures (electricity 7-day) | `docs/evidence/figures/electricity_7d/` |
| Forecast model figures (electricity 30-day) | `docs/evidence/figures/electricity_30d/` |
| Cross-run comparison figure | `docs/evidence/figures/comparison/compare_runs.png` |
| Anomaly / peak-load / forecast service figures | `docs/evidence/figures/services/` |
| Anomaly / peak-load / forecast service figures (water) | `docs/evidence/figures/services_water/` |
| Peak-load tables (electricity) | `docs/evidence/figures/services/peak_dow_profile.csv`, `peak_weekday_weekend.csv`, `peak_monthly_top_days.csv` |
| Peak-load tables (water) | `docs/evidence/figures/services_water/peak_dow_profile.csv`, `peak_weekday_weekend.csv`, `peak_monthly_top_days.csv` |
| Seeded UAT anomaly check | `docs/evidence/figures/services/seeded_check.csv` |
| Experiment summaries and recommendations | `docs/evidence/*.md` |

The evidence is not runtime input. Do not copy the full experiment folders
into the application deployment.

## Evaluation conclusions

### Forecast evidence (Rusun 3, frozen `run_config.json` outputs)

| Horizon | Model | MAE | RMSE | WAPE | Bias | Building error | vs `avg4` MAE |
|---|---|---|---|---|---|---|---|
| 7-day | `avg4` | 21.381 | 29.920 | 18.494% | -0.897 | 8.707% | baseline |
| 7-day | `last_week` | 23.318 | 32.548 | 20.170% | +0.209 | 11.644% | +9.1% |
| 7-day | **LSTM** | **18.125** | **25.039** | **15.678%** | -2.318 | **7.039%** | **-15.2%** |
| 7-day | Ridge | 19.487 | 26.545 | 16.856% | -1.902 | 8.109% | -8.9% |
| 30-day | `avg4` | 79.982 | 113.264 | 15.583% | -22.377 | 6.828% | baseline |
| 30-day | `last_week` | 91.424 | 131.637 | 17.812% | -6.464 | **5.876%** | +14.3% |
| 30-day | **LSTM** | **73.524** | **105.802** | **14.325%** | -20.164 | 6.162% | **-8.1%** |

- 7-day: LSTM is the strongest accuracy model on MAE, RMSE, WAPE, and building
  error, beating `avg4` by ~15.2% and winning 22 of 24 origins; Ridge wins 19 of 24.
- 30-day rolling: LSTM improves MAE ~8.1% versus `avg4`, but `last_week` retains
  the best building-level error. Not yet a billing evaluation.
- Water: `avg4` best (MAE 12.901, RMSE 19.830, WAPE 10.021%, building error
  9.579%); LSTM 13.340; Ridge 14.415. Water units are still unconfirmed.

### Peak-load evidence (electricity, daily, 2025-09 to 2026-07)

- Highest average building demand: Sunday (~5,878/day, +3.9% vs the weekly mean);
  lowest: Friday (~5,381).
- Weekends run ~0.7% above weekdays on average.
- Monthly top days: 2026-04-26 (~7,836, +3.03 z in month), 2026-01-24 (~7,099,
  +2.33), 2025-09-27 (~6,975, +2.01).
- Hourly peak windows need hourly data (not used here).

### Anomaly evidence (electricity, z > 3, 56-day rolling baseline)

- Flags per unit-day: `high_consumption`, `low_consumption`, `zero_reading`,
  `missing_data`, `meter_decrease`.
- Seeded UAT: 4/4 planted anomalies detected — high_consumption (z ~8.2), low_consumption
  (z ~4.2), a zero-reading run, and a missing-data run.
- Water anomalies are computed on tower-aggregated series because per-unit
  readings are mostly count-like; see `docs/evidence/figures/services_water/fig_anomaly.png`.

### Feature forecast

- Supports 7-day and 30-day rolling forecasts (weekly origins); model output
  lands in `forecast_next.csv` from `nusawatt_lstm_v2.py --forecast` and is plotted by
  `nusawatt_services.py --forecast-csv` into `fig_forecast.png`.
- The LSTM forecast is the recommended model output; `avg4` is the level fallback.
- The API reads stored forecast files; the dashboard should call `/api/forecast`,
  not read experiment CSV files directly.

## Key evidence figures

| Metric | Image |
|---|---|
| Forecast error, 7-day horizon | `docs/evidence/figures/electricity_7d/01_metrics.png` |
| Forecast: actual vs predicted, 7-day | `docs/evidence/figures/electricity_7d/03_pred_vs_actual.png` |
| Forecast error, 30-day horizon | `docs/evidence/figures/electricity_30d/01_metrics.png` |
| Forecast: actual vs predicted, 30-day | `docs/evidence/figures/electricity_30d/03_pred_vs_actual.png` |
| Forecast model output vs hist | `docs/evidence/figures/services/fig_forecast.png` |
| Peak load: weekday profile | `docs/evidence/figures/services/peak_dow_profile.csv` |
| Peak load: daily building demand | `docs/evidence/figures/services/fig_peak_load.png` |
| Anomaly detection example | `docs/evidence/figures/services/fig_anomaly.png` |
| Anomaly example (water, tower aggregated) | `docs/evidence/figures/services_water/fig_anomaly.png` |
| Cross-run model comparison | `docs/evidence/figures/comparison/compare_runs.png` |

## Security and data handling

Do not commit:

- `.env` or database credentials;
- raw CSV/database dumps;
- `forecast_store` outputs containing unit-level values;
- per-unit anomaly or forecast exports;
- `.venv`;
- model weight files unless a separate private artifact policy exists.

The selected evidence is appropriate for a private or public repository only
after confirming that the aggregate tables and images do not expose sensitive
resident-level information.
