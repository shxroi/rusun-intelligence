# Final Forecasting Runs Summary

## Scope and configuration

These are the newest frozen runs using the current
[nusawatt_lstm_v3.py](../../nusawatt_lstm_v3.py) implementation and Supabase
data.

Common settings:

- Rusun 3;
- start date 2025-09-01;
- 30-day lookback;
- 1.0 activity threshold;
- three LSTM seeds, except Ridge;
- 40 maximum epochs and six-epoch patience;
- retraining every four origins;
- 28 validation days;
- no holiday features;
- no calibration.

The electricity data contains 360 units across 318 days, with approximately
0.898% missing cells after cleaning. The water data is aggregated into six tower
series and has approximately 0.052% missing aggregated-series days.

## Rusun 3 electricity: 7-day horizon

| Model | MAE | RMSE | WAPE | Bias | Building error |
|---|---:|---:|---:|---:|---:|
| `avg4` | 21.381 | 29.920 | 18.494% | -0.897 | 8.707% |
| `last_week` | 23.318 | 32.548 | 20.170% | +0.209 | 11.644% |
| **LSTM** | **18.125** | **25.039** | **15.678%** | -2.318 | **7.039%** |
| Ridge | 19.487 | 26.545 | 16.856% | -1.902 | 8.109% |

### Paired comparison against `avg4`

- LSTM wins **22 of 24 origins**.
- Ridge wins **19 of 24 origins**.
- Mean paired MAE difference:
  - LSTM: **-3.266** usage units;
  - Ridge: **-1.903** usage units.

### Interpretation

The LSTM is the best model for 7-day Rusun 3 electricity forecasts. It improves
MAE and WAPE by approximately 15.2% versus `avg4`, while also improving
building error. Ridge captures much of the improvement and is simpler, but
remains less accurate than the LSTM.

The LSTM bias is negative, so aggregate totals should still be monitored in
shadow mode.

## Rusun 3 electricity: 30-day horizon

This is a rolling 30-day forecast with weekly origins. It is not yet a true
calendar-month billing evaluation.

| Model | MAE | RMSE | WAPE | Bias | Building error |
|---|---:|---:|---:|---:|---:|
| `avg4` | 79.982 | 113.264 | 15.583% | -22.377 | 6.828% |
| `last_week` | 91.424 | 131.637 | 17.812% | -6.464 | **5.876%** |
| **LSTM** | **73.524** | **105.802** | **14.325%** | -20.164 | 6.162% |

### Paired comparison against `avg4`

- LSTM wins **10 of 12 origins**.
- Mean paired MAE difference: **-6.493** usage units.

### Interpretation

The LSTM improves 30-day unit-level MAE and WAPE by approximately 8.1% versus
`avg4`. However, `last_week` has the best building-level error, and the LSTM
remains negatively biased. This result is promising for monthly-style
forecasting but does not prove billing accuracy.

## Rusun 3 water: tower aggregation, 24 origins

Water readings were aggregated into six tower series. The water unit has not
been confirmed as counts or cubic metres, so the metrics are reported in the
database water-reading units.

| Model | MAE | RMSE | WAPE | Bias | Building error |
|---|---:|---:|---:|---:|---:|
| **`avg4`** | **12.901** | **19.830** | **10.021%** | +0.311 | **9.579%** |
| `last_week` | 17.100 | 26.121 | 13.283% | +0.128 | 12.999% |
| LSTM | 13.340 | 20.294 | 10.362% | -0.194 | 9.949% |
| Ridge | 14.415 | 20.189 | 11.197% | -2.292 | 9.580% |

### Paired comparison against `avg4`

- LSTM wins **9 of 24 origins**.
- Ridge wins **7 of 24 origins**.
- Mean paired MAE difference:
  - LSTM: **+0.439** usage units;
  - Ridge: **+1.514** usage units.

### Interpretation

`avg4` remains the best water method on MAE, RMSE, WAPE, and building error.
The LSTM is close but does not beat the baseline. Ridge performs worse and has
stronger negative bias. No water model should be promoted based on this run.

## Final recommendation

### Rusun 3 electricity

Use the v3 MAE LSTM as a **shadow-mode candidate**, with `avg4` retained as the
fallback. Ridge is a credible simpler comparator but is not the accuracy
leader.

### 30-day electricity

Continue evaluation using true calendar-month periods and billing rules before
using the model for cost or resident billing estimates.

### Rusun 3 water

Keep `avg4` as the operational benchmark and fallback. Confirm water units,
meter semantics, and tower aggregation before further deployment decisions.

### Not approved yet

- official billing;
- automatic tariff or cost decisions;
- universal deployment across Rusun 1, 2, 3, and 4;
- water forecasting in cubic metres;
- production replacement of the baseline.

## Reproducibility artifacts

- [7-day electricity LSTM](./r3_elec_h7/)
- [7-day electricity Ridge](./r3_elec_h7_ridge/)
- [30-day electricity LSTM](./r3_elec_h30/)
- [24-origin water LSTM](./r3_water/)
- [24-origin water Ridge](./r3_water_ridge/)
