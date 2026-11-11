# Loss and Model Comparison Summary

## Scope

These experiments use the Supabase electricity data with the v2 pipeline:

- 24 rolling weekly origins
- 7-day forecast horizon
- 30-day lookback
- Three LSTM seeds
- 1 kWh/day minimum activity threshold
- Retraining every four origins
- No holiday features

The Rusun 3 runs start on 2025-09-01. The Rusun 1 runs start on
2025-04-01.

## Rusun 3 results

| Model | MAE | WAPE | Bias | Building error |
|---|---:|---:|---:|---:|
| `avg4` | 21.381 | 18.494% | -0.897 | 8.707% |
| LSTM MAE | **18.125** | **15.678%** | -2.318 | **7.039%** |
| LSTM Huber | 18.170 | 15.717% | -1.929 | 7.141% |
| LSTM MSE | 18.509 | 16.010% | **-0.546** | 7.371% |
| LSTM Huber + calibration | 19.437 | 16.813% | +2.700 | 9.825% |
| LSTM MAE + calibration | 19.186 | 16.595% | +2.321 | 9.375% |
| Ridge | 19.487 | 16.856% | -1.902 | 8.109% |

### Rusun 3 interpretation

The ordinary MAE LSTM has the best pure forecasting accuracy. Huber is nearly
as accurate and reduces the negative bias, making it the better compromise when
under-estimation matters for billing. MSE produces the closest-to-zero bias but
has worse MAE and building error than Huber.

The validation calibration option overcorrected the forecasts in this run. It
changed negative bias into positive bias and increased building-level error.
Calibration should not be used in production based on these results.

Ridge is competitive and simpler, but its MAE is approximately 7.5% higher than
the MAE LSTM. This indicates that the LSTM contributes useful non-linear
prediction ability beyond the ratio formulation and input features.

## Rusun 1 results

| Model | MAE | WAPE | Bias | Building error |
|---|---:|---:|---:|---:|
| `avg4` | 25.159 | 37.222% | +1.723 | 15.723% |
| `last_week` | 24.153 | 35.733% | +1.029 | 13.274% |
| Ridge | **22.717** | **33.609%** | +2.356 | **12.609%** |
| LSTM Huber + calibration | 22.869 | 33.834% | +2.882 | 15.900% |

### Rusun 1 interpretation

Ridge is the strongest tested option for Rusun 1. It improves both unit-level
accuracy and building-level error relative to `avg4`. However, it still has
positive bias, so aggregate forecasts should continue to be monitored.

Huber with calibration improves MAE but worsens building-level error. It is not
the preferred Rusun 1 model.

## Recommended decision

### Rusun 3

Use the Huber LSTM as the best balanced shadow-mode candidate when billing bias
is important. Retain the MAE LSTM as the pure-accuracy reference.

### Rusun 1

Use Ridge as the current shadow-mode candidate because it improves both MAE and
building error. Do not use it for official billing until its positive bias is
understood and monitored over new periods.

### Calibration

Do not enable `--calibrate` for production. It increased building error in both
tested calibrated runs.

## Limitations

- These are rolling weekly forecasts, not calendar-month billing evaluations.
- The 24 origins overlap, so they are not 24 fully independent monthly tests.
- Rusun 3 and Rusun 1 use different start dates and should not be compared as
  if they were identical datasets.
- Bias is measured in forecast-minus-actual kWh per unit forecast horizon.
- No water model was evaluated here.

## Source files

- [Rusun 3 MAE baseline](./loss_base/summary.csv)
- [Rusun 3 Huber](./loss_huber/summary.csv)
- [Rusun 3 MSE](./loss_mse/summary.csv)
- [Rusun 3 Huber calibrated](./huber_calibrated/summary.csv)
- [Rusun 3 MAE calibrated](./mae_calibrated/summary.csv)
- [Rusun 3 Ridge](./ridge/summary.csv)
- [Rusun 1 Ridge](./rusun1_ridge/summary.csv)
- [Rusun 1 Huber calibrated](./rusun1_huber_calibrated/summary.csv)
