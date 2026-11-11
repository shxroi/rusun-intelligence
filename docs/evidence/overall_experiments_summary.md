# Overall Forecasting Experiments Summary

## Executive conclusion

The strongest result is the v2 hybrid ratio LSTM for **Rusun 3 electricity**.
It consistently improves unit-level weekly forecasts over the `avg4` baseline,
including the longer 24-origin evaluation.

However, the model is not yet a billing or universal building-level solution.
Rusun 1 shows a large unit-level improvement but persistent aggregation and bias
problems. For Rusun 3 water, the simple `avg4` baseline is better than both
LSTM and Ridge.

Recommended status:

- **Rusun 3 electricity:** v2 LSTM shadow-mode candidate.
- **Rusun 1 electricity:** shadow mode only; compare LSTM and Ridge by business
  objective.
- **Rusun 3 water:** use `avg4` as the current benchmark and fallback.
- **Rusun 2 and Rusun 4:** no quality claims until data coverage and identity
  issues are resolved.
- **Official billing:** not approved.

## Data and pipeline validation

### Supabase versus CSV parity

The Supabase and CSV electricity values matched on the overlapping Rusun 3
period:

- 80,738 matched unit-days;
- maximum usage difference approximately 0.00043 kWh;
- 100% of matched values within 0.01 kWh.

The sources do not have identical scopes or date ranges, so this is value parity
on the overlap, not full row-count parity.

### Data-quality findings

- Rusun 3 has the strongest and most usable electricity history.
- Rusun 1 has approximately 1% cleaned missing cells and 199 meter decreases
  converted to missing in the April-start run.
- Rusun 3 water has six aggregated tower series and approximately 0.052% missing
  aggregated-series days after cleaning.
- Rusun 2 has insufficient history for reliable claims.
- Rusun 4 has severe missingness and identity conflicts, including conflicting
  duplicate unit-days.
- Water units have not been confirmed as counts or cubic metres.

## Experiment inventory

| Area | Experiments |
|---|---|
| Earlier electricity models | raw LSTM, ratio LSTM, direct LSTM, monthly baselines |
| Transfer | Rusun 3-trained model transferred to Rusun 1 and Rusun 3 |
| v2 Rusun 3 | 12 origins, 24 origins, ablations, loss comparison, Ridge |
| v2 Rusun 1 | 12/24-origin tests, October-start test, Ridge comparison |
| Long horizon | direct rolling 30-day Rusun 3 forecast |
| Water | tower-level LSTM and Ridge |

## Earlier model results

The initial raw and direct LSTM experiments did not reliably beat recent-history
baselines. The ratio formulation improved results, but the strongest improvement
came from the later v2 hybrid design:

- unit-relative ratio target;
- direct multi-horizon output;
- recent 28-day base;
- zero days retained;
- no month features;
- weighted ratio loss;
- rolling-origin evaluation;
- seed ensemble.

The Rusun 3 transfer experiment was negative: a model trained only on Rusun 3
did not reliably transfer better than local baselines on another building.
Rusun 3 is therefore a useful data-quality benchmark, not an automatic
universal training source.

## Rusun 3 electricity

### Standard v2: 12 origins

| Model | MAE | WAPE | Bias | Building error |
|---|---:|---:|---:|---:|
| `avg4` | 21.680 | 17.814% | -1.730 | 5.788% |
| `last_week` | 23.041 | 18.932% | -0.619 | 8.685% |
| LSTM v2 | **18.560** | **15.250%** | -3.200 | **4.718%** |

The LSTM won 10 of 12 origins. The paired bootstrap interval for LSTM minus
`avg4` MAE was [-4.670, -1.676] kWh.

### Standard v2: 24 origins

| Model | MAE | WAPE | Bias | Building error |
|---|---:|---:|---:|---:|
| `avg4` | 21.381 | 18.494% | -0.897 | 8.707% |
| `last_week` | 23.318 | 20.170% | +0.209 | 11.644% |
| LSTM v2 | **18.125** | **15.678%** | -2.318 | **7.039%** |

The LSTM won **21 of 24 origins** and improved MAE and WAPE by approximately
15.2% versus `avg4`.

### Loss and model comparison: 24 origins

| Model | MAE | WAPE | Bias | Building error |
|---|---:|---:|---:|---:|
| LSTM MAE | **18.125** | **15.678%** | -2.318 | **7.039%** |
| LSTM Huber | 18.170 | 15.717% | -1.929 | 7.141% |
| LSTM MSE | 18.509 | 16.010% | **-0.546** | 7.371% |
| Huber + calibration | 19.437 | 16.813% | +2.700 | 9.825% |
| MAE + calibration | 19.186 | 16.595% | +2.321 | 9.375% |
| Ridge | 19.487 | 16.856% | -1.902 | 8.109% |

The plain MAE LSTM remains the best pure-accuracy model. Huber is nearly tied
and has slightly less negative bias, but the difference is within seed noise.
MSE reduces bias but costs accuracy. Calibration overcorrects and worsens
building error. Ridge captures much of the gain and is simpler, but is still
less accurate than LSTM.

### Ablations

The no-calendar, no-refit, no-weight, one-seed, and minimum-5-kWh runs showed
only modest differences. These experiments do not prove that any individual
component is the causal source of the improvement. The standard configuration
remains a reasonable frozen configuration.

## Rusun 1 electricity

### 24-origin April-start runs

| Model | MAE | WAPE | Bias | Building error |
|---|---:|---:|---:|---:|
| `avg4` | 25.159 | 37.222% | +1.723 | 15.723% |
| `last_week` | 24.153 | 35.733% | +1.029 | 13.274% |
| LSTM MAE | **20.082** | **29.710%** | -5.197 | 13.606% |
| LSTM Huber | **20.040** | **29.647%** | -4.511 | **13.475%** |

The MAE LSTM won **21 of 24 origins**. Huber won **20 of 24 origins**.

The LSTM improves unit-level error, but `last_week` still has lower
building-level error. Negative bias creates under-estimation risk.

### October-start runs

| Model | MAE | WAPE | Bias | Building error |
|---|---:|---:|---:|---:|
| `avg4` | 22.639 | 33.170% | -4.731 | 10.733% |
| `last_week` | 23.154 | 33.925% | -2.666 | 11.155% |
| LSTM MAE | **20.146** | **29.518%** | -10.699 | 16.927% |
| Ridge | 21.590 | 31.634% | -4.070 | **10.096%** |

The October-start result reinforces the risk: LSTM unit MAE improves, but
building error worsens substantially. Ridge is less accurate at unit level but
has the best aggregate error in this comparison.

## Rusun 3 30-day electricity horizon

This is a rolling 30-day test with weekly origins, not a calendar-month billing
test.

| Model | MAE | WAPE | Bias | Building error |
|---|---:|---:|---:|---:|
| `avg4` | 79.982 | 15.583% | -22.377 | 6.828% |
| `last_week` | 91.424 | 17.812% | -6.464 | **5.876%** |
| LSTM v2 | **73.524** | **14.325%** | -20.164 | 6.162% |

The LSTM improves unit-level MAE/WAPE by approximately 8.1% versus `avg4`, but
`last_week` has the best building error. The result is promising but
insufficient for official cost or billing claims.

## Rusun 3 water

Water was aggregated to six tower series because per-unit readings were mostly
binary or count-like.

| Model | MAE | RMSE | WAPE | Bias | Building error |
|---|---:|---:|---:|---:|---:|
| `avg4` | **10.670** | **16.242** | **8.100%** | -0.497 | **5.511%** |
| `last_week` | 15.562 | 24.647 | 11.813% | -0.022 | 9.659% |
| LSTM | 11.349 | 17.454 | 8.615% | -1.078 | 6.040% |
| Ridge | 12.586 | 18.408 | 9.554% | -3.686 | 6.489% |

The water LSTM won **5 of 12 origins**. Ridge won **3 of 12 origins**.
Both were worse than `avg4` on average. `avg4` is the current water benchmark
and fallback.

The water unit remains unresolved. Results must be reported as database units
or counts until cubic-metre semantics are confirmed.

## Cross-experiment conclusions

### What is supported

- Direct Supabase ingestion works for the tested electricity and water paths.
- Supabase and CSV electricity values match on the overlapping Rusun 3 period.
- The v2 ratio LSTM materially improves Rusun 3 electricity unit-level forecasts.
- The Rusun 3 electricity result survives the 24-origin evaluation.
- Ridge captures a substantial part of the v2 improvement and is a valid
  simpler comparator.
- Water aggregation is more appropriate than per-unit water forecasting.
- `avg4` is a strong and difficult-to-beat baseline.

### What is not supported

- Universal promotion across all buildings.
- Official billing or cost-estimation use.
- Guaranteed building-level improvement from LSTM.
- Water forecasting in cubic metres.
- Treating rolling 30-day forecasts as calendar-month forecasts.
- Treating overlapping origins as fully independent observations.

## Frozen recommendation

1. Keep the standard v2 MAE configuration as the Rusun 3 electricity reference.
2. Use `avg4` as the operational fallback everywhere.
3. Keep Rusun 3 electricity LSTM in shadow mode.
4. Evaluate Rusun 1 with both unit-level and building-level gates; do not select
   a final model solely from MAE.
5. Keep `avg4` for water until units and semantics are confirmed.
6. Do not enable the validation calibration option based on current results.
7. Freeze commands, seeds, configuration files, and output directories before
   the final holdout and UAT.

## Remaining work

1. Run water with 24 origins.
2. Run the direct 30-day horizon for cost estimation.
3. Freeze final configurations, seeds, commands, and `run_config.json` files.
4. Run true calendar-month evaluation.
5. Obtain and evaluate an untouched post-training holdout if permitted.
6. Confirm tariff, billing-cycle, fixed-charge, tax, and subsidy rules.
7. Define UAT checks for data freshness, missingness, fallback behavior, model
   output, aggregate totals, bias alerts, and reproducibility.

## Related reports

- [Original v2 experiment comparison](./lstm_v2_experiments_summary.md)
- [Loss and model comparison](./loss_model_comparison_summary.md)
- [Rusun 1 electricity report](./rusun1_electricity_experiments_summary.md)
- [Rusun 3 water report](./rusun3_water_experiments_summary.md)
