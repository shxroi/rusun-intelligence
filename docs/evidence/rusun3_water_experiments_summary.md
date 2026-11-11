# Rusun 3 Water Experiment Summary

## Scope

These experiments used Supabase water data with:

- tower aggregation;
- six tower series (`tower_A` through `tower_F`);
- 12 rolling weekly origins;
- 7-day horizon;
- 30-day lookback;
- three LSTM seeds, or one deterministic Ridge model;
- water reset tolerance of 1.0.

The cleaned input had 34 meter decreases converted to missing. The aggregated
series had approximately 0.052% missing days.

Water values have not been confirmed to represent cubic metres. The metrics
must therefore be interpreted as database water-reading units or counts.

## Results

| Model | MAE | RMSE | WAPE | Bias | Building error |
|---|---:|---:|---:|---:|---:|
| `avg4` | **10.670** | **16.242** | **8.100%** | -0.497 | **5.511%** |
| `last_week` | 15.562 | 24.647 | 11.813% | -0.022 | 9.659% |
| LSTM | 11.349 | 17.454 | 8.615% | -1.078 | 6.040% |
| Ridge | 12.586 | 18.408 | 9.554% | -3.686 | 6.489% |

## Interpretation

`avg4` is the best method on every main metric. The LSTM is close but does not
beat it:

- LSTM MAE is approximately 6.4% higher than `avg4`;
- LSTM RMSE is approximately 7.5% higher;
- LSTM WAPE is approximately 6.4% higher;
- LSTM building error is higher than `avg4`.

Ridge performs worse than both `avg4` and LSTM, especially in bias. The results
do not justify deploying either machine-learning model for water forecasting.

## Recommendation

Use `avg4` as the current water benchmark and fallback. Keep the LSTM and Ridge
results as experimental evidence only.

Before further deployment decisions, confirm:

- whether water readings are counts or cubic metres;
- the meaning of `daily_usage` and `total_usage`;
- whether the 1.0 activity threshold is appropriate;
- whether the tower grouping matches the physical buildings;
- performance on an untouched future period.

## Limitations

- Metrics are not confirmed to be in cubic metres.
- Water was evaluated at tower level, not per apartment.
- The 12 origins overlap and are not independent monthly tests.
- This is a rolling weekly evaluation, not calendar-month billing validation.
- No untouched post-training holdout was used.

## Source files

- [Water tower LSTM](./water_tower/summary.csv)
- [Water tower Ridge](./water_tower_ridge/summary.csv)
