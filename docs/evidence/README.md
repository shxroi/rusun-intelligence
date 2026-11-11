# Experiment evidence

This directory records the validated experiments that support the runtime
integration. It is documentation and review material, not a data source for
the dashboard.

## Final forecast runs

| Folder | Purpose |
|---|---|
| `figures/electricity_7d/` | Rusun 3 electricity, 7-day LSTM, 24 origins |
| `figures/electricity_7d_ridge/` | Simpler Ridge comparator |
| `figures/electricity_30d/` | Rusun 3 electricity, rolling 30-day horizon |
| `figures/water_7d/` | Rusun 3 water, tower aggregation |
| `figures/comparison/` | Cross-run comparison |

Each forecast figure set contains metrics, per-origin errors, forecast versus
actual values, building totals, bias by origin, and horizon-day errors.
Electricity sets also contain usage-decile errors.

## Service outputs

`figures/services/` contains the electricity anomaly, peak-load, and forecast
figures plus aggregate peak summaries. `figures/services_water/` contains the
corresponding water outputs.

`seeded_check.csv` is a synthetic anomaly-detection/UAT result. The original
unit-level `anomalies.csv` files are intentionally not copied here because
they may expose unit identifiers and operational details.

## Main conclusions

- Electricity 7-day LSTM: MAE 18.125, WAPE 15.678%, 22/24 origin wins.
- Electricity 30-day LSTM: MAE 73.524, WAPE 14.325%, 10/12 origin wins.
- Water `avg4`: MAE 12.901, WAPE 10.021%, and 9.579% building error; it
  remains the water benchmark.

These are rolling-origin results with overlapping origins. They do not prove
billing accuracy, universal deployment, or water values in cubic metres.

## Reproducibility

The frozen command settings are stored in `configs/`. The source implementation
used by the production job is `app/nusawatt_lstm_v2.py`; the final research
water evaluation used a newer evaluation script in the original workspace.
The production job has deliberately not been switched to that research script
without a separate integration test.
