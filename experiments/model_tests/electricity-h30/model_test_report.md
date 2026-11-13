# Model tests: Electricity 30-day

Backtest: `experiments\final\r3_elec_h30\backtest_unit_origin.csv` | 12 origins (2026-03-31 to 2026-06-16), 4,251 unit-origin rows, horizon 30 days, 7-day origin spacing. Error sign: forecast - actual.


## Checks and verdicts

| id | check | status | detail |
| --- | --- | --- | --- |
| DATA-01 | no missing values in used columns | PASS | 0 missing cells |
| DATA-02 | one row per (origin, id) | PASS | 0 duplicate keys |
| DATA-03 | no negative forecasts | PASS | 0 rows |
| DATA-04 | avg4 (base x horizon) is positive | PASS | 0 rows |
| GATE-01 | claim gate: >= 8 origins and >= 500 unit-origin rows | PASS | 12 origins, 4,251 rows |
| MOD-08 | relative MAE vs avg4: pooled ratio < 1 and 95% CI entirely below 1 | PASS | 0.919 [0.865, 0.970] (+8.1% vs baseline) |
| MOD-09 | Diebold-Mariano (HLN) vs avg4, absolute-error loss, one-sided p < 0.05 | PASS (INDICATIVE: overlapping targets) | T=12 h=5 stat=-1.86 p=0.0452 wins=10/12 |
| MOD-10 | lstm (walk-forward) unit: coverage within 5 points of 80% | PASS | 76.3% [73.6, 79.1], below 13.0% / above 10.7% |
| MOD-10 | lstm (walk-forward) tower (sum of unit bounds): coverage within 5 points of 80% | OVER (conservative band) | 97.2% [91.7, 100.0], below 2.8% / above 0.0% |
| MOD-10 | lstm (walk-forward) building (sum of bounds): coverage within 5 points of 80% | OVER (conservative band) | 100.0% [100.0, 100.0], below 0.0% / above 0.0% |
| MOD-10 | avg4 (walk-forward) unit: coverage within 5 points of 80% | PASS | 76.7% [73.9, 79.0], below 12.5% / above 10.8% |
| MOD-10 | avg4 (walk-forward) tower (sum of unit bounds): coverage within 5 points of 80% | OVER (conservative band) | 97.2% [91.7, 100.0], below 2.8% / above 0.0% |
| MOD-10 | avg4 (walk-forward) building (sum of bounds): coverage within 5 points of 80% | OVER (conservative band) | 100.0% [100.0, 100.0], below 0.0% / above 0.0% |


## Headline metrics (horizon sums)

| method | n_rows | n_origins | mean_actual | mae | rmse | wape_pct | bias | cv_rmse_pct | nmbe_pct | building_err_pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lstm | 4251 | 12 | 513.262 | 73.524 | 105.802 | 14.325 | -20.164 | 20.614 | -3.929 | 6.162 |
| avg4 | 4251 | 12 | 513.262 | 79.982 | 113.264 | 15.583 | -22.377 | 22.068 | -4.360 | 6.828 |
| last_week | 4251 | 12 | 513.262 | 91.424 | 131.637 | 17.812 | -6.464 | 25.647 | -1.259 | 5.876 |


## MOD-08 Relative MAE (below 1 = better than the baseline)

| model | baseline | measure | rel_mae | ci95_lo | ci95_hi | improvement_pct | note |
| --- | --- | --- | --- | --- | --- | --- | --- |
| lstm | avg4 | pooled (sum |err| model / sum |err| baseline) | 0.919 | 0.865 | 0.970 | 8.074 | CI: bootstrap over origins |
| lstm | avg4 | geometric mean over origins | 0.919 | 0.861 | 0.971 | 8.134 | CI: bootstrap over origins |
| lstm | avg4 | weighted geometric mean over units | 0.927 | 0.902 | 0.952 | 7.349 | CI: bootstrap over units; 0 units excluded (zero MAE); model better for 66% of units |
| lstm | last_week | pooled (sum |err| model / sum |err| baseline) | 0.804 | 0.720 | 0.912 | 19.579 | CI: bootstrap over origins |
| lstm | last_week | geometric mean over origins | 0.820 | 0.731 | 0.920 | 17.965 | CI: bootstrap over origins |
| lstm | last_week | weighted geometric mean over units | 0.767 | 0.740 | 0.795 | 23.314 | CI: bootstrap over units; 0 units excluded (zero MAE); model better for 78% of units |


## MOD-09 Diebold-Mariano with HLN correction (h = 5; H1: model loss lower)

| model | versus | loss | T | h | mean_loss_diff | wins | dm | hln_stat | p_two_sided | p_model_better | significant_at_alpha | status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lstm | avg4 | absolute-error loss (MAE) | 12 | 5 | -6.494 | 10 | -2.977 | -1.857 | 0.090 | 0.045 | True | ok |
| lstm | avg4 | squared-error loss (RMSE) | 12 | 5 | -1655.345 | 8 | -2.014 | -1.256 | 0.235 | 0.118 | False | ok |
| lstm | last_week | absolute-error loss (MAE) | 12 | 5 | -18.001 | 11 | -3.108 | -1.938 | 0.079 | 0.039 | True | ok |
| lstm | last_week | squared-error loss (RMSE) | 12 | 5 | -6186.003 | 10 | -2.711 | -1.691 | 0.119 | 0.059 | False | ok |


## MOD-11 Mincer-Zarnowitz (actual on forecast; ideal intercept 0, slope 1)

| method | space | intercept | se_intercept | slope | se_slope | wald_F | p_joint_a0_b1 | r2 | n | clusters |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lstm | level (kWh) | 35.428 | 12.021 | 0.969 | 0.019 | 4.431 | 0.039 | 0.682 | 4251 | 12 |
| lstm | relative (divided by avg4) | -0.593 | 0.453 | 1.644 | 0.463 | 2.507 | 0.127 | 0.282 | 4251 | 12 |
| avg4 | level (kWh) | 83.018 | 13.093 | 0.876 | 0.016 | 30.315 | 0.000 | 0.649 | 4251 | 12 |
| last_week | level (kWh) | 154.863 | 22.001 | 0.707 | 0.037 | 32.152 | 0.000 | 0.591 | 4251 | 12 |
| last_week | relative (divided by avg4) | 0.806 | 0.349 | 0.279 | 0.313 | 2.670 | 0.113 | 0.021 | 4251 | 12 |


## Bias breakdown

| group | value | method | n | mean_actual | bias | nmbe_pct | mae |
| --- | --- | --- | --- | --- | --- | --- | --- |
| usage_tercile | low | lstm | 1417 | 347.320 | -17.355 | -4.997 | 66.081 |
| usage_tercile | low | avg4 | 1417 | 347.320 | -35.970 | -10.356 | 77.669 |
| usage_tercile | low | last_week | 1417 | 347.320 | -27.031 | -7.783 | 83.673 |
| usage_tercile | mid | lstm | 1417 | 524.295 | -23.775 | -4.535 | 74.411 |
| usage_tercile | mid | avg4 | 1417 | 524.295 | -33.078 | -6.309 | 82.500 |
| usage_tercile | mid | last_week | 1417 | 524.295 | -9.092 | -1.734 | 88.318 |
| usage_tercile | high | lstm | 1417 | 668.170 | -19.360 | -2.898 | 80.081 |
| usage_tercile | high | avg4 | 1417 | 668.170 | 1.917 | 0.287 | 79.776 |
| usage_tercile | high | last_week | 1417 | 668.170 | 16.729 | 2.504 | 102.281 |
| tower | tower_A | lstm | 701 | 505.755 | -2.578 | -0.510 | 108.181 |
| tower | tower_A | avg4 | 701 | 505.755 | -14.758 | -2.918 | 120.040 |
| tower | tower_A | last_week | 701 | 505.755 | -3.023 | -0.598 | 123.231 |
| tower | tower_B | lstm | 714 | 492.892 | -19.747 | -4.006 | 68.113 |
| tower | tower_B | avg4 | 714 | 492.892 | -23.380 | -4.743 | 75.353 |
| tower | tower_B | last_week | 714 | 492.892 | -7.768 | -1.576 | 86.103 |
| tower | tower_C | lstm | 713 | 498.508 | -16.630 | -3.336 | 64.980 |
| tower | tower_C | avg4 | 713 | 498.508 | -24.092 | -4.833 | 72.836 |
| tower | tower_C | last_week | 713 | 498.508 | -11.063 | -2.219 | 95.477 |
| tower | tower_D | lstm | 712 | 582.756 | -34.000 | -5.834 | 67.727 |
| tower | tower_D | avg4 | 712 | 582.756 | -30.446 | -5.224 | 73.597 |
| tower | tower_D | last_week | 712 | 582.756 | -7.379 | -1.266 | 85.724 |
| tower | tower_E | lstm | 720 | 500.341 | -26.325 | -5.261 | 70.381 |
| tower | tower_E | avg4 | 720 | 500.341 | -24.920 | -4.981 | 75.590 |
| tower | tower_E | last_week | 720 | 500.341 | -6.754 | -1.350 | 83.674 |
| tower | tower_F | lstm | 691 | 499.004 | -21.402 | -4.289 | 62.021 |
| tower | tower_F | avg4 | 691 | 499.004 | -16.336 | -3.274 | 62.655 |
| tower | tower_F | last_week | 691 | 499.004 | -2.620 | -0.525 | 74.421 |


## MOD-10 Interval coverage (mode: walkforward; nominal 80%)

| method | level | nominal | n_origins | coverage | ci95_lo | ci95_hi | below_lower | above_upper | rel_width |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lstm (walk-forward) | unit | 0.800 | 6 | 0.763 | 0.736 | 0.791 | 0.130 | 0.107 | 0.409 |
| lstm (walk-forward) | tower (sum of unit bounds) | 0.800 | 6 | 0.972 | 0.917 | 1.000 | 0.028 | 0.000 | 0.409 |
| lstm (walk-forward) | building (sum of bounds) | 0.800 | 6 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.409 |
| avg4 (walk-forward) | unit | 0.800 | 6 | 0.767 | 0.739 | 0.790 | 0.125 | 0.108 | 0.456 |
| avg4 (walk-forward) | tower (sum of unit bounds) | 0.800 | 6 | 0.972 | 0.917 | 1.000 | 0.028 | 0.000 | 0.456 |
| avg4 (walk-forward) | building (sum of bounds) | 0.800 | 6 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.456 |


## Seed spread

| n_seeds | mae_ensemble | mae_seed_min | mae_seed_max | seed_spread_pct_of_ensemble |
| --- | --- | --- | --- | --- |
| 3 | 73.524 | 73.933 | 74.703 | 1.047 |


## Limits

- Walk-forward coverage is a proxy on horizon sums, not the daily bands served by the API.
- DM and bootstrap treat origins as the unit of replication; with few origins results are indicative.
- Tower and building bands sum unit bounds, which tends to over-cover.
