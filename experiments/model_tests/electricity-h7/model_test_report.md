# Model tests: Electricity 7-day

Backtest: `experiments\final\r3_elec_h7\backtest_unit_origin.csv` | 24 origins (2026-01-29 to 2026-07-09), 8,500 unit-origin rows, horizon 7 days, 7-day origin spacing. Error sign: forecast - actual.


## Checks and verdicts

| id | check | status | detail |
| --- | --- | --- | --- |
| DATA-01 | no missing values in used columns | PASS | 0 missing cells |
| DATA-02 | one row per (origin, id) | PASS | 0 duplicate keys |
| DATA-03 | no negative forecasts | PASS | 0 rows |
| DATA-04 | avg4 (base x horizon) is positive | PASS | 0 rows |
| GATE-01 | claim gate: >= 8 origins and >= 500 unit-origin rows | PASS | 24 origins, 8,500 rows |
| MOD-08 | relative MAE vs avg4: pooled ratio < 1 and 95% CI entirely below 1 | PASS | 0.848 [0.803, 0.895] (+15.2% vs baseline) |
| MOD-09 | Diebold-Mariano (HLN) vs avg4, absolute-error loss, one-sided p < 0.05 | PASS | T=24 h=1 stat=-4.92 p=0.0000 wins=22/24 |
| MOD-10 | lstm (walk-forward) unit: coverage within 5 points of 80% | PASS | 77.6% [71.5, 83.0], below 11.4% / above 11.0% |
| MOD-10 | lstm (walk-forward) tower (sum of unit bounds): coverage within 5 points of 80% | OVER (conservative band) | 92.4% [81.8, 100.0], below 3.0% / above 4.5% |
| MOD-10 | lstm (walk-forward) building (sum of bounds): coverage within 5 points of 80% | OVER (conservative band) | 90.9% [77.3, 100.0], below 4.5% / above 4.5% |
| MOD-10 | avg4 (walk-forward) unit: coverage within 5 points of 80% | PASS | 75.2% [67.8, 81.6], below 12.3% / above 12.5% |
| MOD-10 | avg4 (walk-forward) tower (sum of unit bounds): coverage within 5 points of 80% | OVER (conservative band) | 90.9% [78.0, 100.0], below 5.3% / above 3.8% |
| MOD-10 | avg4 (walk-forward) building (sum of bounds): coverage within 5 points of 80% | OVER (conservative band) | 90.9% [77.3, 100.0], below 4.5% / above 4.5% |


## Headline metrics (horizon sums)

| method | n_rows | n_origins | mean_actual | mae | rmse | wape_pct | bias | cv_rmse_pct | nmbe_pct | building_err_pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lstm | 8500 | 24 | 115.610 | 18.125 | 25.039 | 15.678 | -2.318 | 21.658 | -2.005 | 7.039 |
| avg4 | 8500 | 24 | 115.610 | 21.381 | 29.920 | 18.494 | -0.897 | 25.880 | -0.776 | 8.707 |
| last_week | 8500 | 24 | 115.610 | 23.318 | 32.548 | 20.170 | 0.209 | 28.153 | 0.181 | 11.644 |


## MOD-08 Relative MAE (below 1 = better than the baseline)

| model | baseline | measure | rel_mae | ci95_lo | ci95_hi | improvement_pct | note |
| --- | --- | --- | --- | --- | --- | --- | --- |
| lstm | avg4 | pooled (sum |err| model / sum |err| baseline) | 0.848 | 0.803 | 0.895 | 15.230 | CI: bootstrap over origins |
| lstm | avg4 | geometric mean over origins | 0.856 | 0.817 | 0.896 | 14.375 | CI: bootstrap over origins |
| lstm | avg4 | weighted geometric mean over units | 0.860 | 0.846 | 0.875 | 13.958 | CI: bootstrap over units; 0 units excluded (zero MAE); model better for 84% of units |
| lstm | last_week | pooled (sum |err| model / sum |err| baseline) | 0.777 | 0.710 | 0.849 | 22.272 | CI: bootstrap over origins |
| lstm | last_week | geometric mean over origins | 0.803 | 0.746 | 0.865 | 19.654 | CI: bootstrap over origins |
| lstm | last_week | weighted geometric mean over units | 0.779 | 0.766 | 0.793 | 22.064 | CI: bootstrap over units; 0 units excluded (zero MAE); model better for 94% of units |


## MOD-09 Diebold-Mariano with HLN correction (h = 1; H1: model loss lower)

| model | versus | loss | T | h | mean_loss_diff | wins | dm | hln_stat | p_two_sided | p_model_better | significant_at_alpha | status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lstm | avg4 | absolute-error loss (MAE) | 24 | 1 | -3.266 | 22 | -5.030 | -4.924 | 0.000 | 0.000 | True | ok |
| lstm | avg4 | squared-error loss (RMSE) | 24 | 1 | -269.106 | 22 | -4.728 | -4.628 | 0.000 | 0.000 | True | ok |
| lstm | last_week | absolute-error loss (MAE) | 24 | 1 | -5.198 | 21 | -4.396 | -4.304 | 0.000 | 0.000 | True | ok |
| lstm | last_week | squared-error loss (RMSE) | 24 | 1 | -432.433 | 23 | -3.476 | -3.403 | 0.002 | 0.001 | True | ok |


## MOD-11 Mincer-Zarnowitz (actual on forecast; ideal intercept 0, slope 1)

| method | space | intercept | se_intercept | slope | se_slope | wald_F | p_joint_a0_b1 | r2 | n | clusters |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lstm | level (kWh) | 6.324 | 3.070 | 0.965 | 0.019 | 2.247 | 0.128 | 0.702 | 8500 | 24 |
| lstm | relative (divided by avg4) | -0.356 | 0.183 | 1.394 | 0.181 | 4.619 | 0.021 | 0.387 | 8500 | 24 |
| avg4 | level (kWh) | 12.704 | 2.773 | 0.897 | 0.022 | 13.971 | 0.000 | 0.578 | 8500 | 24 |
| last_week | level (kWh) | 28.560 | 7.354 | 0.752 | 0.048 | 16.164 | 0.000 | 0.551 | 8500 | 24 |
| last_week | relative (divided by avg4) | 0.137 | 0.206 | 0.885 | 0.190 | 0.284 | 0.756 | 0.254 | 8500 | 24 |


## Bias breakdown

| group | value | method | n | mean_actual | bias | nmbe_pct | mae |
| --- | --- | --- | --- | --- | --- | --- | --- |
| usage_tercile | low | lstm | 2834 | 78.868 | -3.429 | -4.347 | 15.188 |
| usage_tercile | low | avg4 | 2834 | 78.868 | -4.829 | -6.122 | 19.295 |
| usage_tercile | low | last_week | 2834 | 78.868 | -3.679 | -4.665 | 19.410 |
| usage_tercile | mid | lstm | 2833 | 116.517 | -2.871 | -2.464 | 17.910 |
| usage_tercile | mid | avg4 | 2833 | 116.517 | -1.927 | -1.654 | 20.903 |
| usage_tercile | mid | last_week | 2833 | 116.517 | -0.898 | -0.771 | 23.511 |
| usage_tercile | high | lstm | 2833 | 151.458 | -0.655 | -0.433 | 21.278 |
| usage_tercile | high | avg4 | 2833 | 151.458 | 4.066 | 2.685 | 23.946 |
| usage_tercile | high | last_week | 2833 | 151.458 | 5.205 | 3.436 | 27.036 |
| tower | tower_A | lstm | 1392 | 116.433 | -1.231 | -1.057 | 20.810 |
| tower | tower_A | avg4 | 1392 | 116.433 | -0.488 | -0.419 | 26.678 |
| tower | tower_A | last_week | 1392 | 116.433 | 1.292 | 1.109 | 26.465 |
| tower | tower_B | lstm | 1424 | 111.801 | -2.199 | -1.967 | 16.623 |
| tower | tower_B | avg4 | 1424 | 111.801 | -0.828 | -0.740 | 19.733 |
| tower | tower_B | last_week | 1424 | 111.801 | 0.064 | 0.057 | 21.880 |
| tower | tower_C | lstm | 1430 | 111.117 | -1.955 | -1.759 | 19.033 |
| tower | tower_C | avg4 | 1430 | 111.117 | -1.410 | -1.269 | 21.602 |
| tower | tower_C | last_week | 1430 | 111.117 | -0.302 | -0.272 | 24.675 |
| tower | tower_D | lstm | 1429 | 129.586 | -3.381 | -2.609 | 19.064 |
| tower | tower_D | avg4 | 1429 | 129.586 | -1.914 | -1.477 | 21.726 |
| tower | tower_D | last_week | 1429 | 129.586 | -0.196 | -0.151 | 24.608 |
| tower | tower_E | lstm | 1437 | 110.926 | -2.764 | -2.492 | 16.968 |
| tower | tower_E | avg4 | 1437 | 110.926 | -1.070 | -0.965 | 19.973 |
| tower | tower_E | last_week | 1437 | 110.926 | -0.180 | -0.163 | 21.705 |
| tower | tower_F | lstm | 1388 | 113.782 | -2.350 | -2.066 | 16.269 |
| tower | tower_F | avg4 | 1388 | 113.782 | 0.377 | 0.332 | 18.635 |
| tower | tower_F | last_week | 1388 | 113.782 | 0.618 | 0.543 | 20.584 |


## MOD-10 Interval coverage (mode: walkforward; nominal 80%)

| method | level | nominal | n_origins | coverage | ci95_lo | ci95_hi | below_lower | above_upper | rel_width |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lstm (walk-forward) | unit | 0.800 | 22 | 0.776 | 0.715 | 0.830 | 0.114 | 0.110 | 0.504 |
| lstm (walk-forward) | tower (sum of unit bounds) | 0.800 | 22 | 0.924 | 0.818 | 1.000 | 0.030 | 0.045 | 0.504 |
| lstm (walk-forward) | building (sum of bounds) | 0.800 | 22 | 0.909 | 0.773 | 1.000 | 0.045 | 0.045 | 0.504 |
| avg4 (walk-forward) | unit | 0.800 | 22 | 0.752 | 0.678 | 0.816 | 0.123 | 0.125 | 0.607 |
| avg4 (walk-forward) | tower (sum of unit bounds) | 0.800 | 22 | 0.909 | 0.780 | 1.000 | 0.053 | 0.038 | 0.607 |
| avg4 (walk-forward) | building (sum of bounds) | 0.800 | 22 | 0.909 | 0.773 | 1.000 | 0.045 | 0.045 | 0.607 |


## Seed spread

| n_seeds | mae_ensemble | mae_seed_min | mae_seed_max | seed_spread_pct_of_ensemble |
| --- | --- | --- | --- | --- |
| 3 | 18.125 | 18.132 | 18.291 | 0.877 |


## Limits

- Walk-forward coverage is a proxy on horizon sums, not the daily bands served by the API.
- DM and bootstrap treat origins as the unit of replication; with few origins results are indicative.
- Tower and building bands sum unit bounds, which tends to over-cover.
