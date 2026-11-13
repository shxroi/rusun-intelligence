# Model tests: Water 7-day

Backtest: `experiments\final\r3_water\backtest_unit_origin.csv` | 24 origins (2026-01-29 to 2026-07-09), 144 tower-origin rows, horizon 7 days, 7-day origin spacing. Error sign: forecast - actual.


## Checks and verdicts

| id | check | status | detail |
| --- | --- | --- | --- |
| DATA-01 | no missing values in used columns | PASS | 0 missing cells |
| DATA-02 | one row per (origin, id) | PASS | 0 duplicate keys |
| DATA-03 | no negative forecasts | PASS | 0 rows |
| DATA-04 | avg4 (base x horizon) is positive | PASS | 0 rows |
| GATE-01 | claim gate: >= 8 origins (row rule applies to unit level only) | PASS | 24 origins, 6 series, 144 rows; few series = low power |
| MOD-08 | relative MAE vs avg4: pooled ratio < 1 and 95% CI entirely below 1 | FAIL | 1.034 [0.999, 1.087] (-3.4% vs baseline) |
| MOD-09 | Diebold-Mariano (HLN) vs avg4, absolute-error loss, one-sided p < 0.05 | NOT SIGNIFICANT | T=24 h=1 stat=1.56 p=0.9339 wins=9/24 |
| MOD-10 | lstm (walk-forward) tower: coverage within 5 points of 80% | UNDER (band too narrow) | 62.9% [47.0, 77.3], below 18.2% / above 18.9% |
| MOD-10 | lstm (walk-forward) building (sum of bounds): coverage within 5 points of 80% | UNDER (band too narrow) | 63.6% [45.5, 81.8], below 18.2% / above 18.2% |
| MOD-10 | avg4 (walk-forward) tower: coverage within 5 points of 80% | UNDER (band too narrow) | 62.9% [47.0, 77.3], below 18.2% / above 18.9% |
| MOD-10 | avg4 (walk-forward) building (sum of bounds): coverage within 5 points of 80% | UNDER (band too narrow) | 63.6% [45.5, 81.8], below 18.2% / above 18.2% |


## Headline metrics (horizon sums)

| method | n_rows | n_origins | mean_actual | mae | rmse | wape_pct | bias | cv_rmse_pct | nmbe_pct | building_err_pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lstm | 144 | 24 | 128.743 | 13.340 | 20.294 | 10.362 | -0.194 | 15.763 | -0.151 | 9.949 |
| avg4 | 144 | 24 | 128.743 | 12.901 | 19.830 | 10.021 | 0.311 | 15.403 | 0.242 | 9.579 |
| last_week | 144 | 24 | 128.743 | 17.100 | 26.121 | 13.283 | 0.128 | 20.290 | 0.099 | 12.999 |


## MOD-08 Relative MAE (below 1 = better than the baseline)

| model | baseline | measure | rel_mae | ci95_lo | ci95_hi | improvement_pct | note |
| --- | --- | --- | --- | --- | --- | --- | --- |
| lstm | avg4 | pooled (sum |err| model / sum |err| baseline) | 1.034 | 0.999 | 1.087 | -3.403 | CI: bootstrap over origins |
| lstm | avg4 | geometric mean over origins | 1.038 | 1.004 | 1.082 | -3.850 | CI: bootstrap over origins |
| lstm | avg4 | weighted geometric mean over units | 1.032 | 1.019 | 1.050 | -3.161 | CI: bootstrap over units; 0 units excluded (zero MAE); model better for 0% of units |
| lstm | last_week | pooled (sum |err| model / sum |err| baseline) | 0.780 | 0.595 | 1.002 | 21.987 | CI: bootstrap over origins |
| lstm | last_week | geometric mean over origins | 0.838 | 0.665 | 1.067 | 16.210 | CI: bootstrap over origins |
| lstm | last_week | weighted geometric mean over units | 0.786 | 0.749 | 0.820 | 21.395 | CI: bootstrap over units; 0 units excluded (zero MAE); model better for 100% of units |


## MOD-09 Diebold-Mariano with HLN correction (h = 1; H1: model loss lower)

| model | versus | loss | T | h | mean_loss_diff | wins | dm | hln_stat | p_two_sided | p_model_better | significant_at_alpha | status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lstm | avg4 | absolute-error loss (MAE) | 24 | 1 | 0.439 | 9 | 1.594 | 1.561 | 0.132 | 0.934 | False | ok |
| lstm | avg4 | squared-error loss (RMSE) | 24 | 1 | 18.615 | 8 | 1.149 | 1.125 | 0.272 | 0.864 | False | ok |
| lstm | last_week | absolute-error loss (MAE) | 24 | 1 | -3.760 | 16 | -1.773 | -1.736 | 0.096 | 0.048 | True | ok |
| lstm | last_week | squared-error loss (RMSE) | 24 | 1 | -270.475 | 16 | -1.768 | -1.731 | 0.097 | 0.048 | True | ok |


## MOD-11 Mincer-Zarnowitz (actual on forecast; ideal intercept 0, slope 1)

| method | space | intercept | se_intercept | slope | se_slope | wald_F | p_joint_a0_b1 | r2 | n | clusters |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lstm | level (kWh) | 75.861 | 14.485 | 0.411 | 0.121 | 14.666 | 0.000 | 0.076 | 144 | 24 |
| lstm | relative (divided by avg4) | 1.861 | 0.649 | -0.863 | 0.634 | 7.035 | 0.004 | 0.013 | 144 | 24 |
| avg4 | level (kWh) | 69.221 | 16.491 | 0.461 | 0.139 | 10.176 | 0.001 | 0.085 | 144 | 24 |
| last_week | level (kWh) | 115.147 | 13.777 | 0.105 | 0.097 | 48.557 | 0.000 | 0.011 | 144 | 24 |
| last_week | relative (divided by avg4) | 1.102 | 0.189 | -0.099 | 0.181 | 19.066 | 0.000 | 0.007 | 144 | 24 |


## Bias breakdown

| group | value | method | n | mean_actual | bias | nmbe_pct | mae |
| --- | --- | --- | --- | --- | --- | --- | --- |
| usage_tercile | low | lstm | 49 | 120.265 | -6.560 | -5.454 | 13.756 |
| usage_tercile | low | avg4 | 49 | 120.265 | -5.668 | -4.713 | 13.281 |
| usage_tercile | low | last_week | 49 | 120.265 | -5.898 | -4.904 | 17.776 |
| usage_tercile | mid | lstm | 47 | 134.447 | -4.467 | -3.323 | 11.853 |
| usage_tercile | mid | avg4 | 47 | 134.447 | -3.590 | -2.671 | 11.388 |
| usage_tercile | mid | last_week | 47 | 134.447 | -4.809 | -3.577 | 16.681 |
| usage_tercile | high | lstm | 48 | 131.813 | 10.489 | 7.957 | 14.373 |
| usage_tercile | high | avg4 | 48 | 131.813 | 10.236 | 7.765 | 13.996 |
| usage_tercile | high | last_week | 48 | 131.813 | 11.113 | 8.431 | 16.822 |


## MOD-10 Interval coverage (mode: walkforward; nominal 80%)

| method | level | nominal | n_origins | coverage | ci95_lo | ci95_hi | below_lower | above_upper | rel_width |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lstm (walk-forward) | tower | 0.800 | 22 | 0.629 | 0.470 | 0.773 | 0.182 | 0.189 | 0.311 |
| lstm (walk-forward) | building (sum of bounds) | 0.800 | 22 | 0.636 | 0.455 | 0.818 | 0.182 | 0.182 | 0.311 |
| avg4 (walk-forward) | tower | 0.800 | 22 | 0.629 | 0.470 | 0.773 | 0.182 | 0.189 | 0.301 |
| avg4 (walk-forward) | building (sum of bounds) | 0.800 | 22 | 0.636 | 0.455 | 0.818 | 0.182 | 0.182 | 0.301 |


## Seed spread

| n_seeds | mae_ensemble | mae_seed_min | mae_seed_max | seed_spread_pct_of_ensemble |
| --- | --- | --- | --- | --- |
| 3 | 13.340 | 13.284 | 13.661 | 2.826 |


## Limits

- Walk-forward coverage is a proxy on horizon sums, not the daily bands served by the API.
- DM and bootstrap treat origins as the unit of replication; with few origins results are indicative.
- Tower and building bands sum unit bounds, which tends to over-cover.
