"""Tests for nusawatt_common.

Run from the project root:   .venv\\Scripts\\python -m unittest discover -s tests -v
No TensorFlow needed; the forecaster is tested with a stand-in model.
"""
import os
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app"))
import nusawatt_common as nc  # noqa: E402
from forecast_api import resolve  # noqa: E402  (resolve() in the thin API)


def raw_rows(uid, start, meters, usage, rusun=1, tower="A"):
    """One row per day for one unit. Pass dates via an offset list in `start` = list of day offsets."""
    rows = []
    for off, m, u in zip(start, meters, usage):
        rows.append(dict(
            id=f"{uid}-{off}-{m}", created_at=pd.Timestamp("2026-01-01", tz="UTC"),
            timestamp=None, tower=tower, floor=1, unit_number=uid.split("_")[1],
            electricity_meter=m, usage=u, rusun=rusun, uid=uid,
            date=pd.Timestamp("2026-01-01") + pd.Timedelta(days=off),
        ))
    return rows


def make_raw(rows):
    return pd.DataFrame(rows)


class CleaningTests(unittest.TestCase):
    def test_duplicate_group_is_quarantined_not_resolved(self):
        rows = raw_rows("1_A1", range(6), [100, 110, 120, 130, 140, 150], [10] * 6)
        rows += raw_rows("1_A1", [3], [9000], [0])           # second meter on day 3
        cd = nc.clean_energy(make_raw(rows))
        self.assertTrue(np.isnan(cd.U["1_A1"].iloc[3]))
        self.assertTrue(cd.quarantined["1_A1"].iloc[3])
        self.assertEqual(int(cd.quarantined.values.sum()), 1)
        self.assertEqual(int(cd.U["1_A1"].notna().sum()), 5)

    def test_legacy_rule_keeps_highest_meter(self):
        rows = raw_rows("1_A1", range(4), [100, 110, 120, 130], [10] * 4)
        rows += raw_rows("1_A1", [2], [9000], [77])
        cd = nc.clean_energy(make_raw(rows), dedup="max_meter")
        self.assertEqual(cd.U["1_A1"].iloc[2], 77)
        self.assertEqual(int(cd.quarantined.values.sum()), 0)

    def test_meter_reset_becomes_missing_with_flag(self):
        rows = raw_rows("1_A1", range(6), [100, 110, 120, 5, 15, 25], [10, 10, 10, 10, 10, 10])
        cd = nc.clean_energy(make_raw(rows))
        self.assertTrue(np.isnan(cd.U["1_A1"].iloc[3]))
        self.assertTrue(cd.reset["1_A1"].iloc[3])
        self.assertEqual(int(cd.reset.values.sum()), 1)
        kept = nc.clean_energy(make_raw(rows), reset_policy="keep")
        self.assertEqual(kept.U["1_A1"].iloc[3], 10)

    def test_missing_day_stays_nan_on_complete_calendar(self):
        rows = raw_rows("1_A1", [0, 1, 2, 4, 5], [1, 2, 3, 5, 6], [1, 1, 1, 1, 1])
        cd = nc.clean_energy(make_raw(rows))
        self.assertEqual(len(cd.U), 6)
        self.assertTrue(np.isnan(cd.U["1_A1"].iloc[3]))
        self.assertFalse(cd.quarantined["1_A1"].iloc[3])
        self.assertFalse(cd.reset["1_A1"].iloc[3])

    def test_real_zero_is_not_missing(self):
        rows = raw_rows("1_A1", range(4), [1, 1, 1, 1], [0, 0, 0, 0])
        cd = nc.clean_energy(make_raw(rows))
        self.assertEqual(int(cd.U["1_A1"].notna().sum()), 4)


def synth(n_days=130, n_units=6, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2026-01-01", periods=n_days)
    U = pd.DataFrame(rng.uniform(10, 30, (n_days, n_units)), index=idx,
                     columns=[f"1_U{i}" for i in range(n_units)])
    return U


class LeakageTests(unittest.TestCase):
    def test_scaler_anchor_ignore_rows_after_cutoff(self):
        U = synth()
        ci = 90
        cal = nc.calendar_features(U.index)
        R7, R30 = nc.rolling_features(U)
        a = nc.fit_scaler_anchor(U, R7, R30, cal, ci)
        U2 = U.copy()
        U2.iloc[ci:] = 5000.0
        R7b, R30b = nc.rolling_features(U2)
        b = nc.fit_scaler_anchor(U2, R7b, R30b, cal, ci)
        np.testing.assert_allclose(a["scaler"].min_, b["scaler"].min_)
        np.testing.assert_allclose(a["scaler"].scale_, b["scaler"].scale_)
        self.assertEqual(a["max_usage"], b["max_usage"])
        np.testing.assert_allclose(a["mean_tr"].values, b["mean_tr"].values)

    def test_causal_active_ignores_future(self):
        U = synth()
        a = nc.active_matrix(U)
        U2 = U.copy()
        U2.iloc[80:] = 0.0
        b = nc.active_matrix(U2)
        pd.testing.assert_frame_equal(a.iloc[:81], b.iloc[:81])

    def test_training_and_inference_windows_match(self):
        U = synth(n_units=1)
        ci, t = 100, 80
        cal = nc.calendar_features(U.index)
        R7, R30 = nc.rolling_features(U)
        fit = nc.fit_scaler_anchor(U, R7, R30, cal, ci)
        X, y, tix = nc.training_windows(U, R7, R30, cal, ci, fit)
        train_win = X[list(tix).index(t)]
        ext = U.iloc[:t].tail(60).values
        anchor = np.array([fit["mean_tr"].iloc[0] / fit["max_usage"]])
        inf = nc.inference_window(ext, U.index[t - 1], fit["scaler"], anchor)[0]
        np.testing.assert_allclose(train_win, inf, atol=1e-5)


class ForecastTests(unittest.TestCase):
    class Echo:
        """Stand-in model: predicts the scaled usage of the last day of the window."""
        def predict(self, X, batch_size=None, verbose=0):
            return X[:, -1, 0:1]

    def test_recursive_forecast_starts_after_last_row_and_uses_last_value(self):
        U = synth(n_units=2)
        ci = 100
        cal = nc.calendar_features(U.index)
        R7, R30 = nc.rolling_features(U)
        fit = nc.fit_scaler_anchor(U, R7, R30, cal, ci)
        bundle = dict(model=self.Echo(), scaler=fit["scaler"], max_usage=fit["max_usage"])
        before = U.iloc[:ci]
        wk, daily = nc.forecast_recursive(bundle, before, list(U.columns))
        self.assertEqual(daily.shape, (7, 2))
        last = before.iloc[-1].values
        in_range = (last >= fit["scaler"].data_min_[0]) & (last <= fit["scaler"].data_max_[0])
        np.testing.assert_allclose(daily[0][in_range], last[in_range], rtol=1e-4)
        np.testing.assert_allclose(wk, daily.sum(axis=0))

    def test_future_rows_cannot_change_forecast(self):
        U = synth(n_units=2)
        ci = 100
        cal = nc.calendar_features(U.index)
        R7, R30 = nc.rolling_features(U)
        fit = nc.fit_scaler_anchor(U, R7, R30, cal, ci)
        bundle = dict(model=self.Echo(), scaler=fit["scaler"], max_usage=fit["max_usage"])
        a, _ = nc.forecast_recursive(bundle, U.iloc[:ci], list(U.columns))
        U2 = U.copy()
        U2.iloc[ci:] = 9999
        b, _ = nc.forecast_recursive(bundle, U2.iloc[:ci], list(U.columns))
        np.testing.assert_allclose(a, b)


class EligibilityTests(unittest.TestCase):
    def test_reasons(self):
        n, ci = 75, 60
        idx = pd.date_range("2026-01-01", periods=n)
        rng = np.random.default_rng(1)
        base = lambda: pd.Series(rng.uniform(15, 25, n), index=idx)  # noqa: E731
        U = pd.DataFrame({"ok": base(), "gap": base(), "idle": base() * 0.01, "zeros": base(),
                          "short": base(), "quar": base(), "noactual": base(), "never": np.nan}, index=idx)
        U.loc[idx[40], "gap"] = np.nan
        U.loc[idx[30:60:2][:12], "zeros"] = 0.0           # 12 of 30 days zero
        U.loc[idx[:55], "short"] = np.nan
        U.loc[idx[40], "quar"] = np.nan
        U.loc[idx[62], "noactual"] = np.nan
        Q = pd.DataFrame(False, index=idx, columns=U.columns)
        Q.loc[idx[40], "quar"] = True
        R7, R30 = nc.rolling_features(U)
        r = nc.eligibility(U, R7, R30, ci, quarantined=Q, reset=Q & False)
        self.assertEqual(r["ok"], "eligible")
        self.assertEqual(r["gap"], "missing_days_in_window")
        self.assertEqual(r["idle"], "inactive")
        self.assertEqual(r["zeros"], "too_many_zero_days")
        self.assertEqual(r["short"], "short_history")
        self.assertEqual(r["quar"], "quarantined_in_window")
        self.assertEqual(r["noactual"], "missing_actual_week")
        self.assertEqual(r["never"], "no_data_before_cutoff")
        prod = nc.eligibility(U, R7, R30, ci, quarantined=Q, reset=Q & False, require_actual=False)
        self.assertEqual(prod["noactual"], "eligible")


class ResolveTests(unittest.TestCase):
    """resolve() in forecast_api maps ids to the stored run's level.

    Water: a single unit code must resolve to ITS TOWER, not the building.
    Electricity: unit codes (optionally prefixed with rusun) and tower codes.
    Empty list / 'building' / 'all' -> building total.
    """

    def setUp(self):
        self.run = dict(level="tower", rusun=3, model_version="v2-2026-11", data_cutoff="2026-07-15",
                        interval_coverage=0.8, method="avg4", run_id="r3_water_h7_20260715_v2-2026-11")
        self.F = pd.DataFrame({
            "level": ["tower"] * 4,
            "id": ["tower_A", "tower_B", "tower_C", "tower_D"],
            "value": [1.0, 2.0, 3.0, 4.0],
            "lower": [0.9, 1.9, 2.9, 3.9],
            "upper": [1.1, 2.1, 3.1, 4.1],
            "avg4_value": [7.0, 8.0, 9.0, 10.0],
            "is_fallback": [False] * 4,
            "fallback_reason": [""] * 4,
            "horizon_day": [1, 1, 1, 1],
            "date": ["2026-07-16"] * 4,
        })

    def test_water_single_unit_maps_to_its_tower(self):
        level, ids = resolve(["3A1001"], self.run, self.F, rusun=3)
        self.assertEqual(level, "tower")
        self.assertEqual(ids, ["tower_A"])

    def test_water_unit_without_rusun_prefix_maps_to_tower(self):
        level, ids = resolve(["A1001"], self.run, self.F, rusun=3)
        self.assertEqual(level, "tower")
        self.assertEqual(ids, ["tower_A"])

    def test_water_tower_code_resolves_to_itself(self):
        level, ids = resolve(["tower_A"], self.run, self.F, rusun=3)
        self.assertEqual(level, "tower")
        self.assertEqual(ids, ["tower_A"])

    def test_electricity_unit_code_stays_unit(self):
        F = pd.DataFrame({
            "level": ["unit"] * 3,
            "id": ["3A1001", "3A1002", "3A1003"],
            "value": [1.0, 2.0, 3.0], "lower": [0.9, 1.9, 2.9],
            "upper": [1.1, 2.1, 3.1], "avg4_value": [7.0, 8.0, 9.0],
            "is_fallback": [False] * 3, "fallback_reason": [""] * 3,
            "horizon_day": [1, 1, 1], "date": ["2026-07-16"] * 3,
        })
        run = dict(self.run, level="unit", rusun=3)
        level, ids = resolve(["3A1001"], run, F, rusun=3)
        self.assertEqual(level, "unit")
        self.assertEqual(ids, ["3A1001"])

    def test_electricity_tower_code_resolves_to_tower(self):
        F = pd.DataFrame({
            "level": ["tower"] * 3,
            "id": ["tower_A", "tower_B", "tower_C"],
            "value": [1.0, 2.0, 3.0], "lower": [0.9, 1.9, 2.9],
            "upper": [1.1, 2.1, 3.1], "avg4_value": [7.0, 8.0, 9.0],
            "is_fallback": [False] * 3, "fallback_reason": [""] * 3,
            "horizon_day": [1, 1, 1], "date": ["2026-07-16"] * 3,
        })
        run = dict(self.run, level="tower", rusun=3)
        level, ids = resolve(["4A"], run, F, rusun=3)
        self.assertEqual(level, "tower")
        self.assertEqual(ids, ["tower_A"])

    def test_empty_list_is_building(self):
        level, ids = resolve([], self.run, self.F, rusun=3)
        self.assertEqual(level, "building")
        self.assertEqual(ids, ["building"])

    def test_building_is_building(self):
        level, ids = resolve(["building"], self.run, self.F, rusun=3)
        self.assertEqual(level, "building")
        self.assertEqual(ids, ["building"])


if __name__ == "__main__":
    unittest.main()
