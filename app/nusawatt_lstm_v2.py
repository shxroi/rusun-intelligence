#!/usr/bin/env python3
# ============================================================
# NUSAWATT — LSTM v2: hybrid ratio forecaster (reads Supabase directly)
#
# What is different from 02_model_training.py (and why)
#   1. HYBRID TARGET. The network predicts a multiplicative correction around the
#      unit's own recent level:  forecast_day = base * ratio_day,  base = mean of the last
#      28 days (so base * 7 is exactly avg4).  The output layer starts at ratio = 1, so an
#      untrained / uninformative network reproduces avg4 instead of doing worse.
#   2. DIRECT multi-horizon output (H values at once): no recursive error compounding.
#   3. NO month / day-of-year features (under one year of data they only extrapolate).
#      Calendar = day-of-week (sin/cos), weekend, optional Indonesian public holidays.
#   4. NO positive-only target filter: zero days stay in the data.
#   5. LEVEL-FREE INPUTS: usage is divided by the unit's own base, so low-use units can
#      be included (min-kwh default 1.0 instead of 5) and no global max-scaler exists.
#   6. LOSS = MAE on the ratio, weighted by base  ->  proportional to the kWh error that
#      the backtest reports (no Huber-delta-1 == MSE problem).
#   7. Early stopping picks the epoch count, then the model is REFIT on all data
#      (including the newest weeks) for that many epochs.
#   8. SEED ENSEMBLE: --seeds N models, forecasts averaged; per-seed error also reported.
#  11. WATER (tower/building level): --utility WATER --aggregate tower|building. Per-unit daily
#      water in Rusun 3 is almost binary (0/1 counts), so the series are summed per tower/building
#      first (a day is missing if fewer than 98% of the started units reported). Backward meter
#      jitter of up to 1 count is kept as zero usage instead of being dropped.
#  12. RMSE is reported next to MAE/WAPE/bias (horizon-sum per unit/series).
#  10. OPTIONS: --loss mae|huber|mse (mae learns the MEDIAN and can under-forecast totals),
#      --calibrate (scale forecasts by a factor measured out-of-sample on the validation
#      windows), --model ridge (linear comparator on the same ratio inputs, to test whether
#      the LSTM itself or the hybrid-ratio formulation is responsible for the gain).
#   9. Evaluation: rolling origins, retrained every --retrain-every origins with strictly
#      earlier targets only; paired comparison against avg4 / last_week with a bootstrap
#      over origins.
#
# Data sources
#   --source supabase   reads "meteran-cerdas".meter_readings via SUPABASE_DB_URL
#                       (read-only Postgres connection string; never hard-code it)
#   --source csv        reads the legacy CSV export (--csv path)
#
# Cleaning (rules from the audit; raw data is never modified)
#   duplicate unit-days -> missing | negative usage -> missing
#   meter decrease (total < previous total - 0.02) -> missing (reset/jitter)
#   zeros are KEPT.  Missing days are NOT interpolated; windows touching them are skipped.
#
# Usage
#   python nusawatt_lstm_v2.py --rusun 3 --start-date 2025-09-01
#   python nusawatt_lstm_v2.py --rusun 3 --start-date 2025-09-01 --forecast
#   python nusawatt_lstm_v2.py --source csv --csv energy_meter_rows__3_.csv --rusun 3 \
#          --epochs 3 --seeds 1 --backtest-origins 4        # smoke test
# ============================================================
import argparse
import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

TRAIN_LOG = []          # per-seed training curves, written to training_history.json
_CTX = {"cutoff": ""}
BASE_DAYS = 28          # base = mean of the last 28 days (avg4 = 4 weeks)
RATIO_CLIP = 5.0        # ratio clip for training stability only (evaluation is unclipped)


# ----------------------------------------------------------------------------
# Arguments
# ----------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="NusaWatt hybrid ratio-LSTM (v2)")
    p.add_argument("--source", choices=["supabase", "csv"], default="supabase")
    p.add_argument("--csv", default="")
    p.add_argument("--rusun", default="3", help="rusun number, e.g. 3")
    p.add_argument("--utility", default="ELECTRICITY", choices=["ELECTRICITY", "WATER"])
    p.add_argument("--aggregate", choices=["none", "tower", "building"], default="none",
                   help="sum units into tower/building series before forecasting (use for WATER)")
    p.add_argument("--reset-tolerance", type=float, default=None,
                   help="meter decreases larger than this become missing (default 0.02 kWh; 1.0 for WATER)")
    p.add_argument("--start-date", default="2025-09-01", help="first day used (drops commissioning months)")
    p.add_argument("--horizon", type=int, default=7)
    p.add_argument("--lookback", type=int, default=30)
    p.add_argument("--min-kwh", type=float, default=1.0, help="min trailing mean kWh/day to be eligible")
    p.add_argument("--backtest-origins", type=int, default=12)
    p.add_argument("--retrain-every", type=int, default=4, help="retrain after this many weekly origins")
    p.add_argument("--val-days", type=int, default=28)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--patience", type=int, default=6)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--no-calendar", action="store_true", help="ablation: drop calendar features")
    p.add_argument("--holidays", choices=["none", "package"], default="none",
                   help="package = python 'holidays' lib (Indonesia); verify cuti bersama yourself")
    p.add_argument("--holiday-csv", default="", help="extra holiday dates, one per line (header: date)")
    p.add_argument("--no-refit", action="store_true", help="skip the refit-on-all-data step")
    p.add_argument("--no-weight", action="store_true", help="ablation: unweighted ratio loss")
    p.add_argument("--loss", choices=["mae", "huber", "mse"], default="mae",
                   help="training loss on the ratio (mae = median, mse = mean, huber = pseudo-Huber in between)")
    p.add_argument("--huber-delta", type=float, default=0.1, help="huber delta in ratio units")
    p.add_argument("--calibrate", action="store_true",
                   help="multiply forecasts by a factor fitted out-of-sample on the validation windows")
    p.add_argument("--model", choices=["lstm", "ridge"], default="lstm",
                   help="ridge = linear comparator on the same inputs (no LSTM)")
    p.add_argument("--ridge-alpha", type=float, default=0.01, help="ridge penalty per training sample")
    p.add_argument("--no-daily", action="store_true", help="do not write backtest_daily.csv (as-of daily forecasts per origin)")
    p.add_argument("--forecast", action="store_true", help="also forecast the next H days after the backtest")
    p.add_argument("--out", default="experiments/lstm_v2")
    return p.parse_args()


# ----------------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------------
SQL = r"""
select u.code as unit, u.id::text as unit_id, r.reading_date as date,
       r.daily_usage as usage, r.total_usage as total
from "meteran-cerdas".meter_readings r
join "meteran-cerdas".units     u on u.id = r.unit_id
join "meteran-cerdas".towers    t on t.id = u.tower_id
join "meteran-cerdas".complexes c on c.id = t.complex_id
where r.utility_type::text = %(utility)s and r.deleted_at is null
  and regexp_replace(c.code, '\D', '', 'g') = %(rusun)s
  and r.reading_date >= %(start)s
"""


def load_supabase(a):
    import psycopg2
    dsn = os.environ.get("SUPABASE_DB_URL")
    if not dsn:
        sys.exit("Set SUPABASE_DB_URL to a read-only Postgres connection string first.")
    with psycopg2.connect(dsn) as conn, conn.cursor() as cur:
        cur.execute(SQL, {"utility": a.utility, "rusun": a.rusun, "start": a.start_date})
        df = pd.DataFrame(cur.fetchall(), columns=[d[0] for d in cur.description])
    for c in ("usage", "total"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["date"] = pd.to_datetime(df["date"])
    return df


def load_csv(a):
    raw = pd.read_csv(a.csv, dtype={"rusun": str, "unit_number": str})
    raw = raw[raw["rusun"].astype(str) == str(a.rusun)]
    ts = pd.to_datetime(raw["timestamp"], utc=True).dt.tz_convert("Asia/Makassar")
    meter = "electricity_meter" if "electricity_meter" in raw else raw.columns[raw.columns.str.endswith("_meter")][0]
    df = pd.DataFrame({"unit": raw["rusun"].astype(str) + raw["unit_number"].astype(str),
                       "unit_id": "", "date": ts.dt.tz_localize(None).dt.normalize(),
                       "usage": pd.to_numeric(raw["usage"], errors="coerce"),
                       "total": pd.to_numeric(raw[meter], errors="coerce")})
    return df[df["date"] >= pd.Timestamp(a.start_date)]


def clean(df, tol=0.02):
    """Return (U days x units matrix with NaN for missing, report, unit_id map)."""
    rep = {}
    df = df.sort_values(["unit", "date"]).copy()
    dup = df.duplicated(["unit", "date"], keep=False)
    rep["duplicate_rows_quarantined"] = int(dup.sum())
    df.loc[dup, "usage"] = np.nan
    df = df.drop_duplicates(["unit", "date"])
    neg = df["usage"] < 0
    rep["negative_usage_to_missing"] = int(neg.sum())
    df.loc[neg, "usage"] = np.nan
    prev = df.groupby("unit")["total"].shift()
    reset = df["total"] < prev - tol
    rep["meter_decrease_to_missing"] = int(reset.sum())
    df.loc[reset, "usage"] = np.nan
    U = df.pivot(index="date", columns="unit", values="usage")
    U = U.reindex(pd.date_range(U.index.min(), U.index.max(), freq="D")).astype("float64")
    rep["days"], rep["units"] = int(U.shape[0]), int(U.shape[1])
    rep["missing_cell_share"] = float(U.isna().mean().mean())
    ids = df.drop_duplicates("unit").set_index("unit")["unit_id"].to_dict()
    return U, rep, ids


def aggregate_series(U_df, mode, min_share=0.98):
    """Sum units into tower or building series. A day is NaN if fewer than min_share of the units
    that have already started reporting are present (no silent under-counting)."""
    import re
    codes = list(U_df.columns)
    V = U_df.values
    has = np.isfinite(V).any(axis=0)
    first = np.isfinite(V).argmax(axis=0)
    started = (np.arange(len(U_df))[:, None] >= first[None, :]) & has[None, :]
    groups = {}
    for c in codes:
        m = re.match(r"^\d+([A-Za-z]+)", c)
        groups.setdefault("building" if mode == "building" else f"tower_{m.group(1) if m else 'X'}", []).append(c)
    out = {}
    for name, cols in groups.items():
        idx = [codes.index(c) for c in cols]
        sub, st = V[:, idx], started[:, idx]
        expected, reported = st.sum(1), np.isfinite(sub).sum(1)
        ok = (expected > 0) & (reported >= min_share * expected)
        out[name] = np.where(ok, np.nansum(sub, axis=1), np.nan)
    return pd.DataFrame(out, index=U_df.index)


# ----------------------------------------------------------------------------
# Features
# ----------------------------------------------------------------------------
def calendar_features(index, a, extra_days=0):
    idx = pd.date_range(index[0], periods=len(index) + extra_days, freq="D")
    dow = idx.dayofweek.values
    cols = [np.sin(2 * np.pi * dow / 7), np.cos(2 * np.pi * dow / 7), (dow >= 5).astype(float)]
    hol = set()
    if a.holidays == "package":
        import holidays
        try:
            hol |= set(holidays.country_holidays("ID", years=range(idx[0].year, idx[-1].year + 1)).keys())
        except AttributeError:
            hol |= set(holidays.Indonesia(years=range(idx[0].year, idx[-1].year + 1)).keys())
    if a.holiday_csv:
        hol |= set(pd.to_datetime(pd.read_csv(a.holiday_csv)["date"]).dt.date)
    if a.holidays != "none" or a.holiday_csv:
        cols.append(np.array([d.date() in hol for d in idx], dtype=float))
    cal = np.column_stack(cols).astype("float32")
    if a.no_calendar:
        cal = np.zeros((len(idx), 1), dtype="float32")
    return cal, idx


def samples_at(U, cal, origins, L, H, need_target, min_kwh):
    """Build model inputs for the given origin positions (position of the first forecast day)."""
    Xs, Xf, Xst, Y, B, UI, OI, ACT = [], [], [], [], [], [], [], []
    for o in origins:
        hist = U[o - L:o].T                                   # (units, L)
        base = hist[:, -BASE_DAYS:].mean(axis=1)              # NaN if the 28-day window has a gap
        ok = np.isfinite(hist).all(axis=1) & np.isfinite(base) & (base >= min_kwh)
        if need_target:
            tgt = U[o:o + H].T
            ok &= np.isfinite(tgt).all(axis=1)
        if not ok.any():
            continue
        j = np.where(ok)[0]
        b = base[j]
        ratio = np.clip(hist[j] / b[:, None], 0, RATIO_CLIP)
        seq = np.concatenate([ratio[:, :, None], np.repeat(cal[None, o - L:o], len(j), axis=0)], axis=2)
        fut = np.repeat(cal[None, o:o + H].reshape(1, -1), len(j), axis=0)
        Xs.append(seq)
        Xf.append(fut)
        Xst.append((np.log1p(b) / 4.0)[:, None])
        B.append(b)
        UI.append(j)
        OI.append(np.full(len(j), o))
        if need_target:
            ACT.append(tgt[j])
            Y.append(np.clip(tgt[j] / b[:, None], 0, RATIO_CLIP))
    if not Xs:
        return None
    cat = lambda z: np.concatenate(z, axis=0).astype("float32")
    out = dict(seq=cat(Xs), fut=cat(Xf), stat=cat(Xst), base=np.concatenate(B), unit=np.concatenate(UI),
               origin=np.concatenate(OI))
    if need_target:
        out["y"], out["actual"] = cat(Y), np.concatenate(ACT)
    return out


# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------
def build_model(L, H, c_seq, c_fut, lr, loss="mae", delta=0.1):
    import tensorflow as tf
    from tensorflow.keras import Input, Model, layers
    seq, fut, stat = Input((L, c_seq)), Input((H * c_fut,)), Input((1,))
    x = layers.LSTM(64)(seq)
    x = layers.Dropout(0.2)(x)
    z = layers.Concatenate()([x, fut, stat])
    z = layers.Dense(32, activation="relu")(z)
    z = layers.Dropout(0.1)(z)
    corr = layers.Dense(H, kernel_initializer="zeros", bias_initializer="zeros")(z)   # starts at ratio = 1
    out = layers.Lambda(lambda t: 1.0 + t)(corr)
    m = Model([seq, fut, stat], out)
    ops = getattr(tf.keras, "ops", tf.math)      # Keras 3 ops handle symbolic tensors; Keras 2 uses tf
    def huber(y_true, y_pred):        # pseudo-Huber: smooth, MSE-like near 0 and MAE-like beyond delta
        z = (y_pred - y_true) / delta
        pseudo = delta * delta * (ops.sqrt(1.0 + ops.square(z)) - 1.0)
        return tf.reduce_mean(pseudo, axis=-1) if ops is tf.math else ops.mean(pseudo, axis=-1)
    lf = {"mae": "mae", "mse": "mse", "huber": huber}[loss]
    m.compile(optimizer=tf.keras.optimizers.Adam(lr), loss=lf)
    return m


def weights(S, a):
    if a.no_weight:
        return None
    return (S["base"] / S["base"].mean()).astype("float32")


def fit_models(S_tr, S_val, S_all, a, L, H):
    """Train a.seeds models: early-stop on validation, then refit on all samples."""
    import tensorflow as tf
    c_seq, c_fut = S_tr["seq"].shape[2], S_tr["fut"].shape[1] // H
    models, best_epochs, calibs = [], [], []
    for seed in range(a.seeds):
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(42 + seed)
        m = build_model(L, H, c_seq, c_fut, a.lr, a.loss, a.huber_delta)
        cb = [tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=a.patience, restore_best_weights=True),
              tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5)]
        h = m.fit([S_tr["seq"], S_tr["fut"], S_tr["stat"]], S_tr["y"], sample_weight=weights(S_tr, a),
                  validation_data=([S_val["seq"], S_val["fut"], S_val["stat"]], S_val["y"], weights(S_val, a)),
                  epochs=a.epochs, batch_size=a.batch, callbacks=cb, verbose=0)
        best = int(np.argmin(h.history["val_loss"])) + 1
        best_epochs.append(best)
        TRAIN_LOG.append(dict(cutoff=_CTX["cutoff"], seed=seed, best_epoch=best,
                              loss=[float(v) for v in h.history["loss"]],
                              val_loss=[float(v) for v in h.history["val_loss"]]))
        calibs.append(calibration_factor(m, S_val, a) if a.calibrate else 1.0)
        if not a.no_refit:
            tf.keras.utils.set_random_seed(142 + seed)
            m = build_model(L, H, c_seq, c_fut, a.lr, a.loss, a.huber_delta)
            m.fit([S_all["seq"], S_all["fut"], S_all["stat"]], S_all["y"], sample_weight=weights(S_all, a),
                  epochs=max(best, 3), batch_size=a.batch, verbose=0)
        models.append(m)
    return models, best_epochs, calibs


def calibration_factor(model, S_val, a):
    """Out-of-sample level correction: total actual / total predicted on the validation windows,
    measured with the model that did NOT see them (before the refit)."""
    r = np.clip(model.predict([S_val["seq"], S_val["fut"], S_val["stat"]], batch_size=2048, verbose=0), 0, None)
    pred = float((S_val["base"][:, None] * r).sum())
    return float(np.clip(S_val["actual"].sum() / pred, 0.8, 1.25)) if pred > 0 else 1.0


class RidgeRatio:
    """Weighted ridge regression on the same inputs; predicts ratio = 1 + linear correction."""
    def __init__(self, alpha):
        self.alpha = alpha

    @staticmethod
    def _flat(seq, fut, stat):
        return np.concatenate([seq.reshape(len(seq), -1), fut, stat], axis=1).astype("float64")

    def fit(self, S, w):
        X, d = self._flat(S["seq"], S["fut"], S["stat"]), S["y"].astype("float64") - 1.0
        w = np.ones(len(X)) if w is None else w.astype("float64")
        w = w / w.mean()
        sw = w.sum()
        self.mu = (w[:, None] * X).sum(0) / sw
        Xc = X - self.mu
        self.sd = np.sqrt((w[:, None] * Xc ** 2).sum(0) / sw)
        self.sd[self.sd < 1e-6] = 1.0
        Xs = Xc / self.sd
        self.dmu = (w[:, None] * d).sum(0) / sw
        A = Xs.T @ (w[:, None] * Xs) + self.alpha * len(X) * np.eye(X.shape[1])
        self.beta = np.linalg.solve(A, Xs.T @ (w[:, None] * (d - self.dmu)))
        return self

    def predict(self, X, batch_size=None, verbose=0):
        Xs = (self._flat(*X) - self.mu) / self.sd
        return 1.0 + self.dmu + Xs @ self.beta


def fit_ridge_models(S_tr, S_val, S_all, a, L, H):
    calib = 1.0
    if a.calibrate:
        calib = calibration_factor(RidgeRatio(a.ridge_alpha).fit(S_tr, weights(S_tr, a)), S_val, a)
    m = RidgeRatio(a.ridge_alpha).fit(S_all, weights(S_all, a))
    return [m], [0], [calib]


def predict_ratio(models, S, calibs=None):
    X = [S["seq"], S["fut"], S["stat"]]
    calibs = calibs or [1.0] * len(models)
    per_seed = [np.clip(m.predict(X, batch_size=2048, verbose=0), 0, None) * c for m, c in zip(models, calibs)]
    return np.mean(per_seed, axis=0), per_seed


# ----------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------
def metrics(actual, pred, origin):
    err = pred - actual
    d = pd.DataFrame({"o": origin, "act": actual, "pred": pred, "err": err, "abs": np.abs(err)})
    unit = dict(mae=d["abs"].mean(), rmse=float(np.sqrt((d["err"] ** 2).mean())), wape=100 * d["abs"].sum() / d["act"].sum(), bias=d["err"].mean())
    g = d.groupby("o")
    bld = (100 * (g["pred"].sum() - g["act"].sum()).abs() / g["act"].sum()).mean()
    unit["building_err_pct"] = bld
    return unit


def per_origin_mae(actual, pred, origin):
    return pd.Series(np.abs(pred - actual)).groupby(origin).mean()


def bootstrap_diff(diffs, n=5000, seed=0):
    rng = np.random.default_rng(seed)
    d = np.asarray(diffs)
    means = np.array([rng.choice(d, len(d), replace=True).mean() for _ in range(n)])
    return float(d.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)), float((means < 0).mean())


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    a = parse_args()
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    L, H = a.lookback, a.horizon
    MN = a.model                                    # label used in outputs: "lstm" or "ridge"
    fit_fn = fit_models if a.model == "lstm" else fit_ridge_models
    if a.model == "ridge":
        a.seeds = 1                                 # deterministic, no ensemble

    print("-- Loading + cleaning --")
    raw = load_supabase(a) if a.source == "supabase" else load_csv(a)
    tol = a.reset_tolerance if a.reset_tolerance is not None else (1.0 if a.utility == "WATER" else 0.02)
    U_df, rep, unit_ids = clean(raw, tol)
    rep["reset_tolerance"] = tol
    if a.utility == "WATER" and a.aggregate == "none":
        print("  WARNING: per-unit daily water is mostly 0/1 counts; consider --aggregate tower")
    if a.aggregate != "none":
        U_df, unit_ids = aggregate_series(U_df, a.aggregate), {}
        rep["aggregate"] = a.aggregate
        rep["series"] = list(U_df.columns)
        rep["missing_series_day_share"] = float(U_df.isna().mean().mean())
    print(f"  {rep}")
    U = U_df.values
    T, n_units = U.shape
    cal, cal_idx = calendar_features(U_df.index, a, extra_days=H)
    print(f"  {n_units} units x {T} days ({U_df.index[0].date()} -> {U_df.index[-1].date()}) | "
          f"calendar features: {cal.shape[1]} | horizon {H} | lookback {L}")

    # rolling weekly test origins (position of first forecast day), chronological
    last_origin = T - H
    origins = [last_origin - 7 * k for k in range(a.backtest_origins)][::-1]
    if origins[0] - L - 2 * H - a.val_days < 60:
        sys.exit("Not enough history before the first test origin; reduce --backtest-origins or --start-date.")

    rows, seed_rows, hist_info, daily_rows = [], [], [], []
    for b0 in range(0, len(origins), a.retrain_every):
        block = origins[b0:b0 + a.retrain_every]
        o0 = block[0]
        val_hi, val_lo = o0 - H, o0 - H - a.val_days + 1
        train_hi = val_lo - H
        S_tr = samples_at(U, cal, range(L, train_hi + 1), L, H, True, a.min_kwh)
        S_val = samples_at(U, cal, range(val_lo, val_hi + 1), L, H, True, a.min_kwh)
        S_all = samples_at(U, cal, range(L, val_hi + 1), L, H, True, a.min_kwh)
        print(f"-- Train block cutoff {cal_idx[o0].date()} | train {len(S_tr['base']):,} val {len(S_val['base']):,} "
              f"all {len(S_all['base']):,} windows --")
        _CTX["cutoff"] = str(cal_idx[o0].date())
        models, best, calibs = fit_fn(S_tr, S_val, S_all, a, L, H)
        hist_info.append(dict(cutoff=str(cal_idx[o0].date()), best_epochs=best, calibration=calibs))
        print(f"   best epochs per seed: {best} | calibration factors: {[round(c, 3) for c in calibs]}")
        for o in block:
            S = samples_at(U, cal, [o], L, H, True, a.min_kwh)
            if S is None:
                continue
            ratio, per_seed = predict_ratio(models, S, calibs)
            b = S["base"][:, None]
            pred = np.maximum(b * ratio, 0)
            act = S["actual"]
            rows.append(pd.DataFrame({
                "origin": cal_idx[o].date(), "unit": U_df.columns[S["unit"]],
                "actual": act.sum(1), MN: pred.sum(1),
                "avg4": (S["base"] * H), "last_week": U[o - 7:o].T[S["unit"]].sum(1) * (H / 7.0),
                **{f"{MN}_seed{i}": np.maximum(b * r, 0).sum(1) for i, r in enumerate(per_seed)}}))
            hd = np.tile(np.arange(1, H + 1), len(act))
            daily_rows.append(pd.DataFrame({"origin": str(cal_idx[o].date()), "unit": np.repeat(U_df.columns[S["unit"]].values, H),
                                            "horizon_day": hd, "date": (pd.Timestamp(cal_idx[o]) + pd.to_timedelta(hd - 1, unit="D")).strftime("%Y-%m-%d"),
                                            "forecast": pred.ravel().round(3), "actual": act.ravel().round(3),
                                            "base": np.repeat(S["base"], H).round(3)}))
            seed_rows.append(pd.DataFrame({"origin": cal_idx[o].date(), "day": np.tile(np.arange(1, H + 1), len(act)),
                                           "abs_model": np.abs(pred - act).ravel(),
                                           "abs_avg4": np.abs(b - act).ravel()}))

    R = pd.concat(rows, ignore_index=True)
    R.to_csv(f"{a.out}/backtest_unit_origin.csv", index=False)
    if daily_rows and not a.no_daily:
        pd.concat(daily_rows, ignore_index=True).to_csv(f"{a.out}/backtest_daily.csv", index=False)
    org = R["origin"].astype(str).values
    print("\n== BACKTEST (horizon-sum per unit/series, usage units: kWh or water counts) ==")
    summ = {m: metrics(R["actual"].values, R[m].values, org) for m in ["avg4", "last_week", MN]}
    for i in range(a.seeds):
        summ[f"{MN}_seed{i}"] = metrics(R["actual"].values, R[f"{MN}_seed{i}"].values, org)
    S_df = pd.DataFrame(summ).T.round(3)
    print(S_df.to_string())
    S_df.to_csv(f"{a.out}/summary.csv")

    mae_o = pd.DataFrame({m: per_origin_mae(R["actual"].values, R[m].values, org) for m in ["avg4", "last_week", MN]})
    mae_o.to_csv(f"{a.out}/per_origin_mae.csv")
    diff = (mae_o[MN] - mae_o["avg4"]).values
    mean_d, lo, hi, p_better = bootstrap_diff(diff)
    print(f"\nPaired over {len(diff)} origins: {MN} wins {(diff < 0).sum()}/{len(diff)} | "
          f"mean MAE diff ({MN} - avg4) {mean_d:+.3f} usage units [95% bootstrap CI {lo:+.3f}, {hi:+.3f}] | "
          f"share of resamples where {MN} is better: {p_better:.0%}")
    D = pd.concat(seed_rows).groupby("day")[["abs_model", "abs_avg4"]].mean().round(3)
    D.to_csv(f"{a.out}/per_horizon_day_mae.csv")
    print(f"\nDaily MAE by horizon day ({MN} vs flat avg4 level; the flat baseline ignores weekdays, so use the horizon-sum rows as the fair comparison):")
    print(D.to_string())
    print("\nVerdict rule: promote only if the CI excludes 0 on the negative side AND monthly/bill error also improves.")

    if a.forecast:
        print("\n-- Final model + forecast of the next horizon --")
        o = T                                                     # first day after the data
        val_hi, val_lo = o - H, o - H - a.val_days + 1
        S_tr = samples_at(U, cal, range(L, val_lo - H + 1), L, H, True, a.min_kwh)
        S_val = samples_at(U, cal, range(val_lo, val_hi + 1), L, H, True, a.min_kwh)
        S_all = samples_at(U, cal, range(L, val_hi + 1), L, H, True, a.min_kwh)
        _CTX["cutoff"] = "final"
        models, best, calibs = fit_fn(S_tr, S_val, S_all, a, L, H)
        S = samples_at(U, cal, [o], L, H, False, a.min_kwh)
        ratio, _ = predict_ratio(models, S, calibs)
        pred = np.maximum(S["base"][:, None] * ratio, 0)
        dates = pd.date_range(U_df.index[-1] + pd.Timedelta(days=1), periods=H)
        codes = U_df.columns[S["unit"]]
        fc = pd.DataFrame(pred.T, index=dates, columns=codes).rename_axis("target_date").reset_index() \
            .melt(id_vars="target_date", var_name="unit", value_name="predicted_usage")
        fc["unit_id"] = fc["unit"].map(unit_ids)
        fc["method"], fc["is_fallback"], fc["fallback_reason"] = f"{MN}_v2_hybrid", False, ""
        # eligibility for everyone else -> avg4 fallback where a base exists, otherwise skipped
        elig = []
        for j, code in enumerate(U_df.columns):
            if j in set(S["unit"]):
                continue
            hist = U[o - L:o, j]
            first = np.where(np.isfinite(U[:, j]))[0]
            if len(first) == 0 or first[0] > o - L:
                reason = "insufficient_history"
            elif not np.isfinite(hist).all():
                reason = "gap_in_window"
            else:
                reason = "low_use"
            elig.append(dict(unit=code, unit_id=unit_ids.get(code, ""), reason=reason))
        fc.to_csv(f"{a.out}/forecast_next.csv", index=False)
        pd.DataFrame(elig).to_csv(f"{a.out}/forecast_ineligible_units.csv", index=False)
        print(f"   forecasted {len(codes)} units; {len(elig)} ineligible (reasons in forecast_ineligible_units.csv)")

    if TRAIN_LOG:
        with open(f"{a.out}/training_history.json", "w") as f:
            json.dump(TRAIN_LOG, f)
    with open(f"{a.out}/run_config.json", "w") as f:
        json.dump(dict(args=vars(a), cleaning=rep, training=hist_info, minutes=round((time.time() - t0) / 60, 2)), f,
                  indent=2, default=str)
    print(f"\nDone in {(time.time() - t0) / 60:.1f} min. Outputs in '{a.out}/'")


if __name__ == "__main__":
    main()