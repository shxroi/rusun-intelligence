"""Shared helpers for the NusaWatt intelligence scripts (01, 02, 04 and the audit).

Everything that must be identical between training, backtest and production lives here:
cleaning, the calendar matrix, causal features, the cutoff-safe scaler/anchor, window
construction, the recursive forecaster and the eligibility rules.

Roles used in every output so nobody mistakes the baseline for the product:
    ROLE_CANDIDATE = "lstm"  intended production forecast
    ROLE_BASELINE  = "avg4"  baseline the candidate has to beat
"""
import hashlib
import json
import os
import platform
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

TZ = "Asia/Jakarta"
ROLE_CANDIDATE = "lstm"
ROLE_BASELINE = "avg4"

SEQ_LEN = 30            # days of lookback the BiLSTM sees
HORIZON = 7             # recursive forecast length
ZERO_EPS = 0.5          # usage <= this counts as a "zero" day
MAX_ZERO_RATIO = 0.30   # skip windows with more zero days than this
ACTIVE_MIN_KWH = 5.0    # a unit is "active" if its trailing mean usage >= this
ACTIVE_LOOKBACK = 56    # trailing days used by the causal active definition
METER_TOL = 0.02
FEATURE_NAMES = ["usage", "rolling_7d_avg", "rolling_30d_avg",
                 "is_weekend", "month_sin", "month_cos"]
MODEL_VERSION = "bilstm64_lstm32_seq30_recursive7_v1"
BACKTEST_POLICY = "strict_calendar_causal"

# Minimum coverage before a rusun's accuracy may be reported as a quality claim.
MIN_ORIGINS_FOR_CLAIM = 8
MIN_UNIT_WEEKS_FOR_CLAIM = 500


# ------------------------------------------------------------
# VERSIONING / MANIFESTS
# ------------------------------------------------------------
def file_hash(path, n=8):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:n]


def data_version(path, last_date, clean_tag):
    """e.g. energy_2026-07-15_ab12cd34_quarantine_v1"""
    kind = "water" if "water" in os.path.basename(path).lower() else "energy"
    return f"{kind}_{pd.Timestamp(last_date).date()}_{file_hash(path)}_{clean_tag}"


def write_manifest(out_dir, **fields):
    """Write run_manifest.json. Always records data_version, model_version, backtest_policy."""
    for k in ("data_version", "model_version", "backtest_policy"):
        fields.setdefault(k, "n/a")
    fields["created"] = datetime.now().isoformat(timespec="seconds")
    fields["python"] = platform.python_version()
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "run_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(fields, f, indent=2, default=str)
    return fields


def stamp(df, versions):
    """Add data_version / model_version / backtest_policy columns to an output table."""
    df = df.copy()
    for k in ("data_version", "model_version", "backtest_policy"):
        df[k] = versions.get(k, "n/a")
    return df


# ------------------------------------------------------------
# RAW LOAD
# ------------------------------------------------------------
def load_raw(path, meter_col):
    """Raw rows with parsed WIB calendar date and uid = rusun_unit_number.

    Nothing is dropped, deduplicated or clipped here.
    """
    cols = ["id", "created_at", "timestamp", "tower", "floor", "unit_number",
            meter_col, "usage", "rusun"]
    df = pd.read_csv(path, usecols=cols)
    ts = pd.to_datetime(df["timestamp"], utc=True, format="ISO8601")
    df["date"] = ts.dt.tz_convert(TZ).dt.normalize().dt.tz_localize(None)
    df["created_at"] = pd.to_datetime(df["created_at"], utc=True, format="ISO8601")
    df["uid"] = df["rusun"].astype(str) + "_" + df["unit_number"].astype(str)
    df["usage"] = pd.to_numeric(df["usage"], errors="coerce")
    df[meter_col] = pd.to_numeric(df[meter_col], errors="coerce")
    return df


# ------------------------------------------------------------
# CLEANING
# ------------------------------------------------------------
@dataclass
class CleanData:
    U: pd.DataFrame                 # days x units, usage; NaN = missing / quarantined / reset
    meta: pd.DataFrame              # per unit: rusun, tower, unit_number
    quarantined: pd.DataFrame       # bool, same shape as U
    reset: pd.DataFrame             # bool, same shape as U
    report: dict = field(default_factory=dict)
    policy: dict = field(default_factory=dict)
    clean_tag: str = ""

    def scope(self, uids):
        """Restrict to a subset of units (all pieces stay aligned)."""
        uids = list(uids)
        return CleanData(self.U[uids], self.meta.loc[uids], self.quarantined[uids], self.reset[uids],
                         dict(self.report), dict(self.policy), self.clean_tag)


def clean_energy(raw, meter_col="electricity_meter", dedup="quarantine", reset_policy="nan"):
    """Turn raw rows into a complete daily matrix.

    dedup:
      "quarantine" (provisional default): every (unit, day) with more than one row is unresolvable
          (the rows carry different meter readings, i.e. different physical meters under one id),
          so usage is set to missing and the day is flagged. No row is chosen.
      "max_meter": legacy rule used by the first versions of 01/04 (keep highest meter reading).
    reset_policy:
      "nan": a meter decrease (or negative meter) marks that day's usage as missing and flagged.
      "keep": legacy behaviour, usage is left as reported.

    Both rules look only at the row itself and the previous reading of the same unit, so the
    cleaning is valid at every backtest cutoff (it never uses future data).
    """
    assert dedup in ("quarantine", "max_meter") and reset_policy in ("nan", "keep")
    df = raw
    all_uids = sorted(df["uid"].unique())
    full_dates = pd.date_range(df["date"].min(), df["date"].max(), freq="D")
    dm = df.duplicated(["uid", "date"], keep=False)
    n_dup_groups = int(df.loc[dm, ["uid", "date"]].drop_duplicates().shape[0])

    q_pairs = None
    if dedup == "quarantine":
        q_pairs = df.loc[dm, ["uid", "date"]].drop_duplicates()
        keep = df.loc[~dm].copy()
    else:
        keep = df.sort_values(meter_col, ascending=False).drop_duplicates(["uid", "date"]).copy()

    keep["usage"] = keep["usage"].clip(lower=0)
    keep = keep.sort_values(["uid", "date"])
    d_meter = keep.groupby("uid")[meter_col].diff()
    keep["is_reset"] = (d_meter < -METER_TOL) | (keep[meter_col] < 0)
    if reset_policy == "nan":
        keep.loc[keep["is_reset"], "usage"] = np.nan

    U = keep.pivot(index="date", columns="uid", values="usage").reindex(index=full_dates, columns=all_uids)
    RST = (keep.pivot(index="date", columns="uid", values="is_reset")
           .reindex(index=full_dates, columns=all_uids).fillna(False).astype(bool))
    qarr = np.zeros((len(full_dates), len(all_uids)), dtype=bool)
    if q_pairs is not None and len(q_pairs):
        qarr[full_dates.get_indexer(q_pairs["date"]), pd.Index(all_uids).get_indexer(q_pairs["uid"])] = True
    Q = pd.DataFrame(qarr, index=full_dates, columns=all_uids)
    U = U.mask(Q)

    meta = df.groupby("uid").agg(rusun=("rusun", "first"), tower=("tower", "first"),
                                 floor=("floor", "first"),
                                 unit_number=("unit_number", "first")).loc[all_uids]

    span_missing = span_total = 0
    for col in U.columns:
        s = U[col]
        a, b = s.first_valid_index(), s.last_valid_index()
        if a is None:
            continue
        seg = s.loc[a:b]
        span_missing += int(seg.isna().sum())
        span_total += len(seg)
    tag = f"{dedup}_{reset_policy}"
    report = dict(
        rows_raw=int(len(df)), dup_groups=n_dup_groups,
        quarantined_unit_days=int(Q.values.sum()),
        units_with_quarantine=int(Q.any().sum()),
        quarantined_rusuns=sorted(meta.loc[Q.columns[Q.any().values], "rusun"].astype(str).unique().tolist()),
        quarantined_towers=sorted(meta.loc[Q.columns[Q.any().values], "tower"].astype(str).unique().tolist()),
        reset_unit_days=int(RST.values.sum()), units_with_reset=int(RST.any().sum()),
        missing_pct_inside_span=(span_missing / span_total * 100) if span_total else float("nan"),
        n_units=len(all_uids), first_date=str(full_dates[0].date()), last_date=str(full_dates[-1].date()))
    return CleanData(U, meta, Q, RST, report, dict(dedup=dedup, reset_policy=reset_policy), tag)


def load_clean(path, meter_col="electricity_meter", dedup="quarantine", reset_policy="nan"):
    return clean_energy(load_raw(path, meter_col), meter_col, dedup, reset_policy)


# ------------------------------------------------------------
# FEATURES (causal: look backward only)
# ------------------------------------------------------------
def rolling_features(U):
    return U.rolling(7, min_periods=1).mean(), U.rolling(30, min_periods=7).mean()


def calendar_features(dates):
    dates = pd.DatetimeIndex(dates)
    wk = np.asarray(dates.dayofweek >= 5, dtype=float)
    ms = np.sin(2 * np.pi * np.asarray(dates.month) / 12)
    mc = np.cos(2 * np.pi * np.asarray(dates.month) / 12)
    return np.column_stack([wk, ms, mc])


def active_matrix(U, mode="causal"):
    """Boolean days x units: is the unit 'active' on that day?

    causal:       trailing 56-day mean, known up to the previous day, >= ACTIVE_MIN_KWH.
    full_history: legacy; one constant answer per unit from the mean over ALL days (uses the future).
    """
    if mode == "full_history":
        a = (U.mean() >= ACTIVE_MIN_KWH).values
        return pd.DataFrame(np.tile(a, (len(U), 1)), index=U.index, columns=U.columns)
    m = U.shift(1).rolling(ACTIVE_LOOKBACK, min_periods=14).mean()
    return (m >= ACTIVE_MIN_KWH).fillna(False)


def outage_days(U, meta, active):
    """Days where >=90% of a rusun's active, observed units read zero = data outage, not consumption."""
    out = {}
    for r, cols in meta.groupby("rusun").groups.items():
        cols = list(cols)
        sub, act = U[cols], active[cols]
        obs = (sub.notna() & act).sum(axis=1).replace(0, np.nan)
        zero = ((sub <= 0.01) & act).sum(axis=1)
        out[r] = (zero / obs) >= 0.9
    return pd.DataFrame(out).reindex(U.index).fillna(False).astype(bool)


# ------------------------------------------------------------
# ELIGIBILITY (one definition, with the reason a unit is excluded)
# ------------------------------------------------------------
ELIGIBLE = "eligible"
REASONS = ["no_data_before_cutoff", "short_history", "quarantined_in_window", "reset_in_window",
           "missing_days_in_window", "inactive", "too_many_zero_days", "insufficient_feature_history",
           "missing_actual_week", ELIGIBLE]


def eligibility(U, R7, R30, ci, quarantined=None, reset=None, require_actual=True):
    """First failing reason per unit for the forecast origin at row index ci.

    Eligible = complete 30-day window, active over the last 28 days, at most 30% zero days,
    finite rolling features, and (when require_actual) the next 7 days observed.
    require_actual=False is the production variant: the future week does not exist yet.
    The SAME set is used for every method.
    """
    cols = U.columns
    vals = U.values
    T, N = vals.shape
    win = U.iloc[ci - SEQ_LEN:ci]
    complete = win.notna().all().values
    zero_ok = (((win <= ZERO_EPS).sum() / SEQ_LEN) <= MAX_ZERO_RATIO).values
    active = (win.iloc[-28:].mean() >= ACTIVE_MIN_KWH).values
    feats = (np.isfinite(R7.iloc[ci - SEQ_LEN:ci]).all() & np.isfinite(R30.iloc[ci - SEQ_LEN:ci]).all()).values
    actual_ok = U.iloc[ci:ci + HORIZON].notna().all().values if require_actual else np.ones(N, bool)

    seen = ~np.isnan(vals[:ci])
    any_before = seen.any(axis=0)
    first_pos = np.where(any_before, np.argmax(seen, axis=0), T)
    q_in = quarantined.iloc[ci - SEQ_LEN:ci].any().values if quarantined is not None else np.zeros(N, bool)
    r_in = reset.iloc[ci - SEQ_LEN:ci].any().values if reset is not None else np.zeros(N, bool)

    reason = np.full(N, ELIGIBLE, dtype=object)
    # assign from the last check to the first so that the earliest failing reason wins
    reason[~actual_ok] = "missing_actual_week"
    reason[~feats] = "insufficient_feature_history"
    reason[~zero_ok] = "too_many_zero_days"
    reason[~active] = "inactive"
    reason[~complete] = "missing_days_in_window"
    reason[~complete & r_in] = "reset_in_window"
    reason[~complete & q_in] = "quarantined_in_window"
    reason[~complete & (first_pos > ci - SEQ_LEN)] = "short_history"
    reason[~any_before] = "no_data_before_cutoff"
    return pd.Series(reason, index=cols)


# ------------------------------------------------------------
# BASELINES
# ------------------------------------------------------------
def baseline_forecasts(U, ci, cols):
    """Weekly (7-day) baselines using only rows before ci."""
    return {"last_week": U.iloc[ci - 7:ci][cols].sum().values,
            "avg4": U.iloc[ci - 28:ci][cols].sum().values / 4.0}


# ------------------------------------------------------------
# WINDOWS (single definition for training, backtest and production)
# ------------------------------------------------------------
def fit_scaler_anchor(U, R7, R30, cal, ci):
    """Scaler, max_usage and unit anchors from rows BEFORE ci only.

    Rows after ci never influence the result (tested in tests/test_common.py).
    """
    from sklearn.preprocessing import MinMaxScaler
    Utr = U.iloc[:ci]
    mean_tr = Utr.mean()
    units = list(mean_tr.index[mean_tr >= ACTIVE_MIN_KWH])
    max_usage = float(np.nanmax(Utr[units].values))
    r7v, r30v = R7.iloc[:ci], R30.iloc[:ci]
    rows = []
    for u in units:
        F = np.column_stack([Utr[u].values, r7v[u].values, r30v[u].values, cal[:ci]])
        ok = np.isfinite(F).all(axis=1) & (F[:, 0] > 0.5)
        rows.append(F[ok])
    scaler = MinMaxScaler().fit(np.vstack(rows))
    return dict(scaler=scaler, max_usage=max_usage, mean_tr=mean_tr, units=units, ci=ci)


def training_windows(U, R7, R30, cal, ci, fit):
    """All (window, target) pairs with target day < ci. Returns X, y, t (row index of target)."""
    scaler, max_usage, mean_tr, units = fit["scaler"], fit["max_usage"], fit["mean_tr"], fit["units"]
    Utr, r7v, r30v = U.iloc[:ci], R7.iloc[:ci], R30.iloc[:ci]
    Xs, ys, tix = [], [], []
    for u in units:
        F = np.column_stack([Utr[u].values, r7v[u].values, r30v[u].values, cal[:ci]])
        T = len(F)
        if T <= SEQ_LEN:
            continue
        zero_ind = ((F[:, 0] <= ZERO_EPS) & np.isfinite(F[:, 0])).astype(float)
        Fs = scaler.transform(F)  # NaN stays NaN
        win = sliding_window_view(Fs, SEQ_LEN, axis=0)[:T - SEQ_LEN].transpose(0, 2, 1)
        tgt = Fs[SEQ_LEN:, 0]
        zr = sliding_window_view(zero_ind, SEQ_LEN)[:T - SEQ_LEN].mean(axis=1)
        ok = (np.isfinite(win).all(axis=(1, 2)) & np.isfinite(tgt)
              & (zr <= MAX_ZERO_RATIO) & (tgt > 0.01))
        if not ok.any():
            continue
        anchor = mean_tr[u] / max_usage
        w = win[ok]
        Xs.append(np.concatenate([w, np.full((w.shape[0], SEQ_LEN, 1), anchor)], axis=2).astype(np.float32))
        ys.append(tgt[ok].astype(np.float32))
        tix.append(np.arange(SEQ_LEN, T)[ok])
    return np.concatenate(Xs), np.concatenate(ys), np.concatenate(tix)


def inference_window(ext, last_date, scaler, anchors):
    """Model input for ONE step. ext: (L>=SEQ_LEN, n) usage ending at last_date (inclusive).

    Rolling and calendar features are recomputed from `ext` here, exactly as the training
    features were computed. Returns float32 array (n, SEQ_LEN, len(FEATURE_NAMES) + 1).
    """
    n = ext.shape[1]
    ex = pd.DataFrame(ext)
    r7 = ex.rolling(7, min_periods=1).mean().values[-SEQ_LEN:]
    r30 = ex.rolling(30, min_periods=7).mean().values[-SEQ_LEN:]
    us = ext[-SEQ_LEN:]
    wdates = pd.date_range(end=pd.Timestamp(last_date), periods=SEQ_LEN)
    cal = calendar_features(wdates)
    stack = np.stack([us, r7, r30,
                      np.repeat(cal[:, [0]], n, 1),
                      np.repeat(cal[:, [1]], n, 1),
                      np.repeat(cal[:, [2]], n, 1)], axis=-1)       # (30, n, 6)
    Fs = scaler.transform(stack.transpose(1, 0, 2).reshape(-1, 6)).reshape(n, SEQ_LEN, 6)
    return np.concatenate([Fs, np.repeat(np.asarray(anchors)[:, None, None], SEQ_LEN, 1)],
                          axis=2).astype(np.float32)


def forecast_recursive(bundle, U_before, cols, horizon=HORIZON):
    """Recursive forecast for the `horizon` days after U_before's last row.

    U_before: calendar-indexed usage with every row strictly before the forecast origin.
    The origin is U_before.index[-1] + 1 day, so the same call serves a backtest cutoff and the
    production run. Rolling features are recomputed at every step from history + predictions.
    Returns (weekly_sum[n], daily[horizon, n]) in kWh.
    """
    model, scaler, max_usage = bundle["model"], bundle["scaler"], bundle["max_usage"]
    cols = list(cols)
    hist = U_before.iloc[-60:][cols]
    ext = hist.values.copy()
    last_date = U_before.index[-1]
    anchors = U_before[cols].mean().values / max_usage
    preds = np.zeros((horizon, ext.shape[1]))
    for d in range(horizon):
        Xin = inference_window(ext, last_date + pd.Timedelta(days=d), scaler, anchors)
        ps = np.clip(model.predict(Xin, batch_size=1024, verbose=0).ravel(), 0.0, 1.0)
        raw = np.maximum((ps - scaler.min_[0]) / scaler.scale_[0], 0.0)
        preds[d] = raw
        ext = np.vstack([ext, raw[None, :]])
    return preds.sum(axis=0), preds


def build_model(n_features, lr=1e-3):
    import tensorflow as tf
    from tensorflow.keras.layers import LSTM, BatchNormalization, Bidirectional, Dense, Dropout, Input
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.optimizers import Adam
    m = Sequential([
        Input(shape=(SEQ_LEN, n_features)),
        Bidirectional(LSTM(64, return_sequences=True, kernel_regularizer=tf.keras.regularizers.l2(1e-4))),
        Dropout(0.2),
        LSTM(32, kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
        Dropout(0.2),
        BatchNormalization(),
        Dense(16, activation="relu"),
        Dense(1),
    ])
    m.compile(optimizer=Adam(lr), loss="huber")
    return m


def train_lstm(U, R7, R30, cal, ci, epochs=60, patience=10, batch=512, seed=42, verbose=2):
    """Train on rows before ci. Validation = the last 28 days before ci (temporal, not by unit)."""
    import tensorflow as tf
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    tf.random.set_seed(seed)
    np.random.seed(seed)
    fit = fit_scaler_anchor(U, R7, R30, cal, ci)
    X, y, t = training_windows(U, R7, R30, cal, ci, fit)
    is_val = t >= ci - 28
    Xtr, ytr, Xva, yva = X[~is_val], y[~is_val], X[is_val], y[is_val]
    print(f"    train sequences: {len(Xtr):,} | val: {len(Xva):,} | units: {len(fit['units'])}")
    model = build_model(X.shape[2])
    hist = model.fit(Xtr, ytr, validation_data=(Xva, yva), epochs=epochs, batch_size=batch, verbose=verbose,
                     callbacks=[EarlyStopping(patience=patience, restore_best_weights=True),
                                ReduceLROnPlateau(factor=0.5, patience=max(3, patience // 2), min_lr=1e-6)])
    return dict(model=model, scaler=fit["scaler"], max_usage=fit["max_usage"], ci=ci,
                units=fit["units"], history=hist.history, n_train=len(Xtr), n_val=len(Xva))
