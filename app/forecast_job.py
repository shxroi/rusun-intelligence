#!/usr/bin/env python3
# ============================================================
# NUSAWATT — FORECAST JOB (train / forecast / evaluate)
#
# Turns the experiment code (nusawatt_lstm_v2.py) into a job that a scheduler can run.
# Data is read from Supabase (read-only role, SUPABASE_DB_URL); --source csv is for offline tests.
#
#   train     fit the hybrid ratio LSTM (3 seeds) on all data up to the cutoff, save weights + metadata
#             + empirical interval quantiles under  <store>/models/<utility>_r<rusun>/<version>/
#             (WATER: no model; stores avg4 interval quantiles at tower level)
#   forecast  read the last --history-days days, load the current model, write a new RUN to
#             <store>/runs/<run_id>/ (forecast_daily.csv, history_daily.csv, run.json) and then
#             atomically move <store>/latest_<utility>_h<H>.json to point at it.  No TensorFlow in the API.
#   evaluate  compare stored runs with the actuals once they exist (model vs avg4) -> <store>/metrics.csv
#
# Examples
#   python forecast_job.py train    --rusun 3 --utility ELECTRICITY --start-date 2025-09-01 --horizon 7
#   python forecast_job.py forecast --rusun 3 --utility ELECTRICITY --horizon 7
#   python forecast_job.py train    --rusun 3 --utility WATER --horizon 7
#   python forecast_job.py forecast --rusun 3 --utility WATER --horizon 7
#   python forecast_job.py evaluate --rusun 3 --utility ELECTRICITY --horizon 7
#
# Scheduling (the job is idempotent: the same data cutoff + model version rewrites the same run)
#   forecast: once a day after the previous day's readings are complete, e.g. 03:00 WITA
#   train:    weekly or monthly (or by hand); evaluate: daily or weekly
#   Windows Task Scheduler, cron, or a container cron all work; exit code != 0 means the previous run stays current.
#
# Needs nusawatt_lstm_v2.py in the same folder (override with env NUSAWATT_LIB).
# ============================================================
import argparse
import importlib
import json
import os
import re
import shutil
import sys
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

LIB = importlib.import_module(os.environ.get("NUSAWATT_LIB", "nusawatt_lstm_v2"))
BASE_DAYS = 28


def parse_args():
    p = argparse.ArgumentParser(description="NusaWatt forecast job")
    p.add_argument("command", choices=["train", "forecast", "evaluate"])
    p.add_argument("--source", choices=["supabase", "csv"], default="supabase")
    p.add_argument("--csv", default="")
    p.add_argument("--rusun", default="3")
    p.add_argument("--utility", default="ELECTRICITY", choices=["ELECTRICITY", "WATER"])
    p.add_argument("--store", default="forecast_store")
    p.add_argument("--horizon", type=int, default=7)
    p.add_argument("--lookback", type=int, default=30)
    p.add_argument("--min-kwh", type=float, default=1.0)
    p.add_argument("--start-date", default="2025-09-01", help="train: first day used (skips commissioning)")
    p.add_argument("--val-days", type=int, default=28)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--patience", type=int, default=6)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--loss", choices=["mae", "huber", "mse"], default="mae")
    p.add_argument("--huber-delta", type=float, default=0.1)
    p.add_argument("--no-refit", action="store_true")
    p.add_argument("--no-weight", action="store_true")
    p.add_argument("--coverage", type=float, default=0.8, help="nominal coverage of the empirical interval")
    p.add_argument("--model-version", default="", help="train: name of the new version; forecast: version to use (default: current)")
    p.add_argument("--as-of", default="", help="forecast: last day of data to use (default: newest day in the database)")
    p.add_argument("--history-days", type=int, default=365, help="forecast: days read and exported as history")
    return p.parse_args()


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def tower_of(code):
    m = re.match(r"^\d+([A-Za-z]+)", str(code))
    return f"tower_{m.group(1)}" if m else "tower_X"


def sk(a):
    return f"{a.utility.lower()}_r{a.rusun}"


def lib_ns(a, start_date):
    return argparse.Namespace(source=a.source, csv=a.csv, rusun=a.rusun, utility=a.utility, start_date=start_date)


def load_raw(a, start_date):
    ns = lib_ns(a, start_date)
    return LIB.load_supabase(ns) if a.source == "supabase" else LIB.load_csv(ns)


def data_cutoff(a):
    if a.source == "csv":
        return load_raw(a, "2000-01-01")["date"].max()
    import psycopg2
    dsn = os.environ.get("SUPABASE_DB_URL")
    if not dsn:
        sys.exit("Set SUPABASE_DB_URL to a read-only Postgres connection string first.")
    sql = r"""select max(r.reading_date) from "meteran-cerdas".meter_readings r
              join "meteran-cerdas".units u on u.id = r.unit_id
              join "meteran-cerdas".towers t on t.id = u.tower_id
              join "meteran-cerdas".complexes c on c.id = t.complex_id
              where r.utility_type::text = %(utility)s and r.deleted_at is null
                and regexp_replace(c.code, '\D', '', 'g') = %(rusun)s"""
    with psycopg2.connect(dsn) as conn, conn.cursor() as cur:
        cur.execute(sql, {"utility": a.utility, "rusun": a.rusun})
        return pd.Timestamp(cur.fetchone()[0])


def load_matrix(a, start_date):
    raw = load_raw(a, start_date)
    tol = 1.0 if a.utility == "WATER" else 0.02
    U, rep, ids = LIB.clean(raw, tol)
    if a.utility == "WATER":
        U = LIB.aggregate_series(U, "tower")
        ids = {}
    return U, rep, ids


def cal_ns():
    return argparse.Namespace(no_calendar=False, holidays="none", holiday_csv="")


def utcnow():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_json_atomic(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, default=str)
    os.replace(tmp, path)


def fit_ns(a, seeds, no_refit):
    return argparse.Namespace(seeds=seeds, epochs=a.epochs, patience=a.patience, batch=a.batch, lr=a.lr, loss=a.loss,
                              huber_delta=a.huber_delta, no_refit=no_refit, no_weight=a.no_weight, calibrate=False)


def quantiles(rel, coverage):
    lo, hi = (1 - coverage) / 2, 1 - (1 - coverage) / 2
    return np.quantile(rel, lo, axis=0).tolist(), np.quantile(rel, hi, axis=0).tolist()


def model_paths(a):
    root = f"{a.store}/models/{sk(a)}"
    return root, f"{root}/current_h{a.horizon}.json"


# ----------------------------------------------------------------------------
# train
# ----------------------------------------------------------------------------
def cmd_train(a):
    t0 = time.time()
    U, rep, ids = load_matrix(a, a.start_date)
    as_of = U.index[-1]
    L, H = a.lookback, a.horizon
    print(f"data: {U.shape[1]} series x {U.shape[0]} days ({U.index[0].date()} -> {as_of.date()}) | cleaning {rep}")
    root, pointer = model_paths(a)
    is_water = a.utility == "WATER"
    version = a.model_version or f"{'avg4' if is_water else 'lstm'}-{as_of:%Y%m%d}-h{H}"
    mdir = f"{root}/{version}"
    os.makedirs(mdir, exist_ok=True)
    o = len(U)
    val_hi, val_lo = o - H, o - H - a.val_days + 1

    if is_water:
        rels = []
        for oo in range(val_lo, val_hi + 1):
            base = U.iloc[oo - BASE_DAYS:oo].mean(skipna=False)
            act = U.iloc[oo:oo + H].values.T
            ok = np.isfinite(base.values) & (base.values > 0) & np.isfinite(act).all(axis=1)
            rels.append((act[ok] - base.values[ok][:, None]) / base.values[ok][:, None])
        rel = np.concatenate(rels, axis=0)
        lo, hi = quantiles(rel, a.coverage)
        meta = dict(version=version, method="avg4", utility=a.utility, rusun=a.rusun, horizon=H, lookback=L, level="tower",
                    trained_through=str(as_of.date()), created_at=utcnow(),
                    interval=dict(coverage=a.coverage, avg4_lo=lo, avg4_hi=hi, n_windows=int(len(rel))))
    else:
        cal, _ = LIB.calendar_features(U.index, cal_ns(), extra_days=H)
        S_tr = LIB.samples_at(U.values, cal, range(L, val_lo - H + 1), L, H, True, a.min_kwh)
        S_val = LIB.samples_at(U.values, cal, range(val_lo, val_hi + 1), L, H, True, a.min_kwh)
        S_all = LIB.samples_at(U.values, cal, range(L, val_hi + 1), L, H, True, a.min_kwh)
        if S_tr is None or S_val is None:
            sys.exit("Not enough complete windows to train; check --start-date / data coverage.")
        print(f"windows: train {len(S_tr['base']):,} | validation {len(S_val['base']):,} | all {len(S_all['base']):,}")
        # 1) one early-stopped model that never saw the validation windows -> honest residuals for the interval
        m1, _, _ = LIB.fit_models(S_tr, S_val, S_all, fit_ns(a, 1, True), L, H)
        r1, _ = LIB.predict_ratio(m1, S_val)
        b = S_val["base"][:, None]
        lo, hi = quantiles((S_val["actual"] - b * r1) / b, a.coverage)
        alo, ahi = quantiles((S_val["actual"] - b) / b, a.coverage)
        # 2) production models: early stopping, then refit on everything before the cutoff
        models, best, _ = LIB.fit_models(S_tr, S_val, S_all, fit_ns(a, a.seeds, a.no_refit), L, H)
        for i, m in enumerate(models):
            m.save_weights(f"{mdir}/seed{i}.weights.h5")
        meta = dict(version=version, method="lstm_v2_hybrid", utility=a.utility, rusun=a.rusun, horizon=H, lookback=L,
                    level="unit", min_kwh=a.min_kwh, trained_through=str(as_of.date()), train_start=a.start_date,
                    created_at=utcnow(), seeds=a.seeds, best_epochs=best, loss=a.loss, huber_delta=a.huber_delta, lr=a.lr,
                    c_seq=int(S_tr["seq"].shape[2]), c_fut=int(S_tr["fut"].shape[1] // H),
                    interval=dict(coverage=a.coverage, model_lo=lo, model_hi=hi, avg4_lo=alo, avg4_hi=ahi,
                                  n_windows=int(len(b)), note="empirical quantiles of relative error on the last "
                                  f"{a.val_days} validation days; optimistic out of sample"))
    write_json_atomic(f"{mdir}/meta.json", meta)
    write_json_atomic(pointer, dict(version=version, path=mdir))
    print(f"saved model '{version}' in {mdir} | current pointer: {pointer} | {(time.time() - t0) / 60:.1f} min")


def load_meta(a):
    root, pointer = model_paths(a)
    version = a.model_version
    if not version:
        if not os.path.exists(pointer):
            sys.exit(f"No trained model for horizon {a.horizon}: run 'train' first ({pointer} missing).")
        version = json.load(open(pointer))["version"]
    mdir = f"{root}/{version}"
    return json.load(open(f"{mdir}/meta.json")), mdir


def load_models(meta, mdir):
    models = []
    for i in range(meta["seeds"]):
        m = LIB.build_model(meta["lookback"], meta["horizon"], meta["c_seq"], meta["c_fut"], meta["lr"], meta["loss"],
                            meta["huber_delta"])
        m.load_weights(f"{mdir}/seed{i}.weights.h5")
        models.append(m)
    return models


# ----------------------------------------------------------------------------
# forecast
# ----------------------------------------------------------------------------
def aggregate_rows(units_df):
    """tower and building rows from unit rows (sum of values and of interval bounds)."""
    u = units_df.copy()
    u["tower"] = u["id"].map(tower_of)
    cols = ["value", "lower", "upper", "avg4_value"]
    t = u.groupby(["tower", "date", "horizon_day"], as_index=False).agg(**{c: (c, "sum") for c in cols}, units_included=("id", "nunique"))
    t = t.rename(columns={"tower": "id"}).assign(level="tower", method="aggregate", is_fallback=False, fallback_reason="")
    b = u.groupby(["date", "horizon_day"], as_index=False).agg(**{c: (c, "sum") for c in cols}, units_included=("id", "nunique"))
    b = b.assign(id="building", level="building", method="aggregate", is_fallback=False, fallback_reason="")
    return pd.concat([t, b], ignore_index=True)


def cmd_forecast(a):
    t0 = time.time()
    meta, mdir = load_meta(a)
    H, L = meta["horizon"], meta["lookback"]
    as_of = pd.Timestamp(a.as_of) if a.as_of else data_cutoff(a)
    start = (as_of - pd.Timedelta(days=a.history_days)).date().isoformat()
    U, rep, ids = load_matrix(a, start)
    U = U.loc[:as_of]
    U = U.reindex(pd.date_range(U.index[0], as_of, freq="D"))
    o = len(U)
    if o < L + 1:
        sys.exit("Not enough days in the window for the lookback.")
    dates = pd.date_range(as_of + pd.Timedelta(days=1), periods=H)
    iv = meta["interval"]
    print(f"as of {as_of.date()} | {U.shape[1]} series | model {meta['version']} | forecasting {dates[0].date()} -> {dates[-1].date()}")

    rows, reasons = [], {}
    last = U.iloc[-BASE_DAYS:]
    valid28 = last.notna().sum()
    fb_base = last.mean().where(valid28 >= 14)

    def add(code, vals, lo, hi, base, method, fallback, reason):
        for d in range(H):
            rows.append(dict(level="unit" if meta["level"] == "unit" else "tower", id=code, date=dates[d].date().isoformat(),
                             horizon_day=d + 1, value=vals[d], lower=lo[d], upper=hi[d], avg4_value=base, method=method,
                             is_fallback=fallback, fallback_reason=reason, units_included=1))

    done = set()
    if meta["method"] == "lstm_v2_hybrid":
        cal, _ = LIB.calendar_features(U.index, cal_ns(), extra_days=H)
        S = LIB.samples_at(U.values, cal, [o], L, H, False, meta["min_kwh"])
        if S is not None:
            ratio, _ = LIB.predict_ratio(load_models(meta, mdir), S)
            for k, j in enumerate(S["unit"]):
                base = float(S["base"][k])
                v = np.maximum(base * ratio[k], 0)
                lo = np.minimum(np.maximum(v + np.array(iv["model_lo"]) * base, 0), v)
                hi = np.maximum(v + np.array(iv["model_hi"]) * base, v)
                add(U.columns[j], v, lo, hi, base, "lstm_v2_hybrid", False, "")
                done.add(j)
    for j, code in enumerate(U.columns):
        if j in done:
            continue
        base = fb_base.iloc[j]
        col = U.iloc[:, j].values
        first = np.where(np.isfinite(col))[0]
        if len(first) == 0 or first[0] > o - L:
            reason = "insufficient_history"
        elif not np.isfinite(col[-L:]).all():
            reason = "gap_in_window"
        else:
            reason = "low_use"
        if meta["method"] == "avg4" and np.isfinite(base):
            reason = ""
        if not np.isfinite(base) or base < 0:
            reasons[reason or "no_recent_data"] = reasons.get(reason or "no_recent_data", 0) + 1
            continue
        v = np.full(H, float(base))
        lo = np.minimum(np.maximum(v + np.array(iv["avg4_lo"]) * base, 0), v)
        hi = np.maximum(v + np.array(iv["avg4_hi"]) * base, v)
        add(code, v, lo, hi, float(base), "avg4", meta["method"] != "avg4", reason)
        if meta["method"] != "avg4":
            reasons[reason] = reasons.get(reason, 0) + 1

    F = pd.DataFrame(rows)
    if meta["level"] == "unit":
        F = pd.concat([F, aggregate_rows(F)], ignore_index=True)
    else:
        F = pd.concat([F, aggregate_rows(F.assign(id=F["id"]))[lambda d: d["level"] == "building"]], ignore_index=True)
    for c in ("value", "lower", "upper", "avg4_value"):
        F[c] = F[c].round(3)

    # history export (units or towers, plus tower and building totals)
    hist = []
    for c in U.columns:
        s = U[c].dropna()
        hist.append(pd.DataFrame({"level": "unit" if meta["level"] == "unit" else "tower", "id": c, "date": s.index.strftime("%Y-%m-%d"), "value": s.values.round(3)}))
    T = LIB.aggregate_series(U, "tower") if meta["level"] == "unit" else U
    if meta["level"] == "unit":
        for c in T.columns:
            s = T[c].dropna()
            hist.append(pd.DataFrame({"level": "tower", "id": c, "date": s.index.strftime("%Y-%m-%d"), "value": s.values.round(3)}))
    B = T.sum(axis=1, min_count=T.shape[1]).dropna()
    hist.append(pd.DataFrame({"level": "building", "id": "building", "date": B.index.strftime("%Y-%m-%d"), "value": B.values.round(3)}))
    Hd = pd.concat(hist, ignore_index=True)

    run_id = f"{sk(a)}_h{H}_{as_of:%Y%m%d}_{meta['version']}"
    run = dict(run_id=run_id, status="succeeded", created_at=utcnow(), utility=a.utility, rusun=a.rusun, horizon=H,
               data_cutoff=str(as_of.date()), origin_date=str(dates[0].date()), model_version=meta["version"],
               method=meta["method"], level=meta["level"], series=int(U.shape[1]),
               forecast_series=int(F[F["level"].isin(["unit", "tower"]) & (F["id"] != "building")]["id"].nunique()),
               fallback_reasons=reasons, interval_coverage=iv["coverage"], source=a.source, minutes=round((time.time() - t0) / 60, 2))
    runs = f"{a.store}/runs"
    os.makedirs(runs, exist_ok=True)
    tmp, final = f"{runs}/.tmp_{run_id}", f"{runs}/{run_id}"
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp)
    F.to_csv(f"{tmp}/forecast_daily.csv", index=False)
    Hd.to_csv(f"{tmp}/history_daily.csv", index=False)
    write_json_atomic(f"{tmp}/run.json", run)
    shutil.rmtree(final, ignore_errors=True)
    os.replace(tmp, final)
    ptr_path = f"{a.store}/latest_{a.utility.lower()}_h{H}.json"
    current = json.load(open(ptr_path)) if os.path.exists(ptr_path) else None
    if current and current["data_cutoff"] > run["data_cutoff"]:
        print(f"note: 'latest' stays on the newer cutoff {current['data_cutoff']} (this run is kept for evaluation only)")
    else:
        write_json_atomic(ptr_path, dict(run_id=run_id, path=f"runs/{run_id}", data_cutoff=run["data_cutoff"],
                                         model_version=meta["version"]))
    print(f"run '{run_id}' written: {run['forecast_series']} series forecast, fallbacks {reasons}")


# ----------------------------------------------------------------------------
# evaluate
# ----------------------------------------------------------------------------
def cmd_evaluate(a):
    runs_dir = f"{a.store}/runs"
    if not os.path.isdir(runs_dir):
        sys.exit("No runs yet.")
    rows = []
    for name in sorted(os.listdir(runs_dir)):
        p = f"{runs_dir}/{name}/run.json"
        if name.startswith(".") or not os.path.exists(p):
            continue
        run = json.load(open(p))
        if run["utility"] != a.utility or str(run["rusun"]) != str(a.rusun) or run["horizon"] != a.horizon:
            continue
        origin, H = pd.Timestamp(run["origin_date"]), run["horizon"]
        end = origin + pd.Timedelta(days=H - 1)
        U, _, _ = load_matrix(a, (origin - pd.Timedelta(days=2)).date().isoformat())
        if U.index[-1] < end:
            print(f"  {name}: actuals not complete yet (need {end.date()})")
            continue
        act = U.loc[origin:end]
        F = pd.read_csv(f"{runs_dir}/{name}/forecast_daily.csv")
        lvl = "unit" if run["level"] == "unit" else "tower"
        F = F[F["level"] == lvl]
        g = F.groupby("id").agg(model=("value", "sum"), avg4=("avg4_value", lambda s: s.iloc[0] * H), fb=("is_fallback", "max"))
        g["actual"] = act.sum(min_count=H).reindex(g.index)
        g = g.dropna(subset=["actual"])
        for label, sub in (("all", g), ("model_only", g[~g["fb"]])):
            if not len(sub):
                continue
            e_m, e_b = sub["model"] - sub["actual"], sub["avg4"] - sub["actual"]
            rows.append(dict(run_id=run["run_id"], origin=run["origin_date"], subset=label, n=len(sub),
                             mae_model=e_m.abs().mean(), mae_avg4=e_b.abs().mean(),
                             rmse_model=float(np.sqrt((e_m ** 2).mean())), rmse_avg4=float(np.sqrt((e_b ** 2).mean())),
                             wape_model=100 * e_m.abs().sum() / sub["actual"].sum(), wape_avg4=100 * e_b.abs().sum() / sub["actual"].sum(),
                             bias_model=e_m.mean(), bias_avg4=e_b.mean(),
                             building_err_model=100 * abs(e_m.sum()) / sub["actual"].sum(), building_err_avg4=100 * abs(e_b.sum()) / sub["actual"].sum()))
    if not rows:
        print("nothing to evaluate yet")
        return
    M = pd.DataFrame(rows).round(3)
    path = f"{a.store}/metrics.csv"
    if os.path.exists(path):
        old = pd.read_csv(path)
        M = pd.concat([old[~old["run_id"].isin(M["run_id"])], M], ignore_index=True)
    M.to_csv(path, index=False)
    print(M.to_string(index=False))
    print(f"written {path}")


def main():
    a = parse_args()
    {"train": cmd_train, "forecast": cmd_forecast, "evaluate": cmd_evaluate}[a.command](a)


if __name__ == "__main__":
    main()
