#!/usr/bin/env python3
# ============================================================
# NUSAWATT — thin forecast API (no TensorFlow, no database)
#
# Reads the runs written by forecast_job.py (<store>/latest_<utility>_h<H>.json -> runs/<run_id>/) and returns the
# JSON shape the digital-twin dashboard already expects (src/lib/twin/types.ts):
#   { history: [{date, value}], forecast: [{date, value, lower, upper}], horizonDays, confidence, model }
# plus a "meta" object (method, model version, data cutoff, fallback share) that the UI can show.
#
#   pip install fastapi uvicorn pandas
#   set FORECAST_STORE=forecast_store                 (default)
#   set API_KEY=choose-a-secret                       (optional; then send header  x-api-key)
#   set ALLOWED_ORIGINS=http://localhost:3000,http://localhost:5173
#   uvicorn forecast_api:app --port 8000
#
# Dashboard side:  endpoint: "http://127.0.0.1:8000/api/forecast"   (POST, same body as runForecast sends)
#
# Notes
#  - unitIds may be unit codes ("3A1001" or "A1001"), a tower ("A", "tower_A") or "building"; an empty list = building.
#    Several units are summed, and their interval bounds are summed too (a conservative, wider interval).
#  - metric "water" is forecast per tower (avg4); unit ids are mapped to their towers.
#  - confidence = nominal coverage of the empirical interval (for example 0.8), NOT a model probability.
#  - metric "cost" is not served here (needs the confirmed PLN tariff).
#  - monthKey filters BOTH history and forecast: a past month shows the as-of backtest forecasts (import them with
#    forecast_job.py import-backtest), the month after the cutoff shows the live forecast, and the cutoff month mixes both.
#    meta.forecastType says which one ('live' | 'backtest' | 'mixed' | 'none') so the UI can label it.
# ============================================================
import glob
import json
import os
import re
from typing import List, Optional

import pandas as pd
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

STORE = os.environ.get("FORECAST_STORE", "forecast_store")
API_KEY = os.environ.get("API_KEY", "")
ORIGINS = [o.strip() for o in os.environ.get("ALLOWED_ORIGINS", "http://localhost:3000,http://localhost:5173,http://localhost:8080").split(",") if o.strip()]

app = FastAPI(title="NusaWatt forecast API", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=ORIGINS, allow_methods=["*"], allow_headers=["*"])
_cache = {}


def auth(x_api_key: Optional[str] = Header(default=None)):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="invalid or missing API key")


class ForecastRequest(BaseModel):
    unitIds: List[str] = []
    monthKey: Optional[str] = None
    horizonDays: Optional[int] = None
    metric: str = "electricity"
    endpoint: Optional[str] = None


def database_connection():
    dsn = os.environ.get("SUPABASE_DB_URL")
    if not dsn:
        raise HTTPException(status_code=503, detail="SUPABASE_DB_URL is not configured")
    try:
        import psycopg2
        return psycopg2.connect(dsn)
    except Exception as exc:
        raise HTTPException(status_code=503, detail="could not connect to Supabase database") from exc


def tower_of(code):
    m = re.match(r"^\d+([A-Za-z]+)", str(code))
    return f"tower_{m.group(1)}" if m else "tower_X"


def pointers(utility):
    out = {}
    for p in glob.glob(f"{STORE}/latest_{utility}_h*.json"):
        m = re.search(r"_h(\d+)\.json$", p)
        if m:
            out[int(m.group(1))] = p
    return out


def load_run(pointer_path):
    ptr = json.load(open(pointer_path))
    key = (pointer_path, os.path.getmtime(pointer_path), ptr["run_id"])
    if key not in _cache:
        d = f"{STORE}/{ptr['path']}"
        _cache.clear()
        _cache[key] = dict(run=json.load(open(f"{d}/run.json")), F=pd.read_csv(f"{d}/forecast_daily.csv"),
                           H=pd.read_csv(f"{d}/history_daily.csv"))
    return _cache[key]


def pick_horizon(utility, wanted):
    avail = pointers(utility)
    if not avail:
        raise HTTPException(status_code=503, detail=f"no forecast run for {utility}: run forecast_job.py first")
    ok = [h for h in sorted(avail) if h >= wanted]
    h = ok[0] if ok else max(avail)
    return avail[h], h


def newest_cutoff(utility):
    """data cutoff of the newest run (used to decide whether a month is past, current or future)."""
    avail = pointers(utility)
    if not avail:
        raise HTTPException(status_code=503, detail=f"no forecast run for {utility}: run forecast_job.py first")
    return max(pd.Timestamp(json.load(open(p))["data_cutoff"]) for p in avail.values())


def backtest_points(run, level, ids, ms, me, utility, H):
    """As-of daily forecasts for dates inside [ms, me], taken from the imported rolling backtest
    (each forecast was made before its date; the model was retrained only with earlier data).

    Falls back to the smallest imported backtest horizon when no file matches the picked run's H:
    a 7-day backtest already covers every day of the month (one newest origin per date), and
    without the fallback the cutoff month would report 'live' instead of 'mixed'.
    """
    paths = glob.glob(f"{STORE}/backtest/{utility}_r{run['rusun']}_h{H}.csv")
    if not paths:
        paths = sorted(glob.glob(f"{STORE}/backtest/{utility}_r{run['rusun']}_h*.csv"),
                       key=lambda p: int(re.search(r"_h(\d+)\.csv$", p).group(1)))
    if not paths:
        return None, 0
    key = (paths[0], os.path.getmtime(paths[0]))
    if key not in _cache:
        _cache[key] = pd.read_csv(paths[0])
    B = _cache[key]
    B = B[(B["date"] >= ms.strftime("%Y-%m-%d")) & (B["date"] <= me.strftime("%Y-%m-%d"))]
    if B.empty:
        return None, 0
    B = B.sort_values("horizon_day").drop_duplicates(["unit", "date"], keep="first")      # newest origin per date
    if level == "tower" and run["level"] == "unit":
        B = B[B["unit"].map(tower_of).isin(ids)]
    elif level in ("tower", "unit"):
        B = B[B["unit"].isin(ids)]
    if B.empty:
        return None, 0
    B = B.copy()
    B["value"] = B["base"] if run["method"] == "avg4" else B["forecast"]            # water is served as avg4
    iv = run.get("interval")
    if iv:
        pre = "avg4" if run["method"] == "avg4" else "model"
        lo = pd.Series(iv[f"{pre}_lo"]).reindex(B["horizon_day"] - 1).values
        hi = pd.Series(iv[f"{pre}_hi"]).reindex(B["horizon_day"] - 1).values
        B["lower"] = (B["value"] + lo * B["base"]).clip(lower=0).clip(upper=B["value"])
        B["upper"] = (B["value"] + hi * B["base"]).clip(lower=B["value"])
    else:
        B["lower"], B["upper"] = B["value"], B["value"]
    pts = B.groupby("date", as_index=False)[["value", "lower", "upper"]].sum().sort_values("date")
    return pts, int(B["origin"].nunique())


def resolve(ids, run, F):
    """Return (level, list of ids) in the stored data for the requested ids."""
    level = run["level"]
    all_ids = set(F.loc[F["level"] == level, "id"])
    if not ids or any(i.lower() in ("building", "all") for i in ids):
        return "building", ["building"]
    rusun = str(run["rusun"])
    chosen, towers = [], []
    for i in ids:
        m = re.match(r"^(?:tower_)?([A-Za-z])$", i.strip())
        if m:
            towers.append(f"tower_{m.group(1).upper()}")
            continue
        for cand in (i, f"{rusun}{i}", i.upper(), f"{rusun}{i.upper()}"):
            if cand in all_ids:
                chosen.append(cand)
                break
        else:
            if i in all_ids or i.lower().startswith("tower_"):
                chosen.append(i)
    if level == "tower":              # water: unit ids -> their towers (a unit code need not exist in the water run)
        for i in ids:
            if re.match(r"^(?:tower_)?([A-Za-z])$", i.strip()):
                continue
            code = i if re.match(r"^\d", i) else f"{rusun}{i.upper()}"
            towers.append(tower_of(code))
        t = [x for x in sorted(set(towers)) if x in all_ids]
        return ("tower", t) if t else ("building", ["building"])
    if towers and not chosen:
        return "tower", towers
    if not chosen:
        raise HTTPException(status_code=404, detail="none of the unit ids were found in the latest forecast run")
    return "unit", chosen


@app.get("/api/health")
def health():
    out = {}
    for u in ("electricity", "water"):
        out[u] = {h: json.load(open(p)) for h, p in pointers(u).items()}
    return {"status": "ok", "store": STORE, "latest": out}


@app.get("/api/catalog", dependencies=[Depends(auth)])
def catalog(rusun: str = "3"):
    """Return the real unit/tower identifiers used by forecast_job.py.

    The browser only receives catalog data; the database connection string remains
    in this API process.
    """
    sql = r"""
        select u.id::text as unit_id, u.code as unit_code,
               t.code as tower_code, t.id::text as tower_id
        from "meteran-cerdas".units u
        join "meteran-cerdas".towers t on t.id = u.tower_id
        join "meteran-cerdas".complexes c on c.id = t.complex_id
        where regexp_replace(c.code, '\D', '', 'g') = %(rusun)s
        order by t.code, u.code
    """
    try:
        with database_connection() as conn, conn.cursor() as cur:
            cur.execute(sql, {"rusun": str(rusun)})
            rows = cur.fetchall()
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=502, detail="could not read catalog from Supabase") from exc

    towers = {}
    units = []
    for unit_id, unit_code, tower_code, tower_id in rows:
        tower_code = str(tower_code or "")
        match = re.search(r"([A-Za-z])$", tower_code)
        letter = match.group(1).upper() if match else tower_code
        frontend_tower_id = f"asn{rusun}-t{letter.lower()}"
        towers.setdefault(frontend_tower_id, {
            "id": frontend_tower_id,
            "code": tower_code,
            "name": f"Tower {letter}",
            "rusunId": f"asn{rusun}",
        })
        units.append({
            "id": str(unit_code),
            "databaseId": str(unit_id),
            "code": str(unit_code),
            "towerId": frontend_tower_id,
        })
    return {"rusunId": f"asn{rusun}", "towers": list(towers.values()), "units": units}


@app.post("/api/forecast", dependencies=[Depends(auth)])
def forecast(req: ForecastRequest):
    utility = "water" if req.metric.lower().startswith("water") else "electricity" if req.metric.lower().startswith("elec") else None
    if utility is None:
        raise HTTPException(status_code=501, detail=f"metric '{req.metric}' is not served by this API (electricity and water only)")
    if req.monthKey and not re.match(r"^\d{4}-(0[1-9]|1[0-2])$", req.monthKey):
        raise HTTPException(status_code=422, detail="monthKey must look like 2026-05")
    wanted = req.horizonDays or 7
    ms = me = None
    if req.monthKey:
        ms = pd.Timestamp(f"{req.monthKey}-01")
        me = ms + pd.offsets.MonthEnd(0)
        cutoff0 = newest_cutoff(utility)
        if me > cutoff0:                                    # the month reaches past the data: need enough live horizon
            wanted = max(wanted, int((me - cutoff0).days))
    path, H = pick_horizon(utility, wanted)
    r = load_run(path)
    run, F, Hd = r["run"], r["F"], r["H"]
    cutoff = pd.Timestamp(run["data_cutoff"])
    level, ids = resolve(req.unitIds, run, F)
    f = F[(F["level"] == level) & (F["id"].isin(ids))]
    live = f.groupby(["date", "horizon_day"], as_index=False)[["value", "lower", "upper"]].sum().sort_values("horizon_day")
    live = live[live["horizon_day"] <= min(wanted, run["horizon"])]
    h = Hd[(Hd["level"] == level) & (Hd["id"].isin(ids))].groupby("date", as_index=False)["value"].sum().sort_values("date")
    pts, n_origins, ftype, note = [], 0, "live", "forecast starts the day after dataCutoff"
    if req.monthKey:
        h = h[h["date"].str.startswith(req.monthKey)]
        live = live[(live["date"] >= ms.strftime("%Y-%m-%d")) & (live["date"] <= me.strftime("%Y-%m-%d"))]
        bt, n_origins = (None, 0)
        if ms <= cutoff:
            bt, n_origins = backtest_points(run, level, ids, ms, min(me, cutoff), utility, H)
        if bt is not None:
            pts += list(zip(bt["date"], bt["value"], bt["lower"], bt["upper"]))
        pts += list(zip(live["date"], live["value"], live["lower"], live["upper"]))
        ftype = "none" if not pts else ("mixed" if bt is not None and len(live) else "backtest" if bt is not None else "live")
        note = {"none": "no forecast exists for this month (no backtest imported and the month is not after the data cutoff)",
                "backtest": "past month: as-of forecasts from a rolling backtest, each made before its date; the model was retrained only with earlier data",
                "mixed": "as-of backtest forecasts up to the data cutoff, then the live forecast",
                "live": "forecast starts the day after dataCutoff"}[ftype]
    else:
        h = h.tail(60)
        pts = list(zip(live["date"], live["value"], live["lower"], live["upper"]))
    pts.sort(key=lambda t: t[0])
    fallback_share = float(f.drop_duplicates("id")["is_fallback"].mean()) if len(f) and level != "building" else 0.0
    method = "avg4 baseline" if run["method"] == "avg4" else "LSTM v2 hybrid"
    return {
        "history": [{"date": d, "value": round(float(v), 3)} for d, v in zip(h["date"], h["value"])],
        "forecast": [{"date": d, "value": round(float(v), 3), "lower": round(float(lo), 3), "upper": round(float(up), 3)} for d, v, lo, up in pts],
        "horizonDays": int(len(pts)),
        "confidence": float(run["interval_coverage"]),
        "model": f"{method} ({run['model_version']}); empirical {int(100 * run['interval_coverage'])}% interval",
        "meta": {"method": run["method"], "modelVersion": run["model_version"], "dataCutoff": run["data_cutoff"],
                 "runId": run["run_id"], "level": level, "ids": ids, "forecastType": ftype, "backtestOrigins": n_origins,
                 "fallbackShare": round(fallback_share, 3), "fallbackReasons": run.get("fallback_reasons", {}),
                 "unit": "kWh" if utility == "electricity" else "water units", "note": note},
    }