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
import datetime
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
# Local dev origins: vite dev (5173-5175), TanStack Start preview, and the
# deployed Cloudflare preview. Override with ALLOWED_ORIGINS to restrict.
_DEFAULT_ORIGINS = ",".join([
    "http://localhost:3000",
    "http://localhost:5173",
    "http://localhost:5174",
    "http://localhost:5175",
    "http://localhost:8080",
    "http://127.0.0.1:5173",
    "http://127.0.0.1:5174",
    "http://127.0.0.1:5175",
])
ORIGINS = [o.strip() for o in os.environ.get("ALLOWED_ORIGINS", _DEFAULT_ORIGINS).split(",") if o.strip()]

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


# ============================================================================
# Consumption summary / series / anomalies (real Supabase meter readings)
# ============================================================================
MONTH_RE = re.compile(r"^\d{4}-(0[1-9]|1[0-2])$")


def month_range(month_key):
    """'2026-05' -> (date(2026,5,1), date(2026,5,31))."""
    if not month_key or not MONTH_RE.match(month_key):
        raise HTTPException(status_code=422, detail="monthKey must look like 2026-05")
    y, m = int(month_key[:4]), int(month_key[5:7])
    start = datetime.date(y, m, 1)
    end = datetime.date(y + (m == 12), (m % 12) + 1, 1) - datetime.timedelta(days=1)
    return start, end


def daily_rows(rusun, month_key, unit_ids=None, tower_id=None, floor=None):
    """Daily electricity/water readings per unit for one month.

    Returns list of dicts: unit_id, unit_code, tower_id, tower_code, floor, date, electricity, water.
    Electricity and water are separate rows in meter_readings (utility_type), pivoted here.
    """
    start, end = month_range(month_key)
    sql = r"""
        select u.id::text as unit_id, u.code as unit_code,
               t.id::text as tower_id, t.code as tower_code, u.floor,
               r.reading_date, r.utility_type::text as utility, r.daily_usage
        from "meteran-cerdas".meter_readings r
        join "meteran-cerdas".units     u on u.id = r.unit_id
        join "meteran-cerdas".towers    t on t.id = u.tower_id
        join "meteran-cerdas".complexes c on c.id = t.complex_id
        where r.deleted_at is null
          and regexp_replace(c.code, '\D', '', 'g') = %(rusun)s
          and r.reading_date between %(start)s and %(end)s
    """
    params = {"rusun": str(rusun), "start": start, "end": end}
    if unit_ids:
        sql += " and u.code = any(%(unit_codes)s)"
        params["unit_codes"] = list(unit_ids)
    if tower_id:
        # Accept either the frontend tower id (asn<rusun>-t<letter>, what the
        # catalog returns and the dashboard sends) or a raw tower UUID.
        m = re.match(r"^asn(\d+)-t([a-z])$", str(tower_id))
        if m:
            sql += " and regexp_replace(t.code, '[^A-Za-z]', '', 'g') ilike %(tower_letter)s"
            params["tower_letter"] = m.group(2).upper()
        else:
            sql += " and t.id::text = %(tower_id)s"
            params["tower_id"] = tower_id
    if floor is not None:
        sql += " and u.floor = %(floor)s"
        params["floor"] = int(floor)
    sql += " order by u.code, r.reading_date, r.utility_type"

    with database_connection() as conn, conn.cursor() as cur:
        cur.execute(sql, params)
        rows = [dict(zip([d[0] for d in cur.description], r)) for r in cur.fetchall()]
    for r in rows:
        r["electricity"] = float(r["daily_usage"]) if r["utility"] == "ELECTRICITY" else None
        r["water"] = float(r["daily_usage"]) if r["utility"] == "WATER" else None
    return rows


def _unit_daily_map(rows):
    """Pivot rows -> {unit_code: {date: {electricity, water}}}, keeping unit metadata."""
    meta, out = {}, {}
    for r in rows:
        code = r["unit_code"]
        meta.setdefault(code, {"unitId": r["unit_id"], "code": code, "towerId": r["tower_id"],
                               "towerCode": r["tower_code"], "floor": r["floor"]})
        d = out.setdefault(code, {}).setdefault(str(r["reading_date"]), {"electricity": None, "water": None})
        if r["electricity"] is not None:
            d["electricity"] = r["electricity"]
        if r["water"] is not None:
            d["water"] = r["water"]
    return meta, out


@app.get("/api/summary", dependencies=[Depends(auth)])
def summary(rusun: str = "3", monthKey: str = "", towerId: str = "", floor: int | None = None,
            tariff_kwh: float | None = None):
    """Monthly totals + per-unit summaries for the dashboard stat cards, heatmaps and reports."""
    if not monthKey:
        raise HTTPException(status_code=422, detail="monthKey is required (e.g. 2026-05)")
    rows = daily_rows(rusun, monthKey, tower_id=towerId or None, floor=floor)
    if not rows:
        return {"scope": "rusun", "monthKey": monthKey, "totals": None, "towers": [], "units": [],
                "note": "no readings for this month"}
    meta, per_unit = _unit_daily_map(rows)

    units = []
    tot_e = tot_w = 0.0
    peak = None
    for code, days in per_unit.items():
        e_vals = [d["electricity"] for d in days.values() if d["electricity"] is not None]
        w_vals = [d["water"] for d in days.values() if d["water"] is not None]
        n = len(days) or 1
        sum_e, sum_w = sum(e_vals), sum(w_vals)
        avg_e, avg_w = sum_e / n, sum_w / n
        tot_e += sum_e
        tot_w += sum_w
        mean = avg_e
        sd = (sum((v - mean) ** 2 for v in e_vals) / (len(e_vals) or 1)) ** 0.5 or 1.0
        max_sigma = max(((v - mean) / sd for v in e_vals), default=0.0)
        m = meta[code]
        # frontend tower id must match the catalog (asn<rusun>-t<letter>)
        tm = re.search(r"([A-Za-z])$", m["towerCode"] or "")
        letter = tm.group(1).upper() if tm else (m["towerCode"] or "X")
        units.append({
            "unitId": m["unitId"], "code": code, "towerId": f"asn{rusun}-t{letter.lower()}",
            "towerCode": m["towerCode"], "floor": m["floor"],
            "totalElectricity": round(sum_e, 2), "totalWater": round(sum_w, 3),
            "avgElectricity": round(avg_e, 3), "avgWater": round(avg_w, 3),
            "maxSigma": round(max_sigma, 2),
        })
        if e_vals and (peak is None or sum_e > peak[1]):
            peak = (code, sum_e)

    # tower comparison — keyed by the same frontend tower id the catalog uses
    # (asn<rusun>-t<letter>) so the dashboard can join the two responses.
    towers = {}
    for u in units:
        m = re.search(r"([A-Za-z])$", u["towerCode"] or "")
        letter = m.group(1).upper() if m else (u["towerCode"] or "X")
        frontend_id = f"asn{rusun}-t{letter.lower()}"
        t = towers.setdefault(frontend_id, {"towerId": frontend_id, "name": f"Tower {letter}",
                                             "electricity": 0.0, "water": 0.0, "units": 0})
        t["electricity"] += u["totalElectricity"]
        t["water"] += u["totalWater"]
        t["units"] += 1
    towers_out = [{"towerId": t["towerId"], "name": t["name"], "units": t["units"],
                   "electricity": round(t["electricity"], 1), "water": round(t["water"], 1)}
                  for t in towers.values()]

    return {
        "scope": "floor" if floor is not None else "tower" if towerId else "rusun",
        "monthKey": monthKey,
        "totals": {
            "electricity": round(tot_e, 1), "water": round(tot_w, 1),
            "units": len(units), "towers": len(towers_out),
            "peakUnit": peak[0] if peak else None,
        },
        "towers": towers_out,
        "units": units,
    }


@app.get("/api/series", dependencies=[Depends(auth)])
def series(rusun: str = "3", monthKey: str = "", unitIds: str = "", towerId: str = ""):
    """Daily actuals per unit (and a summed 'total' series) for the charts."""
    if not monthKey:
        raise HTTPException(status_code=422, detail="monthKey is required (e.g. 2026-05)")
    ids = [u for u in unitIds.split(",") if u] or None
    rows = daily_rows(rusun, monthKey, unit_ids=ids, tower_id=towerId or None)
    meta, per_unit = _unit_daily_map(rows)
    out = {}
    for code, days in per_unit.items():
        out[code] = [{"date": d, "electricity": v["electricity"], "water": v["water"]}
                     for d, v in sorted(days.items())]
    dates = sorted({d for days in per_unit.values() for d in days})
    total = []
    for d in dates:
        e = sum((days.get(d, {}).get("electricity") or 0) for days in per_unit.values())
        w = sum((days.get(d, {}).get("water") or 0) for days in per_unit.values())
        total.append({"date": d, "electricity": round(e, 2), "water": round(w, 3)})
    return {"monthKey": monthKey, "units": out, "total": total}


@app.get("/api/anomalies", dependencies=[Depends(auth)])
def anomalies(rusun: str = "3", monthKey: str = "", threshold: float = 2.0):
    """Per-unit z-score anomalies over the month (same idea as the mock detectAnomalies)."""
    rows = daily_rows(rusun, monthKey)
    meta, per_unit = _unit_daily_map(rows)
    out = []
    for code, days in per_unit.items():
        e_vals = [v["electricity"] for v in days.values() if v["electricity"] is not None]
        if len(e_vals) < 3:
            continue
        mean = sum(e_vals) / len(e_vals)
        sd = (sum((v - mean) ** 2 for v in e_vals) / len(e_vals)) ** 0.5 or 1.0
        for d, v in days.items():
            if v["electricity"] is None:
                continue
            sigma = (v["electricity"] - mean) / sd
            if sigma >= threshold:
                m = meta[code]
                out.append({
                    "id": f"{code}-{d}", "unitId": m["unitId"], "unitCode": code,
                    "towerId": m["towerId"], "floor": m["floor"],
                    "timestamp": f"{d} 19:35", "metric": "electricity",
                    "actual": round(v["electricity"], 2), "expected": round(mean, 2),
                    "sigma": round(sigma, 2), "severity": "anomaly" if sigma >= 2.6 else "warning",
                })
    out.sort(key=lambda a: -a["sigma"])
    return {"monthKey": monthKey, "threshold": threshold, "anomalies": out}


@app.get("/api/series/intervals", dependencies=[Depends(auth)])
def series_intervals(unitId: str = "", date: str = ""):
    """5-minute interval readings for one unit-day.

    The internal database stores daily usage only; interval data exists in the real
    resident-facing database. Returns an empty list with a note so the UI can hide the chart.
    """
    if not unitId or not date:
        raise HTTPException(status_code=422, detail="unitId and date are required")
    return {"unitId": unitId, "date": date, "intervals": [],
            "note": "interval data is not available in the internal database (daily usage only)"}