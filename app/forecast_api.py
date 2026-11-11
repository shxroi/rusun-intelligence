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
    return avail[ok[0] if ok else max(avail)]


def tower_letter(code, rusun=None):
    """Extract the single tower letter from a tower-code-style id.

    Recognized forms: 'A', 'tower_A', 'Tower_A', 'TOWER_A', 'r3A', '3A', 'R3A'.
    Returns the uppercase tower letter, or None if the input is not a tower code.
    """
    if not code:
        return None
    m = re.match(r"^(?:tower_)?r?(\d+)?([A-Za-z])$", str(code).strip(), re.IGNORECASE)
    if m:
        return m.group(2).upper()
    return None


def resolve(ids, run, F, rusun):
    """Return (level, list of ids) in the stored data for the requested ids.

    Water: a single unit code maps to ITS TOWER (a unit code need not exist in the
    water run).  A tower-code resolves to that tower row.
    Electricity: unit codes (optionally prefixed with rusun) and tower codes.
    Empty list / 'building' / 'all' -> the building total.
    """
    level = run["level"]
    all_ids = set(F.loc[F["level"] == level, "id"])
    if not ids or any(i.lower() in ("building", "all") for i in ids):
        return "building", ["building"]

    towers = set()
    chosen = []
    rusun = str(rusun)
    for i in ids:
        i = str(i).strip()
        if not i:
            continue
        letter = tower_letter(i)
        if letter is not None:
            towers.add(f"tower_{letter}")
            continue
        if level == "tower":
            # water: ask for a unit -> the stored data has tower rows,
            # so map the unit code to its tower now (not later).
            code = i if re.match(r"^\d", i) else f"{rusun}{i.upper()}"
            towers.add(tower_of(code))
        else:
            # electricity: unit ids, sometimes with rusun prefix. A unit code that
            # already exists in the stored data resolves to that row (no prefix needed).
            for cand in (i, f"{rusun}{i}", i.upper(), f"{rusun}{i.upper()}"):
                if cand in all_ids:
                    chosen.append(cand)
                    break

    if level == "tower":
        # water: units -> their towers; tower-codes -> themselves; drop dups, keep only ids that exist.
        t = sorted(t for t in towers if t in all_ids)
        return ("tower", t) if t else ("building", ["building"])
    if towers and not chosen:
        return "tower", sorted(towers)
    if not chosen:
        raise HTTPException(status_code=404, detail="none of the unit ids were found in the latest forecast run")
    return "unit", chosen


@app.get("/api/health")
def health():
    out = {}
    for u in ("electricity", "water"):
        out[u] = {h: json.load(open(p)) for h, p in pointers(u).items()}
    return {"status": "ok", "store": STORE, "latest": out}


@app.post("/api/forecast", dependencies=[Depends(auth)])
def forecast(req: ForecastRequest):
    utility = "water" if req.metric.lower().startswith("water") else "electricity" if req.metric.lower().startswith("elec") else None
    if utility is None:
        raise HTTPException(status_code=501, detail=f"metric '{req.metric}' is not served by this API (electricity and water only)")
    wanted = req.horizonDays or 7
    r = load_run(pick_horizon(utility, wanted))
    run, F, Hd = r["run"], r["F"], r["H"]
    level, ids = resolve(req.unitIds, run, F, run["rusun"])
    f = F[(F["level"] == level) & (F["id"].isin(ids))]
    fc = f.groupby(["date", "horizon_day"], as_index=False)[["value", "lower", "upper"]].sum().sort_values("horizon_day")
    fc = fc[fc["horizon_day"] <= min(wanted, run["horizon"])]
    h = Hd[(Hd["level"] == level) & (Hd["id"].isin(ids))].groupby("date", as_index=False)["value"].sum().sort_values("date")
    if req.monthKey:
        h = h[h["date"].str.startswith(req.monthKey)]
    else:
        h = h.tail(60)
    fallback_share = float(f.drop_duplicates("id")["is_fallback"].mean()) if len(f) and level != "building" else 0.0
    method = "avg4 baseline" if run["method"] == "avg4" else "LSTM v2 hybrid"
    return {
        "history": [{"date": d, "value": round(float(v), 3)} for d, v in zip(h["date"], h["value"])],
        "forecast": [{"date": d, "value": round(float(v), 3), "lower": round(float(lo), 3), "upper": round(float(up), 3)}
                     for d, v, lo, up in zip(fc["date"], fc["value"], fc["lower"], fc["upper"])],
        "horizonDays": int(len(fc)),
        "confidence": float(run["interval_coverage"]),
        "model": f"{method} ({run['model_version']}); empirical {int(100 * run['interval_coverage'])}% interval",
        "meta": {"method": run["method"], "modelVersion": run["model_version"], "dataCutoff": run["data_cutoff"],
                 "runId": run["run_id"], "level": level, "ids": ids, "fallbackShare": round(fallback_share, 3),
                 "fallbackReasons": run.get("fallback_reasons", {}), "unit": "kWh" if utility == "electricity" else "water units",
                 "note": "history is actual consumption; forecast starts the day after dataCutoff"},
    }
