#!/usr/bin/env python3
# ============================================================
# NUSAWATT — analytical services: anomaly detection (Z-score), peak-load analysis,
# and demo figures for forecast / anomaly / peak load.
#
# Reads the same data as nusawatt_lstm_v2.py (Supabase table "meteran-cerdas".meter_readings via
# SUPABASE_DB_URL, or a legacy CSV) and writes CSV tables + PNG figures to --out.
#
#   python nusawatt_services.py --rusun 3 --start-date 2025-09-01 --out experiments/services
#   python nusawatt_services.py --rusun 3 --start-date 2025-09-01 --seed-demo \
#          --forecast-csv experiments/final/r3_elec_h7_fc/forecast_next.csv --out experiments/services
#   python nusawatt_services.py --rusun 3 --utility WATER --aggregate tower --out experiments/services_water
#
# ANOMALY DETECTION (per unit, daily)
#   z = (usage - mean) / std, where mean/std come from the previous --window days only (causal),
#   with a std floor so quiet units do not produce huge z-scores. Types:
#     high_consumption  z >  threshold            low_consumption  z < -threshold
#     zero_reading      >= --zero-run consecutive zero days on a unit that normally uses >= --min-level
#                       (sensor-fault candidate; can also be an empty unit)
#     missing_data      >= --gap-days consecutive missing days inside a unit's reporting span
#     meter_decrease    cumulative meter went backwards (reset or fault), from the raw rows
#   --seed-demo plants known anomalies in a copy of the data and reports how many are found
#   (this is the check behind UAT cases ANO-02 to ANO-04).
#
# PEAK-LOAD ANALYSIS (daily data)
#   day-of-week demand profile per tower and building, weekday vs weekend, monthly top days and
#   a week x weekday heatmap. Peak HOURS need hourly data, which this script does not use.
# ============================================================
import argparse
import os
import re
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DOW = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
TYPE_COL = {"high_consumption": "#d62728", "low_consumption": "#1f77b4", "zero_reading": "#7f7f7f",
            "missing_data": "#ff7f0e", "meter_decrease": "#9467bd"}
plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": .25, "axes.spines.top": False,
                     "axes.spines.right": False, "font.size": 10})


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
SQL = r"""
select u.code as unit, r.reading_date as date, r.daily_usage as usage, r.total_usage as total
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
                       "date": ts.dt.tz_localize(None).dt.normalize(),
                       "usage": pd.to_numeric(raw["usage"], errors="coerce"),
                       "total": pd.to_numeric(raw[meter], errors="coerce")})
    return df[df["date"] >= pd.Timestamp(a.start_date)]


def clean(df, tol):
    """Matrix days x units (NaN = missing) plus raw-row events (meter decreases, negatives)."""
    df = df.sort_values(["unit", "date"]).copy()
    dup = df.duplicated(["unit", "date"], keep=False)
    df.loc[dup, "usage"] = np.nan
    df = df.drop_duplicates(["unit", "date"])
    prev = df.groupby("unit")["total"].shift()
    bad = (df["total"] < prev - tol) | (df["usage"] < 0)
    events = df.loc[bad, ["unit", "date", "usage"]].assign(expected=np.nan, z=np.nan, type="meter_decrease", run_length=1)
    df.loc[bad, "usage"] = np.nan
    U = df.pivot(index="date", columns="unit", values="usage")
    U = U.reindex(pd.date_range(U.index.min(), U.index.max(), freq="D")).astype("float64")
    return U, events


def tower_of(code):
    m = re.match(r"^\d+([A-Za-z]+)", str(code))
    return f"tower_{m.group(1)}" if m else "tower_X"


def tower_series(U, min_share=0.98):
    """Daily tower totals; a day is NaN if fewer than min_share of the started units reported."""
    codes, V = list(U.columns), U.values
    has = np.isfinite(V).any(0)
    first = np.isfinite(V).argmax(0)
    started = (np.arange(len(U))[:, None] >= first[None, :]) & has[None, :]
    groups = {}
    for c in codes:
        groups.setdefault(tower_of(c), []).append(c)
    out = {}
    for name, cols in sorted(groups.items()):
        idx = [codes.index(c) for c in cols]
        sub, st = V[:, idx], started[:, idx]
        exp, rep = st.sum(1), np.isfinite(sub).sum(1)
        out[name] = np.where((exp > 0) & (rep >= min_share * exp), np.nansum(sub, 1), np.nan)
    return pd.DataFrame(out, index=U.index)


# ----------------------------------------------------------------------------
# Anomaly detection
# ----------------------------------------------------------------------------
def detect_anomalies(U, thr=3.0, window=56, min_periods=28, sigma_floor=0.5, zero_run=3, gap_days=2, min_level=2.0, robust=False):
    prior = U.shift(1)
    if robust:      # median and MAD: one spike does not widen the band for the next 56 days
        mu = prior.rolling(window, min_periods=min_periods).median()
        mad = prior.rolling(window, min_periods=min_periods).apply(lambda v: np.nanmedian(np.abs(v - np.nanmedian(v))), raw=True)
        sd = (1.4826 * mad).clip(lower=sigma_floor)
    else:
        mu = prior.rolling(window, min_periods=min_periods).mean()
        sd = prior.rolling(window, min_periods=min_periods).std().clip(lower=sigma_floor)
    z = (U - mu) / sd
    recs, covered = [], np.zeros(U.shape, dtype=bool)
    for j, c in enumerate(U.columns):
        x, m = U[c].values, mu[c].values
        is0 = np.isfinite(x) & (x <= 1e-9) & np.isfinite(m) & (m >= min_level)
        i = 0
        while i < len(x):
            if is0[i]:
                k = i
                while k + 1 < len(x) and is0[k + 1]:
                    k += 1
                if k - i + 1 >= zero_run:
                    recs.append((c, U.index[i], 0.0, m[i], np.nan, "zero_reading", k - i + 1))
                    covered[i:k + 1, j] = True
                i = k + 1
            else:
                i += 1
        v = np.isfinite(x)
        if v.any():
            f, l = int(np.argmax(v)), len(v) - 1 - int(np.argmax(v[::-1]))
            i = f
            while i <= l:
                if not v[i]:
                    k = i
                    while k + 1 <= l and not v[k + 1]:
                        k += 1
                    if k - i + 1 >= gap_days:
                        recs.append((c, U.index[i], np.nan, np.nan, np.nan, "missing_data", k - i + 1))
                    i = k + 1
                else:
                    i += 1
    zv = z.values
    for typ, mask in (("high_consumption", (zv > thr) & ~covered), ("low_consumption", (zv < -thr) & ~covered)):
        r, c_ = np.where(mask)
        for ri, ci in zip(r, c_):
            recs.append((U.columns[ci], U.index[ri], U.values[ri, ci], mu.values[ri, ci], zv[ri, ci], typ, 1))
    A = pd.DataFrame(recs, columns=["unit", "date", "usage", "expected", "z", "type", "run_length"])
    return A.sort_values(["date", "unit"]).reset_index(drop=True), mu, sd, z


def seed_anomalies(U, seed=7):
    """Plant known anomalies in a copy of U (last ~60 days) for a detection check."""
    rng = np.random.default_rng(seed)
    X, ev = U.copy(), []
    recent = X.iloc[-120:]
    ok = recent.columns[(recent.mean() > 5) & recent.notna().all()]
    if len(ok) < 4:
        return X, ev
    pick = list(rng.choice(ok, 4, replace=False))
    days = X.index[-60:]
    d_hi, d_lo, d_zero, d_gap = days[5], days[18], days[30], days[44]
    X.loc[d_hi:d_hi + pd.Timedelta(days=1), pick[0]] *= 3.0
    ev.append(dict(unit=pick[0], start=d_hi, end=d_hi + pd.Timedelta(days=1), type="high_consumption", what="3x usage for 2 days"))
    X.loc[d_lo:d_lo + pd.Timedelta(days=1), pick[1]] *= 0.1
    ev.append(dict(unit=pick[1], start=d_lo, end=d_lo + pd.Timedelta(days=1), type="low_consumption", what="10% of usage for 2 days"))
    X.loc[d_zero:d_zero + pd.Timedelta(days=4), pick[2]] = 0.0
    ev.append(dict(unit=pick[2], start=d_zero, end=d_zero, type="zero_reading", what="5 zero days"))
    X.loc[d_gap:d_gap + pd.Timedelta(days=3), pick[3]] = np.nan
    ev.append(dict(unit=pick[3], start=d_gap, end=d_gap, type="missing_data", what="4 missing days"))
    return X, ev


def check_seeded(A, ev):
    rows = []
    for e in ev:
        hit = A[(A["unit"] == e["unit"]) & (A["type"] == e["type"]) & (A["date"] >= e["start"]) & (A["date"] <= e["end"])]
        rows.append({**e, "detected": len(hit) > 0, "max_abs_z": float(hit["z"].abs().max()) if len(hit) and hit["z"].notna().any() else np.nan})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Peak load
# ----------------------------------------------------------------------------
def peak_load(T, topk=3):
    S = T.copy()
    S["building"] = T.sum(axis=1, min_count=T.shape[1])
    dow = S.groupby(S.index.dayofweek).mean()
    dow.index = [DOW[i] for i in dow.index]
    wk = pd.DataFrame({"weekday_mean": S[S.index.dayofweek < 5].mean(), "weekend_mean": S[S.index.dayofweek >= 5].mean()})
    wk["weekend_vs_weekday_pct"] = 100 * (wk["weekend_mean"] / wk["weekday_mean"] - 1)
    B = S["building"].dropna()
    rows = []
    for m, g in B.groupby(B.index.to_period("M")):
        z = (g - g.mean()) / (g.std() if g.std() > 0 else 1)
        for d in g.nlargest(topk).index:
            rows.append(dict(month=str(m), date=d.date(), dow=DOW[d.dayofweek], total=g[d], z_in_month=z[d]))
    top = pd.DataFrame(rows)
    hv = B.to_frame("v")
    hv["week"] = (hv.index - pd.to_timedelta(hv.index.dayofweek, unit="D"))
    hv["dow"] = hv.index.dayofweek
    heat = hv.pivot(index="week", columns="dow", values="v")
    b = dow["building"]
    best, worst = b.idxmax(), b.idxmin()
    text = (f"Highest average building demand: {best} ({b[best]:.0f} per day, {100 * (b[best] / b.mean() - 1):+.1f}% vs the weekly mean). "
            f"Lowest: {worst} ({b[worst]:.0f}). Weekends are {wk.loc['building', 'weekend_vs_weekday_pct']:+.1f}% vs weekdays. "
            "Hourly peak windows need hourly data (not used here).")
    return dict(S=S, dow=dow, weekday_weekend=wk, top_days=top, heat=heat, text=text)


# ----------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------
def fig_anomaly(U, A, mu, sd, thr, out, unit_label, seeded=None, name="fig_anomaly", scope_label="all units"):
    counts = A[A["type"] != "missing_data"].groupby("unit").size() if len(A) else pd.Series(dtype=int)
    if seeded is not None and len(seeded):
        cand = list(seeded["unit"])[:1] + list(counts.sort_values(ascending=False).index[:1])
    else:
        cand = list(counts.sort_values(ascending=False).index[:1]) or [U.columns[0]]
    unit = cand[0]
    fig = plt.figure(figsize=(11, 8.2))
    gs = fig.add_gridspec(3, 1, height_ratios=[2.2, 1.2, 1.4], hspace=.38)
    a1, a2, a3 = (fig.add_subplot(gs[i]) for i in range(3))
    x = U[unit]
    a1.plot(x.index, x.values, color="0.25", lw=1.1, label="daily usage")
    a1.plot(mu.index, mu[unit].values, color="tab:blue", lw=1, label="expected (previous 56 days)")
    a1.fill_between(mu.index, (mu[unit] - thr * sd[unit]).values, (mu[unit] + thr * sd[unit]).values, color="tab:blue", alpha=.12,
                    label=f"mean \u00b1 {thr:g} std")
    for typ, g in A[A["unit"] == unit].groupby("type"):
        if typ == "missing_data":
            for _, r in g.iterrows():
                a1.axvspan(r["date"], r["date"] + pd.Timedelta(days=int(r["run_length"])), color=TYPE_COL[typ], alpha=.25, label="missing data")
        else:
            a1.scatter(g["date"], g["usage"], s=36, color=TYPE_COL[typ], zorder=5, label=typ.replace("_", " "))
    h, l = a1.get_legend_handles_labels()
    a1.legend(dict(zip(l, h)).values(), dict(zip(l, h)).keys(), ncol=3, fontsize=8)
    a1.set_ylabel(unit_label)
    a1.set_title(f"Anomaly detection example: {unit}" + (" (with planted anomalies)" if seeded is not None else ""))
    z = ((U[unit] - mu[unit]) / sd[unit])
    a2.plot(z.index, z.values, color="0.3", lw=1)
    a2.axhline(thr, color="tab:red", ls="--", lw=1)
    a2.axhline(-thr, color="tab:red", ls="--", lw=1)
    a2.set_ylabel("Z-score")
    cnt = A.copy()
    cnt["month"] = cnt["date"].dt.to_period("M").astype(str)
    piv = cnt.groupby(["month", "type"]).size().unstack(fill_value=0)
    bottom = np.zeros(len(piv))
    for typ in piv.columns:
        a3.bar(piv.index, piv[typ].values, bottom=bottom, color=TYPE_COL.get(typ, "k"), label=typ.replace("_", " "))
        bottom += piv[typ].values
    a3.set_ylabel(f"flags per month ({scope_label})")
    a3.tick_params(axis="x", rotation=60, labelsize=8)
    if len(piv.columns):
        a3.legend(ncol=3, fontsize=8)
    fig.savefig(f"{out}/{name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_peak(P, out, unit_label):
    S, dow = P["S"], P["dow"]
    fig, ax = plt.subplots(2, 2, figsize=(12, 8.2))
    b = dow["building"]
    colors = ["#d62728" if v == b.max() else "#1f77b4" for v in b.values]
    ax[0, 0].bar(b.index, b.values, color=colors)
    ax[0, 0].axhline(b.mean(), color="k", ls="--", lw=1)
    ax[0, 0].set_title("Average building demand by weekday (peak in red)")
    ax[0, 0].set_ylabel(f"{unit_label} per day")
    towers = [c for c in dow.columns if c != "building"]
    for c in towers:
        ax[0, 1].plot(dow.index, 100 * dow[c] / dow[c].mean(), "-o", ms=4, label=c.replace("tower_", "Tower "))
    ax[0, 1].axhline(100, color="k", lw=.8)
    ax[0, 1].set_title("Weekday shape per tower (100 = the tower's weekly mean)")
    ax[0, 1].legend(ncol=3, fontsize=8)
    heat = P["heat"]
    im = ax[1, 0].imshow(heat.values, aspect="auto", cmap="YlOrRd")
    ax[1, 0].set_xticks(range(7))
    ax[1, 0].set_xticklabels(DOW)
    step = max(1, len(heat) // 8)
    ax[1, 0].set_yticks(range(0, len(heat), step))
    ax[1, 0].set_yticklabels([d.strftime("%d %b %y") for d in heat.index[::step]], fontsize=8)
    ax[1, 0].set_title("Building demand: week (rows) x weekday (columns)")
    ax[1, 0].grid(False)
    fig.colorbar(im, ax=ax[1, 0], shrink=.85, label=unit_label)
    B = S["building"]
    ax[1, 1].plot(B.index, B.values, color="0.55", lw=.9, label="daily building total")
    ax[1, 1].plot(B.index, B.rolling(7, min_periods=4).mean(), color="tab:blue", lw=1.6, label="7-day mean")
    top = P["top_days"]
    if len(top):
        ax[1, 1].scatter(pd.to_datetime(top["date"]), top["total"], color="tab:red", s=26, zorder=5, label="top days per month")
    ax[1, 1].set_title("Daily building total and monthly top days")
    ax[1, 1].legend(fontsize=8)
    ax[1, 1].tick_params(axis="x", rotation=45, labelsize=8)
    fig.suptitle(P["text"], fontsize=9, y=0.995, wrap=True)
    fig.savefig(f"{out}/fig_peak_load.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_forecast(U, T, H, fc_path, out, unit_label, hist_days=60):
    last = U.index[-1]
    recent = U.iloc[-28:]
    base = recent.mean().where(recent.notna().sum() >= 28)
    fc = pd.read_csv(fc_path, parse_dates=["target_date"]) if fc_path and os.path.exists(fc_path) else None
    method = "model forecast" if fc is not None else None
    dates = pd.date_range(last + pd.Timedelta(days=1), periods=H)
    elig = base.dropna()
    show = []
    if len(elig):
        show = [elig.idxmax(), (elig - elig.median()).abs().idxmin()]
    fig, axes = plt.subplots(1 + len(show), 1, figsize=(10.5, 3.4 * (1 + len(show))), sharex=False)
    axes = np.atleast_1d(axes)
    B = T.sum(axis=1, min_count=T.shape[1])
    h = B.iloc[-hist_days:]
    axes[0].plot(h.index, h.values, color="0.25", label="actual (building total)")
    units_fc = None
    if fc is not None:
        tot = fc.groupby("target_date")["predicted_usage"].sum()
        units_fc = set(fc["unit"])
        axes[0].plot(tot.index, tot.values, "-o", ms=4, color="tab:blue", label=f"{method} ({len(units_fc)} units)")
        avg = base[[u for u in base.index if u in units_fc]].sum()
        axes[0].plot(dates, [avg] * H, "--", color="tab:gray", label="avg4 level, same units")
    else:
        avg = base.sum()
        axes[0].plot(dates, [avg] * H, "--", color="tab:gray", label="avg4 level (baseline only)")
    axes[0].axvline(last, color="k", lw=.8, ls=":")
    axes[0].set_title("Building total: history and next " + str(H) + " days" + ("" if fc is not None else
                      "  |  pass --forecast-csv from nusawatt_lstm_v2.py --forecast to add the model"))
    axes[0].set_ylabel(unit_label + " per day")
    axes[0].legend(fontsize=8)
    for ax, u in zip(axes[1:], show):
        hh = U[u].iloc[-hist_days:]
        ax.plot(hh.index, hh.values, color="0.25", label="actual")
        if fc is not None and u in set(fc["unit"]):
            f = fc[fc["unit"] == u].sort_values("target_date")
            ax.plot(f["target_date"], f["predicted_usage"], "-o", ms=4, color="tab:blue", label="model forecast")
        ax.plot(dates, [base[u]] * H, "--", color="tab:gray", label="avg4 level")
        ax.axvline(last, color="k", lw=.8, ls=":")
        ax.set_title(f"Unit {u}")
        ax.set_ylabel(unit_label)
        ax.legend(fontsize=8)
    for ax in axes:
        ax.tick_params(axis="x", rotation=40, labelsize=8)
    fig.tight_layout()
    fig.savefig(f"{out}/fig_forecast.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="NusaWatt analytical services and demo figures")
    p.add_argument("--source", choices=["supabase", "csv"], default="supabase")
    p.add_argument("--env-file", default=".env", help="local KEY=VALUE file (default: .env)")
    p.add_argument("--csv", default="")
    p.add_argument("--rusun", default="3")
    p.add_argument("--utility", default="ELECTRICITY", choices=["ELECTRICITY", "WATER"])
    p.add_argument("--aggregate", choices=["none", "tower"], default="none", help="WATER: use tower")
    p.add_argument("--start-date", default="2025-09-01")
    p.add_argument("--threshold", type=float, default=3.0, help="Z-score threshold")
    p.add_argument("--window", type=int, default=56, help="days used for the rolling mean and std")
    p.add_argument("--sigma-floor", type=float, default=0.5)
    p.add_argument("--zero-run", type=int, default=3)
    p.add_argument("--gap-days", type=int, default=2)
    p.add_argument("--robust", action="store_true", help="median/MAD baseline instead of mean/std (less masking after a spike)")
    p.add_argument("--min-level", type=float, default=2.0, help="usage a unit normally has before a zero run is suspicious")
    p.add_argument("--horizon", type=int, default=7)
    p.add_argument("--forecast-csv", default="", help="forecast_next.csv from nusawatt_lstm_v2.py --forecast")
    p.add_argument("--seed-demo", action="store_true", help="plant known anomalies and report how many are found")
    p.add_argument("--out", default="experiments/services")
    return p.parse_args()


def load_env_file(path):
    """Load simple KEY=VALUE entries without overwriting existing environment variables."""
    if not path or not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key, value = key.strip(), value.strip()
            if value[:1] == value[-1:] and value[:1] in ("'", '"'):
                value = value[1:-1]
            if key:
                os.environ.setdefault(key, value)


def main():
    a = parse_args()
    load_env_file(a.env_file)
    os.makedirs(a.out, exist_ok=True)
    unit_label = "kWh" if a.utility == "ELECTRICITY" else "water units"
    raw = load_supabase(a) if a.source == "supabase" else load_csv(a)
    tol = 1.0 if a.utility == "WATER" else 0.02
    U, raw_events = clean(raw, tol)
    print(f"{U.shape[1]} units x {U.shape[0]} days ({U.index[0].date()} -> {U.index[-1].date()})")
    T = tower_series(U)
    series = T if a.aggregate == "tower" else U
    if a.utility == "WATER" and a.aggregate == "none":
        print("WARNING: per-unit daily water is mostly 0/1 counts, so Z-scores are not meaningful; use --aggregate tower")
    kw = dict(thr=a.threshold, window=a.window, sigma_floor=a.sigma_floor, zero_run=a.zero_run, gap_days=a.gap_days,
              min_level=a.min_level if a.utility == "ELECTRICITY" else 0.0, robust=a.robust)

    print("-- Anomaly detection --")
    A, mu, sd, z = detect_anomalies(series, **kw)
    if a.aggregate == "none" and len(raw_events):
        A = pd.concat([A, raw_events[A.columns]], ignore_index=True).sort_values(["date", "unit"]).reset_index(drop=True)
    A.to_csv(f"{a.out}/anomalies.csv", index=False)
    n_cells = int(np.isfinite(series.values).sum())
    flagged = A[A["type"].isin(["high_consumption", "low_consumption"])]
    print(f"  flags by type: {A['type'].value_counts().to_dict()}")
    print(f"  high/low flags: {len(flagged)} of {n_cells:,} unit-days = {1000 * len(flagged) / max(n_cells, 1):.1f} per 1,000 "
          f"(a Gaussian series would give about 2.7 per 1,000 at 3 sigma)")
    seeded = None
    if a.seed_demo:
        Xs, ev = seed_anomalies(series)
        As, mus, sds, _ = detect_anomalies(Xs, **kw)
        seeded = check_seeded(As, ev)
        seeded.to_csv(f"{a.out}/seeded_check.csv", index=False)
        print(f"  seeded check: {int(seeded['detected'].sum())}/{len(seeded)} planted anomalies detected")
        print(seeded[["unit", "type", "what", "detected", "max_abs_z"]].to_string(index=False))
        fig_anomaly(Xs, As, mus, sds, a.threshold, a.out, unit_label, seeded)
    else:
        fig_anomaly(series, A, mu, sd, a.threshold, a.out, unit_label)

    print("-- Peak load --")
    P = peak_load(T)
    P["dow"].round(2).to_csv(f"{a.out}/peak_dow_profile.csv")
    P["weekday_weekend"].round(2).to_csv(f"{a.out}/peak_weekday_weekend.csv")
    P["top_days"].round(2).to_csv(f"{a.out}/peak_monthly_top_days.csv", index=False)
    open(f"{a.out}/peak_insight.txt", "w").write(P["text"] + "\n")
    print("  " + P["text"])
    fig_peak(P, a.out, unit_label)

    print("-- Forecast figure --")
    fig_forecast(U if a.aggregate == "none" else U, T, a.horizon, a.forecast_csv, a.out, unit_label)
    print(f"Done. Tables and figures in '{a.out}/'")


if __name__ == "__main__":
    main()
