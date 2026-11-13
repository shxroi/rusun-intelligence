#!/usr/bin/env python3
# ============================================================
# NUSAWATT — MODEL TESTS (offline evidence for the thesis / UAT model section)
#
# Reads the backtest written by nusawatt_lstm_v2.py (backtest_unit_origin.csv: one row per
# unit x origin, horizon-sum in kWh) and produces the tests below. No TensorFlow is needed
# for the statistical tests; only --selftest touches your model code.
#
#   MOD-08  Relative MAE vs a baseline (geometric mean; Davydenko & Fildes 2013) + bootstrap CI
#   MOD-09  Diebold-Mariano test with the Harvey-Leybourne-Newbold small-sample correction
#           (Diebold & Mariano 1995; Harvey, Leybourne & Newbold 1997), absolute- and squared-error loss
#   MOD-10  Prediction-interval coverage (unit / tower / building)
#   MOD-11  Mincer-Zarnowitz regression + bias breakdown (Mincer & Zarnowitz 1969)
#   MOD-05  Reproducibility of a re-run against a reference backtest (default tolerance 2%)
#   MOD-12  --selftest: input causality of samples_at (future data cannot change the inputs),
#           window eligibility, the avg4 definition, and (if TensorFlow is installed) that an
#           untrained network reproduces avg4
#   Also: headline metrics (MAE, RMSE, WAPE, bias, CV-RMSE, NMBE, building error), a claim gate
#   (>= 8 origins and >= 500 unit-origin rows, the rule from the old nusawatt_common.py), data checks.
#
# Examples
#   python model_tests.py --backtest experiments/lstm_v2/backtest_unit_origin.csv --horizon 7 \
#          --extra ridge=experiments/ridge/backtest_unit_origin.csv --out experiments/model_tests/elec_h7
#   python model_tests.py --backtest experiments/h30/backtest_unit_origin.csv --horizon 30 \
#          --out experiments/model_tests/elec_h30
#   python model_tests.py --backtest water/backtest_unit_origin.csv --level tower --horizon 7 --out experiments/model_tests/water
#   python model_tests.py --selftest --lib app.nusawatt_lstm_v2 --lib-dir .
#
# IMPORTANT LIMITS (read before quoting results)
#   * Interval coverage here is a WALK-FORWARD PROXY on horizon sums: for each origin the band is
#     built from the relative errors of earlier origins whose targets had already ended. It tests the
#     recipe (empirical relative-error quantiles), NOT the exact daily bands served by forecast_api.py.
#     If your backtest CSV has `lower`/`upper` columns, they are used instead (--interval-mode columns).
#   * Tower and building bands sum the unit bounds, exactly like forecast_job.py / the API.
#   * Weekly origins with a long horizon (e.g. 30 days) overlap; the DM horizon parameter h is set
#     to ceil(horizon / origin spacing) and the result is flagged INDICATIVE when h > 1.
#   * Mincer-Zarnowitz p-values use cluster-robust errors by origin; with few origins they are rough.
#   * Pass/fail labels are decision aids for the UAT, not statistical standards. Thresholds are options.
#   * Install scipy for the numbers you quote (pip install scipy). Without it p-values fall back to normal /
#     chi-square approximations that are too optimistic with few origins; the script prints a warning.
# ============================================================
import argparse
import importlib
import json
import math
import os
import re
import sys

import numpy as np
import pandas as pd

try:
    from scipy import stats as _st
except Exception:  # scipy is optional; normal / chi-square approximations are used instead
    _st = None

MIN_ORIGINS_FOR_CLAIM = 8      # same rule as the old nusawatt_common.py
MIN_ROWS_FOR_CLAIM = 500


# ----------------------------------------------------------------------------
# small helpers
# ----------------------------------------------------------------------------
def tower_of(code):
    m = re.match(r"^\d*([A-Za-z]+)", str(code))
    return f"tower_{m.group(1).upper()}" if m else "tower_X"


def t_cdf(x, df):
    if _st is not None:
        return float(_st.t.cdf(x, df))
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))       # normal approximation


def f_sf_2(x, df2):
    """Survival function of F(2, df2); falls back to chi-square(2) which is exp(-W/2) with W = 2x."""
    if _st is not None and df2 > 0:
        return float(_st.f.sf(x, 2, df2))
    return float(math.exp(-x))


def md_table(df, floatfmt=3):
    if df is None or not len(df):
        return "_(none)_\n"
    d = df.copy()
    for c in d.columns:
        if pd.api.types.is_float_dtype(d[c]):
            d[c] = d[c].map(lambda v: "" if pd.isna(v) else f"{v:.{floatfmt}f}")
    head = "| " + " | ".join(map(str, d.columns)) + " |"
    sep = "| " + " | ".join("---" for _ in d.columns) + " |"
    body = ["| " + " | ".join(str(v) for v in row) + " |" for row in d.itertuples(index=False)]
    return "\n".join([head, sep] + body) + "\n"


def boot_mean_ci(values, B, rng, alpha=0.05):
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    if len(v) < 2:
        return float("nan"), float("nan")
    idx = rng.integers(0, len(v), size=(B, len(v)))
    m = v[idx].mean(axis=1)
    return float(np.percentile(m, 100 * alpha / 2)), float(np.percentile(m, 100 * (1 - alpha / 2)))


# ----------------------------------------------------------------------------
# loading + data checks
# ----------------------------------------------------------------------------
def load_backtest(a):
    df = pd.read_csv(a.backtest)
    baselines = [b for b in a.baselines.split(",") if b]
    need = ["origin", a.id_col, "actual", a.model_col, a.scale_col] + baselines
    missing = [c for c in need if c not in df.columns]
    if missing:
        sys.exit(f"Columns missing in {a.backtest}: {missing}. Found: {list(df.columns)}")
    df["origin"] = pd.to_datetime(df["origin"])
    notes = []
    for spec in a.extra or []:
        if "=" not in spec:
            sys.exit("--extra needs NAME=PATH (the NAME column must exist in that CSV)")
        name, path = spec.split("=", 1)
        e = pd.read_csv(path)
        if name not in e.columns:
            sys.exit(f"--extra {name}: column '{name}' not found in {path}. Found: {list(e.columns)}")
        e["origin"] = pd.to_datetime(e["origin"])
        e = e[["origin", a.id_col, name]]
        before = len(df)
        df = df.merge(e, on=["origin", a.id_col], how="inner")
        notes.append(f"extra '{name}': kept {len(df)} of {before} rows after matching (origin, {a.id_col})")
    return df, baselines, notes


def data_checks(df, a, baselines):
    rows = []
    cols = ["actual", a.model_col, a.scale_col] + baselines
    rows.append(("DATA-01", "no missing values in used columns", "PASS" if not df[cols].isna().any().any() else "FAIL",
                 f"{int(df[cols].isna().sum().sum())} missing cells"))
    dup = int(df.duplicated(["origin", a.id_col]).sum())
    rows.append(("DATA-02", "one row per (origin, id)", "PASS" if dup == 0 else "FAIL", f"{dup} duplicate keys"))
    neg = int((df[[a.model_col] + baselines] < 0).any(axis=1).sum())
    rows.append(("DATA-03", "no negative forecasts", "PASS" if neg == 0 else "FAIL", f"{neg} rows"))
    nz = int((df[a.scale_col] <= 0).sum())
    rows.append(("DATA-04", f"{a.scale_col} (base x horizon) is positive", "PASS" if nz == 0 else "FAIL", f"{nz} rows"))
    n_or, n_rows = df["origin"].nunique(), len(df)
    if a.level == "unit":
        ok = n_or >= MIN_ORIGINS_FOR_CLAIM and n_rows >= MIN_ROWS_FOR_CLAIM
        rows.append(("GATE-01", f"claim gate: >= {MIN_ORIGINS_FOR_CLAIM} origins and >= {MIN_ROWS_FOR_CLAIM} unit-origin rows",
                     "PASS" if ok else "FAIL", f"{n_or} origins, {n_rows:,} rows"))
    else:   # the 500-row rule was written for units; a handful of towers can never reach it
        rows.append(("GATE-01", f"claim gate: >= {MIN_ORIGINS_FOR_CLAIM} origins (row rule applies to unit level only)",
                     "PASS" if n_or >= MIN_ORIGINS_FOR_CLAIM else "FAIL",
                     f"{n_or} origins, {df[a.id_col].nunique()} series, {n_rows:,} rows; few series = low power"))
    return rows


# ----------------------------------------------------------------------------
# headline metrics
# ----------------------------------------------------------------------------
def headline(df, methods, a):
    mean_act = df["actual"].mean()
    rows = []
    for m in methods:
        e = df[m] - df["actual"]                       # sign convention of the script: forecast - actual
        tot = df.groupby("origin")[[m, "actual"]].sum()
        bld = ((tot[m] - tot["actual"]).abs() / tot["actual"] * 100).mean()
        rmse = float(np.sqrt((e ** 2).mean()))
        rows.append(dict(method=m, n_rows=len(df), n_origins=df["origin"].nunique(), mean_actual=mean_act,
                         mae=e.abs().mean(), rmse=rmse, wape_pct=100 * e.abs().sum() / df["actual"].sum(),
                         bias=e.mean(), cv_rmse_pct=100 * rmse / mean_act, nmbe_pct=100 * e.mean() / mean_act,
                         building_err_pct=bld))
    return pd.DataFrame(rows)


def seed_spread(df, a):
    cols = [c for c in df.columns if c.startswith(f"{a.model_col}_seed")]
    if len(cols) < 2:
        return pd.DataFrame()
    maes = {c: float((df[c] - df["actual"]).abs().mean()) for c in cols}
    ens = float((df[a.model_col] - df["actual"]).abs().mean())
    return pd.DataFrame([dict(n_seeds=len(cols), mae_ensemble=ens, mae_seed_min=min(maes.values()),
                              mae_seed_max=max(maes.values()),
                              seed_spread_pct_of_ensemble=100 * (max(maes.values()) - min(maes.values())) / ens)])


# ----------------------------------------------------------------------------
# MOD-08 relative MAE
# ----------------------------------------------------------------------------
def relative_mae(df, model, base, a, rng):
    am, ab = (df[model] - df["actual"]).abs(), (df[base] - df["actual"]).abs()
    d = pd.DataFrame({"origin": df["origin"], "id": df[a.id_col], "am": am, "ab": ab})
    pooled = float(d["am"].sum() / d["ab"].sum())
    # bootstrap over origins for the pooled ratio
    po = d.groupby("origin")[["am", "ab"]].sum()
    idx = rng.integers(0, len(po), size=(a.boot, len(po)))
    ratios = po["am"].values[idx].sum(axis=1) / po["ab"].values[idx].sum(axis=1)
    p_lo, p_hi = np.percentile(ratios, [2.5, 97.5])
    # geometric mean of per-origin relative MAE
    mo = d.groupby("origin")[["am", "ab"]].mean()
    r_o = (mo["am"] / mo["ab"]).replace([np.inf, -np.inf], np.nan).dropna()
    r_o = r_o[r_o > 0]
    gm_o = float(np.exp(np.log(r_o).mean()))
    lo_o, hi_o = boot_mean_ci(np.log(r_o.values), a.boot, rng)
    # weighted geometric mean of per-unit relative MAE (weights = number of origins of the unit)
    gu = d.groupby("id").agg(am=("am", "mean"), ab=("ab", "mean"), n=("am", "size"))
    ok = (gu["am"] > 0) & (gu["ab"] > 0)
    excl = int((~ok).sum())
    gu = gu[ok]
    lr, w = np.log(gu["am"] / gu["ab"]).values, gu["n"].values.astype(float)
    gm_u = float(np.exp((w * lr).sum() / w.sum()))
    ii = rng.integers(0, len(lr), size=(a.boot, len(lr)))
    boot_u = np.exp((w[ii] * lr[ii]).sum(axis=1) / w[ii].sum(axis=1))
    u_lo, u_hi = np.percentile(boot_u, [2.5, 97.5])
    share_better = float((gu["am"] < gu["ab"]).mean())
    out = []
    for name, val, lo, hi, note in (
        ("pooled (sum |err| model / sum |err| baseline)", pooled, p_lo, p_hi, "CI: bootstrap over origins"),
        ("geometric mean over origins", gm_o, float(np.exp(lo_o)), float(np.exp(hi_o)), "CI: bootstrap over origins"),
        ("weighted geometric mean over units", gm_u, u_lo, u_hi,
         f"CI: bootstrap over units; {excl} units excluded (zero MAE); model better for {100 * share_better:.0f}% of units"),
    ):
        out.append(dict(model=model, baseline=base, measure=name, rel_mae=val, ci95_lo=lo, ci95_hi=hi,
                        improvement_pct=100 * (1 - val), note=note))
    return out


# ----------------------------------------------------------------------------
# MOD-09 Diebold-Mariano (+ HLN)
# ----------------------------------------------------------------------------
def dm_hln(d, h):
    d = np.asarray(d, float)
    T = len(d)
    if T < 3:
        return dict(T=T, h=h, status="too few origins")
    dbar = d.mean()

    def gamma(k):
        return float(np.sum((d[k:] - dbar) * (d[:T - k] - dbar)) / T)

    var = (gamma(0) + 2 * sum(gamma(k) for k in range(1, h))) / T       # rectangular window, as in the original DM
    flag = ""
    if var <= 0:                                                       # can happen in small samples
        var = (gamma(0) + 2 * sum((1 - k / h) * gamma(k) for k in range(1, h))) / T
        flag = "bartlett_variance_fallback"
    if var <= 0:
        return dict(T=T, h=h, status="non-positive variance")
    dm = dbar / math.sqrt(var)
    corr = (T + 1 - 2 * h + h * (h - 1) / T) / T
    if corr <= 0:
        return dict(T=T, h=h, dm=dm, status="h too large for T (HLN factor <= 0)")
    hln = dm * math.sqrt(corr)
    return dict(T=T, h=h, mean_loss_diff=dbar, dm=dm, hln_stat=hln,
                p_two_sided=2 * (1 - t_cdf(abs(hln), T - 1)), p_model_better=t_cdf(hln, T - 1),
                wins=int((d < 0).sum()), status=flag or "ok")


def per_origin_loss(df, m, kind):
    e = df[m] - df["actual"]
    s = e.abs() if kind == "abs" else e ** 2
    return s.groupby(df["origin"]).mean().sort_index()


def dm_tables(df, model, others, h, alpha):
    rows = []
    for o in others:
        for kind, label in (("abs", "absolute-error loss (MAE)"), ("sq", "squared-error loss (RMSE)")):
            d = (per_origin_loss(df, model, kind) - per_origin_loss(df, o, kind)).values
            r = dm_hln(d, h)
            r.update(model=model, versus=o, loss=label, significant_at_alpha=bool(r.get("p_model_better", 1) < alpha))
            rows.append(r)
    cols = ["model", "versus", "loss", "T", "h", "mean_loss_diff", "wins", "dm", "hln_stat", "p_two_sided",
            "p_model_better", "significant_at_alpha", "status"]
    return pd.DataFrame(rows).reindex(columns=cols)


# ----------------------------------------------------------------------------
# MOD-11 Mincer-Zarnowitz + bias breakdown
# ----------------------------------------------------------------------------
def mz_regression(y, x, cluster):
    y, x = np.asarray(y, float), np.asarray(x, float)
    X = np.column_stack([np.ones(len(x)), x])
    try:
        xtx_inv = np.linalg.inv(X.T @ X)
    except np.linalg.LinAlgError:
        return None
    beta = xtx_inv @ X.T @ y
    u = y - X @ beta
    codes, uniq = pd.factorize(pd.Series(cluster))
    G, N, K = len(uniq), len(y), 2
    S = np.zeros((G, K))
    np.add.at(S, codes, X * u[:, None])
    c = (G / (G - 1)) * ((N - 1) / (N - K)) if G > 1 else 1.0
    V = c * xtx_inv @ (S.T @ S) @ xtx_inv
    se = np.sqrt(np.clip(np.diag(V), 0, None))
    dlt = beta - np.array([0.0, 1.0])
    try:
        W = float(dlt @ np.linalg.solve(V, dlt))
    except np.linalg.LinAlgError:
        W = float("nan")
    ss_res, ss_tot = float((u ** 2).sum()), float(((y - y.mean()) ** 2).sum())
    return dict(intercept=beta[0], se_intercept=se[0], slope=beta[1], se_slope=se[1], wald_F=W / 2,
                p_joint_a0_b1=f_sf_2(W / 2, G - 1) if np.isfinite(W) else float("nan"), r2=1 - ss_res / ss_tot,
                n=N, clusters=G)


def mz_tables(df, methods, a):
    rows = []
    for m in methods:
        for space in ("level (kWh)", "relative (divided by " + a.scale_col + ")"):
            if space.startswith("relative"):
                if m == a.scale_col:
                    continue                       # forecast / scale is constant -> regression is degenerate
                y, x = df["actual"] / df[a.scale_col], df[m] / df[a.scale_col]
            else:
                y, x = df["actual"], df[m]
            r = mz_regression(y.values, x.values, df["origin"].values)
            if r:
                rows.append(dict(method=m, space=space, **r))
    return pd.DataFrame(rows)


def bias_breakdown(df, methods, a, level):
    d = df.copy()
    d["usage_tercile"] = pd.qcut(d[a.scale_col], 3, labels=["low", "mid", "high"], duplicates="drop")
    groups = [("usage_tercile", "usage_tercile")]
    if level == "unit":
        d["tower"] = d[a.id_col].map(tower_of)
        groups.append(("tower", "tower"))
    rows = []
    for gname, col in groups:
        for key, g in d.groupby(col, observed=True):
            for m in methods:
                e = g[m] - g["actual"]
                rows.append(dict(group=gname, value=str(key), method=m, n=len(g), mean_actual=g["actual"].mean(),
                                 bias=e.mean(), nmbe_pct=100 * e.mean() / g["actual"].mean(), mae=e.abs().mean()))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# MOD-10 interval coverage
# ----------------------------------------------------------------------------
def walkforward_intervals(df, m, a):
    lo_q, hi_q = (1 - a.coverage) / 2, 1 - (1 - a.coverage) / 2
    rel = (df["actual"] - df[m]) / df[a.scale_col]       # same normalisation as forecast_job (error / base)
    origins = sorted(df["origin"].unique())
    parts = []
    for i, o in enumerate(origins):
        usable = [s for s in origins[:i] if (o - s).days >= a.horizon]     # target of s must have ended before o
        usable = usable[-a.wf_origins:]
        if len(usable) < a.wf_min_origins:
            continue
        qlo, qhi = np.quantile(rel[df["origin"].isin(usable)], [lo_q, hi_q])
        cur = df[df["origin"] == o]
        v, b = cur[m].values, cur[a.scale_col].values
        lower = np.minimum(np.maximum(v + qlo * b, 0), v)
        upper = np.maximum(v + qhi * b, v)
        parts.append(pd.DataFrame({"origin": o, "id": cur[a.id_col].values, "actual": cur["actual"].values,
                                   "lower": lower, "upper": upper}))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def coverage_rows(I, label, a, level, rng):
    def per_origin(frame):
        inside = (frame["actual"] >= frame["lower"]) & (frame["actual"] <= frame["upper"])
        return inside
    out = []

    def add(name, series_in, below, above, width):
        lo, hi = boot_mean_ci(series_in.values, a.boot, rng)
        out.append(dict(method=label, level=name, nominal=a.coverage, n_origins=int(len(series_in)),
                        coverage=float(series_in.mean()), ci95_lo=lo, ci95_hi=hi, below_lower=float(below.mean()),
                        above_upper=float(above.mean()), rel_width=float(width.mean())))

    # unit (or tower when --level tower): coverage per origin = share of series inside the band
    g = I.groupby("origin")
    inside = g.apply(lambda f: per_origin(f).mean(), include_groups=False) if _pd_has_include_groups() else \
        g.apply(lambda f: per_origin(f).mean())
    below = g.apply(lambda f: (f["actual"] < f["lower"]).mean(), **_ig())
    above = g.apply(lambda f: (f["actual"] > f["upper"]).mean(), **_ig())
    width = g.apply(lambda f: (f["upper"] - f["lower"]).sum() / f["actual"].sum(), **_ig())
    add("unit" if level == "unit" else "tower", inside, below, above, width)
    if level == "unit":
        T = I.assign(tower=I["id"].map(tower_of)).groupby(["origin", "tower"])[["actual", "lower", "upper"]].sum().reset_index()
        gi = T.groupby("origin")
        add("tower (sum of unit bounds)", gi.apply(lambda f: per_origin(f).mean(), **_ig()),
            gi.apply(lambda f: (f["actual"] < f["lower"]).mean(), **_ig()),
            gi.apply(lambda f: (f["actual"] > f["upper"]).mean(), **_ig()),
            gi.apply(lambda f: (f["upper"] - f["lower"]).sum() / f["actual"].sum(), **_ig()))
    B = I.groupby("origin")[["actual", "lower", "upper"]].sum()
    add("building (sum of bounds)", per_origin(B).astype(float), (B["actual"] < B["lower"]).astype(float),
        (B["actual"] > B["upper"]).astype(float), (B["upper"] - B["lower"]) / B["actual"])
    return out


def _pd_has_include_groups():
    try:
        pd.DataFrame({"a": [1]}).groupby("a").apply(lambda f: 1, include_groups=False)
        return True
    except TypeError:
        return False


def _ig():
    return {"include_groups": False} if _pd_has_include_groups() else {}


def interval_tables(df, model, a, level, rng):
    rows, mode = [], a.interval_mode
    if mode == "auto":
        mode = "columns" if (a.lower_col in df.columns and a.upper_col in df.columns) else "walkforward"
    if mode == "off":
        return pd.DataFrame(), "off"
    if mode == "columns":
        I = pd.DataFrame({"origin": df["origin"], "id": df[a.id_col], "actual": df["actual"],
                          "lower": df[a.lower_col], "upper": df[a.upper_col]})
        rows += coverage_rows(I, model + " (columns)", a, level, rng)
    else:
        for m in dict.fromkeys([model, a.primary_baseline]):
            I = walkforward_intervals(df, m, a)
            if len(I):
                rows += coverage_rows(I, f"{m} (walk-forward)", a, level, rng)
    return pd.DataFrame(rows), mode


# ----------------------------------------------------------------------------
# MOD-05 reproducibility
# ----------------------------------------------------------------------------
def reproducibility(df, a):
    ref = pd.read_csv(a.reference)
    if a.model_col not in ref.columns or "actual" not in ref.columns:
        return [("MOD-05", "reproducibility vs reference", "FAIL", f"reference CSV lacks '{a.model_col}'/'actual'")]
    def mm(x):
        e = x[a.model_col] - x["actual"]
        return e.abs().mean(), float(np.sqrt((e ** 2).mean()))
    (m1, r1), (m2, r2) = mm(df), mm(ref)
    dm_, dr_ = 100 * (m2 - m1) / m1, 100 * (r2 - r1) / r1
    ok = abs(dm_) <= a.tolerance and abs(dr_) <= a.tolerance
    return [("MOD-05", f"re-run reproduces MAE and RMSE within {a.tolerance}%", "PASS" if ok else "FAIL",
             f"MAE {m1:.3f} -> {m2:.3f} ({dm_:+.2f}%), RMSE {r1:.3f} -> {r2:.3f} ({dr_:+.2f}%)")]


# ----------------------------------------------------------------------------
# MOD-12 self-tests against the real model code (nusawatt_lstm_v2.py)
# ----------------------------------------------------------------------------
def selftests(a):
    sys.path.insert(0, a.lib_dir)
    try:
        LIB = importlib.import_module(a.lib)
    except Exception as exc:
        return [("MOD-12", "import model library", "ERROR", f"{a.lib} from '{a.lib_dir}': {exc}")]
    res = []
    rng = np.random.default_rng(1)
    L, H, T, n = 30, 7, 200, 12
    idx = pd.date_range("2025-09-01", periods=T, freq="D")
    dow = idx.dayofweek.values
    U = (10 + 3 * (dow >= 5)[:, None] + rng.normal(0, 1, (T, n)) + rng.uniform(0, 6, n)[None, :]).clip(0.1)
    ns = argparse.Namespace(no_calendar=False, holidays="none", holiday_csv="")
    cal, _ = LIB.calendar_features(idx, ns, extra_days=H)
    o, mink = 150, 1.0

    def same(S0, S1, keys):
        return all(np.array_equal(S0[k], S1[k]) for k in keys)

    keys = ["seq", "fut", "stat", "base", "unit"]
    try:
        S0 = LIB.samples_at(U, cal, [o], L, H, False, mink)
        U2 = U.copy()
        U2[o:] = rng.uniform(500, 900, U2[o:].shape)                      # change everything from the origin onward
        S1 = LIB.samples_at(U2, cal, [o], L, H, False, mink)
        res.append(("MOD-12a", "inputs unchanged when data from the origin onward is altered (forecast mode)",
                    "PASS" if same(S0, S1, keys) else "FAIL", "seq, fut, stat, base, unit compared"))
        St = LIB.samples_at(U, cal, [o], L, H, True, mink)
        U3 = U.copy()
        U3[o:o + H] = rng.uniform(500, 900, U3[o:o + H].shape)           # change the TARGET days only
        S3 = LIB.samples_at(U3, cal, [o], L, H, True, mink)
        res.append(("MOD-12b", "inputs unchanged when only the target days are altered (training mode)",
                    "PASS" if same(St, S3, keys) and not np.array_equal(St["actual"], S3["actual"]) else "FAIL",
                    "targets change, inputs do not"))
        U4 = U.copy()
        U4[o - 1] += 5.0                                                  # CONTROL: last history day must matter
        S4 = LIB.samples_at(U4, cal, [o], L, H, False, mink)
        res.append(("MOD-12c", "control: altering the last history day DOES change the inputs (test can fail)",
                    "PASS" if not same(S0, S4, ["seq", "base"]) else "FAIL", "sensitivity check of this test"))
        U5 = U.copy()
        U5[o - 5, 3] = np.nan
        S5 = LIB.samples_at(U5, cal, [o], L, H, False, mink)
        res.append(("MOD-12d", "a unit with a gap inside the lookback window is excluded (no interpolation)",
                    "PASS" if 3 not in set(S5["unit"]) and 3 in set(S0["unit"]) else "FAIL", "unit 3 given a NaN day"))
        base_ref = U[o - 28:o].mean(axis=0)[S0["unit"]]
        res.append(("MOD-12e", "base equals the mean of the last 28 days, so base x horizon is the avg4 baseline",
                    "PASS" if np.allclose(S0["base"], base_ref, rtol=1e-5) else "FAIL", "max abs diff "
                    f"{float(np.abs(S0['base'] - base_ref).max()):.2e}"))
        U6 = U.copy()
        U6[o - 28:o, 5] = 0.2
        S6 = LIB.samples_at(U6, cal, [o], L, H, False, 1.0)
        res.append(("MOD-12f", "units below --min-kwh are excluded", "PASS" if 5 not in set(S6["unit"]) else "FAIL",
                    "unit 5 set to 0.2 kWh/day"))
    except Exception as exc:
        res.append(("MOD-12", "samples_at tests", "ERROR", f"{type(exc).__name__}: {exc}"))
    try:
        import tensorflow  # noqa: F401
    except Exception:
        res.append(("MOD-12g", "untrained network outputs ratio 1 (reproduces avg4)", "SKIPPED", "TensorFlow not installed"))
        return res
    try:
        c_seq, c_fut = int(S0["seq"].shape[2]), int(S0["fut"].shape[1] // H)
        m = LIB.build_model(L, H, c_seq, c_fut, 1e-3)
        X = [rng.normal(size=(8, L, c_seq)).astype("float32"), rng.normal(size=(8, H * c_fut)).astype("float32"),
             rng.normal(size=(8, 1)).astype("float32")]
        out = m.predict(X, verbose=0)
        res.append(("MOD-12g", "untrained network outputs ratio 1 (reproduces avg4)",
                    "PASS" if np.allclose(out, 1.0, atol=1e-6) else "FAIL", f"max |out-1| = {float(np.abs(out - 1).max()):.2e}"))
    except Exception as exc:
        res.append(("MOD-12g", "untrained network outputs ratio 1", "ERROR", f"{type(exc).__name__}: {exc}"))
    return res


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="NusaWatt model tests (backtest-based)")
    p.add_argument("--backtest", default="", help="backtest_unit_origin.csv from nusawatt_lstm_v2.py")
    p.add_argument("--label", default="", help="name for the report, e.g. 'Electricity 7-day'")
    p.add_argument("--model-col", default="lstm")
    p.add_argument("--baselines", default="avg4,last_week", help="comma list of baseline columns in the CSV")
    p.add_argument("--primary-baseline", default="avg4", help="baseline used for the verdicts")
    p.add_argument("--scale-col", default="avg4", help="column equal to base x horizon (normaliser for relative tests)")
    p.add_argument("--extra", action="append", help="NAME=PATH of another backtest CSV whose NAME column is added (e.g. ridge=...)")
    p.add_argument("--id-col", default="unit")
    p.add_argument("--level", choices=["unit", "tower"], default="unit", help="what the id column holds (tower for water)")
    p.add_argument("--horizon", type=int, default=7, help="forecast horizon in days")
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--dm-h", type=int, default=0, help="DM horizon parameter (0 = ceil(horizon / origin spacing))")
    p.add_argument("--interval-mode", choices=["auto", "columns", "walkforward", "off"], default="auto")
    p.add_argument("--lower-col", default="lower")
    p.add_argument("--upper-col", default="upper")
    p.add_argument("--coverage", type=float, default=0.8, help="nominal coverage of the band")
    p.add_argument("--wf-origins", type=int, default=4, help="walk-forward: earlier origins used to build the band")
    p.add_argument("--wf-min-origins", type=int, default=2)
    p.add_argument("--cov-tol", type=float, default=0.05, help="coverage tolerance for the label (default 5 points; justify your own)")
    p.add_argument("--reference", default="", help="another backtest CSV (re-run of the frozen configuration) for MOD-05")
    p.add_argument("--tolerance", type=float, default=2.0, help="MOD-05 tolerance in percent")
    p.add_argument("--boot", type=int, default=5000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--selftest", action="store_true", help="run MOD-12 against the model library and exit")
    p.add_argument("--lib", default="app.nusawatt_lstm_v2")
    p.add_argument("--lib-dir", default=".")
    p.add_argument("--strict", action="store_true", help="exit with code 1 if any check FAILS")
    p.add_argument("--out", default="experiments/model_tests")
    return p.parse_args()


def verdict_rows(a, rel, dm, ivt, mode, h, T):
    rows = []
    r = rel[(rel["baseline"] == a.primary_baseline) & rel["measure"].str.startswith("pooled")]
    if len(r):
        r = r.iloc[0]
        ok = r["rel_mae"] < 1 and r["ci95_hi"] < 1
        rows.append(("MOD-08", f"relative MAE vs {a.primary_baseline}: pooled ratio < 1 and 95% CI entirely below 1",
                     "PASS" if ok else "FAIL",
                     f"{r['rel_mae']:.3f} [{r['ci95_lo']:.3f}, {r['ci95_hi']:.3f}] ({r['improvement_pct']:+.1f}% vs baseline)"))
    d = dm[(dm["versus"] == a.primary_baseline) & dm["loss"].str.startswith("absolute")]
    if len(d):
        d = d.iloc[0]
        if d["status"] in ("ok", "bartlett_variance_fallback") and pd.notna(d["p_model_better"]):
            st = "PASS" if d["p_model_better"] < a.alpha else "NOT SIGNIFICANT"
            if h > 1:
                st += " (INDICATIVE: overlapping targets)"
            rows.append(("MOD-09", f"Diebold-Mariano (HLN) vs {a.primary_baseline}, absolute-error loss, one-sided p < {a.alpha}",
                         st, f"T={int(d['T'])} h={int(d['h'])} stat={d['hln_stat']:.2f} p={d['p_model_better']:.4f} wins={int(d['wins'])}/{int(d['T'])}"))
        else:
            rows.append(("MOD-09", "Diebold-Mariano (HLN)", "FAIL", str(d["status"])))
    if ivt is not None and len(ivt):
        for _, c in ivt.iterrows():
            diff = c["coverage"] - c["nominal"]
            lab = "PASS" if abs(diff) <= a.cov_tol else ("OVER (conservative band)" if diff > 0 else "UNDER (band too narrow)")
            rows.append(("MOD-10", f"{c['method']} {c['level']}: coverage within {100 * a.cov_tol:.0f} points of {100 * a.coverage:.0f}%",
                         lab, f"{100 * c['coverage']:.1f}% [{100 * c['ci95_lo']:.1f}, {100 * c['ci95_hi']:.1f}], "
                              f"below {100 * c['below_lower']:.1f}% / above {100 * c['above_upper']:.1f}%"))
    return rows


def main():
    a = parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.selftest:
        res = selftests(a)
        S = pd.DataFrame(res, columns=["id", "check", "status", "detail"])
        S.to_csv(f"{a.out}/selftest.csv", index=False)
        print(S.to_string(index=False))
        sys.exit(1 if (a.strict and (S["status"].isin(["FAIL", "ERROR"])).any()) else 0)
    if not a.backtest:
        sys.exit("--backtest is required (or use --selftest)")
    rng = np.random.default_rng(a.seed)
    df, baselines, notes = load_backtest(a)
    if _st is None:
        notes.append("WARNING: scipy not installed; p-values use normal/chi-square approximations (too optimistic). pip install scipy before quoting them.")
        print(notes[-1])
    extras = [s.split("=", 1)[0] for s in (a.extra or [])]
    methods = [a.model_col] + baselines + extras
    others = baselines + extras
    spacing = float(np.median(np.diff(np.sort(df["origin"].unique())).astype("timedelta64[D]").astype(float))) \
        if df["origin"].nunique() > 1 else 7.0
    h = a.dm_h or max(1, int(math.ceil(a.horizon / spacing)))
    label = a.label or f"{a.model_col} backtest"

    checks = [(i, c, s, d) for i, c, s, d in data_checks(df, a, baselines)]
    H = headline(df, methods, a)
    SP = seed_spread(df, a)
    rel = pd.DataFrame([r for b in others for r in relative_mae(df, a.model_col, b, a, rng)])
    DM = dm_tables(df, a.model_col, others, h, a.alpha)
    MZ = mz_tables(df, methods, a)
    BB = bias_breakdown(df, methods, a, a.level)
    IV, mode = interval_tables(df, a.model_col, a, a.level, rng)
    checks += verdict_rows(a, rel, DM, IV, mode, h, df["origin"].nunique())
    if a.reference:
        checks += reproducibility(df, a)
    C = pd.DataFrame(checks, columns=["id", "check", "status", "detail"])

    for name, t in (("headline_metrics", H), ("seed_spread", SP), ("relative_mae", rel), ("dm_tests", DM),
                    ("mincer_zarnowitz", MZ), ("bias_breakdown", BB), ("interval_coverage", IV), ("checks", C)):
        if len(t):
            t.to_csv(f"{a.out}/{name}.csv", index=False)

    spacing_txt = f"{spacing:.0f}-day origin spacing"
    md = [f"# Model tests: {label}\n",
          f"Backtest: `{a.backtest}` | {df['origin'].nunique()} origins ({df['origin'].min().date()} to {df['origin'].max().date()}), "
          f"{len(df):,} {a.level}-origin rows, horizon {a.horizon} days, {spacing_txt}. Error sign: forecast - actual.\n"]
    md += [f"_{n}_\n" for n in notes]
    md += ["\n## Checks and verdicts\n", md_table(C), "\n## Headline metrics (horizon sums)\n", md_table(H.round(3)),
           "\n## MOD-08 Relative MAE (below 1 = better than the baseline)\n", md_table(rel.round(3)),
           f"\n## MOD-09 Diebold-Mariano with HLN correction (h = {h}; H1: model loss lower)\n", md_table(DM.round(4)),
           "\n## MOD-11 Mincer-Zarnowitz (actual on forecast; ideal intercept 0, slope 1)\n", md_table(MZ.round(4)),
           "\n## Bias breakdown\n", md_table(BB.round(3)),
           f"\n## MOD-10 Interval coverage (mode: {mode}; nominal {a.coverage:.0%})\n", md_table(IV.round(3))]
    if len(SP):
        md += ["\n## Seed spread\n", md_table(SP.round(3))]
    md += ["\n## Limits\n",
           "- Walk-forward coverage is a proxy on horizon sums, not the daily bands served by the API.\n"
           "- DM and bootstrap treat origins as the unit of replication; with few origins results are indicative.\n"
           "- Tower and building bands sum unit bounds, which tends to over-cover.\n"]
    with open(f"{a.out}/model_test_report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    with open(f"{a.out}/model_test_summary.json", "w", encoding="utf-8") as f:
        json.dump(dict(label=label, args=vars(a), origins=int(df["origin"].nunique()), rows=int(len(df)), dm_h=h,
                       checks=C.to_dict(orient="records")), f, indent=2, default=str)

    pd.set_option("display.width", 200)
    print(f"== {label} | {df['origin'].nunique()} origins | {len(df):,} rows | h={h} | interval mode: {mode} ==")
    print("\n-- Checks and verdicts --")
    print(C.to_string(index=False, max_colwidth=95))
    print("\n-- Headline --")
    print(H.drop(columns=["n_rows", "n_origins"]).round(3).to_string(index=False))
    print("\n-- Relative MAE --")
    print(rel.drop(columns=["note"]).round(3).to_string(index=False))
    print("\n-- Diebold-Mariano (HLN) --")
    print(DM.drop(columns=["dm"]).round(4).to_string(index=False))
    print("\n-- Mincer-Zarnowitz --")
    print(MZ.round(4).to_string(index=False))
    if len(IV):
        print("\n-- Interval coverage --")
        print(IV.round(3).to_string(index=False))
    print(f"\nWritten to {a.out}/ (model_test_report.md, model_test_summary.json and CSV tables)")
    sys.exit(1 if (a.strict and C["status"].str.startswith(("FAIL", "ERROR")).any()) else 0)


if __name__ == "__main__":
    main()
