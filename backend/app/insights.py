"""Automatic, deterministic insights: KPIs, trend, breakdowns, anomalies, forecast."""
from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd

from .store import store
from .utils import fmt_number, humanize, is_additive, is_money, to_jsonable

PRIMARY_PREFS = ("revenue", "sales", "amount", "profit", "spend")
KPI_PREFS = ("revenue", "sales", "amount", "profit", "spend", "conversions", "units", "quantity",
             "leads", "clicks", "csat", "nps", "score", "cost")
FORECAST_PERIODS = 30


# ---------------------------------------------------------------- roles
def roles(meta: dict[str, Any], df: pd.DataFrame | None = None) -> dict[str, Any]:
    cols = meta["columns"]
    dates = [c["name"] for c in cols if c["semantic"] == "date"]
    metrics = [c["name"] for c in cols if c["semantic"] == "metric"]
    dims = [c["name"] for c in cols if c["semantic"] == "dimension"]
    ids = [c["name"] for c in cols if c["semantic"] == "id"]
    bools = [c["name"] for c in cols if c["dtype"] == "boolean"]
    primary = None
    for pref in PRIMARY_PREFS:
        for m in metrics:
            if pref in m.lower():
                primary = m
                break
        if primary:
            break
    if primary is None and metrics:
        primary = metrics[0]
    dim_card: dict[str, int] = {}
    if df is not None:
        for d in dims:
            dim_card[d] = int(df[d].nunique(dropna=True))
    return {"date": dates[0] if dates else None, "dates": dates, "metrics": metrics, "dims": dims,
            "ids": ids, "bools": bools, "primary": primary, "dim_card": dim_card}


def entity_label(meta: dict[str, Any]) -> str:
    for c in meta["columns"]:
        n = c["name"].lower()
        if c["semantic"] == "id" and n.endswith("_id"):
            base = n[:-3].replace("_", " ").strip()
            if base:
                return base + ("es" if base.endswith(("s", "x", "ch", "sh")) else "s")
    return "records"


def agg_fn(metric: str) -> str:
    return "sum" if is_additive(metric) else "mean"


def metric_label(metric: str) -> str:
    return humanize(metric)


def metric_kind(metric: str) -> str:
    return "currency" if is_money(metric) and "pct" not in metric.lower() else "number"


# ---------------------------------------------------------------- time series helpers
def detect_freq(dates: pd.Series) -> str:
    d = pd.to_datetime(dates.dropna()).dt.normalize().drop_duplicates().sort_values()
    if len(d) < 3:
        return "D"
    med = d.diff().dropna().dt.days.median()
    if med >= 25:
        return "MS"
    if med >= 6:
        return "W"
    return "D"


def series(df: pd.DataFrame, date_col: str, metric: str | None, how: str, freq: str) -> pd.Series:
    d = pd.to_datetime(df[date_col])
    if metric is None:
        vals = pd.Series(1, index=df.index)
        how = "sum"
    elif metric in df and pd.api.types.is_bool_dtype(df[metric]):
        vals = df[metric].astype(float)
    else:
        vals = pd.to_numeric(df[metric], errors="coerce")
    tmp = pd.DataFrame({"d": d.dt.normalize(), "v": vals}).dropna(subset=["d"])
    if tmp.empty:
        return pd.Series(dtype=float)
    if freq == "W":
        # bucket by ISO week start (Monday); natively weekly data keeps one point per week
        key = tmp["d"] - pd.to_timedelta(tmp["d"].dt.dayofweek, unit="D")
        g = tmp.groupby(key)["v"]
        s = g.sum() if how == "sum" else g.mean()
        full = pd.date_range(s.index.min(), s.index.max(), freq="7D")
        s = s.reindex(full)
    else:
        grouped = tmp.set_index("d")["v"].resample(freq)
        s = grouped.sum(min_count=0) if how == "sum" else grouped.mean()
    if how == "sum":
        s = s.fillna(0.0)
    else:
        s = s.interpolate(limit_direction="both")
    return s.astype(float)


def window_value(df: pd.DataFrame, date_col: str, metric: str | None, how: str,
                 start: pd.Timestamp, end: pd.Timestamp) -> float | None:
    d = pd.to_datetime(df[date_col])
    mask = (d > start) & (d <= end)
    if not mask.any():
        return None
    if metric is None:
        return float(mask.sum())
    v = df.loc[mask, metric]
    v = v.astype(float) if pd.api.types.is_bool_dtype(v) else pd.to_numeric(v, errors="coerce")
    return float(v.sum() if how == "sum" else v.mean())


def pct_change(cur: float | None, prev: float | None) -> float | None:
    if cur is None or prev is None or prev == 0 or np.isnan(prev) or np.isnan(cur):
        return None
    return round((cur - prev) / abs(prev) * 100, 2)


# ---------------------------------------------------------------- anomalies & forecast
def find_anomalies(s: pd.Series, metric: str, threshold: float = 3.5, max_n: int = 8) -> list[dict[str, Any]]:
    if len(s) < 14:
        return []
    window = 15 if len(s) > 60 else 7
    expected = s.rolling(window, center=True, min_periods=3).median()
    resid = s - expected
    mad = float(np.median(np.abs(resid.dropna() - np.median(resid.dropna()))))
    scale = 1.4826 * mad
    if scale <= 0 or np.isnan(scale):
        scale = float(resid.std()) or 1.0
    z = resid / scale
    out = []
    for idx in z.abs().sort_values(ascending=False).index:
        zi = float(z[idx])
        if abs(zi) < threshold or len(out) >= max_n:
            break
        out.append({
            "date": idx, "metric": metric, "value": round(float(s[idx]), 2),
            "expected": round(float(expected[idx]), 2), "zscore": round(zi, 2),
            "direction": "spike" if zi > 0 else "dip",
        })
    out.sort(key=lambda a: a["date"])
    return out


def forecast(s: pd.Series, freq: str, metric: str, periods: int = FORECAST_PERIODS) -> dict[str, Any] | None:
    s = s.dropna()
    if len(s) < 14:
        return None
    season = {"D": 7, "W": 52, "MS": 12}.get(freq, 7)
    use_season = len(s) >= 2 * season + 2 and season <= 60 and not (freq == "W")
    nonneg = bool((s >= 0).all())
    step = {"D": pd.Timedelta(days=1), "W": pd.Timedelta(weeks=1)}.get(freq)
    if step is not None:
        future_idx = [s.index[-1] + step * (i + 1) for i in range(periods)]
    else:
        future_idx = list(pd.date_range(s.index[-1], periods=periods + 1, freq="MS")[1:])
    method = "holt-winters"
    try:
        from statsmodels.tsa.holtwinters import ExponentialSmoothing

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = ExponentialSmoothing(
                s.values, trend="add", damped_trend=True,
                seasonal="add" if use_season else None,
                seasonal_periods=season if use_season else None,
                initialization_method="estimated",
            ).fit(optimized=True)
            fc = np.asarray(model.forecast(periods), dtype=float)
            resid = s.values - np.asarray(model.fittedvalues, dtype=float)
        if not np.all(np.isfinite(fc)):
            raise ValueError("non-finite forecast")
    except Exception:
        method = "linear+seasonal"
        y = s.values.astype(float)
        x = np.arange(len(y))
        coef = np.polyfit(x, y, 1)
        trend_fit = np.polyval(coef, x)
        seas = np.zeros(season)
        if use_season:
            detr = y - trend_fit
            for k in range(season):
                seas[k] = np.mean(detr[k::season])
        fx = np.arange(len(y), len(y) + periods)
        fc = np.polyval(coef, fx) + np.array([seas[i % season] for i in fx])
        resid = y - (trend_fit + np.array([seas[i % season] for i in x]))
    sigma = float(np.nanstd(resid)) if len(resid) else 0.0
    rows = []
    for i, (d, v) in enumerate(zip(future_idx, fc)):
        width = 1.96 * sigma * np.sqrt(1 + i / max(len(s), 1) * 4)
        lo, hi = v - width, v + width
        if nonneg:
            v, lo = max(v, 0.0), max(lo, 0.0)
        rows.append({"date": d, "value": round(float(v), 2), "lower": round(float(lo), 2), "upper": round(float(hi), 2)})
    hist_n = 120 if freq == "D" else 104
    history = [{"date": d, "value": round(float(v), 2)} for d, v in s.tail(hist_n).items()]
    return {"metric": metric, "history": history, "forecast": rows, "method": method}


# ---------------------------------------------------------------- main
def _kpi_candidates(r: dict[str, Any], df: pd.DataFrame) -> list[dict[str, Any]]:
    metrics = r["metrics"]
    cands: list[dict[str, Any]] = []
    if r["primary"]:
        cands.append({"metric": r["primary"]})
    if r["ids"]:
        cands.append({"metric": None})  # row count (only meaningful when rows are entities)

    def rank(m: str) -> tuple[int, int]:
        ml = m.lower()
        for i, p in enumerate(KPI_PREFS):
            if p in ml:
                return (0, i)
        return (1 if is_additive(m) else 2, metrics.index(m))

    for m in sorted([m for m in metrics if m != r["primary"]], key=rank):
        if any(w in m.lower() for w in ("price", "discount", "impressions", "cost")) and len(metrics) > 4:
            continue
        cands.append({"metric": m})
    lower = {m.lower(): m for m in metrics}
    derived = []
    rev = next((m for m in metrics if "revenue" in m.lower() or "sales" in m.lower()), None)
    if rev and "profit" in lower:
        derived.append({"derived": "margin", "label": "Profit Margin", "num": lower["profit"], "den": rev})
    if "conversions" in lower and "clicks" in lower:
        derived.append({"derived": "ratio", "label": "Conversion Rate", "num": lower["conversions"], "den": lower["clicks"]})
    for b in r["bools"][:1]:
        derived.append({"derived": "bool", "label": f"{humanize(b)} Rate", "col": b})
    # put derived KPIs at position 3 so they make the cut
    return cands[:3] + derived + cands[3:]


def _kpi(df: pd.DataFrame, r: dict[str, Any], cand: dict[str, Any], entity: str,
         end: pd.Timestamp | None) -> dict[str, Any]:
    date_col = r["date"]
    has_time = date_col is not None and end is not None
    start, prev_start = (end - pd.Timedelta(days=30), end - pd.Timedelta(days=60)) if has_time else (None, None)

    def compute(sub: pd.DataFrame) -> float | None:
        if sub.empty:
            return None
        if "derived" in cand:
            if cand["derived"] == "bool":
                return float(sub[cand["col"]].astype(float).mean() * 100)
            den = pd.to_numeric(sub[cand["den"]], errors="coerce").sum()
            return float(pd.to_numeric(sub[cand["num"]], errors="coerce").sum() / den * 100) if den else None
        m = cand["metric"]
        if m is None:
            return float(len(sub))
        v = pd.to_numeric(sub[m], errors="coerce")
        return float(v.sum() if agg_fn(m) == "sum" else v.mean())

    if "derived" in cand:
        label, fmt = cand["label"], "percent"
    else:
        m = cand["metric"]
        label = humanize(entity) if m is None else (humanize(m) if agg_fn(m) == "sum" else f"Avg {humanize(m)}")
        fmt = "number" if m is None else metric_kind(m)
    if has_time:
        d = pd.to_datetime(df[date_col])
        cur = compute(df[(d > start) & (d <= end)])
        prev = compute(df[(d > prev_start) & (d <= start)])
        value = cur
        change = pct_change(cur, prev)
        period = "Last 30 days"
        # sparkline: weekly values over the last 12 weeks
        spark = []
        for k in range(12, 0, -1):
            w_end = end - pd.Timedelta(days=7 * (k - 1))
            w_start = w_end - pd.Timedelta(days=7)
            v = compute(df[(d > w_start) & (d <= w_end)])
            spark.append(round(v, 4) if v is not None else 0.0)
    else:
        value = compute(df)
        change = None
        period = "All time"
        spark = []
    return {"label": label, "value": round(value, 4) if value is not None else 0.0, "format": fmt,
            "change_pct": change, "period": period, "sparkline": spark}


def compute_insights(ds_id: str) -> dict[str, Any]:
    meta = store.get(ds_id)
    cache = store.cache(ds_id)
    if "insights" in cache:
        return cache["insights"]
    df = store.df(ds_id)
    r = roles(meta, df)
    entity = entity_label(meta)
    date_col, primary = r["date"], r["primary"]
    end = None
    if date_col is not None:
        dts = pd.to_datetime(df[date_col]).dropna()
        if dts.empty:
            date_col = None
            r["date"] = None
        else:
            end = dts.max().normalize() + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)

    # KPIs
    kpis = []
    for cand in _kpi_candidates(r, df):
        if len(kpis) >= 4:
            break
        try:
            kpis.append(_kpi(df, r, cand, entity, end))
        except Exception:
            continue

    trend = None
    anomalies: list[dict[str, Any]] = []
    fc = None
    s = None
    freq = "D"
    how = agg_fn(primary) if primary else "sum"
    if date_col is not None:
        freq = detect_freq(df[date_col])
        s = series(df, date_col, primary, how, freq)
        if len(s) > 0:
            trend = {"date_column": date_col, "metric": primary or "count",
                     "series": [{"date": d, "value": round(float(v), 2)} for d, v in s.items()]}
            anom_s = s
            if freq == "D" and len(s) > 60:
                med = float(np.median(np.abs(s - s.rolling(15, center=True, min_periods=3).median()).dropna()))
                level = float(np.median(np.abs(s))) or 1.0
                if med / level > 0.25:  # too noisy day-to-day: detect on weekly totals/means
                    anom_s = series(df, date_col, primary, how, "W")
            anomalies = find_anomalies(anom_s, primary or "count", threshold=3.5 if how == "sum" else 4.0)
            fc = forecast(s, freq, primary or "count")

    # breakdowns: top 3 dimensions (by usefulness: 2..30 categories)
    breakdowns = []
    dims = [d for d in r["dims"] if 2 <= r["dim_card"].get(d, 0) <= 30 and d not in r["bools"]]
    dims.sort(key=lambda d: (r["dim_card"][d] > 12, r["dims"].index(d)))
    for dim in dims[:3]:
        if primary:
            g = df.groupby(dim, dropna=True)[primary]
            agg = g.sum() if how == "sum" else g.mean()
        else:
            agg = df.groupby(dim, dropna=True).size()
        agg = agg.sort_values(ascending=False).head(12)
        breakdowns.append({"dimension": dim, "metric": primary or "count",
                           "data": [{"label": str(k), "value": round(float(v), 2)} for k, v in agg.items()]})

    highlights = _highlights(df, meta, r, entity, kpis, breakdowns, anomalies, fc, date_col, primary, how)
    result = to_jsonable({"kpis": kpis, "trend": trend, "breakdowns": breakdowns, "anomalies": anomalies,
                          "forecast": fc, "highlights": highlights})
    cache["insights"] = result
    return result


def _highlights(df, meta, r, entity, kpis, breakdowns, anomalies, fc, date_col, primary, how) -> list[str]:
    out: list[str] = []
    kind = metric_kind(primary) if primary else "number"
    plabel = humanize(primary).lower() if primary else entity
    span = ""
    if date_col:
        d = pd.to_datetime(df[date_col]).dropna()
        span = f" between {d.min():%b %d, %Y} and {d.max():%b %d, %Y}"
    if primary:
        v = pd.to_numeric(df[primary], errors="coerce")
        if how == "sum":
            out.append(f"Total {plabel} is **{fmt_number(v.sum(), kind)}** across {len(df):,} {entity}{span}.")
        else:
            out.append(f"Average {plabel} is **{fmt_number(v.mean(), kind)}** across {len(df):,} {entity}{span}.")
    else:
        out.append(f"The dataset contains **{len(df):,} {entity}**{span}.")
    if kpis and kpis[0].get("change_pct") is not None:
        ch = kpis[0]["change_pct"]
        word = "up" if ch >= 0 else "down"
        out.append(f"{kpis[0]['label']} is {word} **{abs(ch):.1f}%** in the last 30 days vs the prior 30 days.")
    for b in breakdowns[:2]:
        if not b["data"]:
            continue
        top, bottom = b["data"][0], b["data"][-1]
        dim = humanize(b["dimension"]).lower()
        if how == "sum" and primary:
            tot = sum(x["value"] for x in b["data"]) or 1
            out.append(f"**{top['label']}** is the leading {dim}, contributing {top['value'] / tot * 100:.0f}% of {plabel} "
                       f"({fmt_number(top['value'], kind)}).")
        elif len(b["data"]) > 1:
            out.append(f"**{top['label']}** has the highest {'average ' + plabel if primary else 'count'} by {dim} "
                       f"({fmt_number(top['value'], kind)}), **{bottom['label']}** the lowest ({fmt_number(bottom['value'], kind)}).")
    if anomalies:
        a = max(anomalies, key=lambda x: abs(x["zscore"]))
        ds = a["date"].strftime("%b %d, %Y") if hasattr(a["date"], "strftime") else str(a["date"])
        out.append(f"Unusual {a['direction']} in {plabel} around **{ds}**: {fmt_number(a['value'], kind)} vs "
                   f"~{fmt_number(a['expected'], kind)} expected.")
    if fc and fc["forecast"]:
        vals = [x["value"] for x in fc["forecast"]]
        if how == "sum":
            out.append(f"Forecast: about **{fmt_number(sum(vals), kind)}** of {plabel} over the next {len(vals)} periods.")
        else:
            out.append(f"Forecast: {plabel} expected to average **{fmt_number(np.mean(vals), kind)}** over the next {len(vals)} periods.")
    return out[:6]
