"""Analyst agent: computes statistics on the query result and picks a chart."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from ..utils import humanize, kind_for, pluralize
from .planner import Plan
from .sql_engineer import SQLResult

MAX_OUTPUT_ROWS = 500
PERIOD_NAMES = {"day", "week", "month", "quarter", "year", "date", "period"}
CHART_TYPES = {"bar", "line", "area", "pie", "scatter", "table", "kpi"}


@dataclass
class Analysis:
    columns: list[str]
    rows: list[list[Any]]
    chart: dict[str, Any]
    stats: dict[str, Any] = field(default_factory=dict)
    kinds: dict[str, str] = field(default_factory=dict)
    shape: str = "table"  # kpi | category | time | scatter | table | empty
    summary: str = ""


def _is_dateish(s: pd.Series, name: str) -> bool:
    if pd.api.types.is_datetime64_any_dtype(s):
        return True
    if s.dtype == object and len(s.dropna()):
        v = s.dropna().iloc[0]
        import datetime as _dt
        if isinstance(v, (_dt.date, _dt.datetime, pd.Timestamp)):
            return True
        if name.lower() in PERIOD_NAMES or name.lower().endswith(("_date", "_month", "_week", "_day")):
            try:
                pd.to_datetime(s.dropna().head(20))
                return True
            except Exception:
                return False
    return False


def _num_cols(df: pd.DataFrame, exclude: set[str]) -> list[str]:
    out = []
    for c in df.columns:
        if c in exclude:
            continue
        s = df[c]
        if pd.api.types.is_bool_dtype(s):
            continue
        if pd.api.types.is_numeric_dtype(s):
            out.append(c)
        elif s.dtype == object and len(s.dropna()):
            conv = pd.to_numeric(s, errors="coerce")
            if conv.notna().sum() == s.notna().sum():
                df[c] = conv
                out.append(c)
    return out


def _title(plan: Plan, x: str | None, y: list[str], kinds_label: dict[str, str]) -> str:
    ylab = " & ".join(kinds_label.get(c, humanize(c)) for c in y[:2]) if y else "Result"
    if plan.intent == "trend" and plan.grain:
        base = f"{plan.grain.title()}ly {ylab}".replace("Dayly", "Daily")
        if plan.dimensions:
            base += f" by {humanize(plan.dimensions[0])}"
        return base
    if plan.intent == "growth" and x:
        base = plan.measures[0].label if plan.measures else ylab
        return f"{base} growth by {humanize(x)}"
    if x:
        prefix = ""
        if plan.limit and plan.order:
            prefix = f"{'Top' if plan.order == 'desc' else 'Bottom'} {plan.limit} "
        return f"{prefix}{pluralize(humanize(x))} by {ylab}" if prefix else f"{ylab} by {humanize(x)}"
    return ylab


def analyze(plan: Plan, res: SQLResult) -> Analysis:
    df = pd.DataFrame(res.rows, columns=res.columns)
    measure_kind = {m.alias: m.kind for m in plan.measures}
    measure_label = {m.alias: m.label for m in plan.measures}
    kinds = {c: measure_kind.get(c, kind_for(c)) for c in df.columns}
    if "growth_pct" in df.columns:
        kinds["growth_pct"] = "percent"
        measure_label["growth_pct"] = "Growth %"
    for c in df.columns:
        for pre in ("previous_", "recent_"):
            if c.startswith(pre) and c[len(pre):] in measure_kind:
                kinds[c] = measure_kind[c[len(pre):]]

    if df.empty:
        return Analysis(res.columns, [], {"type": "table", "title": "No matching rows"}, {"row_count": 0},
                        kinds, "empty", "The query returned no rows.")

    if plan.intent in ("list", "rows"):
        title = "Matching records" if plan.intent == "list" else _title(plan, None, [plan.order_by_metric or ""], {})
        if plan.intent == "rows" and plan.order_by_metric:
            title = f"{'Top' if plan.order == 'desc' else 'Bottom'} {len(df)} records by {humanize(plan.order_by_metric)}"
        stats = {"row_count": len(df)}
        if plan.order_by_metric and plan.order_by_metric in df:
            col = plan.order_by_metric
            stats.update(metric=col, top_value=df[col].iloc[0], total=float(pd.to_numeric(df[col]).sum()))
        return Analysis(list(df.columns), _rows(df), {"type": "table", "title": title}, stats, kinds, "table")

    date_cols = [c for c in df.columns if _is_dateish(df[c], c)]
    nums = _num_cols(df, set(date_cols))
    cats = [c for c in df.columns if c not in nums and c not in date_cols]

    # LLM chart hint (validated) takes precedence when sensible
    hint = res.chart_hint or {}
    stats: dict[str, Any] = {"row_count": len(df)}

    # ---- KPI
    if len(df) == 1 and nums and not cats and not date_cols:
        stats["kpis"] = {c: _f(df[c].iloc[0]) for c in nums}
        chart = {"type": "kpi", "y": nums, "title": res.title_hint or _title(plan, None, nums, measure_label)}
        return Analysis(list(df.columns), _rows(df), chart, stats, kinds, "kpi")

    # ---- time series
    if date_cols and nums:
        x = date_cols[0]
        df[x] = pd.to_datetime(df[x])
        if plan.rank_periods and not cats:
            y = nums[:3]
            df[x] = df[x].map(lambda d: period_label(d, plan.grain))
            stats.update(_category_stats(df, x, y[0], kinds))
            stats.pop("top_share", None)
            stats.pop("top3_share", None)
            stats["period_rank"] = True
            chart = {"type": "bar", "x": x, "y": y, "title": res.title_hint or _title(plan, x, y, measure_label)}
            return Analysis(list(df.columns), _rows(df), chart, stats, kinds, "category")
        if cats:
            cat = cats[0]
            y_metric = nums[0]
            totals = df.groupby(cat)[y_metric].sum().sort_values(ascending=False)
            keep = list(totals.index[:6])
            wide = df[df[cat].isin(keep)].pivot_table(index=x, columns=cat, values=y_metric, aggfunc="sum")
            wide = wide[[k for k in keep if k in wide.columns]].sort_index()
            wide.columns = [str(c) for c in wide.columns]
            wide = wide.reset_index()
            series_cols = [c for c in wide.columns if c != x]
            for c in series_cols:
                kinds[c] = kinds.get(y_metric, "number")
            tot = wide[[x]].copy()
            tot["__total"] = wide[series_cols].sum(axis=1, min_count=1)
            kinds["__total"] = kinds.get(y_metric, "number")
            stats.update(_time_stats(tot, x, "__total", kinds))
            stats["series_totals"] = {str(k): _f(v) for k, v in totals.head(6).items()}
            stats["series_dimension"] = cat
            stats["metric"] = y_metric
            if len(totals) > 6:
                stats["series_note"] = f"Showing top 6 of {len(totals)} {humanize(cat).lower()} values."
            wide = wide.tail(MAX_OUTPUT_ROWS)
            ctype = "line" if len(wide) > 3 else "bar"
            chart = {"type": ctype, "x": x, "y": series_cols,
                     "title": res.title_hint or _title(plan, x, [y_metric], measure_label)}
            return Analysis(list(wide.columns), _rows(wide), chart, stats, kinds, "time")
        df = df.sort_values(x)
        stats.update(_time_stats(df, x, nums[0], kinds))
        out = df.tail(MAX_OUTPUT_ROWS)
        ctype = "area" if len(nums) == 1 and len(out) > 3 else ("line" if len(out) > 3 else "bar")
        if hint.get("type") in ("line", "area", "bar") and len(out) > 1:
            ctype = hint["type"]
        chart = {"type": ctype, "x": x, "y": nums[:4], "title": res.title_hint or _title(plan, x, nums, measure_label)}
        return Analysis(list(out.columns), _rows(out), chart, stats, kinds, "time")

    # ---- categories
    if cats and nums:
        x = cats[0]
        if len(cats) >= 2 and len(nums) == 1:
            second = cats[1]
            totals = df.groupby(x)[nums[0]].sum().sort_values(ascending=plan.order == "asc")
            sec_vals = list(df.groupby(second)[nums[0]].sum().sort_values(ascending=False).index[:6])
            wide = df[df[second].isin(sec_vals)].pivot_table(index=x, columns=second, values=nums[0], aggfunc="sum")
            wide = wide.reindex([i for i in totals.index if i in wide.index])
            wide.columns = [str(c) for c in wide.columns]
            wide = wide.reset_index().head(MAX_OUTPUT_ROWS)
            ys = [c for c in wide.columns if c != x]
            for c in ys:
                kinds[c] = kinds.get(nums[0], "number")
            stats.update(_category_stats(df.groupby(x, as_index=False)[nums[0]].sum()
                                         .sort_values(nums[0], ascending=plan.order == "asc"), x, nums[0], kinds))
            stats["series_dimension"] = second
            chart = {"type": "bar", "x": x, "y": ys,
                     "title": res.title_hint or f"{measure_label.get(nums[0], humanize(nums[0]))} by {humanize(x)} and {humanize(second)}"}
            return Analysis(list(wide.columns), _rows(wide), chart, stats, kinds, "category")
        y = nums[:4]
        if "growth_pct" in nums:
            y = ["growth_pct"]
        stats.update(_category_stats(df, x, y[0], kinds))
        if len(y) > 1:
            stats["secondary"] = [_category_stats(df, x, yy, kinds) for yy in y[1:3]]
        primary_measure = next((m for m in plan.measures if m.alias == y[0]), None)
        additive = primary_measure is None or primary_measure.agg in ("sum", "count")
        ctype = "bar"
        if (plan.share and len(df) <= 8 and len(y) == 1 and additive) or hint.get("type") == "pie" and len(df) <= 10:
            ctype = "pie"
        elif hint.get("type") in ("bar", "table"):
            ctype = hint["type"]
        if len(df) > 60:
            ctype = "table"
        chart = {"type": ctype, "x": x, "y": y, "title": res.title_hint or _title(plan, x, y, measure_label)}
        return Analysis(list(df.columns), _rows(df.head(MAX_OUTPUT_ROWS)), chart, stats, kinds, "category")

    # ---- scatter / distribution
    if len(nums) >= 2 and not cats:
        if plan.intent == "distribution" or (len(nums) == 2 and nums[1] == "count" and df[nums[0]].is_monotonic_increasing):
            x, y = nums[0], [nums[1]]
            stats.update(_category_stats(df.assign(**{x: df[x].astype(str)}), x, y[0], kinds))
            chart = {"type": "bar", "x": x, "y": y, "title": res.title_hint or f"Distribution of {humanize(x.replace('_bucket', ''))}"}
            return Analysis(list(df.columns), _rows(df), chart, stats, kinds, "category")
        x, y = nums[0], nums[1]
        a, b = pd.to_numeric(df[x], errors="coerce"), pd.to_numeric(df[y], errors="coerce")
        r = float(a.corr(b)) if a.notna().sum() > 2 else None
        stats.update(correlation=r, x=x, y=y)
        chart = {"type": "scatter", "x": x, "y": [y], "title": res.title_hint or f"{humanize(y)} vs {humanize(x)}"}
        return Analysis(list(df.columns), _rows(df.head(MAX_OUTPUT_ROWS)), chart, stats, kinds, "scatter")

    chart = {"type": "table", "title": res.title_hint or "Query result"}
    if hint.get("type") in CHART_TYPES and hint.get("x") in df.columns:
        ys = [c for c in (hint.get("y") or []) if c in df.columns]
        if ys:
            chart = {"type": hint["type"], "x": hint["x"], "y": ys, "title": res.title_hint or "Query result"}
    return Analysis(list(df.columns), _rows(df.head(MAX_OUTPUT_ROWS)), chart, stats, kinds, "table")


# ------------------------------------------------------------------ helpers
def period_label(d: Any, grain: str | None) -> str:
    ts = pd.Timestamp(d)
    if grain == "year":
        return f"{ts:%Y}"
    if grain == "quarter":
        return f"Q{(ts.month - 1) // 3 + 1} {ts:%Y}"
    if grain == "month":
        return f"{ts:%b %Y}"
    return f"{ts:%Y-%m-%d}"


def _f(v: Any) -> float | None:
    try:
        f = float(v)
        return None if np.isnan(f) else f
    except (TypeError, ValueError):
        return None


def _rows(df: pd.DataFrame) -> list[list[Any]]:
    return df.astype(object).where(pd.notna(df), None).values.tolist()


def _category_stats(df: pd.DataFrame, x: str, y: str, kinds: dict[str, str]) -> dict[str, Any]:
    s = pd.to_numeric(df[y], errors="coerce")
    labels = df[x].astype(str)
    valid = s.notna()
    s, labels = s[valid], labels[valid]
    if s.empty:
        return {"metric": y, "dimension": x}
    order = s.sort_values(ascending=False)
    top_i, bot_i = order.index[0], order.index[-1]
    total = float(s.sum())
    out = {
        "metric": y, "dimension": x, "kind": kinds.get(y, "number"), "count": int(len(s)),
        "top": {"label": labels[top_i], "value": float(s[top_i])},
        "bottom": {"label": labels[bot_i], "value": float(s[bot_i])},
        "total": total, "mean": float(s.mean()),
    }
    if len(order) > 1:
        out["second"] = {"label": labels[order.index[1]], "value": float(s[order.index[1]])}
    if total and (s >= 0).all():
        out["top_share"] = float(s[top_i] / total * 100)
        out["top3_share"] = float(order.head(3).sum() / total * 100)
    if out["mean"] and (s > 0).all():
        out["bottom_vs_mean_pct"] = float((s[bot_i] - out["mean"]) / abs(out["mean"]) * 100)
        out["top_vs_mean_pct"] = float((s[top_i] - out["mean"]) / abs(out["mean"]) * 100)
    return out


def _time_stats(df: pd.DataFrame, x: str, y: str, kinds: dict[str, str]) -> dict[str, Any]:
    d = df[[x, y]].dropna().sort_values(x)
    if d.empty:
        return {"metric": y}
    vals = pd.to_numeric(d[y], errors="coerce")
    dates = pd.to_datetime(d[x])
    first, last = float(vals.iloc[0]), float(vals.iloc[-1])
    out: dict[str, Any] = {
        "metric": y, "kind": kinds.get(y, "number"), "periods": int(len(d)),
        "start": dates.iloc[0], "end": dates.iloc[-1], "first": first, "last": last,
        "total": float(vals.sum()), "mean": float(vals.mean()),
        "peak": {"date": dates[vals.idxmax()], "value": float(vals.max())},
        "trough": {"date": dates[vals.idxmin()], "value": float(vals.min())},
    }
    if first:
        out["change_pct"] = (last - first) / abs(first) * 100
    if len(vals) >= 2 and vals.iloc[-2]:
        out["last_change_pct"] = (last - float(vals.iloc[-2])) / abs(float(vals.iloc[-2])) * 100
    if len(vals) >= 6:
        half = len(vals) // 2
        a, b = float(vals.iloc[:half].mean()), float(vals.iloc[half:].mean())
        if a:
            out["half_change_pct"] = (b - a) / abs(a) * 100
        x_idx = np.arange(len(vals))
        slope = float(np.polyfit(x_idx, vals.values.astype(float), 1)[0])
        out["slope"] = slope
        out["direction"] = "up" if slope > 0 else "down"
    return out
