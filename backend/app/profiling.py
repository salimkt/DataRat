"""Dataset profiling: column stats, quality score, correlations."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .store import store
from .utils import to_jsonable


def _fmt_bin(v: float) -> str:
    a = abs(v)
    if a >= 1e6:
        return f"{v / 1e6:.1f}M"
    if a >= 1e3:
        return f"{v / 1e3:.1f}K"
    if a >= 10 or v == int(v):
        return f"{v:.0f}"
    return f"{v:.2f}"


def _numeric_hist(s: pd.Series, bins: int = 12) -> list[dict[str, Any]]:
    vals = s.dropna().astype(float).values
    if len(vals) == 0:
        return []
    lo, hi = float(np.min(vals)), float(np.max(vals))
    if lo == hi:
        return [{"bin": _fmt_bin(lo), "count": int(len(vals))}]
    nunique = len(np.unique(vals))
    bins = min(bins, max(nunique, 1))
    counts, edges = np.histogram(vals, bins=bins)
    return [{"bin": f"{_fmt_bin(edges[i])}–{_fmt_bin(edges[i + 1])}", "count": int(c)}
            for i, c in enumerate(counts)]


def _date_hist(s: pd.Series) -> list[dict[str, Any]]:
    d = pd.to_datetime(s.dropna())
    if d.empty:
        return []
    span_days = (d.max() - d.min()).days
    freq = "D" if span_days <= 62 else ("W" if span_days <= 370 else "M")
    per = d.dt.to_period(freq)
    vc = per.value_counts().sort_index()
    if len(vc) > 60:
        vc = d.dt.to_period("Q").value_counts().sort_index()
    return [{"bin": str(k.start_time.date()) if freq != "M" else str(k), "count": int(v)} for k, v in vc.items()]


def profile(ds_id: str) -> dict[str, Any]:
    meta = store.get(ds_id)
    cache = store.cache(ds_id)
    if "profile" in cache:
        return cache["profile"]
    df = store.df(ds_id)
    n_rows, n_cols = len(df), len(df.columns)
    total_cells = max(n_rows * n_cols, 1)
    missing_cells = int(df.isna().sum().sum())
    try:
        dup_rows = int(df.duplicated().sum())
    except TypeError:
        dup_rows = int(df.astype(str).duplicated().sum())

    col_profiles = []
    constant_cols = 0
    for c in meta["columns"]:
        name = c["name"]
        s = df[name]
        nunique = int(s.nunique(dropna=True))
        if nunique <= 1:
            constant_cols += 1
        p: dict[str, Any] = {
            "name": name, "dtype": c["dtype"], "semantic": c["semantic"],
            "missing_pct": round(float(s.isna().mean() * 100), 2) if n_rows else 0.0,
            "unique": nunique,
        }
        if c["dtype"] in ("integer", "float"):
            num = pd.to_numeric(s, errors="coerce")
            if num.notna().any():
                p.update(min=float(num.min()), max=float(num.max()), mean=round(float(num.mean()), 4),
                         median=float(num.median()), std=round(float(num.std()), 4) if num.count() > 1 else 0.0)
                p["histogram"] = _numeric_hist(num)
            if c["semantic"] != "metric" or nunique <= 12:
                vc = s.value_counts().head(10)
                p["top_values"] = [{"value": k, "count": int(v)} for k, v in vc.items()]
        elif c["dtype"] in ("date", "datetime"):
            d = s.dropna()
            if not d.empty:
                p.update(min=d.min(), max=d.max())
                p["histogram"] = _date_hist(d)
        else:
            vc = s.astype(object).where(s.notna(), None).dropna().astype(str).value_counts().head(10) \
                if c["dtype"] != "boolean" else s.value_counts().head(10)
            p["top_values"] = [{"value": k, "count": int(v)} for k, v in vc.items()]
        col_profiles.append(p)

    metric_cols = [c["name"] for c in meta["columns"]
                   if c["dtype"] in ("integer", "float", "boolean") and c["semantic"] in ("metric", "dimension")
                   and (c["dtype"] != "boolean")][:14]
    corr_cols: list[str] = []
    matrix: list[list[float]] = []
    if len(metric_cols) >= 2:
        sub = df[metric_cols].apply(pd.to_numeric, errors="coerce")
        sub = sub.loc[:, sub.nunique() > 1]
        if sub.shape[1] >= 2:
            cm = sub.corr().fillna(0.0)
            corr_cols = list(cm.columns)
            matrix = [[round(float(v), 3) for v in row] for row in cm.values]

    completeness = 1 - missing_cells / total_cells
    uniqueness = 1 - (dup_rows / n_rows if n_rows else 0)
    validity = 1 - (constant_cols / n_cols if n_cols else 0)
    quality = round(100 * (0.6 * completeness + 0.25 * uniqueness + 0.15 * validity), 1)

    result = to_jsonable({
        "rows": n_rows,
        "columns": n_cols,
        "missing_cells_pct": round(missing_cells / total_cells * 100, 2),
        "duplicate_rows": dup_rows,
        "quality_score": quality,
        "column_profiles": col_profiles,
        "correlations": {"columns": corr_cols, "matrix": matrix},
    })
    cache["profile"] = result
    return result
