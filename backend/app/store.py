"""In-memory DuckDB dataset registry."""
from __future__ import annotations

import datetime as _dt
import re
import threading
from typing import Any

import duckdb
import numpy as np
import pandas as pd

from .utils import slugify

MAX_RESULT_ROWS = 5000


class DatasetNotFound(KeyError):
    pass


def _clean_column_names(cols: list[Any]) -> list[str]:
    seen: dict[str, int] = {}
    out = []
    for i, c in enumerate(cols):
        name = re.sub(r"[^0-9a-zA-Z]+", "_", str(c).strip()).strip("_").lower()
        if not name or name.startswith("unnamed"):
            name = f"column_{i + 1}"
        if name[0].isdigit():
            name = f"c_{name}"
        base = name
        k = seen.get(base, 0)
        while name in seen:
            k += 1
            name = f"{base}_{k}"
        seen[base] = k
        seen[name] = 0
        out.append(name)
    return out


def _maybe_parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    for c in df.columns:
        s = df[c]
        if s.dtype == object or pd.api.types.is_string_dtype(s):
            sample = s.dropna().astype(str).head(200)
            if sample.empty:
                continue
            looks_date = sample.str.match(r"^\d{4}[-/]\d{1,2}[-/]\d{1,2}([ T]\d{1,2}:\d{2}(:\d{2})?(\.\d+)?)?(Z|[+-]\d{2}:?\d{2})?$").mean() > 0.9
            looks_date = looks_date or (
                sample.str.match(r"^\d{1,2}[-/]\d{1,2}[-/]\d{4}$").mean() > 0.9
            )
            if looks_date:
                try:
                    parsed = pd.to_datetime(s, errors="coerce", utc=False)
                    if hasattr(parsed.dt, "tz") and parsed.dt.tz is not None:
                        parsed = parsed.dt.tz_convert(None)
                    if parsed.notna().sum() >= 0.9 * s.notna().sum():
                        df[c] = parsed
                except Exception:
                    pass
        elif isinstance(s.dtype, pd.DatetimeTZDtype):
            df[c] = s.dt.tz_convert(None)
    return df


def prepare_frame(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)
    df = df.copy()
    if df.columns.empty:
        raise ValueError("The data has no columns.")
    df.columns = _clean_column_names(list(df.columns))
    for c in df.columns:
        s = df[c]
        if s.dtype == object:
            # nested objects -> JSON-ish strings; mixed types -> strings
            non_null = s.dropna()
            if not non_null.empty and non_null.map(lambda v: isinstance(v, (dict, list))).any():
                import json
                df[c] = s.map(lambda v: json.dumps(v, default=str) if isinstance(v, (dict, list)) else v)
            non_null = df[c].dropna()
            if not non_null.empty:
                types = set(non_null.map(type))
                if len(types) > 1:
                    # try numeric first
                    num = pd.to_numeric(non_null, errors="coerce")
                    if num.notna().all():
                        df[c] = pd.to_numeric(df[c], errors="coerce")
                    else:
                        df[c] = df[c].map(lambda v: None if v is None or (isinstance(v, float) and np.isnan(v)) else str(v))
                elif types == {_dt.date}:
                    df[c] = pd.to_datetime(df[c])
                elif types <= {str}:
                    # numeric strings like "1,234" stay strings unless clean numbers
                    num = pd.to_numeric(non_null, errors="coerce")
                    if num.notna().mean() == 1.0 and len(non_null) > 0:
                        df[c] = pd.to_numeric(df[c], errors="coerce")
    return _maybe_parse_dates(df)


def _dtype_label(s: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(s):
        return "boolean"
    if pd.api.types.is_integer_dtype(s):
        return "integer"
    if pd.api.types.is_float_dtype(s):
        return "float"
    if pd.api.types.is_datetime64_any_dtype(s):
        non_null = s.dropna()
        if not non_null.empty and (non_null.dt.normalize() == non_null).all():
            return "date"
        return "datetime"
    return "string"


def infer_semantic(name: str, s: pd.Series, n_rows: int) -> str:
    n = name.lower()
    dtype = _dtype_label(s)
    if dtype in ("date", "datetime"):
        return "date"
    if dtype == "boolean":
        return "dimension"
    nunique = int(s.nunique(dropna=True))
    is_idish = n == "id" or n.endswith("_id") or n.endswith("_key") or n.endswith("_uuid") or n in ("uuid", "key")
    if dtype in ("integer", "float"):
        if is_idish:
            return "id"
        if dtype == "integer" and n_rows > 50 and nunique == n_rows and ("id" in n.split("_") or "number" in n or n.endswith("_no")):
            return "id"
        if n in ("year", "month", "quarter", "week", "day", "hour") or n.endswith("_year"):
            return "dimension"
        return "metric"
    # strings
    if is_idish:
        return "id"
    if n_rows > 0 and nunique > 50 and nunique >= 0.95 * s.notna().sum() and s.notna().sum() > 20:
        avg_len = s.dropna().astype(str).str.len().mean()
        return "text" if avg_len > 40 else "id"
    if nunique > 0:
        avg_len = s.dropna().astype(str).str.len().mean()
        if avg_len > 60:
            return "text"
    if nunique > 200:
        return "text"
    return "dimension"


class Store:
    def __init__(self) -> None:
        self.con = duckdb.connect(":memory:")
        self.con.execute("SET enable_external_access=false")
        self.con.execute("SET lock_configuration=true")
        self.lock = threading.RLock()
        self.datasets: dict[str, dict[str, Any]] = {}
        self._cache: dict[str, dict[str, Any]] = {}

    # ---------- registry ----------
    def unique_id(self, base: str) -> str:
        base = slugify(base)
        cand, k = base, 2
        while cand in self.datasets:
            cand = f"{base}_{k}"
            k += 1
        return cand

    def register(self, df: pd.DataFrame, name: str, source: str, dataset_id: str | None = None,
                 replace: bool = False) -> dict[str, Any]:
        df = prepare_frame(df)
        if len(df) == 0:
            raise ValueError("The data source returned no rows.")
        with self.lock:
            ds_id = slugify(dataset_id) if dataset_id else slugify(name)
            if not replace or ds_id not in self.datasets:
                ds_id = ds_id if ds_id not in self.datasets else self.unique_id(ds_id)
            tmp = f"__df_{ds_id}"
            self.con.register(tmp, df)
            try:
                self.con.execute(f'CREATE OR REPLACE TABLE "{ds_id}" AS SELECT * FROM "{tmp}"')
            finally:
                self.con.unregister(tmp)
            n_rows = len(df)
            columns = [
                {"name": c, "dtype": _dtype_label(df[c]), "semantic": infer_semantic(c, df[c], n_rows)}
                for c in df.columns
            ]
            meta = {
                "id": ds_id,
                "name": name or ds_id,
                "source": source,
                "rows": int(n_rows),
                "columns": columns,
                "created_at": _dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
            }
            self.datasets[ds_id] = meta
            self._cache.pop(ds_id, None)
            return meta

    def get(self, ds_id: str) -> dict[str, Any]:
        meta = self.datasets.get(ds_id)
        if meta is None:
            raise DatasetNotFound(ds_id)
        return meta

    def list(self) -> list[dict[str, Any]]:
        return list(self.datasets.values())

    def delete(self, ds_id: str) -> None:
        with self.lock:
            self.get(ds_id)
            self.con.execute(f'DROP TABLE IF EXISTS "{ds_id}"')
            self.datasets.pop(ds_id, None)
            self._cache.pop(ds_id, None)

    def cache(self, ds_id: str) -> dict[str, Any]:
        return self._cache.setdefault(ds_id, {})

    # ---------- queries ----------
    def df(self, ds_id: str) -> pd.DataFrame:
        self.get(ds_id)
        with self.lock:
            return self.con.execute(f'SELECT * FROM "{ds_id}"').df()

    def query_df(self, sql: str, params: list | None = None) -> pd.DataFrame:
        with self.lock:
            return self.con.execute(sql, params or []).df()

    def query(self, sql: str, limit: int = MAX_RESULT_ROWS) -> tuple[list[str], list[list[Any]], bool]:
        """Run SQL; return (columns, rows, truncated)."""
        with self.lock:
            timer = threading.Timer(30.0, self._interrupt)
            timer.start()
            try:
                cur = self.con.execute(sql)
                cols = [d[0] for d in (cur.description or [])]
                rows = cur.fetchmany(limit + 1)
            finally:
                timer.cancel()
        truncated = len(rows) > limit
        return cols, [list(r) for r in rows[:limit]], truncated

    def _interrupt(self) -> None:
        try:
            self.con.interrupt()
        except Exception:
            pass

    def preview(self, ds_id: str, limit: int = 100) -> dict[str, Any]:
        self.get(ds_id)
        limit = max(1, min(int(limit), 1000))
        cols, rows, _ = self.query(f'SELECT * FROM "{ds_id}" LIMIT {limit}', limit)
        return {"columns": cols, "rows": rows}

    def schema_text(self, ds_id: str) -> str:
        meta = self.get(ds_id)
        lines = [f'Table "{ds_id}" ({meta["rows"]} rows):']
        for c in meta["columns"]:
            lines.append(f'  - "{c["name"]}" {c["dtype"]} [{c["semantic"]}]')
        return "\n".join(lines)


store = Store()
