"""SQL Engineer agent: writes DuckDB SQL for a plan, executes it, retries on error."""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

from ..sqlguard import UnsafeSQLError, ensure_read_only
from ..store import store
from . import llm
from .planner import Plan, dataset_context, q_ident, _value_hints

INTERNAL_ROW_CAP = 5000
MAX_RETRIES = 2


@dataclass
class SQLResult:
    sql: str
    columns: list[str]
    rows: list[list[Any]]
    source: str = "rules"
    attempts: int = 1
    errors: list[str] = field(default_factory=list)
    chart_hint: dict[str, Any] | None = None
    title_hint: str | None = None
    truncated: bool = False


# ------------------------------------------------------------------ rule SQL
def build_sql(plan: Plan, ds_id: str) -> str:
    t = q_ident(ds_id)
    filters = list(plan.filters)
    meas = ", ".join(f"{m.sql} AS {q_ident(m.alias)}" for m in plan.measures)
    first = q_ident(plan.measures[0].alias) if plan.measures else None
    order = (plan.order or "desc").upper()

    def where(extra: list[str] | None = None) -> str:
        preds = filters + (extra or [])
        return (" WHERE " + " AND ".join(f"({p})" for p in preds)) if preds else ""

    if plan.intent == "list":
        return f"SELECT * FROM {t}{where()} LIMIT {plan.limit or 50}"

    if plan.intent == "rows" and plan.order_by_metric:
        return (f"SELECT * FROM {t}{where([f'{q_ident(plan.order_by_metric)} IS NOT NULL'])} "
                f"ORDER BY {q_ident(plan.order_by_metric)} {order} LIMIT {plan.limit or 10}")

    if plan.intent == "correlation" and len(plan.corr_cols) >= 2:
        x, y = (q_ident(c) for c in plan.corr_cols[:2])
        return (f"SELECT {x}, {y} FROM (SELECT {x}, {y} FROM {t}{where([f'{x} IS NOT NULL', f'{y} IS NOT NULL'])}) "
                f"USING SAMPLE reservoir(500 ROWS) REPEATABLE (42)")

    if plan.intent == "distribution" and plan.corr_cols:
        c = plan.corr_cols[0]
        x = q_ident(c)
        return (
            f"WITH s AS (SELECT CAST({x} AS DOUBLE) AS v FROM {t}{where([f'{x} IS NOT NULL'])}), "
            f"b AS (SELECT MIN(v) AS lo, MAX(v) AS hi FROM s) "
            f"SELECT ROUND(lo + (hi - lo) / 12 * LEAST(FLOOR((v - lo) / NULLIF((hi - lo) / 12, 0)), 11), 2) "
            f"AS {q_ident(c + '_bucket')}, COUNT(*) AS {q_ident('count')} "
            f"FROM s, b GROUP BY 1 ORDER BY 1 NULLS FIRST")

    if plan.intent == "trend" and plan.date_col:
        dc = q_ident(plan.date_col)
        grain = plan.grain or "month"
        period = f"CAST(DATE_TRUNC('{grain}', {dc}) AS DATE) AS {q_ident(grain)}"
        dims = "".join(f", {q_ident(d)}" for d in plan.dimensions[:1])
        sql = f"SELECT {period}{dims}, {meas} FROM {t}{where([f'{dc} IS NOT NULL'])} GROUP BY ALL"
        if plan.rank_periods and not plan.dimensions:
            return f"{sql} ORDER BY {first} {order} NULLS LAST LIMIT {plan.limit or 10}"
        return f"{sql} ORDER BY 1{', 2' if dims else ''}"

    if plan.intent == "growth" and plan.date_col and plan.dimensions:
        return _growth_sql(plan, ds_id, where)

    if plan.intent == "group" and plan.dimensions:
        dims = ", ".join(q_ident(d) for d in plan.dimensions[:2])
        not_null = [f"{q_ident(d)} IS NOT NULL" for d in plan.dimensions[:2]]
        limit = plan.limit or (300 if len(plan.dimensions) > 1 else 50)
        return f"SELECT {dims}, {meas} FROM {t}{where(not_null)} GROUP BY ALL ORDER BY {first} {order} NULLS LAST LIMIT {limit}"

    # kpi
    return f"SELECT {meas} FROM {t}{where()}"


def _growth_sql(plan: Plan, ds_id: str, where) -> str:
    t = q_ident(ds_id)
    dc = q_ident(plan.date_col)
    dim = q_ident(plan.dimensions[0])
    m = plan.measures[0]
    ctx = dataset_context(ds_id)
    rng = ctx["date_range"].get(plan.date_col)
    window = 90
    if rng is not None:
        span = max((rng[1] - rng[0]).days, 2)
        if span < 180:
            window = max(span // 2, 1)
    alias = m.alias
    base_preds = [f"{dim} IS NOT NULL"]
    recent = where(base_preds + [f"{dc} > b.__mx - INTERVAL {window} DAY"])
    prev = where(base_preds + [f"{dc} <= b.__mx - INTERVAL {window} DAY",
                                f"{dc} > b.__mx - INTERVAL {2 * window} DAY"])
    plan.growth_window = window  # type: ignore[attr-defined]
    return (
        f"WITH b AS (SELECT MAX({dc}) AS __mx FROM {t}), "
        f"recent AS (SELECT {dim} AS k, {m.sql} AS v FROM {t}, b{recent} GROUP BY 1), "
        f"prev AS (SELECT {dim} AS k, {m.sql} AS v FROM {t}, b{prev} GROUP BY 1) "
        f"SELECT COALESCE(r.k, p.k) AS {dim}, p.v AS {q_ident('previous_' + alias)}, "
        f"r.v AS {q_ident('recent_' + alias)}, "
        f"ROUND(100.0 * (r.v - p.v) / NULLIF(ABS(p.v), 0), 2) AS {q_ident('growth_pct')} "
        f"FROM recent r FULL OUTER JOIN prev p ON r.k = p.k "
        f"ORDER BY {q_ident('growth_pct')} {(plan.order or 'desc').upper()} NULLS LAST LIMIT {plan.limit or 50}"
    )


# ------------------------------------------------------------------ execution
def execute(sql: str) -> tuple[str, list[str], list[list[Any]], bool]:
    clean = ensure_read_only(sql)
    cols, rows, truncated = store.query(clean, INTERNAL_ROW_CAP)
    return clean, cols, rows, truncated


SQL_SYSTEM = """You are the SQL Engineer agent in a data-analysis team. Write ONE DuckDB SQL query that answers
the business question. Rules:
- Read-only: a single SELECT (or WITH ... SELECT). No DDL/DML, no PRAGMA/ATTACH/COPY, no file functions.
- Quote identifiers with double quotes exactly as given in the schema; the table name is given.
- Aggregate to at most ~200 result rows; use meaningful snake_case aliases for computed columns.
- For time buckets use CAST(DATE_TRUNC('month', "col") AS DATE) AS month (or day/week/quarter/year) and ORDER BY it.
- Rates/percentages: multiply by 100.0 and alias with a _rate / _pct suffix. Guard divisions with NULLIF(...,0).
- Put the label/category or period column first, then the numeric columns.
Respond with ONLY a JSON object:
{"sql": "...", "title": "short chart title",
 "chart": {"type": "bar|line|area|pie|scatter|table|kpi", "x": "column or null", "y": ["numeric columns"]}}"""


def _llm_sql(plan: Plan, ds_id: str, rule_sql: str, previous: list[tuple[str, str]]) -> dict[str, Any]:
    ctx = dataset_context(ds_id)
    parts = [store.schema_text(ds_id), _value_hints(ctx), f"Question: {plan.question}"]
    if getattr(plan, "llm_notes", None):
        parts.append(f"Planner notes: {plan.llm_notes}")
    lp = getattr(plan, "llm_plan", None)
    if lp:
        import json
        parts.append("Planner output: " + json.dumps(lp)[:1500])
    parts.append(f"A heuristic draft query (may be imperfect; improve it if needed):\n{rule_sql}")
    for sql, err in previous:
        parts.append(f"Your previous query failed:\n{sql}\nError: {err}\nFix it.")
    return llm.complete_json(SQL_SYSTEM, "\n\n".join(p for p in parts if p), max_tokens=3000,
                             required_keys=("sql",))


def run(plan: Plan, ds_id: str) -> SQLResult:
    rule_sql = build_sql(plan, ds_id)
    errors: list[str] = []
    attempts = 0
    if llm.llm_available():
        previous: list[tuple[str, str]] = []
        for _ in range(1 + MAX_RETRIES):
            attempts += 1
            try:
                data = _llm_sql(plan, ds_id, rule_sql, previous)
            except llm.LLMError as e:
                errors.append(f"LLM: {str(e)[:120]}")
                break
            sql = str(data.get("sql") or "")
            try:
                clean, cols, rows, trunc = execute(sql)
                chart = data.get("chart") if isinstance(data.get("chart"), dict) else None
                return SQLResult(clean, cols, rows, "llm", attempts, errors, chart,
                                 str(data.get("title") or "") or None, trunc)
            except (UnsafeSQLError, Exception) as e:  # noqa: BLE001 - feed any DB error back
                msg = str(e).splitlines()[0][:300]
                errors.append(msg)
                previous.append((sql, msg))
    # rule-based path (or LLM fallback)
    candidates = [rule_sql]
    if plan.filters:
        p2 = copy.copy(plan)
        p2.filters = []
        candidates.append(build_sql(p2, ds_id))
    candidates.append(f"SELECT * FROM {q_ident(ds_id)} LIMIT 50")
    last_err = None
    for sql in candidates[: 1 + MAX_RETRIES]:
        attempts += 1
        try:
            clean, cols, rows, trunc = execute(sql)
            return SQLResult(clean, cols, rows, "rules", attempts, errors, None, None, trunc)
        except Exception as e:  # noqa: BLE001
            last_err = str(e).splitlines()[0][:300]
            errors.append(last_err)
    raise RuntimeError(f"Could not produce a working query: {last_err}")
