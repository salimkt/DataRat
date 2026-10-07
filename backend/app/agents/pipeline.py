"""Multi-agent pipeline: Planner -> SQL Engineer -> Analyst -> Narrator."""
from __future__ import annotations

import time
from typing import Any

from ..utils import fmt_number, to_jsonable
from . import analyst, llm, narrator, planner, sql_engineer


def _ms(t0: float) -> int:
    return int(round((time.perf_counter() - t0) * 1000))


def run(ds_id: str, question: str) -> dict[str, Any]:
    question = question.strip()
    steps: list[dict[str, Any]] = []

    # 1. Planner
    t0 = time.perf_counter()
    plan = planner.plan(ds_id, question)
    detail = plan.summary()
    if plan.llm_notes:
        detail = f"{plan.llm_notes} — {detail}"
    if plan.matched:
        detail += " · matched " + ", ".join(plan.matched[:6])
    steps.append({"agent": "Planner", "title": "Understood the question",
                  "detail": f"[{'Claude' if plan.source == 'llm' else 'rules'}] {detail}", "duration_ms": _ms(t0)})

    # 2. SQL Engineer
    t0 = time.perf_counter()
    res = sql_engineer.run(plan, ds_id)
    retry_note = f" after {res.attempts} attempts" if res.attempts > 1 else ""
    trunc = " (truncated)" if res.truncated else ""
    steps.append({"agent": "SQL Engineer", "title": "Wrote and executed DuckDB SQL",
                  "detail": f"[{'Claude' if res.source == 'llm' else 'rules'}] Query returned {len(res.rows)} rows"
                            f"{trunc}{retry_note}." + (f" Recovered from: {res.errors[-1]}" if res.errors and res.attempts > 1 else ""),
                  "duration_ms": _ms(t0)})

    # 3. Analyst
    t0 = time.perf_counter()
    an = analyst.analyze(plan, res)
    st = an.stats
    if an.shape == "category" and st.get("top"):
        a_detail = (f"Top {st.get('dimension')}: {st['top']['label']} = "
                    f"{fmt_number(st['top']['value'], st.get('kind', 'number'))}")
        if st.get("top_share") is not None:
            a_detail += f" ({st['top_share']:.0f}% share)"
    elif an.shape == "time" and "change_pct" in st:
        a_detail = f"{st.get('periods')} periods, {st['change_pct']:+.1f}% first→last, peak {fmt_number(st['peak']['value'], st.get('kind', 'number'))}"
    elif an.shape == "kpi":
        a_detail = ", ".join(f"{k} = {fmt_number(v, an.kinds.get(k, 'number'))}" for k, v in (st.get("kpis") or {}).items())
    elif an.shape == "scatter" and st.get("correlation") is not None:
        a_detail = f"Pearson r = {st['correlation']:.2f}"
    else:
        a_detail = f"{st.get('row_count', 0)} rows analysed"
    steps.append({"agent": "Analyst", "title": f"Computed statistics and chose a {an.chart['type']} chart",
                  "detail": a_detail, "duration_ms": _ms(t0)})

    # 4. Narrator
    t0 = time.perf_counter()
    story, src = narrator.narrate(plan, an, ds_id, res.sql)
    steps.append({"agent": "Narrator", "title": "Wrote the executive summary",
                  "detail": f"[{'Claude' if src == 'llm' else 'rules'}] {len(story['recommendations'])} recommendations, "
                            f"{len(story['follow_ups'])} follow-ups",
                  "duration_ms": _ms(t0)})

    chart = {k: v for k, v in an.chart.items() if v is not None}
    return to_jsonable({
        "question": question,
        "steps": steps,
        "sql": res.sql,
        "columns": an.columns,
        "rows": an.rows[:500],
        "chart": chart,
        "narrative": story["narrative"],
        "recommendations": story["recommendations"],
        "follow_ups": story["follow_ups"],
    })
