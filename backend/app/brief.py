"""Executive Brief: LLM-written when available, otherwise templated from insights."""
from __future__ import annotations

import datetime as _dt
import json
from typing import Any

from .agents import llm
from .insights import compute_insights, metric_kind
from .store import store
from .utils import fmt_number, humanize, to_jsonable


def _kpi_line(k: dict[str, Any]) -> str:
    val = fmt_number(k["value"], k["format"])
    ch = k.get("change_pct")
    chs = f" ({'+' if ch >= 0 else ''}{ch:.1f}% vs prior period)" if ch is not None else ""
    return f"- **{k['label']}**: {val} — {k['period'].lower()}{chs}"


def templated_brief(ds_id: str) -> dict[str, Any]:
    meta = store.get(ds_id)
    ins = compute_insights(ds_id)
    kpis = ins["kpis"]
    hl = ins["highlights"]
    summary = " ".join(hl[:3]) if hl else f"{meta['name']} contains {meta['rows']:,} rows."

    sections = []
    if kpis:
        sections.append({"heading": "Performance snapshot", "body": "\n".join(_kpi_line(k) for k in kpis)})

    trend_lines = []
    t = ins.get("trend")
    if t and t["series"]:
        s = t["series"]
        tk = metric_kind(t["metric"])
        first, last = s[0]["value"], s[-1]["value"]
        n = len(s)
        h = max(n // 2, 1)
        a = sum(x["value"] for x in s[:h]) / h
        b = sum(x["value"] for x in s[h:]) / max(n - h, 1)
        if a:
            trend_lines.append(f"- {humanize(t['metric'])} averaged {fmt_number(b, tk)} per period in the second half of "
                               f"the range vs {fmt_number(a, tk)} in the first half ({(b - a) / abs(a) * 100:+.1f}%).")
        trend_lines.append(f"- Series covers {s[0]['date']} to {s[-1]['date']} ({n} periods); "
                           f"latest value {fmt_number(last, tk)} vs {fmt_number(first, tk)} at the start.")
    fc = ins.get("forecast")
    if fc and fc["forecast"]:
        vals = [x["value"] for x in fc["forecast"]]
        fk = metric_kind(fc["metric"])
        trend_lines.append(f"- The {fc['method']} forecast projects {humanize(fc['metric']).lower()} averaging "
                           f"{fmt_number(sum(vals) / len(vals), fk)} per period over the next {len(vals)} periods "
                           f"(range {fmt_number(min(x['lower'] for x in fc['forecast']), fk)}–"
                           f"{fmt_number(max(x['upper'] for x in fc['forecast']), fk)}).")
    if trend_lines:
        sections.append({"heading": "Trends & outlook", "body": "\n".join(trend_lines)})

    bd_lines = []
    for b in ins["breakdowns"]:
        if not b["data"]:
            continue
        bk = metric_kind(b["metric"])
        top = b["data"][0]
        bottom = b["data"][-1]
        tot = sum(x["value"] for x in b["data"]) or 1
        bd_lines.append(f"- **{humanize(b['dimension'])}**: {top['label']} leads with {fmt_number(top['value'], bk)} "
                        f"({top['value'] / tot * 100:.0f}% of shown total); {bottom['label']} trails at "
                        f"{fmt_number(bottom['value'], bk)}.")
    if bd_lines:
        sections.append({"heading": "Where the value is", "body": "\n".join(bd_lines)})

    an = ins["anomalies"]
    if an:
        lines = [f"- {a['direction'].title()} in {humanize(a['metric']).lower()} on {a['date']}: "
                 f"{fmt_number(a['value'], metric_kind(a['metric']))} vs {fmt_number(a['expected'], metric_kind(a['metric']))} expected (z = {a['zscore']})."
                 for a in an[:5]]
        sections.append({"heading": "Risks & anomalies", "body": "\n".join(lines)})
    else:
        sections.append({"heading": "Risks & anomalies", "body": "- No statistically unusual periods detected."})

    actions = []
    if ins["breakdowns"] and ins["breakdowns"][0]["data"]:
        b = ins["breakdowns"][0]
        actions.append(f"- Protect and grow **{b['data'][0]['label']}**, the leading {humanize(b['dimension']).lower()}.")
        if len(b["data"]) > 1:
            actions.append(f"- Build an improvement plan for **{b['data'][-1]['label']}**, the weakest "
                           f"{humanize(b['dimension']).lower()}.")
    if an:
        actions.append("- Run a root-cause review on the flagged anomalies and add monitoring alerts.")
    if kpis and kpis[0].get("change_pct") is not None and kpis[0]["change_pct"] < 0:
        actions.append(f"- {kpis[0]['label']} is down {abs(kpis[0]['change_pct']):.1f}% over the last 30 days: "
                       f"review pipeline and pricing.")
    actions.append("- Use **Ask AI** to drill into any of the figures above.")
    sections.append({"heading": "Recommended actions", "body": "\n".join(actions)})
    return {"summary": summary, "sections": sections, "kpis": kpis}


BRIEF_SYSTEM = """You write crisp executive briefs for business leaders from pre-computed analytics.
Use only the numbers provided. Respond with ONLY a JSON object:
{"title": "...", "summary": "2-4 sentence markdown summary leading with the most important finding",
 "sections": [{"heading": "...", "body": "markdown (bullets welcome)"}]}
Write 4-5 sections such as Performance, Trends & Outlook, Drivers, Risks & Anomalies, Recommended Actions."""


def build_brief(ds_id: str) -> dict[str, Any]:
    meta = store.get(ds_id)
    base = templated_brief(ds_id)
    out = {
        "title": f"Executive Brief — {meta['name']}",
        "generated_at": _dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "summary": base["summary"],
        "sections": base["sections"],
        "kpis": base["kpis"],
    }
    if llm.llm_available():
        try:
            ins = compute_insights(ds_id)
            payload = {"dataset": meta["name"], "rows": meta["rows"],
                       "columns": [c["name"] for c in meta["columns"]],
                       "insights": {**ins, "trend": None,
                                    "forecast": {**ins["forecast"], "history": None} if ins.get("forecast") else None}}
            data = llm.complete_json(BRIEF_SYSTEM, json.dumps(to_jsonable(payload), default=str)[:40000],
                                     max_tokens=4000, required_keys=("summary", "sections"))
            secs = data.get("sections")
            if isinstance(secs, list) and secs and all(isinstance(s, dict) and "heading" in s and "body" in s for s in secs):
                out["summary"] = str(data["summary"])
                out["sections"] = [{"heading": str(s["heading"]), "body": str(s["body"])} for s in secs]
                if data.get("title"):
                    out["title"] = str(data["title"])
        except llm.LLMError:
            pass
    return to_jsonable(out)
