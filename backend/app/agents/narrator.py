"""Narrator agent: executive summary, recommendations and follow-up questions."""
from __future__ import annotations

import json
from typing import Any

import pandas as pd

from ..utils import fmt_number, humanize, is_money, pluralize, sentence_label, to_jsonable
from . import llm
from .analyst import Analysis
from .planner import Plan, dataset_context


def _fmt_date(d: Any, grain: str | None) -> str:
    try:
        ts = pd.Timestamp(d)
    except Exception:
        return str(d)
    if grain == "year":
        return f"{ts:%Y}"
    if grain == "quarter":
        return f"Q{(ts.month - 1) // 3 + 1} {ts:%Y}"
    if grain == "month":
        return f"{ts:%b %Y}"
    if grain == "week":
        return f"the week of {ts:%b %d, %Y}"
    return f"{ts:%b %d, %Y}"


def _at(d: Any, grain: str | None) -> str:
    return ("on " if grain == "day" else "in ") + _fmt_date(d, grain)


def _ctx_phrase(plan: Plan) -> str:
    return f" (filtered to {'; '.join(plan.filter_desc)})" if plan.filter_desc else ""


def _label(plan: Plan, alias: str) -> str:
    for m in plan.measures:
        if m.alias == alias:
            return m.label
    if alias == "growth_pct":
        return "growth"
    return humanize(alias)


def _signed_pct(v: float) -> str:
    return f"{'+' if v >= 0 else ''}{v:.1f}%"


# ------------------------------------------------------------------ rule narration
def narrate_rules(plan: Plan, an: Analysis, ds_id: str) -> dict[str, Any]:
    st = an.stats
    k = an.kinds
    sentences: list[str] = []
    recs: list[str] = []
    ctx = dataset_context(ds_id)
    entity = ctx["entity"]

    if an.shape == "empty":
        sentences.append("No records matched this question" + _ctx_phrase(plan) + ".")
        sentences.append("Try widening the filters or the date range.")
        recs.append("Check that the filter values and date range exist in this dataset.")
    elif an.shape == "kpi":
        kp = st.get("kpis") or {}
        parts = []
        used = set()
        for alias, v in kp.items():
            if alias in used:
                continue
            pct_alias = f"{alias}_pct"
            if pct_alias in kp and kp[pct_alias] is not None and v is not None:
                lab = _label(plan, alias)
                subject = sentence_label(lab.replace(humanize(entity), "").strip()) or "matching"
                parts.append(f"**{fmt_number(v, 'number')}** {entity} were {subject}{_ctx_phrase(plan)} — "
                             f"**{kp[pct_alias]:.1f}%** of all {entity}")
                used.update({alias, pct_alias})
                recs.append(f"Set a target for the {subject} rate and review the gap by team, queue or topic weekly.")
                continue
            parts.append(f"{_label(plan, alias)}: **{fmt_number(v, k.get(alias, 'number'))}**")
            used.add(alias)
        if parts and not any("were" in p for p in parts):
            sentences.append("; ".join(parts) + _ctx_phrase(plan) + ".")
        else:
            sentences.append("; ".join(parts) + ".")
        recs.append("Break this figure down by your key dimensions to see what drives it.")
        recs.append("Track it over time to separate one-off swings from structural change.")
    elif an.shape == "time":
        y = st.get("metric")
        kind = st.get("kind", "number")
        label = _label(plan, y) if y else "The metric"
        if st.get("series_dimension"):
            label = _label(plan, plan.measures[0].alias) if plan.measures else humanize(y)
        grain = plan.grain or "month"
        if "change_pct" in st:
            verb = "grew" if st["change_pct"] >= 0 else "declined"
            sentences.append(
                f"{label} {verb} **{abs(st['change_pct']):.1f}%** from {fmt_number(st['first'], kind)} "
                f"{_at(st['start'], grain)} to **{fmt_number(st['last'], kind)}** {_at(st['end'], grain)}"
                f"{_ctx_phrase(plan)}.")
        if st.get("peak") and st.get("periods", 0) > 2:
            sentences.append(
                f"It peaked at **{fmt_number(st['peak']['value'], kind)}** {_at(st['peak']['date'], grain)}, "
                f"with a low of {fmt_number(st['trough']['value'], kind)} {_at(st['trough']['date'], grain)}.")
        if "last_change_pct" in st and st.get("periods", 0) > 2:
            sentences.append(f"The latest {grain} is {_signed_pct(st['last_change_pct'])} vs the previous {grain}.")
        if st.get("series_totals"):
            top = next(iter(st["series_totals"].items()))
            sentences.append(f"**{top[0]}** contributes the most over the period ({fmt_number(top[1], kind)}).")
        hc = st.get("half_change_pct")
        if hc is not None and hc >= 0:
            recs.append(f"Momentum is positive ({_signed_pct(hc)} second half vs first half): plan capacity, "
                        f"budget and inventory for continued growth.")
        elif hc is not None:
            recs.append(f"{label} is trending down ({_signed_pct(hc)} second half vs first half): run a "
                        f"root-cause review on the drivers behind the decline.")
        if st.get("peak"):
            recs.append(f"Study what happened around the peak {_at(st['peak']['date'], grain)} "
                        f"and replicate the conditions that drove it.")
        recs.append("Set an alert for periods that deviate sharply from the forecast band on the Overview page.")
    elif an.shape == "category":
        y = st.get("metric")
        kind = st.get("kind", "number")
        dim = humanize(st.get("dimension", "")).lower()
        if y == "growth_pct":
            window = getattr(plan, "growth_window", 90)
            ml = _label(plan, plan.measures[0].alias) if plan.measures else "the metric"
            top, bot = st["top"], st["bottom"]
            sentences.append(f"**{top['label']}** shows the fastest {sentence_label(ml)} growth at **{_signed_pct(top['value'])}** "
                             f"(last {window} days vs the prior {window}){_ctx_phrase(plan)}.")
            if st.get("count", 0) > 1:
                sentences.append(f"**{bot['label']}** is the weakest at {_signed_pct(bot['value'])}.")
            recs.append(f"Reinvest in **{top['label']}** while momentum is strong.")
            if bot["value"] < 0:
                recs.append(f"Diagnose the decline in **{bot['label']}** before it compounds.")
            recs.append(f"Review {dim} growth monthly to catch inflection points early.")
        else:
            label = _label(plan, y)
            top, bot = st["top"], st["bottom"]
            ascending = plan.order == "asc"
            lead = bot if ascending else top
            share = f", {st['top_share']:.0f}% of the total" if st.get("top_share") is not None and not ascending \
                and _additive(plan, y) else ""
            word = "lowest" if ascending else "highest"
            if str(st.get("dimension", "")).endswith("_bucket"):
                metric_name = humanize(st["dimension"][:-7]).lower()
                sentences.append(f"The most common {metric_name} range starts at **{fmt_number(top['label'], 'number')}**, "
                                 f"holding **{top['value']:,.0f}** records ({st.get('top_share', 0):.0f}% of the total){_ctx_phrase(plan)}.")
                sentences.append(f"Values span {len(an.rows)} buckets; the distribution is "
                                 f"{'highly skewed' if st.get('top_share', 0) > 50 else 'fairly spread out'}.")
                recs.append(f"Segment the long tail of high {metric_name} values; they often drive a disproportionate share of results.")
                return {"narrative": " ".join(sentences), "recommendations": recs[:3],
                        "follow_ups": follow_ups(plan, an, ds_id)}
            sentences.append(f"**{lead['label']}** has the {word} {sentence_label(label)} "
                             f"at **{fmt_number(lead['value'], kind)}**{share}{_ctx_phrase(plan)}.")
            if st.get("second") and not ascending:
                sentences.append(f"It is followed by **{st['second']['label']}** at {fmt_number(st['second']['value'], kind)}.")
            if st.get("count", 0) > 2:
                other = top if ascending else bot
                other_word = "highest" if ascending else "lowest"
                rel = st.get("top_vs_mean_pct" if ascending else "bottom_vs_mean_pct")
                rel_s = f", {abs(rel):.0f}% {'above' if rel >= 0 else 'below'} the average of {fmt_number(st['mean'], kind)}" if rel is not None else ""
                sentences.append(f"**{other['label']}** is the {other_word} at {fmt_number(other['value'], kind)}{rel_s}.")
            if st.get("top3_share") is not None and st.get("count", 0) > 4 and _additive(plan, y):
                sentences.append(f"The top three {pluralize(dim)} account for {st['top3_share']:.0f}% of {sentence_label(label)}.")
            for sec in st.get("secondary") or []:
                if sec.get("top"):
                    sentences.append(f"On {sentence_label(_label(plan, sec['metric']))}, **{sec['top']['label']}** leads "
                                     f"with {fmt_number(sec['top']['value'], sec.get('kind', 'number'))}.")
            if st.get("period_rank"):
                recs.append(f"Study what drove the result in **{top['label']}** and plan campaigns and capacity around it.")
                recs.append(f"Review why **{bot['label']}** lagged and whether it is seasonal.")
                return {"narrative": " ".join(sentences[:5]), "recommendations": recs[:3],
                        "follow_ups": follow_ups(plan, an, ds_id)}
            recs.append(f"Double down on **{top['label']}**: understand what makes it the top {dim} and replicate it.")
            if st.get("count", 0) > 1:
                recs.append(f"Investigate **{bot['label']}**, the weakest {dim}, for pricing, mix or execution issues.")
            if st.get("top_share", 0) and st.get("top_share", 0) > 40 and _additive(plan, y):
                recs.append(f"Concentration risk: {top['label']} drives {st['top_share']:.0f}% of the total; diversify.")
            else:
                recs.append(f"Set {dim}-level targets for {label.lower()} and review them in the weekly business review.")
    elif an.shape == "scatter":
        r = st.get("correlation")
        if r is not None:
            strength = "strong" if abs(r) >= 0.6 else ("moderate" if abs(r) >= 0.3 else "weak")
            direction = "positive" if r >= 0 else "negative"
            sentences.append(f"{humanize(st['y'])} and {humanize(st['x'])} show a **{strength} {direction}** "
                             f"relationship (r = {r:.2f}).")
            sentences.append("Correlation is not causation, but it points to where to look for drivers.")
            recs.append(f"Use {humanize(st['x']).lower()} as a leading indicator for {humanize(st['y']).lower()}"
                        if abs(r) >= 0.5 else "Look for other drivers; this relationship is too weak to plan on.")
        else:
            sentences.append("Not enough data to estimate a relationship.")
    else:
        n = st.get("row_count", 0)
        sentences.append(f"Returned **{n}** {entity if plan.intent in ('list', 'rows') else 'rows'}{_ctx_phrase(plan)}.")
        if st.get("top_value") is not None and st.get("metric"):
            kind = "currency" if is_money(st["metric"]) else "number"
            sentences.append(f"The top record has {humanize(st['metric']).lower()} of "
                             f"**{fmt_number(st['top_value'], kind)}**.")
        recs.append("Aggregate these records by a dimension to spot patterns.")

    for w in plan.warnings:
        if "LLM" not in w:
            sentences.append(f"_Note: {w}_")
    narrative = " ".join(sentences[:5])
    return {"narrative": narrative, "recommendations": recs[:3], "follow_ups": follow_ups(plan, an, ds_id)}


def _additive(plan: Plan, alias: str) -> bool:
    for m in plan.measures:
        if m.alias == alias:
            return m.agg in ("sum", "count")
    return True


def follow_ups(plan: Plan, an: Analysis, ds_id: str) -> list[str]:
    ctx = dataset_context(ds_id)
    r = ctx["roles"]
    entity = ctx["entity"]
    metric = None
    for m in plan.measures:
        if m.column:
            metric = m.column
            break
    metric = metric or r["primary"]
    mname = sentence_label(humanize(metric)) if metric else entity
    if plan.measures and plan.measures[0].column is None:
        mname = sentence_label(plan.measures[0].label)
    cap = mname[:1].upper() + mname[1:]
    dims = [d for d in r["dims"] if d not in r["bools"] and 2 <= len(ctx["values"].get(d, [])) <= 30]
    unused = [d for d in dims if d not in plan.dimensions]
    out: list[str] = []
    if an.shape == "category" and an.stats.get("top") and not an.stats.get("period_rank") \
            and not str(an.stats.get("dimension", "")).endswith("_bucket"):
        top = an.stats["top"]["label"]
        if r["date"]:
            out.append(f"Monthly {mname} trend for {top}")
        if unused:
            out.append(f"Top 5 {pluralize(humanize(unused[0]).lower())} by {mname} for {top}")
    if an.shape == "time":
        if unused:
            out.append(f"{cap} by {humanize(unused[0]).lower()}")
        if dims and r["date"]:
            out.append(f"Which {humanize(dims[0]).lower()} grew {mname} the fastest?")
    if an.shape in ("kpi", "table", "empty", "scatter"):
        if unused:
            out.append(f"{cap} by {humanize(unused[0]).lower()}")
        if r["date"]:
            out.append(f"Monthly {mname} trend")
    for d in unused[1:]:
        if len(out) >= 3:
            break
        cand = f"Top 5 {pluralize(humanize(d).lower())} by {mname}"
        if cand not in out:
            out.append(cand)
    if len(out) < 3 and r["metrics"]:
        other = next((m for m in r["metrics"] if m != metric), None)
        if other and dims:
            out.append(f"Average {humanize(other).lower()} by {humanize(dims[0]).lower()}")
    if len(out) < 3 and r["date"] and not any(o.startswith("Monthly") for o in out):
        out.append(f"Monthly {mname} trend")
    if len(out) < 3:
        out.append(f"How many {entity} are there?" if r["ids"] else f"{cap} in the last 3 months")
    seen, uniq = set(), []
    for q in out:
        if q.lower() not in seen and q.lower() != plan.question.lower():
            seen.add(q.lower())
            uniq.append(q)
    return uniq[:3]


# ------------------------------------------------------------------ LLM narration
NARRATOR_SYSTEM = """You are the Narrator agent: a sharp management consultant writing for a busy executive.
Given a business question, the SQL used, the result rows and computed statistics, write:
- "narrative": 2-5 sentences of markdown. Lead with the answer, quote concrete numbers (formatted nicely,
  e.g. $1.2M, 34%), bold the key figures. No hedging, no restating the SQL.
- "recommendations": 2-3 specific, actionable recommendations grounded in the data.
- "follow_ups": exactly 3 short follow-up questions the executive could ask next about this dataset.
Respond with ONLY a JSON object: {"narrative": "...", "recommendations": ["..."], "follow_ups": ["..."]}"""


def narrate(plan: Plan, an: Analysis, ds_id: str, sql: str) -> tuple[dict[str, Any], str]:
    fallback = narrate_rules(plan, an, ds_id)
    if not llm.llm_available():
        return fallback, "rules"
    try:
        payload = {
            "question": plan.question,
            "sql": sql,
            "columns": an.columns,
            "rows": to_jsonable(an.rows[:40]),
            "total_rows": len(an.rows),
            "stats": to_jsonable(an.stats),
            "value_formats": an.kinds,
            "dataset_columns": [c["name"] for c in dataset_context(ds_id)["meta"]["columns"]],
        }
        data = llm.complete_json(NARRATOR_SYSTEM, json.dumps(payload, default=str)[:30000], max_tokens=2000,
                                 required_keys=("narrative",))
        narrative = data.get("narrative")
        recs = data.get("recommendations") or []
        fups = data.get("follow_ups") or []
        if not isinstance(narrative, str) or not narrative.strip():
            raise llm.LLMError("empty narrative")
        if not isinstance(recs, list) or not isinstance(fups, list):
            raise llm.LLMError("bad list fields")
        return {
            "narrative": narrative.strip(),
            "recommendations": [str(x) for x in recs if str(x).strip()][:4] or fallback["recommendations"],
            "follow_ups": [str(x) for x in fups if str(x).strip()][:3] or fallback["follow_ups"],
        }, "llm"
    except llm.LLMError:
        return fallback, "rules"
