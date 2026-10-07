"""Planner agent: understands the question and maps it onto the dataset schema.

The rule-based planner is the backbone of the public demo (no API key), so it is
deliberately thorough: column / synonym / value matching, intents (total, top-N,
trend, average, count, compare, growth, share, distribution, correlation, list),
derived business metrics (conversion rate, CTR, ROAS, margin, AOV, ...), boolean
flags (resolved / churned) and relative date filters.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from ..insights import entity_label, roles
from ..store import store
from ..utils import humanize, is_additive, is_money
from . import llm

# ------------------------------------------------------------------ vocab
NUM_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
             "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "fifteen": 15, "twenty": 20}
TOP_WORDS = ("top", "best", "highest", "most", "largest", "biggest", "leading", "greatest", "strongest",
             "maximum", "max", "better", "higher", "more", "longest", "slowest", "positive")
BOTTOM_WORDS = ("bottom", "worst", "lowest", "least", "smallest", "weakest", "minimum", "min", "fewest",
                "poorest", "worse", "lower", "less", "shortest", "negative", "underperforming", "lagging")
TREND_WORDS = ("trend", "trends", "over time", "time series", "timeline", "history", "historical",
               "evolution", "progression", "seasonality", "seasonal", "trajectory", "by date", "per date")
GRAIN_PATTERNS = [
    ("day", r"\b(daily|by day|per day|each day|day by day|day over day|days)\b"),
    ("week", r"\b(weekly|by week|per week|each week|week over week|wow|weeks)\b"),
    ("quarter", r"\b(quarterly|by quarter|per quarter|each quarter|quarter over quarter|qoq|quarters)\b"),
    ("year", r"\b(yearly|annual|annually|by year|per year|each year|year over year|yoy|years)\b"),
    ("month", r"\b(monthly|by month|per month|each month|month over month|mom|months|month)\b"),
]
GROWTH_WORDS = ("growth", "grow", "grew", "growing", "increase", "increased", "increasing", "decrease",
                "decreased", "decline", "declined", "declining", "change", "changed", "improve", "improved",
                "improving", "drop", "dropped", "momentum", "fastest", "accelerat")
AVG_WORDS = ("average", "avg", "mean", "typical", "per order", "per call", "per transaction", "per row",
             "per record", "per customer", "per day")
SUM_WORDS = ("total", "sum", "overall", "combined", "cumulative", "aggregate", "gross")
MEDIAN_WORDS = ("median",)
COUNT_WORDS = ("how many", "number of", "count", "# of", "no of", "quantity of records", "volume of")
SHARE_WORDS = ("share", "percentage", "percent", "proportion", "breakdown", "split", "distribution",
               "mix", "composition", "contribution", "contribute", "pie", "portion")
COMPARE_WORDS = ("compare", "comparison", "vs", "versus", "against", "compared", "difference between")
LIST_WORDS = ("show", "list", "display", "give me", "sample", "examples", "raw", "rows", "records",
              "table", "preview")
CORR_WORDS = ("correlation", "correlate", "correlated", "relationship", "relation", "related", "impact of",
              "effect of", "scatter")
RATE_WORDS = ("rate", "percent", "percentage", "share", "ratio", "proportion", "pct", "how often", "fraction")
NEG_WORDS = ("not", "un", "non", "no", "never", "without", "failed", "open")

# keyword -> candidate column-name fragments (first fragment that exists wins)
SYNONYMS: dict[str, tuple[str, ...]] = {
    "sales": ("revenue", "sales", "amount"),
    "revenue": ("revenue", "sales", "amount", "income"),
    "income": ("revenue", "income", "profit"),
    "turnover": ("revenue", "sales"),
    "earnings": ("profit", "revenue", "earning"),
    "money": ("revenue", "amount"),
    "gmv": ("revenue", "gmv", "sales"),
    "satisfaction": ("csat", "satisfaction", "nps", "rating", "score"),
    "customer satisfaction": ("csat", "satisfaction", "nps"),
    "rating": ("rating", "csat", "score"),
    "ratings": ("rating", "csat", "score"),
    "score": ("score", "csat", "rating"),
    "handle time": ("duration", "handle"),
    "handling time": ("duration", "handle"),
    "call length": ("duration",),
    "call duration": ("duration",),
    "length": ("duration", "length"),
    "talk time": ("duration",),
    "aht": ("duration",),
    "long": ("duration",),
    "hold time": ("wait", "hold"),
    "hold": ("wait", "hold"),
    "wait time": ("wait",),
    "waiting": ("wait",),
    "quantity": ("units", "quantity", "qty"),
    "qty": ("units", "quantity", "qty"),
    "volume": ("units", "quantity", "volume"),
    "items": ("units", "quantity", "items"),
    "units sold": ("units", "quantity"),
    "price": ("unit_price", "price"),
    "prices": ("unit_price", "price"),
    "budget": ("spend", "budget", "cost"),
    "ad spend": ("spend", "cost"),
    "spending": ("spend", "cost"),
    "spent": ("spend", "cost"),
    "expense": ("cost", "spend", "expense"),
    "expenses": ("cost", "spend", "expense"),
    "costs": ("cost", "spend"),
    "mood": ("sentiment",),
    "tone": ("sentiment",),
    "signups": ("conversions", "signups", "leads"),
    "sign ups": ("conversions", "signups", "leads"),
    "acquisitions": ("conversions",),
    "converted": ("conversions",),
    "traffic": ("clicks", "sessions", "visits"),
    "visits": ("visits", "sessions", "clicks"),
    "views": ("impressions", "views"),
    "reach": ("impressions", "reach"),
    "product line": ("product_category", "category"),
    "product lines": ("product_category", "category"),
    "categories": ("category",),
    "segments": ("segment",),
    "customer type": ("customer_segment", "segment"),
    "customer types": ("customer_segment", "segment"),
    "customers": ("customer", "segment"),
    "customer": ("customer", "segment"),
    "market": ("country", "market", "region"),
    "markets": ("country", "market", "region"),
    "nation": ("country",),
    "nations": ("country",),
    "geo": ("region", "country"),
    "geography": ("region", "country"),
    "territory": ("region", "territory"),
    "territories": ("region", "territory"),
    "area": ("region", "area"),
    "areas": ("region", "area"),
    "source": ("channel", "source"),
    "sources": ("channel", "source"),
    "medium": ("channel", "medium"),
    "platform": ("channel", "platform"),
    "platforms": ("channel", "platform"),
    "marketing channel": ("channel",),
    "rep": ("agent", "rep", "employee"),
    "reps": ("agent", "rep", "employee"),
    "representative": ("agent", "rep"),
    "representatives": ("agent", "rep"),
    "staff": ("agent", "employee"),
    "employee": ("agent", "employee"),
    "employees": ("agent", "employee"),
    "people": ("agent", "employee"),
    "issue": ("topic", "issue", "reason"),
    "issues": ("topic", "issue", "reason"),
    "reason": ("topic", "reason"),
    "reasons": ("topic", "reason"),
    "subject": ("topic", "subject"),
    "call reason": ("topic",),
    "department": ("queue", "department", "team"),
    "departments": ("queue", "department", "team"),
    "team": ("queue", "team", "department"),
    "teams": ("queue", "team", "department"),
    "line": ("queue",),
    "resolution": ("resolved",),
    "resolutions": ("resolved",),
    "fixed": ("resolved",),
    "solved": ("resolved",),
    "churn": ("churned", "churn"),
    "sku": ("product", "sku"),
    "skus": ("product", "sku"),
    "item": ("product", "item"),
}
GENERIC_TOKENS = {"id", "sec", "secs", "seconds", "date", "num", "no", "per", "of", "the", "attributed",
                  "total", "avg", "count", "pct", "code", "name", "type", "at", "on", "in", "is", "value",
                  "amount", "time", "ts", "dt", "key", "number", "unit"}

ENTITY_GENERIC = ("rows", "records", "row", "record", "entries", "transactions", "transaction", "deals")


@dataclass
class Measure:
    sql: str
    alias: str
    label: str
    kind: str = "number"  # currency | number | percent
    agg: str = "sum"
    column: str | None = None
    # for growth intent: per-window expression builder
    growth_inner: str | None = None


@dataclass
class Plan:
    question: str
    intent: str = "kpi"  # kpi | group | trend | growth | list | distribution | correlation
    measures: list[Measure] = field(default_factory=list)
    dimensions: list[str] = field(default_factory=list)
    date_col: str | None = None
    grain: str | None = None
    order: str | None = None  # desc | asc
    limit: int | None = None
    filters: list[str] = field(default_factory=list)  # SQL predicates
    filter_desc: list[str] = field(default_factory=list)
    share: bool = False
    corr_cols: list[str] = field(default_factory=list)
    order_by_metric: str | None = None
    warnings: list[str] = field(default_factory=list)
    rank_periods: bool = False
    matched: list[str] = field(default_factory=list)
    source: str = "rules"
    llm_notes: str | None = None

    def summary(self) -> str:
        parts = [f"Intent: {self.intent.replace('_', ' ')}"]
        if self.measures:
            parts.append("measures: " + ", ".join(m.label for m in self.measures))
        if self.dimensions:
            parts.append("by " + ", ".join(self.dimensions))
        if self.grain:
            parts.append(f"grain: {self.grain}")
        if self.order:
            parts.append(("top" if self.order == "desc" else "bottom") + (f" {self.limit}" if self.limit else ""))
        if self.filter_desc:
            parts.append("filters: " + "; ".join(self.filter_desc))
        return " · ".join(parts)


# ------------------------------------------------------------------ helpers
def q_ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def q_lit(v: Any) -> str:
    if isinstance(v, bool):
        return "TRUE" if v else "FALSE"
    if isinstance(v, (int, float)):
        return repr(v)
    return "'" + str(v).replace("'", "''") + "'"


def normalize(text: str) -> str:
    t = text.lower()
    t = t.replace("%", " percent ").replace("&", " and ").replace("#", " number ")
    t = re.sub(r"[’'`]s\b", "", t)
    t = re.sub(r"[^a-z0-9]+", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def _plural_forms(w: str) -> list[str]:
    forms = [w]
    if w.endswith("y") and not w.endswith(("ay", "ey", "oy", "uy")):
        forms.append(w[:-1] + "ies")
    elif w.endswith(("s", "x", "ch", "sh")):
        forms.append(w + "es")
    else:
        forms.append(w + "s")
    return forms


def _find_all(alias: str, q: str) -> list[tuple[int, int]]:
    if not alias:
        return []
    pat = r"(?<![a-z0-9])" + re.escape(alias) + r"(?![a-z0-9])"
    return [(m.start(), m.end()) for m in re.finditer(pat, q)]


def _has(q: str, words) -> bool:
    return any(_find_all(w, q) for w in words)


def dataset_context(ds_id: str) -> dict[str, Any]:
    meta = store.get(ds_id)
    cache = store.cache(ds_id)
    if "context" in cache:
        return cache["context"]
    r = roles(meta)
    values: dict[str, list[Any]] = {}
    for c in meta["columns"]:
        if c["semantic"] == "dimension" and c["dtype"] != "boolean":
            try:
                cnt = store.query_df(f"SELECT COUNT(DISTINCT {q_ident(c['name'])}) AS n FROM {q_ident(ds_id)}")["n"][0]
                if cnt <= 200:
                    vals = store.query_df(
                        f"SELECT DISTINCT {q_ident(c['name'])} AS v FROM {q_ident(ds_id)} WHERE {q_ident(c['name'])} IS NOT NULL")["v"]
                    values[c["name"]] = [v for v in vals.tolist()]
            except Exception:
                pass
    date_range = {}
    for d in r["dates"]:
        try:
            row = store.query_df(f"SELECT MIN({q_ident(d)}) AS lo, MAX({q_ident(d)}) AS hi FROM {q_ident(ds_id)}")
            date_range[d] = (pd.Timestamp(row["lo"][0]), pd.Timestamp(row["hi"][0]))
        except Exception:
            pass
    ctx = {"meta": meta, "roles": r, "values": values, "date_range": date_range,
           "entity": entity_label(meta)}
    cache["context"] = ctx
    return ctx


def _resolve_fragment(frags: tuple[str, ...], cols: list[str]) -> str | None:
    for f in frags:
        for c in cols:
            if c.lower() == f:
                return c
        for c in cols:
            if f in c.lower():
                return c
    return None


# ------------------------------------------------------------------ derived metrics
def _derived_defs(cols: list[str]) -> list[dict[str, Any]]:
    """Business ratios that can be computed from the available columns."""
    def find(*frags):
        return _resolve_fragment(tuple(frags), cols)

    rev = find("revenue", "sales", "amount")
    profit = find("profit")
    spend = find("spend", "cost", "budget")
    clicks = find("clicks")
    impressions = find("impressions", "views")
    conversions = find("conversions", "orders", "signups")
    leads = find("leads")
    out = []

    def add(keys, alias, label, kind, num, den, scale=1.0, minus=None):
        if num and den:
            if minus:
                sql = f"{scale} * (SUM({q_ident(num)}) - SUM({q_ident(minus)})) / NULLIF(SUM({q_ident(den)}), 0)"
            else:
                sql = f"{scale} * SUM({q_ident(num)}) / NULLIF(SUM({q_ident(den)}), 0)"
            out.append({"keys": keys, "alias": alias, "label": label, "kind": kind, "sql": sql,
                        "num": num, "den": den, "scale": scale, "minus": minus})

    add(("lead conversion rate", "lead to conversion", "lead conversion"), "lead_conversion_rate",
        "Lead Conversion Rate", "percent", conversions, leads, 100.0)
    add(("conversion rate", "conversion rates", "conversion", "conversions rate", "cvr", "convert", "converts",
         "converting"), "conversion_rate", "Conversion Rate", "percent", conversions, clicks or leads, 100.0)
    add(("click through rate", "click through", "clickthrough rate", "ctr"), "ctr", "CTR", "percent",
        clicks, impressions, 100.0)
    add(("cost per click", "cpc"), "cpc", "Cost per Click", "currency", spend, clicks)
    add(("cost per acquisition", "cost per conversion", "cpa", "acquisition cost", "cac"), "cpa",
        "Cost per Acquisition", "currency", spend, conversions)
    add(("cost per lead", "cpl"), "cpl", "Cost per Lead", "currency", spend, leads)
    if spend and rev and spend != rev:
        add(("return on ad spend", "roas", "efficient", "efficiency", "return"), "roas", "ROAS", "number", rev, spend)
        add(("roi", "return on investment"), "roi", "ROI", "percent", rev, spend, 100.0, minus=spend)
    add(("lead rate",), "lead_rate", "Lead Rate", "percent", leads, clicks, 100.0)
    if profit and rev and profit != rev:
        add(("profit margin", "margin", "margins", "profitability", "profitable"), "profit_margin",
            "Profit Margin", "percent", profit, rev, 100.0)
    if rev:
        # average order value: revenue per row
        out.append({"keys": ("average order value", "aov", "order value", "basket size", "average deal size",
                             "revenue per order", "sales per order", "average sale", "ticket size"),
                    "alias": "avg_order_value", "label": "Average Order Value", "kind": "currency",
                    "sql": f"SUM({q_ident(rev)}) / NULLIF(COUNT(*), 0)", "num": rev, "den": None})
    return out


# ------------------------------------------------------------------ rule planner
def plan_rules(ds_id: str, question: str) -> Plan:
    ctx = dataset_context(ds_id)
    meta, r = ctx["meta"], ctx["roles"]
    cols = [c["name"] for c in meta["columns"]]
    sem = {c["name"]: c["semantic"] for c in meta["columns"]}
    dtypes = {c["name"]: c["dtype"] for c in meta["columns"]}
    q = normalize(question)
    plan = Plan(question=question)
    entity = ctx["entity"]
    entity_words = set(_plural_forms(entity.rstrip("s"))) | {entity} | set(ENTITY_GENERIC)
    for tok in normalize(meta["id"]).split() + normalize(meta.get("name", "")).split():
        if len(tok) > 3 and tok not in ("sample", "data", "dataset"):
            entity_words.add(tok)

    # ---------- candidate matches: (start, end, priority, kind, target)
    cands: list[tuple[int, int, int, str, Any]] = []

    derived = _derived_defs(cols)
    for d in derived:
        for k in d["keys"]:
            for s, e in _find_all(k, q):
                cands.append((s, e, 6, "derived", d))

    for c in cols:
        cl = c.lower()
        phrase = cl.replace("_", " ")
        aliases = set(_plural_forms(phrase)) | {cl}
        for a in aliases:
            for s, e in _find_all(a, q):
                cands.append((s, e, 5, "col", c))
        toks = [t for t in phrase.split() if t not in GENERIC_TOKENS and len(t) > 2]
        if len(phrase.split()) > 1 and sem[c] in ("metric", "dimension"):
            for t in toks:
                for a in _plural_forms(t):
                    for s, e in _find_all(a, q):
                        cands.append((s, e, 1, "col", c))
        # stem for booleans: resolved -> resolve
        if dtypes[c] == "boolean" and cl.endswith("ed"):
            for s, e in _find_all(cl[:-1], q) + _find_all(cl[:-2], q):
                cands.append((s, e, 4, "col", c))
            for s, e in _find_all("un" + cl, q):
                cands.append((s, e, 7, "boolneg", c))

    for key, frags in SYNONYMS.items():
        target = _resolve_fragment(frags, cols)
        if target is None:
            continue
        for s, e in _find_all(key, q):
            cands.append((s, e, 3, "col", target))

    for col, vals in ctx["values"].items():
        for v in vals:
            nv = normalize(str(v))
            if len(nv) < 2 or nv.isdigit() or nv in ("yes", "no", "true", "false", "none", "other", "all", "na"):
                continue
            for s, e in _find_all(nv, q):
                # single-word values that equal a column alias are weaker
                pri = 4 if " " in nv else 2
                cands.append((s, e, pri, "value", (col, v)))

    # greedy non-overlapping selection: longer span first, then priority
    cands.sort(key=lambda x: (-(x[1] - x[0]), -x[2], x[0]))
    taken: list[tuple[int, int]] = []
    chosen: list[tuple[int, int, int, str, Any]] = []
    for cnd in cands:
        s, e = cnd[0], cnd[1]
        if any(not (e <= ts or s >= te) for ts, te in taken):
            continue
        taken.append((s, e))
        chosen.append(cnd)
    chosen.sort(key=lambda x: x[0])

    metrics: list[str] = []
    dims: list[str] = []
    dates: list[str] = []
    bool_cols: list[tuple[str, bool]] = []
    derived_hits: list[dict[str, Any]] = []
    value_filters: dict[str, list[Any]] = {}
    for s, e, pri, kind, target in chosen:
        if kind == "derived":
            if target not in derived_hits:
                derived_hits.append(target)
            plan.matched.append(f"'{q[s:e]}' → {target['label']}")
        elif kind == "value":
            col, v = target
            value_filters.setdefault(col, [])
            if v not in value_filters[col]:
                value_filters[col].append(v)
            plan.matched.append(f"'{q[s:e]}' → {col} = {v}")
        elif kind == "boolneg":
            bool_cols.append((target, False))
            plan.matched.append(f"'{q[s:e]}' → {target} = false")
        else:
            c = target
            plan.matched.append(f"'{q[s:e]}' → {c}")
            if dtypes[c] == "boolean":
                before = q[max(0, s - 12):s]
                neg = bool(re.search(r"\b(not|non|never|no|without|failed to be)\s*$", before))
                bool_cols.append((c, not neg))
            elif sem[c] == "metric":
                if c not in metrics:
                    metrics.append(c)
            elif sem[c] == "date":
                if c not in dates:
                    dates.append(c)
            elif sem[c] in ("dimension", "text"):
                if c not in dims:
                    dims.append(c)
            elif sem[c] == "id":
                if _has(q, ("unique", "distinct", "different")) and c not in metrics:
                    metrics.append(c)

    # a dimension mentioned only as part of a value phrase should not group
    # e.g. "sales queue" -> queue = Sales  (keep queue only if also "by queue")
    for col in list(value_filters):
        if col in dims:
            by_pat = r"\b(by|per|each|across|every|which|what|all)\s+(the\s+)?" + re.escape(col.replace("_", " ").split()[-1])
            if not re.search(by_pat, q) and len(value_filters[col]) == 1:
                dims.remove(col)

    # ---------- date column & filters
    date_col = dates[0] if dates else r["date"]
    plan.date_col = date_col
    if date_col and date_col in ctx["date_range"]:
        _date_filters(q, plan, date_col, ctx["date_range"][date_col])

    for col, vals in value_filters.items():
        if len(vals) == 1:
            plan.filters.append(f"{q_ident(col)} = {q_lit(vals[0])}")
        else:
            plan.filters.append(f"{q_ident(col)} IN ({', '.join(q_lit(v) for v in vals)})")
        plan.filter_desc.append(f"{col} in {', '.join(map(str, vals))}" if len(vals) > 1 else f"{col} = {vals[0]}")
        if len(vals) > 1 and col not in dims and _has(q, COMPARE_WORDS + ("and", "or")):
            dims.insert(0, col)  # "compare Europe vs Asia" -> group by region

    # ---------- intent words
    is_count = _has(q, COUNT_WORDS)
    is_avg = _has(q, AVG_WORDS)
    is_sum = _has(q, SUM_WORDS)
    is_median = _has(q, MEDIAN_WORDS)
    is_top = _has(q, TOP_WORDS)
    is_bottom = _has(q, BOTTOM_WORDS)
    is_trend = _has(q, TREND_WORDS)
    is_growth = _has(q, GROWTH_WORDS)
    is_share = _has(q, SHARE_WORDS)
    is_corr = _has(q, CORR_WORDS)
    is_list = _has(q, LIST_WORDS)
    is_rate = _has(q, RATE_WORDS)
    # relative date filters ("last 3 months", "this year") are not grains
    q_grain = re.sub(r"\b(last|past|previous|recent|this|current|next)\s+(\d+\s+|(?:" + "|".join(NUM_WORDS)
                     + r")\s+)?(day|week|month|quarter|year)s?\b", " ", q)
    q_grain = re.sub(r"\b(year to date|ytd)\b", " ", q_grain)
    grain = None
    for g, pat in GRAIN_PATTERNS:
        if re.search(pat, q_grain):
            grain = g
            break
    plan.rank_periods = bool(grain and (is_top or is_bottom) and (_has(q, ("which", "what")) or limit_hint(q)))
    years = sorted(set(re.findall(r"\b((?:19|20)\d{2})\b", q)))
    if len(years) >= 2 and date_col:
        grain = grain or "year"

    # explicit N
    limit = None
    m = re.search(r"\b(top|bottom|best|worst|highest|lowest|first|largest|biggest|smallest|leading|worst)\s+(\d+|"
                  + "|".join(NUM_WORDS) + r")\b", q)
    if not m:
        m2 = re.search(r"\b(\d+|" + "|".join(NUM_WORDS) + r")\s+(best|worst|top|highest|lowest|largest|biggest|smallest|most|least|leading)\b", q)
        if m2:
            tok = m2.group(1)
            limit = int(tok) if tok.isdigit() else NUM_WORDS[tok]
    else:
        tok = m.group(2)
        limit = int(tok) if tok.isdigit() else NUM_WORDS[tok]
    if limit is not None and str(limit) in years:
        limit = None
    if limit is not None:
        limit = max(1, min(limit, 100))

    order = None
    if is_bottom and not is_top:
        order = "asc"
    elif is_top:
        order = "desc"
    elif is_bottom:
        # both: whichever appears first
        first_top = min((s for w in TOP_WORDS for s, _ in _find_all(w, q)), default=10**6)
        first_bot = min((s for w in BOTTOM_WORDS for s, _ in _find_all(w, q)), default=10**6)
        order = "asc" if first_bot < first_top else "desc"

    # ---------- measures
    def metric_measure(c: str) -> Measure:
        kind = "currency" if is_money(c) else "number"
        if is_median:
            agg = "median"
        elif is_avg:
            agg = "avg"
        elif is_sum:
            agg = "sum"
        elif plan_intent_hint == "extreme":
            agg = "max" if order == "desc" else "min"
        else:
            agg = "sum" if is_additive(c) else "avg"
        if sem.get(c) == "id":
            return Measure(f"COUNT(DISTINCT {q_ident(c)})", f"unique_{c}", f"Unique {humanize(c)}", "number", "count", c)
        fn = {"sum": "SUM", "avg": "AVG", "median": "MEDIAN", "max": "MAX", "min": "MIN"}[agg]
        alias = c if agg == "sum" else f"{agg}_{c}"
        label = humanize(c) if agg == "sum" else f"{ {'avg': 'Avg', 'median': 'Median', 'max': 'Max', 'min': 'Min'}[agg] } {humanize(c)}"
        return Measure(f"{fn}({q_ident(c)})", alias, label.strip(), kind, agg, c)

    plan_intent_hint = None
    if not dims and not grain and not is_trend and (is_top or is_bottom) and metrics and not is_growth:
        # "what is the highest unit price" vs "top 10 orders by revenue"
        if limit is None and not _has(q, tuple(entity_words)):
            plan_intent_hint = "extreme"

    count_alias = re.sub(r"[^a-z0-9_]", "_", entity.lower()) or "records"
    count_measure = Measure("COUNT(*)", count_alias, humanize(entity), "number", "count")

    measures: list[Measure] = []
    for d in derived_hits:
        measures.append(Measure(d["sql"], d["alias"], d["label"], d["kind"], "ratio"))
    for c in metrics:
        # skip metrics that are just components of a matched derived ratio
        if any(c in (d.get("num"), d.get("den")) for d in derived_hits) and not _has(q, (c.replace("_", " "),)):
            continue
        measures.append(metric_measure(c))

    bool_filter_sql = []
    for c, val in bool_cols:
        pred = f"{q_ident(c)} = {q_lit(val)}"
        if is_rate or (not is_count and not measures and not dims and not grain):
            alias = f"{c}_rate" if val else f"not_{c}_rate"
            expr = f"100.0 * AVG(CASE WHEN {pred} THEN 1.0 ELSE 0.0 END)"
            measures.insert(0, Measure(expr, alias, f"{humanize(c)} Rate" if val else f"Not {humanize(c)} Rate",
                                       "percent", "ratio"))
        elif is_count:
            label = f"{humanize(c)} {humanize(entity)}" if val else f"Not {humanize(c)} {humanize(entity)}"
            alias = f"{c}_{count_alias}" if val else f"not_{c}_{count_alias}"
            measures.insert(0, Measure(f"COUNT(*) FILTER (WHERE {pred})", alias, label, "number", "count"))
            measures.insert(1, Measure(f"100.0 * AVG(CASE WHEN {pred} THEN 1.0 ELSE 0.0 END)",
                                       f"{alias}_pct", f"% {humanize(c)}" if val else f"% Not {humanize(c)}",
                                       "percent", "ratio"))
        else:
            bool_filter_sql.append(pred)
            plan.filter_desc.append(f"{c} = {str(val).lower()}")
    plan.filters.extend(bool_filter_sql)

    if not measures:
        if is_count or _has(q, tuple(entity_words)) or not r["primary"]:
            measures = [count_measure]
        elif r["primary"] and is_money(r["primary"]):
            measures = [metric_measure(r["primary"])]
            if q.strip() and not is_list:
                plan.warnings.append(f"No metric named in the question; using {r['primary']}.")
        else:
            measures = [count_measure]
    elif is_count and not any(m.agg == "count" for m in measures) and not metrics and not derived_hits:
        measures.insert(0, count_measure)
    if is_count and metrics and not derived_hits and not bool_cols and "how many" in q:
        # "how many units were sold" -> sum of units (keep), "how many orders" -> count
        pass

    if _has(q, ("unknown", "conversion", "conversions", "ctr", "roas")) and not derived_hits and not metrics:
        plan.warnings.append("Some requested metrics are not available in this dataset.")

    plan.measures = measures[:4]
    plan.dimensions = dims[:2]
    plan.order = order
    plan.limit = limit
    plan.share = is_share

    # ---------- intent decision
    if is_corr and len([m for m in metrics]) >= 2 and not dims:
        plan.intent = "correlation"
        plan.corr_cols = metrics[:2]
        return plan
    if is_list and not metrics and not derived_hits and not dims and not grain and not is_trend and not is_count \
            and not is_avg and not is_sum:
        plan.intent = "list"
        plan.limit = limit or 50
        return plan
    if _has(q, ("distribution", "histogram", "spread", "range of")) and metrics and not dims and not grain:
        plan.intent = "distribution"
        plan.corr_cols = metrics[:1]
        return plan
    if is_growth and dims and date_col and not grain:
        plan.intent = "growth"
        plan.order = order or "desc"
        return plan
    if (grain or is_trend or (is_growth and not dims) or (dates and not dims)) and date_col:
        plan.intent = "trend"
        plan.grain = grain or _default_grain(ctx["date_range"].get(date_col))
        if len(years) >= 2 and not any("year" in f for f in plan.filter_desc):
            plan.filters.append(f"YEAR({q_ident(date_col)}) IN ({', '.join(years)})")
            plan.filter_desc.append(f"year in {', '.join(years)}")
        return plan
    if dims:
        plan.intent = "group"
        if order is None:
            plan.order = "desc"
        if plan.limit is None and (is_top or is_bottom) and _has(q, ("top", "bottom")):
            plan.limit = 10
        return plan
    if (order and limit) or (order and _has(q, tuple(entity_words)) and (metrics or derived_hits)):
        # row-level ranking: "10 largest orders by revenue"
        if metrics:
            plan.intent = "rows"
            plan.order_by_metric = metrics[0]
            plan.limit = limit or 10
            return plan
    plan.intent = "kpi"
    return plan


def limit_hint(q: str) -> bool:
    return bool(re.search(r"\b(top|bottom|best|worst)\s+(\d+|" + "|".join(NUM_WORDS) + r")\b", q))


def _default_grain(rng: tuple[pd.Timestamp, pd.Timestamp] | None) -> str:
    if not rng:
        return "month"
    days = (rng[1] - rng[0]).days
    if days <= 45:
        return "day"
    if days <= 200:
        return "week"
    if days <= 365 * 6:
        return "month"
    return "year"


MONTHS = ["january", "february", "march", "april", "may", "june", "july", "august", "september",
          "october", "november", "december"]
MONTH_ABBR = [m[:3] for m in MONTHS]


def _date_filters(q: str, plan: Plan, date_col: str, rng: tuple[pd.Timestamp, pd.Timestamp]) -> None:
    lo, hi = rng
    if pd.isna(hi):
        return
    hi_day = pd.Timestamp(hi).normalize()
    dc = q_ident(date_col)

    def add(start: pd.Timestamp, end: pd.Timestamp, desc: str):
        plan.filters.append(f"{dc} >= TIMESTAMP '{start:%Y-%m-%d %H:%M:%S}' AND {dc} < TIMESTAMP '{end:%Y-%m-%d %H:%M:%S}'")
        plan.filter_desc.append(desc)

    m = re.search(r"\b(last|past|previous|recent)\s+(\d+|" + "|".join(NUM_WORDS) + r")\s+(day|week|month|quarter|year)s?\b", q)
    if m:
        n = int(m.group(2)) if m.group(2).isdigit() else NUM_WORDS[m.group(2)]
        unit = m.group(3)
        delta = {"day": pd.DateOffset(days=n), "week": pd.DateOffset(weeks=n), "month": pd.DateOffset(months=n),
                 "quarter": pd.DateOffset(months=3 * n), "year": pd.DateOffset(years=n)}[unit]
        end = hi_day + pd.Timedelta(days=1)
        add(end - delta, end, f"last {n} {unit}s")
        return
    m = re.search(r"\b(last|this|previous|current|past)\s+(week|month|quarter|year)\b", q)
    if m:
        which, unit = m.group(1), m.group(2)
        if unit == "week":
            start = hi_day - pd.Timedelta(days=hi_day.dayofweek)
            if which in ("last", "previous", "past"):
                start -= pd.Timedelta(days=7)
            end = start + pd.Timedelta(days=7)
        elif unit == "month":
            start = hi_day.replace(day=1)
            if which in ("last", "previous", "past"):
                start = (start - pd.Timedelta(days=1)).replace(day=1)
            end = start + pd.DateOffset(months=1)
        elif unit == "quarter":
            qm = 3 * ((hi_day.month - 1) // 3) + 1
            start = hi_day.replace(month=qm, day=1)
            if which in ("last", "previous", "past"):
                start = start - pd.DateOffset(months=3)
            end = start + pd.DateOffset(months=3)
        else:
            start = hi_day.replace(month=1, day=1)
            if which in ("last", "previous", "past"):
                start = start.replace(year=start.year - 1)
            end = start.replace(year=start.year + 1)
        add(pd.Timestamp(start), pd.Timestamp(end), f"{which} {unit}")
        return
    if re.search(r"\b(ytd|year to date)\b", q):
        start = hi_day.replace(month=1, day=1)
        add(start, hi_day + pd.Timedelta(days=1), "year to date")
        return
    qm = re.search(r"\bq([1-4])\s*((?:19|20)\d{2})?\b", q)
    if qm:
        qn = int(qm.group(1))
        year = int(qm.group(2)) if qm.group(2) else hi_day.year
        start = pd.Timestamp(year=year, month=3 * (qn - 1) + 1, day=1)
        add(start, start + pd.DateOffset(months=3), f"Q{qn} {year}")
        return
    mm = re.search(r"\b(" + "|".join(MONTHS + [a for a in MONTH_ABBR if a not in ("mar", "may")]) + r")\b\s*((?:19|20)\d{2})?", q)
    if mm and not re.search(r"\b(by|per|each)\s+month", q):
        name = mm.group(1)
        mi = (MONTHS.index(name) if name in MONTHS else MONTH_ABBR.index(name)) + 1
        year = int(mm.group(2)) if mm.group(2) else (hi_day.year if mi <= hi_day.month else hi_day.year - 1)
        start = pd.Timestamp(year=year, month=mi, day=1)
        add(start, start + pd.DateOffset(months=1), f"{MONTHS[mi - 1].title()} {year}")
        return
    years = sorted(set(re.findall(r"\b((?:19|20)\d{2})\b", q)))
    if len(years) == 1:
        y = int(years[0])
        add(pd.Timestamp(year=y, month=1, day=1), pd.Timestamp(year=y + 1, month=1, day=1), f"year {y}")


# ------------------------------------------------------------------ LLM planner
PLANNER_SYSTEM = """You are the Planner agent in a data-analysis team. Given a table schema and a business
question, identify the analysis intent and the relevant columns. Respond with ONLY a JSON object:
{"intent": "kpi|group|trend|growth|rows|list|distribution|correlation",
 "metrics": [column names], "dimensions": [column names], "date_column": string|null,
 "time_grain": "day|week|month|quarter|year"|null, "filters": [plain-English filter strings],
 "order": "desc|asc"|null, "limit": integer|null, "notes": "one sentence restating the question precisely"}
Only use column names that exist in the schema."""


def plan(ds_id: str, question: str) -> Plan:
    """Rule plan always; enrich with LLM understanding when available."""
    p = plan_rules(ds_id, question)
    if not llm.llm_available():
        return p
    try:
        ctx = dataset_context(ds_id)
        cols = {c["name"] for c in ctx["meta"]["columns"]}
        user = f"{store.schema_text(ds_id)}\n\n{_value_hints(ctx)}\n\nQuestion: {question}"
        data = llm.complete_json(PLANNER_SYSTEM, user, max_tokens=1500, required_keys=("intent",))
        for k in ("metrics", "dimensions"):
            v = data.get(k) or []
            if not isinstance(v, list) or any(x not in cols for x in v):
                raise llm.LLMError(f"planner returned unknown columns in {k}")
        p.source = "llm"
        p.llm_notes = str(data.get("notes") or "")[:400]
        p.llm_plan = data  # type: ignore[attr-defined]
    except llm.LLMError as e:
        p.warnings.append(f"LLM planner unavailable ({str(e)[:80]}); used rules.")
    return p


def _value_hints(ctx: dict[str, Any]) -> str:
    lines = []
    for col, vals in ctx["values"].items():
        shown = ", ".join(map(str, vals[:12]))
        more = f" (+{len(vals) - 12} more)" if len(vals) > 12 else ""
        lines.append(f'- "{col}" values: {shown}{more}')
    for col, (lo, hi) in ctx["date_range"].items():
        lines.append(f'- "{col}" range: {lo} .. {hi}')
    return "Column hints:\n" + "\n".join(lines) if lines else ""
