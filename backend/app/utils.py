"""Small shared helpers: JSON-safe conversion, number formatting, naming."""
from __future__ import annotations

import base64
import datetime as _dt
import math
import re
import uuid
from decimal import Decimal
from typing import Any

import numpy as np
import pandas as pd

MONEY_WORDS = (
    "revenue", "sales", "amount", "profit", "spend", "cost", "price", "income",
    "budget", "value", "gmv", "earning", "payment", "fee", "salary", "margin_amt",
)
RATE_WORDS = ("rate", "pct", "percent", "ratio", "share", "margin")
NON_ADDITIVE_WORDS = (
    "avg", "average", "mean", "rate", "pct", "percent", "ratio", "score", "csat",
    "nps", "sentiment", "rating", "price", "discount", "duration", "wait", "age",
    "margin", "temperature", "latency", "time", "sec", "minutes", "days", "lat",
    "lon", "probability", "index",
)


def to_jsonable(obj: Any) -> Any:
    """Recursively convert numpy/pandas/datetime objects into plain JSON types.

    NaN / Inf / NaT / NA become ``None``; datetimes become ISO strings (date-only
    when the time component is midnight).
    """
    if obj is None or isinstance(obj, (str, bool)):
        return obj
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [to_jsonable(v) for v in obj.tolist()]
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        f = float(obj)
        return None if (math.isnan(f) or math.isinf(f)) else f
    if isinstance(obj, Decimal):
        f = float(obj)
        return None if (math.isnan(f) or math.isinf(f)) else f
    if obj is pd.NaT:
        return None
    try:
        if obj is pd.NA:
            return None
    except Exception:  # pragma: no cover
        pass
    if isinstance(obj, pd.Timestamp):
        if pd.isna(obj):
            return None
        return iso(obj.to_pydatetime())
    if isinstance(obj, np.datetime64):
        if np.isnat(obj):
            return None
        return iso(pd.Timestamp(obj).to_pydatetime())
    if isinstance(obj, _dt.datetime):
        return iso(obj)
    if isinstance(obj, _dt.date):
        return obj.isoformat()
    if isinstance(obj, _dt.time):
        return obj.isoformat()
    if isinstance(obj, (_dt.timedelta, pd.Timedelta)):
        try:
            return obj.total_seconds()
        except Exception:
            return None
    if isinstance(obj, np.timedelta64):
        return to_jsonable(pd.Timedelta(obj))
    if isinstance(obj, uuid.UUID):
        return str(obj)
    if isinstance(obj, (bytes, bytearray, memoryview)):
        b = bytes(obj)
        try:
            return b.decode("utf-8")
        except UnicodeDecodeError:
            return base64.b64encode(b).decode("ascii")
    if isinstance(obj, pd.Series):
        return to_jsonable(obj.tolist())
    if isinstance(obj, pd.DataFrame):
        return to_jsonable(obj.to_dict(orient="records"))
    if hasattr(obj, "item"):
        try:
            return to_jsonable(obj.item())
        except Exception:
            pass
    return str(obj)


def iso(d: _dt.datetime) -> str:
    if d.hour == 0 and d.minute == 0 and d.second == 0 and d.microsecond == 0 and d.tzinfo is None:
        return d.date().isoformat()
    return d.isoformat()


def slugify(text: str, fallback: str = "dataset") -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "_", str(text)).strip("_").lower()
    if not s:
        s = fallback
    if s[0].isdigit():
        s = f"t_{s}"
    return s[:48]


def humanize(name: str) -> str:
    """'product_category' -> 'Product Category'; 'csat' -> 'CSAT'."""
    special = {"csat": "CSAT", "nps": "NPS", "id": "ID", "roi": "ROI", "roas": "ROAS",
               "ctr": "CTR", "cpa": "CPA", "cpc": "CPC", "aov": "AOV", "sec": "(sec)"}
    parts = [p for p in re.split(r"[_\s]+", str(name)) if p]
    out = []
    for p in parts:
        out.append(special.get(p.lower(), p[:1].upper() + p[1:]))
    return " ".join(out) or str(name)


def is_money(name: str) -> bool:
    n = name.lower()
    return any(w in n for w in MONEY_WORDS) and not any(w in n for w in ("pct", "percent", "rate", "ratio"))


def is_additive(name: str) -> bool:
    n = name.lower()
    if is_money(n) and "price" not in n:
        return True
    return not any(w in n for w in NON_ADDITIVE_WORDS)


def fmt_number(v: Any, kind: str = "number") -> str:
    """Compact human formatting: currency / percent / number."""
    if v is None:
        return "n/a"
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    if math.isnan(f):
        return "n/a"
    if kind == "percent":
        return f"{f:.1f}%"
    a = abs(f)
    sign = "-" if f < 0 else ""
    prefix = "$" if kind == "currency" else ""
    if a >= 1e9:
        s = f"{a / 1e9:.2f}B"
    elif a >= 1e6:
        s = f"{a / 1e6:.2f}M"
    elif a >= 1e4:
        s = f"{a / 1e3:.1f}K"
    elif a >= 100 or a == int(a):
        s = f"{a:,.0f}"
    elif a >= 1:
        s = f"{a:,.2f}"
    else:
        s = f"{a:.3f}"
    return f"{sign}{prefix}{s}"


def kind_for(name: str) -> str:
    n = name.lower()
    if any(w in n for w in ("rate", "pct", "percent", "margin", "share")) and not n.endswith("_amt"):
        return "percent"
    if is_money(n):
        return "currency"
    return "number"


def pluralize(phrase: str) -> str:
    """'product category' -> 'product categories', 'country' -> 'countries'."""
    if not phrase:
        return phrase
    head, _, last = phrase.rpartition(" ")
    lw = last.lower()
    if lw.endswith("s") and not lw.endswith("ss"):
        plural = last
    elif lw.endswith("y") and not lw.endswith(("ay", "ey", "oy", "uy")):
        plural = last[:-1] + "ies"
    elif lw.endswith(("s", "x", "ch", "sh")):
        plural = last + "es"
    else:
        plural = last + "s"
    return f"{head} {plural}" if head else plural


def sentence_label(label: str) -> str:
    """'Avg CSAT' -> 'average CSAT'; 'Revenue Attributed' -> 'revenue attributed'."""
    words = []
    for w in label.split():
        if w == "Avg":
            words.append("average")
        elif w.isupper() and len(w) > 1:
            words.append(w)
        else:
            words.append(w.lower())
    return " ".join(words)
