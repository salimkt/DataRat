"""API tests for DataRat (run: pytest -q from backend/)."""
from __future__ import annotations

import io
import json
import os
import sqlite3
import tempfile

import numpy as np
import pandas as pd
import pytest

os.environ.pop("ANTHROPIC_API_KEY", None)
for _k in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "DATARAT_LLM_PROVIDER"):
    os.environ.pop(_k, None)

from fastapi.testclient import TestClient  # noqa: E402

from app import connectors  # noqa: E402
from app.agents import llm  # noqa: E402
from app.main import app  # noqa: E402
from app.sqlguard import UnsafeSQLError, ensure_read_only  # noqa: E402

client = TestClient(app)


@pytest.fixture(autouse=True)
def _no_llm(monkeypatch):
    for k in ("ANTHROPIC_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY", "DATARAT_LLM_PROVIDER"):
        monkeypatch.delenv(k, raising=False)
    yield


def _strict_json(resp):
    """Fail if the body contains NaN/Infinity (invalid JSON)."""
    return json.loads(resp.text, parse_constant=lambda c: (_ for _ in ()).throw(ValueError(f"bad constant {c}")))


# ------------------------------------------------------------------ basics
def test_health():
    r = client.get("/api/health")
    assert r.status_code == 200
    j = r.json()
    assert j["status"] == "ok" and j["llm"] is False and isinstance(j["version"], str)


def test_health_base_url_alone_does_not_enable_llm(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "http://example.invalid")
    assert client.get("/api/health").json()["llm"] is False
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    assert client.get("/api/health").json()["llm"] is True


def test_connectors_shape():
    r = client.get("/api/connectors")
    assert r.status_code == 200
    items = r.json()
    types = {c["type"] for c in items}
    assert types == {"csv", "excel", "json", "parquet", "url", "gsheets", "rest_api", "postgres", "mysql", "sqlite", "sample"}
    for c in items:
        assert set(c) >= {"type", "name", "category", "description", "icon", "upload", "fields"}
        assert c["category"] in ("file", "database", "cloud", "api")
        assert isinstance(c["upload"], bool)
        for f in c["fields"]:
            assert set(f) >= {"key", "label", "type", "required"}
            assert f["type"] in ("text", "password", "textarea", "select")
            if f["type"] == "select":
                assert f["options"]


def test_list_and_get_datasets():
    r = client.get("/api/datasets")
    assert r.status_code == 200
    ids = {d["id"] for d in r.json()}
    assert {"sales", "marketing", "support_calls"} <= ids
    d = client.get("/api/datasets/sales").json()
    assert d["source"] == "sample" and d["rows"] > 4000
    sem = {c["name"]: c["semantic"] for c in d["columns"]}
    assert sem["order_id"] == "id" and sem["order_date"] == "date" and sem["revenue"] == "metric"
    assert sem["region"] == "dimension"
    for c in d["columns"]:
        assert c["semantic"] in ("metric", "dimension", "date", "id", "text")
    assert client.get("/api/datasets/nope").status_code == 404


def test_sample_shapes():
    m = client.get("/api/datasets/marketing").json()
    s = client.get("/api/datasets/support_calls").json()
    assert 700 <= m["rows"] <= 760
    assert 2900 <= s["rows"] <= 3100
    assert [c["name"] for c in m["columns"]] == ["date", "channel", "spend", "impressions", "clicks", "leads",
                                                 "conversions", "revenue_attributed"]
    assert [c["name"] for c in s["columns"]] == ["call_id", "call_date", "agent", "queue", "duration_sec", "wait_sec",
                                                 "sentiment", "resolved", "csat", "topic"]


def test_preview():
    r = client.get("/api/datasets/sales/preview?limit=7")
    assert r.status_code == 200
    j = _strict_json(r)
    assert len(j["rows"]) == 7 and j["columns"][0] == "order_id"
    assert isinstance(j["rows"][0][1], str)  # dates as ISO strings


@pytest.mark.parametrize("ds", ["sales", "marketing", "support_calls"])
def test_profile(ds):
    r = client.get(f"/api/datasets/{ds}/profile")
    assert r.status_code == 200
    p = _strict_json(r)
    assert set(p) >= {"rows", "columns", "missing_cells_pct", "duplicate_rows", "quality_score", "column_profiles", "correlations"}
    assert 0 <= p["quality_score"] <= 100
    assert len(p["column_profiles"]) == p["columns"]
    cm = p["correlations"]
    assert len(cm["matrix"]) == len(cm["columns"]) >= 2
    num = [c for c in p["column_profiles"] if c["semantic"] == "metric"]
    assert all("mean" in c and "histogram" in c for c in num)
    dims = [c for c in p["column_profiles"] if c["semantic"] == "dimension"]
    assert all(c["top_values"] for c in dims)


@pytest.mark.parametrize("ds", ["sales", "marketing", "support_calls"])
def test_insights(ds):
    r = client.get(f"/api/datasets/{ds}/insights")
    assert r.status_code == 200
    j = _strict_json(r)
    assert set(j) == {"kpis", "trend", "breakdowns", "anomalies", "forecast", "highlights"}
    assert 1 <= len(j["kpis"]) <= 6
    for k in j["kpis"]:
        assert set(k) == {"label", "value", "format", "change_pct", "period", "sparkline"}
        assert k["format"] in ("currency", "number", "percent")
        assert isinstance(k["sparkline"], list)
    assert j["trend"]["series"] and set(j["trend"]["series"][0]) == {"date", "value"}
    assert j["breakdowns"] and len(j["breakdowns"]) <= 3
    fc = j["forecast"]
    assert fc and len(fc["forecast"]) == 30 and fc["method"]
    assert set(fc["forecast"][0]) == {"date", "value", "lower", "upper"}
    assert 3 <= len(j["highlights"]) <= 6


def test_sales_anomalies_detect_injected_spike_and_dip():
    an = client.get("/api/datasets/sales/insights").json()["anomalies"]
    dirs = {a["direction"] for a in an}
    assert {"spike", "dip"} <= dirs
    for a in an:
        assert set(a) == {"date", "metric", "value", "expected", "zscore", "direction"}
    spike = [a for a in an if a["direction"] == "spike"][0]
    assert spike["date"].startswith("2025-06")


# ------------------------------------------------------------------ connectors
def _connect(ctype, data=None, files=None):
    payload = {"type": ctype}
    payload.update(data or {})
    return client.post("/api/datasets/connect", data=payload, files=files)


def test_connect_csv_and_delete():
    csv = b"Order Date,Region,Amount,Note\n2025-01-01,East,10.5,a\n2025-01-02,West,,b\n2025-01-03,East,7,\n"
    r = _connect("csv", {"name": "My Orders"}, {"file": ("orders.csv", csv, "text/csv")})
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["id"] == "my_orders" and d["name"] == "My Orders" and d["source"] == "csv" and d["rows"] == 3
    cols = {c["name"]: c for c in d["columns"]}
    assert cols["order_date"]["semantic"] == "date" and cols["amount"]["semantic"] == "metric"
    # JSON-safe: NaN -> null
    p = _strict_json(client.get("/api/datasets/my_orders/preview"))
    assert p["rows"][1][2] is None
    prof = client.get("/api/datasets/my_orders/profile")
    assert prof.status_code == 200 and _strict_json(prof)["missing_cells_pct"] > 0
    ins = client.get("/api/datasets/my_orders/insights")
    assert ins.status_code == 200
    _strict_json(ins)
    ask = client.post("/api/ask", json={"dataset_id": "my_orders", "question": "total amount by region"})
    assert ask.status_code == 200
    assert _strict_json(ask)["rows"][0][0] == "East"
    assert client.delete("/api/datasets/my_orders").json() == {"ok": True}
    assert client.get("/api/datasets/my_orders").status_code == 404


def test_connect_tsv_and_duplicate_ids():
    tsv = b"a\tb\n1\tx\n2\ty\n"
    r1 = _connect("csv", {"name": "tab data"}, {"file": ("t.tsv", tsv, "text/tab-separated-values")})
    r2 = _connect("csv", {"name": "tab data"}, {"file": ("t.tsv", tsv, "text/tab-separated-values")})
    assert r1.status_code == r2.status_code == 200
    assert r1.json()["id"] != r2.json()["id"]
    assert [c["name"] for c in r1.json()["columns"]] == ["a", "b"]
    for r in (r1, r2):
        client.delete(f"/api/datasets/{r.json()['id']}")


def test_connect_excel():
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        pd.DataFrame({"x": [1, 2, 3], "y": ["a", "b", "c"]}).to_excel(w, sheet_name="First", index=False)
        pd.DataFrame({"k": [9, 8]}).to_excel(w, sheet_name="Second", index=False)
    data = buf.getvalue()
    r = _connect("excel", {}, {"file": ("book.xlsx", data, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")})
    assert r.status_code == 200, r.text
    assert r.json()["rows"] == 3 and r.json()["id"] == "book"
    r2 = _connect("excel", {"sheet": "Second", "name": "second sheet"}, {"file": ("book.xlsx", data, "application/octet-stream")})
    assert r2.status_code == 200 and r2.json()["rows"] == 2
    r3 = _connect("excel", {"sheet": "Missing"}, {"file": ("book.xlsx", data, "application/octet-stream")})
    assert r3.status_code == 400
    for d in (r.json()["id"], r2.json()["id"]):
        client.delete(f"/api/datasets/{d}")


@pytest.mark.parametrize("payload,rows", [
    ([{"a": 1, "b": "x"}, {"a": 2, "b": "y"}], 2),
    ({"data": [{"a": 1, "nested": {"c": 3}}, {"a": 2, "nested": {"c": 4}}]}, 2),
])
def test_connect_json(payload, rows):
    r = _connect("json", {"name": "jdata"}, {"file": ("d.json", json.dumps(payload).encode(), "application/json")})
    assert r.status_code == 200, r.text
    assert r.json()["rows"] == rows
    if isinstance(payload, dict):
        assert "nested_c" in [c["name"] for c in r.json()["columns"]]
    client.delete(f"/api/datasets/{r.json()['id']}")


def test_connect_json_ndjson_and_invalid():
    nd = b'{"a": 1}\n{"a": 2}\n{"a": 3}\n'
    r = _connect("json", {"name": "nd"}, {"file": ("d.ndjson", nd, "application/json")})
    assert r.status_code == 200 and r.json()["rows"] == 3
    client.delete(f"/api/datasets/{r.json()['id']}")
    bad = _connect("json", {}, {"file": ("d.json", b"{not json", "application/json")})
    assert bad.status_code == 400


def test_connect_parquet():
    df = pd.DataFrame({"when": pd.date_range("2025-01-01", periods=40, freq="D"),
                       "cat": ["a", "b"] * 20, "val": np.arange(40, dtype=float)})
    buf = io.BytesIO()
    df.to_parquet(buf, index=False)
    r = _connect("parquet", {"name": "pq"}, {"file": ("x.parquet", buf.getvalue(), "application/octet-stream")})
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["rows"] == 40 and {c["name"]: c["semantic"] for c in d["columns"]}["when"] == "date"
    ins = client.get(f"/api/datasets/{d['id']}/insights").json()
    assert ins["trend"] and ins["kpis"]
    client.delete(f"/api/datasets/{d['id']}")


def test_connect_sqlite():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE people (id INTEGER, name TEXT, age INTEGER)")
    con.executemany("INSERT INTO people VALUES (?,?,?)", [(1, "Ann", 30), (2, "Bob", 41), (3, "Cy", 25)])
    con.execute("CREATE TABLE other (z INTEGER)")
    con.commit()
    con.close()
    data = open(path, "rb").read()
    os.remove(path)
    r = _connect("sqlite", {"table": "people"}, {"file": ("db.sqlite", data, "application/octet-stream")})
    assert r.status_code == 200, r.text
    assert r.json()["rows"] == 3 and r.json()["id"] == "people"
    r2 = _connect("sqlite", {"query": "SELECT name, age FROM people WHERE age > 26", "name": "older"},
                  {"file": ("db.sqlite", data, "application/octet-stream")})
    assert r2.status_code == 200 and r2.json()["rows"] == 2
    r3 = _connect("sqlite", {"query": "DELETE FROM people"}, {"file": ("db.sqlite", data, "application/octet-stream")})
    assert r3.status_code == 400
    r4 = _connect("sqlite", {"table": "missing"}, {"file": ("db.sqlite", data, "application/octet-stream")})
    assert r4.status_code == 400
    for d in (r.json()["id"], r2.json()["id"]):
        client.delete(f"/api/datasets/{d}")


def test_connect_sample_reload_after_delete():
    assert client.delete("/api/datasets/marketing").status_code == 200
    assert client.get("/api/datasets/marketing").status_code == 404
    r = _connect("sample", {"dataset": "marketing"})
    assert r.status_code == 200
    assert r.json()["id"] == "marketing" and r.json()["source"] == "sample"
    assert client.get("/api/datasets/marketing").status_code == 200
    # reloading an existing sample keeps the same id
    assert _connect("sample", {"dataset": "marketing"}).json()["id"] == "marketing"
    assert _connect("sample", {"dataset": "nope"}).status_code == 400


def test_connect_errors(monkeypatch):
    assert _connect("bogus").status_code == 400
    assert client.post("/api/datasets/connect", data={}).status_code == 400
    assert _connect("csv").status_code == 400  # no file
    assert _connect("postgres", {"connection_string": "postgresql://u:p@h/db"}).status_code == 400  # missing query
    assert _connect("postgres", {"connection_string": "postgresql://u:p@h/db", "query": "DROP TABLE x"}).status_code == 400
    monkeypatch.setattr(connectors, "MAX_UPLOAD_BYTES", 10)
    r = _connect("csv", {}, {"file": ("big.csv", b"a,b\n" + b"1,2\n" * 20, "text/csv")})
    assert r.status_code == 413


def test_url_connectors_reject_local_and_bad_scheme():
    assert _connect("url", {"url": "http://127.0.0.1:8000/data.csv"}).status_code == 400
    assert _connect("url", {"url": "file:///etc/passwd"}).status_code == 400
    assert _connect("rest_api", {"url": "http://localhost/api"}).status_code == 400
    assert _connect("gsheets", {"sheet_url": "https://example.com/not-a-sheet"}).status_code == 400


def test_gsheets_url_conversion():
    u = connectors.gsheets_csv_url("https://docs.google.com/spreadsheets/d/abc_DEF-123/edit#gid=456")
    assert u == "https://docs.google.com/spreadsheets/d/abc_DEF-123/export?format=csv&gid=456"
    assert connectors.gsheets_csv_url("https://docs.google.com/spreadsheets/d/xyz/edit?usp=sharing") == \
        "https://docs.google.com/spreadsheets/d/xyz/export?format=csv"


def test_rest_json_path_parsing():
    obj = {"meta": {}, "result": {"items": [{"a": 1}, {"a": 2}]}}
    df = connectors._records_from_json(obj, "result.items")
    assert list(df["a"]) == [1, 2]


# ------------------------------------------------------------------ SQL
def test_sql_ok_and_cross_dataset():
    r = client.post("/api/sql", json={"sql": "SELECT region, COUNT(*) AS n FROM sales GROUP BY 1 ORDER BY 2 DESC"})
    assert r.status_code == 200
    j = r.json()
    assert j["columns"] == ["region", "n"] and len(j["rows"]) == 5
    r2 = client.post("/api/sql", json={"dataset_id": "sales", "sql": "WITH m AS (SELECT * FROM marketing) SELECT COUNT(*) FROM m;"})
    assert r2.status_code == 200
    r3 = client.post("/api/sql", json={"sql": "SELECT * FROM sales WHERE region = 'copy; drop'"})
    assert r3.status_code == 200  # keywords inside string literals are fine


@pytest.mark.parametrize("sql", [
    "DROP TABLE sales",
    "DELETE FROM sales",
    "INSERT INTO sales VALUES (1)",
    "SELECT 1; SELECT 2",
    "SELECT 1; DROP TABLE sales",
    "ATTACH 'x.db' AS x",
    "COPY sales TO 'out.csv'",
    "INSTALL httpfs",
    "LOAD httpfs",
    "PRAGMA database_list",
    "EXPORT DATABASE 'dir'",
    "SELECT * FROM read_csv('/etc/hosts')",
    "SELECT * FROM read_csv_auto('/etc/hosts')",
    "SELECT * FROM read_parquet('x.parquet')",
    "SELECT * FROM '/etc/hosts'",
    "SET enable_external_access=true",
    "CREATE TABLE t AS SELECT 1",
    "SELECT getenv('HOME')",
    "",
])
def test_sql_guard_rejects(sql):
    r = client.post("/api/sql", json={"sql": sql})
    assert r.status_code in (400, 422), (sql, r.text)
    if sql:
        with pytest.raises(UnsafeSQLError):
            ensure_read_only(sql)


def test_sql_error_and_unknown_dataset():
    assert client.post("/api/sql", json={"sql": "SELECT nope FROM sales"}).status_code == 400
    assert client.post("/api/sql", json={"dataset_id": "zzz", "sql": "SELECT 1"}).status_code == 404


def test_sql_comment_stripping():
    assert ensure_read_only("-- hello\nSELECT 1 /* c */ ;") .startswith("SELECT 1")


# ------------------------------------------------------------------ ask
def ask(ds, q):
    r = client.post("/api/ask", json={"dataset_id": ds, "question": q})
    assert r.status_code == 200, (q, r.text)
    j = _strict_json(r)
    assert set(j) == {"question", "steps", "sql", "columns", "rows", "chart", "narrative", "recommendations", "follow_ups"}
    assert [s["agent"] for s in j["steps"]] == ["Planner", "SQL Engineer", "Analyst", "Narrator"]
    for s in j["steps"]:
        assert set(s) == {"agent", "title", "detail", "duration_ms"} and isinstance(s["duration_ms"], int)
    assert j["chart"]["type"] in ("bar", "line", "area", "pie", "scatter", "table", "kpi") and j["chart"]["title"]
    assert len(j["rows"]) <= 500
    assert j["narrative"].strip()
    assert isinstance(j["recommendations"], list) and isinstance(j["follow_ups"], list) and j["follow_ups"]
    if j["chart"].get("x"):
        assert j["chart"]["x"] in j["columns"]
    for y in j["chart"].get("y") or []:
        assert y in j["columns"]
    return j


def first_col(j):
    return [r[0] for r in j["rows"]]


ASK_CASES = [
    # (dataset, question, chart types, required columns, extra check)
    ("sales", "total revenue by region", {"bar"}, ["region", "revenue"], lambda j: first_col(j)[0] == "North America"),
    ("sales", "Total Revenue by Region?", {"bar"}, ["region", "revenue"], None),
    ("sales", "top 5 products by profit", {"bar"}, ["product", "profit"], lambda j: len(j["rows"]) == 5),
    ("sales", "top five products by profit", {"bar"}, ["product", "profit"], lambda j: len(j["rows"]) == 5),
    ("sales", "monthly revenue trend", {"area", "line"}, ["month", "revenue"], lambda j: len(j["rows"]) == 24),
    ("sales", "revenue over time", {"area", "line"}, ["revenue"], None),
    ("sales", "how many orders", {"kpi"}, ["orders"], lambda j: j["rows"][0][0] > 4000),
    ("sales", "how many rows", {"kpi"}, [], lambda j: j["rows"][0][0] > 4000),
    ("sales", "average discount", {"kpi"}, ["avg_discount"], None),
    ("sales", "compare revenue and profit by region", {"bar"}, ["region", "revenue", "profit"], None),
    ("sales", "compare Europe vs Asia Pacific revenue", {"bar"}, ["region"], lambda j: len(j["rows"]) == 2),
    ("sales", "which region grew the fastest", {"bar"}, ["region", "growth_pct"], None),
    ("sales", "revenue growth", {"area", "line", "bar"}, ["revenue"], None),
    ("sales", "revenue share by channel", {"pie"}, ["channel", "revenue"], None),
    ("sales", "profit margin by product category", {"bar"}, ["product_category", "profit_margin"], None),
    ("sales", "top 10 orders by revenue", {"table"}, ["order_id", "revenue"], lambda j: len(j["rows"]) == 10),
    ("sales", "revenue in 2025", {"kpi"}, ["revenue"], lambda j: "2025" in j["sql"]),
    ("sales", "revenue last 3 months by region", {"bar"}, ["region"], None),
    ("sales", "quarterly profit", {"area", "line"}, ["quarter", "profit"], lambda j: len(j["rows"]) == 8),
    ("sales", "which month had the highest revenue", {"bar"}, ["month", "revenue"], None),
    ("sales", "units sold by customer segment", {"bar"}, ["customer_segment", "units"], None),
    ("sales", "monthly revenue by region", {"line"}, ["month", "North America"], None),
    ("sales", "revenue by region and channel", {"bar"}, ["region", "Online"], None),
    ("sales", "bottom 3 countries by profit", {"bar"}, ["country"], lambda j: len(j["rows"]) == 3),
    ("sales", "what is the highest unit price", {"kpi"}, ["max_unit_price"], None),
    ("sales", "correlation between discount and profit", {"scatter"}, ["discount", "profit"], None),
    ("sales", "distribution of revenue", {"bar"}, ["revenue_bucket", "count"], None),
    ("sales", "show me some orders", {"table"}, ["order_id"], None),
    ("sales", "revenue 2024 vs 2025", {"bar"}, ["year", "revenue"], lambda j: len(j["rows"]) == 2),
    ("sales", "What were sales last month?", {"kpi"}, ["revenue"], None),
    ("sales", "average order value by channel", {"bar"}, ["avg_order_value"], None),
    ("sales", "how many orders in Europe", {"kpi"}, ["orders"], lambda j: "Europe" in j["sql"]),
    ("sales", "Which product category is most profitable?", {"bar"}, ["product_category"], None),
    ("sales", "something completely unrelated", {"kpi", "table", "bar"}, [], None),
    ("marketing", "which channel has best conversion rate", {"bar"}, ["channel", "conversion_rate"],
     lambda j: first_col(j)[0] == "Events"),
    ("marketing", "total spend by channel", {"bar"}, ["channel", "spend"], None),
    ("marketing", "roas by channel", {"bar"}, ["roas"], None),
    ("marketing", "monthly spend trend", {"area", "line"}, ["month", "spend"], None),
    ("marketing", "cost per acquisition by channel", {"bar"}, ["cpa"], None),
    ("marketing", "ctr by channel", {"bar"}, ["ctr"], None),
    ("marketing", "conversion rate trend", {"area", "line"}, ["conversion_rate"], None),
    ("marketing", "how many leads did paid search generate", {"kpi"}, ["leads"], lambda j: "Paid Search" in j["sql"]),
    ("support_calls", "average csat by queue", {"bar"}, ["queue", "avg_csat"], None),
    ("support_calls", "how many calls were resolved", {"kpi"}, ["resolved_calls"],
     lambda j: 0 < j["rows"][0][0] < 3000 and "resolved" in j["narrative"]),
    ("support_calls", "how many calls were not resolved", {"kpi"}, ["not_resolved_calls"], None),
    ("support_calls", "resolution rate by agent", {"bar"}, ["agent", "resolved_rate"], None),
    ("support_calls", "average wait time by queue", {"bar"}, ["queue", "avg_wait_sec"],
     lambda j: first_col(j)[0] == "Technical Support"),
    ("support_calls", "calls by topic", {"bar"}, ["topic", "calls"], None),
    ("support_calls", "monthly call volume", {"area", "line"}, ["month", "calls"], lambda j: len(j["rows"]) == 12),
    ("support_calls", "which agent has the highest csat", {"bar"}, ["agent", "avg_csat"], None),
    ("support_calls", "average handle time", {"kpi"}, ["avg_duration_sec"], None),
    ("support_calls", "unresolved calls by queue", {"bar"}, ["queue", "calls"], lambda j: "FALSE" in j["sql"]),
    ("support_calls", "average csat for the sales queue", {"kpi"}, ["avg_csat"], lambda j: "'Sales'" in j["sql"]),
    ("support_calls", "what topics drive negative sentiment", {"bar"}, ["topic"], lambda j: first_col(j)[0] == "Outage"),
    ("support_calls", "daily call volume last 30 days", {"area", "line"}, ["day", "calls"], None),
]


@pytest.mark.parametrize("ds,q,charts,cols,check", ASK_CASES, ids=[f"{c[0]}:{c[1]}" for c in ASK_CASES])
def test_ask_battery(ds, q, charts, cols, check):
    j = ask(ds, q)
    assert j["chart"]["type"] in charts, (q, j["chart"], j["sql"])
    for c in cols:
        assert c in j["columns"], (q, c, j["columns"], j["sql"])
    if check:
        assert check(j), (q, j["sql"], j["rows"][:3])


def test_follow_ups_are_answerable():
    for ds, q in [("sales", "total revenue by region"), ("marketing", "monthly spend trend"),
                  ("support_calls", "how many calls were resolved")]:
        for f in ask(ds, q)["follow_ups"]:
            ask(ds, f)


def test_ask_validation():
    assert client.post("/api/ask", json={"dataset_id": "sales", "question": "   "}).status_code == 400
    assert client.post("/api/ask", json={"dataset_id": "missing", "question": "x"}).status_code == 404
    assert client.post("/api/ask", json={"dataset_id": "sales"}).status_code == 422


# ------------------------------------------------------------------ LLM paths (mocked, no network)
def test_llm_sql_path_used_and_validated(monkeypatch):
    calls = []

    def fake(system, user, max_tokens=4000, required_keys=()):
        calls.append(system[:30])
        if "SQL Engineer" in system:
            return {"sql": 'SELECT "channel", SUM("revenue") AS revenue FROM "sales" GROUP BY 1 ORDER BY 2 DESC',
                    "title": "Revenue by channel (LLM)", "chart": {"type": "bar", "x": "channel", "y": ["revenue"]}}
        if "Narrator" in system:
            return {"narrative": "Online leads.", "recommendations": ["Do X"], "follow_ups": ["a", "b", "c"]}
        return {"intent": "group", "metrics": ["revenue"], "dimensions": ["channel"], "notes": "Revenue per channel"}

    monkeypatch.setattr(llm, "llm_available", lambda: True)
    monkeypatch.setattr(llm, "complete_json", fake)
    j = ask("sales", "how is revenue split across channels")
    assert j["chart"]["title"] == "Revenue by channel (LLM)"
    assert j["narrative"] == "Online leads." and j["follow_ups"] == ["a", "b", "c"]
    assert "[Claude]" in j["steps"][1]["detail"]
    assert len(calls) == 3


def test_llm_bad_sql_retries_then_falls_back(monkeypatch):
    attempts = []

    def fake(system, user, max_tokens=4000, required_keys=()):
        if "SQL Engineer" in system:
            attempts.append(user)
            return {"sql": "DROP TABLE sales"}
        raise llm.LLMError("nope")

    monkeypatch.setattr(llm, "llm_available", lambda: True)
    monkeypatch.setattr(llm, "complete_json", fake)
    j = ask("sales", "total revenue by region")
    assert len(attempts) == 3  # first try + 2 retries
    assert "previous query failed" in attempts[-1]
    assert j["sql"].startswith('SELECT "region"')  # rule-based fallback
    assert client.get("/api/datasets/sales").status_code == 200


def test_llm_errors_fall_back_to_rules(monkeypatch):
    def boom(*a, **k):
        raise llm.LLMError("API down")

    monkeypatch.setattr(llm, "llm_available", lambda: True)
    monkeypatch.setattr(llm, "complete_json", boom)
    j = ask("support_calls", "average csat by queue")
    assert "avg_csat" in j["columns"]
    b = client.post("/api/brief", json={"dataset_id": "support_calls"})
    assert b.status_code == 200 and b.json()["sections"]


def test_llm_not_called_without_key(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "http://example.invalid")

    def boom(*a, **k):
        raise AssertionError("LLM must not be called")

    monkeypatch.setattr(llm, "_get_client", boom)
    ask("sales", "total revenue by region")
    client.post("/api/brief", json={"dataset_id": "sales"})
    with pytest.raises(llm.LLMError):
        llm.complete_json("s", "u")


def test_extract_json():
    assert llm.extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert llm.extract_json('Here you go: {"a": [1, 2]} thanks') == {"a": [1, 2]}
    with pytest.raises(llm.LLMError):
        llm.extract_json("no json here")


# ------------------------------------------------------------------ brief
@pytest.mark.parametrize("ds", ["sales", "marketing", "support_calls"])
def test_brief(ds):
    r = client.post("/api/brief", json={"dataset_id": ds})
    assert r.status_code == 200
    j = _strict_json(r)
    assert set(j) == {"title", "generated_at", "summary", "sections", "kpis"}
    assert j["summary"] and len(j["sections"]) >= 3
    assert all(set(s) == {"heading", "body"} for s in j["sections"])
    assert j["kpis"] == client.get(f"/api/datasets/{ds}/insights").json()["kpis"]
    assert client.post("/api/brief", json={"dataset_id": "missing"}).status_code == 404


def test_cors_headers():
    r = client.options("/api/health", headers={"Origin": "https://example.github.io",
                                               "Access-Control-Request-Method": "GET"})
    assert r.headers.get("access-control-allow-origin") in ("*", "https://example.github.io")


# ------------------------------------------------------------------ gemini provider
class _FakeResp:
    def __init__(self, status, payload):
        self.status_code = status
        self._payload = payload
        self.text = json.dumps(payload)

    def json(self):
        return self._payload


def _gemini_ok(obj):
    return _FakeResp(200, {"candidates": [{"finishReason": "STOP", "content": {"parts": [
        {"text": "thinking...", "thought": True}, {"text": json.dumps(obj)}]}}]})


def test_provider_selection(monkeypatch):
    assert llm.provider() is None
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    assert llm.provider() == "anthropic"
    monkeypatch.setenv("GEMINI_API_KEY", "g-test")
    assert llm.provider() == "gemini"  # gemini preferred in auto mode
    assert llm.model_name() == llm.DEFAULT_GEMINI_MODEL
    monkeypatch.setenv("DATARAT_LLM_PROVIDER", "anthropic")
    assert llm.provider() == "anthropic"
    monkeypatch.delenv("GEMINI_API_KEY")
    monkeypatch.setenv("DATARAT_LLM_PROVIDER", "gemini")
    assert llm.provider() is None
    monkeypatch.setenv("GOOGLE_API_KEY", "g-test")
    assert llm.provider() == "gemini"
    h = client.get("/api/health").json()
    assert h["llm"] is True and h["provider"] == "gemini" and h["model"]


def test_gemini_complete_json(monkeypatch):
    import httpx
    monkeypatch.setenv("GEMINI_API_KEY", "g-test")
    calls = []

    def fake_post(url, json=None, headers=None, timeout=None):
        calls.append((url, json, headers))
        return _gemini_ok({"intent": "total", "x": 1})

    monkeypatch.setattr(httpx, "post", fake_post)
    out = llm.complete_json("sys", "user", required_keys=("intent",))
    assert out == {"intent": "total", "x": 1}
    url, body, headers = calls[0]
    assert llm.DEFAULT_GEMINI_MODEL in url and headers["x-goog-api-key"] == "g-test"
    assert body["generationConfig"]["responseMimeType"] == "application/json"
    assert body["system_instruction"]["parts"][0]["text"] == "sys"


def test_gemini_retries_without_thinking_config(monkeypatch):
    import httpx
    monkeypatch.setenv("GEMINI_API_KEY", "g-test")
    bodies = []

    def fake_post(url, json=None, headers=None, timeout=None):
        bodies.append(dict(json["generationConfig"]))
        if "thinkingConfig" in json["generationConfig"]:
            return _FakeResp(400, {"error": {"message": "thinking_level is not supported"}})
        return _gemini_ok({"intent": "x"})

    monkeypatch.setattr(httpx, "post", fake_post)
    assert llm.complete_json("s", "u") == {"intent": "x"}
    assert len(bodies) == 2 and "thinkingConfig" not in bodies[1]


@pytest.mark.parametrize("resp", [
    _FakeResp(403, {"error": {"message": "API key not valid"}}),
    _FakeResp(200, {"candidates": [], "promptFeedback": {"blockReason": "SAFETY"}}),
    _FakeResp(200, {"candidates": [{"finishReason": "MAX_TOKENS", "content": {"parts": [{"text": "{"}]}}]}),
    _FakeResp(200, {"candidates": [{"finishReason": "STOP", "content": {"parts": [{"text": "not json"}]}}]}),
])
def test_gemini_errors_raise_llmerror(monkeypatch, resp):
    import httpx
    monkeypatch.setenv("GEMINI_API_KEY", "g-test")
    monkeypatch.setattr(httpx, "post", lambda *a, **k: resp)
    with pytest.raises(llm.LLMError):
        llm.complete_json("s", "u")


def test_ask_falls_back_when_gemini_fails(monkeypatch):
    import httpx
    monkeypatch.setenv("GEMINI_API_KEY", "g-test")

    def boom(*a, **k):
        raise httpx.ConnectError("down")

    monkeypatch.setattr(httpx, "post", boom)
    r = client.post("/api/ask", json={"dataset_id": "sales", "question": "total revenue by region"})
    assert r.status_code == 200 and r.json()["rows"]
