"""Data source connectors: definitions for the UI + loaders returning DataFrames."""
from __future__ import annotations

import io
import ipaddress
import json
import os
import re
import socket
import sqlite3
import tempfile
from typing import Any
from urllib.parse import parse_qs, urlparse

import httpx
import pandas as pd

from . import samples
from .sqlguard import ensure_read_only

MAX_UPLOAD_BYTES = 50 * 1024 * 1024
MAX_REMOTE_BYTES = 50 * 1024 * 1024
REMOTE_TIMEOUT = 20.0
MAX_DB_ROWS = 500_000


class ConnectorError(ValueError):
    pass


def _f(key, label, type_="text", required=True, placeholder=None, options=None):
    d: dict[str, Any] = {"key": key, "label": label, "type": type_, "required": required}
    if placeholder is not None:
        d["placeholder"] = placeholder
    if options is not None:
        d["options"] = options
    return d


CONNECTORS: list[dict[str, Any]] = [
    {"type": "csv", "name": "CSV / TSV", "category": "file", "icon": "file-text", "upload": True,
     "description": "Upload a comma- or tab-separated file.", "fields": []},
    {"type": "excel", "name": "Excel", "category": "file", "icon": "file-spreadsheet", "upload": True,
     "description": "Upload an .xlsx workbook (first sheet by default).",
     "fields": [_f("sheet", "Sheet name (optional)", required=False, placeholder="Sheet1")]},
    {"type": "json", "name": "JSON", "category": "file", "icon": "braces", "upload": True,
     "description": "Array of records, {\"data\": [...]} or newline-delimited JSON.", "fields": []},
    {"type": "parquet", "name": "Parquet", "category": "file", "icon": "package", "upload": True,
     "description": "Upload an Apache Parquet file.", "fields": []},
    {"type": "url", "name": "Remote URL", "category": "cloud", "icon": "link", "upload": False,
     "description": "Public CSV / JSON / Parquet URL (incl. Google Sheets export links).",
     "fields": [_f("url", "File URL", placeholder="https://example.com/data.csv")]},
    {"type": "gsheets", "name": "Google Sheets", "category": "cloud", "icon": "sheet", "upload": False,
     "description": "A Google Sheet shared as 'Anyone with the link can view'.",
     "fields": [_f("sheet_url", "Sheet share link",
                   placeholder="https://docs.google.com/spreadsheets/d/<id>/edit#gid=0")]},
    {"type": "rest_api", "name": "REST API", "category": "api", "icon": "webhook", "upload": False,
     "description": "GET a JSON endpoint and flatten the records.",
     "fields": [_f("url", "Endpoint URL", placeholder="https://api.example.com/v1/orders"),
                _f("json_path", "JSON path to records (optional)", required=False, placeholder="data.items"),
                _f("headers", "Headers as JSON (optional)", "textarea", required=False,
                   placeholder='{"Authorization": "Bearer <token>"}')]},
    {"type": "postgres", "name": "PostgreSQL", "category": "database", "icon": "database", "upload": False,
     "description": "Run a read-only query against a PostgreSQL database.",
     "fields": [_f("connection_string", "Connection string", "password",
                   placeholder="postgresql://user:pass@host:5432/db"),
                _f("query", "SQL query", "textarea", placeholder="SELECT * FROM orders LIMIT 10000")]},
    {"type": "mysql", "name": "MySQL", "category": "database", "icon": "database", "upload": False,
     "description": "Run a read-only query against a MySQL / MariaDB database.",
     "fields": [_f("connection_string", "Connection string", "password",
                   placeholder="mysql://user:pass@host:3306/db"),
                _f("query", "SQL query", "textarea", placeholder="SELECT * FROM orders LIMIT 10000")]},
    {"type": "sqlite", "name": "SQLite", "category": "database", "icon": "hard-drive", "upload": True,
     "description": "Upload a SQLite database file and pick a table or query.",
     "fields": [_f("table", "Table (optional)", required=False, placeholder="orders"),
                _f("query", "SQL query (optional, overrides table)", "textarea", required=False,
                   placeholder="SELECT * FROM orders")]},
    {"type": "sample", "name": "Sample dataset", "category": "file", "icon": "flask-conical", "upload": False,
     "description": "Reload one of the built-in demo datasets.",
     "fields": [_f("dataset", "Dataset", "select", options=list(samples.SAMPLE_META))]},
]
CONNECTOR_TYPES = {c["type"]: c for c in CONNECTORS}


# ---------------- file parsers ----------------
def _decode(data: bytes) -> str:
    for enc in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def parse_csv(data: bytes, filename: str = "") -> pd.DataFrame:
    text = _decode(data)
    first = text.split("\n", 1)[0]
    if filename.lower().endswith((".tsv", ".tab")) or first.count("\t") > max(first.count(","), 0):
        sep = "\t"
    elif first.count(";") > first.count(",") and first.count(";") > 0:
        sep = ";"
    else:
        sep = ","
    try:
        return pd.read_csv(io.StringIO(text), sep=sep, low_memory=False)
    except Exception as e:
        raise ConnectorError(f"Could not parse CSV: {e}") from e


def parse_excel(data: bytes, sheet: str | None = None) -> pd.DataFrame:
    try:
        return pd.read_excel(io.BytesIO(data), sheet_name=sheet if sheet else 0, engine="openpyxl")
    except ValueError as e:
        raise ConnectorError(f"Could not read Excel sheet: {e}") from e
    except Exception as e:
        raise ConnectorError(f"Could not parse Excel file: {e}") from e


def _records_from_json(obj: Any, json_path: str | None = None) -> pd.DataFrame:
    if json_path:
        for part in [p for p in re.split(r"[.\[\]]+", json_path) if p]:
            if isinstance(obj, list) and part.isdigit():
                obj = obj[int(part)]
            elif isinstance(obj, dict) and part in obj:
                obj = obj[part]
            else:
                raise ConnectorError(f"json_path '{json_path}' not found in response.")
    if isinstance(obj, dict):
        for key in ("data", "records", "rows", "items", "results", "result", "values"):
            if isinstance(obj.get(key), list):
                obj = obj[key]
                break
        else:
            list_vals = [v for v in obj.values() if isinstance(v, list) and v and isinstance(v[0], dict)]
            if list_vals:
                obj = list_vals[0]
            elif obj and all(isinstance(v, list) for v in obj.values()):
                return pd.DataFrame(obj)
            else:
                obj = [obj]
    if not isinstance(obj, list):
        raise ConnectorError("JSON must be an array of records or an object with a 'data' array.")
    if obj and isinstance(obj[0], list):
        header, *body = obj
        if all(isinstance(h, str) for h in header):
            return pd.DataFrame(body, columns=header)
        return pd.DataFrame(obj)
    if obj and not isinstance(obj[0], dict):
        return pd.DataFrame({"value": obj})
    return pd.json_normalize(obj, sep="_", max_level=2)


def parse_json(data: bytes, json_path: str | None = None) -> pd.DataFrame:
    text = _decode(data).strip()
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        try:  # NDJSON
            obj = [json.loads(line) for line in text.splitlines() if line.strip()]
        except json.JSONDecodeError as e:
            raise ConnectorError(f"Invalid JSON: {e}") from e
    return _records_from_json(obj, json_path)


def parse_parquet(data: bytes) -> pd.DataFrame:
    try:
        return pd.read_parquet(io.BytesIO(data))
    except Exception as e:
        raise ConnectorError(f"Could not parse Parquet file: {e}") from e


def parse_sqlite(data: bytes, table: str | None = None, query: str | None = None) -> pd.DataFrame:
    if not data.startswith(b"SQLite format 3"):
        raise ConnectorError("File is not a SQLite database.")
    fd, path = tempfile.mkstemp(suffix=".sqlite")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
        con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        try:
            tables = [r[0] for r in con.execute(
                "SELECT name FROM sqlite_master WHERE type IN ('table','view') AND name NOT LIKE 'sqlite_%' ORDER BY name")]
            if query and query.strip():
                sql = ensure_read_only(query)
            else:
                if not tables:
                    raise ConnectorError("The SQLite database has no tables.")
                t = (table or "").strip() or tables[0]
                if t not in tables:
                    raise ConnectorError(f"Table '{t}' not found. Available: {', '.join(tables)}")
                sql = f'SELECT * FROM "{t}"'
            try:
                df = pd.read_sql_query(sql, con)
            except Exception as e:
                raise ConnectorError(f"SQLite query failed: {e}") from e
            return df.head(MAX_DB_ROWS)
        finally:
            con.close()
    finally:
        try:
            os.remove(path)
        except OSError:
            pass


# ---------------- remote ----------------
def _check_public_url(url: str) -> None:
    p = urlparse(url)
    if p.scheme not in ("http", "https") or not p.hostname:
        raise ConnectorError("Only http(s) URLs are supported.")
    if os.environ.get("DATARAT_ALLOW_PRIVATE_URLS") == "1":
        return
    try:
        infos = socket.getaddrinfo(p.hostname, p.port or (443 if p.scheme == "https" else 80))
    except socket.gaierror as e:
        raise ConnectorError(f"Could not resolve host '{p.hostname}'.") from e
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast:
            raise ConnectorError("URLs pointing to private or local network addresses are not allowed.")


def fetch_url(url: str, headers: dict[str, str] | None = None) -> tuple[bytes, str]:
    """GET a URL with a timeout and size cap. Returns (body, content_type)."""
    url = url.strip()
    _check_public_url(url)
    try:
        with httpx.Client(timeout=REMOTE_TIMEOUT, follow_redirects=True, max_redirects=5) as client:
            with client.stream("GET", url, headers=headers or {}) as resp:
                if resp.status_code >= 400:
                    raise ConnectorError(f"Remote server returned HTTP {resp.status_code}.")
                _check_public_url(str(resp.url))
                cl = resp.headers.get("content-length")
                if cl and cl.isdigit() and int(cl) > MAX_REMOTE_BYTES:
                    raise ConnectorError("Remote file is larger than the 50 MB limit.")
                buf = bytearray()
                for chunk in resp.iter_bytes():
                    buf.extend(chunk)
                    if len(buf) > MAX_REMOTE_BYTES:
                        raise ConnectorError("Remote file is larger than the 50 MB limit.")
                return bytes(buf), resp.headers.get("content-type", "")
    except httpx.TimeoutException as e:
        raise ConnectorError("Timed out fetching the URL.") from e
    except httpx.HTTPError as e:
        raise ConnectorError(f"Could not fetch URL: {e}") from e


def gsheets_csv_url(sheet_url: str) -> str:
    sheet_url = sheet_url.strip()
    m = re.search(r"/spreadsheets/d/([a-zA-Z0-9_-]+)", sheet_url)
    if not m:
        raise ConnectorError("That does not look like a Google Sheets link.")
    if "/export" in sheet_url and "format=csv" in sheet_url:
        return sheet_url
    sid = m.group(1)
    p = urlparse(sheet_url)
    gid = None
    qs = parse_qs(p.query)
    if "gid" in qs:
        gid = qs["gid"][0]
    fm = re.search(r"gid=(\d+)", p.fragment or "")
    if fm:
        gid = fm.group(1)
    url = f"https://docs.google.com/spreadsheets/d/{sid}/export?format=csv"
    if gid:
        url += f"&gid={gid}"
    return url


def parse_by_kind(data: bytes, url_or_name: str, content_type: str = "") -> pd.DataFrame:
    path = urlparse(url_or_name).path.lower() if "://" in url_or_name else url_or_name.lower()
    ct = content_type.lower()
    if path.endswith(".parquet") or "parquet" in ct or data[:4] == b"PAR1":
        return parse_parquet(data)
    if path.endswith((".xlsx", ".xlsm")) or "spreadsheetml" in ct or data[:2] == b"PK":
        return parse_excel(data)
    if path.endswith((".json", ".ndjson", ".jsonl")) or "json" in ct:
        return parse_json(data)
    if "format=csv" in url_or_name or path.endswith((".csv", ".tsv", ".txt")) or "csv" in ct or "text/plain" in ct:
        return parse_csv(data, path)
    stripped = data.lstrip()[:1]
    if stripped in (b"[", b"{"):
        return parse_json(data)
    if b"<html" in data[:500].lower():
        raise ConnectorError("The URL returned an HTML page, not a data file. Is it publicly shared?")
    return parse_csv(data, path)


# ---------------- databases ----------------
def _db_query(conn_str: str, query: str, kind: str) -> pd.DataFrame:
    from sqlalchemy import create_engine, text

    if not conn_str or not conn_str.strip():
        raise ConnectorError("connection_string is required.")
    if not query or not query.strip():
        raise ConnectorError("query is required.")
    sql = ensure_read_only(query)
    cs = conn_str.strip()
    if kind == "postgres":
        cs = re.sub(r"^postgres(ql)?(\+\w+)?://", "postgresql+psycopg2://", cs)
        connect_args = {"connect_timeout": 10}
    else:
        cs = re.sub(r"^mysql(\+\w+)?://", "mysql+pymysql://", cs)
        connect_args = {"connect_timeout": 10, "read_timeout": 60}
    try:
        engine = create_engine(cs, connect_args=connect_args, pool_pre_ping=False)
    except Exception as e:
        raise ConnectorError(f"Invalid connection string: {e}") from e
    try:
        with engine.connect() as conn:
            if kind == "postgres":
                try:
                    conn.execute(text("SET TRANSACTION READ ONLY"))
                except Exception:
                    pass
            result = conn.execute(text(sql))
            cols = list(result.keys())
            rows = result.fetchmany(MAX_DB_ROWS)
            return pd.DataFrame(rows, columns=cols)
    except ConnectorError:
        raise
    except Exception as e:
        raise ConnectorError(f"Database query failed: {str(e).splitlines()[0][:300]}") from e
    finally:
        engine.dispose()


# ---------------- entry point ----------------
def load(ctype: str, fields: dict[str, str], file_bytes: bytes | None = None,
         filename: str | None = None) -> tuple[pd.DataFrame, str]:
    """Load data for a connector. Returns (dataframe, suggested_name)."""
    if ctype not in CONNECTOR_TYPES:
        raise ConnectorError(f"Unknown connector type '{ctype}'.")
    conn = CONNECTOR_TYPES[ctype]
    for fdef in conn["fields"]:
        if fdef["required"] and not (fields.get(fdef["key"]) or "").strip():
            raise ConnectorError(f"Field '{fdef['key']}' is required.")
    base_name = os.path.splitext(os.path.basename(filename or ""))[0] if filename else ""

    if conn["upload"]:
        if not file_bytes:
            raise ConnectorError("A file upload is required for this connector.")
        if len(file_bytes) > MAX_UPLOAD_BYTES:
            raise ConnectorError("File is larger than the 50 MB limit.")
        if ctype == "csv":
            return parse_csv(file_bytes, filename or ""), base_name or "csv_upload"
        if ctype == "excel":
            return parse_excel(file_bytes, (fields.get("sheet") or "").strip() or None), base_name or "excel_upload"
        if ctype == "json":
            return parse_json(file_bytes), base_name or "json_upload"
        if ctype == "parquet":
            return parse_parquet(file_bytes), base_name or "parquet_upload"
        if ctype == "sqlite":
            table = (fields.get("table") or "").strip()
            df = parse_sqlite(file_bytes, table or None, fields.get("query"))
            return df, table or base_name or "sqlite_upload"

    if ctype == "url":
        url = fields["url"].strip()
        if "docs.google.com/spreadsheets" in url:
            url = gsheets_csv_url(url)
        data, ct = fetch_url(url)
        name = os.path.splitext(os.path.basename(urlparse(url).path))[0] or "remote_data"
        return parse_by_kind(data, url, ct), name
    if ctype == "gsheets":
        url = gsheets_csv_url(fields["sheet_url"])
        data, _ = fetch_url(url)
        if b"<html" in data[:500].lower():
            raise ConnectorError("Google returned a sign-in page. Share the sheet as 'Anyone with the link'.")
        return parse_csv(data), "google_sheet"
    if ctype == "rest_api":
        headers: dict[str, str] = {}
        raw = (fields.get("headers") or "").strip()
        if raw:
            try:
                headers = {str(k): str(v) for k, v in json.loads(raw).items()}
            except Exception as e:
                raise ConnectorError("headers must be a JSON object.") from e
        headers.setdefault("Accept", "application/json")
        data, _ = fetch_url(fields["url"], headers)
        try:
            obj = json.loads(_decode(data))
        except json.JSONDecodeError as e:
            raise ConnectorError("The endpoint did not return JSON.") from e
        host = urlparse(fields["url"]).hostname or "api"
        return _records_from_json(obj, (fields.get("json_path") or "").strip() or None), host.split(".")[0] + "_api"
    if ctype in ("postgres", "mysql"):
        return _db_query(fields["connection_string"], fields["query"], ctype), f"{ctype}_query"
    if ctype == "sample":
        ds = fields["dataset"].strip()
        try:
            return samples.generate(ds), ds
        except ValueError as e:
            raise ConnectorError(str(e)) from e
    raise ConnectorError(f"Connector '{ctype}' is not implemented.")  # pragma: no cover
