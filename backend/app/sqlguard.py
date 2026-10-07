"""Read-only SQL guard used by /api/sql and the SQL Engineer agent."""
from __future__ import annotations

import re


class UnsafeSQLError(ValueError):
    pass


BANNED_KEYWORDS = [
    "ATTACH", "DETACH", "COPY", "INSTALL", "LOAD", "PRAGMA", "EXPORT", "IMPORT",
    "INSERT", "UPDATE", "DELETE", "DROP", "CREATE", "ALTER", "TRUNCATE", "MERGE",
    "CALL", "SET", "RESET", "CHECKPOINT", "VACUUM", "USE", "GRANT", "REVOKE",
    "BEGIN", "COMMIT", "ROLLBACK", "ANALYZE", "UPSERT", "SECRET", "EXECUTE", "PREPARE",
    "DEALLOCATE", "SUMMARIZE", "DESCRIBE", "SHOW",
]
BANNED_FUNCTIONS = [
    "read_csv", "read_csv_auto", "read_parquet", "parquet_scan", "parquet_metadata",
    "parquet_schema", "read_json", "read_json_auto", "read_json_objects", "read_ndjson",
    "read_ndjson_auto", "read_ndjson_objects", "read_text", "read_blob", "sniff_csv",
    "csv_scan", "glob", "iceberg_scan", "delta_scan", "sqlite_scan", "sqlite_attach",
    "postgres_scan", "postgres_attach", "mysql_scan", "getenv", "duckdb_secrets",
    "duckdb_settings", "duckdb_extensions", "query_table", "query", "read_xlsx",
    "st_read", "http_get", "httpfs",
]

_KW_RE = re.compile(r"\b(" + "|".join(BANNED_KEYWORDS) + r")\b", re.IGNORECASE)
_FN_RE = re.compile(r"\b(" + "|".join(BANNED_FUNCTIONS) + r")\s*\(", re.IGNORECASE)


def _strip_comments(sql: str) -> str:
    sql = re.sub(r"/\*.*?\*/", " ", sql, flags=re.DOTALL)
    sql = re.sub(r"--[^\n]*", " ", sql)
    return sql


def _mask_literals(sql: str) -> str:
    """Replace contents of '...' string literals and "..." identifiers with blanks."""
    out = []
    i, n = 0, len(sql)
    while i < n:
        c = sql[i]
        if c in ("'", '"'):
            q = c
            j = i + 1
            while j < n:
                if sql[j] == q:
                    if j + 1 < n and sql[j + 1] == q:  # escaped quote
                        j += 2
                        continue
                    break
                j += 1
            out.append(q + "x" * max(0, min(j, n) - i - 1) + (q if j < n else ""))
            i = j + 1
        else:
            out.append(c)
            i += 1
    return "".join(out)


def ensure_read_only(sql: str) -> str:
    """Validate ``sql`` is a single read-only SELECT/WITH statement.

    Returns the cleaned SQL (comments stripped, trailing semicolons removed).
    Raises ``UnsafeSQLError`` otherwise.
    """
    if not sql or not sql.strip():
        raise UnsafeSQLError("SQL is empty.")
    if len(sql) > 20000:
        raise UnsafeSQLError("SQL is too long.")
    cleaned = _strip_comments(sql).strip()
    cleaned = re.sub(r"[;\s]+$", "", cleaned)
    masked = _mask_literals(cleaned)
    if ";" in masked:
        raise UnsafeSQLError("Only a single SQL statement is allowed.")
    first = re.match(r"\s*\(*\s*([A-Za-z]+)", masked)
    if not first or first.group(1).upper() not in ("SELECT", "WITH"):
        raise UnsafeSQLError("Only read-only SELECT / WITH queries are allowed.")
    m = _KW_RE.search(masked)
    if m:
        raise UnsafeSQLError(f"Keyword '{m.group(1).upper()}' is not allowed (read-only mode).")
    m = _FN_RE.search(masked)
    if m:
        raise UnsafeSQLError(f"Function '{m.group(1)}' is not allowed.")
    # FROM 'file.csv' style replacement scans
    if re.search(r"\b(FROM|JOIN)\s+'", masked, re.IGNORECASE):
        raise UnsafeSQLError("Reading files or URLs from SQL is not allowed.")
    return cleaned
