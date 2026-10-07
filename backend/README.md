# DataRat API (backend)

FastAPI + DuckDB (in-memory) + pandas. Implements the contract in [`../SPEC.md`](../SPEC.md).
Boots with three deterministic sample datasets (`sales`, `marketing`, `support_calls`).

## Run locally

```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000     # docs at http://localhost:8000/docs
```

Tests: `pip install pytest && pytest -q`

## Configuration

| env | default | purpose |
|---|---|---|
| `ANTHROPIC_API_KEY` | unset | Enables Claude for the Planner / SQL Engineer / Narrator agents and the brief. Without it a rule-based engine answers questions. (`ANTHROPIC_BASE_URL` alone does not enable the LLM.) |
| `DATARAT_MODEL` | `claude-sonnet-5-5` | Model id |
| `DATARAT_LLM_TIMEOUT` | `45` | Seconds per LLM call |
| `DATARAT_DISABLE_LLM` | unset | `1` forces the rule engine |
| `DATARAT_SQL_MAX_ROWS` | `1000` | Row cap for `/api/sql` |
| `DATARAT_ALLOW_PRIVATE_URLS` | unset | `1` lets the url/rest_api connectors reach private IPs (local dev only) |

## Layout

```
app/main.py          routes (all under /api, CORS *)
app/store.py         DuckDB registry (external file access disabled + locked)
app/connectors.py    csv/tsv, excel, json, parquet, url, gsheets, rest_api, postgres, mysql, sqlite, sample
app/samples.py       seeded sample data (spike 2025-06-14, outage dip 2025-09-01..09)
app/profiling.py     column profiles, quality score, correlations
app/insights.py      KPIs, trend, breakdowns, anomalies (robust z), Holt-Winters forecast, highlights
app/sqlguard.py      read-only SQL guard (SELECT/WITH only, single statement, no file/IO functions)
app/agents/          planner -> sql_engineer -> analyst -> narrator (pipeline.py), llm.py wrapper
app/brief.py         executive brief
```

Safety: uploads capped at 50 MB, remote fetches have a 20 s timeout, 50 MB cap and block private/loopback hosts;
user SQL is guarded and DuckDB runs with `enable_external_access=false`.

## Deploy

`render.yaml` at the repo root defines the `datarat-api` free web service (rootDir `backend`).
Set `ANTHROPIC_API_KEY` in the Render dashboard to turn on the LLM agents.
