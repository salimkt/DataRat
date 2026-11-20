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

Tests: `pip install pytest && pytest -q` (153 tests, offline; no API key needed)

On macOS, LightGBM needs the OpenMP runtime (`brew install libomp`). Without it the wholesale
planner falls back to scikit-learn's HistGradientBoosting automatically (`DATARAT_GBM=sklearn` forces it).

## Configuration

| env | default | purpose |
|---|---|---|
| `ANTHROPIC_API_KEY` | unset | Enables Claude for the Planner / SQL Engineer / Narrator agents and the brief. Without it a rule-based engine answers questions. (`ANTHROPIC_BASE_URL` alone does not enable the LLM.) |
| `DATARAT_MODEL` | `claude-sonnet-5-5` | Model id |
| `DATARAT_LLM_TIMEOUT` | `45` | Seconds per LLM call |
| `DATARAT_DISABLE_LLM` | unset | `1` forces the rule engine |
| `DATARAT_SQL_MAX_ROWS` | `1000` | Row cap for `/api/sql` |
| `DATARAT_ALLOW_PRIVATE_URLS` | unset | `1` lets the url/rest_api connectors reach private IPs (local dev only) |
| `DATARAT_DATE_ORDER` | `dmy` | how to read `nn/nn/yyyy` dates when every value is ambiguous (all parts <= 12): `dmy` or `mdy`. A day above 12 in the column decides automatically. |
| `DATARAT_GBM` | auto | `lightgbm` or `sklearn`: gradient-boosting backend for the wholesale planner |
| `DATARAT_WHOLESALE_DIR` | `backend/var/wholesale` | where the planner CLI writes models, reports and the WhatsApp outbox |
| `DATARAT_WHATSAPP_MODE` | `dry_run` | `off`, `dry_run` (write payloads to an outbox file) or `live` (WhatsApp Cloud API; billed per message) |
| `DATARAT_WHATSAPP_TOKEN` / `DATARAT_WHATSAPP_PHONE_ID` | unset | only used when the mode is `live` |
| `DATARAT_WHATSAPP_TEMPLATE` / `DATARAT_WHATSAPP_LANG` | `reorder_reminder` / `en` | approved template name and language |

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
app/wholesale/       wholesale planner: ingest (Tally/Marg/Vyapar) -> features -> demand + reorder models
                     -> PO plan + call list -> report / Excel / WhatsApp stub; routes under /api/wholesale
```

## Wholesale planner

Design, model comparison and rollout plan: [`../docs/wholesale-forecasting.md`](../docs/wholesale-forecasting.md).

```bash
python -m app.wholesale demo                          # synthetic exports -> train -> report (+ dry-run outbox)
python -m app.wholesale generate --out var/wholesale/data --erp tally
python -m app.wholesale train  --data var/wholesale/data   # weekly
python -m app.wholesale report --data var/wholesale/data --notify   # daily
```

The export folder holds `invoices.csv|xlsx|xml` (item-wise sales register or Tally XML) and,
optionally, `items`, `parties`, `stock` (daily closing stock) and `open_pos` (CSV or Excel).

Safety: uploads capped at 50 MB, remote fetches have a 20 s timeout, 50 MB cap and block private/loopback hosts;
user SQL is guarded and DuckDB runs with `enable_external_access=false`.

## Deploy

`render.yaml` at the repo root defines the `datarat-api` free web service (rootDir `backend`).
Set `ANTHROPIC_API_KEY` in the Render dashboard to turn on the LLM agents.
