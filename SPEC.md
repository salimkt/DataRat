# DataRat — AI Decision Intelligence for Business Leaders

Connect any data source → automatic profiling, KPIs, anomalies, forecasts →
ask questions in plain English → a team of AI agents answers with SQL, charts
and an executive narrative.

## Layout
```
backend/   FastAPI + DuckDB + pandas  (deploys to Render, render.yaml at repo root)
frontend/  React 18 + Vite 5 + TypeScript + Recharts (deploys to GitHub Pages)
```
Node is v18 locally → use Vite 5 (not 6/7). Python 3.10+ (Render uses 3.11).

## Core concepts
- **Dataset**: a table loaded into an in-process DuckDB connection (one per server, in-memory).
  Each dataset has `id` (slug, used as the DuckDB table name), `name`, `source` (connector type),
  `rows`, `columns: [{name, dtype, semantic}]` where semantic ∈ `metric|dimension|date|id|text`.
- Server boots with 3 **sample datasets** preloaded (generated deterministically with numpy seed 42):
  - `sales` (~5k rows, 2 yrs daily): order_id, order_date, region, country, channel, product_category,
    product, customer_segment, units, unit_price, discount, revenue, cost, profit. Include seasonality,
    growth trend and a couple of injected anomalies (e.g. a spike + a dip).
  - `marketing` (~730 rows): date, channel, spend, impressions, clicks, leads, conversions, revenue_attributed.
  - `support_calls` (~3k rows; nods to voice analytics): call_id, call_date, agent, queue, duration_sec,
    wait_sec, sentiment (-1..1), resolved (bool), csat (1-5), topic.

## Connectors (`GET /api/connectors` lists them with a JSON-schema-ish `fields` array for the UI form)
| type | fields | notes |
|---|---|---|
| `csv` | file upload (multipart) | also .tsv |
| `excel` | file upload | .xlsx, first sheet (or `sheet` field) |
| `json` | file upload | array of records or {data:[...]} |
| `parquet` | file upload | |
| `url` | url | remote CSV/JSON/Parquet by extension or content-type (works for public Google Sheets “export?format=csv” links) |
| `gsheets` | sheet_url | converts a share link to the CSV export URL |
| `rest_api` | url, json_path (optional dotted path), headers (optional JSON) | |
| `postgres` | connection_string, query | via SQLAlchemy + psycopg2-binary |
| `mysql` | connection_string, query | via SQLAlchemy + pymysql |
| `sqlite` | file upload, table/query | |
| `sample` | dataset (sales/marketing/support_calls) | reload a sample |

Connector item shape:
`{type, name, category: "file"|"database"|"cloud"|"api", description, icon (lucide icon name), upload: bool, fields:[{key,label,type:"text"|"password"|"textarea"|"select",placeholder?,options?,required}]}`

## REST API (all JSON, prefix `/api`, CORS allow all origins)
- `GET /api/health` → `{status:"ok", llm: bool, version}`
- `GET /api/connectors` → `Connector[]`
- `POST /api/datasets/connect` (multipart/form-data): field `type`, optional `name`, optional `file`, plus connector fields as form fields → `Dataset`
- `GET /api/datasets` → `Dataset[]`
- `GET /api/datasets/{id}` → `Dataset`
- `DELETE /api/datasets/{id}` → `{ok:true}` (samples can be deleted too)
- `GET /api/datasets/{id}/preview?limit=100` → `{columns: string[], rows: any[][]}`
- `GET /api/datasets/{id}/profile` → `Profile`:
  ```
  {rows, columns, missing_cells_pct, duplicate_rows, quality_score (0-100),
   column_profiles:[{name,dtype,semantic,missing_pct,unique,min?,max?,mean?,median?,std?,top_values?:[{value,count}],histogram?:[{bin,count}]}],
   correlations:{columns:string[], matrix:number[][]}}
  ```
- `GET /api/datasets/{id}/insights` → `Insights` (auto, deterministic, no LLM needed):
  ```
  {kpis:[{label, value:number, format:"currency"|"number"|"percent", change_pct:number|null, period:string, sparkline:number[]}],
   trend:{date_column, metric, series:[{date, value}]} | null,
   breakdowns:[{dimension, metric, data:[{label, value}]}],   // top 3 dimension x primary metric
   anomalies:[{date, metric, value, expected, zscore, direction:"spike"|"dip"}],
   forecast:{metric, history:[{date,value}], forecast:[{date,value,lower,upper}], method} | null,  // 30 periods, Holt-Winters via statsmodels or linear+seasonal fallback
   highlights: string[]}  // 3-6 plain-English bullet findings computed from data
  ```
  Primary metric heuristic: prefer columns named revenue/sales/amount/profit/spend, else first numeric metric. Aggregate to daily (or weekly if >400 days ⇒ still daily ok) by date column.
- `POST /api/ask` body `{dataset_id, question}` → `AskResult` — **the multi-agent pipeline**:
  ```
  {question, steps:[{agent, title, detail, duration_ms}],   // trace shown in UI as an agent timeline
   sql, columns:string[], rows:any[][] (≤500),
   chart:{type:"bar"|"line"|"area"|"pie"|"scatter"|"table"|"kpi", x?, y?:string[] , title},
   narrative: string (markdown, 2-5 sentences for an executive),
   recommendations: string[],
   follow_ups: string[]}
  ```
  Agents (each a small class/function, in `backend/app/agents/`):
  1. **Planner** – understands intent, picks relevant columns from schema.
  2. **SQL Engineer** – writes DuckDB SQL; executes; on error retries up to 2× feeding back the error. Read-only guard: only SELECT/WITH allowed.
  3. **Analyst** – computes stats on the result (totals, top/bottom, growth) and chooses chart spec.
  4. **Narrator** – writes executive summary + recommendations + follow-ups.
  LLM: Anthropic Claude via the `anthropic` SDK when `ANTHROPIC_API_KEY` set (model from env `DATARAT_MODEL`, default `claude-sonnet-5-5`; honor `ANTHROPIC_BASE_URL` automatically via SDK). **Without a key, a rule-based fallback must still work** for common questions: "total X by Y", "top N Y by X", "X over time / trend / monthly", "average X", "how many rows", "compare", "growth" — keyword matching against column names. The fallback is what the public demo shows, so make it solid.
- `POST /api/brief` body `{dataset_id}` → `{title, generated_at, summary (markdown), sections:[{heading, body (markdown)}], kpis (same as insights.kpis)}` — an "Executive Brief" (LLM if available, else templated from insights).
- `POST /api/sql` body `{dataset_id?, sql}` → `{columns, rows}` (read-only guard; all datasets queryable by table name).

## Frontend (React + Vite + TS, `base: '/DataRat/'`)
- API base from `import.meta.env.VITE_API_URL` default `https://datarat-api.onrender.com`; settings drawer lets user override (stored in localStorage). Show a friendly "Waking up the analysis server (Render free tier, ~30-50s)…" state with retry.
- Pages (hash router so GH Pages works): 
  - **Overview** (executive dashboard for selected dataset): KPI cards w/ sparklines + change, trend chart with forecast band, breakdown bars, anomaly list, highlights.
  - **Connect**: grid of connector cards by category → modal form → upload/connect → toast.
  - **Datasets**: list, preview table, profile (quality score gauge, column cards, histograms, correlation heatmap).
  - **Ask AI**: chat-like; shows agent timeline (Planner → SQL → Analyst → Narrator), SQL (collapsible), chart, table, narrative, recommendations, follow-up chips. Suggested starter questions per dataset.
  - **SQL Lab**: editor (textarea w/ monospace), run, results table, chart toggle.
  - **Executive Brief**: generate & print-friendly view (window.print), copy as markdown.
- Global dataset switcher in top bar. Sidebar nav. Polished dark/light theme, professional (think Linear/Vercel style), responsive. Icons: lucide-react.
- GitHub Pages deploy via `.github/workflows/deploy-pages.yml` (build frontend, upload `frontend/dist`).
