# 🐀 DataRat — AI Decision Intelligence for Business Leaders

**Connect any data source → instant KPIs, anomalies and forecasts → ask questions in plain English → a team of AI agents answers with SQL, charts and an executive narrative.**

**Live demo:** https://salimkt.github.io/DataRat/ &nbsp;·&nbsp; **API:** https://datarat-api.onrender.com/docs

> The API runs on Render's free tier, so the first request can take 30–50 s while the server wakes up. The UI shows a banner and retries automatically.

---

## Why
Business leaders don't want another BI tool to learn. They want answers like *"Which region is driving the profit drop and what should we do?"* DataRat turns raw tables into a briefing for executives in seconds, and every number links back to the SQL that produced it.

## Features
| | |
|---|---|
| 🔌 **11 connectors** | CSV/TSV, Excel, JSON, Parquet, SQLite, PostgreSQL, MySQL, Google Sheets, remote URL, REST API, sample datasets |
| 📊 **Executive dashboard** | Auto-detected KPIs with 30-day change and sparklines, trend chart with a **Holt-Winters forecast** and confidence band, top breakdowns, plain-English highlights |
| 🚨 **Anomaly detection** | Rolling z-score spikes and dips on the primary metric |
| 🧪 **Data profiling** | Quality score, missing/duplicate analysis, per-column stats and histograms, correlation heatmap, semantic column typing (metric, dimension, date, id) |
| 🤖 **Ask AI (multi-agent)** | Planner → SQL Engineer (self-correcting, up to 2 retries) → Analyst (stats and chart choice) → Narrator (executive summary, recommendations, follow-ups), shown as a live agent timeline |
| 🧑‍💻 **SQL Lab** | Run read-only DuckDB SQL across all connected datasets, with chart toggle |
| 📝 **Executive Brief** | One-click board-ready briefing you can print or export as Markdown |

## Architecture
```
┌────────────── GitHub Pages ──────────────┐        ┌──────────────── Render ────────────────┐
│ React 18 + Vite + TypeScript + Recharts  │  HTTPS │ FastAPI                                │
│ Overview · Connect · Datasets · Ask AI   │ ─────► │  ├─ Connectors (files, DBs, APIs)      │
│ SQL Lab · Executive Brief                │  JSON  │  ├─ DuckDB in-memory analytics engine  │
└──────────────────────────────────────────┘        │  ├─ Profiling · Insights · Forecasting │
                                                    │  └─ Agents: Planner → SQL → Analyst →  │
                                                    │           Narrator  (Claude or rules)  │
                                                    └────────────────────────────────────────┘
```
- **LLM optional.** With `ANTHROPIC_API_KEY` set, the agents use Claude and validate its JSON output against a schema. Without a key, a deterministic rule-based engine answers common business questions: totals, top-N, trends, averages, counts, comparisons, growth, shares, correlations, derived metrics (conversion rate, ROAS, CTR, margin, AOV) and relative date filters.
- **Safe by design.** Read-only SQL guard, DuckDB external file access disabled, upload and remote-fetch size limits, and blocking of private network addresses.

## Run locally
```bash
# backend
cd backend && python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && uvicorn app.main:app --reload --port 8000
pytest -q          # 114 tests

# frontend
cd frontend && npm install
VITE_API_URL=http://localhost:8000 npm run dev
```

## Deploy
- **Frontend:** pushing to `main` runs `.github/workflows/deploy-pages.yml`, which deploys to GitHub Pages. Set the repo variable `VITE_API_URL` to point the UI at a different API.
- **Backend:** in Render, go to **New → Blueprint**, pick this repo, and it reads `render.yaml`. You can optionally add `ANTHROPIC_API_KEY` to turn on the Claude-powered agents.

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/salimkt/DataRat)

## Skills demonstrated
`SQL (DuckDB)` · `Python / pandas / NumPy` · `statsmodels time-series forecasting` · `anomaly detection` · `data quality profiling` · `KPI design` · `data visualization` · `ETL / data connectors` · `FastAPI` · `React + TypeScript` · `LLM multi-agent systems` · `CI/CD`

---
Built by **[Muhammed Salim K T](https://github.com/salimkt)**, AI/ML engineer focused on LLM systems, RAG, multi-agent orchestration and analytics.
