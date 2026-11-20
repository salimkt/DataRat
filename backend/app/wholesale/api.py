"""FastAPI routes for the wholesale planner (mounted under /api/wholesale by app.main).

POST /api/wholesale/demo        synthetic data -> train -> plan; registers the tables in DuckDB
POST /api/wholesale/run         plan from datasets the user connected (e.g. an uploaded Tally export)
GET  /api/wholesale/report      latest report (JSON)
GET  /api/wholesale/report.xlsx latest report as an Excel workbook
GET  /api/wholesale/outbox      WhatsApp payloads that *would* be sent (dry run; nothing is sent)

Training runs synchronously in FastAPI's thread pool with the fast config, so a demo run takes
seconds locally and well under the UI's 3-minute timeout on a small instance.
"""
from __future__ import annotations

import threading
from typing import Any, Optional

import pandas as pd
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field

from ..store import store
from ..utils import to_jsonable
from . import ingest, notify
from .config import PlannerConfig, WhatsAppSettings
from .ingest import IngestError, WholesaleData
from .pipeline import WholesalePlanner, demo_data
from .report import write_excel

router = APIRouter(prefix="/api/wholesale", tags=["wholesale"])
_lock = threading.Lock()
_state: dict[str, Any] = {"report": None, "data": None}

DEMO_START = "2025-01-01"


class DemoBody(BaseModel):
    seed: int = Field(7, ge=0, le=10_000)
    n_buyers: int = Field(60, ge=20, le=200)


class RunBody(BaseModel):
    invoices_dataset_id: str
    items_dataset_id: Optional[str] = None
    parties_dataset_id: Optional[str] = None
    stock_dataset_id: Optional[str] = None
    open_pos_dataset_id: Optional[str] = None
    erp: str = "auto"


def _register(data: WholesaleData, report: dict[str, Any], source: str) -> list[str]:
    """Expose the clean tables and outputs to SQL Lab / Ask AI / the dashboard."""
    tables = {
        "wholesale_invoices": data.lines,
        "wholesale_po_plan": pd.DataFrame(report["po_plan"]),
        "wholesale_call_list": pd.DataFrame(report["call_list"]),
        "wholesale_forecast": pd.DataFrame([
            {"sku": r["sku"], "item_name": r["item_name"], "commodity": r["commodity"], "date": d["date"],
             "forecast": d["value"], "unit": r["unit"]} for r in report["forecast"] for d in r["daily"]]),
    }
    titles = {"wholesale_invoices": "Wholesale invoice lines (clean)", "wholesale_po_plan": "Wholesale PO plan",
              "wholesale_call_list": "Wholesale buyer call list", "wholesale_forecast": "Wholesale 30-day forecast"}
    done = []
    for ds_id, df in tables.items():
        if len(df):
            store.register(df, titles[ds_id], source, dataset_id=ds_id, replace=True)
            done.append(ds_id)
    return done


def _run(data: WholesaleData, cfg: PlannerConfig, source: str) -> dict[str, Any]:
    planner = WholesalePlanner(cfg)
    try:
        planner.fit(data)
    except ValueError as e:
        raise HTTPException(400, f"Not enough history to train: {e}")
    report = planner.plan(data)
    report["datasets"] = _register(data, report, source)
    _state.update(report=report, data=data)
    return report


@router.post("/demo")
def demo(body: DemoBody | None = None):
    body = body or DemoBody()
    with _lock:
        _, data = demo_data(seed=body.seed, n_buyers=body.n_buyers, start=DEMO_START)
        report = _run(data, PlannerConfig.fast(seed=body.seed), "wholesale_demo")
    return JSONResponse(to_jsonable(report))


@router.post("/run")
def run(body: RunBody):
    def df(ds: Optional[str]) -> Optional[pd.DataFrame]:
        return store.df(ds) if ds else None

    with _lock:
        try:
            data = ingest.load(df(body.invoices_dataset_id), df(body.items_dataset_id), df(body.parties_dataset_id),
                               df(body.stock_dataset_id), df(body.open_pos_dataset_id), erp=body.erp)
        except IngestError as e:
            raise HTTPException(400, str(e))
        report = _run(data, PlannerConfig.fast(), "wholesale")
    return JSONResponse(to_jsonable(report))


def _latest() -> dict[str, Any]:
    if _state["report"] is None:
        raise HTTPException(404, "No wholesale report yet. POST /api/wholesale/demo or /api/wholesale/run first.")
    return _state["report"]


@router.get("/report")
def latest_report():
    return JSONResponse(to_jsonable(_latest()))


@router.get("/report.xlsx")
def latest_report_xlsx():
    report = _latest()
    body = write_excel(report)
    name = f"wholesale_plan_{report['summary']['as_of']}.xlsx"
    return Response(body, media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    headers={"Content-Disposition": f'attachment; filename="{name}"'})


@router.get("/outbox")
def outbox():
    """Dry-run WhatsApp payloads for today's list. This endpoint never sends anything."""
    report = _latest()
    settings = WhatsAppSettings.from_env()
    settings.mode = "dry_run"
    payloads = notify.build_payloads(report.get("call_list", []), settings)
    return JSONResponse(to_jsonable({"mode": "dry_run", "messages": len(payloads), "payloads": payloads}))
