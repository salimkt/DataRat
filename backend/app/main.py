"""DataRat API — FastAPI app."""
from __future__ import annotations

import logging
import os
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool
from pydantic import BaseModel
from starlette.datastructures import UploadFile

from . import connectors, samples
from .agents import llm, pipeline
from .brief import build_brief
from .connectors import ConnectorError
from .insights import compute_insights
from .profiling import profile
from .sqlguard import UnsafeSQLError, ensure_read_only
from .store import DatasetNotFound, store
from .utils import to_jsonable

VERSION = "1.0.0"
log = logging.getLogger("datarat")

app = FastAPI(title="DataRat API", version=VERSION,
              description="AI decision intelligence: connect data, get insights, ask questions.")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
                   allow_credentials=False)


def load_samples() -> None:
    for ds_id, title in samples.SAMPLE_META.items():
        if ds_id not in store.datasets:
            store.register(samples.generate(ds_id), title, "sample", dataset_id=ds_id, replace=True)


load_samples()


def ok(data: Any, status: int = 200) -> JSONResponse:
    return JSONResponse(to_jsonable(data), status_code=status)


@app.exception_handler(DatasetNotFound)
async def _not_found(_: Request, exc: DatasetNotFound):
    return JSONResponse({"detail": f"Dataset '{exc.args[0]}' not found."}, status_code=404)


@app.exception_handler(RequestValidationError)
async def _validation(_: Request, exc: RequestValidationError):
    msgs = "; ".join(f"{'.'.join(str(p) for p in e.get('loc', [])[1:])}: {e.get('msg')}" for e in exc.errors())
    return JSONResponse({"detail": msgs or "Invalid request."}, status_code=422)


# ------------------------------------------------------------------ models
class AskBody(BaseModel):
    dataset_id: str
    question: str


class BriefBody(BaseModel):
    dataset_id: str


class SQLBody(BaseModel):
    dataset_id: Optional[str] = None
    sql: str


# ------------------------------------------------------------------ routes
@app.get("/")
def root():
    return {"name": "DataRat API", "version": VERSION, "docs": "/docs", "health": "/api/health"}


@app.get("/api/health")
def health():
    return {"status": "ok", "llm": llm.llm_available(), "provider": llm.provider(),
            "model": llm.model_name() if llm.llm_available() else None, "version": VERSION}


@app.get("/api/connectors")
def list_connectors():
    return connectors.CONNECTORS


@app.post("/api/datasets/connect")
async def connect(request: Request):
    cl = request.headers.get("content-length")
    if cl and cl.isdigit() and int(cl) > connectors.MAX_UPLOAD_BYTES + 1024 * 1024:
        raise HTTPException(413, "Upload is larger than the 50 MB limit.")
    form = await request.form()
    ctype = str(form.get("type") or "").strip()
    if not ctype:
        raise HTTPException(400, "Field 'type' is required.")
    name = str(form.get("name") or "").strip()
    fields: dict[str, str] = {}
    file_bytes: bytes | None = None
    filename: str | None = None
    for key, value in form.multi_items():
        if isinstance(value, UploadFile):
            if key == "file" or file_bytes is None:
                buf = bytearray()
                while True:
                    chunk = await value.read(1024 * 1024)
                    if not chunk:
                        break
                    buf.extend(chunk)
                    if len(buf) > connectors.MAX_UPLOAD_BYTES:
                        raise HTTPException(413, "File is larger than the 50 MB limit.")
                file_bytes = bytes(buf)
                filename = value.filename
        elif key not in ("type", "name"):
            fields[key] = str(value)
    def _load() -> dict[str, Any]:
        df, suggested = connectors.load(ctype, fields, file_bytes, filename)
        if ctype == "sample":
            ds = fields.get("dataset", "").strip()
            return store.register(df, name or samples.SAMPLE_META[ds], "sample", dataset_id=ds, replace=True)
        return store.register(df, name or suggested, ctype, dataset_id=name or suggested)

    try:
        meta = await run_in_threadpool(_load)
    except (ConnectorError, UnsafeSQLError, ValueError) as e:
        raise HTTPException(400, str(e))
    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001
        log.exception("connect failed")
        raise HTTPException(400, f"Could not load data: {str(e).splitlines()[0][:300]}")
    return ok(meta)


@app.get("/api/datasets")
def list_datasets():
    return ok(store.list())


@app.get("/api/datasets/{ds_id}")
def get_dataset(ds_id: str):
    return ok(store.get(ds_id))


@app.delete("/api/datasets/{ds_id}")
def delete_dataset(ds_id: str):
    store.delete(ds_id)
    return {"ok": True}


@app.get("/api/datasets/{ds_id}/preview")
def preview(ds_id: str, limit: int = 100):
    return ok(store.preview(ds_id, limit))


@app.get("/api/datasets/{ds_id}/profile")
def get_profile(ds_id: str):
    return ok(profile(ds_id))


@app.get("/api/datasets/{ds_id}/insights")
def get_insights(ds_id: str):
    return ok(compute_insights(ds_id))


@app.post("/api/ask")
def ask(body: AskBody):
    store.get(body.dataset_id)
    q = body.question.strip()
    if not q:
        raise HTTPException(400, "Question is empty.")
    if len(q) > 1000:
        raise HTTPException(400, "Question is too long (max 1000 characters).")
    try:
        return ok(pipeline.run(body.dataset_id, q))
    except UnsafeSQLError as e:
        raise HTTPException(400, str(e))
    except Exception as e:  # noqa: BLE001
        log.exception("ask failed")
        raise HTTPException(500, f"Analysis failed: {str(e).splitlines()[0][:300]}")


@app.post("/api/brief")
def brief(body: BriefBody):
    store.get(body.dataset_id)
    return ok(build_brief(body.dataset_id))


@app.post("/api/sql")
def run_sql(body: SQLBody):
    if body.dataset_id:
        store.get(body.dataset_id)
    try:
        clean = ensure_read_only(body.sql)
        cols, rows, _ = store.query(clean, limit=int(os.environ.get("DATARAT_SQL_MAX_ROWS", "1000")))
    except UnsafeSQLError as e:
        raise HTTPException(400, str(e))
    except Exception as e:  # noqa: BLE001
        raise HTTPException(400, f"SQL error: {str(e).splitlines()[0][:400]}")
    return ok({"columns": cols, "rows": rows})
