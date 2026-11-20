"""Unified daily report: purchase-order plan + prioritised buyer outreach list + forecasts.

``build_report`` returns a JSON-ready dict (served by the API and saved next to the workbook);
``write_excel`` renders the same content as a formatted workbook for the shop floor.
"""
from __future__ import annotations

import datetime as _dt
import io
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..utils import to_jsonable
from .config import PlannerConfig
from .ingest import WholesaleData
from .seasonality import upcoming_events


def _inr(x: float) -> str:
    """Rs 12,34,567 style (Indian digit grouping)."""
    x = float(x or 0)
    neg, x = x < 0, abs(round(x))
    s = f"{int(x)}"
    if len(s) > 3:
        head, tail = s[:-3], s[-3:]
        parts = []
        while len(head) > 2:
            parts.insert(0, head[-2:])
            head = head[:-2]
        if head:
            parts.insert(0, head)
        s = ",".join(parts + [tail])
    return ("-" if neg else "") + "Rs " + s


def build_report(data: WholesaleData, panel: pd.DataFrame, forecast: pd.DataFrame, plan: pd.DataFrame,
                 calls: pd.DataFrame, metrics: dict[str, Any], cfg: PlannerConfig,
                 model_info: dict[str, Any] | None = None, sigma: dict[str, list[float]] | None = None,
                 sigma_rel: list[float] | None = None) -> dict[str, Any]:
    as_of = pd.Timestamp(data.as_of)
    to_order = plan[plan["status"].isin(["URGENT", "ORDER"])] if len(plan) else plan
    expedite = plan[plan["status"] == "EXPEDITE"] if len(plan) else plan
    urgent = plan[plan["status"] == "URGENT"] if len(plan) else plan
    events = upcoming_events(as_of, 30, cfg.region)
    n_whatsapp = int((calls["channel"] == "WhatsApp").sum()) if len(calls) else 0
    summary = {
        "as_of": as_of.date().isoformat(),
        "business": cfg.business_name,
        "items": int(len(plan)),
        "items_to_order": int(len(to_order)),
        "urgent_items": int(len(urgent)),
        "expedite_items": int(len(expedite)),
        "po_value": float(to_order["est_po_value"].sum()) if len(to_order) else 0.0,
        "buyers_to_contact": int(len(calls)),
        "calls": int(len(calls) - n_whatsapp),
        "whatsapp": n_whatsapp,
        "expected_order_value": float(calls["expected_value"].sum()) if len(calls) else 0.0,
        "upcoming_events": events,
        "history": {"from": data.lines["date"].min().date().isoformat(), "to": data.lines["date"].max().date().isoformat(),
                    "lines": int(len(data.lines)), "buyers": int(data.lines["buyer_id"].nunique()),
                    "stockout_days_share": round(float(panel["stockout"].mean()), 4)},
    }
    report = {
        "title": f"Wholesale plan for {as_of.strftime('%d %b %Y')}",
        "generated_at": _dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "summary": summary,
        "highlights": _highlights(panel, forecast, plan, calls, metrics, events, summary),
        "po_plan": plan.to_dict("records"),
        "call_list": calls.to_dict("records"),
        "forecast": _forecast_table(forecast, data, panel, sigma or {}, sigma_rel or []),
        "metrics": metrics,
        "data_quality": data.quality,
        "model": model_info or {},
        "config": cfg.to_dict(),
    }
    return to_jsonable(report)


def _forecast_table(forecast: pd.DataFrame, data: WholesaleData, panel: pd.DataFrame,
                    sigma: dict[str, list[float]], sigma_rel: list[float], history_days: int = 56) -> list[dict[str, Any]]:
    """Per item: 7/15/30-day totals, the daily path with an ~80 % band (+-1.28 x the backtest
    one-day error) and the recent unconstrained demand history for charting."""
    items = data.items.set_index("sku")
    out = []
    for sku, g in forecast.groupby("sku"):
        g = g.sort_values("k")
        f = g["forecast"].values
        if sku in sigma and sigma[sku]:
            sd = np.full(len(f), float(sigma[sku][0]))
        else:
            sd = (sigma_rel[0] if sigma_rel else 0.5) * f
        hist = panel[panel["sku"] == sku].sort_values("date").tail(history_days)
        out.append({
            "sku": sku, "item_name": items["item_name"].get(sku, sku), "commodity": items["commodity"].get(sku, ""),
            "unit": items["base_unit"].get(sku, ""), "method": g["method"].iat[0],
            "h7": round(float(f[:7].sum()), 1), "h15": round(float(f[:15].sum()), 1), "h30": round(float(f[:30].sum()), 1),
            "daily": [{"date": d.date().isoformat(), "value": round(float(v), 2),
                       "lower": round(float(max(v - 1.28 * e, 0.0)), 2), "upper": round(float(v + 1.28 * e), 2)}
                      for d, v, e in zip(pd.DatetimeIndex(g["date"]), f, sd)],
            "history": [{"date": d.date().isoformat(), "value": round(float(v), 2)}
                        for d, v in zip(pd.DatetimeIndex(hist["date"]), hist["demand"])],
        })
    return out


def _highlights(panel, forecast, plan, calls, metrics, events, summary) -> list[str]:
    h = []
    if summary["items_to_order"]:
        urgent = plan[plan["status"] == "URGENT"]
        txt = f"{summary['items_to_order']} item(s) need a purchase order today, about {_inr(summary['po_value'])} at cost."
        if len(urgent):
            names = ", ".join(f"{r.item_name} (runs out in {int(r.stockout_in_days)} d)" if pd.notna(r.stockout_in_days) else r.item_name
                              for r in urgent.head(3).itertuples())
            txt += f" Urgent: {names}."
        h.append(txt)
    else:
        h.append("No purchase orders are needed today; every item is above its reorder point.")
    unknown = plan[plan["status"] == "CHECK STOCK"] if len(plan) else plan
    if len(unknown):
        h.append(f"No stock snapshot for {len(unknown)} item(s): the plan shows their reorder point (min) and "
                 "order-up-to level (max); add a closing-stock export to get exact order quantities.")
    exp = plan[plan["status"] == "EXPEDITE"] if len(plan) else plan
    if len(exp):
        names = ", ".join(f"{r.item_name} (runs out in {int(r.stockout_in_days)} d)" for r in exp.head(3).itertuples())
        h.append(f"Chase suppliers: open POs land after the projected stockout for {names}.")
    if events:
        e = events[0]
        end = panel["date"].max()
        recent = panel[panel["date"] > end - pd.Timedelta(days=28)].groupby("commodity")["demand"].sum() / 4
        nxt = forecast[forecast["k"] <= 7].merge(panel.drop_duplicates("sku")[["sku", "commodity"]], on="sku")
        nxt = nxt.groupby("commodity")["forecast"].sum()
        ch = ((nxt / recent.reindex(nxt.index)) - 1).dropna()
        ch = ch[np.isfinite(ch)]
        if len(ch):
            top = ch.sort_values(ascending=False).head(2)
            moves = ", ".join(f"{c} {v:+.0%}" for c, v in top.items())
            h.append(f"{e['festival']} is in {e['in_days']} day(s). Next-7-day demand vs a normal recent week: {moves}.")
        else:
            h.append(f"{e['festival']} is in {e['in_days']} day(s).")
    if summary["buyers_to_contact"]:
        h.append(f"Contact {summary['buyers_to_contact']} buyers today ({summary['calls']} calls, {summary['whatsapp']} WhatsApp); "
                 f"expected orders worth about {_inr(summary['expected_order_value'])}.")
        wb = calls[calls["action"] == "Win-back call"] if len(calls) else calls
        if len(wb):
            h.append(f"{len(wb)} regular buyer(s) have gone quiet and are flagged for a win-back call.")
    dm = (metrics or {}).get("demand", {})
    if dm.get("models"):
        ours = dm["models"][0]
        hw = next((m for m in dm["models"] if m["model"] == "holt-winters"), None)
        txt = f"Demand model accuracy on the last {dm.get('test_origins', '?')} weekly backtests: 30-day WAPE {ours.get('wape_30', 0):.0%}"
        if hw:
            txt += f" (Holt-Winters {hw.get('wape_30', 0):.0%})"
        h.append(txt + ".")
    if summary["history"]["stockout_days_share"] > 0.03:
        h.append(f"Stock ran out on {summary['history']['stockout_days_share']:.0%} of item-days in the history; "
                 "those sales were treated as censored demand when training.")
    return h


# ============================================================================ Excel
HEADER_FILL = "1F2937"
STATUS_FILL = {"URGENT": "FCA5A5", "EXPEDITE": "FDBA74", "ORDER": "FDE68A", "CHECK STOCK": "E5E7EB", "OK": "D1FAE5"}


def write_excel(report: dict[str, Any], path: str | Path | None = None) -> bytes:
    """Render the report as a workbook. Returns the bytes; also writes ``path`` if given."""
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter

    wb = Workbook()
    head_font = Font(bold=True, color="FFFFFF")
    head_fill = PatternFill("solid", fgColor=HEADER_FILL)

    def table(ws, rows: list[dict], columns: list[tuple[str, str, str | None]], start_row: int = 1):
        for j, (_, title, _) in enumerate(columns, start=1):
            c = ws.cell(row=start_row, column=j, value=title)
            c.font, c.fill = head_font, head_fill
            c.alignment = Alignment(wrap_text=True, vertical="center")
        for i, r in enumerate(rows, start=start_row + 1):
            for j, (key, _, fmt) in enumerate(columns, start=1):
                v = r.get(key)
                c = ws.cell(row=i, column=j, value=v)
                if fmt:
                    c.number_format = fmt
        for j, (key, title, _) in enumerate(columns, start=1):
            width = max([len(str(title))] + [len(str(r.get(key, ""))) for r in rows[:200]]) + 2
            ws.column_dimensions[get_column_letter(j)].width = min(max(width, 8), 60)
        ws.freeze_panes = ws.cell(row=start_row + 1, column=1)
        if rows:
            ws.auto_filter.ref = f"A{start_row}:{get_column_letter(len(columns))}{start_row + len(rows)}"

    s = report["summary"]
    ws = wb.active
    ws.title = "Summary"
    ws["A1"] = report["title"]
    ws["A1"].font = Font(bold=True, size=14)
    ws["A2"] = f"{s['business']} - generated {report['generated_at']}"
    kv = [("Items to order", s["items_to_order"]), ("Urgent items", s["urgent_items"]),
          ("Open POs to expedite", s.get("expedite_items", 0)),
          ("Estimated PO value (Rs)", round(s["po_value"])), ("Buyers to contact", s["buyers_to_contact"]),
          ("  of which calls", s["calls"]), ("  of which WhatsApp", s["whatsapp"]),
          ("Expected order value (Rs)", round(s["expected_order_value"])),
          ("History", f"{s['history']['from']} to {s['history']['to']} ({s['history']['lines']} lines)")]
    for i, (k, v) in enumerate(kv, start=4):
        ws.cell(row=i, column=1, value=k).font = Font(bold=True)
        c = ws.cell(row=i, column=2, value=v)
        if isinstance(v, (int, float)):
            c.number_format = "#,##0"
    r0 = 4 + len(kv) + 1
    ws.cell(row=r0, column=1, value="Highlights").font = Font(bold=True)
    for i, line in enumerate(report["highlights"], start=r0 + 1):
        ws.cell(row=i, column=1, value=f"- {line}")
    r1 = r0 + len(report["highlights"]) + 2
    ws.cell(row=r1, column=1, value="Upcoming festivals (30 days)").font = Font(bold=True)
    for i, e in enumerate(s["upcoming_events"], start=r1 + 1):
        ws.cell(row=i, column=1, value=e["date"])
        ws.cell(row=i, column=2, value=f"{e['festival']} (in {e['in_days']} d): {', '.join(e['commodities'])}")
    ws.column_dimensions["A"].width = 30
    ws.column_dimensions["B"].width = 70

    po = wb.create_sheet("PO Plan")
    table(po, report["po_plan"], [
        ("status", "Status", None), ("item_name", "Item", None), ("supplier", "Supplier", None),
        ("order_packs", "Order (packs)", "#,##0"), ("pack_unit", "Pack", None), ("order_qty", "Order qty (base unit)", "#,##0"),
        ("unit", "Unit", None), ("est_po_value", "Est. PO value (Rs)", "#,##0"), ("on_hand", "On hand", "#,##0"),
        ("on_order", "On order", "#,##0"), ("reorder_point", "Reorder point (min)", "#,##0"),
        ("order_up_to", "Order-up-to (max)", "#,##0"), ("safety_stock", "Safety stock", "#,##0"),
        ("lead_time_days", "Lead time (d)", "0"), ("days_of_cover", "Days of cover", "0.0"),
        ("stockout_in_days", "Runs out in (d)", "0"), ("next_order_date", "Next order date", None),
        ("forecast_7d", "Forecast 7d", "#,##0"), ("forecast_15d", "Forecast 15d", "#,##0"),
        ("forecast_30d", "Forecast 30d", "#,##0"), ("forecast_method", "Method", None),
    ])
    for i, r in enumerate(report["po_plan"], start=2):
        po.cell(row=i, column=1).fill = PatternFill("solid", fgColor=STATUS_FILL.get(r.get("status"), "FFFFFF"))

    cl = wb.create_sheet("Call List")
    table(cl, report["call_list"], [
        ("rank", "#", "0"), ("buyer_name", "Buyer", None), ("segment", "Segment", None), ("area", "Area", None),
        ("phone", "Phone", None), ("channel", "Channel", None), ("action", "Action", None), ("items", "Suggested items", None),
        ("expected_value", "Expected value (Rs)", "#,##0"), ("max_probability", "Max p(order)", "0.00"),
        ("check_stock", "Check stock", None), ("why", "Why", None), ("message", "Message", None),
    ])

    fc = wb.create_sheet("Forecast")
    rows = [{k: v for k, v in r.items() if k not in ("daily", "history")} for r in report["forecast"]]
    table(fc, rows, [("item_name", "Item", None), ("commodity", "Commodity", None), ("unit", "Unit", None),
                     ("h7", "Next 7 days", "#,##0"), ("h15", "Next 15 days", "#,##0"), ("h30", "Next 30 days", "#,##0"),
                     ("method", "Method", None)])
    daily = wb.create_sheet("Daily Forecast")
    long = [{"item_name": r["item_name"], "date": d["date"], "forecast": d["value"], "unit": r["unit"]}
            for r in report["forecast"] for d in r["daily"]]
    table(daily, long, [("date", "Date", None), ("item_name", "Item", None), ("forecast", "Forecast", "#,##0.0"),
                        ("unit", "Unit", None)])

    mt = wb.create_sheet("Model Metrics")
    dm = report["metrics"].get("demand", {})
    rm = report["metrics"].get("reorder", {})
    mt["A1"] = f"Demand forecast backtest ({dm.get('split_date')} to {dm.get('end_date')}, {dm.get('evaluated_on', '')})"
    mt["A1"].font = Font(bold=True)
    dcols = [("model", "Model", None)] + [(k, k.upper().replace("_", " "), "0.000")
                                          for k in (dm.get("models") or [{}])[0].keys() if k != "model"]
    table(mt, dm.get("models", []), dcols, start_row=2)
    r = 4 + len(dm.get("models", []))
    mt.cell(row=r, column=1, value=f"Buyer reorder backtest ({rm.get('test_from')} to {rm.get('test_to')}, base rate {rm.get('base_rate')})").font = Font(bold=True)
    rcols = [("model", "Model", None)] + [(k, k.upper().replace("_", " "), "0.000")
                                          for k in (rm.get("models") or [{}])[0].keys() if k != "model"]
    table(mt, rm.get("models", []), rcols, start_row=r + 1)
    mt.freeze_panes = None

    dq = wb.create_sheet("Data Quality")
    for i, (k, v) in enumerate(report.get("data_quality", {}).items(), start=1):
        dq.cell(row=i, column=1, value=k).font = Font(bold=True)
        dq.cell(row=i, column=2, value=", ".join(map(str, v)) if isinstance(v, list) else v)
    dq.column_dimensions["A"].width = 24
    dq.column_dimensions["B"].width = 60

    buf = io.BytesIO()
    wb.save(buf)
    data = buf.getvalue()
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_bytes(data)
    return data
