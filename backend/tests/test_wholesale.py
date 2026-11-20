"""Tests for the wholesale planner (run: pytest -q from backend/). Offline, synthetic data only."""
from __future__ import annotations

import io
import json
import math

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.wholesale import features as F
from app.wholesale import ingest as I
from app.wholesale import notify, synthetic
from app.wholesale.config import PlannerConfig, WhatsAppSettings
from app.wholesale.demand import DemandForecaster, croston_sba, purchase_plan
from app.wholesale.pipeline import WholesalePlanner, load_dir
from app.wholesale.reorder import call_list
from app.wholesale.report import _inr, write_excel
from app.wholesale.seasonality import calendar_features, upcoming_events

client = TestClient(app)
START = "2025-01-01"


@pytest.fixture(scope="module")
def world():
    return synthetic.generate(seed=11, n_buyers=50, start=START)


@pytest.fixture(scope="module")
def data(world):
    return I.load(world.invoices, world.items, world.parties, world.stock, world.open_pos)


@pytest.fixture(scope="module")
def panel(data):
    return F.daily_panel(data)


@pytest.fixture(scope="module")
def planner(data):
    p = WholesalePlanner(PlannerConfig.fast(seed=3))
    p.fit(data)
    return p


@pytest.fixture(scope="module")
def report(planner, data):
    return planner.plan(data)


# ------------------------------------------------------------------ synthetic data
def test_synthetic_is_deterministic_and_obviously_fake(world):
    again = synthetic.generate(seed=11, n_buyers=50, start=START)
    pd.testing.assert_frame_equal(world.invoices, again.invoices)
    assert world.parties["phone"].str.startswith("+91 00000").all()
    gst = world.parties["gstin"][world.parties["gstin"] != ""]
    assert gst.str.contains("DEMO").all()
    assert world.meta["lost_sales_share"] > 0  # stockouts exist, so demand is censored
    assert set(world.parties["segment"]) == {"kirana", "hotel", "caterer", "bakery"}
    assert world.as_of == pd.Timestamp(synthetic.DEFAULT_END) + pd.Timedelta(days=1)


# ------------------------------------------------------------------ ingest
@pytest.mark.parametrize("erp", ["tally", "vyapar", "marg", "canonical"])
def test_ingest_all_erp_formats_agree_with_truth(erp):
    w = synthetic.generate(seed=5, n_buyers=25, start="2026-03-01", erp=erp)
    d = I.load(w.invoices, w.items, w.parties, w.stock, w.open_pos)
    assert d.quality["erp"] == erp
    assert d.quality["cancelled_dropped"] > 0 and d.quality["duplicates_dropped"] > 0
    assert d.lines["is_return"].sum() > 0
    billed = d.lines[~d.lines["is_return"]].groupby("sku")["qty"].sum()
    truth = w.truth.groupby("sku")["sales"].sum()
    assert np.allclose(billed.reindex(truth.index).fillna(0), truth, atol=1e-6)
    # "KIRANA 017  " and "Kirana 017" are one buyer
    assert d.lines["buyer_id"].nunique() == d.lines["buyer_name"].nunique()


def test_quantity_and_date_parsing():
    assert I.split_qty("4 Bag") == (4.0, "bag")
    assert I.split_qty("1,250.50 kg") == (1250.5, "kg")
    assert I.split_qty(3) == (3.0, "")
    d = I.parse_dates(pd.Series(["03-Jan-2024", "20240105", "2024-01-06", "07/01/2024", "garbage"]))
    assert list(d.dt.strftime("%Y-%m-%d")[:4]) == ["2024-01-03", "2024-01-05", "2024-01-06", "2024-01-07"]
    assert pd.isna(d.iloc[4])
    assert I.guess_pack("Sunflower Oil 15L Tin") == (15.0, "L")
    assert I.guess_commodity("Toor Dal 30kg") == "pulses"


def test_ingest_without_masters_infers_items():
    raw = pd.DataFrame({
        "Date": ["01-04-2025", "02-04-2025", "02-04-2025"], "Party Name": ["A Store", "a  store", "B Hotel"],
        "Item Name": ["Matta Rice 25kg", "Matta Rice 25kg", "Sugar 50kg"], "Quantity": [2, 1, 3],
        "Unit": ["Bag", "Bag", "Bag"], "Price/Unit": [1300, 1300, 2100], "Amount": [2600, 1300, 6300],
        "Transaction Type": ["Sale", "Sale", "Sale"],
    })
    d = I.load(raw)
    assert d.quality["erp"] == "vyapar"
    assert d.lines["buyer_id"].nunique() == 2
    rice = d.lines[d.lines["commodity"] == "rice"]
    assert rice["qty"].sum() == 75.0 and set(rice["base_unit"]) == {"kg"}
    assert math.isclose(rice["rate"].iat[0], 52.0)


def test_ingest_errors():
    with pytest.raises(I.IngestError):
        I.load(pd.DataFrame({"foo": [1], "bar": [2]}))


def test_tally_xml_parser():
    xml = """<ENVELOPE><BODY><IMPORTDATA><REQUESTDATA>
      <TALLYMESSAGE><VOUCHER VCHTYPE="Sales">
        <DATE>20250401</DATE><VOUCHERTYPENAME>Sales</VOUCHERTYPENAME><VOUCHERNUMBER>1</VOUCHERNUMBER>
        <LEDGERENTRIES.LIST><LEDGERNAME>Kirana 001 - Aluva</LEDGERNAME><ISPARTYLEDGER>Yes</ISPARTYLEDGER></LEDGERENTRIES.LIST>
        <ALLINVENTORYENTRIES.LIST><STOCKITEMNAME>Matta Rice 25kg</STOCKITEMNAME>
          <RATE>1300.00/Bag</RATE><AMOUNT>2600.00</AMOUNT><ACTUALQTY> 2 Bag</ACTUALQTY><BILLEDQTY> 2 Bag</BILLEDQTY>
        </ALLINVENTORYENTRIES.LIST>
      </VOUCHER></TALLYMESSAGE>
      <TALLYMESSAGE><VOUCHER><DATE>20250402</DATE><VOUCHERTYPENAME>Sales</VOUCHERTYPENAME>
        <ISCANCELLED>Yes</ISCANCELLED><PARTYLEDGERNAME>Hotel 002</PARTYLEDGERNAME>
        <ALLINVENTORYENTRIES.LIST><STOCKITEMNAME>Sugar 50kg</STOCKITEMNAME><BILLEDQTY>1 Bag</BILLEDQTY>
          <AMOUNT>2100</AMOUNT></ALLINVENTORYENTRIES.LIST>
      </VOUCHER></TALLYMESSAGE>
    </REQUESTDATA></IMPORTDATA></BODY></ENVELOPE>"""
    raw = I.parse_tally_xml(xml)
    assert len(raw) == 2 and raw["Particulars"].iat[0] == "Kirana 001 - Aluva"
    d = I.load(raw)
    assert d.quality["erp"] == "tally" and d.quality["cancelled_dropped"] == 1
    assert len(d.lines) == 1 and d.lines["qty"].iat[0] == 50.0


# ------------------------------------------------------------------ calendar + features
def test_calendar_features():
    cal = calendar_features(pd.date_range("2025-08-20", "2025-11-30"), "kerala")
    assert cal.loc["2025-09-05", "festival_weight"] == 1.0          # Onam
    assert cal.loc["2025-09-01", "festival_pull"] > 0.5              # stock-up before Onam
    assert cal.loc["2025-10-20", "days_to_major_festival"] == 0      # Diwali
    assert cal.loc["2025-08-20", "sw_monsoon"] == 1 and cal.loc["2025-11-20", "kharif_harvest"] == 1
    ramzan = calendar_features(pd.date_range("2026-03-01", "2026-03-25"), "kerala")["ramzan"]
    assert ramzan.loc["2026-03-10"] == 1 and ramzan.loc["2026-03-22"] == 0
    ev = upcoming_events(pd.Timestamp("2026-11-01"), 30)
    assert any(e["festival"] == "Diwali" and e["in_days"] == 7 for e in ev)


def test_unconstraining_moves_sales_towards_true_demand(world, panel):
    m = panel.merge(world.truth[["date", "sku", "demand"]].rename(columns={"demand": "true"}), on=["date", "sku"])
    so = m[m["stockout"]]
    assert len(so) > 0
    assert abs(so["demand"].sum() - so["true"].sum()) < abs(so["sales"].sum() - so["true"].sum())
    assert (m["demand"] >= m["sales"] - 1e-9).all()


def test_suspect_stockouts_without_stock_file():
    idx = pd.date_range("2025-01-01", periods=60)
    sales = pd.Series(100.0, index=idx)
    orders = pd.Series(3.0, index=idx)
    sales.iloc[40:44] = 0
    orders.iloc[40:44] = 0
    flags = F.suspect_stockouts(sales, orders)
    assert flags.iloc[40:44].all() and not flags.iloc[:40].any()
    assert F.unconstrain(sales, flags).iloc[41] > 50


def test_buyer_snapshot_labels_have_no_leakage(data, panel):
    orders = F._orders(data.lines)
    pri = F.interval_priors(orders, data.parties, data.items)
    cut = pd.Timestamp("2026-06-30")
    snap = F.buyer_snapshots(data, [cut], panel, pri, label_windows=(7,))
    pair = snap.iloc[0]
    o = orders[(orders.buyer_id == pair.buyer_id) & (orders.sku == pair.sku)]
    assert pair["recency"] == (cut - o[o.date <= cut].date.max()).days
    expected = int(((o.date > cut) & (o.date <= cut + pd.Timedelta(days=7))).any())
    assert pair["y_7"] == expected
    assert set(F.BUYER_FEATURES) <= set(snap.columns)


# ------------------------------------------------------------------ models
def test_croston_sba():
    assert croston_sba(np.zeros(10)) == 0.0
    y = np.array([0, 0, 10, 0, 0, 10, 0, 0, 10] * 5, dtype=float)
    assert 2.5 < croston_sba(y) < 3.5


def test_demand_backtest_and_forecast(planner, panel):
    m = planner.metrics["demand"]
    names = [r["model"] for r in m["models"]]
    assert names[0] in ("lightgbm", "gbm (sklearn)") and "holt-winters" in names
    ours, naive = m["models"][0], next(r for r in m["models"] if r["model"].startswith("naive"))
    assert ours["wape_30"] < naive["wape_30"]
    assert ours["mase_daily"] < 1.0
    fc = planner.demand.predict(panel)
    assert (fc.groupby("sku")["k"].max() == planner.cfg.max_horizon).all()
    assert (fc["forecast"] >= 0).all()
    # the item launched 10 days before the end is handled as cold start
    assert fc.loc[fc["sku"] == "OIL-GROUNDNUT", "method"].eq("cold_start").all()
    assert planner.demand.sigma and all(s[0] <= s[-1] * 1.5 for s in planner.demand.sigma.values())


def test_purchase_plan_formulas(data, panel):
    cfg = PlannerConfig(service_level=0.95, review_days=7)
    fc = DemandForecaster(cfg)
    fc.sigma = {"RICE-MATTA": [10.0 * math.sqrt(p) for p in range(1, 31)]}
    dates = pd.date_range(data.as_of, periods=30)
    forecast = pd.DataFrame({"sku": "RICE-MATTA", "date": dates, "k": range(1, 31), "forecast": 100.0, "method": "model"})
    d = I.WholesaleData(lines=data.lines, items=data.items, parties=data.parties,
                        stock=pd.DataFrame({"date": [data.as_of - pd.Timedelta(days=1)], "sku": ["RICE-MATTA"], "closing_qty": [300.0]}),
                        open_pos=pd.DataFrame(columns=["sku", "qty", "expected_date"]), as_of=data.as_of)
    plan = purchase_plan(forecast, d, panel, fc, cfg).iloc[0]
    L, sd = 4, 1.0  # Matta rice in the synthetic item master
    z = 1.6448536269514722
    ss = z * math.sqrt((10 * math.sqrt(L)) ** 2 + (100 * sd) ** 2)
    assert math.isclose(plan["safety_stock"], round(ss, 1), abs_tol=0.11)
    assert math.isclose(plan["reorder_point"], round(400 + ss, 1), abs_tol=0.11)
    up_to = 100 * (L + 7) + z * math.sqrt((10 * math.sqrt(L + 7)) ** 2 + (100 * sd) ** 2)
    packs = max(math.ceil((up_to - 300) / 25), 40)  # MOQ is 40 bags
    assert plan["order_packs"] == packs and plan["order_qty"] == packs * 25
    assert plan["status"] == "URGENT" and plan["stockout_in_days"] == 4  # 300 kg covers 3 days


def test_reorder_model_and_call_list(planner, data, panel):
    m = planner.metrics["reorder"]
    gbm, rule = m["models"][0], m["models"][1]
    assert gbm["pr_auc"] > m["base_rate"] and gbm["pr_auc"] > rule["pr_auc"]
    assert 0 <= gbm[f"precision_at_{m['k']}"] <= 1
    scores = planner.reorder.score(data, panel)
    assert scores["p_reorder"].between(0, 1).all()
    plan = pd.DataFrame(columns=["sku", "status", "on_hand"])
    calls = call_list(scores, data, plan, planner.cfg)
    assert 0 < len(calls) <= planner.cfg.call_list_size
    assert list(calls["rank"]) == list(range(1, len(calls) + 1))
    assert calls["priority_score"].is_monotonic_decreasing
    assert set(calls["channel"]) <= {"Call", "WhatsApp"}
    assert set(calls["action"]) <= {"Reorder reminder", "Win-back call"}
    wa = calls[calls["channel"] == "WhatsApp"]
    opted = set(data.parties.loc[data.parties["whatsapp_opt_in"], "buyer_id"])
    assert set(wa["buyer_id"]) <= opted


# ------------------------------------------------------------------ report, notify, persistence
def test_report_and_excel(report, tmp_path):
    json.dumps(report)  # JSON-safe
    s = report["summary"]
    assert {"items_to_order", "po_value", "buyers_to_contact", "upcoming_events"} <= set(s)
    assert report["po_plan"] and report["call_list"] and report["forecast"] and report["highlights"]
    assert {r["status"] for r in report["po_plan"]} <= {"URGENT", "EXPEDITE", "ORDER", "OK"}  # stock file given
    from openpyxl import load_workbook
    path = tmp_path / "r.xlsx"
    write_excel(report, path)
    wb = load_workbook(path)
    assert wb.sheetnames == ["Summary", "PO Plan", "Call List", "Forecast", "Daily Forecast", "Model Metrics", "Data Quality"]
    assert wb["PO Plan"]["A1"].value == "Status" and wb["PO Plan"].max_row == len(report["po_plan"]) + 1


def test_inr_grouping():
    assert _inr(1234567) == "Rs 12,34,567" and _inr(999) == "Rs 999" and _inr(-100000) == "-Rs 1,00,000"


def test_whatsapp_stub_never_sends(report, tmp_path, monkeypatch):
    out = notify.deliver(report, tmp_path, WhatsAppSettings(mode="dry_run"))
    assert out["mode"] == "dry_run"
    saved = json.loads((tmp_path / f"outbox_{report['summary']['as_of']}.json").read_text())
    assert len(saved) == out["messages"] == sum(r["channel"] == "WhatsApp" for r in report["call_list"])
    for p in saved:
        assert p["request"]["type"] == "template" and p["request"]["to"].startswith("9100000")
    assert notify.deliver(report, tmp_path, WhatsAppSettings(mode="off"))["messages"] == 0
    with pytest.raises(notify.WhatsAppConfigError):
        notify.deliver(report, tmp_path, WhatsAppSettings(mode="live"))
    with pytest.raises(notify.WhatsAppConfigError):
        notify.deliver(report, tmp_path, WhatsAppSettings(mode="carrier-pigeon"))
    monkeypatch.delenv("DATARAT_WHATSAPP_MODE", raising=False)
    assert WhatsAppSettings.from_env().mode == "dry_run"


def test_save_load_and_files_roundtrip(planner, world, tmp_path):
    world.write(tmp_path / "data")
    path = planner.save(tmp_path / "model.joblib")
    again = WholesalePlanner.load(path)
    rep = again.plan(load_dir(tmp_path / "data"))
    assert rep["summary"]["as_of"] == world.as_of.date().isoformat()
    assert 0 < len(rep["po_plan"]) <= len(world.items)


# ------------------------------------------------------------------ API
def test_api_report_404_before_run(monkeypatch):
    from app.wholesale import api
    monkeypatch.setitem(api._state, "report", None)
    assert client.get("/api/wholesale/report").status_code == 404


def test_api_demo_report_xlsx_and_sql():
    r = client.post("/api/wholesale/demo", json={"seed": 3, "n_buyers": 30})
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["po_plan"] and j["call_list"] and j["metrics"]["demand"]["models"]
    assert set(j["datasets"]) >= {"wholesale_invoices", "wholesale_po_plan", "wholesale_call_list", "wholesale_forecast"}
    assert client.get("/api/wholesale/report").json()["summary"]["as_of"] == j["summary"]["as_of"]
    x = client.get("/api/wholesale/report.xlsx")
    assert x.status_code == 200 and x.headers["content-type"].startswith("application/vnd.openxmlformats")
    assert x.content[:2] == b"PK"
    q = client.post("/api/sql", json={"sql": "SELECT commodity, SUM(qty) AS kg FROM wholesale_invoices GROUP BY 1 ORDER BY 2 DESC"})
    assert q.status_code == 200 and len(q.json()["rows"]) >= 4
    ob = client.get("/api/wholesale/outbox").json()
    assert ob["mode"] == "dry_run" and ob["messages"] == len(ob["payloads"])


def test_short_indian_export_keeps_day_first_dates_through_upload(monkeypatch):
    """Every day <= 12, so the date order is ambiguous; it must stay day-first end to end."""
    from app.store import store
    monkeypatch.delenv("DATARAT_DATE_ORDER", raising=False)
    csv = (b"Date,Party Name,Item Name,Quantity,Unit,Price/Unit,Amount,Transaction Type\n"
           b"01/04/2025,A Store,Matta Rice 25kg,2,Bag,1300,2600,Sale\n"
           b"02/04/2025,B Hotel,Sugar 50kg,1,Bag,2100,2100,Sale\n")
    up = client.post("/api/datasets/connect", data={"type": "csv", "name": "short_vyapar"},
                     files={"file": ("short.csv", csv, "text/csv")})
    assert up.status_code == 200, up.text
    d = I.load(store.df(up.json()["id"]))
    assert list(d.lines["date"].dt.strftime("%Y-%m-%d")) == ["2025-04-01", "2025-04-02"]


def test_api_run_on_uploaded_export():
    w = synthetic.generate(seed=4, n_buyers=30, start="2025-03-01", erp="vyapar")
    files = {"file": ("vyapar_sales.csv", w.invoices.to_csv(index=False).encode(), "text/csv")}
    up = client.post("/api/datasets/connect", data={"type": "csv", "name": "vyapar_sales"}, files=files)
    assert up.status_code == 200, up.text
    ds = up.json()["id"]
    r = client.post("/api/wholesale/run", json={"invoices_dataset_id": ds})
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["data_quality"]["erp"] == "vyapar" and j["po_plan"]
    # no stock export was connected: nothing is ordered blindly
    assert {r["status"] for r in j["po_plan"]} == {"CHECK STOCK"} and all(r["order_qty"] == 0 for r in j["po_plan"])
    bad = client.post("/api/wholesale/run", json={"invoices_dataset_id": "sales"})
    assert bad.status_code == 400
    assert client.post("/api/wholesale/run", json={"invoices_dataset_id": "nope"}).status_code == 404
