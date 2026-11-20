"""Deterministic synthetic data for a B2B staples wholesaler (no real customer or invoice data).

What it simulates
-----------------
* ~120 buyers in four segments (kirana stores, hotels, caterers, bakeries) in Kochi-style areas.
  Each buyer keeps its own stock of each item it uses, consumes it daily (with festival, Ramzan,
  monsoon and wedding-season effects) and visits the wholesaler on a renewal process. On a visit
  it buys the items that are running low, so "running low" is literally the reorder trigger.
* Caterers are event-driven: lumpy, intermittent orders a few days before weddings/functions.
* Bulk festive hoarding (bigger orders 1-3 weeks before Onam/Diwali/Eid) and forward buying when
  prices are rising, followed by longer gaps.
* Price paths with harvest cycles and shocks (pulse inflation, edible-oil duty step, sugar rise).
* Churning buyers (visits thin out, then stop) and new buyers / new items (cold start).
* The wholesaler's own warehouse with a naive "gut feel" replenishment policy and occasional
  supplier delays, so some days are stocked out: invoices then show *sales*, not *demand*
  (censored demand). The latent demand is returned separately as ground truth.
* ERP-export noise: dd-Mon-yyyy dates, "4 Bag" / "100 kg" quantity strings, cancelled vouchers,
  credit notes, duplicated rows and inconsistent party-name spacing/case.

Names are obviously synthetic ("Kirana 017 - Edappally"), phone numbers use the invalid
"+91 00000 xxxxx" range and GSTINs contain "DEMO", so nothing can be mistaken for a real party.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .seasonality import calendar_features, festival_multiplier

DEFAULT_START = "2024-01-01"
DEFAULT_END = "2026-10-31"

# sku, name, commodity, base unit, pack unit, pack size, price Rs/base unit, lead time (mean, sd),
# MOQ packs, supplier, HSN heading, launch offset (days before end; None = from the start)
ITEM_SPECS = [
    ("RICE-MATTA", "Matta Rice 25kg", "rice", "kg", "Bag", 25, 52.0, 4, 1.0, 40, "Supplier R1 (rice mill)", "1006", None),
    ("RICE-SONA", "Sona Masoori Rice 25kg", "rice", "kg", "Bag", 25, 60.0, 6, 2.0, 40, "Supplier R2 (rice trader)", "1006", None),
    ("RICE-BASMATI", "Basmati Rice 25kg", "rice", "kg", "Bag", 25, 98.0, 8, 2.0, 20, "Supplier R2 (rice trader)", "1006", None),
    ("RICE-JAYA", "Jaya Rice 25kg", "rice", "kg", "Bag", 25, 48.0, 5, 1.0, 20, "Supplier R1 (rice mill)", "1006", 50),
    ("SUGAR-S30", "Sugar S-30 50kg", "sugar", "kg", "Bag", 50, 42.0, 3, 1.0, 20, "Supplier S1 (sugar distributor)", "1701", None),
    ("DAL-TOOR", "Toor Dal 30kg", "pulses", "kg", "Bag", 30, 145.0, 7, 2.0, 10, "Supplier P1 (pulses importer)", "0713", None),
    ("DAL-CHANA", "Chana Dal 30kg", "pulses", "kg", "Bag", 30, 85.0, 7, 2.0, 10, "Supplier P1 (pulses importer)", "0713", None),
    ("DAL-URAD", "Urad Dal 30kg", "pulses", "kg", "Bag", 30, 118.0, 7, 3.0, 10, "Supplier P1 (pulses importer)", "0713", None),
    ("DAL-MOONG", "Moong Dal 30kg", "pulses", "kg", "Bag", 30, 112.0, 7, 2.0, 10, "Supplier P1 (pulses importer)", "0713", None),
    ("OIL-SUNFLOWER", "Sunflower Oil 15L Tin", "oil", "L", "Tin", 15, 128.0, 5, 1.0, 30, "Supplier O1 (oil refinery)", "1512", None),
    ("OIL-PALM", "Palmolein Oil 15L Tin", "oil", "L", "Tin", 15, 102.0, 5, 1.0, 30, "Supplier O1 (oil refinery)", "1511", None),
    ("OIL-COCONUT", "Coconut Oil 15L Tin", "oil", "L", "Tin", 15, 210.0, 4, 1.0, 20, "Supplier O2 (coconut oil mill)", "1513", None),
    ("OIL-GROUNDNUT", "Groundnut Oil 15L Tin", "oil", "L", "Tin", 15, 175.0, 5, 1.0, 10, "Supplier O1 (oil refinery)", "1508", 10),
]

AREAS = ["Edappally", "Kakkanad", "Vyttila", "Aluva", "Fort Kochi", "Kaloor", "Palarivattom",
         "Thrippunithura", "Kalamassery", "Perumbavoor", "Angamaly", "Mattancherry"]

# daily usage (base units/day for an average buyer of the segment) per item
SEGMENTS = {
    "kirana": dict(share=0.55, visit_mean=6.0, visit_shape=4.0, discount=0.0, gst_share=0.6, use={
        "RICE-MATTA": (0.85, 9.0), "RICE-SONA": (0.5, 5.0), "RICE-BASMATI": (0.25, 1.2), "RICE-JAYA": (0.5, 4.0),
        "SUGAR-S30": (0.9, 6.0), "DAL-TOOR": (0.7, 1.6), "DAL-CHANA": (0.5, 1.0), "DAL-URAD": (0.5, 1.1),
        "DAL-MOONG": (0.4, 0.8), "OIL-SUNFLOWER": (0.6, 2.2), "OIL-PALM": (0.5, 2.0),
        "OIL-COCONUT": (0.7, 1.5), "OIL-GROUNDNUT": (0.3, 1.0)}),
    "hotel": dict(share=0.22, visit_mean=3.5, visit_shape=6.0, discount=0.01, gst_share=0.9, use={
        "RICE-MATTA": (0.8, 22.0), "RICE-SONA": (0.6, 12.0), "RICE-BASMATI": (0.5, 6.0), "RICE-JAYA": (0.3, 8.0),
        "SUGAR-S30": (0.9, 7.0), "DAL-TOOR": (0.6, 3.0), "DAL-CHANA": (0.4, 1.5), "DAL-URAD": (0.8, 4.0),
        "DAL-MOONG": (0.3, 1.0), "OIL-SUNFLOWER": (0.8, 6.0), "OIL-PALM": (0.6, 5.0),
        "OIL-COCONUT": (0.7, 3.0), "OIL-GROUNDNUT": (0.2, 2.0)}),
    "caterer": dict(share=0.10, visit_mean=0.0, visit_shape=0.0, discount=0.02, gst_share=0.8, use={
        "RICE-MATTA": (0.9, 150.0), "RICE-BASMATI": (0.6, 60.0), "SUGAR-S30": (0.9, 40.0),
        "DAL-TOOR": (0.7, 20.0), "DAL-MOONG": (0.5, 10.0), "OIL-SUNFLOWER": (0.8, 30.0),
        "OIL-COCONUT": (0.7, 25.0)}),
    "bakery": dict(share=0.13, visit_mean=5.0, visit_shape=4.0, discount=0.005, gst_share=0.8, use={
        "SUGAR-S30": (1.0, 18.0), "OIL-PALM": (0.8, 6.0), "OIL-SUNFLOWER": (0.5, 3.0),
        "OIL-COCONUT": (0.3, 1.5), "DAL-CHANA": (0.3, 1.0), "DAL-MOONG": (0.2, 0.8)}),
}


@dataclass
class SyntheticWholesale:
    invoices: pd.DataFrame        # raw ERP export (format depends on `erp`)
    items: pd.DataFrame           # stock item master
    parties: pd.DataFrame         # party (ledger) master
    stock: pd.DataFrame           # daily closing stock per item (stock summary export)
    open_pos: pd.DataFrame        # purchase orders placed but not yet received at the end date
    truth: pd.DataFrame           # ground truth: latent demand vs sales per item-day (never in an ERP)
    as_of: pd.Timestamp           # first day after the history
    erp: str = "tally"
    meta: dict = field(default_factory=dict)

    def write(self, out_dir: str | Path, fmt: str = "csv") -> dict[str, str]:
        """Write the ERP-style files the CLI / API ingest (truth goes to a separate file)."""
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        paths = {}
        frames = {"invoices": self.invoices, "items": self.items, "parties": self.parties,
                  "stock": self.stock, "open_pos": self.open_pos, "ground_truth": self.truth}
        for name, df in frames.items():
            if fmt == "xlsx" and name == "invoices":
                p = out / f"{name}.xlsx"
                df.to_excel(p, index=False)
            else:
                p = out / f"{name}.csv"
                df.to_csv(p, index=False)
            paths[name] = str(p)
        return paths


def _price_paths(days: pd.DatetimeIndex, rng: np.random.Generator) -> dict[str, np.ndarray]:
    n = len(days)
    month = days.month.values
    out = {}
    for spec in ITEM_SPECS:
        sku, commodity, base = spec[0], spec[2], spec[6]
        walk = np.cumsum(rng.normal(0, 0.004, n))
        walk -= np.linspace(0, walk[-1], n) * 0.7  # keep the drift bounded
        level = np.exp(walk)
        if commodity == "rice":  # new kharif crop arrives Oct-Dec -> softer prices
            level *= 1 - 0.04 * np.isin(month, [10, 11, 12]) + 0.03 * np.isin(month, [6, 7, 8])
        if commodity == "pulses":
            level *= 1 - 0.03 * np.isin(month, [3, 4])
        shock = np.ones(n)
        if sku == "DAL-TOOR":  # 2024 pulse inflation: +30 % ramp then easing
            a, b, c = (days.get_indexer([pd.Timestamp(x)])[0] for x in ("2024-03-01", "2024-07-15", "2024-12-31"))
            if a >= 0 and c >= 0:
                shock[a:b] = np.linspace(1, 1.3, b - a)
                shock[b:c] = np.linspace(1.3, 1.08, c - b)
                shock[c:] = 1.08
        if commodity == "oil" and sku != "OIL-COCONUT":  # import duty step
            i = days.get_indexer([pd.Timestamp("2024-09-14")])[0]
            if i >= 0:
                shock[i:] *= 1.18
        if sku == "SUGAR-S30":
            a, b = (days.get_indexer([pd.Timestamp(x)])[0] for x in ("2025-08-01", "2025-10-15"))
            if a >= 0 and b >= 0:
                shock[a:b] = np.linspace(1, 1.12, b - a)
                shock[b:] = 1.12
        if sku == "RICE-MATTA":
            i = days.get_indexer([pd.Timestamp("2026-05-01")])[0]
            if i >= 0:
                shock[i:] *= np.linspace(1, 1.1, n - i)
        out[sku] = base * level * shock
    return out


def generate(seed: int = 7, start: str = DEFAULT_START, end: str = DEFAULT_END, n_buyers: int = 120,
             erp: str = "tally", stockout_pressure: float = 1.0) -> SyntheticWholesale:
    """Build the whole synthetic world. Deterministic for a given seed and arguments."""
    rng = np.random.default_rng(seed)
    days = pd.date_range(start, end, freq="D")
    n_days = len(days)
    skus = [s[0] for s in ITEM_SPECS]
    spec = {s[0]: s for s in ITEM_SPECS}
    sku_idx = {s: i for i, s in enumerate(skus)}
    launch = {s[0]: (0 if s[12] is None else max(0, n_days - s[12])) for s in ITEM_SPECS}
    cal = calendar_features(days, "kerala")
    prices = _price_paths(days, rng)
    # 14-day price momentum drives forward buying / waiting
    momentum = {s: pd.Series(p).pct_change(14).fillna(0).values for s, p in prices.items()}
    fest = {c: festival_multiplier(days, c, "kerala") for c in ("rice", "sugar", "pulses", "oil")}
    pull = cal["festival_pull"].values
    monsoon = cal["sw_monsoon"].values
    ramzan = cal["ramzan"].values
    wedding = cal["wedding_index"].values
    dow = days.dayofweek.values
    month = days.month.values

    # ------------------------------------------------------------- buyers
    seg_names = list(SEGMENTS)
    shares = np.array([SEGMENTS[s]["share"] for s in seg_names])
    counts = np.floor(shares * n_buyers).astype(int)
    counts[0] += n_buyers - counts.sum()
    buyers = []
    k = 0
    for seg, cnt in zip(seg_names, counts):
        for j in range(cnt):
            k += 1
            late = rng.random()
            if late < 0.10:
                start_day = int(rng.integers(n_days // 3, n_days - 90))
            elif late < 0.16:
                start_day = int(rng.integers(n_days - 60, n_days - 7))  # brand-new buyers (cold start)
            else:
                start_day = int(rng.integers(0, 30))
            churn_day = int(rng.integers(n_days // 2, n_days - 45)) if rng.random() < 0.10 else n_days
            area = AREAS[int(rng.integers(len(AREAS)))]
            label = {"kirana": "Kirana", "hotel": "Hotel", "caterer": "Caterer", "bakery": "Bakery"}[seg]
            buyers.append(dict(
                buyer_no=k, segment=seg, name=f"{label} {k:03d} - {area}", area=area,
                start_day=start_day, churn_day=churn_day, size=float(rng.lognormal(0, 0.45)),
                gstin=(f"32DEMO{k:05d}X1Z5" if rng.random() < SEGMENTS[seg]["gst_share"] else ""),
                phone=f"+91 00000 {k:05d}", whatsapp_opt_in=bool(rng.random() < 0.85),
            ))

    # ------------------------------------------------------------- buyer-side requests
    req_day, req_buyer, req_sku, req_qty = [], [], [], []
    for b in buyers:
        seg = SEGMENTS[b["segment"]]
        items = [s for s, (p, _) in seg["use"].items() if rng.random() < p]
        if not items:
            items = [max(seg["use"], key=lambda s: seg["use"][s][0])]
        if b["segment"] == "caterer":
            # functions arrive as a Poisson process driven by the wedding index and festivals
            lam = 0.07 * b["size"] * (0.3 + wedding + 0.5 * pull)
            lam[b["churn_day"]:] = 0
            lam[: b["start_day"]] = 0
            events = np.where(rng.random(n_days) < lam)[0]
            for e in events:
                order_day = e - int(rng.integers(1, 4))
                if order_day < max(b["start_day"], 0):
                    continue
                guests = rng.lognormal(0, 0.5)
                for s in items:
                    if order_day < launch[s] or rng.random() < 0.2:
                        continue
                    pack = spec[s][5]
                    qty = seg["use"][s][1] * guests * rng.uniform(0.7, 1.3)
                    req_day.append(order_day); req_buyer.append(b["buyer_no"]); req_sku.append(s)
                    req_qty.append(max(1, int(np.ceil(qty / pack))) * pack)
            continue
        # regular buyers: daily consumption per item, visits on a renewal process
        rates = {}
        for s in items:
            commodity = spec[s][2]
            base = seg["use"][s][1] * b["size"]
            season = fest[commodity].copy()
            if b["segment"] == "hotel":
                season *= 1 + 0.25 * ramzan * (commodity in ("rice", "sugar", "oil")) - 0.08 * monsoon
            if b["segment"] == "bakery":
                season *= 1 + 0.35 * (month == 12) * (commodity in ("sugar", "oil"))
            season *= np.where(dow == 6, 0.85, 1.0)
            noise = rng.lognormal(0, 0.12, n_days)
            r = base * season * noise
            r[: launch[s]] = 0
            rates[s] = r
        cum = {s: np.concatenate([[0.0], np.cumsum(r)]) for s, r in rates.items()}
        inv = {s: rates[s][max(b["start_day"], launch[s]) : max(b["start_day"], launch[s]) + 7].sum() for s in items}
        cover = {s: rng.uniform(6, 12) for s in items}
        day = b["start_day"]
        last_day = {s: day for s in items}
        while day < min(b["churn_day"], n_days):
            mean_gap = seg["visit_mean"]
            if b["churn_day"] < n_days and day > b["churn_day"] - 75:
                mean_gap *= 1.8  # visits thin out before a buyer churns
            gap = max(1, int(round(rng.gamma(seg["visit_shape"], mean_gap / seg["visit_shape"]))))
            day += gap
            if day >= min(b["churn_day"], n_days):
                break
            if dow[day] == 6 and rng.random() < 0.7:  # most shops avoid Sunday trips
                day += 1
                if day >= n_days:
                    break
            for s in items:
                if day < launch[s]:
                    continue
                used = cum[s][day] - cum[s][last_day[s]]
                inv[s] = max(0.0, inv[s] - used)
                last_day[s] = day
                rate_now = rates[s][max(0, day - 7) : day + 1].mean() + 1e-9
                stock_up = 1.0 + 0.9 * pull[day] * (spec[s][2] in ("sugar", "oil", "rice"))
                mom = momentum[s][day]
                stock_up *= 1.45 if mom > 0.04 else (0.85 if mom < -0.04 else 1.0)
                if inv[s] < 0.6 * cover[s] * rate_now * stock_up:
                    pack = spec[s][5]
                    want = cover[s] * rate_now * stock_up * rng.uniform(1.4, 1.9) - inv[s]
                    packs = max(1, int(np.ceil(want / pack)))
                    req_day.append(day); req_buyer.append(b["buyer_no"]); req_sku.append(s)
                    req_qty.append(packs * pack)
                    inv[s] += packs * pack
    req = pd.DataFrame({"day": req_day, "buyer_no": req_buyer, "sku": req_sku, "qty": req_qty})
    req = req.sample(frac=1.0, random_state=seed).sort_values("day", kind="stable").reset_index(drop=True)

    # ------------------------------------------------------------- wholesaler warehouse (censoring)
    n_sku = len(skus)
    demand = np.zeros((n_days, n_sku))
    np.add.at(demand, (req["day"].values, req["sku"].map(sku_idx).values), req["qty"].values)
    filled = np.zeros(len(req))
    on_hand = np.array([demand[launch[s] : launch[s] + 28, i].sum() * 0.8 for i, s in enumerate(skus)])
    pipeline: list[tuple[int, int, float, int]] = []  # (arrival day, sku idx, qty, order day)
    closing = np.zeros((n_days, n_sku))
    sales = np.zeros((n_days, n_sku))
    # supplier disruptions: a few windows per item where lead time jumps by 8-14 days
    disrupt = {i: set() for i in range(n_sku)}
    for i in range(n_sku):
        for _ in range(int(rng.integers(1, 4) * stockout_pressure)):
            d0 = int(rng.integers(60, n_days - 30))
            disrupt[i].update(range(d0, d0 + 21))
    rows_by_day = np.split(np.arange(len(req)), np.searchsorted(req["day"].values, np.arange(1, n_days)))
    req_sku_idx = req["sku"].map(sku_idx).values
    req_qty_arr = req["qty"].values.astype(float)
    for d in range(n_days):
        arrived = [p for p in pipeline if p[0] == d]
        for p in arrived:
            on_hand[p[1]] += p[2]
        pipeline = [p for p in pipeline if p[0] > d]
        for r in rows_by_day[d]:
            i = req_sku_idx[r]
            pack = spec[skus[i]][5]
            give = min(req_qty_arr[r], np.floor(on_hand[i] / pack) * pack)
            filled[r] = give
            on_hand[i] -= give
            sales[d, i] += give
        closing[d] = on_hand
        if dow[d] == 0:  # Monday review with a rule of thumb based on *observed sales*
            for i, s in enumerate(skus):
                if d < launch[s]:
                    continue
                recent = sales[max(0, d - 28) : d + 1, i].mean()
                on_order = sum(p[2] for p in pipeline if p[1] == i)
                pos = on_hand[i] + on_order
                lo, hi = 9 / stockout_pressure, 22
                if pos < lo * recent or (recent == 0 and pos == 0):
                    pack = spec[s][5]
                    qty = max(spec[s][9] * pack, np.ceil((hi * max(recent, 1) - pos) / pack) * pack)
                    lt = max(1, int(round(rng.normal(spec[s][7], spec[s][8]))))
                    if d in disrupt[i]:
                        lt += int(rng.integers(8, 15))
                    pipeline.append((d + lt, i, float(qty), d))
    req["filled"] = filled

    truth = pd.DataFrame({
        "date": np.repeat(days.values, n_sku), "sku": np.tile(skus, n_days),
        "demand": demand.ravel(), "sales": sales.ravel(), "closing_stock": closing.ravel(),
    })
    pack_arr = np.tile([spec[s][5] for s in skus], n_days)
    truth["stockout"] = (truth["closing_stock"] < pack_arr) & (truth["demand"] > truth["sales"])
    launch_date = {s: days[launch[s]] for s in skus}
    truth = truth[truth["date"].values >= truth["sku"].map(launch_date).values].reset_index(drop=True)

    # ------------------------------------------------------------- invoices (only what was billed)
    lines = req[req["filled"] > 0].copy()
    lines["date"] = days.values[lines["day"].values]
    bmap = {b["buyer_no"]: b for b in buyers}
    lines["segment"] = lines["buyer_no"].map(lambda x: bmap[x]["segment"])
    disc = lines["segment"].map(lambda s: SEGMENTS[s]["discount"]).values
    base_price = np.array([prices[s][d] for s, d in zip(lines["sku"].values, lines["day"].values)])
    lines["rate"] = np.round(base_price * (1 - disc) * (1 + rng.normal(0, 0.005, len(lines))), 2)
    lines = lines.sort_values(["day", "buyer_no", "sku"]).reset_index(drop=True)
    inv_key = lines["day"].astype(str) + "-" + lines["buyer_no"].astype(str)
    inv_no = pd.factorize(inv_key)[0] + 1
    fy = np.where(lines["date"].dt.month >= 4, lines["date"].dt.year, lines["date"].dt.year - 1)
    lines["invoice_no"] = [f"SI/{y % 100:02d}-{(y + 1) % 100:02d}/{n:06d}" for y, n in zip(fy, inv_no)]
    lines["party"] = lines["buyer_no"].map(lambda x: bmap[x]["name"])
    lines["gstin"] = lines["buyer_no"].map(lambda x: bmap[x]["gstin"])
    lines["voucher_type"] = "Sales"
    # ERP noise --------------------------------------------------------------
    n = len(lines)
    pick = rng.random(n)
    lines.loc[pick < 0.015, "party"] = lines.loc[pick < 0.015, "party"].str.upper()
    lines.loc[(pick >= 0.015) & (pick < 0.03), "party"] = lines.loc[(pick >= 0.015) & (pick < 0.03), "party"] + "  "
    cancelled = lines.sample(frac=0.003, random_state=seed + 1).copy()
    cancelled["voucher_type"] = "Sales (Cancelled)"
    returns = lines.sample(frac=0.005, random_state=seed + 2).copy()
    returns["voucher_type"] = "Credit Note"
    returns["filled"] = returns["sku"].map(lambda s: spec[s][5]).astype(float)  # one damaged pack
    returns["day"] = np.minimum(returns["day"] + rng.integers(1, 6, len(returns)), n_days - 1)
    returns["date"] = days.values[returns["day"].values]
    returns["invoice_no"] = [f"CN/{i:05d}" for i in range(1, len(returns) + 1)]
    dupes = lines.sample(frac=0.002, random_state=seed + 3)
    raw = pd.concat([lines, cancelled, returns, dupes], ignore_index=True).sort_values(["day", "invoice_no"], kind="stable")
    raw = raw.reset_index(drop=True)
    raw["pack_size"] = raw["sku"].map(lambda s: spec[s][5])
    raw["pack_unit"] = raw["sku"].map(lambda s: spec[s][4])
    raw["base_unit"] = raw["sku"].map(lambda s: spec[s][3])
    raw["item_name"] = raw["sku"].map(lambda s: spec[s][1])
    raw["amount"] = np.round(raw["filled"] * raw["rate"], 2)
    invoices = _to_erp(raw, erp, rng)

    items = pd.DataFrame([{
        "sku": s[0], "item_name": s[1], "commodity": s[2], "base_unit": s[3], "pack_unit": s[4],
        "pack_size": s[5], "lead_time_days": s[7], "lead_time_sd": s[8], "moq_packs": s[9],
        "supplier": s[10], "hsn": s[11], "capacity_packs": int(max(200, demand[:, sku_idx[s[0]]].mean() * 60 / s[5])),
    } for s in ITEM_SPECS])
    parties = pd.DataFrame([{
        "party": b["name"], "segment": b["segment"], "area": b["area"], "gstin": b["gstin"],
        "phone": b["phone"], "whatsapp_opt_in": b["whatsapp_opt_in"],
    } for b in buyers])
    stock = truth[["date", "sku", "closing_stock"]].copy()
    stock["item_name"] = stock["sku"].map(lambda s: spec[s][1])
    stock = stock[["date", "item_name", "closing_stock"]].rename(columns={"closing_stock": "closing_qty"})
    open_pos = pd.DataFrame([{
        "item_name": spec[skus[p[1]]][1], "qty": p[2], "order_date": days[p[3]].date().isoformat(),
        "expected_date": (days[-1] + pd.Timedelta(days=p[0] - (n_days - 1))).date().isoformat(),
    } for p in pipeline], columns=["item_name", "qty", "order_date", "expected_date"])
    meta = {
        "seed": seed, "start": str(days[0].date()), "end": str(days[-1].date()), "n_buyers": n_buyers,
        "lines_billed": int(len(lines)), "requests": int(len(req)),
        "lost_sales_share": float(1 - req["filled"].sum() / max(req["qty"].sum(), 1)),
        "churned_buyers": int(sum(b["churn_day"] < n_days for b in buyers)),
        "new_buyers_last_60d": int(sum(b["start_day"] >= n_days - 60 for b in buyers)),
    }
    return SyntheticWholesale(invoices=invoices, items=items, parties=parties, stock=stock,
                              open_pos=open_pos, truth=truth, as_of=days[-1] + pd.Timedelta(days=1),
                              erp=erp, meta=meta)


def _qty_text(qty: float, pack: float, pack_unit: str, base_unit: str, use_base: bool) -> str:
    if use_base:
        return f"{qty:g} {'kg' if base_unit == 'kg' else 'Ltr'}"
    return f"{qty / pack:g} {pack_unit}"


def _to_erp(raw: pd.DataFrame, erp: str, rng: np.random.Generator) -> pd.DataFrame:
    """Shape the billed lines like a typical item-wise sales register export of the given ERP."""
    # deterministic per (invoice, item) so an exported duplicate row stays an exact duplicate
    use_base = (pd.util.hash_pandas_object(raw[["invoice_no", "sku"]], index=False).values % 10) == 0
    qty_text = [_qty_text(q, p, pu, bu, ub) for q, p, pu, bu, ub in
                zip(raw["filled"], raw["pack_size"], raw["pack_unit"], raw["base_unit"], use_base)]
    rate_pack = np.round(raw["rate"] * raw["pack_size"], 2)
    if erp == "tally":
        return pd.DataFrame({
            "Date": raw["date"].dt.strftime("%d-%b-%Y"), "Particulars": raw["party"],
            "Voucher Type": raw["voucher_type"], "Voucher No.": raw["invoice_no"],
            "GSTIN/UIN": raw["gstin"], "Stock Item": raw["item_name"], "Quantity": qty_text,
            "Rate": rate_pack, "Per": raw["pack_unit"], "Value": raw["amount"],
        })
    if erp == "vyapar":
        return pd.DataFrame({
            "Date": raw["date"].dt.strftime("%d/%m/%Y"), "Invoice No.": raw["invoice_no"],
            "Party Name": raw["party"], "Transaction Type": raw["voucher_type"].str.replace("Credit Note", "Sale Return"),
            "Item Name": raw["item_name"], "Quantity": raw["filled"] / raw["pack_size"], "Unit": raw["pack_unit"],
            "Price/Unit": rate_pack, "Amount": raw["amount"],
        })
    if erp == "marg":
        return pd.DataFrame({
            "Bill Date": raw["date"].dt.strftime("%d-%m-%Y"), "Bill No": raw["invoice_no"],
            "Party": raw["party"], "Type": raw["voucher_type"], "Product": raw["item_name"],
            "Qty": raw["filled"] / raw["pack_size"], "Unit": raw["pack_unit"], "Rate": rate_pack,
            "Amount": raw["amount"],
        })
    if erp == "canonical":
        return pd.DataFrame({
            "date": raw["date"], "invoice_no": raw["invoice_no"], "voucher_type": raw["voucher_type"],
            "party": raw["party"], "gstin": raw["gstin"], "item_name": raw["item_name"],
            "qty": raw["filled"], "unit": raw["base_unit"], "rate": raw["rate"], "amount": raw["amount"],
        })
    raise ValueError(f"Unknown ERP format '{erp}'. Use tally, vyapar, marg or canonical.")
