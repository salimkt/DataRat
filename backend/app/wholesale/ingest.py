"""Raw ERP exports -> clean canonical tables.

Accepted inputs
---------------
* Item-wise sales register / day book exported to Excel or CSV from Tally Prime, Marg ERP or
  Vyapar (column names differ; ALIASES maps them), or an already canonical file.
* Tally XML (vouchers fetched over Tally's XML/HTTP interface or exported as XML): parse_tally_xml.
* Optional masters: items (pack size, lead time, MOQ, supplier), parties (segment, phone,
  WhatsApp opt-in), daily closing stock, open purchase orders.

Canonical invoice line (one row per invoice x item)
---------------------------------------------------
date, invoice_no, voucher_type, is_return, buyer_id, buyer_name, gstin, sku, item_name,
commodity, qty (base unit: kg or L), base_unit, rate (Rs per base unit), amount

Cleaning rules (counted in the returned quality report)
------------------------------------------------------
drop cancelled vouchers and non-sales vouchers; mark credit notes / sale returns as returns;
drop exact duplicate rows; parse dd-mm-yyyy / dd-Mon-yyyy / ISO / Tally YYYYMMDD dates;
parse "4 Bag", "100 kg", "15 Ltr" quantity strings and convert to the base unit via the pack
size; normalise party names (case, spacing, punctuation) so one shop is one buyer; map unknown
items to a slug and guess their commodity from the name; drop zero/negative quantities.
"""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from ..utils import slugify

ALIASES: dict[str, tuple[str, ...]] = {
    "date": ("date", "invoice_date", "bill_date", "voucher_date", "vch_date", "txn_date", "transaction_date"),
    "invoice_no": ("invoice_no", "voucher_no", "vch_no", "bill_no", "invoice_number", "ref_no", "voucher_number"),
    "voucher_type": ("voucher_type", "vch_type", "transaction_type", "type", "txn_type", "document_type"),
    "party": ("party", "party_name", "particulars", "customer", "customer_name", "buyer", "ledger", "party_ledger"),
    "gstin": ("gstin", "gstin_uin", "party_gstin", "buyer_gstin", "gst_no"),
    "item_name": ("item_name", "stock_item", "product", "item", "product_name", "sku", "stock_item_name"),
    "qty": ("qty", "quantity", "billed_qty", "actual_qty", "billed_quantity"),
    "unit": ("unit", "uom", "per", "unit_name"),
    "rate": ("rate", "price_unit", "price_per_unit", "unit_price", "price", "sale_price"),
    "amount": ("amount", "value", "net_amount", "total", "line_amount", "taxable_value", "item_amount"),
    "status": ("status", "is_cancelled", "cancelled"),
}

UNIT_TO_BASE = {  # unit word -> (base unit, factor) ; "pack" units are resolved via pack size
    "kg": ("kg", 1.0), "kgs": ("kg", 1.0), "kilogram": ("kg", 1.0), "g": ("kg", 0.001), "gm": ("kg", 0.001),
    "qtl": ("kg", 100.0), "quintal": ("kg", 100.0), "ton": ("kg", 1000.0), "mt": ("kg", 1000.0),
    "l": ("L", 1.0), "ltr": ("L", 1.0), "litre": ("L", 1.0), "liter": ("L", 1.0), "ml": ("L", 0.001),
}
PACK_UNITS = {"bag", "bags", "tin", "tins", "box", "boxes", "pkt", "packet", "packets", "nos", "pcs",
              "sack", "jar", "can", "drum", "bottle", "carton", "ctn"}
COMMODITY_WORDS = {
    "rice": ("rice", "matta", "basmati", "sona", "ponni", "jaya", "biriyani", "kaima", "jeerakasala"),
    "sugar": ("sugar", "jaggery", "sharkara"),
    "pulses": ("dal", "dhal", "gram", "chana", "toor", "tur", "urad", "moong", "masoor", "rajma", "lentil", "peas"),
    "oil": ("oil", "ghee", "vanaspati", "palmolein"),
}


class IngestError(ValueError):
    pass


@dataclass
class WholesaleData:
    """Everything the planner needs, already cleaned."""
    lines: pd.DataFrame
    items: pd.DataFrame
    parties: pd.DataFrame
    stock: pd.DataFrame | None = None       # date, sku, closing_qty (base units)
    open_pos: pd.DataFrame | None = None    # sku, qty, expected_date
    as_of: pd.Timestamp | None = None
    quality: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.as_of is None:
            self.as_of = pd.Timestamp(self.lines["date"].max()).normalize() + pd.Timedelta(days=1)


# --------------------------------------------------------------------------- helpers
def norm_col(c: Any) -> str:
    return re.sub(r"[^0-9a-zA-Z]+", "_", str(c).strip()).strip("_").lower()


def _find(cols: list[str], role: str) -> str | None:
    for alias in ALIASES[role]:
        if alias in cols:
            return alias
    return None


def detect_erp(columns: list[str]) -> str:
    cols = {norm_col(c) for c in columns}
    if {"particulars", "stock_item"} <= cols or "gstin_uin" in cols:
        return "tally"
    if {"bill_date", "product"} <= cols or {"bill_no", "product"} <= cols:
        return "marg"
    if {"party_name", "item_name", "price_unit"} <= cols or "transaction_type" in cols:
        return "vyapar"
    return "canonical"


def norm_party(name: Any) -> str:
    s = re.sub(r"[^0-9a-z]+", " ", str(name).lower()).strip()
    return re.sub(r"\s+", " ", s)


def guess_commodity(name: str) -> str:
    n = str(name).lower()
    for commodity, words in COMMODITY_WORDS.items():
        if any(re.search(rf"\b{w}", n) for w in words):
            return commodity
    return "other"


def guess_pack(name: str) -> tuple[float, str]:
    """'Matta Rice 25kg' -> (25, 'kg'); 'Sunflower Oil 15L Tin' -> (15, 'L')."""
    m = re.search(r"(\d+(?:\.\d+)?)\s*(kg|kgs|ltr|litre|l)\b", str(name).lower())
    if not m:
        return 1.0, "kg"
    return float(m.group(1)), ("L" if m.group(2).startswith("l") else "kg")


def parse_dates(s: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(s):
        return pd.to_datetime(s).dt.tz_localize(None) if getattr(s.dt, "tz", None) else pd.to_datetime(s)
    txt = s.astype(str).str.strip()
    out = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")
    patterns = [
        (r"^\d{8}$", "%Y%m%d"),                      # Tally XML
        (r"^\d{4}-\d{2}-\d{2}", None),               # ISO
        (r"^\d{1,2}-[A-Za-z]{3}-\d{2,4}$", None),    # 1-Apr-2024 / 1-Apr-24
        (r"^\d{1,2}[/-]\d{1,2}[/-]\d{2,4}$", None),  # Indian day-first
    ]
    for pat, fmt in patterns:
        m = txt.str.match(pat) & out.isna()
        if m.any():
            if fmt:
                out[m] = pd.to_datetime(txt[m], format=fmt, errors="coerce")
            else:
                out[m] = pd.to_datetime(txt[m], dayfirst=not pat.startswith(r"^\d{4}"), errors="coerce")
    rest = out.isna() & txt.ne("") & txt.ne("nan") & txt.ne("NaT")
    if rest.any():
        out[rest] = pd.to_datetime(txt[rest], dayfirst=True, errors="coerce")
    return out.dt.normalize()


_QTY_RE = re.compile(r"^\s*(-?[\d,]*\.?\d+)\s*([A-Za-z]*)")


def split_qty(v: Any) -> tuple[float, str]:
    """'4 Bag' -> (4.0, 'bag'); '1,250.00 kg' -> (1250.0, 'kg'); 3 -> (3.0, '')."""
    if isinstance(v, (int, float, np.integer, np.floating)) and not pd.isna(v):
        return float(v), ""
    m = _QTY_RE.match(str(v))
    if not m:
        return float("nan"), ""
    return float(m.group(1).replace(",", "")), m.group(2).lower()


def _num(s: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(s):
        return s.astype(float)
    return pd.to_numeric(s.astype(str).str.replace(r"[^\d.\-]", "", regex=True), errors="coerce")


# --------------------------------------------------------------------------- masters
def prepare_items(items: pd.DataFrame | None, names: list[str] | None = None) -> pd.DataFrame:
    """Normalise the item master; build one from item names if none was supplied."""
    if items is None or len(items) == 0:
        names = sorted(set(names or []))
        rows = []
        for n in names:
            size, unit = guess_pack(n)
            rows.append({"sku": slugify(n).upper(), "item_name": n, "commodity": guess_commodity(n),
                         "base_unit": unit, "pack_unit": "Pack", "pack_size": size})
        items = pd.DataFrame(rows, columns=["sku", "item_name", "commodity", "base_unit", "pack_unit", "pack_size"])
    items = items.copy()
    items.columns = [norm_col(c) for c in items.columns]
    if "item_name" not in items:
        for alt in ("name", "stock_item", "product", "item"):
            if alt in items:
                items = items.rename(columns={alt: "item_name"})
                break
    if "item_name" not in items:
        raise IngestError("The item master needs an item name column.")
    if "sku" not in items:
        items["sku"] = items["item_name"].map(lambda n: slugify(n).upper())
    defaults = {"commodity": None, "base_unit": None, "pack_unit": "Pack", "pack_size": None,
                "lead_time_days": 7, "lead_time_sd": 2.0, "moq_packs": 1, "supplier": "Unassigned",
                "capacity_packs": np.nan}
    for col, default in defaults.items():
        if col not in items:
            items[col] = default
    guessed = items["item_name"].map(guess_pack)
    items["commodity"] = items["commodity"].fillna(items["item_name"].map(guess_commodity))
    items["pack_size"] = pd.to_numeric(items["pack_size"], errors="coerce").fillna(guessed.map(lambda g: g[0]))
    items["base_unit"] = items["base_unit"].fillna(guessed.map(lambda g: g[1]))
    for col in ("lead_time_days", "lead_time_sd", "moq_packs", "capacity_packs"):
        items[col] = pd.to_numeric(items[col], errors="coerce")
    items["lead_time_days"] = items["lead_time_days"].fillna(7).clip(lower=1)
    items["lead_time_sd"] = items["lead_time_sd"].fillna(2.0).clip(lower=0)
    items["moq_packs"] = items["moq_packs"].fillna(1).clip(lower=1)
    items["key"] = items["item_name"].map(norm_party)
    return items.drop_duplicates("sku").reset_index(drop=True)


def prepare_parties(parties: pd.DataFrame | None) -> pd.DataFrame:
    cols = ["buyer_id", "party", "segment", "area", "gstin", "phone", "whatsapp_opt_in"]
    if parties is None or len(parties) == 0:
        return pd.DataFrame(columns=cols)
    p = parties.copy()
    p.columns = [norm_col(c) for c in p.columns]
    pc = _find(list(p.columns), "party") or ("name" if "name" in p else None)
    if pc is None:
        raise IngestError("The party master needs a party / name column.")
    p = p.rename(columns={pc: "party"})
    if "segment" not in p:
        grp = next((c for c in ("group", "under", "ledger_group", "category", "customer_type") if c in p), None)
        p["segment"] = p[grp].map(_segment_from_group) if grp else "unknown"
    for c in ("area", "gstin", "phone"):
        if c not in p:
            p[c] = ""
    if "whatsapp_opt_in" not in p:
        p["whatsapp_opt_in"] = False
    p["whatsapp_opt_in"] = p["whatsapp_opt_in"].astype(str).str.lower().isin(("true", "1", "yes", "y"))
    p["buyer_id"] = p["party"].map(norm_party)
    return p[cols].drop_duplicates("buyer_id").reset_index(drop=True)


def _segment_from_group(g: Any) -> str:
    g = str(g).lower()
    for seg, words in {"hotel": ("hotel", "restaurant", "hotels"), "caterer": ("cater",),
                       "bakery": ("bakery", "sweet", "bakeries"), "kirana": ("kirana", "retail", "store", "shop")}.items():
        if any(w in g for w in words):
            return seg
    return "unknown"


# --------------------------------------------------------------------------- invoices
def clean_invoices(raw: pd.DataFrame, items: pd.DataFrame, erp: str = "auto") -> tuple[pd.DataFrame, dict[str, Any]]:
    if raw is None or len(raw) == 0:
        raise IngestError("The invoice export is empty.")
    q: dict[str, Any] = {"rows_in": int(len(raw))}
    df = raw.copy()
    df.columns = [norm_col(c) for c in df.columns]
    q["erp"] = detect_erp(list(df.columns)) if erp in (None, "", "auto") else erp
    cols = list(df.columns)
    found = {role: _find(cols, role) for role in ALIASES}
    missing = [r for r in ("date", "party", "item_name", "qty") if not found[r]]
    if missing:
        raise IngestError(f"Could not find column(s) for {', '.join(missing)}. Columns seen: {', '.join(cols[:30])}")
    out = pd.DataFrame({role: df[c] for role, c in found.items() if c})
    before = len(out)
    out = out.drop_duplicates()
    q["duplicates_dropped"] = int(before - len(out))
    vt = out["voucher_type"].astype(str).str.lower() if "voucher_type" in out else pd.Series("sales", index=out.index)
    status = out["status"].astype(str).str.lower() if "status" in out else pd.Series("", index=out.index)
    cancelled = vt.str.contains("cancel") | status.isin(("yes", "true", "1", "cancelled"))
    is_return = vt.str.contains("credit note|return|cn")
    is_sale = vt.str.contains("sale|invoice|tax|retail|^$|nan") | is_return
    q["cancelled_dropped"] = int(cancelled.sum())
    q["non_sales_dropped"] = int((~is_sale & ~cancelled).sum())
    keep = ~cancelled & is_sale
    out, is_return = out[keep].copy(), is_return[keep]
    out["is_return"] = is_return.values
    out["date"] = parse_dates(out["date"])
    q["bad_dates_dropped"] = int(out["date"].isna().sum())
    out = out[out["date"].notna()]

    # items -> sku
    key_to_sku = dict(zip(items["key"], items["sku"]))
    out["item_key"] = out["item_name"].map(norm_party)
    out["sku"] = out["item_key"].map(key_to_sku)
    unknown = out["sku"].isna()
    q["unknown_items"] = sorted(out.loc[unknown, "item_name"].astype(str).unique().tolist())[:50]
    out.loc[unknown, "sku"] = out.loc[unknown, "item_name"].map(lambda n: slugify(n).upper())
    meta = items.set_index("sku")
    # quantities -> base unit
    parsed = out["qty"].map(split_qty)
    qty_num = parsed.map(lambda t: t[0]).astype(float)
    unit_txt = parsed.map(lambda t: t[1])
    if "unit" in out:
        unit_txt = unit_txt.where(unit_txt != "", out["unit"].astype(str).str.lower().str.strip())
    pack = out["sku"].map(meta["pack_size"]).fillna(out["item_name"].map(lambda n: guess_pack(n)[0])).astype(float)
    base_unit = out["sku"].map(meta["base_unit"]).fillna(out["item_name"].map(lambda n: guess_pack(n)[1]))

    def factor(u: str, p: float) -> float:
        u = (u or "").lower().rstrip(".")
        if u in UNIT_TO_BASE:
            return UNIT_TO_BASE[u][1]
        if u in PACK_UNITS or u == "pack" or u == "":
            return p
        return p

    out["qty_base"] = qty_num.abs() * np.array([factor(u, p) for u, p in zip(unit_txt, pack)])
    amount = _num(out["amount"]).abs() if "amount" in out else pd.Series(np.nan, index=out.index)
    rate_raw = _num(out["rate"]) if "rate" in out else pd.Series(np.nan, index=out.index)
    out["amount_val"] = amount.fillna(rate_raw * qty_num.abs())
    q["zero_qty_dropped"] = int((~(out["qty_base"] > 0)).sum())
    out = out[out["qty_base"] > 0]
    out["rate_base"] = out["amount_val"] / out["qty_base"]
    # parties
    out["buyer_name"] = out["party"].astype(str).str.strip().str.replace(r"\s+", " ", regex=True)
    out["buyer_id"] = out["party"].map(norm_party)
    gstin = out["gstin"].fillna("").astype(str).str.strip().str.upper() if "gstin" in out else ""
    canon = pd.DataFrame({
        "date": out["date"].values,
        "invoice_no": out["invoice_no"].astype(str).values if "invoice_no" in out else "",
        "voucher_type": out["voucher_type"].astype(str).values if "voucher_type" in out else "Sales",
        "is_return": out["is_return"].values,
        "buyer_id": out["buyer_id"].values,
        "buyer_name": out["buyer_name"].values,
        "gstin": gstin.values if isinstance(gstin, pd.Series) else gstin,
        "sku": out["sku"].values,
        "item_name": out["item_name"].astype(str).values,
        "commodity": out["sku"].map(meta["commodity"]).fillna(out["item_name"].map(guess_commodity)).values,
        "qty": out["qty_base"].round(3).values,
        "base_unit": base_unit.values,
        "rate": out["rate_base"].round(3).values,
        "amount": out["amount_val"].round(2).values,
    })
    canon["gstin"] = canon["gstin"].replace({"NAN": "", "NONE": ""})
    # a party name variant ("KIRANA 017  ") collapses to the same buyer_id; keep the most common spelling
    canon["buyer_name"] = canon.groupby("buyer_id")["buyer_name"].transform(lambda s: s.mode().iat[0])
    canon = canon.sort_values(["date", "invoice_no", "sku"]).reset_index(drop=True)
    q.update({
        "rows_out": int(len(canon)), "returns": int(canon["is_return"].sum()),
        "buyers": int(canon["buyer_id"].nunique()), "items": int(canon["sku"].nunique()),
        "date_min": canon["date"].min().date().isoformat() if len(canon) else None,
        "date_max": canon["date"].max().date().isoformat() if len(canon) else None,
    })
    if canon.empty:
        raise IngestError("No sales lines left after cleaning.")
    return canon, q


def prepare_stock(stock: pd.DataFrame | None, items: pd.DataFrame) -> pd.DataFrame | None:
    """Daily closing stock export -> date, sku, closing_qty (base units)."""
    if stock is None or len(stock) == 0:
        return None
    s = stock.copy()
    s.columns = [norm_col(c) for c in s.columns]
    dc = _find(list(s.columns), "date")
    ic = _find(list(s.columns), "item_name")
    qc = next((c for c in ("closing_qty", "closing_quantity", "closing_stock", "closing_balance", "qty", "quantity") if c in s), None)
    if not (dc and ic and qc):
        raise IngestError("The stock file needs date, item and closing quantity columns.")
    key_to_sku = dict(zip(items["key"], items["sku"]))
    meta = items.set_index("sku")
    sku = s[ic].map(norm_party).map(key_to_sku).fillna(s[ic].map(lambda n: slugify(n).upper()))
    parsed = s[qc].map(split_qty)
    qty = parsed.map(lambda t: t[0]).astype(float)
    unit = parsed.map(lambda t: t[1])
    pack = sku.map(meta["pack_size"]).fillna(1.0).astype(float)
    fac = np.array([UNIT_TO_BASE[u][1] if u in UNIT_TO_BASE else (p if u in PACK_UNITS else 1.0)
                    for u, p in zip(unit, pack)])
    out = pd.DataFrame({"date": parse_dates(s[dc]), "sku": sku, "closing_qty": (qty * fac).clip(lower=0)})
    return out.dropna(subset=["date"]).groupby(["date", "sku"], as_index=False)["closing_qty"].sum()


def prepare_open_pos(pos: pd.DataFrame | None, items: pd.DataFrame) -> pd.DataFrame | None:
    if pos is None or len(pos) == 0:
        return pd.DataFrame(columns=["sku", "qty", "expected_date"])
    p = pos.copy()
    p.columns = [norm_col(c) for c in p.columns]
    ic = _find(list(p.columns), "item_name")
    qc = _find(list(p.columns), "qty")
    if not (ic and qc):
        raise IngestError("The open purchase orders file needs item and qty columns.")
    key_to_sku = dict(zip(items["key"], items["sku"]))
    sku = p[ic].map(norm_party).map(key_to_sku).fillna(p[ic].map(lambda n: slugify(n).upper()))
    exp = parse_dates(p["expected_date"]) if "expected_date" in p else pd.Series(pd.NaT, index=p.index)
    return pd.DataFrame({"sku": sku, "qty": _num(p[qc]).fillna(0), "expected_date": exp})


def load(invoices: pd.DataFrame, items: pd.DataFrame | None = None, parties: pd.DataFrame | None = None,
         stock: pd.DataFrame | None = None, open_pos: pd.DataFrame | None = None, erp: str = "auto",
         as_of: str | pd.Timestamp | None = None) -> WholesaleData:
    """One call from raw exports to WholesaleData."""
    if items is None or len(items) == 0:
        tmp = invoices.copy()
        tmp.columns = [norm_col(c) for c in tmp.columns]
        ic = _find(list(tmp.columns), "item_name")
        if ic is None:
            raise IngestError("Could not find the item column in the invoice export.")
        items_m = prepare_items(None, tmp[ic].dropna().astype(str).str.strip().tolist())
    else:
        items_m = prepare_items(items)
    lines, quality = clean_invoices(invoices, items_m, erp)
    # add items that were sold but missing from the master
    extra = sorted(set(lines["sku"]) - set(items_m["sku"]))
    if extra:
        names = lines.drop_duplicates("sku").set_index("sku").loc[extra, "item_name"].tolist()
        add = prepare_items(None, names)
        add["sku"] = extra
        items_m = pd.concat([items_m, add], ignore_index=True)
    parties_m = prepare_parties(parties)
    seen = lines.drop_duplicates("buyer_id")[["buyer_id", "buyer_name", "gstin"]]
    if len(parties_m):
        parties_m = seen.merge(parties_m.drop(columns=["gstin"]), on="buyer_id", how="left")
    else:
        parties_m = seen.assign(segment="unknown", area="", phone="", whatsapp_opt_in=False, party=seen["buyer_name"])
    parties_m["segment"] = parties_m["segment"].fillna("unknown")
    parties_m["whatsapp_opt_in"] = parties_m["whatsapp_opt_in"].fillna(False).astype(bool)
    for c in ("area", "phone"):
        parties_m[c] = parties_m[c].fillna("")
    st = prepare_stock(stock, items_m)
    quality["stock_snapshots"] = 0 if st is None else int(len(st))
    pos = prepare_open_pos(open_pos, items_m)
    data = WholesaleData(lines=lines, items=items_m, parties=parties_m, stock=st, open_pos=pos,
                         as_of=None if as_of is None else pd.Timestamp(as_of).normalize(), quality=quality)
    return data


# --------------------------------------------------------------------------- Tally XML
def parse_tally_xml(xml: bytes | str) -> pd.DataFrame:
    """Flatten Tally voucher XML (ENVELOPE ... VOUCHER ... ALLINVENTORYENTRIES.LIST) into rows
    with Tally-register column names, ready for clean_invoices(erp="tally").

    ElementTree does not resolve external entities; only parse XML from your own Tally instance.
    """
    root = ET.fromstring(xml if isinstance(xml, (bytes, str)) else bytes(xml))
    rows = []
    for v in root.iter("VOUCHER"):
        def t(tag: str, node: ET.Element = v) -> str:
            el = node.find(tag)
            return (el.text or "").strip() if el is not None and el.text else ""
        vtype = t("VOUCHERTYPENAME") or v.get("VCHTYPE", "")
        if t("ISCANCELLED").lower() == "yes":
            vtype = f"{vtype} (Cancelled)"
        # The party is PARTYLEDGERNAME in most exports; Tally's sample XML instead marks the
        # party ledger entry with ISPARTYLEDGER=Yes.
        party = t("PARTYLEDGERNAME") or t("PARTYNAME")
        if not party:
            for le in v.findall("LEDGERENTRIES.LIST") + v.findall("ALLLEDGERENTRIES.LIST"):
                if t("ISPARTYLEDGER", le).lower() == "yes":
                    party = t("LEDGERNAME", le)
                    break
        entries = v.findall("ALLINVENTORYENTRIES.LIST") + v.findall("INVENTORYENTRIES.LIST")
        for e in entries:
            rate = t("RATE", e)
            rows.append({
                "Date": t("DATE"), "Particulars": party,
                "Voucher Type": vtype, "Voucher No.": t("VOUCHERNUMBER"), "GSTIN/UIN": t("PARTYGSTIN"),
                "Stock Item": t("STOCKITEMNAME", e), "Quantity": t("BILLEDQTY", e) or t("ACTUALQTY", e),
                "Rate": rate.split("/")[0] if rate else "", "Value": t("AMOUNT", e),
            })
    if not rows:
        raise IngestError("No vouchers with inventory entries found in the Tally XML.")
    return pd.DataFrame(rows)
