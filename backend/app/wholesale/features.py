"""Feature engineering.

1. ``daily_panel``: one row per item x day with gross sales, price, stockout flag and the
   *unconstrained* demand estimate (sales on stockout days are censored, so they are lifted to a
   baseline estimated from nearby in-stock days).
2. ``demand_frame``: direct multi-horizon training/inference rows. For a forecast origin t (the
   last day with data) and horizon k = 1..H, the target is demand on day t+k divided by the item's
   recent scale, so one global model serves fast and slow items alike. Features use data up to t
   only, plus calendar regressors of the target day (known in advance).
3. ``buyer_snapshots``: buyer x item RFM features at a cutoff date, with labels "orders within
   the next N days". Reorder intervals are shrunk towards a segment x commodity prior
   (empirical Bayes) so buyers with one or two orders still get a sensible estimate.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .ingest import WholesaleData
from .seasonality import CALENDAR_COLUMNS, calendar_features

EPS = 1e-6
DOW_WEEKS = 8


# ============================================================================ demand panel
def daily_panel(data: WholesaleData) -> pd.DataFrame:
    lines = data.lines[~data.lines["is_return"]]
    end = pd.Timestamp(data.as_of) - pd.Timedelta(days=1)
    lines = lines[lines["date"] <= end]
    pack = data.items.set_index("sku")["pack_size"].to_dict()
    g = lines.groupby(["sku", "date"]).agg(sales=("qty", "sum"), amount=("amount", "sum"),
                                           orders=("qty", "size"), buyers=("buyer_id", "nunique"))
    frames = []
    stock = data.stock
    stock_idx = stock.set_index(["sku", "date"])["closing_qty"] if stock is not None and len(stock) else None
    for sku, grp in g.groupby(level="sku"):
        grp = grp.droplevel("sku")
        dates = pd.date_range(grp.index.min(), end, freq="D")
        f = grp.reindex(dates)
        f[["sales", "amount", "orders", "buyers"]] = f[["sales", "amount", "orders", "buyers"]].fillna(0)
        price = (f["amount"] / f["sales"].replace(0, np.nan))
        f["price"] = price.ffill().bfill()
        f["sku"] = sku
        p = float(pack.get(sku, 1.0) or 1.0)
        if stock_idx is not None and sku in stock_idx.index.get_level_values(0):
            closing = stock_idx.loc[sku].reindex(dates).ffill()
            f["closing"] = closing.values
            f["stockout"] = (closing < p).fillna(False).values
            f["stockout_source"] = "stock"
        else:
            f["closing"] = np.nan
            f["stockout"] = suspect_stockouts(f["sales"], f["orders"]).values
            f["stockout_source"] = "inferred"
        f["demand"] = unconstrain(f["sales"], f["stockout"])
        frames.append(f.rename_axis("date").reset_index())
    panel = pd.concat(frames, ignore_index=True)
    panel["commodity"] = panel["sku"].map(data.items.set_index("sku")["commodity"]).fillna("other")
    return panel


def suspect_stockouts(sales: pd.Series, orders: pd.Series) -> pd.Series:
    """Without stock snapshots: a run of >= 2 zero-sale days for an item that normally gets
    >= 1.5 orders a day is treated as a probable stockout (censored)."""
    busy = orders.shift(1).rolling(28, min_periods=7).mean() >= 1.5
    zero = sales.eq(0)
    run = zero.groupby((~zero).cumsum()).cumsum()
    run_len = run.groupby((~zero).cumsum()).transform("max")
    return (zero & busy & (run_len >= 2)).fillna(False)


def unconstrain(sales: pd.Series, stockout: pd.Series) -> pd.Series:
    """Censored-demand correction: on stockout days demand >= sales, so replace sales with the
    larger of sales and an in-stock baseline (centred 29-day mean of uncensored days x weekday
    factor). Uncensored days are left as observed."""
    so = stockout.astype(bool)
    if not so.any():
        return sales.astype(float)
    unc = sales.where(~so)
    base = unc.rolling(29, center=True, min_periods=5).mean()
    base = base.fillna(unc.rolling(56, min_periods=1).mean()).fillna(unc.mean()).fillna(0)
    dow = pd.Series(sales.index.dayofweek, index=sales.index)
    prof = unc.groupby(dow).mean() / max(unc.mean(), EPS)
    factor = dow.map(prof).fillna(1.0).clip(0.3, 2.0)
    return pd.Series(np.where(so, np.maximum(sales, base * factor), sales), index=sales.index).astype(float)


# ============================================================================ demand frame
DEMAND_FEATURES = [
    "k", "r7", "r14", "r28", "r91", "sd28", "zero28", "since_nz", "last", "trend",
    "dow_mean", "ly", "price_rel90", "price_ch14", "price_ch30", "so28", "age",
] + [f"t_{c}" for c in CALENDAR_COLUMNS] + ["sku_cat", "commodity_cat"]


def _origin_features(f: pd.DataFrame) -> pd.DataFrame:
    """Per-item features indexed by origin date (each uses data up to and including that date)."""
    d = f["demand"]
    scale = d.rolling(56, min_periods=1).mean().clip(lower=EPS)
    o = pd.DataFrame(index=f.index)
    o["scale"] = scale
    for w in (7, 14, 28, 91):
        o[f"r{w}"] = d.rolling(w, min_periods=1).mean() / scale
    o["sd28"] = d.rolling(28, min_periods=3).std() / scale
    o["zero28"] = d.eq(0).rolling(28, min_periods=1).mean()
    nz = d.gt(0)
    last_nz = pd.Series(np.where(nz, np.arange(len(d)), np.nan), index=d.index).ffill()
    o["since_nz"] = np.minimum(np.arange(len(d)) - last_nz.fillna(-60).values, 60)
    o["last"] = d / scale
    o["trend"] = o["r7"] - o["r28"]
    price = f["price"]
    o["price_rel90"] = price / price.rolling(90, min_periods=1).mean()
    o["price_ch14"] = price.pct_change(14, fill_method=None)
    o["price_ch30"] = price.pct_change(30, fill_method=None)
    o["so28"] = f["stockout"].astype(float).rolling(28, min_periods=1).sum()
    o["age"] = np.arange(len(d), dtype=float)
    dow = f.index.dayofweek
    for j in range(7):
        m = d[dow == j].rolling(DOW_WEEKS, min_periods=1).mean()
        o[f"dowm_{j}"] = m.reindex(f.index).ffill() / scale
    # centred weekly mean, used for "same period last year"
    o["c7"] = d.rolling(7, center=True, min_periods=3).mean()
    return o


def demand_frame(panel: pd.DataFrame, origins: dict[str, list[pd.Timestamp]] | None, horizon: int,
                 region: str, with_target: bool, sku_categories: list[str], commodity_categories: list[str]
                 ) -> pd.DataFrame:
    """Rows = (item, origin, k). ``origins`` maps sku -> origin dates (None = last date only)."""
    out = []
    cal_cache: dict = {}
    for sku, f in panel.groupby("sku", sort=True):
        f = f.set_index("date").sort_index()
        o = _origin_features(f)
        org = [f.index[-1]] if origins is None else [t for t in origins.get(sku, []) if t in o.index]
        if not org:
            continue
        org = pd.DatetimeIndex(org)
        ks = np.arange(1, horizon + 1)
        t_rep = np.repeat(org.values, horizon)
        k_rep = np.tile(ks, len(org))
        tgt = pd.DatetimeIndex(t_rep) + pd.to_timedelta(k_rep, unit="D")
        base = o.loc[t_rep]
        rows = pd.DataFrame({"sku": sku, "origin": t_rep, "date": tgt.values, "k": k_rep})
        for c in ("scale", "r7", "r14", "r28", "r91", "sd28", "zero28", "since_nz", "last", "trend",
                  "price_rel90", "price_ch14", "price_ch30", "so28", "age"):
            rows[c] = base[c].values
        tdow = tgt.dayofweek.values
        dowm = base[[f"dowm_{j}" for j in range(7)]].values
        rows["dow_mean"] = dowm[np.arange(len(rows)), tdow]
        ly_dates = tgt - pd.Timedelta(days=364)
        rows["ly"] = o["c7"].reindex(ly_dates).values / base["scale"].values
        key = (tgt.min(), tgt.max())
        if key not in cal_cache:
            cal_cache[key] = calendar_features(pd.date_range(*key, freq="D"), region)
        cal = cal_cache[key].reindex(tgt)
        for c in CALENDAR_COLUMNS:
            rows[f"t_{c}"] = cal[c].values
        if with_target:
            dem = f["demand"].reindex(tgt)
            rows["y_demand"] = dem.values
            rows["y_sales"] = f["sales"].reindex(tgt).values
            rows["y_stockout"] = f["stockout"].reindex(tgt).values
            rows["y"] = dem.values / base["scale"].values
        out.append(rows)
    if not out:
        return pd.DataFrame(columns=["sku", "origin", "date"] + DEMAND_FEATURES)
    df = pd.concat(out, ignore_index=True)
    df["commodity"] = df["sku"].map(panel.drop_duplicates("sku").set_index("sku")["commodity"])
    df["sku_cat"] = pd.Categorical(df["sku"], categories=sku_categories)
    df["commodity_cat"] = pd.Categorical(df["commodity"], categories=commodity_categories)
    return df


# ============================================================================ buyer snapshots
BUYER_FEATURES = [
    "recency", "n_orders", "n_30", "n_90", "qty_90", "amount_90", "mean_int", "std_int", "last5_int",
    "mean_int_shrunk", "due_ratio", "z_due", "gap_left", "cv_int", "last_qty_ratio", "tenure",
    "recency_any", "items_90", "price_ch14", "price_rel90", "item_so7", "w_pull", "w_wedding", "w_ramzan",
    "segment_cat", "commodity_cat",
]


def _orders(lines: pd.DataFrame) -> pd.DataFrame:
    l = lines[~lines["is_return"]]
    return (l.groupby(["buyer_id", "sku", "date"], as_index=False)
             .agg(qty=("qty", "sum"), amount=("amount", "sum")))


def interval_priors(orders: pd.DataFrame, parties: pd.DataFrame, items: pd.DataFrame,
                    until: pd.Timestamp | None = None) -> dict[tuple[str, str], float]:
    """Median reorder interval per (segment, commodity) from pairs with >= 3 orders."""
    o = orders if until is None else orders[orders["date"] <= until]
    seg = parties.set_index("buyer_id")["segment"]
    com = items.set_index("sku")["commodity"]
    o = o.sort_values("date")
    gaps = o.groupby(["buyer_id", "sku"])["date"].diff().dt.days
    tmp = o.assign(gap=gaps).dropna(subset=["gap"])
    per_pair = tmp.groupby(["buyer_id", "sku"])["gap"].agg(["median", "size"])
    per_pair = per_pair[per_pair["size"] >= 2].reset_index()
    per_pair["segment"] = per_pair["buyer_id"].map(seg).fillna("unknown")
    per_pair["commodity"] = per_pair["sku"].map(com).fillna("other")
    pri = per_pair.groupby(["segment", "commodity"])["median"].median().to_dict()
    pri[("*", "*")] = float(per_pair["median"].median()) if len(per_pair) else 14.0
    return pri


def _window_calendar(cal: pd.DataFrame, h: int) -> pd.DataFrame:
    """For each date c: max festival pull / mean wedding index / any Ramzan over (c, c+h]."""
    rev = cal.iloc[::-1]
    out = pd.DataFrame(index=cal.index)
    out["w_pull"] = rev["festival_pull"].rolling(h, min_periods=1).max().iloc[::-1].shift(-1)
    out["w_wedding"] = rev["wedding_index"].rolling(h, min_periods=1).mean().iloc[::-1].shift(-1)
    out["w_ramzan"] = rev["ramzan"].rolling(h, min_periods=1).max().iloc[::-1].shift(-1)
    return out.ffill()


def buyer_snapshots(data: WholesaleData, cutoffs: list[pd.Timestamp], panel: pd.DataFrame,
                    priors: dict[tuple[str, str], float], label_windows: tuple[int, ...] = (),
                    lookback: int = 365, prior_weight: float = 3.0, region: str = "kerala", window: int = 7,
                    segment_categories: list[str] | None = None,
                    commodity_categories: list[str] | None = None) -> pd.DataFrame:
    orders = _orders(data.lines)
    orders = orders[orders["date"] < pd.Timestamp(data.as_of)]
    seg = data.parties.set_index("buyer_id")["segment"]
    com = data.items.set_index("sku")["commodity"]
    cut = pd.DatetimeIndex(sorted(set(pd.Timestamp(c).normalize() for c in cutoffs)))
    c_days = cut.values.astype("datetime64[D]").astype(np.int64)
    rows = []
    for (buyer, sku), g in orders.groupby(["buyer_id", "sku"], sort=False):
        d = g["date"].values.astype("datetime64[D]").astype(np.int64)
        order = np.argsort(d, kind="stable")
        d, qty, amt = d[order], g["qty"].values[order], g["amount"].values[order]
        idx = np.searchsorted(d, c_days, side="right")  # orders known at the cutoff (end of day c)
        ok = idx >= 1
        if not ok.any():
            continue
        last = np.where(ok, d[np.maximum(idx - 1, 0)], 0)
        ok &= (c_days - last) <= lookback
        if not ok.any():
            continue
        sel = np.where(ok)[0]
        i, c = idx[sel], c_days[sel]
        n = i.astype(float)
        cq = np.concatenate([[0.0], np.cumsum(qty)])
        ca = np.concatenate([[0.0], np.cumsum(amt)])
        gaps = np.diff(d).astype(float)
        cg = np.concatenate([[0.0], np.cumsum(gaps)])
        cg2 = np.concatenate([[0.0], np.cumsum(gaps ** 2)])
        n_int = n - 1
        mean_int = np.where(n_int > 0, cg[np.maximum(i - 1, 0)] / np.maximum(n_int, 1), np.nan)
        var = np.where(n_int > 1, cg2[np.maximum(i - 1, 0)] / np.maximum(n_int, 1) - mean_int ** 2, np.nan)
        std_int = np.sqrt(np.clip(var, 0, None))
        k5 = np.minimum(n_int, 5)
        last5 = np.where(n_int > 0, (d[i - 1] - d[np.maximum(i - 1 - k5.astype(int), 0)]) / np.maximum(k5, 1), np.nan)
        i30 = np.searchsorted(d, c - 30, side="right")
        i90 = np.searchsorted(d, c - 90, side="right")
        i365 = np.searchsorted(d, c - 365, side="right")
        typical = pd.Series(qty).rolling(5, min_periods=1).median().values
        segment = seg.get(buyer, "unknown")
        commodity = com.get(sku, "other")
        prior = priors.get((segment, commodity), priors.get(("*", "*"), 14.0))
        shrunk = (np.nan_to_num(mean_int) * n_int + prior_weight * prior) / (n_int + prior_weight)
        std_shr = (np.nan_to_num(std_int) * n_int + prior_weight * 0.6 * prior) / (n_int + prior_weight)
        recency = (c - d[i - 1]).astype(float)
        r = pd.DataFrame({
            "buyer_id": buyer, "sku": sku, "cutoff": cut[sel], "segment": segment, "commodity": commodity,
            "recency": recency, "n_orders": n, "n_30": (i - i30).astype(float), "n_90": (i - i90).astype(float),
            "qty_90": cq[i] - cq[i90], "amount_90": ca[i] - ca[i90], "amount_365": ca[i] - ca[i365],
            "mean_int": mean_int, "std_int": std_int, "last5_int": last5,
            "mean_int_shrunk": shrunk, "due_ratio": recency / np.maximum(shrunk, 1.0),
            "z_due": (recency - shrunk) / (std_shr + 1.0), "gap_left": shrunk - recency,
            "cv_int": std_int / np.maximum(mean_int, 1.0),
            "last_qty": qty[i - 1], "typical_qty": typical[i - 1],
            "last_qty_ratio": qty[i - 1] / np.maximum(typical[i - 1], EPS),
            "last_rate": amt[i - 1] / np.maximum(qty[i - 1], EPS),
            "tenure": (c - d[0]).astype(float),
        })
        for h in label_windows:
            j = np.searchsorted(d, c + h, side="right")
            r[f"y_{h}"] = (j - i > 0).astype(int)
            r[f"qty_next_{h}"] = cq[j] - cq[i]
        rows.append(r)
    if not rows:
        return pd.DataFrame(columns=["buyer_id", "sku", "cutoff"] + BUYER_FEATURES)
    df = pd.concat(rows, ignore_index=True)
    # buyer-level context: days since any order, breadth of the basket
    df["recency_any"] = df.groupby(["buyer_id", "cutoff"])["recency"].transform("min")
    df["items_90"] = df.groupby(["buyer_id", "cutoff"])["recency"].transform(lambda s: float((s <= 90).sum()))
    # item-level context at the cutoff: price momentum and stockouts at the wholesaler
    pi = panel.set_index(["sku", "date"])
    price = pi["price"]
    p14 = price.groupby(level=0).pct_change(14, fill_method=None)
    rel = price / price.groupby(level=0).transform(lambda s: s.rolling(90, min_periods=1).mean())
    so7 = pi["stockout"].astype(float).groupby(level=0).transform(lambda s: s.rolling(7, min_periods=1).sum())
    key = pd.MultiIndex.from_arrays([df["sku"], df["cutoff"]])
    df["price_ch14"] = p14.reindex(key).values
    df["price_rel90"] = rel.reindex(key).values
    df["item_so7"] = so7.reindex(key).values
    cal = calendar_features(pd.date_range(cut.min(), cut.max() + pd.Timedelta(days=40), freq="D"), region)
    df = df.join(_window_calendar(cal, window), on="cutoff")
    df["segment_cat"] = pd.Categorical(df["segment"], categories=segment_categories or sorted(df["segment"].unique()))
    df["commodity_cat"] = pd.Categorical(df["commodity"], categories=commodity_categories or sorted(df["commodity"].unique()))
    return df
