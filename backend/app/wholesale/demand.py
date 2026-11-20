"""Item demand forecasting (7 / 15 / 30 days) and purchase-order quantities.

Model: one global gradient-boosted model, direct multi-horizon. Each training row is
(item, origin t, horizon k) -> demand(t+k) / scale(t). Direct (not recursive) forecasting avoids
compounding errors and gives a full daily path, so any window (lead time, lead time + review
period, 7/15/30 days) is a simple sum.

Evaluation: a hold-out backtest over the last ``backtest_days`` compares the model with
naive (28-day mean), seasonal naive (8-week weekday profile), Holt-Winters (statsmodels, the
method DataRat's dashboard already uses) and Croston-SBA. Metrics are computed on in-stock days
only, because sales on stockout days understate demand.

Inventory policy (min/max, i.e. continuous-review (s, S), with daily review):
    D(P)   = forecast demand over the next P days
    SS(P)  = z * sqrt( sigma_e(P)^2 + dbar^2 * sigma_LT^2 )
             sigma_e(P): std of P-day forecast errors from the backtest, dbar: mean daily demand,
             sigma_LT: lead-time std (days), z = inverse normal of the service level
    ROP    = D(L) + SS(L)                       reorder point ("min")
    S      = D(L + R) + SS(L + R)               order-up-to level ("max")
    IP     = on hand + on order                 inventory position
    order  = S - IP if IP <= ROP, rounded up to whole packs, at least the MOQ, capped by capacity
    status = URGENT (order now; stock runs out within the lead time), ORDER, EXPEDITE (enough is on
             order but it arrives after the projected stockout: chase the supplier), OK, or
             CHECK STOCK (no stock snapshot: compare the min/max levels with a physical count)
"""
from __future__ import annotations

import math
import warnings
from statistics import NormalDist
from typing import Any

import numpy as np
import pandas as pd

from .config import PlannerConfig
from .features import DEMAND_FEATURES, demand_frame
from .ingest import WholesaleData
from .models import backend_name, feature_importance, fit_quiet, make_regressor

MIN_ORIGIN_HISTORY = 28


class DemandForecaster:
    def __init__(self, cfg: PlannerConfig):
        self.cfg = cfg
        self.model: Any = None
        self.sku_categories: list[str] = []
        self.commodity_categories: list[str] = []
        self.sigma: dict[str, list[float]] = {}     # sku -> RMSE of P-day sums, P = 1..H
        self.sigma_rel: list[float] = []            # pooled relative error by P (fallback)
        self.metrics: dict[str, Any] = {}
        self.importance: list[dict[str, Any]] = []
        self.backend = backend_name()

    # ------------------------------------------------------------------ training
    def _origins(self, panel: pd.DataFrame, start_offset: int, until: pd.Timestamp, step: int,
                 last: pd.Timestamp | None = None) -> dict[str, list[pd.Timestamp]]:
        out = {}
        for sku, g in panel.groupby("sku"):
            first = g["date"].min() + pd.Timedelta(days=start_offset)
            stop = until if last is None else min(until, last)
            if first > stop:
                continue
            out[sku] = list(pd.date_range(first, stop, freq=f"{step}D"))
        return out

    def _xy(self, frame: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        frame = frame[frame["y"].notna() & np.isfinite(frame["y"])]
        w = np.where(frame["y_stockout"].astype("boolean").fillna(False).to_numpy(bool), 0.5, 1.0)  # imputed targets count less
        return frame[DEMAND_FEATURES], frame["y"].clip(lower=0).values, w

    def fit(self, panel: pd.DataFrame, until: pd.Timestamp | None = None) -> "DemandForecaster":
        H = self.cfg.max_horizon
        end = panel["date"].max() if until is None else pd.Timestamp(until)
        self.sku_categories = sorted(panel["sku"].unique())
        self.commodity_categories = sorted(panel["commodity"].unique())
        origins = self._origins(panel, MIN_ORIGIN_HISTORY, end - pd.Timedelta(days=1), self.cfg.origin_step_days)
        frame = demand_frame(panel[panel["date"] <= end], origins, H, self.cfg.region, True,
                             self.sku_categories, self.commodity_categories)
        X, y, w = self._xy(frame)
        self.model = fit_quiet(make_regressor(self.cfg), X, y, w)
        self.importance = feature_importance(self.model, DEMAND_FEATURES)
        return self

    def _predict_frame(self, frame: pd.DataFrame) -> np.ndarray:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ratio = np.clip(self.model.predict(frame[DEMAND_FEATURES]), 0, None)
        return ratio * frame["scale"].values

    # ------------------------------------------------------------------ inference
    def predict(self, panel: pd.DataFrame) -> pd.DataFrame:
        """Daily forecast for k = 1..H from the last date of each item. Columns: sku, date, k,
        forecast, method ('model' or 'cold_start')."""
        H = self.cfg.max_horizon
        frame = demand_frame(panel, None, H, self.cfg.region, False, self.sku_categories, self.commodity_categories)
        frame["forecast"] = self._predict_frame(frame) if len(frame) else []
        frame["method"] = "model"
        out = frame[["sku", "date", "k", "forecast", "method"]].copy()
        # cold start: items with very little history get a commodity-based prior instead
        ages = panel.groupby("sku")["date"].agg(lambda s: (s.max() - s.min()).days + 1)
        cold = ages[ages < self.cfg.min_history_days].index
        if len(cold):
            out = out[~out["sku"].isin(cold)]
            out = pd.concat([out, self._cold_start(panel, list(cold))], ignore_index=True)
        return out.sort_values(["sku", "k"]).reset_index(drop=True)

    def _cold_start(self, panel: pd.DataFrame, skus: list[str]) -> pd.DataFrame:
        """New item: blend its own (short) mean with a fraction of the commodity's average item,
        weighted by days of history, and shape it by the commodity weekday profile."""
        H = self.cfg.max_horizon
        end = panel["date"].max()
        recent = panel[panel["date"] > end - pd.Timedelta(days=56)]
        rows = []
        for sku in skus:
            g = panel[panel["sku"] == sku]
            com = g["commodity"].iat[0]
            peers = recent[(recent["commodity"] == com) & (~recent["sku"].isin(skus))]
            peer_mean = peers.groupby("sku")["demand"].mean().mean() if len(peers) else 0.0
            peer_mean = 0.0 if pd.isna(peer_mean) else float(peer_mean)
            n = len(g)
            own = float(g["demand"].mean()) if n else 0.0
            wt = n / (n + 14.0)
            level = wt * own + (1 - wt) * 0.25 * peer_mean
            prof = peers.groupby(peers["date"].dt.dayofweek)["demand"].mean()
            prof = (prof / prof.mean()).reindex(range(7)).fillna(1.0) if len(prof) and prof.mean() > 0 else pd.Series(1.0, index=range(7))
            dates = pd.date_range(g["date"].max() + pd.Timedelta(days=1), periods=H, freq="D")
            for k, d in enumerate(dates, start=1):
                rows.append({"sku": sku, "date": d, "k": k, "forecast": level * prof[d.dayofweek], "method": "cold_start"})
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------ backtest
    def backtest(self, panel: pd.DataFrame) -> dict[str, Any]:
        """Fit on data before the hold-out, score on it, store error sigmas for safety stock."""
        cfg, H = self.cfg, self.cfg.max_horizon
        end = panel["date"].max()
        split = end - pd.Timedelta(days=cfg.backtest_days)
        self.fit(panel, until=split)
        last_origin = end - pd.Timedelta(days=H)
        test_origins = {}
        for sku, g in panel.groupby("sku"):
            first = max(split, g["date"].min() + pd.Timedelta(days=MIN_ORIGIN_HISTORY))
            if first <= last_origin:
                test_origins[sku] = list(pd.date_range(first, last_origin, freq="D"))
        if not test_origins:
            return {"error": "not enough history for a backtest"}
        frame = demand_frame(panel, test_origins, H, cfg.region, True, self.sku_categories, self.commodity_categories)
        frame["pred"] = self._predict_frame(frame)
        self._sigmas(frame)
        # model comparison on a weekly subset of origins (keeps Holt-Winters affordable)
        weekly = sorted({t for v in test_origins.values() for t in v})[::7]
        sub = frame[frame["origin"].isin(weekly)].copy()
        preds = {"lightgbm" if self.backend == "lightgbm" else "gbm (sklearn)": sub["pred"].values}
        if cfg.run_baselines:
            preds.update(_baselines(panel, sub))
        train_part = panel[panel["date"] <= split]
        mase_scale = _mase_scale(train_part)
        table = []
        for name, p in preds.items():
            table.append({"model": name, **_score(sub, p, cfg.horizons, mase_scale)})
        best = min(table, key=lambda r: r[f"wape_{cfg.horizons[0]}"])["model"]
        self.metrics = {
            "backend": self.backend, "split_date": split.date().isoformat(), "end_date": end.date().isoformat(),
            "test_origins": len(weekly), "items": int(sub["sku"].nunique()),
            "evaluated_on": "in-stock days only (stockout days have censored sales)",
            "models": table, "best_model": best,
        }
        return self.metrics

    def _sigmas(self, frame: pd.DataFrame) -> None:
        H = self.cfg.max_horizon
        f = frame.sort_values(["sku", "origin", "k"])
        rel_num = np.zeros(H)
        rel_den = np.zeros(H)
        for sku, g in f.groupby("sku"):
            piv_p = g.pivot(index="origin", columns="k", values="pred").reindex(columns=range(1, H + 1))
            piv_y = g.pivot(index="origin", columns="k", values="y_demand").reindex(columns=range(1, H + 1))
            cp, cy = piv_p.cumsum(axis=1), piv_y.cumsum(axis=1)
            err = (cp - cy)
            ok = piv_y.notna().all(axis=1)
            err, cy = err[ok], cy[ok]
            if len(err) < 3:
                continue
            self.sigma[sku] = [float(x) for x in np.sqrt((err ** 2).mean(axis=0)).values]
            rel_num += (err ** 2).sum(axis=0).values
            rel_den += (cy ** 2).sum(axis=0).values
        self.sigma_rel = [float(x) for x in np.sqrt(rel_num / np.maximum(rel_den, 1e-9))]

    def sigma_for(self, sku: str, P: int, forecast_sum: float) -> float:
        H = self.cfg.max_horizon
        P = int(max(1, P))
        if sku in self.sigma:
            s = self.sigma[sku]
            return s[min(P, H) - 1] * (math.sqrt(P / H) if P > H else 1.0)
        rel = self.sigma_rel[min(P, H) - 1] if self.sigma_rel else 0.5
        return 1.5 * rel * forecast_sum  # unknown item: be more cautious


# ============================================================================ baselines + metrics
def _baselines(panel: pd.DataFrame, sub: pd.DataFrame) -> dict[str, np.ndarray]:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    out = {"naive (28d mean)": np.zeros(len(sub)), "seasonal naive (weekday)": np.zeros(len(sub)),
           "holt-winters": np.zeros(len(sub)), "croston-sba": np.zeros(len(sub))}
    series = {s: g.set_index("date")["demand"].sort_index() for s, g in panel.groupby("sku")}
    for (sku, origin), g in sub.groupby(["sku", "origin"]):
        y = series[sku][:origin]
        pos = sub.index.get_indexer(g.index)
        ks = g["k"].values
        tgt_dow = pd.DatetimeIndex(g["date"]).dayofweek.values
        out["naive (28d mean)"][pos] = y.iloc[-28:].mean()
        last8 = y.iloc[-56:]
        prof = last8.groupby(last8.index.dayofweek).mean()
        out["seasonal naive (weekday)"][pos] = prof.reindex(tgt_dow).fillna(last8.mean()).values
        hist = y.iloc[-365:]
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fit = ExponentialSmoothing(hist.values, trend="add", damped_trend=True, seasonal="add",
                                           seasonal_periods=7, initialization_method="estimated").fit()
                fc = np.clip(fit.forecast(int(ks.max())), 0, None)
            out["holt-winters"][pos] = fc[ks - 1]
        except Exception:  # noqa: BLE001
            out["holt-winters"][pos] = y.iloc[-28:].mean()
        out["croston-sba"][pos] = croston_sba(y.iloc[-365:].values)
    return out


def croston_sba(y: np.ndarray, alpha: float = 0.1) -> float:
    """Croston's method with the Syntetos-Boylan bias correction; returns the flat daily rate."""
    nz = np.where(y > 0)[0]
    if len(nz) == 0:
        return 0.0
    z, p, last = float(y[nz[0]]), float(nz[0] + 1), nz[0]
    for i in nz[1:]:
        z = alpha * y[i] + (1 - alpha) * z
        p = alpha * (i - last) + (1 - alpha) * p
        last = i
    return (1 - alpha / 2) * z / max(p, 1.0)


def _mase_scale(train: pd.DataFrame) -> dict[str, float]:
    """In-sample MAE of the weekly seasonal naive forecast, per item (MASE denominator)."""
    out = {}
    for sku, g in train.groupby("sku"):
        y = g.sort_values("date")["demand"].values
        out[sku] = float(np.mean(np.abs(y[7:] - y[:-7]))) if len(y) > 14 else float("nan")
    return out


def _score(sub: pd.DataFrame, pred: np.ndarray, horizons: tuple[int, ...], mase_scale: dict[str, float]) -> dict[str, float]:
    df = sub[["sku", "origin", "k", "y_sales", "y_stockout"]].copy()
    df["p"] = pred
    instock = ~df["y_stockout"].astype("boolean").fillna(False).to_numpy(bool)
    df["a"] = np.where(instock, df["y_sales"].fillna(0), 0.0)
    df["p_m"] = np.where(instock, df["p"], 0.0)
    res: dict[str, float] = {}
    for h in horizons:
        w = df[df["k"] <= h].groupby(["sku", "origin"])[["a", "p_m"]].sum()
        res[f"wape_{h}"] = round(float((w["p_m"] - w["a"]).abs().sum() / max(w["a"].sum(), 1e-9)), 4)
        res[f"bias_{h}"] = round(float((w["p_m"] - w["a"]).sum() / max(w["a"].sum(), 1e-9)), 4)
    d = df[instock]
    mase = []
    for sku, g in d.groupby("sku"):
        sc = mase_scale.get(sku)
        if sc and np.isfinite(sc) and sc > 0:
            mase.append(float(np.mean(np.abs(g["p"] - g["y_sales"].fillna(0)))) / sc)
    res["mase_daily"] = round(float(np.mean(mase)), 4) if mase else float("nan")
    return res


# ============================================================================ purchase plan
def purchase_plan(forecast: pd.DataFrame, data: WholesaleData, panel: pd.DataFrame, fc: DemandForecaster,
                  cfg: PlannerConfig) -> pd.DataFrame:
    z = NormalDist().inv_cdf(cfg.service_level)
    as_of = pd.Timestamp(data.as_of)
    items = data.items.set_index("sku")
    last_price = panel.sort_values("date").groupby("sku")["price"].last()
    on_hand = {}
    if data.stock is not None and len(data.stock):
        st = data.stock[data.stock["date"] < as_of].sort_values("date")
        on_hand = st.groupby("sku")["closing_qty"].last().to_dict()
    pos = data.open_pos if data.open_pos is not None else pd.DataFrame(columns=["sku", "qty", "expected_date"])
    rows = []
    for sku, g in forecast.groupby("sku"):
        if sku not in items.index:
            continue
        it = items.loc[sku]
        f = g.sort_values("k")["forecast"].values
        H = len(f)

        def D(P: int) -> float:
            P = int(P)
            if P <= H:
                return float(f[:P].sum())
            return float(f.sum() + f[-7:].mean() * (P - H))

        L = int(math.ceil(float(it["lead_time_days"])))
        sd_L = float(it["lead_time_sd"])
        R = int(cfg.review_days)
        pack = float(it["pack_size"] or 1.0)
        dbar = D(L) / max(L, 1)
        ss_L = z * math.sqrt(fc.sigma_for(sku, L, D(L)) ** 2 + (dbar * sd_L) ** 2)
        ss_LR = z * math.sqrt(fc.sigma_for(sku, L + R, D(L + R)) ** 2 + (dbar * sd_L) ** 2)
        rop = D(L) + ss_L
        up_to = D(L + R) + ss_LR
        stock_known = sku in on_hand
        oh = float(on_hand.get(sku, 0.0))
        p_sku = pos[pos["sku"] == sku]
        on_order = float(p_sku["qty"].sum())
        ip = oh + on_order
        order_qty, packs = 0.0, 0
        if stock_known and ip <= rop:
            packs = int(math.ceil(max(up_to - ip, 0) / pack))
            packs = max(packs, int(it.get("moq_packs", 1) or 1))
            cap = it.get("capacity_packs")
            if cap is not None and not pd.isna(cap):
                packs = int(min(packs, max(0, float(cap) - ip / pack)))
            order_qty = packs * pack
        # projected stockout date without a new order (open POs arrive on their expected date)
        arrivals = np.zeros(H)
        for _, p in p_sku.iterrows():
            k = 1 if pd.isna(p["expected_date"]) else int((pd.Timestamp(p["expected_date"]) - as_of).days) + 1
            arrivals[min(max(k, 1), H) - 1] += float(p["qty"])
        level = oh + np.cumsum(arrivals) - np.cumsum(f)
        out_k = int(np.argmax(level < 0)) + 1 if (level < 0).any() else None
        if not stock_known:
            # no stock snapshot for this item: show min/max levels for a manual check, order nothing
            status = "CHECK STOCK"
        elif ip <= rop:
            # a new PO placed today arrives after L days: running out sooner is urgent
            status = "URGENT" if (out_k is not None and out_k <= L) else "ORDER"
        elif out_k is not None and out_k <= L:
            # enough is on order, but it lands after the shelf runs empty: chase the supplier
            status = "EXPEDITE"
        else:
            status = "OK"
        next_order = None
        if stock_known and ip > rop:
            cross = np.where(ip - np.cumsum(f) <= rop)[0]
            next_order = (as_of + pd.Timedelta(days=int(cross[0]))).date().isoformat() if len(cross) else None
        price = float(last_price.get(sku, np.nan))
        cost_ratio = float(it["purchase_cost_ratio"]) if "purchase_cost_ratio" in it and not pd.isna(it.get("purchase_cost_ratio")) else 0.94
        rows.append({
            "sku": sku, "item_name": it["item_name"], "commodity": it["commodity"], "supplier": it.get("supplier", ""),
            "unit": it["base_unit"], "pack_unit": it.get("pack_unit", "Pack"), "pack_size": pack,
            "forecast_7d": round(D(7), 1), "forecast_15d": round(D(15), 1), "forecast_30d": round(D(30), 1),
            "lead_time_days": L, "lead_time_sd": sd_L, "lead_time_demand": round(D(L), 1),
            "safety_stock": round(ss_L, 1), "reorder_point": round(rop, 1), "order_up_to": round(up_to, 1),
            "on_hand": round(oh, 1) if stock_known else None, "on_order": round(on_order, 1),
            "inventory_position": round(ip, 1), "days_of_cover": round(oh / max(D(7) / 7, 1e-9), 1) if stock_known else None,
            "stockout_in_days": out_k if stock_known else None, "status": status, "order_qty": round(order_qty, 1), "order_packs": packs,
            "next_order_date": next_order,
            "est_po_value": round(order_qty * price * cost_ratio, 0) if order_qty and np.isfinite(price) else 0.0,
            "forecast_method": g["method"].iat[0],
        })
    df = pd.DataFrame(rows)
    if len(df):
        rank = {"URGENT": 0, "EXPEDITE": 1, "ORDER": 2, "CHECK STOCK": 3, "OK": 4}
        df = df.sort_values(["status", "est_po_value"], key=lambda s: s.map(rank) if s.name == "status" else -s)
    return df.reset_index(drop=True)
