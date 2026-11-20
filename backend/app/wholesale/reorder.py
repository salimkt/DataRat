"""Buyer x item reorder prediction and the daily next-best-action (outreach) list.

Two classifiers share the RFM / interval features from ``features.buyer_snapshots``:
  p_reorder   P(buyer orders this item within the next 7 days)      -> "reorder reminder"
  p_retain    P(buyer orders this item within the next 30 days)     -> churn risk = 1 - p_retain

Training uses weekly snapshots of history; evaluation uses *daily* snapshots of a hold-out
period, because the list is produced every day. Labels never overlap the hold-out: a training
snapshot's label window ends before the first test day.

Baseline for comparison: the classic rule "days since last order / usual interval" (due ratio).
"""
from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd

from .config import PlannerConfig
from .features import BUYER_FEATURES, buyer_snapshots, interval_priors, _orders
from .ingest import WholesaleData
from .models import backend_name, feature_importance, fit_quiet, make_classifier

SEGMENTS = ["bakery", "caterer", "hotel", "kirana", "unknown"]


class ReorderModel:
    def __init__(self, cfg: PlannerConfig):
        self.cfg = cfg
        self.clf_reorder: Any = None
        self.clf_retain: Any = None
        self.priors: dict = {}
        self.commodity_categories: list[str] = []
        self.metrics: dict[str, Any] = {}
        self.importance: list[dict[str, Any]] = []
        self.backend = backend_name()

    # ------------------------------------------------------------------ helpers
    def _snap(self, data: WholesaleData, panel: pd.DataFrame, cutoffs, labels=()) -> pd.DataFrame:
        return buyer_snapshots(
            data, cutoffs, panel, self.priors, label_windows=labels, lookback=self.cfg.active_lookback_days,
            prior_weight=self.cfg.interval_prior_weight, region=self.cfg.region,
            window=self.cfg.reorder_window_days, segment_categories=SEGMENTS,
            commodity_categories=self.commodity_categories)

    @staticmethod
    def _proba(clf: Any, X: pd.DataFrame) -> np.ndarray:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return clf.predict_proba(X)[:, 1]

    def _train(self, data: WholesaleData, panel: pd.DataFrame, end: pd.Timestamp) -> None:
        """Fit both classifiers on weekly snapshots whose label windows end on or before ``end``."""
        cfg = self.cfg
        start = data.lines["date"].min() + pd.Timedelta(days=56)
        h7, h30 = cfg.reorder_window_days, cfg.retention_window_days
        cut7 = list(pd.date_range(start, end - pd.Timedelta(days=h7), freq=f"{cfg.snapshot_step_days}D"))
        cut30 = [c for c in cut7 if c <= end - pd.Timedelta(days=h30)]
        snap = self._snap(data, panel, cut7, labels=(h7, h30))
        if snap.empty or snap[f"y_{h7}"].nunique() < 2:
            raise ValueError("Not enough buyer history to train the reorder model.")
        self.clf_reorder = fit_quiet(make_classifier(cfg), snap[BUYER_FEATURES], snap[f"y_{h7}"])
        s30 = snap[snap["cutoff"].isin(cut30)]
        if len(s30) and s30[f"y_{h30}"].nunique() == 2:
            self.clf_retain = fit_quiet(make_classifier(cfg), s30[BUYER_FEATURES], s30[f"y_{h30}"])
        else:
            self.clf_retain = None
        self.importance = feature_importance(self.clf_reorder, BUYER_FEATURES)

    # ------------------------------------------------------------------ public API
    def backtest(self, data: WholesaleData, panel: pd.DataFrame) -> dict[str, Any]:
        cfg = self.cfg
        h7, h30 = cfg.reorder_window_days, cfg.retention_window_days
        end = pd.Timestamp(data.as_of) - pd.Timedelta(days=1)
        test_start = end - pd.Timedelta(days=cfg.backtest_days)
        self.commodity_categories = sorted(data.items["commodity"].unique())
        orders = _orders(data.lines)
        self.priors = interval_priors(orders, data.parties, data.items, until=test_start)
        self._train(data, panel, test_start)
        test_cut = list(pd.date_range(test_start + pd.Timedelta(days=1), end - pd.Timedelta(days=h7), freq="D"))
        snap = self._snap(data, panel, test_cut, labels=(h7, h30))
        y = snap[f"y_{h7}"].values
        p = self._proba(self.clf_reorder, snap[BUYER_FEATURES])
        # rule baseline: rank by due ratio, but treat pairs lapsed > 2x their interval as dormant
        rule = np.where(snap["due_ratio"].values <= 2.0, snap["due_ratio"].values, 0.0)
        k = cfg.call_list_size
        from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
        res = {
            "backend": self.backend, "test_from": test_cut[0].date().isoformat(),
            "test_to": test_cut[-1].date().isoformat(), "test_days": len(test_cut),
            "rows": int(len(snap)), "base_rate": round(float(y.mean()), 4), "k": k,
            "models": [
                {"model": "gbm classifier", "roc_auc": round(float(roc_auc_score(y, p)), 4),
                 "pr_auc": round(float(average_precision_score(y, p)), 4),
                 "brier": round(float(brier_score_loss(y, p)), 4), **_at_k(snap, p, y, k)},
                {"model": "rule: due ratio (<= 2x)", "roc_auc": round(float(roc_auc_score(y, rule)), 4),
                 "pr_auc": round(float(average_precision_score(y, rule)), 4), "brier": None, **_at_k(snap, rule, y, k)},
            ],
        }
        # churn / retention head: evaluated where a 30-day label exists, on regular buyers
        if self.clf_retain is not None:
            s30 = snap[(snap["cutoff"] <= end - pd.Timedelta(days=h30)) & (snap["n_orders"] >= 4)]
            if len(s30) and s30[f"y_{h30}"].nunique() == 2:
                churn = 1 - s30[f"y_{h30}"].values
                risk = 1 - self._proba(self.clf_retain, s30[BUYER_FEATURES])
                res["churn"] = {"rows": int(len(s30)), "base_rate": round(float(churn.mean()), 4),
                                "roc_auc": round(float(roc_auc_score(churn, risk)), 4),
                                "pr_auc": round(float(average_precision_score(churn, risk)), 4),
                                "rule_pr_auc": round(float(average_precision_score(churn, s30["due_ratio"].values)), 4),
                                "rule": "due ratio (higher = more overdue)"}
        # cold start slice: pairs with fewer than 3 orders at the cutoff
        cold = snap["n_orders"] < 3
        if cold.sum() > 50 and len(np.unique(y[cold])) == 2:
            res["cold_start"] = {"rows": int(cold.sum()), "pr_auc": round(float(average_precision_score(y[cold], p[cold])), 4),
                                 "rule_pr_auc": round(float(average_precision_score(y[cold], rule[cold])), 4)}
        self.metrics = res
        return res

    def fit(self, data: WholesaleData, panel: pd.DataFrame) -> "ReorderModel":
        end = pd.Timestamp(data.as_of) - pd.Timedelta(days=1)
        self.commodity_categories = sorted(data.items["commodity"].unique())
        self.priors = interval_priors(_orders(data.lines), data.parties, data.items)
        self._train(data, panel, end)
        return self

    def score(self, data: WholesaleData, panel: pd.DataFrame) -> pd.DataFrame:
        """Scores for every active buyer x item pair as of the last data day."""
        end = pd.Timestamp(data.as_of) - pd.Timedelta(days=1)
        snap = self._snap(data, panel, [end])
        if snap.empty:
            return snap
        snap["p_reorder"] = self._proba(self.clf_reorder, snap[BUYER_FEATURES])
        snap["p_retain"] = self._proba(self.clf_retain, snap[BUYER_FEATURES]) if self.clf_retain is not None else np.nan
        snap["churn_risk"] = 1 - snap["p_retain"]
        return snap


def _at_k(snap: pd.DataFrame, score: np.ndarray, y: np.ndarray, k: int) -> dict[str, float]:
    """Mean precision / recall of the top-k pairs per day (the daily call list budget)."""
    df = pd.DataFrame({"cutoff": snap["cutoff"].values, "s": score, "y": y})
    prec, rec = [], []
    for _, g in df.groupby("cutoff"):
        top = g.nlargest(k, "s")
        prec.append(top["y"].mean())
        rec.append(top["y"].sum() / max(g["y"].sum(), 1))
    return {f"precision_at_{k}": round(float(np.mean(prec)), 4), f"recall_at_{k}": round(float(np.mean(rec)), 4)}


# ============================================================================ next-best-action
def call_list(scores: pd.DataFrame, data: WholesaleData, plan: pd.DataFrame, cfg: PlannerConfig) -> pd.DataFrame:
    """One row per buyer: what to say, how to reach them and why, ranked by expected value."""
    if scores is None or scores.empty:
        return pd.DataFrame()
    s = scores.copy()
    items = data.items.set_index("sku")
    parties = data.parties.set_index("buyer_id")
    s["exp_qty"] = s["typical_qty"]
    s["exp_value"] = s["typical_qty"] * s["last_rate"]
    s["pack_size"] = s["sku"].map(items["pack_size"]).fillna(1.0)
    s["pack_unit"] = s["sku"].map(items["pack_unit"]).fillna("Pack")
    s["item_name"] = s["sku"].map(items["item_name"]).fillna(s["sku"])
    s["reorder_score"] = s["p_reorder"] * s["exp_value"]
    regular = s["n_orders"] >= 4
    overdue = s["recency"] > np.maximum(1.5 * s["mean_int_shrunk"], 14)
    s["is_reorder"] = s["p_reorder"] >= cfg.reorder_threshold
    s["is_winback"] = regular & overdue & (s["churn_risk"].fillna(0) >= cfg.winback_threshold) & ~s["is_reorder"]
    s["winback_score"] = s["churn_risk"].fillna(0) * s["amount_365"] / 12.0
    low_stock = set(plan.loc[plan["status"] == "URGENT", "sku"]) if len(plan) else set()
    on_hand = plan.set_index("sku")["on_hand"].to_dict() if len(plan) else {}
    rows = []
    for buyer, g in s.groupby("buyer_id"):
        re = g[g["is_reorder"]].sort_values("reorder_score", ascending=False)
        wb = g[g["is_winback"]].sort_values("winback_score", ascending=False)
        if re.empty and wb.empty:
            continue
        party = parties.loc[buyer] if buyer in parties.index else pd.Series(dtype=object)
        name = party.get("buyer_name", buyer) if len(party) else buyer
        segment = g["segment"].iat[0]
        if not re.empty:
            top = re.head(4)
            action = "Reorder reminder"
            priority = float(top["reorder_score"].sum())
            value = float(top["exp_value"].sum())
            parts, why, checks = [], [], []
            for _, r in top.iterrows():
                packs = max(1, int(round(r["exp_qty"] / r["pack_size"])))
                parts.append(f"{r['item_name']} ~{packs} {r['pack_unit']}")
                why.append(f"{r['item_name']}: usually every {r['mean_int_shrunk']:.0f} d, last {r['recency']:.0f} d ago (p={r['p_reorder']:.2f})")
                oh = on_hand.get(r["sku"])
                if r["sku"] in low_stock or (oh is not None and not pd.isna(oh) and oh < r["exp_qty"]):
                    checks.append(r["item_name"])
            items_txt = "; ".join(parts)
            p_max = float(top["p_reorder"].max())
        else:
            top = wb.head(3)
            action = "Win-back call"
            priority = float(top["winback_score"].sum())
            value = float(top["amount_365"].sum() / 12.0)
            items_txt = "; ".join(top["item_name"])
            why = [f"{r['item_name']}: usually every {r['mean_int_shrunk']:.0f} d, silent for {r['recency']:.0f} d (churn risk {r['churn_risk']:.2f})"
                   for _, r in top.iterrows()]
            checks = []
            p_max = float(top["p_reorder"].max())
        opt_in = bool(party.get("whatsapp_opt_in", False)) if len(party) else False
        channel = "Call"
        if action == "Reorder reminder" and opt_in and segment in ("kirana", "bakery", "unknown") and value < cfg.call_value_threshold:
            channel = "WhatsApp"
        rows.append({
            "buyer_id": buyer, "buyer_name": name, "segment": segment,
            "area": party.get("area", "") if len(party) else "", "phone": party.get("phone", "") if len(party) else "",
            "action": action, "channel": channel, "items": items_txt, "expected_value": round(value, 0),
            "max_probability": round(p_max, 3), "priority_score": round(priority, 1),
            "why": " | ".join(why), "check_stock": ", ".join(checks),
            "message": _message(cfg, name, action, items_txt),
        })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = out.sort_values("priority_score", ascending=False).head(cfg.call_list_size).reset_index(drop=True)
    out.insert(0, "rank", np.arange(1, len(out) + 1))
    return out


def _message(cfg: PlannerConfig, name: str, action: str, items: str) -> str:
    if action == "Reorder reminder":
        return (f"Hello {name}, this is {cfg.business_name}. Time to restock? Suggested: {items}. "
                f"Reply with your quantities and we will deliver at today's rate.")
    return (f"Hello {name}, this is {cfg.business_name}. We have not seen your order for {items} for a while. "
            f"Is there anything we can do better? Reply here to place your next order.")
