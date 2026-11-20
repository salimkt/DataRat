"""WholesalePlanner: weekly ``fit`` (backtest + retrain), daily ``plan`` (inference + report).

    planner = WholesalePlanner(PlannerConfig())
    metrics = planner.fit(data)          # weekly, e.g. Sunday night
    planner.save("var/wholesale/model.joblib")
    ...
    planner = WholesalePlanner.load("var/wholesale/model.joblib")
    report = planner.plan(fresh_data)    # daily, before the shop opens
"""
from __future__ import annotations

import datetime as _dt
import time
from pathlib import Path
from typing import Any

import pandas as pd

from . import ingest, synthetic
from .config import PlannerConfig
from .demand import DemandForecaster, purchase_plan
from .features import daily_panel
from .ingest import WholesaleData
from .models import backend_name
from .reorder import ReorderModel, call_list
from .report import build_report


class WholesalePlanner:
    def __init__(self, cfg: PlannerConfig | None = None):
        self.cfg = cfg or PlannerConfig()
        self.demand = DemandForecaster(self.cfg)
        self.reorder = ReorderModel(self.cfg)
        self.metrics: dict[str, Any] = {}
        self.trained_at: str | None = None
        self.trained_through: str | None = None

    def fit(self, data: WholesaleData) -> dict[str, Any]:
        t0 = time.time()
        panel = daily_panel(data)
        demand_metrics = self.demand.backtest(panel)
        self.demand.fit(panel)
        reorder_metrics = self.reorder.backtest(data, panel)
        self.reorder.fit(data, panel)
        self.metrics = {"demand": demand_metrics, "reorder": reorder_metrics,
                        "train_seconds": round(time.time() - t0, 1)}
        self.trained_at = _dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
        self.trained_through = (pd.Timestamp(data.as_of) - pd.Timedelta(days=1)).date().isoformat()
        return self.metrics

    def plan(self, data: WholesaleData) -> dict[str, Any]:
        if self.demand.model is None or self.reorder.clf_reorder is None:
            raise RuntimeError("The planner is not trained yet; call fit() first.")
        panel = daily_panel(data)
        forecast = self.demand.predict(panel)
        po = purchase_plan(forecast, data, panel, self.demand, self.cfg)
        scores = self.reorder.score(data, panel)
        calls = call_list(scores, data, po, self.cfg)
        info = {"backend": backend_name(), "trained_at": self.trained_at, "trained_through": self.trained_through,
                "demand_top_features": self.demand.importance[:8], "reorder_top_features": self.reorder.importance[:8]}
        return build_report(data, panel, forecast, po, calls, self.metrics, self.cfg, info,
                            sigma=self.demand.sigma, sigma_rel=self.demand.sigma_rel)

    def save(self, path: str | Path) -> Path:
        import joblib

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        return path

    @staticmethod
    def load(path: str | Path) -> "WholesalePlanner":
        import joblib

        obj = joblib.load(path)
        if not isinstance(obj, WholesalePlanner):
            raise TypeError(f"{path} does not contain a WholesalePlanner.")
        return obj


# ============================================================================ file helpers
def _read_any(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in (".xlsx", ".xls"):
        return pd.read_excel(path)
    if path.suffix.lower() == ".xml":
        return ingest.parse_tally_xml(path.read_bytes())
    return pd.read_csv(path)


def load_dir(folder: str | Path, erp: str = "auto", as_of: str | None = None) -> WholesaleData:
    """Read an export folder: invoices.(csv|xlsx|xml) required; items, parties, stock, open_pos
    (csv or xlsx) optional. ground_truth.csv (synthetic only) is ignored."""
    folder = Path(folder)

    def find(stem: str, required: bool = False) -> pd.DataFrame | None:
        for ext in (".csv", ".xlsx", ".xls", ".xml"):
            p = folder / f"{stem}{ext}"
            if p.exists():
                return _read_any(p)
        if required:
            raise FileNotFoundError(f"{folder} has no {stem}.csv / .xlsx / .xml")
        return None

    return ingest.load(find("invoices", True), find("items"), find("parties"), find("stock"), find("open_pos"),
                       erp=erp, as_of=as_of)


def demo_data(seed: int = 7, n_buyers: int = 120, start: str = synthetic.DEFAULT_START,
              end: str = synthetic.DEFAULT_END, erp: str = "tally") -> tuple[synthetic.SyntheticWholesale, WholesaleData]:
    world = synthetic.generate(seed=seed, n_buyers=n_buyers, start=start, end=end, erp=erp)
    data = ingest.load(world.invoices, world.items, world.parties, world.stock, world.open_pos, erp=erp)
    return world, data
