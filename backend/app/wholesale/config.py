"""Planner settings. Every knob has a business meaning; defaults suit a small staples wholesaler."""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class PlannerConfig:
    # ---- demand forecasting / purchase orders
    horizons: tuple[int, ...] = (7, 15, 30)  # days; report totals for each
    max_horizon: int = 30                    # the model forecasts day 1..max_horizon
    service_level: float = 0.95              # cycle service level -> z for safety stock
    review_days: int = 7                     # R: order cycle (how often POs are placed per item)
    origin_step_days: int = 3                # spacing of training forecast origins
    backtest_days: int = 84                  # length of the hold-out period for model comparison
    min_history_days: int = 14               # below this an item is treated as cold start
    run_baselines: bool = True               # also score naive / seasonal naive / Holt-Winters / SBA
    # ---- buyer reorder / churn
    reorder_window_days: int = 7             # "will order within the next N days"
    retention_window_days: int = 30          # used for churn / win-back risk
    snapshot_step_days: int = 7              # spacing of training snapshots
    active_lookback_days: int = 365          # pairs silent for longer are dropped as dormant
    interval_prior_weight: float = 3.0       # empirical-Bayes shrinkage of reorder intervals
    call_list_size: int = 40                 # rows in the daily outreach list
    reorder_threshold: float = 0.45          # p(order in window) to suggest a reminder
    winback_threshold: float = 0.6           # churn risk to suggest a win-back call
    call_value_threshold: float = 60_000.0   # Rs; kirana/bakery orders above this get a call, not WhatsApp
    # ---- model
    n_estimators: int = 400
    learning_rate: float = 0.05
    num_leaves: int = 31
    min_child_samples: int = 40
    seed: int = 7
    # ---- business
    business_name: str = "DataRat Wholesale (demo)"
    region: str = "kerala"
    currency: str = "Rs"

    @classmethod
    def fast(cls, **kw: Any) -> "PlannerConfig":
        """Small, quick settings for tests and the hosted demo."""
        base = dict(n_estimators=120, learning_rate=0.1, origin_step_days=5, backtest_days=56,
                    snapshot_step_days=7)
        base.update(kw)
        return cls(**base)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["horizons"] = list(self.horizons)
        return d


def data_dir() -> Path:
    """Where models, reports and the WhatsApp outbox are written (git-ignored by default)."""
    p = Path(os.environ.get("DATARAT_WHOLESALE_DIR", Path(__file__).resolve().parents[2] / "var" / "wholesale"))
    p.mkdir(parents=True, exist_ok=True)
    return p


@dataclass
class WhatsAppSettings:
    """Delivery mode for the outreach list.

    mode = "off"      nothing is written or sent
    mode = "dry_run"  (default) message payloads are written to an outbox JSON file only
    mode = "live"     POST to the WhatsApp Business Cloud API. Requires *all* of
                      DATARAT_WHATSAPP_MODE=live, DATARAT_WHATSAPP_TOKEN, DATARAT_WHATSAPP_PHONE_ID
                      and an approved template name. Template messages are billed per message by Meta.
    """
    mode: str = "dry_run"
    token: str | None = None
    phone_number_id: str | None = None
    template: str = "reorder_reminder"
    language: str = "en"
    api_version: str = "v21.0"
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_env(cls) -> "WhatsAppSettings":
        return cls(
            mode=os.environ.get("DATARAT_WHATSAPP_MODE", "dry_run").strip().lower() or "dry_run",
            token=os.environ.get("DATARAT_WHATSAPP_TOKEN") or None,
            phone_number_id=os.environ.get("DATARAT_WHATSAPP_PHONE_ID") or None,
            template=os.environ.get("DATARAT_WHATSAPP_TEMPLATE", "reorder_reminder"),
            language=os.environ.get("DATARAT_WHATSAPP_LANG", "en"),
            api_version=os.environ.get("DATARAT_WHATSAPP_API_VERSION", "v21.0"),
        )
