"""Indian demand calendar: festivals, Ramzan, wedding season, monsoon, harvest and sugar season.

Festival dates for lunar festivals move every year, so they are listed per year. They are
approximate (some festivals fall a day apart between states) and only cover 2023-2027: before
going live, replace FESTIVALS with the state government's holiday list for the trading region.

Each festival has a ``weight`` (rough relative impact on staples demand) and the items it lifts
most, so the model can learn item-specific festive uplift. Seasonal indices (wedding, monsoon,
harvest) are monthly priors; they are *features*, the model learns their actual effect from data.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Festival:
    name: str
    dates: tuple[str, ...]
    weight: float                      # 0..1 relative impact
    commodities: tuple[str, ...]       # rice / sugar / pulses / oil
    regions: tuple[str, ...] = ("all",)


FESTIVALS: tuple[Festival, ...] = (
    Festival("Makar Sankranti / Pongal", ("2023-01-15", "2024-01-15", "2025-01-14", "2026-01-14", "2027-01-15"),
             0.5, ("rice", "sugar", "pulses", "oil")),
    Festival("Holi", ("2023-03-08", "2024-03-25", "2025-03-14", "2026-03-04", "2027-03-22"),
             0.3, ("sugar", "oil"), ("north", "all")),
    Festival("Vishu", ("2023-04-15", "2024-04-14", "2025-04-14", "2026-04-15", "2027-04-15"),
             0.5, ("rice", "sugar", "oil"), ("kerala",)),
    Festival("Eid al-Fitr", ("2023-04-22", "2024-04-11", "2025-03-31", "2026-03-20", "2027-03-10"),
             0.8, ("rice", "sugar", "oil")),
    Festival("Eid al-Adha", ("2023-06-29", "2024-06-17", "2025-06-07", "2026-05-27", "2027-05-17"),
             0.6, ("rice", "oil")),
    Festival("Onam", ("2023-08-29", "2024-09-15", "2025-09-05", "2026-08-26", "2027-09-12"),
             1.0, ("rice", "sugar", "pulses", "oil"), ("kerala",)),
    Festival("Ganesh Chaturthi", ("2023-09-19", "2024-09-07", "2025-08-27", "2026-09-14", "2027-09-04"),
             0.4, ("sugar", "oil", "pulses"), ("west", "south", "all")),
    Festival("Dussehra / Navratri", ("2023-10-24", "2024-10-12", "2025-10-02", "2026-10-20", "2027-10-09"),
             0.6, ("sugar", "oil", "pulses")),
    Festival("Diwali", ("2023-11-12", "2024-10-31", "2025-10-20", "2026-11-08", "2027-10-29"),
             0.9, ("sugar", "oil", "pulses", "rice")),
    Festival("Christmas", ("2023-12-25", "2024-12-25", "2025-12-25", "2026-12-25", "2027-12-25"),
             0.6, ("sugar", "oil"), ("kerala", "all")),
)

# Monthly priors (Jan..Dec). Wedding index: Nov-Feb and Apr-May are peak muhurat months across most
# of India; Kerala adds Chingam (mid-Aug..mid-Sep) and avoids Karkidakam (mid-Jul..mid-Aug).
WEDDING_INDEX = {
    "all": (0.8, 0.9, 0.4, 0.7, 0.8, 0.5, 0.2, 0.3, 0.4, 0.4, 1.0, 0.9),
    "kerala": (0.7, 0.8, 0.4, 0.8, 0.8, 0.4, 0.15, 0.7, 0.9, 0.4, 0.8, 0.7),
}
# Southwest monsoon: onset over Kerala around 1 June, season June-September.
SW_MONSOON = (0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0)
# Kharif (rice) harvest arrivals Oct-Dec; rabi (wheat, chana) Mar-Apr; sugar crushing ~Nov-Apr.
KHARIF_HARVEST = (0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1)
RABI_HARVEST = (0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0)
SUGAR_CRUSHING = (1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1)

MAX_GAP = 60  # cap for days-to / days-since festival features


def festivals_for(region: str = "all") -> list[tuple[pd.Timestamp, Festival]]:
    region = (region or "all").lower()
    out = []
    for f in FESTIVALS:
        if "all" in f.regions or region in f.regions:
            out.extend((pd.Timestamp(d), f) for d in f.dates)
    return sorted(out, key=lambda x: x[0])


def _days_to_next(dates: pd.DatetimeIndex, events: np.ndarray) -> np.ndarray:
    """Days from each date to the next event (0 on the day), capped at MAX_GAP."""
    d = dates.values.astype("datetime64[D]").astype(np.int64)
    ev = np.sort(events.astype("datetime64[D]").astype(np.int64))
    idx = np.searchsorted(ev, d, side="left")
    nxt = np.where(idx < len(ev), ev[np.minimum(idx, len(ev) - 1)] - d, MAX_GAP)
    return np.minimum(nxt, MAX_GAP).astype(float)


def _days_since_last(dates: pd.DatetimeIndex, events: np.ndarray) -> np.ndarray:
    d = dates.values.astype("datetime64[D]").astype(np.int64)
    ev = np.sort(events.astype("datetime64[D]").astype(np.int64))
    idx = np.searchsorted(ev, d, side="right") - 1
    prev = np.where(idx >= 0, d - ev[np.maximum(idx, 0)], MAX_GAP)
    return np.minimum(prev, MAX_GAP).astype(float)


def calendar_features(dates: pd.DatetimeIndex | pd.Series, region: str = "kerala") -> pd.DataFrame:
    """One row per date with calendar regressors (all known in advance, so safe for forecasting)."""
    dates = pd.DatetimeIndex(pd.to_datetime(dates)).normalize()
    fest = festivals_for(region)
    major = np.array([d for d, f in fest if f.weight >= 0.6], dtype="datetime64[ns]")
    all_ev = np.array([d for d, _ in fest], dtype="datetime64[ns]")
    weight_by_day = {}
    for d, f in fest:
        weight_by_day[d] = max(weight_by_day.get(d, 0.0), f.weight)
    eid = np.array([d for d, f in fest if f.name == "Eid al-Fitr"], dtype="datetime64[ns]")
    month = dates.month.values - 1
    w_idx = WEDDING_INDEX.get(region, WEDDING_INDEX["all"])
    to_major = _days_to_next(dates, major) if len(major) else np.full(len(dates), MAX_GAP, float)
    # Ramzan: the ~29 days before Eid al-Fitr (iftar demand for hotels: rice, sugar, oil).
    to_eid = _days_to_next(dates, eid) if len(eid) else np.full(len(dates), MAX_GAP, float)
    # Festival "pull" decays with distance: buyers stock up 1-3 weeks ahead.
    weights = np.array([weight_by_day.get(d, 0.0) for d in dates])
    pull = np.zeros(len(dates))
    for d, f in fest:
        delta = (d - dates).days.values.astype(float)  # >0: festival ahead
        pull = np.maximum(pull, np.where((delta >= 0) & (delta <= 21), f.weight * np.exp(-delta / 8.0), 0.0))
    return pd.DataFrame({
        "dow": dates.dayofweek.values,
        "dom": dates.day.values,
        "month": dates.month.values,
        "weekofyear": dates.isocalendar().week.values.astype(int),
        "month_start_week": (dates.day.values <= 7).astype(int),  # kiranas restock after salary day
        "festival_weight": weights,
        "festival_pull": pull,
        "days_to_major_festival": to_major,
        "days_since_festival": _days_since_last(dates, all_ev) if len(all_ev) else MAX_GAP,
        "ramzan": ((to_eid > 0) & (to_eid <= 29)).astype(int),
        "wedding_index": np.array(w_idx)[month],
        "sw_monsoon": np.array(SW_MONSOON)[month],
        "kharif_harvest": np.array(KHARIF_HARVEST)[month],
        "rabi_harvest": np.array(RABI_HARVEST)[month],
        "sugar_crushing": np.array(SUGAR_CRUSHING)[month],
    }, index=dates)


CALENDAR_COLUMNS = list(calendar_features(pd.DatetimeIndex(["2025-01-01"])).columns)


def upcoming_events(as_of: pd.Timestamp, days: int = 30, region: str = "kerala") -> list[dict]:
    """Festivals in the next ``days`` days, for the report header."""
    as_of = pd.Timestamp(as_of).normalize()
    out = []
    for d, f in festivals_for(region):
        delta = (d - as_of).days
        if 0 <= delta <= days:
            out.append({"date": d.date().isoformat(), "festival": f.name, "in_days": int(delta),
                        "weight": f.weight, "commodities": list(f.commodities)})
    return out


def festival_multiplier(dates: pd.DatetimeIndex, commodity: str, region: str = "kerala") -> np.ndarray:
    """Ground-truth festive uplift used by the synthetic generator (not by the models)."""
    dates = pd.DatetimeIndex(dates)
    mult = np.ones(len(dates))
    for d, f in festivals_for(region):
        if commodity not in f.commodities:
            continue
        delta = (d - dates).days.values.astype(float)
        bump = np.where((delta >= 0) & (delta <= 14), 0.9 * f.weight * np.exp(-delta / 6.0), 0.0)
        bump += np.where((delta < 0) & (delta >= -3), -0.25 * f.weight, 0.0)  # post-festival lull
        mult = mult + bump
    return np.clip(mult, 0.5, None)
