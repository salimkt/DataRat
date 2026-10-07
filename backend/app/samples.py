"""Deterministic sample datasets (numpy seed 42)."""
from __future__ import annotations

import numpy as np
import pandas as pd

SAMPLE_META = {
    "sales": "Sales Orders (sample)",
    "marketing": "Marketing Performance (sample)",
    "support_calls": "Support Calls (sample)",
}

SPIKE_DATE = "2025-06-14"
DIP_START, DIP_END = "2025-09-01", "2025-09-09"


def generate(name: str) -> pd.DataFrame:
    if name == "sales":
        return gen_sales()
    if name == "marketing":
        return gen_marketing()
    if name == "support_calls":
        return gen_support_calls()
    raise ValueError(f"Unknown sample dataset '{name}'. Choose one of: {', '.join(SAMPLE_META)}")


def gen_sales() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    days = pd.date_range("2024-01-01", "2025-12-31", freq="D")
    n_days = len(days)
    t = np.arange(n_days) / n_days
    doy = days.dayofyear.values
    # growth trend + yearly seasonality (Q4 peak, summer lull) + weekday effect
    seasonal = 1 + 0.25 * np.cos(2 * np.pi * (doy - 340) / 365.25)
    weekday = np.where(days.dayofweek.values >= 5, 0.75, 1.08)
    lam = 6.2 * (0.85 + 0.35 * t) * seasonal * weekday
    counts = rng.poisson(lam)
    spike_i = int(np.where(days == pd.Timestamp(SPIKE_DATE))[0][0])
    counts[spike_i] += 38  # flash-sale spike
    dip_mask = (days >= pd.Timestamp(DIP_START)) & (days <= pd.Timestamp(DIP_END))
    counts[dip_mask] = 0  # checkout outage dip (no orders for 9 days)

    regions = {
        "North America": ["United States", "Canada", "Mexico"],
        "Europe": ["Germany", "United Kingdom", "France", "Spain"],
        "Asia Pacific": ["Japan", "Australia", "India", "Singapore"],
        "Latin America": ["Brazil", "Chile", "Colombia"],
        "Middle East & Africa": ["UAE", "South Africa", "Saudi Arabia"],
    }
    region_w = np.array([0.36, 0.27, 0.21, 0.09, 0.07])
    channels = ["Online", "Retail", "Partner", "Direct Sales"]
    channel_w = np.array([0.42, 0.26, 0.18, 0.14])
    catalog = {
        "Electronics": [("Laptop Pro 14", 1450, 0.72), ("Wireless Earbuds", 129, 0.55),
                        ("4K Monitor", 389, 0.68), ("Smartphone X", 899, 0.74)],
        "Furniture": [("Ergo Chair", 329, 0.58), ("Standing Desk", 549, 0.61),
                      ("Bookshelf", 159, 0.52)],
        "Office Supplies": [("Paper Pack", 24, 0.45), ("Gel Pens (12)", 12, 0.40),
                            ("Desk Organizer", 36, 0.48), ("Label Printer", 89, 0.57)],
        "Software": [("Analytics Suite", 499, 0.18), ("Security Plus", 199, 0.15),
                     ("Cloud Backup", 99, 0.22)],
    }
    cat_names = list(catalog)
    cat_w = np.array([0.34, 0.2, 0.28, 0.18])
    segments = ["Enterprise", "SMB", "Consumer"]
    seg_w = np.array([0.28, 0.37, 0.35])

    total = int(counts.sum())
    order_dates = np.repeat(days.values, counts)
    region_idx = rng.choice(len(regions), size=total, p=region_w)
    region_names = list(regions)
    cat_idx = rng.choice(len(cat_names), size=total, p=cat_w)
    spike_lo, spike_hi = int(counts[:spike_i].sum()), int(counts[: spike_i + 1].sum())
    rows = []
    disc_choices = np.array([0.0, 0.0, 0.05, 0.1, 0.15, 0.2])
    for i in range(total):
        region = region_names[region_idx[i]]
        country = regions[region][rng.integers(len(regions[region]))]
        cat = cat_names[cat_idx[i]]
        prod, base, cost_ratio = catalog[cat][rng.integers(len(catalog[cat]))]
        seg = segments[rng.choice(3, p=seg_w)]
        units = int(rng.integers(1, 6) if base > 300 else rng.integers(1, 16))
        if seg == "Enterprise":
            units = int(units * 2)
        unit_price = round(base * rng.uniform(0.92, 1.08), 2)
        discount = float(rng.choice(disc_choices))
        if spike_lo <= i < spike_hi:
            discount = 0.1
            units *= 2
        revenue = round(units * unit_price * (1 - discount), 2)
        cost = round(units * unit_price * cost_ratio, 2)
        rows.append((
            pd.Timestamp(order_dates[i]), region, country,
            channels[rng.choice(4, p=channel_w)], cat, prod, seg,
            units, unit_price, discount, revenue, cost, round(revenue - cost, 2),
        ))
    df = pd.DataFrame(rows, columns=[
        "order_date", "region", "country", "channel", "product_category", "product",
        "customer_segment", "units", "unit_price", "discount", "revenue", "cost", "profit",
    ])
    df.insert(0, "order_id", [f"ORD-{100000 + i}" for i in range(len(df))])
    return df


def gen_marketing() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    weeks = pd.date_range("2024-01-01", periods=105, freq="W-MON")
    channels = {
        # name: (weekly spend, cpm, ctr, lead rate, conversion rate (of clicks), revenue per conversion)
        "Paid Search": (14000, 38.0, 0.045, 0.11, 0.025, 160),
        "Paid Social": (11000, 12.0, 0.016, 0.07, 0.021, 140),
        "Display": (7000, 4.0, 0.006, 0.04, 0.009, 120),
        "Email": (2500, 8.0, 0.032, 0.16, 0.030, 150),
        "Affiliate": (5200, 18.0, 0.022, 0.09, 0.034, 155),
        "Video": (8800, 18.0, 0.009, 0.05, 0.014, 130),
        "Events": (6000, 60.0, 0.012, 0.21, 0.048, 380),
    }
    rows = []
    for wi, w in enumerate(weeks):
        season = 1 + 0.22 * np.cos(2 * np.pi * (w.dayofyear - 330) / 365.25)
        growth = 1 + 0.25 * wi / len(weeks)
        for ch, (sp, cpm, ctr, lr, cr, rpc) in channels.items():
            spend = sp * season * growth * rng.uniform(0.8, 1.2)
            impressions = int(spend / cpm * 1000 * rng.uniform(0.9, 1.1))
            clicks = int(impressions * ctr * rng.uniform(0.85, 1.15))
            leads = int(clicks * lr * rng.uniform(0.8, 1.2))
            conversions = int(clicks * cr * rng.uniform(0.75, 1.25))
            revenue = conversions * rpc * rng.uniform(0.85, 1.15)
            rows.append((w, ch, round(spend, 2), impressions, clicks, leads, conversions, round(revenue, 2)))
    return pd.DataFrame(rows, columns=["date", "channel", "spend", "impressions", "clicks",
                                       "leads", "conversions", "revenue_attributed"])


def gen_support_calls() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n = 3000
    start = pd.Timestamp("2025-01-01")
    day_offsets = np.sort(rng.integers(0, 365, size=n))
    seconds = rng.integers(8 * 3600, 19 * 3600, size=n)
    dates = start + pd.to_timedelta(day_offsets, unit="D") + pd.to_timedelta(seconds, unit="s")
    agents = ["Ava Thompson", "Liam Chen", "Sofia Garcia", "Noah Patel", "Mia Johnson",
              "Ethan Kim", "Isabella Rossi", "Lucas Silva", "Amelia Brown", "Mason Nguyen",
              "Harper Davis", "Elijah Wilson"]
    skill = rng.uniform(-0.25, 0.25, size=len(agents))
    queues = {
        "Billing": (["Billing dispute", "Refund request", "Invoice question"], 360, 0.12),
        "Technical Support": (["Outage", "Password reset", "Integration error", "Performance issue"], 620, 0.0),
        "Sales": (["Upgrade inquiry", "Pricing question", "Demo request"], 300, 0.2),
        "Account Management": (["Cancellation", "Contract renewal", "Plan change"], 480, -0.05),
        "Returns": (["Shipping delay", "Damaged item", "Return label"], 400, -0.1),
    }
    qnames = list(queues)
    qw = np.array([0.24, 0.32, 0.14, 0.13, 0.17])
    rows = []
    for i in range(n):
        ai = int(rng.integers(len(agents)))
        q = qnames[rng.choice(len(qnames), p=qw)]
        topics, base_dur, q_sent = queues[q]
        topic = topics[rng.integers(len(topics))]
        wait = float(rng.gamma(2.0, 45.0))
        if q == "Technical Support":
            wait *= 1.6
        duration = int(max(45, rng.normal(base_dur, base_dur * 0.35)))
        sentiment = float(np.clip(rng.normal(0.15 + q_sent + skill[ai] - wait / 900, 0.35), -1, 1))
        if topic == "Outage":
            sentiment = float(np.clip(sentiment - 0.3, -1, 1))
        p_res = 1 / (1 + np.exp(-(1.2 + 2.2 * sentiment)))
        resolved = bool(rng.random() < p_res)
        csat_raw = 3.1 + 1.6 * sentiment + (0.5 if resolved else -0.7) + rng.normal(0, 0.55)
        csat = int(np.clip(round(csat_raw), 1, 5))
        rows.append((f"CALL-{50000 + i}", dates[i], agents[ai], q, duration, int(wait),
                     round(sentiment, 3), resolved, csat, topic))
    return pd.DataFrame(rows, columns=["call_id", "call_date", "agent", "queue", "duration_sec",
                                       "wait_sec", "sentiment", "resolved", "csat", "topic"])
