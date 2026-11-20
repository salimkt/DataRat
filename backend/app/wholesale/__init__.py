"""Wholesale planner: demand forecasting, purchase-order planning and buyer reorder prediction
for a B2B staples trader (rice, sugar, pulses, cooking oil).

Modules
-------
config       PlannerConfig (horizons, service level, call-list size, ...) and env settings
seasonality  Indian festival / wedding / monsoon / harvest calendar features
synthetic    deterministic synthetic ERP export generator (no real data needed)
ingest       raw Tally / Marg / Vyapar / canonical invoice exports -> clean canonical lines
features     daily demand panel (with stockout unconstraining) and buyer-item RFM features
models       gradient-boosting backend (LightGBM if it loads, else scikit-learn HistGradientBoosting)
demand       global multi-horizon demand forecaster, backtest vs baselines, PO quantities
reorder      buyer x item reorder (7 d) and retention (30 d) classifiers, next-best-action list
report       unified daily report (dict / JSON) and the formatted Excel workbook
notify       WhatsApp delivery stub (dry-run outbox by default; never sends unless configured)
pipeline     WholesalePlanner: fit (weekly) / plan (daily) / save / load
api          FastAPI routes mounted under /api/wholesale
"""
