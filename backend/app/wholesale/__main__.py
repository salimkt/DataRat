"""Command line: ``python -m app.wholesale <command>`` (run from backend/).

  demo      generate synthetic exports, train, plan, write the Excel/JSON report and WhatsApp outbox
  generate  write synthetic ERP exports to a folder (tally | vyapar | marg | canonical)
  train     weekly job: backtest + retrain on an export folder, save the model
  report    daily job: load the model, plan from the latest exports, write the report (+ outbox)

Examples
  python -m app.wholesale demo
  python -m app.wholesale generate --out var/wholesale/data --erp tally
  python -m app.wholesale train  --data var/wholesale/data
  python -m app.wholesale report --data var/wholesale/data --notify
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from . import notify
from .config import PlannerConfig, data_dir
from .pipeline import WholesalePlanner, demo_data, load_dir
from .report import write_excel


def _cfg(args: argparse.Namespace) -> PlannerConfig:
    return PlannerConfig.fast() if getattr(args, "fast", False) else PlannerConfig()


def _print_metrics(m: dict) -> None:
    d, r = m.get("demand", {}), m.get("reorder", {})
    if d.get("models"):
        print(f"\nDemand backtest ({d['split_date']} .. {d['end_date']}, {d['test_origins']} weekly origins, "
              f"{d['items']} items, {d['evaluated_on']}):")
        print(pd.DataFrame(d["models"]).to_string(index=False))
    if r.get("models"):
        print(f"\nBuyer reorder backtest ({r['test_from']} .. {r['test_to']}, {r['test_days']} daily lists, "
              f"{r['rows']} buyer-item rows, base rate {r['base_rate']:.1%}):")
        print(pd.DataFrame(r["models"]).to_string(index=False))
        if "churn" in r:
            c = r["churn"]
            print(f"Churn head (no order in 30 d, regular buyers): ROC-AUC {c['roc_auc']}, PR-AUC {c['pr_auc']} "
                  f"(rule {c['rule_pr_auc']}), base rate {c['base_rate']:.1%}")
        if "cold_start" in r:
            c = r["cold_start"]
            print(f"Cold-start pairs (<3 orders): PR-AUC {c['pr_auc']} (rule {c['rule_pr_auc']}), n={c['rows']}")
    print(f"\nTraining time: {m.get('train_seconds')} s")


def _write_outputs(report: dict, out: Path, do_notify: bool) -> dict:
    as_of = report["summary"]["as_of"]
    xlsx = out / f"report_{as_of}.xlsx"
    js = out / f"report_{as_of}.json"
    write_excel(report, xlsx)
    js.write_text(json.dumps(report, indent=1, ensure_ascii=False))
    paths = {"excel": str(xlsx), "json": str(js)}
    if do_notify:
        paths["whatsapp"] = notify.deliver(report, out)
    return paths


def _summary(report: dict) -> None:
    s = report["summary"]
    print(f"\n{report['title']}")
    for h in report["highlights"]:
        print(f"  - {h}")
    po = pd.DataFrame(report["po_plan"])
    if len(po):
        cols = ["status", "item_name", "order_packs", "pack_unit", "on_hand", "reorder_point", "order_up_to",
                "forecast_7d", "forecast_30d", "stockout_in_days"]
        print("\nPurchase-order plan:")
        print(po[cols].to_string(index=False))
    cl = pd.DataFrame(report["call_list"])
    if len(cl):
        print(f"\nOutreach list (top 10 of {len(cl)}):")
        print(cl[["rank", "buyer_name", "action", "channel", "expected_value", "items"]].head(10).to_string(index=False))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m app.wholesale", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("demo", help="synthetic end-to-end run")
    d.add_argument("--out", default=None)
    d.add_argument("--seed", type=int, default=7)
    d.add_argument("--buyers", type=int, default=120)
    d.add_argument("--fast", action="store_true", help="smaller, quicker model settings")
    g = sub.add_parser("generate", help="write synthetic ERP exports")
    g.add_argument("--out", required=True)
    g.add_argument("--erp", default="tally", choices=["tally", "vyapar", "marg", "canonical"])
    g.add_argument("--format", default="csv", choices=["csv", "xlsx"])
    g.add_argument("--seed", type=int, default=7)
    g.add_argument("--buyers", type=int, default=120)
    t = sub.add_parser("train", help="weekly retrain")
    t.add_argument("--data", required=True)
    t.add_argument("--erp", default="auto")
    t.add_argument("--model", default=None)
    t.add_argument("--fast", action="store_true")
    r = sub.add_parser("report", help="daily plan")
    r.add_argument("--data", required=True)
    r.add_argument("--erp", default="auto")
    r.add_argument("--model", default=None)
    r.add_argument("--out", default=None)
    r.add_argument("--notify", action="store_true", help="WhatsApp outbox (dry run unless configured live)")
    args = ap.parse_args(argv)

    base = data_dir()
    if args.cmd == "generate":
        from . import synthetic
        world = synthetic.generate(seed=args.seed, n_buyers=args.buyers, erp=args.erp)
        paths = world.write(args.out, args.format)
        print(json.dumps({"files": paths, "meta": world.meta}, indent=1))
        return 0
    if args.cmd == "demo":
        out = Path(args.out) if args.out else base / "demo"
        world, data = demo_data(seed=args.seed, n_buyers=args.buyers)
        files = world.write(out / "data")
        print(f"Synthetic exports written to {out / 'data'} ({world.meta['lines_billed']} billed lines, "
              f"lost-sales share {world.meta['lost_sales_share']:.1%})")
        print("Ingest quality:", json.dumps({k: v for k, v in data.quality.items() if k != 'unknown_items'}))
        planner = WholesalePlanner(_cfg(args))
        metrics = planner.fit(data)
        planner.save(out / "model.joblib")
        _print_metrics(metrics)
        report = planner.plan(data)
        paths = _write_outputs(report, out, True)
        (out / "metrics.json").write_text(json.dumps(metrics, indent=1))
        _summary(report)
        print("\nFiles:", json.dumps({**paths, "model": str(out / "model.joblib"), "inputs": files}, indent=1))
        return 0
    if args.cmd == "train":
        data = load_dir(args.data, args.erp)
        planner = WholesalePlanner(_cfg(args))
        metrics = planner.fit(data)
        path = planner.save(args.model or base / "model.joblib")
        _print_metrics(metrics)
        print(f"\nModel saved to {path}")
        return 0
    if args.cmd == "report":
        data = load_dir(args.data, args.erp)
        planner = WholesalePlanner.load(args.model or base / "model.joblib")
        report = planner.plan(data)
        out = Path(args.out) if args.out else base
        paths = _write_outputs(report, out, args.notify)
        _summary(report)
        print("\nFiles:", json.dumps(paths, indent=1))
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
