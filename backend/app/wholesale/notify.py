"""WhatsApp delivery of the outreach list: a stub by default.

* ``dry_run`` (default): builds WhatsApp Cloud API template payloads for opted-in buyers on the
  WhatsApp channel and writes them to ``outbox_<date>.json``. Nothing leaves the machine.
* ``off``: does nothing.
* ``live``: posts each payload to the Cloud API. Only when DATARAT_WHATSAPP_MODE=live *and* a
  token and phone-number id are configured. Business-initiated messages must use a template that
  Meta has approved, and Meta bills template messages per delivered message, so keep this off
  until the template, opt-ins and budget are in place. The tests never exercise this path.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .config import WhatsAppSettings


class WhatsAppConfigError(RuntimeError):
    pass


def build_payloads(call_list: list[dict[str, Any]], settings: WhatsAppSettings,
                   opted_in: set[str] | None = None) -> list[dict[str, Any]]:
    out = []
    for row in call_list:
        if row.get("channel") != "WhatsApp":
            continue
        if opted_in is not None and row.get("buyer_id") not in opted_in:
            continue
        to = re.sub(r"\D", "", str(row.get("phone", "")))
        if not to:
            continue
        out.append({
            "buyer_id": row.get("buyer_id"),
            "preview": row.get("message"),
            "request": {
                "messaging_product": "whatsapp", "to": to, "type": "template",
                "template": {
                    "name": settings.template, "language": {"code": settings.language},
                    "components": [{"type": "body", "parameters": [
                        {"type": "text", "text": str(row.get("buyer_name", ""))[:60]},
                        {"type": "text", "text": str(row.get("items", ""))[:500]},
                    ]}],
                },
            },
        })
    return out


def deliver(report: dict[str, Any], out_dir: str | Path, settings: WhatsAppSettings | None = None,
            opted_in: set[str] | None = None) -> dict[str, Any]:
    settings = settings or WhatsAppSettings.from_env()
    if settings.mode == "off":
        return {"mode": "off", "messages": 0}
    payloads = build_payloads(report.get("call_list", []), settings, opted_in)
    if settings.mode == "dry_run":
        path = Path(out_dir) / f"outbox_{report['summary']['as_of']}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payloads, indent=2, ensure_ascii=False))
        return {"mode": "dry_run", "messages": len(payloads), "outbox": str(path)}
    if settings.mode == "live":
        return _send_live(payloads, settings)
    raise WhatsAppConfigError(f"Unknown DATARAT_WHATSAPP_MODE '{settings.mode}' (use off, dry_run or live).")


def _send_live(payloads: list[dict[str, Any]], settings: WhatsAppSettings) -> dict[str, Any]:  # pragma: no cover
    if not (settings.token and settings.phone_number_id):
        raise WhatsAppConfigError("Live WhatsApp needs DATARAT_WHATSAPP_TOKEN and DATARAT_WHATSAPP_PHONE_ID.")
    import httpx

    url = f"https://graph.facebook.com/{settings.api_version}/{settings.phone_number_id}/messages"
    headers = {"Authorization": f"Bearer {settings.token}"}
    sent, errors = 0, []
    with httpx.Client(timeout=20.0) as client:
        for p in payloads:
            r = client.post(url, headers=headers, json=p["request"])
            if r.status_code < 300:
                sent += 1
            else:
                errors.append({"buyer_id": p["buyer_id"], "status": r.status_code, "body": r.text[:300]})
    return {"mode": "live", "messages": len(payloads), "sent": sent, "errors": errors}
