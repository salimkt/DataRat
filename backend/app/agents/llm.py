"""Thin LLM wrapper (Google Gemini or Anthropic Claude) that returns validated JSON.

Provider selection (``DATARAT_LLM_PROVIDER`` = auto | gemini | anthropic, default auto):
Gemini is used when ``GEMINI_API_KEY`` (or ``GOOGLE_API_KEY``) is set, otherwise Claude
when ``ANTHROPIC_API_KEY`` is set (``ANTHROPIC_BASE_URL`` alone is not enough).
Every caller must be ready to fall back to rules when ``complete_json`` raises.
"""
from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

log = logging.getLogger("datarat.llm")

DEFAULT_MODEL = "claude-sonnet-5-5"
DEFAULT_GEMINI_MODEL = "gemini-3.8-flash"
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
_client = None


class LLMError(RuntimeError):
    pass


def _gemini_key() -> str:
    return (os.environ.get("GEMINI_API_KEY", "") or os.environ.get("GOOGLE_API_KEY", "")).strip()


def provider() -> str | None:
    """The active LLM provider ("gemini" / "anthropic"), or None when no key is configured."""
    if os.environ.get("DATARAT_DISABLE_LLM") == "1":
        return None
    choice = os.environ.get("DATARAT_LLM_PROVIDER", "auto").strip().lower()
    has_gemini = bool(_gemini_key())
    has_anthropic = bool(os.environ.get("ANTHROPIC_API_KEY", "").strip())
    if choice == "gemini":
        return "gemini" if has_gemini else None
    if choice == "anthropic":
        return "anthropic" if has_anthropic else None
    if has_gemini:
        return "gemini"
    if has_anthropic:
        return "anthropic"
    return None


def llm_available() -> bool:
    return provider() is not None


def model_name() -> str:
    if provider() == "gemini":
        return os.environ.get("GEMINI_MODEL", "").strip() or DEFAULT_GEMINI_MODEL
    return os.environ.get("DATARAT_MODEL", "").strip() or DEFAULT_MODEL


def _get_client():
    global _client
    if _client is None:
        import anthropic

        # base_url is picked up from ANTHROPIC_BASE_URL by the SDK automatically
        _client = anthropic.Anthropic(
            api_key=os.environ["ANTHROPIC_API_KEY"].strip(),
            timeout=float(os.environ.get("DATARAT_LLM_TIMEOUT", "45")),
            max_retries=1,
        )
    return _client


def extract_json(text: str) -> Any:
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start = min([i for i in (text.find("{"), text.find("[")) if i >= 0], default=-1)
    if start < 0:
        raise LLMError("No JSON found in model output.")
    end = max(text.rfind("}"), text.rfind("]"))
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError as e:
        raise LLMError(f"Invalid JSON from model: {e}") from e


def _gemini_text(system: str, user: str, max_tokens: int) -> str:
    import httpx

    body: dict[str, Any] = {
        "system_instruction": {"parts": [{"text": system}]},
        "contents": [{"role": "user", "parts": [{"text": user}]}],
        "generationConfig": {
            "responseMimeType": "application/json",
            "temperature": 0.2,
            # thinking tokens count against the output budget, so leave headroom
            "maxOutputTokens": max(max_tokens * 2, 8192),
            "thinkingConfig": {"thinkingLevel": "low"},
        },
    }
    url = GEMINI_URL.format(model=model_name())
    headers = {"x-goog-api-key": _gemini_key(), "Content-Type": "application/json"}
    timeout = float(os.environ.get("DATARAT_LLM_TIMEOUT", "45"))
    try:
        resp = httpx.post(url, json=body, headers=headers, timeout=timeout)
        if resp.status_code == 400 and "thinking" in resp.text.lower():
            # older / non-thinking models reject thinkingConfig; retry without it
            body["generationConfig"].pop("thinkingConfig")
            resp = httpx.post(url, json=body, headers=headers, timeout=timeout)
    except httpx.HTTPError as e:
        raise LLMError(f"Gemini connection error: {e}") from e
    if resp.status_code != 200:
        try:
            msg = resp.json().get("error", {}).get("message", "")
        except ValueError:
            msg = resp.text[:200]
        raise LLMError(f"Gemini API error {resp.status_code}: {msg}")
    data = resp.json()
    cands = data.get("candidates") or []
    if not cands:
        reason = (data.get("promptFeedback") or {}).get("blockReason", "no candidates")
        raise LLMError(f"Gemini returned no answer ({reason}).")
    cand = cands[0]
    finish = cand.get("finishReason", "")
    if finish == "MAX_TOKENS":
        raise LLMError("Model output was truncated.")
    if finish in ("SAFETY", "RECITATION", "PROHIBITED_CONTENT", "BLOCKLIST", "SPII"):
        raise LLMError(f"Model declined the request ({finish}).")
    parts = (cand.get("content") or {}).get("parts") or []
    return "".join(p.get("text", "") for p in parts if not p.get("thought"))


def _anthropic_text(system: str, user: str, max_tokens: int) -> str:
    try:
        import anthropic
    except ImportError as e:  # pragma: no cover
        raise LLMError("anthropic package not installed") from e
    try:
        resp = _get_client().messages.create(
            model=model_name(),
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
            output_config={"effort": "low"},
        )
    except anthropic.APIStatusError as e:
        raise LLMError(f"Anthropic API error {e.status_code}: {e.message}") from e
    except anthropic.APIConnectionError as e:
        raise LLMError(f"Anthropic connection error: {e}") from e
    except Exception as e:  # TypeError on older SDKs, etc.
        raise LLMError(f"LLM call failed: {e}") from e
    if resp.stop_reason == "refusal":
        raise LLMError("Model declined the request.")
    if resp.stop_reason == "max_tokens":
        raise LLMError("Model output was truncated.")
    return "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")


def complete_json(system: str, user: str, max_tokens: int = 4000,
                  required_keys: tuple[str, ...] = ()) -> dict[str, Any]:
    """Call the configured LLM and return a parsed JSON object. Raises LLMError on any failure."""
    prov = provider()
    if prov is None:
        raise LLMError("LLM not configured (set GEMINI_API_KEY or ANTHROPIC_API_KEY).")
    if prov == "gemini":
        text = _gemini_text(system, user, max_tokens)
    else:
        text = _anthropic_text(system, user, max_tokens)
    data = extract_json(text)
    if not isinstance(data, dict):
        raise LLMError("Model returned JSON that is not an object.")
    missing = [k for k in required_keys if k not in data]
    if missing:
        raise LLMError(f"Model JSON missing keys: {missing}")
    return data
