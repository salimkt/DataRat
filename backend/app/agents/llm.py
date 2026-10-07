"""Thin wrapper around the Anthropic SDK that returns validated JSON.

The LLM is used only when ``ANTHROPIC_API_KEY`` is set (``ANTHROPIC_BASE_URL``
alone is not enough). Every caller must be ready to fall back to rules when
``complete_json`` raises.
"""
from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

log = logging.getLogger("datarat.llm")

DEFAULT_MODEL = "claude-sonnet-5-5"
_client = None


class LLMError(RuntimeError):
    pass


def llm_available() -> bool:
    if os.environ.get("DATARAT_DISABLE_LLM") == "1":
        return False
    return bool(os.environ.get("ANTHROPIC_API_KEY", "").strip())


def model_name() -> str:
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


def complete_json(system: str, user: str, max_tokens: int = 4000,
                  required_keys: tuple[str, ...] = ()) -> dict[str, Any]:
    """Call Claude and return a parsed JSON object. Raises LLMError on any failure."""
    if not llm_available():
        raise LLMError("LLM not configured (ANTHROPIC_API_KEY unset).")
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
    text = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
    data = extract_json(text)
    if not isinstance(data, dict):
        raise LLMError("Model returned JSON that is not an object.")
    missing = [k for k in required_keys if k not in data]
    if missing:
        raise LLMError(f"Model JSON missing keys: {missing}")
    return data
