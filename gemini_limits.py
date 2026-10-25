"""Fail-closed Gemini budget for controller-authorized guarded jobs.

Legacy jobs without GUARDED_GEMINI_LIMITS_JSON retain existing behavior.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import threading
from pathlib import Path

CONTRACT_VERSION = "openshorts-guarded-gemini-v1"
REQUIRED = {
    "contract_version": CONTRACT_VERSION,
    "max_calls": 8,
    "attempts_per_stage": 1,
    "sdk_retries": 0,
    "max_output_tokens_per_call": 2048,
    "max_total_output_tokens": 8192,
    "max_input_tokens_per_call": 262144,
    "max_total_input_tokens": 1048576,
    "auto_layout": False,
    "screencast_gemini": False,
    "hook_grounding": False,
    "vision_fallback": "fail_closed",
}
_lock = threading.RLock()


class GeminiLimitError(RuntimeError):
    pass


def configured():
    return bool(os.getenv("GUARDED_GEMINI_LIMITS_JSON"))


def limits():
    if not configured():
        return None
    try:
        value = json.loads(os.environ["GUARDED_GEMINI_LIMITS_JSON"])
    except (TypeError, ValueError) as exc:
        raise GeminiLimitError("guarded Gemini limits are malformed") from exc
    if value != REQUIRED:
        raise GeminiLimitError("guarded Gemini limits do not match the enforced contract")
    if not os.getenv("GUARDED_GEMINI_RECEIPT_PATH") or len(os.getenv("GUARDED_GEMINI_RECEIPT_KEY", "")) < 32:
        raise GeminiLimitError("guarded Gemini receipt configuration is incomplete")
    return value


def guarded_client(genai, types, api_key):
    """Construct the synchronous SDK client with no transport retry policy."""
    if not configured():
        return genai.Client(api_key=api_key)
    limits()
    options = types.HttpOptions(retry_options=None)
    client = genai.Client(api_key=api_key, http_options=options)
    actual = getattr(getattr(client, "_api_client", None), "_http_options", None)
    if actual is None or getattr(actual, "retry_options", "missing") is not None:
        raise GeminiLimitError("zero SDK retries cannot be proven")
    return client


def _path():
    return Path(os.environ["GUARDED_GEMINI_RECEIPT_PATH"])


def _sign(payload):
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hmac.new(os.environ["GUARDED_GEMINI_RECEIPT_KEY"].encode(), raw,
                    hashlib.sha256).hexdigest()


def _initial():
    return {"contract_version": CONTRACT_VERSION, "limits": REQUIRED,
            "calls_reserved": 0, "calls_completed": 0,
            "input_tokens": 0, "cached_tokens": 0,
            "output_tokens": 0, "thinking_tokens": 0,
            "reserved_output_tokens": 0, "in_flight": False,
            "stages": [], "status": "active"}


def _load():
    path = _path()
    if not path.exists():
        return _initial()
    value = json.loads(path.read_text(encoding="utf-8"))
    signature = value.pop("integrity_sha256", None)
    if not hmac.compare_digest(signature or "", _sign(value)):
        raise GeminiLimitError("Gemini receipt integrity failed")
    if value.get("contract_version") != CONTRACT_VERSION or value.get("limits") != REQUIRED:
        raise GeminiLimitError("Gemini receipt contract changed")
    return value


def _save(value):
    path = _path(); path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(value); payload["integrity_sha256"] = _sign(value)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as output:
        json.dump(payload, output, sort_keys=True, separators=(",", ":"))
        output.flush(); os.fsync(output.fileno())
    os.replace(temporary, path)


def _input_upper_bound(contents):
    """Conservative byte bound: every text token consumes at least one byte.

    Guarded jobs disable all multimodal paths, so bytes/Parts are prohibited.
    """
    def visit(value):
        if isinstance(value, str): return len(value.encode("utf-8"))
        if isinstance(value, (list, tuple)): return sum(visit(item) for item in value)
        if isinstance(value, dict): return len(json.dumps(value, ensure_ascii=False).encode("utf-8"))
        raise GeminiLimitError("guarded Gemini input must be text-only")
    return visit(contents)


def generate_content(client, *, model, contents, config, stage):
    """Reserve limits, make exactly one SDK call, and persist verified usage."""
    if not configured():
        return client.models.generate_content(model=model, contents=contents, config=config)
    cap = limits()
    upper = _input_upper_bound(contents)
    if upper > cap["max_input_tokens_per_call"]:
        raise GeminiLimitError("Gemini per-call input limit exceeded")
    with _lock:
        receipt = _load()
        if receipt["in_flight"]:
            raise GeminiLimitError("prior Gemini call outcome is ambiguous")
        if receipt["calls_reserved"] >= cap["max_calls"]:
            raise GeminiLimitError("Gemini call limit exhausted")
        if any(item.get("stage") == stage for item in receipt["stages"]):
            raise GeminiLimitError("Gemini stage attempt already consumed")
        if receipt["input_tokens"] + upper > cap["max_total_input_tokens"]:
            raise GeminiLimitError("Gemini total input limit exhausted")
        remaining = cap["max_total_output_tokens"] - receipt["reserved_output_tokens"]
        allowance = min(cap["max_output_tokens_per_call"], remaining)
        if allowance <= 0:
            raise GeminiLimitError("Gemini total output limit exhausted")
        if config is None or not hasattr(config, "model_copy"):
            raise GeminiLimitError("Gemini output limit cannot be applied")
        config = config.model_copy(update={"max_output_tokens": allowance})
        receipt["calls_reserved"] += 1
        receipt["reserved_output_tokens"] += allowance
        receipt["in_flight"] = True
        receipt["stages"].append({"stage": stage, "model": model, "state": "outcome_unknown",
                                  "input_upper_bound": upper,
                                  "output_tokens_reserved": allowance})
        _save(receipt)
    response = client.models.generate_content(model=model, contents=contents, config=config)
    usage = getattr(response, "usage_metadata", None)
    required = ("prompt_token_count", "candidates_token_count")
    if usage is None or any(getattr(usage, name, None) is None for name in required):
        raise GeminiLimitError("Gemini usage metadata is missing")
    input_tokens = int(usage.prompt_token_count or 0)
    cached = int(getattr(usage, "cached_content_token_count", 0) or 0)
    output = int(usage.candidates_token_count or 0)
    thinking = int(getattr(usage, "thoughts_token_count", 0) or 0)
    if min(input_tokens, cached, output, thinking) < 0 or input_tokens + cached > upper:
        raise GeminiLimitError("Gemini input usage exceeds its reservation")
    if output + thinking > allowance:
        raise GeminiLimitError("Gemini output usage exceeds its reservation")
    with _lock:
        receipt = _load()
        receipt["in_flight"] = False
        receipt["calls_completed"] += 1
        receipt["input_tokens"] += input_tokens
        receipt["cached_tokens"] += cached
        receipt["output_tokens"] += output
        receipt["thinking_tokens"] += thinking
        receipt["stages"][-1].update({"state": "completed", "input_tokens": input_tokens,
            "cached_tokens": cached, "output_tokens": output, "thinking_tokens": thinking})
        receipt["status"] = "complete" if receipt["calls_completed"] == cap["max_calls"] else "active"
        _save(receipt)
    return response


def receipt():
    if not configured(): return None
    value = _load()
    if value["in_flight"] or value["calls_reserved"] != value["calls_completed"]:
        raise GeminiLimitError("Gemini receipt has an ambiguous call")
    payload = dict(value); payload["integrity_sha256"] = _sign(value)
    return payload
