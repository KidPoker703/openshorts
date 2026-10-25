import json
from types import SimpleNamespace

import pytest

import gemini_limits as guard


class Config:
    def __init__(self, **values): self.values = values
    def model_copy(self, update): return Config(**{**self.values, **update})


class Models:
    def __init__(self): self.calls = []
    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        usage = SimpleNamespace(prompt_token_count=3, cached_content_token_count=2,
            candidates_token_count=20, thoughts_token_count=3)
        return SimpleNamespace(usage_metadata=usage)


@pytest.fixture
def configured_guard(monkeypatch, tmp_path):
    monkeypatch.setenv("GUARDED_GEMINI_LIMITS_JSON", json.dumps(guard.REQUIRED))
    monkeypatch.setenv("GUARDED_GEMINI_RECEIPT_PATH", str(tmp_path / "receipt.json"))
    monkeypatch.setenv("GUARDED_GEMINI_RECEIPT_KEY", "s" * 64)
    return tmp_path


def test_guard_enforces_output_cap_receipt_and_one_attempt_per_stage(configured_guard):
    client = SimpleNamespace(models=Models())
    guard.generate_content(client, model="gemini-3.1-flash-lite", contents="bounded",
                           config=Config(), stage="score")
    assert client.models.calls[0]["config"].values["max_output_tokens"] == 2048
    receipt = guard.receipt()
    assert receipt["calls_reserved"] == receipt["calls_completed"] == 1
    assert receipt["stages"][0]["model"] == "gemini-3.1-flash-lite"
    assert receipt["cached_tokens"] == 2 and receipt["thinking_tokens"] == 3
    with pytest.raises(guard.GeminiLimitError, match="stage attempt"):
        guard.generate_content(client, model="gemini-3.1-flash-lite", contents="again",
                               config=Config(), stage="score")


def test_ambiguous_call_is_durable_and_blocks_restart(configured_guard):
    class Failing:
        def generate_content(self, **kwargs): raise TimeoutError("ambiguous")
    client = SimpleNamespace(models=Failing())
    with pytest.raises(TimeoutError):
        guard.generate_content(client, model="gemini-3.1-flash-lite", contents="x",
                               config=Config(), stage="detail")
    with pytest.raises(guard.GeminiLimitError, match="ambiguous"):
        guard.generate_content(client, model="gemini-3.1-flash-lite", contents="x",
                               config=Config(), stage="other")


def test_zero_sdk_retries_must_be_provable(configured_guard):
    class HttpOptions:
        def __init__(self, retry_options): self.retry_options = retry_options
    class Client:
        def __init__(self, **kwargs):
            assert kwargs["http_options"].retry_options is None
            self._api_client = SimpleNamespace(_http_options=kwargs["http_options"])
    result = guard.guarded_client(SimpleNamespace(Client=Client),
                                  SimpleNamespace(HttpOptions=HttpOptions), "not-a-real-key")
    assert result._api_client._http_options.retry_options is None


def test_missing_usage_multimodal_and_tampered_receipt_fail_closed(configured_guard):
    with pytest.raises(guard.GeminiLimitError, match="text-only"):
        guard.generate_content(SimpleNamespace(models=Models()), model="m",
                               contents=object(), config=Config(), stage="vision")
    path = configured_guard / "receipt.json"
    path.write_text('{"tampered":true}', encoding="utf-8")
    with pytest.raises(guard.GeminiLimitError, match="integrity"):
        guard.receipt()
