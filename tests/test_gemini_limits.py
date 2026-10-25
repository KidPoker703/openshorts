import json
import socket
import threading
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
    monkeypatch.setenv("GUARDED_GEMINI_JOB_IDENTITY", "a" * 64)
    monkeypatch.setenv("GUARDED_GEMINI_ENGINE_JOB_ID", "engine-job-1")
    monkeypatch.setenv("GUARDED_GEMINI_MODEL", "gemini-3.1-flash-lite")
    return tmp_path


def test_guard_enforces_output_cap_receipt_and_one_attempt_per_stage(configured_guard):
    client = SimpleNamespace(models=Models())
    guard.generate_content(client, model="gemini-3.1-flash-lite", contents="bounded",
                           config=Config(), stage_id="score-001")
    assert client.models.calls[0]["config"].values["max_output_tokens"] == 2048
    receipt = guard.receipt()
    assert receipt["calls_reserved"] == receipt["calls_completed"] == 1
    assert receipt["stages"][0]["model"] == "gemini-3.1-flash-lite"
    assert receipt["job_identity"] == "a" * 64
    assert receipt["reserved_input_tokens"] == guard.REQUIRED["max_input_tokens_per_call"]
    assert receipt["cached_tokens"] == 2 and receipt["thinking_tokens"] == 3
    with pytest.raises(guard.GeminiLimitError, match="stage attempt"):
        guard.generate_content(client, model="gemini-3.1-flash-lite", contents="again",
                               config=Config(), stage_id="score-001")


def test_ambiguous_call_is_durable_and_blocks_restart(configured_guard):
    class Failing:
        def generate_content(self, **kwargs): raise TimeoutError("ambiguous")
    client = SimpleNamespace(models=Failing())
    with pytest.raises(TimeoutError):
        guard.generate_content(client, model="gemini-3.1-flash-lite", contents="x",
                               config=Config(), stage_id="detail-001")
    with pytest.raises(guard.GeminiLimitError, match="ambiguous"):
        guard.generate_content(client, model="gemini-3.1-flash-lite", contents="x",
                               config=Config(), stage_id="other-001")


def test_zero_sdk_retries_must_be_provable(configured_guard):
    class HttpOptions:
        def __init__(self, retry_options): self.retry_options = retry_options
    class Client:
        def __init__(self, **kwargs):
            assert kwargs["http_options"].retry_options is None
            self._api_client = SimpleNamespace(_http_options=kwargs["http_options"],
                _retry=SimpleNamespace(stop=SimpleNamespace(max_attempt_number=1)))
    result = guard.guarded_client(SimpleNamespace(Client=Client),
                                  SimpleNamespace(HttpOptions=HttpOptions), "not-a-real-key")
    assert result._api_client._http_options.retry_options is None


def test_missing_usage_multimodal_and_tampered_receipt_fail_closed(configured_guard):
    with pytest.raises(guard.GeminiLimitError, match="text-only"):
        guard.generate_content(SimpleNamespace(models=Models()), model="m",
                               contents=object(), config=Config(), stage_id="vision-001")
    path = configured_guard / "receipt.json"
    path.write_text('{"tampered":true}', encoding="utf-8")
    with pytest.raises(guard.GeminiLimitError, match="integrity"):
        guard.receipt()


def test_total_input_reservations_exhaust_before_fifth_call(configured_guard):
    client = SimpleNamespace(models=Models())
    for index in range(4):
        guard.generate_content(client, model="gemini-3.1-flash-lite", contents="bounded",
            config=Config(), stage_id=f"stage-{index}")
    with pytest.raises(guard.GeminiLimitError, match="total input"):
        guard.generate_content(client, model="gemini-3.1-flash-lite", contents="bounded",
            config=Config(), stage_id="stage-4")


def test_existing_interprocess_lock_fails_closed(configured_guard):
    lock = configured_guard / "receipt.json.lock"; lock.write_text("other process")
    with pytest.raises(guard.GeminiLimitError, match="locked"):
        guard.generate_content(SimpleNamespace(models=Models()),
            model="gemini-3.1-flash-lite", contents="bounded",
            config=Config(), stage_id="score-001")


def test_installed_sdk_sync_transport_makes_exactly_one_http_attempt(configured_guard):
    genai = pytest.importorskip("google.genai")
    types = pytest.importorskip("google.genai.types")
    server = socket.socket(); server.bind(("127.0.0.1", 0)); server.listen(4)
    attempts = []
    def serve():
        connection, _ = server.accept()
        attempts.append(connection.recv(65536)); connection.close(); server.close()
    thread = threading.Thread(target=serve, daemon=True); thread.start()
    base_url = f"http://127.0.0.1:{server.getsockname()[1]}"
    factory = SimpleNamespace(Client=lambda api_key, http_options: genai.Client(
        api_key=api_key, http_options=types.HttpOptions(
            base_url=base_url, retry_options=http_options.retry_options)))
    client = guard.guarded_client(factory, types, "offline-test")
    assert client._api_client._retry.stop.max_attempt_number == 1
    with pytest.raises(Exception):
        client.models.generate_content(model="gemini-3.1-flash-lite", contents="offline")
    thread.join(timeout=2)
    assert len(attempts) == 1
