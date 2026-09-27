"""The shared LLM client (etl/llm.py): formats, retries, validation, cache key, logging. No network."""

from __future__ import annotations

import json
import sqlite3

import httpx
import pytest

from etl import llm


def _openai_reply(text: str, rid: str = "chatcmpl-1"):
    return {"id": rid, "choices": [{"message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 11, "completion_tokens": 7}}


def _anthropic_reply(text: str, rid: str = "msg-1"):
    return {"id": rid, "content": [{"type": "thinking", "thinking": "..."}, {"type": "text", "text": text}],
            "usage": {"input_tokens": 5, "output_tokens": 3}, "stop_reason": "end_turn"}


def _client(handler, **overrides) -> llm.LLMClient:
    s = llm.LLMSettings(api=overrides.pop("api", "openai"), base_url="http://llm.test", model="test-model",
                        retry_base_delay=0.0, timeout=5.0, **overrides)
    return llm.LLMClient(s, transport=httpx.MockTransport(handler))


def test_openai_request_carries_seed_temperature_and_parses_reply():
    seen = {}

    def handler(request: httpx.Request):
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content)
        seen["auth"] = request.headers.get("authorization")
        return httpx.Response(200, json=_openai_reply('```json\n{"a": 1}\n```'))

    c = _client(handler, api_key="k", seed=42, temperature=0.0)
    res = c.call_json("hello")
    assert seen["url"] == "http://llm.test/v1/chat/completions"
    assert seen["body"]["seed"] == 42 and seen["body"]["temperature"] == 0.0 and seen["body"]["model"] == "test-model"
    assert seen["auth"] == "Bearer k"
    assert res.parsed == {"a": 1} and res.input_tokens == 11 and res.request_id == "chatcmpl-1" and res.attempts == 1


def test_anthropic_format_skips_thinking_blocks():
    seen = {}

    def handler(request: httpx.Request):
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content)
        seen["key"] = request.headers.get("x-api-key")
        return httpx.Response(200, json=_anthropic_reply('[1, 2]'))

    c = _client(handler, api="anthropic", api_key="k")
    res = c.call_json("hello")
    assert seen["url"] == "http://llm.test/v1/messages" and "seed" not in seen["body"] and seen["key"] == "k"
    assert res.parsed == [1, 2] and res.output_tokens == 3
    assert c.settings.fingerprint()["seed_applied"] is False


def test_retries_5xx_then_succeeds():
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        if calls["n"] < 3:
            return httpx.Response(503, text="busy")
        return httpx.Response(200, json=_openai_reply('{"ok": true}'))

    res = _client(handler).call_json("p")
    assert res.parsed == {"ok": True} and calls["n"] == 3 and res.attempts == 3


def test_client_error_is_not_retried():
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        return httpx.Response(400, text="bad request")

    with pytest.raises(llm.LLMError):
        _client(handler).call_json("p")
    assert calls["n"] == 1


def test_unparsable_then_valid_reply_and_repair_note():
    bodies = []

    def handler(request):
        bodies.append(json.loads(request.content)["messages"][0]["content"])
        if len(bodies) == 1:
            return httpx.Response(200, json=_openai_reply("Sure! Here you go: not json at all"))
        return httpx.Response(200, json=_openai_reply('{"x": 2}'))

    res = _client(handler).call_json("prompt A")
    assert res.parsed == {"x": 2} and res.attempts == 2
    assert bodies[0] == "prompt A" and bodies[1].startswith("prompt A") and "rejected" in bodies[1]


def test_validation_failure_is_retried_inside_the_loop():
    n = {"i": 0}

    def handler(request):
        n["i"] += 1
        return httpx.Response(200, json=_openai_reply(json.dumps({"primary_diagnosis": {}} if n["i"] == 1 else {"primary_diagnosis": {"name": "x"}, "clinical_findings": []})))

    def validate(p):
        return (isinstance(p, dict) and p.get("primary_diagnosis", {}).get("name") and "clinical_findings" in p) or "missing primary_diagnosis or clinical_findings"

    res = _client(handler).call_json("p", validate=validate)
    assert res.attempts == 2 and res.repairs == ["missing primary_diagnosis or clinical_findings"]


def test_attempts_exhausted_raises():
    def handler(request):
        return httpx.Response(200, json=_openai_reply("nope"))

    with pytest.raises(llm.LLMError, match="after 3 attempts"):
        _client(handler, max_attempts=3).call_json("p")


def test_cache_key_covers_the_whole_prompt_and_settings():
    c = _client(lambda r: httpx.Response(500))
    base = "x" * 300
    assert c.input_hash("s", base) == c.input_hash("s", base)
    assert c.input_hash("s", base) != c.input_hash("s", base + "y")            # a change beyond 200 characters
    assert c.input_hash("s", base) != c.input_hash("t", base)
    other = llm.LLMClient(llm.LLMSettings(base_url="http://llm.test", model="test-model", seed=1), transport=httpx.MockTransport(lambda r: httpx.Response(500)))
    assert c.input_hash("s", base) != other.input_hash("s", base)              # seed is part of the key


def test_log_call_records_sampling_settings_and_cache_lookup_round_trips():
    conn = sqlite3.connect(":memory:")
    conn.executescript("""CREATE TABLE llm_call_log (call_id INTEGER PRIMARY KEY AUTOINCREMENT, stage TEXT NOT NULL,
        model TEXT NOT NULL, prompt_template TEXT, input_tokens INTEGER, output_tokens INTEGER, cost_usd REAL,
        latency_ms INTEGER, request_id TEXT, input_hash TEXT, output_json TEXT, raw_response TEXT, error TEXT,
        created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')));""")
    c = _client(lambda r: httpx.Response(200, json=_openai_reply('{"v": 1}', rid="req-9")), seed=7)
    res = c.call_json("p")
    h = c.input_hash("stage_x", "p")
    llm.log_call(conn, "stage_x", h, res, prompt_template="t", client=c)
    row = conn.execute("select model, request_id, temperature, seed, attempts, endpoint, output_json from llm_call_log").fetchone()
    assert row == ("test-model", "req-9", 0.0, 7, 1, "http://llm.test/v1/chat/completions", '{"v": 1}')
    assert llm.cache_lookup(conn, "stage_x", h) == {"v": 1}
    assert llm.cache_lookup(conn, "stage_x", "other") is None
    llm.log_call(conn, "stage_x", "h2", None, error="boom", client=c)
    assert conn.execute("select error, attempts from llm_call_log where input_hash='h2'").fetchone() == ("boom", None)


def test_settings_from_env_legacy_gateway(monkeypatch):
    for k in ("SH_LLM_API", "SH_LLM_BASE_URL", "SH_LLM_GATEWAY_URL", "SH_LLM_MODEL"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("SH_LLM_GATEWAY_URL", "http://gw:8080/")
    s = llm.LLMSettings.from_env()
    assert s.api == "anthropic" and s.endpoint == "http://gw:8080/v1/messages"
    monkeypatch.setenv("SH_LLM_BASE_URL", "http://open:8000")
    monkeypatch.setenv("SH_LLM_MODEL", "m")
    monkeypatch.setenv("SH_LLM_TEMPERATURE", "0.2")
    s = llm.LLMSettings.from_env()
    assert s.api == "openai" and s.endpoint == "http://open:8000/v1/chat/completions" and s.model == "m" and s.temperature == 0.2
