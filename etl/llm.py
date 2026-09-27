"""The one LLM client of the ETL (ROADMAP Stage 6).

Every LLM-calling stage (1d, 4c, 5, 6, 7, 8, 9, 10) goes through `LLMClient`: one place for the endpoint,
the request format, sampling parameters, retries, the cache key and the `llm_call_log` row.

Configuration (environment):
    SH_LLM_API           openai | anthropic      request format (default openai: any OpenAI-compatible
                                                 server — vLLM, llama.cpp, Ollama, OpenRouter, ...)
    SH_LLM_BASE_URL      http://localhost:8000   server root; /v1/chat/completions or /v1/messages is appended
    SH_LLM_MODEL         model name as the server knows it
    SH_LLM_API_KEY       optional bearer key / x-api-key
    SH_LLM_TEMPERATURE   0.0                     sent on every request
    SH_LLM_SEED          0                       sent on every request (openai format; anthropic has none)
    SH_LLM_TOP_P         unset
    SH_LLM_MAX_TOKENS    4096
    SH_LLM_TIMEOUT       360                     seconds; grows by 60 s per attempt
    SH_LLM_MAX_ATTEMPTS  3                       total attempts per call (HTTP, timeout, parse, validation)
    SH_LLM_RETRY_DELAY   2.0                     base of the exponential backoff
    SH_LLM_GATEWAY_URL   legacy: selects the anthropic format at that URL when SH_LLM_BASE_URL is unset

Cache key: SHA-256 over (stage, model, temperature, seed, top_p, max_tokens, the complete prompt). A change
anywhere in the input is a new key. Retries after an unparsable or structurally invalid reply append a repair
note to the prompt; the key is computed by the caller from the original prompt, so the cache is unaffected.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable

import httpx

from etl.utils.logging import get_logger

log = get_logger("etl.llm")

Validator = Callable[[Any], Any]
"""validate(parsed) -> True/None when valid; False or a message string when not; may raise ValueError."""


# ---------------------------------------------------------------------------
# settings
# ---------------------------------------------------------------------------

def _env_float(name: str, default: float | None) -> float | None:
    v = os.environ.get(name)
    return float(v) if v not in (None, "") else default


def _env_int(name: str, default: int | None) -> int | None:
    v = os.environ.get(name)
    return int(v) if v not in (None, "") else default


@dataclass(frozen=True)
class LLMSettings:
    api: str = "openai"
    base_url: str = "http://localhost:8000"
    model: str = "kimi-k2.5"
    api_key: str | None = None
    temperature: float = 0.0
    seed: int | None = 0
    top_p: float | None = None
    max_tokens: int = 4096
    timeout: float = 360.0
    max_attempts: int = 3
    retry_base_delay: float = 2.0

    @classmethod
    def from_env(cls) -> "LLMSettings":
        legacy = os.environ.get("SH_LLM_GATEWAY_URL")
        base = os.environ.get("SH_LLM_BASE_URL") or legacy or cls.base_url
        api = os.environ.get("SH_LLM_API") or ("anthropic" if legacy and not os.environ.get("SH_LLM_BASE_URL") else "openai")
        if api not in ("openai", "anthropic"):
            raise ValueError(f"SH_LLM_API must be 'openai' or 'anthropic', not {api!r}")
        return cls(
            api=api, base_url=base.rstrip("/"), model=os.environ.get("SH_LLM_MODEL", cls.model),
            api_key=os.environ.get("SH_LLM_API_KEY") or None,
            temperature=_env_float("SH_LLM_TEMPERATURE", 0.0), seed=_env_int("SH_LLM_SEED", 0),
            top_p=_env_float("SH_LLM_TOP_P", None), max_tokens=_env_int("SH_LLM_MAX_TOKENS", 4096),
            timeout=_env_float("SH_LLM_TIMEOUT", 360.0), max_attempts=_env_int("SH_LLM_MAX_ATTEMPTS", 3),
            retry_base_delay=_env_float("SH_LLM_RETRY_DELAY", 2.0),
        )

    @property
    def endpoint(self) -> str:
        return f"{self.base_url}/v1/chat/completions" if self.api == "openai" else f"{self.base_url}/v1/messages"

    def fingerprint(self) -> dict:
        """What the manifest records about the generator (never the key)."""
        d = asdict(self)
        d.pop("api_key", None)
        d["endpoint"] = self.endpoint
        d["seed_applied"] = self.api == "openai" and self.seed is not None
        return d


@dataclass
class LLMResult:
    text: str
    parsed: Any
    input_tokens: int = 0
    output_tokens: int = 0
    request_id: str | None = None
    attempts: int = 1
    latency_ms: int = 0
    finish_reason: str | None = None
    repairs: list[str] = field(default_factory=list)


class LLMError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# parsing
# ---------------------------------------------------------------------------

def strip_fences(text: str) -> str:
    t = (text or "").strip()
    if t.startswith("```"):
        t = "\n".join(line for line in t.split("\n") if not line.strip().startswith("```"))
    return t.strip()


def parse_json(text: str) -> Any:
    """The reply as JSON: fenced or bare; the first {...} / [...] block when the model added prose."""
    t = strip_fences(text)
    try:
        return json.loads(t, strict=False)
    except json.JSONDecodeError:
        pass
    starts = [i for i in (t.find("{"), t.find("[")) if i >= 0]
    if not starts:
        raise ValueError("no JSON object or array in the reply")
    i = min(starts)
    closer = "}" if t[i] == "{" else "]"
    j = t.rfind(closer)
    if j <= i:
        raise ValueError("unterminated JSON in the reply")
    return json.loads(t[i:j + 1], strict=False)


def _check(validate: Validator | None, parsed: Any) -> str | None:
    """None when valid, else the reason."""
    if validate is None:
        return None
    try:
        r = validate(parsed)
    except ValueError as exc:
        return str(exc) or "structural validation failed"
    if r is None or r is True:
        return None
    if r is False:
        return "structural validation failed"
    return str(r)


# ---------------------------------------------------------------------------
# client
# ---------------------------------------------------------------------------

class LLMClient:
    def __init__(self, settings: LLMSettings | None = None, transport: httpx.BaseTransport | None = None):
        self.settings = settings or LLMSettings.from_env()
        self._transport = transport
        self.calls = 0

    # -- one request ---------------------------------------------------------
    def _body(self, prompt: str) -> dict:
        s = self.settings
        if s.api == "openai":
            body: dict = {"model": s.model, "messages": [{"role": "user", "content": prompt}],
                          "temperature": s.temperature, "max_tokens": s.max_tokens}
            if s.seed is not None:
                body["seed"] = s.seed
        else:
            body = {"model": s.model, "max_tokens": s.max_tokens, "temperature": s.temperature,
                    "messages": [{"role": "user", "content": prompt}]}
        if s.top_p is not None:
            body["top_p"] = s.top_p
        return body

    def _headers(self) -> dict:
        s = self.settings
        h = {"Content-Type": "application/json"}
        if s.api == "openai":
            if s.api_key:
                h["Authorization"] = f"Bearer {s.api_key}"
        else:
            h["anthropic-version"] = "2023-06-01"
            if s.api_key:
                h["x-api-key"] = s.api_key
        return h

    def complete(self, prompt: str, timeout: float | None = None) -> LLMResult:
        """One request. Raises httpx errors and ValueError (no text / bad JSON envelope)."""
        s = self.settings
        t0 = time.time()
        with httpx.Client(timeout=timeout or s.timeout, transport=self._transport) as client:
            resp = client.post(s.endpoint, json=self._body(prompt), headers=self._headers())
            resp.raise_for_status()
        self.calls += 1
        data = json.loads(resp.text, strict=False)
        if s.api == "openai":
            choices = data.get("choices") or []
            if not choices:
                raise ValueError("no choices in the reply")
            msg = choices[0].get("message") or {}
            text = msg.get("content") or ""
            usage = data.get("usage") or {}
            res = LLMResult(text=text, parsed=None, input_tokens=int(usage.get("prompt_tokens") or 0),
                            output_tokens=int(usage.get("completion_tokens") or 0), request_id=data.get("id"),
                            finish_reason=choices[0].get("finish_reason"))
        else:
            text = next((item.get("text", "") for item in data.get("content", []) if item.get("type") == "text"), "")
            usage = data.get("usage") or {}
            res = LLMResult(text=text, parsed=None, input_tokens=int(usage.get("input_tokens") or 0),
                            output_tokens=int(usage.get("output_tokens") or 0), request_id=data.get("id"),
                            finish_reason=data.get("stop_reason"))
        if not res.text:
            raise ValueError("no text block found in the reply")
        res.latency_ms = int((time.time() - t0) * 1000)
        return res

    # -- the retry loop ------------------------------------------------------
    def call_json(self, prompt: str, validate: Validator | None = None, max_attempts: int | None = None) -> LLMResult:
        """A JSON reply that passes `validate`, or LLMError after `max_attempts` total attempts.

        Retries: HTTP 429/5xx and transport errors (exponential backoff), timeouts (longer timeout),
        unparsable JSON and validation failures (a repair note is appended to the prompt).
        """
        s = self.settings
        attempts = max_attempts or s.max_attempts
        current = prompt
        repairs: list[str] = []
        last: Exception | None = None
        for attempt in range(1, attempts + 1):
            try:
                res = self.complete(current, timeout=s.timeout + (attempt - 1) * 60)
            except httpx.HTTPStatusError as exc:
                last = exc
                code = exc.response.status_code
                if code == 429 or code >= 500:
                    delay = s.retry_base_delay * (2 ** (attempt - 1))
                    log.warning("HTTP %s from %s, retrying in %.0fs (attempt %d/%d)", code, s.endpoint, delay, attempt, attempts)
                    if attempt < attempts:
                        time.sleep(delay)
                    continue
                raise LLMError(f"HTTP {code} from {s.endpoint}: {exc.response.text[:200]}") from exc
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last = exc
                log.warning("%s calling %s (attempt %d/%d)", type(exc).__name__, s.endpoint, attempt, attempts)
                if attempt < attempts:
                    time.sleep(s.retry_base_delay)
                continue
            except ValueError as exc:                       # bad envelope
                last = exc
                log.warning("bad reply envelope: %s (attempt %d/%d)", exc, attempt, attempts)
                continue
            try:
                parsed = parse_json(res.text)
                reason = _check(validate, parsed)
            except (json.JSONDecodeError, ValueError) as exc:
                parsed, reason = None, f"reply is not valid JSON: {exc}"
            if reason is None:
                res.parsed = parsed
                res.attempts = attempt
                res.repairs = repairs
                return res
            last = LLMError(reason)
            repairs.append(reason)
            log.warning("invalid reply (%s), retrying (attempt %d/%d)", reason, attempt, attempts)
            current = (f"{prompt}\n\n[Your previous reply was rejected: {reason}. "
                       f"Reply with valid JSON in exactly the requested structure and nothing else.]")
        raise LLMError(f"LLM call failed after {attempts} attempts: {last}") from last

    # -- cache key -------------------------------------------------------------
    def input_hash(self, stage: str, prompt: str) -> str:
        s = self.settings
        key = json.dumps({"stage": stage, "model": s.model, "temperature": s.temperature, "seed": s.seed,
                          "top_p": s.top_p, "max_tokens": s.max_tokens, "prompt": prompt}, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(key.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# llm_call_log helpers
# ---------------------------------------------------------------------------

LOG_COLUMNS = {"temperature": "REAL", "seed": "INTEGER", "attempts": "INTEGER", "endpoint": "TEXT"}


def ensure_log_columns(conn) -> None:
    have = {r[1] for r in conn.execute("PRAGMA table_info(llm_call_log)").fetchall()}
    if not have:
        return
    for col, typ in LOG_COLUMNS.items():
        if col not in have:
            conn.execute(f"ALTER TABLE llm_call_log ADD COLUMN {col} {typ}")
    conn.commit()


def cache_lookup(conn, stage: str, input_hash: str) -> Any | None:
    row = conn.execute(
        "SELECT output_json FROM llm_call_log WHERE stage = ? AND input_hash = ? AND error IS NULL LIMIT 1",
        (stage, input_hash)).fetchone()
    if row and row[0]:
        try:
            return json.loads(row[0], strict=False)
        except (json.JSONDecodeError, TypeError):
            return None
    return None


def log_call(conn, stage: str, input_hash: str, result: LLMResult | None, error: str | None = None,
             prompt_template: str | None = None, latency_ms: int | None = None, output_json: str | None = None,
             client: "LLMClient | None" = None) -> None:
    """One row per call, success or failure, with the sampling settings that produced it."""
    c = client or get_client()
    s = c.settings
    ensure_log_columns(conn)
    if result is not None and output_json is None:
        output_json = json.dumps(result.parsed, ensure_ascii=False)
    conn.execute(
        """INSERT INTO llm_call_log (stage, model, prompt_template, input_tokens, output_tokens, latency_ms,
                                     request_id, input_hash, output_json, raw_response, error,
                                     temperature, seed, attempts, endpoint)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (stage, s.model, prompt_template,
         result.input_tokens if result else 0, result.output_tokens if result else 0,
         latency_ms if latency_ms is not None else (result.latency_ms if result else None),
         result.request_id if result else None, input_hash,
         output_json if result else None, result.text if result else None, error,
         s.temperature, s.seed, result.attempts if result else None, s.endpoint),
    )


# ---------------------------------------------------------------------------
# module singleton
# ---------------------------------------------------------------------------

_CLIENT: LLMClient | None = None


def get_client() -> LLMClient:
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = LLMClient()
    return _CLIENT


def set_client(client: LLMClient | None) -> None:
    """Install a client (tests, or a process that configured one explicitly); None resets to the environment."""
    global _CLIENT
    _CLIENT = client


def reachable(settings: LLMSettings | None = None, timeout: float = 5.0) -> str | None:
    """None when the endpoint answers HTTP at all, else the reason (for the preflight)."""
    s = settings or LLMSettings.from_env()
    try:
        with httpx.Client(timeout=timeout) as c:
            c.get(s.base_url + ("/v1/models" if s.api == "openai" else "/"), headers=LLMClient(s)._headers())
        return None
    except httpx.HTTPError as exc:
        return f"{type(exc).__name__}: {exc}"
