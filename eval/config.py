"""Evaluation pipeline configuration: model registry, DB helper, cost rates."""

import os
from dataclasses import dataclass, field
from enum import Enum


# ---------------------------------------------------------------------------
# Adapter types
# ---------------------------------------------------------------------------

class AdapterType(str, Enum):
    OPENAI_COMPATIBLE = "openai_compatible"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"


# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    """Configuration for a single LLM model."""
    name: str
    adapter_type: AdapterType
    base_url: str
    api_key_env: str  # environment variable name holding the API key
    model_id: str     # model identifier for the API
    max_tokens: int = 16384
    temperature: float = 0.0
    timeout_secs: float = 120.0
    extra: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Model registry (10 models + 3 retrieval baselines)
# ---------------------------------------------------------------------------

# Self-hosted Anthropic-compatible LLM gateway used by the "kimi-2.5*" and "glm-5-agent"
# entries (the paper ran its generator and two agent models through one). Not part of
# the release: set SH_LLM_GATEWAY_URL to your own endpoint, or use the OpenRouter entries.
LLM_GATEWAY_URL = os.environ.get("SH_LLM_GATEWAY_URL", "http://localhost:8080")

OPENAI_URL = "https://api.openai.com/v1"
OPENROUTER_URL = "https://openrouter.ai/api/v1"


def _openrouter(name: str, model_id: str, **extra) -> "ModelConfig":
    return ModelConfig(name=name, adapter_type=AdapterType.OPENAI_COMPATIBLE, base_url=OPENROUTER_URL,
                       api_key_env="OPENROUTER_API_KEY", model_id=model_id, extra=extra)


def _openai(name: str, model_id: str) -> "ModelConfig":
    return ModelConfig(name=name, adapter_type=AdapterType.OPENAI_COMPATIBLE, base_url=OPENAI_URL,
                       api_key_env="OPENAI_API_KEY", model_id=model_id)


# Stage 8 protocol panel (EVAL_PROTOCOL.md §4): chosen on Artificial Analysis intelligence vs price
# (September 2026) among tool-calling models; Kimi K2.5 is the data generator and is never ranked.
# RL candidates: open weights small enough to train (dense ≤ 30B or a small MoE), tool calling, hosted on
# OpenRouter for a cheap hosted baseline; the paired before/after comparison uses our own serving of both
# the base and the trained weights (EVAL_PROTOCOL.md §5).
RL_CANDIDATES: dict[str, "ModelConfig"] = {
    "qwen3.5-9b": _openrouter("qwen3.5-9b", "qwen/qwen3.5-9b"),                 # 9B dense, Apache-2.0, $0.10/$0.15
    "qwen3.5-27b": _openrouter("qwen3.5-27b", "qwen/qwen3.5-27b"),              # 27B dense, $0.20/$1.56
    "muse-glimmer-30b": _openrouter("muse-glimmer-30b", "meta/muse-glimmer-30b"),   # 30B dense, Apache-2.0, agent-tuned, $0.30/$1.20
    "qwen3.5-35b-a3b": _openrouter("qwen3.5-35b-a3b", "qwen/qwen3.5-35b-a3b"),  # 35B MoE / 3B active, $0.31/$1.25
}

PROTOCOL_PANEL: dict[str, "ModelConfig"] = {
    "gpt-6-sol": _openai("gpt-6-sol", "gpt-6-sol"),                          # frontier anchor, $2/$10
    "gpt-6-luna": _openai("gpt-6-luna", "gpt-6-luna"),                       # value frontier, $0.10/$0.50
    "mimo-v2.6-pro": _openrouter("mimo-v2.6-pro", "xiaomi/mimo-v2.6-pro"),   # top open weights (AA 46), $0.43/$0.87
    "muse-spark-1.3": _openrouter("muse-spark-1.3", "meta/muse-spark-1.3-contributor"),   # Meta open weights (AA 48), contributor tier $0.10/$0.20
    "deepseek-v4-pro": _openrouter("deepseek-v4-pro", "deepseek/deepseek-v4-pro"),      # AA 44, $0.35/$0.70
    "glm-5.3-flash": _openrouter("glm-5.3-flash", "z-ai/glm-5.3-flash"),     # cheapest tool-capable, $0.04/$0.14
    "opus-5.5": _openrouter("opus-5.5", "anthropic/claude-opus-5.5"),        # AA #1 (58), $4/$20; optional anchor
    "kimi-k2.5": _openrouter("kimi-k2.5", "moonshotai/kimi-k2.5"),           # the generator: separate row
    # current frontier / value ids (September 2026) for later anchors
    "gpt-6-astra": _openrouter("gpt-6-astra", "openai/gpt-6-astra"),
    "sonnet-5": _openrouter("sonnet-5", "anthropic/claude-sonnet-5"),
    "fable-5.1": _openrouter("fable-5.1", "anthropic/claude-fable-5.1"),
    "gemini-3.8-flash": _openrouter("gemini-3.8-flash", "google/gemini-3.8-flash"),
    "deepseek-v4-flash": _openrouter("deepseek-v4-flash", "deepseek/deepseek-v4-flash"),
    "mimo-v2.6-flash": _openrouter("mimo-v2.6-flash", "xiaomi/mimo-v2.6-flash"),
}
PROTOCOL_PANEL.update(RL_CANDIDATES)

# Paper-era entries whose OpenRouter ids no longer resolve (kept for the record; `scripts/refresh_model_registry.py --check` reports them)
LEGACY_UNAVAILABLE = {"gpt-5.3", "opus-4.6"}

MODEL_REGISTRY: dict[str, ModelConfig] = {
    # --- Proprietary (via OpenRouter) ---
    "gpt-5.3": ModelConfig(
        name="gpt-5.3",
        adapter_type=AdapterType.OPENAI_COMPATIBLE,
        base_url="https://openrouter.ai/api/v1",
        api_key_env="OPENROUTER_API_KEY",
        model_id="openai/gpt-5.3-chat",
    ),
    "opus-4.6": ModelConfig(
        name="opus-4.6",
        adapter_type=AdapterType.OPENAI_COMPATIBLE,
        base_url="https://openrouter.ai/api/v1",
        api_key_env="OPENROUTER_API_KEY",
        model_id="anthropic/claude-opus-4-6",
    ),
    "gemini-3.1": ModelConfig(
        name="gemini-3.1",
        adapter_type=AdapterType.OPENAI_COMPATIBLE,
        base_url="https://openrouter.ai/api/v1",
        api_key_env="OPENROUTER_API_KEY",
        model_id="google/gemini-3.1-pro-preview",
    ),
    # --- Open source (via OpenRouter) ---
    "llama-4": ModelConfig(
        name="llama-4",
        adapter_type=AdapterType.OPENAI_COMPATIBLE,
        base_url="https://openrouter.ai/api/v1",
        api_key_env="OPENROUTER_API_KEY",
        model_id="meta-llama/llama-4-scout",
    ),
    "qwen-3": ModelConfig(
        name="qwen-3",
        adapter_type=AdapterType.OPENAI_COMPATIBLE,
        base_url="https://openrouter.ai/api/v1",
        api_key_env="OPENROUTER_API_KEY",
        model_id="qwen/qwen3.5-397b-a17b",
    ),
    "deepseek-v3.2": ModelConfig(
        name="deepseek-v3.2",
        adapter_type=AdapterType.OPENAI_COMPATIBLE,
        base_url="https://openrouter.ai/api/v1",
        api_key_env="OPENROUTER_API_KEY",
        model_id="deepseek/deepseek-v3.2",  # was -speciale (404 on OpenRouter; renamed)
        max_tokens=32768,  # reasoning model needs extra headroom
    ),
    "mistral-3": ModelConfig(
        name="mistral-3",
        adapter_type=AdapterType.OPENAI_COMPATIBLE,
        base_url="https://openrouter.ai/api/v1",
        api_key_env="OPENROUTER_API_KEY",
        model_id="mistralai/mistral-large-2512",
    ),
    "gemma-3": ModelConfig(
        name="gemma-3",
        adapter_type=AdapterType.OPENAI_COMPATIBLE,
        base_url="https://openrouter.ai/api/v1",
        api_key_env="OPENROUTER_API_KEY",
        model_id="google/gemma-3-27b-it",
        # Of the five providers serving this model, only DeepInfra supports function
        # calling; unpinned routing lands on tool-less providers and 429s, stalling
        # the agent loop. Pinning is required for the agentic arms.
        extra={"provider": {"only": ["DeepInfra"], "allow_fallbacks": False}},
    ),
    "glm-5": ModelConfig(
        name="glm-5",
        adapter_type=AdapterType.OPENAI_COMPATIBLE,
        base_url="https://openrouter.ai/api/v1",
        api_key_env="OPENROUTER_API_KEY",
        model_id="z-ai/glm-5",
        max_tokens=32768,  # verbose output needs extra headroom
    ),
    # --- Self-hosted Anthropic-compatible gateway (SH_LLM_GATEWAY_URL) ---
    # The paper served Kimi 2.5 and the GLM-5 agent through a private gateway; point
    # SH_LLM_GATEWAY_URL at your own, or use the OpenRouter entries above/below.
    "kimi-2.5": ModelConfig(
        name="kimi-2.5",
        adapter_type=AdapterType.ANTHROPIC,
        base_url=LLM_GATEWAY_URL,
        api_key_env="",
        model_id="kimi-k2.5",
        timeout_secs=360.0,
        extra={"disable_thinking": True},
    ),
    "kimi-2.5-thinking": ModelConfig(
        name="kimi-2.5-thinking",
        adapter_type=AdapterType.ANTHROPIC,
        base_url=LLM_GATEWAY_URL,
        api_key_env="",
        model_id="kimi-k2.5",
        timeout_secs=600.0,  # thinking is ~2-3x slower
        extra={},            # no disable_thinking → gateway auto-enables thinking
    ),
    # OpenRouter-served Kimi-2.5-thinking — fallback for the robustness study when the
    # self-hosted gateway is unavailable. Same base version (kimi-k2.5) via Moonshot's official
    # endpoint, thinking enabled via OpenRouter's `reasoning` param. Kept as a distinct
    # name so provenance (OpenRouter, not the gateway) is explicit in the results.
    "kimi-2.5-thinking-or": ModelConfig(
        name="kimi-2.5-thinking-or",
        adapter_type=AdapterType.OPENAI_COMPATIBLE,
        base_url="https://openrouter.ai/api/v1",
        api_key_env="OPENROUTER_API_KEY",
        model_id="moonshotai/kimi-k2.5",
        max_tokens=32768,     # thinking needs output headroom
        timeout_secs=600.0,   # thinking is ~2-3x slower
        extra={"reasoning": {"enabled": True}},
    ),
    # OpenRouter-served Kimi-2.5 (non-thinking) — the gateway-down fallback pair for
    # kimi-2.5-thinking-or. Reasoning disabled to mirror the gateway's disable_thinking.
    "kimi-2.5-or": ModelConfig(
        name="kimi-2.5-or",
        adapter_type=AdapterType.OPENAI_COMPATIBLE,
        base_url="https://openrouter.ai/api/v1",
        api_key_env="OPENROUTER_API_KEY",
        model_id="moonshotai/kimi-k2.5",
        timeout_secs=360.0,
        extra={"reasoning": {"enabled": False}},
    ),
    "kimi-2.5-improved": ModelConfig(
        name="kimi-2.5-improved",
        adapter_type=AdapterType.ANTHROPIC,
        base_url=LLM_GATEWAY_URL,
        api_key_env="",
        model_id="kimi-k2.5",
        timeout_secs=360.0,
        extra={"disable_thinking": True, "prompt_revision": "v1.31"},
    ),
    "kimi-2.5-thinking-improved": ModelConfig(
        name="kimi-2.5-thinking-improved",
        adapter_type=AdapterType.ANTHROPIC,
        base_url=LLM_GATEWAY_URL,
        api_key_env="",
        model_id="kimi-k2.5",
        timeout_secs=600.0,
        extra={"prompt_revision": "v1.31"},  # thinking auto-enabled (no disable_thinking)
    ),
    # --- GLM-5 agent (via the same self-hosted gateway) ---
    # The gateway used for the paper served one provider at a time (Kimi or GLM-5), and
    # for GLM-5 it served /v1/messages but returned OpenAI-format responses
    # when ZhipuAI/GLM is active. Tools must be passed in OpenAI format (not
    # Anthropic input_schema). The `gateway_openai_response` flag tells the
    # adapter to skip Anthropic→OpenAI tool conversion and parse OpenAI response.
    "glm-5-agent": ModelConfig(
        name="glm-5-agent",
        adapter_type=AdapterType.ANTHROPIC,
        base_url=LLM_GATEWAY_URL,
        api_key_env="",
        model_id="glm-5",
        timeout_secs=360.0,
        max_tokens=16384,
        extra={"gateway_openai_response": True},
    ),
}

# Retrieval baselines (not LLM models — handled separately in retrieval task)
RETRIEVAL_BASELINES = {"bm25", "sapbert", "hybrid"}


# ---------------------------------------------------------------------------
# Cost rates: USD per 1M tokens (input, output)
# ---------------------------------------------------------------------------

MODEL_REGISTRY.update(PROTOCOL_PANEL)


def _live_prices() -> dict[str, tuple[float, float]]:
    """eval/model_prices.json: $ per 1M (input, output) per registry name, refreshed by scripts/refresh_model_registry.py."""
    import json as _json
    from pathlib import Path as _Path
    p = _Path(__file__).with_name("model_prices.json")
    if not p.exists():
        return {}
    try:
        return {k: (float(v["input"]), float(v["output"])) for k, v in _json.loads(p.read_text()).get("models", {}).items()}
    except (ValueError, KeyError, TypeError):
        return {}


# $ per 1M tokens (input, output); the Stage-8 runner re-prices from OpenRouter's live list when it can
COST_RATES: dict[str, tuple[float, float]] = {
    "gpt-6-sol": (2.00, 10.00), "gpt-6-luna": (0.10, 0.50), "mimo-v2.6-pro": (0.43, 0.87),
    "muse-spark-1.3": (0.10, 0.20), "deepseek-v4-pro": (0.35, 0.70), "glm-5.3-flash": (0.04, 0.14),
    "opus-5.5": (4.00, 20.00), "kimi-k2.5": (0.45, 2.25),
    "qwen3.5-9b": (0.10, 0.15), "qwen3.5-27b": (0.20, 1.56), "muse-glimmer-30b": (0.30, 1.20), "qwen3.5-35b-a3b": (0.31, 1.25),
    "gpt-6-astra": (10.0, 50.0), "sonnet-5": (2.0, 10.0), "fable-5.1": (10.0, 50.0), "gemini-3.8-flash": (0.75, 3.75),
    "deepseek-v4-flash": (0.05, 0.09), "mimo-v2.6-flash": (0.14, 0.28),
    "gpt-5.3":       (1.75, 14.00),
    "opus-4.6":      (5.00, 25.00),
    "gemini-3.1":    (2.00, 12.00),
    "llama-4":       (0.11, 0.34),
    "qwen-3":        (0.55, 3.50),
    "deepseek-v3.2": (0.27, 0.41),
    "mistral-3":     (0.50, 1.50),
    "gemma-3":       (0.10, 0.30),
    "kimi-2.5":          (0.00, 0.00),
    "kimi-2.5-thinking": (0.00, 0.00),  # self-hosted, no cost
    "kimi-2.5-thinking-or": (0.375, 2.025),  # OpenRouter moonshotai/kimi-k2.5
    "kimi-2.5-or": (0.375, 2.025),  # OpenRouter moonshotai/kimi-k2.5 (non-thinking)
    "kimi-2.5-improved": (0.00, 0.00),  # self-hosted, revised prompts (v1.31)
    "kimi-2.5-thinking-improved": (0.00, 0.00),  # self-hosted, thinking + revised prompts
    "glm-5":             (0.80, 2.56),
    "glm-5-agent":       (0.00, 0.00),  # self-hosted gateway
}
COST_RATES.update(_live_prices())   # live OpenRouter prices win over the table when refreshed


def estimate_cost_usd(model_name: str, input_tokens: int, output_tokens: int) -> float:
    """Estimate cost in USD for a single API call."""
    rates = COST_RATES.get(model_name, (0.0, 0.0))
    return (input_tokens * rates[0] + output_tokens * rates[1]) / 1_000_000


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------

PG_URL = os.environ.get("DATABASE_URL", "postgresql://epic_sim:dev_password@localhost:5432/epic_sim")


def get_pg_connection():
    """Synchronous PostgreSQL connection for the evaluation pipeline."""
    import psycopg
    return psycopg.connect(PG_URL, autocommit=False)


# ---------------------------------------------------------------------------
# Evaluation tasks and strategies
# ---------------------------------------------------------------------------

EVAL_TASKS = [
    "patient_diagnosis",
    "context_summarization",
    "evidence_retrieval",
    "imaging_indication",
    # Stage 7 families (derived from the index-encounter diagnosis instances)
    "differential_diagnosis",
    "test_selection",
    "error_detection",
    "lab_triage",
    "atypical_diagnosis",
]

# Max output tokens per task (used for token budget calculations)
TASK_MAX_TOKENS: dict[str, int] = {
    "patient_diagnosis": 2000,
    "context_summarization": 1024,
    "evidence_retrieval": 512,
    "imaging_indication": 512,
    "differential_diagnosis": 512,
    "test_selection": 512,
    "error_detection": 512,
    "lab_triage": 512,
    "atypical_diagnosis": 2000,
}

# Context window sizes per model (for input token budget calculations)
MODEL_CONTEXT_WINDOWS: dict[str, int] = {
    "gpt-5.3": 128000,
    "opus-4.6": 200000,
    "gemini-3.1": 1000000,
    "llama-4": 128000,
    "qwen-3": 131072,
    "deepseek-v3.2": 128000,
    "mistral-3": 128000,
    "gemma-3": 128000,
    "glm-5": 128000,
    "kimi-2.5": 131072,
    "kimi-2.5-thinking": 131072,
    "kimi-2.5-thinking-or": 262144,
    "kimi-2.5-or": 262144,
    "kimi-2.5-improved": 131072,
    "kimi-2.5-thinking-improved": 131072,
    "glm-5-agent": 128000,
}

PROMPT_STRATEGIES = ["zero_shot", "few_shot", "cot", "structured"]


def get_api_key(model_name: str) -> str:
    """Resolve API key from environment for the given model."""
    cfg = MODEL_REGISTRY[model_name]
    if not cfg.api_key_env:
        return ""
    key = os.environ.get(cfg.api_key_env, "")
    if not key:
        raise ValueError(
            f"API key not set: export {cfg.api_key_env}=... for model {model_name}"
        )
    return key
