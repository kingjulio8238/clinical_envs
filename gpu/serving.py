"""The before/after serving configuration (P3), free of Modal so it can be tested anywhere.

Both evaluations run on ART's managed vLLM runtime — the engine that samples the policy during ART training (vLLM
0.25.1 + ART's patches, with native LoRA for Qwen3.5's linear-attention modules). One server starts on the base; a
trained adapter is loaded into it at run time under `RL_SERVED_NAME`, so the two runs differ only in the model name
their requests carry. gpu/common.py launches it; gpu/vllm_eval.py runs the client against it.
"""
from __future__ import annotations

MODEL = "Qwen/Qwen3.5-9B"
RL_SERVED_NAME = "qwen3.5-9b-rl"
"""The trained policy's name on the server (base + LoRA); eval/config.py `qwen3.5-9b-rl` requests it."""
ENGINE_ARGS = {"max_model_len": 65536, "gpu_memory_utilization": 0.90, "max_num_seqs": 128,
               "limit_mm_per_prompt": {"image": 0, "video": 0},
               "generation_config": "vllm",   # as ART's training server: unset sampling fields take vLLM's defaults
               # off by default for this hybrid model; on, every turn reuses the cached chart history instead of
               # re-reading it (82% prefix-cache hits, prompt compute -69%, generation +42%: g1 vs g1-prefix, 2026-10-01)
               "enable_prefix_caching": True}
SERVER_ARGS = {"enable_auto_tool_choice": True, "tool_call_parser": "qwen3_coder", "reasoning_parser": "qwen3"}


def launch_config(lora_rank: int = 16, engine_extra: dict | None = None) -> dict:
    """Arguments of ART's runtime server (`art.vllm_runtime.VllmRuntimeLaunchConfig`): the base, LoRA enabled (the
    runtime always enables it) with room for one adapter of `lora_rank`."""
    engine = dict(ENGINE_ARGS, max_loras=1, max_lora_rank=max(16, int(lora_rank)), **(engine_extra or {}))
    return {"base_model": MODEL, "port": 8000, "host": "127.0.0.1", "cuda_visible_devices": "0", "lora_path": None,
            "served_model_name": MODEL, "engine_args": engine, "server_args": dict(SERVER_ARGS)}


def check_served(models: list[dict], lora_path: str | None) -> str:
    """The model name the client must request; raises unless the server serves exactly what was asked for."""
    ids = [m.get("id") for m in models]
    if MODEL not in ids:
        raise RuntimeError(f"base model {MODEL} not served: {ids}")
    if not lora_path:
        return MODEL
    if not any(m.get("id") == RL_SERVED_NAME and str(m.get("root", "")).rstrip("/") == lora_path.rstrip("/") for m in models):
        raise RuntimeError(f"adapter {lora_path} not served as {RL_SERVED_NAME}: {models}")
    return RL_SERVED_NAME


PROBE_MESSAGES = [{"role": "user", "content": "Fever, neck stiffness and photophobia for one day. Most likely diagnosis?"}]


def logprob_gap(a: list[float], b: list[float]) -> float:
    """Mean absolute difference of the chosen tokens' log-probabilities (greedy, same prompt). A trained adapter must
    give a gap above 0; 0 means requests for the adapter are answered by the base."""
    n = min(len(a), len(b))
    return sum(abs(x - y) for x, y in zip(a[:n], b[:n])) / n if n else 0.0


