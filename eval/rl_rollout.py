"""RL rollouts (RL readiness C1): one training episode against any OpenAI-compatible chat client — the trainer's policy
server (ART's vLLM + LoRA) in training, a local vLLM server in C2/C4 — recording the policy's choices so a trainer can
compute token log-probabilities, with the reward from the frozen scorer.

    from eval.rl_rollout import RolloutConfig, rollout, start_up
    info = start_up()                        # refuses to run on a scorer that differs from eval/reward_lock.json
    result = await rollout(client, "policy", inst, cfg)
    result.reward, result.record["signals"], result.messages_and_choices   # ART: art.Trajectory(messages_and_choices=..., reward=...)

The episode itself is eval.episode.EpisodeDriver, identical to evaluation; only the model call lives here.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from typing import Any

from eval.episode import LOCAL_LIMITS, EpisodeDriver, EpisodeLimits

TRAIN_UNITS = ("patient_diagnosis", "differential_diagnosis", "evidence_retrieval", "test_selection")
"""RL round 1 (audit/RL_READINESS_TODO.md B-stage outcome); atypical_diagnosis is the held-out transfer test."""
TRANSFER_UNITS = ("atypical_diagnosis",)

EVAL_SAMPLING = {"temperature": 1.0, "top_p": 0.95, "presence_penalty": 1.5, "extra_body": {"top_k": 20}}
"""Qwen3.5 thinking-mode defaults (model card); used for training rollouts and for every before/after evaluation."""


def start_up() -> dict:
    """Frozen-reward check at trainer start-up (A4/C1). Raises eval.reward_version.RewardDrift on a drifted scorer."""
    from eval import reward_version as RV
    return RV.require_frozen()


@dataclass
class RolloutConfig:
    budget: int = 40
    per_turn_tokens: int = LOCAL_LIMITS["per_turn"]
    per_episode_tokens: int = LOCAL_LIMITS["per_episode"]
    sampling: dict = field(default_factory=lambda: dict(EVAL_SAMPLING))
    seed: int | None = None

    def limits(self, task: str) -> EpisodeLimits:
        return EpisodeLimits.for_unit(task, self.budget, per_turn=self.per_turn_tokens, per_episode=self.per_episode_tokens)


@dataclass
class RolloutResult:
    reward: float
    record: dict
    messages_and_choices: list[Any]
    error: str | None
    tools: list[dict] = field(default_factory=list)


_local = threading.local()


def _env():
    from eval import protocol_run as R
    return R._env()


def _parse_calls(message) -> list[dict]:
    out = []
    for i, tc in enumerate(getattr(message, "tool_calls", None) or []):
        fn = tc.function
        try:
            args = json.loads(fn.arguments) if isinstance(fn.arguments, str) else (fn.arguments or {})
        except (json.JSONDecodeError, TypeError):
            args = {}
        out.append({"id": tc.id or f"call_{i}", "name": fn.name, "arguments": args if isinstance(args, dict) else {}})
    return out


async def rollout(client, model: str, inst: dict, cfg: RolloutConfig | None = None, env=None) -> RolloutResult:
    """One agent episode. `client`: an openai.AsyncOpenAI-compatible client. `inst`: an instance dict with `task`."""
    cfg = cfg or RolloutConfig()
    env = env or _env()
    task = inst["task"]
    drv = EpisodeDriver(env, inst, "agent", cfg.budget, cfg.limits(task))
    system = {"role": "system", "content": drv.system}
    choices: list[Any] = []
    turn = 0
    while not drv.finished:
        kwargs = dict(model=model, messages=[system] + drv.messages, tools=drv.tools, tool_choice="auto",
                      max_completion_tokens=cfg.per_turn_tokens, **{k: v for k, v in cfg.sampling.items() if k != "extra_body"})
        if cfg.sampling.get("extra_body"):
            kwargs["extra_body"] = dict(cfg.sampling["extra_body"])
        if cfg.seed is not None:
            kwargs["seed"] = cfg.seed * 1000 + turn
        try:
            resp = await client.chat.completions.create(**kwargs)
        except Exception as exc:  # noqa: BLE001 — an infrastructure failure: recorded, excluded from training
            drv.fail(exc)
            break
        choice = resp.choices[0]
        choices.append(choice)
        usage = getattr(resp, "usage", None)
        try:
            drv.observe(choice.message.content, _parse_calls(choice.message),
                        output_tokens=getattr(usage, "completion_tokens", 0) or 0, input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
                        truncated=choice.finish_reason == "length")
        except Exception as exc:  # noqa: BLE001
            drv.fail(exc)
            break
        turn += 1
    rec = drv.finish()
    # the trajectory: the system prompt, then every message, with each assistant turn replaced by the policy's Choice
    mac: list[Any] = [system]
    it = iter(choices)
    for m in drv.messages:
        if m["role"] == "assistant":
            ch = next(it, None)
            mac.append(ch if ch is not None else m)
        else:
            mac.append(m)
    return RolloutResult(reward=float(rec["reward"]), record=rec, messages_and_choices=mac, error=rec.get("error"), tools=drv.tools)
