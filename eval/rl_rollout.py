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

DEV_MODULUS = 10
"""RL checkpoint-selection dev set (audit/RL_SUCCESS_CRITERIA.md, P4): train-split patients with
sha256("rl-dev:<patient_id>") % 10 == 0 (52 of 600 patients) are never trained on; checkpoints are chosen on their
instances. Public, heldout and private stay untouched until the final before/after evaluation."""


def is_dev_patient(patient_id) -> bool:
    import hashlib
    return int(hashlib.sha256(f"rl-dev:{patient_id}".encode()).hexdigest(), 16) % DEV_MODULUS == 0


def train_prompts(db, units, seed: int = 0, keep: set | None = None, drop: set | None = None) -> list[dict]:
    """Training prompts: train-split instances of non-dev patients — only a C4 keep-list when given, or all but a C4
    drop-list (prompts measured to have no reward spread)."""
    import random
    out = [{**i, "task": u} for u in units for i in db.instances(u, "train")
           if not is_dev_patient(i["patient_id"]) and (keep is None or i["gt_id"] in keep)
           and (drop is None or i["gt_id"] not in drop)]
    random.Random(seed).shuffle(out)
    return out


def dev_instances(db, units, per_unit: int, seed: int = 0) -> list[dict]:
    """Checkpoint-selection instances: up to `per_unit` train-split instances of dev patients per unit, one per
    patient first (seeded), so selection never touches public / heldout / private."""
    import random
    out = []
    for u in units:
        insts = [i for i in db.instances(u, "train") if is_dev_patient(i["patient_id"])]
        rng = random.Random(f"{seed}:{u}:dev")
        rng.shuffle(insts)
        first, rest, seen = [], [], set()
        for i in insts:
            (rest if i["patient_id"] in seen else first).append(i)
            seen.add(i["patient_id"])
        out += [{**i, "task": u} for i in (first + rest)[:per_unit]]
    return out


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
    logprobs: bool = False
    """Training rollouts request token log-probabilities: ART's trainer refuses a Choice without them
    ("Trainable vLLM Choice is missing logprobs", art/preprocessing/tokenize.py)."""

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
        if cfg.logprobs:
            kwargs["logprobs"] = True
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
