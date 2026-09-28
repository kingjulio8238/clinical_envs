"""One episode of the environment, independent of how the model is called (RL readiness C1/C3).

`EpisodeDriver` owns everything that makes an episode an episode — the prompt and tools, executing tool calls, the
single-arm corrective turn, submissions in tool or text form, the nudge after a turn without action, the limits, the
forced final submission, the reward and the monitor signals. The protocol runner (hosted or local models) and the RL
rollout (a trainer's policy) both drive it:

    drv = EpisodeDriver(env, inst, arm="agent", budget=40, limits=EpisodeLimits(max_turns=43))
    while not drv.finished:
        reply = call_model(drv.system, drv.messages, drv.tools, max_tokens=drv.limits.max_output_tokens_per_turn)
        drv.observe(text=reply.text, tool_calls=[{"id", "name", "arguments": dict}], output_tokens=...)
    rec = drv.finish()

Limits are deterministic (C3): turns, output tokens per turn (enforced by the caller's request), output tokens per
episode. A wall-clock deadline exists only for hosted APIs (`wall_clock_s`), never in local or training runs.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass

SUBMIT_KEYS = {"active_diagnoses", "chronic_conditions", "summary", "rankings", "clinical_question", "differential",
               "icd10", "section_type", "relevant"}
SECONDARY_METRICS = ("diagnosis_named", "diagnosis_coded")
UNIT_MAX_TURNS = {"error_detection": 16}
"""Per-unit turn caps (Stage B2); other units: budget + 3."""


@dataclass
class EpisodeLimits:
    max_turns: int
    max_output_tokens_per_turn: int | None = 4096
    max_episode_output_tokens: int | None = None
    wall_clock_s: float | None = None

    @classmethod
    def for_unit(cls, task: str, budget: int = 40, *, per_turn: int | None = 4096, per_episode: int | None = None,
                 wall_clock_s: float | None = None) -> "EpisodeLimits":
        return cls(UNIT_MAX_TURNS.get(task, budget + 3), per_turn, per_episode, wall_clock_s)


LOCAL_LIMITS = {"per_turn": 4096, "per_episode": 32768}
"""Deterministic limits for local evaluation and training (C3): 4,096 output tokens per turn and 32,768 per episode
(hosted Qwen: median 5k, p90 11k, p99 ~30k output tokens per episode on the round-1 units)."""


def submission_from_text(text: str) -> dict | None:
    t = (text or "").strip()
    if t.startswith("```"):
        t = "\n".join(l for l in t.split("\n") if not l.strip().startswith("```"))
    for cand in (t, t[t.find("{"): t.rfind("}") + 1] if "{" in t else ""):
        if not cand:
            continue
        try:
            obj = json.loads(cand, strict=False)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and (set(obj) & SUBMIT_KEYS):
            return obj
        if isinstance(obj, list):
            return {"differential": obj}
    return None


class EpisodeDriver:
    def __init__(self, env, inst: dict, arm: str, budget: int, limits: EpisodeLimits):
        self.env, self.inst, self.arm, self.budget, self.limits = env, inst, arm, budget, limits
        self.task = inst.get("task") or ""
        self.t0 = time.time()
        self.done = False
        self.stopped: str | None = None
        self.last_parsed: dict | None = None
        self.rec: dict = {"gt_id": inst["gt_id"], "patient_id": inst["patient_id"], "encounter_id": inst.get("encounter_id"),
                          "parent_gt_id": inst["gt"].get("parent_gt_id"), "involvement": inst["gt"].get("involvement"),
                          "reward": 0.0, "metric": None, "steps": 0, "turns": 0, "orders": 0, "order_log": [], "unmatched_orders": 0,
                          "input_tokens": 0, "output_tokens": 0, "truncated_turns": 0, "cost_usd": 0.0, "error": None,
                          "forced": False, "submitted": False, "submission": None, "latency_ms": 0, "limit_reason": None,
                          "limits": asdict(limits)}
        ro = env.reset(gt_id=inst["gt_id"], budget=budget)
        self.ro = ro
        self.task = ro.task or self.task
        self.system = ro.instructions
        if arm == "single":
            user = (f"{ro.intro}\n\nYou cannot call tools in this setting. The visible chart follows; answer in one message with "
                    f"the JSON the task asks for (or call {ro.submit_tool}).\n\n{env.visible_chart()}")
            self.tools = [t for t in ro.tools if t["function"]["name"] == ro.submit_tool]
        else:
            user = ro.intro
            self.tools = ro.tools
        self.messages: list[dict] = [{"role": "user", "content": user}]

    # ------------------------------------------------------------------
    @property
    def finished(self) -> bool:
        if self.done or self.stopped:
            return True
        lim = self.limits
        if self.rec["turns"] >= lim.max_turns:
            self.stopped = f"turn limit {lim.max_turns}"
        elif lim.max_episode_output_tokens and self.rec["output_tokens"] >= lim.max_episode_output_tokens:
            self.stopped = f"episode output-token limit {lim.max_episode_output_tokens}"
        elif lim.wall_clock_s and time.time() - self.t0 > lim.wall_clock_s - 5:
            self.stopped = f"episode deadline {lim.wall_clock_s:.0f}s"
            self.rec["error"] = f"episode deadline {lim.wall_clock_s:.0f}s reached at turn {self.rec['turns'] + 1}"
        return self.stopped is not None

    def stop(self, reason: str) -> None:
        self.stopped = reason

    def _set_result(self, reward, info, submission, forced=None, submitted=True):
        self.rec.update(reward=float(reward), metric=info.get("reward_metric"),
                        forced=bool(info.get("forced")) if forced is None else forced,
                        submitted=submitted, submission=submission)
        self.rec["metrics"] = {k: v for k, v in (info.get("metrics") or {}).items() if k in SECONDARY_METRICS}

    def observe(self, text: str | None, tool_calls: list[dict] | None = None, *, output_tokens: int = 0,
                input_tokens: int = 0, truncated: bool = False, latency_ms: int = 0) -> None:
        """Apply one assistant reply. `tool_calls`: [{"id", "name", "arguments": dict}] as the model returned them."""
        env, ro, rec = self.env, self.ro, self.rec
        rec["turns"] += 1
        rec["output_tokens"] += int(output_tokens or 0)
        rec["input_tokens"] += int(input_tokens or 0)
        rec["latency_ms"] += int(latency_ms or 0)
        rec["truncated_turns"] += int(bool(truncated))
        calls = tool_calls or []
        raw = [{"id": c.get("id") or f"call_{rec['turns']}_{i}", "type": "function",
                "function": {"name": c["name"], "arguments": json.dumps(c.get("arguments") or {})}} for i, c in enumerate(calls)]
        if self.arm == "single" and calls and all(c["name"] != ro.submit_tool for c in calls) and not rec.get("format_retry"):
            # the single arm has no chart tools: one corrective turn, the calls never executed (Stage 8)
            rec["format_retry"] = [c["name"] for c in calls]
            self.messages.append({"role": "assistant", "content": text or "", "tool_calls": raw})
            for r, c in zip(raw, calls):
                self.messages.append({"role": "tool", "tool_call_id": r["id"], "name": c["name"],
                                      "content": "Not available in this setting: no tools can be called."})
            self.messages.append({"role": "user", "content": f"Tools are not available here. Answer now from the chart above: "
                                                            f"call {ro.submit_tool} or reply with the JSON answer."})
            return
        if calls:
            self.messages.append({"role": "assistant", "content": text or "", "tool_calls": raw})
            for r, c in zip(raw, calls):
                name, args = c["name"], c.get("arguments") or {}
                obs, reward, done, info = env.step(name, args)
                self.messages.append({"role": "tool", "tool_call_id": r["id"], "name": name,
                                      "content": info.get("observation_text") or json.dumps(obs, default=str)[:8000]})
                if done:
                    self.done = True
                    self._set_result(reward, info, args)
                    return
        else:
            parsed = submission_from_text(text or "")
            self.messages.append({"role": "assistant", "content": text or ""})
            if parsed is not None:
                self.last_parsed = parsed
                obs, reward, done, info = env.step(ro.submit_tool, parsed)
                self.done = True
                self._set_result(reward, info, parsed)
                return
            self.messages.append({"role": "user", "content": f"Take an action: call a tool, or call {ro.submit_tool} with your "
                                                            f"answer. {max(0, self.budget - env.ep['steps'])} actions remain."})
        if self.arm == "single":
            self.stopped = self.stopped or "single arm: one answer"

    def fail(self, exc: BaseException) -> None:
        """Record an error (API failure after retries); the episode scores 0."""
        detail = ""
        cause = exc
        while cause is not None and not detail:
            try:
                r = getattr(cause, "response", None)
            except Exception:  # noqa: BLE001
                r = None
            if r is not None:
                try:
                    detail = f" | {r.text[:300]}"
                except Exception:  # noqa: BLE001
                    detail = " | (streamed response)"
            cause = cause.__cause__
        self.rec["error"] = f"{type(exc).__name__}: {str(exc)[:300]}{detail}"
        self.stopped = self.stopped or "error"

    def finish(self) -> dict:
        rec, env, ro = self.rec, self.env, self.ro
        try:
            if not self.done and not rec["error"]:
                # forced final submission: the best answer seen, else empty (scored 0), exactly like the server
                obs, reward, _, info = env.step(ro.submit_tool, self.last_parsed or {})
                self._set_result(reward, info, self.last_parsed, forced=True, submitted=self.last_parsed is not None)
                rec["limit_reason"] = self.stopped
            elif not self.done and rec["error"] and rec["error"].startswith("episode deadline"):
                obs, reward, _, info = env.step(ro.submit_tool, self.last_parsed or {})
                self._set_result(reward, info, self.last_parsed, forced=True, submitted=self.last_parsed is not None)
                rec["limit_reason"] = self.stopped
            rec["steps"] = env.ep["steps"] if env.ep else rec["steps"]
            rec["order_log"] = list((env.ep or {}).get("orders") or [])
        except Exception as exc:  # noqa: BLE001
            if not rec["error"]:
                self.fail(exc)
        if rec["error"] and not rec["error"].startswith("episode deadline"):
            rec.update(reward=0.0)
            try:
                env.close()
            except Exception:  # noqa: BLE001
                pass
        rec["orders"] = len(rec["order_log"])
        rec["unmatched_orders"] = sum(1 for o in rec["order_log"] if not o.get("matched"))
        rec["wall_s"] = round(time.time() - self.t0, 2)
        from eval.rl_monitor import episode_signals
        rec["signals"] = episode_signals(self.task, rec)
        return rec
