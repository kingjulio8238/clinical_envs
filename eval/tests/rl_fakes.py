"""CPU stand-ins for the RL pipeline tests and `scripts/train_rl.py --dry-run`: an openai.AsyncOpenAI-shaped client whose
policy is scripted (oracle answers through the environment's own oracle, or a bad policy), returning objects with the
same attributes as openai's ChatCompletion (choices[0].message.content / .tool_calls / .finish_reason, usage)."""

from __future__ import annotations

import json
from types import SimpleNamespace as NS


class ScriptedAsyncClient:
    def __init__(self, policy: str = "oracle", fail_after: int | None = None):
        self.policy, self.fail_after, self.calls = policy, fail_after, 0
        self.requests: list[dict] = []
        self.chat = NS(completions=NS(create=self._create))

    async def _create(self, **kw):
        from eval import protocol_run as R
        self.calls += 1
        self.requests.append(kw)
        if self.fail_after is not None and self.calls > self.fail_after:
            raise RuntimeError("policy server down")
        env = R._env()
        tools = [t["function"]["name"] for t in kw.get("tools") or []]
        submit = next((t for t in tools if t.startswith("submit")), None)
        n_tool_msgs = sum(1 for m in kw["messages"] if m.get("role") == "tool")
        if self.policy == "loop":                     # never submits: exercises the limits
            call = {"name": "view_problem_list", "arguments": {"patient_id": env.ep["patient_id"]}}
        elif self.policy == "text":                   # answers in text, not a tool call
            args = env.oracle(); args.pop("tests_ordered", None)
            return self._reply(content=json.dumps(args))
        else:
            orders = env.oracle_orders() if "order_test" in tools else []
            n_orders = sum(1 for m in kw["messages"] if m.get("role") == "tool" and m.get("name") == "order_test")
            if n_orders < len(orders):
                call = {"name": "order_test", "arguments": {"name": orders[n_orders]}}
            elif n_tool_msgs == 0 and "view_encounters" in tools:
                call = {"name": "view_encounters", "arguments": {"patient_id": env.ep["patient_id"]}}
            else:
                args = env.oracle(); args.pop("tests_ordered", None)
                if self.policy == "empty":
                    args = {}
                call = {"name": submit, "arguments": args}
        tc = NS(id=f"call_{self.calls}", type="function", function=NS(name=call["name"], arguments=json.dumps(call["arguments"])))
        return self._reply(content="", tool_calls=[tc])

    def _reply(self, content: str = "", tool_calls=None, finish="stop"):
        msg = NS(role="assistant", content=content, tool_calls=tool_calls)
        choice = NS(index=0, message=msg, finish_reason="tool_calls" if tool_calls else finish, logprobs=None)
        return NS(choices=[choice], usage=NS(prompt_tokens=100, completion_tokens=20))
