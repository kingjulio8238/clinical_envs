"""EVAL_PROTOCOL.md arithmetic and the protocol runner, with a scripted adapter (no network)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from eval import degenerate as D
from eval import protocol as P
from eval import protocol_run as R
from eval.adapters import ModelResponse

pytestmark = pytest.mark.reward_hacking


def test_bootstrap_and_paired_difference_are_deterministic():
    vals = [0.0, 0.5, 1.0, 1.0, 0.25]
    lo, hi = P.bootstrap_ci(vals)
    assert lo <= sum(vals) / 5 <= hi and (lo, hi) == P.bootstrap_ci(vals)
    a = {1: 1.0, 2: 1.0, 3: 0.8, 4: 0.9}
    b = {1: 0.2, 2: 0.4, 3: 0.3, 9: 1.0}
    d = P.paired_diff(a, b)
    assert d["n"] == 3 and d["mean"] == pytest.approx((0.8 + 0.6 + 0.5) / 3) and d["significant"]
    same = P.paired_diff(a, a)
    assert same["mean"] == 0 and not same["significant"]


def test_normalization_uses_floors_file():
    floors = {"splits": {"public": {"patient_diagnosis": {"metrics": {"weighted_problem_list_f1_neutral":
              {"floor": 0.2, "ceiling": 1.0, "headroom": 0.8}}}}}}
    assert P.normalize("patient_diagnosis", "public", 0.6, floors) == pytest.approx(0.5)
    assert P.normalize("patient_diagnosis", "public", 0.1, floors) < 0
    assert P.normalize("nope", "public", 0.6, floors) is None


def _write_run(root: Path, model: str, task: str, arm: str, rewards: dict[int, float], parent=None):
    d = root / f"{model}__{task}__{arm}__public__s0"
    d.mkdir(parents=True)
    (d / "manifest.json").write_text(json.dumps({"model": model, "task": task, "arm": arm, "split": "public"}))
    with (d / "predictions.jsonl").open("w") as fh:
        for g, r in rewards.items():
            fh.write(json.dumps({"gt_id": g, "reward": r, "steps": 3, "cost_usd": 0.01, "error": None, "forced": False,
                                 "parent_gt_id": (parent or {}).get(g)}) + "\n")


def test_leaderboard_ranks_normalized_and_separates_the_generator(tmp_path):
    _write_run(tmp_path, "modelA", "patient_diagnosis", "agent", {1: 0.9, 2: 0.8, 3: 1.0})
    _write_run(tmp_path, "modelB", "patient_diagnosis", "agent", {1: 0.3, 2: 0.2, 3: 0.4})
    _write_run(tmp_path, "kimi-k2.5", "patient_diagnosis", "agent", {1: 1.0, 2: 1.0, 3: 1.0})
    _write_run(tmp_path, "modelA", "patient_diagnosis", "single", {1: 0.5, 2: 0.5, 3: 0.5})
    runs = P.load_runs(tmp_path)
    md, summary = P.leaderboard(runs, "public")
    lines = [l for l in md.splitlines() if l.startswith("| model") or l.startswith("| kimi") or l.startswith("| *gen")]
    body = md.splitlines()
    ia = next(i for i, l in enumerate(body) if l.startswith("| modelA |"))
    ib = next(i for i, l in enumerate(body) if l.startswith("| modelB |"))
    ig = next(i for i, l in enumerate(body) if l.startswith("| *generator"))
    ik = next(i for i, l in enumerate(body) if l.startswith("| kimi-k2.5 |"))
    assert ia < ib < ig < ik, "ranked panel first, then the generator row after the separator"
    assert "best" in body[ia] and "*" in body[ib]          # B's paired difference to A is significant
    abl = P.ablations(runs, "public")
    assert "tools − no tools" in abl and "modelA" in abl


class ScriptedAdapter:
    """Orders the discriminating tests (if any) then submits the oracle; or fails when asked to."""

    def __init__(self, env_getter, fail: bool = False):
        self.env_getter, self.fail = env_getter, fail
        self.config = type("C", (), {"name": "scripted", "temperature": 0.0})()

    def call_multi_turn_with_retry(self, system, messages, tools=None, max_retries=None):
        if self.fail:
            raise RuntimeError("provider down")
        env = self.env_getter()
        ordered = sum(1 for m in messages if m.get("role") == "tool" and m.get("name") == "order_test")
        orders = env.oracle_orders() if any(t["function"]["name"] == "order_test" for t in (tools or [])) else []
        if ordered < len(orders):
            call = {"name": "order_test", "arguments": {"name": orders[ordered]}}
        else:
            args = env.oracle(); args.pop("tests_ordered", None)
            call = {"name": env.submit_tool, "arguments": args}
        raw = [{"id": f"call_{len(messages)}", "type": "function", "function": {"name": call["name"], "arguments": json.dumps(call["arguments"])}}]
        return ModelResponse(text="", input_tokens=1000, output_tokens=50, latency_ms=5, raw_json=None, tool_calls=[call], raw_tool_calls=raw)


@pytest.fixture(scope="module")
def db():
    if not D.DEFAULT_DB.exists():
        pytest.skip("release DB not present")
    return D.ReleaseDB()


@pytest.mark.parametrize("task", ["patient_diagnosis", "test_selection", "specialty_absent"])
def test_runner_scores_oracle_agent_one_and_records_cost(tmp_path, db, task):
    adapter = ScriptedAdapter(R._env)
    out = R.run("glm-5.3-flash", task, n=3, arm="agent", seed=1, workers=1, out_root=tmp_path, adapter=adapter, prices=(1.0, 2.0), quiet=True)
    preds = [json.loads(l) for l in (out / "predictions.jsonl").read_text().splitlines()]
    assert len(preds) == 3 and all(p["reward"] >= 1.0 - 1e-9 for p in preds), preds
    assert all(p["input_tokens"] > 0 and p["cost_usd"] > 0 and p["error"] is None for p in preds)
    if task == "test_selection":
        assert all(p["orders"] >= 1 for p in preds)
    mf = json.loads((out / "manifest.json").read_text())
    assert mf["model"] == "glm-5.3-flash" and mf["n_sampled"] == 3 and mf["prices_per_million"] == {"input": 1.0, "output": 2.0}
    assert len({p["patient_id"] for p in preds}) == 3            # one instance per patient


def test_runner_records_failures_as_zero_and_resumes(tmp_path, db):
    out = R.run("glm-5.3-flash", "patient_diagnosis", n=2, arm="agent", seed=2, workers=1, out_root=tmp_path,
                adapter=ScriptedAdapter(R._env, fail=True), prices=(1.0, 2.0), quiet=True)
    preds = [json.loads(l) for l in (out / "predictions.jsonl").read_text().splitlines()]
    assert len(preds) == 2 and all(p["reward"] == 0.0 and p["error"] and "provider down" in p["error"] for p in preds)
    # resume: nothing left to do, file unchanged
    out2 = R.run("glm-5.3-flash", "patient_diagnosis", n=2, arm="agent", seed=2, workers=1, out_root=tmp_path,
                 adapter=ScriptedAdapter(R._env), prices=(1.0, 2.0), quiet=True)
    assert out2 == out and len((out / "predictions.jsonl").read_text().splitlines()) == 2


def test_single_arm_gives_the_visible_chart_and_hides_results(tmp_path, db):
    seen = {}

    class Peek(ScriptedAdapter):
        def call_multi_turn_with_retry(self, system, messages, tools=None, max_retries=None):
            seen["user"] = messages[0]["content"]; seen["tools"] = [t["function"]["name"] for t in tools]
            return super().call_multi_turn_with_retry(system, messages, tools, max_retries)

    out = R.run("glm-5.3-flash", "test_selection", n=1, arm="single", seed=3, workers=1, out_root=tmp_path, adapter=Peek(R._env), prices=(1.0, 2.0), quiet=True)
    assert "[HPI]" in seen["user"] or "[CHIEF COMPLAINT]" in seen["user"]
    assert "[LABS]" not in seen["user"].split("ENCOUNTER")[-1]      # the index visit's results are hidden in the single arm too
    assert seen["tools"] == ["submit_workup"]
    preds = [json.loads(l) for l in (out / "predictions.jsonl").read_text().splitlines()]
    assert preds[0]["submitted"]
