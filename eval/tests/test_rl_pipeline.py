"""RL pipeline (stage C, non-GPU): the shared episode driver, deterministic limits, rollouts against an OpenAI-shaped
client, the frozen-reward start-up check, monitors, and a CPU dry run of the training loop. No network, no GPU."""

from __future__ import annotations

import asyncio
import json

import pytest

from eval import degenerate as D
from eval import protocol_run as R
from eval import rl_monitor as M
from eval.episode import EpisodeLimits
from eval.rl_rollout import TRAIN_UNITS, RolloutConfig, rollout, start_up
from eval.tests.rl_fakes import ScriptedAsyncClient
from eval.tests.test_protocol import ScriptedAdapter

pytestmark = pytest.mark.reward_hacking


@pytest.fixture(scope="module")
def db():
    if not D.DEFAULT_DB.exists():
        pytest.skip("release DB not present")
    return R.shared_db()


def _inst(db, task, split="train", k=0):
    return {**db.instances(task, split)[k], "task": task}


@pytest.mark.parametrize("task", TRAIN_UNITS + ("atypical_diagnosis",))
def test_oracle_rollout_scores_one_and_records_the_policy_choices(db, task):
    client = ScriptedAsyncClient("oracle")
    res = asyncio.run(rollout(client, "policy", _inst(db, task), RolloutConfig()))
    assert res.error is None and res.reward >= 1 - 1e-9, res.record
    mac = res.messages_and_choices
    assert mac[0]["role"] == "system" and mac[1]["role"] == "user"
    n_choices = sum(1 for m in mac if not isinstance(m, dict))
    assert n_choices == client.calls == res.record["turns"] and res.tools
    assert res.record["signals"]["entries"] >= 1 and "limits" in res.record
    req = client.requests[0]
    assert req["max_completion_tokens"] == 4096 and req["temperature"] == 1.0 and req["extra_body"]["top_k"] == 20


def test_text_answers_are_parsed_like_tool_submissions(db):
    res = asyncio.run(rollout(ScriptedAsyncClient("text"), "p", _inst(db, "patient_diagnosis"), RolloutConfig()))
    assert res.reward >= 1 - 1e-9 and res.record["submitted"]


def test_turn_and_token_limits_are_deterministic(db):
    inst = _inst(db, "patient_diagnosis")
    res = asyncio.run(rollout(ScriptedAsyncClient("loop"), "p", inst, RolloutConfig(budget=40)))
    assert res.record["limit_reason"] == "turn limit 43" and res.record["forced"] and res.reward == 0.0
    cfg = RolloutConfig(per_episode_tokens=60)                    # 20 output tokens per scripted turn
    res2 = asyncio.run(rollout(ScriptedAsyncClient("loop"), "p", inst, cfg))
    assert res2.record["limit_reason"] == "episode output-token limit 60" and res2.record["turns"] == 3
    assert res2.record["limits"]["wall_clock_s"] is None       # no wall clock in local / training runs (C3)
    ed = asyncio.run(rollout(ScriptedAsyncClient("loop"), "p", _inst(db, "error_detection"), RolloutConfig()))
    assert ed.record["limit_reason"] == "turn limit 16"


def test_policy_server_failure_is_an_error_not_a_score(db):
    res = asyncio.run(rollout(ScriptedAsyncClient("oracle", fail_after=1), "p", _inst(db, "differential_diagnosis"), RolloutConfig()))
    assert res.error and "policy server down" in res.error and res.reward == 0.0


def test_start_up_refuses_a_drifted_reward(monkeypatch):
    from eval import reward_version as RV
    assert start_up()["reward_drift"] == []
    monkeypatch.setattr(RV, "drift", lambda: ["eval/scoring.py"])
    with pytest.raises(RV.RewardDrift):
        start_up()


def test_monitor_signals_flag_probe_patterns():
    sink = {"active_diagnoses": [{"icd10": "", "name": " ".join(f"word{i}" for i in range(30)), "acuity": "acute"}] * 12}
    sig = M.episode_signals("patient_diagnosis", {"submission": sink, "turns": 2, "steps": 1, "metrics": {}})
    assert sig["probe_many_entries"] and sig["probe_long_name"] and sig["probe_duplicates"] and sig["names_only_entries"] == 12
    ok = M.episode_signals("patient_diagnosis", {"submission": {"active_diagnoses": [{"icd10": "I10", "name": "Hypertension"}]}})
    assert not (ok["probe_many_entries"] or ok["probe_long_name"] or ok["probe_duplicates"])


def test_alert_rules():
    base = {"entries": 3.0, "answer_chars": 300.0, "max_name_tokens": 4.0, "diagnosis_named": 0.5, "reward_mean": 0.3}
    cur = {"entries": 7.0, "answer_chars": 350.0, "max_name_tokens": 4.0, "diagnosis_named": 0.35, "reward_mean": 0.4,
           "probe_many_entries": 0.08}
    hist = [{"train_reward": 0.30, "heldout_reward": 0.30}, {"train_reward": 0.35, "heldout_reward": 0.29},
            {"train_reward": 0.42, "heldout_reward": 0.28}]
    a = M.alerts(cur, base, hist)
    assert any("probe_many_entries" in x for x in a) and any("entries 7.0" in x for x in a)
    assert any("naming fell" in x for x in a) and any("heldout" in x for x in a)
    assert M.alerts(base, base, hist[:1]) == []


def test_local_protocol_runs_record_deterministic_limits(tmp_path, db):
    out = R.run_units("glm-5.3-flash", ["patient_diagnosis"], 1, seed=13, workers=1, out_root=tmp_path, adapter=ScriptedAdapter(R._env),
                      prices=(0.0, 0.0), quiet=True, local=True)["patient_diagnosis"]
    mf = json.loads((out / "manifest.json").read_text())
    assert mf["local"] is True and mf["limits"]["wall_clock_s"] is None and mf["limits"]["max_episode_output_tokens"] == 32768
    pred = json.loads((out / "predictions.jsonl").read_text().splitlines()[0])
    assert pred["reward"] >= 1 - 1e-9 and "signals" in pred


def test_training_loop_dry_run(tmp_path, monkeypatch, db):
    import scripts.train_rl as T
    monkeypatch.setattr(T, "ROOT", tmp_path)
    rc = T.main(["--run-name", "dry", "--dry-run", "--max-steps", "2", "--rollouts-per-group", "2", "--groups-per-step", "3",
                 "--val-every", "2", "--val-per-unit", "2"])
    assert rc == 0
    steps = [json.loads(l) for l in (tmp_path / "results" / "rl" / "dry" / "steps.jsonl").read_text().splitlines()]
    assert [s["step"] for s in steps] == [1, 2] and all(s["episodes"] == 6 and s["exceptions"] == 0 for s in steps)
    assert steps[0]["train_reward"] >= 1 - 1e-9 and steps[1]["heldout_reward"] >= 1 - 1e-9
    assert set(steps[1]["heldout_by_unit"]) == set(TRAIN_UNITS) | {"atypical_diagnosis"}
    cfg = json.loads((tmp_path / "results" / "rl" / "dry" / "config.json").read_text())
    assert cfg["reward_version"] and cfg["reward_drift"] == []


def test_group_variance_summary_and_prompt_filter(tmp_path, db):
    """C4: prompts whose k rewards are equal carry no GRPO signal and are dropped; the rest form the training keep-list."""
    import scripts.group_variance as G

    class Alternating(ScriptedAdapter):
        """Oracle on even samples, an empty answer on odd ones — except for one prompt that always fails."""

        def call_multi_turn_with_retry(self, system, messages, tools=None, max_retries=None):
            from eval.adapters import _CALL_SEED
            env = self.env_getter()
            sample = _CALL_SEED.base % 100
            if env.ep["gt_id"] == self.always_zero or sample % 2:
                from eval.adapters import ModelResponse
                call = {"name": env.submit_tool, "arguments": {}}
                raw = [{"id": "c", "type": "function", "function": {"name": env.submit_tool, "arguments": "{}"}}]
                return ModelResponse(text="", input_tokens=1, output_tokens=1, latency_ms=1, raw_json=None, tool_calls=[call], raw_tool_calls=raw)
            return super().call_multi_turn_with_retry(system, messages, tools, max_retries)

    first = R.sample_instances(db, "patient_diagnosis", "train", 3, 0)
    ad = Alternating(R._env)
    ad.always_zero = first[0]["gt_id"]
    argv = ["--model", "glm-5.3-flash", "--units", "patient_diagnosis", "--per-unit", "3", "--k", "4", "--workers", "1", "--out", str(tmp_path)]
    orig = G.create_adapter
    G.create_adapter = lambda cfg: ad
    try:
        assert G.main(argv) == 0
    finally:
        G.create_adapter = orig
    summ = json.loads((tmp_path / "glm-5.3-flash" / "summary.json").read_text())["per_unit"]["patient_diagnosis"]
    keep = json.loads((tmp_path / "glm-5.3-flash" / "prompts.json").read_text())["keep"]
    assert summ["complete"] == 3 and summ["zero_variance"] == 1 and summ["all_zero"] == 1
    assert ad.always_zero not in keep and len(keep) == 2


def test_local_model_payload_carries_sampling_and_seed(monkeypatch):
    """The vLLM request: Qwen3.5 thinking sampling (top_p, top_k, presence_penalty), a per-episode seed, no streaming."""
    import httpx
    import eval.adapters as A
    from eval.config import MODEL_REGISTRY
    seen = {}

    def handler(request):
        seen.update(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": "{}"}, "finish_reason": "length"}],
                                         "usage": {"prompt_tokens": 1, "completion_tokens": 4096}})
    real = httpx.Client
    monkeypatch.setattr(A.httpx, "Client", lambda *a, **k: real(transport=httpx.MockTransport(handler)))
    ad = A.create_adapter(MODEL_REGISTRY["qwen3.5-9b-local"])
    A._CALL_SEED.base, A._CALL_SEED.n = 4242, 0
    r = ad.call_multi_turn("s", [{"role": "user", "content": "u"}])
    assert seen["top_k"] == 20 and seen["top_p"] == 0.95 and seen["presence_penalty"] == 1.5 and seen["temperature"] == 1.0
    assert seen["seed"] == 4242 * 1000 + 1 and "stream" not in seen and seen["max_tokens"] == 4096
    assert r.finish_reason == "length"
