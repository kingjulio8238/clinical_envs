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
    hist = [{"train_reward": 0.30, "dev_reward": 0.30}, {"train_reward": 0.35, "dev_reward": 0.29},
            {"train_reward": 0.42, "dev_reward": 0.28}]
    a = M.alerts(cur, base, hist)
    assert any("probe_many_entries" in x for x in a) and any("entries 7.0" in x for x in a)
    assert any("naming fell" in x for x in a) and any("while dev" in x for x in a)
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
    assert steps[0]["train_reward"] >= 1 - 1e-9 and steps[1]["dev_reward"] >= 1 - 1e-9
    assert set(steps[1]["dev_by_unit"]) == set(TRAIN_UNITS) | {"atypical_diagnosis"}
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

    from eval.rl_rollout import is_dev_patient
    first = [i for i in R.sample_instances(db, "patient_diagnosis", "train", 10 ** 6, 0) if not is_dev_patient(i["patient_id"])][:3]
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



def test_dev_patients_are_never_training_prompts(db):
    """P4: checkpoints are selected on train-split dev patients that no training batch ever contains; the dev set never
    touches public / heldout / private; every training unit has dev instances."""
    from eval.rl_rollout import dev_instances, is_dev_patient, train_prompts
    units = list(TRAIN_UNITS) + ["atypical_diagnosis"]
    prompts = train_prompts(db, units)
    dev = dev_instances(db, units, 10 ** 6)
    assert prompts and dev and not ({i["patient_id"] for i in prompts} & {i["patient_id"] for i in dev})
    assert all(not is_dev_patient(i["patient_id"]) for i in prompts) and all(is_dev_patient(i["patient_id"]) for i in dev)
    train_ids = {i["gt_id"] for u in units for i in db.instances(u, "train")}
    assert {i["gt_id"] for i in dev} <= train_ids                       # dev is a slice of train only
    by_unit = {u: sum(1 for i in dev if i["task"] == u) for u in units}
    assert all(n >= 50 for n in by_unit.values()), by_unit
    small = dev_instances(db, ["patient_diagnosis"], 20)
    assert len(small) == 20 and len({i["patient_id"] for i in small}) == 20      # one per patient first



def test_art_configuration_for_qwen35_9b():
    """P6: unvalidated-arch opt-in, bf16 LoRA (the trained policy is the served policy), 64k sequences, vision off,
    Qwen3.5 tool / reasoning parsers, the validated Qwen3.5 dense LoRA targets, logprobs on training rollouts.
    (Keys were checked against ART 0.5.20's TypedDicts: audit/PRE_OCT1_TODO.md P6.)"""
    import argparse
    import scripts.train_rl as T
    a = argparse.Namespace(max_seq_length=65536, gpu_memory_utilization=0.8, lora_rank=16, lora_alpha=32)
    internal, server = T.art_configs(a)
    assert internal["allow_unvalidated_arch"] is True
    assert internal["init_args"]["load_in_4bit"] is False and internal["init_args"]["load_in_16bit"] is True
    assert internal["init_args"]["max_seq_length"] == internal["engine_args"]["max_model_len"] == 65536
    assert internal["engine_args"]["limit_mm_per_prompt"] == {"image": 0, "video": 0}
    assert server["server_args"]["tool_call_parser"] == "qwen3_coder" and server["server_args"]["reasoning_parser"] == "qwen3"
    lc = T.lora_config(a)
    assert {"in_proj_qkv", "in_proj_z", "out_proj"} <= set(lc["target_modules"]) and lc["rank"] == 16
    client = ScriptedAsyncClient("oracle")
    asyncio.run(rollout(client, "p", {**R.shared_db().instances("patient_diagnosis", "train")[0], "task": "patient_diagnosis"},
                        RolloutConfig(logprobs=True)))
    assert client.requests[0]["logprobs"] is True


def test_trained_checkpoint_serving_configuration():
    """P3: one engine for the before and the after (ART's runtime, the training engine), the adapter loaded into the
    base server under its own name, the same parsers as training, name routing that fails instead of scoring the base
    under the trained name, and a probe that catches an adapter which changes nothing."""
    import argparse
    import importlib
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "gpu"))
    S = importlib.import_module("serving")
    import scripts.train_rl as T
    cfg = S.launch_config(32)
    assert cfg["served_model_name"] == S.MODEL and cfg["lora_path"] is None
    assert cfg["engine_args"]["max_lora_rank"] == 32 and S.launch_config()["engine_args"]["max_lora_rank"] == 16
    assert cfg["engine_args"]["max_model_len"] == 65536 and cfg["engine_args"]["generation_config"] == "vllm"
    _, server = T.art_configs(argparse.Namespace(max_seq_length=65536, gpu_memory_utilization=0.8, lora_rank=16, lora_alpha=32))
    assert all(cfg["server_args"][k] == v for k, v in server["server_args"].items())     # training parsers = eval parsers
    base = [{"id": S.MODEL, "root": S.MODEL}]
    ckpt = "/results/rl/r/.art/clinical-envs/models/qwen35-9b-clinical/checkpoints/0040"
    assert S.check_served(base, None) == S.MODEL
    with pytest.raises(RuntimeError):
        S.check_served(base, ckpt)                                                        # adapter not loaded
    with pytest.raises(RuntimeError):
        S.check_served(base + [{"id": S.RL_SERVED_NAME, "root": "/elsewhere"}], ckpt)       # a different adapter
    assert S.check_served(base + [{"id": S.RL_SERVED_NAME, "root": ckpt + "/"}], ckpt) == S.RL_SERVED_NAME
    assert S.logprob_gap([-0.1, -0.2], [-0.1, -0.2]) == 0.0 and S.logprob_gap([-0.1, -0.2], [-0.3, -0.2, -5]) > 0
    # registry routing: the trained name is requested only when the job says an adapter is loaded
    import eval.config as C
    if {"SH_VLLM_MODEL", "SH_VLLM_BASE_MODEL"} & set(__import__("os").environ):
        pytest.skip("serving env vars set in this shell")
    assert C.MODEL_REGISTRY["qwen3.5-9b-rl"].model_id == S.RL_SERVED_NAME
    assert C.MODEL_REGISTRY["qwen3.5-9b-local"].model_id == S.MODEL
    rl, local = C.MODEL_REGISTRY["qwen3.5-9b-rl"], C.MODEL_REGISTRY["qwen3.5-9b-local"]
    assert (rl.max_tokens, rl.temperature, rl.extra, rl.base_url) == (local.max_tokens, local.temperature, local.extra, local.base_url)
