"""The in-process environment (eval/local_env.py) behaves like the HTTP environment (Stage 5).

Three layers: (1) the Stage-1 reward-hacking policies score identically through `LocalEnv.step(submit)`
and through the scorer directly, and the oracle reaches the ceiling; (2) the visibility and budget rules
hold; (3) when a server is reachable (SH_ENV_URL, default http://127.0.0.1:8000 if it answers /health),
observations and rewards match the HTTP env on sampled episodes.
"""

from __future__ import annotations

import json
import os

import httpx
import pytest

from eval import degenerate as D
from eval.local_env import EpisodeError, LocalEnv

pytestmark = pytest.mark.reward_hacking

SPLIT = "public"
URL = os.environ.get("SH_ENV_URL", "http://127.0.0.1:8000")
TOKEN = os.environ.get("EPIC_SIM_SCORER_TOKEN", "dev-scorer-token-change-in-production")


@pytest.fixture(scope="session")
def env():
    if not D.DEFAULT_DB.exists():
        pytest.skip(f"{D.DEFAULT_DB.name} not present")
    return LocalEnv()


def _units(env, task, n=25):
    return env.db.instances(task, SPLIT)[:n]


def _submit(env, inst, prediction):
    env.reset(gt_id=inst["gt_id"])
    obs, reward, done, info = env.step(env.submit_tool, prediction)
    assert done and obs["status"] == "submitted"
    return reward, info


# ---------------------------------------------------------------------------
# 1. reward parity with the scorer: Stage-1 policies and oracles
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("task,policy", [
    ("patient_diagnosis", "copy_problem_list_plus_chronic"), ("patient_diagnosis", "profile_chronic"),
    ("patient_diagnosis", "copy_problem_list"), ("evidence_retrieval", "single_hpi"),
    ("evidence_retrieval", "section_type_prior"), ("context_summarization", "chart_dump"),
    ("context_summarization", "echo_structured_hints"), ("specialty_involved", "chart_dump"),
    ("specialty_absent", "abstain_always"), ("specialty_involved", "phrase_only"), ("imaging_indication", "restate_order"),
    ("imaging_indication", "chief_complaint"),
])
def test_stage1_policies_score_identically_through_the_env(env, task, policy):
    insts = _units(env, task, 12)
    preds = D.run_policy(env.db, task, policy, insts)
    direct = D.per_item(env.db, task, preds, insts)
    through_env = [_submit(env, i, p)[0] for i, p in zip(insts, preds)]
    assert through_env == pytest.approx(direct, abs=1e-9)


@pytest.mark.parametrize("task", ["patient_diagnosis", "context_summarization", "specialty_involved", "specialty_absent",
                                  "evidence_retrieval", "imaging_indication"])
def test_oracle_reaches_the_ceiling(env, task):
    rewards = []
    for inst in _units(env, task, 15):
        env.reset(gt_id=inst["gt_id"])
        rewards.append(env.step(env.submit_tool, env.oracle())[1])
    floor = 0.99 if task == "evidence_retrieval" else 1.0
    assert min(rewards) >= floor - 1e-9, rewards


def test_copying_the_problem_list_does_not_solve_diagnosis_through_the_env(env):
    insts = _units(env, "patient_diagnosis", 60)
    rewards = []
    for inst in insts:
        ro = env.reset(gt_id=inst["gt_id"])
        problems = env.step("view_problem_list", {"patient_id": ro.patient_id})[0]
        chart = env.step("view_encounter_detail", {"encounter_id": ro.encounter_id})[0]
        names = [p["display_name"] for p in problems]
        icd = env.db.name_to_icd()
        pred = {"active_diagnoses": [{"icd10": icd.get(n.lower(), ""), "name": n, "acuity": "chronic"} for n in names], "chronic_conditions": []}
        rewards.append(env.step(env.submit_tool, pred)[1])
        assert all(s["section_type"] not in ("assessment", "plan") for s in chart["sections"])
    assert sum(rewards) / len(rewards) <= 0.25


def test_empty_and_malformed_submissions_score_zero(env):
    inst = _units(env, "patient_diagnosis", 1)[0]
    reward, info = _submit(env, inst, {})
    assert reward == 0.0 and info["malformed"] is True
    reward, info = _submit(env, inst, {"nonsense": 1})
    assert reward == 0.0 and info["malformed"] is True


# ---------------------------------------------------------------------------
# 2. visibility and budget
# ---------------------------------------------------------------------------

def _dx_with_later_encounters(env):
    for inst in env.db.instances("patient_diagnosis", SPLIT):
        encs = env._encounter_rows(inst["patient_id"])
        order = {r[0]: r[6] for r in encs}
        if any(order[e] > order[inst["encounter_id"]] for e in order):
            return inst, {e for e in order if order[e] > order[inst["encounter_id"]]}
    pytest.skip("no index-encounter instance with later encounters")


def test_point_in_time_cutoff_and_hidden_sections(env):
    inst, future = _dx_with_later_encounters(env)
    ro = env.reset(gt_id=inst["gt_id"])
    assert f"encounter_id = {inst['encounter_id']}" in ro.intro
    seen = {e["encounter_id"] for e in env.step("view_encounters", {"patient_id": ro.patient_id})[0]}
    assert inst["encounter_id"] in seen and not (seen & future)
    assert "error" in env.step("view_encounter_detail", {"encounter_id": next(iter(future))})[0]
    detail = env.step("view_encounter_detail", {"encounter_id": inst["encounter_id"]})[0]
    assert detail["sections"] and not {s["section_type"] for s in detail["sections"]} & {"assessment", "plan"}
    chart = env.step("open_chart", {"patient_id": ro.patient_id})[0]
    assert not {e["encounter_id"] for e in chart["recent_encounters"]} & future
    for r in env.step("view_results", {"patient_id": ro.patient_id, "result_type": "labs"})[0]:
        assert r["encounter_id"] not in future
    for s in env.step("search_chart", {"patient_id": ro.patient_id, "query": "pain"})[0]:
        assert s["encounter_id"] not in future and s["section_type"] not in ("assessment", "plan")


def test_problem_list_is_documented_history_without_codes(env):
    inst = _units(env, "patient_diagnosis", 1)[0]
    ro = env.reset(gt_id=inst["gt_id"])
    problems = env.step("view_problem_list", {"patient_id": ro.patient_id})[0]
    chronic = [str(c if not isinstance(c, dict) else c.get("name") or c.get("condition") or "")
               for c in (env.db.profile(ro.patient_id).get("chronic_conditions") or []) if c]
    assert [p["display_name"] for p in problems] == chronic
    assert all(set(p) == {"display_name", "source"} for p in problems)
    chart = env.step("open_chart", {"patient_id": ro.patient_id})[0]
    assert chart["active_problems"] == problems
    label_names = {d["display_name"].lower() for d in inst["gt"]["active_diagnoses"] + inst["gt"]["chronic_conditions"]}
    assert not label_names & {p["display_name"].lower() for p in problems}


def test_budget_forces_submission_and_errors_are_observations(env):
    inst = _units(env, "context_summarization", 1)[0]
    ro = env.reset(gt_id=inst["gt_id"], budget=3)
    assert env.step("view_encounters", {"patient_id": ro.patient_id})[3]["remaining"] == 2
    assert "error" in env.step("no_such_tool", {})[0]                    # counts as a step
    assert "scored through submit_summary" in env.step("submit_diagnosis", {})[0]["error"]   # does not
    assert "missing argument" in env.step("view_encounter_detail", {})[0]["error"]           # does
    obs, _, done, info = env.step("view_encounters", {"patient_id": ro.patient_id})
    assert not done and "Action budget of 3 spent" in obs["error"] and info["remaining"] == 0
    obs, reward, done, info = env.step(ro.submit_tool, {"summary": "x"})
    assert done and info["forced"] is True
    with pytest.raises(EpisodeError):
        env.step("view_encounters", {"patient_id": ro.patient_id})


def test_sampling_is_seeded_and_task_checked(env):
    a = env.reset(task="imaging_indication", split=SPLIT, seed=3)
    b = env.reset(task="imaging_indication", split=SPLIT, seed=3)
    assert a.gt_id == b.gt_id and a.task == "imaging_indication" and a.submit_tool == "submit_pre_read"
    with pytest.raises(EpisodeError):
        env.reset(gt_id=a.gt_id, task="patient_diagnosis")


# ---------------------------------------------------------------------------
# 3. parity with the HTTP environment (needs the compose stack)
# ---------------------------------------------------------------------------

def _server():
    try:
        if httpx.get(f"{URL}/health", timeout=2).status_code == 200:
            return httpx.Client(base_url=URL, headers={"X-Scorer-Token": TOKEN}, timeout=120)
    except httpx.HTTPError:
        pass
    return None


def _canon(obj):
    return json.dumps(obj, sort_keys=True, default=str)


@pytest.mark.parametrize("task", ["patient_diagnosis", "context_summarization", "evidence_retrieval", "imaging_indication"])
def test_observations_and_rewards_match_the_http_env(env, task):
    client = _server()
    if client is None:
        pytest.skip(f"no environment server at {URL}")
    for inst in _units(env, task, 4):
        ro = env.reset(gt_id=inst["gt_id"])
        ep = client.post("/env/reset", json={"gt_id": inst["gt_id"]}).json()
        for key in ("task", "split", "variant", "patient_id", "encounter_id", "budget", "submit_tool", "instructions", "intro", "task_inputs", "tools"):
            assert getattr(ro, key) == ep[key], (inst["gt_id"], key)
        pid = ro.patient_id
        encs = env.step("view_encounters", {"patient_id": pid})[0]
        calls = [("open_chart", {"patient_id": pid}), ("view_encounters", {"patient_id": pid}),
                 ("view_encounter_detail", {"encounter_id": encs[0]["encounter_id"]}), ("view_problem_list", {"patient_id": pid}),
                 ("view_medications", {"patient_id": pid}), ("view_results", {"patient_id": pid, "result_type": "labs"}),
                 ("view_results", {"patient_id": pid, "result_type": "imaging"}), ("view_section", {"section_id": encs and 1}),
                 ("view_encounter_detail", {"encounter_id": 999999999}), ("no_such_tool", {})]
        for name, args in calls:
            local = env.step(name, args)[0]
            remote = client.post("/env/step", json={"episode_id": ep["episode_id"], "name": name, "arguments": args}).json()["observation"]
            assert _canon(local) == _canon(remote), (inst["gt_id"], name)
        # search: same candidate set (ranking may differ: FTS5 bm25 vs Postgres ts_rank)
        local = env.step("search_chart", {"patient_id": pid, "query": "pain"})[0]
        remote = client.post("/env/step", json={"episode_id": ep["episode_id"], "name": "search_chart",
                                                "arguments": {"patient_id": pid, "query": "pain"}}).json()["observation"]
        assert {s["section_id"] for s in local} == {s["section_id"] for s in remote}, (inst["gt_id"], "search_chart")
        oracle = client.get(f"/env/oracle/{ep['episode_id']}").json()
        assert _canon(oracle["arguments"]) == _canon(env.oracle())
        r_remote = client.post("/env/step", json={"episode_id": ep["episode_id"], "name": oracle["submit_tool"], "arguments": oracle["arguments"]}).json()
        _, r_local, done, info = env.step(env.submit_tool, oracle["arguments"])
        assert done and r_local == pytest.approx(r_remote["reward"], abs=1e-9)
        assert info["reward_metric"] == r_remote["info"]["reward_metric"]
