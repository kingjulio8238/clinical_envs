"""Stage 7 task families: label construction, scorer properties, environment behaviour."""

from __future__ import annotations

import pytest

from eval import degenerate as D
from eval import scoring_tasks7 as S
from eval import stage7 as S7
from eval.local_env import LocalEnv

pytestmark = pytest.mark.reward_hacking
SPLIT = "public"


@pytest.fixture(scope="session")
def db():
    if not D.DEFAULT_DB.exists():
        pytest.skip(f"{D.DEFAULT_DB.name} not present")
    d = D.ReleaseDB()
    if not d.instances("differential_diagnosis", SPLIT):
        pytest.skip("Stage-7 instances not built (scripts/build_stage7_tasks.py)")
    return d


@pytest.fixture(scope="session")
def env(db):
    return LocalEnv()


# ---------------------------------------------------------------------------
# labels
# ---------------------------------------------------------------------------

def test_every_family_derives_from_a_scorable_index_instance(db):
    parents = {i["gt_id"]: i for i in db.instances("patient_diagnosis", SPLIT)}
    for task in S7.NEW_TASKS:
        insts = db.instances(task, SPLIT)
        assert insts, task
        for i in insts[:200]:
            p = parents[i["gt"]["parent_gt_id"]]
            assert i["patient_id"] == p["patient_id"] and i["encounter_id"] == p["encounter_id"] and i["split"] == p["split"]
            assert i["gt"]["derived"] == "stage7_v1"


def test_differential_label_is_correct_plus_distractors(db):
    for i in db.instances("differential_diagnosis", SPLIT)[:100]:
        gt = i["gt"]
        assert gt["correct"] and gt["distractors"]
        assert not {d["diagnosis_id"] for d in gt["correct"]} & {d["diagnosis_id"] for d in gt["distractors"]}
        assert all(d["icd10"] for d in gt["distractors"]) and not {d["icd10"] for d in gt["correct"]} & {d["icd10"] for d in gt["distractors"]}


def test_test_selection_labels_are_documented_discriminating_tests(db):
    for i in db.instances("test_selection", SPLIT)[:100]:
        gt = i["gt"]
        rows = S7.orderable_from_gt(gt)
        names = {f["name"] for f in rows if f["present"]}
        assert set(gt["discriminating"]) <= names and gt["n_needed"] == len(gt["discriminating"]) >= 1
        assert all(f["edge"] in S7.DISCRIMINATING_EDGES for f in rows if f["name"] in gt["discriminating"])
        assert set(gt["hidden_section_types"]) == S7.RESULT_SECTIONS


def test_error_injection_is_a_real_change_of_the_stated_type(db):
    seen = set()
    for i in db.instances("error_detection", SPLIT)[:300]:
        gt = i["gt"]
        orig, inj = S7.error_pair(gt)
        assert gt["error_type"] in S7.ERROR_TYPES and orig != inj
        seen.add(gt["error_type"])
        ov = gt["section_overrides"][str(gt["section_id"])]
        text = next(x for sid, st, x in S7.index_sections(db.conn, i["encounter_id"]) if sid == gt["section_id"])
        assert orig in text and S7.apply_override(text, ov) != text
        if gt["error_type"] == "laterality":
            assert orig.lower().count("left") != inj.lower().count("left") and orig.lower().count("right") != inj.lower().count("right")
    assert len(seen) >= 3, seen


def test_lab_triage_labels_partition_the_results(db):
    for i in db.instances("lab_triage", SPLIT)[:100]:
        gt = i["gt"]
        assert gt["relevant"] and gt["background"] and not set(gt["relevant"]) & set(gt["background"])
        assert gt["most_urgent"] is None or gt["most_urgent"] in gt["relevant"]


def test_atypical_masks_mention_of_a_strong_finding_and_keeps_the_label(db):
    parents = {i["gt_id"]: i for i in db.instances("patient_diagnosis", SPLIT)}
    for i in db.instances("atypical_diagnosis", SPLIT)[:100]:
        gt = i["gt"]
        p = parents[gt["parent_gt_id"]]["gt"]
        key = lambda ds: [(d.get("icd10"), d.get("acuity"), d.get("diagnosis_id")) for d in ds]
        assert key(gt["active_diagnoses"]) == key(p["active_diagnoses"]) and key(gt["chronic_conditions"]) == key(p["chronic_conditions"])
        assert gt["masked_findings"] and gt["section_overrides"]
        for sid, ov in gt["section_overrides"].items():
            text = next(x for s, st, x in S7.index_sections(db.conn, i["encounter_id"]) if s == int(sid))
            new = S7.apply_override(text, ov)
            assert S7.MASK in new and len(new) < len(text) + len(S7.MASK) * len(ov["replacements"])


# ---------------------------------------------------------------------------
# scorers
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("task", S7.NEW_TASKS)
def test_oracle_scores_one_and_empty_scores_zero(db, task):
    insts = db.instances(task, SPLIT)[:40]
    ora = [D.oracle(db, i) for i in insts]
    assert min(D.per_item(db, task, ora, insts)) >= 1.0 - 1e-9
    assert max(D.per_item(db, task, [{} for _ in insts], insts)) == 0.0


def test_differential_rewards_rank_and_penalizes_the_problem_list(db):
    insts = db.instances("differential_diagnosis", SPLIT)[:60]
    reversed_ = [{"differential": list(reversed(D.oracle(db, i)["differential"]))} for i in insts]
    ora = D.primary(db, "differential_diagnosis", [D.oracle(db, i) for i in insts], insts)
    rev = D.primary(db, "differential_diagnosis", reversed_, insts)
    assert ora == pytest.approx(1.0) and rev < ora
    copy = D.primary(db, "differential_diagnosis", D.run_policy(db, "differential_diagnosis", "copy_problem_list", insts), insts)
    assert copy <= 0.1, copy


def test_workup_score_components(db):
    i = db.instances("test_selection", SPLIT)[0]
    gt = i["gt"]
    dx = {"icd10": gt["diagnosis"]["icd10"], "name": gt["diagnosis"]["name"]}
    assert S.score_test_selection_item({**dx, "tests_ordered": list(gt["discriminating"])}, gt)["workup_score"] == pytest.approx(1.0)
    no_orders = S.score_test_selection_item({**dx, "tests_ordered": []}, gt)
    assert no_orders["workup_score"] == pytest.approx(S.NO_EVIDENCE_COVERAGE)
    shotgun = S.score_test_selection_item({**dx, "tests_ordered": gt["discriminating"] + ["a", "b", "c", "d", "e", "f", "g", "h", "i"]}, gt)
    assert shotgun["workup_coverage"] == 1.0 and shotgun["workup_parsimony"] < 0.5 and shotgun["workup_score"] < 0.5
    wrong = S.score_test_selection_item({"icd10": "Z00.00", "name": "x", "tests_ordered": list(gt["discriminating"])}, gt)
    assert wrong["workup_score"] == 0.0
    # names in the submission are matched, not trusted verbatim: unmatched names cost parsimony and earn no coverage
    assert S.score_test_selection_item({**dx, "tests_ordered": ["nonexistent test"]}, gt)["workup_score"] == pytest.approx(S.NO_EVIDENCE_COVERAGE)


def test_error_detection_half_credit_and_aliases(db):
    gt = {"section_type": "physical_exam", "error_type": "laterality"}
    assert S.score_error_detection_item({"section_type": "physical exam", "error_type": "left/right swap"}, gt)["error_detection_score"] == 1.0
    assert S.score_error_detection_item({"section_type": "labs", "error_type": "laterality"}, gt)["error_detection_score"] == 0.5
    assert S.score_error_detection_item({"section_type": "labs", "error_type": "age"}, gt)["error_detection_score"] == 0.0


def test_triage_all_results_is_penalized(db):
    insts = db.instances("lab_triage", SPLIT)[:80]
    allr = D.primary(db, "lab_triage", D.run_policy(db, "lab_triage", "all_results", insts), insts)
    ora = D.primary(db, "lab_triage", [D.oracle(db, i) for i in insts], insts)
    assert ora == pytest.approx(1.0) and allr < 0.75, allr


# ---------------------------------------------------------------------------
# environments
# ---------------------------------------------------------------------------

def test_test_selection_episode_hides_results_and_orders_reveal_documented_findings(env):
    inst = env.db.instances("test_selection", SPLIT)[0]
    gt = inst["gt"]
    ro = env.reset(gt_id=inst["gt_id"])
    assert ro.submit_tool == "submit_workup" and any(t["function"]["name"] == "order_test" for t in ro.tools)
    for task in S7.NEW_TASKS:                                     # every task's brief offers its submit tool
        r = env.reset(gt_id=env.db.instances(task, SPLIT)[0]["gt_id"])
        assert any(t["function"]["name"] == r.submit_tool for t in r.tools), task
        assert (task == "test_selection") == any(t["function"]["name"] == "order_test" for t in r.tools), task
    ro = env.reset(gt_id=inst["gt_id"])
    detail = env.step("view_encounter_detail", {"encounter_id": ro.encounter_id})[0]
    assert not {s["section_type"] for s in detail["sections"]} & S7.RESULT_SECTIONS
    labs = env.step("view_results", {"patient_id": ro.patient_id, "result_type": "labs"})[0]
    assert all(r["encounter_id"] != ro.encounter_id for r in labs)
    hits = env.step("search_chart", {"patient_id": ro.patient_id, "query": gt["discriminating"][0]})[0]
    assert all(not (h["encounter_id"] == ro.encounter_id and h["section_type"] in S7.RESULT_SECTIONS) for h in hits)
    missing = env.step("order_test", {"name": "positron emission tomography of the left toe"})[0]
    assert missing["matched"] is None and "not performed" in missing["result"]
    obs = env.step("order_test", {"name": gt["discriminating"][0]})[0]
    assert obs["matched"] == gt["discriminating"][0]
    assert env.state()["orders"] and len(env.state()["orders"]) == 2
    _, reward, done, info = env.step("submit_workup", {"icd10": gt["diagnosis"]["icd10"], "name": gt["diagnosis"]["name"], "tests_ordered": ["fake claim"]})
    assert done and info["reward_metric"] == "workup_score"
    assert reward == pytest.approx(1.0 * 1.0 * min(1.0, gt["n_needed"] / 2))          # the trace counted 2 orders, the claim did not
    # other tasks reject order_test
    env.reset(gt_id=env.db.instances("differential_diagnosis", SPLIT)[0]["gt_id"])
    assert "error" in env.step("order_test", {"name": "CBC"})[0]


def test_error_and_atypical_episodes_serve_the_altered_text(env):
    inst = env.db.instances("error_detection", SPLIT)[0]
    gt = inst["gt"]
    ro = env.reset(gt_id=inst["gt_id"])
    orig, inj = S7.error_pair(gt)
    sec = env.step("view_section", {"section_id": gt["section_id"]})[0]
    assert inj in sec["section_text"] and orig not in sec["section_text"]
    detail = env.step("view_encounter_detail", {"encounter_id": ro.encounter_id})[0]
    assert any(inj in s["section_text"] for s in detail["sections"])
    _, reward, done, _ = env.step("submit_error", {"section_type": gt["section_type"], "error_type": gt["error_type"], "description": "x"})
    assert done and reward == 1.0
    inst = env.db.instances("atypical_diagnosis", SPLIT)[0]
    ro = env.reset(gt_id=inst["gt_id"])
    detail = env.step("view_encounter_detail", {"encounter_id": ro.encounter_id})[0]
    joined = " ".join(s["section_text"] for s in detail["sections"])
    assert S7.MASK in joined
    for sid, ov in inst["gt"]["section_overrides"].items():
        for orig, _ in ov["replacements"]:
            assert orig not in joined
    _, reward, done, info = env.step(ro.submit_tool, env.oracle())
    assert done and reward == pytest.approx(1.0) and info["reward_metric"] == "weighted_problem_list_f1_neutral"


def test_oracle_episodes_reach_the_ceiling_in_the_env(env):
    for task in S7.NEW_TASKS:
        for inst in env.db.instances(task, SPLIT)[:5]:
            env.reset(gt_id=inst["gt_id"])
            for name in env.oracle_orders():
                env.step("order_test", {"name": name})
            args = env.oracle()
            args.pop("tests_ordered", None)
            _, reward, done, _ = env.step(env.submit_tool, args)
            assert done and reward >= 1.0 - 1e-9, (task, inst["gt_id"], reward)
