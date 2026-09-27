"""Stage 2 regression suite: no consumer of the simulator or the eval harness can read the labels.

Runs against the released SQLite database in-process (the simulator services are exercised through
`sqlite+aiosqlite`, as the reward-hacking suite does). Postgres-backed end-to-end coverage of the same
rules lives in epic_sim/tests (test_env.py, test_epic.py, test_fhir.py).
"""

from __future__ import annotations

import asyncio
import collections
import json
import sqlite3
import warnings

import pytest

from eval import degenerate as D
from eval import private_labels
from eval.examples import FROZEN_EXAMPLES_PATH
from epic_sim.app.services import visibility

pytestmark = pytest.mark.reward_hacking

SPLIT = "public"


@pytest.fixture(scope="session")
def db():
    if not D.DEFAULT_DB.exists():
        pytest.skip(f"{D.DEFAULT_DB.name} not present")
    return D.ReleaseDB()


def _run(coro):
    return asyncio.run(coro)


async def _with_session(url, fn):
    """Call `fn(session)` against the release DB through the real async services."""
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
    engine = create_async_engine(url)
    try:
        async with async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)() as s:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return await fn(s)
    finally:
        await engine.dispose()


# ---------------------------------------------------------------------------
# 2.1 problem list = documented history
# ---------------------------------------------------------------------------

def test_problem_list_service_returns_documented_history_without_codes(db, orm_sqlite_url):
    from epic_sim.app.services import epic_service
    insts = db.instances("patient_diagnosis", SPLIT)[:25]
    pids = [i["patient_id"] for i in insts]

    async def go(s):
        return [await epic_service.get_problem_list(s, p) for p in pids]
    tool = _run(_with_session(orm_sqlite_url, go))
    for inst, problems in zip(insts, tool):
        profile = [str(c) for c in db.profile(inst["patient_id"]).get("chronic_conditions") or []]
        assert [p.display_name for p in problems] == profile
        assert all(p.icd10_code is None and p.snomed_id is None and p.diagnosis_id is None for p in problems)
        labels = {d["display_name"].lower() for d in inst["gt"]["active_diagnoses"]}
        # the only overlap allowed is a profile that itself names a tested diagnosis (a Stage 4 data defect)
        assert not ({p.display_name.lower() for p in problems} & labels - {c.lower() for c in profile})


def test_problems_from_profile_handles_shapes():
    ps = visibility.problems_from_profile({"chronic_conditions": ["Hypertension", {"name": "T2DM"}, "", None]})
    assert [p.display_name for p in ps] == ["Hypertension", "T2DM"] and ps[0].problem_id == "chart-0"
    assert visibility.problems_from_profile(None) == [] and visibility.problems_from_profile("not json") == []


# ---------------------------------------------------------------------------
# 2.2 point-in-time cutoff
# ---------------------------------------------------------------------------

def test_allowed_encounters_for_imaging_instance(db, orm_sqlite_url):
    inst = next(i for i in db.instances("imaging_indication", SPLIT)
                if db.q("select encounter_order from longitudinal_encounters where encounter_id=?", i["encounter_id"])[0][0] > 0)
    allowed = _run(_with_session(orm_sqlite_url, lambda s: visibility.allowed_encounter_ids(s, inst["gt_id"])))
    orders = dict(db.q("select encounter_id, encounter_order from longitudinal_encounters where patient_id=?", inst["patient_id"]))
    target = orders[inst["encounter_id"]]
    assert allowed == {e for e, o in orders.items() if o <= target}
    assert len(allowed) < len(orders)
    # index-encounter diagnosis instances are point-in-time too (Stage 4); patient-scoped units have no cutoff
    dx_inst = db.instances("patient_diagnosis", SPLIT)[0]
    assert _run(_with_session(orm_sqlite_url, lambda s: visibility.allowed_encounter_ids(s, dx_inst["gt_id"]))) == db.encounters_upto(dx_inst["patient_id"], dx_inst["encounter_id"])
    summ_inst = db.instances("context_summarization", SPLIT)[0]
    assert _run(_with_session(orm_sqlite_url, lambda s: visibility.allowed_encounter_ids(s, summ_inst["gt_id"]))) is None


def test_filter_future_encounters_covers_tool_and_fhir_shapes():
    allowed = {1, 2}
    tools = [{"encounter_id": 1, "x": 1}, {"encounter_id": 3, "x": 3}]
    assert visibility.filter_future_encounters(tools, allowed) == [{"encounter_id": 1, "x": 1}]
    nested = {"recent_encounters": tools, "sections": [{"encounter_id": 3, "section_type": "hpi"}]}
    out = visibility.filter_future_encounters(nested, allowed)
    assert out == {"recent_encounters": [{"encounter_id": 1, "x": 1}], "sections": []}
    bundle = {"resourceType": "Bundle", "entry": [
        {"resource": {"resourceType": "Encounter", "id": 3}},
        {"resource": {"resourceType": "DocumentReference", "encounter": {"reference": "Encounter/3"}}},
        {"resource": {"resourceType": "DocumentReference", "encounter": {"reference": "Encounter/2"}}},
    ]}
    assert len(visibility.filter_future_encounters(bundle, allowed)["entry"]) == 1


# ---------------------------------------------------------------------------
# 2.3 outcome sections hidden in every read path
# ---------------------------------------------------------------------------

def test_encounter_detail_and_section_hide_outcome_sections(db, orm_sqlite_url):
    from epic_sim.app.services import epic_service
    eid, sid = db.q("select encounter_id, id from encounter_ehr_sections where section_type='assessment' limit 1")[0]

    async def go(s):
        return (await epic_service.get_encounter_detail(s, eid, "attending"),
                await epic_service.get_encounter_section(s, sid, "attending"))
    detail, section = _run(_with_session(orm_sqlite_url, go))
    assert detail is not None and {x.section_type for x in detail.sections}.isdisjoint({"assessment", "plan"})
    assert section is None


def test_strip_outcome_sections_shapes():
    obj = {"sections": [{"section_type": "hpi", "t": 1}, {"section_type": "plan", "t": 2}], "assessment": "MI", "hpi": "ok"}
    assert visibility.strip_outcome_sections(obj) == {"sections": [{"section_type": "hpi", "t": 1}], "hpi": "ok"}


# ---------------------------------------------------------------------------
# 2.4 retrieval query is one diagnosis, judgments from the release rule
# ---------------------------------------------------------------------------

def test_retrieval_instances_are_per_diagnosis(db):
    insts = db.instances("evidence_retrieval", SPLIT)
    assert insts and all(len(i["gt"]["query_diagnoses"]) == 1 for i in insts)
    # the query names one of the patient's correct diagnoses, never the whole set (the Stage-1 leak)
    corr = collections.defaultdict(set)
    for pid, q in db.q("select patient_id, source_question_ids from longitudinal_encounters"):
        for qid in json.loads(q):
            corr[pid].update(d for (d,) in db.q("select diagnosis_id from question_diagnoses where question_id=? and role='correct'", qid))
    for i in insts:
        q = {d["diagnosis_id"] for d in i["gt"]["query_diagnoses"]}
        assert len(q) == 1 and q <= corr[i["patient_id"]]
        assert q < corr[i["patient_id"]] or len(corr[i["patient_id"]]) == 1
    superseded = db.q("select count(*) from benchmark_ground_truth where task='evidence_retrieval' and is_diagnostic=0 "
                      "and exclusion_reason like 'superseded%'")[0][0]
    assert superseded == 1268


def test_judgments_follow_the_content_grading_rule(db):
    """Stored judgments equal the content rule (scripts/regrade_retrieval_by_content.Graph) recomputed."""
    from scripts.regrade_retrieval_by_content import Graph
    conn = sqlite3.connect(f"file:{D.DEFAULT_DB}?mode=ro", uri=True)
    graph = Graph(conn)
    for inst in db.instances("evidence_retrieval", SPLIT)[:30]:
        d = inst["gt"]["query_diagnoses"][0]["diagnosis_id"]
        assert graph.grade(inst["patient_id"], d) == db.judgments(inst["gt_id"])


# ---------------------------------------------------------------------------
# 2.6 few-shot examples come from train
# ---------------------------------------------------------------------------

def test_frozen_few_shot_examples_are_train_only(db):
    doc = json.loads(FROZEN_EXAMPLES_PATH.read_text())
    meta = doc["_meta"]
    assert meta["split"] == "train"
    split = dict(db.q("select distinct patient_id, split from benchmark_ground_truth where patient_id is not null"))
    for task, pids in meta["patient_ids"].items():
        assert pids and all(split[p] == "train" for p in pids), task
        assert doc[task].count("Example ") >= 1
    # and no evaluation-split chief complaint appears in any block
    eval_cc = {cc for (cc,) in db.q("select distinct l.chief_complaint from longitudinal_encounters l join benchmark_ground_truth b "
                                    "on b.patient_id=l.patient_id where b.split in ('public','heldout','private') and length(l.chief_complaint)>40")}
    blob = " ".join(v for k, v in doc.items() if k != "_meta")
    assert not any(cc in blob for cc in eval_cc)


# ---------------------------------------------------------------------------
# 2.7 private split: labels are not in the release
# ---------------------------------------------------------------------------

def test_private_split_has_no_labels_in_release(db):
    rows = db.q("select task, ground_truth from benchmark_ground_truth where split='private'")
    assert len(rows) > 0
    for task, gt in rows:
        gt = json.loads(gt)
        assert gt.get(private_labels.REMOVED_FLAG) is True
        assert not any(k in gt for k in private_labels.LABEL_KEYS[task])
    assert db.q("select count(*) from relevance_judgments r join benchmark_ground_truth b using(gt_id) where b.split='private'")[0][0] == 0
    # inputs survive: retrieval query, clinical question, specialty name
    assert all("query_diagnoses" in json.loads(g) for (g,) in db.q("select ground_truth from benchmark_ground_truth where split='private' and task='evidence_retrieval' and is_diagnostic"))
    assert all("specialty" in json.loads(g) for (g,) in db.q("select ground_truth from benchmark_ground_truth where split='private' and json_extract(ground_truth,'$.variant')='specialty_conditioned'"))


def test_private_overlay_restores_labels_when_present(db):
    overlay = private_labels.get()
    if overlay is None:
        pytest.skip("private label overlay not present (operator-only)")
    ids = overlay.gt_ids()
    release = {g for (g,) in db.q("select gt_id from benchmark_ground_truth where split='private'")}
    assert ids == release
    gt = overlay.ground_truth(min(ids))
    assert gt and not gt.get(private_labels.REMOVED_FLAG)
    insts = db.instances("patient_diagnosis", "private")
    assert insts and D.primary(db, "patient_diagnosis", [D.oracle(db, insts[0])], insts[:1]) == pytest.approx(1.0)


def test_strip_labels_keeps_inputs():
    gt = {"query_diagnoses": [{"diagnosis_id": 1}], "num_passages": 3, "grade_distribution": {}}
    out = private_labels.strip_labels("evidence_retrieval", gt)
    assert out == {"query_diagnoses": [{"diagnosis_id": 1}], private_labels.REMOVED_FLAG: True}


# ---------------------------------------------------------------------------
# 2.8 reward only for the private split
# ---------------------------------------------------------------------------

def test_score_response_is_redacted_for_private():
    from epic_sim.app.routers.score import redact_metrics
    r = {"split": "private", "reward": 0.5, "metrics": {"recall": 1.0}}
    assert redact_metrics(r)["metrics"] == {} and redact_metrics(r)["reward"] == 0.5
    assert redact_metrics({**r, "split": "public"})["metrics"] == {"recall": 1.0}
