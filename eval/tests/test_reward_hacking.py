"""Reward-hacking regression suite (audit/ROADMAP.md Stage 1).

Every test runs a degenerate policy (eval/degenerate.py) through the repo's own scorer on the
released database and asserts a gate. Gates that the current scorer or data fail are marked
`xfail(strict=True)` with the roadmap stage that fixes them: the suite is green today, and when a
fix lands the test XPASSes, strict mode turns that into a failure, and the marker must be removed.
So a fix cannot land without being recorded here, and a regression cannot land silently.

Runs in-process on benchmark_v1.3.db: no Postgres, no simulator, no LLM.  ~1 minute.

    pytest eval/tests/test_reward_hacking.py -m reward_hacking -q
"""

from __future__ import annotations

import pytest

from eval import degenerate as D
from eval import floors as F
from eval.scoring import _is_abstention, hallucination_rate

pytestmark = pytest.mark.reward_hacking

SPLIT = "public"
S2 = "ROADMAP Stage 2 (close label leaks)"
S3 = "ROADMAP Stage 3 (harden reward functions)"
S4 = "ROADMAP Stage 4 (repair data / redefine patient diagnosis)"

# Gates. Numbers are what a sound metric must satisfy, not what the scorer does today.
GATES = {
    "single_passage_p5": 0.2,          # one submitted passage can fill at most 1 of 5 slots
    "content_blind_margin": 0.05,      # a content-blind ranking may beat random by at most this
    "chart_dump_summary": 0.5,         # pasting the chart must score below half the ceiling
    "negated_recall": 0.0,             # "denies fever" must not satisfy "Fever"
    "phrase_abstention": 0.1,          # a stock phrase inside a full summary must not count as abstaining
    "absent_by_name_accuracy": 0.65,   # the specialty name alone must not predict absent/involved
    "copy_problem_list": 0.25,         # copying the chart's own problem list must not solve diagnosis
    "tool_echo_margin": 0.05,          # the problem-list tool must reveal nothing beyond the profile
    "documented_truth": 0.95,          # reference + documented comorbidities must score ~1
    "bare_category_credit": 0.9,       # 3-character categories must not earn full credit
    "hallucination_flagged": 0.9,      # invented sentences must be flagged
    "empty_counted": 0.5,              # [perfect, empty] must average to 0.5, not 1.0
}


# ---------------------------------------------------------------------------
# fixtures (one pass over the public split, shared by every test)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def db():
    if not D.DEFAULT_DB.exists():
        pytest.skip(f"{D.DEFAULT_DB.name} not present")
    return D.ReleaseDB()


@pytest.fixture(scope="session")
def floors(db):
    return F.compute_floors(db, SPLIT)


def _unit(floors, task, metric=None):
    return floors[task]["metrics"][metric or F.METRICS[task][0]]


def _pol(floors, task, name, metric=None):
    return _unit(floors, task, metric)["policies"][name]


# ---------------------------------------------------------------------------
# ceilings and the committed floors file
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("task", ["patient_diagnosis", "context_summarization", "specialty_absent", "imaging_indication"])
def test_oracle_reaches_ceiling(floors, task):
    assert _unit(floors, task)["ceiling"] == pytest.approx(1.0, abs=1e-9)


def test_retrieval_oracle_near_ceiling(floors):
    # a patient with fewer than 5 relevant sections caps the oracle's P@5 below 1
    assert _unit(floors, "evidence_retrieval")["ceiling"] >= 0.99


@pytest.mark.xfail(strict=True, reason=f"{S3}: 540 involved items have no critical finding and score 0 for any answer (#23)")
def test_specialty_oracle_scores_one_on_every_item(db):
    insts = db.instances("specialty_involved", SPLIT)
    rewards = D.per_item(db, "specialty_involved", [D.oracle(db, i) for i in insts], insts)
    assert min(rewards) == pytest.approx(1.0), f"{sum(r < 1 for r in rewards)} unwinnable items"


def test_committed_floors_are_current(floors):
    stored = F.load().get("splits", {}).get(SPLIT, {})
    diffs = F.matches(floors, stored)
    assert not diffs, "eval/floors.json is stale; run `python -m eval.floors --split public --write`:\n" + "\n".join(diffs)


# ---------------------------------------------------------------------------
# empty / malformed submissions
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("task", ["patient_diagnosis", "evidence_retrieval", "context_summarization",
                                  "specialty_involved", "imaging_indication"])
def test_empty_submission_scores_zero(floors, task):
    assert _pol(floors, task, "empty") == 0.0


def test_empty_submission_is_a_free_abstention(floors):
    # Documented, not gated: on absent items an empty answer is the correct answer, so a policy that
    # outputs nothing collects the whole absent unit. The unit therefore has no headroom.
    assert _pol(floors, "specialty_absent", "empty") == 1.0
    assert F.normalize(0.5, "specialty_absent", floors={"splits": {SPLIT: floors}}) is None


@pytest.mark.xfail(strict=True, reason=f"{S3}: _compute_patient_diagnosis_metrics skips empty predictions instead of scoring 0 (#19)")
def test_empty_diagnosis_prediction_is_counted(db):
    inst = db.instances("patient_diagnosis", SPLIT)[0]
    both = D.primary(db, "patient_diagnosis", [D.oracle(db, inst), {}], [inst, inst])
    assert both == pytest.approx(GATES["empty_counted"], abs=0.01)


# ---------------------------------------------------------------------------
# evidence retrieval
# ---------------------------------------------------------------------------

@pytest.mark.xfail(strict=True, reason=f"{S3}: precision_at_k divides by min(k, len(results)) (#1)")
def test_single_passage_cannot_saturate_p5(floors):
    assert _pol(floors, "evidence_retrieval", "single_hpi") <= GATES["single_passage_p5"]


@pytest.mark.xfail(strict=True, reason=f"{S3}: grades are a function of section type, not text (#22)")
@pytest.mark.parametrize("metric", ["precision_5", "ndcg_10"])
def test_content_blind_ranking_is_no_better_than_random(floors, metric):
    rnd = _pol(floors, "evidence_retrieval", "random", metric)
    typ = _pol(floors, "evidence_retrieval", "section_type_prior", metric)
    assert typ <= rnd + GATES["content_blind_margin"], f"type prior {typ:.3f} vs random {rnd:.3f}"


def test_random_ranking_floor_is_recorded(floors):
    e = _unit(floors, "evidence_retrieval", "ndcg_10")
    assert "random" in e["policies"] and 0 < e["policies"]["random"] < e["ceiling"]


# ---------------------------------------------------------------------------
# summarization (whole patient)
# ---------------------------------------------------------------------------

@pytest.mark.xfail(strict=True, reason=f"{S3}: clinical_f1 is recall-only; a chart dump scores ~0.67")
def test_chart_dump_is_not_a_good_summary(floors):
    assert _pol(floors, "context_summarization", "chart_dump") <= GATES["chart_dump_summary"] * _unit(floors, "context_summarization")["ceiling"]


@pytest.mark.xfail(strict=True, reason=f"{S2}: the structured prompt injects the patient's key findings, which are the scored must-include names (#12)")
def test_structured_hints_do_not_contain_scored_findings(db):
    insts = db.instances("context_summarization", SPLIT)
    overlap = n = 0
    for i in insts:
        hinted = set(db.key_finding_names(i["patient_id"])[:10])
        must = {f["display_name"] for f in i["gt"]["must_include_findings"]}
        overlap += len(hinted & must); n += len(hinted)
    assert overlap == 0, f"{overlap}/{n} hinted names are must-include findings"


@pytest.mark.xfail(strict=True, reason=f"{S3}: the whole-patient matcher (phrase_in_text) has no negation guard (F§13)")
def test_negated_finding_is_not_credited(db):
    gt = {"must_include_findings": [{"display_name": n} for n in D.NEGATED_FINDINGS["must_include"]]}
    inst = D.Instance(gt_id=-1, task="context_summarization", patient_id=-1, encounter_id=None, split=SPLIT, gt=gt)
    assert D.primary(db, "context_summarization", [{"summary": D.NEGATED_FINDINGS["summary"]}], [inst]) == GATES["negated_recall"]


# ---------------------------------------------------------------------------
# specialty-conditioned summarization
# ---------------------------------------------------------------------------

@pytest.mark.xfail(strict=True, reason=f"{S3}: _is_abstention fires on a stock phrase anywhere in the text (#4)")
def test_stock_phrase_in_full_summary_is_not_abstention(floors, db):
    assert not _is_abstention(db.chart_text(db.instances("specialty_absent", SPLIT)[0]["patient_id"]) + "\n" + D.ABSTAIN_PHRASE)
    assert _pol(floors, "specialty_absent", "chart_dump_plus_phrase") <= GATES["phrase_abstention"]


def test_stock_phrase_does_not_change_involved_score(floors):
    # the exploit is free: appending the phrase costs nothing on involved items
    assert _pol(floors, "specialty_involved", "chart_dump_plus_phrase") == pytest.approx(_pol(floors, "specialty_involved", "chart_dump"), abs=1e-9)


@pytest.mark.xfail(strict=True, reason=f"{S3}: absent specialties are the first two alphabetically, so the name predicts the label")
def test_specialty_name_does_not_predict_absence(db):
    inv, ab = db.instances("specialty_involved", SPLIT), db.instances("specialty_absent", SPLIT)
    names = set(db.most_absent_specialties("train"))
    correct = sum(i["gt"]["specialty"] in names for i in ab) + sum(i["gt"]["specialty"] not in names for i in inv)
    assert correct / (len(inv) + len(ab)) <= GATES["absent_by_name_accuracy"]


# ---------------------------------------------------------------------------
# patient diagnosis
# ---------------------------------------------------------------------------

@pytest.mark.xfail(strict=True, reason=f"{S4}: 80% of reference diagnoses appear verbatim in later problem lists (#2)")
def test_copying_the_problem_list_does_not_solve_diagnosis(floors):
    assert _pol(floors, "patient_diagnosis", "copy_problem_list") <= GATES["copy_problem_list"]


@pytest.mark.xfail(strict=True, reason=f"{S2}: epic_service.get_problem_list returns the reference diagnoses (#10)")
def test_problem_list_tool_reveals_nothing_beyond_the_profile(floors):
    tool = _pol(floors, "patient_diagnosis", "echo_problem_list_tool")
    profile = _pol(floors, "patient_diagnosis", "profile_chronic")
    assert tool <= profile + GATES["tool_echo_margin"], f"tool echo {tool:.3f} vs profile-only {profile:.3f}"


@pytest.mark.xfail(strict=True, reason=f"{S3}: the chart-neutral set covers profile conditions only, not documented secondary diagnoses (#13)")
def test_documented_comorbidities_are_not_penalized(db):
    insts = db.instances("patient_diagnosis", SPLIT)
    assert D.primary(db, "patient_diagnosis", [D.dx_reference_plus_secondary(db, i) for i in insts], insts) >= GATES["documented_truth"]


@pytest.mark.xfail(strict=True, reason=f"{S3}: _match_icd10_sets gives full credit at the 3-character category")
def test_bare_categories_do_not_earn_full_credit(db):
    insts = db.instances("patient_diagnosis", SPLIT)
    assert D.primary(db, "patient_diagnosis", [D.dx_reference_categories(db, i) for i in insts], insts) < GATES["bare_category_credit"]


# ---------------------------------------------------------------------------
# imaging indication and the hallucination metric
# ---------------------------------------------------------------------------

def test_imaging_floor_is_recorded(floors):
    e = _unit(floors, "imaging_indication")
    assert e["floor_policy"] in ("restate_order", "chief_complaint") and e["floor"] < 0.5


@pytest.mark.xfail(strict=True, reason=f"{S3}: hallucination_rate is token overlap against the whole chart; invented sentences score 0 (#14)")
def test_invented_sentences_are_flagged(db):
    charts = [db.chart_text(i["patient_id"]) for i in db.instances("context_summarization", SPLIT)]
    assert hallucination_rate([D.INVENTED_SENTENCES] * len(charts), charts) >= GATES["hallucination_flagged"]


# ---------------------------------------------------------------------------
# normalization
# ---------------------------------------------------------------------------

def test_normalized_score_is_zero_at_floor_and_one_at_ceiling(floors):
    doc = {"splits": {SPLIT: floors}}
    for task in ("patient_diagnosis", "evidence_retrieval", "context_summarization", "imaging_indication"):
        e = _unit(floors, task)
        assert F.normalize(e["floor"], task, floors=doc) == pytest.approx(0.0)
        assert F.normalize(e["ceiling"], task, floors=doc) == pytest.approx(1.0)
