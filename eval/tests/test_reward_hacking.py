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
    "content_blind_headroom_share": 0.5,   # a content-blind ranking may capture at most half the headroom above random
    "chart_dump_summary": 0.5,         # pasting the chart must score below half the ceiling
    "negated_recall": 0.0,             # "denies fever" must not satisfy "Fever"
    "phrase_abstention": 0.1,          # a stock phrase inside a full summary must not count as abstaining
    "name_classifier_margin": 0.02,    # a specialty-name classifier may not beat the majority class by more
    "copy_problem_list": 0.25,         # copying the chart's own problem list must not solve diagnosis
    "tool_echo_margin": 0.05,          # the problem-list tool must reveal nothing beyond the profile
    "documented_truth": 0.95,          # reference + documented comorbidities must score ~1
    "bare_category_credit": 0.9,       # 3-character categories must not earn full credit
    "hallucination_flagged": 0.8,      # invented sentences must be flagged (a few charts do carry PE / fracture terms)
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
                                  "specialty_involved", "specialty_absent", "imaging_indication"])
def test_empty_submission_scores_zero(floors, task):
    assert _pol(floors, task, "empty") == 0.0


def test_empty_submission_is_not_an_abstention(floors):
    # Stage 3: abstention is the explicit `abstain` field. An empty answer scores 0 everywhere; the
    # constant explicit abstention still collects the whole absent unit (a binary item has no other
    # correct answer), so that unit has no headroom by construction and is never normalized.
    assert _pol(floors, "specialty_absent", "empty") == 0.0
    assert _pol(floors, "specialty_absent", "abstain_always") == 1.0
    assert _pol(floors, "specialty_involved", "abstain_always") == 0.0
    assert F.normalize(0.5, "specialty_absent", floors={"splits": {SPLIT: floors}}) is None


def test_empty_diagnosis_prediction_is_counted(db):
    inst = db.instances("patient_diagnosis", SPLIT)[0]
    both = D.primary(db, "patient_diagnosis", [D.oracle(db, inst), {}], [inst, inst])
    assert both == pytest.approx(GATES["empty_counted"], abs=0.01)


# ---------------------------------------------------------------------------
# evidence retrieval
# ---------------------------------------------------------------------------

def test_single_passage_cannot_saturate_p5(floors):
    assert _pol(floors, "evidence_retrieval", "single_hpi") <= GATES["single_passage_p5"]


@pytest.mark.parametrize("metric", ["precision_5", "ndcg_10"])
def test_content_blind_ranking_captures_little_headroom(floors, metric):
    """A ranking that never reads a section (train-split mean grade per section type) may capture
    at most half of the headroom above random. Before Stage 3 it captured 59% on nDCG@10 and sat at
    the ceiling on P@5 (0.97 vs 0.999), because grades were a function of section type (#22)."""
    e = _unit(floors, "evidence_retrieval", metric)
    rnd, typ = e["policies"]["random"], e["policies"]["section_type_prior"]
    share = (typ - rnd) / (e["ceiling"] - rnd) if e["ceiling"] > rnd else 1.0
    assert share <= GATES["content_blind_headroom_share"], f"type prior {typ:.3f}, random {rnd:.3f}, ceiling {e['ceiling']:.3f}: share {share:.2f}"


def test_random_ranking_floor_is_recorded(floors):
    e = _unit(floors, "evidence_retrieval", "ndcg_10")
    assert "random" in e["policies"] and 0 < e["policies"]["random"] < e["ceiling"]


# ---------------------------------------------------------------------------
# summarization (whole patient)
# ---------------------------------------------------------------------------

def test_chart_dump_is_not_a_good_summary(floors):
    assert _pol(floors, "context_summarization", "chart_dump") <= GATES["chart_dump_summary"] * _unit(floors, "context_summarization")["ceiling"]


def test_structured_prompts_carry_no_graph_concepts(db):
    """Stage 2.5: the 'structured' strategy may add ontology *guidance*, never patient-specific
    concepts from the label graph. Exercised on the real prompt builders with inputs that carry
    the graph fields the old loaders used to fill."""
    from eval.tasks import imaging, patient_diagnosis, summarization

    inst = db.instances("context_summarization", SPLIT)[0]
    pid = inst["patient_id"]
    key = db.key_finding_names(pid)[:10]
    must = [f["display_name"] for f in inst["gt"]["must_include_findings"]]
    s_inp = summarization.SummarizationInput(
        gt_id=inst["gt_id"], patient_id=pid, clinical_question="q", must_include_findings=inst["gt"]["must_include_findings"],
        ehr_text="RECORD", ground_truth=inst["gt"], variant="unconditioned",
        key_findings_hint=[{"name": n, "type": "symptom"} for n in key])
    _, user = summarization.format_prompt(s_inp, "structured")
    assert "RECORD" in user and not any(n in user for n in must + key)

    d_inst = db.instances("patient_diagnosis", SPLIT)[0]
    d_inp = patient_diagnosis.PatientDiagnosisInput(
        gt_id=d_inst["gt_id"], patient_id=d_inst["patient_id"], ehr_text="RECORD", ehr_token_estimate=1,
        ground_truth=d_inst["gt"], organ_systems=["Cardiovascular System"],
        key_findings=[{"name": "Polyuria", "snomed_id": "28442001", "type": "symptom"}])
    _, user = patient_diagnosis.format_prompt(d_inp, "structured")
    assert "Polyuria" not in user and "28442001" not in user and "I00" not in user

    i_inst = db.instances("imaging_indication", SPLIT)[0]
    i_inp = imaging.ImagingInput(
        gt_id=i_inst["gt_id"], encounter_id=i_inst["encounter_id"], patient_id=i_inst["patient_id"], modality="ct",
        body_region="abdomen", clinical_indication="pain", ehr_text="RECORD", ground_truth=i_inst["gt"],
        relevant_findings_with_loinc=[{"name": "Leukocytosis", "loinc_code": "6690-2", "type": "lab_value"}],
        differential_icd10_codes=[{"name": "Acute appendicitis", "icd10_code": "K35.80", "role": "correct"}])
    _, user = imaging.format_prompt(i_inp, "structured")
    assert "Acute appendicitis" not in user and "K35.80" not in user and "Leukocytosis" not in user


def test_negated_finding_is_not_credited(db):
    gt = {"must_include_findings": [{"display_name": n} for n in D.NEGATED_FINDINGS["must_include"]]}
    inst = D.Instance(gt_id=-1, task="context_summarization", patient_id=-1, encounter_id=None, split=SPLIT, gt=gt)
    assert D.primary(db, "context_summarization", [{"summary": D.NEGATED_FINDINGS["summary"]}], [inst]) == GATES["negated_recall"]


# ---------------------------------------------------------------------------
# specialty-conditioned summarization
# ---------------------------------------------------------------------------

def test_stock_phrase_in_full_summary_is_not_abstention(floors, db):
    assert not _is_abstention(db.chart_text(db.instances("specialty_absent", SPLIT)[0]["patient_id"]) + "\n" + D.ABSTAIN_PHRASE)
    assert _pol(floors, "specialty_absent", "chart_dump_plus_phrase") <= GATES["phrase_abstention"]


def test_stock_phrase_does_not_help_involved_items(floors):
    # appending the phrase must not raise the involved score (it used to be free, and bought the absent unit)
    assert _pol(floors, "specialty_involved", "chart_dump_plus_phrase") <= _pol(floors, "specialty_involved", "chart_dump") + 1e-9


def test_specialty_name_does_not_predict_absence(db):
    """A classifier that abstains iff the specialty name has a train-split absent rate above 0.5 must
    not beat the majority class. Absent specialties are drawn in proportion to how often each
    specialty is involved, so P(absent | name) stays near the base rate."""
    inv, ab = db.instances("specialty_involved", SPLIT), db.instances("specialty_absent", SPLIT)
    rate = {s: r for s, r in db.q("select json_extract(ground_truth,'$.specialty'), avg(json_extract(ground_truth,'$.involvement')='absent') "
                                  "from benchmark_ground_truth where task='context_summarization' and split='train' "
                                  "and json_extract(ground_truth,'$.variant')='specialty_conditioned' group by 1")}
    abstain_on = {s for s, r in rate.items() if r > 0.5}
    correct = sum(i["gt"]["specialty"] in abstain_on for i in ab) + sum(i["gt"]["specialty"] not in abstain_on for i in inv)
    majority = max(len(inv), len(ab)) / (len(inv) + len(ab))
    assert correct / (len(inv) + len(ab)) <= majority + GATES["name_classifier_margin"], (abstain_on, correct / (len(inv) + len(ab)), majority)


# ---------------------------------------------------------------------------
# patient diagnosis
# ---------------------------------------------------------------------------

def test_copying_the_problem_list_does_not_solve_diagnosis(floors):
    # Stage 4: the task is index-encounter diagnosis and the chart up to the index visit never names its
    # diagnosis; copying the problem list yields earlier (chart-neutral) diagnoses only.
    assert _pol(floors, "patient_diagnosis", "copy_problem_list") <= GATES["copy_problem_list"]


def test_problem_list_tool_reveals_nothing_beyond_the_profile(floors):
    # Fixed in Stage 2.1: the tool returns the chart's documented history (profile conditions), which
    # carries no codes, so echoing it can only score what the profile-only policy scores.
    tool = _pol(floors, "patient_diagnosis", "echo_problem_list_tool")
    profile = _pol(floors, "patient_diagnosis", "profile_chronic")
    assert tool <= profile + GATES["tool_echo_margin"], f"tool echo {tool:.3f} vs profile-only {profile:.3f}"


def test_documented_comorbidities_are_not_penalized(db):
    insts = db.instances("patient_diagnosis", SPLIT)
    assert D.primary(db, "patient_diagnosis", [D.dx_reference_plus_secondary(db, i) for i in insts], insts) >= GATES["documented_truth"]


def test_bare_categories_do_not_earn_full_credit(db):
    insts = db.instances("patient_diagnosis", SPLIT)
    assert D.primary(db, "patient_diagnosis", [D.dx_reference_categories(db, i) for i in insts], insts) < GATES["bare_category_credit"]


# ---------------------------------------------------------------------------
# imaging indication and the hallucination metric
# ---------------------------------------------------------------------------

def test_imaging_floor_is_recorded(floors):
    e = _unit(floors, "imaging_indication")
    assert e["floor_policy"] in ("restate_order", "chief_complaint") and e["floor"] < 0.5


def test_invented_sentences_are_flagged(db):
    insts = db.instances("context_summarization", SPLIT)
    charts = [db.chart_text(i["patient_id"]) for i in insts]
    terms = [db.patient_terms(i["patient_id"]) for i in insts]
    rate = hallucination_rate([D.INVENTED_SENTENCES] * len(charts), charts, concept_extractor=db.concept_extractor(),
                              patient_terms=terms)
    assert rate >= GATES["hallucination_flagged"], rate


# ---------------------------------------------------------------------------
# normalization
# ---------------------------------------------------------------------------

def test_normalized_score_is_zero_at_floor_and_one_at_ceiling(floors):
    doc = {"splits": {SPLIT: floors}}
    for task in ("patient_diagnosis", "evidence_retrieval", "context_summarization", "imaging_indication"):
        e = _unit(floors, task)
        assert F.normalize(e["floor"], task, floors=doc) == pytest.approx(0.0)
        assert F.normalize(e["ceiling"], task, floors=doc) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Stage 3 terms: graded credit, acuity, per-item = batch, grounding, quota, content grades, abstain field
# ---------------------------------------------------------------------------

def test_icd_credit_is_graded():
    from eval.scoring import _icd_credit
    assert _icd_credit("E11.01", "E11.01") == 1.0
    assert _icd_credit("E11.00", "E11.01") == 0.75      # same subcategory E11.0
    assert _icd_credit("E11.9", "E11.01") == 0.5        # same category only
    assert _icd_credit("E11.65", "E11.9") == 1.0        # unspecified reference: any E11 is full credit
    assert _icd_credit("I10", "I10") == 1.0 and _icd_credit("I11.0", "I10") == 0.0
    # injury codes key on 5 characters: S44.21 and S44.22 are different injuries; the 7th character
    # (encounter) is the subcategory tier
    assert _icd_credit("S44.21XD", "S44.21XA") == 0.75 and _icd_credit("S44.21XA", "S44.22XA") == 0.0
    assert _icd_credit("S44.21XA", "S54.21XA") == 0.0


def test_acuity_is_scored():
    from eval.scoring import score_patient_diagnosis_item
    gt = {"active_diagnoses": [{"icd10": "J96.02", "acuity": "acute"}], "chronic_conditions": []}
    right = score_patient_diagnosis_item({"active_diagnoses": [{"icd10": "J96.02", "acuity": "acute"}]}, gt)
    wrong = score_patient_diagnosis_item({"active_diagnoses": [{"icd10": "J96.02", "acuity": "chronic"}]}, gt)
    missing = score_patient_diagnosis_item({"active_diagnoses": [{"icd10": "J96.02"}]}, gt)
    assert right["weighted_problem_list_f1_neutral"] == 1.0
    assert wrong["weighted_problem_list_f1_neutral"] == pytest.approx(0.5)
    assert missing["weighted_problem_list_f1_neutral"] == pytest.approx(0.5)


@pytest.mark.parametrize("task", ["patient_diagnosis", "specialty_involved", "context_summarization", "imaging_indication"])
def test_single_item_reward_equals_batch_mean(db, task):
    insts = db.instances(task, SPLIT)[:12]
    preds = [D.oracle(db, i) if k % 2 else {} for k, i in enumerate(insts)]
    per = D.per_item(db, task, preds, insts)
    assert D.primary(db, task, preds, insts) == pytest.approx(sum(per) / len(per), abs=1e-9)


def test_summary_oracle_is_fully_grounded(db):
    insts = db.instances("context_summarization", SPLIT)[:40]
    m = D.score(db, "context_summarization", [D.oracle(db, i) for i in insts], insts)
    assert m["clinical_f1"] == pytest.approx(1.0) and m["grounded_precision"] == pytest.approx(1.0)


def test_long_summary_is_discounted(db):
    inst = db.instances("context_summarization", SPLIT)[0]
    short = D.oracle(db, inst)["summary"]
    padded = short + " " + " ".join(["The patient was seen in clinic."] * 200)   # grounded filler, > budget
    a = D.primary(db, "context_summarization", [{"summary": short}], [inst])
    b = D.primary(db, "context_summarization", [{"summary": padded}], [inst])
    assert a == pytest.approx(1.0) and b < 0.4


def test_must_include_covers_every_encounter_when_possible(db):
    orders = dict(db.q("select patient_id, max(encounter_order) from longitudinal_encounters group by 1"))
    for inst in db.instances("context_summarization", SPLIT):
        got = {f.get("encounter_order") for f in inst["gt"]["must_include_findings"]}
        assert orders[inst["patient_id"]] in got, inst["gt_id"]


def test_retrieval_grades_depend_on_content(db):
    """Byte-identical sections within an instance carry the same grade (they did not: #22)."""
    stype_text = {f"ees_{i}": t for i, t in db.q("select id, section_text from encounter_ehr_sections")}
    for inst in db.instances("evidence_retrieval", SPLIT)[:300]:
        by_text = {}
        for pid, g in db.judgments(inst["gt_id"]).items():
            by_text.setdefault(stype_text[pid], set()).add(g)
        assert all(len(v) == 1 for v in by_text.values()), inst["gt_id"]


def test_abstention_requires_the_field(db):
    insts = db.instances("specialty_absent", SPLIT)[:20]
    text_only = [{"summary": "No active problems in this specialty."} for _ in insts]
    flagged = [{"summary": "", "abstain": True} for _ in insts]
    assert D.primary(db, "specialty_absent", text_only, insts) == 0.0
    assert D.primary(db, "specialty_absent", flagged, insts) == 1.0
    inv = db.instances("specialty_involved", SPLIT)[:20]
    assert D.primary(db, "specialty_involved", flagged, inv) == 0.0      # wrong abstention scores 0
