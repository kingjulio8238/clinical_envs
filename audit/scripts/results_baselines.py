"""§5 comparison: zero-model baselines on the exact patient subsets behind Tables 2 and 3.

Subsets
  public      all 200 public patients (Table 2 models)
  agentic100  the Table 3 selection rule (eval/agents/runner.py:select_patients): public patients
              sorted by descending encounter count, first 100 -- before its ceiling exclusion,
              which needs the unreleased evaluation_predictions table
  phys13      the 13 physician-study patients (scripts/retrieval_sections_only.py:IDS)

Run from the repo root:  .venv/bin/python audit/scripts/results_baselines.py
"""
import collections
import importlib.util
import json
import re
import sqlite3
import sys

sys.path.insert(0, ".")
from eval.imaging_concepts import ConceptExtractor  # noqa: E402
from eval.scoring import compute_all_metrics  # noqa: E402

c = sqlite3.connect("benchmark_v1.3.db")
q = lambda sql, *a: c.execute(sql, a).fetchall()  # noqa: E731

PUBLIC = [p for (p,) in q("select patient_id from benchmark_ground_truth where task='patient_diagnosis' and split='public'")]
NENC = dict(q("select patient_id, num_encounters from longitudinal_patients"))
SUBSETS = {
    "public": set(PUBLIC),
    "agentic100": set(sorted(PUBLIC, key=lambda p: (-NENC[p], p))[:100]),
    "phys13": {2610, 2850, 1726, 2046, 2834, 2285, 1853, 2631, 2323, 1969, 1741, 2549, 2376},
}


class _Cur:  # sqlite cursor shim tolerant of the absent terminology table
    def __init__(self): self.c = c.cursor()
    def execute(self, sql, *a):
        try: self.c.execute(sql, *a)
        except sqlite3.OperationalError: self.c.execute("select 1 where 0")
    def fetchall(self): return self.c.fetchall()


spec = importlib.util.spec_from_file_location("cns", "scripts/chart_neutral_sets.py")
cns = importlib.util.module_from_spec(spec); spec.loader.exec_module(cns)
NEUTRAL, _, _ = cns.neutral_sets(_Cur())
NAME2ICD = {}
for code, n in q("select icd10_code, display_name from diagnoses where icd10_code is not null"):
    NAME2ICD.setdefault(n.lower(), code)
CHART = collections.defaultdict(str)
for pid, t in q("select patient_id, note_text from longitudinal_encounters order by patient_id, encounter_order"):
    CHART[pid] += t + "\n"
PROFILE = {pid: json.loads(p) for pid, p in q("select patient_id, profile from longitudinal_patients")}
STYPE = {f"ees_{i}": t for i, t in q("select id, section_type from encounter_ehr_sections")}
PRIOR = {"hpi": 2.58, "pathology": 2.44, "imaging": 2.37, "other_studies": 2.3, "physical_exam": 2.26, "ros": 2.19,
         "chief_complaint": 2.19, "labs": 2.12, "pmh": 1.89, "family_history": 1.89, "social_history": 1.89,
         "psh": 1.88, "vitals": 1.69, "medications": 0.73, "allergies": 0.68}  # train-split mean grade (type_baseline.py)
inv = []
for did, sn, dn, sd in q("select diagnosis_id, snomed_id, display_name, snomed_desc from diagnoses"):
    inv += [(f"S{sn}" if sn else f"D{did}", dn), (f"S{sn}" if sn else f"D{did}", sd)]
for fid, sn, dn, sd in q("select finding_id, snomed_id, display_name, snomed_desc from clinical_findings where finding_type<>'demographic'"):
    inv += [(f"S{sn}" if sn else f"F{fid}", dn), (f"S{sn}" if sn else f"F{fid}", sd)]
EXT = ConceptExtractor(inv)
ORDER = dict(q("select gt_id, clinical_indication from imaging_orders"))


def rows(task, pids, where="1"):
    return [(g, p, e, json.loads(t)) for g, p, e, t in
            q(f"select b.gt_id, coalesce(b.patient_id, l.patient_id), b.encounter_id, b.ground_truth "
              f"from benchmark_ground_truth b left join longitudinal_encounters l using(encounter_id) "
              f"where b.task=? and {where}", task) if p in pids]


def copy_dx(pid):
    act = [{"icd10": NAME2ICD[n.lower()], "acuity": "acute"} for n in
           set(re.findall(r"^- (.+?) \(diagnosed \d{4}-\d\d-\d\d\)", CHART[pid], re.M)) if n.lower() in NAME2ICD]
    chron = [{"icd10": sorted(cns.map_condition(str(s))[0])[0]} for s in PROFILE[pid].get("chronic_conditions") or []
             if cns.map_condition(str(s))[0]]
    return {"active_diagnoses": act, "chronic_conditions": chron}


def with_neutral(p, g):
    return dict(g, _neutral_categories=NEUTRAL.get(p, []))


for name, pids in SUBSETS.items():
    out = {"n_patients": len(pids)}
    R = rows("patient_diagnosis", pids)
    m = compute_all_metrics("patient_diagnosis", [copy_dx(p) for _, p, _, _ in R], [with_neutral(p, g) for _, p, _, g in R])
    out["dx_copy_chart"] = round(m["weighted_problem_list_f1_neutral"], 3)
    # the agent harness's view_problem_list returns exactly the reference diagnoses (epic_service.get_problem_list)
    m = compute_all_metrics("patient_diagnosis", [{"active_diagnoses": [{"icd10": d["icd10"]} for d in g.get("active_diagnoses", []) + g.get("chronic_conditions", [])]}
                                                  for _, _, _, g in R], [with_neutral(p, g) for _, p, _, g in R])
    out["dx_echo_problem_list_tool"] = round(m["weighted_problem_list_f1_neutral"], 3)
    R = rows("context_summarization", pids, "json_extract(b.ground_truth,'$.variant') is null")
    m = compute_all_metrics("context_summarization", [{"summary": CHART[p]} for _, p, _, _ in R], [g for *_, g in R])
    out["summ_dump_chart"] = round(m["clinical_f1"], 3)
    R = rows("evidence_retrieval", pids)
    P, G = [], []
    for gid, p, _, _ in R:
        j = dict(q("select passage_id, relevance_grade from relevance_judgments where gt_id=?", gid))
        P.append({"rankings": [{"passage_id": x} for x in sorted(j, key=lambda x: -PRIOR.get(STYPE[x], 0))]}); G.append({"_judgments": j})
    m = compute_all_metrics("evidence_retrieval", P, G)
    out["retr_type_prior_P@5"], out["retr_type_prior_nDCG@10"] = round(m["precision_5"], 3), round(m["ndcg_10"], 3)
    R = rows("imaging_indication", pids)
    m = compute_all_metrics("imaging_indication", [{"clinical_question": ORDER.get(gid, "")} for gid, *_ in R], [g for *_, g in R],
                            concept_extractor=EXT)
    out["img_restate_order"] = round(m["clinical_question_concept_f1"], 3)
    print(name, json.dumps(out))
print("agentic100 encounter-count mix:", dict(sorted(collections.Counter(NENC[p] for p in SUBSETS["agentic100"]).items())),
      "| public:", dict(sorted(collections.Counter(NENC[p] for p in PUBLIC).items())))
