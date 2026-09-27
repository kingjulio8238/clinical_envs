"""Appendix F: ground-truth construction claims vs the released DB (+ CMS FY2025 table for imaging codes).

Run from the repo root:  .venv/bin/python audit/scripts/appendix_f_checks.py <path/to/icd10cm_order_2025.txt>
"""
import collections, json, re, sqlite3, sys
sys.path.insert(0, ".")
from eval.scoring import compute_all_metrics

c = sqlite3.connect("benchmark_v1.3.db")
q = lambda sql, *a: c.execute(sql, a).fetchall()

# 1) retrieval: identical section text in the same patient, different grade => grade is not a function of content
grp = collections.defaultdict(list)
for pid, st, txt, g in q("""select l.patient_id, s.section_type, s.section_text, r.relevance_grade
                            from relevance_judgments r join encounter_ehr_sections s on s.id = cast(substr(r.passage_id,5) as int)
                            join longitudinal_encounters l on l.encounter_id = s.encounter_id"""):
    grp[(pid, st, txt)].append(g)
dup = [v for v in grp.values() if len(v) > 1]
diff = [v for v in dup if len(set(v)) > 1]
print(f"retrieval: groups of byte-identical sections within a patient: {len(dup)}; graded differently: {len(diff)} "
      f"({len(diff)/len(dup):.0%}); sections involved: {sum(len(v) for v in diff)}")
# 2) whole-patient summarization matcher: negation
gt = {"must_include_findings": [{"display_name": "Fever"}, {"display_name": "Ketonuria"}]}
m = compute_all_metrics("context_summarization", [{"summary": "Patient denies fever. No ketonuria."}], [gt])
print(f"whole-patient clinical_f1 for 'Patient denies fever. No ketonuria.' vs must-include [Fever, Ketonuria]: {m['clinical_f1']:.2f}")
# 3) imaging: indication names the diagnosis? reference differential codes valid?
cms = {}
for line in open(sys.argv[1], encoding="latin-1"):
    cms[line[6:13].strip()] = line[14] == "1"
cats = {k[:3] for k in cms}
st = collections.Counter(); names = 0; n = 0
corr = {}
for eid, sq in q("select encounter_id, source_question_ids from longitudinal_encounters"):
    for (nm,) in q("select lower(d.display_name) from question_diagnoses qd join diagnoses d using(diagnosis_id) where qd.question_id=? and role='correct'", json.loads(sq)[0]):
        corr[eid] = nm
for eid, g, ind in q("select b.encounter_id, b.ground_truth, o.clinical_indication from benchmark_ground_truth b join imaging_orders o using(gt_id) where b.task='imaging_indication'"):
    n += 1; g = json.loads(g)
    names += bool(corr.get(eid)) and corr[eid] in (ind or "").lower()
    for d in g.get("differential_context", []):
        k = (d.get("icd10") or "").replace(".", "").upper()
        st["billable" if cms.get(k) else "header" if k in cms else "invalid code, valid category" if k[:3] in cats else "invalid category"] += 1
print(f"imaging: orders whose indication contains the correct diagnosis name: {names}/{n}")
print("imaging: reference differential ICD codes vs CMS FY2025:", dict(st))
# 4) specialty: absent items per patient; which specialties
print("specialty absent items per patient:", dict(q("select k, count(*) from (select patient_id, sum(json_extract(ground_truth,'$.involvement')='absent') k from benchmark_ground_truth where json_extract(ground_truth,'$.variant')='specialty_conditioned' group by 1) group by k")))
