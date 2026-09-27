"""Appendix E: clustering, profile, timeline and note-assembly claims against the released DB.

Run from the repo root:  .venv/bin/python audit/scripts/appendix_e_checks.py
"""
import collections, json, sqlite3
c = sqlite3.connect("benchmark_v1.3.db")
q = lambda sql, *a: c.execute(sql, a).fetchall()

corr = collections.defaultdict(list); dxs = collections.defaultdict(set)
for qid, did, role, n in q("select question_id, diagnosis_id, role, lower(d.display_name) from question_diagnoses join diagnoses d using(diagnosis_id)"):
    if role == "correct": corr[qid].append(n)
    if role in ("correct", "secondary"): dxs[qid].add(did)
enc = collections.defaultdict(list)
for pid, sq, o in q("select patient_id, source_question_ids, encounter_order from longitudinal_encounters order by patient_id, encounter_order"):
    enc[pid].append(json.loads(sq)[0])
# clusters: does any non-seed member fail to share a dx with SOME member vs with the seed?  (seed unknown after
# re-ordering, so test the paper's claim directly: is there a member sharing nothing with some other member
# that is linked to everyone else -- i.e. is every cluster a 'star'?)
star = sum(1 for qs in enc.values() if any(all(dxs[m] & dxs[x] for x in qs if x != m) for m in qs))
print(f"clusters with at least one member linked to ALL others (star centre = seed): {star}/{len(enc)}")
print("visit types:", dict(q("select encounter_type, count(*) from longitudinal_encounters group by 1")))
print("patients whose first visit is not at the calendar anchor:", q("select count(*) from longitudinal_encounters where encounter_order=0 and encounter_date!='2020-01-15'")[0][0],
      "| patients with no encounter_order 0 (encounter removed after generation):",
      q("select count(*) from (select patient_id from longitudinal_encounters group by 1 having min(encounter_order)>0)")[0][0])
print("same-day encounter pairs within a patient:", q("select count(*) from (select patient_id, encounter_date from longitudinal_encounters group by 1,2 having count(*)>1)")[0][0])
print("sections: is_modified", q("select sum(is_modified), count(*) from encounter_ehr_sections")[0],
      "| columns:", [r[1] for r in q("pragma table_info(encounter_ehr_sections)")])
print("generation_method by first(1)/later(0) encounter:", dict(q("select (encounter_order=0)||'-'||generation_method, count(*) from longitudinal_encounters group by 1")))
viol = 0
for pid, prof in q("select patient_id, profile from longitudinal_patients"):
    cc = [x.lower() for x in json.loads(prof).get("chronic_conditions") or []]
    own = {n for qid in enc[pid] for n in corr[qid]}
    viol += any(d == x or (len(d) > 6 and d in x) for d in own for x in cc)
print(f"profiles listing one of the patient's own tested diagnoses as a chronic condition: {viol}/{len(enc)}")
