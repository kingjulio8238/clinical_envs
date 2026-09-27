"""Reproduce the data-level numbers in audit/FINDINGS.md from benchmark_v1.3.db alone.

Run from the repo root:  .venv/bin/python audit/scripts/data_checks.py
"""
import collections
import json
import sqlite3
import sys

sys.path.insert(0, ".")
from eval.scoring import compute_all_metrics  # noqa: E402
from eval.semantic_match import phrase_in_text  # noqa: E402

c = sqlite3.connect("benchmark_v1.3.db")
q = lambda sql, *a: c.execute(sql, a).fetchall()  # noqa: E731

# ---- graph / extraction --------------------------------------------------------
q2corr = dict(q("select question_id, diagnosis_id from question_diagnoses where role='correct'"))
q2dx = collections.defaultdict(set)
for qid, did in q("select question_id, diagnosis_id from question_diagnoses where role in ('correct','secondary')"):
    q2dx[qid].add(did)
dx = {d: (n, a, code, desc) for d, n, a, code, desc in
      q("select diagnosis_id, display_name, acuity, icd10_code, icd10_desc from diagnoses")}
ref_ids = set(q2corr.values())
print("diagnoses with ICD code but NULL description (llm_unvalidated):",
      sum(1 for v in dx.values() if v[2] and not v[3]),
      "| of which reference dx:", sum(1 for d in ref_ids if dx[d][2] and not dx[d][3]), "/", len(ref_ids))
print("questions whose correct dx is also listed as a distractor:",
      q("select count(distinct a.question_id) from question_diagnoses a join question_diagnoses b "
        "on a.question_id=b.question_id and a.diagnosis_id=b.diagnosis_id "
        "and a.role='correct' and b.role='distractor'")[0][0])

# ---- patients / clustering -----------------------------------------------------
enc = collections.defaultdict(list)  # pid -> [(eid, qid)] in encounter order
for pid, eid, sq in q("select patient_id, encounter_id, source_question_ids from longitudinal_encounters "
                      "order by patient_id, encounter_order"):
    enc[pid].append((eid, json.loads(sq)[0]))
disconnected = 0
for es in enc.values():
    qs = [qq for _, qq in es]; seen = {qs[0]}; grew = True
    while grew:
        grew = False
        for x in qs:
            if x not in seen and any(q2dx[x] & q2dx[s] for s in seen):
                seen.add(x); grew = True
    disconnected += len(seen) < len(qs)
print("clusters not connected by shared correct/secondary dx:", disconnected, "/", len(enc))
print("encounters per patient:", dict(sorted(collections.Counter(len(v) for v in enc.values()).items())))
print("patients with the same reference dx in >1 encounter:",
      sum(1 for es in enc.values() if len({q2corr[x] for _, x in es}) < len(es)))
print("encounter dates by day-of-month:", dict(q("select substr(encounter_date,9,2), count(*) from longitudinal_encounters group by 1")))

# ---- temporal leakage ----------------------------------------------------------
sec = collections.defaultdict(dict)
for eid, t, txt in q("select encounter_id, section_type, section_text from encounter_ehr_sections"):
    sec[eid][t] = (txt or "").lower()
psh_pat = [es for es in enc.values() if sum(1 for e, _ in es if "psh" in sec[e]) > 1]
print("patients whose PSH text is identical in every encounter:",
      sum(1 for es in psh_pat if len({sec[e]["psh"] for e, _ in es if "psh" in sec[e]}) == 1), "/", len(psh_pat))
for kw, proc in (("appendicitis", "appendectomy"), ("ectopic", "salpingectomy")):
    hits = [(e, proc in sec[e].get("psh", "")) for es in enc.values() for e, x in es if kw in dx[q2corr[x]][0].lower()]
    print(f"{kw} encounters whose own PSH already lists {proc}: {sum(h for _, h in hits)} / {len(hits)}")
fut = fut_acute = 0
for es in enc.values():
    later = {q2corr[x] for _, x in es[1:]} - {q2corr[es[0][1]]}
    hit = [d for d in later if dx[d][0].lower() in sec[es[0][0]].get("pmh", "")]
    fut += bool(hit); fut_acute += any(dx[d][1] == "acute" for d in hit)
print(f"patients whose encounter-0 PMH names a later encounter's dx: {fut} (acute: {fut_acute})")
own = tot = 0
for es in enc.values():
    for i, (e, x) in enumerate(es):
        if q2corr[x] not in {q2corr[y] for _, y in es[:i]} and "hpi" in sec[e]:
            tot += 1; own += dx[q2corr[x]][0].lower() in sec[e]["hpi"]
print(f"first-occurrence encounters whose HPI names that encounter's dx: {own} / {tot}")

# ---- patient diagnosis ---------------------------------------------------------
notes = collections.defaultdict(str)
for pid, t in q("select patient_id, note_text from longitudinal_encounters"):
    notes[pid] += t.lower() + "\n"
hit = total = 0
for pid, g in q("select patient_id, ground_truth from benchmark_ground_truth where task='patient_diagnosis'"):
    g = json.loads(g)
    for d in g.get("active_diagnoses", []) + g.get("chronic_conditions", []):   # private rows are stripped
        total += 1; hit += f"{d['display_name'].lower()} (diagnosed" in notes[pid]
print(f"reference dx appearing verbatim as '<name> (diagnosed ...)' in the chart: {hit}/{total} = {hit/total:.1%}")
gt = json.loads(q("select ground_truth from benchmark_ground_truth where gt_id=7304")[0][0])
bare = {"active_diagnoses": [{"icd10": "E11.9"}, {"icd10": "K35"}, {"icd10": "J96"}]}
print("patient 1973 reward for bare categories E11.9/K35/J96:",
      compute_all_metrics("patient_diagnosis", [bare], [gt])["weighted_problem_list_f1_neutral"])

# ---- evidence retrieval ---------------------------------------------------------
pd_ref = {p: {d["diagnosis_id"] for d in json.loads(g).get("active_diagnoses", []) + json.loads(g).get("chronic_conditions", [])}
          for p, g in q("select patient_id, ground_truth from benchmark_ground_truth where task='patient_diagnosis'")}
same = sum(1 for p, g in q("select patient_id, ground_truth from benchmark_ground_truth where task='evidence_retrieval'")
           if {d["diagnosis_id"] for d in json.loads(g)["query_diagnoses"]} == pd_ref[p])
print(f"retrieval query diagnoses == patient-diagnosis reference: {same}/{len(pd_ref)}")
print("public share of sections graded >=2:",
      q("select round(avg(relevance_grade>=2),3) from relevance_judgments join benchmark_ground_truth using(gt_id) where split='public'")[0][0])
stype = {f"ees_{i}": t for i, t in q("select id, section_type from encounter_ehr_sections")}
P, G = [], []
for (gid,) in q("select gt_id from benchmark_ground_truth where task='evidence_retrieval' and split='public' and is_diagnostic"):   # superseded duplicates carry no judgments
    j = dict(q("select passage_id, relevance_grade from relevance_judgments where gt_id=?", gid))
    one = [p for p in j if stype[p] == "hpi"][:1] or list(j)[:1]
    P.append({"rankings": [{"passage_id": one[0]}]}); G.append({"_judgments": j})
print("public P@5 when submitting a single HPI section:",
      round(compute_all_metrics("evidence_retrieval", P, G)["precision_5"], 3))

# ---- summarization -------------------------------------------------------------
print("whole-patient matcher, 'Fever' in 'Patient denies fever.':", phrase_in_text("Fever", "Patient denies fever."))
print("distinct whole-patient clinical questions:",
      q("select count(distinct json_extract(ground_truth,'$.clinical_question')) from benchmark_ground_truth "
        "where task='context_summarization' and json_extract(ground_truth,'$.variant') is null")[0][0])
print("specialty items: absent share by specialty (top 4):")
for r in q("select json_extract(ground_truth,'$.specialty') s, sum(json_extract(ground_truth,'$.involvement')='absent'), count(*) "
           "from benchmark_ground_truth where json_extract(ground_truth,'$.variant')='specialty_conditioned' "
           "group by 1 order by 2 desc limit 4"):
    print("   ", r)

# ---- splits --------------------------------------------------------------------
split = {}
for pid, s in q("select coalesce(b.patient_id, e.patient_id), b.split from benchmark_ground_truth b "
                "left join longitudinal_encounters e using(encounter_id)"):
    assert split.setdefault(pid, s) == s, f"patient {pid} spans splits"
print("patients per split:", dict(collections.Counter(split.values())))
print("held-out instances shipped with labels:",
      q("select count(*) from benchmark_ground_truth where split='heldout' and length(ground_truth)>10")[0][0])
keyed = collections.defaultdict(set)
for pid, es in enc.items():
    keyed[pid] = {q2corr[x] for _, x in es}
D = {s: set().union(*(keyed[p] for p in keyed if split[p] == s)) for s in ("train", "heldout")}
cat = lambda ds: {(dx[d][2] or "")[:3] for d in ds}  # noqa: E731
print(f"held-out dx seen in train: {len(D['heldout'] & D['train'])/len(D['heldout']):.1%} (diagnosis_id); "
      f"{len(cat(D['heldout']) & cat(D['train']))/len(cat(D['heldout'])):.1%} (3-char ICD, the scorer's match level)")
