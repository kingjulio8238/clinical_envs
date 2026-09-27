"""§6 checks: does chart-neutral scoring really stop chart-documented conditions being penalized, and
what does the case mix look like.

Run from the repo root:  .venv/bin/python audit/scripts/limitations_checks.py
"""
import collections, importlib.util, json, sqlite3, sys
sys.path.insert(0, ".")
from eval.scoring import compute_all_metrics

c = sqlite3.connect("benchmark_v1.3.db")
q = lambda sql, *a: c.execute(sql, a).fetchall()

class _Cur:
    def __init__(self): self.c = c.cursor()
    def execute(self, sql, *a):
        try: self.c.execute(sql, *a)
        except sqlite3.OperationalError: self.c.execute("select 1 where 0")
    def fetchall(self): return self.c.fetchall()
spec = importlib.util.spec_from_file_location("cns", "scripts/chart_neutral_sets.py")
cns = importlib.util.module_from_spec(spec); spec.loader.exec_module(cns)
NEUTRAL, _, _ = cns.neutral_sets(_Cur())

code = dict(q("select diagnosis_id, icd10_code from diagnoses"))
sec = collections.defaultdict(set)
for pid, sq in q("select patient_id, source_question_ids from longitudinal_encounters"):
    for qid in json.loads(sq):
        for (d,) in q("select diagnosis_id from question_diagnoses where question_id=? and role='secondary'", qid):
            if code.get(d): sec[pid].add(code[d])
R = [(p, json.loads(g)) for p, g in q("select patient_id, ground_truth from benchmark_ground_truth where task='patient_diagnosis' and split='public'")]
P, G, pen = [], [], []
for p, g in R:
    ref = [d["icd10"] for d in g.get("active_diagnoses", []) + g.get("chronic_conditions", []) if d.get("icd10")]
    refcat = {x[:3] for x in ref}; neu = set(NEUTRAL.get(p, []))
    extra = sorted({x for x in sec[p] if x[:3] not in refcat})
    pen.append(sum(1 for x in extra if x[:3] not in neu))
    P.append({"active_diagnoses": [{"icd10": x} for x in ref + extra]}); G.append(dict(g, _neutral_categories=sorted(neu)))
m = compute_all_metrics("patient_diagnosis", P, G)
print(f"public: reference + graph-secondary dx (conditions the source vignette documents): w-F1 {m['weighted_problem_list_f1_neutral']:.3f}, "
      f"precision_neutral {m['problem_list_precision_neutral']:.3f}; patients with >=1 penalized documented condition: "
      f"{sum(1 for x in pen if x)}/{len(pen)} (mean {sum(pen)/len(pen):.1f} penalized per patient)")

# case mix: how 'classic' are the presentations the labels rest on
rel = collections.Counter()
for (r,) in q("""select df.relationship from question_findings qf
                 join question_diagnoses qd on qd.question_id=qf.question_id and qd.role='correct'
                 join diagnosis_findings df on df.diagnosis_id=qd.diagnosis_id and df.finding_id=qf.finding_id
                 where qf.relevance='key'"""):
    rel[r] += 1
tot = sum(rel.values())
print("key findings by typed relation to the correct dx:", {k: f"{v/tot:.1%}" for k, v in rel.most_common()})
print("section presence (share of encounters):", {t: round(n / 5602, 2) for t, n in q(
    "select section_type, count(distinct encounter_id) from encounter_ehr_sections group by 1 order by 2")})
