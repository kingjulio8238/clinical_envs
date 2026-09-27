"""Zero-model baseline for patient_diagnosis: copy the chart's problem list."""
import sqlite3, json, re, sys, collections, importlib.util
sys.path.insert(0, '.')
from eval.scoring import compute_all_metrics
c = sqlite3.connect('benchmark_v1.3.db')
class Cur:  # sqlite cursor shim that tolerates the missing terminology table
    def __init__(s): s.c = c.cursor()
    def execute(s, q, *a):
        try: s.c.execute(q, *a)
        except sqlite3.OperationalError: s.c.execute("select 1 where 0")
    def fetchall(s): return s.c.fetchall()
spec = importlib.util.spec_from_file_location("cns", "scripts/chart_neutral_sets.py"); cns = importlib.util.module_from_spec(spec); spec.loader.exec_module(cns)
neutral, _, _ = cns.neutral_sets(Cur())
name2icd = {}
for code, n in c.execute("select icd10_code, display_name from diagnoses where icd10_code is not null"):
    name2icd.setdefault(n.lower(), code)
notes = collections.defaultdict(str)
for pid, t in c.execute("select patient_id, note_text from longitudinal_encounters"): notes[pid] += t + "\n"
prof = {pid: json.loads(p) for pid, p in c.execute("select patient_id, profile from longitudinal_patients")}
split = sys.argv[1] if len(sys.argv) > 1 else 'public'
rows = c.execute("select patient_id, ground_truth from benchmark_ground_truth where task='patient_diagnosis' and split=?", (split,)).fetchall()
def run(policy):
    P, G = [], []
    for pid, gt in rows:
        gt = json.loads(gt); gt["_neutral_categories"] = neutral.get(pid, [])
        act = []
        if 'plist' in policy:
            for n in set(re.findall(r"^- (.+?) \(diagnosed \d{4}-\d\d-\d\d\)", notes[pid], re.M)):
                if n.lower() in name2icd: act.append({"icd10": name2icd[n.lower()], "acuity": "acute"})
        chron = []
        if 'chronic' in policy:
            for s in prof[pid].get("chronic_conditions") or []:
                cats, _ = cns.map_condition(str(s))
                chron += [{"icd10": k} for k in sorted(cats)[:1]]
        P.append({"active_diagnoses": act, "chronic_conditions": chron}); G.append(gt)
    m = compute_all_metrics("patient_diagnosis", P, G)
    return {k: round(m[k], 3) for k in ("weighted_problem_list_f1_neutral", "weighted_problem_list_recall", "problem_list_precision_neutral", "problem_list_recall")}
for pol in (('chronic',), ('plist',), ('plist', 'chronic')):
    print(split, '+'.join(pol), run(pol))
