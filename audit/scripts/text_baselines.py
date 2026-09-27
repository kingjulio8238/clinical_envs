"""Zero-model baselines for the text tasks (summarization, specialty, imaging) on one split."""
import sqlite3, json, sys, collections, hashlib
sys.path.insert(0, '.')
from eval.scoring import compute_all_metrics
from eval.imaging_concepts import ConceptExtractor
c = sqlite3.connect('benchmark_v1.3.db')
split = sys.argv[1] if len(sys.argv) > 1 else 'public'
chart = collections.defaultdict(str)
for pid, t in c.execute("select patient_id, note_text from longitudinal_encounters order by patient_id, encounter_order"): chart[pid] += t + "\n"
def items(where):
    return [(pid, json.loads(g)) for pid, g in c.execute(f"select patient_id, ground_truth from benchmark_ground_truth where split=? and {where}", (split,))]
out = {}
whole = items("task='context_summarization' and json_extract(ground_truth,'$.variant') is null")
for name, f in (("empty", lambda p: ""), ("dump_chart", lambda p: chart[p])):
    m = compute_all_metrics("context_summarization", [{"summary": f(p)} for p, _ in whole], [g for _, g in whole])
    out[f"whole/{name}"] = round(m["clinical_f1"], 3)
spec = items("json_extract(ground_truth,'$.variant')='specialty_conditioned'")
inv = [(p, g) for p, g in spec if g["involvement"] != "absent"]; ab = [(p, g) for p, g in spec if g["involvement"] == "absent"]
for name, f in (("dump_chart", lambda p: chart[p]), ("dump_chart+phrase", lambda p: chart[p] + "\nNo significant distress.")):
    mi = compute_all_metrics("context_summarization", [{"summary": f(p)} for p, _ in inv], [g for _, g in inv])
    ma = compute_all_metrics("context_summarization", [{"summary": f(p)} for p, _ in ab], [g for _, g in ab])
    out[f"spec/{name}"] = {"conditioned_f1(involved)": round(mi["conditioned_f1"], 3), "leakage": round(mi["leakage_rate"], 3),
                           "abstention_acc(absent)": round(ma["abstention_accuracy"], 3)}
# imaging: ConceptExtractor inventory built from sqlite (same query as eval.imaging_concepts.load_inventory)
inv_rows = []
for did, sn, dn, sd in c.execute("select diagnosis_id, snomed_id, display_name, snomed_desc from diagnoses"):
    cid = f"S{sn}" if sn else f"D{did}"; inv_rows += [(cid, dn), (cid, sd)]
for fid, sn, dn, sd in c.execute("select finding_id, snomed_id, display_name, snomed_desc from clinical_findings where finding_type <> 'demographic'"):
    cid = f"S{sn}" if sn else f"F{fid}"; inv_rows += [(cid, dn), (cid, sd)]
ext = ConceptExtractor(inv_rows)
img = [(eid, json.loads(g)) for eid, g in c.execute("select encounter_id, ground_truth from benchmark_ground_truth where task='imaging_indication' and split=?", (split,))]
order = dict(c.execute("select gt_id, clinical_indication from imaging_orders"))
gtid = [r[0] for r in c.execute("select gt_id from benchmark_ground_truth where task='imaging_indication' and split=?", (split,))]
cc = dict(c.execute("select encounter_id, chief_complaint from longitudinal_encounters"))
for name, f in (("restate_order", lambda i, e: order.get(gtid[i], "")), ("chief_complaint", lambda i, e: cc[e])):
    m = compute_all_metrics("imaging_indication", [{"clinical_question": f(i, e)} for i, (e, _) in enumerate(img)], [g for _, g in img], concept_extractor=ext)
    out[f"imaging/{name}"] = round(m["clinical_question_concept_f1"], 3)
s = json.dumps(out, sort_keys=True)
print(split, s); print("digest", hashlib.md5(s.encode()).hexdigest())
