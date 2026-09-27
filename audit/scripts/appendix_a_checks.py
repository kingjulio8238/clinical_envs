"""Appendix A (secondary metrics): what each metric can and cannot detect.

Run from the repo root:  .venv/bin/python audit/scripts/appendix_a_checks.py
"""
import collections, json, random, sqlite3, sys
import numpy as np
sys.path.insert(0, ".")
from eval.scoring import compute_all_metrics, hallucination_rate

c = sqlite3.connect("benchmark_v1.3.db")
q = lambda sql, *a: c.execute(sql, a).fetchall()
chart = collections.defaultdict(str)
for pid, t in q("select patient_id, note_text from longitudinal_encounters order by patient_id, encounter_order"):
    chart[pid] += t + "\n"

# ---- Table A internal consistency (numbers transcribed from the paper) -------------
T2 = {"Gemini": .681, "GPT": .703, "Kimi": .732, "Opus": .615, "DeepSeek": .660, "GLM": .657, "Qwen": .653, "Mistral": .676, "Llama": .536, "Gemma": .287}
T2S = {"Gemini": .502, "GPT": .489, "Kimi": .532, "Opus": .550, "DeepSeek": .399, "GLM": .484, "Qwen": .425, "Mistral": .416, "Llama": .403, "Gemma": .385}
PREC = {"Gemini": .606, "GPT": .645, "Kimi": .687, "Opus": .535, "DeepSeek": .653, "GLM": .627, "Qwen": .668, "Mistral": .636, "Llama": .552, "Gemma": .249}
REC = {"Gemini": .787, "GPT": .781, "Kimi": .787, "Opus": .738, "DeepSeek": .682, "GLM": .698, "Qwen": .652, "Mistral": .728, "Llama": .542, "Gemma": .346}
OMI = {"Gemini": .498, "GPT": .511, "Kimi": .468, "Opus": .450, "DeepSeek": .601, "GLM": .516, "Qwen": .575, "Mistral": .584, "Llama": .597, "Gemma": .615}
hm = lambda p, r: 2 * p * r / (p + r)
print("Table 2 dx F1 minus HM(TableA Prec, Rec):", {m: round(T2[m] - hm(PREC[m], REC[m]), 3) for m in T2})
print("Table 2 Summ + Table A Summ.Omiss:", {m: round(T2S[m] + OMI[m], 3) for m in T2})
ms = list(T2)
print("Spearman-free check, corr(Prec, Rec) across models:", round(np.corrcoef([PREC[m] for m in ms], [REC[m] for m in ms])[0, 1], 2),
      "| excluding Gemma:", round(np.corrcoef([PREC[m] for m in ms if m != "Gemma"], [REC[m] for m in ms if m != "Gemma"])[0, 1], 2))

# ---- hallucination_rate: can it detect fabrication? ------------------------------
R = [(p, json.loads(g)) for p, g in q("select patient_id, ground_truth from benchmark_ground_truth where task='context_summarization' "
                                      "and json_extract(ground_truth,'$.variant') is null and split='public'")]
pids = [p for p, _ in R]; refs = [g["reference_summary"] for _, g in R]
own = hallucination_rate(refs, [chart[p] for p in pids])
other = hallucination_rate(refs, [chart[pids[(i + 1) % len(pids)]] for i in range(len(pids))])
fab = ("The patient has a history of metastatic pancreatic adenocarcinoma and was started on FOLFIRINOX. "
       "He was admitted for a pulmonary embolism and was treated with apixaban. "
       "His course was complicated by a left hip fracture after a fall at home.")
fabricated = hallucination_rate([fab] * len(pids), [chart[p] for p in pids])
print(f"hallucination_rate: reference summary vs OWN chart {own:.3f}; vs a DIFFERENT patient's chart {other:.3f}; "
      f"3 invented sentences (cancer, PE, hip fracture) vs each chart {fabricated:.3f}")

# ---- ICD specificity: what a category-only prediction scores ---------------------
from eval.scoring import _icd10_specificity_score, _match_icd10_sets
spec = []
for (g,) in q("select ground_truth from benchmark_ground_truth where task='patient_diagnosis' and split='public'"):
    g = json.loads(g); codes = [d["icd10"] for d in g["active_diagnoses"] + g["chronic_conditions"] if d.get("icd10")]
    m, _, _ = _match_icd10_sets([x[:3] for x in codes], codes); s = _icd10_specificity_score(m)
    if s is not None: spec.append(s)
print(f"icd10_specificity when predicting ONLY 3-char categories: {np.mean(spec):.3f} (Table A range 0.824-0.940)")

# ---- acuity: majority-class floor ------------------------------------------------
acu = collections.Counter()
for (g,) in q("select ground_truth from benchmark_ground_truth where task='patient_diagnosis' and split='public'"):
    g = json.loads(g)
    for d in g["active_diagnoses"]: acu["acute_on_chronic" if d.get("acuity") == "acute_on_chronic" else "acute"] += 1
    for d in g["chronic_conditions"]: acu["chronic"] += 1
tot = sum(acu.values()); print("reference acuity mix (public):", {k: f"{v/tot:.1%}" for k, v in acu.items()})

# ---- retrieval MRR / MAP@10: chance levels ----------------------------------------
rng = random.Random(0); P, G = [], []
for (gid,) in q("select gt_id from benchmark_ground_truth where task='evidence_retrieval' and split='public'"):
    j = dict(q("select passage_id, relevance_grade from relevance_judgments where gt_id=?", gid))
    for _ in range(20):
        ids = list(j); rng.shuffle(ids); P.append({"rankings": [{"passage_id": x} for x in ids]}); G.append({"_judgments": j})
m = compute_all_metrics("evidence_retrieval", P, G)
print(f"random ranking (20 shuffles x 200): MRR {m['mrr']:.3f}, MAP@10 {m['map_10']:.3f}, P@5 {m['precision_5']:.3f}, nDCG@10 {m['ndcg_10']:.3f}")
