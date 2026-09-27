"""How much of the whole-patient summarization score the 'ontology-grounded structured' prompt hands over.

eval/tasks/summarization.py:_load_key_findings puts up to 10 of the patient's key-finding names into the
prompt ("Your summary must address each of these findings"); the scorer (clinical_f1) is recall of
must-include names, which are drawn from the same key findings. The SQL has no ORDER BY, so which 10 is
unspecified; we average over random draws and also report alphabetical order.

Run from the repo root:  .venv/bin/python audit/scripts/summ_hint_leak.py
"""
import collections, json, random, sqlite3, sys
sys.path.insert(0, ".")
from eval.scoring import compute_all_metrics

c = sqlite3.connect("benchmark_v1.3.db")
key = collections.defaultdict(set)
for pid, sq in c.execute("select patient_id, source_question_ids from longitudinal_encounters"):
    for qid in json.loads(sq):
        for (n,) in c.execute("select distinct cf.display_name from question_findings qf join clinical_findings cf "
                              "using(finding_id) where qf.question_id=? and qf.relevance='key'", (qid,)):
            key[pid].add(n)
R = [(p, json.loads(g)) for p, g in c.execute(
    "select patient_id, ground_truth from benchmark_ground_truth where task='context_summarization' "
    "and json_extract(ground_truth,'$.variant') is null and split='public'")]
share = [len({f["display_name"] for f in g["must_include_findings"]} & key[p]) / max(1, len(g["must_include_findings"])) for p, g in R]
print(f"public: must-include names that are key findings: {sum(share)/len(share):.1%}; mean distinct key findings per patient: "
      f"{sum(len(key[p]) for p, _ in R)/len(R):.1f}")
def score(pick):
    P = [{"summary": "\n".join(f"- {n}" for n in pick(p))} for p, _ in R]
    return compute_all_metrics("context_summarization", P, [g for _, g in R])["clinical_f1"]
rs = [score(lambda p, rng=random.Random(s): rng.sample(sorted(key[p]), min(10, len(key[p])))) for s in range(20)]
print(f"summary = the 10 hinted names verbatim: clinical_f1 {sum(rs)/len(rs):.3f} (random 10, 20 draws; range {min(rs):.3f}-{max(rs):.3f}); "
      f"alphabetical 10: {score(lambda p: sorted(key[p])[:10]):.3f}")
