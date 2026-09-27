"""Content-blind retrieval baseline: rank chart sections by section type only."""
import sqlite3, sys, collections
sys.path.insert(0, '.')
from eval.scoring import compute_all_metrics
c = sqlite3.connect('benchmark_v1.3.db')
split = sys.argv[1] if len(sys.argv) > 1 else 'public'
sec = {f"ees_{i}": (t, o, e) for i, t, o, e in c.execute("select id, section_type, section_order, encounter_id from encounter_ehr_sections")}
eorder = {e: o for e, o in c.execute("select encounter_id, encounter_order from longitudinal_encounters")}
# fixed type prior learned from the TRAIN split's mean grade per section type
tr = collections.defaultdict(list)
for pid, g in c.execute("select r.passage_id, r.relevance_grade from relevance_judgments r join benchmark_ground_truth b using(gt_id) where b.split='train'"):
    tr[sec[pid][0]].append(g)
prior = {t: sum(v) / len(v) for t, v in tr.items()}
print("type prior (train mean grade):", {k: round(v, 2) for k, v in sorted(prior.items(), key=lambda x: -x[1])})
P, G = [], []
for (gt_id,) in c.execute("select gt_id from benchmark_ground_truth where task='evidence_retrieval' and split=?", (split,)).fetchall():
    j = dict(c.execute("select passage_id, relevance_grade from relevance_judgments where gt_id=?", (gt_id,)).fetchall())
    ranked = sorted(j, key=lambda p: (-prior.get(sec[p][0], 0), -eorder[sec[p][2]]))  # passage set from the instance's own chart
    P.append({"rankings": [{"passage_id": p} for p in ranked]}); G.append({"_judgments": j})
m = compute_all_metrics("evidence_retrieval", P, G)
print(split, {k: round(v, 3) for k, v in m.items()})
