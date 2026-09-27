"""Appendix H: internal consistency of Tables H.1-H.3 and attempts to reproduce them from the release.

Run from the repo root:  .venv/bin/python audit/scripts/appendix_h_checks.py
"""
import collections, importlib.util, itertools, json, sqlite3
import numpy as np
from scipy.spatial.distance import jensenshannon
from scipy.stats import spearmanr

CH = ["A00-B99", "C00-D49", "D50-D89", "E00-E89", "F01-F99", "G00-G99", "H00-H59", "H60-H95", "I00-I99", "J00-J99", "K00-K95",
      "L00-L99", "M00-M99", "N00-N99", "O00-O9A", "P00-P96", "Q00-Q99", "R00-R99", "S00-T88", "V00-Y99", "Z00-Z99"]
# Table H.2, transcribed (SH, source, Synthea)
H2 = {"E00-E89": (13.8, 10.1, 3.1), "I00-I99": (12.6, 9.6, 1.4), "Z00-Z99": (9.3, 2.9, 56.8), "F01-F99": (6.0, 6.7, 9.4),
      "N00-N99": (5.9, 6.0, 1.3), "A00-B99": (5.5, 7.5, 0.1), "S00-T88": (5.4, 8.1, 2.5), "K00-K95": (5.2, 5.4, 9.9),
      "C00-D49": (5.0, 7.1, 0.3), "D50-D89": (4.8, 5.4, 1.1), "J00-J99": (4.4, 4.3, 7.9), "M00-M99": (4.2, 5.5, 1.4),
      "G00-G99": (4.1, 6.3, 1.2), "O00-O9A": (3.6, 3.9, 0.5), "R00-R99": (3.5, 1.8, 2.5), "Q00-Q99": (1.9, 3.9, 0.1),
      "P00-P96": (1.5, 1.6, 0.0), "L00-L99": (1.4, 2.0, 0.1), "H00-H59": (0.8, 1.2, 0.0), "H60-H95": (0.6, 0.6, 0.5), "V00-Y99": (0.2, 0.1, 0.0)}
sh, src, syn = (np.array([H2[c][i] for c in CH]) for i in range(3))
print("Table H.2 column sums (SH, source, Synthea):", round(sh.sum(), 1), round(src.sum(), 1), round(syn.sum(), 1))
js = lambda a, b: jensenshannon(a / a.sum(), b / b.sum(), base=2) ** 2
print(f"from Table H.2 itself: JSD(SH,src)={js(sh, src):.3f} rho={spearmanr(sh, src).correlation:.2f} | "
      f"JSD(SH,Synthea)={js(sh, syn):.3f} rho={spearmanr(sh, syn).correlation:.2f} | JSD(src,Synthea)={js(src, syn):.3f}"
      f"   (paper: 0.029/0.83, 0.326/0.70, 0.436)")
nz = [i for i, c in enumerate(CH) if c != "Z00-Z99"]
r = lambda v, i: v[i] / v[nz].sum()
iI, iE = CH.index("I00-I99"), CH.index("E00-E89")
print(f"SH/Synthea ratio as printed: circulatory {sh[iI]/syn[iI]:.1f}x, endocrine {sh[iE]/syn[iE]:.1f}x; "
      f"after excluding Z00-Z99 from both: {r(sh,iI)/r(syn,iI):.1f}x, {r(sh,iE)/r(syn,iE):.1f}x")

def chapter(code):
    k = (code or "")[:3].upper()
    bounds = [("A00", "B99"), ("C00", "D49"), ("D50", "D89"), ("E00", "E89"), ("F01", "F99"), ("G00", "G99"), ("H00", "H59"), ("H60", "H95"),
              ("I00", "I99"), ("J00", "J99"), ("K00", "K95"), ("L00", "L99"), ("M00", "M99"), ("N00", "N99"), ("O00", "O9A"), ("P00", "P96"),
              ("Q00", "Q99"), ("R00", "R99"), ("S00", "T88"), ("V00", "Y99"), ("Z00", "Z99")]
    return next((CH[i] for i, (lo, hi) in enumerate(bounds) if lo <= k <= hi), None)

c = sqlite3.connect("benchmark_v1.3.db")
used = {json.loads(s)[0] for (s,) in c.execute("select source_question_ids from longitudinal_encounters")}
code = dict(c.execute("select diagnosis_id, icd10_code from diagnoses"))
qd = collections.defaultdict(list)
for qid, did, role in c.execute("select question_id, diagnosis_id, role from question_diagnoses"): qd[qid].append((did, role))
def dist(qids, roles):
    cnt = collections.Counter(chapter(code[d]) for q in qids for d, r in qd[q] if r in roles); cnt.pop(None, None)
    v = np.array([cnt[ch] for ch in CH], float); return 100 * v / v.sum()
print("\nreproduction attempts (released DB; SH = 5,602 used questions, source = all 7,003):")
for label, roles in (("correct only", {"correct"}), ("correct+secondary", {"correct", "secondary"}), ("all roles incl. distractors", {"correct", "secondary", "distractor"})):
    a, b = dist(used, roles), dist(qd.keys(), roles)
    print(f"  {label:28s} SH Z={a[-1]:.1f}% E={a[3]:.1f}% I={a[8]:.1f}% | JSD(SH,src)={js(a, b):.4f} rho={spearmanr(a, b).correlation:.2f}")
print("  -> Table H.2's SH column (Z 9.3% vs source 2.9%) is not reproducible from question-level diagnoses; the SH unit\n"
      "     likely includes LLM-profile conditions (chart-neutral mapping assigns Z87/Z90/Z98 to smoking and surgery).")

# comorbidity ORs (Table H.3) with two candidate per-patient condition sets
spec = importlib.util.spec_from_file_location("cns", "scripts/chart_neutral_sets.py"); cns = importlib.util.module_from_spec(spec); spec.loader.exec_module(cns)
class Cur:
    def __init__(s): s.c = c.cursor()
    def execute(s, q, *a):
        try: s.c.execute(q, *a)
        except sqlite3.OperationalError: s.c.execute("select 1 where 0")
    def fetchall(s): return s.c.fetchall()
neutral, _, _ = cns.neutral_sets(Cur())
pq = collections.defaultdict(set)
for pid, s in c.execute("select patient_id, source_question_ids from longitudinal_encounters"): pq[pid].add(json.loads(s)[0])
graph = {p: {(code[d] or "")[:3] for q in qs for d, r in qd[q] if r in ("correct", "secondary") and code[d]} for p, qs in pq.items()}
withprof = {p: graph[p] | set(neutral.get(p, [])) for p in pq}
PAIRS = [("E11", "N18", 14.3), ("I48", "I63", 16.01), ("I10", "I50", 9.43), ("E78", "I25", 21.63), ("J44", "J96", 70.41),
         ("E11", "I25", 5.16), ("N18", "D64", 4.14), ("K70", "I85", 100.2), ("E66", "G47", 4.62), ("I10", "N18", 11.94)]
def OR(sets, a, b):
    t = [[0.5, 0.5], [0.5, 0.5]]
    for s in sets.values(): t[a in s][b in s] += 1
    return t[1][1] * t[0][0] / (t[1][0] * t[0][1]), int(t[1][1] - 0.5)
print("\nTable H.3 odds ratios (Haldane) — paper vs graph dx (correct+secondary) vs graph + profile-derived categories:")
for a, b, paper in PAIRS:
    o1, n1 = OR(graph, a, b); o2, n2 = OR(withprof, a, b)
    print(f"  {a}-{b}: paper {paper:6.2f} | graph {o1:6.2f} (both n={n1}) | +profile {o2:6.2f} (both n={n2})")

# how much of each pair's co-occurrence comes from a single source vignette (not from longitudinal assembly)
qcat = collections.defaultdict(set)
for q, rows in qd.items():
    for d, r in rows:
        if r in ("correct", "secondary") and code[d]: qcat[q].add(code[d][:3])
print("\nshare of co-occurring patients where BOTH categories come from one source question:")
for a, b, _ in PAIRS:
    both = [p for p, qs in pq.items() if any(a in qcat[x] for x in qs) and any(b in qcat[x] for x in qs)]
    same = [p for p in both if any(a in qcat[x] and b in qcat[x] for x in pq[p])]
    print(f"  {a}-{b}: {len(same)}/{len(both)}")
