"""Reproduce the §4 numbers in audit/FINDINGS.md.

Run from the repo root:  .venv/bin/python audit/scripts/realism_checks.py
"""
import collections
import json
import sqlite3

import numpy as np
from scipy.spatial.distance import jensenshannon
from scipy.stats import binomtest, spearmanr

c = sqlite3.connect("benchmark_v1.3.db")

# ---- §4.1 reported statistics: internal consistency ----------------------------
print("53/100 two-sided binomial p:", round(binomtest(53, 100, 0.5).pvalue, 3))
p9 = binomtest(9, 10, 0.5).pvalue
print(f"best rater 9/10: two-sided p={p9:.4f}, Holm x10={min(1, 10 * p9):.3f}; "
      f"one-sided Holm x10={min(1, 10 * binomtest(9, 10, 0.5, alternative='greater').pvalue):.3f}")
print("min correct/100 for p<.05:", next(k for k in range(50, 101) if binomtest(k, 100, 0.5).pvalue < 0.05))

# ---- Appendix H: ICD-10 chapter distribution, benchmark vs extracted questions ---
CHAPTERS = [("A00", "B99"), ("C00", "D49"), ("D50", "D89"), ("E00", "E89"), ("F01", "F99"), ("G00", "G99"),
            ("H00", "H59"), ("H60", "H95"), ("I00", "I99"), ("J00", "J99"), ("K00", "K95"), ("L00", "L99"),
            ("M00", "M99"), ("N00", "N99"), ("O00", "O9A"), ("P00", "P96"), ("Q00", "Q99"), ("R00", "R99"),
            ("S00", "T88"), ("U00", "U85"), ("V00", "Y99"), ("Z00", "Z99")]


def chapter(code):
    k = (code or "")[:3].upper()
    return next((i for i, (lo, hi) in enumerate(CHAPTERS) if lo <= k <= hi), None)


qcode = dict(c.execute("select question_id, d.icd10_code from question_diagnoses "
                       "join diagnoses d using(diagnosis_id) where role='correct'"))
used = {json.loads(s)[0] for (s,) in c.execute("select source_question_ids from longitudinal_encounters")}
src = collections.Counter(chapter(qcode[q]) for q in qcode)
ben = collections.Counter(chapter(qcode[q]) for q in used)
src.pop(None, None); ben.pop(None, None)
keys = sorted(set(src) | set(ben))
p = np.array([src[k] for k in keys], float); q = np.array([ben[k] for k in keys], float)
print(f"chapter JSD (base 2) benchmark({len(used)}) vs extracted questions({len(qcode)}): "
      f"{jensenshannon(p / p.sum(), q / q.sum(), base=2) ** 2:.4f}; Spearman {spearmanr(p, q).correlation:.3f}")

# ---- §4.2 how much dx-dx edges move specialty labels ---------------------------
n = rel = neu = 0; size = collections.Counter(); crit = 0
for (g,) in c.execute("select ground_truth from benchmark_ground_truth "
                      "where json_extract(ground_truth,'$.variant')='specialty_conditioned' "
                      "and json_extract(ground_truth,'$.involvement')!='absent'"):
    t = json.loads(g)["tiers"]; n += 1
    rel += bool(t["relevant"]); neu += bool(t["neutral"])
    for k in ("primary", "relevant", "neutral"):
        size[k] += len(t[k])
    crit += sum(1 for f in t["relevant"] if f.get("importance") == "critical")
print(f"involved specialty items {n}: relevant tier non-empty {rel} ({rel / n:.1%}), neutral non-empty {neu} ({neu / n:.1%})")
print("findings per tier:", dict(size), "| critical relevant findings:", crit)
print("diagnosis_relations shipped in release DB:",
      bool(c.execute("select 1 from sqlite_master where name='diagnosis_relations'").fetchone()))
