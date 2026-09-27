"""Appendix D: graph counts, ontology coverage, and ICD-10-CM validity of the released codes.

Needs the public CMS FY2025 'code descriptions in tabular order' file (the same one the entrypoint
downloads):  https://www.cms.gov/files/zip/2025-code-descriptions-tabular-order.zip
Run from the repo root:  .venv/bin/python audit/scripts/appendix_d_checks.py <path/to/icd10cm_order_2025.txt>
"""
import collections, json, sqlite3, sys

cms = {}
for line in open(sys.argv[1], encoding="latin-1"):
    code, billable, desc = line[6:13].strip(), line[14] == "1", line[77:].strip()
    cms[code] = (billable, desc)
cats = {c[:3] for c in cms}
print(f"CMS FY2025 table: {len(cms)} codes, {sum(b for b, _ in cms.values())} billable")

c = sqlite3.connect("benchmark_v1.3.db")
ref = {d for (d,) in c.execute("select distinct diagnosis_id from question_diagnoses where role='correct'")}
rows = c.execute("select diagnosis_id, icd10_code, icd10_desc from diagnoses").fetchall()
def status(code):
    k = (code or "").replace(".", "").upper()
    if not k: return "uncoded"
    if k in cms: return "billable" if cms[k][0] else "header (non-billable)"
    return "valid 3-char category, invalid code" if k[:3] in cats else "invalid category"
for label, subset in (("all 9,623 nodes", rows), ("2,975 reference (correct-answer) nodes", [r for r in rows if r[0] in ref])):
    by = collections.defaultdict(collections.Counter)
    for d, code, desc in subset:
        by["validated (desc present)" if desc else ("unvalidated (desc NULL)" if code else "uncoded")][status(code)] += 1
    print(f"\n{label}:")
    for k, v in by.items():
        print(f"  {k:26s} n={sum(v.values()):5d}  {dict(v)}")
# encounters whose reference code cannot be scored at the category level
enc = collections.Counter()
for (sq,) in c.execute("select source_question_ids from longitudinal_encounters"):
    for (code,) in c.execute("select d.icd10_code from question_diagnoses qd join diagnoses d using(diagnosis_id) "
                             "where qd.question_id=? and qd.role='correct'", (json.loads(sq)[0],)):
        enc[status(code)] += 1
print("\nreference code status over the 5,602 benchmark encounters:", dict(enc))
