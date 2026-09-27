"""Appendix G: locate Panel C in the release and compare the synthetic register with Panels A/B.

Run from the repo root:  .venv/bin/python audit/scripts/appendix_g_checks.py
"""
import re, sqlite3, statistics
c = sqlite3.connect("benchmark_v1.3.db")
r = c.execute("""select s.id, e.patient_id, e.encounter_id, e.encounter_order, e.encounter_date, e.encounter_type, e.generation_method
                 from encounter_ehr_sections s join longitudinal_encounters e using(encounter_id)
                 where s.section_type='hpi' and s.section_text like 'The patient is a 69-year-old male with a history of neurogenic bladder%'""").fetchone()
print("Panel C = section %s, patient %s, encounter %s (order %s, %s, %s, generation %s)" % r)
pid = r[1]
print("  split:", c.execute("select split from benchmark_ground_truth where patient_id=? limit 1", (pid,)).fetchone()[0],
      "| in the 13-patient physician subset (scripts/retrieval_sections_only.py):", pid in {2610, 2850, 1726, 2046, 2834, 2285, 1853, 2631, 2323, 1969, 1741, 2549, 2376})
for o, d, t, dx in c.execute("""select e.encounter_order, e.encounter_date, e.encounter_type, d.display_name from longitudinal_encounters e
        join question_diagnoses qd on qd.question_id=json_extract(e.source_question_ids,'$[0]') and qd.role='correct'
        join diagnoses d using(diagnosis_id) where e.patient_id=? order by 1""", (pid,)):
    print(f"  enc {o} {d} {t:10s} {dx}")
notes = [t for (t,) in c.execute("select note_text from longitudinal_encounters")]
print("synthetic notes containing a de-identification mask ('[redacted]' or '___'):", sum(1 for t in notes if "[redacted]" in t.lower() or "___" in t))
hpi = [t for (t,) in c.execute("select section_text from encounter_ehr_sections where section_type='hpi'")]
abbr = r"\b(s/p|SOB|CP|c/o|h/o|x3|N/V|abd|pt|w/)\b"
print(f"synthetic HPIs using any common shorthand (s/p, SOB, CP, c/o, h/o, x3, N/V, abd, pt, w/): "
      f"{sum(1 for t in hpi if re.search(abbr, t))}/{len(hpi)}; mean HPI length {statistics.mean(len(t.split()) for t in hpi):.0f} words")
