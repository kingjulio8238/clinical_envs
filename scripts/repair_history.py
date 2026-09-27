"""Stage 4: make each note's history reflect its visit date, and strip tested diagnoses from profiles.

Defects (audit/FINDINGS.md #5, #21, F§12): the profile LLM listed the patient's own tested diagnoses as
"chronic conditions" (500/1,268 profiles), so every note's Active Problem List named later encounters'
diagnoses from visit 0; the profile's surgical history was pasted into every note, so the operation that
treats a visit's diagnosis appears in that visit's own note; and 210 polished HPIs name the encounter's own new
diagnosis.

Rules (deterministic):
  profile   chronic_conditions / comorbidities entries that name a keyed (correct) diagnosis are removed;
            `primary_diagnoses` is nulled (it is the label list)
  PMH       profile-derived problem-list lines naming a keyed diagnosis are removed from every note
  HPI       at a diagnosis's first-keyed encounter, its display name in the HPI is replaced by
            "the presenting problem"
  PSH       a procedure that treats a keyed diagnosis (PROCEDURE_MAP) is visible only in encounters
            after that diagnosis's first-keyed encounter
  notes     note_text is rebuilt from the sections (the release assembler's format); is_modified set

Also regenerates the released patient_profiles.db / .jsonl from the repaired tables. Idempotent.
Retrieval judgments are content-graded, so run scripts/regrade_retrieval_by_content.py --force afterwards.

    python scripts/repair_history.py --db benchmark_v1.3.db [--dry-run]
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import shutil
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

MARK = "history_v1"
MASK = "the presenting problem"

# procedure keyword (regex, lowercase) -> diagnosis keywords it treats
PROCEDURE_MAP: list[tuple[str, tuple[str, ...]]] = [
    (r"appendectomy|appendicectomy", ("appendic",)),
    (r"cholecystectomy", ("cholecyst", "cholelith", "gallstone", "biliary colic", "choledocholith")),
    (r"salping|ectopic", ("ectopic",)),
    (r"thyroidectomy", ("thyroid", "graves", "hashimoto", "thyrotox", "hyperthyroid", "goiter")),
    (r"mastectomy|lumpectomy", ("breast cancer", "breast carcinoma", "ductal carcinoma", "lobular carcinoma")),
    (r"colectomy|hemicolectomy|bowel resection|colostomy|hartmann", ("colon cancer", "colorectal", "diverticul", "bowel obstruction", "ischemic colitis", "volvulus", "toxic megacolon")),
    (r"arthroplasty|hip replacement|knee replacement|open reduction|orif|hip pinning", ("fracture", "osteoarthritis", "avascular necrosis", "osteonecrosis")),
    (r"cabg|bypass graft|coronary artery bypass|pci|angioplasty|stent", ("myocardial infarction", "coronary", "angina", "acute coronary")),
    (r"splenectomy", ("thrombocytopenic purpura", "splenic", "hereditary spherocytosis")),
    (r"nephrectomy", ("renal cell", "wilms", "kidney cancer", "renal mass")),
    (r"hysterectomy|myomectomy", ("uterine", "fibroid", "leiomyoma", "endometrial", "adenomyosis", "cervical cancer")),
    (r"tonsillectomy|adenoidectomy", ("tonsil", "peritonsillar", "adenoid")),
    (r"craniotomy|burr hole|evacuation", ("subdural", "epidural hematoma", "intracerebral", "brain tumor", "glioma", "meningioma")),
    (r"laminectomy|discectomy|spinal fusion", ("disc herniation", "spinal stenosis", "cauda equina", "radiculopathy")),
    (r"pacemaker|icd placement|defibrillator", ("heart block", "bradycardia", "sick sinus", "ventricular tachycardia", "cardiomyopathy")),
    (r"valve replacement|valvuloplasty|valve repair", ("stenosis", "regurgitation", "endocarditis", "rheumatic")),
    (r"amputation", ("gangrene", "osteomyelitis", "diabetic foot", "necrotizing")),
    (r"transplant", ("end-stage renal", "end stage renal", "cirrhosis", "heart failure", "hepatocellular")),
    (r"parathyroidectomy", ("hyperparathyroid", "parathyroid")),
    (r"adrenalectomy", ("pheochromocytoma", "cushing", "adrenal", "aldosteron")),
    (r"thymectomy", ("myasthenia", "thymoma")),
    (r"orchiectomy", ("testicular", "torsion")),
    (r"prostatectomy|turp", ("prostat",)),
    (r"herniorrhaphy|hernia repair", ("hernia",)),
    (r"carotid endarterectomy", ("carotid", "stroke", "transient ischemic")),
    (r"whipple|pancreatectomy", ("pancreatic", "pancreas")),
    (r"lobectomy|pneumonectomy|wedge resection", ("lung cancer", "lung carcinoma", "pulmonary nodule", "bronchogenic")),
    (r"thrombectomy|embolectomy", ("embolism", "thrombosis", "stroke")),
]

_NORM_STRIP = re.compile(r"\(.*?\)|\b(history of|hx of|status post|s/p|stage \w+|type \d|mellitus|chronic|acute|disease|disorder|syndrome)\b|[^a-z0-9 ]")


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", _NORM_STRIP.sub(" ", (s or "").lower())).strip()


def names_match(a: str, b: str) -> bool:
    """Same condition: equal after normalization, one normalized name inside the other (>= 8 chars), or the
    raw diagnosis name inside the raw text (catches parentheticals such as
    "peripheral artery disease (aortoiliac occlusive disease/Leriche syndrome)")."""
    ra, rb = (a or "").lower().strip(), (b or "").lower().strip()
    if len(ra) >= 8 and ra in rb or len(rb) >= 8 and rb in ra:
        return True
    na, nb = norm(a), norm(b)
    if not na or not nb:
        return False
    if na == nb:
        return True
    short, long_ = sorted((na, nb), key=len)
    return len(short) >= 8 and short in long_


def load(conn: sqlite3.Connection):
    enc = collections.defaultdict(list)   # pid -> [(order, eid, date, type, attending, department, qid)]
    for pid, o, eid, d, t, att, dep, sq in conn.execute(
            "select patient_id, encounter_order, encounter_id, encounter_date, encounter_type, attending_name, department, source_question_ids "
            "from longitudinal_encounters order by patient_id, encounter_order"):
        enc[pid].append((o, eid, d, t, att, dep, json.loads(sq)[0]))
    correct = collections.defaultdict(list)  # qid -> [display_name]
    for q, n in conn.execute("select qd.question_id, d.display_name from question_diagnoses qd join diagnoses d using(diagnosis_id) where qd.role='correct'"):
        correct[q].append(n)
    sections = collections.defaultdict(dict)  # eid -> {type: (id, text, order)}
    for sid, eid, st, txt, so in conn.execute("select id, encounter_id, section_type, section_text, section_order from encounter_ehr_sections"):
        sections[eid][st] = (sid, txt or "", so)
    return enc, correct, sections


def first_keyed(patient_encs, correct) -> dict[str, int]:
    """diagnosis display name -> encounter_order at which it is first keyed."""
    out: dict[str, int] = {}
    for o, _, _, _, _, _, q in patient_encs:
        for n in correct[q]:
            out.setdefault(n, o)
    return out


def procedure_hidden(line: str, keyed: dict[str, int], order: int) -> bool:
    low = line.lower()
    for proc_rx, dx_keys in PROCEDURE_MAP:
        if re.search(proc_rx, low):
            for dx, f in keyed.items():
                if any(k in dx.lower() for k in dx_keys) and order <= f:
                    return True
    return False


def assemble_note(date, attending, department, etype, sections: dict[str, tuple]) -> str:
    """The release assembler's format (etl/stages/s09_encounters._assemble_note_text)."""
    order = ["chief_complaint", "hpi", "pmh", "psh", "medications", "allergies", "family_history", "social_history", "ros",
             "vitals", "physical_exam", "labs", "imaging", "pathology", "other_studies", "assessment", "plan"]
    headers = {"chief_complaint": "CHIEF COMPLAINT", "hpi": "HISTORY OF PRESENT ILLNESS", "pmh": "PAST MEDICAL HISTORY",
               "psh": "PAST SURGICAL HISTORY", "medications": "MEDICATIONS", "allergies": "ALLERGIES", "family_history": "FAMILY HISTORY",
               "social_history": "SOCIAL HISTORY", "ros": "REVIEW OF SYSTEMS", "vitals": "VITAL SIGNS", "physical_exam": "PHYSICAL EXAMINATION",
               "labs": "LABORATORY DATA", "imaging": "IMAGING", "pathology": "PATHOLOGY", "other_studies": "OTHER STUDIES",
               "assessment": "ASSESSMENT AND PLAN", "plan": "PLAN"}
    texts = {st: v[1] for st, v in sections.items() if v[1]}
    parts = ["ENCOUNTER NOTE", f"Date: {date}", f"Provider: {attending}, MD", f"Department: {department}", f"Visit Type: {etype}", ""]
    for st in order:
        text = texts.get(st)
        if not text:
            continue
        if st == "plan" and "assessment" in texts:
            continue
        if st == "assessment" and "plan" in texts:
            parts += ["ASSESSMENT AND PLAN:", text + "\n\n" + texts["plan"], ""]
            continue
        parts += [f"{headers.get(st, st.upper())}:", text, ""]
    return "\n".join(parts).rstrip()


def export_profiles(conn: sqlite3.Connection, root: Path) -> None:
    """Regenerate patient_profiles.db / .jsonl (scripts/export_patient_profiles.py column sets)."""
    pcols = ["patient_id", "profile", "age", "sex", "race_ethnicity", "insurance", "num_encounters", "primary_diagnoses", "comorbidities", "generation_seed"]
    ecols = ["encounter_id", "patient_id", "encounter_date", "encounter_type", "chief_complaint", "attending_name", "department", "encounter_order", "note_text", "generation_method"]
    out_db = root / "patient_profiles.db"
    tmp = root / "patient_profiles.db.tmp"
    if tmp.exists():
        tmp.unlink()
    exp = sqlite3.connect(tmp)
    exp.execute(f"create table patients ({', '.join(pcols)})")
    exp.execute(f"create table encounters ({', '.join(ecols)})")
    exp.executemany(f"insert into patients values ({','.join('?' * len(pcols))})", conn.execute(f"select {', '.join(pcols)} from longitudinal_patients order by patient_id"))
    exp.executemany(f"insert into encounters values ({','.join('?' * len(ecols))})", conn.execute(f"select {', '.join(ecols)} from longitudinal_encounters order by patient_id, encounter_order"))
    exp.commit(); exp.execute("vacuum"); exp.close()
    shutil.move(tmp, out_db)
    with open(root / "patient_profiles.jsonl", "w") as fh:
        for row in conn.execute(f"select {', '.join(pcols)} from longitudinal_patients order by patient_id"):
            p = dict(zip(pcols, row))
            for k in ("profile", "primary_diagnoses", "comorbidities"):
                if isinstance(p.get(k), str):
                    try: p[k] = json.loads(p[k])
                    except ValueError: pass
            p["encounters"] = [dict(zip(ecols, r)) for r in conn.execute(
                f"select {', '.join(ecols)} from longitudinal_encounters where patient_id=? order by encounter_order", (p["patient_id"],))]
            fh.write(json.dumps(p, ensure_ascii=False) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--db", default="benchmark_v1.3.db")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    conn = sqlite3.connect(a.db)
    if conn.execute("select count(*) from release_info where key='history_repair'").fetchone()[0]:
        print("already applied; nothing to do")
        return 0
    enc, correct, sections = load(conn)
    stats = collections.Counter()
    edited_sections: dict[int, str] = {}
    edited_encounters: set[int] = set()

    for pid, profile_json, comorb in conn.execute("select patient_id, profile, comorbidities from longitudinal_patients").fetchall():
        encs = enc[pid]
        keyed = first_keyed(encs, correct)
        profile = json.loads(profile_json or "{}")
        # -- profile ---------------------------------------------------------------
        removed = []
        kept = []
        for c in profile.get("chronic_conditions") or []:
            name = c if not isinstance(c, dict) else (c.get("name") or c.get("condition") or "")
            hit = next((d for d in keyed if names_match(str(name), d)), None)
            (removed if hit else kept).append(str(name))
        if removed:
            profile["chronic_conditions"] = kept
            profile["removed_tested_conditions"] = removed
            stats["profiles_changed"] += 1
            stats["conditions_removed"] += len(removed)
            try:
                com = [x for x in json.loads(comorb or "[]") if not any(names_match(str(x), d) for d in keyed)]
            except ValueError:
                com = None
            if not a.dry_run:
                conn.execute("update longitudinal_patients set profile=?, comorbidities=coalesce(?, comorbidities), primary_diagnoses=null where patient_id=?",
                             (json.dumps(profile), json.dumps(com) if com is not None else None, pid))
        elif not a.dry_run:
            conn.execute("update longitudinal_patients set primary_diagnoses=null where patient_id=?", (pid,))
        # -- notes -----------------------------------------------------------------
        for o, eid, date, etype, att, dep, q in encs:
            secs = sections[eid]
            changed = False
            # PMH: drop profile lines that name a keyed diagnosis
            if "pmh" in secs:
                sid, txt, _ = secs["pmh"]
                lines = txt.split("\n")
                new_lines = [ln for ln in lines if not (ln.startswith("- ") and "(diagnosed" not in ln and any(names_match(ln[2:], d) for d in keyed))]
                if new_lines != lines:
                    stats["pmh_lines_removed"] += len(lines) - len(new_lines)
                    new_txt = "\n".join(new_lines)
                    secs["pmh"] = (sid, new_txt, secs["pmh"][2]); edited_sections[sid] = new_txt; changed = True
            # HPI: mask the diagnosis first keyed at this encounter
            if "hpi" in secs:
                sid, txt, _ = secs["hpi"]
                new_txt = txt
                for d, f in keyed.items():
                    if f == o and d in correct[q]:
                        pat = re.compile(re.escape(d), re.I)
                        if pat.search(new_txt):
                            new_txt = pat.sub(MASK, new_txt); stats["hpi_masked"] += 1
                if new_txt != txt:
                    secs["hpi"] = (sid, new_txt, secs["hpi"][2]); edited_sections[sid] = new_txt; changed = True
            # PSH: hide procedures that treat a diagnosis not yet keyed
            if "psh" in secs:
                sid, txt, _ = secs["psh"]
                lines = txt.split("\n")
                new_lines = [ln for ln in lines if not procedure_hidden(ln, keyed, o)]
                if new_lines != lines:
                    stats["psh_lines_hidden"] += len(lines) - len(new_lines)
                    new_txt = "\n".join(new_lines).strip() or "No prior surgical history documented."
                    secs["psh"] = (sid, new_txt, secs["psh"][2]); edited_sections[sid] = new_txt; changed = True
            if changed:
                edited_encounters.add(eid)
                if not a.dry_run:
                    conn.execute("update longitudinal_encounters set note_text=? where encounter_id=?", (assemble_note(date, att, dep, etype, secs), eid))
    print(f"profiles changed {stats['profiles_changed']} (conditions removed {stats['conditions_removed']}); "
          f"PMH lines removed {stats['pmh_lines_removed']}; HPI masked {stats['hpi_masked']}; PSH lines hidden {stats['psh_lines_hidden']}; "
          f"sections edited {len(edited_sections)}; encounters rebuilt {len(edited_encounters)}")
    if a.dry_run:
        return 0
    with conn:
        conn.executemany("update encounter_ehr_sections set section_text=?, is_modified=1 where id=?", [(t, sid) for sid, t in edited_sections.items()])
        conn.execute("insert or replace into release_info (key, value) values ('history_repair', ?)",
                     (f"{MARK}: profile tested-diagnosis strip ({stats['profiles_changed']} profiles), PMH profile lines removed ({stats['pmh_lines_removed']}), "
                      f"HPI self-naming masked ({stats['hpi_masked']}), PSH procedures shown only after their diagnosis ({stats['psh_lines_hidden']} lines); "
                      f"primary_diagnoses nulled (labels live in benchmark_ground_truth); notes rebuilt (scripts/repair_history.py, Stage 4)",))
    export_profiles(conn, Path(a.db).resolve().parent)
    conn.execute("vacuum")
    print("applied; patient_profiles.db / .jsonl regenerated")
    return 0


if __name__ == "__main__":
    sys.exit(main())
