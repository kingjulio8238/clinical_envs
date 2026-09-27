"""Stage 2.6: regenerate eval/few_shot_examples.json from TRAIN-split patients.

The release's frozen few-shot blocks were public-split instances with their reference answers, and
the live selector picked from public-split runs (audit/FINDINGS.md #12, F§6 5.1). This builds the
same blocks (eval.examples.format_few_shot_block) from the first N train instances per task, from
the SQLite release DB, and records the patients used so that a test can check they never sit in an
evaluation split.

    python scripts/build_few_shot_examples.py --db benchmark_v1.3.db
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval.examples import FROZEN_EXAMPLES_PATH, format_few_shot_block  # noqa: E402

TASKS = ("patient_diagnosis", "context_summarization", "evidence_retrieval", "imaging_indication")


def chart(conn, patient_id: int, max_order: int | None = None) -> str:
    """Mirror eval.tasks.patient_diagnosis._assemble_patient_ehr (and imaging's temporal cut)."""
    rows = conn.execute(
        "select le.encounter_date, le.encounter_type, le.chief_complaint, ees.section_type, ees.section_text "
        "from encounter_ehr_sections ees join longitudinal_encounters le using(encounter_id) "
        "where le.patient_id=? and ees.section_type not in ('assessment','plan') "
        + ("and le.encounter_order<=? " if max_order is not None else "")
        + "order by le.encounter_order, ees.section_order",
        (patient_id, max_order) if max_order is not None else (patient_id,)).fetchall()
    parts, cur = [], None
    for date, etype, cc, stype, text in rows:
        key = f"{date}|{etype}"
        if key != cur:
            cur = key
            parts.append(f"\n{'=' * 60}\nENCOUNTER: {date} ({etype}){' — ' + cc if cc else ''}\n{'=' * 60}")
        parts.append(f"[{stype.upper().replace('_', ' ')}]\n{text}")
    return "\n\n".join(parts)


def expected(conn, task: str, gt_id: int, gt: dict) -> dict:
    """Mirror eval.examples._format_expected_output."""
    if task == "patient_diagnosis":
        return {"active_diagnoses": [{"icd10": d.get("icd10", ""), "name": d.get("display_name", ""), "acuity": d.get("acuity", "acute")}
                                     for d in gt.get("active_diagnoses", [])],
                "chronic_conditions": [{"icd10": d.get("icd10", ""), "name": d.get("display_name", ""), "acuity": d.get("acuity", "chronic")}
                                       for d in gt.get("chronic_conditions", [])]}
    if task == "context_summarization":
        return {"summary": gt.get("reference_summary", "")}
    if task == "evidence_retrieval":
        rows = conn.execute("select passage_id, relevance_grade from relevance_judgments where gt_id=? "
                            "order by relevance_grade desc, passage_id limit 10", (gt_id,)).fetchall()
        return {"rankings": [{"passage_id": p, "grade": g} for p, g in rows]}
    return {"clinical_question": gt.get("inferred_clinical_question", ""), "pre_read_summary": gt.get("pre_read_summary", ""),
            "must_include_findings": gt.get("must_include_findings", []), "differential": gt.get("differential_context", [])}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--db", default="benchmark_v1.3.db")
    ap.add_argument("--n", type=int, default=3)
    a = ap.parse_args()
    conn = sqlite3.connect(f"file:{a.db}?mode=ro", uri=True)
    doc = {"_meta": {"split": "train", "selection": f"first {a.n} train instances per task by gt_id (whole-patient rows for summarization)",
                     "source": "scripts/build_few_shot_examples.py", "patient_ids": {}, "gt_ids": {}}}
    for task in TASKS:
        extra = " and json_extract(b.ground_truth,'$.variant') is null" if task == "context_summarization" else ""
        rows = conn.execute(
            "select b.gt_id, coalesce(b.patient_id, l.patient_id), b.ground_truth, l.encounter_order from benchmark_ground_truth b "
            "left join longitudinal_encounters l using(encounter_id) where b.task=? and b.split='train' and b.is_diagnostic "
            f"and json_extract(b.ground_truth,'$._labels_removed') is null{extra} order by b.gt_id limit ?", (task, a.n)).fetchall()
        examples = []
        for gt_id, pid, gt, order in rows:
            gt = json.loads(gt)
            if task == "evidence_retrieval":
                names = [conn.execute("select display_name from diagnoses where diagnosis_id=?", (d["diagnosis_id"],)).fetchone()[0]
                         for d in gt["query_diagnoses"]]
                text = f"DIAGNOSIS: {', '.join(names)}"
            else:
                text = chart(conn, pid, order if task == "imaging_indication" else None)
            examples.append({"gt_id": gt_id, "task": task, "input_text": text, "expected_output": expected(conn, task, gt_id, gt)})
        doc[task] = format_few_shot_block(examples, task)
        doc["_meta"]["patient_ids"][task] = [r[1] for r in rows]
        doc["_meta"]["gt_ids"][task] = [r[0] for r in rows]
        print(f"{task}: gt_ids {[r[0] for r in rows]} patients {[r[1] for r in rows]} ({len(doc[task])} chars)")
    FROZEN_EXAMPLES_PATH.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n")
    print(f"wrote {FROZEN_EXAMPLES_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
