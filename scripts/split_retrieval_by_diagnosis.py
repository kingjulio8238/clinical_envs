"""Stage 2.4: one evidence-retrieval instance per (patient, diagnosis).

The released retrieval instance is one per patient and its query is the concatenated names of *all*
the patient's reference diagnoses, i.e. the patient-diagnosis answer key (audit/FINDINGS.md #3).
This script replaces it, in the SQLite release DB and in place, with one instance per reference
diagnosis whose query names only that diagnosis. Judgments are derived with the release's own
grading rule (etl/stages/s10_ground_truth.py:_grade_section) restricted to the single diagnosis, so
the label semantics are unchanged: the script first proves that the same rule applied to the full
diagnosis set reproduces every stored judgment exactly, and aborts otherwise.

The patient-level rows are kept for provenance but superseded (is_diagnostic = 0,
exclusion_reason set), so every loader, the scorer's instance list and /env skip them.

    python scripts/split_retrieval_by_diagnosis.py --db benchmark_v1.3.db --dry-run
    python scripts/split_retrieval_by_diagnosis.py --db benchmark_v1.3.db
"""

from __future__ import annotations

import argparse
import collections
import json
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from etl.stages.s10_ground_truth import _grade_section  # noqa: E402

MARK = "per_diagnosis_v1"
SUPERSEDED = "superseded by per-diagnosis retrieval instances (Stage 2.4)"


def load_graph(conn: sqlite3.Connection):
    enc_findings: dict[int, dict] = collections.defaultdict(dict)   # question_id -> {fid: (relevance, ehr_section, ftype)}
    for qid, fid, rel, sec, ftype in conn.execute(
            "select qf.question_id, qf.finding_id, qf.relevance, qf.ehr_section, cf.finding_type "
            "from question_findings qf join clinical_findings cf using(finding_id)"):
        enc_findings[qid][fid] = (rel, sec, ftype)
    df_lookup = {(d, f): r for d, f, r in conn.execute("select diagnosis_id, finding_id, relationship from diagnosis_findings")}
    enc_q = {e: json.loads(sq)[0] for e, sq in conn.execute("select encounter_id, source_question_ids from longitudinal_encounters")}
    sec = {f"ees_{i}": (e, t) for i, e, t in conn.execute("select id, encounter_id, section_type from encounter_ehr_sections")}
    return enc_findings, df_lookup, enc_q, sec


def grade(passage_id, dx_ids, enc_findings, df_lookup, enc_q, sec) -> int:
    eid, stype = sec[passage_id]
    return _grade_section(stype, enc_findings.get(enc_q[eid], {}), set(dx_ids), df_lookup)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--db", default="benchmark_v1.3.db")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    conn = sqlite3.connect(a.db)
    if conn.execute("select count(*) from benchmark_ground_truth where json_extract(ground_truth,'$.derived')=?", (MARK,)).fetchone()[0]:
        print("already applied; nothing to do")
        return 0
    enc_findings, df_lookup, enc_q, sec = load_graph(conn)
    parents = conn.execute("select gt_id, patient_id, split, difficulty, created_at, ground_truth from benchmark_ground_truth "
                           "where task='evidence_retrieval' and is_diagnostic order by gt_id").fetchall()
    judg = collections.defaultdict(dict)
    for g, p, r in conn.execute("select gt_id, passage_id, relevance_grade from relevance_judgments"):
        judg[g][p] = r

    # 1. prove the rule reproduces the stored labels with the full diagnosis set
    mismatches = checked = 0
    for gt_id, *_, gt in parents:
        dx = [d["diagnosis_id"] for d in json.loads(gt)["query_diagnoses"]]
        for pid, stored in judg[gt_id].items():
            checked += 1
            if grade(pid, dx, enc_findings, df_lookup, enc_q, sec) != stored:
                mismatches += 1
    print(f"reproduction check: {checked} stored judgments, {mismatches} mismatches")
    if mismatches:
        print("ABORT: the grading rule does not reproduce the release; refusing to derive per-diagnosis labels")
        return 1

    # 2. derive per-diagnosis instances
    next_id = conn.execute("select max(gt_id) from benchmark_ground_truth").fetchone()[0] + 1
    new_rows, new_judg = [], []
    for gt_id, patient_id, split, difficulty, created_at, gt in parents:
        gt = json.loads(gt)
        for d in gt["query_diagnoses"]:
            dx = d["diagnosis_id"]
            grades = {pid: grade(pid, [dx], enc_findings, df_lookup, enc_q, sec) for pid in judg[gt_id]}
            dist = collections.Counter(grades.values())
            new_gt = {"query_diagnoses": [{"diagnosis_id": dx}], "num_passages": len(grades),
                      "grade_distribution": {str(k): dist.get(k, 0) for k in range(4)},
                      "parent_gt_id": gt_id, "derived": MARK}
            new_rows.append((next_id, "evidence_retrieval", "patient", patient_id, json.dumps(new_gt), difficulty, split,
                             1, sum(1 for v in grades.values() if v >= 2), created_at, 1))
            for pid, v in sorted(grades.items()):
                new_judg.append((next_id, pid, "encounter_section", v, f"section_type={sec[pid][1]}; diagnosis_id={dx}", "rule_based"))
            next_id += 1
    print(f"{len(parents)} patient-level instances -> {len(new_rows)} per-diagnosis instances, {len(new_judg)} judgments")
    if a.dry_run:
        return 0

    with conn:
        conn.executemany("insert into benchmark_ground_truth (gt_id, task, granularity, patient_id, ground_truth, difficulty, split, "
                         "num_diagnoses, num_evidence, created_at, is_diagnostic) values (?,?,?,?,?,?,?,?,?,?,?)", new_rows)
        conn.executemany("insert into relevance_judgments (gt_id, passage_id, passage_source, relevance_grade, rationale, source) "
                         "values (?,?,?,?,?,?)", new_judg)
        conn.execute("update benchmark_ground_truth set is_diagnostic=0, exclusion_reason=? where task='evidence_retrieval' "
                     "and json_extract(ground_truth,'$.derived') is null", (SUPERSEDED,))
        conn.execute("insert or replace into release_info (key, value) values ('retrieval_instances', ?)",
                     (f"one instance per (patient, reference diagnosis): {len(new_rows)} instances derived from {len(parents)} "
                      f"patient-level rows with the release grading rule restricted to one diagnosis (Stage 2.4); "
                      f"patient-level rows kept with is_diagnostic=0",))
    conn.execute("vacuum")
    print("applied")
    return 0


if __name__ == "__main__":
    sys.exit(main())
