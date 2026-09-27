"""Stage 4: patient diagnosis as index-encounter diagnosis.

The longitudinal problem-list task was 80% copyable: its labels appear verbatim as "(diagnosed <date>)"
lines in later notes (audit #2). This materializes one instance per encounter at which a *new* correct
diagnosis is keyed for the patient: the input is the chart up to and including that encounter (no
assessment/plan), the label is that encounter's newly keyed diagnosis, and the patient's earlier keyed
diagnoses are chart-neutral for the instance (`neutral_extra`). Encounters that repeat an earlier
diagnosis yield no instance (documented in `release_info`). The 1,268 longitudinal rows are kept with
`is_diagnostic = 0` as the problem-list extraction task.

Private patients: the full row goes to the overlay and the release keeps the inputs only. Idempotent.

    python scripts/redefine_patient_diagnosis.py --db benchmark_v1.3.db [--overlay private/labels_v1.3.db] [--dry-run]
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval import private_labels  # noqa: E402

MARK = "index_encounter_v1"
SUPERSEDED = "superseded by index-encounter diagnosis (Stage 4): longitudinal problem-list extraction"
CHRONIC = ("chronic", "acute_on_chronic")


def build_instances(conn: sqlite3.Connection, parents: dict[int, tuple[int, str, str]]):
    """parents: patient_id -> (gt_id, split, difficulty). Yields (patient_id, encounter row, gt dict, is_diagnostic)."""
    dx = {d: (icd, sn, name, ac) for d, icd, sn, name, ac in conn.execute("select diagnosis_id, icd10_code, snomed_id, display_name, acuity from diagnoses")}
    correct = collections.defaultdict(list)
    for q, d in conn.execute("select question_id, diagnosis_id from question_diagnoses where role='correct' order by id"):
        correct[q].append(d)
    nondx = set()
    for (g,) in conn.execute("select ground_truth from benchmark_ground_truth where task='patient_diagnosis' and granularity='patient'"):
        for e in json.loads(g).get("active_diagnoses", []) + json.loads(g).get("chronic_conditions", []):
            if e.get("excluded_nondiagnostic"):
                nondx.add(e["diagnosis_id"])
    encs = collections.defaultdict(list)
    for pid, o, eid, date, sq in conn.execute("select patient_id, encounter_order, encounter_id, encounter_date, source_question_ids from longitudinal_encounters order by patient_id, encounter_order"):
        encs[pid].append((o, eid, date, json.loads(sq)[0]))
    def nname(d: int) -> str:
        return " ".join(re.sub(r"[^a-z0-9 ]", " ", dx[d][2].lower()).split())

    out, repeats = [], 0
    for pid, rows in encs.items():
        if pid not in parents:
            continue
        seen: list[int] = []
        seen_names: set[str] = set()
        for o, eid, date, q in rows:
            # new = keyed here for the first time, by node id AND by normalized name (unmerged name-duplicate
            # nodes would otherwise leave a "(diagnosed ...)" line naming the label in the chart)
            new = [d for d in correct[q] if d not in seen and d in dx and nname(d) not in seen_names]
            if not new:
                repeats += 1
                seen += [d for d in correct[q] if d not in seen]
                seen_names |= {nname(d) for d in correct[q] if d in dx}
                continue
            entries = []
            for d in new:
                icd, sn, name, ac = dx[d]
                e = {"icd10": icd, "acuity": ac or "unspecified", "snomed_id": sn, "diagnosis_id": d, "display_name": name,
                     "first_encounter_id": eid, "first_encounter_date": date}
                if d in nondx:
                    e["excluded_nondiagnostic"] = True
                entries.append(e)
            gt = {
                "active_diagnoses": [e for e in entries if e["acuity"] not in CHRONIC],
                "chronic_conditions": [e for e in entries if e["acuity"] in CHRONIC],
                "encounter_diagnosis_map": {str(eid): [{"role": "correct", "diagnosis_id": d} for d in new]},
                "index_encounter": {"encounter_id": eid, "encounter_order": o, "encounter_date": date},
                "neutral_extra": sorted({(dx[d][0] or "")[:3] for d in seen if dx[d][0]}),
                "derived": MARK, "parent_gt_id": parents[pid][0],
            }
            # unwinnable if every label is excluded or has no billable ICD-10 code (the scorer matches by code)
            is_dx = 0 if all(e.get("excluded_nondiagnostic") or not e["icd10"] for e in entries) else 1
            out.append((pid, o, eid, gt, is_dx))
            seen += [d for d in correct[q] if d not in seen]
            seen_names |= {nname(d) for d in correct[q] if d in dx}
    return out, repeats


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--db", default="benchmark_v1.3.db")
    ap.add_argument("--overlay", default=str(private_labels.DEFAULT_PATH))
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    conn = sqlite3.connect(a.db)
    if conn.execute("select count(*) from benchmark_ground_truth where json_extract(ground_truth,'$.derived')=?", (MARK,)).fetchone()[0]:
        print("already applied; nothing to do")
        return 0
    ov_path = Path(a.overlay)
    ov = sqlite3.connect(ov_path) if ov_path.exists() else None
    parents = {p: (g, s, d) for g, p, s, d in conn.execute(
        "select gt_id, patient_id, split, difficulty from benchmark_ground_truth where task='patient_diagnosis' and granularity='patient'")}
    instances, repeats = build_instances(conn, parents)
    next_id = max(conn.execute("select max(gt_id) from benchmark_ground_truth").fetchone()[0],
                  ov.execute("select max(gt_id) from benchmark_ground_truth").fetchone()[0] if ov else 0) + 1
    by_split = collections.Counter(parents[p][1] for p, *_ in instances)
    print(f"{len(instances)} index-encounter instances from {sum(len(v) for v in [instances])} first-occurrence encounters; "
          f"{repeats} repeat encounters excluded; by split {dict(by_split)}; non-diagnostic {sum(1 for *_, d in instances if not d)}")
    if a.dry_run:
        return 0
    created_at = conn.execute("select created_at from benchmark_ground_truth where gt_id=?", (next(iter(parents.values()))[0],)).fetchone()[0]
    rows_release, rows_overlay = [], []
    for pid, o, eid, gt, is_dx in instances:
        parent_gt, split, difficulty = parents[pid]
        reason = None if is_dx else "index-encounter label is excluded or has no billable ICD-10 code (Stage 4)"
        base = (next_id, "patient_diagnosis", "encounter", pid, eid, difficulty, split, len(gt["active_diagnoses"]) + len(gt["chronic_conditions"]), created_at, is_dx, reason)
        if split == private_labels.PRIVATE_SPLIT and ov is not None:
            rows_overlay.append(base[:5] + (json.dumps(gt),) + base[5:])
            rows_release.append(base[:5] + (json.dumps(private_labels.strip_labels("patient_diagnosis", gt)),) + base[5:])
        else:
            rows_release.append(base[:5] + (json.dumps(gt),) + base[5:])
        next_id += 1
    sql = ("insert into benchmark_ground_truth (gt_id, task, granularity, patient_id, encounter_id, ground_truth, difficulty, split, num_diagnoses, created_at, is_diagnostic, exclusion_reason) "
           "values (?,?,?,?,?,?,?,?,?,?,?,?)")
    with conn:
        conn.executemany(sql, rows_release)
        conn.execute("update benchmark_ground_truth set is_diagnostic=0, exclusion_reason=? where task='patient_diagnosis' and granularity='patient'", (SUPERSEDED,))
        conn.execute("insert or replace into release_info (key, value) values ('patient_diagnosis_instances', ?)",
                     (f"{MARK}: one instance per encounter at which a new correct diagnosis is keyed ({len(instances)} instances; {repeats} repeat "
                      f"encounters excluded); input = chart up to the index encounter; earlier keyed diagnoses are chart-neutral (neutral_extra); "
                      f"the 1,268 longitudinal rows are kept with is_diagnostic=0 as the problem-list extraction task (scripts/redefine_patient_diagnosis.py, Stage 4)",))
    if ov is not None:
        with ov:
            ov.executemany(sql, rows_overlay)
            ov.execute("update benchmark_ground_truth set is_diagnostic=0, exclusion_reason=? where task='patient_diagnosis' and granularity='patient'", (SUPERSEDED,))
        print(f"overlay: {len(rows_overlay)} private instances written")
    conn.execute("vacuum")
    print(f"applied: {len(rows_release)} rows in the release")
    return 0


if __name__ == "__main__":
    sys.exit(main())
