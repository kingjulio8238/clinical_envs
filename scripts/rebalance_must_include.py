"""Stage 3: whole-patient must-include findings with a per-encounter quota.

The release took the first 20 key findings in chronological order, so for 18-31% of patients the
most recent visit contributed nothing (audit F§1 Stage 4). This selects round-robin across
encounters, critical findings first (a pathognomonic or highly-suggestive edge to that encounter's
correct diagnosis), deduplicated by name, capped at 20. Applies to the release DB and the private
overlay. Idempotent.

    python scripts/rebalance_must_include.py --db benchmark_v1.3.db [--overlay private/labels_v1.3.db]
"""

from __future__ import annotations

import argparse
import collections
import json
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval import private_labels  # noqa: E402

MARK = "round_robin_v1"
CAP = 20
CRITICAL = ("pathognomonic", "highly_suggestive")


def build(conn: sqlite3.Connection):
    enc = collections.defaultdict(list)     # patient -> [(order, qid)]
    for pid, order, sq in conn.execute("select patient_id, encounter_order, source_question_ids from longitudinal_encounters"):
        enc[pid].append((order, json.loads(sq)[0]))
    keyf = collections.defaultdict(list)    # qid -> [(id, fid, name, ftype)]
    for q, rid, fid, name, ftype in conn.execute(
            "select qf.question_id, qf.id, qf.finding_id, cf.display_name, cf.finding_type from question_findings qf "
            "join clinical_findings cf using(finding_id) where qf.relevance='key' and coalesce(qf.present,1)=1 "
            "and cf.finding_type <> 'demographic' order by qf.id"):
        keyf[q].append((rid, fid, name, ftype))
    correct = collections.defaultdict(set)
    for q, d in conn.execute("select question_id, diagnosis_id from question_diagnoses where role='correct'"):
        correct[q].add(d)
    edges = {(d, f): r for d, f, r in conn.execute("select diagnosis_id, finding_id, relationship from diagnosis_findings")}

    def select(pid: int) -> list[dict]:
        queues = []
        for order, q in sorted(enc[pid]):
            crit = [(rid, fid, name, ftype, True) for rid, fid, name, ftype in keyf[q]
                    if any(edges.get((d, fid)) in CRITICAL for d in correct[q])]
            rest = [(rid, fid, name, ftype, False) for rid, fid, name, ftype in keyf[q]
                    if not any(edges.get((d, fid)) in CRITICAL for d in correct[q])]
            queues.append((order, collections.deque(crit + rest)))
        out, seen = [], set()
        while len(out) < CAP and any(qu for _, qu in queues):
            for order, qu in queues:
                while qu:
                    rid, fid, name, ftype, is_crit = qu.popleft()
                    if name.lower() not in seen:
                        seen.add(name.lower())
                        out.append({"display_name": name, "finding_type": ftype, "encounter_order": order, "critical": is_crit})
                        break
                if len(out) >= CAP:
                    break
        return out
    return select


def apply(store: sqlite3.Connection, select, last_order: dict[int, int], dry_run: bool) -> tuple[int, int]:
    """Rewrite must_include_findings on every whole-patient row of `store`. Returns
    (rows updated, rows whose latest encounter is represented)."""
    rows = store.execute("select gt_id, patient_id, ground_truth from benchmark_ground_truth where task='context_summarization' "
                         "and json_extract(ground_truth,'$.variant') is null and json_extract(ground_truth,'$.must_include_findings') is not null").fetchall()
    n = latest = 0
    for gt_id, pid, gt in rows:
        gt = json.loads(gt)
        sel = select(pid)
        n += 1
        latest += any(f["encounter_order"] == last_order.get(pid) for f in sel)
        if not dry_run:
            gt["must_include_findings"] = sel
            gt["selection"] = MARK
            store.execute("update benchmark_ground_truth set ground_truth=? where gt_id=?", (json.dumps(gt), gt_id))
    return n, latest


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--db", default="benchmark_v1.3.db")
    ap.add_argument("--overlay", default=str(private_labels.DEFAULT_PATH))
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    conn = sqlite3.connect(a.db)
    if conn.execute("select count(*) from benchmark_ground_truth where json_extract(ground_truth,'$.selection')=?", (MARK,)).fetchone()[0]:
        print("already applied; nothing to do")
        return 0
    select = build(conn)
    last_order = dict(conn.execute("select patient_id, max(encounter_order) from longitudinal_encounters group by 1"))
    with conn:
        n, latest = apply(conn, select, last_order, a.dry_run)
    print(f"release: {n} whole-patient items rebalanced; latest encounter represented in {latest}")
    ov_path = Path(a.overlay)
    if ov_path.exists():
        ov = sqlite3.connect(ov_path)
        with ov:
            n, latest = apply(ov, select, last_order, a.dry_run)
        print(f"overlay: {n} private items rebalanced; latest encounter represented in {latest}")
    if not a.dry_run:
        conn.execute("insert or replace into release_info (key, value) values ('must_include_selection', ?)",
                     ("round_robin_v1: key findings selected round-robin across encounters, critical first, deduplicated, cap 20 "
                      "(scripts/rebalance_must_include.py, Stage 3)",))
        conn.commit()
        print("applied")
    return 0


if __name__ == "__main__":
    sys.exit(main())
