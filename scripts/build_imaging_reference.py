"""Stage 3: a deterministic reference for the imaging-indication task.

The release scored the inferred clinical question against a question Kimi 2.5 wrote from the full
profile (which carried future history: "stump appendicitis given prior surgery"). This stores, per
imaging instance, `reference_terms` built from the graph only:

  diagnosis    the ordering encounter's correct diagnosis (display name)
  differential the question's distractor diagnoses
  findings     the question's key findings stated as present, of type symptom / sign / imaging finding

The scorer extracts concepts from these terms and computes concept F1 against the prediction
(eval.scoring.reference_terms_text). The LLM question stays as a secondary metric. Terms are labels,
so for private instances they go to the overlay. Idempotent.

    python scripts/build_imaging_reference.py --db benchmark_v1.3.db [--overlay private/labels_v1.3.db]
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

MARK = "graph_terms_v1"
FINDING_TYPES = ("symptom", "sign", "imaging_finding")


def build(conn: sqlite3.Connection):
    enc_q = {e: json.loads(sq)[0] for e, sq in conn.execute("select encounter_id, source_question_ids from longitudinal_encounters")}
    dx = collections.defaultdict(lambda: {"correct": [], "distractor": []})
    for q, role, name in conn.execute("select qd.question_id, qd.role, d.display_name from question_diagnoses qd join diagnoses d using(diagnosis_id) "
                                      "where qd.role in ('correct','distractor') order by qd.id"):
        dx[q][role].append(name)
    keyf = collections.defaultdict(list)
    for q, name in conn.execute("select qf.question_id, cf.display_name from question_findings qf join clinical_findings cf using(finding_id) "
                                f"where qf.relevance='key' and coalesce(qf.present,1)=1 and cf.finding_type in {FINDING_TYPES} order by qf.id"):
        keyf[q].append(name)

    def terms(encounter_id: int) -> dict:
        q = enc_q[encounter_id]
        seen, findings = set(), []
        for n in keyf[q]:
            if n.lower() not in seen:
                seen.add(n.lower()); findings.append(n)
        return {"diagnosis": dx[q]["correct"], "differential": [n for n in dx[q]["distractor"] if n not in dx[q]["correct"]],
                "findings": findings, "source": MARK}
    return terms


def apply(store: sqlite3.Connection, terms, rows, dry_run: bool) -> int:
    n = 0
    for gt_id, eid, gt in rows:
        gt = json.loads(gt)
        gt["reference_terms"] = terms(eid)
        n += 1
        if not dry_run:
            store.execute("update benchmark_ground_truth set ground_truth=? where gt_id=?", (json.dumps(gt), gt_id))
    return n


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--db", default="benchmark_v1.3.db")
    ap.add_argument("--overlay", default=str(private_labels.DEFAULT_PATH))
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    conn = sqlite3.connect(a.db)
    if conn.execute("select count(*) from benchmark_ground_truth where json_extract(ground_truth,'$.reference_terms.source')=?", (MARK,)).fetchone()[0]:
        print("already applied; nothing to do")
        return 0
    terms = build(conn)
    rows = conn.execute("select gt_id, encounter_id, ground_truth from benchmark_ground_truth where task='imaging_indication' and split != 'private' order by gt_id").fetchall()
    with conn:
        n = apply(conn, terms, rows, a.dry_run)
    sample = terms(rows[0][1])
    print(f"release: {n} imaging items; e.g. gt {rows[0][0]}: {sample}")
    ov_path = Path(a.overlay)
    if ov_path.exists():
        ov = sqlite3.connect(ov_path)
        rows = ov.execute("select gt_id, encounter_id, ground_truth from benchmark_ground_truth where task='imaging_indication' order by gt_id").fetchall()
        with ov:
            n = apply(ov, terms, rows, a.dry_run)
        print(f"overlay: {n} private imaging items")
    if not a.dry_run:
        conn.execute("insert or replace into release_info (key, value) values ('imaging_reference', ?)",
                     ("graph_terms_v1: reference_terms = correct diagnosis + distractor differential + key symptom/sign/imaging findings of the "
                      "ordering encounter's question; concept F1 is scored against them (scripts/build_imaging_reference.py, Stage 3)",))
        conn.commit()
        print("applied")
    return 0


if __name__ == "__main__":
    sys.exit(main())
