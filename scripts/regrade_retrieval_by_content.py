"""Stage 3: grade every retrieval judgment from what the section text states.

The release graded a section by the finding *types* its encounter's question carried, never by the
text (audit #22: 62% of byte-identical sections carried different grades; a content-blind ranking by
section type scored P@5 0.97). For each per-diagnosis instance this script rebuilds the judgments:

  finding set F(d)  = findings of the patient's source questions whose correct diagnosis is d
                      (with their relevance), plus findings with a typed edge to d
  section grade     = max over f in F(d) that the section text states with the right polarity
                      (present findings stated, absent findings negated), of the release rule:
                        3  key finding with a pathognomonic / highly-suggestive edge to d
                        2  key finding otherwise, or supporting finding with a commonly-seen edge
                        1  supporting or background finding otherwise
                        0  nothing stated
  presence          = eval.concept_match.mentioned_polarity (lexical-not-negated / value polarity)

Applies to the release DB and, for the private split, the operator's label overlay. Idempotent.

    python scripts/regrade_retrieval_by_content.py --db benchmark_v1.3.db [--overlay private/labels_v1.3.db] [--dry-run]
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
from eval.concept_match import Phrase, TextIndex, mentioned_polarity  # noqa: E402

MARK = "content_v1"
EDGE_RELEVANCE = {"pathognomonic": "supporting", "highly_suggestive": "supporting", "commonly_seen": "supporting",
                  "rules_out": "supporting", "risk_factor": "background", "protective": "background"}


def rule(relevance: str | None, edge: str | None) -> int:
    if relevance == "key" and edge in ("pathognomonic", "highly_suggestive"):
        return 3
    if relevance == "key":
        return 2
    if relevance == "supporting" and edge == "commonly_seen":
        return 2
    if relevance in ("supporting", "background"):
        return 1
    return 0


class Graph:
    def __init__(self, conn: sqlite3.Connection):
        self.sections = collections.defaultdict(list)          # patient -> [(passage_id, TextIndex, text)]
        for pid, sid, txt in conn.execute(
                "select l.patient_id, s.id, s.section_text from encounter_ehr_sections s join longitudinal_encounters l using(encounter_id) "
                "where s.section_type not in ('assessment','plan') order by l.encounter_order, s.section_order"):
            self.sections[pid].append((f"ees_{sid}", TextIndex(txt or ""), txt or ""))
        self.q_findings = collections.defaultdict(list)        # qid -> [(fid, relevance, present)]
        for q, fid, rel, pres in conn.execute("select question_id, finding_id, relevance, present from question_findings"):
            self.q_findings[q].append((fid, rel, 1 if pres is None else int(pres)))
        self.finding = {fid: (name, ftype) for fid, name, ftype in conn.execute("select finding_id, display_name, finding_type from clinical_findings")}
        self.edges = {}
        self.edges_by_dx = collections.defaultdict(dict)
        for d, fid, rel in conn.execute("select diagnosis_id, finding_id, relationship from diagnosis_findings"):
            self.edges[(d, fid)] = rel
            self.edges_by_dx[d][fid] = rel
        self.correct = collections.defaultdict(set)            # qid -> {dx}
        for q, d in conn.execute("select question_id, diagnosis_id from question_diagnoses where role='correct'"):
            self.correct[q].add(d)
        self.patient_q = collections.defaultdict(list)
        for pid, sq in conn.execute("select patient_id, source_question_ids from longitudinal_encounters order by encounter_order"):
            self.patient_q[pid] += json.loads(sq)
        self._phrase: dict[int, Phrase] = {}

    def phrase(self, fid: int) -> Phrase:
        if fid not in self._phrase:
            self._phrase[fid] = Phrase(self.finding[fid][0])
        return self._phrase[fid]

    def finding_set(self, patient: int, d: int) -> dict[int, tuple[str, int]]:
        F: dict[int, tuple[str, int]] = {}
        for q in self.patient_q[patient]:
            if d in self.correct[q]:
                for fid, rel, pres in self.q_findings[q]:
                    if fid in self.finding and self.finding[fid][1] != "demographic" and rel:
                        F.setdefault(fid, (rel, pres))
        for fid, rel in self.edges_by_dx.get(d, {}).items():
            if fid not in F and fid in self.finding and self.finding[fid][1] != "demographic":
                F[fid] = (EDGE_RELEVANCE.get(rel, "background"), 1)
        return F

    def grade(self, patient: int, d: int) -> dict[str, int]:
        F = self.finding_set(patient, d)
        out = {}
        for pid, tidx, _ in self.sections[patient]:
            best = 0
            for fid, (rel, pres) in F.items():
                pol = mentioned_polarity(self.phrase(fid), tidx)
                if pol != ("present" if pres else "absent"):
                    continue
                best = max(best, rule(rel, self.edges.get((d, fid))))
                if best == 3:
                    break
            out[pid] = best
        return out


def regrade(conn: sqlite3.Connection, store: sqlite3.Connection, graph: Graph, rows, dry_run: bool) -> collections.Counter:
    """`rows` = (gt_id, patient_id, gt_json) instances whose judgments live in `store`."""
    dist = collections.Counter()
    for gt_id, patient, gt in rows:
        gt = json.loads(gt)
        d = gt["query_diagnoses"][0]["diagnosis_id"]
        grades = graph.grade(patient, d)
        dist.update(grades.values())
        # identical text must get an identical grade within the instance
        by_text = collections.defaultdict(set)
        for pid, _, txt in graph.sections[patient]:
            by_text[txt].add(grades[pid])
        assert all(len(v) == 1 for v in by_text.values()), f"gt {gt_id}: identical sections graded differently"
        if dry_run:
            continue
        gt.update({"num_passages": len(grades), "grade_distribution": {str(k): sum(1 for v in grades.values() if v == k) for k in range(4)},
                   "graded": MARK})
        store.execute("delete from relevance_judgments where gt_id=?", (gt_id,))
        store.executemany("insert into relevance_judgments (gt_id, passage_id, passage_source, relevance_grade, rationale, source) values (?,?,?,?,?,?)",
                          [(gt_id, pid, "encounter_section", g, None, "content_rule") for pid, g in sorted(grades.items())])
        store.execute("update benchmark_ground_truth set ground_truth=?, num_evidence=? where gt_id=?",
                      (json.dumps(gt), sum(1 for v in grades.values() if v >= 2), gt_id))
    return dist


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--db", default="benchmark_v1.3.db")
    ap.add_argument("--overlay", default=str(private_labels.DEFAULT_PATH))
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    conn = sqlite3.connect(a.db)
    if conn.execute("select count(*) from benchmark_ground_truth where json_extract(ground_truth,'$.graded')=?", (MARK,)).fetchone()[0]:
        print("already applied to the release; nothing to do")
        return 0
    graph = Graph(conn)
    before = collections.Counter(g for (g,) in conn.execute("select relevance_grade from relevance_judgments"))
    rows = conn.execute("select gt_id, patient_id, ground_truth from benchmark_ground_truth where task='evidence_retrieval' and is_diagnostic "
                        "and split != 'private' order by gt_id").fetchall()
    with conn:
        dist = regrade(conn, conn, graph, rows, a.dry_run)
    print(f"release: {len(rows)} instances; grades before {dict(sorted(before.items()))} after {dict(sorted(dist.items()))}")
    ov_path = Path(a.overlay)
    if ov_path.exists():
        ov = sqlite3.connect(ov_path)
        rows = ov.execute("select gt_id, patient_id, ground_truth from benchmark_ground_truth where task='evidence_retrieval' and is_diagnostic order by gt_id").fetchall()
        with ov:
            dist = regrade(conn, ov, graph, rows, a.dry_run)
        print(f"overlay: {len(rows)} private instances regraded; grades {dict(sorted(dist.items()))}")
    else:
        print("no overlay present; private instances not regraded")
    if not a.dry_run:
        conn.execute("insert or replace into release_info (key, value) values ('retrieval_grading', ?)",
                     ("content_v1: a section's grade is the release rule applied to the query diagnosis's findings that the section text "
                      "states with the right polarity (scripts/regrade_retrieval_by_content.py, Stage 3)",))
        conn.commit(); conn.execute("vacuum")
        print("applied")
    return 0


if __name__ == "__main__":
    sys.exit(main())
