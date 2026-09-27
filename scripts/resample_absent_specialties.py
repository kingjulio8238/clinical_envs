"""Stage 3: choose each patient's two absent-specialty items at random, in proportion to involvement.

The release took the first two non-involved specialties alphabetically, so every patient got a
Cardiology and a Dermatology item and the specialty name predicted the label (94% / 77% absent;
audit F§1). This resamples the two absent specialties per patient with a seed derived from the
patient id, rewriting `specialty` and `clinical_question` on the existing rows (in the release and,
for private patients, in the overlay too, since both carry those input fields). Candidates are drawn
with probability proportional to how often the specialty is *involved* across the corpus, so that
P(absent | specialty name) stays near the base rate and the name alone predicts nothing. Idempotent.

    python scripts/resample_absent_specialties.py --db benchmark_v1.3.db [--overlay private/labels_v1.3.db]
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval import private_labels  # noqa: E402
from etl.ontology.specialty_map import SPECIALTIES  # noqa: E402

SEED = 20260927
MARK = "resampled_v2"
QUESTION = ("Summarize this patient's chart from a {L} perspective: the active {L} problems and the comorbidities, "
            "labs, and medications relevant to {L} care.")


def label(s: str) -> str:
    return s.replace("_", "/")


def plan(conn: sqlite3.Connection, overlay: sqlite3.Connection | None) -> dict[int, tuple[str, str]]:
    """gt_id -> (new specialty, new clinical question) for every absent row. Private rows carry no
    `involvement` in the release, so they are read from the overlay."""
    sql = ("select gt_id, patient_id, json_extract(ground_truth,'$.specialty'), json_extract(ground_truth,'$.involvement') "
           "from benchmark_ground_truth where json_extract(ground_truth,'$.variant')='specialty_conditioned' {w} order by gt_id")
    rows = conn.execute(sql.format(w="and split != 'private'")).fetchall()
    if overlay is not None:
        rows += overlay.execute(sql.format(w="")).fetchall()
    by_patient = collections.defaultdict(list)
    for r in rows:
        by_patient[r[1]].append(r)
    weight = collections.Counter(s for _, _, s, inv in rows if inv != "absent")   # involvement frequency per specialty
    out = {}
    for pid, items in by_patient.items():
        involved = {s for _, _, s, inv in items if inv != "absent"}
        absent = [g for g, _, _, inv in items if inv == "absent"]
        cands = sorted(set(SPECIALTIES) - involved)
        rng = random.Random(f"{pid}:{SEED}")
        picks: list[str] = []
        while len(picks) < len(absent):
            s = rng.choices(cands, weights=[weight.get(c, 0) + 1 for c in cands])[0]
            if s not in picks:
                picks.append(s)
        for g, s in zip(absent, picks):
            out[g] = (s, QUESTION.format(L=label(s)))
    return out


def apply(store: sqlite3.Connection, changes: dict[int, tuple[str, str]], dry_run: bool) -> int:
    n = 0
    for gt_id, gt in store.execute("select gt_id, ground_truth from benchmark_ground_truth where gt_id in (%s)" % ",".join(map(str, changes)) or "0").fetchall():
        gt = json.loads(gt)
        s, q = changes[gt_id]
        gt["specialty"], gt["clinical_question"], gt["absent_selection"] = s, q, MARK
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
    if conn.execute("select count(*) from benchmark_ground_truth where json_extract(ground_truth,'$.absent_selection')=?", (MARK,)).fetchone()[0]:
        print("already applied; nothing to do")
        return 0
    ov_path = Path(a.overlay)
    ov = sqlite3.connect(ov_path) if ov_path.exists() else None
    changes = plan(conn, ov)
    before = collections.Counter(s for s, _ in (conn.execute("select json_extract(ground_truth,'$.specialty'), 1 from benchmark_ground_truth "
                                                              "where gt_id in (%s)" % ",".join(map(str, changes))).fetchall()))
    after = collections.Counter(s for s, _ in changes.values())
    print(f"absent items: {len(changes)}; specialties before (top 3) {before.most_common(3)}; after (top 3) {after.most_common(3)}")
    with conn:
        n = apply(conn, changes, a.dry_run)
    print(f"release: {n} rows updated")
    if ov is not None:
        with ov:
            n = apply(ov, changes, a.dry_run)
        print(f"overlay: {n} private rows updated")
    if not a.dry_run:
        conn.execute("insert or replace into release_info (key, value) values ('absent_specialties', ?)",
                     (f"resampled_v1: two absent specialties per patient drawn at random from the non-involved ones "
                      f"(seed {SEED}; scripts/resample_absent_specialties.py, Stage 3)",))
        conn.commit()
        print("applied")
    return 0


if __name__ == "__main__":
    sys.exit(main())
