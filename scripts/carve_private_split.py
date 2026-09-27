"""Stage 2.7: a private held-out split whose labels are not in the repository.

v1.3 shipped the 'heldout' labels inside benchmark_v1.3.db, so that split is a second validation
set, not a private test (audit/FINDINGS.md #3). This script carves N patients out of 'train' into a
new 'private' split, stratified like scripts/build_rl_split.py (dominant ICD-10 letter x encounter
bucket, fixed seed), and moves their labels out of the release:

  private/labels_v1.3.db   (gitignored)  full ground_truth rows + relevance_judgments
  benchmark_v1.3.db                       inputs only ({"_labels_removed": true, ...}), no judgments

The scorer overlays the labels back when the file is present (eval/private_labels.py).

    python scripts/carve_private_split.py --db benchmark_v1.3.db --n 200 --dry-run
    python scripts/carve_private_split.py --db benchmark_v1.3.db --n 200
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
from eval.private_labels import DEFAULT_PATH, PRIVATE_SPLIT, strip_labels  # noqa: E402


def bucket(n: int) -> str:
    return "2-3" if n <= 3 else "4-5" if n <= 5 else "6+"


def choose(conn: sqlite3.Connection, n: int, seed: int) -> list[int]:
    """Stratified, seeded choice of n train patients (proportional allocation with carry)."""
    pool = [p for (p,) in conn.execute("select distinct patient_id from benchmark_ground_truth where split='train' and patient_id is not null")]
    nenc = dict(conn.execute("select patient_id, num_encounters from longitudinal_patients"))
    letters = collections.defaultdict(collections.Counter)
    for pid, code in conn.execute(
            "select l.patient_id, d.icd10_code from longitudinal_encounters l, json_each(l.source_question_ids) j "
            "join question_diagnoses qd on qd.question_id = j.value and qd.role='correct' "
            "join diagnoses d on d.diagnosis_id = qd.diagnosis_id"):
        letters[pid][(code or "?")[0]] += 1
    strata = collections.defaultdict(list)
    for p in pool:
        strata[(letters[p].most_common(1)[0][0] if letters[p] else "?", bucket(nenc[p]))].append(p)
    rng = random.Random(seed)
    quota, carry, chosen = n / len(pool), 0.0, []
    for key in sorted(strata):
        ps = sorted(strata[key])
        rng.shuffle(ps)
        want = len(ps) * quota + carry
        k = int(round(want))
        carry = want - k
        chosen += ps[:k]
    return sorted(chosen)


def rebuild_split_check(conn: sqlite3.Connection) -> None:
    """SQLite cannot alter a CHECK constraint: recreate benchmark_ground_truth with 'private' allowed."""
    ddl = conn.execute("select sql from sqlite_master where type='table' and name='benchmark_ground_truth'").fetchone()[0]
    if "'private'" in ddl:
        return
    new_ddl = ddl.replace("CHECK (split IN ('train','public','heldout'))", "CHECK (split IN ('train','public','heldout','private'))")
    assert new_ddl != ddl, "unexpected CHECK constraint text"
    idx = [r[0] for r in conn.execute("select sql from sqlite_master where type='index' and tbl_name='benchmark_ground_truth' and sql is not null")]
    cols = [r[1] for r in conn.execute("pragma table_info(benchmark_ground_truth)")]
    conn.execute("pragma foreign_keys=off")
    conn.execute(new_ddl.replace('"benchmark_ground_truth"', '"benchmark_ground_truth__new"', 1))
    conn.execute(f'insert into benchmark_ground_truth__new ({",".join(cols)}) select {",".join(cols)} from benchmark_ground_truth')
    conn.execute("drop table benchmark_ground_truth")
    conn.execute("alter table benchmark_ground_truth__new rename to benchmark_ground_truth")
    for sql in idx:
        conn.execute(sql)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--db", default="benchmark_v1.3.db")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=20260927)
    ap.add_argument("--out", default=str(DEFAULT_PATH))
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    conn = sqlite3.connect(a.db)
    if conn.execute("select count(*) from benchmark_ground_truth where split=?", (PRIVATE_SPLIT,)).fetchone()[0]:
        print("private split already present; nothing to do")
        return 0
    patients = choose(conn, a.n, a.seed)
    gt_ids = [g for (g,) in conn.execute(
        "select b.gt_id from benchmark_ground_truth b left join longitudinal_encounters l using(encounter_id) "
        f"where coalesce(b.patient_id, l.patient_id) in ({','.join('?' * len(patients))})", patients)]
    n_j = conn.execute(f"select count(*) from relevance_judgments where gt_id in ({','.join('?' * len(gt_ids))})", gt_ids).fetchone()[0]
    print(f"private: {len(patients)} patients, {len(gt_ids)} instances, {n_j} judgments (seed {a.seed})")
    if a.dry_run:
        return 0

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        print(f"refusing to overwrite {out}")
        return 1
    ph = ",".join("?" * len(gt_ids))
    with conn:
        rebuild_split_check(conn)
        conn.execute(f"update benchmark_ground_truth set split=? where gt_id in ({ph})", [PRIVATE_SPLIT, *gt_ids])
        # overlay: full rows for the private instances
        conn.execute("attach database ? as priv", (str(out),))
        conn.execute("create table priv.benchmark_ground_truth as select * from main.benchmark_ground_truth where split=?", (PRIVATE_SPLIT,))
        conn.execute(f"create table priv.relevance_judgments as select * from main.relevance_judgments where gt_id in ({ph})", gt_ids)
        conn.execute("create table priv.release_info (key text primary key, value text)")
        conn.execute("insert into priv.release_info values ('private_split', ?)",
                     (json.dumps({"n_patients": len(patients), "seed": a.seed, "patients": patients}),))
        conn.execute("create index priv.idx_priv_gt on benchmark_ground_truth(gt_id)")
        conn.execute("create index priv.idx_priv_rj on relevance_judgments(gt_id)")
        # release: strip labels, drop judgments
        for gt_id, task, gt in conn.execute(f"select gt_id, task, ground_truth from main.benchmark_ground_truth where gt_id in ({ph})", gt_ids).fetchall():
            conn.execute("update main.benchmark_ground_truth set ground_truth=? where gt_id=?", (json.dumps(strip_labels(task, json.loads(gt))), gt_id))
        conn.execute(f"delete from main.relevance_judgments where gt_id in ({ph})", gt_ids)
        conn.execute("insert or replace into main.release_info (key, value) values ('splits', ?)", (
            "public = 200 patients (reported benchmark); heldout = 268 patients (labels shipped: second validation set); "
            f"train = {800 - len(patients)} patients (RL training pool); private = {len(patients)} patients carved from train "
            f"(seed {a.seed}), labels NOT in this file (eval/private_labels.py)",))
    conn.execute("detach database priv")
    conn.execute("vacuum")
    print(f"applied; labels written to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
