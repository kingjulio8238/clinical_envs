"""Slim the release database without losing information (Stage 7: room for the new task families under
GitHub's 100 MB file limit).

Two reversible, lossless changes:
  1. `relevance_judgments.passage_source` / `source` hold one constant each ('encounter_section',
     'content_rule') on every row and `rationale` is always NULL; the constants are set to NULL here and
     restored by `epic_sim.migrate.sqlite_to_pg` when loading Postgres (recorded in `release_info`).
  2. every `benchmark_ground_truth.ground_truth` JSON is re-serialized without whitespace and without
     ASCII escaping (same object, smaller text).

    python scripts/slim_release.py --db benchmark_v1.3.db [--overlay private/labels_v1.3.db] [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

JUDGMENT_DEFAULTS = {"passage_source": "encounter_section", "source": "content_rule"}


def slim(conn: sqlite3.Connection, dry_run: bool = False) -> dict:
    out: dict = {}
    consts = conn.execute("select count(*), sum(passage_source is not null), sum(source is not null), "
                          "count(distinct coalesce(passage_source,'')), count(distinct coalesce(source,'')) from relevance_judgments").fetchone()
    n, ps, so, dps, dso = consts
    nullable = []                                       # only a column whose every non-NULL value is the default is nulled
    for col, default in JUDGMENT_DEFAULTS.items():
        vals = [v[0] for v in conn.execute(f"select distinct {col} from relevance_judgments where {col} is not null").fetchall()]
        if vals == [default]:
            nullable.append(col)
    out["judgments"] = {"rows": n, "non_null_before": ps, "nulled_columns": nullable}
    before = conn.execute("select sum(length(ground_truth)) from benchmark_ground_truth").fetchone()[0] or 0
    rows = conn.execute("select gt_id, ground_truth from benchmark_ground_truth").fetchall()
    compact = [(json.dumps(json.loads(g), separators=(",", ":"), ensure_ascii=False), gid) for gid, g in rows if g]
    after = sum(len(c) for c, _ in compact)
    out["ground_truth_json"] = {"rows": len(compact), "bytes_before": before, "bytes_after": after}
    if dry_run:
        return out
    with conn:
        for col in nullable:
            conn.execute(f"update relevance_judgments set {col} = null")
        conn.executemany("update benchmark_ground_truth set ground_truth = ? where gt_id = ?", compact)
        try:
            conn.execute("insert or replace into release_info (key, value) values ('release_slimming', ?)",
                         ("relevance_judgments.passage_source/source are NULL in the release and mean 'encounter_section'/'content_rule' "
                          "(restored by epic_sim.migrate.sqlite_to_pg); ground_truth JSON is stored compactly (scripts/slim_release.py, Stage 7)",))
        except sqlite3.OperationalError:
            pass                                                        # the overlay has no release_info
    conn.execute("vacuum")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--db", default="benchmark_v1.3.db")
    ap.add_argument("--overlay", default="private/labels_v1.3.db")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    for path in (a.db, a.overlay):
        if not Path(path).exists():
            continue
        size0 = Path(path).stat().st_size
        conn = sqlite3.connect(path)
        r = slim(conn, a.dry_run)
        conn.close()
        print(f"{path}: {json.dumps(r)}; {size0 / 1e6:.1f} MB -> {Path(path).stat().st_size / 1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
