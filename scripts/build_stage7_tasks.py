"""Stage 7: build the new task families from the released graph and the index-encounter instances.

Every instance derives from a scorable index-encounter `patient_diagnosis` row (its `parent_gt_id`) and
inherits the patient's split and cutoff. Private-split labels go to the operator's overlay; the release
keeps the stripped inputs. Idempotent (a `derived` marker per family). See audit/STAGE7_TODO.md.

    python scripts/build_stage7_tasks.py --db benchmark_v1.3.db [--overlay private/labels_v1.3.db] [--dry-run]
        [--cap-train N]   cap the train-split instances per family (DB size)
    python scripts/build_stage7_tasks.py --patch-order-context   add the order context (Stage-8 audit R1) to an
        already-built release + overlay in place (idempotent; a fresh build writes it directly)
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
from eval import stage7 as S7  # noqa: E402

MARK = "stage7_v1"


def _parents(conn: sqlite3.Connection, ov: sqlite3.Connection | None) -> list[dict]:
    rows = conn.execute("select gt_id, patient_id, encounter_id, split, difficulty, ground_truth, created_at from benchmark_ground_truth "
                        "where task='patient_diagnosis' and granularity='encounter' and is_diagnostic order by gt_id").fetchall()
    out = []
    for gt_id, pid, eid, split, diff, gt, created in rows:
        gt = json.loads(gt)
        if gt.get(private_labels.REMOVED_FLAG):
            full = ov.execute("select ground_truth from benchmark_ground_truth where gt_id=?", (gt_id,)).fetchone() if ov else None
            if not full:
                continue
            gt = json.loads(full[0])
        out.append({"gt_id": gt_id, "patient_id": pid, "encounter_id": eid, "split": split, "difficulty": diff, "gt": gt, "created_at": created})
    return out


def _labels(p: dict) -> list[dict]:
    return [d for d in p["gt"].get("active_diagnoses", []) + p["gt"].get("chronic_conditions", []) if not d.get("excluded_nondiagnostic")]


def build(conn: sqlite3.Connection, parents: list[dict]) -> dict[str, list[tuple[dict, dict, int]]]:
    """task -> [(parent, gt, is_diagnostic)]."""
    out: dict[str, list] = {t: [] for t in S7.NEW_TASKS}
    ages = dict(conn.execute("select patient_id, age from longitudinal_patients").fetchall())
    for k, p in enumerate(parents):
        eid = p["encounter_id"]
        labels = _labels(p)
        label_ids = [d["diagnosis_id"] for d in labels]
        base = {"index_encounter": p["gt"].get("index_encounter"), "parent_gt_id": p["gt_id"], "derived": MARK}

        # differential
        label_codes = {d.get("icd10") for d in labels if d.get("icd10")}
        # a distractor must be coded (the reward is ICD credit) and must not be the label itself (the source
        # questions list the answer among the choices: audit "correct dx also listed as a distractor")
        distr = [d for d in S7.distractors(conn, eid) if d["icd10"] and d["diagnosis_id"] not in label_ids and d["icd10"] not in label_codes]
        if distr and labels:
            out["differential_diagnosis"].append((p, {**base, "correct": [{"diagnosis_id": d["diagnosis_id"], "icd10": d["icd10"], "display_name": d["display_name"]} for d in labels],
                                                       "distractors": distr}, 1))

        # test selection
        orderable = S7.encounter_findings(conn, eid, S7.ORDERABLE_TYPES, label_ids)
        ctx = S7.order_context(S7.index_sections(conn, eid), orderable)
        orderable = [{**f, "context": ctx.get(f["name"], "")} for f in orderable]
        disc = [f["name"] for f in orderable if f["present"] and f["edge"] in S7.DISCRIMINATING_EDGES]
        if disc and labels and labels[0].get("icd10"):
            out["test_selection"].append((p, {**base, "diagnosis": {"icd10": labels[0]["icd10"], "name": labels[0]["display_name"], "diagnosis_id": labels[0]["diagnosis_id"]},
                                               "orderable": [[f[k2] for k2 in S7.ORDERABLE_FIELDS] for f in orderable],   # compact rows
                                               "discriminating": disc, "n_needed": len(disc), "hidden_section_types": sorted(S7.RESULT_SECTIONS)}, 1))

        # error detection (rotate the type by position so the families are balanced)
        sections = S7.index_sections(conn, eid)
        for j in range(len(S7.ERROR_TYPES)):
            et = S7.ERROR_TYPES[(k + j) % len(S7.ERROR_TYPES)]
            lab = S7.inject_error(sections, et, stated_age=ages.get(p["patient_id"]))
            if lab is not None:
                lab = {k2: v for k2, v in lab.items() if k2 not in ("original", "injected")}   # stored once, in the override
                out["error_detection"].append((p, {**base, **lab}, 1))
                break

        # lab triage
        triage = [f for f in S7.encounter_findings(conn, eid, S7.TRIAGE_TYPES, label_ids) if f["present"]]
        relevant = [f["name"] for f in triage if f["relevance"] in ("key", "supporting")]
        background = [f["name"] for f in triage if f["relevance"] not in ("key", "supporting")]
        keys = [f for f in triage if f["relevance"] == "key"]
        if len(triage) >= 3 and keys and background:
            urgent = max(keys, key=lambda f: (S7.EDGE_RANK.get(f["edge"], 0), -triage.index(f)))
            out["lab_triage"].append((p, {**base, "relevant": relevant, "background": background,
                                          "most_urgent": urgent["name"] if S7.EDGE_RANK.get(urgent["edge"], 0) else None}, 1))

        # atypical variant
        allf = S7.encounter_findings(conn, eid, ("symptom", "sign", "lab_value", "imaging_finding", "procedure_result", "vital_sign", "history_item"), label_ids)
        strong = [f["name"] for f in allf if f["present"] and f["edge"] in S7.DISCRIMINATING_EDGES]
        if strong and labels:
            overrides, masked = S7.mask_findings(sections, strong)
            if masked:
                slim = lambda d: {k2: d[k2] for k2 in ("icd10", "acuity", "display_name", "diagnosis_id", "excluded_nondiagnostic") if k2 in d}
                out["atypical_diagnosis"].append((p, {**base, "variant": "atypical", "masked_findings": masked, "section_overrides": overrides,
                                                      "active_diagnoses": [slim(d) for d in p["gt"].get("active_diagnoses", [])],
                                                      "chronic_conditions": [slim(d) for d in p["gt"].get("chronic_conditions", [])],
                                                      "neutral_extra": p["gt"].get("neutral_extra", [])}, 1))
    return out


def widen_task_check(conn: sqlite3.Connection) -> bool:
    """The release table carries CHECK (task IN (...)); SQLite cannot alter it, so rebuild the table with
    the Stage-7 task names added. Returns True when the table was rebuilt."""
    ddl = conn.execute("select sql from sqlite_master where name='benchmark_ground_truth'").fetchone()
    if not ddl or "CHECK (task IN (" not in ddl[0] or "'differential_diagnosis'" in ddl[0]:
        return False
    old = ddl[0][ddl[0].index("CHECK (task IN ("):]
    old = old[: old.index("))") + 2]
    tasks = ", ".join(f"'{t}'" for t in ("patient_diagnosis", "context_summarization", "evidence_retrieval", "imaging_indication", *S7.NEW_TASKS))
    new_ddl = ddl[0].replace(old, f"CHECK (task IN ({tasks}))").replace('CREATE TABLE "benchmark_ground_truth"', 'CREATE TABLE "benchmark_ground_truth_new"')
    indexes = [r[0] for r in conn.execute("select sql from sqlite_master where type='index' and tbl_name='benchmark_ground_truth' and sql is not null")]
    with conn:
        conn.execute(new_ddl)
        conn.execute("insert into benchmark_ground_truth_new select * from benchmark_ground_truth")
        conn.execute("drop table benchmark_ground_truth")
        conn.execute("alter table benchmark_ground_truth_new rename to benchmark_ground_truth")
        for ix in indexes:
            conn.execute(ix)
    return True


ORDER_CTX_MARK = "order_ctx_v1"


def patch_order_context(conn: sqlite3.Connection, ov: sqlite3.Connection | None) -> int:
    """Rewrite every test_selection row's `orderable` with the context field, in the release (public/train) and
    the overlay (private labels). Returns the number of rows rewritten."""
    n = 0
    for db in (conn, ov):
        if db is None:
            continue
        rows = db.execute("select gt_id, encounter_id, ground_truth from benchmark_ground_truth where task='test_selection'").fetchall()
        upd = []
        for gt_id, eid, gt in rows:
            gt = json.loads(gt)
            if "orderable" not in gt:                      # a stripped private row: its labels live in the overlay
                continue
            fs = [{k: v for k, v in f.items() if k != "context"} for f in S7.orderable_from_gt(gt)]
            ctx = S7.order_context(S7.index_sections(conn, eid), fs)
            gt["orderable"] = [[f[k] if k != "context" else ctx.get(f["name"], "") for k in S7.ORDERABLE_FIELDS] for f in fs]
            upd.append((json.dumps(gt), gt_id))
        with db:
            db.executemany("update benchmark_ground_truth set ground_truth=? where gt_id=?", upd)
        n += len(upd)
    with conn:
        conn.execute("insert or replace into release_info (key, value) values ('stage7_order_context', ?)",
                     (f"{ORDER_CTX_MARK}: test_selection orderable findings carry the content tokens of the result text that "
                      f"documents them (order matching by test name; scripts/build_stage7_tasks.py --patch-order-context)",))
    return n


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--db", default="benchmark_v1.3.db")
    ap.add_argument("--overlay", default=str(private_labels.DEFAULT_PATH))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--cap-train", type=int, default=None, help="cap train-split instances per family")
    ap.add_argument("--patch-order-context", action="store_true")
    a = ap.parse_args()
    conn = sqlite3.connect(a.db)
    if a.patch_order_context:
        ov = sqlite3.connect(a.overlay) if Path(a.overlay).exists() else None
        n = patch_order_context(conn, ov)
        conn.execute("vacuum")
        print(f"order context: {n} test_selection rows rewritten")
        return 0
    if conn.execute("select count(*) from benchmark_ground_truth where json_extract(ground_truth,'$.derived')=?", (MARK,)).fetchone()[0]:
        print("already applied; nothing to do")
        return 0
    ov_path = Path(a.overlay)
    ov = sqlite3.connect(ov_path) if ov_path.exists() else None
    parents = _parents(conn, ov)
    fam = build(conn, parents)
    if a.cap_train:
        for t, rows in fam.items():
            kept, n_train = [], 0
            for p, gt, isd in rows:
                if p["split"] == "train":
                    if n_train >= a.cap_train:
                        continue
                    n_train += 1
                kept.append((p, gt, isd))
            fam[t] = kept
    for t, rows in fam.items():
        by = collections.Counter(p["split"] for p, _, _ in rows)
        print(f"{t:24s} {len(rows):5d} instances  {dict(sorted(by.items()))}")
    if a.dry_run:
        return 0
    next_id = max(conn.execute("select max(gt_id) from benchmark_ground_truth").fetchone()[0],
                  ov.execute("select max(gt_id) from benchmark_ground_truth").fetchone()[0] if ov else 0) + 1
    sql = ("insert into benchmark_ground_truth (gt_id, task, granularity, patient_id, encounter_id, ground_truth, difficulty, split, num_diagnoses, created_at, is_diagnostic) "
           "values (?,?,?,?,?,?,?,?,?,?,?)")
    rel, ovr = [], []
    for t, rows in fam.items():
        for p, gt, isd in rows:
            base = (next_id, t, "encounter", p["patient_id"], p["encounter_id"], p["difficulty"], p["split"], None, p["created_at"], isd)
            if p["split"] == private_labels.PRIVATE_SPLIT and ov is not None:
                ovr.append(base[:5] + (json.dumps(gt),) + base[5:])
                rel.append(base[:5] + (json.dumps(private_labels.strip_labels(t, gt)),) + base[5:])
            else:
                rel.append(base[:5] + (json.dumps(gt),) + base[5:])
            next_id += 1
    counts = {t: len(rows) for t, rows in fam.items()}
    if widen_task_check(conn):
        print("release: task CHECK widened to the Stage-7 task names")
    if ov is not None and widen_task_check(ov):
        print("overlay: task CHECK widened")
    with conn:
        conn.executemany(sql, rel)
        conn.execute("insert or replace into release_info (key, value) values ('stage7_tasks', ?)",
                     (f"{MARK}: differential_diagnosis, test_selection, error_detection, lab_triage, atypical_diagnosis derived from the "
                      f"index-encounter diagnosis instances ({json.dumps(counts)}); labels from the graph tables and deterministic text "
                      f"transforms of the index encounter (scripts/build_stage7_tasks.py, Stage 7)",))
    if ov is not None:
        with ov:
            ov.executemany(sql, ovr)
        print(f"overlay: {len(ovr)} private instances written")
    conn.execute("vacuum")
    print(f"applied: {len(rel)} rows in the release")
    return 0


if __name__ == "__main__":
    sys.exit(main())
