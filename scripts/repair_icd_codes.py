"""Stage 4: validate, repair, flag and de-duplicate the ICD-10-CM codes of the diagnosis nodes, with provenance.

The extraction accepted the LLM's code whenever it was billable and kept unvalidated codes otherwise
(audit #6, #20): 346 reference diagnoses carry header or invalid codes, and billable-but-wrong codes exist
("Leukocytosis" -> D72.819 decreased WBC). Conservative, deterministic repairs:

  validate   every code against the CMS FY2025 tabular list (downloaded to data/ontology if absent);
             status: billable | header | invalid_code | invalid_category | uncoded
  header     -> the billable child whose description says "unspecified", else the first child
  invalid    -> the longest valid prefix's header rule; otherwise uncoded (NULL) and flagged
  semantic   billable codes whose description contradicts the name (with/without) or shares no content token
             are flagged; a with/without contradiction is repaired when exactly one sibling agrees with the name
  merge      nodes with the same normalized display name AND the same SNOMED id collapse into one
             (billable, best description match); graph edges and ground-truth references are repointed
  provenance columns on `diagnoses` (icd10_status, icd10_repaired_from, icd10_repair_reason, icd10_flag,
             merged_into) and a `diagnosis_merges` table; label ICD codes refreshed from the nodes

Applies to the release DB and, for label JSON, the private overlay. Idempotent.

    python scripts/repair_icd_codes.py --db benchmark_v1.3.db [--overlay private/labels_v1.3.db] [--dry-run]
"""

from __future__ import annotations

import argparse
import collections
import difflib
import io
import json
import re
import sqlite3
import sys
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval import private_labels  # noqa: E402

CMS_URL = "https://www.cms.gov/files/zip/2025-code-descriptions-tabular-order.zip"
ONTOLOGY_DIR = Path(__file__).resolve().parent.parent / "data" / "ontology"
STOP = set("of the and with without or in to due other unspecified specified a an by for not on as at type".split())
_WITH = re.compile(r"\bwith(?:out)?\b\s+([a-z][a-z\- ]{2,}?)(?=,|$| with| without| and| or)")


def cms_table() -> dict[str, tuple[bool, str]]:
    """code (no dot) -> (billable, description); downloads the public CMS file if needed."""
    ONTOLOGY_DIR.mkdir(parents=True, exist_ok=True)
    txt = next(ONTOLOGY_DIR.glob("icd10cm*order*2025*.txt"), None)
    if txt is None:
        data = urllib.request.urlopen(CMS_URL, timeout=120).read()
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            name = next(n for n in z.namelist() if re.search(r"icd10cm_order_2025\.txt$", n))
            txt = ONTOLOGY_DIR / Path(name).name
            txt.write_bytes(z.read(name))
    table = {}
    for line in txt.read_text(encoding="latin-1").splitlines():
        table[line[6:13].strip()] = (line[14] == "1", line[77:].strip())
    return table


def nodot(code: str | None) -> str:
    return (code or "").replace(".", "").upper().strip()


def dotted(code: str) -> str:
    return code if len(code) <= 3 else f"{code[:3]}.{code[3:]}"


def tokens(s: str) -> set[str]:
    return {w for w in re.findall(r"[a-z]+", (s or "").lower()) if w not in STOP and len(w) > 2}


def norm_name(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", (s or "").lower())).strip()


def qualifiers(s: str) -> dict[str, bool]:
    """{'perforation': True} for 'with perforation', False for 'without perforation'."""
    out = {}
    for m in re.finditer(r"\b(with|without)\s+([a-z][a-z\- ]*?)(?=,|;|$|\s+(?:with|without|and|or)\b)", (s or "").lower()):
        key = m.group(2).strip().split(" or ")[0].strip()
        if key:
            out[key] = m.group(1) == "with"
    return out


def contradiction(name: str, desc: str) -> str | None:
    qn, qd = qualifiers(name), qualifiers(desc)
    for k, v in qn.items():
        for kd, vd in qd.items():
            if (k in kd or kd in k) and v != vd:
                return f"name says {'with' if v else 'without'} {k}; code says {'with' if vd else 'without'} {kd}"
    return None


class Repairer:
    def __init__(self, cms):
        self.cms = cms
        self.children = collections.defaultdict(list)     # prefix -> billable codes under it
        for code, (bill, _) in cms.items():
            if bill:
                for n in range(3, len(code)):
                    self.children[code[:n]].append(code)
        self.categories = {c[:3] for c in cms}

    def status(self, code: str) -> str:
        if not code:
            return "uncoded"
        if code in self.cms:
            return "billable" if self.cms[code][0] else "header"
        return "invalid_code" if code[:3] in self.categories else "invalid_category"

    def child_for(self, prefix: str) -> str | None:
        kids = sorted(set(self.children.get(prefix, [])))
        if not kids:
            return None
        unspec = [k for k in kids if "unspecified" in self.cms[k][1].lower()]
        return (unspec or kids)[0]

    def repair(self, code: str, name: str) -> tuple[str | None, str | None]:
        """(new code or None, reason) for a header / invalid code."""
        st = self.status(code)
        if st == "header":
            k = self.child_for(code)
            return (k, f"header -> child ({'unspecified' if k and 'unspecified' in self.cms[k][1].lower() else 'first billable'})") if k else (None, None)
        if st == "invalid_code":
            # only a subcategory-level prefix (>= 4 characters) is specific enough to stand in for the code
            for n in range(len(code) - 1, 3, -1):
                if code[:n] in self.cms:
                    k = code[:n] if self.cms[code[:n]][0] else self.child_for(code[:n])
                    if k:
                        return k, f"invalid code -> longest valid prefix {dotted(code[:n])}"
            return "", "invalid code; no valid subcategory prefix; uncoded"
        return None, None

    def sibling_fix(self, code: str, name: str) -> tuple[str | None, str]:
        """Repair a with/without contradiction by the unique sibling that agrees with the name."""
        sibs = [c for c in set(self.children.get(code[:5], [])) | set(self.children.get(code[:4], [])) if c != code]
        agree = [c for c in sibs if contradiction(name, self.cms[c][1]) is None and tokens(self.cms[c][1]) & tokens(name)]
        if not agree:
            return None, "contradiction; no agreeing sibling"
        best = max(agree, key=lambda c: difflib.SequenceMatcher(None, name.lower(), self.cms[c][1].lower()).ratio())
        cur = difflib.SequenceMatcher(None, name.lower(), self.cms[code][1].lower()).ratio() if code in self.cms else 0
        if difflib.SequenceMatcher(None, name.lower(), self.cms[best][1].lower()).ratio() >= cur:
            return best, f"with/without contradiction -> sibling {dotted(best)}"
        return None, "contradiction; sibling not closer"


def ensure_columns(conn: sqlite3.Connection) -> None:
    have = {r[1] for r in conn.execute("pragma table_info(diagnoses)")}
    for col in ("icd10_status", "icd10_repaired_from", "icd10_repair_reason", "icd10_flag"):
        if col not in have:
            conn.execute(f"alter table diagnoses add column {col} TEXT")
    if "merged_into" not in have:
        conn.execute("alter table diagnoses add column merged_into INTEGER")
    conn.execute("create table if not exists diagnosis_merges (old_id INTEGER PRIMARY KEY, new_id INTEGER, reason TEXT)")


def refresh_labels(store: sqlite3.Connection, dx: dict[int, tuple], merge_map: dict[int, int], dry_run: bool) -> int:
    """Repoint diagnosis ids and refresh ICD codes inside ground-truth JSON."""
    n = 0
    for gt_id, task, gt in store.execute("select gt_id, task, ground_truth from benchmark_ground_truth where task in ('patient_diagnosis','evidence_retrieval')").fetchall():
        g = json.loads(gt); changed = False
        if task == "patient_diagnosis":
            for key in ("active_diagnoses", "chronic_conditions"):
                seen_ids = set(); kept = []
                for e in g.get(key, []):
                    d = merge_map.get(e.get("diagnosis_id"), e.get("diagnosis_id"))
                    if d in dx and (d != e.get("diagnosis_id") or e.get("icd10") != dx[d][0] or e.get("display_name") != dx[d][2]):
                        e.update({"diagnosis_id": d, "icd10": dx[d][0], "snomed_id": dx[d][1], "display_name": dx[d][2]}); changed = True
                    if d in seen_ids:
                        changed = True; continue        # two entries collapsed into one node
                    seen_ids.add(d); kept.append(e)
                if kept != g.get(key, []):
                    g[key] = kept
            for eid, lst in (g.get("encounter_diagnosis_map") or {}).items():
                for e in lst:
                    d = merge_map.get(e.get("diagnosis_id"), e.get("diagnosis_id"))
                    if d != e.get("diagnosis_id"):
                        e["diagnosis_id"] = d; changed = True
        else:
            for e in g.get("query_diagnoses", []):
                d = merge_map.get(e.get("diagnosis_id"), e.get("diagnosis_id"))
                if d != e.get("diagnosis_id"):
                    e["diagnosis_id"] = d; changed = True
        if changed:
            n += 1
            if not dry_run:
                store.execute("update benchmark_ground_truth set ground_truth=? where gt_id=?", (json.dumps(g), gt_id))
    return n


def supersede_duplicate_queries(store: sqlite3.Connection) -> int:
    """After merges, two per-diagnosis retrieval rows of one patient may name the same node: keep the first."""
    seen, n = set(), 0
    for gt_id, pid, gt in store.execute("select gt_id, patient_id, ground_truth from benchmark_ground_truth where task='evidence_retrieval' "
                                        "and is_diagnostic order by gt_id").fetchall():
        key = (pid, json.loads(gt)["query_diagnoses"][0]["diagnosis_id"])
        if key in seen:
            store.execute("update benchmark_ground_truth set is_diagnostic=0, exclusion_reason='duplicate query diagnosis after node merge (Stage 4)' where gt_id=?", (gt_id,))
            n += 1
        seen.add(key)
    return n


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--db", default="benchmark_v1.3.db")
    ap.add_argument("--overlay", default=str(private_labels.DEFAULT_PATH))
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    conn = sqlite3.connect(a.db)
    if conn.execute("select count(*) from release_info where key='icd10_repair'").fetchone()[0]:
        print("already applied; nothing to do")
        return 0
    rep = Repairer(cms_table())
    ensure_columns(conn) if not a.dry_run else None
    rows = conn.execute("select diagnosis_id, icd10_code, icd10_desc, snomed_id, display_name from diagnoses order by diagnosis_id").fetchall()
    ref = {d for (d,) in conn.execute("select distinct diagnosis_id from question_diagnoses where role='correct'")}
    stats = collections.Counter()
    updates = {}   # id -> dict(code, desc, status, repaired_from, reason, flag)

    for d, code, desc, sn, name in rows:
        c = nodot(code)
        st = rep.status(c)
        stats[f"status:{st}"] += 1
        new, reason, flag = None, None, None
        if st in ("header", "invalid_code"):
            new, reason = rep.repair(c, name)
        elif st == "invalid_category":
            reason, new = "invalid category; uncoded", ""
        if st == "billable":
            cd = contradiction(name, rep.cms[c][1])
            if cd:
                fix, why = rep.sibling_fix(c, name)
                if fix:
                    new, reason = fix, why
                else:
                    flag = cd
            elif not (tokens(name) & tokens(rep.cms[c][1])):
                flag = "no shared content token between name and code description"
        if new is not None:
            stats["repaired" if new else "uncoded_by_repair"] += 1
            if d in ref:
                stats["reference_repaired" if new else "reference_uncoded"] += 1
        if flag:
            stats["flagged"] += 1
            if d in ref:
                stats["reference_flagged"] += 1
        final = new if new is not None else c
        updates[d] = {"code": dotted(final) if final else None, "desc": rep.cms[final][1] if final in rep.cms else None,
                      "status": rep.status(final) if final else "uncoded", "repaired_from": code if new is not None else None,
                      "reason": reason if new is not None else None, "flag": flag, "snomed": sn, "name": name}

    # merges: same normalized name + same SNOMED id
    groups = collections.defaultdict(list)
    for d, u in updates.items():
        if u["snomed"]:
            groups[(norm_name(u["name"]), u["snomed"])].append(d)
    merge_map: dict[int, int] = {}
    for key, ids in groups.items():
        if len(ids) < 2:
            continue
        def rank(i):
            u = updates[i]
            sim = difflib.SequenceMatcher(None, u["name"].lower(), (u["desc"] or "").lower()).ratio()
            return (u["status"] == "billable", u["flag"] is None, sim, -i)
        keep = max(ids, key=rank)
        for i in ids:
            if i != keep:
                merge_map[i] = keep
    stats["merged_nodes"] = len(merge_map)
    stats["reference_merged"] = sum(1 for i in merge_map if i in ref)
    print("; ".join(f"{k} {v}" for k, v in sorted(stats.items())))
    examples = [(d, u["repaired_from"], u["code"], u["reason"]) for d, u in updates.items() if u["repaired_from"] and d in ref][:6]
    print("example reference repairs:", examples)
    print("diag: T812 in CMS:", "T812" in rep.cms, "| status T812XXA:", rep.status("T812XXA"))
    if a.dry_run:
        return 0

    with conn:
        for d, u in updates.items():
            conn.execute("update diagnoses set icd10_code=?, icd10_desc=?, icd10_status=?, icd10_repaired_from=?, icd10_repair_reason=?, icd10_flag=? where diagnosis_id=?",
                         (u["code"], u["desc"], u["status"], u["repaired_from"], u["reason"], u["flag"], d))
        # the UNIQUE(icd10_code, snomed_id) constraint may now collide for nodes that are not name-duplicates:
        # keep the constraint honest by merging those too (same code + same snomed = same concept)
        dupes = conn.execute("select icd10_code, snomed_id, group_concat(diagnosis_id) from diagnoses where icd10_code is not null and snomed_id is not null "
                             "group by 1,2 having count(*)>1").fetchall()
        for code, sn, ids in dupes:
            ids = sorted(int(x) for x in ids.split(","))
            keep = next((i for i in ids if i not in merge_map), ids[0])
            for i in ids:
                if i != keep and i not in merge_map:
                    merge_map[i] = keep
        # resolve chains
        for i in list(merge_map):
            while merge_map[i] in merge_map:
                merge_map[i] = merge_map[merge_map[i]]
        for old, new in merge_map.items():
            conn.execute("insert or replace into diagnosis_merges (old_id, new_id, reason) values (?,?,?)", (old, new, "same normalized name and SNOMED id, or same code and SNOMED id"))
            # tombstone: keep the node row for provenance but release its code, so that the simulator's
            # UNIQUE(icd10_code, snomed_id) holds (the canonical node carries the code)
            conn.execute("update diagnoses set merged_into=?, icd10_repaired_from=coalesce(icd10_repaired_from, icd10_code), "
                         "icd10_repair_reason=coalesce(icd10_repair_reason, 'merged into canonical node ' || ?), "
                         "icd10_code=null, icd10_desc=null, icd10_status='merged' where diagnosis_id=?", (new, new, old))
            # repoint edges; the SQLite release has no unique indexes, so guard explicitly against the
            # simulator's UNIQUE(question, dx, role) / UNIQUE(dx, finding, relationship)
            for table, cols, key in (("question_diagnoses", "(question_id, diagnosis_id, role, confidence, source)", ("question_id", "role")),
                                     ("diagnosis_findings", "(diagnosis_id, finding_id, relationship, frequency, evidence_source)", ("finding_id", "relationship"))):
                guard = " and ".join(f"t.{k} = o.{k}" for k in key)
                conn.execute(f"insert into {table} {cols} select {cols[1:-1].replace('diagnosis_id', '?')} from {table} o where o.diagnosis_id=? "
                             f"and not exists (select 1 from {table} t where t.diagnosis_id=? and {guard})", (new, old, new))
                conn.execute(f"delete from {table} where diagnosis_id=?", (old,))
        dx = {d: (c, s, n) for d, c, s, n in conn.execute("select diagnosis_id, icd10_code, snomed_id, display_name from diagnoses")}
        n_gt = refresh_labels(conn, dx, merge_map, False)
        n_dup = supersede_duplicate_queries(conn)
        conn.execute("insert or replace into release_info (key, value) values ('icd10_repair', ?)",
                     (f"CMS FY2025 validation with provenance columns (icd10_status, icd10_repaired_from, icd10_repair_reason, icd10_flag, merged_into): "
                      f"{stats['repaired']} codes repaired, {stats['flagged']} flagged, {len(merge_map)} nodes merged (diagnosis_merges); "
                      f"ground-truth ICD codes refreshed in {n_gt} rows (scripts/repair_icd_codes.py, Stage 4)",))
    ov_path = Path(a.overlay)
    if ov_path.exists():
        ov = sqlite3.connect(ov_path)
        with ov:
            n_ov = refresh_labels(ov, dx, merge_map, False)
            n_dup_ov = supersede_duplicate_queries(ov)
            # the release rows of private retrieval instances must carry the same is_diagnostic flag
            for (g,) in ov.execute("select gt_id from benchmark_ground_truth where task='evidence_retrieval' and is_diagnostic=0 and exclusion_reason like 'duplicate%'"):
                conn.execute("update benchmark_ground_truth set is_diagnostic=0, exclusion_reason='duplicate query diagnosis after node merge (Stage 4)' where gt_id=?", (g,))
            conn.commit()
        print(f"overlay: {n_ov} label rows refreshed; {n_dup_ov} duplicate retrieval queries superseded")
    conn.execute("vacuum")
    print(f"applied: {n_gt} release label rows refreshed; {len(merge_map)} nodes merged; {n_dup} duplicate retrieval queries superseded")
    return 0


if __name__ == "__main__":
    sys.exit(main())
