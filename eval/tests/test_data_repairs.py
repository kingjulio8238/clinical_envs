"""Stage 4 regression suite: the release data no longer leaks labels through history, profiles or codes.

Invariants, not snapshots: each test states a property the repaired data must keep. Runs on the released
SQLite file; the ICD test uses the public CMS FY2025 table when it is present under data/ontology.
"""

from __future__ import annotations

import collections
import json
import re
import sqlite3

import pytest

from eval import degenerate as D

pytestmark = pytest.mark.reward_hacking

SPLIT = "public"


@pytest.fixture(scope="session")
def db():
    if not D.DEFAULT_DB.exists():
        pytest.skip(f"{D.DEFAULT_DB.name} not present")
    return D.ReleaseDB()


@pytest.fixture(scope="session")
def keyed(db):
    """patient -> {diagnosis display name (lower): first encounter_order}."""
    corr = collections.defaultdict(list)
    for q, n in db.q("select qd.question_id, d.display_name from question_diagnoses qd join diagnoses d using(diagnosis_id) where qd.role='correct'"):
        corr[q].append(n.lower())
    out: dict[int, dict[str, int]] = collections.defaultdict(dict)
    for pid, o, sq in db.q("select patient_id, encounter_order, source_question_ids from longitudinal_encounters order by patient_id, encounter_order"):
        for n in corr[json.loads(sq)[0]]:
            out[pid].setdefault(n, o)
    return out


# ---------------------------------------------------------------------------
# history and profiles
# ---------------------------------------------------------------------------

def test_profiles_do_not_list_tested_diagnoses(db, keyed):
    from scripts.repair_history import names_match
    offenders = []
    for pid, prof in db.q("select patient_id, profile from longitudinal_patients"):
        for c in json.loads(prof).get("chronic_conditions") or []:
            if any(names_match(str(c), d) for d in keyed[pid]):
                offenders.append((pid, c))
    assert not offenders, offenders[:5]


def test_primary_diagnoses_column_is_not_a_label_list(db):
    assert db.q("select count(*) from longitudinal_patients where primary_diagnoses is not null")[0][0] == 0


def test_hpi_never_names_its_own_new_diagnosis(db, keyed):
    hits = 0
    for pid, o, eid, sq in db.q("select patient_id, encounter_order, encounter_id, source_question_ids from longitudinal_encounters"):
        names = {n for n in keyed[pid] if keyed[pid][n] == o}
        hpi = (db.q("select section_text from encounter_ehr_sections where encounter_id=? and section_type='hpi'", eid) or [("",)])[0][0] or ""
        hits += any(n in hpi.lower() for n in names)
    assert hits == 0


def test_procedures_follow_their_diagnosis(db, keyed):
    from scripts.repair_history import procedure_hidden
    leaks = []
    for pid, o, eid in db.q("select patient_id, encounter_order, encounter_id from longitudinal_encounters"):
        psh = (db.q("select section_text from encounter_ehr_sections where encounter_id=? and section_type='psh'", eid) or [("",)])[0][0] or ""
        for line in psh.split("\n"):
            if procedure_hidden(line, {n: f for n, f in keyed[pid].items()}, o):
                leaks.append((pid, o, line))
    assert not leaks, leaks[:5]


def test_profile_problem_lines_do_not_name_keyed_diagnoses(db, keyed):
    from scripts.repair_history import names_match
    leaks = 0
    for pid, eid in db.q("select patient_id, encounter_id from longitudinal_encounters"):
        pmh = (db.q("select section_text from encounter_ehr_sections where encounter_id=? and section_type='pmh'", eid) or [("",)])[0][0] or ""
        for line in pmh.split("\n"):
            if line.startswith("- ") and "(diagnosed" not in line and any(names_match(line[2:], d) for d in keyed[pid]):
                leaks += 1
    assert leaks == 0


def test_note_text_matches_sections(db):
    """note_text is rebuilt from the sections: every section body appears verbatim in its note."""
    missing = 0
    for eid, note in db.q("select encounter_id, note_text from longitudinal_encounters where encounter_id in "
                          "(select encounter_id from longitudinal_encounters order by encounter_id limit 400)"):
        for (txt,) in db.q("select section_text from encounter_ehr_sections where encounter_id=? and section_type not in ('assessment','plan')", eid):
            if txt and txt.strip() and txt.strip() not in note:
                missing += 1
    assert missing == 0


# ---------------------------------------------------------------------------
# index-encounter diagnosis
# ---------------------------------------------------------------------------

def test_index_encounter_instances_cover_first_occurrences(db, keyed):
    insts = db.instances("patient_diagnosis", SPLIT)
    assert insts and all(i["encounter_id"] for i in insts)
    assert all(i["gt"].get("derived") == "index_encounter_v1" for i in insts)
    # one instance per encounter at which a diagnosis (by node id) is keyed for the first time
    corr = collections.defaultdict(list)
    for q, d in db.q("select question_id, diagnosis_id from question_diagnoses where role='correct'"):
        corr[q].append(d)
    first_by_id: dict[int, dict[int, int]] = collections.defaultdict(dict)
    for pid, o, sq in db.q("select patient_id, encounter_order, source_question_ids from longitudinal_encounters order by patient_id, encounter_order"):
        for d in corr[json.loads(sq)[0]]:
            first_by_id[pid].setdefault(d, o)
    per_patient = collections.Counter(i["patient_id"] for i in insts)
    for pid, n in per_patient.items():
        assert n <= len(set(first_by_id[pid].values()))
    superseded = db.q("select count(*) from benchmark_ground_truth where task='patient_diagnosis' and granularity='patient' and is_diagnostic=0")[0][0]
    assert superseded == 1268


def test_index_chart_has_no_dated_line_for_its_label(db):
    """The '(diagnosed <date>)' problem-list lines up to the index encounter never name the label."""
    leaks = 0
    for i in db.instances("patient_diagnosis", SPLIT):
        names = {d["display_name"].lower() for d in i["gt"]["active_diagnoses"] + i["gt"]["chronic_conditions"]}
        chart = db.chart_text(i["patient_id"], i["encounter_id"]).lower()
        dated = re.findall(r"^- (.+?) \(diagnosed \d{4}-\d\d-\d\d\)", chart, re.M)
        leaks += any(n in dated for n in names)
    assert leaks == 0


def test_neutral_extra_is_earlier_keyed_codes(db):
    """neutral_extra = 3-character categories of the diagnoses (by node id) keyed at earlier encounters."""
    code = {d: (c or "")[:3] for d, c in db.q("select diagnosis_id, icd10_code from diagnoses")}
    corr = collections.defaultdict(list)
    for q, d in db.q("select question_id, diagnosis_id from question_diagnoses where role='correct'"):
        corr[q].append(d)
    encs = collections.defaultdict(list)
    for pid, o, sq in db.q("select patient_id, encounter_order, source_question_ids from longitudinal_encounters order by patient_id, encounter_order"):
        encs[pid].append((o, json.loads(sq)[0]))
    for i in db.instances("patient_diagnosis", SPLIT)[:300]:
        o = i["gt"]["index_encounter"]["encounter_order"]
        earlier = {code[d] for oo, q in encs[i["patient_id"]] if oo < o for d in corr[q] if code.get(d)}
        assert set(i["gt"]["neutral_extra"]) == earlier, i["gt_id"]


def test_copy_policy_is_near_zero_on_index_instances(db):
    insts = db.instances("patient_diagnosis", SPLIT)[:150]
    assert D.primary(db, "patient_diagnosis", D.run_policy(db, "patient_diagnosis", "copy_problem_list_plus_chronic", insts), insts) <= 0.15


# ---------------------------------------------------------------------------
# ICD codes and provenance
# ---------------------------------------------------------------------------

def test_icd_codes_are_billable_with_provenance(db):
    cms = next(D.ROOT.joinpath("data", "ontology").glob("icd10cm*order*2025*.txt"), None)
    if cms is None:
        pytest.skip("CMS FY2025 table not present under data/ontology")
    billable = {line[6:13].strip() for line in cms.read_text(encoding="latin-1").splitlines() if line[14] == "1"}
    bad = [(d, c) for d, c in db.q("select diagnosis_id, icd10_code from diagnoses where icd10_code is not null and merged_into is null")
           if c.replace(".", "").upper() not in billable]
    assert not bad, bad[:5]
    assert db.q("select count(*) from diagnoses where icd10_status is null")[0][0] == 0
    assert db.q("select count(*) from diagnoses where icd10_repaired_from is not null and icd10_repair_reason is null")[0][0] == 0


def test_merged_nodes_have_no_edges_and_labels_use_node_codes(db):
    assert db.q("select count(*) from question_diagnoses where diagnosis_id in (select diagnosis_id from diagnoses where merged_into is not null)")[0][0] == 0
    assert db.q("select count(*) from diagnosis_findings where diagnosis_id in (select diagnosis_id from diagnoses where merged_into is not null)")[0][0] == 0
    assert db.q("select count(*) from diagnosis_merges")[0][0] == db.q("select count(*) from diagnoses where merged_into is not null")[0][0]
    code = dict(db.q("select diagnosis_id, icd10_code from diagnoses"))
    for i in db.instances("patient_diagnosis", SPLIT)[:300]:
        for e in i["gt"]["active_diagnoses"] + i["gt"]["chronic_conditions"]:
            assert e["icd10"] == code[e["diagnosis_id"]], (i["gt_id"], e)
    for i in db.instances("evidence_retrieval", SPLIT)[:300]:
        assert i["gt"]["query_diagnoses"][0]["diagnosis_id"] in code          # never a merged-away node


def test_release_info_records_every_repair(db):
    keys = {k for (k,) in db.q("select key from release_info")}
    assert {"history_repair", "patient_diagnosis_instances", "icd10_repair", "retrieval_grading"} <= keys
