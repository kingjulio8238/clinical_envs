"""RL readiness stage B (task set): lab_triage redesign (B1), error_detection turn cap (B2), privileged floor
policies (B3). test_selection's re-measurement (B4) is a result, recorded in audit/RL_READINESS_TODO.md. No network."""

from __future__ import annotations

import json

import pytest

from eval import degenerate as D
from eval import floors as F
from eval import protocol_run as R
from eval import stage7 as S7
from eval.scoring_tasks7 import score_lab_triage_item
from eval.tests.test_protocol import ScriptedAdapter

pytestmark = pytest.mark.reward_hacking


@pytest.fixture(scope="module")
def db():
    if not D.DEFAULT_DB.exists():
        pytest.skip("release DB not present")
    return R.shared_db()


def _gt(db, gt_id=None):
    insts = db.instances("lab_triage", "public")
    return next(i for i in insts if gt_id is None or i["gt_id"] == gt_id)["gt"]


# ---------------------------------------------------------------------------
# B1 lab_triage
# ---------------------------------------------------------------------------

def test_triage_rows_are_built_for_every_instance(db):
    for split in ("public", "heldout", "train"):
        insts = db.instances("lab_triage", split)
        assert insts and all(S7.triage_rows(i["gt"]) for i in insts), split
        rows = [r for i in insts for r in S7.triage_rows(i["gt"])]
        assert sum(bool(r["context"]) for r in rows) / len(rows) > 0.9      # documented text found for >90% of results


def test_analyte_names_reach_interpretation_labels(db):
    """A clinician flags "Platelet count" and "Blood pressure"; the labels say "Thrombocytopenia" and "Severe
    hypertension". Both must count, and naming a background result must cost specificity."""
    gt = next(i["gt"] for i in db.instances("lab_triage", "public")
              if {"Thrombocytopenia", "Severe hypertension"} <= {r["name"] for r in S7.triage_rows(i["gt"])})
    rows = S7.triage_rows(gt)
    rel = [r["name"] for r in rows if r["relevant"]]
    good = score_lab_triage_item({"relevant": ["Platelet count", "Blood pressure"] + [n for n in rel if n not in ("Thrombocytopenia", "Severe hypertension")],
                                  "most_urgent": gt["most_urgent"]}, gt)
    assert good["triage_sensitivity"] == pytest.approx(1.0) and good["triage_specificity"] == pytest.approx(1.0)
    bg = [r["name"] for r in rows if not r["relevant"]]
    worse = score_lab_triage_item({"relevant": rel + bg[:1], "most_urgent": gt["most_urgent"]}, gt)
    assert worse["triage_score"] < good["triage_score"]


@pytest.mark.parametrize("split", ["public", "heldout"])
def test_flagging_everything_or_nothing_earns_no_triage_credit(db, split):
    """Youden's J: sensitivity + specificity − 1. Flagging every result (read from the chart, or even the label
    names) scores 0 on the flagging term; the redesigned floor must stay ≤ 0.3 (the old F1 floor was 0.63)."""
    insts = db.instances("lab_triage", split)
    for pol in ("all_results", "all_documented"):
        m = D.score(db, "lab_triage", [D.POLICIES["lab_triage"][pol](db, i) for i in insts], insts)
        assert m["triage_youden"] <= 0.12, (pol, m["triage_youden"])
    floors = F.load()["splits"][split]["lab_triage"]["metrics"]["triage_score"]
    assert floors["floor"] <= 0.3 and floors["ceiling"] >= 0.99, floors


def test_triage_urgent_must_be_the_labelled_result(db):
    gt = _gt(db)
    rows = S7.triage_rows(gt)
    wrong = next(r["name"] for r in rows if r["name"] != gt["most_urgent"])
    right = score_lab_triage_item({"relevant": [gt["most_urgent"]], "most_urgent": gt["most_urgent"]}, gt)
    miss = score_lab_triage_item({"relevant": [gt["most_urgent"]], "most_urgent": wrong}, gt)
    assert right["triage_urgent_hit"] == 1.0 and miss["triage_urgent_hit"] == 0.0


def test_legacy_rows_without_results_keep_the_f1_rule():
    gt = {"relevant": ["A", "B"], "background": ["C"], "most_urgent": "A"}
    assert "triage_relevant_f1" in score_lab_triage_item({"relevant": ["A"], "most_urgent": "A"}, gt)


# ---------------------------------------------------------------------------
# B2 error_detection turn cap
# ---------------------------------------------------------------------------

def test_error_detection_episodes_are_turn_capped(tmp_path, db):
    out = R.run("glm-5.3-flash", "error_detection", n=1, seed=12, workers=1, out_root=tmp_path, adapter=ScriptedAdapter(R._env),
                prices=(1.0, 1.0), quiet=True)
    assert json.loads((out / "manifest.json").read_text())["max_turns"] == R.UNIT_MAX_TURNS["error_detection"] == 16
    out2 = R.run("glm-5.3-flash", "patient_diagnosis", n=1, seed=12, workers=1, out_root=tmp_path, adapter=ScriptedAdapter(R._env),
                 prices=(1.0, 1.0), quiet=True)
    assert json.loads((out2 / "manifest.json").read_text())["max_turns"] == 43


# ---------------------------------------------------------------------------
# B3 privileged policies do not set floors
# ---------------------------------------------------------------------------

def test_privileged_policies_are_gates_not_floors():
    """Echoing the rubric's finding names (summarization) and flagging the label names (triage) read information no
    model is served; they are recorded as `privileged_floor`, and the floor is the best policy a model could run."""
    doc = F.load()["splits"]["public"]
    summ = doc["context_summarization"]["metrics"]["clinical_f1"]
    assert summ["floor_policy"] not in F.PRIVILEGED_POLICIES and summ["privileged_floor_policy"] == "echo_structured_hints"
    assert summ["privileged_floor"] <= 0.5                                   # still CI-gated (test_reward_hacking)
    tri = doc["lab_triage"]["metrics"]["triage_score"]
    assert tri["floor_policy"] not in F.PRIVILEGED_POLICIES


# ---------------------------------------------------------------------------
# B4 open decision (closed): is-a forms of a generic reference
# ---------------------------------------------------------------------------

def _node(db, name):
    return db.conn.execute("select diagnosis_id, icd10_code from diagnoses where display_name=? and merged_into is null", (name,)).fetchone()


@pytest.mark.parametrize("pred,ref,expect", [
    ("Serratia marcescens bacteremia", "Gram-negative rod infection", 0.5),
    ("Glucagonoma (pancreatic alpha-cell neuroendocrine tumor)", "Pancreatic neuroendocrine tumor localization", 0.5),
    ("Disseminated histoplasmosis", "Histoplasmosis", 0.75),          # a qualifier, not another disease
    ("Pneumonia", "Gram-negative rod infection", 0.0),               # not a form of it
    ("Pancreatic cancer", "Pancreatic neuroendocrine tumor localization", 0.0),
])
def test_isa_forms_of_a_generic_reference(db, pred, ref, expect):
    from eval.scoring import dx_credit
    did, code = _node(db, ref)
    assert dx_credit("Z99.9", code, pred, ref, did) == pytest.approx(expect)


def test_isa_table_is_frozen_and_audited():
    """The LLM-curated is-a table is part of the reward lock, and the judge's over-credit audit of every answer it
    newly credits stays ≤ 10%."""
    from pathlib import Path
    from eval import reward_version as RV
    assert "eval/diagnosis_isa_aliases.json" in RV.REWARD_FILES and RV.drift() == []
    audit = Path(__file__).resolve().parents[2] / "results" / "isa_overcredit_audit.json"
    if not audit.exists():
        pytest.skip("results/isa_overcredit_audit.json not present")
    d = json.loads(audit.read_text())
    assert d["judged"] > 0 and d["over_credit_rate"] <= 0.10, d["over_credit_rate"]
