"""Scorers of the Stage-7 task families (audit/STAGE7_TODO.md). Dispatched from eval.scoring.compute_all_metrics.

Per-item scores; a batch metric is the mean of the items, so a single episode's reward equals the corpus
metric on that instance (the Stage-3 convention). Every scorer maps an empty or malformed submission to 0.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from eval.stage7 import ERROR_TYPES, match_name, match_names, order_matches, ordered_names, orderable_from_gt

DIFFERENTIAL_K = 5
DISTRACTOR_GAIN = 0.5
NO_EVIDENCE_COVERAGE = 0.25
TRIAGE_W_RELEVANT, TRIAGE_W_URGENT = 0.6, 0.4

PRIMARY = {
    "differential_diagnosis": "differential_ndcg_5",
    "test_selection": "workup_score",
    "error_detection": "error_detection_score",
    "lab_triage": "triage_score",
    "atypical_diagnosis": "weighted_problem_list_f1_neutral",
}


def _icd_credit(pred: str, gt: str, pred_name: str = "", gt_name: str = "", gt_id=None) -> float:
    """ICD credit, or the concept name credit when higher (eval.scoring.dx_credit: Stage-8 R2, RL readiness A1)."""
    from eval.scoring import dx_credit
    return dx_credit(pred or "", gt or "", pred_name or "", gt_name or "", gt_id)


# ---------------------------------------------------------------------------
# differential diagnosis
# ---------------------------------------------------------------------------

def _dx_list(pred: Any) -> list[dict]:
    if not isinstance(pred, dict):
        return []
    items = pred.get("differential")
    if not isinstance(items, list):
        return []
    return [d for d in items if isinstance(d, dict) and (d.get("icd10") or d.get("name"))][:DIFFERENTIAL_K]


def score_differential_item(pred: dict, gt: dict) -> dict:
    """nDCG@5 with gain 1.0 for the correct diagnosis and 0.5 per distractor, each with graded ICD credit,
    matched one-to-one greedily; the ideal ordering is correct first, then the distractors."""
    correct = [d for d in gt.get("correct", []) if isinstance(d, dict)]
    distr = [d for d in gt.get("distractors", []) if isinstance(d, dict)]
    targets = [(d.get("icd10") or "", 1.0, d.get("display_name") or d.get("name") or "", d.get("diagnosis_id")) for d in correct] + \
              [(d.get("icd10") or "", DISTRACTOR_GAIN, d.get("display_name") or d.get("name") or "", d.get("diagnosis_id")) for d in distr]
    preds = _dx_list(pred)
    used: set[int] = set()
    gains: list[float] = []
    top1 = 0.0
    for i, p in enumerate(preds):
        best, best_j = 0.0, None
        for j, (code, gain, tname, tid) in enumerate(targets):
            if j in used or not code:
                continue
            c = _icd_credit(p.get("icd10") or "", code, str(p.get("name") or ""), tname, tid) * gain
            if c > best:
                best, best_j = c, j
        if best_j is not None:
            used.add(best_j)
            if i == 0 and best_j < len(correct):
                top1 = best
        gains.append(best)
    dcg = sum(g / math.log2(i + 2) for i, g in enumerate(gains))
    ideal = sorted((t[1] for t in targets), reverse=True)[:DIFFERENTIAL_K]
    idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal))
    from eval.scoring import diagnosis_named_coded
    named, coded = diagnosis_named_coded([str(p.get("icd10") or "") for p in preds], [str(p.get("name") or "") for p in preds],
                                         [d.get("icd10") or "" for d in correct],
                                         [d.get("display_name") or d.get("name") or "" for d in correct],
                                         [d.get("diagnosis_id") for d in correct])
    return {"differential_ndcg_5": (dcg / idcg) if idcg else 0.0, "differential_top1": top1,
            "diagnosis_named": named, "diagnosis_coded": coded,
            "differential_distractor_recall": (sum(1 for j in used if j >= len(correct)) / len(distr)) if distr else 0.0,
            "n_predicted": len(preds)}


# ---------------------------------------------------------------------------
# test selection (agentic workup)
# ---------------------------------------------------------------------------

def score_test_selection_item(pred: dict, gt: dict) -> dict:
    """workup_score = icd_credit(diagnosis) x coverage x parsimony.

    coverage  = 1 when at least one discriminating test was ordered, else NO_EVIDENCE_COVERAGE;
    parsimony = min(1, n_needed / n_orders) (n_needed = number of discriminating tests, at least 1);
    orders come from the environment's trace (`tests_ordered`, each with what it matched) or, for a
    single-turn submission, from the submission's own `tests_ordered` names matched against the
    encounter's orderable findings."""
    if not isinstance(pred, dict):
        pred = {}
    dx = pred.get("diagnosis") if isinstance(pred.get("diagnosis"), dict) else pred
    gdx = gt.get("diagnosis") or {}
    credit = _icd_credit(str(dx.get("icd10") or ""), str(gdx.get("icd10") or ""), str(dx.get("name") or ""), str(gdx.get("name") or ""),
                         gdx.get("diagnosis_id"))
    from eval.scoring import diagnosis_named_coded
    named, coded = diagnosis_named_coded([str(dx.get("icd10") or "")], [str(dx.get("name") or "")], [str(gdx.get("icd10") or "")],
                                         [str(gdx.get("name") or "")], [gdx.get("diagnosis_id")])
    disc = [d for d in gt.get("discriminating", [])]
    orders = pred.get("tests_ordered") or []
    if orders and isinstance(orders[0], str):                       # single-turn: names only, matched like order_test
        orderable = orderable_from_gt(gt)
        matched = {n for o in orders for n in order_matches(str(o), orderable)}
        n_orders = len(orders)
    else:
        matched = ordered_names([o for o in orders if isinstance(o, dict)])
        n_orders = len(orders)
    hit = bool(matched & set(disc))
    coverage = 1.0 if hit else NO_EVIDENCE_COVERAGE
    n_needed = max(1, len(disc))
    parsimony = min(1.0, n_needed / n_orders) if n_orders else 1.0
    return {"workup_score": credit * coverage * parsimony, "workup_icd_credit": credit, "workup_coverage": coverage,
            "workup_parsimony": parsimony, "n_orders": n_orders, "discriminating_ordered": int(hit),
            "diagnosis_named": named, "diagnosis_coded": coded}


# ---------------------------------------------------------------------------
# error detection
# ---------------------------------------------------------------------------

_SECTION_ALIASES = {"exam": "physical_exam", "physical exam": "physical_exam", "physical examination": "physical_exam",
                    "history": "hpi", "history of present illness": "hpi", "lab": "labs", "laboratory": "labs",
                    "vital signs": "vitals", "radiology": "imaging"}
_TYPE_ALIASES = {"value": "implausible_value", "implausible": "implausible_value", "lab value": "implausible_value",
                 "numeric": "implausible_value", "side": "laterality", "left right": "laterality", "left/right": "laterality",
                 "age": "age_contradiction", "sex": "sex_contradiction", "gender": "sex_contradiction", "pronoun": "sex_contradiction"}


def _canon(value: Any, aliases: dict[str, str], valid: tuple[str, ...] | None = None) -> str:
    v = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    plain = str(value or "").strip().lower()
    if plain in aliases:
        v = aliases[plain]
    if valid and v not in valid:
        for k, canon in aliases.items():
            if k in plain:
                return canon
    return v


def score_error_detection_item(pred: dict, gt: dict) -> dict:
    if not isinstance(pred, dict):
        pred = {}
    sec_ok = _canon(pred.get("section_type"), _SECTION_ALIASES) == str(gt.get("section_type") or "")
    typ_ok = _canon(pred.get("error_type"), _TYPE_ALIASES, ERROR_TYPES) == str(gt.get("error_type") or "")
    return {"error_detection_score": 0.5 * sec_ok + 0.5 * typ_ok, "error_section_hit": float(sec_ok), "error_type_hit": float(typ_ok)}


# ---------------------------------------------------------------------------
# lab triage
# ---------------------------------------------------------------------------

def score_lab_triage_item(pred: dict, gt: dict) -> dict:
    if not isinstance(pred, dict):
        pred = {}
    relevant = list(gt.get("relevant", []))
    background = list(gt.get("background", []))
    cand = relevant + background
    names = [str(x) for x in (pred.get("relevant") or []) if isinstance(x, (str, int, float))] if isinstance(pred.get("relevant"), list) else []
    m = match_names(names, cand)
    matched = {v for v in m.values() if v}
    tp = len(matched & set(relevant))
    precision = tp / len(names) if names else 0.0
    recall = tp / len(relevant) if relevant else (1.0 if not names else 0.0)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    urgent_gt = gt.get("most_urgent")
    urgent_hit = 0.0
    if urgent_gt and pred.get("most_urgent"):
        urgent_hit = 1.0 if match_name(str(pred["most_urgent"]), cand) == urgent_gt else 0.0
    score = TRIAGE_W_RELEVANT * f1 + TRIAGE_W_URGENT * urgent_hit if urgent_gt else f1
    return {"triage_score": score, "triage_relevant_f1": f1, "triage_relevant_precision": precision,
            "triage_relevant_recall": recall, "triage_urgent_hit": urgent_hit, "n_flagged": len(names)}


# ---------------------------------------------------------------------------
# batch dispatch
# ---------------------------------------------------------------------------

_ITEM = {"differential_diagnosis": score_differential_item, "test_selection": score_test_selection_item,
         "error_detection": score_error_detection_item, "lab_triage": score_lab_triage_item}


def compute_metrics(task: str, predictions: list[dict], ground_truths: list[dict]) -> dict:
    if task == "atypical_diagnosis":
        from eval.scoring import _compute_patient_diagnosis_metrics
        return _compute_patient_diagnosis_metrics(predictions, ground_truths)
    fn = _ITEM[task]
    items = [fn(p, g) for p, g in zip(predictions, ground_truths)]
    if not items:
        return {}
    out: dict[str, Any] = {}
    for k in items[0]:
        vals = [it[k] for it in items if it.get(k) is not None]
        out[k] = float(np.mean(vals)) if vals else 0.0
    out["n_scored"] = len(items)
    return out
