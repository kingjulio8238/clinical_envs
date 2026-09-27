"""Scoring engine: task-specific and cross-task metrics.

Task-specific:
  - Diagnosis: top-k accuracy, hierarchical F1, MRR
  - Summarization: ROUGE-L, BERTScore, clinical F1, hallucination rate, omission rate
  - Retrieval: NDCG@10, MAP@10, Recall@K, Precision@5, MRR, latency-adjusted NDCG
  - Imaging: clinical question F1, summary ROUGE-L, differential coverage, findings recall

Cross-task:
  - Cost-normalized performance
  - Inter-model agreement (Cohen's kappa)
"""

import json
import logging
import math
import re
from collections import defaultdict

import numpy as np

from eval import concept_match, semantic_match, value_match

log = logging.getLogger(__name__)


# Acuity weights for acuity-weighted recall (higher-severity findings count more).
_ACUITY_WEIGHTS = {
    "acute": 1.0,
    "acute_on_chronic": 1.0,
    "chronic": 0.6,
    "unspecified": 0.8,
}


def _finding_names(findings: list) -> list[str]:
    """Extract display names from a findings list (dicts or bare strings).

    Entries marked `excluded_nondiagnostic` are skipped: these are non-clinical
    concepts (study design, statistics, ethics) that leaked in from non-diagnostic
    board questions and are not valid targets for chart-grounded reasoning.
    """
    names: list[str] = []
    for f in findings or []:
        if isinstance(f, str):
            if f:
                names.append(f)
        elif isinstance(f, dict):
            if f.get("excluded_nondiagnostic"):
                continue
            n = f.get("display_name") or f.get("name") or ""
            if n:
                names.append(n)
    return names


def _list_recall(summaries: list[str], findings_lists: list[list]) -> float:
    """Mean per-item recall of a findings list within each summary (semantic).

    Items whose findings list is empty are skipped (no denominator).
    """
    recalls = []
    for summary, findings in zip(summaries, findings_lists):
        names = _finding_names(findings)
        if not names:
            continue
        hits = semantic_match.count_present(names, summary or "")
        recalls.append(hits / len(names))
    return float(np.mean(recalls)) if recalls else 0.0


def _cascade_present(names: list[str], summary: str) -> tuple[int, int]:
    """Matcher cascade for one summary: lexical-AND-not-negated first, then the
    polarity-safe value-normalizer on the residual. Returns (n_matched, n_value_added)."""
    summary = summary or ""
    matched = value_added = 0
    for name in names:
        if semantic_match.phrase_in_text(name, summary) and not value_match.negated_mention(name, summary):
            matched += 1
        elif value_match.value_polarity_match(name, summary):
            matched += 1
            value_added += 1
    return matched, value_added


def _list_recall_cascade(summaries: list[str], findings_lists: list[list]) -> tuple[float, int]:
    """Cascade (lexical+negation+value-normalizer) mean recall + total value-normalizer credits."""
    recalls, added = [], 0
    for summary, findings in zip(summaries, findings_lists):
        names = _finding_names(findings)
        if not names:
            continue
        n, a = _cascade_present(names, summary)
        recalls.append(n / len(names))
        added += a
    return (float(np.mean(recalls)) if recalls else 0.0), added


# ============================================================================
# DIAGNOSIS METRICS
# ============================================================================

def _normalize_icd10(code: str) -> str:
    """Strip dots, uppercase."""
    return code.replace(".", "").replace(" ", "").upper() if code else ""


def top_k_accuracy(predictions: list[dict], ground_truths: list[dict], k: int = 1) -> float:
    """Exact ICD-10 match within top-k predicted diagnoses."""
    correct = 0
    total = 0
    for pred, gt in zip(predictions, ground_truths):
        gt_primary = gt.get("primary_diagnosis", {})
        gt_icd = _normalize_icd10(gt_primary.get("icd10", ""))
        if not gt_icd:
            continue
        total += 1

        # Also accept alternatives
        acceptable = {gt_icd}
        for alt in gt.get("acceptable_alternatives", []):
            alt_icd = _normalize_icd10(alt.get("icd10", ""))
            if alt_icd:
                acceptable.add(alt_icd)

        pred_diagnoses = pred.get("diagnoses", [])
        for dx in pred_diagnoses[:k]:
            pred_icd = _normalize_icd10(dx.get("icd10", ""))
            if pred_icd in acceptable:
                correct += 1
                break

    return correct / total if total > 0 else 0.0


def hierarchical_f1(predictions: list[dict], ground_truths: list[dict]) -> float:
    """ICD-10 chapter-level (first 3 chars) match F1."""
    tp = fp = fn = 0
    for pred, gt in zip(predictions, ground_truths):
        gt_primary = gt.get("primary_diagnosis", {})
        gt_chapter = _normalize_icd10(gt_primary.get("icd10", ""))[:3]
        if not gt_chapter:
            continue

        pred_diagnoses = pred.get("diagnoses", [])
        pred_chapters = set()
        for dx in pred_diagnoses[:5]:
            ch = _normalize_icd10(dx.get("icd10", ""))[:3]
            if ch:
                pred_chapters.add(ch)

        if gt_chapter in pred_chapters:
            tp += 1
        else:
            fn += 1
        # FP: predicted chapters not matching GT
        fp += len(pred_chapters - {gt_chapter})

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def mean_reciprocal_rank(predictions: list[dict], ground_truths: list[dict]) -> float:
    """MRR: 1/rank of first correct ICD-10 in ranked list."""
    rr_sum = 0.0
    total = 0
    for pred, gt in zip(predictions, ground_truths):
        gt_primary = gt.get("primary_diagnosis", {})
        gt_icd = _normalize_icd10(gt_primary.get("icd10", ""))
        if not gt_icd:
            continue
        total += 1

        acceptable = {gt_icd}
        for alt in gt.get("acceptable_alternatives", []):
            alt_icd = _normalize_icd10(alt.get("icd10", ""))
            if alt_icd:
                acceptable.add(alt_icd)

        for rank, dx in enumerate(pred.get("diagnoses", []), start=1):
            pred_icd = _normalize_icd10(dx.get("icd10", ""))
            if pred_icd in acceptable:
                rr_sum += 1.0 / rank
                break

    return rr_sum / total if total > 0 else 0.0


# ============================================================================
# PATIENT DIAGNOSIS METRICS
# ============================================================================

class SeverityTier:
    CRITICAL = 3.0
    MODERATE = 2.0
    ROUTINE = 1.0


CRITICAL_CODES_3CHAR = {
    "A41", "I21", "I26", "I63", "I60", "I61",
    "J96", "N17", "K72", "E87", "T78",
}

CRITICAL_CHAPTERS = {"A", "B", "C", "D", "I", "J", "S", "T"}


def _normalize_icd10_category(code: str) -> str:
    """Normalize and extract 3-char ICD-10 category."""
    norm = _normalize_icd10(code)
    return norm[:3] if len(norm) >= 3 else norm


def _assign_severity_tier(icd10_code: str, acuity: str) -> float:
    """Assign severity tier weight based on ICD-10 category and acuity."""
    norm = _normalize_icd10(icd10_code)
    cat3 = norm[:3] if len(norm) >= 3 else norm
    first_char = norm[0] if norm else ""

    if cat3 in CRITICAL_CODES_3CHAR:
        return SeverityTier.CRITICAL

    is_active = acuity in ("acute", "acute_on_chronic")
    if is_active and first_char in CRITICAL_CHAPTERS:
        return SeverityTier.CRITICAL
    if is_active and first_char not in CRITICAL_CHAPTERS:
        return SeverityTier.MODERATE
    if not is_active and first_char in CRITICAL_CHAPTERS:
        return SeverityTier.MODERATE

    return SeverityTier.ROUTINE


def _match_icd10_sets(
    pred_codes: list[str], gt_codes: list[str]
) -> tuple[list[tuple[str, str]], list[str], list[str]]:
    """Match predicted to GT codes at 3-char category level.

    Returns (matched_pairs, unmatched_pred, unmatched_gt).
    Greedy matching: prefer exact full-code match, then longest prefix.
    """
    pred_norm = [_normalize_icd10(c) for c in pred_codes]
    gt_norm = [_normalize_icd10(c) for c in gt_codes]

    matched: list[tuple[str, str]] = []
    used_pred: set[int] = set()
    used_gt: set[int] = set()

    # Pass 1: exact full-code match
    for gi, gc in enumerate(gt_norm):
        if gi in used_gt:
            continue
        for pi, pc in enumerate(pred_norm):
            if pi in used_pred:
                continue
            if pc == gc:
                matched.append((pc, gc))
                used_pred.add(pi)
                used_gt.add(gi)
                break

    # Pass 2: 3-char category match (longest shared prefix as tiebreaker)
    for gi, gc in enumerate(gt_norm):
        if gi in used_gt:
            continue
        gc_cat = gc[:3]
        # Injury/poisoning: use 5-char dedup
        dedup_len = 5 if gc_cat and gc_cat[0] in ("S", "T") else 3
        gc_key = gc[:dedup_len]

        best_pi = None
        best_prefix_len = -1
        for pi, pc in enumerate(pred_norm):
            if pi in used_pred:
                continue
            pc_key = pc[:dedup_len]
            if pc_key == gc_key:
                prefix_len = len(_shared_prefix(pc, gc))
                if prefix_len > best_prefix_len:
                    best_prefix_len = prefix_len
                    best_pi = pi

        if best_pi is not None:
            matched.append((pred_norm[best_pi], gc))
            used_pred.add(best_pi)
            used_gt.add(gi)

    unmatched_pred = [pred_norm[i] for i in range(len(pred_norm)) if i not in used_pred]
    unmatched_gt = [gt_norm[i] for i in range(len(gt_norm)) if i not in used_gt]

    return matched, unmatched_pred, unmatched_gt


def _shared_prefix(a: str, b: str) -> str:
    """Return the shared prefix of two strings."""
    i = 0
    while i < len(a) and i < len(b) and a[i] == b[i]:
        i += 1
    return a[:i]


def _icd10_specificity_score(matched_pairs: list[tuple[str, str]]) -> float | None:
    """Mean character-level agreement beyond the 3-char prefix."""
    if not matched_pairs:
        return None

    scores = []
    for pred, gt in matched_pairs:
        # If GT ends in 9 at subcategory (unspecified convention), full credit
        if len(gt) >= 4 and gt[3] == "9":
            scores.append(1.0)
            continue
        prefix_len = len(_shared_prefix(pred, gt))
        max_len = max(len(pred), len(gt))
        scores.append(prefix_len / max_len if max_len > 0 else 1.0)

    return float(np.mean(scores))


def problem_list_recall(pred_codes: list[str], gt_codes: list[str]) -> float:
    """Fraction of GT diagnoses found in predicted set at 3-char category level."""
    if not gt_codes:
        return 1.0
    matched, _, _ = _match_icd10_sets(pred_codes, gt_codes)
    return len(matched) / len(gt_codes)


def problem_list_precision(pred_codes: list[str], gt_codes: list[str]) -> float:
    """Fraction of predicted diagnoses matching any GT diagnosis."""
    if not pred_codes:
        return 1.0 if not gt_codes else 0.0
    matched, _, _ = _match_icd10_sets(pred_codes, gt_codes)
    return len(matched) / len(pred_codes)


def problem_list_f1(recall: float, precision: float) -> float:
    """Harmonic mean of recall and precision."""
    if recall + precision == 0:
        return 0.0
    return 2 * recall * precision / (recall + precision)


def weighted_problem_list_recall(
    matched_gt_codes: list[str], matched_gt_acuities: list[str],
    all_gt_codes: list[str], all_gt_acuities: list[str],
) -> float:
    """Severity-weighted recall."""
    if not all_gt_codes:
        return 1.0
    total_weight = sum(
        _assign_severity_tier(c, a) for c, a in zip(all_gt_codes, all_gt_acuities)
    )
    matched_weight = sum(
        _assign_severity_tier(c, a) for c, a in zip(matched_gt_codes, matched_gt_acuities)
    )
    return matched_weight / total_weight if total_weight > 0 else 0.0


ACUITY_NORMALIZATION = {
    "acute": "acute",
    "chronic": "chronic",
    "acute_on_chronic": "acute_on_chronic",
    "subacute": "acute",
    "resolving": "acute",
    "stable": "chronic",
    "exacerbation": "acute_on_chronic",
    "flare": "acute_on_chronic",
    "acute on chronic": "acute_on_chronic",
    "acute-on-chronic": "acute_on_chronic",
}


def _normalize_acuity(raw: str | None) -> str | None:
    """Normalize an acuity label. Returns None if unrecognized."""
    if not raw:
        return None
    return ACUITY_NORMALIZATION.get(raw.lower().strip())


def acuity_accuracy(
    matched_pairs_with_labels: list[tuple[str, str]]
) -> float | None:
    """Among matched diagnoses with valid acuity labels, fraction correct.

    Each tuple is (predicted_acuity, gt_acuity) already normalized.
    """
    eligible = [(p, g) for p, g in matched_pairs_with_labels if p and g]
    if not eligible:
        return None
    correct = sum(1 for p, g in eligible if p == g)
    return correct / len(eligible)


def acuity_confusion_matrix(
    matched_pairs_with_labels: list[tuple[str, str]]
) -> dict[str, int]:
    """3x3 confusion matrix. Keys are 'gt_label:pred_label'."""
    matrix: dict[str, int] = defaultdict(int)
    for pred_acuity, gt_acuity in matched_pairs_with_labels:
        if pred_acuity and gt_acuity:
            matrix[f"{gt_acuity}:{pred_acuity}"] += 1
    return dict(matrix)


ICD_CREDIT = {"exact": 1.0, "subcategory": 0.75, "category": 0.5}
"""Graded credit for a predicted ICD-10-CM code against a reference code (Stage 3). A reference that
is itself unspecified (3 characters, or a '9' in the 4th position) is fully credited by any match in
its category: the record supports nothing more specific."""
ACUITY_MISMATCH_FACTOR = 0.5
"""A matched diagnosis whose predicted acuity is missing or differs from a known reference acuity
keeps half its credit. The task's output schema carries acuity; the old metric never read it."""


def _icd_levels(code: str) -> tuple[int, int]:
    """(category length, subcategory length): injury/poisoning codes (S, T) key on 5 characters."""
    cat = 5 if code[:1] in ("S", "T") else 3
    return cat, cat + 1


def _icd_credit(pred: str, gt: str) -> float:
    pred, gt = _normalize_icd10(pred), _normalize_icd10(gt)
    if not pred or not gt:
        return 0.0
    if pred == gt:
        return ICD_CREDIT["exact"]
    cat, sub = _icd_levels(gt)
    if len(pred) < cat or pred[:cat] != gt[:cat]:
        return 0.0
    unspecified = len(gt) <= cat or gt[cat:cat + 1] == "9"
    if unspecified:
        return ICD_CREDIT["exact"]
    if len(pred) >= sub and len(gt) >= sub and pred[:sub] == gt[:sub]:
        return ICD_CREDIT["subcategory"]
    return ICD_CREDIT["category"]


NAME_CREDIT = {"equivalent": 0.75, "related": 0.5}
"""Diagnosis credit by name when the codes disagree (Stage-8 audit R2): the model named the reference diagnosis
but coded it differently (I20.9 for "stable angina due to CAD", H90.5 for presbycusis) or the reference code is
off (P01.1, a newborn code, on a maternal PPROM label). `equivalent`: the same name after normalization, any code;
`related`: one name contains the other or they overlap by Jaccard >= 0.5, within the same ICD block (first two
characters) and with no contradicting qualifier (left/right, acute/chronic, type 1/2, non-, with/without ...).
Always below an exact code, so coding stays rewarded."""
_DX_STOP = {"of", "the", "a", "an", "and", "in", "on", "to", "due", "for", "by", "or", "at", "as", "unspecified", "nos",
            "disease", "disorder", "syndrome", "condition", "other", "specified"}
_DX_SYN = {"mi": "myocardial infarction", "stemi": "st elevation myocardial infarction", "nstemi": "non st elevation myocardial infarction",
           "chf": "heart failure", "hf": "heart failure", "copd": "chronic obstructive pulmonary", "htn": "hypertension",
           "dm": "diabetes mellitus", "t1dm": "type 1 diabetes mellitus", "t2dm": "type 2 diabetes mellitus", "ckd": "chronic kidney",
           "aki": "acute kidney injury", "uti": "urinary tract infection", "dvt": "deep vein thrombosis", "pe": "pulmonary embolism",
           "cad": "coronary artery", "gerd": "gastroesophageal reflux", "afib": "atrial fibrillation", "af": "atrial fibrillation",
           "pprom": "preterm premature rupture membranes", "prom": "premature rupture membranes", "tia": "transient ischemic attack",
           "ards": "acute respiratory distress", "dka": "diabetic ketoacidosis", "sle": "systemic lupus erythematosus",
           "ibs": "irritable bowel", "bph": "benign prostatic hyperplasia", "hiv": "human immunodeficiency virus", "tb": "tuberculosis",
           "ii": "2", "iii": "3", "iv": "4"}      # not "i": "I-cell disease" is no type 1
_DX_POLAR = ({"left", "right"}, {"acute", "chronic"}, {"benign", "malignant"}, {"primary", "secondary"}, {"upper", "lower"},
             {"anterior", "posterior"}, {"inferior", "superior"}, {"with", "without"}, {"congenital", "acquired"},
             {"early", "late"}, {"unilateral", "bilateral"}, {"proximal", "distal"}, {"central", "peripheral"})


def _dx_tokens(name: str, keep_parentheticals: bool = False) -> frozenset[str]:
    t = (name or "").lower()
    if not keep_parentheticals:
        t = re.sub(r"\([^)]*\)", " ", t)                            # parentheticals are glosses ("(PPROM)")
    t = t.replace("-", " ").replace("/", " ").replace(",", " ")
    out = set()
    for w in re.findall(r"[a-z0-9]+", t):
        for part in _DX_SYN.get(w, w).split():
            if part in _DX_STOP or (len(part) == 1 and not part.isdigit()):      # "tourette's" -> tourette
                continue
            if len(part) > 4 and part.endswith("s") and not part.endswith("ss") and not part.endswith("is"):
                part = part[:-1]
            out.add(part)
    return frozenset(out)


def _dx_conflict(a: frozenset[str], b: frozenset[str]) -> bool:
    if ("non" in a) != ("non" in b):
        return True
    for group in _DX_POLAR:
        ga, gb = a & group, b & group
        if ga and gb and ga != gb:
            return True
    da, db = {x for x in a if x.isdigit()}, {x for x in b if x.isdigit()}
    return bool(da and db and not (da <= db or db <= da))           # type 1 vs type 2; "2" is within "2, 4"


def name_credit(pred_name: str, gt_name: str, pred_code: str = "", gt_code: str = "") -> float:
    pa, ga = _dx_tokens(pred_name), _dx_tokens(gt_name)
    if not pa or not ga or _dx_conflict(pa, ga):
        return 0.0
    if pa == ga:
        return NAME_CREDIT["equivalent"]
    # the prediction names the reference in full inside a more specific name ("Sepsis due to pneumonia with septic
    # shock" for "Septic shock", "Acute GVHD following allogeneic HSCT" for "Acute graft-versus-host disease"):
    # related in any ICD block, when the reference has at least two content words (a one-word reference such as
    # "hypertension" is contained in too many other diseases)
    pf = _dx_tokens(pred_name, keep_parentheticals=True)
    if len(ga) >= 2 and ga <= pf and not _dx_conflict(pf, ga):
        return NAME_CREDIT["related"]
    pc, gc = _normalize_icd10(pred_code), _normalize_icd10(gt_code)
    if not pc or not gc or pc[:2] != gc[:2]:
        return 0.0
    if pa <= ga or ga <= pa or len(pa & ga) / len(pa | ga) >= 0.5:
        return NAME_CREDIT["related"]
    return 0.0


def dx_credit(pred_code: str, gt_code: str, pred_name: str = "", gt_name: str = "") -> float:
    """ICD credit, or the name credit when that is higher (never above the exact-code credit)."""
    c = _icd_credit(pred_code, gt_code)
    if c >= ICD_CREDIT["exact"] or not (pred_name and gt_name):
        return c
    return max(c, name_credit(pred_name, gt_name, pred_code, gt_code))


def _match_graded(pred_codes: list[str], gt_codes: list[str], pred_names: list[str] | None = None,
                  gt_names: list[str] | None = None) -> list[tuple[int, int, float]]:
    """Greedy one-to-one matching by descending credit. Returns (pred_index, gt_index, credit)."""
    pn = pred_names or [""] * len(pred_codes)
    gn = gt_names or [""] * len(gt_codes)
    cred = {(gi, pi): dx_credit(p, g, pn[pi], gn[gi]) for gi, g in enumerate(gt_codes) for pi, p in enumerate(pred_codes)}
    pairs = sorted(((c, -gi, -pi) for (gi, pi), c in cred.items() if c > 0), reverse=True)
    used_p, used_g, out = set(), set(), []
    for credit, ngi, npi in pairs:
        gi, pi = -ngi, -npi
        if gi in used_g or pi in used_p:
            continue
        used_g.add(gi); used_p.add(pi)
        out.append((pi, gi, credit))
    return out


def _dx_entries(pred: dict, with_names: bool = False):
    """Predicted (codes, normalized acuities[, names]). Chronic-list entries are chronic; missing or
    unrecognized acuity is None (scored as a mismatch)."""
    active = pred.get("active_diagnoses") or []
    chronic = pred.get("chronic_conditions") or []
    if not active and not chronic:
        active = pred.get("diagnoses") or []          # flat fallback
    codes, acuities, names = [], [], []
    for dx in active:
        if isinstance(dx, dict):
            codes.append(dx.get("icd10", "") or ""); acuities.append(_normalize_acuity(dx.get("acuity")))
            names.append(str(dx.get("name") or dx.get("display_name") or ""))
    for dx in chronic:
        if isinstance(dx, dict):
            codes.append(dx.get("icd10", "") or ""); acuities.append("chronic")
            names.append(str(dx.get("name") or dx.get("display_name") or ""))
    return (codes, acuities, names) if with_names else (codes, acuities)


def score_patient_diagnosis_item(pred: dict, gt: dict) -> dict:
    """Per-instance patient-diagnosis metrics (the RL reward is `weighted_problem_list_f1_neutral`).

    credit(match) = ICD credit (1 / 0.75 / 0.5) x acuity factor (1, or 0.5 when the reference acuity is
    known and the prediction's is missing or different). Weighted recall weights each reference entry by
    its severity tier; precision is the mean credit over predictions, with unmatched predictions whose
    3-character category lies in the patient's chart-neutral set removed from the denominator.
    An empty or malformed prediction scores 0 (it used to be skipped).
    """
    gt_active = [d for d in gt.get("active_diagnoses", []) if isinstance(d, dict) and not d.get("excluded_nondiagnostic")]
    gt_chronic = [d for d in gt.get("chronic_conditions", []) if isinstance(d, dict) and not d.get("excluded_nondiagnostic")]
    gt_codes = [d.get("icd10", "") or "" for d in gt_active] + [d.get("icd10", "") or "" for d in gt_chronic]
    gt_acuity = [d.get("acuity") or "unspecified" for d in gt_active] + ["chronic" for _ in gt_chronic]
    gt_acuity = [a if a in ("acute", "chronic", "acute_on_chronic") else None for a in gt_acuity]
    weights = [_assign_severity_tier(c, a or "acute") for c, a in zip(gt_codes, gt_acuity)]
    gt_names = [str(d.get("display_name") or d.get("name") or "") for d in gt_active + gt_chronic]

    pred_codes, pred_acuity, pred_names = _dx_entries(pred if isinstance(pred, dict) else {}, with_names=True)
    neutral = {c[:3] for c in (gt.get("_neutral_categories") or [])}
    matched = _match_graded(pred_codes, gt_codes, pred_names, gt_names)

    credit_by_gt: dict[int, float] = {}
    icd_by_gt: dict[int, float] = {}
    acuity_pairs: list[tuple[str, str]] = []
    for pi, gi, icd_credit in matched:
        factor = 1.0
        if gt_acuity[gi] is not None:
            if pred_acuity[pi] != gt_acuity[gi]:
                factor = ACUITY_MISMATCH_FACTOR
            acuity_pairs.append((pred_acuity[pi] or "", gt_acuity[gi]))
        credit_by_gt[gi] = icd_credit * factor
        icd_by_gt[gi] = icd_credit
    matched_pred = {pi for pi, _, _ in matched}
    unmatched_pred = [pi for pi in range(len(pred_codes)) if pi not in matched_pred]
    neutral_unmatched = [pi for pi in unmatched_pred if _normalize_icd10(pred_codes[pi])[:3] in neutral]

    total_credit = sum(credit_by_gt.values())
    n_gt, n_pred = len(gt_codes), len(pred_codes)
    recall = total_credit / n_gt if n_gt else (1.0 if not n_pred else 0.0)
    w_recall = (sum(credit_by_gt.get(gi, 0.0) * w for gi, w in enumerate(weights)) / sum(weights)) if n_gt and sum(weights) else recall
    precision = total_credit / n_pred if n_pred else 0.0
    denom_neutral = n_pred - len(neutral_unmatched)
    precision_neutral = total_credit / denom_neutral if denom_neutral else (1.0 if n_pred and total_credit else 0.0)
    if not n_pred:
        precision = precision_neutral = 0.0
    eligible = [(p, g) for p, g in acuity_pairs if p]
    return {
        "problem_list_recall": recall,
        "problem_list_precision": precision,
        "problem_list_f1": problem_list_f1(recall, precision),
        "weighted_problem_list_recall": w_recall,
        "weighted_problem_list_f1": problem_list_f1(w_recall, precision),
        "problem_list_precision_neutral": precision_neutral,
        "problem_list_f1_neutral": problem_list_f1(recall, precision_neutral),
        "weighted_problem_list_f1_neutral": problem_list_f1(w_recall, precision_neutral),
        "n_neutral_predictions": len(neutral_unmatched),
        "icd10_specificity_score": (sum(icd_by_gt.values()) / len(icd_by_gt)) if icd_by_gt else None,
        "acuity_accuracy": (sum(1 for p, g in acuity_pairs if p == g) / len(acuity_pairs)) if acuity_pairs else None,
        "acuity_pairs": acuity_pairs,
        "empty_prediction": n_pred == 0,
    }


def _compute_patient_diagnosis_metrics(predictions: list[dict], ground_truths: list[dict]) -> dict:
    """Batch = mean of the per-item metrics, so a single-item reward equals the aggregate."""
    items = [score_patient_diagnosis_item(p, g) for p, g in zip(predictions, ground_truths)]
    keys = ("problem_list_recall", "problem_list_precision", "problem_list_f1", "weighted_problem_list_recall",
            "weighted_problem_list_f1", "problem_list_precision_neutral", "problem_list_f1_neutral",
            "weighted_problem_list_f1_neutral")
    out = {k: float(np.mean([it[k] for it in items])) if items else 0.0 for k in keys}
    spec = [it["icd10_specificity_score"] for it in items if it["icd10_specificity_score"] is not None]
    pairs = [pr for it in items for pr in it["acuity_pairs"]]
    out.update({
        "n_neutral_predictions": sum(it["n_neutral_predictions"] for it in items),
        "icd10_specificity_score": float(np.mean(spec)) if spec else 0.0,
        "acuity_accuracy": acuity_accuracy(pairs),
        "acuity_confusion_matrix": acuity_confusion_matrix(pairs),
        "n_scored": len(items),
        "n_tier_c": sum(1 for it in items if it["empty_prediction"]),   # empty predictions, scored 0
    })
    return out


# ============================================================================
# SUMMARIZATION METRICS
# ============================================================================

def rouge_l(predictions: list[str], references: list[str]) -> float:
    """ROUGE-L F1 score averaged across items."""
    try:
        from rouge_score import rouge_scorer
    except ImportError:
        log.warning("rouge-score not installed; skipping ROUGE-L")
        return 0.0

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = []
    for pred, ref in zip(predictions, references):
        if not pred or not ref:
            scores.append(0.0)
            continue
        result = scorer.score(ref, pred)
        scores.append(result["rougeL"].fmeasure)
    return float(np.mean(scores)) if scores else 0.0


def clinical_f1(predictions: list[str], must_include_findings: list[list[dict]],
                chart_texts: list[str] | None = None, concept_extractor=None) -> float:
    """Mean per-item whole-patient summary score (see score_summary_item)."""
    if not predictions:
        return 0.0
    charts = chart_texts or [None] * len(predictions)
    return float(np.mean([score_summary_item(p, m, c, concept_extractor)["clinical_f1"]
                          for p, m, c in zip(predictions, must_include_findings, charts)]))


def finding_recall(summary: str, findings: list, concept_extractor=None) -> float:
    """Share of must-include findings the summary states as present (lexical-not-negated, value
    polarity, or concept match). 0 when there are no findings."""
    names = _finding_names(findings)
    if not names:
        return 0.0
    return concept_match.count_present(names, summary or "", concept_extractor) / len(names)


def score_summary_item(summary: str, must_include: list, chart_text: str | None = None,
                       concept_extractor=None, patient_terms: str | None = None) -> dict:
    """Whole-patient summary reward (`clinical_f1`, Stage 3):

        recall    = must-include findings stated as present (negation-aware, concept-level)
        precision = share of the summary's clinical concepts the record supports (chart text or the
                    patient's annotated finding/diagnosis names)
        length    = 1 up to SUMMARY_WORD_BUDGET words, then budget / words
        clinical_f1 = HM(recall, precision) x length   (recall x length when no chart text or the
                      summary carries no clinical concept)

    The old metric was recall only: pasting the chart scored 0.67 and "denies fever" credited "Fever".
    """
    summary = summary or ""
    recall = finding_recall(summary, must_include, concept_extractor)
    precision = None
    if chart_text and concept_extractor is not None:
        precision = concept_match.grounded_precision(summary, chart_text, concept_extractor, patient_terms)
    length = concept_match.length_factor(summary)
    core = concept_match.harmonic(recall, precision) if precision is not None else recall
    return {"clinical_f1": core * length, "finding_recall": recall, "grounded_precision": precision,
            "length_factor": length, "omission_rate": 1.0 - recall}


def omission_rate(predictions: list[str], must_include_findings: list[list[dict]]) -> float:
    """Fraction of must-include findings NOT stated as present (1 - finding recall)."""
    if not predictions:
        return 1.0
    return 1.0 - float(np.mean([finding_recall(p, m) for p, m in zip(predictions, must_include_findings)]))


def hallucination_rate(predictions: list[str], ehr_texts: list[str],
                       jaccard_threshold: float = 0.15, concept_extractor=None,
                       patient_terms: list[str | None] | None = None) -> float:
    """Fraction of summary sentences not grounded in the source chart.

    With a concept extractor (the scorer always passes one): a sentence is hallucinated when it
    carries clinical concepts none of which the chart states. Without one, the legacy token-overlap
    test is used; it cannot detect fabrication (audit #14) and is kept only for backward comparison.
    """
    if not predictions:
        return 0.0
    rates = []
    terms = patient_terms or [None] * len(predictions)
    for pred, ehr, pt in zip(predictions, ehr_texts, terms):
        if not pred:
            rates.append(0.0)
            continue
        if concept_extractor is not None:
            r = concept_match.hallucination_rate_one(pred, ehr or "", concept_extractor, pt)
            rates.append(0.0 if r is None else r)
            continue
        sentences = _split_sentences(pred)
        if not sentences:
            rates.append(0.0)
            continue
        ehr_tokens = set((ehr or "").lower().split())
        hallucinated = 0
        for sent in sentences:
            sent_tokens = set(sent.lower().split())
            if not sent_tokens:
                continue
            overlap = len(sent_tokens & ehr_tokens) / len(sent_tokens)
            if overlap < jaccard_threshold:
                hallucinated += 1
        rates.append(hallucinated / len(sentences))
    return float(np.mean(rates)) if rates else 0.0


def _split_sentences(text: str) -> list[str]:
    """Simple sentence splitter."""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s for s in sentences if len(s) > 10]


# ============================================================================
# RETRIEVAL METRICS
# ============================================================================

def ndcg_at_k(ranked_results: list[list[dict]], judgments: list[dict[str, int]],
              k: int = 10) -> float:
    """NDCG@k with graded relevance (0-3)."""
    scores = []
    for results, judg in zip(ranked_results, judgments):
        dcg = 0.0
        for i, item in enumerate(results[:k]):
            pid = item.get("passage_id", "")
            grade = judg.get(pid, 0)
            dcg += (2**grade - 1) / math.log2(i + 2)

        # Ideal DCG: sort all judged passages by grade
        ideal_grades = sorted(judg.values(), reverse=True)[:k]
        idcg = sum((2**g - 1) / math.log2(i + 2) for i, g in enumerate(ideal_grades))

        scores.append(dcg / idcg if idcg > 0 else 0.0)
    return float(np.mean(scores)) if scores else 0.0


def map_at_k(ranked_results: list[list[dict]], judgments: list[dict[str, int]],
             k: int = 10, threshold: int = 2) -> float:
    """Mean Average Precision@k with binary relevance (grade >= threshold)."""
    aps = []
    for results, judg in zip(ranked_results, judgments):
        relevant_count = 0
        precision_sum = 0.0
        for i, item in enumerate(results[:k]):
            pid = item.get("passage_id", "")
            if judg.get(pid, 0) >= threshold:
                relevant_count += 1
                precision_sum += relevant_count / (i + 1)

        total_relevant = sum(1 for g in judg.values() if g >= threshold)
        ap = precision_sum / min(total_relevant, k) if total_relevant > 0 else 0.0
        aps.append(ap)
    return float(np.mean(aps)) if aps else 0.0


def recall_at_k(ranked_results: list[list[dict]], judgments: list[dict[str, int]],
                k: int = 10, threshold: int = 2) -> float:
    """Recall@k with binary relevance."""
    recalls = []
    for results, judg in zip(ranked_results, judgments):
        total_relevant = sum(1 for g in judg.values() if g >= threshold)
        if total_relevant == 0:
            recalls.append(0.0)
            continue
        retrieved_relevant = 0
        for item in results[:k]:
            pid = item.get("passage_id", "")
            if judg.get(pid, 0) >= threshold:
                retrieved_relevant += 1
        recalls.append(retrieved_relevant / total_relevant)
    return float(np.mean(recalls)) if recalls else 0.0


def precision_at_k(ranked_results: list[list[dict]], judgments: list[dict[str, int]],
                   k: int = 5, threshold: int = 2) -> float:
    """Precision@k with binary relevance."""
    precisions = []
    for results, judg in zip(ranked_results, judgments):
        relevant = 0
        for item in results[:k]:
            pid = item.get("passage_id", "")
            if judg.get(pid, 0) >= threshold:
                relevant += 1
        precisions.append(relevant / k)   # fixed-k denominator: a short list fills fewer slots (Stage 3)
    return float(np.mean(precisions)) if precisions else 0.0


def retrieval_mrr(ranked_results: list[list[dict]], judgments: list[dict[str, int]],
                  threshold: int = 2) -> float:
    """MRR: 1/rank of first relevant passage (grade >= threshold)."""
    rr_sum = 0.0
    total = len(ranked_results)
    for results, judg in zip(ranked_results, judgments):
        for rank, item in enumerate(results, start=1):
            pid = item.get("passage_id", "")
            if judg.get(pid, 0) >= threshold:
                rr_sum += 1.0 / rank
                break
    return rr_sum / total if total > 0 else 0.0


def latency_adjusted_ndcg(ranked_results: list[list[dict]], judgments: list[dict[str, int]],
                          latencies_ms: list[int], k: int = 10) -> float:
    """NDCG@10 / log2(1 + latency_seconds). Penalizes slow retrieval."""
    base_ndcg = ndcg_at_k(ranked_results, judgments, k)
    if not latencies_ms:
        return base_ndcg
    avg_latency_s = np.mean(latencies_ms) / 1000.0
    penalty = math.log2(1 + avg_latency_s)
    return base_ndcg / penalty if penalty > 0 else base_ndcg


# ============================================================================
# IMAGING METRICS
# ============================================================================

def clinical_question_f1(predicted_questions: list[str], reference_questions: list[str]) -> float:
    """Token-level F1 between predicted and reference clinical questions."""
    f1_scores = []
    for pred, ref in zip(predicted_questions, reference_questions):
        pred_tokens = set(pred.lower().split())
        ref_tokens = set(ref.lower().split())
        if not pred_tokens or not ref_tokens:
            f1_scores.append(0.0)
            continue
        tp = len(pred_tokens & ref_tokens)
        precision = tp / len(pred_tokens)
        recall = tp / len(ref_tokens)
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        f1_scores.append(f1)
    return float(np.mean(f1_scores)) if f1_scores else 0.0


def differential_coverage(predicted_diffs: list[list[dict]],
                          reference_diffs: list[list[dict]]) -> float:
    """Fraction of reference differential diagnoses mentioned in prediction."""
    coverages = []
    for pred_list, ref_list in zip(predicted_diffs, reference_diffs):
        if not ref_list:
            continue
        # Build set of reference ICD-10s and names
        ref_codes = set()
        ref_names = set()
        for dx in ref_list:
            code = _normalize_icd10(dx.get("icd10", ""))
            if code:
                ref_codes.add(code)
            name = dx.get("diagnosis", "").lower()
            if name:
                ref_names.add(name)

        hits = 0
        for ref_dx in ref_list:
            ref_code = _normalize_icd10(ref_dx.get("icd10", ""))
            ref_name = ref_dx.get("diagnosis", "").lower()
            found = False
            for pred_dx in pred_list:
                pred_code = _normalize_icd10(pred_dx.get("icd10", ""))
                pred_name = pred_dx.get("diagnosis", "").lower()
                if (ref_code and pred_code and ref_code == pred_code) or \
                   (ref_name and pred_name and (ref_name in pred_name or pred_name in ref_name)):
                    found = True
                    break
            if found:
                hits += 1
        coverages.append(hits / len(ref_list))
    return float(np.mean(coverages)) if coverages else 0.0


def imaging_findings_recall(predicted_findings: list[list[str]],
                            reference_findings: list[list[str]]) -> float:
    """Recall of must_include_findings for imaging task."""
    recalls = []
    for pred_list, ref_list in zip(predicted_findings, reference_findings):
        if not ref_list:
            continue
        pred_lower = " ".join(pred_list).lower()
        hits = 0
        for ref_finding in ref_list:
            if ref_finding.lower() in pred_lower:
                hits += 1
        recalls.append(hits / len(ref_list))
    return float(np.mean(recalls)) if recalls else 0.0


# ============================================================================
# CROSS-TASK METRICS
# ============================================================================

def cost_normalized_performance(primary_metric: float, model_name: str,
                                input_tokens: list[int], output_tokens: list[int]) -> float:
    """Primary metric / mean cost per item in USD."""
    from eval.config import estimate_cost_usd

    if not input_tokens:
        return primary_metric  # No cost data

    total_cost = sum(
        estimate_cost_usd(model_name, inp, out)
        for inp, out in zip(input_tokens, output_tokens)
    )
    mean_cost = total_cost / len(input_tokens) if input_tokens else 0.0
    if mean_cost <= 0:
        return float("inf") if primary_metric > 0 else 0.0
    return primary_metric / mean_cost


def inter_model_agreement(predictions_by_model: dict[str, list[bool]]) -> dict[str, float]:
    """Pairwise Cohen's kappa between models on binary correct/incorrect.

    Returns dict with 'mean_kappa' and pairwise 'model_a_vs_model_b' keys.
    """
    try:
        from sklearn.metrics import cohen_kappa_score
    except ImportError:
        log.warning("scikit-learn not installed; skipping Cohen's kappa")
        return {"mean_kappa": 0.0}

    models = sorted(predictions_by_model.keys())
    kappas = {}
    all_k = []

    for i, m1 in enumerate(models):
        for m2 in models[i + 1:]:
            p1 = predictions_by_model[m1]
            p2 = predictions_by_model[m2]
            if len(p1) != len(p2):
                continue
            # Convert to int for kappa
            y1 = [int(x) for x in p1]
            y2 = [int(x) for x in p2]
            try:
                k = cohen_kappa_score(y1, y2)
            except Exception:
                k = 0.0
            key = f"{m1}_vs_{m2}"
            kappas[key] = k
            all_k.append(k)

    kappas["mean_kappa"] = float(np.mean(all_k)) if all_k else 0.0
    return kappas


# ============================================================================
# DISPATCHER
# ============================================================================

def compute_all_metrics(task: str, predictions: list[dict], ground_truths: list[dict],
                        **kwargs) -> dict[str, float]:
    """Compute all metrics for a task. Returns flat dict of metric_name → value."""
    if task == "patient_diagnosis":
        return _compute_patient_diagnosis_metrics(predictions, ground_truths)
    elif task == "context_summarization":
        return _compute_summarization_metrics(predictions, ground_truths, **kwargs)
    elif task == "evidence_retrieval":
        return _compute_retrieval_metrics(predictions, ground_truths, **kwargs)
    elif task == "imaging_indication":
        return _compute_imaging_metrics(predictions, ground_truths, **kwargs)
    elif task in ("differential_diagnosis", "test_selection", "error_detection", "lab_triage", "atypical_diagnosis"):
        from eval.scoring_tasks7 import compute_metrics          # Stage 7 families
        return compute_metrics(task, predictions, ground_truths)
    else:
        raise ValueError(f"Unknown task: {task}")


def _compute_diagnosis_metrics(predictions: list[dict], ground_truths: list[dict]) -> dict:
    return {
        "top_1_accuracy": top_k_accuracy(predictions, ground_truths, k=1),
        "top_3_accuracy": top_k_accuracy(predictions, ground_truths, k=3),
        "top_5_accuracy": top_k_accuracy(predictions, ground_truths, k=5),
        "hierarchical_f1": hierarchical_f1(predictions, ground_truths),
        "mrr": mean_reciprocal_rank(predictions, ground_truths),
    }


def _extract_pred_summaries(predictions: list[dict]) -> list[str]:
    """Pull the summary string from predictions (handles structured sections)."""
    summaries = []
    for p in predictions:
        s = p.get("summary", "")
        if isinstance(s, str) and s:
            summaries.append(s)
            continue
        # Structured current-visit / conditioned output: join section fields.
        parts = []
        for key in ("reason_for_visit", "interval_changes", "active_problems",
                    "context", "summary"):
            v = p.get(key)
            if isinstance(v, str) and v:
                parts.append(v)
            elif isinstance(v, list):
                parts.append(" ".join(str(x) for x in v))
        summaries.append(" ".join(parts) if parts else (json.dumps(s) if s else ""))
    return summaries


def _extract_abstains(predictions: list[dict]) -> list[bool]:
    """The explicit abstention flag of each submission (specialty-conditioned items)."""
    return [bool(isinstance(p, dict) and p.get("abstain") is True) for p in predictions]


def _dominant_variant(ground_truths: list[dict]) -> str:
    """The variant shared by the GT rows in this run (default 'unconditioned')."""
    counts: dict[str, int] = defaultdict(int)
    for gt in ground_truths:
        counts[gt.get("variant") or "unconditioned"] += 1
    return max(counts, key=counts.get) if counts else "unconditioned"


def _acuity_weighted_recall(summaries: list[str],
                            must_include_lists: list[list]) -> float | None:
    """Recall weighted by per-finding acuity. None if findings carry no acuity."""
    has_acuity = any(
        isinstance(f, dict) and f.get("acuity")
        for fl in must_include_lists for f in (fl or [])
    )
    if not has_acuity:
        return None
    num = den = 0.0
    for summary, findings in zip(summaries, must_include_lists):
        for f in findings or []:
            if not isinstance(f, dict):
                continue
            name = f.get("display_name") or f.get("name") or ""
            if not name:
                continue
            w = _ACUITY_WEIGHTS.get(f.get("acuity") or "unspecified", 0.8)
            den += w
            if semantic_match.phrase_in_text(name, summary or ""):
                num += w
    return (num / den) if den else 0.0


def _compute_summarization_metrics(predictions: list[dict], ground_truths: list[dict],
                                    ehr_texts: list[str] | None = None,
                                    **kwargs) -> dict:
    pred_summaries = _extract_pred_summaries(predictions)
    variant = _dominant_variant(ground_truths)

    if variant == "current_visit":
        return _compute_current_visit_metrics(
            pred_summaries, ground_truths, ehr_texts
        )
    if variant == "specialty_conditioned":
        return _compute_specialty_conditioned_metrics(
            pred_summaries, ground_truths, ehr_texts, abstains=_extract_abstains(predictions),
            concept_extractor=kwargs.get("concept_extractor"))

    # --- Unconditioned whole-patient summary (default) ---
    extractor = kwargs.get("concept_extractor")
    charts = list(ehr_texts) if ehr_texts else [gt.get("_chart_text") for gt in ground_truths]
    ref_summaries = [gt.get("reference_summary", "") for gt in ground_truths]
    must_include = [gt.get("must_include_findings", []) for gt in ground_truths]
    terms = [gt.get("_patient_terms") for gt in ground_truths]
    items = [score_summary_item(p, m, c, extractor, t) for p, m, c, t in zip(pred_summaries, must_include, charts, terms)]
    grounded = [it["grounded_precision"] for it in items if it["grounded_precision"] is not None]
    metrics = {
        "rouge_l": rouge_l(pred_summaries, ref_summaries),
        "clinical_f1": float(np.mean([it["clinical_f1"] for it in items])) if items else 0.0,
        "finding_recall": float(np.mean([it["finding_recall"] for it in items])) if items else 0.0,
        "grounded_precision": float(np.mean(grounded)) if grounded else None,
        "length_factor": float(np.mean([it["length_factor"] for it in items])) if items else 0.0,
        "omission_rate": float(np.mean([it["omission_rate"] for it in items])) if items else 1.0,
        "mean_summary_words": float(np.mean([len((s or "").split()) for s in pred_summaries])) if pred_summaries else 0.0,
    }
    aw = _acuity_weighted_recall(pred_summaries, must_include)
    if aw is not None:
        metrics["acuity_weighted_recall"] = aw
    if any(charts):
        metrics["hallucination_rate"] = hallucination_rate(
            pred_summaries, [c or "" for c in charts], concept_extractor=extractor, patient_terms=terms)
    return metrics


def _compute_current_visit_metrics(pred_summaries: list[str],
                                   ground_truths: list[dict],
                                   ehr_texts: list[str] | None = None) -> dict:
    """Current-visit summarization: recall of active findings + interval deltas.

    The EHR input was truncated to encounters <= the index encounter, so
    hallucination is scored against that same truncated context.
    """
    ref_summaries = [gt.get("reference_summary", "") for gt in ground_truths]
    must_include = [gt.get("must_include_findings", []) for gt in ground_truths]
    new_lists = [(gt.get("deltas") or {}).get("new", []) for gt in ground_truths]
    resolved_lists = [(gt.get("deltas") or {}).get("resolved", []) for gt in ground_truths]
    off_target_lists = [gt.get("off_target_findings", []) for gt in ground_truths]
    future_lists = [gt.get("future_findings", []) for gt in ground_truths]
    # "Should-not-include" set = irrelevant (background/distractor) + future-only
    # (temporal leakage). Both penalize precision.
    neg_lists = [(o or []) + (f or []) for o, f in zip(off_target_lists, future_lists)]

    recall = clinical_f1(pred_summaries, must_include)
    # Fractions of each "should-not-include" set the summary pulled in (lower is
    # better). precision = 1 - (fraction of the combined negative set included).
    off_target_rate = _list_recall(pred_summaries, off_target_lists)
    future_leakage_rate = _list_recall(pred_summaries, future_lists)
    precision = 1.0 - _list_recall(pred_summaries, neg_lists)
    selection_f1 = (
        2 * recall * precision / (recall + precision)
        if (recall + precision) > 0 else 0.0
    )

    metrics = {
        "rouge_l": rouge_l(pred_summaries, ref_summaries),
        "clinical_f1": recall,
        "omission_rate": 1.0 - recall,
        "delta_coverage_new": _list_recall(pred_summaries, new_lists),
        "delta_coverage_resolved": _list_recall(pred_summaries, resolved_lists),
        "off_target_rate": off_target_rate,
        "future_leakage_rate": future_leakage_rate,
        "selection_f1": selection_f1,
        "mean_summary_words": float(np.mean([len((s or "").split()) for s in pred_summaries])) if pred_summaries else 0.0,
    }
    if ehr_texts:
        metrics["hallucination_rate"] = hallucination_rate(pred_summaries, ehr_texts)
    return metrics


def _is_abstention(prediction) -> bool:
    """A submission abstains iff it carries an explicit `abstain: true`. Text never abstains: the
    old rule (<= 12 words, or a stock phrase such as "no significant" anywhere in the text) let a full
    summary with "No significant distress." appended score 1.0 on every absent item (audit #4)."""
    return bool(isinstance(prediction, dict) and prediction.get("abstain") is True)


def _tier(gt: dict, name: str) -> list:
    return (gt.get("tiers") or {}).get(name, []) or []


def _critical(findings: list) -> list:
    return [f for f in findings if isinstance(f, dict) and f.get("importance") == "critical"]


def score_specialty_item(summary: str, abstain: bool, gt: dict, chart_text: str | None = None,
                         concept_extractor=None) -> dict:
    """Specialty-conditioned reward (Stage 3).

    Involved item (the specialty has an active problem):
        target    = critical primary+relevant findings, or all of them when the item has none
                    (540 items used to score 0 for every answer, audit #23)
        recall    = target findings stated as present (negation-aware, concept-level)
        precision = 1 - share of the excluded (off-specialty) sample the summary mentions
        conditioned_f1 = HM(recall, precision) x length factor; 0 if the submission abstains
    Absent item (no active problem): abstention_accuracy = 1 iff `abstain` is true.
    """
    summary = summary or ""
    primary, relevant = _tier(gt, "primary"), _tier(gt, "relevant")
    involved = gt.get("involvement") in ("high", "low", "involved") or bool(primary or relevant)
    excluded = _tier(gt, "excluded_sample")
    leakage = finding_recall(summary, excluded, concept_extractor) if excluded else 0.0
    if not involved:
        return {"involved": False, "abstention_accuracy": 1.0 if abstain else 0.0,
                "absent_leakage_rate": 0.0 if abstain else leakage}
    combined = primary + relevant
    crit = _critical(combined)
    target = crit or combined
    # an excluded finding whose name is implied by the target findings themselves ("Hypotension" vs
    # "Orthostatic hypotension") cannot be avoided by any faithful summary: it is not leakage
    if excluded and combined:
        target_text = ". ".join(_finding_names(combined))
        excluded = [e for e in excluded if not concept_match.present(_finding_names([e])[0] if _finding_names([e]) else "", target_text, concept_extractor)]
        leakage = finding_recall(summary, excluded, concept_extractor) if excluded else 0.0
    recall_target = finding_recall(summary, target, concept_extractor) if not abstain else 0.0
    recall_all = finding_recall(summary, combined, concept_extractor) if not abstain else 0.0
    precision = 1.0 - leakage
    length = concept_match.length_factor(summary)
    out = {
        "involved": True,
        "wrong_abstention": abstain,
        "primary_recall_critical": finding_recall(summary, _critical(primary) or primary, concept_extractor) if not abstain else 0.0,
        "primary_recall_complete": finding_recall(summary, primary, concept_extractor) if not abstain else 0.0,
        "relevant_recall_critical": finding_recall(summary, _critical(relevant) or relevant, concept_extractor) if relevant and not abstain else None,
        "relevant_recall_complete": finding_recall(summary, relevant, concept_extractor) if relevant and not abstain else None,
        "leakage_rate": leakage if not abstain else 0.0,
        "leakage_rate_strict": finding_recall(summary, excluded + _tier(gt, "neutral"), concept_extractor) if (excluded or _tier(gt, "neutral")) and not abstain else 0.0,
        "conditioned_f1": 0.0 if abstain else concept_match.harmonic(recall_target, precision) * length,
        "conditioned_f1_complete": 0.0 if abstain else concept_match.harmonic(recall_all, precision) * length,
        "critical_fallback": not crit,
        "length_factor": length,
    }
    out["omission_rate"] = 1.0 - out["primary_recall_complete"]
    return out


def _compute_specialty_conditioned_metrics(pred_summaries: list[str],
                                           ground_truths: list[dict],
                                           ehr_texts: list[str] | None = None,
                                           abstains: list[bool] | None = None,
                                           concept_extractor=None) -> dict:
    """Batch = mean of per-item scores over involved and absent items respectively."""
    abstains = abstains or [False] * len(pred_summaries)
    charts = list(ehr_texts) if ehr_texts else [gt.get("_chart_text") for gt in ground_truths]
    items = [score_specialty_item(s, a, gt, c, concept_extractor)
             for s, a, gt, c in zip(pred_summaries, abstains, ground_truths, charts)]
    inv = [it for it in items if it["involved"]]
    ab = [it for it in items if not it["involved"]]

    def mean(key, pool):
        vals = [it[key] for it in pool if it.get(key) is not None]
        return float(np.mean(vals)) if vals else 0.0

    metrics = {
        "n_involved": len(inv),
        "n_absent": len(ab),
        "mean_summary_words": float(np.mean([len((s or "").split()) for s in pred_summaries])) if pred_summaries else 0.0,
    }
    if inv:
        metrics.update({
            "primary_recall_critical": mean("primary_recall_critical", inv),
            "primary_recall_complete": mean("primary_recall_complete", inv),
            "relevant_recall_critical": mean("relevant_recall_critical", inv),
            "relevant_recall_complete": mean("relevant_recall_complete", inv),
            "leakage_rate": mean("leakage_rate", inv),
            "leakage_rate_strict": mean("leakage_rate_strict", inv),
            "conditioned_f1": mean("conditioned_f1", inv),
            "conditioned_f1_complete": mean("conditioned_f1_complete", inv),
            "omission_rate": mean("omission_rate", inv),
            "wrong_abstention_rate": float(np.mean([it["wrong_abstention"] for it in inv])),
            "critical_fallback_rate": float(np.mean([it["critical_fallback"] for it in inv])),
            "length_factor": mean("length_factor", inv),
        })
        if any(charts):
            inv_idx = [i for i, it in enumerate(items) if it["involved"]]
            metrics["hallucination_rate"] = hallucination_rate(
                [pred_summaries[i] for i in inv_idx], [charts[i] or "" for i in inv_idx], concept_extractor=concept_extractor)
    if ab:
        metrics["abstention_accuracy"] = mean("abstention_accuracy", ab)
        metrics["absent_leakage_rate"] = mean("absent_leakage_rate", ab)
    return metrics


def _normalize_passage_id(pid: str, judg_keys: set[str]) -> str:
    """Normalize passage_id to match judgment keys.

    Agents may submit raw numeric IDs (e.g., '2085') while judgments use
    prefixed IDs (e.g., 'ees_2085'). Try the raw ID first, then common prefixes.
    """
    pid_str = str(pid)
    if pid_str in judg_keys:
        return pid_str
    # Try adding common prefixes
    for prefix in ("ees_", "fc_"):
        prefixed = f"{prefix}{pid_str}"
        if prefixed in judg_keys:
            return prefixed
    # Try stripping prefix if agent included one
    if "_" in pid_str:
        stripped = pid_str.split("_", 1)[1]
        if stripped in judg_keys:
            return stripped
    # Any other spelling of a section id ("ehr_section_1234", "section 1234", "S1234"): the trailing number is
    # the encounter_ehr_sections id. The paper's agent prompt showed "ehr_section_1234" as the format, so
    # agents following it scored 0 on every ranking (found by the Stage-8 atomic smoke).
    import re as _re
    m = _re.search(r"(\d+)\s*$", pid_str)
    if m and f"ees_{m.group(1)}" in judg_keys:
        return f"ees_{m.group(1)}"
    return pid_str


def _compute_retrieval_metrics(predictions: list[dict], ground_truths: list[dict],
                                latencies_ms: list[int] | None = None,
                                **kwargs) -> dict:
    # predictions should be list of {"rankings": [{"passage_id", "grade"}, ...]}
    ranked_results = []
    judgments_list = []
    for pred, gt in zip(predictions, ground_truths):
        raw_rankings = pred.get("rankings", [])
        judg = gt.get("_judgments", {})
        judg_keys = set(judg.keys())
        # Normalize passage IDs to match judgment keys
        normalized = []
        for item in raw_rankings:
            norm_item = dict(item)
            norm_item["passage_id"] = _normalize_passage_id(
                item.get("passage_id", ""), judg_keys
            )
            normalized.append(norm_item)
        ranked_results.append(normalized)
        judgments_list.append(judg)

    metrics = {
        "ndcg_10": ndcg_at_k(ranked_results, judgments_list, k=10),
        "map_10": map_at_k(ranked_results, judgments_list, k=10),
        "recall_5": recall_at_k(ranked_results, judgments_list, k=5),
        "recall_10": recall_at_k(ranked_results, judgments_list, k=10),
        "recall_20": recall_at_k(ranked_results, judgments_list, k=20),
        "precision_5": precision_at_k(ranked_results, judgments_list, k=5),
        "mrr": retrieval_mrr(ranked_results, judgments_list),
    }

    if latencies_ms:
        metrics["latency_adjusted_ndcg"] = latency_adjusted_ndcg(
            ranked_results, judgments_list, latencies_ms, k=10
        )

    return metrics


def _compute_imaging_metrics(predictions: list[dict], ground_truths: list[dict],
                             concept_extractor=None, **kwargs) -> dict:
    """Imaging-indication metrics. `concept_extractor` (eval.imaging_concepts.ConceptExtractor,
    built from the loaded database) adds the paper's primary metric, concept-level F1 of the
    inferred clinical question; without it only the token-level F1 is reported."""
    pred_questions = [p.get("clinical_question", "") for p in predictions]
    ref_questions = [gt.get("inferred_clinical_question", "") for gt in ground_truths]
    pred_summaries = [p.get("pre_read_summary", "") for p in predictions]
    ref_summaries = [gt.get("pre_read_summary", "") for gt in ground_truths]
    pred_diffs = [p.get("differential", []) for p in predictions]
    ref_diffs = [gt.get("differential_context", []) for gt in ground_truths]
    pred_findings = [p.get("must_include_findings", []) for p in predictions]
    ref_findings = [gt.get("must_include_findings", []) for gt in ground_truths]

    metrics = {
        "clinical_question_f1": clinical_question_f1(pred_questions, ref_questions),
        "summary_rouge_l": rouge_l(pred_summaries, ref_summaries),
        "differential_coverage": differential_coverage(pred_diffs, ref_diffs),
        "findings_recall": imaging_findings_recall(pred_findings, ref_findings),
    }
    if concept_extractor is not None:
        from eval.imaging_concepts import concept_f1_batch
        # Stage 3: the primary reference is the deterministic concept set built from the graph
        # (reference_terms: the encounter's correct diagnosis, its differential and its key findings);
        # the LLM-authored question is scored as a secondary metric when both exist.
        det_refs = [reference_terms_text(gt) for gt in ground_truths]
        if all(det_refs):
            metrics.update(concept_f1_batch(concept_extractor, pred_questions, det_refs))
            llm = concept_f1_batch(concept_extractor, pred_questions, ref_questions)
            metrics["clinical_question_concept_f1_llm"] = llm["clinical_question_concept_f1"]
        else:
            metrics.update(concept_f1_batch(concept_extractor, pred_questions, ref_questions))
    return metrics


def reference_terms_text(gt: dict) -> str:
    """The deterministic imaging reference as text: graph names of the correct diagnosis, the
    differential (distractor diagnoses) and the encounter's key findings, or '' when absent."""
    terms = gt.get("reference_terms")
    if not isinstance(terms, dict):
        return ""
    names = [n for key in ("diagnosis", "differential", "findings") for n in (terms.get(key) or [])]
    return ". ".join(str(n) for n in names if n)
