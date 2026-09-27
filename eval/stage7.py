"""Shared, pure helpers for the Stage-7 task families (see audit/STAGE7_TODO.md).

Used by the builder (scripts/build_stage7_tasks.py), the scorers (eval/scoring_tasks7.py) and both
environments (order matching, section overrides, hidden result sections). Everything here is
deterministic and free of I/O except the SQLite readers, which take a `sqlite3.Connection`.
"""

from __future__ import annotations

import json
import re
import sqlite3
from typing import Any

ORDERABLE_TYPES = ("lab_value", "imaging_finding", "procedure_result")
RESULT_SECTIONS = frozenset({"labs", "imaging", "pathology", "other_studies"})
TRIAGE_TYPES = ("lab_value", "vital_sign")
DISCRIMINATING_EDGES = ("pathognomonic", "highly_suggestive")
EDGE_RANK = {"pathognomonic": 3, "highly_suggestive": 2, "commonly_seen": 1}
ERROR_TYPES = ("implausible_value", "laterality", "age_contradiction", "sex_contradiction")
NEW_TASKS = ("differential_diagnosis", "test_selection", "error_detection", "lab_triage", "atypical_diagnosis")
ENCOUNTER_TASKS = NEW_TASKS                     # all Stage-7 tasks are bound to an index encounter
MASK = "[finding not documented]"
NOT_DOCUMENTED = "not performed / no result documented at this visit"

_STOP = {"of", "the", "a", "an", "and", "in", "on", "with", "for", "to", "level", "levels", "test", "positive", "negative"}
_SYN = {"wbc": "white", "cbc": "complete", "hgb": "hemoglobin", "plt": "platelet", "cr": "creatinine", "cxr": "chest xray",
        "ct": "computed tomography", "mri": "magnetic resonance", "us": "ultrasound", "ecg": "electrocardiogram",
        "ekg": "electrocardiogram", "crp": "reactive protein", "esr": "sedimentation", "bnp": "natriuretic", "tsh": "thyroid stimulating",
        "ua": "urinalysis", "lfts": "liver", "bmp": "metabolic", "cmp": "metabolic", "hba1c": "hemoglobin a1c", "inr": "prothrombin",
        "abg": "arterial gas", "lp": "lumbar puncture", "csf": "cerebrospinal", "echo": "echocardiogram", "x-ray": "xray"}


# ---------------------------------------------------------------------------
# names
# ---------------------------------------------------------------------------

PANELS: dict[str, tuple[str, ...]] = {
    "cbc": ("white blood cell count", "hemoglobin", "hematocrit", "platelet count", "leukocyte count", "neutrophil count"),
    "complete blood count": ("white blood cell count", "hemoglobin", "hematocrit", "platelet count", "leukocyte count", "neutrophil count"),
    "bmp": ("sodium", "potassium", "chloride", "bicarbonate", "blood urea nitrogen", "creatinine", "glucose", "calcium"),
    "basic metabolic panel": ("sodium", "potassium", "chloride", "bicarbonate", "blood urea nitrogen", "creatinine", "glucose", "calcium"),
    "cmp": ("sodium", "potassium", "chloride", "bicarbonate", "blood urea nitrogen", "creatinine", "glucose", "calcium",
            "albumin", "total bilirubin", "alkaline phosphatase", "ast", "alt", "total protein"),
    "comprehensive metabolic panel": ("sodium", "potassium", "chloride", "bicarbonate", "blood urea nitrogen", "creatinine", "glucose",
                                      "calcium", "albumin", "total bilirubin", "alkaline phosphatase", "ast", "alt", "total protein"),
    "lfts": ("ast", "alt", "alkaline phosphatase", "total bilirubin", "albumin"), "liver function tests": ("ast", "alt", "alkaline phosphatase", "total bilirubin", "albumin"),
    "coags": ("prothrombin time", "inr", "partial thromboplastin time"), "coagulation panel": ("prothrombin time", "inr", "partial thromboplastin time"),
    "urinalysis": ("urine protein", "urine blood", "urine leukocyte esterase", "urine nitrite", "urine glucose", "urine ketones", "urine specific gravity", "urinalysis"),
    "lipid panel": ("total cholesterol", "ldl cholesterol", "hdl cholesterol", "triglycerides"),
    "thyroid panel": ("thyroid stimulating hormone", "free t4", "free thyroxine", "t3"),
    "abg": ("arterial ph", "pco2", "po2", "arterial blood gas"), "arterial blood gas": ("arterial ph", "pco2", "po2", "arterial blood gas"),
}
"""Order panels: one order reveals every member the encounter documented (one step, one order)."""


def norm_tokens(name: str) -> frozenset[str]:
    """Token set of a test / finding name after lower-casing, abbreviation expansion, light stemming."""
    t = (name or "").lower()
    t = re.sub(r"\bx[- ]?rays?\b|\bradiograph(?:y|s)?\b", "xray", t)
    t = t.replace("-", " ").replace("/", " ")
    words = re.findall(r"[a-z0-9]+", t)
    out = set()
    for w in words:
        w = _SYN.get(w, w)
        for part in w.split():
            if part in _STOP or len(part) < 2:
                continue
            if len(part) > 4 and part.endswith("s"):
                part = part[:-1]
            out.add(part)
    return frozenset(out)


def name_similarity(a: str, b: str) -> float:
    """Jaccard of the normalized token sets, 1.0 when one name's tokens are contained in the other's."""
    ta, tb = norm_tokens(a), norm_tokens(b)
    if not ta or not tb:
        return 0.0
    if ta <= tb or tb <= ta:
        return 1.0
    return len(ta & tb) / len(ta | tb)


def match_name(query: str, candidates: list[str], threshold: float = 0.5) -> str | None:
    """The candidate name the query denotes, or None: an identical token set wins outright ("Baseline
    hemoglobin" is not "Hemoglobin" when both are documented), then the highest similarity."""
    qt = norm_tokens(query)
    for c in candidates:
        if qt and norm_tokens(c) == qt:
            return c
    best, best_s = None, 0.0
    for c in candidates:
        s = name_similarity(query, c)
        if s > best_s:
            best, best_s = c, s
    return best if best_s >= threshold else None


def match_names(queries: list[str], candidates: list[str], threshold: float = 0.5) -> dict[str, str | None]:
    """One-to-one assignment: identical token sets first, then the best remaining similarity >= threshold."""
    out: dict[str, str | None] = {q: None for q in queries}
    free = list(dict.fromkeys(candidates))
    ctoks = {c: norm_tokens(c) for c in free}
    for q in queries:
        qt = norm_tokens(q)
        exact = next((c for c in free if ctoks[c] == qt and qt), None)
        if exact is not None:
            out[q] = exact
            free.remove(exact)
    pairs = sorted(((name_similarity(q, c), q, c) for q in queries if out[q] is None for c in free), reverse=True)
    used_q: set[str] = set()
    for sim, q, c in pairs:
        if sim < threshold:
            break
        if q in used_q or c not in free:
            continue
        out[q] = c
        used_q.add(q)
        free.remove(c)
    return out


# ---------------------------------------------------------------------------
# SQLite readers (release schema)
# ---------------------------------------------------------------------------

def encounter_question(conn: sqlite3.Connection, encounter_id: int) -> int | None:
    row = conn.execute("select source_question_ids from longitudinal_encounters where encounter_id = ?", (encounter_id,)).fetchone()
    if not row or not row[0]:
        return None
    try:
        q = json.loads(row[0])
    except (json.JSONDecodeError, TypeError):
        return None
    return int(q[0]) if isinstance(q, list) and q else None


def encounter_findings(conn: sqlite3.Connection, encounter_id: int, types: tuple[str, ...], label_dx: list[int]) -> list[dict]:
    """The encounter's findings of the given types with their typed edge to the label diagnoses."""
    q = encounter_question(conn, encounter_id)
    if q is None:
        return []
    rows = conn.execute(
        f"select cf.finding_id, cf.display_name, cf.finding_type, qf.present, qf.value_text, qf.relevance "
        f"from question_findings qf join clinical_findings cf using(finding_id) "
        f"where qf.question_id = ? and cf.finding_type in ({','.join('?' * len(types))}) order by qf.id", (q, *types)).fetchall()
    edges: dict[int, str] = {}
    if label_dx:
        for fid, rel in conn.execute(
                f"select finding_id, relationship from diagnosis_findings where diagnosis_id in ({','.join('?' * len(label_dx))})",
                label_dx).fetchall():
            if EDGE_RANK.get(rel, 0) > EDGE_RANK.get(edges.get(fid), 0):
                edges[fid] = rel
    out = []
    for fid, name, ftype, present, value, rel in rows:
        out.append({"finding_id": fid, "name": name, "type": ftype, "present": 1 if present is None else int(present),
                    "value": value, "relevance": rel, "edge": edges.get(fid)})
    return out


def distractors(conn: sqlite3.Connection, encounter_id: int) -> list[dict]:
    q = encounter_question(conn, encounter_id)
    if q is None:
        return []
    return [{"diagnosis_id": d, "icd10": icd, "display_name": name} for d, icd, name in conn.execute(
        "select d.diagnosis_id, d.icd10_code, d.display_name from question_diagnoses qd join diagnoses d using(diagnosis_id) "
        "where qd.question_id = ? and qd.role = 'distractor' and d.merged_into is null order by qd.id", (q,)).fetchall()]


def index_sections(conn: sqlite3.Connection, encounter_id: int) -> list[tuple[int, str, str]]:
    return [(i, t, x or "") for i, t, x in conn.execute(
        "select id, section_type, section_text from encounter_ehr_sections where encounter_id = ? "
        "and section_type not in ('assessment', 'plan') order by section_order, id", (encounter_id,)).fetchall()]


# ---------------------------------------------------------------------------
# error injection (deterministic; returns None when the section does not admit the error)
# ---------------------------------------------------------------------------

_NUM_LINE = re.compile(r"^(?P<head>[-•*\s]*[^:\n]{2,60}:\s*)(?P<num>\d{1,3}(?:,\d{3})*(?:\.\d+)?)(?P<tail>\s*[^\n]*)$", re.M)
_LATERAL = re.compile(r"\b(left|right)\b", re.I)
_AGE = re.compile(r"\b(\d{1,3})-year-old\b")
_PRONOUN = {"he": "she", "she": "he", "his": "her", "her": "his", "him": "her", "himself": "herself", "herself": "himself"}


def inject_implausible_value(text: str) -> tuple[str, dict] | None:
    for m in _NUM_LINE.finditer(text):
        raw = m.group("num")
        value = float(raw.replace(",", ""))
        if value <= 0:
            continue
        new = value * 10 if value < 1000 else value / 100
        new_s = f"{new:.1f}".rstrip("0").rstrip(".") if new < 100 else f"{int(round(new)):,}"
        line = m.group(0)
        injected = m.group("head") + new_s + m.group("tail")
        return text[:m.start()] + injected + text[m.end():], {"original": line.strip(), "injected": injected.strip()}
    return None


def inject_laterality(text: str) -> tuple[str, dict] | None:
    m = _LATERAL.search(text)
    if not m:
        return None
    word = m.group(1)
    swap = {"left": "right", "right": "left"}[word.lower()]
    swap = swap.capitalize() if word[0].isupper() else swap
    sent = _sentence_at(text, m.start())
    new_text = text[:m.start()] + swap + text[m.end():]
    return new_text, {"original": sent, "injected": _sentence_at(new_text, m.start())}


def inject_age(text: str, stated_age: int | None) -> tuple[str, dict] | None:
    m = _AGE.search(text)
    if not m:
        return None
    age = int(m.group(1))
    if stated_age is not None and abs(age - stated_age) > 3:
        return None                                  # the text already disagrees with the demographics; skip
    new_age = age + 40 if age + 40 <= 95 else max(1, age - 40)
    sent = _sentence_at(text, m.start())
    new_text = text[:m.start(1)] + str(new_age) + text[m.end(1):]
    return new_text, {"original": sent, "injected": _sentence_at(new_text, m.start())}


def inject_sex(text: str) -> tuple[str, dict] | None:
    for sent_start, sent in _sentences(text):
        words = re.findall(r"\b(he|she|his|her|him|himself|herself)\b", sent, re.I)
        if len(words) >= 2:
            new_sent = re.sub(r"\b(he|she|his|her|him|himself|herself)\b",
                              lambda mm: _keep_case(mm.group(1), _PRONOUN[mm.group(1).lower()]), sent, flags=re.I)
            new_text = text[:sent_start] + new_sent + text[sent_start + len(sent):]
            return new_text, {"original": sent, "injected": new_sent}
    return None


def _keep_case(src: str, repl: str) -> str:
    return repl.capitalize() if src[0].isupper() else repl


def _sentences(text: str):
    for m in re.finditer(r"[^.!?\n]+[.!?]?", text):
        s = m.group(0)
        if s.strip():
            yield m.start(), s


def _sentence_at(text: str, pos: int) -> str:
    for start, s in _sentences(text):
        if start <= pos < start + len(s):
            return s.strip()
    return text.strip()[:200]


def inject_error(sections: list[tuple[int, str, str]], error_type: str, stated_age: int | None = None) -> dict | None:
    """Try one error type on the sections it applies to; returns the label dict or None."""
    prefs = {"implausible_value": ("labs", "vitals"), "laterality": ("physical_exam", "hpi", "imaging"),
             "age_contradiction": ("hpi",), "sex_contradiction": ("hpi", "physical_exam")}
    fn = {"implausible_value": lambda t: inject_implausible_value(t), "laterality": inject_laterality,
          "age_contradiction": lambda t: inject_age(t, stated_age), "sex_contradiction": inject_sex}[error_type]
    for wanted in prefs[error_type]:
        for sid, st, text in sections:
            if st != wanted or not text:
                continue
            r = fn(text)
            if r is not None:
                new_text, lab = r
                return {"section_id": sid, "section_type": st, "error_type": error_type, **lab,
                        "section_overrides": {str(sid): {"original": lab["original"], "injected": lab["injected"]}}}
    return None


# ---------------------------------------------------------------------------
# masking (atypical variants)
# ---------------------------------------------------------------------------

def mask_findings(sections: list[tuple[int, str, str]], names: list[str]) -> tuple[dict[str, dict], list[str]]:
    """Mask every sentence that mentions one of the finding names. Returns (section_overrides, masked names)."""
    from eval.concept_match import Phrase, TextIndex, mentioned_polarity
    overrides: dict[str, dict] = {}
    masked: list[str] = []
    phrases = [(n, Phrase(n)) for n in names]
    for sid, st, text in sections:
        if st in RESULT_SECTIONS or st in ("demographics", "medications", "allergies", "family_history", "social_history"):
            pass                                            # masking a result is fine; the exclusions are chart context
        if not text:
            continue
        new_text = text
        hit_here = []
        for start, sent in list(_sentences(text)):
            idx = TextIndex(sent)
            for n, ph in phrases:
                if mentioned_polarity(ph, idx) == "present":
                    hit_here.append((sent, n))
                    break
        if not hit_here:
            continue
        repl = []
        for sent, n in hit_here:
            masked_sent = sent[: len(sent) - len(sent.lstrip())] + MASK + ("." if sent.rstrip().endswith(".") else "")
            repl.append([sent, masked_sent])
            masked.append(n)
        overrides[str(sid)] = {"replacements": repl}          # compact: only the masked sentences are stored
    return overrides, sorted(set(masked))


# ---------------------------------------------------------------------------
# applying overrides / hidden sections to observations (both environments)
# ---------------------------------------------------------------------------

def apply_override(text: str | None, ov: dict) -> str:
    """An override is either one {original, injected} pair or {replacements: [[original, injected], ...]}."""
    out = text or ""
    if ov.get("full"):
        return ov["injected"]
    for orig, inj in ov.get("replacements") or ([[ov["original"], ov["injected"]]] if "original" in ov else []):
        out = out.replace(orig, inj, 1)
    return out


def apply_episode_rules(tool: str, args: dict, obs: Any, overrides: dict[str, dict] | None,
                        hidden_types: set[str] | frozenset[str] | None, index_encounter: int | None) -> Any:
    """Section overrides (error_detection, atypical_diagnosis) and hidden result sections (test_selection)."""
    overrides = overrides or {}
    hidden = set(hidden_types or ())
    if not overrides and not hidden:
        return obs

    def fix_section(d: dict) -> dict | None:
        sid = d.get("section_id")
        st = str(d.get("section_type") or "")
        if hidden and st in hidden and d.get("encounter_id") == index_encounter:
            return None
        if sid is not None and str(sid) in overrides and "section_text" in d:
            return {**d, "section_text": apply_override(d.get("section_text"), overrides[str(sid)])}
        return d

    if isinstance(obs, dict):
        if "sections" in obs and isinstance(obs["sections"], list):          # view_encounter_detail
            out = [fix_section(s) if isinstance(s, dict) else s for s in obs["sections"]]
            return {**obs, "sections": [s for s in out if s is not None]}
        if "section_id" in obs:                                                # view_section
            fixed = fix_section(obs)
            return fixed if fixed is not None else {"error": "This section is not available in this task."}
        return obs
    if isinstance(obs, list):
        if tool == "view_results" and hidden:
            rt = {"labs": "labs", "imaging": "imaging", "pathology": "pathology"}.get(str((args or {}).get("result_type")), "")
            if rt in hidden:
                return [r for r in obs if not (isinstance(r, dict) and r.get("encounter_id") == index_encounter)]
        out = [fix_section(x) if isinstance(x, dict) else x for x in obs]     # search_chart, view_medications
        return [x for x in out if x is not None]
    return obs


ORDERABLE_FIELDS = ("name", "type", "present", "value", "relevance", "edge")


def orderable_from_gt(gt: dict) -> list[dict]:
    """The encounter's orderable findings as dicts; stored compactly as rows in the ground truth."""
    rows = gt.get("orderable") or []
    return [dict(zip(ORDERABLE_FIELDS, r)) if isinstance(r, list) else r for r in rows]


def error_pair(gt: dict) -> tuple[str, str]:
    """(original, injected) of an error_detection label, stored once in its section override."""
    ov = next(iter((gt.get("section_overrides") or {}).values()), {})
    return ov.get("original", gt.get("original", "")), ov.get("injected", gt.get("injected", ""))


def order_result(name: str, orderable: list[dict]) -> dict:
    """What `order_test` returns: the documented result(s) of the matched finding(s), or not-documented.
    A panel name (CBC, BMP, LFTs, ...) reveals every member the encounter documented."""
    documented = list(orderable)                      # present findings and explicit negatives alike
    names = [f["name"] for f in documented]
    queries = [name]
    key = " ".join(norm_tokens(name)) if name else ""
    for panel, members in PANELS.items():
        if norm_tokens(panel) == norm_tokens(name):
            queries = list(members)
            break
    hits: list[dict] = []
    for q in queries:
        hit = match_name(q, names)
        if hit is not None and all(h["matched"] != hit for h in hits):
            f = next(x for x in documented if x["name"] == hit)
            if f.get("present"):
                result = f.get("value") or "present / abnormal as documented"
            else:
                result = f"negative / not present{(': ' + str(f['value'])) if f.get('value') else ''}"
            hits.append({"matched": hit, "result": result, "type": f.get("type")})
    if not hits:
        return {"test": name, "matched": None, "result": NOT_DOCUMENTED}
    if len(hits) == 1:
        return {"test": name, **hits[0]}
    return {"test": name, "matched": [h["matched"] for h in hits], "results": hits}


def ordered_names(order_results: list[dict]) -> set[str]:
    """Finding names an episode's orders revealed (for coverage)."""
    out: set[str] = set()
    for r in order_results:
        m = r.get("matched")
        if isinstance(m, list):
            out.update(m)
        elif m:
            out.add(m)
    return out
