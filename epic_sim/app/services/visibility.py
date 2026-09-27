"""Server-side visibility rules: what any consumer of the simulator may see.

One place for the three rules the paper's `/env` endpoint applied on its own (and the Epic tool API,
FHIR and the paper's agent harness did not), so that no consumer can read the labels:

1. **Outcome sections are hidden.** Assessment and plan state the conclusion of a visit; they are
   removed from every read path (`hidden_sections()`), not only from `/env` observations.
2. **The problem list is the chart's documented history.** The profile's chronic conditions, i.e.
   what the notes themselves list under "Active Problem List", never the graph-derived correct
   diagnoses of the patient's source questions, which are the patient-diagnosis reference itself
   (`documented_problems`).
3. **Point-in-time cutoff.** For an imaging-indication instance nothing after the ordering encounter
   is visible (`allowed_encounter_ids`, `filter_future_encounters`). Any consumer that identifies its
   instance (an `/env` episode, an agent session with `gt_id`, a FHIR call with `X-Session-Id`) gets
   the cutoff applied.

The functions are pure or read-only; audit/FINDINGS.md #5, #10 and F§6 5.3 are the defects they close.
"""

from __future__ import annotations

import json
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from epic_sim.app.config import settings
from epic_sim.app.models.benchmark import BenchmarkGroundTruth
from epic_sim.app.models.longitudinal import LongitudinalEncounter, LongitudinalPatient
from epic_sim.app.schemas.epic import ProblemEntry

OUTCOME_SECTIONS: frozenset[str] = frozenset({"assessment", "plan"})
POINT_IN_TIME_TASKS: frozenset[str] = frozenset({"imaging_indication", "patient_diagnosis",
                                                  "differential_diagnosis", "test_selection", "error_detection",
                                                  "lab_triage", "atypical_diagnosis"})
"""Tasks whose instance is bound to an encounter (encounter_id set): nothing after it is visible.
Longitudinal rows (encounter_id NULL) have no cutoff."""


def hidden_sections() -> frozenset[str]:
    """Section types no consumer may read (empty only if EPIC_SIM_HIDE_OUTCOME_SECTIONS=0)."""
    return OUTCOME_SECTIONS if settings.hide_outcome_sections else frozenset()


def is_hidden(section_type: Any) -> bool:
    value = getattr(section_type, "value", section_type)
    return str(value or "").lower() in hidden_sections()


# ---------------------------------------------------------------------------
# 1. outcome sections
# ---------------------------------------------------------------------------

def strip_outcome_sections(obj: Any) -> Any:
    """Recursively drop assessment/plan entries from a tool or API result."""
    hidden = hidden_sections()
    if not hidden:
        return obj
    if isinstance(obj, list):
        return [strip_outcome_sections(i) for i in obj
                if not (isinstance(i, dict) and str(i.get("section_type", "")).lower() in hidden)]
    if isinstance(obj, dict):
        return {k: strip_outcome_sections(v) for k, v in obj.items()
                if not (k.lower() in hidden and isinstance(v, str))}
    return obj


# ---------------------------------------------------------------------------
# 2. documented history
# ---------------------------------------------------------------------------

def problems_from_profile(profile: dict | str | None) -> list[ProblemEntry]:
    """The chart's problem list: the profile's chronic conditions, in order, as they appear under
    'Active Problem List' in every note. No codes: the chart lists names."""
    if isinstance(profile, str):
        try:
            profile = json.loads(profile or "{}")
        except ValueError:
            profile = {}
    out = []
    for i, c in enumerate(((profile or {}).get("chronic_conditions") or [])):
        if c is None:
            continue
        name = c if not isinstance(c, dict) else (c.get("name") or c.get("condition") or "")
        name = str(name or "").strip()
        if name:
            out.append(ProblemEntry(diagnosis_id=None, display_name=name, source="chart_history",
                                    problem_id=f"chart-{i}"))
    return out


async def documented_problems(db: AsyncSession, patient_id: int) -> list[ProblemEntry]:
    row = await db.execute(select(LongitudinalPatient.profile).where(LongitudinalPatient.patient_id == patient_id))
    profile = row.scalar_one_or_none()
    return problems_from_profile(profile)


# ---------------------------------------------------------------------------
# 3. point in time
# ---------------------------------------------------------------------------

async def allowed_encounter_ids(db: AsyncSession, gt_id: int | None) -> set[int] | None:
    """Encounters visible for a benchmark instance, or None when the task has no cutoff."""
    if gt_id is None:
        return None
    row = (await db.execute(
        select(BenchmarkGroundTruth.task, BenchmarkGroundTruth.patient_id, BenchmarkGroundTruth.encounter_id)
        .where(BenchmarkGroundTruth.gt_id == gt_id))).first()
    if row is None:
        return None
    task, patient_id, encounter_id = row
    task = getattr(task, "value", task)
    if task not in POINT_IN_TIME_TASKS or encounter_id is None:
        return None
    target = (await db.execute(
        select(LongitudinalEncounter.patient_id, LongitudinalEncounter.encounter_order)
        .where(LongitudinalEncounter.encounter_id == encounter_id))).first()
    if target is None:
        return None
    pid, order = target
    rows = await db.execute(
        select(LongitudinalEncounter.encounter_id)
        .where(LongitudinalEncounter.patient_id == pid)
        .where(LongitudinalEncounter.encounter_order <= order))
    return {r[0] for r in rows.all()}


def filter_future_encounters(obj: Any, allowed: set[int]) -> Any:
    """Recursively drop anything tied to an encounter outside `allowed`: encounter records
    (`encounter_id`), sections that carry one, and FHIR resources whose `encounter.reference`
    points past the cutoff."""
    if isinstance(obj, list):
        return [filter_future_encounters(i, allowed) for i in obj if _visible(i, allowed)]
    if isinstance(obj, dict):
        if not _visible(obj, allowed):   # a single encounter (or section) past the cutoff
            return {"error": "This encounter is after the task's index encounter and is not accessible in this episode."}
        return {k: filter_future_encounters(v, allowed) for k, v in obj.items()}
    return obj


def _visible(item: Any, allowed: set[int]) -> bool:
    if not isinstance(item, dict):
        return True
    eid = item.get("encounter_id")
    if eid is None and isinstance(item.get("id"), int) and item.get("resourceType") == "Encounter":
        eid = item["id"]
    if eid is None:
        ref = item.get("encounter")
        if isinstance(ref, dict) and isinstance(ref.get("reference"), str) and ref["reference"].startswith("Encounter/"):
            eid = ref["reference"].split("/", 1)[1]
    if eid is None:
        res = item.get("resource")
        if isinstance(res, dict):  # FHIR bundle entry
            return _visible(res, allowed)
        return True
    try:
        return int(eid) in allowed
    except (TypeError, ValueError):
        return True


def apply(obj: Any, allowed: set[int] | None) -> Any:
    """Every rule that applies to a rendered result: hide outcome sections, then the cutoff."""
    obj = strip_outcome_sections(obj)
    return filter_future_encounters(obj, allowed) if allowed is not None else obj
