"""FHIR R4 endpoints — read and search for all clinical resources.

Implements Epic-style FHIR R4 patterns:
- Bearer token auth on all endpoints
- Role-based section filtering (partial observability)
- Pagination via Bundle.link
- _count, _sort, date range search
"""

from __future__ import annotations

import json
import re
import uuid
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Body, Depends, Header, HTTPException, Query, Response, status
from fastapi.responses import JSONResponse
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from epic_sim.app.auth.oauth2 import CurrentUser, get_current_user
from epic_sim.app.auth.rbac import (
    ROLE_SECTION_ACCESS,
    can_access_resource,
    can_write_resource,
    get_allowed_sections,
)
from epic_sim.app.fhir.capability import CAPABILITY_STATEMENT
from epic_sim.app.fhir.resources.allergy_intolerance import AllergyIntolerance
from epic_sim.app.fhir.resources.condition import Condition
from epic_sim.app.fhir.resources.diagnostic_report import DiagnosticReport
from epic_sim.app.fhir.resources.document_reference import DocumentReference
from epic_sim.app.fhir.resources.encounter import Encounter
from epic_sim.app.fhir.resources.medication_request import MedicationRequest
from epic_sim.app.fhir.resources.observation import CATEGORY_TYPES, Observation
from epic_sim.app.fhir.resources.patient import Patient
from epic_sim.app.fhir.resources.service_request import ServiceRequest
from epic_sim.app.fhir.types import Bundle, BundleEntry, BundleEntrySearch, BundleLink, OperationOutcome, OperationOutcomeIssue, Reference
from epic_sim.app.models.base import get_db
from epic_sim.app.services import session_service, visibility
from epic_sim.app.services.session_service import get_redis
from epic_sim.app.models.benchmark import ImagingOrder
from epic_sim.app.models.fhir_store import FhirWrite
from epic_sim.app.models.longitudinal import (
    EncounterEhrSection,
    LongitudinalEncounter,
    LongitudinalPatient,
)
from epic_sim.app.models.ontology import ClinicalFinding, Diagnosis
from epic_sim.app.models.relationships import QuestionDiagnosis, QuestionFinding

router = APIRouter()


def _require_scope(user: CurrentUser, resource_type: str) -> None:
    """Raise 403 if the user lacks the required scope for a resource type."""
    if not can_access_resource(user.role, resource_type, user.scopes):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Insufficient scope for {resource_type}",
        )


def _make_bundle(
    entries: list[dict],
    total: int,
    base_url: str = "",
    offset: int = 0,
    count: int = 10,
) -> dict:
    """Build a FHIR Bundle with pagination links."""
    links = [BundleLink(relation="self", url=base_url)]
    if offset + count < total:
        links.append(BundleLink(relation="next", url=f"{base_url}&_offset={offset + count}"))
    if offset > 0:
        prev_offset = max(0, offset - count)
        links.append(BundleLink(relation="previous", url=f"{base_url}&_offset={prev_offset}"))

    bundle = Bundle(
        type="searchset",
        total=total,
        link=links,
        entry=[
            BundleEntry(
                resource=e,
                search=BundleEntrySearch(mode="match"),
            )
            for e in entries
        ],
    )
    return bundle.model_dump(by_alias=True, exclude_none=True)


# ── Metadata ─────────────────────────────────────────────────

@router.get("/metadata")
async def capability_statement():
    """Return the FHIR CapabilityStatement (no auth required)."""
    return CAPABILITY_STATEMENT


# ── Patient ──────────────────────────────────────────────────

@router.get("/Patient/{patient_id}")
async def read_patient(
    patient_id: int,
    user: CurrentUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    _require_scope(user, "Patient")
    result = await db.execute(
        select(LongitudinalPatient).where(LongitudinalPatient.patient_id == patient_id)
    )
    patient = result.scalar_one_or_none()
    if patient is None:
        raise HTTPException(status_code=404, detail="Patient not found")
    return Patient.from_db(patient).model_dump(by_alias=True, exclude_none=True)


@router.get("/Patient")
async def search_patient(
    name: str | None = None,
    gender: str | None = None,
    _id: str | None = Query(None, alias="_id"),
    _count: int = Query(10, le=100, alias="_count"),
    _offset: int = Query(0, alias="_offset"),
    user: CurrentUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    _require_scope(user, "Patient")
    q = select(LongitudinalPatient)

    if _id:
        q = q.where(LongitudinalPatient.patient_id == int(_id))
    if gender:
        sex_map = {"male": "M", "female": "F"}
        q = q.where(LongitudinalPatient.sex == sex_map.get(gender, gender))
    if name:
        # Search in profile JSONB for name field
        q = q.where(
            func.cast(LongitudinalPatient.profile["name"], type_=func.text()).ilike(f"%{name}%")
        )

    # Count
    count_q = select(func.count()).select_from(q.subquery())
    total = (await db.execute(count_q)).scalar() or 0

    # Paginate
    q = q.order_by(LongitudinalPatient.patient_id).offset(_offset).limit(_count)
    result = await db.execute(q)
    patients = result.scalars().all()

    entries = [Patient.from_db(p).model_dump(by_alias=True, exclude_none=True) for p in patients]
    return _make_bundle(entries, total, offset=_offset, count=_count)


# ── Encounter ────────────────────────────────────────────────

async def _cutoff(redis, x_session_id: str | None, db: AsyncSession) -> set[int] | None:
    """Point-in-time cutoff for a FHIR caller that identifies its benchmark instance through an
    agent session (`X-Session-Id`, created with a gt_id at POST /sessions). None = no cutoff."""
    if not (redis and x_session_id):
        return None
    session = await session_service.get_session(redis, x_session_id)
    if session is None or getattr(session, "gt_id", None) is None:
        return None
    return await visibility.allowed_encounter_ids(db, session.gt_id)


def _apply_cutoff(bundle: dict, allowed: set[int] | None) -> dict:
    if allowed is None:
        return bundle
    out = visibility.filter_future_encounters(bundle, allowed)
    out["total"] = len(out.get("entry") or [])
    return out


def _question_ids(source_question_ids: str | None) -> list[int]:
    """`longitudinal_encounters.source_question_ids` is a JSON array ("[6235]"); one question per encounter."""
    if not source_question_ids:
        return []
    try:
        parsed = json.loads(source_question_ids)
    except (json.JSONDecodeError, ValueError):
        parsed = [x.strip().strip("[]") for x in source_question_ids.split(",")]
    if not isinstance(parsed, list):
        parsed = [parsed]
    return [int(x) for x in parsed if isinstance(x, int) or str(x).strip().isdigit()]


async def _measurement_encounters(db: AsyncSession, patient: int, allowed: set[int] | None) -> dict[int, tuple[int, str | None]]:
    """question_id -> (encounter_id, encounter_date) for the patient's visible encounters."""
    q = select(LongitudinalEncounter.encounter_id, LongitudinalEncounter.encounter_date,
               LongitudinalEncounter.source_question_ids).where(LongitudinalEncounter.patient_id == patient)
    if allowed is not None:
        q = q.where(LongitudinalEncounter.encounter_id.in_(sorted(allowed)))
    out: dict[int, tuple[int, str | None]] = {}
    for eid, date, sq in (await db.execute(q)).all():
        for qid in _question_ids(sq):
            out.setdefault(qid, (eid, date))
    return out


def _date_ok(value: str | None, param: str | None) -> bool:
    """FHIR date search on an ISO date string: eq (default), ge, le, gt, lt prefixes."""
    if not param:
        return True
    if value is None:
        return False
    for prefix in ("ge", "le", "gt", "lt", "eq", "ne"):
        if param.startswith(prefix):
            bound = param[2:]
            return {"ge": value >= bound, "le": value <= bound, "gt": value > bound, "lt": value < bound,
                    "eq": value.startswith(bound), "ne": not value.startswith(bound)}[prefix]
    return value.startswith(param)


async def _written(db: AsyncSession, resource_type: str, patient: int, user: CurrentUser, x_session_id: str | None) -> list[dict]:
    """Resources created through the write API that this caller may see: the same agent session, or,
    without a session, the same user's session-less writes."""
    q = select(FhirWrite).where(FhirWrite.resource_type == resource_type, FhirWrite.patient_id == patient)
    if x_session_id:
        q = q.where(FhirWrite.session_id == x_session_id)
    else:
        q = q.where(FhirWrite.session_id.is_(None), FhirWrite.user_id == str(user.user_id))
    return [w.resource for w in (await db.execute(q.order_by(FhirWrite.created_at, FhirWrite.id))).scalars().all()]


async def _written_by_id(db: AsyncSession, resource_type: str, rid: str, user: CurrentUser, x_session_id: str | None) -> dict | None:
    w = (await db.execute(select(FhirWrite).where(FhirWrite.id == rid, FhirWrite.resource_type == resource_type))).scalar_one_or_none()
    if w is None:
        return None
    visible = (w.session_id == x_session_id) if x_session_id else (w.session_id is None and w.user_id == str(user.user_id))
    return w.resource if visible else None


def _written_matches(res: dict, category: str | None = None, code: str | None = None, date: str | None = None,
                     encounter: int | None = None) -> bool:
    if category and not any(c.get("code") == category for cc in res.get("category", []) for c in cc.get("coding", [])):
        return False
    if code:
        v = code.rpartition("|")[2].lower()
        cc = res.get("code") or {}
        if v not in {(c.get("code") or "").lower() for c in cc.get("coding", [])} and v not in (cc.get("text") or "").lower():
            return False
    if date and not _date_ok(res.get("effectiveDateTime") or res.get("authoredOn") or res.get("recordedDate"), date):
        return False
    if encounter is not None and (res.get("encounter") or {}).get("reference") != f"Encounter/{encounter}":
        return False
    return True


def _condition_resource(patient: int, problem) -> dict:
    """A Condition for one entry of the chart's documented problem list. The chart lists names, so
    no ICD-10/SNOMED code is attached; codes for the patient's *reference* diagnoses are the label."""
    return {
        "resourceType": "Condition",
        "id": f"{patient}-{problem.problem_id}",
        "clinicalStatus": {"coding": [{"system": "http://terminology.hl7.org/CodeSystem/condition-clinical", "code": "active"}]},
        "verificationStatus": {"coding": [{"system": "http://terminology.hl7.org/CodeSystem/condition-ver-status", "code": "confirmed"}]},
        "category": [{"coding": [{"system": "http://terminology.hl7.org/CodeSystem/condition-category",
                                  "code": "problem-list-item", "display": "Problem List Item"}]}],
        "code": {"text": problem.display_name},
        "subject": {"reference": f"Patient/{patient}"},
        "note": [{"text": "Documented history from the chart's problem list."}],
    }


@router.get("/Encounter/{encounter_id}")
async def read_encounter(
    encounter_id: int,
    user: CurrentUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    redis=Depends(get_redis),
    x_session_id: str | None = Header(default=None),
):
    _require_scope(user, "Encounter")
    result = await db.execute(
        select(LongitudinalEncounter).where(LongitudinalEncounter.encounter_id == encounter_id)
    )
    enc = result.scalar_one_or_none()
    allowed = await _cutoff(redis, x_session_id, db)
    if enc is None or (allowed is not None and encounter_id not in allowed):
        raise HTTPException(status_code=404, detail="Encounter not found")
    return Encounter.from_db(enc).model_dump(by_alias=True, exclude_none=True)


@router.get("/Encounter")
async def search_encounter(
    patient: int | None = None,
    date: str | None = None,
    _count: int = Query(10, le=100, alias="_count"),
    _offset: int = Query(0, alias="_offset"),
    _sort: str | None = Query(None, alias="_sort"),
    user: CurrentUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    redis=Depends(get_redis),
    x_session_id: str | None = Header(default=None),
):
    _require_scope(user, "Encounter")
    allowed = await _cutoff(redis, x_session_id, db)
    q = select(LongitudinalEncounter)
    if allowed is not None:
        q = q.where(LongitudinalEncounter.encounter_id.in_(sorted(allowed)))

    if patient:
        q = q.where(LongitudinalEncounter.patient_id == patient)
    if date:
        # Support ge/le prefixes: date=ge2020-01-01
        if date.startswith("ge"):
            q = q.where(LongitudinalEncounter.encounter_date >= date[2:])
        elif date.startswith("le"):
            q = q.where(LongitudinalEncounter.encounter_date <= date[2:])
        else:
            q = q.where(LongitudinalEncounter.encounter_date == date)

    count_q = select(func.count()).select_from(q.subquery())
    total = (await db.execute(count_q)).scalar() or 0

    # Sort
    if _sort == "-date":
        q = q.order_by(LongitudinalEncounter.encounter_date.desc())
    elif _sort == "date":
        q = q.order_by(LongitudinalEncounter.encounter_date.asc())
    else:
        q = q.order_by(LongitudinalEncounter.encounter_id)

    q = q.offset(_offset).limit(_count)
    result = await db.execute(q)
    encounters = result.scalars().all()

    entries = [Encounter.from_db(e).model_dump(by_alias=True, exclude_none=True) for e in encounters]
    return _make_bundle(entries, total, offset=_offset, count=_count)


# ── Condition ────────────────────────────────────────────────

@router.get("/Condition/{condition_id}")
async def read_condition(
    condition_id: str,
    user: CurrentUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    x_session_id: str | None = Header(default=None),
):
    """One entry of a patient's documented problem list (ids `<patient>-chart-<n>`) or a Condition
    created through the write API in this session."""
    _require_scope(user, "Condition")
    m = re.fullmatch(r"(\d+)-(chart-\d+)", condition_id)
    if m:
        patient = int(m.group(1))
        for p in await visibility.documented_problems(db, patient):
            if p.problem_id == m.group(2):
                return _condition_resource(patient, p)
    written = await _written_by_id(db, "Condition", condition_id, user, x_session_id)
    if written is not None:
        return written
    raise HTTPException(status_code=404, detail="Condition not found")


@router.get("/Condition")
async def search_condition(
    patient: int | None = None,
    category: str | None = None,
    code: str | None = None,
    _count: int = Query(10, le=100, alias="_count"),
    _offset: int = Query(0, alias="_offset"),
    user: CurrentUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    x_session_id: str | None = Header(default=None),
):
    """A patient's conditions: the chart's documented problem list (Stage 2.1), plus the Conditions this
    session created through the write API.

    Until Stage 2.1 this returned the graph-derived diagnoses of the patient's source questions
    (correct role for problem-list-item, every role for encounter-diagnosis) with ICD-10 and SNOMED
    codes, i.e. the patient-diagnosis reference (audit/FINDINGS.md #10). Coded reference diagnoses
    are labels and are not served through any API.
    """
    _require_scope(user, "Condition")
    if patient is None:
        raise HTTPException(status_code=400, detail="'patient' parameter is required")
    problems = await visibility.documented_problems(db, patient)
    if code:
        needle = code.split("|", 1)[-1].lower()
        problems = [p for p in problems if needle in p.display_name.lower()]
    entries = [_condition_resource(patient, p) for p in problems]
    entries += [w for w in await _written(db, "Condition", patient, user, x_session_id) if _written_matches(w, code=code)]
    return _make_bundle(entries[_offset:_offset + _count], len(entries), offset=_offset, count=_count)


# ── Observation ──────────────────────────────────────────────

def _code_matches(cf: ClinicalFinding, code: str) -> bool:
    system, _, value = code.rpartition("|")
    v = value.lower()
    if system == "http://loinc.org":
        return (cf.loinc_code or "").lower() == v
    if system == "http://snomed.info/sct":
        return (cf.snomed_id or "").lower() == v
    return v in {(cf.loinc_code or "").lower(), (cf.snomed_id or "").lower()} or v in (cf.display_name or "").lower()


def _dump(resource) -> dict:
    return resource.model_dump(by_alias=True, exclude_none=True)


@router.get("/Observation/{observation_id}")
async def read_observation(
    observation_id: str,
    user: CurrentUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    redis=Depends(get_redis),
    x_session_id: str | None = Header(default=None),
):
    """One measurement (`id` = question_findings.id) of a charted encounter, or an Observation this
    session created. Measurements of encounters past the session's cutoff are not found."""
    _require_scope(user, "Observation")
    if observation_id.isdigit():
        row = (await db.execute(
            select(ClinicalFinding, QuestionFinding)
            .join(QuestionFinding, ClinicalFinding.finding_id == QuestionFinding.finding_id)
            .where(QuestionFinding.id == int(observation_id)))).first()
        if row is not None:
            cf, qf = row
            candidates = (await db.execute(
                select(LongitudinalEncounter)
                .where(LongitudinalEncounter.source_question_ids.like(f"%{qf.question_id}%")))).scalars().all()
            enc = next((e for e in candidates if qf.question_id in _question_ids(e.source_question_ids)), None)
            if enc is not None:
                allowed = await _cutoff(redis, x_session_id, db)
                if allowed is None or enc.encounter_id in allowed:
                    return _dump(Observation.from_measurement(cf, qf, enc.patient_id, enc.encounter_id, enc.encounter_date))
        raise HTTPException(status_code=404, detail="Observation not found")
    written = await _written_by_id(db, "Observation", observation_id, user, x_session_id)
    if written is not None:
        return written
    raise HTTPException(status_code=404, detail="Observation not found")


@router.get("/Observation")
async def search_observation(
    patient: int | None = None,
    category: str | None = None,
    code: str | None = None,
    date: str | None = None,
    encounter: int | None = None,
    _count: int = Query(10, le=100, alias="_count"),
    _offset: int = Query(0, alias="_offset"),
    user: CurrentUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    redis=Depends(get_redis),
    x_session_id: str | None = Header(default=None),
):
    """A patient's measurements, one Observation per finding per encounter, oldest encounter first,
    followed by the Observations this session created. Honors the session's point-in-time cutoff."""
    _require_scope(user, "Observation")
    if patient is None:
        raise HTTPException(status_code=400, detail="'patient' parameter is required")

    allowed = await _cutoff(redis, x_session_id, db)
    encs = await _measurement_encounters(db, patient, allowed)
    if encounter is not None:
        encs = {q: v for q, v in encs.items() if v[0] == encounter}
    if date:
        encs = {q: v for q, v in encs.items() if _date_ok(v[1], date)}

    entries: list[tuple[tuple, dict]] = []
    if encs:
        q = (select(ClinicalFinding, QuestionFinding)
             .join(QuestionFinding, ClinicalFinding.finding_id == QuestionFinding.finding_id)
             .where(QuestionFinding.question_id.in_(sorted(encs))))
        if category:
            types = CATEGORY_TYPES.get(category)
            q = q.where(ClinicalFinding.finding_type.in_(types)) if types else q.where(False)
        for cf, qf in (await db.execute(q)).all():
            if code and not _code_matches(cf, code):
                continue
            eid, edate = encs[qf.question_id]
            entries.append(((edate or "", eid, qf.id), _dump(Observation.from_measurement(cf, qf, patient, eid, edate))))
    entries.sort(key=lambda t: t[0])
    resources = [r for _, r in entries]
    resources += [w for w in await _written(db, "Observation", patient, user, x_session_id)
                  if _written_matches(w, category, code, date, encounter)]
    return _make_bundle(resources[_offset:_offset + _count], len(resources), offset=_offset, count=_count)


# ── DiagnosticReport ─────────────────────────────────────────

@router.get("/DiagnosticReport")
async def search_diagnostic_report(
    patient: int | None = None,
    category: str | None = None,
    _count: int = Query(10, le=100, alias="_count"),
    _offset: int = Query(0, alias="_offset"),
    user: CurrentUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    redis=Depends(get_redis),
    x_session_id: str | None = Header(default=None),
):
    """Imaging / pathology / labs sections as reports; a labs report lists its lab Observations in
    `result`. Honors role section access and the session's point-in-time cutoff."""
    _require_scope(user, "DiagnosticReport")

    if patient is None:
        raise HTTPException(status_code=400, detail="'patient' parameter is required")

    allowed = get_allowed_sections(user.role)
    report_types = {"imaging", "pathology", "labs"}
    if allowed is not None:
        report_types = report_types & allowed
    if category:
        cat_map = {"RAD": "imaging", "SP": "pathology", "LAB": "labs"}
        report_types = report_types & {cat_map.get(category, category)}
    if not report_types:
        return _make_bundle([], 0, offset=_offset, count=_count)

    q = (
        select(EncounterEhrSection, LongitudinalEncounter.encounter_date, LongitudinalEncounter.source_question_ids)
        .join(LongitudinalEncounter, EncounterEhrSection.encounter_id == LongitudinalEncounter.encounter_id)
        .where(LongitudinalEncounter.patient_id == patient)
        .where(EncounterEhrSection.section_type.in_(report_types))
    )
    cutoff = await _cutoff(redis, x_session_id, db)
    if cutoff is not None:
        q = q.where(EncounterEhrSection.encounter_id.in_(sorted(cutoff)))

    total = (await db.execute(select(func.count()).select_from(q.subquery()))).scalar() or 0
    rows = (await db.execute(q.order_by(EncounterEhrSection.id).offset(_offset).limit(_count))).all()

    # lab Observations per encounter on this page, for DiagnosticReport.result
    q_by_enc = {sec.encounter_id: _question_ids(sq) for sec, _, sq in rows if sec.section_type == "labs"}
    labs_by_q: dict[int, list[int]] = {}
    qids = sorted({qid for v in q_by_enc.values() for qid in v})
    if qids:
        lab_rows = await db.execute(
            select(QuestionFinding.id, QuestionFinding.question_id)
            .join(ClinicalFinding, ClinicalFinding.finding_id == QuestionFinding.finding_id)
            .where(QuestionFinding.question_id.in_(qids), ClinicalFinding.finding_type.in_(["lab_value"]))
            .order_by(QuestionFinding.id))
        for oid, qid in lab_rows.all():
            labs_by_q.setdefault(qid, []).append(oid)

    entries = []
    for section, enc_date, _ in rows:
        report = DiagnosticReport.from_ehr_section(section, patient_id=patient, encounter_date=enc_date,
                                                   section_type=section.section_type)
        if section.section_type == "labs":
            report.result = [Reference(reference=f"Observation/{oid}")
                             for qid in q_by_enc.get(section.encounter_id, []) for oid in labs_by_q.get(qid, [])]
        entries.append(_dump(report))
    return _make_bundle(entries, total, offset=_offset, count=_count)


# ── ServiceRequest ───────────────────────────────────────────

@router.get("/ServiceRequest/{request_id}")
async def read_service_request(
    request_id: str,
    user: CurrentUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    redis=Depends(get_redis),
    x_session_id: str | None = Header(default=None),
):
    _require_scope(user, "ServiceRequest")
    if request_id.isdigit():
        order = (await db.execute(select(ImagingOrder).where(ImagingOrder.order_id == int(request_id)))).scalar_one_or_none()
        if order is not None:
            enc = (await db.execute(select(LongitudinalEncounter.patient_id)
                                    .where(LongitudinalEncounter.encounter_id == order.encounter_id))).scalar_one()
            allowed = await _cutoff(redis, x_session_id, db)
            if allowed is None or order.encounter_id in allowed:
                return _dump(ServiceRequest.from_db(order, patient_id=enc))
        raise HTTPException(status_code=404, detail="ServiceRequest not found")
    written = await _written_by_id(db, "ServiceRequest", request_id, user, x_session_id)
    if written is not None:
        return written
    raise HTTPException(status_code=404, detail="ServiceRequest not found")


@router.get("/ServiceRequest")
async def search_service_request(
    patient: int | None = None,
    status_param: str | None = Query(None, alias="status"),
    _count: int = Query(10, le=100, alias="_count"),
    _offset: int = Query(0, alias="_offset"),
    user: CurrentUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    redis=Depends(get_redis),
    x_session_id: str | None = Header(default=None),
):
    _require_scope(user, "ServiceRequest")

    if patient is None:
        raise HTTPException(status_code=400, detail="'patient' parameter is required")

    q = (
        select(ImagingOrder)
        .join(LongitudinalEncounter, ImagingOrder.encounter_id == LongitudinalEncounter.encounter_id)
        .where(LongitudinalEncounter.patient_id == patient)
    )
    cutoff = await _cutoff(redis, x_session_id, db)
    if cutoff is not None:
        q = q.where(ImagingOrder.encounter_id.in_(sorted(cutoff)))
    orders = (await db.execute(q.order_by(ImagingOrder.order_id))).scalars().all()
    entries = [_dump(ServiceRequest.from_db(o, patient_id=patient)) for o in orders]
    entries += await _written(db, "ServiceRequest", patient, user, x_session_id)
    if status_param:
        entries = [e for e in entries if e.get("status") == status_param]
    return _make_bundle(entries[_offset:_offset + _count], len(entries), offset=_offset, count=_count)


# ── DocumentReference ────────────────────────────────────────

@router.get("/DocumentReference")
async def search_document_reference(
    patient: int | None = None,
    type_param: str | None = Query(None, alias="type"),
    _count: int = Query(10, le=100, alias="_count"),
    _offset: int = Query(0, alias="_offset"),
    user: CurrentUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    redis=Depends(get_redis),
    x_session_id: str | None = Header(default=None),
):
    """Search DocumentReferences (EHR sections) for a patient.

    Applies role-based section filtering (partial observability), hides the outcome sections
    (assessment/plan) for every role, and honors the session's point-in-time cutoff.
    """
    _require_scope(user, "DocumentReference")

    if patient is None:
        raise HTTPException(status_code=400, detail="'patient' parameter is required")

    # Role-based section filtering
    allowed = get_allowed_sections(user.role)

    q = (
        select(EncounterEhrSection, LongitudinalEncounter.encounter_date)
        .join(LongitudinalEncounter, EncounterEhrSection.encounter_id == LongitudinalEncounter.encounter_id)
        .where(LongitudinalEncounter.patient_id == patient)
    )

    if allowed is not None:
        q = q.where(EncounterEhrSection.section_type.in_(allowed))
    if visibility.hidden_sections():
        q = q.where(EncounterEhrSection.section_type.not_in(sorted(visibility.hidden_sections())))
    cutoff = await _cutoff(redis, x_session_id, db)
    if cutoff is not None:
        q = q.where(EncounterEhrSection.encounter_id.in_(sorted(cutoff)))

    if type_param:
        q = q.where(EncounterEhrSection.section_type == type_param)

    count_subq = select(func.count()).select_from(q.subquery())
    total = (await db.execute(count_subq)).scalar() or 0

    q = q.order_by(EncounterEhrSection.encounter_id, EncounterEhrSection.section_order)
    q = q.offset(_offset).limit(_count)
    result = await db.execute(q)

    entries = []
    for section, enc_date in result.all():
        doc = DocumentReference.from_ehr_section(section, patient_id=patient, encounter_date=enc_date)
        entries.append(doc.model_dump(by_alias=True, exclude_none=True))

    return _make_bundle(entries, total, offset=_offset, count=_count)


# ── MedicationRequest / AllergyIntolerance (section-backed) ──

async def _section_resources(db, patient: int, section_type: str, cutoff: set[int] | None, factory) -> list[dict]:
    q = (
        select(EncounterEhrSection, LongitudinalEncounter.encounter_date)
        .join(LongitudinalEncounter, EncounterEhrSection.encounter_id == LongitudinalEncounter.encounter_id)
        .where(LongitudinalEncounter.patient_id == patient)
        .where(EncounterEhrSection.section_type == section_type)
    )
    if cutoff is not None:
        q = q.where(EncounterEhrSection.encounter_id.in_(sorted(cutoff)))
    rows = (await db.execute(q.order_by(EncounterEhrSection.encounter_id))).all()
    return [_dump(factory(section, patient_id=patient, encounter_date=enc_date)) for section, enc_date in rows]


@router.get("/MedicationRequest/{request_id}")
async def read_medication_request(
    request_id: str,
    user: CurrentUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    x_session_id: str | None = Header(default=None),
):
    _require_scope(user, "MedicationRequest")
    written = await _written_by_id(db, "MedicationRequest", request_id, user, x_session_id)
    if written is not None:
        return written
    raise HTTPException(status_code=404, detail="MedicationRequest not found")


@router.get("/MedicationRequest")
async def search_medication_request(
    patient: int | None = None,
    _count: int = Query(10, le=100, alias="_count"),
    _offset: int = Query(0, alias="_offset"),
    user: CurrentUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    redis=Depends(get_redis),
    x_session_id: str | None = Header(default=None),
):
    _require_scope(user, "MedicationRequest")
    if patient is None:
        raise HTTPException(status_code=400, detail="'patient' parameter is required")
    cutoff = await _cutoff(redis, x_session_id, db)
    entries = await _section_resources(db, patient, "medications", cutoff, MedicationRequest.from_ehr_section)
    entries += await _written(db, "MedicationRequest", patient, user, x_session_id)
    return _make_bundle(entries[_offset:_offset + _count], len(entries), offset=_offset, count=_count)


@router.get("/AllergyIntolerance")
async def search_allergy_intolerance(
    patient: int | None = None,
    _count: int = Query(10, le=100, alias="_count"),
    _offset: int = Query(0, alias="_offset"),
    user: CurrentUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    redis=Depends(get_redis),
    x_session_id: str | None = Header(default=None),
):
    _require_scope(user, "AllergyIntolerance")
    if patient is None:
        raise HTTPException(status_code=400, detail="'patient' parameter is required")
    cutoff = await _cutoff(redis, x_session_id, db)
    entries = await _section_resources(db, patient, "allergies", cutoff, AllergyIntolerance.from_ehr_section)
    return _make_bundle(entries[_offset:_offset + _count], len(entries), offset=_offset, count=_count)


# ── Writes (FHIR create) ─────────────────────────────────────

_WRITABLE: dict[str, set[str]] = {
    "Observation": {"status", "code"},
    "ServiceRequest": {"status", "intent", "code"},
    "MedicationRequest": {"status", "intent"},
    "Condition": {"code"},
}


def _outcome(status_code: int, code: str, diagnostics: str) -> JSONResponse:
    body = OperationOutcome(issue=[OperationOutcomeIssue(severity="error", code=code, diagnostics=diagnostics)])
    return JSONResponse(status_code=status_code, content=body.model_dump(by_alias=True, exclude_none=True))


@router.post("/{resource_type}", status_code=201)
async def create_resource(
    resource_type: str,
    response: Response,
    body: dict[str, Any] = Body(...),
    user: CurrentUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    redis=Depends(get_redis),
    x_session_id: str | None = Header(default=None),
):
    """FHIR `create` for Observation, ServiceRequest, MedicationRequest and Condition (Stage 4b).

    The server assigns the id and `meta`, returns the stored resource with a `Location` header, and
    answers errors with an OperationOutcome. Writes land in `fhir_writes`, never in the benchmark tables,
    and are visible only to the agent session that made them (`X-Session-Id`), or to the same user when
    no session is given. There is no update or delete.
    """
    if resource_type not in _WRITABLE:
        return _outcome(404, "not-supported", f"create is not supported for {resource_type}")
    if not can_write_resource(user.role, resource_type, user.scopes):
        return _outcome(403, "forbidden", f"Insufficient scope for {resource_type} create")
    if not isinstance(body, dict) or body.get("resourceType") != resource_type:
        return _outcome(400, "structure", f"body must be a {resource_type} resource (resourceType mismatch)")
    missing = sorted(f for f in _WRITABLE[resource_type] if not body.get(f))
    if resource_type == "MedicationRequest" and not (body.get("medicationCodeableConcept") or body.get("medicationReference")):
        missing.append("medication[x]")
    if missing:
        return _outcome(400, "required", f"missing required element(s): {', '.join(missing)}")
    if "id" in body or "meta" in body:
        return _outcome(400, "invalid", "id and meta are server-assigned; omit them")
    m = re.fullmatch(r"Patient/(\d+)", str((body.get("subject") or {}).get("reference", "")))
    if not m:
        return _outcome(400, "required", "subject.reference must be 'Patient/<id>'")
    patient = int(m.group(1))
    if (await db.execute(select(LongitudinalPatient.patient_id).where(LongitudinalPatient.patient_id == patient))).scalar_one_or_none() is None:
        return _outcome(422, "not-found", f"Patient/{patient} does not exist")
    if x_session_id:
        session = await session_service.get_session(redis, x_session_id) if redis else None
        if session is None:
            return _outcome(422, "not-found", "X-Session-Id does not name an active session")
        if session.patient_id not in (None, patient):
            return _outcome(422, "business-rule", "the session is bound to a different patient")
    enc_ref = (body.get("encounter") or {}).get("reference")
    if enc_ref is not None:
        em = re.fullmatch(r"Encounter/(\d+)", str(enc_ref))
        if not em:
            return _outcome(400, "value", "encounter.reference must be 'Encounter/<id>'")
        eid = int(em.group(1))
        owner = (await db.execute(select(LongitudinalEncounter.patient_id).where(LongitudinalEncounter.encounter_id == eid))).scalar_one_or_none()
        allowed = await _cutoff(redis, x_session_id, db)
        if owner != patient or (allowed is not None and eid not in allowed):
            return _outcome(422, "business-rule", f"Encounter/{eid} is not an accessible encounter of Patient/{patient}")

    rid = str(uuid.uuid4())
    resource = dict(body)
    resource["id"] = rid
    resource["meta"] = {"versionId": "1", "lastUpdated": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    db.add(FhirWrite(id=rid, resource_type=resource_type, patient_id=patient, session_id=x_session_id or None,
                     user_id=str(user.user_id), resource=resource))
    await db.commit()
    response.headers["Location"] = f"/fhir/{resource_type}/{rid}"
    response.headers["ETag"] = 'W/"1"'
    return resource
