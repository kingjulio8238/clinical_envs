"""FHIR R4 Observation resource.

One Observation per *measurement*: a `question_findings` row, i.e. a finding as extracted from one
encounter's source. The resource carries the encounter and its date, the presence flag as an
`interpretation` (POS/NEG), the parsed value with a UCUM unit (blood pressure as components, a stated
normal range as `referenceRange`), and the raw text the parser did not consume in `note`. Findings
extracted as *absent* were served as positive observations before Stage 4b (audit F§16).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import Field

from epic_sim.app.fhir.types import CodeableConcept, Coding, DomainResource, Quantity, Reference
from epic_sim.app.fhir.units import interpretation, parse_value, presence_concept

if TYPE_CHECKING:
    from epic_sim.app.models.ontology import ClinicalFinding
    from epic_sim.app.models.relationships import QuestionFinding

# Map finding_type → FHIR Observation category
_CATEGORY_MAP = {
    "lab_value": ("laboratory", "Laboratory"),
    "vital_sign": ("vital-signs", "Vital Signs"),
    "imaging_finding": ("imaging", "Imaging"),
    "procedure_result": ("procedure", "Procedure"),
    "symptom": ("exam", "Exam"),
    "sign": ("exam", "Exam"),
    "history_item": ("social-history", "Social History"),
    "medication": ("therapy", "Therapy"),
    "demographic": ("social-history", "Social History"),
}

# FHIR category → finding types (search filter)
CATEGORY_TYPES = {
    "laboratory": ["lab_value"],
    "vital-signs": ["vital_sign"],
    "imaging": ["imaging_finding"],
    "procedure": ["procedure_result"],
    "exam": ["symptom", "sign"],
    "social-history": ["history_item", "demographic"],
    "therapy": ["medication"],
}


class Observation(DomainResource):
    resource_type: str = Field("Observation", alias="resourceType")
    status: str = "final"
    category: list[CodeableConcept] = Field(default_factory=list)
    code: CodeableConcept | None = None
    subject: Reference | None = None
    encounter: Reference | None = None
    effective_date_time: str | None = Field(None, alias="effectiveDateTime")
    value_quantity: dict | None = Field(None, alias="valueQuantity")
    value_string: str | None = Field(None, alias="valueString")
    value_codeable_concept: dict | None = Field(None, alias="valueCodeableConcept")
    interpretation: list[dict] = Field(default_factory=list)
    note: list[dict] = Field(default_factory=list)
    reference_range: list[dict] = Field(default_factory=list, alias="referenceRange")
    component: list[dict] = Field(default_factory=list)

    @classmethod
    def from_measurement(
        cls,
        cf: ClinicalFinding,
        qf: QuestionFinding,
        patient_id: int,
        encounter_id: int | None,
        encounter_date: str | None,
    ) -> Observation:
        codings = []
        if getattr(cf, "loinc_code", None):
            codings.append(Coding(system="http://loinc.org", code=cf.loinc_code, display=cf.loinc_desc or cf.display_name))
        if cf.snomed_id:
            codings.append(Coding(system="http://snomed.info/sct", code=cf.snomed_id, display=cf.snomed_desc or cf.display_name))
        code = CodeableConcept(coding=codings, text=cf.display_name)

        finding_type = getattr(cf.finding_type, "value", cf.finding_type) or "symptom"
        cat_code, cat_display = _CATEGORY_MAP.get(finding_type, ("exam", "Exam"))
        category = [CodeableConcept(coding=[Coding(
            system="http://terminology.hl7.org/CodeSystem/observation-category", code=cat_code, display=cat_display)])]

        present = bool(qf.present) if qf.present is not None else True
        parsed = parse_value(qf.value_text, qf.value_numeric)
        value_quantity = parsed.quantity.fhir() if parsed.quantity else None
        value_string = parsed.string
        value_concept = None
        if value_quantity is None and value_string is None and not parsed.components:
            value_concept = presence_concept(present)
        notes = [{"text": parsed.note}] if parsed.note else []

        return cls(
            id=str(qf.id),
            category=category,
            code=code,
            subject=Reference(reference=f"Patient/{patient_id}"),
            encounter=Reference(reference=f"Encounter/{encounter_id}") if encounter_id else None,
            effective_date_time=encounter_date,
            value_quantity=value_quantity,
            value_string=value_string,
            value_codeable_concept=value_concept,
            interpretation=[interpretation(present)],
            note=notes,
            reference_range=parsed.reference_range,
            component=parsed.components,
        )
