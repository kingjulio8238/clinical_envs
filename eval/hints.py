"""Ontology guidance for the structured (ontology-grounded) prompt strategy.

Until Stage 2.5 the structured strategy injected *graph-derived concepts* into the prompt: the
patient's key findings ("your summary must address each of these findings"), the source questions'
organ systems (the answer's ICD chapters), the encounter's correct diagnosis listed among "likely
differentials", and the pathognomonic findings of the query diagnoses. Those are the labels, or a
function of them; echoing the summarization hints alone scored 0.43 (audit/FINDINGS.md #12, #17).

The strategy now differs from zero-shot only by *how* the model is asked to ground its answer:
which coding systems to use and how to normalize concepts. No patient-specific concept is added.
Each helper returns the same fixed guidance block; the `{structured_hints}` template slot is kept.
"""

from __future__ import annotations

_GUIDANCE = (
    "Ground your answer in standard clinical ontologies. Name diagnoses with the most specific "
    "ICD-10-CM code the record supports (prefer a billable code; use an unspecified '.9' code only "
    "when the record gives no detail), express findings and problems as SNOMED CT concepts where a "
    "standard term exists, and refer to laboratory tests by their LOINC-style analyte name with the "
    "value and unit exactly as documented. Use only what the record itself states.\n\n"
)


def format_diagnosis_hints() -> str:
    """Guidance for patient-diagnosis (and the retired single-vignette diagnosis) prompts."""
    return _GUIDANCE


def format_summarization_hints() -> str:
    return _GUIDANCE


def format_retrieval_hints() -> str:
    return _GUIDANCE


def format_imaging_hints() -> str:
    return _GUIDANCE
