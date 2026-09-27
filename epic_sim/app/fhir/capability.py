"""FHIR R4 CapabilityStatement for the Epic EHR simulation."""

from epic_sim.app.config import settings

CAPABILITY_STATEMENT = {
    "resourceType": "CapabilityStatement",
    "id": "epic-sim",
    "status": "active",
    "date": "2026-09-27",
    "kind": "instance",
    "fhirVersion": "4.0.1",
    "format": ["json"],
    "implementation": {
        "description": "Epic EHR Simulation — FHIR R4 Server for Clinical AI Evaluation",
        "url": settings.fhir_base_url,
    },
    "rest": [
        {
            "mode": "server",
            "security": {
                "service": [
                    {
                        "coding": [
                            {
                                "system": "http://terminology.hl7.org/CodeSystem/restful-security-service",
                                "code": "SMART-on-FHIR",
                            }
                        ]
                    }
                ],
                "description": "OAuth2 bearer token required for all clinical endpoints",
            },
            "resource": [
                {
                    "type": "Patient",
                    "interaction": [
                        {"code": "read"},
                        {"code": "search-type"},
                    ],
                    "searchParam": [
                        {"name": "_id", "type": "token"},
                        {"name": "name", "type": "string"},
                        {"name": "gender", "type": "token"},
                        {"name": "_count", "type": "number"},
                    ],
                },
                {
                    "type": "Encounter",
                    "interaction": [
                        {"code": "read"},
                        {"code": "search-type"},
                    ],
                    "searchParam": [
                        {"name": "patient", "type": "reference"},
                        {"name": "date", "type": "date"},
                        {"name": "class", "type": "token"},
                        {"name": "_count", "type": "number"},
                        {"name": "_sort", "type": "string"},
                    ],
                },
                {
                    "type": "Condition",
                    "interaction": [
                        {"code": "read"},
                        {"code": "search-type"},
                        {"code": "create"},
                    ],
                    "searchParam": [
                        {"name": "patient", "type": "reference"},
                        {"name": "category", "type": "token"},
                        {"name": "code", "type": "token"},
                        {"name": "_count", "type": "number"},
                    ],
                },
                {
                    "type": "Observation",
                    "documentation": "One Observation per measurement (a finding as extracted from one encounter). "
                                     "interpretation POS/NEG carries the presence flag; effectiveDateTime and encounter come "
                                     "from the source encounter; quantities carry UCUM units, blood pressure as components, "
                                     "stated normal ranges as referenceRange. Honors the session cutoff (X-Session-Id).",
                    "interaction": [
                        {"code": "read"},
                        {"code": "search-type"},
                        {"code": "create"},
                    ],
                    "searchParam": [
                        {"name": "patient", "type": "reference"},
                        {"name": "category", "type": "token"},
                        {"name": "code", "type": "token"},
                        {"name": "date", "type": "date"},
                        {"name": "encounter", "type": "reference"},
                        {"name": "_count", "type": "number"},
                        {"name": "_offset", "type": "number"},
                    ],
                },
                {
                    "type": "DiagnosticReport",
                    "interaction": [
                        {"code": "read"},
                        {"code": "search-type"},
                    ],
                    "searchParam": [
                        {"name": "patient", "type": "reference"},
                        {"name": "category", "type": "token"},
                        {"name": "_count", "type": "number"},
                    ],
                },
                {
                    "type": "ServiceRequest",
                    "interaction": [
                        {"code": "read"},
                        {"code": "search-type"},
                        {"code": "create"},
                    ],
                    "searchParam": [
                        {"name": "patient", "type": "reference"},
                        {"name": "status", "type": "token"},
                        {"name": "_count", "type": "number"},
                    ],
                },
                {
                    "type": "DocumentReference",
                    "interaction": [
                        {"code": "read"},
                        {"code": "search-type"},
                    ],
                    "searchParam": [
                        {"name": "patient", "type": "reference"},
                        {"name": "type", "type": "token"},
                        {"name": "_count", "type": "number"},
                    ],
                },
                {
                    "type": "MedicationRequest",
                    "interaction": [
                        {"code": "read"},
                        {"code": "search-type"},
                        {"code": "create"},
                    ],
                    "searchParam": [
                        {"name": "patient", "type": "reference"},
                        {"name": "_count", "type": "number"},
                    ],
                },
                {
                    "type": "AllergyIntolerance",
                    "interaction": [
                        {"code": "read"},
                        {"code": "search-type"},
                    ],
                    "searchParam": [
                        {"name": "patient", "type": "reference"},
                        {"name": "_count", "type": "number"},
                    ],
                },
                {
                    "type": "CodeSystem",
                    "interaction": [{"code": "search-type"}],
                    "operation": [
                        {"name": "lookup", "definition": "http://hl7.org/fhir/OperationDefinition/CodeSystem-lookup"},
                    ],
                },
                {
                    "type": "ValueSet",
                    "interaction": [{"code": "search-type"}],
                    "operation": [
                        {"name": "expand", "definition": "http://hl7.org/fhir/OperationDefinition/ValueSet-expand"},
                        {"name": "validate-code", "definition": "http://hl7.org/fhir/OperationDefinition/ValueSet-validate-code"},
                    ],
                },
            ],
        }
    ],
}
