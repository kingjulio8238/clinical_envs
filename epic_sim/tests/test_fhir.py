"""Tests for FHIR R4 endpoints."""

import pytest
from httpx import AsyncClient


pytestmark = pytest.mark.asyncio(loop_scope="session")

# Patient IDs start at 1672 based on our data
KNOWN_PATIENT_ID = 1672


async def test_patient_search(client: AsyncClient, attending_token: str):
    """Patient search returns a Bundle with correct total."""
    headers = {"Authorization": f"Bearer {attending_token}"}
    resp = await client.get("/fhir/Patient?_count=5", headers=headers)
    assert resp.status_code == 200
    data = resp.json()
    assert data["resourceType"] == "Bundle"
    assert data["type"] == "searchset"
    assert data["total"] == 1268  # Known count from migration
    assert len(data["entry"]) == 5


async def test_patient_read(client: AsyncClient, attending_token: str):
    """Read a specific patient."""
    headers = {"Authorization": f"Bearer {attending_token}"}
    resp = await client.get(f"/fhir/Patient/{KNOWN_PATIENT_ID}", headers=headers)
    assert resp.status_code == 200
    data = resp.json()
    assert data["resourceType"] == "Patient"
    assert data["id"] == str(KNOWN_PATIENT_ID)
    assert len(data["identifier"]) > 0
    assert data["identifier"][0]["value"] == str(KNOWN_PATIENT_ID)


async def test_patient_not_found(client: AsyncClient, attending_token: str):
    headers = {"Authorization": f"Bearer {attending_token}"}
    resp = await client.get("/fhir/Patient/999999", headers=headers)
    assert resp.status_code == 404


async def test_encounter_search(client: AsyncClient, attending_token: str):
    """Encounter search for a patient returns encounters."""
    headers = {"Authorization": f"Bearer {attending_token}"}
    resp = await client.get(f"/fhir/Encounter?patient={KNOWN_PATIENT_ID}&_count=50", headers=headers)
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] > 0
    # Check encounter structure
    entry = data["entry"][0]["resource"]
    assert entry["resourceType"] == "Encounter"
    assert entry["status"] == "finished"
    assert "class" in entry
    assert "subject" in entry
    assert entry["subject"]["reference"] == f"Patient/{KNOWN_PATIENT_ID}"


async def test_encounter_date_filter(client: AsyncClient, attending_token: str):
    """Encounter search with date filter."""
    headers = {"Authorization": f"Bearer {attending_token}"}
    resp = await client.get(
        f"/fhir/Encounter?patient={KNOWN_PATIENT_ID}&date=ge2020-06-01",
        headers=headers,
    )
    assert resp.status_code == 200
    data = resp.json()
    # All returned encounters should be on or after 2020-06-01
    for entry in data["entry"]:
        period = entry["resource"].get("period", {})
        assert period.get("start", "9999") >= "2020-06-01"


async def test_condition_search(client: AsyncClient, attending_token: str):
    """Condition search returns diagnoses for a patient."""
    headers = {"Authorization": f"Bearer {attending_token}"}
    resp = await client.get(f"/fhir/Condition?patient={KNOWN_PATIENT_ID}", headers=headers)
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] > 0
    # Check condition structure
    cond = data["entry"][0]["resource"]
    assert cond["resourceType"] == "Condition"
    assert "code" in cond
    assert "clinicalStatus" in cond


async def test_condition_requires_patient(client: AsyncClient, attending_token: str):
    """Condition search requires patient parameter."""
    headers = {"Authorization": f"Bearer {attending_token}"}
    resp = await client.get("/fhir/Condition", headers=headers)
    assert resp.status_code == 400


async def test_observation_search(client: AsyncClient, attending_token: str):
    """Observation search returns clinical findings."""
    headers = {"Authorization": f"Bearer {attending_token}"}
    resp = await client.get(f"/fhir/Observation?patient={KNOWN_PATIENT_ID}&_count=5", headers=headers)
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] > 0
    obs = data["entry"][0]["resource"]
    assert obs["resourceType"] == "Observation"
    assert "category" in obs
    assert "code" in obs


async def test_document_reference_search(client: AsyncClient, attending_token: str):
    """DocumentReference search returns EHR sections."""
    headers = {"Authorization": f"Bearer {attending_token}"}
    resp = await client.get(f"/fhir/DocumentReference?patient={KNOWN_PATIENT_ID}&_count=5", headers=headers)
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] > 0
    doc = data["entry"][0]["resource"]
    assert doc["resourceType"] == "DocumentReference"
    assert doc["status"] == "current"


async def test_service_request_search(client: AsyncClient, attending_token: str):
    """ServiceRequest search returns imaging orders."""
    headers = {"Authorization": f"Bearer {attending_token}"}
    resp = await client.get(f"/fhir/ServiceRequest?patient={KNOWN_PATIENT_ID}", headers=headers)
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] > 0
    sr = data["entry"][0]["resource"]
    assert sr["resourceType"] == "ServiceRequest"
    assert sr["intent"] == "order"


async def test_medication_request_search(client: AsyncClient, attending_token: str):
    """MedicationRequest search returns medication sections."""
    headers = {"Authorization": f"Bearer {attending_token}"}
    resp = await client.get(f"/fhir/MedicationRequest?patient={KNOWN_PATIENT_ID}", headers=headers)
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] > 0


async def test_allergy_intolerance_search(client: AsyncClient, attending_token: str):
    """AllergyIntolerance search returns allergy sections."""
    headers = {"Authorization": f"Bearer {attending_token}"}
    resp = await client.get(f"/fhir/AllergyIntolerance?patient={KNOWN_PATIENT_ID}", headers=headers)
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] > 0


async def test_pagination(client: AsyncClient, attending_token: str):
    """Pagination works correctly via _count and _offset."""
    headers = {"Authorization": f"Bearer {attending_token}"}

    # Page 1
    resp1 = await client.get("/fhir/Patient?_count=3&_offset=0", headers=headers)
    assert resp1.status_code == 200
    page1 = resp1.json()
    assert len(page1["entry"]) == 3

    # Page 2 should have different patients
    resp2 = await client.get("/fhir/Patient?_count=3&_offset=3", headers=headers)
    assert resp2.status_code == 200
    page2 = resp2.json()
    assert len(page2["entry"]) == 3

    # IDs should be different
    ids1 = {e["resource"]["id"] for e in page1["entry"]}
    ids2 = {e["resource"]["id"] for e in page2["entry"]}
    assert ids1.isdisjoint(ids2), "Pagination returned duplicate patients"


# ---------------------------------------------------------------------------
# Stage 4b: Observation fidelity and the write API
# ---------------------------------------------------------------------------

import psycopg  # noqa: E402

from epic_sim.app.config import settings  # noqa: E402


def _pg():
    return psycopg.connect(settings.database_url_sync.replace("postgresql+psycopg://", "postgresql://"))


def _measurement_rows(patient_id: int, encounter_ids: set[int] | None = None):
    """(qf.id, encounter_id, present, value_text, finding_type) for the patient's charted measurements."""
    with _pg() as conn:
        encs = conn.execute("SELECT encounter_id, source_question_ids FROM longitudinal_encounters WHERE patient_id = %s",
                            (patient_id,)).fetchall()
        out = []
        for eid, sq in encs:
            if encounter_ids is not None and eid not in encounter_ids:
                continue
            qid = int(str(sq).strip("[]").split(",")[0])
            out += [(r[0], eid, r[1], r[2], r[3]) for r in conn.execute(
                "SELECT qf.id, qf.present, qf.value_text, cf.finding_type::text FROM question_findings qf "
                "JOIN clinical_findings cf USING (finding_id) WHERE qf.question_id = %s", (qid,)).fetchall()]
        return out


async def _all_observations(client: AsyncClient, headers: dict, patient_id: int, **params) -> tuple[int, list[dict]]:
    entries, offset, total = [], 0, None
    while True:
        qs = "&".join(f"{k}={v}" for k, v in params.items())
        resp = await client.get(f"/fhir/Observation?patient={patient_id}&_count=100&_offset={offset}" + (f"&{qs}" if qs else ""),
                                headers=headers)
        assert resp.status_code == 200, resp.text
        data = resp.json()
        total = data["total"]
        entries += [e["resource"] for e in data.get("entry", [])]
        offset += 100
        if offset >= total:
            return total, entries


async def test_observation_is_one_resource_per_measurement(client: AsyncClient, attending_token: str):
    headers = {"Authorization": f"Bearer {attending_token}"}
    rows = _measurement_rows(KNOWN_PATIENT_ID)
    total, entries = await _all_observations(client, headers, KNOWN_PATIENT_ID)
    assert total == len(rows) == len(entries) > 100
    assert {e["id"] for e in entries} == {str(r[0]) for r in rows}                       # ids are question_findings ids
    by_id = {str(r[0]): r for r in rows}
    for e in entries:
        qf_id, eid, present, _, _ = by_id[e["id"]]
        assert e["encounter"]["reference"] == f"Encounter/{eid}"
        assert e["effectiveDateTime"] and e["subject"]["reference"] == f"Patient/{KNOWN_PATIENT_ID}"
        assert e["interpretation"][0]["coding"][0]["code"] == ("POS" if present else "NEG")
    # oldest encounter first
    dates = [e["effectiveDateTime"] for e in entries]
    assert dates == sorted(dates)


async def test_absent_findings_are_negative_not_positive(client: AsyncClient, attending_token: str):
    headers = {"Authorization": f"Bearer {attending_token}"}
    rows = _measurement_rows(KNOWN_PATIENT_ID)
    absent_no_value = next(r for r in rows if r[2] == 0 and not r[3])
    resp = await client.get(f"/fhir/Observation/{absent_no_value[0]}", headers=headers)
    assert resp.status_code == 200
    obs = resp.json()
    assert obs["interpretation"][0]["coding"][0]["code"] == "NEG"
    assert obs["valueCodeableConcept"]["coding"][0]["code"] == "2667000"                 # SNOMED Absent
    assert "valueQuantity" not in obs and "valueString" not in obs


async def test_observation_units_components_and_reference_range(client: AsyncClient, attending_token: str):
    headers = {"Authorization": f"Bearer {attending_token}"}
    with _pg() as conn:
        bp = conn.execute("SELECT qf.id FROM question_findings qf JOIN longitudinal_encounters l ON l.source_question_ids = '[' || qf.question_id || ']' "
                          "WHERE qf.value_text ~ '^[0-9]{2,3}/[0-9]{2,3} mmHg$' LIMIT 1").fetchone()
        lab = conn.execute("SELECT qf.id FROM question_findings qf JOIN longitudinal_encounters l ON l.source_question_ids = '[' || qf.question_id || ']' "
                           "WHERE qf.value_text ~ '^[0-9.]+ mg/dL \\(Normal: [0-9.]+-[0-9.]+\\)$' LIMIT 1").fetchone()
    assert bp and lab, "fixture measurements not found"
    obs = (await client.get(f"/fhir/Observation/{bp[0]}", headers=headers)).json()
    assert "valueQuantity" not in obs
    assert [c["code"]["coding"][0]["code"] for c in obs["component"]] == ["8480-6", "8462-4"]
    assert all(c["valueQuantity"]["code"] == "mm[Hg]" for c in obs["component"])
    obs = (await client.get(f"/fhir/Observation/{lab[0]}", headers=headers)).json()
    q = obs["valueQuantity"]
    assert q["unit"] == "mg/dL" and q["system"] == "http://unitsofmeasure.org" and q["code"] == "mg/dL"
    rr = obs["referenceRange"][0]
    assert rr["low"]["value"] < rr["high"]["value"] and rr["low"]["code"] == "mg/dL"


async def test_observation_search_params(client: AsyncClient, attending_token: str):
    headers = {"Authorization": f"Bearer {attending_token}"}
    rows = _measurement_rows(KNOWN_PATIENT_ID)
    labs = [r for r in rows if r[4] == "lab_value"]
    total, entries = await _all_observations(client, headers, KNOWN_PATIENT_ID, category="laboratory")
    assert total == len(labs) and all(e["category"][0]["coding"][0]["code"] == "laboratory" for e in entries)
    eid = rows[0][1]
    total, entries = await _all_observations(client, headers, KNOWN_PATIENT_ID, encounter=eid)
    assert total == sum(1 for r in rows if r[1] == eid) and all(e["encounter"]["reference"] == f"Encounter/{eid}" for e in entries)
    first_date = min(e["effectiveDateTime"] for _, e in [(0, x) for x in (await _all_observations(client, headers, KNOWN_PATIENT_ID))[1]])
    total, entries = await _all_observations(client, headers, KNOWN_PATIENT_ID, date=f"le{first_date}")
    assert total > 0 and all(e["effectiveDateTime"] <= first_date for e in entries)
    total, entries = await _all_observations(client, headers, KNOWN_PATIENT_ID, code="hemoglobin")
    assert all("hemoglobin" in e["code"]["text"].lower() for e in entries)
    resp = await client.get("/fhir/Observation/999999999", headers=headers)
    assert resp.status_code == 404


async def test_observation_honors_session_cutoff(client: AsyncClient, attending_token: str):
    headers = {"Authorization": f"Bearer {attending_token}"}
    with _pg() as conn:
        row = conn.execute(
            """
            SELECT g.gt_id, g.patient_id, g.encounter_id FROM benchmark_ground_truth g
            JOIN longitudinal_encounters e ON e.encounter_id = g.encounter_id
            WHERE g.task::text = 'patient_diagnosis' AND g.granularity::text = 'encounter' AND g.split::text = 'public'
              AND e.encounter_order < (SELECT max(encounter_order) FROM longitudinal_encounters WHERE patient_id = g.patient_id)
            ORDER BY g.gt_id LIMIT 1
            """).fetchone()
        gt_id, pid, eid = row
        allowed = {r[0] for r in conn.execute(
            "SELECT encounter_id FROM longitudinal_encounters WHERE patient_id = %s AND encounter_order <= "
            "(SELECT encounter_order FROM longitudinal_encounters WHERE encounter_id = %s)", (pid, eid)).fetchall()}
    sess = await client.post("/epic/sessions", headers=headers, json={"gt_id": gt_id})
    assert sess.status_code == 201, sess.text
    sh = {**headers, "X-Session-Id": sess.json()["session_id"]}
    visible = _measurement_rows(pid, allowed)
    everything = _measurement_rows(pid)
    assert len(visible) < len(everything)
    total, entries = await _all_observations(client, sh, pid)
    assert total == len(visible) and {e["id"] for e in entries} == {str(r[0]) for r in visible}
    future = next(r for r in everything if r[1] not in allowed)
    assert (await client.get(f"/fhir/Observation/{future[0]}", headers=sh)).status_code == 404
    assert (await client.get(f"/fhir/Observation/{future[0]}", headers=headers)).status_code == 200   # no session: no cutoff
    # the section-backed searches honor the same cutoff
    for rt in ("DiagnosticReport", "MedicationRequest", "AllergyIntolerance", "ServiceRequest"):
        data = (await client.get(f"/fhir/{rt}?patient={pid}&_count=100", headers=sh)).json()
        for e in data.get("entry", []):
            ref = e["resource"].get("encounter", {}).get("reference", "Encounter/0")
            assert int(ref.split("/")[1]) in allowed, rt


async def test_labs_report_lists_its_observations(client: AsyncClient, attending_token: str):
    headers = {"Authorization": f"Bearer {attending_token}"}
    data = (await client.get(f"/fhir/DiagnosticReport?patient={KNOWN_PATIENT_ID}&category=LAB&_count=100", headers=headers)).json()
    assert data["total"] > 0
    report = data["entry"][0]["resource"]
    assert report["result"], "labs report should reference its lab Observations"
    obs_id = report["result"][0]["reference"].split("/")[1]
    obs = (await client.get(f"/fhir/Observation/{obs_id}", headers=headers)).json()
    assert obs["category"][0]["coding"][0]["code"] == "laboratory"
    assert obs["encounter"]["reference"] == report["encounter"]["reference"]


async def test_capability_statement_declares_create_and_search_params(client: AsyncClient):
    cap = (await client.get("/fhir/metadata")).json()
    obs = next(r for r in cap["rest"][0]["resource"] if r["type"] == "Observation")
    assert {"read", "search-type", "create"} <= {i["code"] for i in obs["interaction"]}
    assert {"patient", "category", "code", "date", "encounter"} <= {p["name"] for p in obs["searchParam"]}


_OBS_BODY = {
    "resourceType": "Observation", "status": "final",
    "category": [{"coding": [{"system": "http://terminology.hl7.org/CodeSystem/observation-category", "code": "vital-signs"}]}],
    "code": {"coding": [{"system": "http://loinc.org", "code": "8310-5", "display": "Body temperature"}], "text": "Body temperature"},
    "subject": {"reference": f"Patient/{KNOWN_PATIENT_ID}"},
    "effectiveDateTime": "2022-05-01",
    "valueQuantity": {"value": 38.4, "unit": "°C", "system": "http://unitsofmeasure.org", "code": "Cel"},
}


async def test_write_round_trip_is_session_scoped(client: AsyncClient, attending_token: str):
    headers = {"Authorization": f"Bearer {attending_token}"}
    s1 = (await client.post("/epic/sessions", headers=headers, json={"patient_id": KNOWN_PATIENT_ID})).json()["session_id"]
    s2 = (await client.post("/epic/sessions", headers=headers, json={"patient_id": KNOWN_PATIENT_ID})).json()["session_id"]
    h1, h2 = {**headers, "X-Session-Id": s1}, {**headers, "X-Session-Id": s2}
    before = (await client.get(f"/fhir/Observation?patient={KNOWN_PATIENT_ID}&category=vital-signs&_count=1", headers=h1)).json()["total"]
    resp = await client.post("/fhir/Observation", headers=h1, json=_OBS_BODY)
    assert resp.status_code == 201, resp.text
    created = resp.json()
    assert created["id"] and created["meta"]["versionId"] == "1"
    assert resp.headers["Location"] == f"/fhir/Observation/{created['id']}"
    # readable and searchable in the same session
    assert (await client.get(f"/fhir/Observation/{created['id']}", headers=h1)).json()["valueQuantity"]["value"] == 38.4
    after = (await client.get(f"/fhir/Observation?patient={KNOWN_PATIENT_ID}&category=vital-signs&_count=1", headers=h1)).json()["total"]
    assert after == before + 1
    # invisible to another session and to a session-less caller
    assert (await client.get(f"/fhir/Observation/{created['id']}", headers=h2)).status_code == 404
    assert (await client.get(f"/fhir/Observation?patient={KNOWN_PATIENT_ID}&category=vital-signs&_count=1", headers=h2)).json()["total"] == before
    assert (await client.get(f"/fhir/Observation/{created['id']}", headers=headers)).status_code == 404
    # the benchmark tables are untouched
    with _pg() as conn:
        assert conn.execute("SELECT count(*) FROM fhir_writes WHERE id = %s", (created["id"],)).fetchone()[0] == 1
        assert conn.execute("SELECT count(*) FROM question_findings WHERE value_numeric = 38.4 AND value_text IS NULL").fetchone()[0] == 0


async def test_write_orders_and_problems(client: AsyncClient, attending_token: str):
    headers = {"Authorization": f"Bearer {attending_token}"}
    sid = (await client.post("/epic/sessions", headers=headers, json={"patient_id": KNOWN_PATIENT_ID})).json()["session_id"]
    h = {**headers, "X-Session-Id": sid}
    sr = {"resourceType": "ServiceRequest", "status": "active", "intent": "order",
          "code": {"coding": [{"system": "http://loinc.org", "code": "24323-8", "display": "Comprehensive metabolic panel"}]},
          "subject": {"reference": f"Patient/{KNOWN_PATIENT_ID}"}}
    resp = await client.post("/fhir/ServiceRequest", headers=h, json=sr)
    assert resp.status_code == 201, resp.text
    found = (await client.get(f"/fhir/ServiceRequest?patient={KNOWN_PATIENT_ID}&status=active&_count=100", headers=h)).json()
    assert resp.json()["id"] in {e["resource"]["id"] for e in found["entry"]}
    assert (await client.get(f"/fhir/ServiceRequest/{resp.json()['id']}", headers=h)).status_code == 200
    mr = {"resourceType": "MedicationRequest", "status": "active", "intent": "order",
          "medicationCodeableConcept": {"text": "Ceftriaxone 1 g IV daily"}, "subject": {"reference": f"Patient/{KNOWN_PATIENT_ID}"}}
    resp = await client.post("/fhir/MedicationRequest", headers=h, json=mr)
    assert resp.status_code == 201, resp.text
    assert (await client.get(f"/fhir/MedicationRequest/{resp.json()['id']}", headers=h)).status_code == 200
    cond = {"resourceType": "Condition", "code": {"text": "Community-acquired pneumonia"},
            "subject": {"reference": f"Patient/{KNOWN_PATIENT_ID}"}}
    resp = await client.post("/fhir/Condition", headers=h, json=cond)
    assert resp.status_code == 201, resp.text
    conds = (await client.get(f"/fhir/Condition?patient={KNOWN_PATIENT_ID}&_count=100", headers=h)).json()
    assert resp.json()["id"] in {e["resource"]["id"] for e in conds["entry"]}
    assert (await client.get(f"/fhir/Condition/{resp.json()['id']}", headers=h)).status_code == 200


async def test_write_validation_and_scopes(client: AsyncClient, attending_token: str, nurse_token: str, lab_tech_token: str):
    headers = {"Authorization": f"Bearer {attending_token}"}
    # scopes: nurse may write Observation, not ServiceRequest; lab tech may write nothing
    nurse = {"Authorization": f"Bearer {nurse_token}"}
    resp = await client.post("/fhir/ServiceRequest", headers=nurse, json={"resourceType": "ServiceRequest", "status": "active",
                             "intent": "order", "code": {"text": "CBC"}, "subject": {"reference": f"Patient/{KNOWN_PATIENT_ID}"}})
    assert resp.status_code == 403 and resp.json()["resourceType"] == "OperationOutcome"
    assert (await client.post("/fhir/Observation", headers=nurse, json=_OBS_BODY)).status_code == 201
    assert (await client.post("/fhir/Observation", headers={"Authorization": f"Bearer {lab_tech_token}"}, json=_OBS_BODY)).status_code == 403
    # validation
    bad = {**_OBS_BODY}; bad.pop("code")
    resp = await client.post("/fhir/Observation", headers=headers, json=bad)
    assert resp.status_code == 400 and "code" in resp.json()["issue"][0]["diagnostics"]
    resp = await client.post("/fhir/Observation", headers=headers, json={**_OBS_BODY, "resourceType": "Condition"})
    assert resp.status_code == 400
    resp = await client.post("/fhir/Observation", headers=headers, json={**_OBS_BODY, "subject": {"reference": "Patient/999999"}})
    assert resp.status_code == 422
    resp = await client.post("/fhir/Observation", headers=headers, json={**_OBS_BODY, "encounter": {"reference": "Encounter/1"}})
    assert resp.status_code == 422                                                   # not this patient's encounter
    resp = await client.post("/fhir/Patient", headers=headers, json={"resourceType": "Patient"})
    assert resp.status_code == 404 and resp.json()["issue"][0]["code"] == "not-supported"
    # a session-less write is visible to its author only
    resp = await client.post("/fhir/Observation", headers=headers, json=_OBS_BODY)
    assert resp.status_code == 201
    assert (await client.get(f"/fhir/Observation/{resp.json()['id']}", headers=headers)).status_code == 200
    assert (await client.get(f"/fhir/Observation/{resp.json()['id']}", headers=nurse)).status_code == 404
