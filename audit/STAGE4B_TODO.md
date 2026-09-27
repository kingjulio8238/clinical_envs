# Stage 4b: FHIR fidelity — DONE 2026-09-27

Scope: ROADMAP Stage 4b only. Goal: make `Observation` trustworthy (audit F§16: the 20,861 absent findings
are served as positive observations with no date, encounter, or units, collapsed to one resource per
finding) and lift the "read-only FHIR" limitation with a bounded write path.
Done = every box checked, simulator suite green in Docker, live smoke recorded, validation filled in.

## Design decisions (fixed before coding)
- **Unit of an Observation = one measurement**: one `question_findings` row (a finding as extracted from one
  encounter's source), `id = <question_findings.id>`. The encounter is the one whose `source_question_ids`
  carries the row's question (exactly one per encounter, verified); `effectiveDateTime` = its date.
- **Presence**: every Observation carries `interpretation` POS/NEG (v3-ObservationInterpretation). A finding
  with no value gets `valueCodeableConcept` SNOMED 52101004 Present / 2667000 Absent; a finding with a value
  keeps it (an absent "elevated ALT" still reports the measured 30 U/L) and NEG says the finding is negative.
- **Values and units** come from `value_text`, parsed deterministically (`epic_sim/app/fhir/units.py`): number
  (thousands separators honoured — the ETL's `value_numeric` dropped them: "250,000/mm³" → 250.0), comparator
  (`<0.04`), unit token mapped to a UCUM code, blood pressure as systolic/diastolic `component`s (LOINC
  8480-6 / 8462-4), a "(normal: 22-29)" parenthetical as `referenceRange`, any remainder kept in `note`.
  Unknown unit tokens are not guessed: the quantity is served without a unit and the raw text in `note`.
- **Point-in-time**: `Observation` search and read honour the session cutoff (`X-Session-Id`) like Encounter
  and DocumentReference; before this the findings of later encounters were readable in point-in-time tasks.
- **Search params**: `patient` (required), `category`, `code` (LOINC/SNOMED `system|code` or display text),
  `date` (ge/le/eq), `encounter`, `_count`, `_offset`. `DiagnosticReport` (labs) lists its lab Observations in
  `result`.
- **Writes** (optional item, done bounded): FHIR `create` for Observation, ServiceRequest, MedicationRequest
  and Condition. Server-assigned id, `meta.versionId/lastUpdated`, `Location` header, OperationOutcome on
  errors. Requires the `patient/<Type>.write` scope (attending, resident: all four; nurse: Observation).
  Stored in a new `fhir_writes` table (never in the benchmark tables); visible to the same `X-Session-Id`
  (or, without a session, the same user), so episodes never see each other's writes. Read back by id and in
  search. No update/delete.

## Observation
- [x] presence flags: interpretation POS/NEG on every resource; Present/Absent value when there is no value
- [x] dates: `effectiveDateTime` and `encounter` reference from the source encounter
- [x] units: UCUM quantity, comparator, BP components, referenceRange, note for the unparsed remainder
- [x] one resource per measurement (`id` = question_findings.id); read by id; nothing served for an unknown id
- [x] session cutoff applied to search and read
- [x] `code`, `date`, `encounter` search params implemented; capability statement updated
- [x] DiagnosticReport labs → `result` references

## Writes
- [x] write scopes (`Observation.write`, `MedicationRequest.write` added; role grants) and `can_write_resource`
- [x] `fhir_writes` table: ORM model + Alembic migration + `sqlite_to_pg` untouched (runtime table)
- [x] POST /fhir/{Observation,ServiceRequest,MedicationRequest,Condition}: validation, 201 + Location, OperationOutcome
- [x] written resources readable by id and merged into search for the same session / user only
- [x] capability statement declares `create`

## Tests (Docker suite)
- [x] units parser unit tests (pure, no DB)
- [x] Observation: total == measurement count for the patient; every entry has encounter, date, interpretation
- [x] absent finding served NEG (and Absent when no value); BP components; UCUM unit; referenceRange
- [x] cutoff: with a diagnosis-instance session, no Observation past the index encounter; read of a later one → 404
- [x] writes: 201 + read back + visible in the same session, invisible to another; 403 without scope; 400 OperationOutcome for a bad body
- [x] existing RBAC tests still pass

## Docs
- [x] README FHIR paragraph; ROADMAP Stage 4b status; STATE next move; memory

## Validate
- [x] `docker compose build app`, migration applied, simulator suite green
- [x] live smoke: patient 1672 Observation totals (178 all / 48 with the gt 68569 session), one absent, one BP, one write round-trip
- [x] `pytest eval/tests etl/tests` still green (no data change expected)
- [x] committed and pushed; main synced

## Validation record
- Simulator suite in Docker: 103 passed, 8 skipped (84 before + 9 parser + 10 FHIR tests). Host: eval + ETL 76 passed, 1 skipped.
- Parser over all 138,777 measurements (0.3 s, 0 errors): 32,630 quantities with a UCUM unit, 4,111 without (pH,
  percentiles, reflex grades, pack-years — not guessed), 2,212 blood pressures as components, 1,034 reference
  ranges, 217 comparators, 62,300 string values, 37,524 presence-only findings.
- Live smoke (patient 1672): 178 Observations without a session, 48 with the gt 68569 session (encounters 6802,
  6797); `Observation/348056` (Fever, absent) → NEG / Absent, 2020-01-15, Encounter/6802; BP 112/76 as LOINC
  8480-6 / 8462-4 components in mm[Hg]; Hemoglobin 12.5 g/dL with UCUM; ServiceRequest create → 201 + Location,
  readable in the session (200), invisible to another caller (404); a create referencing an encounter past the
  cutoff → 422 OperationOutcome.
- Also closed: DiagnosticReport, ServiceRequest, MedicationRequest and AllergyIntolerance searches ignored the
  session cutoff (labs of later encounters were readable in point-in-time tasks); they honour it now.
