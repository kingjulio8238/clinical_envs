# Stage 4: repair data in place (database) — DONE 2026-09-27

Scope: ROADMAP Stage 4 only. Goal: fix leaks and noise directly in the release DB (and the private overlay).
Done = every box checked, the last Stage-1 xfail (`test_copying_the_problem_list_does_not_solve_diagnosis`)
flipped and un-marked, `audit/scripts/data_checks.py` leak counters at 0, validation filled in.

## Design decisions (fixed before coding)
- Every repair is a deterministic, idempotent script with a `release_info` key; scripts rewrite
  `encounter_ehr_sections.section_text` and rebuild `longitudinal_encounters.note_text` from the sections.
- Point-in-time rule for history: a chart line about diagnosis d or its treatment is visible only in encounters
  dated *after* the encounter at which d is first keyed. Profile conditions that name a keyed diagnosis are removed
  from the profile and from every note; procedures that treat a keyed diagnosis are visible only after it.
- Patient diagnosis becomes **index-encounter diagnosis**: one instance per encounter whose correct diagnosis is
  new for the patient; input = chart up to and including that encounter (no assessment/plan); label = that
  diagnosis. Earlier-keyed diagnoses are chart-neutral for the instance. The 1,268 longitudinal rows are kept but
  superseded (`is_diagnostic = 0`, "problem-list extraction").
- ICD repairs are conservative and recorded per node: validate against CMS FY2025 (downloaded if absent), repair
  header codes to their unspecified/first billable child, flag semantic mismatches (with/without contradictions,
  no shared content token), repair with/without contradictions when a sibling code fits, merge only nodes with
  the same normalized name **and** SNOMED id (or the same code and SNOMED id); provenance columns on `diagnoses`
  + a `diagnosis_merges` table. Merged-away nodes stay as tombstones (`icd10_status = 'merged'`) with their code
  released, so the simulator's `UNIQUE(icd10_code, snomed_id)` and the edge-table uniqueness constraints hold.

Order of the chain (matters: `neutral_extra` and label codes depend on the repaired codes):
`repair_history.py` → `repair_icd_codes.py` → `redefine_patient_diagnosis.py` → `regrade_retrieval_by_content.py --force`.

## Medical / surgical history
- [x] PMH: no note names a diagnosis first keyed at that encounter or later — 4,536 profile-derived problem lines removed; residual 11 patients name a later diagnosis inside source prose (not copyable lines; left)
- [x] HPI: the polished HPI of a first-occurrence encounter does not name its own diagnosis — 0/4,388 (was 210); 203 HPIs masked with "the presenting problem"
- [x] PSH: procedures that treat a keyed diagnosis appear only in encounters after it — 348 procedures hidden; salpingectomy anachronisms 0/12; the 2/6 remaining appendectomy lines follow an earlier appendicitis visit
- [x] `note_text` rebuilt for every edited encounter (2,660); `is_modified` set

## Profiles
- [x] tested (keyed) diagnoses stripped from `profile.chronic_conditions` (564 profiles changed, 1,128 conditions removed, kept under `removed_tested_conditions`); `primary_diagnoses` nulled — profiles naming a tested diagnosis 1/1,268 (was 500)
- [x] `patient_profiles.db` / `.jsonl` regenerated

## Diagnosis dataset
- [x] index-encounter instances materialized (release + overlay for private patients): 4,545 from 5,602 encounters; 1,057 repeat encounters excluded (target was "up to 5,602")
- [x] loaders / env context / prompts / degenerate policies use the chart up to the index encounter (`assemble_index_encounter_ehr`, `POINT_IN_TIME_TASKS`, `chart_text(pid, upto_encounter_id)`, index preamble in `build_patient_intro`)
- [x] earlier-keyed diagnoses chart-neutral per instance (`neutral_extra`, id-based)
- [x] longitudinal rows superseded; counts documented — 4,424 scorable (train 2,115 / heldout 905 / public 693 / private 711); 121 `is_diagnostic = 0` (73 all-non-diagnostic labels, 48 whose only label has no billable code after the CMS repair, which makes them unwinnable under code matching)
- [x] copyability: regex-copy policy 0.036 on the public unit (was 0.843); `test_copying_the_problem_list_does_not_solve_diagnosis` un-marked

## ICD codes
- [x] every code validated against CMS FY2025; `icd10_status` per node (billable 6,195 / uncoded 2,459 / merged 969)
- [x] header / invalid codes repaired where a deterministic child exists (1,226); otherwise flagged (1,544)
- [x] semantic check (with/without, token overlap) stored per node (`icd10_flag`); contradictions repaired via sibling when unambiguous
- [x] duplicates merged (969); references repointed in graph tables (guarded against the simulator's unique constraints) and GT JSON (1,407 release + overlay rows); 30 duplicate retrieval queries superseded
- [x] provenance columns + `diagnosis_merges` table; Alembic `c9d0e1f2a3b4` + ORM `DiagnosisMerge`; `sqlite_to_pg` loads it

## Cross-cutting
- [x] `data_checks.py` leak counters → 0 for HPI self-naming and salpingectomy; audit scripts tolerate stripped private rows; new regression suite `eval/tests/test_data_repairs.py` (13 invariants)
- [x] `eval/floors.json` regenerated for all four splits (public dx floor 0.036, ceiling 1.0; retrieval nDCG@10 floor 0.420; summarization 0.487; specialty 0.184; imaging 0.219)
- [x] DATA_CARD / README / ROADMAP / STATE updated

## Validate
- [x] `pytest eval/tests etl/tests` green (see Validation)
- [x] Postgres reloaded count-exact (9,623 diagnoses, 51,018 question_diagnoses, 51,376 diagnosis_findings, 21,140 ground-truth rows, 239,545 judgments); simulator suite in Docker 84 passed / 8 skipped; `/env` smoke on index-encounter episodes: `view_encounters` returns only encounters ≤ index, a later encounter's detail is refused, the index detail carries no assessment/plan and does not name the label, `get_problem_list` does not name the label, oracle reward 1.0 on 3/3 episodes
- [x] committed and pushed; main synced

## Validation record
- Two live defects surfaced only by the Postgres load and were fixed in the scripts and the data: merged nodes kept
  their (code, SNOMED) pair (UniqueViolation on `uq_diagnoses_icd10_snomed`) and edge repointing used `insert or
  ignore` against a SQLite table with no unique index (57 + 624 duplicate edges; UniqueViolation on
  `uq_qd_question_dx_role`). Lesson recorded: SQLite acceptance tests do not exercise the simulator's constraints;
  the Postgres reload is part of Stage acceptance.
- Oracle ceiling on public was 0.997 before excluding the 2 public (48 overall) instances whose only label lost its
  code; the scorer matches by code, so they were unwinnable and are now `is_diagnostic = 0` with a reason.
