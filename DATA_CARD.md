# Data Card — Synthetic Hospital v1.3

## Summary

1,268 synthetic patients and 5,602 clinical encounters, distributed as SQLite
(`patient_profiles.db`) and JSON Lines (`patient_profiles.jsonl`), plus the benchmark
database (`benchmark_v1.3.db`) holding the chart sections, ground truth, relevance
judgments, imaging orders, knowledge graph, and split labels. The data supports evaluation
of clinical AI on patient diagnosis, context summarization, evidence retrieval, and imaging
indication over multi-visit patient timelines, and training on the released training split.

The records are fully synthetic. They are generated from AI-generated question vignettes
combined with a knowledge graph derived from medical facts (see Provenance).

## Files


| File                     | Format     | Contents                                                          |
| ------------------------ | ---------- | ----------------------------------------------------------------- |
| `patient_profiles.db`    | SQLite 3   | Two tables: `patients`, `encounters`                              |
| `patient_profiles.jsonl` | JSON Lines | One patient object per line; encounters nested under `encounters` |
| `benchmark_v1.3.db`      | SQLite 3   | Benchmark tables (below); load into the simulator with `python -m epic_sim.migrate.sqlite_to_pg` |

## Schema



### `patients` (1,268 rows — one per synthetic patient)


| Column              | Description                                                                                       |
| ------------------- | ------------------------------------------------------------------------------------------------- |
| `patient_id`        | Unique patient identifier                                                                         |
| `profile`           | JSON blob: demographics, chronic conditions, surgical/family history, allergies, home medications |
| `age`               | Age at first encounter                                                                            |
| `sex`               | Patient sex                                                                                       |
| `race_ethnicity`    | Race / ethnicity                                                                                  |
| `insurance`         | Insurance type                                                                                    |
| `num_encounters`    | Number of encounters for this patient                                                             |
| `primary_diagnoses` | JSON array of the patient's primary diagnoses                                                     |
| `comorbidities`     | JSON array of comorbid conditions                                                                 |
| `generation_seed`   | Seed used for reproducible generation                                                             |


> **Note:** a `pcp_name` field exists in the internal database but is **intentionally
> excluded** from this release — it was a non-informative placeholder (`Dr. Smith-<id>`).
> The meaningful provider field is `encounters.attending_name`.



### `encounters` (5,602 rows — one per clinical visit, ~4.4 per patient)


| Column              | Description                                                                       |
| ------------------- | --------------------------------------------------------------------------------- |
| `encounter_id`      | Unique encounter identifier                                                       |
| `patient_id`        | Foreign key to `patients`                                                         |
| `encounter_date`    | Date of the visit                                                                 |
| `encounter_type`    | `outpatient`, `ed`, `inpatient`, `icu`, `telehealth`, `procedure`, or `follow_up` |
| `chief_complaint`   | Presenting complaint                                                              |
| `attending_name`    | Attending provider (synthetic)                                                    |
| `department`        | Clinical department                                                               |
| `encounter_order`   | 0-based sequence position within the patient's timeline                           |
| `note_text`         | Full clinical note for the visit                                                  |
| `generation_method` | `hybrid` (template assembly + LLM prose polish) or `template`                     |


To reconstruct a patient's full record, join on `patient_id` and order by `encounter_order`.
The JSONL already nests encounters in that order.


### `benchmark_v1.3.db`

| Table | Rows | Contents |
|---|---|---|
| `longitudinal_patients` | 1,268 | Patient profiles (same fields as `patients` above) |
| `longitudinal_encounters` | 5,602 | Encounters with `note_text`; `source_question_ids` are opaque integer keys into the annotation tables |
| `encounter_ehr_sections` | 59,964 | The note split into typed sections (`hpi`, `pmh`, `medications`, `labs`, ...) — the passage unit for retrieval. `search_vector` ships NULL (PostgreSQL recomputes it on load). The simulator never serves `assessment` / `plan` |
| `benchmark_ground_truth` | 16,595 (15,527 scorable) | One row per task instance: `task`, `granularity`, `patient_id` / `encounter_id`, `ground_truth` (JSON), `split`; 1,268 superseded patient-level retrieval rows carry `is_diagnostic = 0` |
| `relevance_judgments` | 239,545 | Graded relevance (0–3) of each chart section for the per-diagnosis evidence-retrieval instance it belongs to, graded from the section **content** (`scripts/regrade_retrieval_by_content.py`; private-split judgments are held outside the release) |
| `imaging_orders` | 1,865 | Underspecified imaging orders for the imaging-indication task, keyed to their ground truth |
| `diagnoses`, `clinical_findings`, `diagnosis_findings` | 9,623 (8,654 live; 969 merged) / 36,620 / ≤52,082 | The ontology-grounded knowledge graph: diagnoses (ICD-10-CM validated against CMS FY2025, with provenance columns; SNOMED CT), findings (SNOMED CT, LOINC), and typed diagnosis–finding relations |
| `question_findings`, `question_diagnoses`, `board_questions` | 138,777 / 51,075 / 7,003 | Per-source-question annotation links (finding and diagnosis ids with roles) and question metadata (subject, organ system, difficulty). No question text is included |
| `release_info` | — | Version, export date, task list, split definitions, and change notes |

**Tasks** (`benchmark_ground_truth.task`, scorable rows): `patient_diagnosis` (4,545 **index-encounter** instances, `granularity = encounter`: diagnose one visit from the chart up to it; 4,424 scorable — 121 carry `is_diagnostic = 0` because every label is a non-diagnostic entry or has no billable ICD-10 code after the CMS repair; the 1,268 longitudinal problem-list rows are kept with `is_diagnostic = 0`), `context_summarization` (7,613: 1,268 whole-patient plus 6,345 specialty-conditioned), `evidence_retrieval` (4,551 scorable, one per patient × reference diagnosis), `imaging_indication` (1,865).

**Stage-7 task families** (`granularity = encounter`, each derived from a scorable index-encounter `patient_diagnosis` row — `parent_gt_id` — and inheriting its split; labels come from the graph tables and deterministic transforms of the index encounter's own sections; the train split is capped at 1,500 per family to keep the file under GitHub's 100 MB limit): `differential_diagnosis` (3,809: the visit's correct diagnosis plus its distractor diagnoses, ranked), `test_selection` (1,960: the diagnosis plus the *discriminating* documented tests — present lab / imaging / procedure findings with a `pathognomonic` or `highly_suggestive` edge — with the visit's result sections hidden and an `order_test` tool), `error_detection` (3,483: one injected documentation error per visit — implausible value, laterality swap, age or sex contradiction — served as a section override), `lab_triage` (1,248: the visit's key/supporting lab and vital findings vs. background, and the most urgent one), `atypical_diagnosis` (2,692: the parent's labels on a chart whose sentences stating a strong finding are masked). `release_info.stage7_tasks` records the counts. The ground truth of these rows carries the serving rules (`section_overrides`, `hidden_section_types`, the compact `orderable` table) as well as the labels; both are stripped for the private split.

**Release slimming**: `relevance_judgments.passage_source` and `source` are NULL in the release and mean `encounter_section` / `content_rule` (every row carried those constants; `epic_sim.migrate.sqlite_to_pg` restores them on load), and `ground_truth` JSON is stored without whitespace (`release_info.release_slimming`).

**Splits** (`benchmark_ground_truth.split`), assigned at the patient level:

| Split | Patients | Instances | Role |
|---|---|---|---|
| `public` | 200 | 2,375 | The reported benchmark |
| `heldout` | 268 | 3,214 | Labels shipped in this file: use as a second validation set, not as a private test |
| `train` | 600 | 7,310 | Training pool, including RL with the graph-derived rewards |
| `private` | 200 | 2,628 | Carved from the original 800-patient train split (`scripts/carve_private_split.py`, seed 20260927). **Labels are not in this file**: the rows keep their inputs only (`"_labels_removed": true`) and the scorer overlays the labels from an operator-held file (`eval/private_labels.py`) |

Instance counts include the per-diagnosis evidence-retrieval instances (below).

**Data repairs in this fork (Stage 4, `scripts/repair_history.py`, `redefine_patient_diagnosis.py`, `repair_icd_codes.py`)**: history is point-in-time (profile conditions that name a tested diagnosis removed from profiles and notes; procedures that treat a diagnosis shown only after it; the polished HPI never names the visit's own new diagnosis; `note_text` rebuilt); `longitudinal_patients.primary_diagnoses` is NULL (labels live in `benchmark_ground_truth`); ICD-10-CM codes validated against CMS FY2025 with per-node provenance (`icd10_status`, `icd10_repaired_from`, `icd10_repair_reason`, `icd10_flag`, `merged_into`; `diagnosis_merges` table); 969 duplicate nodes (same name and SNOMED id) merged and label references repointed. `patient_profiles.db` / `.jsonl` were regenerated from the repaired tables.

**Label revisions in this fork** (all deterministic, scripted, applied to the release and the private overlay): retrieval judgments graded from section content; whole-patient must-include findings selected round-robin across encounters (critical first, cap 20); absent specialties resampled per patient in proportion to involvement; imaging `reference_terms` (correct diagnosis, differential, key findings) as the scored reference. See `SCORING_CHANGES.md` and `release_info`.

**Evidence retrieval is one instance per (patient, reference diagnosis)** (4,581 instances; `ground_truth.derived = "per_diagnosis_v1"`, `parent_gt_id` points at the original patient-level row). The query names a single diagnosis, so it is no longer the patient's full diagnosis answer key; judgments are derived with the release's own section-grading rule restricted to that diagnosis (`scripts/split_retrieval_by_diagnosis.py`, which first proves the rule reproduces every original judgment). The 1,268 patient-level rows are kept with `is_diagnostic = 0` and `exclusion_reason` set; their judgments were dropped for file size and are exactly reproducible with the same script.

The held-out split was drawn from the same pool as the training split by stratified sampling over dominant ICD-10 chapter and encounter count (seed 20260922); membership is in `scripts/build_rl_split.py`. Every encounter derives from a distinct source question, so no patient shares source material with any other.

**Excluded from the benchmark database**: source text of any kind (raw source cards, board-question vignettes and answers, fact cards and their links, source EHR sections), the licensed SNOMED CT / LOINC terminology tables (load your own with `epic_sim.migrate.load_terminology`), model outputs, and LLM call logs. Relevance judgments are restricted to chart sections; fact-card judgments are not shipped.

**Changes from v1.2**: the single-encounter diagnosis-accuracy task was retired; split labels are `public` / `heldout` / `train` (formerly `val` / `test`); the chart text was cleaned of self-contradictory medication and history statements, sentence fragments, leading whitespace, and duplicate problem-list entries, with every note rebuilt from its sections.

## Provenance

Each patient is formed by deterministically clustering compatible question vignettes into
one person (Stage 8a, no LLM), writing a coherent profile from that cluster (Stage 8b, LLM),
planning a dated visit timeline (Stage 9a, LLM), and generating each visit note by template
assembly plus LLM polish (Stage 9b). Provenance pointers back to source questions
(`source_question_ids`) are removed from the profile files; in `benchmark_v1.3.db` they are kept
as opaque integer keys because the annotation tables are keyed on them, but no question text is
included.

## Copyright & licensing

The released data is fully synthetic and carries no copyright concern. The profile files contain no terminology codes. The benchmark database's knowledge-graph tables carry per-concept ICD-10-CM codes (US public domain), LOINC codes (redistributable with attribution), and SNOMED CT identifiers (concept ids only, no descriptions beyond the display names; use of SNOMED CT content requires a UMLS/affiliate license in non-member countries). The full terminology tables are not included.

## Intended use & limitations

**Known label exposure through the simulator API.** The simulator's problem-list tool and the
`active_problems` field of the chart summary (and the FHIR Condition resource) return the graph-derived
correct diagnoses with ICD-10 codes, i.e. the patient-diagnosis reference. The `/env` endpoints and the
Harbor tasks substitute the chart's documented history and hide assessment/plan sections; agent
evaluations that call the raw tool API directly are not protected.

Intended for benchmarking clinical AI agents. The data is synthetic and must **not** be used
for clinical decision-making or treated as real patient data. Clinical content originates in
medical education material and may contain simplifications or artifacts of the generation
process.