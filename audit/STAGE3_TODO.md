# Stage 3: harden rewards (scorer) — todo

Scope: ROADMAP Stage 3 only. Goal: make each score measure the claimed task. **Status: DONE 2026-09-27**;
validation below. Definitions: `SCORING_CHANGES.md`.

## Design decisions (held)
- Metric names unchanged; definitions changed. Every batch metric is the mean of per-item rewards, so `/score` on one
  item equals the aggregate.
- Label changes were applied by scripts to the release DB **and** the private overlay (all idempotent, `release_info` keys):
  `regrade_retrieval_by_content.py`, `rebalance_must_include.py`, `resample_absent_specialties.py`, `build_imaging_reference.py`.

## General
- [x] missing / malformed answers score 0 on every task (dx per-item zero; runner stores failures as `{}` via `_FailedResponse`)

## Diagnosis
- [x] chart-neutral set includes graph `secondary` diagnoses (`scripts/chart_neutral_sets.py:documented_secondary`) — #13
- [x] graded ICD credit 1 / 0.75 / 0.5; unspecified reference fully credited by its category (`_icd_credit`)
- [x] acuity scored (× 0.5 when missing or wrong against a known reference acuity)

## Retrieval
- [x] fixed-k P@k — #1
- [x] judgments regraded from content; identical text ⇒ identical grade (asserted per instance) — #22
- [x] reward = `ndcg_10` (score_one, degenerate, floors, report, score router, tests)

## Summarization (whole-patient)
- [x] concept-level, negation-aware recall (`eval/concept_match.py`) — F§13
- [x] grounded-concept precision (chart text ∪ patient's annotated terms, by concept surface forms) — oracle 1.0
- [x] length factor (budget 350 words)
- [x] per-encounter round-robin must-include; latest encounter represented for 1,268/1,268 patients

## Specialty
- [x] explicit `abstain` field (prompts, submit schema, env normalizer, parser); text never abstains — #4
- [x] absent specialties resampled, involvement-weighted; name-only classifier ≤ majority + 0.02
- [x] critical-fallback to complete recall (540 items) — #23; oracle 1.0 on every involved item
- [x] excluded findings implied by the target set are not leakage
- [x] per-item conditioned_f1 = batch mean

## Imaging
- [x] deterministic `reference_terms` (correct dx, distractor differential, key symptom/sign/imaging findings); LLM question demoted to `_llm`

## Cross-cutting
- [x] hallucination metric = sentences with no supported concept — #14 (invented sentences flagged ≥ 0.8)
- [x] Stage-1 xfails owned by this stage un-marked (12) and passing; new gates: graded credit, acuity, per-item = batch,
  grounded oracle, length discount, quota coverage, content-consistent grades, abstain field
- [x] `eval/floors.json` regenerated for public / heldout / train / private (see table)
- [x] README, DATA_CARD, ROADMAP, STATE, `SCORING_CHANGES.md`

## Validate
- [x] eval suites (reward-hacking + leaks), floors currency deferred: **52 passed, 2 failed → fixed, 1 xfailed (Stage 4)** in 3:11; final CI run below
- [x] simulator suite in the container (image rebuilt, Postgres reloaded from the Stage-3 SQLite, overlay installed): **84 passed, 8 skipped**
- [x] live `/env`: oracle reward 1.000 and empty 0.000 on all four tasks; `/score/tasks` reports `ndcg_10` for retrieval;
  absent item: text → 0.0, `abstain: true` → 1.0; private item → reward only (`metrics = {}`)
- [x] final `pytest eval/tests etl/tests` (incl. floors currency): `floors.json is current`; **104 passed, 1 skipped, 1 xfailed** (Stage 4) in 2:59, plus one upstream test (`test_sc_abstention`) updated to the explicit `abstain` field and re-run green
- [x] committed and pushed; main synced

## Floors before → after (public split; chart-only policies)
| unit | metric | ceiling | Stage 2 floor | Stage 3 floor | floor policy now |
|---|---|---|---|---|---|
| patient_diagnosis | w-F1 neutral | 1.000 | 0.836 | **0.570** | copy problem list + profile (acuity/credit now bite) |
| evidence_retrieval | nDCG@10 (reward) | 1.000 | 0.808 | **0.432** | section-type prior (random 0.25; one HPI 0.07) |
| evidence_retrieval | P@5 | 0.918 | 0.995 | **0.398** | section-type prior |
| context_summarization | clinical_f1 | 1.000 | 0.668 (chart dump) | **0.487** (echo 10 names; chart dump 0.32) | length + grounding |
| specialty_involved | conditioned_f1 | 1.000 | 0.478 | **0.183** | chart dump |
| specialty_absent | abstention_acc | 1.000 | 1.000 | 1.000 | constant `abstain: true` (binary unit, not normalized) |
| imaging_indication | concept F1 | 1.000 | 0.217 | 0.219 | chief complaint / restate order |

Heldout / train floors: dx 0.601 / 0.605; nDCG@10 0.438 / 0.442; summarization 0.459 / 0.468; specialty involved 0.180 / 0.172;
imaging 0.207 / 0.214. Private (computed with the overlay): dx 0.613; nDCG@10 0.447; P@5 0.410; summarization 0.475; specialty involved 0.158; imaging 0.225.

## Notes
- A local Postgres on 127.0.0.1:5432 shadows the compose one for host connections; run the simulator suite inside the
  container (`docker compose run --rm app pytest epic_sim/tests`) after `docker compose build app`.
- Cold cost of the summarization reward is ~0.75 s per patient (chart concept extraction, cached per process); warm cost ~10 ms.
- Deferred from the roadmap: rate limiting on `/score` (Stage 5).
