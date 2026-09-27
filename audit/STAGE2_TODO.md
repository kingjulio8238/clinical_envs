# Stage 2: close label leaks (serving layer) — todo

Scope: ROADMAP Stage 2 only. Goal: remove answer information from what models see. **Status: DONE
2026-09-27**; validation below.

## Items
- [x] 2.1 `epic_service.get_problem_list` returns the chart's documented history (profile chronic conditions, no codes); `open_chart.active_problems`, `view_problem_list`, FHIR `Condition` follow (`epic_sim/app/services/visibility.py:documented_problems`)
- [x] 2.2 point-in-time cutoff from one module (`visibility.allowed_encounter_ids` / `filter_future_encounters`) for every consumer that identifies an instance: agent sessions with `gt_id`, `/env` episodes, FHIR with `X-Session-Id` (Encounter, DocumentReference)
- [x] 2.3 assessment/plan hidden server-side in every read path: encounter detail, section, chart search (keyword + semantic candidates), FHIR DocumentReference, `/env`; setting `EPIC_SIM_HIDE_OUTCOME_SECTIONS` (default on)
- [x] 2.4 retrieval query is one diagnosis: 4,581 per-(patient, diagnosis) instances derived with the release grading rule (`scripts/split_retrieval_by_diagnosis.py`; rule reproduced on 58,926 stored judgments, 0 mismatches); patient-level rows superseded (`is_diagnostic = 0`)
- [x] 2.5 "structured" strategy carries ontology guidance only; graph-derived hint loaders removed from all four task loaders (`eval/hints.py`, `eval/tasks/*.py`)
- [x] 2.6 few-shot selection from train runs only; frozen examples regenerated from train patients 1672/1676/1677 with `_meta` (`scripts/build_few_shot_examples.py`, `eval/few_shot_examples.json`)
- [x] 2.7 `private` split: 200 patients / 2,628 instances carved from train (seed 20260927); labels and 54,948 judgments moved to gitignored `private/labels_v1.3.db`; release keeps inputs only; scorer overlay `eval/private_labels.py` (`SH_PRIVATE_LABELS_DB`); Alembic `b8c9d0e1f2a3` adds the enum value; ORM `SplitType`/`EvalTask` enums fixed
- [x] 2.8 `/score` and `/env` return the reward only for private items (`EPIC_SIM_VERBOSE_SCORE_SPLITS`); `503` without the overlay
- [ ] (deferred to Stage 5) rate limiting on `/score`; needs a Redis-backed counter

## Cross-cutting
- [x] Stage-1 xfails owned by this stage flipped and un-marked: `test_problem_list_tool_reveals_nothing_beyond_the_profile`, structured-hints test rewritten against the real prompt builders
- [x] `eval/tests/test_label_leaks.py` (13 tests) + `eval/tests/conftest.py` (ORM-compatible SQLite copy)
- [x] `eval/floors.json` regenerated for public, heldout, train, private (floors CLI accepts `private`)
- [x] DATA_CARD / README / ROADMAP updated; release DB 89 MB (recomputable `search_vector` and superseded judgments dropped)

## Validate
- [x] `pytest eval/tests/test_reward_hacking.py eval/tests/test_label_leaks.py`: **31 passed, 13 xfailed in 59 s**; `eval.floors --check` current
- [x] pre-existing suites: eval (other) + etl `50 passed, 1 skipped`
- [x] Postgres reloaded from the new SQLite in the compose stack: counts match (16,595 GT rows, 239,545 judgments, `private` 2,628), `search_vector` recomputed for all 59,964 sections
- [x] simulator suite in the container: see final run below
- [x] end-to-end through the live stack: imaging `/env` episode sees 2/3 encounters, no A&P in detail, `open_chart.active_problems` == profile with no codes; dx episode `view_problem_list` has no codes; retrieval intro names one diagnosis; `/score` private item → `503` without overlay, `reward` with `metrics = {}` with overlay; train item keeps 14 metrics; FHIR: Encounter total 3 → 2 with imaging session, future `Encounter/{id}` → 404, Condition entries == profile names with no coding, DocumentReference `type=assessment` total 0
- [x] final container run recorded in the Stage 2 commit message

## Notes
- The running example of the paper (patient 1973) landed in the private split.
- `EPIC_SIM_FORCE_RELOAD=1 docker compose up -d` does not reach the container (compose does not pass the variable); reload with `docker compose exec app python -m epic_sim.migrate.sqlite_to_pg`. Fix in Stage 6.
- The app image bakes the code: after code changes run `docker compose up -d --build app`.
