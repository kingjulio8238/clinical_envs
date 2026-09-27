# Stage 7: new tasks — DONE 2026-09-27

Scope: ROADMAP Stage 7 only. Goal: new task families that use what no current task uses — the 27,424
distractor rows, the typed diagnosis–finding relations, the structured findings behind the notes — each
verifiable by construction, each with degenerate floors and an oracle ceiling, each served by both
environments with the visibility rules of Stage 2/4. Done = every box checked, floors regenerated,
Postgres reloaded, live episodes of each task recorded, validation filled in.

## Design decisions (fixed before coding)
- **Every new instance derives from a scorable index-encounter diagnosis instance** (`parent_gt_id`), so it
  inherits the patient's split, the point-in-time cutoff and the private-label handling. Labels are built
  from the graph tables only (`question_diagnoses`, `question_findings`, `diagnosis_findings`,
  `clinical_findings`) and from deterministic text transforms of the index encounter's own sections; no
  LLM, no clinical thresholds of mine.
- **`differential_diagnosis`** — rank the differential at the index visit. Label: the encounter's correct
  diagnosis and its distractor diagnoses (the answer choices the vignette was written to discriminate).
  Submission: a ranked list of ≤5 `{icd10, name}`. Reward `differential_ndcg_5`: gain 1.0 for the correct
  diagnosis, 0.5 for each distractor, both with the Stage-3 graded ICD credit (1 / 0.75 / 0.5), normalized
  by the ideal ordering. Copying the chart's problem list earns nothing (documented history ≠ this visit's
  differential); a frequency prior over train diagnoses is the floor to beat.
- **`test_selection`** — the agentic task. The index encounter's result sections (labs, imaging, pathology,
  other studies) are hidden. A new tool, `order_test(name)`, returns the result of a test *as documented at
  the index encounter* (matched by name to the encounter's lab / imaging / procedure findings; "not
  performed / no result documented" otherwise) and costs a step. Submission `submit_workup{icd10, name}`; the
  environment records the orders. Label: the diagnosis plus the *discriminating* tests — present findings
  of orderable types with a `pathognomonic` or `highly_suggestive` edge to the label. Reward
  `workup_score = icd_credit × coverage × parsimony`, coverage = 1 if at least one discriminating test was
  ordered else 0.25 (a diagnosis without evidence is a guess), parsimony = min(1, n_needed / n_orders)
  with n_needed = number of discriminating tests (≥1); a diagnosis with no orders gets coverage 0.25.
  Ordering everything is penalized by parsimony; ordering nothing is capped at 0.25.
- **`error_detection`** — controlled imperfection with labels. One error is injected into one section of
  the index encounter by a deterministic transform chosen by rotation: an implausible lab value (a numeric
  result ×10), a laterality swap (left ↔ right), an age contradiction (the HPI's "N-year-old" moved by +40
  years against the stated demographics), a sex contradiction (pronouns flipped in one HPI sentence). The
  environment serves the altered text (section overrides); the label is `{section_type, error_type,
  original, injected}`. Submission `{section_type, error_type, description}`; reward `error_detection_score`
  = 0.5·[section type correct] + 0.5·[error type correct]. Floor: the most common (section, type) pair.
- **`lab_triage`** — which of this visit's results matter. Label from the graph: `relevant` = the index
  encounter's present lab / vital findings whose relevance is key or supporting; `background` = the rest;
  `most_urgent` = the key finding with the strongest typed edge to the diagnosis (pathognomonic >
  highly_suggestive > commonly_seen). Submission `{relevant: [names], most_urgent: name}`; reward
  `triage_score` = 0.6·F1(relevant, name-matched) + 0.4·[most_urgent matched]. Flagging every result as
  relevant is penalized by precision.
- **`atypical_diagnosis`** — robustness variant of patient diagnosis. The sentences of the index encounter
  that state a present `pathognomonic` / `highly_suggestive` finding are masked ("[finding not
  documented]"); the label is unchanged; the reward is the patient-diagnosis reward. Robustness =
  score(atypical) − score(original) per model.
- **Serving.** Both environments apply, per episode: the cutoff (all new tasks are encounter-bound), hidden
  result sections for `test_selection`, section overrides for `error_detection` / `atypical_diagnosis`,
  and the `order_test` tool (HTTP: `env_service.step`; in-process: `LocalEnv`). The overrides and the
  orderable table live in the ground truth (labels, never returned) and are stripped for the private split.
- **Size.** `benchmark_v1.3.db` must stay under GitHub's 100 MB limit (95 MB today): overrides store only
  the changed sentence, the orderable table stores names / values / flags, and each family is capped per
  split if needed.

## Data (scripts/build_stage7_tasks.py; release + private overlay; idempotent)
- [x] `differential_diagnosis` instances (index instances with ≥1 distractor)
- [x] `test_selection` instances (index instances with ≥1 present discriminating orderable finding)
- [x] `error_detection` instances (one injected error per eligible index encounter, rotating type)
- [x] `lab_triage` instances (≥3 present lab/vital findings, ≥1 key)
- [x] `atypical_diagnosis` instances (≥1 maskable sentence)
- [x] `release_info` key; DATA_CARD counts; DB < 100 MB

## Scoring (eval/scoring_tasks7.py, dispatched from compute_all_metrics)
- [x] `differential_ndcg_5` (+ `differential_top1`), `workup_score` (+ components), `error_detection_score`, `triage_score`; atypical → patient-diagnosis metrics
- [x] PRIMARY_METRIC in score_one / degenerate / floors; LABEL_KEYS; `_attach_context` for atypical (neutral set)
- [x] oracle for every task scores 1.0

## Environments
- [x] prompts: TASK_GOALS, SUBMIT_TOOL_SCHEMAS (`submit_differential`, `submit_workup`, `submit_error`, `submit_triage`, reuse `submit_diagnosis`), patient intro
- [x] `order_test` tool (definition + env-only execution); hidden result sections; section overrides — HTTP env
- [x] same in `LocalEnv`; parity test on sampled episodes
- [x] `EvalTask` enum + Alembic migration adding the enum values; `EVAL_TASKS`; task loaders for the single-turn harness; `POINT_IN_TIME_TASKS`

## Floors and gates
- [x] degenerate policies per task (empty; differential: copy_problem_list, train frequency prior; workup: no_orders, order_everything, order_everything_plus_prior_dx; error: majority (section,type); triage: all_results, empty; atypical: the dx policies)
- [x] `eval/floors.json` regenerated (4 splits); reward-hacking gates: differential copy ≤ 0.1, order-everything ≤ 0.5·ceiling, all-results triage ≤ 0.5, majority error ≤ 0.6
- [x] `eval/tests/test_stage7_tasks.py`: label construction invariants, scorer properties, env behaviour (hidden sections, overrides, order_test reveals only documented results, orders counted)

## Validate
- [x] `pytest eval/tests etl/tests` green; simulator suite green in Docker after the enum migration
- [x] Postgres reloaded; live `/env` episode per new task (LocalEnv and HTTP), oracle 1.0, `order_test` smoke
- [x] committed and pushed; main synced

## Validation record
- Instances (`scripts/build_stage7_tasks.py --cap-train 1500`): differential_diagnosis 3,725 (public 666 / heldout 873 /
  train 1,500 / private 686), test_selection 1,960 (314 / 379 / 965 / 302), error_detection 3,483 (574 / 781 / 1,500 /
  628), lab_triage 1,248 (172 / 266 / 599 / 211), atypical_diagnosis 2,692 (427 / 545 / 1,314 / 406); 13,108 rows; the
  release grew 86.3 → 95.9 MB after `scripts/slim_release.py` took it from 95.3 to 86.3 MB losslessly (judgment
  constants NULL, restored on Postgres load; compact JSON). Differential distractors exclude uncoded nodes and the
  label itself (the source questions list the answer among the choices — 1,999 such rows in the audit).
- Public floors (`eval/floors.json`): differential 0.027 (copy the problem list; frequency prior lower), test_selection
  0.001 (order everything + problem-list diagnosis), error_detection 0.494 (majority (section, type): the half-credit
  metric gives a guess of either half), lab_triage 0.631 (flag every result: most results at these visits are
  key/supporting, so the headroom is the precision on the background ones and the most-urgent pick), atypical 0.035.
  Oracle 1.0 on every family, in the scorer and through both environments.
- Tests: `eval/tests/test_stage7_tasks.py` 18 passed (labels derive from scorable index rows; distractors coded and
  disjoint from the label; discriminating tests are documented present findings with the right edges; injected
  errors change the text as typed; triage labels partition; atypical keeps the label and masks; oracle 1 / empty 0;
  reversed differential < oracle; copy-the-problem-list ≤ 0.1; workup components incl. shotgun penalty and a claimed
  order that the trace does not carry; error half credit and aliases; all-results triage < 0.75; env hides result
  sections, refuses order_test outside test_selection, reveals only documented results, counts orders from the trace,
  serves overrides and masks; every brief offers its submit tool and order_test only for test_selection).
- Live HTTP: all five families reset/step/submit through the compose stack with oracle reward 1.0; test_selection
  hid the index visit's result sections and `order_test('Proteinuria')` returned the documented result.
- Simulator suite: 102 passed + `test_get_tools_returns_14` (order_test) after the enum migration `e1f2a3b4c5d6`.
- Known limits: error injection is four deterministic transforms (no copy-forward or medication–allergy conflicts
  yet: those need the medication/allergy sections that `open_chart` reads from SQL, not from section overrides);
  lab-triage and error-detection floors are high by construction of their metrics; trend/threshold tasks over the
  FHIR time series remain open.
