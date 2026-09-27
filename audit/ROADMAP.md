# Where the fork should extend Synthetic Hospital

After Phase 1 (this roadmap + RL on train + evals) is done, continue with `audit/SCALE_UP_PLAN.md`.

Companion to `audit/FINDINGS.md` (evidence referenced as F§n / headline #n). Written 2026-09-27, after the
audit of paper §3–§6.

## What the fork should be

The upstream repo is a **benchmark with a paper attached**. Its most valuable, non-replicable assets are:
- a patient-disjoint, ontology-linked longitudinal corpus;
- a deterministic in-process scorer;
- an `/env` reset/step API.

Its weakest parts are the reward functions and the label hygiene. Those are exactly what an RL environment
cannot tolerate: a policy optimizes whatever the reward actually pays for, and today it pays for:
- copying problem lists (#2);
- submitting one HPI section (#1);
- appending "No significant distress." (#4);
- omitting documented conditions (#13).

**Recommendation:** position the fork as a *reward-hardened clinical RL environment built on the Synthetic
Hospital corpus*, not as a re-run of the paper. Almost every fix below is **deterministic and runs on the
released DB alone**, with no source content, no LLM gateway and no GPU. Only the last stages need new
generation.

Order is by dependency. Each item has an acceptance test; most tests are already scripted in `audit/scripts/`.

---

## Stage 1: A reward-hacking regression suite (build first; everything else is measured by it)

**Status: DONE (2026-09-27).** `eval/degenerate.py` (policies + oracles), `eval/floors.py` + `eval/floors.json`
(floors/ceilings for public, heldout, train; `normalize()`), `eval/tests/test_reward_hacking.py` (31 tests: 16 pass,
15 strict xfails naming Stages 2/3/4), `eval.report` normalized column, `.github/workflows/tests.yml`. Public floors:
dx 0.836 (copy problem list + profile; 1.000 via the tool API), retrieval P@5 0.995 (one HPI; ceiling 0.999),
nDCG@10 0.808 (type prior), summarization 0.668 (chart dump), specialty involved 0.478, specialty absent 1.000
(empty answer; **no headroom**), imaging 0.217 (restate order).

Turn every exploit found in the audit into a test that fails CI.

| Degenerate policy | Current reward | Gate |
|---|---|---|
| empty / malformed submission | 0 | = 0 |
| retrieval: one HPI section | P@5 0.995 | ≤ chance |
| retrieval: section-type ranking (content-blind) | P@5 0.97 | ≤ chance + ε |
| summarization: paste the chart | 0.676 | well below a good summary (needs a precision term) |
| summarization: echo finding names | 0.428 | ≈ 0 without clinical context |
| specialty: any text + "No significant distress." | absent 1.000 | abstention only via an explicit field |
| diagnosis: regex-copy problem list | 0.843 | ≈ share of labels that are legitimately copyable |
| diagnosis: echo the `view_problem_list` tool | 0.993 | tool no longer returns labels |
| diagnosis: reference + documented secondary conditions | 0.662 | ≈ 1.0 (no penalty for documented truth) |
| imaging: restate the order | 0.217 | report as floor |
| hallucination metric: invented sentences | 0.000 | must flag them (replace token-overlap grounding with concept-level support) |
| retrieval: random shuffle | nDCG@10 0.524, MRR 0.831 | report as floor; a metric whose floor ≈ the best model is retired or re-graded |

Also report every score as **(model − floor) / (ceiling − floor)**, with floor = best degenerate policy and
ceiling = oracle. Absolute numbers on this benchmark are not interpretable without it (F§6).

## Stage 2: Close the label leaks (serving layer; no data change)

**Status: DONE (2026-09-27).** `epic_sim/app/services/visibility.py` holds the three rules (hidden outcome
sections, documented-history problem list, point-in-time cutoff) and every consumer uses it: `epic_service`
(problem list, encounter detail, section, search), the agent tool router (session-bound cutoff), FHIR
(`Condition` from the chart, `DocumentReference`/`Encounter` filtered, `X-Session-Id` cutoff) and `/env`.
Retrieval is one instance per (patient, diagnosis) (`scripts/split_retrieval_by_diagnosis.py`, 4,581 instances,
grading rule reproduced on 58,926 judgments with 0 mismatches). The "structured" strategy carries ontology
guidance only (`eval/hints.py`); few-shot examples are train-only (`scripts/build_few_shot_examples.py`). A
200-patient `private` split has its labels in a gitignored overlay (`scripts/carve_private_split.py`,
`eval/private_labels.py`); `/score` and `/env` return the reward only for it. Regression tests:
`eval/tests/test_label_leaks.py`; the two Stage-1 xfails owned by this stage are unmarked. Not done: rate limiting
on `/score` (needs a Redis-backed counter; noted for Stage 5). "No data change" was not held to: item 4 required
new retrieval rows and item 7 moved labels out of the release DB (89 MB after dropping the recomputable
`search_vector` text and the superseded judgments).


1. **Fix `epic_service.get_problem_list`** (`epic_sim/app/services/epic_service.py:297`) at the source, so every consumer is safe: `open_chart.active_problems`, `view_problem_list`, FHIR Condition, and the paper's `eval/agents` harness. Today only `/env` substitutes the profile history (#10).
2. **Point-in-time filtering in one place.** Move `filter_future_encounters` from `env_service` into the service layer so the agent harness and FHIR honor it too (F§6 5.3).
3. **Hide assessment/plan server-side for all consumers**, not just in the client harness.
4. **Retrieval query ≠ answer key.** Today the query is the patient-diagnosis reference (1,268/1,268, #3). Options:
   - one instance per (patient, diagnosis), which also multiplies RL states;
   - never serve a patient's retrieval and diagnosis instances to the same policy in eval.
5. **Prompt strategies must not contain labels.** Drop or quarantine the "structured" hints that inject key findings, the correct diagnosis or organ systems (#12). Few-shot examples come from **train** only, never from `public` (F§6 5.1).
6. **Held-out integrity.**
   - v1.3 shipped held-out labels, so treat `heldout` as a second validation set.
   - For a real private test, carve a fork-internal split from `train` and never publish its labels.
   - Make `/score` return only the reward (not per-metric breakdowns) for private splits, and rate-limit it; the full metric dict lets a caller probe labels.
   - Replace the default scorer token.

## Stage 3: Harden the reward functions (scorer; deterministic)

**Status: DONE (2026-09-27).** See `SCORING_CHANGES.md` for the definitions. Per-item = batch mean everywhere;
zero-fill; graded ICD credit and acuity; documented conditions neutral; fixed-k P@k; content-graded retrieval
judgments with nDCG@10 as reward; whole-patient summary = HM(negation-aware concept recall, grounded precision) ×
length, must-include rebalanced per encounter; explicit `abstain` field, involvement-weighted absent specialties,
critical-fallback for the 540 zero items; imaging concept F1 against graph reference terms; concept-grounded
hallucination rate. All 12 Stage-3 xfails un-marked and passing; floors regenerated. `eval/concept_match.py` is the
shared matcher. Floors before → after (public): retrieval P@5 0.995 → 0.38 (type prior; one HPI 0.06), nDCG@10
0.81 → 0.40; summarization chart dump 0.67 → 0.32; dx copy 0.84 → 0.60; specialty absent still 1.0 for a constant
`abstain: true` (binary unit, not normalized).


**All tasks**
- Zero-fill: every requested item gets a score. Today patient diagnosis skips empty predictions and the single-turn runner drops API failures (F§10 C.3), so means cover only answered items.

**Patient diagnosis**
- Extend the chart-neutral set to conditions documented in *encounters*: at minimum the graph's `secondary` diagnoses, and ideally anything the chart states (#13).
- Replace flat 3-char matching with hierarchical credit (full code > subcategory > category). Today bare `K35` / `J96` earn full credit, and chronic E11.9 earns credit for acute HHS E11.01.
- Score predicted acuity. The paper's task output includes it; the metric ignores it.
- Severity weights by clinical severity, not ICD first letter.

**Evidence retrieval**
- Fixed-k denominator for P@k (never `min(k, len(results))`).
- **Re-grade from section content**: a section is relevant if its text actually contains the query diagnosis' key or supporting findings (names and `value_text` are in `question_findings`), using the negation-aware cascade already in `eval/scoring.py:_list_recall_cascade`. Today grades are a function of section *type* only (F§1 Stage 4).
- Headline nDCG over the re-graded pool.

**Summarization (whole-patient)**
- Add a precision/faithfulness term. Recall-only lets a chart dump score 0.68. Options:
  - penalize off-target findings (as the specialty variant already does);
  - penalize unsupported sentences (`hallucination_rate` exists);
  - penalize length.
- Stratify must-include findings per encounter instead of taking the first 20 chronologically. 18–31% of patients currently have nothing required from the latest visit.
- Match at concept level (reuse `eval/imaging_concepts.ConceptExtractor`) with a negation guard, instead of display-name substring. Today "Patient denies fever" satisfies "Fever".

**Specialty-conditioned**
- 540/3,809 involved items have no critical finding and score 0 for any answer as single-item rewards. Fall back to complete recall, or drop them from RL sampling. Also make per-item reward and corpus metric agree (F§13).
- Explicit `abstain: bool` output field, instead of "≤12 words or contains a stock phrase".
- Sample absent specialties at random and balance them. Today they are the first two alphabetically, so every patient gets Cardiology and Dermatology, which the name alone predicts.

**Imaging**
- The reference question is Kimi-authored and was written with the full (future-leaking) profile.
- Near term: score against a **deterministic** reference: concepts of the encounter's correct diagnosis plus key findings from the graph.
- Later: regenerate the reference with a point-in-time context and a model family not on the panel.

## Stage 4: Repair the data in place (release DB only; deterministic)

**Status: DONE (2026-09-27).** `scripts/repair_history.py` (profiles: 564 changed, 1,128 tested conditions removed;
4,536 PMH profile lines removed; 203 HPIs masked; 348 PSH procedures hidden until after their diagnosis; 2,660 notes
rebuilt; `primary_diagnoses` nulled; profile files regenerated), `scripts/repair_icd_codes.py` (CMS FY2025 validation,
1,226 codes repaired, 1,544 flagged, 969 nodes merged into tombstones that release their code so the simulator's
`UNIQUE(icd10_code, snomed_id)` and edge uniqueness hold, provenance columns + `diagnosis_merges`, labels refreshed),
`scripts/redefine_patient_diagnosis.py` (4,545 index-encounter instances, 4,424 scorable: 73 all-non-diagnostic and 48
whose only label lost its code are `is_diagnostic = 0`; 1,057 repeat encounters excluded; earlier keyed diagnoses
chart-neutral; longitudinal rows superseded), retrieval regraded on the edited text. Verified live: Postgres reload
count-exact, simulator suite green in Docker, `/env` index-encounter episodes hide later encounters and the label, and
the oracle scores 1.0. Acceptance:
HPI self-naming 0/4,388 (was 210); profiles naming a tested diagnosis 1/1,268 (was 500); salpingectomy anachronisms
0/12; the two remaining appendectomy lines follow an earlier appendicitis visit; copy-the-problem-list policy 0.07 on
the new unit (was 0.84). The last Stage-1 xfail is un-marked. Residual: 99/705 public index charts still contain the
label's name somewhere, mostly in family history ("Father: type 2 diabetes") and source-vignette prose; these are not
copyable lines and were left untouched (masking clinical content would alter the case).


1. **Temporal consistency.**
   - Remove from each encounter's PSH any procedure whose triggering diagnosis is first keyed at or after that encounter. The appendicitis→appendectomy style map already exists in `scripts/chart_neutral_sets.py:SURG`.
   - Remove later-keyed diagnoses from encounter-0+ PMH (322 patients). The root cause is that 500/1,268 profiles list the patient's own tested diagnoses as chronic conditions (F§12). Strip those from `profile.chronic_conditions` before rendering the PMH, and recompute the chart-neutral sets.
   - Handle the 18 charts whose encounter 0 was removed after generation. Either renumber them and regenerate the first HPI, or drop them.
   - Mask the encounter's own diagnosis in polished HPIs (210 cases).
   - Acceptance: `data_checks.py` leak counters go to 0.
2. **Make patient diagnosis non-copyable.** Re-define the task as **point-in-time index-encounter diagnosis**: chart truncated at encounter *k*, label = encounter *k*'s correct diagnosis.
   - That label appears nowhere in the truncated chart, so the task becomes diagnosis rather than extraction (#2).
   - It raises instances from 1,268 to up to 5,602, excluding repeats of an earlier diagnosis or marking them as follow-ups.
   - Keep the longitudinal problem-list task as a separate extraction task and name it as such.
3. **ICD hygiene.**
   - Validate every code against CMS ICD-10-CM, which is public and already fetched by the entrypoint. `audit/scripts/appendix_d_checks.py` already does this: 90 of the 5,602 encounter reference codes are uncoded or have an invalid category.
  - Add a semantic check even for *billable* codes. Step (i) accepted e.g. "Leukocytosis" → D72.819 (*decreased* WBC) and "Septic arthritis" → B26.85 (mumps arthritis) (F§11).
   - Add a semantic check of code description vs display name (with/without contradictions, SapBERT or token similarity).
   - Fix the 346 unvalidated reference codes.
   - Apply the 88 blocked corrections by merging the duplicate nodes (`scripts/write_dup_note.py`).
   - Dedupe same-concept nodes (e.g. 8 "Leukocytosis" nodes).
   - Drop the 1,965 correct-also-distractor rows.
4. **Surface realism tells:** jitter encounter dates (all on the 15th); flag or merge same-diagnosis-every-visit patients (656 with repeats; 1771 has MDD ×7).
5. **Persist provenance:** write `resolution_method` per node (computed at `s05:692`, never stored); ship `diagnosis_relations`.

## Stage 4b: FHIR fidelity (serving layer)

- `Observation`: carry the presence flag (as `interpretation` or `dataAbsentReason`, or drop absent findings), `effectiveDateTime` and the encounter reference from the source encounter, units, and one resource per measurement rather than per finding. Today the 20,861 absent findings are served as positive observations (F§16).
- `Condition`: fix at the service layer (Stage 2.1).
- Add FHIR writes (orders, problem-list updates) if MedAgentBench-style action tasks are wanted.

**Status: DONE (2026-09-27).** `Observation` is one resource per measurement (`id` = `question_findings.id`; the
patient's 178 measurements were 1 resource per distinct finding before) with `encounter`, `effectiveDateTime`,
`interpretation` POS/NEG (absent findings with no value carry SNOMED Absent), values parsed from `value_text` by
`epic_sim/app/fhir/units.py` (UCUM units for 32,630 of the 38,953 numeric measurements, 2,212 blood pressures as
components, 1,034 reference ranges, 217 comparators; unknown units are never guessed — 4,111 quantities are served
without one and the raw text in `note`); `code`, `date` and `encounter` search params; the session cutoff applies to
`Observation`, `DiagnosticReport`, `ServiceRequest`, `MedicationRequest` and `AllergyIntolerance` (only Encounter and
DocumentReference honoured it before); a labs `DiagnosticReport` lists its Observations in `result`. FHIR `create` for
Observation, ServiceRequest, MedicationRequest and Condition with write scopes, server-assigned ids, `Location`,
OperationOutcome errors and a session-scoped `fhir_writes` table (Alembic `d0e1f2a3b4c5`). `epic_sim/tests/test_fhir.py`
+ `test_units.py` cover it; see `audit/STAGE4B_TODO.md`.

## Stage 5: Environment throughput (profile first)

The `/env` step path is HTTP → FastAPI → Postgres → Redis. The scorer is already in-process and deterministic.

- **Profile first:** measure step latency and the rollout-throughput ceiling once Docker is up.
- **Likely win if the step dominates:** an **in-process gym-style env over the read-only SQLite DB**, with tools as SQL, no network, and trivially parallel across workers. This matches the "environment step is the critical path" rule for RL.
- **Keep parity:** the HTTP env stays for Harbor/external agents, and both must pass the Stage 1 suite.

**Status: DONE (2026-09-27).** Baseline (`eval/bench_env.py`, compose stack on an 8-core laptop, single uvicorn
worker): 13.6 steps/s for one sequential client, 105 steps/s at 4 concurrent clients, falling to 64 at 64 clients
(p50 reset 769 ms); under load the app container sits at 100% of one core while Postgres is at 40–60% and Redis at
4%, so the bottleneck was the single-process Python server (framework + ORM + JSON), not the databases. Component
floors: framework 2.2 ms, Redis +1.2 ms, a Postgres tool 5–8 ms, the scorer 19 ms, reset 19 ms. Fix:
`eval/local_env.py` (`LocalEnv`), the same environment in-process over the SQLite release — same brief, tools,
observation shapes, visibility rules, budget and reward, checked by `eval/tests/test_local_env.py` (28 tests, incl.
byte-for-byte observation and reward parity against the live server). With HTTP gone the scorer dominated (60 ms
mean per submit, 1.6 s worst: first-touch concept extraction over a chart), so two hot paths in
`eval/imaging_concepts.py` were made output-identical but 4× faster each (precomputed curated-pattern table +
per-call token stems; fuzzy-snap vocabulary bucketed by first letter and length); `floors --check` stays bit-exact
and runs 3× faster. Result on the same 64 public episodes, single process: **470 steps/s vs 13.6 (35×) and vs the
server's 4-client peak of 105 (4.5×)**; tool steps 0.05–0.3 ms, search 2.4 ms, submit 2.8 ms p50 / 12.9 ms mean.
Multi-process scaling was measured on a host with load average 30 (an unrelated 4-core job running), so it is
reported but not trusted: see `audit/STAGE5_TODO.md` and `audit/bench/*.json`.

## Stage 6: Reproducibility of the pipeline (only if the fork will regenerate data)

- **Pipeline wiring:** wire stages 7–12 into `etl/main.py`, and make `run()` execute every sub-step.
- **Fail loudly on missing curated / finding-site / miscoding CSVs.** Today a missing `curated_diagnosis_edges.csv` silently changes the specialty labels.
- **Stage 5 correctness:** retry structural-validation failures (today they are logged and dropped); key the LLM cache on a hash of the full input (today it uses the first 200 characters); store `resolution_method` and the segmentation review flags.
- **LLM endpoint:** replace the private gateway with a configurable OpenAI-compatible endpoint; set temperature and seed; cache calls.
- **Split scripts:** remove the dependency on the retired `diagnosis_accuracy` task; ship the split JSON; fix `split="val"` defaults and the non-random `select_patients` in `eval/agents/runner.py`.
- **Source content:** needs our own. The ETL is source-agnostic via YAML deck/PDF profiles. The upstream tags (`GPT4Anki::…`) suggest a public LLM-generated deck, which is a **contamination risk** for frontier-model evaluation. A private source corpus is the only route to a truly unseen test set.

## Stage 7: New capability (only after Stages 1–4, so rewards stay honest)

These address the paper's own limitations and are all verifiable by construction:

- **Controlled imperfection with labels.** Inject copy-forward errors, medication-list discrepancies, contradictory allergies and missing results, with an injection log as ground truth. This yields new tasks (contradiction detection, medication reconciliation), and it replaces today's accidental, unlabeled errors.
- **Differential diagnosis and test selection.** The graph holds **27,477 distractor rows** and typed relations (`rules_out`, `highly_suggestive`, `pathognomonic`) that no current task uses. Two new tasks:
  - rank the differential;
  - choose the test that best discriminates the correct diagnosis from its distractors, rewarded from the typed relations.
  
  This is the genuinely agentic task the paper lacks: evidence gathering is *necessary*, not optional (cf. §5.3).
- **Structured longitudinal data.** Labs with LOINC and numeric values exist in `question_findings`. Serve them as FHIR Observation time series and add trend and threshold tasks with numeric, exactly checkable rewards.
- **Atypical presentations.** Drop pathognomonic or highly suggestive findings from a vignette to create atypical variants with unchanged labels, then measure robustness.

## Stage 8: Evaluation protocol for anything the fork reports

- Report floors (Stage 1), oracle ceilings, and **paired bootstrap CIs**. The metric's patient-level SD at n=200 is ≈0.014 per system.
- Keep the benchmark-building model (Kimi 2.5) off the ranked panel, or report it separately.
- Run strategy ablations without label-bearing hints, with few-shot examples from train only.
- Ablations must actually vary the manipulated content: App. B's "graph" arm carried no graph information (F§9). Add a shuffled-content control and hold persona/instructions fixed.
- Sample agentic evaluation patients at random, stratified by encounter count.
- Smoke-test before any paid model run. Re-running Table 2 is worthwhile only for targeted questions, e.g. one model with vs without the summarization hints to size the leak (F§6).
