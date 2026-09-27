# Synthetic Hospital v1.3: paper vs. code audit

Scope: paper §3 (Benchmark construction), §3.1 (Evaluation splits), Table 1, §4 (Assessing benchmark realism), §5 (Results), §6 (Conclusion and limitations) and Appendices A–I,
checked against this fork at
commit `b047385` and the released `benchmark_v1.3.db`. Audited 2026-09-27.

Method: code reading with file:line evidence, direct SQL over the released DB, and zero-model baselines scored
with the repo's own scorer (`eval.scoring.compute_all_metrics`, the function behind `/score` and `/env`). Every
number below is reproducible from `audit/scripts/` (run from the repo root with the `.venv` built from
`requirements-core.txt`):

| Script | Reproduces |
|---|---|
| `data_checks.py` | all data-level counts (leakage, clustering, splits, overlap, exploits) |
| `copy_baseline.py <split>` | copy-the-problem-list baseline, patient diagnosis |
| `type_baseline.py <split>` | content-blind section-type ranking, evidence retrieval |
| `text_baselines.py <split>` | dump-the-chart / abstention-phrase / restate-the-order baselines |
| `realism_checks.py` | §4: study statistics, chapter-distribution JSD, graph-edge share of specialty labels |
| `results_baselines.py` | §5: zero-model baselines on the Table 2 (public), Table 3 (agentic 100) and physician (13) patient subsets |
| `summ_hint_leak.py` | §5: the score handed over by the "ontology-grounded" summarization prompt |
| `limitations_checks.py` | §6: chart-neutral penalty on documented conditions; case mix; section missingness |
| `appendix_a_checks.py` | App. A: Table A consistency, hallucination-metric sensitivity, specificity/acuity/retrieval floors |
| `appendix_c_checks.py` | App. C: Table C.1 winners and cell count, Table C.2 margins, Spearman null for n=4 |
| `appendix_e_checks.py` | App. E: cluster shape, timeline anchors, removed encounters, provenance columns, profile constraint violations |
| `appendix_f_checks.py <cms.txt>` | App. F: identical-text retrieval grades, summarization negation, imaging indication/differential codes |
| `appendix_g_checks.py` | App. G: locate Panel C; de-identification masks and shorthand in the synthetic corpus |
| `appendix_h_checks.py` | App. H: Table H.1/H.2 consistency, reproduction attempts, comorbidity ORs and their single-vignette origin |
| `appendix_d_checks.py <cms.txt>` | App. D: graph counts; every released ICD code checked against the public CMS FY2025 table |

Status legend: **OK** = claim holds; **PARTIAL** = holds with material caveats; **FALSE** = contradicted by code or data.

---

## 0. Headline

The released artifact is internally consistent: the counts match, the running example matches, the split is
patient-disjoint and scoring is deterministic. But several claims that matter for how results should be read
do not hold, four of the five reward functions can be gamed without reading the chart, and the §5 results are
confounded by label leakage in prompts and in the agent tool API:

| # | Issue | Evidence | Impact |
|---|---|---|---|
| 1 | **Retrieval reward saturates.** P@5 divides by `min(k, len(results))` | one HPI section scores **P@5 0.995** (public); ranking by section type alone scores 0.972; chance is ≈0.69 | the retrieval RL reward carries almost no signal |
| 2 | **Patient diagnosis is mostly copying** | 80.0% of reference dx appear verbatim as `<name> (diagnosed YYYY-MM-DD)` in later notes; a regex copy scores **w-F1 0.843 public / 0.845 held-out** | the task measures extraction plus ICD coding, not diagnosis |
| 3 | **Held-out labels are not private** | all 2,536 held-out rows ship with full GT; also, every retrieval query **equals** the patient-diagnosis answer key (1,268/1,268) | "accessible only through our scorer" is false twice over |
| 4 | **Specialty abstention can be gamed with a phrase** | adding "No significant distress." to any summary makes absent items score **1.0**, with involved items unchanged | 40% of specialty items (1,600 in train) are free |
| 5 | **Future events leak into earlier notes** | surgical history is timeless: 812/935 patients have identical PSH in every note; 6/6 appendicitis encounters already list the appendectomy | outcome leakage; wrong imaging references |
| 6 | **ICD grounding is LLM-first, not deterministic** | `_resolve_icd10` accepts the LLM's code whenever it exists in CMS; 346/2,975 reference dx carry an *unvalidated* LLM code | "LLM codes are only candidates" is false |
| 7 | **No physician-review artifact** | no review table, flag or script exists for patient records | "a physician reviews generated records" cannot be verified |
| 8 | **§4.1 realism study has no artifacts in the repo** | no MIMIC converter, no study records, judgments or analysis, no Epic-like UI (the simulator is API-only), no Appendix H code | the study cannot be audited or reproduced; the reported statistics are internally consistent but underpowered for an "indistinguishable" claim |
| 9 | **§4.2 validates a component that barely touches the labels** | dx–dx edges credit findings in only **117/3,809 (3.1%)** involved specialty items; 15 of the 111 "recovered" pairs are curated edges built *from the reference set's own misses* | "ontology-derived labels agree with physician judgment" is supported for a sliver of one task |
| 10 | **§5 agentic patient-diagnosis gains come from reading the answer key** | the paper's agent harness exposes `view_problem_list` / `open_chart.active_problems`, which return **exactly the reference diagnoses with ICD codes**; echoing that tool scores **0.993** on the Table 3 patients | the +0.14 to +0.34 "self-retrieval helps" result (§5.3) is a label leak, not evidence gathering |
| 11 | **Zero-model baselines beat every model in Table 2 on 3 of 5 columns** | copy the chart's problem list: dx **0.843** vs best 0.732; paste the chart: summarization **0.676** vs best 0.550; section-type ranking: P@5 **0.97** / nDCG **0.81** vs best 0.833 / 0.536 | "no model approaches ceiling" and "retrieval is more mature" are artifacts of the metrics |
| 12 | **The Table 2 summarization prompt contains the labels** | the "ontology-grounded" strategy puts 10 of the patient's key-finding names into the prompt with "do not omit them"; echoing those names alone scores **0.428**, above 5 of the 10 models | the Summ. column is contaminated; few-shot examples are also drawn from the **public** split with their reference answers |
| 13 | **Chart-neutral scoring does not protect chart-documented conditions** | listing the reference **plus** the other conditions the source vignettes document (AKI, septic shock, hyperkalemia…) drops w-F1 from ≈1.0 to **0.662**; 182/200 public patients are penalized | §6's "chart-neutral scoring prevents them from being penalized" is false; as an RL reward it teaches omission |
| 14 | **The hallucination metric cannot detect fabrication** | three invented sentences (metastatic pancreatic cancer, PE, hip fracture) score hallucination **0.000** against every chart; a *different patient's* summary scores 0.023 | App. A's "models rarely fabricate unsupported findings" is unsupported |
| 15 | **Table 2's nDCG@10 is at chance** | a random shuffle of the chart sections scores nDCG@10 **0.524**, MRR 0.831 and MAP@10 0.549; 5/10 models are at or below random on nDCG@10 (best 0.536), and 2/10 on MRR and MAP | the retrieval column does not separate models from chance |
| 16 | **App. B's "graph" condition contains no graph information** | the "graph" prompt is the specialty zero-shot template, whose clinical question is one fixed sentence for all 6,345 items (only the specialty name varies); no comorbidity from the graph is injected. The "117 public" instances are in fact public 26 + heldout 24 + train 67 | App. B measures prompt wording, not whether graph-derived context helps |
| 17 | **Prompt strategy was tuned on the evaluation split, and "structured" wins exactly where it injects the scored items** | the C.1 pilot runs on `split="public"` (`strategy_experiment.py:78`); few-shot examples come from public items; structured wins summarization for all 3 pilot models (+0.07 mean), and its hints are the key findings the metric counts | App. C.1 itself avoids structured for specialty summarization "to avoid leaking ontology-derived relevance labels", yet uses the same mechanism for whole-patient summarization |
| 18 | **The generator-robustness study (C.2) has no code and cannot test the larger bias** | no re-rendering code or data in the repo; only narratives and profiles were re-rendered, while the Kimi-proposed ICD codes and Kimi-authored imaging references were kept as labels; "ordering preserved (ρ ≥ 0.80)" with 4 models has P = 0.167 under a random ordering | App. C.2 does not show the benchmark is free of Kimi advantage |
| 19 | **Failed answers are dropped, not zeroed, on patient diagnosis** | the scorer skips empty predictions (perfect + empty → 1.000); single-turn API failures are never stored | agentic and Table 2 diagnosis scores are over answered items only, which inflates models that fail often; this contradicts "scored as zero rather than excluded" |
| 20 | **App. D overstates ICD grounding and contradicts itself on LLM codes** | table-validated ICD coverage is **61.3%**, not 75.9% (75.9 = validated + the 14.6% unvalidated the caption calls "further"); 24.1% of diagnosis nodes are uncoded. "LLM codes are never accepted on their own" contradicts its own step (i), which accepted 83.2% of mentions on billability alone; e.g. "Leukocytosis" → D72.819 *Decreased* white blood cell count; "Septic arthritis" → B26.85 *Mumps* arthritis | the "verifiable ground truth" rests on unvalidated LLM coding for most labels |
| 21 | **The profile LLM breaks its own stated constraints, and the prose does reach the labels** | 500/1,268 profiles (39%) list one of the patient's **own tested diagnoses** as a chronic condition (the prompt forbids it); the running example's profile adds a "prior appendectomy" before the appendicitis visit. That prose drives the chart-neutral set (Appendectomy → K35–K37 neutral) and the imaging reference ("stump appendicitis given prior surgery") | App. E's "the benchmark labels never touch this prose" is false |
| 22 | **Retrieval grades are not a function of the section text** | 62% of byte-identical sections within a patient (3,614 groups, 16,658 sections) carry **different** grades. E.g. patient 1973's identical PSH, family- and social-history sections are graded 1, 2, 2 across its three encounters | the retrieval label cannot be learned from content |
| 23 | **540 specialty items (14%) are unwinnable as single-item rewards** | involved items with no critical finding score `conditioned_f1 = 0` for **any** answer, including a perfect one (e.g. 1973 Pulmonology). Batch scoring silently skips them for recall, so the benchmark metric and the RL reward disagree by construction | dead RL items; per-item rewards do not average to the reported metric |
| 24 | **The realism-study conversion shown in App. G leaves a perfect tell** | Panel B (the "converted" real note "used during physician review") still carries four `[redacted]` masks; **0 of 5,602** synthetic notes contain a mask. §4.1 said masks were repaired or dropped. Panel C is presented as a "comparable abdominal presentation" but is a urinary one | either physicians saw a trivially detectable marker, which makes the 53% hard to interpret, or Panel B is not what they saw |
| 25 | **App. H's distribution and comorbidity evidence is unreleased, and it holds by construction** | "mapping and analysis code are released": no Synthea, JSD or odds-ratio code exists in the repo. On the release, benchmark vs source gives JSD **0.001** (80% subset), not 0.029. for 7 of the 10 comorbidity pairs, most co-occurring patients (64–96%) have both conditions inside a **single source vignette**. The 9×/4× circulatory/endocrine ratios shrink to 4.3×/2.1× once Synthea's Z-chapter mass is removed | App. H does not show emergent clinical structure. It also calls the benchmark "not a training corpus", contradicting the RL training split of §3.1 |
| 26 | **"Complete provenance" and "narrative errors cannot corrupt the ground truth" (App. I, Table I) are false** | the profile prose (chronic conditions, surgical history, meds) and the rewritten HPIs have no graph links; that prose sets the chart-neutral sets and the imaging references (F§12). The graph itself is LLM annotation of source text, so its errors are labels (D72.819, K35.890, 1,965 answer-as-distractor rows). FHIR `Observation` serves the 20,861 **absent** findings with no negation, dates or units | Table I's "Provenance: Full / ground truth independent of narrative" is SH's central differentiator, and it does not hold |

---

## 1. §3 Benchmark construction

### Running example (patient 1973): OK

Every stated fact matches the DB:
- 58-year-old man with 3 encounters, built from source questions 1382, 2439 and 1539. Source ages were 52, 58 and 58.
- The encounters are linked through shared nodes 524 (type 2 diabetes) and 529 (acute kidney injury).
- Dates are 2020-01 (perforated appendicitis with septic shock), 2020-09 (hyperosmolar state, +8 months) and 2021-05 (ICU hypercapnic respiratory failure, +16 months).
- Question 2439 → E11.01 with 26 typed findings: polyuria is symptom/key and metformin is medication/background.
- Labels match: 3 acute diagnoses weighted 2/2/3, 34 sections with 13 graded 3, 20 must-include findings, and specialty items for Endocrinology, Gastroenterology, General Surgery and Pulmonology plus absent-specialty items for Cardiology and Dermatology.
- The imaging order is "CT abdomen/pelvis, stat: RLQ pain, fever".

The running example itself carries two defects:
- **Encounter 0 (appendicitis) already lists "Appendectomy (complicated by perforation)"** under past surgical history, the outcome of that very visit. The LLM-written imaging reference then rationalises it as *"stump appendicitis given prior surgery"*.
- **Diagnosis 4019 "Acute appendicitis *with* perforation and septic shock" is coded K35.890 "…*without* perforation or gangrene".**

### Stage 1: Knowledge ingestion. OK

- Parsing is rule-based. APKG notes go to `raw_cards` (s01); PDFs go to `fact_cards` with `extraction_method='rule_based'` (s01d:299-330). Provenance is kept through `raw_card_id` / `source_qid`.
- The "rule-based with LLM fallback" classifier (s02) only ever returns `rule_based`, so the LLM branch is dead code. s03's LLM enrichment is skipped (s03:496, 545).
- **Pipeline wiring:** `etl/main.py` dispatches only stages 1–6 (main.py:40-72); stages 7–12 log "not yet implemented". s07–s10f exist only as standalone `python -m` modules. Stage 5's `run()` does only 5a+5b, and Stage 6's `run()` does only 6a. SNOMED, SapBERT, LOINC and the LLM relationship passes need separate CLI flags. The README's "`--stage 1..12`, in order" does not run the pipeline.
- The source material is not shipped, so the ETL cannot be re-run from the release in any case.

### Stage 2: Ontology grounding and knowledge graph. FALSE on determinism

| Claim | Status | Evidence |
|---|---|---|
| Kimi 2.5 extracts primary, differential and secondary diagnoses plus typed findings | OK | `MODEL="kimi-k2.5"` via the unreleased gateway `SH_LLM_GATEWAY_URL` (s05:44-45, 272-283). 9 finding types; relevance is key/supporting/background/distractor |
| "ground deterministically…; LLM-proposed codes are treated only as candidates" | **FALSE** (ICD-10) | `_resolve_icd10` (s05_ontology.py:574-602): (1) if the LLM code exists in CMS and is not a header, accept it as `llm_validated`; (2) exact description match; (3) difflib ratio ≥0.70; (4) otherwise **store the raw LLM code** as `llm_unvalidated`. "Validated" means only that the code exists; nothing checks it against the diagnosis name |
| | data | 1,409/9,623 diagnoses carry a code with NULL description (unvalidated). **346 of the 2,975 reference diagnoses** do, e.g. J44.8, E87.2 and K35.2 (header codes) |
| | data | K35.890 for "with perforation" passes step 1 or step 3 (ratio 0.722). Dedup keys on `icd10:{code}` (s05:605), so distinct diagnoses mapped to one code merge |
| SNOMED CT / LOINC grounding | OK | deterministic. SNOMED discards the LLM ids and re-matches with SapBERT (≥0.70 dx, ≥0.65 findings); LOINC is exact, then component, then fuzzy ≥0.90 |
| a separate LLM pass types diagnosis–finding relations | OK | s06b (Kimi): pathognomonic / highly_suggestive / commonly_seen / risk_factor / protective / rules_out. s06d (dx↔dx edges) is deterministic |
| fact cards linked to graph concepts | OK, but LLM | s06c uses a Kimi prompt, then exact → substring → fuzzy ≥0.80 name resolution. Fact cards are not in the release |
| vignettes segmented into EHR sections "while preserving the source text" | PARTIAL | segmentation is **LLM** (s07:111-130). Verbatim is a prompt instruction only; no substring check (s07:231-252 only logs warnings) |
| "Every node retains its source identifier and grounding method" | **FALSE** | `resolution_method` is computed (s05:692) but never inserted (s05:717-725). `diagnoses` / `clinical_findings` have no method or source columns. The only provenance is `question_*.source`, which is `'llm'` on every row |

Extraction quality notes:
- **1,965 of 7,002 questions (28%)** list the correct diagnosis also as a distractor.
- Example value error: Q2439 records "Fever" as present at 37.2 °C.

### Stage 3: Longitudinal patient generation. PARTIAL

| Claim | Status | Evidence |
|---|---|---|
| deterministic graph clustering before any narrative | OK | Step 8a (s08:721-889) makes no LLM call and uses no randomness. Ties fall to input order |
| demographically compatible | OK, and stricter | same sex; same age bucket (0–17 / 18–35 / 36–60 / 61+) with within-bucket tolerance of 2 / 5 / **7** / 10 years (s08:47-58, 547-573). Smoking and alcohol contradictions are also barred (s08:530-544), which the paper does not state. Ages 60 and 61 can never cluster |
| each added encounter shares a correct or secondary dx with an existing member | PARTIAL, stricter | the candidate must share an exact `diagnosis_id` with the **seed** (s08:673-679); distractors are excluded. Data: 0/1,268 clusters are disconnected |
| constrained greedy; each question in ≤1 patient; 1,268 patients from 5,602 questions | OK | cap by cluster acuity: acute-only 3, chronic-single 5, chronic-multi 8; minimum 2. Sizes: 2→350, 3→355, 4→58, 5→116, 6→29, 7→29, 8→331. 7,003 → 5,602: questions with no sex/age or no compatible partner are **dropped** (no singleton patients). No question is reused |
| Kimi realises profile, timeline and limited HPI continuity; problem and med lists propagated deterministically | PARTIAL | three LLM calls (8b profile, 9a timeline, 9b HPI polish), none with temperature or seed set. The **profile call sees all encounters including later ones** (s08:1159-1196) with no temporal guard. The HPI-polish prompt includes the **current encounter's diagnosis** (s09:164, 1445). Medication merge is deterministic (s09:924-951) |
| "A physician reviews generated records" | **no evidence** | no review table, flag or script for patient records. The only clinician review in the repo covers the specialty map (`etl/ontology/specialty_map.py`) |
| narrative "does not determine benchmark ground truth" | PARTIAL | the timeline LLM **chooses encounter order and dates** (s09:642-711); only 336/1,268 patients keep the seed as encounter 0. Order feeds `first_encounter_date`, the summarization cut, the imaging prior-context window, the `/env` point-in-time cut and the "diagnosed {date}" problem-list lines |

Consequences visible in the data (all from `data_checks.py`):
- **Timeless surgical history:** 812/935 patients have the same PSH text in every encounter, so surgery that treats encounter *k* appears in notes 0…*k*−1. All **6/6** appendicitis encounters already list the appendectomy; **5/12** ectopic-pregnancy encounters already list the salpingectomy.
- **Future diagnoses in the first note:** in 322 patients, encounter 0's PMH names a diagnosis first keyed at a later encounter (47 of them acute).
- **HPI leaks the answer:** in 210/4,417 first-occurrence encounters the HPI names that encounter's own diagnosis.
- **Repetition:** 656/1,268 patients have the same reference diagnosis keyed in more than one encounter; patient 1771 is "diagnosed" with MDD at 7 of 8 visits.
- **Synthetic dates:** every encounter date falls on the 15th of the month.

### Stage 4: Benchmark construction (labels). PARTIAL

**"Labels are deterministic functions of the graph, except the imaging reference": PARTIAL.**
- Every *primary* reward is deterministic. Scoring is identical under `PYTHONHASHSEED` 1 and 2.
- However, the chart-neutral sets are built from the LLM-generated profile text, not the graph.
- The LLM `reference_summary` feeds `rouge_l`, a secondary metric.

**Patient diagnosis**
- **Reference:** the correct-role diagnosis nodes per encounter. Node acuity decides active vs chronic (s10:439-507). Chronic conditions that exist only in the profile (e.g. 1973's diabetes, hypertension and CKD) are not in the reference.
- **Severity weights (scoring.py:196-233):** CRITICAL=3 for a hard-coded list of 3-char codes, or for active diagnoses in ICD letters {A,B,C,D,I,J,S,T}; MODERATE=2 otherwise when active; ROUTINE=1. The weight depends on the first letter, not clinical severity: septic shock coded K35 gets 2.
- **Matching (`_match_icd10_sets`):** exact code first, then **3-char category** (5-char for S/T). SNOMED and names are not used. Bare `E11.9, K35, J96` scores **1.0** on patient 1973, and chronic "T2DM" E11.9 takes full credit for acute HHS E11.01.
- **Chart-neutral rule:** an unmatched prediction whose 3-char category is in the patient's neutral set is dropped from the precision denominator; recall is unchanged. The set comes from **profile** chronic conditions, surgical history (regex, e.g. Appendectomy→K35–K37 + Z90 + Z98), smoking and alcohol (`scripts/chart_neutral_sets.py`). So "documented in the chart" means *profile-documented*. Listing every profile condition is free: precision_neutral is 1.0 for the chronic-only policy.
- **F1:** `weighted_problem_list_f1_neutral` = HM(mean severity-weighted recall, mean **unweighted** chart-neutral precision). Predicted acuity never enters it.

**Evidence retrieval**
- **Grading ignores section content.** `_grade_section` (s10:1452-1493) grades a section from the key findings of its encounter's questions whose `ehr_section` or finding type maps to that section *type* (`SECTION_FINDING_TYPE_MAP`). The section text is never read, and every stored rationale is just `section_type=<type>`.
- **Grades are type-driven:** train mean grade is HPI 2.58, imaging 2.37, exam 2.26, … medications 0.73, allergies 0.68. The PMH, PSH, family and social history sections are near-identical (1.88–1.89).
- **Relevance is the majority class:** 69.2% of public sections are graded ≥2, which is the P@5 relevance threshold.
- **Baselines (public / held-out):** section-type prior P@5 **0.972 / 0.969**, nDCG@10 0.805 / 0.806. A single HPI section gives P@5 **0.995** (nDCG@10 0.199).
- **The query leaks the other task's labels:** it is the display names of the patient's reference diagnoses (`eval/tasks/retrieval.py:91`; `/env` passes them as `diagnosis_names`, env_service.py:100-106). It equals the patient-diagnosis reference for **1,268/1,268** patients.
- **Stale stored metadata:** the GT's `num_passages: 120` / `grade_distribution` still counts fact cards, although only 34 section judgments ship for 1973.

**Context summarization (whole-patient)**
- **Selection:** must-include = key findings in chronological order, name-deduplicated, **capped at 20** (s10:775-792). 919/1,268 patients hit the cap.
- **Latest visit drops out:** in 228 patients no must-include name comes from the most recent encounter. By the stricter key-finding definition it is 395, and 667 have no finding *first introduced* at the final encounter. For 1973, nothing from the ICU visit (the current problem) is required.
- **"clinical_f1" is recall** (scoring.py:571-580). The matcher is abbreviation-expanded substring or ≥70% token overlap, with **no negation guard**: `phrase_in_text("Fever", "Patient denies fever.") → True`.
- **Lexical ceiling:** pasting the entire chart scores only **0.676**, because names like "Severe hyperglycemia" do not appear literally in "Serum glucose: 680 mg/dL".
- **The question is constant:** the "clinical question" is identical for 1,267/1,268 items.

**Specialty-conditioned summarization**
- **Ownership:** ICD chapter → home specialty (`specialty_map.py:186`).
- **Tiers:**
  - primary: owned by the specialty.
  - relevant: Class-1 or curated edge to an owned diagnosis.
  - neutral: Class-2/3 edges only; ignored in scoring.
  - excluded: no link, sampled ≤15.
  - `tau_residual=0.5` is stored in every row but unused.
- **Absent items are not sampled.** They are the **first two alphabetically** among non-involved specialties (s10f:77). Every patient therefore has Cardiology and Dermatology items, which are absent 77% and 94% of the time. The item's name alone predicts abstention.
- **Involved items:** `conditioned_f1` = HM(critical-only primary∪relevant recall, 1 − leakage over the ≤15 excluded samples).
- **Absent items:** binary `_is_abstention` is true if the output is ≤12 words **or contains** any of "no significant", "no active", "not applicable", … (scoring.py:1014-1028).
- **Exploit:** chart dump + "No significant distress." → abstention **1.0** with conditioned_f1 unchanged (0.435).

**Imaging indication**
- **What the LLM author sees:** the **full profile JSON** (future surgical and chronic history), this encounter's sections including assessment and plan, and prior chief complaints (last 5). It is given **only the first** correct diagnosis (s10:1735-1747).
- **Concept F1 (`eval/imaging_concepts.py`):** graph names + SNOMED descriptions + curated synonyms, matched on phrase windows with fuzzy snap at 0.86. Deterministic.
- **Baselines (public):** restating the terse order scores **0.217**; the chief complaint scores 0.179.

### Stage 5: Domain-faithful simulation. Not yet exercised

- **Not run:** the Docker daemon was not running during the audit, so FHIR, RBAC and the `/env` loop were not exercised.
- **Label exposure:** the README concedes that the raw tool API (`view_problem_list`, `open_chart.active_problems`, FHIR Condition) returns the diagnosis labels, and that the paper's `eval/agents` harness left them exposed.
- **Default secret:** the scorer token defaults to the public string `dev-scorer-token-change-in-production` (`epic_sim/app/config.py:33`).

---

## 2. §3.1 Evaluation splits

| Claim | Status | Evidence |
|---|---|---|
| patient-level, disjoint, instances inherit the split | OK | 0 patients span splits (imaging rows joined via encounter) |
| public 200 / 1,859; train 800 / 7,619; held-out 268 / 2,536 | OK | exact |
| public "stratified by patient difficulty and encounter count" | OK, not reproducible | `eval/split.py` (seed 42, difficulty × {2-3, 4-5, 6+}). It reads the **retired `diagnosis_accuracy` task**, so it cannot run on v1.3. Released marginals match: public easy/med/hard 27/34/40% vs train 28/33/40% |
| held-out "labels accessible only through our scorer" | **FALSE** | all 2,536 held-out GT rows ship in `benchmark_v1.3.db` (the README concedes this). No split gating exists in `/score` or `/env`. The retrieval query reveals held-out diagnosis labels, and `/score` returns full per-metric breakdowns, which allow label probing even if the DB were withheld |
| train/held-out stratified by dominant ICD-10 chapter × encounter count, "within one percentage point" | PARTIAL | `scripts/build_rl_split.py` (seed 20260922). "Chapter" is the **first letter of the ICD code**, not the ICD-10 chapter (e.g. D spans two chapters). Largest marginal gap is **1.55 pp** (4–5-encounter bucket: 13.4% train vs 14.9% held-out); letter marginals are within 0.9 pp. `data/rl_split.json` is not shipped, and the script needs Postgres |
| no chart or source material crosses splits | OK | each question is used exactly once |
| 54% of held-out dx occur in training, 46% unseen | OK, but misleading | 414/761 = **54.4%** at `diagnosis_id`. The scorer matches at 3-char ICD, where **87.0%** of held-out categories are seen in train. Instance-weighted, 61.1% are seen. "Disease-level generalization" under the actual metric covers ~13% of categories |
| each task yields a deterministic score in [0,1] "from the underlying graph" | PARTIAL | deterministic (hash-seed test) and clamped to [0,1] (score_one.py). Not all from the graph: the imaging reference is LLM-authored and the chart-neutral sets come from profile text |
| 7,619 instances = distinct starting states over 800 patients; ≈30,000 trajectories at k=4 | OK arithmetically (30,476) | only 800 charts. 4,037 (53%) are specialty items differing only by specialty name, and 1,600 of them are absent items solvable by the abstention phrase |
| chart-neutral scoring throughout, including as training reward | OK | `PRIMARY_METRIC["patient_diagnosis"]="weighted_problem_list_f1_neutral"` (score_one.py:38). Implication: an RL policy learns that listing every profile condition is free |

---

## 3. Table 1

| Row | Counts | Status | Notes |
|---|---|---|---|
| Patient diagnosis: "Severity-weighted F1", output ICD-10 + acuity | 200/268/800 OK | PARTIAL | only recall is severity-weighted; precision is unweighted and chart-neutral. **Predicted acuity is not scored** by the primary metric. Matching is at 3-char category |
| Summarization: "Finding-level F1", input clinical question + EHR | 200/268/800 OK | **FALSE** (name) | the metric is must-include **recall**, with no precision term. The clinical question is constant |
| Specialty summarization: "Specialty-relevance F1" | 983/1,325/4,037 OK | PARTIAL | 40% of items (absent) use binary abstention accuracy, not F1. F1 on involved items uses critical-only recall and leakage over ≤15 sampled findings |
| Evidence retrieval: P@5, nDCG@10, input "Diagnosis + patient record" | 200/268/800 OK | metric broken | see §1: P@5 0.995 with one section. The "diagnosis" input is the patient-diagnosis answer key |
| Imaging indication: "Question concept F1", output question + pre-read | 276/407/1,182 OK | PARTIAL | the pre-read is not in the primary metric. The reference is LLM-authored from a context that includes future history |
| Total 12,014 | OK | | |

Appendix A (secondary metrics) and Appendices D–F were not available to check.

---

## 4. Zero-model baselines (the repo's own scorer)

| Task (primary metric) | Policy (reads no chart unless stated) | Public | Held-out |
|---|---|---|---|
| Patient diagnosis (w-F1 neutral) | profile chronic conditions only | 0.352 | 0.324 |
| | regex-copy "(diagnosed …)" lines, names→codes via the DB | 0.828 | 0.838 |
| | both | **0.843** | **0.845** |
| Evidence retrieval (P@5) | rank by section type prior | **0.972** | 0.969 |
| | submit one HPI section | **0.995** | |
| Summarization (clinical_f1) | paste the entire chart | 0.676 | |
| Specialty, involved (conditioned_f1) | paste the entire chart | 0.435 | |
| Specialty, absent (abstention acc.) | chart + "No significant distress." | **1.000** | |
| Imaging (concept F1) | restate the terse order | 0.217 | |

The copy baseline uses the benchmark's name→ICD table, so it is an upper bound for "copy + perfect coding".

---

## 5. §4 Assessing benchmark realism

### 4.1 Realism: physician real-vs-synthetic study. Not verifiable from the repo

**No artifact exists** for any part of the study:

| Stated component | In repo? | Evidence |
|---|---|---|
| deterministic MIMIC-IV → Synthetic Hospital converter (section parsing, date shift, name substitution, lab/med/vitals re-rendering, radiology→impression) | **no** | `grep -ri mimic` over code returns nothing |
| harmonization applied to both arms (drop synthetic social history; drop real demographics and A&P; chart-length budget) | **no** | |
| the 5 real + 5 matched synthetic records, the matching procedure (domain, sex, age, chart length) | **no** | which 5 synthetic patients were shown, and how they were chosen, is unknown |
| "our Epic-like interface" used by the physicians | **no** | `epic_sim` is a JSON/FHIR API (`routers/epic.py` etc.); no HTML, templates or static front end |
| the 100 judgments, confidence ratings, analysis code | **no** | no Fleiss/binomial/bootstrap code anywhere |
| Appendix G examples; Appendix H (JSD, Spearman, comorbidity structure, Synthea comparison) | **no** | no Synthea or JSD code |

Checks I could run:
- **The reported statistics are internally consistent** (`realism_checks.py`):
  - 53/100 gives a two-sided binomial p of **0.617**.
  - Sensitivity 52% (26/50) and specificity 54% (27/50) imply 53% accuracy and "real" chosen in 49% of judgments.
  - The best rater at 9/10 has a two-sided p of 0.0215, which is **0.215** after Holm with 10 raters. That matches the paper's 0.22 only as a two-sided test; one-sided it would be 0.107.
- **The "18 section types" do not match the code.** The synthetic renderer has **17** (`s09_encounters.py:69-73`). The 18th is presumably `demographics`, which the note assembler never emits (dead code, see §1 Stage 3). The paper says harmonization removed demographics only from the *real* arm.
- **The section's title contradicts its first finding.** The title says "synthetic records are **in**distinguishable from real", but the first finding reads "Synthetic and real records are **not reliably in**distinguishable". The latter says the opposite of the intended claim (presumably "not reliably distinguishable").

Design limits of the evidence (these are statistics, not code, but they determine what the claim can bear):
- **The sample is 10 records, not 100 judgments.** Every physician rated the same 10 records, so the judgments are not independent. A judgment-level bootstrap CI (43–63%) ignores that clustering.
- **Absence of evidence is not equivalence.** At n=100, only accuracy ≥61% would reach p<0.05, and the reported CI does not exclude 63% accuracy. No equivalence test (e.g. TOST) is reported.
- **The real arm was reformatted into the synthetic schema.** This is deliberate, and it means the test compares synthetic charts against *synthetic-formatted* real charts. That is a test of clinical content only; it says nothing about the benchmark's surface realism.
- **Selection is unknown.** The data audit in §1 found systematic tells in a *random* chart that a curated five could avoid:
  - every encounter date falls on the 15th;
  - surgical history is identical across encounters and includes future operations;
  - the same diagnosis is "new" at up to 7 visits;
  - encounter-0 notes name later diagnoses.
  
  Whether the real arm's shifted dates also landed on the 15th is not stated.
- **One rater population across studies.** 10 physicians rated realism, while §3's record review and §4.2's reference set each cite "a (licensed) physician". Rater identity and overlap are not reported.

**Appendix H distribution claim** ("preserves the ICD-10 chapter distribution of its source corpus, JSD = 0.029, ρ = 0.83"):
- **Not reproducible, and the released data gives very different numbers.** Comparing the 5,602 used questions with all 7,002 extracted questions (true ICD-10 chapters) gives **JSD 0.0013 (base 2) and Spearman 0.991**.
- The paper's "source corpus" must therefore be something not shipped, such as the raw card corpus.
- Either way, the test is weak: the benchmark is an 80% subset of its source, so near-identical marginals are expected by construction.

### 4.2 Verifiable ground truth: diagnosis-relationship reference set. PARTIAL, with circularity

The workflow is in the repo (`scripts/comorbidity_coverage.py`, `scripts/decide_overlap_analysis.py`,
`etl/stages/s06d_diagnosis_relations.py`), but none of the inputs or outputs are:
`comorbidity_coverage_list.md` (the 119 pairs), `comorbidity_coverage_report.csv`,
`curated_diagnosis_edges.csv`, `finding_site_review.csv`, `data/benchmark_v1.2_copy.db`, and the
`diagnosis_relations` table (absent from `benchmark_v1.3.db`). **111/119 cannot be recomputed.**

| Claim | Status | Evidence |
|---|---|---|
| pre-registered set of 119 required relationships, expected type assigned "before graph construction" | unverifiable | the script calls the list "pre-registered" (comorbidity_coverage.py:1). It reads `benchmark_v1.2_copy.db`, so the list post-dates the v1.2 diagnosis nodes. Git history is squashed (2 commits), so timing cannot be checked |
| 96 recovered through SNOMED relationships, shared anatomical sites and shared findings | PARTIAL, lenient scoring | **Hit criterion:** each side of a pair resolves to a *set* of diagnoses via ICD prefixes and keyword substrings. Some sets are very broad: "cancer"/"malignant" → every `C` code, "pregnancy" → every `O` code, "congenital anomaly" → every `Q` code, "anemia" → D50–D64. A hit is **any** edge between **any** member of A and **any** member of B (`best_class`, comorbidity_coverage.py:302-313). `HIT(stronger)` also counts. **Finding-site rows** pass if either side has a `shared_finding_site` edge with *anything*, not with its partner (`fs_hit`, :334) |
| 15 recovered "through physician-curated edges" | **circular** | the script writes every MISS/WEAKER row to `curated_diagnosis_edges_TEMPLATE.csv` for the clinician to fill (:5-10, 350-372). The builder then adds those rows as `source='curated'` edges (s06d:307-343). The 15 are the reference set's own misses, patched and re-counted as recovered. **Honest pre-curation recall: 96/119 = 81%**, and that uses the lenient criterion above |
| curated edges are targeted | no | each filled row expands to an ICD-prefix × ICD-prefix **cartesian product** of diagnoses (s06d:333-341), so one curated pair can create many gold edges. If the CSV is missing, the builder **silently** adds none (s06d:308-309), which changes labels on rebuild with no warning |
| 8 accepted gaps, not recovered by lowering the shared-finding threshold | OK | `decide_overlap_analysis.py` hard-codes exactly 8 DECIDE rows {26, 49, 71, 79, 96, 101, 103, 117} and computes their residual overlap |
| links are "ontology-derived" | PARTIAL | Class-1/2 edges come from SNOMED. Class-3 "shared findings" edges are weighted by the **LLM-typed** diagnosis–finding relations (s06b; `_DF_REL_WEIGHT`), and curated edges are manual |
| this validates that "ontology-derived labels agree with physician judgment" | **overreach** | see below |

**How much the validated component moves the labels** (released data; `realism_checks.py`):
- **Only definitional and curated edges earn credit.** In `s10f_specialty.py:103-114`, only Class-1 (`due_to`) and curated edges make a finding *relevant*, i.e. credited. Class-2 (associated_with, after, pathological_process, shared finding site) and Class-3 (shared findings) edges only put findings in the **neutral** tier, which is neither credited nor penalized. So most of the "96 recovered via SNOMED relationships, shared sites and shared findings" do **not** make any finding count as relevant.
- Across all 3,809 involved specialty items:
  - The relevant tier is non-empty in **117 (3.1%)**; the neutral tier in 172 (4.5%).
  - Finding counts by tier: primary **48,627**, relevant 1,270, neutral 1,653.
  - Only **212** critical-importance relevant findings exist in the whole benchmark, and these are the only relevant findings that enter the headline `conditioned_f1`.
- **The labels rest on something §4.2 does not validate.** Specialty labels are driven almost entirely by the **ICD-chapter → specialty ownership map** (`specialty_map.py`). That map had a separate "second-clinician review" (specialty_map.py:11-34), which the paper does not report.

**Label validity that §4.2 does not cover** (the heading claims labels in general):
- **Nothing validates the other tasks' labels:** reference diagnoses and their ICD codes, acuity, retrieval grades, must-include findings and imaging references.
- **The repo's own ICD audit shows known wrong codes shipped:**
  - `scripts/audit_miscoding.py` flags diagnoses whose ICD chapter disagrees with the SNOMED→ICD crosswalk.
  - `scripts/write_dup_note.py` records **352 corrections, of which 264 were applied and 88 were blocked** by a uniqueness constraint. Those 88 were *left with their known-wrong codes* ("Decision (2026-06-07)").
  - Neither audit is mentioned in §4, and its CSVs are not shipped.
- **Collateral evidence of label noise:** K35.890 "without perforation" on a perforated appendicitis (the running example), and 346 reference diagnoses with unvalidated LLM codes (§1).
- **False-positive relations are unquantified**, as the paper says. They matter for scoring: a spurious Class-1 or curated edge turns an off-target finding into credited "relevant".

## 6. §5 Results

Model outputs, run logs and the physician-study export are **not in the repo**:
- `results/` and `data/` are gitignored;
- `scripts/retrieval_sections_only.py` reads run ids from an unreleased `evaluation_predictions` table and `gui/clinical-eval-platform/blob_pull_content.json`;
- there is no Appendix A or C.

So Tables 2 and 3 cannot be re-derived without re-running the models. What can be checked is:
1. the protocol in code;
2. the arithmetic of the tables;
3. how zero-model baselines score on the **same patient subsets** (`results_baselines.py`).

### 5.1 Experimental setup

| Claim | Status | Evidence |
|---|---|---|
| 10 models | OK, with naming drift | `eval/config.py:45-192`. The code's "all models" list has **11** entries, including `kimi-2.5-improved` and `kimi-2.5-thinking-improved` (`strategy_experiment.py:29-33`). Their `prompt_revision: "v1.31"` flag is **read nowhere**, so "improved" differs only in name. Kimi is served from the authors' private gateway; the other nine go through OpenRouter. Temperature is 0.0 and there is one run per model, with no seeds and no CIs |
| 4 strategies compared on 3 pilot models "spanning the panel's strength range"; best per task locked | PARTIAL | pilot models are GPT 5.3, DeepSeek V3.2 and Mistral Large, 75 items each (`strategy_experiment.py:28,35`). By Table 2's own mean rank they sit at #1, #4 and #8 of 10; the two weakest (Llama 4 Scout, Gemma 3) are not represented |
| locked strategy for summarization = "ontology-grounded" | **conflict** | the code's `LOCKED_STRATEGIES` has `context_summarization: "few_shot"` (`strategy_experiment.py:38-43`). The paper, README and Table 2 caption say ontology-grounded. Either way the column is contaminated (next row) |
| strategies are fair comparisons | **FALSE** | the "structured / ontology-grounded" prompts inject **graph labels**. Summarization gets 10 key-finding names with "Your summary must address each of these findings… do not omit them" (`summarization.py:246-269`, `hints.py:181-199`); must-include names are 100% key findings. Imaging gets the question's **correct** diagnosis among the "differentials" (`imaging.py:109-146`, role IN ('correct','distractor')). Patient diagnosis gets the answer's organ systems. Any ablation where "structured" wins is confounded by label leakage |
| few-shot examples | **contaminated** | selected "by model agreement" from `split = 'public'` zero-shot runs (`examples.py:108-175`). The shipped frozen examples (`eval/few_shot_examples.json`) are **public-split patients** 1761/1937/1942 (dx), 1698/1682/1715 (summ) and 2000/1690/1748 (imaging), with their **reference answers** as "Expected output". Test items appear in the prompt, and selection used test-set model performance |
| Kimi is a neutral panelist | **no** | Kimi 2.5 **built the benchmark**: it proposed the ICD codes that were accepted as labels (§1 Stage 2), wrote the profiles, timelines and HPIs, and **authored the imaging reference questions** scored by concept F1 (`s10_ground_truth.py` imports `MODEL` = `kimi-k2.5`). Its #1 patient-diagnosis score (0.732) and #2 imaging score are therefore partly self-agreement. Its Table 2 retrieval row also comes from run **21**, while every other model is from runs 79–119 (`retrieval_sections_only.py:17`), i.e. a much earlier pipeline state |

### 5.2 Single-turn evaluation (Table 2)

**Arithmetic:** OK.
- Mean rank reproduces from the printed values using **P@5** as the retrieval column; nDCG@10 is not ranked.
- 8/10 rows match exactly. Gemini (4.50 vs 4.40) and Qwen (5.90 vs 6.00) differ only by the P@5 tie at 0.809, which unrounded values would break.
- "Top two separated by 0.03, top four by 0.06" and "three different models achieve the best score" both check out.

**The headline readings do not survive zero-model baselines** (the repo's scorer on the Table 2 patients; no model, no chart reading unless stated):

| Column | Best model | Physicians (13 pts) | Zero-model baseline, public 200 | Same baseline, physicians' 13 pts |
|---|---|---|---|---|
| Patient dx, sev.-wgt. F1 | 0.732 (Kimi) | 0.664 | regex-copy the chart's problem list + profile: **0.843** | **0.853** |
| Summ., "Find. F1" | 0.550 (Opus) | 0.051 | paste the whole chart: **0.676**; echo the 10 hinted names: 0.428 | 0.638 |
| Spec. Summ., "Rel. F1" | 0.680 (Opus) | – | chart dump: involved 0.435; +"No significant distress.": absent **1.000** | – |
| Retrieval P@5 | 0.833 (GPT) | 0.888 | rank by section type: **0.967–0.972**; one HPI section: 0.995 | **0.938** |
| Retrieval nDCG@10 | 0.536 (GPT) | 0.505 | rank by section type: **0.805** | **0.787** |
| Imaging concept F1 | 0.518 (GPT) | 0.282 | restate the terse order: 0.217 | 0.185 |

Claim by claim:
- **"No model approaches ceiling … the benchmark meaningfully separates systems": FALSE as stated.** Three of five columns have a no-model floor *above* every model. The distance from 1.0 reflects metric construction, not clinical difficulty:
  - summarization is lexical name recall, and the chart itself scores only 0.68;
  - retrieval grades come from section type;
  - diagnosis is copyable.
- **"Frontier compression; no measurable gains in diagnostic accuracy": unsupported.**
  - Diagnosis is 80% copyable, so the column measures extraction plus ICD formatting.
  - Kimi, which proposed the reference codes, leads.
  - The 0.03 top-two gap has no CI. The patient-level bootstrap SD of the metric at n=200 is **0.014** for a single system (copy baseline), so the gap is about 1.5 unpaired SEs.
- **"Performance tracks task difficulty; locating evidence is more mature": FALSE.**
  - Every model's P@5 (0.741–0.833) is **below** a content-blind section-type ranking (0.967–0.972), and chance is 0.69.
  - nDCG@10 is 0.48–0.54 for models vs 0.81 for the type prior.
  - The retrieval column is 1/5 of the mean rank, so the rank partly measures proximity to a type prior.
- **Physician reference: PARTIAL.**
  - It uses 7 physicians on 13 public patients against models on all 200. The table does not match the subsets; the script computes a matched version but the table does not use it.
  - Physicians 3 and 17 are excluded in code (`retrieval_sections_only.py`: `physicianId in (3, 17)`), and the paper does not report the exclusion.
  - On the same 13 patients, the regex copy (0.853) beats physicians (0.664) at diagnosis.
  - Physicians' summarization score of 0.051 against models' ~0.5 shows the metric rewards listing finding names, not clinical summarization.
- **"Imaging is scored by concept F1 over graph-linked diagnoses and findings": OK.** The references are Kimi-authored.

### 5.3 Agentic evaluation (Table 3)

**Harness facts** (`eval/agents/`):

| Claim | Status | Evidence |
|---|---|---|
| 13-tool Epic-style API, 40-action budget | OK | 13 tools in `epic_sim/app/routers/agent.py:38-179`; `DEFAULT_BUDGET = 40` (`runner.py:50`) |
| three conditions | OK | "context" arm = chart preloaded into the first message, **tools still available** (`runner.py:262-266`); self-retrieval = patient id only; LLM = single turn |
| "the same 100 public-split patients" | **biased sample** | `select_patients` sorts public patients by **descending encounter count** and takes the first 100 (`runner.py:154-156`), after excluding "ceiling" patients that ≥7 models solved in earlier zero-shot runs (`runner.py:133-150`; needs the unreleased runs table). Before the exclusion, the 100 are **52 eight-encounter charts** + all 4–7 + 11 three-encounter charts and **no** two-encounter patients (public: 59 two-encounter, 52 eight-encounter). The default `split="val"` no longer exists in v1.3 (`runner.py:62`) |
| outcome sections hidden | PARTIAL | assessment/plan are stripped **client-side** (`structured_agent.py:17-37`). There is **no future-encounter filter** for imaging agents (the `/env` endpoint has one: `env_service.py:402-403`; the paper's harness does not) |
| failed sessions scored 0; token usage (145k vs 60k) | unverifiable | no session logs shipped |

**The answer key is a tool.**
- `view_problem_list` and the `active_problems` field of `open_chart` call `epic_service.get_problem_list`. It returns the patient's `role='correct'` diagnoses with ICD-10 and SNOMED (`epic_service.py:297-332`), which **is** the patient-diagnosis reference.
- The README's "Known issues" admits the paper's agentic scores through this harness "are inflated".
- Echoing that tool's output scores **0.993** on the Table 3 patient rule (`results_baselines.py`).
- Consequences for §5.3:
  - **"Self-retrieving agents outperform the single-turn LLM for all three models (+0.14 to +0.34)": confounded.** One `open_chart` call reveals the labels. Llama 4 Scout's 0.894, higher than any single-turn model in Table 2, is what copying that list looks like.
  - **Llama's +0.240 "loop effect" on diagnosis is also confounded.** The full-context agent has the same tools.
  - **"Multi-turn interaction … can help when evidence gathering is itself central to the task": unsupported** by this experiment. The only positive cell is the leaked one.

**Other checks:**
- **Arithmetic:** every loop and self-retrieval effect equals the difference of its row to ±0.001 (rounding). "Losses of 0.07–0.19 on summarization and retrieval": OK (0.074–0.187). "Within ±0.03 in five of nine pairs": OK (−0.024, +0.006, −0.024, +0.016, −0.005).
- **Table 3's single-turn summarization cells** (0.342 / 0.260 / 0.255) sit **~0.15 below** the same models in Table 2 (0.489 / 0.416 / 0.403). The other three tasks agree within 0.02 (dx 0.705 vs 0.703, retrieval 0.826 vs 0.833, imaging 0.519 vs 0.518). This is consistent with Table 2's summarization prompt carrying the label hints (5.1) and Table 3's plain chart not carrying them. The patient sets also differ (100 long charts vs all 200).
- **Other same-subset floors** (Table 3's 100 patients): regex copy dx **0.865** (above all three single-turn LLMs, 0.55–0.71); chart-dump summarization 0.688 (above every condition, max 0.342); section-type P@5 **0.984** (above every condition, max 0.826); restate-the-order imaging 0.200.

## 7. §6 Conclusion and limitations

| Stated limitation or claim | Status | What the code and data show |
|---|---|---|
| physician study is small and record-level, not population-level | OK as stated | it also has **no artifacts in the repo** (§5 of this doc), which the limitations do not say |
| education-derived case mix may under-represent rare or atypical presentations | PARTIAL | the risk runs the other way on rarity: board questions over-sample "zebras" (e.g. 100 Medical Genetics specialty items). Presentations lean textbook: key findings are typed pathognomonic 3.2%, highly_suggestive 26.6%, commonly_seen 55.9%. **Unstated and more important:** the source tags read `GPT4Anki::Subject::System::…` (`s03_extract_board.py:151-152`), i.e. an LLM-generated Anki deck. If that deck is public, frontier models may have trained on the very vignettes. There is no contamination analysis |
| does not simulate missingness, contradictions or documentation errors | misleading | **Missingness occurs naturally but is unlabeled:** vitals appear in 59% of encounters, labs 67%, imaging 33%, ROS 8%. **Contradictions and errors occur by accident, uncontrolled and unlabeled:** timeless surgical history listing future operations; the same diagnosis "new" at up to 7 visits; 1,965 questions with the answer also a distractor; K35.890 "without perforation" on a perforated appendix; encounter-0 notes naming later diagnoses. The v1.3 change notes themselves describe removing "self-contradictory medication and history statements" from 11,446 sections. The honest statement is "contains unlabeled generation errors" |
| agentic evaluation: 3 models, 3 conditions, 100 patients, one scaffold | incomplete | omits the three issues that invalidate its main positive result: the scaffold's `view_problem_list` / `open_chart` return the diagnosis labels; the 100 patients are the longest charts after a ceiling exclusion; and imaging agents are not time-restricted |
| graph-derived rewards are "verifiable but not exhaustive"; chart-documented conditions absent from the reference "receive no additional reward, although chart-neutral scoring prevents them from being penalized" | **FALSE** | the neutral set covers only **profile** chronic conditions, surgical history, smoking and alcohol (`scripts/chart_neutral_sets.py`). Conditions documented in the *encounters* are not covered, e.g. the graph's `secondary` diagnoses (1973: AKI, septic shock, lactic acidosis, hyperkalemia). A prediction of the reference **plus** those secondary conditions scores **w-F1 0.662** (precision 0.495) on public, with **182/200** patients penalized, 4.6 penalized conditions each (`limitations_checks.py`). As a training reward this teaches the policy to under-report documented conditions |
| "verifiable ground truth" | PARTIAL | deterministic scoring, yes. But the labels include LLM-accepted ICD codes (346 unvalidated reference codes), 88 known miscodings deliberately kept (`scripts/write_dup_note.py`), and LLM-authored imaging references |
| "open, reproducible foundation" | PARTIAL | open data and scorer, yes. Not reproducible: the source content is not shipped; the LLM gateway is private; `etl/main.py` runs stages 1–6 only; the curated / finding-site / miscoding CSVs, the 119-pair list and `diagnosis_relations` are not shipped (and missing curated CSVs are **silently** ignored); split scripts need the retired `diagnosis_accuracy` task and Postgres; no model outputs or physician data |
| "without relying on restricted patient data" | OK | for using the benchmark. The §4.1 realism study itself used MIMIC-IV, which is credentialed access |

**Material limitations missing from §6** (each established earlier in this doc):
- **Label leakage:**
  - diagnosis labels are copyable from later problem lists (80%);
  - the retrieval query equals the diagnosis answer key;
  - the agent tool API returns the labels;
  - the summarization prompt strategy injects scored findings;
  - few-shot examples come from the public split.
- **Reward exploits:** single-section P@5, the abstention phrase, and the recall-only summarization metric.
- **Held-out labels are shipped.**
- **The benchmark-building model (Kimi 2.5) is also a ranked panelist.**

See `audit/ROADMAP.md` for what the fork should change.

## 8. Appendix A: Secondary metrics (Table A)

No model outputs ship, so Table A cannot be recomputed. What can be checked is each metric's definition and
what it is able to detect, the table's arithmetic, and the floors on the same data (`appendix_a_checks.py`).

### Metric definitions (eval/scoring.py)

| Column | Definition in code | What it can and cannot measure |
|---|---|---|
| Patient Dx **ICD Spec** | mean shared-prefix / max-length over **matched pairs only**; a reference whose 4th character is `9` gets full credit (`_icd10_specificity_score`, :304-319) | predicting only 3-char categories still scores **0.731** (floor); models 0.824–0.940. An over-general code is already credited by the primary F1, which matches at 3 characters |
| **Acuity** | accuracy over matched pairs; missing predicted acuity defaults to `acute`; unspecified reference acuity → acute | reference mix is 47% acute / 53% chronic, so the majority-class floor is ≈0.53 (on matched pairs, which may differ); models 0.654–0.798 |
| **Prec / Rec** | caption says chart-neutral precision; recall unlabeled | Table 2 F1 is 0.002–0.011 **below** HM(Prec, Rec) for every model. So "Rec" is not the severity-weighted recall that Table 2 uses (likely unweighted), and the columns do not compose to Table 2 |
| Summ. **Omiss.** | `omission_rate = 1 − clinical_f1` (:583-585) | **identical by definition to 1 − Table 2 Summ.**: Table 2 + Omiss. = 1.000 for all 10 models. It adds no information |
| Summ. / Spec. **Hall.** | a sentence is hallucinated if <15% of its whitespace tokens appear **anywhere** in the patient's chart (:588-615), stopwords included | **cannot detect fabrication:** invented sentences score **0.000**; another patient's reference summary 0.023; the correct summary 0.000 |
| Spec. **Omiss.** | `1 − primary_recall_complete` (all primary findings, not only critical) | OK as defined |
| Spec. **Leak.** | negation-aware recall of the **≤15-finding `excluded_sample`** | measures leakage against 15 sampled findings, not the whole off-specialty set. It rewards terse output: Llama has the lowest leakage 0.087 **and** the highest omission 0.794 |
| Spec. **Abst.** | ≤12 words, or contains a stock phrase | exploitable: "No significant distress." appended to any text → 1.000 (#4) |
| Retrieval **MAP@10 / MRR** | binary relevance at grade ≥2 | random shuffle: **MRR 0.831, MAP@10 0.549** (P@5 0.694, nDCG@10 0.524). Models: MRR 0.811–0.906, MAP 0.521–0.588 |
| Imaging **DiffCov** | fraction of the (Kimi-authored) reference differentials matched by exact ICD **or** name substring in either direction | lenient: a generic prediction like "appendicitis" matches "stump appendicitis with perforation" |
| Imaging **FindRec** | a reference finding counts only if its **whole phrase** is a case-insensitive substring of the joined prediction list (:780-793) | reference findings are LLM phrases, median 6 words (e.g. "Periappendiceal fat stranding and fluid"). The 0.09–0.22 range reflects string matching, not clinical recall |

### Claims

| Claim | Status | Evidence |
|---|---|---|
| "high-recall models achieve recall by over-generating diagnoses with lower precision" | **not supported** | across the 10 models corr(Prec, Rec) = **+0.85** (+0.35 excluding Gemma). Kimi ties for the highest recall (0.787) **and** has the highest precision (0.687). Only Gemini and Opus fit the pattern |
| DeepSeek has the highest acuity accuracy without leading either F1 | OK | 0.798. The floor is ≈0.53, and accuracy is computed on each model's own matched subset, so denominators differ across models |
| "MRR is uniformly high … all models place at least one relevant passage near the top" | true but vacuous | random order gives MRR 0.831; DeepSeek (0.811) and Llama (0.829) are **below random** |
| "the gap is in grading the rest of the list, captured by NDCG@10 (Table 3)" | **FALSE**, plus a wrong cross-reference | NDCG@10 is in Table 2, not Table 3. Random nDCG@10 = 0.524; Gemini 0.497, DeepSeek 0.476, Llama 0.481, GLM 0.508 and Qwen 0.508 are below it, and the best model (GPT, 0.536) is +0.012 above. NDCG@10 does not separate models from chance. A content-blind type prior scores 0.805 |
| "Gemma 3 trades the highest findings recall for the lowest differential coverage, the only model where the trade-off is that severe" | **FALSE** | Gemma has the highest FindRec (0.224), but the **lowest DiffCov is Llama's (0.347 < 0.369)**, and Llama also has the 2nd-highest FindRec (0.190). The trade-off is at least as severe for Llama |
| "Hallucination Rate remains consistently low … modern frontier models rarely fabricate unsupported clinical findings" | **unsupported** | the metric returns 0.000 for fully invented sentences. Near-zero values are a property of the metric, not of the models. "Across all prompting strategies" refers to data (App. C) not shown here |
| "the dominant failure mode is omission … 0.45–0.62 whole-patient, 0.47–0.79 specialty" | ranges OK; inference tautological | whole-patient omission is 1 − the primary metric, which is recall-only, so omission is its only possible failure mode. The metric has no way to register over-inclusion (a chart dump scores 0.676) or fabrication (above). "Even under the best-performing structured strategy" is the strategy whose prompt lists the scored findings (#12) |
| "ICD-10 code specificity remains uniformly high, suggesting errors arise from the wrong diagnosis rather than an overly general code" | **by construction** | specificity is computed only on pairs already matched at 3 characters; category-only prediction floors at 0.731; the primary F1 gives full credit to over-general codes. The design cannot attribute error to over-generality |
| bold marks | OK | each bolded value is the column optimum. Recall has a Gemini/Kimi tie at 0.787 and only Gemini is bolded |
| caption: "Summ. Omiss. is whole-patient summarization omission under the structured strategy" | informative | this **confirms Table 2's Summ. column used the structured (ontology-grounded) strategy**, i.e. the prompt with 10 hinted must-include names (#12), not the `few_shot` that the code's `LOCKED_STRATEGIES` names |

## 9. Appendix B: Effect of specialty-conditioned context

Code: `scripts/ablation_relevance.py`, with prompts in `eval/prompts.py:360-440`. No per-row CSVs or
outputs ship.

| Claim | Status | Evidence |
|---|---|---|
| "On the 117 credited-relevant instances **in the public split**" | **FALSE** | the script loads `split="val"` **+** `split="test"` (`ablation_relevance.py`, `main()`), i.e. the pre-v1.3 names for public + (train ∪ heldout). The 117 credited-relevant rows are **public 26, heldout 24, train 67**. The ablation therefore ran on held-out patients, and only 22% of it is the public split |
| "three held-out foundation models" (Opus 4.6, GPT 5.3, GLM 5) | unclear | all three are ranked panelists in Table 2. The script's default model is `kimi-2.5-thinking` |
| condition **graph** "included graph-derived relevant comorbidities" | **FALSE** | `graph = format_prompt(inp, "zero_shot")` → `SPECIALTY_ZERO_SHOT`, filled with the ground-truth `clinical_question`. That question is **one fixed template** for all 6,345 specialty items: "Summarize this patient's chart from a {S} perspective: the active {S} problems and the comorbidities, labs, and medications relevant to {S} care." `structured_hints` is empty for zero-shot. **No diagnosis, comorbidity or finding from the graph enters the prompt.** The script's own docstring calls it the "graph-aware GT question" |
| condition **neutral** "provided only the specialty framing" | OK | `SPECIALTY_NEUTRAL`, with a different system persona ("clinician" vs "subspecialty consultant") and no include/omit instruction |
| what graph − neutral isolates | **confounded** | the two prompts differ in persona, in an explicit instruction to *include* cross-system comorbidities, labs and medications, and in an instruction to *omit* unrelated systems. They do not differ in graph content. The contrast measures instruction wording |
| Δ graph − neutral: Opus −0.018 (p=0.38), GPT +0.020 (p=0.09), GLM +0.044 (p<0.001) | unverifiable | no outputs ship. The test is a paired Wilcoxon plus a row bootstrap, with no correction across 3 models × 2 contrasts |
| "graph-derived specialty context **consistently** altered finding selection" | **not supported** | one negative (ns), one positive (ns) and one positive (significant) do not make a consistent effect. And there was no graph-derived context (above) |
| neutral − omit "positive for all three … the specialty-conditioned labels encode information that models can exploit when it is made available" | **circular** | `SPECIALTY_SAME_ONLY` instructs "Do NOT include comorbidities, labs, or medications from other organ systems", and the metric is recall of exactly those cross-system (relevant-tier) findings. A drop under "omit" is the instruction being followed. No label information was "made available" in any condition. No effect sizes or p-values are reported for this contrast |
| scope | note | the relevant tier exists in 117/3,809 involved items (3.1%); its 1,270 findings are 2.5% of specialty tier findings (F§5 4.2). The ablation concerns a sliver of one task, as its own docstring says ("NOT the headline conditioned_f1, which is ~98% primary-driven") |
| reproducible on v1.3 | **no** | the `val`/`test` split names no longer exist, so the script selects 0 rows. It also needs Postgres and, by default, the private Kimi gateway |

**What a valid version would do** (cost, if re-run: 3 conditions × 117 rows × 3 models = 1,053 calls):
- inject the item's actual gold-linked comorbidity names in the graph arm, with identical persona and instructions across arms;
- add a shuffled-graph control (another patient's comorbidities) to separate "graph content" from "longer prompt";
- restrict to one split, or report per split.

## 10. Appendix C: Analysis

### C.1 Prompting strategy analysis

| Claim | Status | Evidence |
|---|---|---|
| four strategies share the same system prompt and JSON schema | OK | all 6 task registries in `eval/prompts.py` have one system string across strategies |
| few-shot examples are "selected from public-split cases that the largest number of models answered correctly under zero-shot" | OK, but this **is** test contamination | `examples.py:108-175` selects from `split='public'` runs. The shipped examples are public patients, with their reference answers as "Expected output" (F§6 5.1). The public split is the reported benchmark |
| "CoT augments the few-shot prompt with task-specific reasoning steps" | **FALSE** | no CoT template has a `{few_shot_examples}` slot, and each task's `format_prompt` fills examples only when `strategy == "few_shot"` (e.g. `patient_diagnosis.py:199`). CoT is zero-shot + reasoning steps |
| structured adds "ontology-normalized representations of concepts **already present in the clinical record**" | **misleading** | the hints are read from the **label graph**, not the record (`eval/hints.py`, `eval/tasks/*._load_*_hints`). **Summarization:** 10 key findings = the must-include population, with "do not omit them". **Patient diagnosis:** ICD-10 chapter ranges of the source questions' organ systems, i.e. the answer's chapters, plus key findings. **Imaging:** the question's **correct** diagnosis with its ICD code, listed among distractors as "likely differentials" (`imaging.py:109-146`). **Retrieval:** pathognomonic findings of the query diagnoses. Whether a finding is in the text matters less than *which* findings are selected, and that selection is the label |
| pilot = 3 models "chosen to span the strength range" | PARTIAL | GPT 5.3, DeepSeek V3.2 and Mistral Large rank #1, #8 and #5 by Table 2 mean rank (2.4 / 7.4 / 4.6). The two weakest models (Llama, Gemma), where "scaffolding is most likely to surface gains", are absent |
| pilot items | **test-split tuning** | `run_round(..., split="public", pilot=75)` (`strategy_experiment.py:78-80`). Strategies were selected on the same split they were then reported on |
| "60 strategy×model×task conditions" | **FALSE** | Table C.1 has 3 × 4 × 4 = **48** cells |
| caption: "no prompting strategy is best for any task" | **self-contradictory** | the text says "no single strategy dominates"; the table shows a clear per-task winner in 3 of 4 tasks |
| "zero-shot wins retrieval, few-shot imaging, structured summarization" | OK | all 3 pilot models agree for each of these tasks |
| "CoT wins patient diagnosis" | PARTIAL | only on the 3-model mean (CoT 0.681 vs few-shot 0.665). Per model: GPT is best with **few-shot** (0.732 vs CoT 0.710), DeepSeek with **zero-shot** (0.664; CoT 0.633 is its joint-worst), and Mistral with CoT. Locking CoT moves models unequally (Mistral +0.121 vs its zero-shot, DeepSeek −0.031). This is the opposite of "controlling" for prompting |
| structured "improves Finding-level F1 for all three pilot models without raising hallucination" | true, but it is **leakage** | the gain (+0.07 mean) comes from putting the scored names in the prompt. Echoing the hints alone scores 0.428 (F§6). The hallucination metric cannot detect fabrication (F§8) |
| "structured hints … help precisely where the task rewards findings completeness" | **restates the leak** | injecting the items that a recall metric counts raises that recall |
| "Mistral with CoT outperforms GPT 5.3 zero-shot" | OK | 0.699 vs 0.676 on patient diagnosis |
| final lock: "zero-shot for diagnosis and retrieval, CoT for patient diagnosis, structured for whole-patient summarization, zero-shot for specialty (to avoid leaking ontology-derived relevance labels), few-shot for imaging" | inconsistent | "diagnosis" is the task retired in v1.3. Code `LOCKED_STRATEGIES` says `few_shot` for summarization (F§6). The paper accepts that structured hints leak labels for specialty summarization, but uses the same mechanism for whole-patient summarization, where the hints are the scored must-include findings |

### C.2 Robustness to narrative generation

**No code or data exists in the repo.** There is no re-rendering pipeline, no GPT-rendered corpus, no 100-patient
sample definition and no analysis script. The only trace is a config comment that `kimi-2.5-thinking-or` was a
"fallback for the robustness study" (`eval/config.py:138`).

| Claim | Status | Evidence / reasoning |
|---|---|---|
| "every benchmark label is identical across conditions; only the free-text documentation changes" | PARTIAL | profiles were regenerated too. The patient-diagnosis **chart-neutral set is computed from the profile** (`scripts/chart_neutral_sets.py`), so the scoring function changes with the rendering. If profiles changed, PSH/PMH changed, and the future-leak surface changed with them |
| the design tests whether Kimi benefits from the benchmark's construction | **no** | only narratives and profiles were swapped. Kimi's larger footprint is in the **labels**, and those were held fixed: its proposed ICD codes were accepted as reference codes (F§1 Stage 2), and it authored the imaging reference questions and summary references. The test can detect style familiarity only |
| stability is evidence of "graph-derived state, not narrative" | expected by construction | the scored content is largely deterministic template text that re-rendering does not touch: "(diagnosed …)" problem lists (80% of diagnosis labels), verbatim vitals and labs, and section-type retrieval grades. Stability on patient diagnosis is expected whatever the generator |
| Kimi mean \|Δ\| 0.034, GPT 0.042, Opus 0.028, GLM 0.037 | OK arithmetically | recomputed: 0.0338 / 0.0423 / 0.0283 / 0.0368 |
| primary analysis: Kimi's margin over the field should shrink under GPT rendering | not reported as such | from Table C.2, Kimi's margin change is dx **+0.034**, summ −0.013, retr +0.022, img −0.008: small and mixed. The paper cites only the dx gain |
| TOST equivalence at ±0.05 | **promised, not reported** | Table C.2 shows only Holm-Wilcoxon daggers. Retrieval \|Δ\| exceeds 0.05 for all 4 models, so equivalence would fail there. The other cells have no reported CIs |
| "only the Opus retrieval change is Holm-significant" | unusual | GPT's −0.144 is larger than Opus's −0.087 and not significant, consistent with the small, unequal N (48–135 per cell, from 100 patients). Why some cells have 48 items is not explained |
| "ordering preserved (ρ ≥ 0.80) on three of four tasks" | **uninformative** | with 4 models ρ takes only 11 values. ρ = 0.8 means one adjacent swap, and P(ρ ≥ 0.8) under a random ordering is **0.167** |
| "GPT does worse on its own rendering, the opposite of a home-field advantage" | overreach | the −0.144 is on retrieval P@5, whose grades are section-type driven and whose best score is achievable without reading content (F§6). It cannot speak to familiarity |
| model naming | inconsistent | text "GLM 5 (744B-A17B)" vs table "744B-A40B". "Section 5 argues … such an advantage is unlikely": the §5 text has no such argument |
| "the specialty-relevance task is omitted; it is not patient-scoped in this holdout" | unexplained | specialty items are patient-scoped in the DB (one row per patient × specialty) |

### C.3 Agentic evaluation, extended discussion

| Claim | Status | Evidence |
|---|---|---|
| 40-action budget; on exhaustion one final turn with retrieval tools disabled; unparseable → 0 | OK | `agent_loop.py:296-350` offers only the submit tool, and `forced_submit` is logged |
| failed sessions "scored as zero rather than excluded" | **FALSE for patient diagnosis** | a failed session stores `{}` (`agents/runner.py:371`). `_compute_patient_diagnosis_metrics` **skips** predictions with no diagnoses (`n_tier_c += 1; continue`) instead of scoring 0. Verified: one perfect item + one empty answer → 1.000 (`n_scored` 1). The code's own comment: "a model that answered 43 of 100 patients was ranked on those 43 alone" (`agent_loop.py:288-291`). The forced final turn reduces this but does not remove it. The other tasks do score `{}` as 0. Single-turn runs (Table 2) are affected too: API exceptions are never stored (`eval/runner.py:105-124`), so they drop out of every task's mean |
| three conditions and effect definitions | OK | as in F§6 5.3. The effect arithmetic reproduces |
| "To prevent information leakage, both agent conditions hide the assessment and plan sections, **matching the information available to the single-turn baseline**" | **FALSE** | both agent conditions keep `open_chart` and `view_problem_list`, whose `active_problems` **are the reference diagnoses with ICD codes** (`epic_service.py:110-150, 297-332`). The single-turn baseline never sees these. Imaging agents also have no point-in-time filter, while the single-turn imaging input is cut at the order (`imaging.py:83-94`) |
| "self-retrieval effect within ±0.03 in five of nine", "gains no larger than +0.14", "loop effect negative in all nine", "LLM beats best agentic condition by 0.07–0.15 (summ), 0.02–0.19 (retr), imaging within 0.02 / 0.20" | OK | all recomputed from Table 3 |
| "self-retrieving agent exceeds full-context agent by 0.10–0.18" on patient diagnosis | OK arithmetically (0.099–0.183) | but see the next row |
| interpretation: "tool use helps the model identify and assemble information … difficult to compress into a single preconstructed prompt" | **unsupported; leak explains it** | a self-retrieving agent's natural first call, `open_chart(patient_id)`, returns the answer key. The full-context agent already has the chart and calls tools less. That predicts exactly "self-retrieval > full-context agent on diagnosis only". Echoing the tool scores 0.993 (F§6). Llama's only positive loop effect (+0.240) is also on diagnosis |
| failed sessions 145k vs 60k input tokens; weaker models degrade from context management | unverifiable | no session logs ship. The harness does compress context (`agent_loop.py:_compress_if_needed`), which is not mentioned |
| "a scripted retrieval pipeline followed by single-turn inference may be more effective than a general-purpose agent" | plausible, untested | no scripted-retrieval condition was run |

## 11. Appendix D: Extraction and ontology grounding details

### Code constants (`etl/stages/s05_ontology.py`, `s06_relationships.py`, `s07_ehr_sections.py`, `etl/ontology/*`)

| Claim | Status | Evidence |
|---|---|---|
| "constants below are those in the released code … at the level needed to reproduce it" | PARTIAL | the constants mostly match (below), but the default path does not run them. `etl/main.py:40-72` wires stages 1–6 only. Stage 5 `run()` = 5a + 5b; the SNOMED passes (5d), SapBERT (5e) and LOINC (5f) are **CLI-flag only** (s05:1617-1641). Stage 6 `run()` = 6a only; 6b (dx–finding typing) and 6c (fact linking) are CLI-only. The ontology files (`data/ontology/`) and source content are not shipped |
| kimi-k2.5, default sampling, 4,096 max output tokens | OK | s05:45, s05:279. No temperature or top_p is sent, so reruns are not deterministic |
| "every call keyed by a SHA-256 hash of its input" | **FALSE** | 5a keys on `stage:qid:question_text[:200]` (s05:250-253), i.e. only the first 200 characters. A later edit to a vignette beyond character 200 would hit the stale cache. 6b keys on dx id + finding ids, 6c on fact ids, 7 on qid + 16 hex characters of the vignette hash. `llm_call_log` stores tokens and raw output; `cost_usd` and `request_id` are never written, and failed calls store `raw_response` NULL |
| truncation: vignette 2,000, answer+explanation 500, distractors 200 each | PARTIAL | the constants exist (s05:54-56). Only the explanation is cut to 500; the stem and choices are not truncated as described. Distractor explanations are capped at the **first 4** (s05:244) |
| prompt blocks; 21 categories, 4 acuities, 9 finding types; presence and relevance fields | OK | s05:62-117. The prompt also asks for a `snomed_suggestion` |
| "structural-validation failures are retried up to three times" | **FALSE** | `_validate_extraction` runs **after** `_call_with_retry` returns (s05:430-433). A failure raises and is logged, never retried. `MAX_RETRIES = 3` covers only parse, HTTP and timeout errors, and it means 3 total attempts |
| controlled-vocabulary normalization by a fixed synonym table | PARTIAL | tables at s05:119-172. Unknown finding types become `sign`, unknown acuity becomes `unspecified`, and **unknown categories pass through raw** (s05:206), hence 34 category values in the DB |
| ICD cascade (i)–(v) and FY2025 CMS table | OK as mechanics | (i) accept if the code exists and is billable (`is_header` holds CMS's valid flag, `icd10.py:48`); (ii) exact description; (iii) ≥1 shared 3+-character token prefilter, up to 300 candidates, then difflib ≥0.70; (iv) the raw suggestion is kept with `icd10_desc = NULL`; (v) unresolved. **`resolution_method` is never stored** (s05:723-731), so the only "flag" is the NULL description |
| merge "keeping the highest-priority resolution" | OK | priority llm_validated < exact < fuzzy < llm_unvalidated < unresolved (s05:706-710). Findings merge on `lower(name)\|type`, first seen wins |
| "SNOMED identifiers proposed by the LLM are discarded" | PARTIAL | they **are inserted** in 5b and only wiped when 5d is run via the CLI. Through `main.py` the LLM's SNOMED ids stay. Released coverage matches a 5d run (88.0/97.5%) |
| SNOMED passes and thresholds (exact → Extended Map (dx) → fuzzy 0.95/0.90 → SapBERT 0.70/0.65), US Edition 2025-09-01 | OK | s05:835-916, 966-987, 1066, 1108; `snomed.py:32-42`. The docstring still says 0.85/0.80 (stale). Findings SapBERT uses only the last name variant. Normalization applies to findings only |
| LOINC 2.82; candidate expansion; long name → component → fuzzy 0.90; serum/plasma > quantitative > "most established" | OK, with nuance | `loinc.py:1,27`. Pass 1 already indexes both long name and component. "Most established" is implemented as the **shortest LOINC_NUM string** (`loinc.py:190-198`). Fuzzy hits are chosen by tie-break, not score |
| diagnosis–finding pass: one call per correct dx, ≤40 co-occurring findings with counts, 6 types, frequency | OK | s06:57-63, 511, 547, 645. Findings beyond the top 40 by co-occurrence are **dropped**, not typed. "Omit unrelated" is prompt-only |
| fact linking: batches of 10; exact → substring → fuzzy 0.80; no new concepts | OK | s06:70, 955-1060. The substring step picks the **longest** overlap although the comment says shortest. Fact text is truncated to 300 characters |
| segmentation: 18 types; "copy the source text verbatim and add nothing"; regex checks "flag sections … for review" | PARTIAL | 18 types (s07:41-60). The prompt says "preserve the clinical language exactly" (s07:120); the output is never compared with the vignette. Regex flags are a `log.debug` plus a total count (s07:457, 496). **There is no review queue or stored flag**, and sections are inserted regardless |

### Counts and coverage against the released DB and the public CMS table (`appendix_d_checks.py`)

| Claim (Table D / text) | Released data | Status |
|---|---|---|
| 7,003 board questions; 138,777 finding mentions; 9,623 diagnoses; 36,620 findings; 51,075 question–diagnosis edges | exact | OK |
| 53,746 diagnosis mentions | 51,075 question–diagnosis edges after merging (7,002 correct, 16,596 secondary, 27,477 distractor) | consistent (duplicates merge) |
| CMS FY2025 table: 97,584 codes, 74,260 billable | exact | OK |
| **ICD-10-CM 75.9% grounded, "counting only table-validated codes; a further 14.6% carry a flagged, unvalidated code"** | validated (description present) **5,898 = 61.3%**; unvalidated **1,409 = 14.6%**; uncoded **2,316 = 24.1%** | **FALSE**: 75.9% = validated **+** unvalidated. The caption double-counts the 14.6%. Table-validated coverage is **61.3%**, and nearly a quarter of diagnosis nodes have no code |
| unvalidated codes: "a flagged code that is correct at the three-character level still scores correctly" | of the 1,409 unvalidated: 895 real **header** (non-billable) codes, 458 invalid codes in a valid category, **56 invalid categories**. Reference nodes: 223 header, 113 invalid-in-valid-category, **10 invalid category**, 189 uncoded. Over the 5,602 encounters, **90 reference codes (1.6%) cannot match at any level** (14 invalid category + 76 uncoded) | PARTIAL: "valid at 3 characters" is not "correct" (next row) |
| step (i) accepts "if it is a billable code" | 5,897/5,898 validated nodes are billable | OK: the check is billability, with no semantic check |
| "The LLM's suggested code is treated as a candidate and **never accepted on its own**" | step (i) *is* acceptance on its own when the code is billable, and that is **83.2%** of mentions by the paper's own count | **FALSE, self-contradictory**. Examples where a billable-but-wrong LLM code was accepted: "Leukocytosis" → **D72.819 "Decreased white blood cell count"** (the opposite; node 969, 36 question links); "Septic arthritis" → **B26.85 "Mumps arthritis"**; "Acute appendicitis with perforation…" → K35.890 "…without perforation" (running example); "Obesity" → E66.3 "Overweight" |
| "Mentions are then merged on the resolved code (or lower-cased name when uncoded)" | no ICD code maps to two nodes and no uncoded name repeats | OK, and the consequence is **fragmentation**: 721 display names map to 1,822 nodes, e.g. "Leukocytosis" ×8 codes, "Septic arthritis" ×9 codes (6 lateralities), "Radial nerve injury" ×9. Findings: 2,025 names → 4,271 nodes. Clustering and sharing work on `diagnosis_id`, so these fragments cannot link patients |
| organ-system category "(21 values)", normalized by a fixed synonym table | **34** distinct values: 21 main ones, "Other" as the largest (1,698 = 17.6%), plus 13 stray singletons (e.g. "Anti-infective", "Cellular process", "Surgical", "Pediatric") | PARTIAL |
| acuity ∈ {acute, chronic, acute-on-chronic, unspecified} | acute 3,602; chronic 3,589; acute_on_chronic 259; **unspecified 2,173 (22.6%)** | OK. Scoring maps unspecified-in-active to acute |
| nine finding types; absent findings retained as negations | 9 types exactly; 20,861/138,777 (15.0%) findings have `present = 0` | OK |
| SNOMED coverage 88.0% (8,469/9,623) dx; 97.5% (35,700/36,620) findings | exact | OK |
| LOINC 28.9% (1,009/3,488 lab values) | exact | OK |
| diagnosis–finding pass: 2,975 correct-answer dx, ≤40 findings per call → 52,082 edges over 2,952 dx | exact; the max per diagnosis is 40, and every edge's diagnosis is a correct-answer node | OK. Distribution: commonly_seen 30,629, highly_suggestive 9,258, risk_factor 7,439, rules_out 3,120, pathognomonic 1,194, protective 442. `evidence_source` = `board_question` for all rows, i.e. the typing is LLM output |
| fact cards 53,999, 85.6% linked; 68,560 / 103,879 fact edges; 51,940 EHR sections over 6,990 questions | not in the release | unverifiable |
| mention-level resolution shares 83.2 / 0.2 / 4.3 / 7.3 / 5.0% | needs `llm_call_log` (not shipped) | unverifiable |
| running example: HHS → E11.01; T2DM E11.9; AKI N17.9; 26 findings | exact | OK, with a clinical caveat: E11.01 is HHS **with coma**, while the vignette describes lethargy and confusion, which fits E11.00 (without coma) |
| "This graph … serves as the source of all benchmark labels" | the imaging reference and the chart-neutral sets are not graph-derived (F§1 Stage 4) | PARTIAL |

## 12. Appendix E: Patient clustering and record generation details

Much of this was verified in F§1 Stage 3. This section covers what is new in Appendix E, using
`appendix_e_checks.py` plus the code.

### Question nodes and compatibility

| Claim | Status | Evidence |
|---|---|---|
| age/sex by regex from the demographics section → raw vignette → pronouns; four age buckets; D(q) = correct ∪ secondary; acuity chronic if any correct dx is chronic or acute-on-chronic; smoking and alcohol from social history | OK | `s08_patients.py:47-58, 462-468, 530-578` (F§1). Questions without both age and sex are unclustered (s08:616-620) |
| compatibility: same sex, same bucket, \|Δage\| ≤ 2/5/7/10, no smoking or alcohol contradiction; linked if D(a) ∩ D(b) ≠ ∅, distractors excluded | OK | s08:530-578 |

### Constrained greedy clustering

| Claim | Status | Evidence |
|---|---|---|
| visit nodes by decreasing degree; candidates = unclustered c with s∼c **and s⌢c**, ordered by \|D(c) ∩ D(s)\| desc | OK | s08:659, 673-684 |
| admit if (i) c∼m for all m ∈ C and (ii) c⌢m for **at least one** m ∈ C, so "later encounters may share a diagnosis with an intermediate encounter **but not with the seed**" | **self-contradictory** | the candidate pool is already restricted to nodes linked to the **seed** (s08:673-679), so (ii) is always satisfied by the seed and is redundant. No member can link only through an intermediate. **All 1,268 clusters are stars:** in every patient at least one encounter shares a diagnosis with every other (`appendix_e_checks.py`) |
| size target 3 / 5 / 8 by acuity, re-evaluated after each admission; singletons discarded; patient age = median member age; deterministic | OK | s08:61-65, 669-707, 817 (`int(median(ages))`). Ties fall to input order |
| 1,268 patients over 5,602 questions; 1,401 excluded; sizes 350/355/58/116/29/29/331 | OK | exact |

### Profile generation (8b)

| Claim | Status | Evidence |
|---|---|---|
| prompt carries the correct diagnoses and, per source question, CC + HPI[:300] + PMH/meds/social[:200] | OK (F§1) | the prompt includes **every** encounter's correct diagnosis, including later ones (s08:1159-1196) |
| twelve fixed keys | OK | `REQUIRED_PROFILE_KEYS` (s08:1265-1269). The paper lists only 10 in its parenthesis; `sex` and `age_at_first_encounter` are the other two |
| constraints: nothing contradicts a source excerpt; **chronic conditions are background comorbidities, not the diagnoses being tested**; home meds fit those conditions | **violated at scale** | **500/1,268 (39%)** profiles list one of the patient's **own correct diagnoses** as a chronic condition (e.g. 1680 "systemic lupus erythematosus", 1683 "graves disease"). Because the PMH problem list is prefixed with the profile's chronic conditions **from encounter 0**, these tested diagnoses appear in every note before the visit that "diagnoses" them. This is one driver of the 80% copyability (F§1 Stage 4). Surgical history built from future outcomes (6/6 appendicitis encounters) also contradicts the source presentations |
| "profiles are checked for required keys and for age (±5 years) and sex" | PARTIAL | `_validate_profile` only **returns warnings**. They are logged (`log.warning`) and the profile is written regardless (s08:1463-1480). Nothing is rejected, regenerated or stored |

### Timeline planning (9a)

| Claim | Status | Evidence |
|---|---|---|
| visit types 3,079 / 1,755 / 383 / 296 / 45 / 29 / 15 | OK | exact |
| rules: first visit at month 0, non-decreasing dates, span ≤ 10 years, each source question used once; "validated programmatically"; "dates anchored to a fixed calendar start" | PARTIAL | max span is exactly 120 months (OK). But **10** patients' first visit is not at the 2020-01-15 anchor, and **18** patients have **no encounter_order 0**: their first visit was removed after generation, so their charts start at order 1 or 2. Their first note carries an LLM-rewritten ("hybrid") HPI written against a prior-encounter summary that no longer exists in the chart. 103 same-day encounter pairs exist. Validation failures are warnings only (F§1) |
| "chronic care before acute events" | not enforced | prompt rule only |

### Note assembly and HPI rewriting (9b)

| Claim | Status | Evidence |
|---|---|---|
| PMH prefixed with an active problem list of earlier encounters' correct diagnoses (with dates) plus profile chronic conditions; profile home meds appended when absent; allergies/FH/SH/PSH filled from profile only when the source lacks them | OK | s09:905, 924-951, 1293-1307. This deterministic step is what makes 80% of patient-diagnosis labels copyable (#2) |
| "Every section row records its source section identifier and whether it was modified, so provenance is recoverable line by line" | **FALSE for the release** | `encounter_ehr_sections` ships `id, encounter_id, section_type, section_text, is_modified, section_order, search_vector`. There is **no source-section id**, and the source sections are excluded from the release (DATA_CARD). `is_modified` = 38,550 matches. The 28,433 "no source section" count cannot be checked |
| HPI rewrite prompt includes profile, advanced age, metadata, **the current diagnosis**, prior-encounter summary; "introduce no new findings or assessments"; ≤150% length; 50–250% flagged | PARTIAL | constraints are prompt-only. `_validate_polished_hpi` returns warnings that are logged, not stored or enforced (s09:1120-1131). Giving the model the diagnosis leaks it: 210/4,417 first-occurrence HPIs name their own new diagnosis (F§1 Stage 3) |
| 4,211 rewritten; "the remainder (all first encounters, plus a small number without an HPI section)" | OK | 4,211 hybrid (all non-first); 1,250 first-encounter templates; 141 non-first templates without an HPI. The 18 order-0-less charts are included in the non-first counts |

### Running example

| Claim | Status | Evidence |
|---|---|---|
| A 52 y, B 58 y, C 58 y, all men in the middle bucket; shared T2DM and AKI nodes; all correct dx acute → closes at 3 | OK | question-level data (F§1) |
| profile: T2DM, HTN, CKD 3b, six meds, **a prior appendectomy**, FH diabetes/renal, former smoker | OK as reported, and **it is the defect** | the profile gives a patient whose month-0 visit is perforated appendicitis a *prior* appendectomy. That violates the stated "nothing may contradict a source excerpt" and puts the outcome of visit A into visit A's own PSH |
| timeline A→B→C = ED m0, ED m8, ICU m16; B's PMH problem list; B's HPI opens with prior septic shock | OK | matches the released notes exactly |
| "**The benchmark labels never touch this prose**" | **FALSE** | (a) The imaging reference for visit A is **LLM-authored from the profile**: "Evaluate for acute appendicitis (including stump appendicitis given prior surgery)…", and its pre-read cites "prior appendectomy". (b) The patient-diagnosis **chart-neutral set** is computed from the profile: "Appendectomy" makes K35–K37, Z90 and Z98 neutral for this patient (`chart_neutral_sets.py:SURG`). (c) The problem list that makes labels copyable is deterministic template text, not graph-only |
| "the imaging item … pairs its order with a **graph-derived** clinical question" | **FALSE** | the reference question is an LLM output (s10 stage 10E, `MODEL = kimi-k2.5`), anchored to one graph diagnosis but written from profile and encounter prose |
| 34 sections graded 13/11/6/4 "from the diagnosis–finding graph" | counts OK; mechanism PARTIAL | grades come from section **type** × the encounter's key-finding types; the section text is not read (F§1 Stage 4) |
| patient is in the training split | OK | `split = train` |

## 13. Appendix F: Ground-truth construction details

Much of this was verified in F§1 Stage 4. This section covers what is new, using `appendix_f_checks.py`
and direct scorer calls.

### Patient diagnosis

| Claim | Status | Evidence |
|---|---|---|
| reference = D\* in encounter order, with id, ICD, SNOMED, name, acuity, first encounter and date; chronic/AoC → chronic list; encounter→dx map | OK | s10:439-507; mean **3.61** dx per patient (paper 3.6) |
| "Secondary diagnoses … are deliberately not in the reference; **they are handled at scoring time by the chart-neutral rule**" | **FALSE** | the chart-neutral set is built from the **profile** only: chronic conditions, surgical history, smoking and alcohol (`scripts/chart_neutral_sets.py`). Graph `secondary` diagnoses are not included. Reference + secondary diagnoses scores **0.662**; 182/200 public patients are penalized (F§7, #13) |
| severity tiers: 3 for eleven categories (A41, I21, I26, I63, I60, I61, J96, N17, K72, E87, T78) or acute in chapters A–D, I, J, S, T; 2 for other acute or chronic in those chapters; 1 otherwise | OK | `scoring.py:196-233`. Caveat: "life-threatening" covers all of **E87** (any fluid/electrolyte disorder, e.g. mild hyponatremia) and **T78** (includes unspecified allergy). "Neoplastic/hematologic" is the letters C and D, and injury is S/T |
| primary = HM(tier-weighted recall, unweighted precision) | OK, as a **corpus-level** HM of means | `_compute_patient_diagnosis_metrics`. The per-item reward (`score_one`) is the same formula on one item, so mean reward ≠ reported metric. Empty predictions are skipped rather than zeroed (#19) |
| difficulty = score of encounter count, organ-system count, chronic+acute co-presence | OK | `s10:236-259` |

### Evidence retrieval

| Claim | Status | Evidence |
|---|---|---|
| corpus = patient's sections except assessment/plan | OK | 58,926 judgments = exactly the 59,964 − 1,038 non-A&P sections; 0 A&P sections are judged |
| a section's grade = max over "the findings of its encounter's source question that fall in that section (by finding type …)" with the (r, e) rule | OK as described, and it is the defect | a finding "falls in" a section by its **type** (symptoms and history → HPI; history items → PMH, PSH, FH, SH …), not by appearing in the text. Grades are therefore a property of the encounter's finding inventory. **62% of byte-identical sections within a patient carry different grades** (3,614 groups, 16,658 sections). In 1973, the identical PSH ("Appendectomy…; Inguinal hernia repair"), family- and social-history texts are graded **1, 2, 2** across the three encounters |
| grade totals 12,759 / 29,263 / 10,622 / 6,282 | OK | exact |
| P@5 relevance at grade ≥2; nDCG@10 on 0–3 | OK | P@5 uses `min(k, len)` (#1). The shipped GT JSON still carries stale fact-card `num_passages` / `grade_distribution` (e.g. 1973: 120 passages vs 34 judged) |
| 1973: second encounter's labs = grade 3 because severe hyperglycaemia is key + highly suggestive for HHS | OK | edge (3649, 15986) = `highly_suggestive`; ees_14511 = 3 |

### Context summarization (unconditioned)

| Claim | Status | Evidence |
|---|---|---|
| must-include = key findings of all source questions, name-deduped, capped at 20 (mean 18.8) | OK | mean **18.76**. The cap is chronological, so for 1973 "the cap is reached before the third encounter's findings", i.e. the current ICU problem is never required (18–31% of patients, F§1) |
| "fraction … present under an **abbreviation- and negation-aware** matcher" | **FALSE** | whole-patient `clinical_f1` uses `_list_recall` → `phrase_in_text`, with no negation guard. The negation-aware cascade is used only by the specialty variant. "Patient denies fever. No ketonuria." scores **1.00** against must-include [Fever, Ketonuria] |
| LLM narrative used only for ROUGE-L | OK | secondary metric only |
| clinical question fixed | OK, with one exception | 1,267/1,268; one item asks about obstetric history |

### Context summarization (specialty-conditioned)

| Claim | Status | Evidence |
|---|---|---|
| ICD range → twenty specialties, second-clinician review; I10–I59 → Cardiology, I60–I69 → Neurology, G00–G09 → Neurology + ID | OK | `specialty_map.py:19-60` |
| "up to two absent specialties yield abstention items" | PARTIAL | **exactly** two per patient (1,268 × 2 = 2,536), and they are the first two alphabetically among non-involved specialties (s10f:77), which is not stated |
| tiers: primary / relevant (class-1 or curated) / neutral (class-2/3) / excluded; critical = pathognomonic or highly-suggestive owner edge | OK | `s10f_specialty.py:40-79, 103-114` |
| headline = HM(critical primary∪relevant recall, 1 − leakage); leakage = share of excluded findings mentioned | PARTIAL | leakage is over the ≤15-finding `excluded_sample`, not all excluded findings. **540/3,809 involved items (14%) have no critical finding.** On a single item (the RL reward) they score `conditioned_f1 = 0` for every answer. For 1973 Pulmonology, a correct summary naming the ABG, labored respiration and accessory muscle use scores 0.0. Batch scoring skips them for recall (`_list_recall_cascade`: `if not names: continue`), so benchmark and reward disagree |
| neutral tier keeps the boundary "independent of the associative edges whose validity Section 4.2 assesses" | note | this concedes that most of §4.2's "96 recovered" relations (class-2/3) do not enter credited labels (F§5) |
| 1973: endocrinology critical findings "led by severe hyperglycaemia, confusion, hypotension"; GI + general surgery (rebound tenderness, leukocytosis, free fluid); pulmonology (ABG, accessory muscle use) | OK | matches 59017–59020. The **GI and General Surgery items have byte-identical labels** (both own K35); pulmonology has no critical finding and is the unwinnable case above |
| 6,345 items (3,809 involved, 2,536 absent) | OK | exact |

### Imaging indication

| Claim | Status | Evidence |
|---|---|---|
| 1,865 items; 675 X-ray, 529 US, 342 CT, 179 MRI, 50 CTA, 90 other | OK | "other" = fluoroscopy 40 + other 20 + nuclear 15 + mammography 13 + PET-CT 2 |
| LLM inputs: profile, non-imaging sections, imaging section, correct dx, ≤5 prior chief complaints | OK | s10:1560-1747. "Non-imaging sections" includes assessment and plan, and the profile carries future history (F§12) |
| indication must not name the diagnosis | OK | 1/1,865 indications contain the correct diagnosis name |
| reference differential of "2–4 coded diagnoses" | PARTIAL | sizes are 3 (337) or 4 (1,528). Of 7,123 reference differential codes, **942 are non-billable headers, 169 are invalid codes and 4 are invalid categories** (15.6%) against CMS FY2025, e.g. 1973's "stump appendicitis" → K35.3, a header meaning "with localized peritonitis" |
| "the reference question is anchored to the graph label rather than to the model's own reading of the case" | PARTIAL | the LLM is told the correct diagnosis, but it also reads the profile. For 1973 the reference asks about **stump** appendicitis because of the profile's fictitious prior appendectomy (F§12) |
| responses validated for required fields and indication length | not re-checked | validation code only |

### Provenance and exclusions

| Claim | Status | Evidence |
|---|---|---|
| GT rows store task, granularity, patient/encounter/question and difficulty | OK | schema |
| "every LLM call in this stage is cached by input hash and logged" | unverifiable | `llm_call_log` is not shipped; the stage-5 cache key is not a full-input hash (F§11) |
| non-diagnostic rule set; "all 12,014 items are diagnostic" | OK at item level | 85 **entries** inside patient-diagnosis references are still flagged `excluded_nondiagnostic` and skipped when scoring (public 11, heldout 21, train 53) |

## 14. Appendix G: Real and synthetic note examples

No MIMIC-IV conversion code, converted records or study packets are in the repo (F§5). Panels A and B cannot be
checked against source. Panel C can: it is in the release (`appendix_g_checks.py`).

| Claim | Status | Evidence |
|---|---|---|
| Panel B = "the same admission after conversion to the Synthetic Hospital note format used during physician review"; the normalization "preserved the clinical content while removing formatting differences" | **inconsistent with §4.1** | §4.1: "de-identification masks are **repaired** by … substituting consistent fictional names for masked providers, or **dropping a clause whose subject was masked**". Panel B instead keeps **four** masks, rendered as `[redacted]` ("[redacted] is a [redacted] man", "up to [redacted] aching pain", "currently [redacted] at baseline"). The only change from Panel A is `___` → `[redacted]` plus line re-wrapping. **None of the 5,602 synthetic notes contains a mask.** If physicians saw Panel B's format, every real record carried a marker that identifies it, and chance-level accuracy (53%) would be surprising. If they did not, Panel B does not show the reviewed format |
| "removing formatting differences" | PARTIAL | the register difference remains and is systematic. Panel A/B is telegraphic ("No CP or SOB", "s/p", "x3 weeks"). Only **20/5,414 (0.4%)** synthetic HPIs use any common shorthand (s/p, SOB, CP, c/o, h/o, x3, N/V, abd, pt, w/); the synthetic HPIs are flowing prose averaging 53 words. §4.1 says this register was deliberately left for physicians to detect |
| Panel C: "the full HPI of a Synthetic Hospital patient **with a comparable abdominal presentation**" | **FALSE** | Panel C is a **urinary** presentation (overflow incontinence, neurogenic bladder), not abdominal. It is released section 33133: patient **2285**, encounter 9783, order 4, 2025-07-15, ED. Verbatim match |
| Panel C as "an example of the synthetic documentation evaluated by physicians" | OK, with context | patient 2285 is a **public-split** patient and one of the 13 physician-study patients. The HPI is an **LLM-rewritten** ("hybrid") HPI, not template text. Its opening names the encounter's reference diagnosis ("neurogenic bladder secondary to sacral nerve root injury"). That is legitimate here, since the same diagnosis was keyed at encounter 2 (2023), which the HPI cites. It is a repeat-diagnosis encounter of the kind that makes labels copyable (F§1) |
| Panel C internal consistency | OK | age 69 fits the profile (64 at 2020-01, so 69 at 2025-07); "initially evaluated in 2023" matches encounter 2; "recent diagnosis of gastric adenocarcinoma" matches encounter 3 (2025-01) |
| redistribution of MIMIC text | note | Panels A/B reproduce MIMIC-IV note text (de-identified). The paper states the DUA limits this to a shortened excerpt. Nothing MIMIC-derived is in the repo |

## 15. Appendix H: Case-mix and distributional validation

**No code or data for this appendix is in the repo.** There is no Synthea cohort, no SNOMED→chapter mapping (the
"50-entry manually curated" table), no JSD, Spearman or odds-ratio script, and no pre-registration document.
`grep -ri 'synthea|haldane|jensen|odds ratio'` finds nothing relevant. The paper's statement "The mapping and analysis
code are released with the benchmark" is **false for this repo**. What can be checked is the tables' internal consistency and
their reproducibility from the release (`appendix_h_checks.py`).

| Claim | Status | Evidence |
|---|---|---|
| Table H.1 values follow from Table H.2 | OK | from H.2: JSD(SH, source) = 0.029, ρ = 0.83; JSD(SH, Synthea) = 0.325, ρ = 0.69; JSD(source, Synthea) = 0.433. These match the printed 0.029/0.83, 0.326/0.70 and 0.436 to rounding. Column sums are 99.7 / 100.0 / 100.1 |
| SH "closely matches the ICD-10 chapter distribution of its source corpus (JSD 0.029, ρ 0.83)" | **not reproducible; the test is trivial** | on the released data, the 5,602 used questions vs all 7,003 give **JSD 0.0010–0.0013, ρ 0.99** for every choice of unit (correct / +secondary / +distractors). The benchmark is an 80% subset of its source, so agreement is guaranteed (the paper concedes "agreement … is expected"). The printed H.2 cannot come from a common unit: SH has Z00–Z99 = **9.3%** vs source 2.9%, a 3× excess no question-level count reproduces (correct-only gives SH Z = 2.6%). The SH column most likely adds profile conditions, e.g. the chart-neutral mapping turns smoking and surgery into Z87/Z90/Z98. That would compare LLM-profile content against a source that has none |
| SH "differs substantially from Synthea"; "circulatory and endocrine ≈ 9× and 4× more frequent" | arithmetic OK; interpretation inflated | 12.6/1.4 = 9.0× and 13.8/3.1 = 4.5×, but Synthea's column is **56.8% Z00–Z99**. Removing Z from both gives **4.3×** and **2.1×**. The Z mass is largely Synthea's coding of social and administrative SNOMED "findings" (a known Synthea property), so the divergence partly measures coding conventions, not case mix |
| "differences reflect the benchmark's intended emphasis on diagnostically informative cases" | untested | no analysis separates case-mix emphasis from coding-convention differences |
| Synthea v4.0.0, seed 20260707, Massachusetts, 11,481 patients, 403,751 conditions, 94.1% mapped | unverifiable | nothing shipped. The jar hash is truncated in the text ("ed43c20a…") |
| comorbidity pairs: "all ten show OR > 4" (Table H.3) | **not reproducible; unit-dependent** | with patient-level 3-character categories from graph correct+secondary diagnoses, 9/10 are > 4 but **N18–D64 = 2.60**. Adding profile-derived categories gives 3.21. None of the ten printed ORs is matched by either unit (e.g. J44–J96: paper 70.4, release 19.6/15.7). No CIs are given. Several cells are tiny: **K70–I85 rests on 2 co-occurring patients (OR 67, 95% CI 11–413)**, and I48–I63 on 7 |
| "the generation process preserves … clinically expected disease co-occurrence in the assembled longitudinal patients" | **misattributed** | for 7 of the 10 pairs, most co-occurring patients have both conditions **inside one source vignette**: E11–N18 49/62, I48–I63 6/7, I10–I50 45/47, E78–I25 24/29, J44–J96 18/19, E11–I25 14/22, I10–N18 66/72. The remaining three are K70–I85 1/2, E66–G47 8/16 and N18–D64 5/16. The associations come from textbook vignettes that co-mention comorbidities, not from longitudinal assembly. The clustering is also *defined* by shared correct/secondary diagnoses (F§12), which induces co-occurrence of linking conditions |
| pairs drawn "from the clinical reference set used in Section 4.2" | note | that set also drove the curated graph edges (F§5), so it is not an independent reference |
| "pre-registered three-way comparison" | unverifiable | no pre-registration artifact |
| "designed as an evaluation benchmark rather than a population simulator **or training corpus** … not … for model training" | **contradicts §3.1 / README** | the release ships an 800-patient **training split** "released for training, including reinforcement learning with the graph-derived rewards" (README; §3.1 "the training split additionally supports learning with verifiable rewards") |

## 16. Appendix I: Extended related work

This appendix is mostly literature. It was checked for (a) the claims it makes about Synthetic Hospital and
(b) one characterization of prior work that is checkable.

| Claim about Synthetic Hospital | Status | Evidence |
|---|---|---|
| "no prior benchmark combines open data with coverage of all five [chart-based capabilities] … longitudinal problem-list construction as a scored task … remains unoccupied" | PARTIAL | the capability is claimed, but SH's problem-list task is **80% copyable** from the chart's own problem list (#2). A zero-model regex scores 0.843, above every model. As built, it measures problem-list *extraction*. "EHR operation" was evaluated through a harness whose tools return the labels (#10) |
| "Synthetic Hospital provides the **same FHIR affordances** [as MedAgentBench] … with the problem-list, summarization and graded-retrieval tasks they lack" | **overstated** | the FHIR router is **read-only**: GET only for Patient, Encounter, Condition, Observation, DiagnosticReport, ServiceRequest, DocumentReference, MedicationRequest and AllergyIntolerance (`epic_sim/app/routers/fhir.py:90-644`); writes exist only on the Epic-style JSON API. **`Observation` is built from graph findings** (`fhir/resources/observation.py:43-110`) and has problems: it has **no presence flag**, so the 20,861 extracted **absent** findings (11,797 with no value text at all) are served as positive observations. It has **no effective date or encounter link**, so there is no point-in-time view, and quantities have **no units**. It is de-duplicated per finding, so repeated measurements collapse into one. `Condition` returns the diagnosis labels (#10) |
| real-EHR ground truth "is only what was charted, so a model failure cannot be separated from an incomplete record"; SH's ground truth "does not depend on what happened to be documented" | **inverted in practice** | SH's reference omits conditions the chart documents: graph secondary diagnoses are not in the reference and not neutral, so listing them is penalized (0.662; 182/200 public patients; F§13). And 80% of the reference is literally what the chart documents in its problem lists |
| "derives every patient from **public** educational material" | inconsistent | the README says rebuilding "requires source material that **cannot be redistributed**", and the source is not shipped. The tags point to an LLM-generated Anki deck (`GPT4Anki::…`, `s03_extract_board.py:151`). "Public" and "non-redistributable" cannot both hold. If it is public, there is contamination risk (F§7) |
| "Every diagnosis, finding, laboratory result and **narrative statement** is linked through a typed knowledge graph to ontology-grounded source concepts, yielding **complete provenance**" | **FALSE** | profile content (chronic conditions, surgical history, home meds, family and social history) and the LLM-rewritten HPIs have no graph links: e.g. 1973's "prior appendectomy", or the 500 profiles that list tested diagnoses. Ontology grounding is incomplete (ICD validated 61.3%, SNOMED 88.0% of diagnoses, LOINC 28.9% of labs; F§11). The per-node grounding method is never stored, and the release drops the source-section ids (F§12) |
| Table I: SH "Provenance: **Full**", "Ground truth: **Ontology-grounded**", "No Real Data: Yes" | **Full / ontology-grounded: FALSE**; No Real Data: OK | provenance as above. Ground truth includes billable-but-wrong LLM ICD codes accepted by step (i) (F§11), LLM-typed diagnosis–finding edges that drive retrieval grades and specialty criticality, and LLM-authored imaging references. The benchmark itself needs no real data; the realism study used MIMIC-IV |
| Table I caption: "our ground truth is derived from the ontology-grounded graph **independently of the generated narrative, so narrative errors cannot corrupt it**" (vs SimSUM, which "annotates its own generated text, so generation errors can enter the labels") | **FALSE** | (a) Narrative → labels: the profile narrative sets the chart-neutral set (Appendectomy → K35–K37 neutral) and drives the imaging reference ("stump appendicitis given prior surgery"). (b) The graph *is* LLM annotation (Kimi extraction of source text), so annotation errors enter the labels just as the caption says of SimSUM: e.g. "Leukocytosis" → D72.819 (*decreased* WBC), "Septic arthritis" → mumps arthritis, 1,965 questions listing the answer as a distractor. The difference from SimSUM is an intermediate graph, not error-free labels |
| "To our knowledge, the first benchmark to combine fully synthetic longitudinal EHRs, ontology-grounded provenance, verifiable ground truth and unrestricted open access" | PARTIAL | fully synthetic and open, yes. "Ontology-grounded provenance" and "verifiable ground truth" are qualified by every row above |
| cross-references | broken | the text twice cites "(Table )" with no number |

| Claim about prior work | Status | Evidence |
|---|---|---|
| LongHealth "comprises 20 **single-encounter** multiple-choice cases" | **likely mischaracterized** | LongHealth has 20 fictional patient cases of 5,090–6,754 words each, with 400 multiple-choice questions on information extraction, negation and **sorting**, drawn from single and **multiple** patient documents ([arXiv:2401.14490](https://arxiv.org/abs/2401.14490); [PMC12290132](https://pmc.ncbi.nlm.nih.gov/articles/PMC12290132/)). Multi-document cases with temporal sorting are not "single-encounter" |
| other characterizations (MedAgentBench 100 patients, MedAlign 983 instructions / 276 records, etc.) | not checked | outside the codebase |

---

## 17. Not yet verified

- The simulator end to end (Docker was down): FHIR, RBAC, `/env` hiding of assessment/plan, budget enforcement, Harbor export and oracle.
- Re-running any model (Tables 2/3): needs OpenRouter spend plus the Kimi gateway; the model outputs and physician export are not shipped.
- The ETL cannot be re-run: the source content is not shipped and the LLM gateway is private.
- §4 physician-study data, the 119-pair list and the curated/finding-site CSVs: need the authors' artifacts.
