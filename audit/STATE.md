# Synthetic Hospital v1.3: current repo state vs the paper

Synthesis of `audit/FINDINGS.md` (evidence, cited as #n for headline items or F§n for sections) and
`audit/ROADMAP.md` (fixes). Covers the whole paper: §3–§6 and Appendices A–I. Fork at `b047385`; audited 2026-09-27.
All 15 scripts in `audit/scripts/` re-run cleanly against the released DB (App. D/F also need the public CMS FY2025
ICD-10-CM file).

## Bottom line

**The released artifact is real, consistent and usable as a corpus. The paper's claims about its validity and its
results mostly are not supported by what ships.**
- The data, counts, splits, running example and scorer arithmetic all check out.
- What fails is the layer the paper sells:
  - "verifiable, ontology-grounded ground truth with complete provenance";
  - rewards that measure clinical skill;
  - physician-validated realism;
  - the experimental conclusions.
- Four of the five task rewards can be maxed or gamed without reading the chart. The labels leak through the chart text, the retrieval query, the tool API and the prompt strategy. Every human-study and distributional result has no artifact in the repo.

## Component scorecard

Verdict: **Works** = as the paper describes · **Differs** = present, but the material behavior differs · **Broken** =
present but defective for its stated purpose · **Absent** = not in the repo.

| Component | Paper says | Repo state | Verdict |
|---|---|---|---|
| **Released data** (`benchmark_v1.3.db`, profiles) | 1,268 patients, 5,602 encounters, 12,014 instances, graph | every count exact; loads in SQLite; scorer runs in-process with no Postgres | **Works** |
| **Splits** | patient-disjoint public/heldout/train; held-out private | disjoint, sizes exact. Held-out labels **ship** (#3); stratification within 1.55 pp, not 1 pp; split scripts need the retired task and Postgres | **Differs** |
| **ETL pipeline** | deterministic 5-stage pipeline, reproducible, constants documented | `main.py` runs stages 1–6 only; SNOMED/SapBERT/LOINC/typing/linking are CLI-only; source content, ontology files, curated CSVs and the LLM gateway are not shipped; validation failures are not retried; the cache key is truncated (F§11) | **Absent** as a runnable pipeline; **Differs** as code |
| **Knowledge graph / grounding** | LLM codes "only candidates", deterministic grounding, full provenance | LLM codes accepted on billability alone (83% of mentions), including wrong ones (D72.819 for leukocytosis); validated ICD 61.3%, not 75.9%; no per-node method stored; concepts fragmented across nodes (F§11, #6, #20) | **Differs** |
| **Patient records** | deterministic clustering, constrained LLM prose, physician-reviewed | clustering deterministic (stars around the seed). The profile LLM breaks its constraints (500/1,268 list their own tested diagnoses); surgical history is timeless (future surgery in earlier notes); 18 charts missing encounter 0; no review artifact (#5, #7, #21) | **Differs** |
| **Patient-diagnosis task** | longitudinal diagnosis, severity-weighted chart-neutral F1 | 80% of labels are copyable from later problem lists (regex copy 0.843 > best model); 3-char matching; acuity unscored; documented secondary conditions **penalized**; empty answers dropped (#2, #13, #19) | **Broken** as a diagnosis measure |
| **Evidence retrieval** | graded relevance from findings; P@5 / nDCG@10 | grades depend on section type, not text (62% of identical sections graded differently); P@5 `min(k, len)` means one HPI scores 0.995; nDCG@10 at chance for 5/10 models; the query is the diagnosis answer key (#1, #15, #22) | **Broken** |
| **Summarization (whole-patient)** | finding-level F1, negation-aware | recall only; not negation-aware; chronological cap drops the latest visit (18–31%); chart dump scores 0.676 > best model; hallucination metric blind (#14) | **Broken** as F1 |
| **Specialty summarization** | specialty-relevance F1 over a validated graph | absent items gamed by a stock phrase (#4); absent specialties are the first two alphabetically; 540 items unwinnable as rewards (#23); graph edges affect 3.1% of items (#9) | **Broken** as reward, **Differs** as label |
| **Imaging indication** | concept F1 vs graph-anchored reference | concept F1 deterministic; reference LLM-authored from future-leaking profile; 15.6% of reference differential codes invalid or header (F§13) | **Differs** |
| **Scorer** (`eval/scoring.py`, `score_one`) | deterministic, verifiable rewards in [0,1] | deterministic and bounded (hash-seed test). Per-item reward ≠ corpus metric for dx and specialty | **Works** mechanically, **Broken** semantically |
| **Simulator** (FastAPI, FHIR, RBAC, Epic API) | production-style, Epic-faithful | code present, **never run in this audit** (Docker down). Raw API returns the labels (`view_problem_list`, `open_chart`, FHIR Condition); FHIR read-only; `Observation` drops negation, dates and units (#10, F§16) | **Unverified**; known leaks |
| **`/env` RL interface** | reset/step, hidden outcome, budget, rewards | best-protected path (replaces problem list, point-in-time imaging, server-side A&P hiding) but inherits data-level leaks (copyable PMH, retrieval query) and scorer exploits | **Unverified**; partially protected |
| **Single-turn eval harness** | locked strategies, 10 models | runs; few-shot examples are **public-split** items; "structured" prompts inject labels; code-locked summarization strategy (`few_shot`) ≠ paper (structured); API failures dropped (#12, #17) | **Differs** |
| **Agent harness** (`eval/agents`) | 13 tools, 40 actions, matched information | tools return the labels; imaging not time-restricted; the 100 patients are the longest charts minus a ceiling filter; default split `val` no longer exists (#10) | **Broken** for dx; **Differs** |
| **Paper results** (Tables 2, 3, A, C.1, C.2) | 10-model leaderboard, agentic decomposition, robustness | **no model outputs, run logs or physician export shipped**; Kimi served via private gateway; arithmetic reproduces; conclusions fail against zero-model floors (#11, #15) | **Absent** (not reproducible) |
| **Physician studies** (§3 review, §4.1 realism, §4.2 reference set, Table 2 physicians) | 10- and 7-physician studies, 119-pair set | nothing shipped: no records, judgments, converter, UI, 119-pair list or curated CSVs; App. G's panel shows an identifying mask; §4.2's 15 curated recoveries are circular (#7, #8, #24) | **Absent** |
| **Distribution analyses** (App. H) | Synthea comparison, comorbidity ORs, "code released" | no code or data; the release gives JSD 0.001, not 0.029; comorbidity co-occurrence mostly within single vignettes (#25) | **Absent** |

## Paper claims, section by section

| Section | Holds | Fails or unsupported |
|---|---|---|
| §3 Construction | pipeline shape; running example; counts | deterministic grounding; LLM codes "only candidates"; per-node provenance; physician review; labels independent of prose |
| §3.1 Splits | sizes; patient-disjoint; no source reuse; 54% diagnosis overlap | held-out "only via scorer"; within-1-pp stratification; "46% unseen diseases" (13% at the scored 3-char level) |
| Table 1 | all instance counts | "Finding-level F1" is recall; "Specialty F1" is 40% abstention accuracy; acuity output unscored |
| §4.1 Realism | reported statistics are internally consistent | no artifacts; 10 records, not 100 independent judgments; no equivalence test; "18 section types" vs 17 rendered |
| §4.2 Relations | 8 accepted gaps documented | 111/119 unreproducible; 15 recoveries circular; lenient hit criterion; validates 3.1% of the label surface |
| §5.1 Setup | same system prompt across strategies | strategies tuned on the evaluation split; few-shot contaminated; pilot doesn't span the panel; Kimi built the benchmark |
| §5.2 Table 2 | mean rank and gaps reproduce | "no model near ceiling", "retrieval more mature", "frontier compressed": all overturned by zero-model floors |
| §5.3 Table 3 | every effect equals its row difference | "self-retrieval helps diagnosis" = answer key via tool; biased patient sample; failed diagnosis sessions dropped, not zeroed |
| §6 Limitations | small physician study acknowledged | "chart-neutral prevents penalty" (false); omits every leak and exploit; "reproducible foundation" |
| App. A | bold marks; ranges | hallucination metric blind to fabrication; nDCG@10 and MRR at chance; omission = 1 − primary; two comparative claims contradicted by the table |
| App. B | – | "graph" arm contains no graph content; "117 public" spans all splits; neutral−omit circular; script cannot run on v1.3 |
| App. C | C.3 arithmetic; forced-final-turn mechanics | C.1 cell count, CoT definition, test-split tuning; C.2 no code, cannot test label bias, ρ with n=4 uninformative; C.3 "matched information" false |
| App. D | table sizes; SNOMED/LOINC coverage; typed edges | ICD coverage (61.3% not 75.9%); "never accepted on its own"; retries; full-input hashing; default pipeline |
| App. E | cluster sizes; visit types; note assembly | clustering-rule description self-contradictory; profile constraints violated (500 patients); "labels never touch prose"; line-level provenance in the release |
| App. F | every count; severity tiers; tier definitions | "negation-aware"; "secondary diagnoses handled by chart-neutral"; retrieval grades are type-driven; 540 unwinnable items |
| App. G | Panel C is real release text | Panel B keeps masks (an identifying marker); Panel C is not abdominal |
| App. H | H.1 follows from H.2 | nothing reproducible or shipped; comparison trivially true; ORs from single vignettes; "not a training corpus" contradicts §3.1 |
| App. I | "No Real Data: Yes" | "complete provenance", "narrative errors cannot corrupt ground truth", "same FHIR affordances"; LongHealth mischaracterized |

## What the repo is fit for today

| Use | Fit | Why |
|---|---|---|
| **A longitudinal synthetic clinical corpus** (reading, retrieval prototyping, tool/API development) | **Yes, with caveats** | open, patient-disjoint, rich structure. Know about temporal leaks, the timeless surgical history and ICD noise |
| **Reproducing the paper's numbers** | **No** | no outputs, a private gateway, missing physician and relation artifacts |
| **Comparing models with the paper's metrics** | **Not meaningfully** | floors exceed models on 3/5 columns; the metrics reward copying, section type and name-listing |
| **RL training with the shipped rewards** | **No** | a policy would learn to copy the problem list, submit one HPI section, append "No significant distress.", omit documented conditions, and call `open_chart` for the answer. 14% of specialty items give zero reward regardless of the answer |
| **RL environment after fixes** | **Yes, likely** | most defects are fixable **deterministically on the released DB**, without source content, LLMs or GPUs (ROADMAP Stages 1–4) |

## What is still unverified

- **The simulator and `/env` end to end.** Docker was down, so FHIR, RBAC, `/env` hiding and budget, step latency, and the Harbor oracle are untested. This is the cheapest next check and the one gating Stage 5 (throughput) of the roadmap.
- **Any model run.** A targeted run is warranted only where it would change a decision, e.g. one model with vs without the summarization hints to size #12.

## Recommended next move

ROADMAP **Stage 1 is done** (reward-hacking suite, floors/ceilings, normalized reporting, CI). Next is **Stage 2**: close
the label leaks at the serving layer, starting with `epic_service.get_problem_list`; the suite's strict xfails
`test_problem_list_tool_reveals_nothing_beyond_the_profile` and `test_structured_hints_do_not_contain_scored_findings`
are its acceptance tests. In parallel, bring the Docker stack up and smoke one `/env` episode per task, the only large
unverified surface.

## Engineering quality

Well packaged, but weak at the core. The deployment, serving and scoring *plumbing* is good. The *correctness* of labels
and metrics, the testing and the reproducibility are research-prototype grade.

| Area | Grade | Evidence |
|---|---|---|
| Packaging and deployment | **Strong** | 3-target Dockerfile (app / with-data / all-in-one), compose stack, Apptainer def, Harbor exporter with oracle, idempotent entrypoint, 10 Alembic migrations, public ICD-10 auto-fetch |
| Architecture of scoring and env | **Good** | one scorer (`compute_all_metrics`) shared by batch eval, `/score` and `/env`; in-process and deterministic; the `/env` design (server-side hiding, budget, scorer token) is sound |
| Documentation honesty | **Good** | README and DATA_CARD disclose label exposure via the raw API, the LLM-authored imaging reference and the unlicensed terminology tables |
| Metric correctness | **Poor** | basic bugs that property tests would catch: P@k `min(k, len)`; empty critical set → 0; no negation in the whole-patient matcher; empty predictions skipped, not zeroed; per-item reward ≠ corpus metric |
| Label and data correctness | **Poor** | LLM output accepted without semantic checks; constraint violations logged, not enforced (profile, timeline, HPI length, segmentation); leaks fixed at the edge (`/env`) rather than at the source (`epic_service`) |
| Tests and CI | **Weak** | 143 tests (epic_sim 92, eval 31, etl 20). No CI config, no lint or type config. eval tests cover specialty, imaging concepts and a *current-visit* variant absent from the release. **No tests of the patient-diagnosis, retrieval or whole-patient summarization metrics, none of any ETL stage, and none of reward properties** |
| Failure handling | **Weak** | 45 broad `except Exception` handlers. Silent paths: a missing curated CSV silently changes labels; validation failures are dropped, not retried; API failures vanish from means |
| Code hygiene | **Weak** | ~43k LOC of accreted research code; 31 files still reference v1.2 paths, `val`/`test` splits or the retired `diagnosis_accuracy` task; dead config (`prompt_revision`), dead branches (s02 LLM fallback, `demographics` section), stale docstrings and GT fields |
| Reproducibility | **Poor** | `etl/main.py` runs stages 1–6 of ~12; the source, ontology files, curated CSVs, `llm_call_log` and model outputs are not shipped; a private LLM gateway; analysis scripts need Postgres state that is not released |
