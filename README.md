<p align="center">
  <img src="synthetic_hospital_logo.svg" alt="Synthetic Hospital emblem: a teal cross with a pulse line" width="160">
</p>
<h1 align="center">
      Synthetic Hospital:<br>
      A Medical Benchmark & EHR Simulation Platform</h1>

Converts USMLE-style medical education source content into a ground-truth benchmark database of synthetic longitudinal patient records, served through an Epic-faithful EHR simulation platform (FHIR R4 + OAuth2/RBAC), for evaluating clinical AI agents on four longitudinal-chart tasks: patient diagnosis (index-encounter diagnosis from the chart up to that visit; the longitudinal problem-list task is kept as a superseded extraction task), context summarization (with whole-patient, current-visit, and specialty-conditioned variants), evidence retrieval, and imaging indication. The benchmark ships with a labelled training split so it can also be used as a verifiable-reward environment.

> **Paper:** Park, Chen, Dettmers. _Synthetic Hospital: An Open, Verifiable, Physician-Validated Longitudinal EHR Benchmark._ Preprint, 2026. arXiv link: (https://arxiv.org/abs/2609.30027).
> If you use this work, please cite it (see [CITATION.cff](CITATION.cff)).

## Architecture

```
source content                        (not distributed — see Data Setup)
      │
      ▼
  etl/  (stages s01–s10: ingest → classify → extract → fact cards →
         ontology mapping → relationships → EHR sections → patients →
         encounters → ground truth)          [SQLite: data/benchmark.db]
      │
      ▼
  epic_sim/  FastAPI EHR simulator (FHIR R4, OAuth2, RBAC, Postgres)
      │
      ▼
  eval/  task runners, scoring, semantic matching, reports
```

## Released Dataset (v1.3)

| File | Contents |
|---|---|
| `patient_profiles.db`, `patient_profiles.jsonl` | 1,268 synthetic longitudinal patients and their 5,602 clinical notes (SQLite and JSON Lines) |
| `benchmark_v1.3.db` | The benchmark database: chart sections, ground truth for the four tasks (private-split labels excluded), section-level relevance judgments, imaging orders, the ontology-grounded knowledge graph, and the public / held-out / train / private split labels |

See **[DATA_CARD.md](DATA_CARD.md)** for schemas, provenance, and what was excluded. The data is fully synthetic and not for clinical use.

**Splits.** Patients are partitioned at the patient level: `public` (200 patients; the reported benchmark), `heldout` (268; labels shipped, so a second validation set), `train` (600; released for training, including reinforcement learning with the graph-derived rewards) and `private` (200, carved from train; **labels are not in the repository**, see *Private split* below). No patient shares source material with any other. Evidence retrieval is one instance per (patient, reference diagnosis), so its query never equals the patient's diagnosis answer key.

**What the simulator never serves** (`epic_sim/app/services/visibility.py`): assessment and plan sections; the graph-derived diagnoses (the problem list is the chart's documented history); and, for an imaging-indication or index-encounter diagnosis instance, any encounter after the index one. These rules apply to the Epic tool API, FHIR and `/env` alike. The "structured" prompt strategy adds ontology guidance only, no patient-specific concepts, and few-shot examples come from the train split.

**FHIR.** `Observation` is one resource per measurement (a finding as extracted from one encounter; `id` = `question_findings.id`) with the source encounter and its date, the presence flag as `interpretation` POS/NEG (a finding with no value carries SNOMED Present/Absent), values parsed from the source text with UCUM units, blood pressure as systolic/diastolic components and a stated normal range as `referenceRange`; the unparsed remainder is kept in `note`. A labs `DiagnosticReport` lists its lab Observations in `result`. Every search honours the session cutoff (`X-Session-Id`). The server accepts FHIR `create` for `Observation`, `ServiceRequest`, `MedicationRequest` and `Condition` (scope `patient/<Type>.write`; attending and resident hold all four, nurse `Observation.write`): writes go to the `fhir_writes` table, never to the benchmark tables, and are visible only to the session (or, without one, the user) that made them.

## Repository Map

| Path | Purpose |
|---|---|
| `etl/` | Source content → benchmark DB pipeline. `parsers/`, `stages/` (s01–s10), `ontology/` (SNOMED, ICD-10, LOINC, SapBERT embeddings), `deck_profiles/` and `pdf_profiles/` (declarative source descriptions; see `stages/EXTENDING.md`) |
| `epic_sim/` | EHR simulation platform: FastAPI app, FHIR R4 resources, OAuth2 + RBAC, Alembic migrations, SQLite→Postgres migration scripts, tests |
| `eval/` | Evaluation framework: CLI, task definitions (`tasks/`), agentic harness (`agents/`), scoring, ICD-10 validation, semantic matching, reporting |
| `scripts/` | Auditing and analysis scripts (ontology mapping QA, dataset tiers, ablations, paper tables, Harbor export, image publishing) |
| `docker/`, `Dockerfile`, `docker-compose.yml` | Container build and entrypoints (compose stack, self-contained and all-in-one images) |
| `harbor/`, `apptainer/` | Harbor task templates and the Apptainer definition |
| `benchmark_v1.3.db`, `patient_profiles.db`, `patient_profiles.jsonl` | The released data (see `DATA_CARD.md`) |

## Quickstart (Docker, one command)

The compose stack builds the simulator image, starts Postgres 16 and Redis 7, loads
`benchmark_v1.3.db` into Postgres on first boot, and serves the EHR simulator:

```bash
docker compose up -d                 # first boot loads the database (~10 s), then serves http://localhost:8000/docs
docker compose logs -f app           # watch the load; "[entrypoint] database ready: 1268 patients" means done
```

Everything else runs inside the same image, against the loaded database:

```bash
docker compose run --rm app python -m eval.cli --help
docker compose run --rm app python -m eval.cli run --task patient_diagnosis --model <model> --split public --strategy cot
docker compose run --rm app pytest eval/tests
```

Set `OPENROUTER_API_KEY` in your shell (or a `.env` file) before running models; results are written to
`./results` on the host. Host ports are configurable if the defaults are taken:
`POSTGRES_PORT=55432 REDIS_PORT=56379 APP_PORT=58000 docker compose up -d`.

The load is idempotent: restarting the stack skips it when patients are already present
(`EPIC_SIM_FORCE_RELOAD=1` reloads from the SQLite file). See `docker/entrypoint.sh` for the full list
of environment switches.

### Terminology (ICD-10-CM, SNOMED CT, LOINC)

The FHIR terminology endpoints (`$lookup`, `$expand`, `$validate-code`) and the trigram code search read
from a `terminology_codes` table that the release cannot pre-fill: SNOMED CT needs a UMLS licence and
LOINC a free registration. Everything else in the benchmark works without them.

- **ICD-10-CM** is public domain and is fetched from CMS automatically on first boot (set
  `EPIC_SIM_DOWNLOAD_ICD10=0` to disable, or drop the CMS "code descriptions in tabular order" `.txt`
  or `.zip` into the ontology directory to load offline).
- **SNOMED CT and LOINC**: download your own copies, unpack them anywhere under a directory, mount it
  at `/app/data/ontology` (uncomment the volume in `docker-compose.yml`), and restart the stack. The
  entrypoint discovers the files by pattern (`sct2_Concept_Snapshot*.txt` with its
  `sct2_Description_Snapshot-en*.txt`; `LoincTable/Loinc.csv`), so any release version and either the
  original folder layout or flat files work, and loads only the systems not yet present. Expect a few
  minutes for SNOMED CT. Nothing else needs to change; the terminology tests stop skipping once the
  systems are loaded.
- Manual install: `python -m epic_sim.migrate.load_terminology --auto --download-icd10` does the same
  from the host; `--list` shows what was found.

## Quickstart (manual)

```bash
# 1. Install
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Infrastructure (Postgres 16 + Redis 7)
docker compose up -d

# 3. EHR simulator
uvicorn epic_sim.app.main:app --reload   # http://localhost:8000/docs

# 4. Run tests
pytest epic_sim/tests
pytest eval/tests
```

## Data Setup

The benchmark database `benchmark_v1.3.db` **is distributed** (see Released Dataset) and is all you need to run the evaluations: load it into the simulator with the commands below. Rebuilding it from scratch requires source material that cannot be redistributed:

1. **Source materials**: the ETL is source-agnostic. Describe your own Anki decks or PDFs with a small YAML profile (`etl/deck_profiles/`, `etl/pdf_profiles/`; see `etl/stages/EXTENDING.md`) and place the files under `apkg/` (paths in `etl/config.py`). The material the paper used is not included.
2. **Ontologies** (place in `data/ontology/`):
   - SNOMED CT US Edition + Transitive Closure — [UMLS license required](https://www.nlm.nih.gov/healthit/snomedct/)
   - LOINC — [free registration](https://loinc.org/downloads/)
   - ICD-10-CM 2025 tabular XML — [CDC/CMS, public domain](https://www.cms.gov/medicare/coding-billing/icd-10-codes)
3. **SapBERT embeddings**: generated locally by `etl/ontology/sapbert_embedder.py` (no download needed).

Then run the ETL:

```bash
python -m etl.main --all                         # stages 1..12, in order (or --stage N, --from 5 --to 10)
python -m etl.main --stage 5 --pilot 20          # a 20-record pilot of one stage
python -m etl.preflight --all                    # what is missing, without running anything
```

### Bring your own source corpus

The pipeline ships no source content and no LLM. What it needs from you, and how it behaves:

- **Sources.** Anki `.apkg` decks (and PDFs) described by a YAML profile (`etl/deck_profiles/`,
  `etl/pdf_profiles/`, `etl/stages/EXTENDING.md`), placed under `apkg/` or given with `--sources DIR`
  (`SH_APKG_DIR`). A private corpus is the only route to a test set that frontier models have not seen: the
  source families the original benchmark used are public study decks, so anything built from them is a
  contamination risk for evaluation (fine for training).
- **An LLM endpoint.** Any OpenAI-compatible server (vLLM, llama.cpp, Ollama, OpenRouter, ...):
  `SH_LLM_BASE_URL`, `SH_LLM_MODEL`, optional `SH_LLM_API_KEY`; `SH_LLM_API=anthropic` selects the messages
  format the original gateway spoke. Every request is sent with `temperature 0` and `SH_LLM_SEED` (default 0),
  retried on HTTP/timeout/parse **and** structural-validation failures, cached on a SHA-256 of the full
  prompt plus the sampling settings, and logged to `llm_call_log` with model, temperature, seed, attempts,
  endpoint and request id. `scripts/mock_llm_server.py` is a deterministic stand-in for dry runs.
- **Preflight.** `etl.main` checks a stage's inputs before running it and exits 2 with the list of what is
  missing and how to fix it (source decks, the ICD-10-CM table — `python -m etl.stages.s05_ontology
  --download-icd10` — SNOMED RF2, LOINC, the clinician-curated CSVs stage 11 needs, a reachable endpoint).
  Stage 11 refuses to build the typed diagnosis graph without `finding_site_review.csv` and
  `curated_diagnosis_edges.csv` unless you pass `--allow-missing-curated`, because their absence changes the
  specialty labels silently.
- **Manifest.** Every stage appends to `data/pipeline_manifest.json`: LLM fingerprint, pipeline seed, SHA-256
  of each input file, git commit, summary. Two runs with the same manifest inputs and a deterministic model
  reproduce the same database; with a sampling model they reproduce the same *procedure*.
- **Contamination check.** `python scripts/source_fingerprints.py --write --db data/benchmark.db` stores the
  SHA-256 and hashed 8-word shingles of every source vignette (no text); `--check corpus.jsonl` reports which
  items a public dump or a training corpus contains, exactly or by shingle overlap.

What remains irreproducible by construction: the original source decks and the LLM outputs that built
`benchmark_v1.3.db` (kimi-k2.5 through a private gateway, unseeded). The released database is the artifact;
the pipeline is how to build your own.

And load into the simulator:

```bash
alembic -c epic_sim/alembic/alembic.ini upgrade head   # creates the schema (run first on a fresh database)
python -m epic_sim.migrate.sqlite_to_pg                 # loads benchmark_v1.3.db (repo root) into Postgres
python -m epic_sim.migrate.load_terminology             # optional; needs the ontology files above
```

Or let the container do all of this: `docker compose up -d` (Quickstart above).

## Running Evaluations

Models are accessed via OpenRouter; set `OPENROUTER_API_KEY`. The registry entries named `kimi-2.5*` and
`glm-5-agent` were served through a private Anthropic-compatible gateway for the paper; they read
`SH_LLM_GATEWAY_URL` (default `http://localhost:8080`) and need your own gateway, whereas the `*-or`
entries reach the same models through OpenRouter.

```bash
python -m eval.cli run --task patient_diagnosis --model <model> --split public --strategy cot
python -m eval.cli score --run-id <id>
python -m eval.cli report --split public --format csv
python -m eval.cli matrix --task patient_diagnosis --split public   # all models
```

Tasks: `patient_diagnosis`, `context_summarization`, `evidence_retrieval`, `imaging_indication`, and the Stage-7 families `differential_diagnosis`, `test_selection`, `error_detection`, `lab_triage`, `atypical_diagnosis` (see *New task families* below). Splits: `public`, `heldout`, `train`. See `python -m eval.cli --help`.

### New task families (Stage 7)

Five encounter-level tasks derived from the index-encounter diagnosis instances, using the parts of the graph no earlier task used (the 27k distractor rows, the typed diagnosis–finding relations, the structured findings behind the notes). Every label is built from the graph tables or a deterministic transform of the index encounter's own text; every task has degenerate floors and an oracle at 1.0; both environments serve them with the point-in-time cutoff.

| Task | Submit tool | What is scored | Reward |
|---|---|---|---|
| `differential_diagnosis` | `submit_differential` | ranked list (≤5) vs. the visit's correct diagnosis (gain 1) and its distractors (gain 0.5), graded ICD credit | `differential_ndcg_5` |
| `test_selection` | `order_test` … `submit_workup` | the index visit's result sections are hidden; `order_test(name)` returns the result as documented (or "not performed") and costs a step; the reward counts the episode's orders, never the submission's claims | `workup_score` = ICD credit × evidence (a discriminating test ordered, else 0.25) × parsimony (needed / ordered) |
| `error_detection` | `submit_error` | one injected error (implausible value, laterality swap, age or sex contradiction) served through a section override | 0.5 × section hit + 0.5 × type hit |
| `lab_triage` | `submit_triage` | which of the visit's lab / vital results bear on the diagnosis (key/supporting vs. background) and the most urgent one | 0.6 × F1 + 0.4 × urgent hit |
| `atypical_diagnosis` | `submit_diagnosis` | the patient-diagnosis task on a chart whose sentences stating a pathognomonic / highly-suggestive finding are masked | patient-diagnosis reward; robustness = atypical − original |

Locked prompting strategies used in the paper: CoT (patient diagnosis), ontology-grounded structured (whole-patient summarization), zero-shot (retrieval, specialty-conditioned summarization), few-shot (imaging). `eval.cli score` reports the primary metrics as **redefined in this fork** (see [SCORING_CHANGES.md](SCORING_CHANGES.md); names unchanged, semantics hardened): patient diagnosis by graded-ICD, acuity-aware, severity-weighted F1 under the chart-neutral rule (`weighted_problem_list_f1_neutral`), retrieval by `ndcg_10` over content-graded chart sections (`precision_5` also returned), summarization by HM(negation-aware finding recall, chart-grounded concept precision) × length factor (`clinical_f1`), and the imaging clinical question by concept F1 against graph-derived reference terms (`clinical_question_concept_f1`; extractor in `eval/imaging_concepts.py`, built from the loaded database with no external ontology files).

`context_summarization` has three variants selected via `--granularity`: whole-patient (default), `encounter` (current-visit, point-in-time), and `specialty` (specialty-conditioned; optionally scope to a frozen cohort with `--tier small|medium|large`). Example:

```bash
python -m eval.cli run --task context_summarization --granularity specialty --model <model> --split public
```

Agentic (tool-use) evaluation against the live EHR simulator uses `eval/agents/` with the read-only bash sandbox defined in `docker-compose.override.yml`. The `bash_readonly` role that the sandbox uses is created automatically by the container entrypoint (or by `scripts/setup_bash_readonly.sql` in a manual install).

## Protocol results (Stage 8)

Every number below follows [EVAL_PROTOCOL.md](EVAL_PROTOCOL.md) (floors and ceilings, 95% bootstrap intervals,
paired comparisons, failures scored 0, random seeded samples); the paper's Tables 2–3 and Appendix A are
**non-protocol**. Public split, agentic episodes in the in-process environment; full tables with intervals in
[results/leaderboard.md](results/leaderboard.md), the RL decision in [results/rl_decision.md](results/rl_decision.md).

| unit | floor | Qwen3.5-9B, tools (n=120) | Qwen3.5-9B, no tools | GPT-6 Sol, tools (n=40) |
|---|---|---|---|---|
| patient_diagnosis | 0.036 | 0.332 | 0.246 | 0.520 |
| atypical_diagnosis | 0.035 | 0.347 (n=77) | — | 0.647 (n=23) |
| differential_diagnosis | 0.027 | 0.408 | 0.340 | 0.526 |
| test_selection | 0.001 | 0.273 | 0.106 | 0.325 |
| evidence_retrieval | 0.420 | 0.563 | — | 0.700 |
| context_summarization | 0.487 | 0.525 | — | 0.619 |
| specialty_conditioned | 0.407 | 0.595 | — | 0.589 |
| imaging_indication | 0.219 | 0.249 | — | 0.289 |
| error_detection | 0.494 | 0.892 | — | 1.000 |
| lab_triage | 0.631 | 0.425 | — | 0.487 |

Tools help the RL candidate on all three ablated units (paired, intervals above 0). Four units show learnable
headroom for RL (above the floor, below saturation, a significant gap to the anchor): patient_diagnosis,
atypical_diagnosis, evidence_retrieval, differential_diagnosis. lab_triage is below its flag-everything floor for
both models and error_detection is saturated. Reproduce with `bash scripts/run_stage8_panel.sh results/logs`, then
`python scripts/rescore_protocol_runs.py && python -m eval.protocol report && python scripts/stage8_rl_decision.py`.

## Floors, ceilings and the reward-hacking suite

Raw scores on this benchmark are not interpretable on their own: on the public split a content-blind ranking of
sections by type scores P@5 0.97, a regex that copies the chart's own problem list scores severity-weighted F1 0.84,
and pasting the whole chart scores summarization "F1" 0.67 (see `audit/FINDINGS.md`). `eval/floors.json` records,
per split and task, the **floor** (best zero-model policy in `eval/degenerate.py`) and the **ceiling** (label
oracle); `eval.report` appends `<metric>_normalized = (score − floor) / (ceiling − floor)` next to every primary metric.

```bash
python -m eval.floors --split public            # print floors / ceilings / headroom
python -m eval.floors --split public --check    # CI: fail if eval/floors.json is stale
pytest eval/tests/test_reward_hacking.py -q     # ~40 s, SQLite only, no Postgres
```

`eval/tests/test_reward_hacking.py` runs every known exploit through the scorer and gates it. Gates the current
scorer or data fail are `xfail(strict=True)`, each naming the `audit/ROADMAP.md` stage that fixes it: the suite is
green today, and a fix that lands without updating the suite (and `eval/floors.json`) fails CI
(`.github/workflows/tests.yml`). Install `requirements-dev.txt` to run it; it drives the simulator's real
problem-list service over the SQLite file through `aiosqlite`.

## Private split

`scripts/carve_private_split.py` moved the labels of 200 patients out of `benchmark_v1.3.db` into
`private/labels_v1.3.db` (gitignored). Their rows keep the inputs only (`"_labels_removed": true`, no relevance
judgments). Whoever holds the overlay scores them by pointing `SH_PRIVATE_LABELS_DB` at it (default
`private/labels_v1.3.db`); `/score` and `/env` then return the **reward only** for private items, never the metric
breakdown, and `503` when the overlay is absent. A copy of the labels is available to collaborators on request.

## Scoring Endpoint (rewards)

The simulator exposes the paper's scorer over HTTP so that an external trainer can obtain a reward
for any submission without importing the evaluation code. It is guarded by a shared secret
(`EPIC_SIM_SCORER_TOKEN`, sent as `X-Scorer-Token`) that the policy under training must not hold; the
labels themselves are never returned.

```bash
TOKEN=dev-scorer-token-change-in-production        # set EPIC_SIM_SCORER_TOKEN in production
curl -s -H "X-Scorer-Token: $TOKEN" "http://localhost:8000/score/tasks"
curl -s -H "X-Scorer-Token: $TOKEN" "http://localhost:8000/score/instances?task=patient_diagnosis&split=train&limit=3"
curl -s -H "X-Scorer-Token: $TOKEN" -H "Content-Type: application/json" http://localhost:8000/score \
  -d '{"gt_id": 123, "prediction": {"active_diagnoses": [{"icd10": "E11.01", "acuity": "acute"}], "chronic_conditions": []}}'
```

The response carries `reward` in [0, 1], the metric it was taken from, and every metric the scorer
computed (reward only for the private split). Rewards are the tasks' primary metrics as defined in
[SCORING_CHANGES.md](SCORING_CHANGES.md): graded-ICD, acuity-aware, severity-weighted chart-neutral F1 for
patient diagnosis; HM(finding recall, grounded precision) × length for summarization (conditioned F1, or
abstention accuracy from the explicit `abstain` field, for the specialty variant); nDCG at 10 over
content-graded chart sections for retrieval (P@5 also returned); and concept F1 of the inferred clinical
question against graph-derived reference terms for imaging. Submissions use the same JSON schemas the
agent tools accept (`submit_diagnosis`, `submit_summary`, `submit_rankings`, `submit_pre_read`); a malformed
or empty submission scores 0 rather than raising. `POST /score/batch` scores up to 256 items per call. The
same function is available in-process as `eval.score_one.score_submission`.

## RL Environment (reset and step)

The simulator can be driven as a reinforcement-learning environment. An episode is one benchmark
instance played as the paper's tool-use task: the policy gets the paper's agent system prompt, the
patient assignment, and the EHR tools as function schemas (13, plus `order_test` in test-selection episodes); it acts by calling tools; the task's
submit tool ends the episode and returns the reward from the scoring endpoint above. Episode state
lives in Redis (started by `docker compose`).

```python
from eval.env_client import SyntheticHospitalEnv

env = SyntheticHospitalEnv("http://localhost:8000", scorer_token=TOKEN)
obs = env.reset(task="patient_diagnosis", split="train", seed=0)   # or reset(gt_id=...)
# obs.instructions (system prompt), obs.intro (first user message), obs.tools (function schemas)
o, reward, done, info = env.step("open_chart", {"patient_id": obs.patient_id})
o, reward, done, info = env.step(obs.submit_tool, {"active_diagnoses": [...], "chronic_conditions": []})
# done is True, reward in [0, 1], info["metrics"] holds every metric
```

The same four calls are plain HTTP (`POST /env/reset`, `POST /env/step`, `GET /env/state/{id}`,
`POST /env/close`, all with `X-Scorer-Token`), so any language works. `scripts/env_demo.py` runs a
scripted policy end to end and is the template to replace with a model. What the server enforces:

- **Hidden outcome.** Assessment and plan sections are stripped from every observation, as in the
  paper's baselines, and the problem list returned by `open_chart` / `view_problem_list` is the chart's
  documented history (the profile's chronic conditions) rather than the simulator's graph-derived
  encounter diagnoses, which are the patient-diagnosis reference itself. Imaging-indication episodes
  cannot observe encounters after the imaging order.
- **Budget.** Default 40 actions (`budget=` on reset). Once spent, only the submit tool is accepted,
  mirroring the paper's forced final turn; unknown tools and bad arguments come back as error
  observations and still cost a step.
- **Reward.** Computed once at submission with the paper's primary metrics; a malformed or missing
  submission scores 0. Labels are never returned, and the policy holds no credential: the harness
  keeps the scorer token.
- **Reproducibility.** `reset(task=, split=, seed=)` samples deterministically; `env.instances()` lists
  every instance id so you can build your own curriculum over the 7,310 training instances.

### In-process environment (no server)

For training throughput, `eval.local_env.LocalEnv` is the same environment without HTTP, Postgres or
Redis: it reads `benchmark_v1.3.db` directly, serves the same 9 tools with the same observations, applies
the same visibility rules and budget, and computes the same reward in-process. One `LocalEnv` per process;
processes scale independently.

```python
from eval.local_env import LocalEnv

env = LocalEnv()                      # + env.warmup() once per process (~15 s of caches)
obs = env.reset(task="patient_diagnosis", split="train", seed=0)
o, reward, done, info = env.step("view_encounters", {"patient_id": obs.patient_id})
o, reward, done, info = env.step(obs.submit_tool, env.oracle())     # env.oracle(): label-derived answer
```

`eval/tests/test_local_env.py` checks parity: the Stage-1 degenerate policies and the oracles score
identically through `LocalEnv` and through the scorer, and, when a server is up, every tool observation
and reward matches the HTTP env on sampled episodes (`search_chart` returns the same candidate set; its
rank order comes from an FTS5 bm25 index instead of Postgres `ts_rank`). `python -m eval.bench_env
{http,local,components}` is the throughput harness; `audit/bench/*.json` holds the measurements
(see `audit/STAGE5_TODO.md` for the before/after table).

## Container images, Harbor tasks, and Apptainer

The Dockerfile has three targets:

| Target | Build | Contents |
|---|---|---|
| `app` (default, used by compose) | `docker compose build` | code and dependencies; the database is bind-mounted |
| `with-data` | `docker pull ghcr.io/sparkcpark/synthetic-hospital:1.3-data` | plus `benchmark_v1.3.db` baked in; self-contained, used by Harbor tasks |
| `allinone` | `docker pull ghcr.io/sparkcpark/synthetic-hospital:1.3-allinone` | plus PostgreSQL and Redis inside the image, run as an unprivileged user with no other services |

The two data-bearing images are published to GitHub Container Registry; `scripts/publish_images.sh`
builds and pushes them (maintainers). To build them locally instead, use the same tags:
`docker build --target with-data -t ghcr.io/sparkcpark/synthetic-hospital:1.3-data .`

**Harbor.** `scripts/harbor_export.py` turns benchmark instances into
[Harbor](https://harborframework.com) tasks, one directory per instance, with the paper's agent
prompt as `instruction.md`, a compose environment (simulator with the episode pre-started, Postgres and
Redis on a network the agent cannot reach), an `sh-agent` command line for the agent, a verifier that
writes the episode reward to `/logs/verifier/reward.json`, and an oracle solution that scores 1.0. See
`harbor/README.md`:

```bash
python scripts/harbor_export.py --out harbor_tasks --task patient_diagnosis --split train --limit 50
uvx harbor run -p harbor_tasks -a oracle                  # self-check, then swap in any Harbor agent
```

**Apptainer.** For clusters without Docker, `apptainer/synthetic_hospital.def` converts the all-in-one
image into a `.sif` that starts everything as the calling user with a bound state directory; see
`apptainer/README.md`. The plain `docker run --rm -p 8000:8000 ghcr.io/sparkcpark/synthetic-hospital:1.3-allinone` form
works anywhere Docker does and needs no compose file.

## Reproducing Paper Results

1. Load `benchmark_v1.3.db` into the simulator (Data Setup above). The ETL that built it is deterministic for non-LLM stages; LLM-assisted stages record model + prompt provenance.
2. Run the eval matrix for each task/model pair reported in the paper.
3. Ablations: `scripts/ablation_relevance.py`, `scripts/run_specialty_sweep.py`.

## Known issues and caveats

- **The paper's agentic patient-diagnosis scores are inflated.** Until Stage 2 of this fork,
  `view_problem_list`, the `active_problems` field of `open_chart` and the FHIR Condition resource
  returned the graph-derived correct diagnoses with ICD-10 codes, i.e. the label; the `eval/agents`
  harness used for the paper's agentic runs hid assessment/plan client-side but left that list. Every
  serving path now returns the chart's documented history instead and hides assessment/plan server-side.
  Use the environment endpoints for any new agent evaluation.
- Later encounter notes carry an "Active Problem List" of earlier encounters' diagnoses by construction
  (that is the chart); only the current encounter's diagnosis and the coded list are withheld.
- The imaging-indication reference question is LLM-authored (anchored to the graph diagnosis); the other
  three tasks' labels are deterministic functions of the graph.
- The terminology tables ship empty (licensing); see Terminology above.

## License & Citation

Code: [LICENSE](LICENSE). The released data files are fully synthetic and redistributable; the source material the pipeline was built from is not included.

`benchmark_v1.3.db` (88 MB) is committed directly (under GitHub's 100 MB file limit), so a plain `git clone` brings the data with no Git LFS setup; it is also attached to each GitHub release as a downloadable asset.
