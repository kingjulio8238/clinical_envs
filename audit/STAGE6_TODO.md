# Stage 6: pipeline reproducibility — DONE 2026-09-27

Scope: ROADMAP Stage 6 only. Goal: make the ETL that built the benchmark runnable end to end by someone with
their own source corpus and an open LLM endpoint, deterministic where the tooling allows it, and loud about
what it cannot do. Done = every box checked, the synthetic end-to-end smoke recorded, validation filled in.
Nothing here regenerates the released data; the released DB is untouched.

## Design decisions (fixed before coding)
- **One LLM client** (`etl/llm.py`) replaces the seven per-stage copies of `_call_kimi` / `_call_with_retry`
  / cache-key / `llm_call_log` code. Configuration is by environment: `SH_LLM_API` (`openai` chat-completions
  — the open default — or `anthropic` messages, which the private gateway spoke), `SH_LLM_BASE_URL`,
  `SH_LLM_MODEL`, `SH_LLM_API_KEY`, `SH_LLM_TEMPERATURE` (0), `SH_LLM_SEED` (0), `SH_LLM_TOP_P`,
  `SH_LLM_MAX_TOKENS`, `SH_LLM_TIMEOUT`, `SH_LLM_MAX_ATTEMPTS`. The legacy `SH_LLM_GATEWAY_URL` still works
  (it selects the anthropic format at that URL). Stage modules keep their function names (`_call_with_retry`,
  `_compute_input_hash*`, `MODEL`) as thin wrappers, so the stage logic is unchanged.
- **Retry covers validation.** `LLMClient.call_json(prompt, validate=...)` retries HTTP 429/5xx, timeouts,
  transport errors, unparsable JSON **and** structural-validation failures (the audit found validation ran
  after the retry loop, so a malformed extraction was logged and dropped). A retry after a bad reply appends a
  short repair note to the prompt; the cache key stays the original prompt's.
- **Cache key = the full input**: SHA-256 over stage, model, temperature, seed, top_p, max_tokens and the
  complete prompt. The audit found 5a keyed on the first 200 characters of the question, 7 on 16 hex characters
  of the vignette hash, 6b/6c/8b/9a/9b/10c/10e on record ids only, so an edited prompt or record hit a stale
  cache. Every `llm_call_log` row now also records `temperature`, `seed`, `attempts`, `endpoint` and the real
  `request_id` (10c/10e stored the raw response text in that column).
- **Seeded generation**: temperature 0 and a fixed seed are sent on every request (OpenAI format; the anthropic
  format has no seed and the manifest says so). Nothing in the deterministic stages uses `random`; the one
  pipeline seed (`SH_PIPELINE_SEED`) is recorded in the manifest and applied to `random` at start for any
  future stage that needs it.
- **Fail loudly**: `etl/preflight.py` checks, before a stage runs, that its inputs exist — the source
  directory with at least one profile-matched deck, the ICD-10-CM table, the SNOMED RF2 files, LOINC, the
  clinician-curated CSVs for stage 11 (`finding_site_review.csv`, `curated_diagnosis_edges.csv`), an
  answering LLM endpoint for LLM stages — and lists everything missing with the environment variable or path
  that fixes it. Stage 11 raises instead of silently changing the specialty labels when the curated files are
  absent (`--allow-missing-curated` restores the old fallback, explicitly).
- **All stages in `etl/main.py`**: 1 ingest, 2 classify, 3 board extraction, 4 fact cards (+ topic
  enrichment), 5 ontology (5a extract, 5b ICD map, 5d SNOMED, 5e SapBERT when an index exists, 5f LOINC),
  6 relationships (6a, 6b, 6c), 7 EHR sections, 8 patients (8a, 8b), 9 encounters (9a, 9b), 10 ground truth
  (10a–10e), 11 typed diagnosis relations (s06d), 12 specialty tiers (s10f). `--stage N`, `--from/--to`,
  `--all`, `--pilot`, `--workers`, `--dry-run`, `--sources DIR`, `--skip-preflight`. Every stage appends to
  `data/pipeline_manifest.json`: settings fingerprint, seeds, SHA-256 of every input file, git commit, summary.
- **Bring your own corpus**: the deck/PDF profile system already makes the ETL source-agnostic; this stage adds
  the missing pieces — `--sources DIR`, the preflight, a synthetic end-to-end smoke that needs no source
  content, and `scripts/source_fingerprints.py`, which hashes every source item (normalized text, plus 8-word
  shingles) so a benchmark built from a private corpus can be checked against any public dump for
  contamination without shipping the text.

## Wiring and configuration
- [x] `etl/main.py` runs stages 1–12 (`--stage`, `--from/--to`, `--all`, `--pilot`, `--workers`, `--dry-run`, `--sources`)
- [x] `data/pipeline_manifest.json` written per stage (settings, seeds, input hashes, git commit, summary)
- [x] `etl/preflight.py`: per-stage requirements; missing inputs listed with the fix; exit code 2
- [x] stage 11 fails loudly without the curated CSVs unless `--allow-missing-curated`

## LLM client
- [x] `etl/llm.py`: openai + anthropic formats, env configuration, seed/temperature on every request
- [x] retry loop covers HTTP/timeout/transport/parse/validation; repair note on retry; attempts recorded
- [x] cache key = full input; `llm_call_log` gains `temperature`, `seed`, `attempts`, `endpoint`
- [x] stages 1d, 4c, 5, 6, 7, 8, 9, 10 use the client; validators wired into the retry: 5a extraction,
      6b list, 6c dict, 7 list, 8b profile keys, 9a timeline, 9b polished HPI, 10c/10c_visit/10e dicts
- [x] 10c/10e store the real request id

## Source corpus
- [x] `scripts/source_fingerprints.py --write` (hashes + shingles, no text) and `--check corpus.jsonl` (overlap report)
- [x] README "Bring your own source corpus": profiles, preflight, endpoint, seeds, fingerprints, what is and is not reproducible

## Tests
- [x] `etl/tests/test_llm_client.py` (mock transport): both formats, retry on 500 → success, parse retry, validation retry with repair note, attempts exhausted raises, seed/temperature in payload, full-input cache key
- [x] `etl/tests/test_preflight.py`: missing inputs reported; `--allow-missing-curated`
- [x] `etl/tests/test_main_wiring.py`: every stage 1–12 resolves to a callable
- [x] `etl/tests/test_fingerprints.py`: normalization, shingles, overlap
- [x] synthetic end-to-end smoke: a scratch DB with the ETL schema and 3 synthetic board questions, a mock LLM
      server, `python -m etl.main --stage 5 --pilot 3` then `--stage 7`: rows in `llm_call_log` with the new
      columns, cache hit on rerun, one forced validation failure retried and recovered
- [x] `pytest eval/tests etl/tests` green

## Validate
- [x] preflight on this checkout lists exactly the missing inputs (no source decks, no SNOMED/LOINC, no curated CSVs)
- [x] committed and pushed; main synced

## Validation record
- `pytest etl/tests`: 38 passed, 1 skipped (registry, LLM client ×10, wiring/preflight ×8, synthetic end-to-end smoke).
- Smoke (mock endpoint, `--fail-first 1`): stage 5 on 3 synthetic questions made 4 requests (3 + 1 retry after the
  forced invalid reply), `llm_call_log` rows carry model/temperature 0/seed 123/attempts {1,1,2}/endpoint/request id,
  5b validated K35.80 against the CMS table (2 diagnosis nodes, 3 findings); a rerun made 0 requests (full cache
  hits); stage 7 wrote 12 sections through the same client; the manifest recorded runs [5, 5, 7] with the LLM
  fingerprint (no key), pipeline seed and the ICD table's SHA-256.
- Preflight on this checkout (`python -m etl.preflight --all --no-llm`): missing = source decks, board_questions
  (no data/benchmark.db), SNOMED RF2, LOINC, finding_site_review.csv, curated_diagnosis_edges.csv — exactly the
  inputs the repository does not ship.
- Not done here, by design: regenerating the released data (no source corpus, no model), and the split scripts /
  `select_patients`, which operate on the released database and are Stage-8 material.
