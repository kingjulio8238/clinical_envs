# Stage 5: environment throughput — DONE 2026-09-27

Scope: ROADMAP Stage 5 only. Goal: measure the `/env` step path before optimizing, name the bottleneck, and
— if the HTTP → FastAPI → Postgres → Redis path dominates — add an in-process environment over the read-only
SQLite release that behaves identically (observations, visibility rules, budget, reward) and passes the
Stage-1 reward-hacking gates. Done = every box checked, the baseline and the after numbers recorded side by
side, parity tests green, validation filled in.

## Design decisions (fixed before coding)
- **One benchmark harness for both paths** (`eval/bench_env.py`): the same scripted episode (reset →
  open_chart → view_encounters → view_encounter_detail → search_chart → view_results → view_problem_list →
  view_medications → oracle submit) on the same public gt_ids (fixed seed, all four tasks), the same timing
  points (per-tool latency percentiles, episodes/s, steps/s), the same concurrency sweep (threads for HTTP,
  processes for in-process). Numbers are written to `audit/bench/*.json` so before/after is a diff, not a memory.
- **Decompose before optimizing.** The HTTP baseline separates the framework floor (`GET /health`), Redis
  (`GET /env/state`), Postgres reads per tool (`/env/step`), the scorer (`submit`) and the reset (Postgres +
  prompt build), sequentially and under load. The bottleneck is whatever grows fastest with concurrency.
- **In-process env = same code where it is pure, same SQL where it is not.** `eval/local_env.py` reuses
  `env_service.normalize_submission / render / _tool_schemas`, `visibility.strip_outcome_sections /
  filter_future_encounters`, the prompt builders and the Epic pydantic schemas, so observation shapes are
  identical by construction; the tool queries are re-issued against SQLite with the same ordering and filters.
  The reward is `eval.scoring.compute_all_metrics` with the same per-instance context
  (`eval.degenerate.attach_context`), i.e. exactly what `/score` and `/env` compute, with the private overlay
  honoured and the metric breakdown redacted for the private split like the server.
- **Search is the one intentional difference.** Postgres ranks `search_chart` with `ts_rank` over an English
  tsvector; SQLite gets an FTS5 index (porter stemmer, bm25) built once per process from the release text. Same
  candidate set and filters (patient, role, hidden sections, cutoff), different rank order for ties.
- **Parity is tested, not asserted**: `eval/tests/test_local_env.py` checks (a) the Stage-1 policies score
  identically through `LocalEnv.step(submit)` and through the scorer directly, (b) the oracle reaches the
  ceiling on every task, (c) the visibility rules hold, and (d) when a server is reachable (`SH_ENV_URL`),
  observations and rewards match the HTTP env byte-for-byte on sampled episodes.
- **Scorer optimizations must be output-identical**: verified on 300 real sections (`_cur_hits`) and every
  unseen token of 400 sections (`_snap`), then by `python -m eval.floors --split public --check` (bit-exact).

## Baseline (HTTP env, compose stack on this laptop: single uvicorn worker, Postgres 16, Redis 7)
- [x] component floors (sequential, 200 reps; `audit/bench/http_components.json`)

  | call | p50 ms | p95 ms |
  |---|---|---|
  | GET /health (framework floor) | 2.2 | 4.1 |
  | GET /env/state (Redis + framework) | 3.4 | 5.3 |
  | step view_problem_list (Redis + 1 small Postgres query) | 7.6 | 13.2 |
  | step view_encounter_detail (Redis + Postgres, large payload) | 10.7 | 18.3 |
  | step search_chart (Redis + tsvector) | 8.6 | 16.0 |
  | POST /env/reset (Postgres context + prompt build) | 19.4 | 25.5 |
  | step submit (scorer) | 18.7 | 22.9 |

- [x] concurrency sweep, 64 public episodes × 8 steps (`audit/bench/http_baseline.json`)

  | clients | steps/s | episodes/s | reset p50 | detail p50 | search p50 | submit p50 |
  |---|---|---|---|---|---|---|
  | 1 | 13.6 | 1.7 | 23 ms | 11 ms | 10 ms | 33 ms |
  | 4 | **104.8** | 13.1 | 45 | 18 | 16 | 46 |
  | 16 | 97.4 | 12.2 | 103 | 89 | 112 | 103 |
  | 64 | 64.1 | 8.0 | 769 | 278 | 227 | 274 |

- [x] bottleneck named: under 16 clients the app container runs at **100% of one core** while Postgres is at
  40–60% and Redis at 4% (`docker stats`), and latency grows linearly with clients from 4 up — the single-process
  Python server (framework + SQLAlchemy + JSON + Redis round trips per step), not the databases. Ceiling ≈ 105
  steps/s per server process.

## In-process environment
- [x] `eval/local_env.py`: `LocalEnv` with `instances / reset / step / state / close / oracle`, same return types as `eval.env_client.SyntheticHospitalEnv`
- [x] all 9 read tools over SQLite with the Epic schemas; unknown tool / missing argument / budget / wrong submit tool behave like the server
- [x] visibility: outcome sections, documented problem list, point-in-time cutoff for imaging and index-encounter diagnosis
- [x] reward: same scorer, same context, private overlay, redaction; `oracle()` builder
- [x] FTS5 chart search
- [x] `eval/bench_env.py local` (single process) and `--workers N` (processes, warmed before the clock starts)

## Scorer hot paths (found by profiling the in-process env: submit was 60.7 ms mean, 1.6 s worst)
- [x] `imaging_concepts.ConceptExtractor._cur_hits`: curated-pattern table precomputed once, token stems once per call (was 382 parts × N tokens × `stems()`): 1.24 → 0.31 ms per section, identical output on 300 sections
- [x] `imaging_concepts.ConceptExtractor._snap`: vocabulary bucketed by (first letter, length) instead of a full scan per unseen token: 2.24 → 0.50 ms per token, identical snaps
- [x] `LocalEnv.warmup()` preloads the release caches and imports `rouge_score` (1.4 s import of nltk/scipy) so first-touch cost is paid once per process
- [x] `floors --check public` bit-exact; the check itself runs in 2 min 10 s (was ~6.5 min)

## Tests
- [x] `eval/tests/test_local_env.py`: 28 tests — 11 Stage-1 policy parities, 6 oracle ceilings, copy-the-problem-list gate through the env, malformed submissions, cutoff + hidden sections, documented problem list, budget forcing, seeded sampling, 4 live HTTP-parity tests (ran against the compose stack: no skips)
- [x] `pytest eval/tests etl/tests` green (see Validation); simulator suite untouched

## After (same harness, same 64 gt_ids; `audit/bench/local_64.json` before the scorer fixes, `local_64_after.json` after)

| path | steps/s | episodes/s | reset p50 | detail p50 | search p50 | submit p50 / mean |
|---|---|---|---|---|---|---|
| HTTP, 1 client (baseline) | 13.6 | 1.7 | 23.4 ms | 10.6 ms | 10.0 ms | 33 ms |
| HTTP, 4 clients (server peak) | 104.8 | 13.1 | 45 | 18 | 16 | 46 |
| LocalEnv, 1 process, before scorer fixes | 122.7 | 15.3 | 0.24 | 0.15 | 2.5 | 6.0 / 60.7 |
| **LocalEnv, 1 process, after** | **470.5** | **58.8** | 0.21 | 0.14 | 2.4 | 2.8 / 12.9 |

Speedup: **35× over the sequential HTTP env, 4.5× over the server's saturation peak, per process.** Tool steps
cost 0.05–0.3 ms; the remaining per-episode cost is the reward (summarization concept matching on first touch of
a patient, ~50 ms; later visits of the same patient ~10 ms), then FTS5 search (2.4 ms).

Process scaling (256 episodes; `local_scaling.json` before, `local_scaling_after.json` after; 1,024 episodes in
`local_scaling_1024.json`): 1 → 464, 2 → 602 (630 at 1,024 ep), 4 → 544 (696 at 1,024 ep), 8 → 382 steps/s.
**Not a clean measurement**: the host's load average was 27–30 on 8 cores during the sweep (an unrelated
OpenFOAM container held 4.3 cores throughout, plus Docker Desktop), so 4 and 8 workers were competing for ~3
free cores. Per-process throughput is the trustworthy number; scaling should be re-measured on an idle machine.
Each worker also pays its own first-touch cost per patient (caches are per process).

## Validate
- [x] parity test run against the live compose stack recorded (28 passed, 38.7 s)
- [x] floors bit-exact after the scorer optimizations
- [x] committed and pushed; main synced
