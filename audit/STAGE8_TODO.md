# Stage 8: evaluation protocol — todo

Scope: ROADMAP Stage 8 only. Goal: a protocol every number this fork reports follows (floors, ceilings,
paired intervals, the generator outside the ranked panel, one-factor ablations, random agentic samples,
failures counted), the code that enforces it, and the first protocol-conformant results committed under
`results/`. Done = every box checked, a leaderboard with intervals for the chosen panel on the public
split, validation filled in.

## Design decisions (fixed before coding)
- **Protocol = `EVAL_PROTOCOL.md`**; every table in README / DATA_CARD / results cites it, and the paper's
  Tables 2–3 / Appendix A are labeled non-protocol.
- **One runner for every task and both arms** (`eval/protocol_run.py`): a model plays instances as
  `LocalEnv` episodes (parity-tested against the HTTP env), so the reward is the environment's, the cutoff /
  hidden sections / overrides / budget are enforced, and the new Stage-7 tasks (incl. `order_test`) run the
  same way as the old ones. `arm=agent` = tool loop; `arm=single` = one call over the visible chart with
  only the submit tool: the tools-vs-no-tools ablation on identical instances. No Postgres needed for runs;
  artifacts are JSONL + manifest per run.
- **Sampling**: seeded random instances, at most one per patient first (`sample_instances`); the legacy
  agent runner's `select_patients` now samples at random with a seed (default public; `val` rejected; the
  ceiling filter is off by default and documented as the paper's bias).
- **Failures score 0 and stay in `n`** (API errors after the adapter's retries, unparsable answers, budget
  exhaustion → forced empty submission), each recorded with its error string.
- **Statistics** (`eval/protocol.py`): normalized score from `eval/floors.json`; 95% bootstrap intervals
  (2,000 resamples, seed 0); paired bootstrap of per-instance differences vs the best model; ablation tables
  paired on instances (tools − no tools; atypical − typical via `parent_gt_id`).
- **Panel selection** (Artificial Analysis intelligence vs price, September 2026, tool-calling models on
  OpenRouter / OpenAI): a frontier anchor on a smaller sample, the value frontier, the best open-weights
  models, the cheapest tool-capable model, and Kimi K2.5 (the generator) in its own unranked row. Prices are
  re-read from OpenRouter's live list at run time and written to the manifest.
- **Cost discipline**: smoke 3 instances per model first; project from measured tokens; `--max-usd` per
  run; running spend printed every 10 episodes; launch only when the projection fits the balance.

## Protocol and code
- [x] `EVAL_PROTOCOL.md` (instances/splits, floors/ceilings, intervals, panel vs generator, ablations, agentic sampling, artifacts, cost discipline)
- [x] `eval/protocol.py`: bootstrap CI, paired differences, normalization, leaderboard + ablation tables, `report` CLI
- [x] `eval/protocol_run.py`: LocalEnv episodes with the model adapters, both arms, seeded sampling, failure = 0, resume, cost cap, manifest
- [x] registry: protocol panel + RL candidates (`qwen3.5-9b`, `qwen3.5-27b`, `muse-glimmer-30b`, `qwen3.5-35b-a3b`) + current frontier ids; `scripts/refresh_model_registry.py --check/--update` validates every id against OpenRouter and refreshes `eval/model_prices.json` (legacy `gpt-5.3`, `opus-4.6` ids no longer served, kept for the record)
- [x] runner fixes: `select_patients` seeded random / default public / ceiling filter off; `eval/split.py` and `scripts/build_rl_split.py` no longer reference the retired task
- [x] `eval/tests/test_protocol.py` green (8) (arithmetic; scripted-adapter runs: oracle agent → 1.0 incl. `order_test`, failure → 0 recorded, resume, single arm sees the visible chart only)

## Runs (public split; OPENROUTER_API_KEY in .env; balance was $0.54 of $130 on 2026-09-27 → top-up needed)
Decision (2026-09-27): the fork's question is whether RL on this environment improves a model, so the paid
panel is one RL candidate + one anchor, not a leaderboard of ten. Models: **qwen3.5-9b** (9B dense, open
weights, tool calling, $0.10/$0.15 hosted; trainable on 1–2 GPUs) as the RL candidate; **gpt-6-sol**
($2/$10) on 40 instances/unit as the prompting ceiling; Kimi K2.5 and other models only after RL shows a gain.
- [x] smoke glm-5.3-flash, patient_diagnosis: 3 episodes, reward 0.42, 12.4k tokens, 6.3 steps, $0.0007/episode
- [x] smoke qwen3.5-9b, test_selection: 3 episodes, reward 0.00 (wrong diagnoses; 1–3 orders each), 28k tokens
      (one episode 22k output tokens: verbose reasoning), 6 steps, $0.0033/episode → 9 units × 120 ≈ $3.5
- [x] qwen3.5-9b agent arm: 10 scoring units × 120 (atypical 77 paired) = 1,157 episodes, 10 errors (0.9%; 7 in error_detection, see below)
- [x] qwen3.5-9b single arm (no tools) on patient_diagnosis / test_selection / differential_diagnosis, 360 episodes (rerun with the format retry, H15)
- [x] gpt-6-sol agent arm, 40/unit (atypical 23) = 383 episodes, 0 errors, $7.96 (OpenAI credits)
- [ ] (later, after RL shows a gain) Kimi K2.5 separate row, further open models
- [x] `python -m eval.protocol report` → `results/leaderboard.md`, `results/summary.json`; predictions + manifests committed
- [x] `scripts/rescore_protocol_runs.py`: every run re-scored with the final scorer (as-run value kept in `reward_asrun`;
      unchanged units reproduce their as-run rewards exactly)
- [x] reward-noise audit (`scripts/reward_noise_audit.py`, GPT-6 Sol as judge, audit only): `results/reward_noise_audit*.json`
- [x] RL go/no-go (`scripts/stage8_rl_decision.py`): `results/rl_decision.md`

## Full-run log (2026-09-27, 19:07–20:36)
- OpenRouter $9.93 → $5.46 (billed $4.47: Qwen runs $3.96 recorded + the superseded single arm $0.18 + calls lost to
  restarts/retries, 13% over recorded); OpenAI $7.96 + audit
- 19:25 crash of the Qwen agent process at 199/1157: a provider error inside the 200 stream ("Upstream error from
  Venice: Stream interrupted") was raised as a non-retried HTTPStatusError, and reading its unread streaming body in
  the error handler killed the run → H16: mid-stream errors are transport errors (retried), error reporting can no
  longer raise; resumed (no completed episode lost), later restarted at 32 workers (all workers were waiting on
  generation: 16% CPU, 32 connections)
- H15: the single arm executed tool calls the arm does not offer (Qwen called `order_test` in 43% of single-arm
  test_selection episodes → forced 0) → never executed; one corrective turn; rerun (forced 52 → 4, reward 0.058 →
  0.094; the remaining forced are 16,384-token thinking loops)
- R8 (reward-noise audit of GPT-6 Sol's zeros during the run): a clinically correct name inside a more specific name
  in another ICD block scored 0 ("Sepsis due to pneumonia with septic shock" for Septic shock, acute GVHD after HSCT,
  infant physiologic reflux) → related credit (0.5) across blocks when the reference has ≥ 2 content words; the
  audit of Qwen's zeros then found "I-cell disease" read as type 1 (a roman-numeral bug) → fixed; all runs rescored
  (Sol patient_diagnosis 0.482 → 0.514; every other change ≤ 0.02); floors unchanged
- error_detection: 7 of 120 Qwen episodes error — the model loops (14–28 turns, 10k–48k output tokens) until the
  15-min deadline cuts a call; the unit is near saturation (Qwen 0.89, Sol 1.00) and fails the ≤ 2% error gate

## Results (public split, protocol-conformant; `results/leaderboard.md`)
| unit | floor | Qwen3.5-9B tools (n=120) raw (norm) | Qwen no tools | GPT-6 Sol tools (n=40) |
|---|---|---|---|---|
| patient_diagnosis | 0.036 | 0.305 (+0.28) | 0.226 (+0.20) | 0.514 (+0.50) |
| atypical_diagnosis | 0.035 | 0.271 (+0.24) (n=77) | — | 0.625 (+0.61) (n=23) |
| differential_diagnosis | 0.027 | 0.390 (+0.37) | 0.324 (+0.31) | 0.524 (+0.51) |
| test_selection | 0.001 | 0.247 (+0.25) | 0.095 (+0.09) | 0.315 (+0.31) |
| evidence_retrieval | 0.420 | 0.563 (+0.25) | — | 0.700 (+0.48) |
| context_summarization | 0.487 | 0.525 (+0.07) | — | 0.619 (+0.26) |
| specialty_conditioned | 0.407 | 0.595 (+0.32) | — | 0.589 (+0.31) |
| imaging_indication | 0.219 | 0.249 (+0.04) | — | 0.289 (+0.09) |
| error_detection | 0.494 | 0.892 (+0.79) | — | 1.000 (+1.00) |
| lab_triage | 0.631 | 0.425 (−0.56) | — | 0.487 (−0.39) |

- **Tools help** (paired, Qwen, n=120): +0.079 patient_diagnosis, +0.065 differential, +0.148 test_selection, all
  intervals above 0. **Atypical − typical** is not significant for either model (Qwen −0.03, Sol +0.07).
- **RL go/no-go: GO** — 4 units have headroom (normalized 0.05–0.80), a significant anchor gap ≥ 0.10 and ≤ 2%
  errors: patient_diagnosis (gap 0.21), atypical_diagnosis (0.39), evidence_retrieval (0.15), differential (0.13).
  test_selection's gap (0.09, interval above 0) and specialty's (0.09, interval spans 0) fall just short.
- **Reward noise** (judge: clinically the same diagnosis but scored 0, among Qwen's answered zeros):
  patient_diagnosis 9%, atypical 21%, differential 0%, test_selection 13%; the residue is synonymy the token rules
  cannot see (Asherman ↔ intrauterine adhesions, HSP ↔ IgA vasculitis, arsenic toxicity ↔ poisoning) — a
  concept-level (SNOMED) matcher is the follow-up.
- **Findings for the task set**: lab_triage is below its floor for both models (flag-everything beats them: the
  metric rewards recall of every key/supporting result, the models flag a few) — re-specify before RL;
  error_detection is saturated; imaging_indication and context_summarization have little headroom (normalized
  ≤ 0.1 for Qwen, ≤ 0.26 for Sol).

## Apply to what the fork reports
- [x] README results section cites the protocol and the leaderboard; paper numbers labeled non-protocol
- [x] memory

## Validate
- [x] `pytest eval/tests etl/tests` green; floors current (4 splits); Docker parity + simulator suite green
- [x] committed and pushed; main synced

## Pre-launch audit (2026-09-27, after the 36-unit atomic smoke) — every issue found, and its fix
Reward validity
- [x] R1 test_selection: `order_test` matched test names against *finding* names ("Bilateral symmetric high-frequency hearing loss"), so natural orders ("audiogram", "fern test", "stress test") returned "not performed" and evidence coverage collapsed → test-name matcher (result words stripped, prefix stems, canonical test lexicon mapping findings to the tests that report them)
  - done: each orderable finding carries `context`, the content tokens of the index encounter's result-section lines/sentences that document it (by name, ≥60% of its name tokens, its documented number, or its documented wording); an order matches by name as before **or** when all its content words appear in a finding's context (prefix stems, test-name synonyms: audiogram→audiometry, sonography→ultrasound, MPI, SPECT→nuclear, LDH, RBC). Generic orders ("labs", "imaging", "blood test") match by name only (<5% of instances, tested). 90.5% of discriminating findings have a context (the rest are exam maneuvers visible in the chart). `scripts/build_stage7_tasks.py` writes it on a fresh build; `--patch-order-context` rewrote the 1,960 built rows in place (release + overlay; `release_info.stage7_order_context`; DB 96.4 MB). The single-turn scorer uses the same matcher.
- [x] R2 diagnosis credit is ICD-only: synonymous codes in other categories (I20.9 vs I25.119, H90.5 vs H91.13) and miscoded labels (P01.1, a newborn code, for maternal PPROM) score 0 for clinically correct answers (9/9 smoke test_selection answers correct, 7/9 credited 0) → name-equivalence credit (0.75) in every diagnosis scorer; floors re-derived
  - done: `eval.scoring.dx_credit` = max(ICD credit, name credit); name credit 0.75 for the same normalized name (parentheticals dropped, abbreviations expanded, digits kept), 0.5 for containment/Jaccard ≥ 0.5 within the same ICD block, 0 on a contradicting qualifier (left/right, acute/chronic, type 1/2, non-, with/without ...). Used by patient_diagnosis/atypical (greedy matching), differential and test_selection. Rescored smoke: Qwen differential 0.14→0.39, test_selection single 0.06→0.13; Sol and patient_diagnosis unchanged. Floors unchanged (the problem-list copy policy already maps names to exact codes).
- [x] R3 specialty_absent alone has floor 1.0 (always abstain) — no headroom; the task is the mixture → combined `specialty_conditioned` unit (per-row metric) in degenerate/floors/runner; the paid run samples the mixture (public floor 0.41, abstain_always; 120 sampled = 61 involved / 59 absent)
- [ ] R4 error_detection: 1.00 for all three models in the smoke (floor 0.49) — probably saturated; measured in the full run, excluded from RL targets if so (finding, no code change)
- [ ] R5 lab_triage: floor 0.63 (flag every result) — little headroom by construction (finding)
Harness / protocol
- [x] H1 cost was list-price only; OpenRouter actually billed ~2x (smoke: $0.143 list vs $0.28 balance change, incl. abandoned calls) → OpenRouter usage accounting per call (`usage.cost`), actual-cost projection, in-runner balance guard
- [x] H2 provider stalls (OpenRouter held calls 5–12 min) → per-call cap bounded by the 15-min episode deadline, provider logged per call. The re-smoke showed a flat 120 s cap kills healthy long-thinking calls (a single-arm Qwen call died after 3×120 s; agent turns on slower providers were retried and re-billed) → cap = 30 s + max_tokens / 15 tok/s (303 s at 4,096, 1,122 s at 16,384), still bounded by the episode deadline
- [x] H3 units ran sequentially, each unit's stragglers idling the pool → one interleaved pool across all units per model
- [x] H4 one 326 MB LocalEnv per worker thread limits concurrency → one shared read-only release DB + search index per process
- [x] H5 predictions did not record the orders or what they matched → orders (name, matched finding) and unmatched count per episode
- [x] H6 single arm counted chart assembly as 5 environment steps → chart assembled outside the step counter
- [x] H7 atypical − typical ablation was not paired by construction (independent samples) → atypical instances sampled from the patient_diagnosis sample's parents (120 → 77 paired atypical instances; the rest of the parents have no atypical variant)
- [x] H8 Kimi (generator row, checklist item) missing from the plan → small-n run (20/unit) after the RL candidate, only if the OpenRouter balance keeps ≥ $2 margin (`KIMI=1 bash scripts/run_stage8_panel.sh`, off by default)
- [x] H9 sampling differs by provider (GPT-6 Sol only accepts default temperature; reasoning_effort none with tools; Qwen temperature 0, thinking on) → recorded per manifest, stated in the report
- [x] H10 plan/docs drift: GLM dropped (not an RL candidate); unit counts; run script → updated
- [x] H12 (re-smoke) abandoned calls ran on a ThreadPoolExecutor whose workers are joined at exit, so finished runs stayed alive behind stalled calls → each call on a daemon thread
- [x] H13 (re-smoke) the single arm at 4,096 output tokens truncated Qwen's thinking before any answer (1/3 patient_diagnosis scored 0 for that reason) → single arm gets the registry's 16,384 (one call = the whole thinking allowance); a remaining 16,384-token thinking loop is a real model failure and scores 0
- [x] R6 (re-smoke, 20-episode matcher checks on Sol and Qwen) orders named by analyte missed results named by interpretation ("serum sodium" / BMP vs "Hyponatremia", ABG vs "Hypercapnia"), "iron studies" missed (the stemmed "studie" escaped the narrative list), "Lyme serology" missed → interpretation→analyte lexicon, stemmed stop lists, shared-stem token match (serology/serologic), lyme→borrelia. Remaining unmatched orders are tests the chart never documents (Legionella urine antigen, CT head, Histoplasma antigen) — correct "not performed" answers. Qwen repeated one undocumented order ~34× in one episode: model behaviour (budget exhaustion, parsimony-penalized), an RL target, not a harness fault
- [x] H14 (re-smoke) with the long cap a provider that holds the connection without generating cost a whole episode deadline (2/3 single-arm differential calls sat 900 s) → OpenRouter calls are streamed; a call fails fast when no token arrives within 150 s or none follows for 60 s (keep-alive comments do not count), and closing the stream cancels the generation. Round-5 re-smoke: 0 errors in 38 episodes, one slow-provider retry
- [x] R7 (finding) specialty_involved scores documented findings, not diagnosis names: a Dermatology summary naming hidradenitis suppurativa (Hurley II) but none of its findings scores 0; audited in the full run's reward-noise pass
- [x] H11 hosted Qwen is a preview; the RL before/after must be re-measured under identical local serving (vLLM) → stated in EVAL_PROTOCOL.md §4 (done) and the report
Readiness gate (all must pass before launch)
- [x] tests for R1–R3, R6, H1–H7, H14 (`eval/tests/test_stage8_audit.py`, 44 tests); `pytest eval/tests etl/tests` 236 passed / 1 skipped; floors regenerated for all 4 splits (only specialty_conditioned added and test_selection's order-everything floor moved, 0.0007→0.0009 public)
- [x] Docker image rebuilt, Postgres reloaded, HTTP parity green for every Stage-7 task and the specialty unit (order_test observations and rewards identical), simulator suite 103 passed / 8 skipped
- [x] re-smoke on the final code: Qwen agent 10 units (29 episodes) + no-tools 3 units (9) with streaming, GPT-6 Sol 10 units (29); 0 errors; OpenRouter billed $0.0712 for $0.0694 recorded (2.6% gap); matcher checks 20 episodes each on Sol and Qwen
- [x] projection (`scripts/project_stage8_cost.py`, x1.5 margin): OpenRouter $2.77 → $4.15 against a $9.93 balance; OpenAI $6.93 → $10.40 (free credits); wall clock ~1.2 h (Qwen agent, 16 workers) — caps in `scripts/run_stage8_panel.sh`: Qwen agent $4.50, no-tools $0.30, Sol $12, min balance $1
