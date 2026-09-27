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
- [ ] qwen3.5-9b agent arm: 120 instances × 9 scoring units (≈ $3.5 × 1.5 margin)
- [ ] qwen3.5-9b single arm (no tools) on 3 units — tools ablation (≈ $0.5)
- [ ] gpt-6-sol agent arm on 40/unit (360 episodes × ~15k tokens ≈ $12 × 1.5 margin)
- [ ] (later, after RL shows a gain) Kimi K2.5 separate row (~$8), further open models
- [ ] `python -m eval.protocol report` → `results/leaderboard.md`, `results/summary.json`; predictions + manifests committed

## Apply to what the fork reports
- [ ] README results section cites the protocol and the leaderboard; paper numbers labeled non-protocol
- [ ] DATA_CARD / ROADMAP / STATE updated; memory

## Validate
- [ ] `pytest eval/tests etl/tests` green
- [ ] committed and pushed; main synced
